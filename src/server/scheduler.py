"""Continuous-batching scheduler.

One engine thread owns the GPU. Requests wait in a bounded queue (full -> QueueFull -> HTTP 429), then move
waiting -> prefilling -> running. Each loop:
  1. admit waiting requests while there is a batch slot and enough free KV blocks (their prompt blocks are
     reserved now, so a prefill never runs out halfway);
  2. run ONE packed prefill forward over at most `prefill_chunk` prompt tokens, taken first-come-first-served
     from the prefilling requests: several short prompts share one pass over the weights, and a long prompt is
     split into chunks across iterations (chunked prefill), so running requests are never stalled for long;
  3. run ONE batched decode step for every running request (each weight read once for the whole batch);
  4. sample the batch's next tokens in one batched pass and stream them; finished requests free their KV
     blocks immediately, so a waiting request can take the slot on the very next step.
If the KV pool runs dry mid-generation, the newest running request is preempted: its blocks are freed and it is
re-queued to recompute its prefix later (vLLM-style recompute preemption); its client sees no gap in the text.
"""
import collections
import concurrent.futures
import inspect
import itertools
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from backend.pinning import lock_in_memory, model_tensors
from sampler import SamplingParams, sample, sample_batch
from state import OutOfBlocks


class QueueFull(Exception):
    pass


@dataclass(eq=False)                                  # identity semantics: hashable, compared by object
class Request:
    id: str
    prompt_ids: list[int]
    params: SamplingParams
    max_new_tokens: int
    out: object = field(default_factory=queue.Queue)       # anything with .put: ("token", text)... then ("done", reason) or ("error", msg)
    arrived: float = field(default_factory=time.perf_counter)
    generated: list[int] = field(default_factory=list)
    emitted: str = ""
    prefix_offset: int = 0                                # incremental detokenization window (see _emit)
    read_offset: int = 0
    state: object = None
    to_prefill: list[int] = field(default_factory=list)  # prompt (+ tokens made before a preemption) to prefill
    prefill_pos: int = 0                                  # how many of them are already in the state
    first_token_at: float | None = None
    last_token_at: float | None = None
    generator: torch.Generator | None = None
    cancelled: bool = False
    finish_reason: str | None = None


def _settle(fut: concurrent.futures.Future, result=None, error: Exception | None = None) -> None:
    """Complete a job's future unless the requester already cancelled it (it may do so at any moment)."""
    try:
        fut.set_exception(error) if error is not None else fut.set_result(result)
    except concurrent.futures.InvalidStateError:
        pass


class Scheduler:
    def __init__(self, engine, metrics, max_batch: int = 8, max_waiting: int = 64, kv_blocks: int = 1024,
                 block_size: int = 16, max_model_len: int = 4096, prefill_chunk: int = 512,
                 batch_wait_ms: float = 5.0, lock_weights: bool = False):
        if not 1 <= max_batch <= 32:                          # batched decode kernels cover up to 32 rows
            raise ValueError("max_batch must be between 1 and 32")
        self.eng, self.metrics = engine, metrics
        self.model, self.tok = engine.model, engine.tokenizer
        if prefill_chunk < 1:
            raise ValueError("prefill_chunk must be at least 1")
        if not 0 <= batch_wait_ms <= 1000:                    # also rejects NaN and inf (a wait that long raises)
            raise ValueError("batch_wait_ms must be between 0 and 1000")
        cap = getattr(engine, "max_model_len", None)          # a model's own limit (registry and model)
        self.max_batch, self.max_waiting = max_batch, max_waiting
        self.max_model_len = min(max_model_len, cap) if cap else max_model_len
        self.prefill_chunk = prefill_chunk
        self.batch_wait = batch_wait_ms / 1e3
        # the pool's units: blocks of every layer (Qwen3.5), or of one layer group (Tiny Aya, whose sliding layers keep
        # a ring of blocks per sequence, sized for chunks of at most prefill_chunk tokens)
        self.pool = self.model.new_paged_pool(kv_blocks, block_size, max_seqs=max(max_batch, 2),
                                              max_chunk=prefill_chunk)
        self.block_size = block_size
        self.waiting: collections.deque[Request] = collections.deque()
        self.prefilling: list[Request] = []                  # admitted, prompt partly in the state
        self.running: list[Request] = []                     # decoding
        self.active: set[Request] = set()                     # submitted and not finished (waiting, prefilling, running)
        self.cond = threading.Condition()
        self.accepting = True
        self._stop = False
        self.jobs: collections.deque = collections.deque()   # (fn, future): whole-model work, e.g. a decision
        self.job = None                                       # the job in progress: (generator, future)
        self._ids = itertools.count()
        self._arrivals = 0                                    # submitted so far: the batching window counts these
        self._warmup()                                        # every layer has run: the weight cache is complete
        # keep the weights in RAM (MPS only): otherwise an idle server's weights get compressed or swapped and the
        # next request waits for them to come back (backend/pinning.py)
        self.weights_locked, self.lock_error = 0, None
        if lock_weights:
            try:
                self.weights_locked = lock_in_memory(model_tensors(self.model))
            except (OSError, ValueError) as e:                # serve anyway, without the guarantee
                self.lock_error = str(e)
        metrics.set("weights_locked_bytes", self.weights_locked)
        metrics.set("kv_blocks_total", self.pool.allocator.num_blocks)            # in units of the pool
        metrics.set("kv_unit_bytes", self.pool.unit_bytes)
        metrics.set("kv_blocks_free", self.pool.allocator.num_free)
        self.thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self.thread.start()

    def _warmup(self) -> None:
        """Page in every weight and build every fused/quantized tensor before /ready says yes (weights load lazily),
        through both the single-sequence and the batched decode paths. Fails fast on a bad model/backend combo."""
        states = [self.model.new_paged_state(self.pool) for _ in range(2)]
        self.model.forward_packed([(torch.tensor([0]), st) for st in states])
        self.model.decode_batch([0, 0], states)
        for st in states:
            st.free()

    # ------------------------------------------------------------------ public API (any thread)
    def submit(self, prompt_ids: list[int], params: SamplingParams, max_new_tokens: int, out=None) -> Request:
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) + max_new_tokens > self.max_model_len:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({max_new_tokens}) exceeds {self.max_model_len}")
        if self._blocks_for(len(prompt_ids) + max_new_tokens) > self.pool.allocator.num_blocks:
            raise ValueError("request can never fit in the KV pool")    # would be preempted forever otherwise
        req = Request(f"req-{next(self._ids)}", list(prompt_ids), params, max_new_tokens)
        if out is not None:
            req.out = out
        if params.seed is not None:
            req.generator = torch.Generator().manual_seed(params.seed)
        with self.cond:
            if not self.accepting or len(self.waiting) + len(self.jobs) >= self.max_waiting:
                self.metrics.inc("requests_rejected_total")
                raise QueueFull("server busy")
            self.waiting.append(req)
            self.active.add(req)
            self._arrivals += 1
            self.metrics.inc("requests_total")
            self.metrics.inc("prompt_tokens_total", len(prompt_ids))
            self.cond.notify()
        return req

    def cancel(self, req: Request) -> None:
        """Any thread. A queued request is dropped now (its queue slot frees at once); a running one stops at the
        engine's next step."""
        req.cancelled = True
        with self.cond:
            queued = req in self.waiting
            if queued:
                self.waiting.remove(req)
        if queued:
            self._finish(req, "cancelled")

    def run_job(self, fn) -> concurrent.futures.Future:
        """Any thread: run fn() on the engine thread (the GPU's only user); the future gets its result or exception.
        If fn returns a generator, the engine advances it one step per loop iteration (the return value is the
        result), so decode steps for running streams keep going in between. Jobs run one at a time, in order; one
        whose future is cancelled before or during its run is dropped. Shares the waiting queue's capacity."""
        fut = concurrent.futures.Future()
        with self.cond:
            if not self.accepting or len(self.waiting) + len(self.jobs) >= self.max_waiting:
                self.metrics.inc("jobs_rejected_total")
                raise QueueFull("server busy")
            self.jobs.append((fn, fut))
            self.cond.notify()
        return fut

    def ready(self) -> bool:
        return self.accepting and self.thread.is_alive() and len(self.waiting) + len(self.jobs) < self.max_waiting

    def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Graceful: stop accepting, let running/waiting requests finish (up to the timeout), then stop the loop."""
        with self.cond:
            self.accepting = False
            self.cond.notify()
        deadline = time.perf_counter() + drain_timeout
        while self.active and time.perf_counter() < deadline:  # includes a request mid-prefill (in neither list)
            time.sleep(0.05)
        with self.cond:
            self._stop = True
            self.cond.notify()
        self.thread.join(timeout=30)                          # the loop exits after its current step
        for r in list(self.active):                           # drain timed out: tell the stragglers
            self._finish(r, "error", "server shutting down")
        pending = [self.job[1]] if self.job else []
        self.job = None
        while self.jobs:
            pending.append(self.jobs.popleft()[1])
        for fut in pending:
            _settle(fut, error=RuntimeError("server shutting down"))

    # ------------------------------------------------------------------ engine thread
    def _loop(self) -> None:
        while True:
            with self.cond:
                idle = not self.prefilling and not self.running
                while not (self._stop or self.waiting or self.prefilling or self.running or self.jobs or self.job):
                    self.cond.wait()
                if self._stop:
                    return
                if self.job is None and self.jobs:
                    self.job = self.jobs.popleft()
                if idle and self.batch_wait and self.waiting:
                    self._gather()
            if self.job is not None:
                self._job_step()
            try:
                self._admit()
                if self.prefilling:
                    self._prefill_step()
                if self.running:
                    self._decode_step()
            except Exception as e:                            # never let one bad step kill the server
                for r in list(self.prefilling) + list(self.running):
                    self._finish(r, "error", str(e))
            self._update_gauges()

    def _gather(self) -> None:
        """Called with the lock held when an idle engine has just been handed work: keep collecting arrivals while
        they come within `batch_wait` of each other (at most 2 x batch_wait in total), so requests that arrive
        together (clients whose requests all finished on the same step) share one packed prefill pass instead of
        the first one being prefilled alone. A lone request waits at most `batch_wait`."""
        deadline = time.perf_counter() + 2 * self.batch_wait
        while not self._stop and len(self.waiting) < self.max_batch:
            before = self._arrivals                           # arrivals, not queue length: a cancel is not one
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self.cond.wait(min(self.batch_wait, remaining))
            if self._arrivals == before:
                return                                        # nothing arrived within batch_wait: go

    def _blocks_for(self, n_tokens: int) -> int:
        """Pool units a sequence of n_tokens positions holds (only grows with n_tokens)."""
        return self.pool.blocks_for(n_tokens)

    def _admit(self) -> None:
        """Move waiting requests to prefilling while there is a batch slot, a state slot and room in the KV pool.
        No compute here: the prompt's blocks are reserved and the prefill happens in chunks (_prefill_step)."""
        while True:
            with self.cond:
                in_flight = len(self.running) + len(self.prefilling)
                if not self.waiting or in_flight >= self.max_batch:
                    return
                req = self.waiting[0]
                if req.cancelled:
                    self.waiting.popleft(); self._finish(req, "cancelled"); continue
                ids = req.prompt_ids + req.generated          # a preempted request recomputes what it produced
                # headroom: keep one step's growth free per sequence in flight (a unit per group), since each may
                # cross a block boundary on its next step; otherwise the newcomer would be preempted right away and
                # its prefill wasted
                if self._blocks_for(len(ids) + 1) + in_flight * self.pool.max_step_units > self.pool.allocator.num_free:
                    return                                    # wait for running requests to free blocks
                if not self.pool.free_seqs:
                    return                                    # wait for a DeltaNet state slot too
                self.waiting.popleft()
            try:
                req.state = self.model.new_paged_state(self.pool)
                req.state.reserve(len(ids))                   # every prompt block now: chunks can never run out
            except Exception as e:                            # fail this request, not the server
                self._finish(req, "error", f"admission failed: {e}")
                continue
            req.to_prefill, req.prefill_pos = ids, 0
            self.prefilling.append(req)

    def _prefill_step(self) -> None:
        """One packed forward over the next chunks of the prefilling requests (first come, first served), at most
        `prefill_chunk` tokens in total. Requests whose whole prompt is now in their state get their first token."""
        for r in [r for r in self.prefilling if r.cancelled]:   # all of them, not only those the budget reaches:
            self._finish(r, "cancelled")                       # their blocks and slots free up on this step
        chunks, taken, budget = [], [], self.prefill_chunk
        for r in self.prefilling:
            if budget == 0:
                break
            n = min(len(r.to_prefill) - r.prefill_pos, budget)
            chunks.append((torch.tensor(r.to_prefill[r.prefill_pos:r.prefill_pos + n]), r.state))
            taken.append((r, n))
            budget -= n
        if not chunks:
            return
        try:
            logits = self.model.forward_packed(chunks)
        except Exception as e:                                # fail these requests, not the server
            for r, _ in taken:
                self._finish(r, "error", f"prefill failed: {e}")
            return
        self.metrics.inc("prefill_steps_total")
        self.metrics.inc("prefill_tokens_total", sum(n for _, n in taken))
        done = []
        for row, (r, n) in enumerate(taken):
            r.prefill_pos += n
            if r.prefill_pos == len(r.to_prefill):
                done.append((row, r))
        if not done:
            return
        for _, r in done:
            self.prefilling.remove(r)
        tokens = self._sample_rows([r for _, r in done], logits[[row for row, _ in done]])
        for (_, r), token in zip(done, tokens):
            if token is None:
                continue                                      # sampling failed: already finished with an error
            self.running.append(r)
            self._emit(r, token)                              # first token (or, after preemption, the next one)

    def _decode_step(self) -> None:
        active = [r for r in self.running if r.finish_reason is None]
        for r in active:
            if r.cancelled:
                self._finish(r, "cancelled")
        active = [r for r in self.running if r.finish_reason is None]
        # every sequence needs room for one more position; preempt the newest until they fit
        while active and sum(self._blocks_for(r.state.length + 1) - self._blocks_for(r.state.length) for r in active) \
                > self.pool.allocator.num_free:
            victim = active.pop()
            self._preempt(victim)
        if not active:
            return
        logits = self.model.decode_batch([r.generated[-1] for r in active], [r.state for r in active])
        self.metrics.inc("decode_steps_total")
        self.metrics.inc("decode_sequences_total", len(active))
        for r, token in zip(active, self._sample_rows(active, logits)):
            if token is not None:
                self._emit(r, token)

    def _sample_rows(self, reqs: list[Request], logits: torch.Tensor) -> list[int | None]:
        """Next token for each request from its row of logits, in one batched pass (the penalty and the top-k
        selection run once for the whole batch; see sampler.sample_batch). A request whose row cannot be sampled
        is finished with an error (None returned for it); its batch-mates go on unaffected."""
        V = self.tok.vocab_size()
        prev = [r.prompt_ids + r.generated for r in reqs]
        try:
            # on the CPU: with unified memory, copying a batch-8 of 248k logits takes ~0.5 ms, and torch.topk over
            # them takes 1.0 ms there vs 4.5 ms on MPS (docs/bench/raw/profiling.md section 9). Copy, then slice off
            # the padded vocabulary rows: copying the strided slice costs twice as much
            tokens = sample_batch(logits.float().cpu()[:, :V], prev, [r.params for r in reqs],
                                  [r.generator for r in reqs])
        except Exception:                                     # the copy or the batch-wide selection failed, before
            tokens = []                                       # any generator was drawn from: safe to go row by row
            for r, row, p in zip(reqs, logits, prev):
                try:
                    tokens.append(sample(row[:V].float().cpu(), p, r.params, r.generator))
                except Exception as e:
                    tokens.append(e)
        out = []
        for r, t in zip(reqs, tokens):
            if isinstance(t, Exception):
                self._finish(r, "error", f"sampling failed: {t}")
                out.append(None)
            else:
                out.append(t)
        return out

    def _emit(self, req: Request, token: int) -> None:
        now = time.perf_counter()
        if token in self.eng.eos_ids:
            self._finish(req, "stop"); return
        if req.first_token_at is None:
            req.first_token_at = now
            self.metrics.ttft.observe(now - req.arrived)
        else:
            self.metrics.tpot.observe(now - req.last_token_at)
        req.last_token_at = now
        req.generated.append(token)
        self.metrics.inc("generation_tokens_total")
        # Incremental detokenization: decode only a short window instead of the whole output every token.
        # prefix = text of the last emitted tokens, full = that plus the new ones; the difference is the new text.
        # Held back while it ends in U+FFFD (a multi-byte character split across tokens).
        prefix = self.tok.decode(req.generated[req.prefix_offset:req.read_offset])
        full = self.tok.decode(req.generated[req.prefix_offset:])
        if len(full) > len(prefix) and not full.endswith("\ufffd"):
            piece = full[len(prefix):]
            req.out.put(("token", piece))
            req.emitted += piece
            req.prefix_offset, req.read_offset = req.read_offset, len(req.generated)
        if len(req.generated) >= req.max_new_tokens:
            self._finish(req, "length")

    def _preempt(self, req: Request) -> None:
        req.state.free()
        req.state = None
        self.running.remove(req)
        with self.cond:
            self.waiting.appendleft(req)                      # resume first, recomputing prompt + generated
        self.metrics.inc("requests_preempted_total")

    def _finish(self, req: Request, reason: str, error: str | None = None) -> None:
        if req.finish_reason is not None:
            return
        req.finish_reason = reason
        text = self.tok.decode(req.generated) if req.generated else ""
        if len(text) > len(req.emitted):
            req.out.put(("token", text[len(req.emitted):]))
            req.emitted = text
        if req.state is not None:                             # free first: whoever sees "done" sees the blocks back
            req.state.free()
            req.state = None
        if req in self.running:
            self.running.remove(req)
        if req in self.prefilling:
            self.prefilling.remove(req)
        self.active.discard(req)
        req.out.put(("error", error) if error else ("done", reason))
        self.metrics.inc("requests_finished_total")
        if reason in ("stop", "length"):                      # completed requests only; errors/cancels are counted
            self.metrics.e2e.observe(time.perf_counter() - req.arrived)

    def refresh_gauges(self) -> None:
        """Any thread (/metrics calls it at scrape time). The engine loop refreshes the gauges once per iteration,
        after admitting; a request that waits for less than one iteration (behind a ~0.8 s prefill pass, say) would
        otherwise never show up in waiting_requests, the signal an autoscaler watches."""
        with self.cond:
            self._update_gauges()

    def _job_step(self) -> None:
        """Advance the job in progress by one step (a plain function runs whole). A cancelled job is dropped; a
        failed one fails alone, its traceback cleared so the frames' tensors are freed now, not by the next job."""
        fn, fut = self.job
        try:
            if fut.cancelled():
                self.job = None
                return
            if not inspect.isgenerator(fn):
                fn = fn()
                if not inspect.isgenerator(fn):
                    self.job = None
                    _settle(fut, result=fn)
                    return
                self.job = (fn, fut)
            next(fn)
        except StopIteration as done:
            self.job = None
            _settle(fut, result=done.value)
        except Exception as e:                                # the job fails, not the server
            self.job = None
            _settle(fut, error=e.with_traceback(None))
        finally:
            fn = fut = None

    def _update_gauges(self) -> None:
        self.metrics.set("running_requests", len(self.running))
        self.metrics.set("prefilling_requests", len(self.prefilling))
        self.metrics.set("waiting_requests", len(self.waiting))
        self.metrics.set("waiting_jobs", len(self.jobs) + (self.job is not None))
        self.metrics.set("kv_blocks_free", self.pool.allocator.num_free)
        self.metrics.set("kv_bytes_held", (self.pool.allocator.num_blocks - self.pool.allocator.num_free)
                         * self.pool.unit_bytes)
