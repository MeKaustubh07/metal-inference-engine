"""Packed and chunked prefill: several prompts in one forward pass == each prompt alone, and a scheduler that splits
prompts into chunks (interleaved with decode steps) produces the same text as sequential decoding while running
requests keep generating during a long prefill.

usage: test_prefill.py [--target qwen|tiny|aya-int4|aya-int8]   the serving target (tests/serving_targets.py; default
qwen). The Metal section loads Qwen3.5 on the GPU, so it runs with the qwen target only (N/A on the others). On aya
(Tiny Aya on Metal, not exact) served greedy runs follow the near-tie rule; small pools; GPU memory and aborts gated."""
import gc
import sys
import time

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from engine import load_engine
from sampler import SamplingParams
from server.metrics import Metrics
from server.scheduler import Scheduler
from serving_targets import gpu_gates, load_target, na, notes, target_from_argv, unload_target

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}{notes()}")

GREEDY = SamplingParams(temperature=0, repetition_penalty=1.0)
PROMPTS = ["The capital of France is", "def fibonacci(n):", "Hello world",
           "The history of computing is a story of abstraction. Each generation of engineers built tools that hid"]
TARGET = target_from_argv()
T = load_target(TARGET)                                      # the target's one model in memory, reused below
KV, POOL = (32, 32) if T.gpu else (256, 128)                 # KV blocks of the Schedulers and packed_vs_alone's pool:
                                                             # small on the GPU (memory is gated; aya needs <= 18)


def raw(eng, text):                                          # a raw prompt's ids as /v1/completions builds them (BOS
    return eng.tokenizer.encode(text, add_bos=True)          # first, if the model has one), on any engine


def sequential(model, V, ids, n, eos):                       # -> (greedy ids, logits rows: one per id, plus the row
    st = model.new_state(len(ids) + n + 1)                   # that chose an end token if it stopped on one: T.agree's)
    lg = model.forward(torch.tensor(ids), state=st, last_only=True)[0]
    out, rows = [], []
    for _ in range(n):
        rows.append(lg[:V].float().cpu())
        t = int(lg[:V].argmax())
        if t in eos:
            break
        out.append(t)
        lg = model.forward(torch.tensor([t]), state=st)[0]
    return out, rows


def wait_for(cond, what, limit=120):                         # a busy-wait with a deadline: "" or, past it, a note for
    end = time.perf_counter() + limit                        # the check's label (the check then fails)
    while not cond():
        if time.perf_counter() > end:
            return f"; waited {limit} s for {what} in vain"
        time.sleep(0.001)
    return ""


def packed_vs_alone(eng, tol, blocks=128):
    m, V = eng.model, eng.tokenizer.vocab_size()
    ids = [raw(eng, p) for p in PROMPTS]
    pool = m.new_paged_pool(blocks, 16, max_seqs=16)
    alone, alone_st = [], []
    for i, x in enumerate(ids):                              # the long prompt as 10 tokens + the rest
        st = m.new_paged_state(pool)
        if i == 3:
            m.forward(torch.tensor(x[:10]), state=st)
            x = x[10:]
        alone.append(m.forward(torch.tensor(x), state=st, last_only=True)[0][:V].float().cpu())
        alone_st.append(st)
    sts = [m.new_paged_state(pool) for _ in ids]
    m.forward(torch.tensor(ids[3][:10]), state=sts[3])
    lg = m.forward_packed([(torch.tensor(x), st) for x, st in zip(ids[:3], sts[:3])] +
                          [(torch.tensor(ids[3][10:]), sts[3])])[:, :V].float().cpu()
    d = max((lg[i] - alone[i]).abs().max().item() for i in range(4))
    nxt = [int(a.argmax()) for a in alone]
    step = (m.decode_batch(nxt, sts) - m.decode_batch(nxt, alone_st)).abs().max().item()
    check(f"{eng.name} {m.b.name}: 4 prompts packed in one pass (one continuing a half-prefilled state) == each "
          f"alone, max|dlogit| {d:.1e} < {tol}; decoding on from the packed states: max|d| {step:.1e}",
          d < tol and step < tol and [s.length for s in sts] == [s.length for s in alone_st])
    # a pack above 32 rows takes the GEMM path while each prompt alone uses the batched kernels: the result must
    # still not depend on what it was packed with (both keep fp32 activations). The prompts are the first 18 and 25
    # ids of a text long enough for any tokenizer (Qwen3.5's: PROMPTS[3] alone, and with " the details of the layer
    # below."), so the pack has 43 rows whatever the target
    text = PROMPTS[3] + " the details of the layer below."
    base = raw(eng, f"{text} {text}")
    long = [base[:18], base[:25]]
    alone = [m.forward(torch.tensor(x), state=m.new_paged_state(pool), last_only=True)[0][:V].float().cpu()
             for x in long]
    lg = m.forward_packed([(torch.tensor(x), m.new_paged_state(pool)) for x in long])[:, :V].float().cpu()
    d = max((lg[i] - alone[i]).abs().max().item() for i in range(2))
    check(f"{eng.name} {m.b.name}: a {sum(map(len, long))}-token pack (GEMM path) == its {len(long[0])}- and "
          f"{len(long[1])}-token prompts alone (batched kernels), max|dlogit| {d:.1e} < {tol}",
          d < tol and max(map(len, long)) <= 32 < sum(map(len, long)))


def run_scheduler(eng, prompts, n, chunk, kv_blocks=256, max_batch=4, lock_weights=False):
    """Submit all prompts; return texts, the scheduler's counters, the Requests (generated ids, finish reasons), the
    emit order [(request id, tokens)] and the scheduler."""
    s = Scheduler(eng, Metrics(), max_batch=max_batch, kv_blocks=kv_blocks, prefill_chunk=chunk,
                  lock_weights=lock_weights)
    order, emit = [], s._emit
    def logged(req, token):
        emit(req, token); order.append((req.id, len(req.generated)))
    s._emit = logged
    with s.cond:                                             # all queued before the engine thread looks (RLock)
        reqs = [s.submit(raw(eng, p), GREEDY, n) for p in prompts]
    for r in reqs:
        while r.out.get(timeout=600)[0] == "token":
            pass
    s.shutdown(5)
    return [eng.tokenizer.decode(r.generated) for r in reqs], dict(s.metrics.counters), reqs, order, s


# ---------------------------------------------------------------- the target (qwen, tiny: CPU fp32, exact; aya: near-ties)
eng = T.engine
packed_vs_alone(eng, T.logit_tol, POOL)
T.tidy()                                                     # on the GPU: free each section's pools (no-op on the CPU)
V, eos = eng.tokenizer.vocab_size(), set(eng.eos_ids)
refs = [sequential(eng.model, V, raw(eng, p), 12, eos) for p in PROMPTS]
def agrees(reqs):                                            # every served greedy run ended normally, == its reference
    ok = [T.agree(ref, r.generated, r.finish_reason, what=f"prompt {j}")
          for j, (ref, r) in enumerate(zip(refs, reqs))]
    return all(ok) and all(r.finish_reason in ("stop", "length") for r in reqs)
c, got = run_scheduler(eng, PROMPTS, 12, chunk=4, kv_blocks=KV)[1:3]
prompt_tokens = sum(len(raw(eng, p)) for p in PROMPTS)
check(f"{eng.name}: prefill in 4-token chunks ({c['prefill_steps_total']} packed passes for {prompt_tokens} prompt "
      f"tokens) == sequential greedy", agrees(got) and c["prefill_tokens_total"] == prompt_tokens
      and c["prefill_steps_total"] >= prompt_tokens // 4)
T.tidy()
c, got = run_scheduler(eng, PROMPTS, 12, chunk=512, kv_blocks=KV)[1:3]
check(f"{eng.name}: all 4 prompts packed into one prefill pass ({c['prefill_steps_total']} pass) == sequential greedy",
      agrees(got) and c["prefill_steps_total"] == 1)
T.tidy()

# a request that is already generating keeps generating while a long prompt is prefilled chunk by chunk (slow_steps:
# a fast model's 60 steps take ~0.1 s, so a stalled test thread could see the request done before the long prompt
# arrives, and the check would pass without testing anything)
s = Scheduler(eng, Metrics(), max_batch=4, kv_blocks=KV, prefill_chunk=16)
T.slow_steps(s)
order, emit = [], s._emit
def logged(req, token):
    emit(req, token); order.append((req.id, len(req.generated)))
s._emit = logged
short = s.submit(raw(eng, PROMPTS[0]), GREEDY, 60)
short.out.get(timeout=120)                                   # it is decoding now
long_ids = raw(eng, " ".join(["Inference engines turn trained weights into answers."] * 12))
long = s.submit(long_ids, GREEDY, 4)
while long.out.get(timeout=600)[0] == "token":
    pass
first_long = next(i for i, (rid, _) in enumerate(order) if rid == long.id)
short_during = sum(1 for rid, _ in order[:first_long] if rid == short.id)
chunks = -(-len(long_ids) // 16)
while short.out.get(timeout=120)[0] == "token":
    pass
ok = short_during >= chunks - 1                              # on failure the label says how the short request ended
check(f"a running request kept generating while a {len(long_ids)}-token prompt was prefilled in {chunks} chunks "
      f"({short_during} of its tokens came before the long prompt's first token)"
      + ("" if ok else f"; the short request made {len(short.generated)} tokens ({short.finish_reason})"), ok)
s.shutdown(5)
del s, emit; T.tidy()

# a request cancelled while it waits behind a long prompt's chunks is finished at the next prefill step (not after
# the long prompt's remaining passes), and its reserved KV blocks come back
s = Scheduler(eng, Metrics(), max_batch=4, kv_blocks=KV, prefill_chunk=4)
with s.cond:
    first = s.submit(long_ids, GREEDY, 2)
    queued = s.submit(raw(eng, PROMPTS[1]), GREEDY, 2)
stuck = wait_for(lambda: queued in s.prefilling or queued.finish_reason, "the queued request's admission")
steps_at_cancel = s.metrics.counters["prefill_steps_total"]
s.cancel(queued)
stuck += wait_for(lambda: queued.finish_reason, "the cancel to finish it")
late = s.metrics.counters["prefill_steps_total"] - steps_at_cancel
while first.out.get(timeout=600)[0] == "token":
    pass
s.shutdown(5)
check(f"a request cancelled behind a {len(long_ids)}-token prompt's 4-token chunks ends within {late} prefill step(s) "
      f"(the long prompt needed {-(-len(long_ids) // 4)}); all KV blocks and state slots back{stuck}",
      not stuck and queued.finish_reason == "cancelled" and late <= 1
      and s.pool.allocator.num_free == s.pool.allocator.num_blocks and len(s.pool.free_seqs) == 4)
del s; T.tidy()

# requests arriving ~1 ms apart at an idle engine share one prefill pass (the batching window), and they are
# not held back when arrivals stop
s = Scheduler(eng, Metrics(), max_batch=4, kv_blocks=KV, batch_wait_ms=5)
reqs = []
for p in PROMPTS[:3]:
    reqs.append(s.submit(raw(eng, p), GREEDY, 4))
    time.sleep(0.001)
for r in reqs:
    while r.out.get(timeout=120)[0] == "token":
        pass
passes = s.metrics.counters["prefill_steps_total"]
s.shutdown(5)
check(f"3 requests arriving ~1 ms apart at an idle engine share {passes} prefill pass (batching window)", passes == 1)
del s; T.tidy()

# a cancel during the window is not an arrival: the survivor waits one window, not two (200 ms scale for timer slack)
s = Scheduler(eng, Metrics(), max_batch=4, kv_blocks=KV, batch_wait_ms=200)
spans, gather = [], s._gather
def timed_gather():
    t0 = time.perf_counter(); gather(); spans.append(time.perf_counter() - t0)
s._gather = timed_gather
with s.cond:
    a = s.submit(raw(eng, PROMPTS[0]), GREEDY, 2)
    b = s.submit(raw(eng, PROMPTS[1]), GREEDY, 2)
time.sleep(0.02)
s.cancel(b)
while a.out.get(timeout=120)[0] == "token":
    pass
s.shutdown(5)
check(f"a request cancelled inside a 200 ms batching window does not extend it for the others "
      f"(window {spans[0] * 1e3:.0f} ms; 2 windows would be 400)", b.finish_reason == "cancelled" and spans[0] < 0.3)
del s, gather; T.tidy()
bad = []
for v in [float("inf"), float("nan"), 1e13, -1]:
    try:
        Scheduler(eng, Metrics(), batch_wait_ms=v, kv_blocks=KV)
    except ValueError:
        bad.append(v)
check(f"batch_wait_ms inf / nan / 1e13 / -1 rejected at construction (a wait that long would kill the engine "
      f"thread): {len(bad)} of 4", len(bad) == 4)
gpu_gates(T, check)                                          # while the target is loaded (nothing on a CPU target)
freed = ("the GPU target's model is freed and the MPS cache emptied" if T.gpu
         else "the CPU target's model is freed before the Metal section")
del eng, T, refs                                             # the Metal section loads Qwen on the GPU
check(freed, unload_target(TARGET))

# ---------------------------------------------------------------- Metal: packed == alone; hybrid pool + chunks + preemption
if TARGET == "qwen":
    for name, backend, tol in [("qwen3.5-0.8b", "metal", 1e-2), ("qwen3.5-0.8b", "metal-int4", 1e-2)]:
        eng = load_engine(name, backend)
        packed_vs_alone(eng, tol)
        if backend == "metal-int4":
            V, eos = eng.tokenizer.vocab_size(), set(eng.eos_ids)
            prompts = [PROMPTS[1], PROMPTS[3], PROMPTS[2]]  # ~4 / ~19 / ~2 tokens + 30 new: 8+ blocks needed, 6 exist
            want = [eng.tokenizer.decode(sequential(eng.model, V, raw(eng, p), 30, eos)[0]) for p in prompts]
            texts, c, _, _, sch = run_scheduler(eng, prompts, 30, chunk=4, kv_blocks=6, max_batch=3,
                                                lock_weights=True)
            check(f"{name} {backend}: 4-token prefill chunks + a KV pool small enough to preempt "
                  f"({c['requests_preempted_total']} preemptions) == decoding alone, with the weights locked in RAM "
                  f"({sch.weights_locked / 2**20:.0f} MiB); every block and state slot returned",
                  texts == want and c["requests_preempted_total"] >= 1
                  and sch.pool.allocator.num_free == sch.pool.allocator.num_blocks
                  and len(sch.pool.free_seqs) == 3 and sch.weights_locked > 0 and sch.lock_error is None)
        del eng
        gc.collect(); torch.mps.empty_cache()
else:                                                        # these load Qwen3.5 on the GPU
    for backend in ("metal", "metal-int4"):
        na(f"qwen3.5-0.8b {backend}: 4 prompts packed in one pass == each alone (Qwen3.5 on Metal)")
        na(f"qwen3.5-0.8b {backend}: a 43-token pack (GEMM path) == its prompts alone (Qwen3.5 on Metal)")
    na("qwen3.5-0.8b metal-int4: 4-token prefill chunks + a KV pool small enough to preempt == decoding alone, "
       "weights locked in RAM (Qwen3.5 on Metal)")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
