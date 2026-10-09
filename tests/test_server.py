"""Week 15: serving. Batched decode == sequential decode for the hybrid Qwen3.5 model (fp32 CPU, exact greedy), sequences
joining/leaving mid-batch, batched quantized kernels == per-row kernels, Metal batched decode within bf16 noise,
preemption-by-recompute, and the HTTP API: completions, chat, SSE streaming, 429 backpressure, validation,
metrics, cancellation, seeded sampling and graceful drain.

--target tiny serves a small random Cohere2 instead (tests/serving_targets.py; its sliding-window rings wrap over
HTTP), aya-int4 / aya-int8 the real Tiny Aya on Metal, whose batched replies are compared with sequential greedy under
the near-tie rule (Target.agree): the Metal sections (Qwen3.5-0.8B on the GPU) print N/A beside them."""
import asyncio
import gc
import json
import sys
import time

import httpx
import torch

sys.path.insert(0, "src")
from chat import format_chat
from engine import load_engine
from quant import quantize
from sampler import SamplingParams
from server.app import OutputFilter, create_app
from server.metrics import Metrics
from server.scheduler import QueueFull, Request, Scheduler
from serving_targets import (PROMPTS, Served, agree_selfcheck, gpu_gates, load_target, na, notes, target_from_argv,
                             unload_target)

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}{notes()}")

GREEDY = SamplingParams(temperature=0, repetition_penalty=1.0)
HINDI = "नमस्ते दुनिया, यह एक परीक्षण है।"      # multi-byte text; Qwen3.5 has a token per character here, so the UTF-8
                                                # holdback is checked separately by feeding the detokenizer a split emoji


def sequential(model, V, ids, n, eos=()):
    """One sequence alone with a contiguous cache: prefill, then one forward per token. -> (tokens, logits)."""
    st = model.new_state(len(ids) + n + 1)
    lg = model.forward(torch.tensor(ids), state=st, last_only=True)[0]
    toks, logs = [], []
    for _ in range(n):
        logs.append(lg[:V].float().cpu()); t = int(lg[:V].argmax())
        if t in eos:
            break
        toks.append(t)
        lg = model.forward(torch.tensor([t]), state=st)[0]
    return toks, logs


def batched(model, V, prompts, n, join_at):
    """decode_batch over paged states; prompt j is prefilled at step join_at[j] and leaves after n tokens."""
    pool = model.new_paged_pool(32, 16)                 # a few sequences of at most 32 positions: 2 blocks each
    states, toks, logs, done = {}, [[] for _ in prompts], [[] for _ in prompts], set()
    step = 0
    while len(done) < len(prompts):
        for j, ids in enumerate(prompts):
            if join_at[j] == step:
                states[j] = model.new_paged_state(pool)
                lg = model.forward(torch.tensor(ids), state=states[j], last_only=True)[0]
                logs[j].append(lg[:V].float().cpu()); toks[j].append(int(lg[:V].argmax()))
        active = [j for j in states if j not in done]
        for j in active:
            if len(toks[j]) == n:
                states[j].free(); done.add(j)            # leaves the batch; its blocks go back to the pool
        active = [j for j in active if j not in done]
        if active:
            lg = model.decode_batch([toks[j][-1] for j in active], [states[j] for j in active])
            for j, l in zip(active, lg):
                logs[j].append(l[:V].float().cpu()); toks[j].append(int(l[:V].argmax()))
        step += 1
    return toks, logs, pool


# ---------------------------------------------------------------- 1. batched == sequential (exact; aya: near-ties)
TARGET = target_from_argv()
T = load_target(TARGET)                             # qwen (default): Qwen3.5-0.8B; tiny: a random Cohere2 (fp32 CPU)
name, eng, raw = T.name, T.engine, T.raw_ids        # the one model loaded, reused by the HTTP suites below
V, N = eng.tokenizer.vocab_size(), 10
ids = [raw(p) for p in PROMPTS[:3]]                 # as the server encodes a raw prompt (BOS first, if any)
ref = [sequential(eng.model, V, x, N) for x in ids]
got, logs, pool = batched(eng.model, V, ids, N, join_at=[0, 0, 3])    # third joins mid-flight
check(f"{name}: batched greedy == sequential for 3 prompts (one joins at step 3)",
      all([T.agree(ref[j], got[j], what=f"prompt {j}") for j in range(3)]))
diff = max((logs[j][i] - ref[j][1][i]).abs().max().item() for j in range(3) for i in range(N)
           if got[j][:i] == ref[j][0][:i])                       # steps after a divergence follow other contexts
tol = f"{T.logit_tol:.0e}".replace("e-0", "e-")                     # qwen, tiny 2e-3; aya 3e-2
check(f"{name}: batched logits == sequential logits (max|diff| {diff:.1e} < {tol})", diff < T.logit_tol)
check(f"{name}: every KV block and DeltaNet state slot returned to the pool after sequences leave",
      pool.allocator.num_free == pool.allocator.num_blocks and len(pool.free_seqs) == 8)
del pool                                            # a GPU target holds no pool it no longer uses

# ---------------------------------------------------------------- 2a. output filter (reasoning split, stop strings)
f = OutputFilter(True, [])                          # the server's filter on fixed text: any target runs these
got = f.feed("Let me think") + f.feed(" about it.</think>\n\nThe answer") + f.feed(" is 4.") + f.flush()
check("thinking mode: text before </think> is reasoning, after it content",
         got == [("reasoning", "Let me think"), ("reasoning", " about it."), ("content", "The answer"),
                 ("content", " is 4.")])
f = OutputFilter(True, [])
got = f.feed("ok</think>") + f.feed("\n") + f.feed("\nAnswer\n") + f.flush()
check("newlines after </think> are dropped even when they arrive in later pieces",
         got == [("reasoning", "ok"), ("content", "Answer\n")])
f = OutputFilter(True, [])
check("thinking mode cut off by max_tokens: everything is reasoning, no content",
         f.feed("still thinking") + f.flush() == [("reasoning", "still thinking")])
f = OutputFilter(False, ["END", "\n\n"])
got = f.feed("abc E") + f.feed("N") + f.feed("D tail")
check("stop string split across pieces: content cut before it, nothing after", "".join(t for _, t in got) == "abc "
      and f.stopped)
f = OutputFilter(False, ["END"])
check("held-back tail that never becomes a stop string is flushed",
      "".join(t for _, t in f.feed("abcEN") + f.flush()) == "abcEN" and not f.stopped)
check("the near-tie rule (D9) on made-up logits: equal, cancelled early, a near-tie and an end-token near-tie accepted; "
      "a clear divergence or a wrong length refused; exact targets demand equality", agree_selfcheck())

# ---------------------------------------------------------------- 2. HTTP API over the scheduler (the target)
V, tok, EOS = eng.tokenizer.vocab_size(), eng.tokenizer, set(eng.eos_ids)
API_LEN = 256 if T.chat_len() + 64 <= 256 else 512            # api_suite's max_model_len (aya: a chat is 377 ids)
_refs = {}


def reference(prompt, n, chat=False):
    """Greedy alone, once per (prompt, n, chat), from the ids the server builds -> (tokens, logits rows if inexact)."""
    if (prompt, n, chat) not in _refs:
        toks, logs = sequential(eng.model, V, tok.encode(prompt) if chat else raw(prompt), n, EOS)
        _refs[prompt, n, chat] = toks, ([] if T.exact else logs)
    return _refs[prompt, n, chat]


def ref_text(prompt, n, chat=False):
    return tok.decode(reference(prompt, n, chat)[0])


def same(text, prompt, n, served, chat=False, what=""):
    """A served greedy reply == greedy alone: its text (exact targets), else its ids under the near-tie rule (D9)."""
    if T.exact:
        return text == ref_text(prompt, n, chat)
    r = served.last(tok.encode(prompt) if chat else raw(prompt))           # the latest request for that prompt
    return text == tok.decode(r.generated) and T.agree(reference(prompt, n, chat), r.generated, r.finish_reason, what)


def words(n):
    """'word ' repeated to about n tokens (Qwen3.5: 'word ' * n, one token each; tiny: three each)."""
    return "word " * (n // (len(tok.encode("word word ")) - len(tok.encode("word "))))


def sse(body: str):
    lines = [l[6:] for l in body.split("\n") if l.startswith("data: ")]
    return [json.loads(l) for l in lines if l != "[DONE]"], lines[-1] if lines else None


async def api_suite():
    app = create_app(eng, max_batch=4, max_waiting=8, kv_blocks=128, max_model_len=API_LEN, prefill_chunk=T.prefill_chunk)
    sched, served = app.state.scheduler, Served(app.state.scheduler)   # served: the ids behind each reply
    T.slow_steps(sched)                                           # streams stay in flight (cancel, shared steps)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300) as c:
        r1, r2 = await c.get("/health"), await c.get("/ready")
        check("/health and /ready return 200", r1.status_code == 200 and r2.status_code == 200)

        lens, greedy = [8, 12, 16, 10], {"temperature": 0, "repetition_penalty": 1.0}
        rs = await asyncio.gather(*[c.post("/v1/completions", json={"prompt": p, "max_tokens": n, **greedy})
                                    for p, n in zip(PROMPTS, lens)])
        check("4 concurrent completions (continuous batching) == sequential greedy text",
              all([same(rs[j].json()["choices"][0]["text"], PROMPTS[j], lens[j], served, what=f"prompt {j}")
                   for j in range(4)]))
        c_ = sched.metrics.counters
        mean_batch = c_["decode_sequences_total"] / max(1, c_["decode_steps_total"])
        check(f"they really shared decode steps (mean decode batch {mean_batch:.2f} > 1.5)", mean_batch > 1.5)
        u = rs[1].json()["usage"]
        check("usage counts prompt and completion tokens",
              u["prompt_tokens"] == len(raw(PROMPTS[1])) and u["completion_tokens"] == 12
              and u["total_tokens"] == u["prompt_tokens"] + 12)
        check("finish_reason is 'length' at max_tokens", rs[2].json()["choices"][0]["finish_reason"] == "length")

        seen, submit = [], sched.submit                        # what an empty request samples with: the model's own
        sched.submit = lambda ids, params, *a, **k: (seen.append(params), submit(ids, params, *a, **k))[1]
        r = await c.post("/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 2})
        sched.submit = submit
        check(f"a request that sets no sampling runs with the model's registry values ({eng.sampling})",
              r.status_code == 200 and seen == [SamplingParams(**eng.sampling)])

        r = await c.post("/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 12, "stream": True, **greedy})
        chunks, last = sse(r.text)
        streamed = "".join(ch["choices"][0]["text"] for ch in chunks)
        check("SSE stream: text/event-stream, chunks concatenate to the non-stream text, ends with [DONE]",
              r.headers["content-type"].startswith("text/event-stream") and same(streamed, PROMPTS[0], 12, served)
              and last == "[DONE]" and chunks[-1]["choices"][0]["finish_reason"] == "length")

        msgs = [{"role": "user", "content": "What is 2+2? Answer with one number."}]
        r = await c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 16, **greedy})
        body = r.json()
        prompt, msg = format_chat(msgs, style=eng.chat_style), body["choices"][0]["message"]
        check("chat completion == sequential greedy on the chat-formatted prompt; role assistant",
              msg.get("role") == "assistant" and len(msg) == 2 and same(msg.get("content"), prompt, 16, served, chat=True)
              and body["usage"]["prompt_tokens"] == len(tok.encode(prompt)))
        r = await c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 16, "stream": True, **greedy})
        chunks, last = sse(r.text)
        check("chat SSE: first delta carries role 'assistant'; deltas concatenate to the non-stream message",
              chunks[0]["choices"][0]["delta"].get("role") == "assistant"
              and "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
              == body["choices"][0]["message"]["content"] and last == "[DONE]")

        rn = await c.post("/v1/completions", json={"prompt": HINDI, "max_tokens": 16, **greedy})
        rs_ = await c.post("/v1/completions", json={"prompt": HINDI, "max_tokens": 16, "stream": True, **greedy})
        streamed = "".join(ch["choices"][0]["text"] for ch in sse(rs_.text)[0])
        ref = ref_text(HINDI, 16)                           # a trained model writes whole characters; a random one
        check(f"Hindi (multi-byte) stream == non-stream == sequential{', no U+FFFD' if T.real else ''}: {ref!r}",
              streamed == rn.json()["choices"][0]["text"] and same(streamed, HINDI, 16, served)
              and ("\ufffd" not in streamed or not T.real))
        if not T.real:                                      # stray bytes, U+FFFD in its own decode too
            na("Hindi reply has no U+FFFD (whole characters: a trained model's)")

        ref = ref_text(PROMPTS[1], 20)
        stop, want = ref[5:8], ref[: ref.find(ref[5:8])]
        rn = await c.post("/v1/completions", json={"prompt": PROMPTS[1], "max_tokens": 20, "stop": [stop], **greedy})
        rs_ = await c.post("/v1/completions", json={"prompt": PROMPTS[1], "max_tokens": 20, "stop": stop,
                                                    "stream": True, **greedy})
        chunks, _ = sse(rs_.text)
        check(f"stop string {stop!r}: text cut before it, finish_reason 'stop', same when streaming",
              rn.json()["choices"][0]["text"] == want and rn.json()["choices"][0]["finish_reason"] == "stop"
              and "".join(ch["choices"][0]["text"] for ch in chunks) == want
              and chunks[-1]["choices"][0]["finish_reason"] == "stop")

        r = await c.post("/v1/completions", json={"prompt": PROMPTS[0], "max_completion_tokens": 5, **greedy})
        check("max_completion_tokens is honoured", r.json()["usage"]["completion_tokens"] == 5)
        tiny, normal = await asyncio.gather(
            c.post("/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 8, "temperature": 1e-40}),
            c.post("/v1/completions", json={"prompt": PROMPTS[3], "max_tokens": 8, **greedy}))
        check("temperature 1e-40 behaves as greedy and does not break the request batched with it",
              tiny.status_code == normal.status_code == 200
              and same(tiny.json()["choices"][0]["text"], PROMPTS[0], 8, served, what="1e-40")
              and same(normal.json()["choices"][0]["text"], PROMPTS[3], 8, served, what="greedy"))

        bad = [await c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0}),
               await c.post("/v1/completions", json={"prompt": "hi", "temperature": -1}),
               await c.post("/v1/chat/completions", json={"messages": [{"role": "robot", "content": "x"}]}),
               await c.post("/v1/chat/completions", json={"messages": []})]
        bad.append(await c.post("/v1/completions", json={"prompt": "hi", "n": 2}))
        check("invalid parameters (incl. n > 1) -> 422", all(b.status_code == 422 for b in bad))
        codes = [(await c.post("/v1/chat/completions", json={"messages": m})).status_code for m in [
            [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
            [{"role": "system", "content": "only a system prompt"}]]]
        codes.append((await c.post("/v1/completions", content=b'{"prompt": "bad \\ud800 text"}',    # valid JSON escape
                                   headers={"content-type": "application/json"})).status_code)
        check(f"system message not first / no user message -> 400; lone surrogate -> 422, not a crash ({codes})",
              codes == [400, 400, 422])
        r = await c.post("/v1/completions", json={"prompt": words(API_LEN - 6), "max_tokens": 50})
        check("prompt + max_tokens over max_model_len -> 400", r.status_code == 400)   # the prompt fits; 50 more do not
        room = API_LEN - len(raw(words(API_LEN - 16)))              # the same request with no max_tokens: the
        r = await c.post("/v1/completions", json={"prompt": words(API_LEN - 16), "temperature": 0})   # default is cut
        made = r.json()["usage"]["completion_tokens"] if r.status_code == 200 else None    # to the room left
        check(f"no max_tokens and a prompt that leaves {room} tokens of room: 200, at most {room} made ({made})",
              0 < room < 128 and r.status_code == 200 and 0 < made <= room)

        seeded = {"prompt": PROMPTS[3], "max_tokens": 12, "temperature": 0.9, "seed": 1234}
        a, b = await asyncio.gather(c.post("/v1/completions", json=seeded), c.post("/v1/completions", json=seeded))
        check("same seed -> same sampled text, even when the two requests are batched together",
              a.json()["choices"][0]["text"] == b.json()["choices"][0]["text"])

        m = (await c.get("/metrics")).text
        check("/metrics exposes counters, gauges and latency histograms",
              all(k in m for k in ["engine_requests_total", "engine_generation_tokens_total", "engine_kv_blocks_free",
                                   "engine_time_to_first_token_seconds_bucket", "engine_time_per_output_token_seconds_count",
                                   "engine_request_latency_seconds_sum"]))

    # cancellation: tokens stop, blocks come back
    req = sched.submit(raw(PROMPTS[0]), GREEDY, 200)
    req.out.get(timeout=120)
    sched.cancel(req)
    while (item := req.out.get(timeout=120))[0] == "token":
        pass
    time.sleep(0.2)
    check("cancel stops generation early and frees its KV blocks and state slot",
          item[1] == "cancelled" and len(req.generated) < 200
          and sched.pool.allocator.num_free == sched.pool.allocator.num_blocks and len(sched.pool.free_seqs) == 4)

    # graceful drain: in-flight request finishes, new ones are refused
    req = sched.submit(raw(PROMPTS[1]), GREEDY, 10)
    sched.shutdown(drain_timeout=120)
    items = [req.out.get() for _ in range(req.out.qsize())]  # the engine thread has stopped
    try:
        sched.submit(raw("x"), GREEDY, 4); refused = False
    except QueueFull:
        refused = True
    check("graceful shutdown drains the in-flight request, then refuses new ones and reports not-ready",
          items[-1] == ("done", "length") and len(req.generated) == 10 and refused and not sched.ready())

    # the server's incremental detokenizer holds back a character split across tokens (two emoji, 3 + 2 pieces each)
    ids = tok.encode(" Hi 🫠 é中文🫠")
    r = Request("utf8", [], GREEDY, 100)
    for t in ids:
        sched._emit(r, t)
    pieces = [r.out.get()[1] for _ in range(r.out.qsize())]
    check(f"UTF-8 holdback: {len(ids)} tokens -> {len(pieces)} pieces {pieces}, none with U+FFFD, joined == decode(ids)",
          "".join(pieces) == tok.decode(ids) and not any("\ufffd" in p for p in pieces) and len(pieces) < len(ids))


async def backpressure_suite():
    app = create_app(eng, max_batch=1, max_waiting=1, kv_blocks=64, max_model_len=256, prefill_chunk=T.prefill_chunk)
    T.slow_steps(app.state.scheduler)                        # the running request still holds the slot when others come
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300) as c:
        rs = await asyncio.gather(*[c.post("/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 24,
                                                                     "temperature": 0}) for _ in range(6)])
        codes = [r.status_code for r in rs]
        rejected = [r for r in rs if r.status_code == 429]
        check(f"bounded queue: over-capacity requests get 429 + Retry-After (codes {codes})",
              len(rejected) >= 1 and all(c in (200, 429) for c in codes) and all("retry-after" in r.headers for r in rejected))
        m = (await c.get("/metrics")).text
        check("rejections are counted in metrics", f"engine_requests_rejected_total {len(rejected)}" in m)
    sched = app.state.scheduler
    running = sched.submit(raw(PROMPTS[2]), GREEDY, 60)
    running.out.get(timeout=120)                             # it holds the only batch slot
    queued = sched.submit(raw(PROMPTS[0]), GREEDY, 8)        # fills the only queue slot
    sched.cancel(queued)
    try:
        sched.submit(raw(PROMPTS[1]), GREEDY, 8); room = True
    except QueueFull:
        room = False
    check("cancelling a queued request frees its queue slot at once (no 429 for the next client)",
          room and queued.finish_reason == "cancelled")
    sched.shutdown(120)


async def preemption_suite():
    # 3 sequences x (prompt + 29 positions: the 30th token is never fed back) need 3 + 3 + 2 blocks of 16 = 8 > 6, so
    # the pool runs dry mid-generation; units (4 a block): tiny 9 + 9 + 8 > 24; aya 32 > 20: 2 blocks fewer, dry early
    prompts = [PROMPTS[0], PROMPTS[1], PROMPTS[3]]
    probe = eng.model.new_paged_pool(1, 16, max_seqs=1, max_chunk=T.prefill_chunk)   # one block of every layer
    need = sum(probe.blocks_for(len(raw(p)) + 29) for p in prompts)
    app = create_app(eng, max_batch=4, max_waiting=8, kv_blocks=(need - 2) // probe.allocator.num_blocks - 2 * (not T.exact),
                     max_model_len=96, prefill_chunk=T.prefill_chunk)
    served = Served(app.state.scheduler)
    del probe
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300) as c:
        rs = await asyncio.gather(*[c.post("/v1/completions", json={"prompt": p, "max_tokens": 30, "temperature": 0,
                                                                     "repetition_penalty": 1.0}) for p in prompts])
        m = (await c.get("/metrics")).text
    pre = [l for l in m.splitlines() if l.startswith("engine_requests_preempted_total")]
    n_pre = int(float(pre[0].split()[-1])) if pre else 0
    check(f"KV pool exhaustion preempts ({n_pre} preemptions) and recomputes: outputs still == sequential greedy",
          n_pre >= 1 and all([same(r.json()["choices"][0]["text"], p, 30, served, what=f"prompt {PROMPTS.index(p)}")
                              for r, p in zip(rs, prompts)]))
    pool = app.state.scheduler.pool
    check("all KV blocks and state slots free after preempted requests finish",
          pool.allocator.num_free == pool.allocator.num_blocks and len(pool.free_seqs) == 4)
    app.state.scheduler.shutdown(5)


def uvicorn_suite():
    """The real server over a real socket: incremental SSE, disconnect handling, lifespan shutdown."""
    import logging
    import socket
    import threading
    import uvicorn
    logged = []                                                   # the server's JSON request log lines
    grab = logging.Handler(); grab.emit = lambda rec: logged.append(rec.getMessage())
    logging.getLogger("engine.server").addHandler(grab); logging.getLogger("engine.server").setLevel(logging.INFO)
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    app = create_app(eng, max_batch=2, max_waiting=4, kv_blocks=64, max_model_len=512, prefill_chunk=T.prefill_chunk)
    sched = app.state.scheduler
    T.slow_steps(sched)                                           # tokens are still coming when the client gives up
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           timeout_graceful_shutdown=5))
    th = threading.Thread(target=server.run, daemon=True); th.start()
    deadline = time.perf_counter() + 60                           # a server that never starts fails below
    while not server.started and time.perf_counter() < deadline:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    t0, stamps = time.perf_counter(), []
    body = {"prompt": PROMPTS[2], "max_tokens": 24 if T.exact else 64, "temperature": 0, "repetition_penalty": 1.0,
            "stream": True}                                       # aya makes 24 tokens in ~1 s: 64 keep qwen's margin
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith("data: {"):
                stamps.append(time.perf_counter() - t0)
    check(f"real uvicorn: SSE chunks arrive as they are generated (first {stamps[0]:.2f}s, last {stamps[-1]:.2f}s)",
          len(stamps) >= 10 and stamps[0] < 0.5 * stamps[-1])
    before = sched.metrics.counters["generation_tokens_total"]
    try:
        httpx.post(f"{url}/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 400, "temperature": 0},
                   timeout=1.5)
    except httpx.TimeoutException:
        pass
    deadline = time.perf_counter() + 15
    while sched.active and time.perf_counter() < deadline:
        time.sleep(0.1)
    made = sched.metrics.counters["generation_tokens_total"] - before
    check(f"real uvicorn: a non-streaming request whose client gave up is cancelled ({made} of 400 tokens made), "
          f"blocks and state slot freed", not sched.active and made < 400
          and sched.pool.allocator.num_free == sched.pool.allocator.num_blocks and len(sched.pool.free_seqs) == 2)
    last = json.loads([m for m in logged if '"request_finished"' in m][-1])
    check(f"its request_finished log line says finish_reason {last['finish_reason']!r} (the engine stops it one step "
          f"after the handler logs)", last["finish_reason"] == "cancelled" and last["completion_tokens"] < 400)
    # a stop string also ends the request through cancel(), but the request finished normally: the log line must say
    # what the client was told
    ref = ref_text(PROMPTS[1], 20)                                # api_suite's, cached: no model work beside the engine
    r = httpx.post(f"{url}/v1/completions", json={"prompt": PROMPTS[1], "max_tokens": 20, "stop": [ref[5:8]],
                                                  "temperature": 0, "repetition_penalty": 1.0}, timeout=120)
    last = json.loads([m for m in logged if '"request_finished"' in m][-1])
    check(f"a stop-string completion is logged with finish_reason {last['finish_reason']!r}, what the client got "
          f"({r.json()['choices'][0]['finish_reason']!r})",
          r.json()["choices"][0]["finish_reason"] == "stop" and last["finish_reason"] == "stop")

    # /metrics reports the queue as it is now, even while the engine is in the middle of a step
    deadline = time.perf_counter() + 60                           # the stop-string request above ends one step later
    while sched.active and time.perf_counter() < deadline:
        time.sleep(0.01)
    entered, gate, orig = threading.Event(), threading.Event(), sched._decode_step
    def held():
        entered.set(); gate.wait(30); orig()
    sched._decode_step = held
    first = sched.submit(raw(PROMPTS[0]), GREEDY, 3)
    entered.wait(30)                                              # first is running; the engine is held mid-step
    second = sched.submit(raw(PROMPTS[1]), GREEDY, 3)
    scraped = {l.split()[0]: float(l.split()[1]) for l in httpx.get(f"{url}/metrics").text.splitlines()
               if l.startswith(("engine_waiting_requests ", "engine_running_requests "))}
    sched._decode_step = orig; gate.set()
    for r in (first, second):
        while r.out.get(timeout=120)[0] == "token":
            pass
    check(f"/metrics shows a request queued behind a step still in progress (waiting "
          f"{scraped['engine_waiting_requests']:.0f}, running {scraped['engine_running_requests']:.0f})",
          scraped["engine_waiting_requests"] == 1 and scraped["engine_running_requests"] == 1)
    server.should_exit = True
    th.join(30)
    check("real uvicorn: shutdown runs the lifespan (engine thread stopped, not ready)",
          not th.is_alive() and not sched.thread.is_alive() and not sched.ready())


for suite in (api_suite, backpressure_suite, preemption_suite):
    asyncio.run(suite()); T.tidy()            # tidy: a GPU target frees each suite's apps, schedulers and pools
uvicorn_suite()
gpu_gates(T, check)                            # GPU targets: peak memory and Metal aborts, while the model is loaded
freed = ("the GPU target's model is freed and the MPS cache emptied" if T.gpu else
         "the CPU target's model is freed before the Metal sections")
del eng, T, raw                                # the Metal sections load Qwen on the GPU: free the target's model first
check(freed, unload_target(TARGET))
if TARGET != "qwen":                           # the Metal sections load Qwen3.5-0.8B on the GPU: beside its target only
    for s in ("bf16", "int8", "int4"):
        na(f"batched {s} matvec (M=2,3,8,13,32; with/without bias and residual) == per-row kernel == dense")
    for b in ("metal", "metal-int4"):
        na(f"qwen3.5-0.8b {b}: decode_batch (B up to 4, one joins late) greedy == single-sequence greedy")
    na("scheduler over a HybridPool on Metal INT4: outputs == decoding alone, all KV blocks and state slots returned")
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

# ---------------------------------------------------------------- 3. Metal: batched quantized kernels, batched decode
from backend.metal import MetalBackend
mb = MetalBackend("int4")
torch.manual_seed(0)
for scheme in ["bf16", "int8", "int4"]:
    w0 = torch.randn(383, 1056, dtype=torch.bfloat16) * 0.05  # odd N (last SIMD group has one row), K % 128 != 0
    w = w0.to("mps") if scheme == "bf16" else quantize(w0, scheme).to("mps")
    dense_w = w.float() if scheme == "bf16" else w.dequantize().float()
    bias = torch.randn(383, dtype=torch.bfloat16, device="mps")
    worst = 0.0
    for M in [2, 3, 8, 13, 32]:                              # 13 and 32 run as groups of 8
        x, res = torch.randn(M, 1056, device="mps"), torch.randn(M, 383, device="mps")
        for bb, rr in [(None, None), (bias, None), (None, res), (bias, res)]:
            yb = mb.linear(x, w, bb, rr)
            rows = torch.cat([mb.linear(x[i:i + 1], w, bb, None if rr is None else rr[i:i + 1]) for i in range(M)])
            dense = x @ dense_w.T + (0 if bb is None else bb.float()) + (0 if rr is None else rr)
            worst = max(worst, (yb - rows).abs().max().item(), (yb - dense).abs().max().item())
    check(f"batched {scheme} matvec (M=2,3,8,13,32; with/without bias and residual) == per-row kernel == dense "
          f"(worst {worst:.1e})", worst < 1e-3)

for name, backend in [("qwen3.5-0.8b", "metal"), ("qwen3.5-0.8b", "metal-int4")]:
    # batched decode (pooled DeltaNet state, paged attention, batched matvec kernels) must give each request the
    # same output as decoding it alone: bit-identical for bf16; INT4 dequantizes in a different order (rounding)
    eng = load_engine(name, backend)
    V, N = eng.tokenizer.vocab_size(), 12
    ids = [eng.tokenizer.encode(p) for p in PROMPTS]
    ref = [sequential(eng.model, V, x, N) for x in ids]
    got, logs, pool = batched(eng.model, V, ids, N, join_at=[0, 0, 0, 2])
    worst = max((a - b).abs().max().item() for j in range(4) for a, b in zip(logs[j], ref[j][1]))
    check(f"{name} {backend}: decode_batch (B up to 4, one joins late) greedy == single-sequence greedy, "
          f"max|dlogit| {worst:.1e} < 1e-2", all(got[j] == ref[j][0] for j in range(4)) and worst < 1e-2)
    if name == "qwen3.5-0.8b" and backend == "metal-int4":
        # the scheduler over a HybridPool, with a KV pool small enough to force preemption: every request still
        # matches decoding it alone, and every KV block and DeltaNet state slot comes back
        sched = Scheduler(eng, Metrics(), max_batch=3, kv_blocks=7, max_model_len=96)
        reqs = [sched.submit(x, GREEDY, 30) for x in ids[:3]]
        texts = []
        for r in reqs:
            while (item := r.out.get(timeout=300))[0] == "token":
                pass
            texts.append(eng.tokenizer.decode(r.generated))
        want = [eng.tokenizer.decode(sequential(eng.model, V, x, 30, set(eng.eos_ids))[0]) for x in ids[:3]]
        c_ = sched.metrics.counters
        check(f"scheduler over a HybridPool on Metal INT4: {c_['requests_preempted_total']} preemptions, outputs == "
              f"decoding alone, all KV blocks and state slots returned",
              c_["requests_preempted_total"] >= 1 and texts == want
              and sched.pool.allocator.num_free == sched.pool.allocator.num_blocks and len(sched.pool.free_seqs) == 3)
        sched.shutdown(5)
    del eng; gc.collect(); torch.mps.empty_cache()

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
