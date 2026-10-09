"""Order-invariant decisions (src/decision.py, POST /v1/decide): a forked state is independent of its original,
scoring options from forks == scoring each option alone, shuffling the options only shuffles their scores and
embeddings (exactly), options are complete answers, the context is prefilled in chunks, the server runs decisions
stepwise between decode steps and drops cancelled ones, and the choice / boolean / score modes and the endpoint
behave.

--target qwen (default), tiny, aya-int4 or aya-int8 (tests/serving_targets.py): on a random model the checks of
meaning (Paris, 7 + 3, review scores, yes / no answers) print N/A, and so does the Metal section (it loads Qwen on the
GPU; N/A beside Tiny Aya too). On aya-int4 / aya-int8 they are reported, not gated, except Paris and decision 0, and
the tolerances are the target's (D10)."""
import asyncio
import gc
import math
import sys
import threading
import time

import httpx
import torch

sys.path.insert(0, "src")
from chat import format_chat
from decision import boundary_mask, prepare, prompt_ids, score, score_options, score_steps
from engine import load_engine
from sampler import SamplingParams
from server.app import create_app
from server.metrics import Metrics
from server.scheduler import QueueFull, Scheduler
from serving_targets import gpu_gates, load_target, na, notes, report, target_from_argv, unload_target
from state import ContiguousKVCache

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}{notes()}")

TARGET = target_from_argv()
T = load_target(TARGET)                                # the engine every section below serves (CPU fp32, or Metal)
meaning = check if T.gate_meaning else report          # checks of meaning: gated, or reported on a quantized model
TOL = T.logprob_tol                                    # scores vs references; printed as written (qwen 2e-4)
TOL_TEXT = f"{TOL:.0e}".replace("e-0", "e-")
QUESTION = "What is the capital of France? Answer with the city name."
OPTIONS = ["Paris", "Lyon", "Marseille is the capital", "Nice"]        # single- and multi-token options
PERM = [2, 0, 3, 1]


def alone(m, ctx, o, V, boundary=None):
    """log P(o | ctx) from one pass over ctx + o with a fresh state: the reference. With boundary (a boundary_mask),
    + log P(a boundary token follows o), from the last row. The head runs only on the rows from the context's last
    token on (the ones that predict o, and the one after it), not on the context's: a 377-token Tiny Aya chat x 262k
    ids would be 0.4 GB of logits. That is at least 2 rows, and those rows get exactly what a head over every row gave
    them: on the CPU any 2 or more rows do (a single row takes BLAS's vector path, which rounds differently), and on
    Metal Qwen's 25-token context makes them the last of the 8-row groups a 26-30-row head ran in."""
    h = m.packed_hidden([(torch.tensor(ctx + o), m.new_state(len(ctx) + len(o)))])[len(ctx) - 1:]
    lg = m.head(h)[:, :V].float().cpu()
    lp = lg - torch.logsumexp(lg, 1, keepdim=True)
    s = sum(lp[j, o[j]].item() for j in range(len(o)))
    return s if boundary is None else s + torch.logsumexp(lp[-1].masked_fill(~boundary, float("-inf")), 0).item()


def rewind_checks(m, ctx, ids, V):
    """Options one at a time (a memory budget of one byte): each scored on the context's own state, which is rewound
    after it, with no fork. -> (scores == each option from its own fork, bit-exact; the context state afterwards is
    exactly what its prefill left). The reference scores one option per call with room to fork, today's path."""
    real_fork, real_new, made = ContiguousKVCache.fork, m.new_state, []
    def refuse(self):
        raise AssertionError("forked")
    ContiguousKVCache.fork = refuse
    m.new_state = lambda *a, **k: made.append(real_new(*a, **k)) or made[-1]
    try:
        got = score_options(m, ctx, ids, V, embeddings=True, budget=1)
    except AssertionError:
        got = None
    finally:
        ContiguousKVCache.fork = real_fork
        del m.new_state
    ref = [score_options(m, ctx, [o], V, embeddings=True, budget=1 << 40)[0] for o in ids]
    same = got is not None and all(a.logprob == b.logprob and torch.equal(a.embedding, b.embedding)
                                   for a, b in zip(got, ref))
    st, n = made[0], len(ctx)
    fresh = real_new(n + max(map(len, ids)))
    for i in range(0, n, 512):                                       # the context prefill, as decision does it
        m.forward(torch.tensor(ctx[i:i + 512]), state=fresh, last_only=True)
    intact = (st.length == n and torch.equal(st.kv.k[:, :n], fresh.kv.k[:, :n])
              and torch.equal(st.kv.v[:, :n], fresh.kv.v[:, :n]) and torch.equal(st.S, fresh.S)
              and torch.equal(st.conv_tail, fresh.conv_tail))
    return same, intact


def core(eng, tol):
    m, tok, V = eng.model, eng.tokenizer, eng.tokenizer.vocab_size()
    name = f"{eng.name} {m.b.name}"
    ctx = prompt_ids(tok, "", QUESTION, True, eng.chat_style)
    ids = [tok.encode(o) for o in OPTIONS]

    # a fork continues exactly like the original would, and feeding the fork leaves the original untouched (room for
    # the longest option: 5 tokens on Qwen, 10 on the tiny target's small vocabulary; attention reads views of the
    # cache, so its capacity never changes a value)
    room = len(ctx) + max(map(len, ids))
    st = m.new_state(room); m.forward(torch.tensor(ctx), state=st, last_only=True)
    fresh = m.new_state(room); m.forward(torch.tensor(ctx), state=fresh, last_only=True)
    f = st.fork()
    via_fork = m.forward(torch.tensor(ids[0]), state=f, last_only=True)
    m.forward(torch.tensor(ids[2]), state=st.fork())                    # another fork, fed something else
    orig = m.forward(torch.tensor(ids[0]), state=st, last_only=True)
    ref = m.forward(torch.tensor(ids[0]), state=fresh, last_only=True)
    check(f"{name}: a fork continues like the original ({(via_fork - ref).abs().max().item():.1e}), and feeding a "
          f"fork leaves the original exactly untouched ({(orig - ref).abs().max().item():.1e})",
          torch.equal(orig.cpu(), ref.cpu()) and (via_fork - ref).abs().max().item() < tol and f.length == st.length)

    a = score_options(m, ctx, ids, V, embeddings=True)
    want = [alone(m, ctx, o, V) for o in ids]
    d = max(abs(s.logprob - w) for s, w in zip(a, want))
    check(f"{name}: forked scores == each option scored alone as context + option, max |dlogprob| {d:.1e} < {tol} "
          f"({[round(s.logprob, 3) for s in a]})", d < tol)

    b = score_options(m, ctx, [ids[i] for i in PERM], V, embeddings=True)
    same = all(b[k].logprob == a[i].logprob and torch.equal(b[k].embedding, a[i].embedding) for k, i in enumerate(PERM))
    check(f"{name}: shuffling the options only shuffles their scores and embeddings (bit-exact)", same)

    c = score_options(m, ctx, ids, V, budget=1)                          # one option per packed pass
    d = max(abs(x.logprob - y.logprob) for x, y in zip(a, c))
    check(f"{name}: options in groups of one (memory budget) == all in one pass, max |d| {d:.1e} < {tol}", d < tol)
    same, intact = rewind_checks(m, ctx, ids, V)
    check(f"{name}: in groups of one, each option is scored on the context's own state with no fork, == from its own "
          f"fork (bit-exact)", same)
    check(f"{name}: afterwards the context's state is exactly what its prefill left (length, K/V, recurrent state)",
          intact)
    return a


# ---------------------------------------------------------------- tiny Cohere2, CPU: decide without forks
# window 8; contexts below, at, one past and well past it; options of 6, 1 and 3 tokens, the longest first in the
# canonical order, so a shorter option reads where a longer one wrote; fp32 and bf16 KV
import tempfile

from tiny_cohere2 import build

with tempfile.TemporaryDirectory() as tmp:
    _, make = build(tmp, window=8)
    opts = [[3, 11, 12, 13, 14, 15], [7], [9, 9, 9]]
    for kv in (torch.float32, torch.bfloat16):
        tm, runs = make(kv_dtype=kv), []
        for n in (5, 8, 9, 30, 40):
            ctx = torch.randint(20, 500, (n,), generator=torch.Generator().manual_seed(n)).tolist()
            runs.append(rewind_checks(tm, ctx, opts, 512))
        check(f"tiny Cohere2 (window 8), {str(kv)[6:]} KV, contexts of 5-40 tokens: options scored on the context's own "
              f"state with no fork == from their own forks (bit-exact), and the context left as its prefill left it "
              f"({sum(a and b for a, b in runs)}/{len(runs)})", all(a and b for a, b in runs))

# ---------------------------------------------------------------- the target: CPU fp32 (exact), or Tiny Aya on Metal
eng = T.engine
core(eng, TOL)
from decision import decide
if T.real:                                                       # checks of meaning: a trained model's answers
    yes = decide(eng, "boolean", "", "Is Paris the capital of France?")
    no = decide(eng, "boolean", "", "Is Lyon the capital of France?")
    meaning(f"boolean: Paris is the capital -> {yes['decision']} (P(yes) {yes['probability']:.2f}); Lyon -> "
            f"{no['decision']} (P(yes) {no['probability']:.2f})", yes["decision"] is True and no["decision"] is False)
    labels = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
    pos = decide(eng, "score", "", "How positive is this review: 'an absolute delight, I loved every minute'?", labels)
    neg = decide(eng, "score", "", "How positive is this review: 'a dull, painful waste of two hours'?", labels)
    meaning(f"score: a glowing review rates {pos['expected']:.2f}, a scathing one {neg['expected']:.2f} (0-4 scale)",
            pos["expected"] > 2.5 > neg["expected"])
else:
    na("boolean: Paris is the capital -> True; Lyon -> False")
    na("score: a glowing review rates above 2.5, a scathing one below (0-4 scale)")
pick = decide(eng, "choice", "", QUESTION, OPTIONS)
total = sum(o["probability"] for o in pick["options"])
if T.real:
    check(f"choice: picks {OPTIONS[pick['decision']]!r}; probabilities sum to 1 ({total:.6f})",
          pick["decision"] == 0 and abs(total - 1) < 1e-6)
else:                                                            # what the choice means does not apply; the sum does
    na("choice: picks 'Paris'")
    check(f"choice: probabilities sum to 1 ({total:.6f})", abs(total - 1) < 1e-6)
bad = 0
for args in [("pick", "", "q", ["a", "b"]), ("boolean", "", "q", ["a", "b", "c"]), ("choice", "", "q", ["a"]),
             ("score", "", "q", None), ("choice", "", "q", ["a", ""])]:
    try:
        prepare(eng, *args)
    except ValueError:
        bad += 1
check(f"prepare rejects an unknown type, 3 boolean options, 1 choice, no score labels, an empty option ({bad}/5)",
      bad == 5)

# options are whole words: their tokens, then a token that cannot continue the last word or number. Every eos id ends
# one; the dict shows the one a chat closes its turns with (Qwen's <|im_end|>, the tiny target's <|end|>)
tok, V = eng.tokenizer, eng.tokenizer.vocab_size()
mask = boundary_mask(tok, V)
chat = tok.encode(format_chat([{"role": "user", "content": "Hi"}], style=eng.chat_style))
ends = [e for e in sorted(eng.eos_ids) if e in chat]
kinds = {t: bool(mask[tok.encode(t)[0]]) for t in ["0", "es", " Paris", ",", "\n"]} | \
    {tok.decode([e]): bool(mask[e]) for e in ends}
check(f"boundary tokens: {kinds}", kinds == {"0": False, "es": False, " Paris": True, ",": True, "\n": True}
      | {tok.decode([e]): True for e in ends} and ends and all(mask[e] for e in eng.eos_ids))
_, _, plain_ids = prepare(eng, "choice", "", "q", ["Paris", " Lyon"], chat=False)
check("plain prompts put one space before each option", plain_ids == [tok.encode(" Paris"), tok.encode(" Lyon")])
ctx = prompt_ids(tok, "", QUESTION, True, eng.chat_style)
ids = [tok.encode(o) for o in OPTIONS]
with_b = score_options(eng.model, ctx, ids, V, boundary=mask)
want = [alone(eng.model, ctx, o, V, mask) for o in ids]          # reference: one sequence, boundary at its last row
d = max(abs(x.logprob - w) for x, w in zip(with_b, want))
check(f"forked scores with the boundary term == one-sequence reference, max |d| {d:.1e} < {TOL_TEXT}", d < TOL)
if T.real:
    r = decide(eng, "choice", "", "What is 7 + 3? Answer with the number only.", ["1", "10", "7"])
    p1, p10 = math.exp(r["options"][0]["logprob"]), math.exp(r["options"][1]["logprob"])
    meaning(f"'10' is not swallowed by its prefix '1': 7 + 3 -> {['1', '10', '7'][r['decision']]!r} "
            f"(P('1' as a word) {p1:.3f}, P('10') {p10:.3f}, disjoint: sum <= 1)", r["decision"] == 1 and p1 + p10 <= 1)
else:
    na("'10' is not swallowed by its prefix '1': 7 + 3 -> '10'")

# the context is prefilled in chunks (as the scheduler does): 4-token chunks give the same scores as one pass
ctx = prompt_ids(eng.tokenizer, "Inference engines turn trained weights into answers. " * 6, QUESTION, True,
                 eng.chat_style)
ids = [eng.tokenizer.encode(o) for o in OPTIONS]
big, small = (score_options(eng.model, ctx, ids, eng.tokenizer.vocab_size(), chunk=c) for c in (512, 4))
d = max(abs(x.logprob - y.logprob) for x, y in zip(big, small))
check(f"a {len(ctx)}-token context prefilled in 4-token chunks scores like one pass (max |d| {d:.1e} < {TOL_TEXT})",
      d < TOL)

# on the server: a decision runs stepwise (one context chunk per engine step) while a stream keeps generating; a
# cancelled queued decision never runs; queued decisions count against readiness. Chunks of 16, the scheduler's and
# the decision's alike (the check counts them); the target's step delay keeps a fast model's stream running meanwhile.
# The whole-run reference is computed first, before any engine thread runs: one thread uses the model at a time
GREEDY = SamplingParams(temperature=0, repetition_penalty=1.0)
opts, ctx2, opt_ids = prepare(eng, "choice", "Inference engines turn trained weights into answers. " * 8,
                              "What is this text about?", ["Computing", "Cooking"])
whole = score(eng, "choice", opts, ctx2, opt_ids, chunk=16)
s = Scheduler(eng, Metrics(), max_batch=2, max_waiting=2, kv_blocks=64, prefill_chunk=16)
T.slow_steps(s)
stream = s.submit(T.raw_ids("The history of computing is"), GREEDY, 400)
stream.out.get(timeout=120)                                      # it is decoding now
n0 = len(stream.generated)
res = s.run_job(lambda: score_steps(eng, "choice", opts, ctx2, opt_ids, chunk=16)).result(timeout=600)
during = len(stream.generated) - n0
chunks = -(-len(ctx2) // 16)
check(f"a decision over a {len(ctx2)}-token context ran in {chunks} chunks while a stream generated {during} tokens, "
      f"and gives the same result as running it whole", during >= chunks - 1 and res == whole)
s.cancel(stream)
gate, ran = threading.Event(), []
first = s.run_job(lambda: gate.wait(30))                         # holds the engine thread
while s.job is None:
    time.sleep(0.01)
dropped = s.run_job(lambda: ran.append("dropped"))
later = s.run_job(lambda: ran.append("later"))
busy = not s.ready()
try:
    s.run_job(lambda: None); full = False
except QueueFull:
    full = True
dropped.cancel(); gate.set()
later.result(timeout=30)
check(f"a cancelled queued decision never runs ({ran}); with 2 queued, /ready is false and a third is turned away",
      ran == ["later"] and busy and full)
s.shutdown(5)
del s
T.tidy()                                                         # GPU targets: free its pool before the next section


async def http_suite():
    app = create_app(eng, max_batch=2, max_waiting=4, kv_blocks=64, max_model_len=512, prefill_chunk=T.prefill_chunk)
    sched = app.state.scheduler
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300) as c:
        body = {"type": "choice", "question": QUESTION, "options": OPTIONS}
        r = await c.post("/v1/decide", json=body)
        rp = await c.post("/v1/decide", json={**body, "options": [OPTIONS[i] for i in PERM]})
        j, jp = r.json(), rp.json()
        probs = [o["probability"] for o in j["options"]]
        # picking 'Paris' (decision 0) and answering yes are meaning: a trained model's; the rest holds for any model
        check(f"/v1/decide choice: decision {j['decision']} ({OPTIONS[j['decision']]!r}); the shuffled request picks "
              f"the same option with the same probabilities",
              r.status_code == 200 and (j["decision"] == 0 or not T.real)
              and OPTIONS[PERM[jp["decision"]]] == OPTIONS[j["decision"]]
              and all(jp["options"][k]["probability"] == probs[i] for k, i in enumerate(PERM)))
        rb = await c.post("/v1/decide", json={"type": "boolean", "question": "Is Paris the capital of France?",
                                              "embeddings": True})
        jb = rb.json()
        check(f"/v1/decide boolean with embeddings: {jb['decision']}, one {len(jb['options'][0]['embedding'])}-dim "
              f"embedding per option", rb.status_code == 200 and isinstance(jb["decision"], bool)
              and (jb["decision"] is True or not (T.real and T.gate_meaning))
              and len(jb["options"][0]["embedding"]) == eng.model.config.hidden_size)
        if T.real and not T.gate_meaning:                        # yes is meaning: reported on a quantized model
            report(f"/v1/decide boolean: Paris is the capital -> {jb['decision']}", jb["decision"] is True)
        codes = [(await c.post("/v1/decide", json=b)).status_code for b in (
            {"type": "pick", "question": "q", "options": ["a", "b"]},              # unknown type
            {"type": "choice", "question": "q", "options": ["a"]},                 # one option
            {"type": "choice", "question": "q", "options": ["a", "x" * 2000]},     # option too long
            {"type": "choice", "question": "q " * 600, "options": ["a", "b"]})]    # prompt over max_model_len
        check(f"/v1/decide rejects bad input: {codes} == [422, 422, 400, 400]", codes == [422, 422, 400, 400])
        m = (await c.get("/metrics")).text
        n = [l for l in m.splitlines() if l.startswith("engine_decisions_total ")]
        check(f"decisions counted in /metrics ({n[0] if n else 'missing'})", n and float(n[0].split()[1]) == 3)
    sched.shutdown(5)
    check("no KV blocks used by decisions (they run on their own forked states)",
          sched.pool.allocator.num_free == sched.pool.allocator.num_blocks)


asyncio.run(http_suite())
T.tidy()                                                         # GPU targets: free the app's scheduler and pool
gpu_gates(T, check)                                              # GPU targets: peak memory, Metal aborts
freed = "the GPU target's model is freed and the MPS cache emptied" if T.gpu else \
    "the CPU target's model is freed before the Metal section"
del eng, T                                                       # the Metal section loads Qwen on the GPU
check(freed, unload_target(TARGET))

# ---------------------------------------------------------------- Metal INT4: the same properties within INT4/bf16 noise
if TARGET != "qwen":                                             # this section serves Qwen on the GPU
    na("Metal INT4 (Qwen3.5-0.8B): forks, shuffles, groups of one and the rewind within INT4/bf16 noise")
elif torch.backends.mps.is_available():
    eng = load_engine("qwen3.5-0.8b", "metal-int4")
    core(eng, 2e-2)
    del eng
    gc.collect(); torch.mps.empty_cache()

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
