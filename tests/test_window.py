"""Tiny Aya port, M5 part B: the sliding window (each query sees itself and the W - 1 keys before it, as transformers'
`kv_idx > q_idx - sliding_window` on top of causal), checked against a float64 oracle written from absolute positions
(tests/tiny_cohere2.py), on the CPU. Every test crosses the window: windows that do not divide the block size, chunks
longer than the window, chunks that start or end at it, and freed KV blocks filled with NaN so a stray read shows.
"""
import sys
import tempfile

import torch

sys.path.insert(0, "src")
import ops
from backend.torch_ref import TorchBackend
from tiny_cohere2 import build, hf_greedy, hf_run, ref_attention, rel

torch.manual_seed(0)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


# 1. ops.attention with a window == the oracle, exhaustively over small cases: windows 1..10, chunks of 1..8 queries
# starting anywhere in 0..24, given any key suffix that still holds every key a query needs (from 0 up to the first
# query's window start). Large scores, so one wrongly masked or unmasked key moves the output far past the tolerance.
def case(W, T, start, lo, window):
    q = torch.randn(T, 2, 4) * 3
    k, v = torch.randn(start + T, 1, 4) * 3, torch.randn(start + T, 1, 4)
    want = ref_attention(q, k, v, torch.arange(start, start + T), torch.arange(start + T), W)
    got = ops.attention(q, k[lo:], v[lo:], causal=True, window=window)
    return (got.double() - want).abs().max().item()


worst, n, off_by_one = 0.0, 0, {}
for W in range(1, 11):
    for T in range(1, 9):
        for start in range(25):
            for lo in range(max(0, start - W + 1) + 1):
                worst = max(worst, case(W, T, start, lo, W)); n += 1
            if start + T > W:                                         # the window excludes a key: W +- 1 must show
                for wrong in (W - 1, W + 1):
                    if wrong >= 1:
                        off_by_one[wrong, W] = max(off_by_one.get((wrong, W), 0.0), case(W, T, start, 0, wrong))
check(f"windowed ops.attention == oracle on all {n} cases (W 1-10, T 1-8, start 0-24, every valid key suffix), "
      f"worst {worst:.1e}", worst < 1e-5)
missed = [k for k, d in off_by_one.items() if d < 1e-3]
check(f"sensitivity: a window off by one (W - 1 or W + 1) differs from the oracle for all {len(off_by_one)} "
      "window sizes", not missed)

# 2. realistic heads and scales, a chunk longer than the window and one ending exactly at it
for T, S, W, scale in ((5, 40, 7, 1), (5, 40, 7, 30), (12, 12, 4, 1), (6, 13, 13, 1), (1, 300, 64, 1)):
    q = torch.randn(T, 8, 16) * scale
    k, v = torch.randn(S, 2, 16), torch.randn(S, 2, 16)
    want = ref_attention(q, k, v, torch.arange(S - T, S), torch.arange(S), W)
    d = (ops.attention(q, k, v, window=W).double() - want).abs().max().item()
    check(f"8/2 heads, {T} quer{'y' if T == 1 else 'ies'} over {S} keys, window {W}, q-scale {scale}: max|diff| vs "
          f"oracle {d:.1e}", d < 1e-5)

# 3. below the window nothing changes: no window, or a window that excludes no key, gives exactly today's result
q, k, v = torch.randn(7, 8, 16), torch.randn(20, 2, 16), torch.randn(20, 2, 16)
base = ops.attention(q, k, v)
check("window=None and a window >= every query's position are bit-identical to no window",
      torch.equal(ops.attention(q, k, v, window=None), base) and torch.equal(ops.attention(q, k, v, window=20), base))
try:
    ops.attention(q, k, v, window=0); refused = False
except ValueError:
    refused = True
check("window 0 refused", refused)

# 3b. the scores are computed in place (one [heads, T, S] buffer instead of three, which matters at 8K): the same
# operations in the same order as the formula before M5, written out here, so the result is bit-identical on the CPU
def before_m5(q, k, v):
    T, Hq, d = q.shape
    S, Hkv, _ = k.shape
    kh, vh = (x.float().repeat_interleave(Hq // Hkv, dim=1).transpose(0, 1) for x in (k, v))
    scores = (q.float().transpose(0, 1) @ kh.transpose(1, 2)) / d ** 0.5
    scores = scores.masked_fill(torch.ones(T, S, dtype=torch.bool).triu(diagonal=S - T + 1), float("-inf"))
    return (ops.softmax(scores, dim=-1) @ vh).transpose(0, 1)
same = all(torch.equal(ops.attention(q, k, v), before_m5(q, k, v))
           for q, k, v in ((torch.randn(T, 8, 16) * sc, torch.randn(S, 2, 16), torch.randn(S, 2, 16))
                           for T, S, sc in ((7, 20, 1), (1, 300, 1), (64, 64, 30), (128, 512, 1))))
check("in-place scores, masking and softmax are bit-identical to the formula before M5 (4 shapes, CPU)", same)

# 4. the paged reference reads only the window: blocks wholly before it are NaN (as if freed or reused) and the result
# is still finite and equal to attention over exactly those positions. Windows 16 (aligned), 20 and 6 (not)
rb = TorchBackend("cpu", torch.float32)
bs, lens_l = 4, [1, 6, 7, 20, 21, 43]
for W in (16, 20, 6):
    nbs = [-(-n // bs) for n in lens_l]
    perm = torch.randperm(sum(nbs)).tolist()
    tabs = [perm[sum(nbs[:i]):sum(nbs[:i + 1])] for i in range(len(lens_l))]
    tables = torch.tensor([t + [0] * (max(nbs) - len(t)) for t in tabs], dtype=torch.int32)
    kp, vp = torch.randn(sum(nbs), bs, 2, 16), torch.randn(sum(nbs), bs, 2, 16)
    gathered = [(kp[t].flatten(0, 1)[:n].clone(), vp[t].flatten(0, 1)[:n].clone()) for t, n in zip(tabs, lens_l)]
    for t, n in zip(tabs, lens_l):                                   # poison every block before the window's start
        for b in t[:max(0, n - W) // bs]:
            kp[b], vp[b] = float("nan"), float("nan")
    q = torch.randn(len(lens_l), 8, 16)
    out = rb.paged_attention(q, kp, vp, tables, torch.tensor(lens_l, dtype=torch.int32), bs, window=W)
    want = torch.cat([ops.attention(q[i:i + 1], K[max(0, n - W):], V[max(0, n - W):])
                      for i, ((K, V), n) in enumerate(zip(gathered, lens_l))])
    check(f"paged reference, window {W}, lengths {lens_l}, block size {bs}: finite with the blocks before the window "
          f"NaN, max|diff| {(out - want).abs().max().item():.1e}", bool(torch.isfinite(out).all())
          and torch.allclose(out, want, rtol=0, atol=1e-6))

# 5. Cohere2Model with an 8-token window vs transformers' full forward, over 40 positions (5 windows): every path the
# engine has: stateless, chunked prefill on contiguous and paged states (chunks ending at the window, longer than it,
# token by token; block sizes 4 and 3, which do not align with it), decode, batched decode, packed prefill and forks
torch.set_grad_enabled(False)
with tempfile.TemporaryDirectory() as tmp:
    hf, make = build(tmp, window=8, seed=4)
    model = make()
    ids = torch.randint(3, 512, (40,), generator=torch.Generator().manual_seed(5))
    ref_logits, ref = hf_run(hf, ids)
    cap = {}
    logits = model.forward(ids, capture=cap)
    worst = max(rel(cap[k], ref[k]) for k in ref)
    check(f"model, window 8, 40 positions, stateless: embedding, 8 layers, final norm and logits == HF, worst "
          f"{max(worst, rel(logits, ref_logits)):.1e}", worst < 1e-5 and rel(logits, ref_logits) < 1e-5)

    schedules = [[40], [8, 32], [5, 3, 32], [3, 17, 20], [9, 7, 8, 16], [1] * 40]
    def chunked(new_state, sched):
        st, outs, pos = new_state(), [], 0
        for n in sched:
            outs.append(model.forward(ids[pos:pos + n], state=st)); pos += n
        return torch.cat(outs), st
    for kind, bs in (("contiguous", None), ("paged", 4), ("paged", 3)):
        new = (lambda: model.new_state(64)) if bs is None else \
            (lambda bs=bs: model.new_paged_state(model.new_paged_pool(num_blocks=64, block_size=bs, max_seqs=4)))
        worst = max(rel(chunked(new, sch)[0], ref_logits) for sch in schedules)
        check(f"model, {kind}{f' (blocks of {bs})' if bs else ''}: chunked prefill == HF for {len(schedules)} chunk "
              f"schedules (ending at the window, longer than it, token by token), worst {worst:.1e}", worst < 1e-5)

    prompt = ids[:6]                                            # greedy: 20 tokens, from inside the window to 3.25
    want = hf_greedy(hf, prompt, 20)                            # windows past it
    for kind in ("contiguous", "paged"):
        st = model.new_state(64) if kind == "contiguous" else \
            model.new_paged_state(model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=2))
        nxt, got = int(model.forward(prompt, state=st, last_only=True)[0].argmax()), []
        for _ in range(20):
            got.append(nxt)
            nxt = int(model.forward(torch.tensor([nxt]), state=st)[0].argmax())
        check(f"model, {kind}: greedy 20 tokens from a 6-token prompt (to position 26) == HF", got == want)

    pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=4)
    lens, toks = (5, 13, 30), (7, 9, 11)
    paged = [model.new_paged_state(pool) for _ in lens]
    alone = [model.new_state(64) for _ in lens]
    for st, a, n in zip(paged, alone, lens):
        model.forward(ids[:n], state=st); model.forward(ids[:n], state=a)
    worst = 0.0
    for step in range(3):                                       # three steps: 30 -> 33 crosses blocks too
        b = model.decode_batch([t + step for t in toks], paged)
        one = torch.cat([model.forward(torch.tensor([t + step]), state=a) for t, a in zip(toks, alone)])
        worst = max(worst, rel(b, one))
    check(f"model: batched paged decode of 3 sequences (5, 13, 30 positions) == each alone, 3 steps, worst {worst:.1e}",
          worst < 1e-5)

    pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=2)
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)
    model.forward(ids[:11], state=b)
    packed = model.forward_packed([(ids[:7], a), (ids[11:25], b)])          # b's chunk crosses its window again
    sep = torch.stack([model.forward(ids[:7])[-1], model.forward(ids[:25])[-1]])
    check(f"model: packed prefill with a continuing chunk past the window == separate forwards, relative error "
          f"{rel(packed, sep):.1e}", rel(packed, sep) < 1e-5)

    base = model.new_state(64); model.forward(ids[:30], state=base)
    fork = base.fork()
    x, y = model.forward(torch.tensor([7]), state=base)[0], model.forward(torch.tensor([9]), state=fork)[0]
    ok = rel(x, model.forward(torch.cat([ids[:30], torch.tensor([7])]))[-1]) < 1e-5 and \
        rel(y, model.forward(torch.cat([ids[:30], torch.tensor([9])]))[-1]) < 1e-5
    check("model: a fork at position 30 and the original continue independently and exactly", ok)

    # 6. the sliding layers never read keys before their window: in a paged pool, poison (NaN) every block that lies
    # wholly before the next query's window, in the sliding layers only (the full layers still need them), then
    # decode on and continue with another chunk: the result stays finite and equal to an unpoisoned run
    pool = model.new_paged_pool(num_blocks=32, block_size=4, max_seqs=2)
    st, clean = model.new_paged_state(pool), model.new_state(64)
    model.forward(ids[:30], state=st); model.forward(ids[:30], state=clean)
    first_needed = 30 - 8 + 1                                  # the next query (position 30) sees keys 23 .. 30
    for i in range(model.config.num_hidden_layers):
        if model.config.is_sliding(i):
            for blk in st.kv.block_table[:first_needed // 4]:
                pool.kv.k[i, blk], pool.kv.v[i, blk] = float("nan"), float("nan")
    outs = [model.forward(torch.tensor([t]), state=st) for t in (3, 4, 5)] + [model.forward(ids[30:40], state=st)]
    want = [model.forward(torch.tensor([t]), state=clean) for t in (3, 4, 5)] + [model.forward(ids[30:40], state=clean)]
    got, want = torch.cat(outs), torch.cat(want)
    check(f"model: sliding layers' blocks before the window poisoned with NaN: 3 decode steps and a 10-token chunk "
          f"stay finite and == an unpoisoned run ({rel(got, want):.1e})",
          bool(torch.isfinite(got).all()) and rel(got, want) < 1e-6)

    # 7. bf16 KV past the window: paged == contiguous exactly, the same chunking
    m16 = make(kv_dtype=torch.bfloat16)
    def run16(st):
        return torch.cat([m16.forward(ids[:5], state=st), m16.forward(ids[5:8], state=st),
                          m16.forward(ids[8:40], state=st)] + [m16.forward(torch.tensor([t]), state=st) for t in (3, 4)])
    c = run16(m16.new_state(64))
    g = run16(m16.new_paged_state(m16.new_paged_pool(num_blocks=64, block_size=3, max_seqs=2)))
    check(f"model, bf16 KV past the window: paged (blocks of 3) == contiguous, max difference "
          f"{(c - g).abs().max().item():.1e}", torch.allclose(c, g, rtol=0, atol=1e-6))

    # 8. the long answer key (scripts/golden_aya.py --long) prefills its prompt in chunks over one transformers
    # DynamicCache, one decoder layer at a time: chunks crossing the window must give HF's full forward
    sys.path.insert(0, "scripts")
    from golden_aya import StreamedCohere2
    from transformers.cache_utils import DynamicCache
    streamed = StreamedCohere2(tmp)
    cache, hs, pos = DynamicCache(config=streamed.config), [], 0
    for n in (5, 3, 17, 15):                                   # 5 + 3 ends at the window; 17 is longer than it
        hs.append(streamed.run_all([ids[pos:pos + n].tolist()], [cache])[0]); pos += n
    h = torch.cat(hs)
    check(f"answer-key generator: chunked prefill (5, 3, 17, 15) over one cache == HF's full forward past the "
          f"window: final norm {rel(h, ref['final_norm']):.1e}, logits {rel(streamed.head(h), ref_logits):.1e}",
          rel(h, ref["final_norm"]) < 1e-5 and rel(streamed.head(h), ref_logits) < 1e-5)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
