"""Tiny Aya port, M2: Cohere2Model == transformers' Cohere2ForCausalLM on small random models, fp32 on the CPU.

No download: each model is built from a config with random weights (large ones, so attention is far from uniform and
a RoPE or LayerNorm mistake cannot hide in the noise), saved with save_pretrained and loaded by both. Checked: every
layer and the logits (8 query heads per 2 KV heads: Tiny Aya's ratio of 4); that the test notices broken RoPE; the
LayerNorm eps at every norm; greedy tokens; cached == uncached; paged == contiguous; batched decode == one at a time;
packed prefill == separate, with a continuing chunk; forks that diverge at the same position; an untied head refused;
and the length cap (max_position_embeddings) on a model with an 8-token window and a 16-position cap: exact up to 16
positions with the window binding, refused past them, for fresh and continuing sequences, packs and mixed batches, with
no state moved and no KV block taken (the window itself: tests/test_window.py).
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Cohere2Config as HFConfig, Cohere2ForCausalLM

sys.path.insert(0, "src")
import models.cohere2 as cohere2_module
from backend.torch_ref import TorchBackend
from config import Cohere2Config
from models.cohere2 import Cohere2Model
from tiny_cohere2 import build, hf_greedy, hf_run, rel

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")


with tempfile.TemporaryDirectory() as tmp:
    hf, make = build(tmp, window=4096)
    model = make()
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(3, 512, (23,), generator=g)

    # 1. every layer and the logits
    ref_logits, ref = hf_run(hf, prompt)
    cap = {}
    logits = model.forward(prompt, capture=cap)
    worst = max(rel(cap[k], ref[k]) for k in ref)
    check(f"all {len(ref)} captured tensors (embedding, 8 layers, final norm) == HF, worst relative error "
          f"{worst:.1e}", worst < 1e-5 and len(ref) == 10)
    check(f"logits (x logit_scale 0.25) == HF, relative error {rel(logits, ref_logits):.1e}",
          rel(logits, ref_logits) < 1e-5)

    # 2. the comparison can fail: without the q/k row reordering, RoPE pairs the wrong numbers
    real = cohere2_module.interleaved_to_half
    cohere2_module.interleaved_to_half = lambda heads, d: torch.arange(heads * d)
    broken = make().forward(prompt)
    cohere2_module.interleaved_to_half = real
    check(f"sensitivity: half-split RoPE without the reordering is caught (relative error "
          f"{rel(broken, ref_logits):.1e})", rel(broken, ref_logits) > 1e-2)

    # 3. greedy tokens, and cached == uncached
    want = hf_greedy(hf, prompt, 12)
    state = model.new_state(64)
    step = model.forward(prompt, state=state, last_only=True)[0]
    got, worst = [], 0.0
    for _ in range(12):
        nxt = int(step.argmax())
        got.append(nxt)
        full = model.forward(torch.tensor(prompt.tolist() + got))[-1]          # no cache: the whole sequence again
        step = model.forward(torch.tensor([nxt]), state=state)[0]
        worst = max(worst, rel(step, full))
    check(f"greedy: 12 tokens identical to HF's ({got[:6]} ...)", got == want)
    sys.path.insert(0, "scripts")
    from golden_aya import StreamedCohere2                        # the M3 answer-key generator for the real model
    s_logits, s_greedy = StreamedCohere2(tmp).greedy(prompt.tolist(), 12)
    check("answer-key generator (one HF layer at a time) == the full HF model bit for bit, same greedy tokens",
          torch.equal(s_logits, ref_logits) and s_greedy == want)
    trio = [prompt[:5].tolist(), prompt.tolist(), prompt[3:14].tolist()]   # lockstep: different lengths, own caches
    t_logits, t_greedy, _ = StreamedCohere2(tmp).greedy_all(trio, 6)
    ok = all(rel(lg, hf(torch.tensor([p])).logits[0]) < 1e-6 and g == hf_greedy(hf, torch.tensor(p), 6)
             for p, lg, g in zip(trio, t_logits, t_greedy))
    check("answer-key generator, 3 prompts of different lengths in lockstep == HF on each alone", ok)
    check(f"cached decode == recomputing the whole sequence, relative error {worst:.1e}", worst < 1e-5)

    # 4. paged == contiguous, prompt in two chunks and then decode
    def run(st):
        out = [model.forward(prompt[:9], state=st), model.forward(prompt[9:], state=st)]
        out += [model.forward(torch.tensor([t]), state=st) for t in got[:6]]
        return torch.cat(out)
    pool = model.new_paged_pool(num_blocks=64, block_size=4, max_seqs=8)
    paged_state = model.new_paged_state(pool)
    a, b = run(model.new_state(64)), run(paged_state)
    check(f"paged KV (4-token blocks) == contiguous, max difference {(a - b).abs().max().item():.1e}",
          torch.allclose(a, b, rtol=0, atol=1e-6) and len(paged_state.kv.block_table) == -(-(23 + 6) // 4))
    paged_state.free()

    # 5. batched decode == one sequence at a time (paged and contiguous), and packed prefill == separate prefills
    prompts = [prompt[:5], prompt[:17], prompt[3:14]]
    for kind in ("paged", "contiguous"):
        new = (lambda: model.new_paged_state(pool)) if kind == "paged" else (lambda: model.new_state(64))
        alone, batched = [new() for _ in prompts], [new() for _ in prompts]
        for p, s in zip(prompts, alone):
            model.forward(p, state=s)
        packed = model.forward_packed([(p, s) for p, s in zip(prompts, batched)])
        separate = torch.stack([model.forward(p)[-1] for p in prompts])
        worst = rel(packed, separate)
        for t in (7, 11, 13):
            one = torch.cat([model.forward(torch.tensor([t]), state=s) for s in alone])
            worst = max(worst, rel(model.decode_batch([t] * 3, batched), one))
        for s in alone + batched:
            s.free()
        check(f"{kind}: packed prefill == separate, batched decode == one at a time, relative error {worst:.1e}",
              worst < 1e-5)
    for kind in ("paged", "contiguous"):                          # a chunk continuing a state, packed with a fresh one
        new = (lambda: model.new_paged_state(pool)) if kind == "paged" else (lambda: model.new_state(64))
        cont, fresh = new(), new()
        model.forward(prompt[:9], state=cont)
        packed = model.forward_packed([(prompt[:5], fresh), (prompt[9:], cont)])
        separate = torch.stack([model.forward(prompt[:5])[-1], model.forward(prompt)[-1]])
        check(f"{kind}: a packed chunk continuing a state == the whole prompt alone ({rel(packed, separate):.1e})",
              rel(packed, separate) < 1e-5)
        cont.free(); fresh.free()
    h = model.packed_hidden([(prompt, None)])
    check("packed_hidden then head == forward", rel(model.head(h), ref_logits) < 1e-5)

    # streaming (stream=True, the fp32 CPU mode for the real model), on bf16 weights so that widening is real: each
    # weight widened when used and dropped, the head widened 100 rows at a time (the table never whole); same numbers
    class Bf16:
        def get(self, name):
            return model.weights.get(name).to(torch.bfloat16)

        def tensor_names(self):
            return model.weights.tensor_names()
    rows, cohere2_module.HEAD_ROWS = cohere2_module.HEAD_ROWS, 100
    streamed, normal = Cohere2Model(model.config, Bf16(), stream=True), Cohere2Model(model.config, Bf16())
    widened, prepare = [], streamed.b.prepare
    streamed.b.prepare = lambda w, name=None: (widened.append(tuple(w.shape)), prepare(w, name))[1]
    s_state, n_state = streamed.new_state(64), normal.new_state(64)
    a = torch.cat([streamed.forward(prompt, state=s_state), streamed.decode_batch([7], [s_state])])
    b = torch.cat([normal.forward(prompt, state=n_state), normal.decode_batch([7], [n_state])])
    cohere2_module.HEAD_ROWS = rows
    kept = [k for k, v in streamed._cache.items() if v.dim() > 1]
    table_parts = sorted({r for r, _ in widened if r <= 100})
    check(f"stream=True on bf16 weights: the same logits ({rel(a, b):.1e}), no widened weight kept, and the 512-row "
          f"table only ever widened in slices of {table_parts[-1]} rows", rel(a, b) < 1e-6 and not kept
          and (512, 128) not in widened and (100, 128) in widened)
    for be, why in ((types.SimpleNamespace(device=torch.device("meta")), "a non-CPU backend"),
                    (TorchBackend("cpu", torch.bfloat16), "bf16 weights on the CPU")):
        try:
            Cohere2Model(model.config, model.weights, backend=be, stream=True); check(f"stream=True refuses {why}", False)
        except ValueError:
            check(f"stream=True refuses {why}", True)
    pc = {}                                             # packed_hidden's capture: rows in chunk order, every layer
    model.packed_hidden([(prompt, None), (prompt[:5], None)], capture=pc)
    check("packed_hidden(capture=): the whole prompt and a 5-token prefix in one pack, every layer == HF",
          set(pc) == set(ref) and max(max(rel(pc[k][:23], ref[k]), rel(pc[k][23:], ref[k][:5])) for k in ref) < 1e-5)

    # 6. forks: an independent copy of a sequence so far. Both continue at the SAME position with different tokens;
    # the original must not see the copy's write
    base = model.new_state(64)
    model.forward(prompt, state=base)
    fork = base.fork()
    x, y = model.forward(torch.tensor([7]), state=base), model.forward(torch.tensor([9]), state=fork)
    w = model.forward(torch.tensor([5]), state=base)[0]
    check("fork: copy and original diverge at the same position, and the original stays exact",
          rel(w, model.forward(torch.tensor(prompt.tolist() + [7, 5]))[-1]) < 1e-5
          and rel(y[0], model.forward(torch.tensor(prompt.tolist() + [9]))[-1]) < 1e-5)

    # 6b. bf16 KV cache (Tiny Aya served on Metal): K/V rounded to bf16 as stored, widened back to fp32 when read.
    # Half the bytes, a small deviation from HF's fp32 K/V (measured on the real model in test_aya_quant), and every
    # equality between the engine's own paths still exact
    m16 = Cohere2Model(model.config, model.weights, kv_dtype=torch.bfloat16)
    s16, s32 = m16.new_state(64), model.new_state(64)
    p16, p32 = m16.forward(prompt, state=s16), model.forward(prompt, state=s32)
    check(f"bf16 KV: stored as bfloat16, {s16.bytes_used()} bytes vs {s32.bytes_used()} for fp32; prefill logits "
          f"within {rel(p16, p32):.1e} (relative) of fp32 KV",
          s16.kv.k.dtype == torch.bfloat16 and 2 * s16.bytes_used() == s32.bytes_used() and rel(p16, p32) < 2e-2)

    def run16(st):
        out = [m16.forward(prompt[:9], state=st), m16.forward(prompt[9:], state=st)]
        out += [m16.forward(torch.tensor([t]), state=st) for t in got[:6]]
        return torch.cat(out)
    pool16 = m16.new_paged_pool(num_blocks=64, block_size=4, max_seqs=8)
    c16, g16, c32 = run16(m16.new_state(64)), run16(m16.new_paged_state(pool16)), run(model.new_state(64))
    check(f"bf16 KV: paged pool (bfloat16) == contiguous, max difference {(c16 - g16).abs().max().item():.1e}; the "
          f"whole run within {rel(c16, c32):.1e} of fp32 KV",
          pool16.kv.k.dtype == torch.bfloat16 and torch.allclose(c16, g16, rtol=0, atol=1e-6) and rel(c16, c32) < 2e-2)

    def prefilled(new, ids):
        st = new(); m16.forward(ids, state=st); return st
    pair = (prompt[:5], prompt[:17])
    batched = m16.decode_batch([7, 9], [prefilled(lambda: m16.new_paged_state(pool16), p) for p in pair])
    alone = torch.cat([m16.forward(torch.tensor([t]), state=prefilled(lambda: m16.new_state(64), p))
                       for t, p in zip((7, 9), pair)])
    check(f"bf16 KV: batched paged decode == one at a time, relative error {rel(batched, alone):.1e}",
          rel(batched, alone) < 1e-5)
    # chunked vs one pass is NOT exact in bf16: the K/V of later layers differ by fp32 rounding (other GEMM shapes),
    # and a value near a bf16 rounding boundary then rounds one step (2^-8) the other way. The difference must stay
    # far below bf16's own deviation from fp32 KV
    st1, st2 = m16.new_state(64), m16.new_state(64)
    one = m16.forward(prompt, state=st1)
    two = torch.cat([m16.forward(prompt[:9], state=st2), m16.forward(prompt[9:], state=st2)])
    flips = int((st1.kv.k[:, :23] != st2.kv.k[:, :23]).sum() + (st1.kv.v[:, :23] != st2.kv.v[:, :23]).sum())
    check(f"bf16 KV: prompt in two chunks (9 + 14) vs one pass: {flips} of {2 * st1.kv.k[:, :23].numel()} cached "
          f"values one bf16 step apart, logits within {rel(two, one):.1e} (bf16 vs fp32 KV: {rel(p16, p32):.1e})",
          rel(two, one) < rel(p16, p32) / 10)
    base16 = m16.new_state(64)
    m16.forward(prompt, state=base16)
    f16 = base16.fork()
    x16, y16 = m16.forward(torch.tensor([7]), state=base16), m16.forward(torch.tensor([7]), state=f16)
    check("bf16 KV: a fork is bfloat16 too and continues exactly like the original",
          f16.kv.k.dtype == torch.bfloat16 and torch.equal(x16, y16))
    lp, lq = torch.log_softmax(ref_logits, -1), torch.log_softmax(p16, -1)
    kl16 = (lp.exp() * (lp - lq)).sum(-1)
    same = int((p16.argmax(-1) == ref_logits.argmax(-1)).sum())
    check(f"bf16 KV vs HF (fp32 K/V): same argmax at {same}/{len(prompt)} positions, KL mean {kl16.mean():.1e} "
          f"max {kl16.max():.1e}", same == len(prompt))
    try:
        Cohere2Model(model.config, model.weights, kv_dtype=torch.float16); refused = False
    except ValueError:
        refused = True
    check("kv_dtype other than float32 / bfloat16 refused (the attention kernels read only those)", refused)

with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as shards:
    import golden_aya                                           # 7. the answer key on a checkpoint shaped like the
    hf16, _ = build(tmp, window=16, seed=4)                     # real one: bf16 in several files (an index), the head
    hf16.to(torch.bfloat16).save_pretrained(shards, max_shard_size="300KB")      # in 100-row slices, prompts crossing
    hf = Cohere2ForCausalLM.from_pretrained(shards, dtype=torch.float32, attn_implementation="eager").eval()  # the window
    long, short = (torch.randint(3, 512, (n,), generator=torch.Generator().manual_seed(n)).tolist() for n in (30, 7))
    rows, golden_aya.HEAD_ROWS = golden_aya.HEAD_ROWS, 100
    caps = [{}, {}]
    k_logits, k_greedy, _ = golden_aya.StreamedCohere2(shards).greedy_all([long, short], 6, caps, keep_from=[20, 0])
    golden_aya.HEAD_ROWS = rows
    ok = os.path.exists(f"{shards}/model.safetensors.index.json")
    for p, lg, g, cap, k0 in zip((long, short), k_logits, k_greedy, caps, (20, 0)):
        h_logits, h_cap = hf_run(hf, torch.tensor(p))
        ok &= torch.equal(lg, h_logits[k0:]) and g == hf_greedy(hf, torch.tensor(p), 6) and set(cap) == set(h_cap) \
            and all(torch.equal(cap[k], h_cap[k]) for k in h_cap)
    check("answer key on bf16 shards, head in slices, 30 tokens past a 16-token window: every capture and logit "
          "bit-identical to HF, same greedy", ok)

with tempfile.TemporaryDirectory() as tmp:                     # 8. eps: the residual stream's variance grows past
    hf, make = build(tmp, window=4096, eps=1.0)                 # ~1000 after layer 0, so only a large eps shows
    prompt = torch.randint(3, 512, (23,), generator=torch.Generator().manual_seed(1))   # whether every norm uses it
    ref_logits, ref = hf_run(hf, prompt)
    cap = {}
    logits = make().forward(prompt, capture=cap)
    worst = max(rel(cap[k], ref[k]) for k in ref)
    check(f"eps 1.0: every layer, the final norm and the logits == HF (worst {worst:.1e})",
          worst < 1e-5 and rel(logits, ref_logits) < 1e-5)

with tempfile.TemporaryDirectory() as tmp:                     # 9. an untied head would be ignored: refused
    build(tmp, window=4096, tied=False)
    try:
        Cohere2Config.from_json(f"{tmp}/config.json"); check("a checkpoint with an untied lm_head is refused", False)
    except ValueError:
        check("a checkpoint with an untied lm_head is refused", True)
    try:
        golden_aya.StreamedCohere2(tmp); check("... and the answer key refuses it too", False)
    except ValueError:
        check("... and the answer key refuses it too", True)

with tempfile.TemporaryDirectory() as tmp:                     # 10. the length cap (max_position_embeddings)
    hf, make = build(tmp, window=8, seed=2, max_pos=16)          # window 8 binds inside the cap of 16
    model = make()
    ids = torch.randint(3, 512, (17,), generator=torch.Generator().manual_seed(3))
    ref16, _ = hf_run(hf, ids[:16])
    check(f"window 8, cap 16: 16 positions exact, the window binding (relative error "
          f"{rel(model.forward(ids[:16]), ref16):.1e})", rel(model.forward(ids[:16]), ref16) < 1e-5)
    hf_full = Cohere2ForCausalLM._from_config(HFConfig.from_pretrained(tmp, sliding_window=4096),
                                              attn_implementation="eager").eval()
    hf_full.load_state_dict(hf.state_dict())
    binds = rel(hf(ids[None, :16]).logits[0, -1], hf_full(ids[None, :16]).logits[0, -1])
    check(f"window 8: at 16 positions HF's window does exclude keys (logits move by {binds:.1e})", binds > 1e-4)
    try:
        model.forward(ids); check("cap 16: a 17-position forward is refused", False)
    except ValueError:
        check("cap 16: a 17-position forward is refused", True)
    for kind in ("contiguous", "paged"):                       # a continuing chunk that would cross the cap
        pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
        st = model.new_state(32) if kind == "contiguous" else model.new_paged_state(pool)
        model.forward(ids[:9], state=st)
        blocks, free = (len(st.kv.block_table) if kind == "paged" else 0), pool.allocator.num_free
        try:
            model.forward(ids[9:17], state=st); refused = False                 # 9 + 8 = 17 positions
        except ValueError:
            refused = True
        untouched = st.length == 9 and pool.allocator.num_free == free and \
            (kind == "contiguous" or len(st.kv.block_table) == blocks)
        rest = model.forward(ids[9:16], state=st)[-1]                          # and it still continues exactly
        check(f"cap 16, {kind}: a continuing chunk crossing it is refused, nothing moved, then exact",
              refused and untouched and rel(rest, ref16[-1]) < 1e-5)
    pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)
    model.forward(ids[:12], state=b)
    free = pool.allocator.num_free
    try:
        model.forward_packed([(ids[:3], a), (ids[12:17], b)]); refused = False
    except ValueError:
        refused = True
    check("cap 16: a pack with one chunk crossing it is refused whole, nothing moved",
          refused and a.length == 0 and b.length == 12 and pool.allocator.num_free == free)
    pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)          # the one at the cap comes LAST,
    model.forward(ids[:5], state=a)                                          # at a block boundary (16 = 4 x 4)
    model.forward(ids[:16], state=b)
    free, tables = pool.allocator.num_free, (len(a.kv.block_table), len(b.kv.block_table))
    try:
        model.decode_batch([5, 5], [a, b]); refused = False
    except ValueError:
        refused = True
    check("cap 16: a batched decode with one sequence at it is refused, no block taken, no state moved",
          refused and (a.length, b.length) == (5, 16) and pool.allocator.num_free == free
          and (len(a.kv.block_table), len(b.kv.block_table)) == tables)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
