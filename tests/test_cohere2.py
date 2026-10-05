"""Tiny Aya port, M2: Cohere2Model == transformers' Cohere2ForCausalLM on small random models, fp32 on the CPU.

No download: each model is built from a config with random weights (large ones, so attention is far from uniform and
a RoPE or LayerNorm mistake cannot hide in the noise), saved with save_pretrained and loaded by both. Checked: every
layer and the logits (8 query heads per 2 KV heads: Tiny Aya's ratio of 4); that the test notices broken RoPE; the
LayerNorm eps at every norm; greedy tokens; cached == uncached; paged == contiguous; batched decode == one at a time;
packed prefill == separate, with a continuing chunk; forks that diverge at the same position; an untied head refused;
and the sliding-window guard on a model with a 16-token window: exact up to 16 positions, refused past them, for fresh
and continuing sequences, packs and mixed batches, with no state moved and no KV block taken.
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Cohere2Config as HFConfig, Cohere2ForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

sys.path.insert(0, "src")
import models.cohere2 as cohere2_module
from backend.torch_ref import TorchBackend
from config import Cohere2Config
from models.cohere2 import Cohere2Model
from weight_loader import SafetensorsFile

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def build(tmp: str, window: int, seed: int = 0, eps: float = 1e-5, tied: bool = True):
    """A random 8-layer Cohere2 (sliding, sliding, sliding, full, x2) with 8 query heads per 2 KV heads of 16 dims,
    saved and loaded by HF and by the engine."""
    torch.manual_seed(seed)
    hc = HFConfig(vocab_size=512, hidden_size=128, intermediate_size=320, num_hidden_layers=8, num_attention_heads=8,
                  num_key_value_heads=2, max_position_embeddings=8192, layer_norm_eps=eps, logit_scale=0.25,
                  sliding_window=window, rope_parameters={"rope_type": "default", "rope_theta": 50000.0},
                  tie_word_embeddings=tied, initializer_range=0.3, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    hf = Cohere2ForCausalLM._from_config(hc, attn_implementation="eager").eval()
    for m in hf.modules():                                     # LayerNorm weights start at 1: make them matter
        if type(m).__name__ == "Cohere2LayerNorm":
            m.weight.copy_(1 + 0.3 * torch.randn_like(m.weight))
    hf.save_pretrained(tmp)
    return hf, lambda: Cohere2Model(Cohere2Config.from_json(f"{tmp}/config.json"),
                                    SafetensorsFile(f"{tmp}/model.safetensors"))


def hf_run(hf, ids):
    """HF logits [T, vocab] and the outputs of the embedding, every layer and the final norm."""
    cap, hooks = {}, []
    m = hf.model
    hooks.append(m.embed_tokens.register_forward_hook(lambda _m, _i, o: cap.__setitem__("embed", o[0])))
    for i, layer in enumerate(m.layers):
        hooks.append(layer.register_forward_hook(
            lambda _m, _i, o, i=i: cap.__setitem__(f"l{i}_out", (o[0] if isinstance(o, tuple) else o)[0])))
    hooks.append(m.norm.register_forward_hook(lambda _m, _i, o: cap.__setitem__("final_norm", o[0])))
    logits = hf(ids[None]).logits[0]
    for h in hooks:
        h.remove()
    return logits, cap


def hf_greedy(hf, ids, n):
    out = ids.tolist()
    for _ in range(n):
        out.append(int(hf(torch.tensor([out])).logits[0, -1].argmax()))
    return out[len(ids):]


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

with tempfile.TemporaryDirectory() as tmp:                     # 10. the sliding-window guard
    hf, make = build(tmp, window=16, seed=2)
    model = make()
    ids = torch.randint(3, 512, (17,), generator=torch.Generator().manual_seed(3))
    ref16, _ = hf_run(hf, ids[:16])
    check(f"window 16: 16 positions are exact (relative error {rel(model.forward(ids[:16]), ref16):.1e})",
          rel(model.forward(ids[:16]), ref16) < 1e-5)
    hf_full = Cohere2ForCausalLM._from_config(HFConfig.from_pretrained(tmp, sliding_window=4096),
                                              attn_implementation="eager").eval()
    hf_full.load_state_dict(hf.state_dict())
    binds = rel(hf(ids[None]).logits[0, -1], hf_full(ids[None]).logits[0, -1])
    check(f"window 16: at 17 positions HF's window does exclude a key (logits move by {binds:.1e})", binds > 1e-4)
    try:
        model.forward(ids); check("window 16: a 17-position forward is refused", False)
    except ValueError:
        check("window 16: a 17-position forward is refused", True)
    for kind in ("contiguous", "paged"):                       # a continuing chunk that would cross the window
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
        check(f"window 16, {kind}: a continuing chunk crossing it is refused, nothing moved, then exact",
              refused and untouched and rel(rest, ref16[-1]) < 1e-5)
    pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)
    model.forward(ids[:12], state=b)
    free = pool.allocator.num_free
    try:
        model.forward_packed([(ids[:3], a), (ids[12:17], b)]); refused = False
    except ValueError:
        refused = True
    check("window 16: a pack with one chunk crossing it is refused whole, nothing moved",
          refused and a.length == 0 and b.length == 12 and pool.allocator.num_free == free)
    pool = model.new_paged_pool(num_blocks=16, block_size=4, max_seqs=8)
    a, b = model.new_paged_state(pool), model.new_paged_state(pool)          # the one at the window comes LAST,
    model.forward(ids[:5], state=a)                                          # at a block boundary (16 = 4 x 4)
    model.forward(ids[:16], state=b)
    free, tables = pool.allocator.num_free, (len(a.kv.block_table), len(b.kv.block_table))
    try:
        model.decode_batch([5, 5], [a, b]); refused = False
    except ValueError:
        refused = True
    check("window 16: a batched decode with one sequence at it is refused, no block taken, no state moved",
          refused and (a.length, b.length) == (5, 16) and pool.allocator.num_free == free
          and (len(a.kv.block_table), len(b.kv.block_table)) == tables)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
