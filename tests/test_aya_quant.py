"""Tiny Aya port, M4: INT8 and INT4 on the Metal backend against the fp32 answer key (scripts/golden_aya.py).

The metrics of tests/test_qwen35_2b.py: the KL divergence of the next-token distribution from the reference's, and
top-1 flips at positions where the reference's top two are more than 0.5 apart (a near-tie may flip on rounding alone);
here the KL is pooled over all positions, decode steps included. Positions: every position of the 7 raw prompts, the
last 8 of the 2 chat prompts, and the 9 decode steps of every prompt (the reference's own greedy tokens fed back, so
both see the same history). The gates use every position; the KL at served positions is reported too, leaving out
positions inside the chat template (the token read is a template token and the next one is fixed by the template; the
chat's last position, which predicts the reply's first token, stays in) and decode steps from the reference's own stop
token on, where a server would have stopped: there the reference is unsure (top probability ~0.5) and the KL swings
most. Also reported: greedy agreement, resident weights and single-stream decode speed. Each scheme is measured
twice on the same loaded weights, with an fp32 and a bf16 KV cache (M5): the gates judge both caches (the shipped one
is bf16), and its cost is the change in KL. Needs the gated files, the answer key and the .qt files (scripts/quantize.py); otherwise
SKIP, or FAIL naming the command.
"""
import gc
import glob
import os
import sys
import time

import torch

sys.path.insert(0, "src")
from engine import load_engine
from ops import softmax
from quant import QuantTensor

D, G = "models/tiny-aya-global", "tests/golden_aya"
if not os.path.exists(f"{D}/model.safetensors.index.json") or not torch.backends.mps.is_available():
    print(f"SKIP: needs the gated files in {D} and a Metal GPU")
    sys.exit(0)
goldens = sorted(glob.glob(f"{G}/*.pt"), key=lambda f: int(os.path.basename(f)[:-3]))
if not goldens:
    print("FAIL: no answer key: run scripts/.venv/bin/python scripts/golden_aya.py")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


recs = [torch.load(f) for f in goldens]


def measure(m, tok, eos_ids) -> dict:
    """The model's next-token distributions vs the answer key's, at every recorded position."""
    V = tok.vocab_size()
    kls, served, flips, clear_n, agree, tot = [], [], 0, 0, 0, 0
    template = set(tok.special.values()) - {tok.bos_id}
    for r in recs:
        st = m.new_state(len(r["ids"]) + 16)
        ours = [m.forward(r["ids"], state=st).cpu()[r["logits_from"]:, :V]]
        for t in r["greedy"][:-1].tolist():                    # the reference's tokens, so both see the same history
            ours.append(m.forward(torch.tensor([t]), state=st).cpu()[:, :V])
        ours = torch.cat(ours)
        ref = torch.cat([r["logits"], r["step_logits"]])[:, :V]
        p, q = softmax(ref), softmax(ours)
        kls.append((p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1))
        n, seq = len(r["ids"]), r["ids"].tolist() + r["greedy"].tolist()   # seq[i]: the token position i reads
        stop = next((n + k for k, t in enumerate(r["greedy"].tolist()) if t in eos_ids), len(seq))
        served.append(torch.tensor([i < stop and (i >= n - 1 or seq[i] not in template)
                                    for i in range(r["logits_from"], r["logits_from"] + len(ours))]))
        top2 = ref.topk(2, -1).values
        clear = (top2[:, 0] - top2[:, 1]) > 0.5
        flips += int(((ours.argmax(-1) != ref.argmax(-1)) & clear).sum())
        clear_n += int(clear.sum())
        g = m.new_state(len(r["ids"]) + 16)                    # free greedy: tokens equal to the reference's
        nxt, out = int(m.forward(r["ids"], state=g, last_only=True)[0].argmax()), []
        for _ in range(10):
            out.append(nxt)
            nxt = int(m.forward(torch.tensor([nxt]), state=g)[0].argmax())
        same = next((j for j, (a, b) in enumerate(zip(out, r["greedy"].tolist())) if a != b), 10)
        agree, tot = agree + same, tot + 10
    kl, is_served = torch.cat(kls), torch.cat(served)
    return dict(kl=kl.mean().item(), kl_max=kl.max().item(), kl_served=kl[is_served].mean().item(),
                n_served=int(is_served.sum()), flips=flips, clear=clear_n, agree=agree, tot=tot, n=len(kl))


def decode_speed(m, tok) -> float:
    st = m.new_state(64)
    m.forward(torch.tensor(tok.encode("The capital of France is", add_bos=True)), state=st, last_only=True)
    for _ in range(3):
        m.forward(torch.tensor([12]), state=st, last_only=True)
    m.b.sync(); t0 = time.perf_counter()
    for _ in range(20):
        int(m.forward(torch.tensor([12]), state=st, last_only=True)[0].argmax())
    return 20 / (time.perf_counter() - t0)


summary = {}
for scheme in ("int8", "int4"):
    qt = f"{D}/model.{scheme}.qt"
    if not os.path.exists(qt):
        policy = " --policy configs/quant/tiny-aya-global.json" if scheme == "int4" else ""
        check(f"{scheme}: {qt} exists (scripts/quantize.py {D}/model.safetensors.index.json {qt} --scheme {scheme}"
              f"{policy})", False)
        continue
    eng = load_engine("tiny-aya-global", f"metal-{scheme}", qt)
    m, tok = eng.model, eng.tokenizer
    shipped = m.kv_dtype                                       # the registry's: bf16
    for kv in (torch.float32, torch.bfloat16):                 # the same weights, the KV cache in each dtype
        m.kv_dtype = kv
        s = summary[scheme, kv] = measure(m, tok, eng.eos_ids) | {"tps": decode_speed(m, tok)}
        print(f"      {scheme}, {str(kv)[6:]} KV: {s['n']} positions | KL vs fp32 mean {s['kl']:.4f} (max "
              f"{s['kl_max']:.3f}; at the {s['n_served']} served positions {s['kl_served']:.4f}) | top-1 flips "
              f"{s['flips']}/{s['clear']} non-tied | greedy == reference for {s['agree']}/{s['tot']} tokens | decode "
              f"{s['tps']:.1f} tok/s", flush=True)
    m.kv_dtype = shipped
    gb = sum(w.nbytes if isinstance(w, QuantTensor) else w.numel() * w.element_size() for w in m._cache.values()) / 1e9
    d = {k: summary[scheme, torch.bfloat16][k] - summary[scheme, torch.float32][k] for k in ("kl", "kl_served")}
    summary[scheme, "delta"] = d
    print(f"      {scheme}: weights {gb:.2f} GB | bf16 KV changes the KL by {d['kl']:+.4f} ({d['kl_served']:+.4f} at "
          f"served positions)", flush=True)
    del eng, m
    gc.collect()
    torch.mps.empty_cache()

for scheme, kl_max, flips_max in (("int8", 0.05, 0.02), ("int4", 0.15, 0.10)):   # the gates, on both KV caches:
    for kv in (torch.float32, torch.bfloat16):                                    # a broken one must not hide
        if (scheme, kv) in summary:
            s = summary[scheme, kv]
            check(f"{scheme}, {str(kv)[6:]} KV: KL vs the fp32 answer key < {kl_max} ({s['kl']:.4f}) and top-1 flips "
                  f"<= {flips_max:.0%} of non-tied positions ({s['flips']}/{s['clear']})",
                  s["kl"] < kl_max and s["flips"] <= flips_max * s["clear"])
if ("int8", "delta") in summary:
    d = summary["int8", "delta"]
    check(f"int8: bf16 KV changes the KL by at most 0.01 either way ({d['kl']:+.4f})", abs(d["kl"]) <= 0.01)
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
