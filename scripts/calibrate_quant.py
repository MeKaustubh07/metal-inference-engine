"""Measure INT4 sensitivity per weight tensor and write a mixed-precision policy (any supported model).

For each 2-D weight, quantize ONLY that tensor to INT4 (everything else at the baseline precision) and measure the KL
divergence of the next-token distribution against reference logits (HF goldens), separately at position 0 (the
attention-sink first token) and at later positions. Tensors whose damage exceeds the threshold stay INT8 in INT4 mode.

Baseline bf16 (the default, Qwen3.5): bf16 weights on the GPU. Baseline int8 (a model whose bf16 weights do not fit,
Tiny Aya): every weight INT8, quantized from the checkpoint at load; each tensor in turn is rebuilt in INT4 from the
checkpoint, measured, and put back. A reference that keeps only some positions' logits (logits_from) is compared there.

usage: calibrate_quant.py --model qwen3.5-0.8b|qwen3.5-2b|tiny-aya-global [--baseline bf16|int8] [--golden-dir DIR]
                          [--threshold 0.005] [--out F]
"""
import argparse
import glob
import json
import sys

import torch

sys.path.insert(0, "src")
from engine import load_engine
from ops import softmax
from quant import QuantTensor, quantize


def kl_scores(model, goldens, vocab):
    """Mean KL at position 0 (over the references that keep it) and at the later positions."""
    pos0, later = [], []
    for g in goldens:
        start = g.get("logits_from", 0)
        p = softmax(g["logits"][:, :vocab].float()); q = softmax(model.forward(g["ids"]).cpu()[start:, :vocab])
        kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)
        if start == 0:
            pos0.append(kl[0].item()); later.append(kl[1:].mean().item())
        else:
            later.append(kl.mean().item())
    return sum(pos0) / max(len(pos0), 1), sum(later) / len(later)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--baseline", choices=["bf16", "int8"], default="bf16",
                    help="precision of every other tensor while one is measured in INT4 (int8: bf16 does not fit)")
    ap.add_argument("--golden-dir", default=None, help="HF reference logits (default: the model's golden folder)")
    ap.add_argument("--threshold", type=float, default=0.005, help="KL damage (nats) above which a tensor stays INT8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"configs/quant/{a.model}.json"
    golden_dir = a.golden_dir or {"qwen3.5-0.8b": "tests/golden_qwen35", "qwen3.5-2b": "tests/golden_qwen35_2b",
                                  "tiny-aya-global": "tests/golden_aya"}[a.model]
    files = sorted(glob.glob(f"{golden_dir}/*.pt"))
    if not files:                                                        # else every KL is 0 and nothing is kept INT8
        sys.exit(f"no goldens in {golden_dir}; run scripts/{'golden_aya' if a.model == 'tiny-aya-global' else 'golden_qwen35'}.py")
    goldens = [torch.load(f) for f in files]
    if a.baseline == "bf16":
        eng = load_engine(a.model, "mps")                                # bf16 weights on the GPU
    else:
        eng = load_engine(a.model, "metal-int8")                         # quantized from the checkpoint at load
    m, vocab = eng.model, eng.tokenizer.vocab_size()
    base0, base_later = kl_scores(m, goldens, vocab)                      # also fills the resident cache
    if a.baseline == "bf16":
        keys = [k for k, w in m._cache.items() if isinstance(w, torch.Tensor) and w.ndim == 2
                and w.dtype == torch.bfloat16 and "embed" not in k]
    else:
        keys = [k for k, w in m._cache.items() if isinstance(w, QuantTensor) and "embed" not in k]
    damage = {}
    for n, key in enumerate(keys):
        orig = m._cache[key]
        if a.baseline == "bf16":
            m._cache[key] = quantize(orig.float().cpu(), "int4").dequantize().to(orig.device, torch.bfloat16)
        else:                                       # the next forward rebuilds only this tensor, in INT4, from the
            del m._cache[key]                       # checkpoint (every other tensor is still cached in INT8)
            m.b.scheme = "int4"
        try:
            k0, kl = kl_scores(m, goldens, vocab)
        finally:                                    # put the baseline back even if a forward fails
            if a.baseline == "int8":
                m.b.scheme = "int8"
            m._cache[key] = orig
        damage[key] = {"pos0": round(k0 - base0, 4), "later": round(kl - base_later, 4)}
        print(f"\r{n + 1}/{len(keys)} tensors measured", end="", flush=True)
    keep = sorted(k for k, v in damage.items() if max(v["pos0"], v["later"]) > a.threshold)
    policy = {"model": a.model, "scheme": "int4", "threshold_nats": a.threshold, "reference": golden_dir,
              f"baseline_kl_{a.baseline}": {"pos0": round(base0, 4), "later": round(base_later, 4)},
              "keep_int8": keep, "always_int8": ["embed_tokens.weight"],
              "damage": dict(sorted(damage.items(), key=lambda kv: -max(kv[1].values())))}
    json.dump(policy, open(out, "w"), indent=1)
    print(f"\n{len(keep)} of {len(damage)} tensors stay INT8\nwrote {out}")


if __name__ == "__main__":
    main()
