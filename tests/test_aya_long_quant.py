"""Tiny Aya port, M5 B6/B7: the quantized engine on Metal past the 4096-token sliding window. INT8 and INT4, each with
an fp32 and a bf16 KV cache, against the long fp32 answer key (scripts/golden_aya.py --long, ~4.8K tokens in 8
languages): KL at the key's rows before and after position 4096, the 9 decode steps, greedy agreement (gates as in
tests/test_aya_quant.py). Then the bf16 KV cache's cost out to ~7.6K tokens (950 tokens per language), engine against
engine (fp32 vs bf16 KV, INT4; no fp32 answer key reaches that far). The fp32 CPU engine against the same key is
tests/test_aya_long.py. Needs the gated files, the key and the .qt files; otherwise SKIP, or FAIL naming the command."""
import gc
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import long_texts
from engine import load_engine
from tokenizer import Tokenizer

D, KEY = "models/tiny-aya-global", "tests/golden_aya_long/0.pt"
if not os.path.exists(f"{D}/model.safetensors.index.json") or not torch.backends.mps.is_available():
    print(f"SKIP: needs the gated files in {D} and a Metal GPU")
    sys.exit(0)
if not os.path.exists(KEY):
    print("FAIL: no long answer key: run scripts/.venv/bin/python scripts/golden_aya.py --long")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


t0 = time.perf_counter()
key = torch.load(KEY)
ids, n, W = key["ids"], len(key["ids"]), 4096
tok = Tokenizer(f"{D}/tokenizer.json")
from tokenizers import Tokenizer as HFTokenizer
hf_tok = HFTokenizer.from_file(f"{D}/tokenizer.json")
class HFEnc:
    @staticmethod
    def encode(text):
        return hf_tok.encode(text, add_special_tokens=False).ids


# 1. past the window, against the key: INT8 and INT4, fp32 and bf16 KV
V, rows = tok.vocab_size(), key["rows"]


def kl(ref, ours):
    lp, lq = torch.log_softmax(ref[:, :V].double(), -1), torch.log_softmax(ours[:, :V].double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def prefill(m, ids, state, chunk=512):
    return torch.cat([m.packed_hidden([(ids[i:i + chunk], state)]) for i in range(0, len(ids), chunk)])


summary = {}
for scheme in ("int8", "int4"):
    qt = f"{D}/model.{scheme}.qt"
    if not os.path.exists(qt):
        check(f"{scheme}: {qt} exists (scripts/quantize.py)", False)
        continue
    eng = load_engine("tiny-aya-global", f"metal-{scheme}", qt)
    m = eng.model
    for kv in (torch.float32, torch.bfloat16):
        m.kv_dtype = kv
        s = m.new_state(n + 16)
        lg = m.head(prefill(m, ids, s)[rows]).float().cpu()
        stp = torch.cat([m.forward(torch.tensor([t]), state=s) for t in key["greedy"][:-1].tolist()]).float().cpu()
        k_rows, k_steps = kl(key["logits"], lg), kl(key["step_logits"], stp)
        agree = int(lg[-1, :V].argmax() == key["greedy"][0]) + int((stp[:, :V].argmax(-1) == key["greedy"][1:]).sum())
        summary[scheme, kv] = dict(before=k_rows[rows < W].mean().item(), after=k_rows[rows >= W].mean().item(),
                                   steps=k_steps.mean().item(), max=max(k_rows.max().item(), k_steps.max().item()),
                                   agree=agree, all=torch.cat([k_rows, k_steps]).mean().item())
        x = summary[scheme, kv]
        print(f"      {scheme}, {str(kv)[6:]} KV: KL vs the fp32 key, rows before {W} {x['before']:.4f}, from it on "
              f"{x['after']:.4f}, decode steps {x['steps']:.4f} (max {x['max']:.3f}); greedy == key {agree}/10",
              flush=True)
        del s                                              # one KV state at a time, and the GPU cache emptied, so
        gc.collect()                                       # the runs do not pile up (8 GB)
        torch.mps.empty_cache()
    m.kv_dtype = torch.bfloat16
    if scheme == "int8":
        del eng, m
        gc.collect()
        torch.mps.empty_cache()
if ("int8", torch.bfloat16) in summary:
    a, b = summary["int8", torch.bfloat16], summary["int8", torch.float32]
    check(f"int8 past the window: KL vs the fp32 key < 0.05 ({a['all']:.4f}; {a['after']:.4f} from position {W}), and "
          f"bf16 KV adds at most 0.01 ({a['all'] - b['all']:+.4f})", a["all"] < 0.05 and a["all"] - b["all"] <= 0.01)
if ("int4", torch.bfloat16) in summary:
    a = summary["int4", torch.bfloat16]
    check(f"int4 past the window: KL vs the fp32 key < 0.15 ({a['all']:.4f}; {a['after']:.4f} from position {W})",
          a["all"] < 0.15)

# 2. the bf16 KV cache out to ~7.6K tokens (INT4 on Metal): the same text, 950 tokens per language, through an fp32
# and a bf16 KV state in lockstep; KL(fp32 KV || bf16 KV) at every 64th row, around the window's edge and at the last
# 8, then 32 greedy tokens from each. No fp32 answer key reaches this far, so this is engine against engine
if ("int4", torch.bfloat16) in summary:
    long_ids = torch.tensor(tok.encode(long_texts.build(HFEnc, 950), add_bos=True))
    N = len(long_ids)
    pick = torch.tensor(sorted(set(range(0, N, 64)) | set(range(W - 6, W + 5)) | set(range(N - 8, N))))
    out, gen = {}, {}
    for kv in (torch.float32, torch.bfloat16):
        m.kv_dtype = kv
        s = m.new_state(N + 40)
        hq = prefill(m, long_ids, s)
        out[kv] = m.head(hq[pick]).float().cpu()
        nxt, g = int(out[kv][-1, :V].argmax()), []
        for _ in range(32):
            g.append(nxt)
            nxt = int(m.forward(torch.tensor([nxt]), state=s)[0, :V].argmax())
        gen[kv] = g
        del s, hq
        gc.collect()
        torch.mps.empty_cache()
    k = kl(out[torch.float32], out[torch.bfloat16])
    same = next((j for j, (x, y) in enumerate(zip(gen[torch.float32], gen[torch.bfloat16])) if x != y), 32)
    check(f"bf16 KV out to {N} tokens (INT4): KL(fp32 KV || bf16 KV) mean {k.mean().item():.4f} (before {W} "
          f"{k[pick < W].mean().item():.4f}, from it on {k[pick >= W].mean().item():.4f}, max {k.max().item():.3f}); "
          f"greedy identical for {same}/32 tokens", k.mean().item() < 0.01)
print(f"      {time.perf_counter() - t0:.0f} s in all")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
