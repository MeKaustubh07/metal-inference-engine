"""Tiny Aya port, M5 B6/B7: the quantized engine on Metal past the 4096-token sliding window. INT8 and INT4, each with
an fp32 and a bf16 KV cache, against the long fp32 answer key (scripts/golden_aya.py --long, ~4.8K tokens in 8
languages): KL at the key's rows before and after position 4096, the 9 decode steps, greedy agreement (gates as in
tests/test_aya_quant.py, on every run, the decode steps also on their own), the GPU memory each run reaches against
Metal's recommended working set, and that Metal aborted no command buffer (macOS's log). Then the bf16 KV cache's cost
out to ~7.6K tokens (950 tokens per language), engine against engine (fp32 vs bf16 KV, INT4; no fp32 answer key
reaches that far). The fp32 CPU engine against the same key is tests/test_aya_long.py. Needs the gated files, the key
and the .qt files; otherwise SKIP, or FAIL naming the command."""
import gc
import os
import subprocess
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


t0, started = time.perf_counter(), time.strftime("%Y-%m-%d %H:%M:%S")
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


def gpu_peak():
    """This run's GPU memory high point: the allocator's peak heap bytes, plus what Metal holds outside them now."""
    A = torch.accelerator
    return A.max_memory_reserved() + max(0, torch.mps.driver_allocated_memory() - A.memory_reserved())


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
        torch.accelerator.reset_peak_memory_stats()      # the last run ended with empty_cache
        lg = m.head(prefill(m, ids, s)[rows]).float().cpu()
        stp = torch.cat([m.forward(torch.tensor([t]), state=s) for t in key["greedy"][:-1].tolist()]).float().cpu()
        k_rows, k_steps = kl(key["logits"], lg), kl(key["step_logits"], stp)
        agree = int(lg[-1, :V].argmax() == key["greedy"][0]) + int((stp[:, :V].argmax(-1) == key["greedy"][1:]).sum())
        summary[scheme, kv] = dict(before=k_rows[rows < W].mean().item(), after=k_rows[rows >= W].mean().item(),
                                   steps=k_steps.mean().item(), max=max(k_rows.max().item(), k_steps.max().item()),
                                   agree=agree, all=torch.cat([k_rows, k_steps]).mean().item(),
                                   gpu=gpu_peak())
        x = summary[scheme, kv]
        print(f"      {scheme}, {str(kv)[6:]} KV: KL vs the fp32 key, rows before {W} {x['before']:.4f}, from it on "
              f"{x['after']:.4f}, decode steps {x['steps']:.4f} (max {x['max']:.3f}); greedy == key {agree}/10; "
              f"GPU memory {x['gpu'] / 2**30:.2f} GiB",
              flush=True)
        del s                                              # one KV state at a time, and the GPU cache emptied, so
        gc.collect()                                       # the runs do not pile up (8 GB)
        torch.mps.empty_cache()
    m.kv_dtype = torch.bfloat16
    if scheme == "int8":
        del eng, m
        gc.collect()
        torch.mps.empty_cache()
for (scheme, kv), x in summary.items():                    # every run: a broken one must not hide behind the other
    T = {"int8": 0.05, "int4": 0.15}[scheme]
    check(f"{scheme}, {str(kv)[6:]} KV past the window: KL vs the fp32 key < {T} over the key's rows and decode steps "
          f"({x['all']:.4f}; {x['after']:.4f} from position {W}) and over the decode steps alone ({x['steps']:.4f})",
          x["all"] < T and x["steps"] < T)
if summary:                                                # Metal aborts command buffers it cannot fit, and torch
    rec = torch.mps.recommended_max_memory()              # 2.14 then returns garbage silently (src/backend/__init__)
    worst = max(summary, key=lambda r: summary[r]["gpu"])  # 0.95: where MLX starts freeing its cache
    check(f"GPU memory: every run stays under 0.95 x Metal's recommended {rec / 2**30:.2f} GiB (highest: "
          f"{worst[0]}, {str(worst[1])[6:]} KV, {summary[worst]['gpu'] / 2**30:.2f} GiB)",
          all(x["gpu"] <= 0.95 * rec for x in summary.values()))
if ("int8", torch.bfloat16) in summary and ("int8", torch.float32) in summary:
    d = summary["int8", torch.bfloat16]["all"] - summary["int8", torch.float32]["all"]
    check(f"int8: bf16 KV changes the KL by at most 0.01 either way ({d:+.4f})", abs(d) <= 0.01)

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
# Metal logs every command buffer it aborts (torch 2.14 never reads their status, so nothing else says so)
ABORT = "Execution of the command buffer was aborted"
try:
    log = subprocess.run(["/usr/bin/log", "show", "--start", started, "--style", "compact", "--predicate",
                          f'processID == {os.getpid()} AND eventMessage CONTAINS "{ABORT}"'],
                         capture_output=True, text=True, timeout=300)
    rc, err, out = log.returncode, log.stderr.strip(), log.stdout
except subprocess.TimeoutExpired:
    rc, err, out = None, "timed out after 300 s", ""
if rc != 0:                                                 # it needs an admin account and no sandbox
    check(f"Metal aborted no command buffer of this process: macOS's log not readable (log show: {rc}, "
          f"{err[:150]})", False)
else:
    aborts = [line.split(ABORT)[-1].strip() for line in out.splitlines() if ABORT in line]
    check(f"Metal aborted no command buffer of this process (macOS log: {len(aborts)}"
          + (f", e.g. {aborts[0][:90]})" if aborts else ")"), not aborts)
print(f"      {time.perf_counter() - t0:.0f} s in all")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
