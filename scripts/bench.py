"""Benchmark harness: TTFT, TPOT, prefill/decode throughput, peak memory, and the bandwidth ceiling.

usage: bench.py [--model qwen3.5-0.8b] [--backend cpu|mps|metal|metal-int8|metal-int4] [--weights FILE.qt]
                [--prompt-lens 16,256,1024] [--new 32] [--no-cache] [--batch 1,2,4,8] [--prefill-chunk N] [--paged]
                [--kv-blocks N] [--kv-report] [--trace] [--decide CTX [--options N] [--option-tokens K]]
                [--out docs/bench/x.md]

--batch measures continuous-batching decode: B sequences advance together through model.decode_batch, and the
aggregate rate is B tokens per step. --prefill-chunk N prefills each prompt N tokens at a time (TTFT is the whole
prefill), as the server does; --paged puts the sequence in the model's paged pool (Tiny Aya: the sliding layers'
rings) instead of a contiguous cache, sized for that one sequence, or with --kv-blocks N as serve.py --kv-blocks N
sizes the server's (blocks of every layer). --kv-report prints the KV bytes one sequence holds at 1K-8K positions,
with the pool's own layout and with one table for every layer, and how many sequences fit in a fixed budget. Memory
is the process's physical footprint (CPU and Metal alike, as Activity Monitor shows it), now and its lifetime peak.
--trace (with --paged and one prompt length): ONE prefill in this fresh process, the footprint after every chunk: what
one request adds to a fresh process. A long-running server also keeps what the allocator cached for earlier requests,
which repeated runs in one process (no --trace) measure.
--decide CTX (Metal): one /v1/decide-sized scoring call (decision.score_options) over a CTX-token context with N
options of K tokens, next to a pool of --kv-blocks blocks (or the model's) as the server holds it: the GPU memory peak
(the allocator's own), the seconds, Metal's aborted command buffers (macOS's log) and every score in full, so a change
to decide can be checked bit for bit.
"""
import argparse
import os
import resource
import subprocess
import sys
import time

import torch

sys.path.insert(0, "src")
from engine import load_engine
from tokenizer import Tokenizer

TEXT = ("The history of computing is a story of abstraction. Each generation of engineers built tools that "
        "hid the details of the layer below, so the next generation could think in bigger pieces. ")
BANDWIDTH = 100e9   # M2 unified memory, bytes/s


def peak_rss_mib() -> float:
    """Peak resident set of the CPU process (MiB). Does NOT include memory the Metal driver holds for the GPU."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20      # ru_maxrss is bytes on macOS


def gpu_mib() -> float:
    return torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else 0.0


def footprint_mib() -> tuple[float, float]:
    """This process's physical footprint now and its lifetime peak (MiB): CPU and Metal memory alike (macOS
    proc_pid_rusage, RUSAGE_INFO_V4: after a 16-byte uuid, uint64 fields; phys_footprint is the 8th,
    lifetime_max_phys_footprint the 29th). ru_maxrss misses the GPU's buffers."""
    import ctypes
    import os
    buf = (ctypes.c_uint64 * 64)()
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, buf) != 0:
        return float("nan"), float("nan")
    return buf[2 + 7] / 2**20, buf[2 + 28] / 2**20


def footprint_trace(model, tok, n: int, chunk: int, kv_blocks: int, sync) -> str:
    """Weights made resident with a tiny prompt, then ONE n-token prefill into a paged pool (kv_blocks blocks, or sized
    for the sequence), in chunk-token pieces: the physical footprint after each (MiB) and the peak."""
    model.forward(torch.tensor(prompt_ids(tok, 8)), last_only=True)
    sync()
    weights = footprint_mib()[0]
    probe = model.new_paged_pool(1, 16, max_seqs=1, max_chunk=chunk)
    blocks = kv_blocks or -(-probe.blocks_for(n) // getattr(probe.kv, "n_groups", 1))
    state = model.new_paged_state(model.new_paged_pool(blocks, 16, max_seqs=1, max_chunk=chunk))
    ids, trace = torch.tensor(prompt_ids(tok, n)), []
    for i in range(0, n, chunk):
        model.forward(ids[i:i + chunk], state=state, last_only=True)
        sync()
        trace.append(round(footprint_mib()[0]))
    return (f"{n}-token prefill, chunks of {chunk}, paged pool of {blocks} blocks: footprint {weights:.0f} MiB with "
            f"the weights resident, then after each chunk {trace}; peak {footprint_mib()[1]:.0f} MiB")


def kv_report(model, chunk: int = 512) -> list[str]:
    """KV bytes per sequence from the pool's own arithmetic (exact: units held x unit bytes), next to one table of
    blocks for every layer, at 1K-8K positions, and the sequences that fit in 1.125 / 1.5 GiB. chunk: the longest
    forward (the prefill chunk), which sizes a sliding-window ring."""
    dtype0 = getattr(model, "kv_dtype", torch.float32)
    ring = getattr(model.new_paged_pool(1, 16, max_seqs=1, max_chunk=chunk).kv, "ring", None)
    lines = [f"KV per sequence, prefill chunks of {chunk}"
             + (f" (sliding layers: rings of {ring} blocks)" if ring else ""), "",
             "| KV dtype | positions | one table, every layer (MiB) | pool's layout (MiB) | ratio |",
             "|---|---:|---:|---:|---:|"]
    per_seq = {}
    for dt in (torch.float32, torch.bfloat16) if hasattr(model, "kv_dtype") else (torch.float32,):
        if hasattr(model, "kv_dtype"):
            model.kv_dtype = dt
        pool = model.new_paged_pool(1, 16, max_seqs=1, max_chunk=chunk)
        kv = pool.kv
        n_layers = sum(len(g) for g in kv.groups) if hasattr(kv, "groups") else kv.k.shape[0]
        block_all = 2 * n_layers * kv.k[0, 0].numel() * kv.k.element_size()          # one block of every layer
        for n in (1024, 2048, 4096, 6144, 8192):
            flat, ours = -(-n // 16) * block_all, pool.blocks_for(n) * pool.unit_bytes
            per_seq[dt, n] = (flat, ours)
            lines.append(f"| {str(dt)[6:]} | {n} | {flat / 2**20:.0f} | {ours / 2**20:.0f} | {ours / flat:.2f} |")
    if hasattr(model, "kv_dtype"):
        model.kv_dtype = dtype0
    lines += ["", "| KV dtype | positions | budget (GiB) | sequences, one table | sequences, pool's layout |",
              "|---|---:|---:|---:|---:|"]
    for (dt, n), (flat, ours) in per_seq.items():
        if n in (4096, 8192):
            for gib in (1.125, 1.5):
                fit = (int(gib * 2**30 // flat), int(gib * 2**30 // ours))
                lines.append(f"| {str(dt)[6:]} | {n} | {gib} | {fit[0]} | {fit[1]} |")
    return lines


def metal_aborts(since: str) -> int | None:
    """Command buffers Metal aborted for this process since `since` (it logs each one; torch 2.14 reports none)."""
    abort = "Execution of the command buffer was aborted"
    r = subprocess.run(["/usr/bin/log", "show", "--start", since, "--style", "compact", "--predicate",
                        f'processID == {os.getpid()} AND eventMessage CONTAINS "{abort}"'],
                       capture_output=True, text=True, timeout=300)
    return sum(abort in line for line in r.stdout.splitlines()) if r.returncode == 0 else None


def decide_bench(eng, ctx_len: int, n_opts: int, opt_tokens: int, kv_blocks: int) -> str:
    """One /v1/decide-sized scoring call beside the server's pool: the context's cache and its forks live outside it."""
    from decision import score_options
    model, tok, A = eng.model, eng.tokenizer, torch.accelerator
    pool = model.new_paged_pool(kv_blocks, 16, max_seqs=8, max_chunk=512)
    st = model.new_paged_state(pool)
    model.forward_packed([(torch.tensor([0]), st)])          # warm-up, as Scheduler._warmup
    st.free()
    g = torch.Generator().manual_seed(0)                       # ordinary tokens, clear of the special ids
    opts = [torch.randint(1000, 100000, (opt_tokens,), generator=g).tolist() for _ in range(n_opts)]
    ctx = prompt_ids(tok, ctx_len)
    torch.mps.synchronize()
    A.reset_peak_memory_stats()
    since, t0 = time.strftime("%Y-%m-%d %H:%M:%S"), time.perf_counter()
    scores = score_options(model, ctx, opts, tok.vocab_size())
    torch.mps.synchronize()
    secs = time.perf_counter() - t0
    peak = A.max_memory_reserved() + max(0, torch.mps.driver_allocated_memory() - A.memory_reserved())
    rec = torch.mps.recommended_max_memory()
    return (f"/v1/decide, {len(ctx)}-token context, {n_opts} options x {opt_tokens} tokens, pool of {kv_blocks} "
            f"blocks: peak GPU memory {peak / 2**30:.2f} GiB ({peak / rec:.2f}x Metal's recommended {rec / 2**30:.2f}), "
            f"{secs:.1f} s, Metal aborts {metal_aborts(since)}\n"
            f"scores {[repr(s.logprob) for s in scores]}")


def prompt_ids(tok: Tokenizer, n: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n:
        ids += tok.encode(TEXT)
    return ids[:n]


def timed(fn, sync):
    sync(); t0 = time.perf_counter(); out = fn(); sync()
    return out, time.perf_counter() - t0


_pools: dict = {}                                       # bench_one's paged pools, built once (as the server does)


def bench_one(model, ids: list[int], new: int, use_cache: bool, sync, chunk: int = 0, paged: bool = False,
              kv_blocks: int = 0) -> dict:
    x = torch.tensor(ids)
    if use_cache:
        step = chunk or len(ids)
        if paged:                                           # the server's pool: kv_blocks blocks of every layer, or
            probe = model.new_paged_pool(1, 16, max_seqs=1, max_chunk=step)   # just this sequence's units (rings
            groups = getattr(probe.kv, "n_groups", 1)                           # counted), built ONCE and reused:
            blocks = kv_blocks or -(-probe.blocks_for(len(ids) + new) // groups)   # a new pool per run left the
            if (blocks, step) not in _pools:                                    # last ones in the GPU's cache
                _pools[blocks, step] = model.new_paged_pool(blocks, 16, max_seqs=1, max_chunk=step)
            state = model.new_paged_state(_pools[blocks, step])
        else:
            state = model.new_state(len(ids) + new)

        def prefill():
            for i in range(0, len(ids), step):
                out = model.forward(x[i:i + step], state=state, last_only=True)
            return out
        logits, ttft = timed(prefill, sync)
        nxt = int(logits[0].argmax())
        steps = []
        for _ in range(new - 1):
            logits, dt = timed(lambda: model.forward(torch.tensor([nxt]), state=state, last_only=True), sync)
            nxt = int(logits[0].argmax()); steps.append(dt)
        if paged:
            state.free()                                    # its units and slot back to the reused pool
    else:
        seq = list(ids)
        logits, ttft = timed(lambda: model.forward(torch.tensor(seq), last_only=True), sync)
        seq.append(int(logits[0].argmax()))
        steps = []
        for _ in range(new - 1):
            logits, dt = timed(lambda: model.forward(torch.tensor(seq), last_only=True), sync)
            seq.append(int(logits[0].argmax())); steps.append(dt)
    tpot = sorted(steps)[len(steps) // 2]                       # median decode step
    return {"prompt": len(ids), "cache": use_cache, "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3,
            "prefill_tps": len(ids) / ttft, "decode_tps": 1 / tpot}


def bench_batch(model, tok, B: int, steps: int, sync) -> float:
    """Median seconds per decode_batch step for B sequences (16-token prompts, paged KV)."""
    pool = model.new_paged_pool(B * -(-(16 + steps + 1) // 16), 16, max_seqs=B)   # room for prompt + every step
    states = [model.new_paged_state(pool) for _ in range(B)]
    for st in states:
        model.forward(torch.tensor(prompt_ids(tok, 16)), state=st, last_only=True)
    toks = [0] * B
    times = []
    for _ in range(steps):
        logits, dt = timed(lambda: model.decode_batch(toks, states), sync)
        toks = logits.argmax(-1).tolist(); times.append(dt)
    for st in states:
        st.free()
    return sorted(times)[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--backend", choices=["cpu", "mps", "metal", "metal-int8", "metal-int4"], default="cpu")
    ap.add_argument("--weights", default=None, help="pre-quantized .qt file")
    ap.add_argument("--prompt-lens", default="16,256,1024")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--no-cache", action="store_true", help="also measure recompute-everything decoding")
    ap.add_argument("--batch", default=None, help="comma-separated batch sizes for the decode_batch benchmark")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="prefill N prompt tokens at a time (0: all at once)")
    ap.add_argument("--paged", action="store_true", help="the model's paged pool instead of a contiguous cache")
    ap.add_argument("--kv-blocks", type=int, default=0, help="--paged: the pool's size, as serve.py --kv-blocks")
    ap.add_argument("--kv-report", action="store_true", help="KV bytes per sequence and sequences per budget")
    ap.add_argument("--trace", action="store_true", help="one paged prefill, the footprint after every chunk")
    ap.add_argument("--decide", type=int, default=0, help="a /v1/decide scoring call over this many context tokens")
    ap.add_argument("--options", type=int, default=16, help="--decide: options (the API allows 16)")
    ap.add_argument("--option-tokens", type=int, default=64, help="--decide: tokens per option (the API allows 64)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.new < 2:
        ap.error("--new must be >= 2 (TPOT needs at least one decode step after the first token)")

    eng = load_engine(a.model, a.backend, a.weights)
    model, tok, backend = eng.model, eng.tokenizer, eng.model.b
    sync = torch.mps.synchronize if backend.device.type == "mps" else (lambda: None)
    if a.kv_report:
        print("\n".join(kv_report(model, a.prefill_chunk or 512)) + "\n")
    if a.decide:
        if backend.device.type != "mps":
            ap.error("--decide measures GPU memory: use a Metal backend")
        print(decide_bench(eng, a.decide, a.options, a.option_tokens, a.kv_blocks or eng.kv_blocks or 1024))
        return
    if a.trace:                                             # one request's prefill, as a server sees it
        print(footprint_trace(model, tok, int(a.prompt_lens.split(",")[0]), a.prefill_chunk or 512, a.kv_blocks, sync))
        return

    bench_one(model, prompt_ids(tok, 8), 4, True, sync)                       # warm-up: weights resident, kernels compiled
    weight_bytes = sum(w.nbytes if hasattr(w, "scheme") else w.numel() * w.element_size() for w in model._cache.values())
    ceiling = BANDWIDTH / weight_bytes

    def median_run(n: int, new: int, use_cache: bool) -> dict:
        kw = dict(chunk=a.prefill_chunk, paged=a.paged, kv_blocks=a.kv_blocks)
        bench_one(model, prompt_ids(tok, n), 2, use_cache, sync, **kw)      # warm-up at THIS shape (graphs, allocator)
        runs = [bench_one(model, prompt_ids(tok, n), new, use_cache, sync, **kw) for _ in range(3)]
        return sorted(runs, key=lambda r: r["ttft_ms"])[1]                 # median of 3 by TTFT

    rows = []
    for n in [int(v) for v in a.prompt_lens.split(",") if v]:
        rows.append(median_run(n, a.new, True))
        if a.no_cache:
            rows.append(median_run(n, min(a.new, 6), False))

    pool_note = "" if not a.paged else \
        f", paged pool of {a.kv_blocks} blocks" if a.kv_blocks else ", paged pool sized for the sequence"
    lines = [f"### {backend.name} — {eng.name}, {a.new} new tokens, M2 8 GB",
             "", f"Weights resident: {weight_bytes / 1e9:.2f} GB → bandwidth ceiling ≈ {ceiling:.0f} tok/s "
             f"(every decode step reads every weight once at ~100 GB/s).", f"Memory: peak CPU RSS {peak_rss_mib():.0f} MiB, "
             f"Metal driver {gpu_mib():.0f} MiB, physical footprint {footprint_mib()[0]:.0f} MiB now and "
             f"{footprint_mib()[1]:.0f} MiB at its peak. Each row: median of 3 runs after a warm-up at that prompt "
             f"length{f', prefilled {a.prefill_chunk} tokens at a time' if a.prefill_chunk else ''}"
             f"{pool_note}.", ""]
    if rows:
        lines += ["| prompt tokens | KV cache | TTFT (ms) | TPOT (ms) | prefill tok/s | decode tok/s | % of ceiling |",
                  "|---:|:---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['prompt']} | {'yes' if r['cache'] else 'no'} | {r['ttft_ms']:.0f} | {r['tpot_ms']:.1f} | "
                     f"{r['prefill_tps']:.0f} | {r['decode_tps']:.1f} | {100 * r['decode_tps'] / ceiling:.0f}% |")
    if a.batch:
        sizes = [int(v) for v in a.batch.split(",")]
        lines += ["", f"| batch | step (ms) | per-sequence tok/s | aggregate tok/s | aggregate vs batch {sizes[0]} |",
                  "|---:|---:|---:|---:|---:|"]
        base = None
        for B in sizes:
            bench_batch(model, tok, B, 3, sync)                              # warm-up at this batch size
            step = bench_batch(model, tok, B, a.new, sync)
            base = base or B / step
            lines.append(f"| {B} | {step * 1e3:.1f} | {1 / step:.1f} | {B / step:.1f} | {B / step / base:.2f}x |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        with open(a.out, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
