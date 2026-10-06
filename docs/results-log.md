# Results log: every measurement, in the order it happened

The raw material for the project write-up: what was measured at each step, what it showed, and what was decided.
Hardware throughout: MacBook Air M2 (Mac14,2: 4P+4E CPU, 8-core GPU), 8 GB unified memory, ~100 GB/s. Commits
are in `git log`; detailed tables live in `docs/bench/` (raw tool output in `docs/bench/raw/`, test output in
`docs/bench/raw/tests/`, the last run with Qwen2.5-0.5B in `docs/bench/raw/tests/2026-09-30-with-qwen2.5/`, numbers
first recorded in working notes in `docs/bench/raw/early_measurements.md`).

## Phase 0 — TypeScript on Node (Aug 30 – Sep 5, commits `bd2b183`, `a3d84d0`)

Hypothesis: build the engine on `@stdlib/blas`, the BLAS routines the author contributes to. (Numbers from the
working notes of 2026-09-05; `raw/early_measurements.md`.)

| measurement (decode matvec 4864×896, prefill GEMM 64×896×4864) | result |
|---|---|
| naive typed-array loop, fp32 | 3.8 ms (2.3 GFLOP/s) |
| `@stdlib/blas-base-sgemv` v0.1.1 | 5.8 ms (slower than the naive loop) |
| prefill: naive vs best `sgemm` layout | 301 ms vs 580 ms |
| `sdot` vs naive dot | 0.77 vs 2.28 GFLOP/s |
| same ops in PyTorch on MPS | 0.15 ms and 0.66 ms (25× and 450× faster than the naive JS loop) |
| weights | JS has no bf16: a 1 GB checkpoint becomes ~3 GB resident; 0.3% of weights are fp16-subnormal |

Loader and BPE tokenizer were correct (51/51 vs HF on the first run). **Decision (Sep 5):** move the numeric
core to Python + PyTorch (bf16 native, a route to hand-written Metal kernels, same language as the answer key).

## Correctness on Qwen2.5-0.5B (W1–W6)

| week | commit | result |
|---|---|---|
| W1 loader | `2f11d14` | mmap + zero-copy bf16 views: open 7 ms, 4/4 integrity checks, peak RSS 215 MB (vs ~3 GB in TS) |
| W2 tokenizer | `e12611d` | byte-level BPE, 51/51 identical to HF `tokenizers` |
| W3 embedding, RMSNorm | `40f4c66` | embedding exact (diff 0), RMSNorm ≤ 2.4e-7 vs HF fp32 (`raw/early_measurements.md`, `raw/tests/2026-09-30-with-qwen2.5/test_ops.txt`) |
| W4 block | `8e74eb4` | RoPE q/k 1.3e-4/1.8e-4 abs (keys reach 130: ~1e-6 relative); layer-0 attention ≤ 4e-6; full block ≤ 3.3e-5 (`raw/early_measurements.md`, `raw/tests/2026-09-30-with-qwen2.5/test_block.txt`) |
| W5 model | `390685e` | 24 layers, logits ≤ 3e-4 vs HF; greedy continuation "Paris" (`raw/tests/2026-09-30-with-qwen2.5/test_model.txt`) |
| W6 sampling, chat, REPL | `c907515` | ChatML template identical to HF; top-k/top-p/temperature/repetition as HF's warpers |
| W6 review | `88555fe` | review workflow: 9 confirmed defects (top-p boundary, padding ids sampled, stream tail) fixed; greedy == HF `generate()` token for token on 5 prompts |

## Speed on Qwen2.5-0.5B (W7–W12)

| step | commit | decode tok/s | note |
|---|---|---:|---|
| CPU fp32 reference | `028caba` | 10.2 | backend protocol, KV cache (W7 commit; 23.8 at a 16-token prompt in the W8 harness, `decode.md`) |
| MPS bf16 (PyTorch ops) | `028caba` | 20.8 | |
| KV cache effect (CPU, 1024 tokens) | `f79f354` | 53 ms vs 1,512 ms per token | ~28× |
| hand-written Metal kernels | `0a36780` | ~41 | matvec, RMSNorm, RoPE, decode attention |
| + fusion alone | | ~37–39 | dispatch count was not the bottleneck |
| + remove `int(positions[0])` sync per layer | | ~68 | 24 forced CPU↔GPU waits per token gone |
| + embedding lookup on GPU | `0a36780` | **~73** | CPU issue 23.1 ms → 1.6 ms per token (72.7 at a 16-token prompt in `decode.md`; the W9 commit's end-to-end run said 66) |
| native Obj-C++ runtime (same MSL) | `7dd7f61` | 63 / 75 / 89 GB/s matvec with cold weights (`decode.md`; the commit's 79–88 GB/s was cache-served) | tiled GEMM correct but 4× slower than MPS → prefill keeps MPS |
| review of W7–8 (56 agents, 18 confirmed) | `63d80d2` | | atomic block allocation, sound benchmark method |
| review of W9–10 (48 agents) | `595b5ce` | | matvec numbers were cache-inflated: 49–83 GB/s with cold weights (commit; `decode.md` quotes 58–83 for the shapes it lists) |
| INT8 (block 32, symmetric) | `fb6cff0` | ~89 (85.0 at 16 tokens; mean of 16/256/1024-token prompts) | 0.52 GB, perplexity 11.30 vs bf16 11.34, greedy identical |
| INT4 symmetric (first attempt) | | | perplexity +28% (14.54), top-1 62% |
| INT4 asymmetric + INT8 embedding | | | perplexity 11.50, but KL 1.85: position-0 "attention sink" damage |
| **INT4 asymmetric + calibrated policy** | `fb6cff0` | **~97** (99.3 at 16 tokens; mean) | 19 tensors kept INT8, 0.41 GB, KL 0.114, top-1 94% |

Details: `docs/bench/decode.md`, `docs/bench/quant.md`.

## The port to Qwen3.5 (W13–W14)

| step | commit | result |
|---|---|---|
| Qwen3.5-0.8B text path | `c8ac5c8` | 24 layers ≤ 4e-6 relative vs HF fp32, logits 6e-5, greedy identical incl. Hindi |
| fused DeltaNet decode kernels | `c8ac5c8` | 17 → 46 tok/s (0.8B, bf16) |
| tokenizer finding | `c8ac5c8` | HF `AutoTokenizer` resolves this checkpoint to `Qwen2Tokenizer` (older regex, splits combining marks); the engine follows `tokenizer.json` |
| review fixes | `4b31841` | exact chat-template rules, 7 missing special tokens, atomic state reservation; goldens with 57- and 126-token prompts (only the 126-token one crosses the 64-token chunk; the commit message says both) |
| 0.8B suite (7 prompts) | | worst layer 3.8e-6 relative, logits ≤ 6.4e-5, top-1 100%, greedy 70/70 = HF; paged hybrid state == uncached (4.3e-5) (`raw/tests/2026-09-30-with-qwen2.5/test_qwen35.txt`) |
| Qwen3.5-2B bf16 vs HF bf16 | `32de01f` | KL 0.0004, 0/190 top-1 flips, greedy 64/70 (one divergence at a 0.015-logit near-tie) |
| 2B INT8 | `32de01f` | 2.00 GB, KL 0.0007, 0 flips, greedy 70/70 |
| 2B INT4 (5 tensors kept INT8) | `32de01f` | 1.41 GB, KL 0.045, 3/190 flips, answers "Paris" |
| bug found by the 2B suite | `32de01f` | quantized prefill dequantized the 248k-row tied head at once (2 GB for the fp32 result alone plus dequantization intermediates → Metal command-buffer error, KL 5.98 / 6.00, 180/190 flips); fixed with row chunks (`raw/early_measurements.md`) |

## Serving (W15–W16)

| step | commit | result |
|---|---|---|
| server + continuous batching | `2278c0e` | batched == sequential greedy for both families (fp32 CPU: tokens identical, logits within 6e-5); preemption == uninterrupted; SSE == non-stream |
| review of W15 (106 agents, 50 findings) | `2278c0e` / `ae478b6` | 9 confirmed (all fixed) + more fixed from the rejected pile: non-stream disconnect never cancelled, tiny temperature crashed a whole batch, queued cancels held queue slots, missing `regex` in requirements (landed with the requirements files in `378d540`) (`raw/early_measurements.md`) |
| bugs found by tests | `2278c0e` | lone-surrogate JSON crashed FastAPI's 422 handler (500); drain lost a request mid-prefill; disconnect only checked when idle |
| **first load test** (2B INT4, 16 req/level, 64 tokens) | | 30.0 / 30.8 / 35.5 / 31.4 tok/s at 1/2/4/8 clients: **batching gave almost nothing** |
| profile of one batch-8 step | | 111 ms linears + 78 ms per-sequence attention/DeltaNet (5.7–9.8 ms per sequence at B = 1–8) (`raw/profiling.md`) |
| kernel experiment: x tile in threadgroup memory | | 2–3× slower at M = 1, 1.1–1.4× faster at M = 8 (M = 2–4 not measured); beaten by 2 rows per SIMD group → not adopted |
| kernel experiment: 4 rows per SIMD group | | 1.4–1.5× at M = 8, slower at M ≤ 2 |
| kernel experiment: 2 rows per SIMD group | | INT8 head at M = 8: 35 → 14 ms; adopted for all M ≥ 2 |
| pooled DeltaNet state (2 dispatches per layer for the batch) + 2-rows kernels | `ae478b6` | batch-8 step 189 → 97 ms (188.6 → 96.5) |
| paged-attention kernel (one dispatch per layer) | `ae478b6` | batch-8 step 97 → 69 ms; per-sequence overhead 0.8 ms |
| decode step, 2B INT4 | `ae478b6` | B=1 26.5 → 20 ms; B=8 186 → 69 ms (43 → 116 tok/s aggregate) |
| batching review (20 agents) | `cc40106` | 1 confirmed (no scheduler test on the pooled state) → added, with defensive fixes: a freed pooled state could alias a reused slot, admission waits for a free state slot, K % 4 guard, max_batch validated |
| prefill profile, 256 tokens INT4 | | 1.67 s of 1.92 s in quantized linears: 0.55 s torch-op dequant + 1.1 s fp32 GEMM |
| one-pass dequant → bf16 + bf16 GEMM | `34b1b59` | 128 → 305 tok/s (256 tokens) |
| MPS GEMM on M2 (M = 256–1024) | | **fp32 2.5 TFLOPS, fp16 2.6, bf16 1.4** (no native bf16 math before M3) |
| one-pass dequant → fp32 + fp32 GEMM | `f9280fa` | **395 tok/s (256), 446 tok/s (1024)**, exact vs the reference path |
| final engine benchmark, INT4 (2 runs) | `f9280fa` | decode B=1 50.1 / 49.7 tok/s; batch 8 115.8 / 112.7; prefill 256/1024: 398/450 and 395/444 tok/s |
| final engine benchmark, INT8 | `f9280fa` | decode B=1 39.2; batch 8 **126.4** (INT8 ahead of INT4 from batch 2 (54.3 vs 53.6, within noise) and clearly at 4–8; likely a compute-bound shared-weight step where nibble unpacking costs more than bytes, not profiled) |
| final engine benchmark, bf16 | | decode 20.6; batch 8 70.5 (3.4×); 1024-token prefill 9.8 s, likely paging (5,177 MiB of GPU memory) |
| **final load test** (same setup as the first) | `f9280fa` (the prefill kernels it adds do not touch these 19–24-token prompts) | 40.3 / 41.5 / 52.9 / 55.9 tok/s at 1/2/4/8 clients; TTFT p50 221 ms → 1.4 s; 0 rejected |
| why the server trails the engine (56 vs 116 at 8) | | lockstep waves (every request 64 tokens) start with 8 back-to-back prefills (~0.29 s each); then 108–110 ms per token vs a 69 ms step + ~12 ms CPU sampling, i.e. ~28 ms per step of unprofiled engine-loop work (`serving.md`) |

## Closing the server gap (2026-09-29/30; `docs/bench/serving.md`, raw in `raw/profiling.md` §7–10)

| step | result |
|---|---|
| re-measure the old server (bbeb6c6, same scheduler as `f9280fa`) | 39.3 / 41.3 / 61.2 / 77.0 and 36.5 / 40.5 / 61.6 / 73.6 tok/s at 1/2/4/8 clients (`raw/loadgen_2b_int4_2026-09-29_before_run*.md`), not 55.9 at 8; the 7 before-runs kept in the two A/Bs below: 67.3–77.6 at 8 clients, 39.8–40.3 at 1 (published: 40.3) |
| profile of a server decode step, batch 8 (timers around the scheduler's methods + GPU synchronize) | HTTP 84.1 ms = 69.2 GPU + 4.1 issue + 8.8 sampling + 0.2 emit + 1.7 rest; in-process 79.4. **The ~28 ms did not reproduce**: the last-admitted request of an 8-client wave decoded at 77–90 ms/token in 9 re-measured runs vs 108–110 on 2026-09-28, a run slower at 4 and 8 clients (cause not established). Gap to 116 tok/s in the final A/B's before-runs (104 vs 69 ms per 8 tokens): ~27 ms one-at-a-time prefill (~1.7 s per wave, ~0.21 s per prompt), ~9 ms per-step work (steps ~78 vs 69 ms; mostly sampling, 8.8 ms in the profiled run): **about three quarters prefill** |
| prefill time vs prompt length, one forward | T = 1 / 23 / 184 / 256: 26 / 222 / 542 / 676 ms: prefill grows slowly with length (23 tokens cost 41% of 184); up to 32 rows the batched kernels re-read every weight once per 8 rows, above 32 the fp32 expansion + GEMM starts high and grows slowly (linears 184 ms at 33 rows, 194 at 64, 336 at 184) |
| packed prefill (`forward_packed`) vs each prompt alone | max \|Δlogit\| 2.9e-6 INT4 Metal, 0 bf16 Metal, ≤ 2.7e-5 fp32 CPU; 43-token packs (GEMM path) ≤ 2.9e-5 |
| bf16 weights above 32 rows: fp32 GEMM instead of bf16 activations (so packing does not change a prompt's numerics) | Qwen2.5-0.5B bf16 perplexity 11.336 → 11.350 (`raw/tests/2026-09-30-with-qwen2.5/test_quant.txt`); Qwen3.5-2B bf16 vs HF unchanged except one near-tie gap 0.015 → 0.019 |
| first load test of packed + chunked prefill | no clear win: 72.3 tok/s at 8 (that session's baseline 77.0); in the server log each 8-client wave's first request was prefilled alone (~20 tokens) and the other 7, arriving a few ms later, together (~155 tokens); that request then ran a step ahead, so the next wave split the same way → **batching window** (default 5 ms, twice that at most) |
| batched sampling with the selection on MPS | saved nothing: 8.8 → 8.4 ms per batch-8 step (HTTP) |
| why: `torch.topk` over 8 × 248,320 | **MPS 4.5 ms for any k (1–64), CPU 1.0 ms**; copying the batch's logits to the CPU 0.48 ms (0.99 for the strided vocab slice) |
| batched sampling on CPU logits | 2.2 ms vs 6.7 ms for the old per-row sampler (microbenchmark); in-process server profile: sampling 6.9 → 3.2 ms per step, batch-8 step 79.4 → 73.3 ms (no HTTP profile of the final code) |
| packed prefill pass, 8 chat prompts (175 tokens) | 0.82 s warm, 1.33 s the first time (one at a time: ~1.8 s; one 184-token prompt: 0.54 s → per-sequence attention/DeltaNet costs ~40 ms per extra packed sequence) |
| A/B 1 (intermediate: selection on MPS), 10 runs, 3 excluded by one rule (8-client TPOT p50 > 120 ms; in each the slowdown began mid-run; in the one run with CPU samples, macOS background services took the top CPU slots while the server's own CPU use fell) | 72.4 → 88.1 tok/s at 8 clients (`raw/loadgen_2b_int4_2026-09-29_ab.md`) |
| reviews (3 rounds, 2-vote verification) | round 1 (14 agents): 2 confirmed: one request's large top_k slowed the whole batch's sampling → per-row fallback above 256; a bf16 pack over 32 tokens differed from the prompt alone → fp32 GEMM. Round 2 (16 agents): 5 confirmed (two of them the same issue): a failed batch retried rows with already-advanced seeded generators → per-row isolation; a cancelled request behind the chunk budget kept its blocks → all cancelled ones finish first; parity and tie tests too weak → strengthened, mutation-checked. Round 3 (8 agents): `--batch-wait-ms inf` killed the engine thread on the first request → bounded 0–1000; a cancel during the window counted as an arrival (split vote) → arrivals counted |
| **A/B 2, final code, 8 runs (ABBA + BAAB), none excluded** | **77.0 → 94.3 tok/s at 8 clients** (ranges 76.8–77.6 vs 92.5–95.0); TTFT p50 1,056 → 838 ms, p95 1,710 → 892; TPOT 91.5 → 72.6 ms; 4 clients 62.1 → 69.5, 2 clients 42.0 → 45.3; 1 client unchanged (TTFT +8 ms; the window costs a lone request ~6 ms in-process, the rest not isolated) (`raw/loadgen_2b_int4_2026-09-30_ab_final.md`) |
| what is left of the gap to 116 | lockstep waves: 0.82 s packed prefill (which yields the first tokens) + 63 × ~73 ms steps ≈ 5.4 s per 512 tokens ≈ 94 tok/s; prefill ~15%, sampling ~4% |
| tests | 14 suites pass; `test_sampling` 23 (batched == per-row on CPU and MPS over 300 random batches), `test_prefill` 18, `test_server` 45 |

## Deployment tour (2026-09-30; `docs/bench/serving.md`, raw in `raw/deployment_tour_2026-09-30.md`)

| step | result |
|---|---|
| server run by hand: health, ready, metrics, chat, completions, streaming, seeds, validation | all as designed |
| load test at 8 clients against the deployed server | 94.8 tok/s, TTFT p50 817 ms (A/B: 94.3) |
| 80 simultaneous requests | 72 accepted (8 in flight + 64 queued), 8 × 429 with Retry-After: 1; all 80 clients dropped → cleaned up within ~1 s |
| SIGTERM mid-stream | 150-token stream finished with [DONE], new connections refused, exit 2.9 s after the signal |
| **found: idle server loses its weights** | first token 0.24 s back to back vs 0.69–0.81 s after 1–4 min idle; 1.4–2.2 GB paged back in (28 MB when warm). Weights are Metal buffers in unified memory, compressed by macOS while idle |
| fix tried: Metal residency set + requestResidency | no effect (in-process prefill after 60 / 180 s idle: 494 / 617 ms) |
| **fix: mlock the weight buffers** (`src/backend/pinning.py`) | in-process prefill after 60 s idle 438 / 396 → 229 / 235 ms, after 3 min 530 / 433 → 264 / 278 ms (warm 186–190); through the server, after 4 min idle 0.512 / 0.668 → 0.323 / 0.447 s (warm 0.25). KV/state pools (~0.5 GB) not locked yet |
| found: queue gauges refreshed once per engine step | peak `waiting` 0 during the load test → read at scrape time |
| found: request log `finish_reason: null` for cancelled running requests | 17 of 73 in the burst → reason as sent to the client; review caught stop-string completions logged `cancelled`, fixed |
| also seen | first request of a new prompt length +0.1–0.2 s (MPS shape compilation), not fixed |
| Stage 7 (2026-10-01): the CPU container, Qwen3.5-0.8B fp32, Docker Desktop VM 4.80 GiB | ready 50.7 s, healthcheck healthy at 52.9 s, 3.49 GiB; first request: first token 10.1 s, 56 tokens in 510 s (**0.11 tok/s**; natively on the Mac's CPU 11.7) (`raw/docker_2026-10-01.md`) |
| why | not denormals (0 of 5,050,368 state values; flushing them changes nothing), not threads (1 thread: 53 s/token, 8 threads: 18 s/token). At 1 thread a step reads ~3 GB of weights at ~60 MB/s, a storage rate: the weights do not stay resident in the VM. Conclusion: the image is for Linux hosts and CI; on a Mac, run natively |
| tests | `test_kernels` 26, `test_server` 48, `test_prefill` 18; with the server fixes reverted, the 2 new server checks fail |

## Container (Docker Engine 29.5.3 in Docker Desktop, 3.83 GiB VM, 8 vCPU, arm64; transcript in `raw/early_measurements.md`)

| check | result |
|---|---|
| `docker build` | 73 s, image 1.37 GB (python:3.14-slim + CPU torch wheel) |
| Qwen2.5-0.5B, CPU fp32 | ready in ~12 s, 2.4 GB, answers "Paris.", healthcheck healthy, ~4 tok/s decode (64 tokens in 15.7 s) |
| `docker stop` mid-stream (120 tokens) | stream completed with [DONE], lifespan shutdown, exit at 26.8 s (< 30 s stop timeout) |
| Qwen3.5-0.8B, CPU fp32 | output " Paris.\nThe capital of" (same as the HF golden), but peaks at 3.54 of 3.83 GiB and pages: 6 tokens in 114 s → give the VM more memory (≥ 6 GB suggested, untested) |

## Comparison with llama.cpp and MLX-LM (Qwen3.5-2B, same machine, one engine at a time)

| metric | ours INT4 | ours INT8 | llama.cpp Q4_1 | llama.cpp Q4_K_M | llama.cpp Q8_0 | MLX-LM 4-bit |
|---|---:|---:|---:|---:|---:|---:|
| decode, 1 stream (ours: `decode_batch` step; llama.cpp: llama-bench / batched-bench) | 50.1 | 39.2 | 42.3 / 53.3 | 32.5 | 28.6 / 38.6 | 69.9 |
| prefill 256 / 1024 | 398 / 450 | 389 / 443 | 601 / 452 | 445 / 375 | 513 / 438 | 499 / 448 |
| batch 8 aggregate | 115.8 | 126.4 | 61.9 | — | 66.1 | 73.3 |

llama.cpp build 11146 (Homebrew), bartowski GGUFs; MLX-LM 0.31.3, mlx-community 4-bit. Analysis: `docs/bench/compare.md`.

Load-test and benchmark tables: `docs/bench/serving.md`, `docs/bench/qwen35.md`, `docs/bench/compare.md`.

## Removing the bring-up model and the baselines (2026-09-30)

Disk space for the next model (Aya): the served Qwen3.5-2B and its fp32 reference twin Qwen3.5-0.8B stay, everything
else goes. `4f9581b` is the last commit with Qwen2.5-0.5B.

| step | result |
|---|---|
| removed from disk (never in git) | `models/qwen2.5-0.5b` (1.8 GB), `tests/golden/` (Qwen2.5 HF answer keys, 19 MB), the comparison baselines `models/gguf` (4.5 GB) and `models/qwen3.5-2b-mlx-4bit` (1.6 GB; `docs/bench/compare.md` names their repos): ~7.9 GB |
| removed from the repo (in git history) | `src/models/qwen2.py`, `src/inspect_model.py` (its loader integrity check moved to `test_qwen35`), `scripts/golden_layers.py`, `tests/golden_tokens.json`; suites `test_ops`, `test_block`, `test_model`, which only covered the Qwen2 bring-up (`test_qwen35` checks every Qwen3.5 layer, the final norm and the logits against HF fp32) |
| moved | Qwen2.5 INT4 policy and its calibration data → `raw/quant_policy_qwen2.5-0.5b.json`; the last run of the 14 suites → `raw/tests/2026-09-30-with-qwen2.5/` |
| ported to Qwen3.5-0.8B | `test_tokenizer` (+ the 2B's tokenizer files byte-equal to the 0.8B's), `test_sampling`, `test_cache` (hybrid byte accounting; 1- and 2-token prefills from an empty DeltaNet state; MPS fp32/bf16 and Metal, 57 tokens on Metal), `test_paged` (running out of blocks or DeltaNet state slots), `test_kernels` (0.8B shapes, d = 256 decode attention, the paged-attention kernel, fp32-weight RMSNorm), `test_quant`, `test_server` (HTTP/scheduler checks on the hybrid model; UTF-8 holdback fed a split emoji), `test_prefill`; `test_qwen35` also checks Metal prefill top-1 at non-tied positions |
| first run after the removal | 11 suites, 220 checks passed, 2 failed (`test_quant`: the two INT4 thresholds below); tokenizer 66/66 cases |
| 0.8B on Metal vs HF fp32 (7 prompts; perplexity on one English paragraph) | bf16 1.51 GB, perplexity 12.855, KL 0.0000, top-1 100%, 0/183 flips at non-tied positions; INT8 0.80 GB, 12.844, KL 0.0013, 96%, 0/183; INT4 0.58 GB, 14.354 (+11.7%), KL 0.215, 87%, 5/183; INT4 + policy (10 tensors INT8) 0.60 GB, 13.856 (+7.8%), KL 0.068, 88%, 1/183 (`raw/tests/test_quant.txt`) |
| the 2 failures: thresholds re-based | both were set on Qwen2.5-0.5B, where INT4 cost +1.3% perplexity (`raw/tests/2026-09-30-with-qwen2.5/test_quant.txt`). Plain INT4 perplexity bound 5% → 15% of bf16, a regression guard above the measured +11.7%, plus a new check that the calibrated policy lowers it. Calibrated INT4 "mean top-1 agreement ≥ 90%" (88% here, pulled down by near-ties on the 4- and 10-token prompts) → the served 2B's gate: KL < 0.15 and top-1 flips ≤ 10% of non-tied positions (1/183) |
| full run, archived with `run_tests.py --save` | 11 suites, **223 checks passed**, 440 s (tokenizer 0.3 s, sampling 13.4, cache 76.5, paged 29.2, kernels 9.0, native 2.5, quant 64.3, qwen35 34.0, qwen35_2b 52.3, server 97.2, prefill 61.4) (`raw/tests/run_tests.txt`) |
| Metal bf16 prefill vs HF fp32 | 0 top-1 flips at the 183 non-tied positions of 218; greedy 70/70 (`raw/tests/test_qwen35.txt`) |
| cache on bf16 backends (budget: 1.5 × the backend's max logit error vs HF fp32) | MPS bf16 cached vs uncached 0.19 within 0.29, argmax equal at all 8 non-tied positions of 10, greedy 10/10; Metal 0.00 within 0.00 at 10 and 57 tokens (`raw/tests/test_cache.txt`) |
| Dockerfile default | `qwen3.5-0.8b` on the CPU (was `qwen2.5-0.5b`); needs a VM of ≥ 6 GB, not yet measured |

## Order-invariant decisions (2026-10-02; `docs/bench/decision.md`)

Inspired by a livestream that made Qwen3 choice-order invariant with a custom attention mask and position ids (JEV
decision networks). On the hybrid model the recurrent layers have no mask, so each option gets a fork of the context's
state instead.

| step | result |
|---|---|
| `fork()` (KV + DeltaNet state + conv tail) | a fork continues exactly like the original; feeding it leaves the original untouched (0.0) |
| forked scores vs each option alone | 5.3e-5 (CPU fp32), 5.7e-6 (Metal INT4) |
| shuffled options | scores and embeddings bit-identical on CPU; on Metal not at first (the batched kernels group rows, so a row's rounding depends on where it lands) → options packed in a canonical order: bit-identical on Metal too |
| first dataset run (probabilities normalized in the caller's order) | 36/40 shuffled questions identical → normalize over the sorted scores: 40/40 |
| review (41 agents): options scored as prefixes ("1" credited with "10") | fixed first with an end-of-turn token: it measured terseness (Lyon "is" the capital, P(yes) 0.57) → replaced with a word-boundary term: "7 + 3" picks "10" (P 0.981 vs 0.003) |
| review: unchunked context prefill, all queued decisions run back to back, abandoned decisions still run, failed jobs kept tensors alive, jobs invisible to `/ready` | chunked prefill; stepwise jobs between decode steps; cancel on disconnect; tracebacks cleared; jobs counted |
| zero-shot accuracy, Qwen3.5-2B INT4, 300 questions of `avbiswas/bev-decision` | choice 44.0% (random 24.1%, first option 29.0%); boolean 73.0% (majority 70.0%); score 33.0% exact, 68.0% within one (`raw/decision_eval_2026-10-02.md`) |
| cost vs re-reading the context per option | 2.7-3.7× with 4 options, 6.4-14.0× with 16 (89-1,279-token contexts) (`raw/decision_cost_2026-10-02.md`) |
| discarded | a first cost run with an unchunked baseline (44.7× / 191.9× at 1,279 tokens): ~45 s per forward that did not reproduce (3.02 s unchunked, 3.09 s chunked) |
| tests | `test_decision` 24 checks |

## Porting Tiny Aya, M1: everything before the model (2026-10-04/05; plan in `docs/tiny-aya-plan.md`)

| step | commit | result |
|---|---|---|
| research | — | 4 reports, each checked by an adversarial verifier, then a check of the plan: the architecture recounts to exactly 3,349,227,520 parameters; the official config is hash-verified |
| download | — | revision `af89d219`, pinned; both shards' SHA-256 equal their Hugging Face LFS hashes |
| tokenizer | `86e502d` | every Isolated Split, NFC only when the file asks, BOS from the post-processor. The old code kept only the first regex's matches: for Tiny Aya, digit runs only (4/286 cases in the research probe). Now 185/185 vs HF for both models (Qwen's 66 earlier cases unchanged) |
| config, sharded loader | `034ec7a` | `Cohere2Config` == transformers on 13 fields; the 290 tensors bit-identical to safetensors' own reader |
| chat template | `02f13e3` | the model's own Jinja template, rendered as transformers renders it: == `apply_chat_template` on 10 conversations x 2, text and ids; a one-line question is 372 prompt tokens, ~360 of them the fixed preamble |
| registry, BOS policy | `2f664f4` | stops 3 / 6 / 261001, the card's sampling, a 4096 cap (the sliding window); raw prompts get BOS, chat prompts and decision options do not |
| review fixes | after `2f664f4` | `\b` with Oniguruma's word characters (a Persian number before ZWNJ, `۱۹۷۰‌ها`, was grouped differently); template errors 400 not 500; `enable_thinking` refused for Tiny Aya (it returned an empty reply); legacy rope `"type"` and invalid windows refused; vocab and stop-id guard at load; a licence guard that catches rewrapped copies. Tokenizers 191/191, `test_aya` 40/40 |

## Porting Tiny Aya, M2: the model on the CPU (2026-10-05)

| check (`tests/test_cohere2.py`, random 8-layer models, 8 query / 2 KV heads, fp32 CPU) | result |
|---|---|
| every layer, final norm and logits vs HF `Cohere2ForCausalLM` | within 1.5e-6 relative; with eps 1.0 (so every norm's eps matters) 2.6e-6 |
| interleaved RoPE by reordering q/k rows at load (no new kernel) | exact up to float rounding; without the reordering the logits are off by 75% |
| greedy, cached == uncached, paged == contiguous, batched / packed == alone | 12 greedy tokens identical to HF; 2.9e-6; bit-identical; 2.5e-6 |
| the sliding-window guard (window 16) | 16 positions exact; HF's window starts to matter at 17 (logits move 6.6%); fresh, continuing, packed and batched requests past it are refused with no state moved and no KV block taken |
| review: 3 lenses + 3 verifiers; 63 planted bugs | no model bug; 52 planted bugs caught, 11 missed (3 harmless); new checks catch the 8 others; untied heads refused; Cohere2's fused qkv joins the INT4 policy groups; the scheduler applies the model's own length cap |

## Porting Tiny Aya, M3: the real model, fp32, in 8 GB (2026-10-05)

The full model in fp32 is 13.4 GB; one layer is 0.31 GB. Both sides run one layer at a time.

| step | result |
|---|---|
| answer key (`scripts/golden_aya.py`): transformers' own `Cohere2DecoderLayer`s built one at a time from the safetensors files, its mask helpers, rotary embedding and `DynamicCache`; all prompts in lockstep | on random models bit-identical to the full transformers model (also on bf16 shards, with the head in slices and prompts crossing a 16-token window), greedy == an argmax loop over the full model; on Tiny Aya 9 prompts (826 tokens) and 10 greedy tokens each in 212 s, peak RSS 2.68 GB |
| first attempt | each prompt reloaded all 36 layers at every step (90 passes over 6.7 GB on a Mac already swapping); stopped after ~3 min and rewritten in lockstep (~10 passes) |
| engine, `Cohere2Model(stream=True)`: weights widened per use and dropped, head in 32,768-row slices; one packed prefill, then batched decode | ids from the engine's tokenizer and chat template == HF's on all 9 prompts; every layer within 6.5e-6 relative (worst single token 9.4e-6), logits 5.7e-6 (worst token 9.2e-6), chat (377-token prompts) final hidden state of every position 1.5e-5 and last-8 logits 9.7e-6 per token, all 81 decode steps' logits 1.4e-5; greedy identical on 9/9 (English, Hindi, Arabic, Chinese, Swahili, Python code, two chats); 178 s, peak RSS 2.69 GB |
| M3 review (2 lenses + 2 verifiers) | no bug; the fast suite now exercises streaming on bf16 weights (with a record of what is widened), packed captures and the answer key on bf16 shards with the head in slices and prompts crossing the window; the real-model test compares an explicit list of 38 tensors, the embedding bit for bit, per-token errors (the BOS token's activations are up to 50x the others', so whole-tensor errors could have hidden theirs; they did not: per token is ~1.5x) and every decode step's logits; README credits Tiny Aya (licence, AUP, not affiliated, citation) |
| what Tiny Aya says | "The capital of France is" → " Paris."; Arabic → Cairo; Chinese → Beijing; Swahili → Nairobi; a one-word chat answer → "Paris." then `<|END_RESPONSE|>` (the first stop token a chat turn emits) |

## Porting Tiny Aya, M4: INT8 / INT4 on Metal (2026-10-05)

| step | result |
|---|---|
| memory | `quantize()` in whole-row chunks of 2^24 weights (bit-identical; the 262,144-row embedding needed ~6.5 GB of fp32 temporaries in one piece); `save_qt` writes the header first, then one tensor at a time (byte-identical to the old writer on synthetic files and on Qwen3.5-0.8B INT4 with its policy, 4x faster) |
| files | INT8 3.56 GB in 27 s, INT4 2.34 GB in 25 s, ~3 GB peak RSS (mostly the checkpoint's mapped pages) |
| kernels | Metal LayerNorm (two threadgroup reductions) within 6.2e-6 of `ops.layer_norm` (inputs offset by 100); decode attention 16/4 heads x 128 to 4096 positions 1.7e-7; paged 3.6e-7 |
| quality vs the M3 fp32 key, 169 positions (all raw positions, last 8 of each chat, every decode step) | INT8: KL 0.0006, top-1 flips 0/116, greedy 90/90 tokens. INT4 (calibrated): KL 0.105, 0.054 at the 150 served positions (in-template positions and decode steps from the reference's own stop token on left out), 9/116 flips, greedy 44/90 |
| where INT4 differs most | 5 of the 6 positions with KL > 0.5 are after the model's own `<EOS_TOKEN>` or inside the chat template, where the reference is unsure (top probability ~0.5) and no server would use the prediction; the sixth is a real miss mid-sentence in Swahili. Without those 6, mean KL 0.050 (Qwen3.5-2B INT4: 0.045 vs HF bf16, a per-prompt mean over prompt positions only, so a near but not identical statistic) |
| calibration (`--baseline int8`: every tensor INT8 from the checkpoint, one at a time rebuilt in INT4) | 33 min for 144 tensors; 1 above 0.005 nats (layer 0's `down_proj`, +0.012), median 0.0003; damages sum to 0.073 nats, the top 10 hold 36%, the MLP 80%. Unlike Qwen3.5 (a few o_proj / down_proj at position 0), Tiny Aya's INT4 error is spread thinly: the policy moves KL only 0.109 -> 0.105 |
| decode speed, single stream, fp32 KV | INT8 19-21 tok/s, INT4 29-32 tok/s over three runs (estimate ~30) |
| review (4 areas, each finding checked by a skeptic) | the LayerNorm kernel clean; fixed: calibrate_quant's scheme restore broke the bf16 (Qwen3.5) path (smoke-tested again), its missing-goldens hint, `save_qt` now writes to a temporary file replaced only when complete and `QtFile` refuses a truncated file at open (checked with an interrupted synthetic write), the served-position mask (BOS and each chat's last position were wrongly left out, post-stop decode steps wrongly kept), the INT4 hint without `--policy`, the quantize.py docstring |

## Porting Tiny Aya, M5 part A: bf16 KV cache (2026-10-05)

| step | result |
|---|---|
| kernels | both decode-attention kernels are templates over the KV type (float, bfloat), instantiated under the old names and `_bf16`; on a bf16 cache they are bit-identical to the fp32 kernels on the widened copy (Tiny Aya 16/4 x 128 to 8192 positions, Qwen 8/2 x 256 and 14/2 x 64, paged with scrambled tables). `compile_shader` does not check argument types (a bf16 tensor in a `float*` kernel is read as garbage without an error), so the backend picks the kernel by dtype and refuses any other |
| speed | at 4096 positions the bf16 kernel takes 807 us vs 849 us for fp32: the kernel is limited by its parallelism (one threadgroup per query head), not by reading K/V, so bf16 saves memory, not decode time; splitting each head's keys over several threadgroups is a later performance step |
| per model | `kv_dtype` on the registry entry: Tiny Aya bf16 (72 KiB per token instead of 144), Qwen3.5 fp32 |
| tiny random model, CPU | bf16 KV: half the bytes, logits within 5.9e-3 of fp32 KV, same argmax as HF at 23/23 positions (KL 6.6e-6); paged == contiguous exactly, batched == alone, forks exact. Chunked vs one pass is not exact in bf16: 18 of 11,776 cached values land one bf16 step apart, because fp32 rounding differences (other GEMM shapes) push values across a bf16 rounding boundary (1.1e-4 on the logits) |
| real model vs the M3 key (169 positions, up to ~386 tokens) | INT8: KL 0.0006 with fp32 KV, 0.0006 with bf16 (+0.0000), 0 flips, greedy 90/90 in both. INT4: 0.1050 vs 0.1036 (-0.0014; +0.0001 at served positions), 9 flips and 44/90 in both. Decode speed unchanged (INT8 ~21, INT4 ~30 tok/s). Long contexts (4K-8K) are measured in part B |

## Porting Tiny Aya, M5 part B1-B5: the sliding window (2026-10-05)

| step | result |
|---|---|
| semantics | transformers' `sliding_window_overlay`: a query at position q sees key k iff q - W < k <= q (W keys, itself included); Tiny Aya W = 4096 on 27 of 36 layers |
| reference attention | `ops.attention(window=W)`: drops keys no query of the chunk sees, ORs a band mask; == a float64 oracle written from absolute positions on all 18,320 small cases (W 1-10, chunks 1-8, starts 0-24, every valid key suffix), worst 3.8e-6; W - 1 and W + 1 are caught for every W in 1-10 (19 off-by-one windows); no window, or W >= every position, is bit-identical to before |
| Metal decode | paged kernel: loops start at s0 = (S > W ? S - W : 0); single-sequence decode: a view of the last W keys. fp32 and bf16, 16/4 x 128, at W - 1, W, W + 1, 2W + 3 and 4095-8192: within 4.8e-7; paged with every block before the window NaN: finite, within 6.6e-7 |
| RoPE drift (found by the M5 design review) | the kernel computed its own pow() for each pair's frequency; one ulp off, and the angle position x frequency grew the error with the position: Tiny Aya 2.0e-5 relative at 386, 2.0e-4 at 4095, 4.1e-4 at 8191 (Qwen3.5 3.6e-4 at 8191). It now reads transformers' fp32 table (computed once on the CPU): ~1e-7 at every position, 8e-8 at 31999. Qwen3.5-2B's quality numbers are unchanged to 4 decimals (KL 0.0004 / 0.0007 / 0.0454) |
| model | sliding layers read only their window's suffix of the KV cache (`read(layer, end, lo)`), full layers everything; the cap is config.max_position_embeddings (8192), refused atomically before any state moves. Tiny random Cohere2 with W = 8 over 40 positions vs HF: every layer 2.2e-6; chunked prefill (6 schedules incl. chunks ending at the window, longer than it, token by token) on contiguous and paged (blocks of 4 and 3) 5.2e-6; greedy 20 tokens to position 26 identical; batched, packed and forks exact; the sliding layers' blocks before the window NaN-poisoned: output unchanged (0.0) |
| 8K memory | attention scores scaled, masked and softmaxed in place (one [16, T, S] buffer instead of three; bit-identical on the CPU); generate.py prefills in 512-token chunks; the server's cap defaults to the model's own |

## Porting Tiny Aya, M5 part B6-B7: the real model past the window, and the 8192 cap (2026-10-05)

| step | result |
|---|---|
| long text | `scripts/long_texts.py`: ~600 tokens each of 8 public-domain texts (Project Gutenberg en, fr, de, es, zh, ja pinned by SHA-256; Wikisource hi, ar at pinned revisions, extracted text pinned by SHA-256), downloaded on demand into the gitignored `models/long_texts/`, never committed. 4,804 tokens with BOS; 950 per language gives 7,603 |
| long answer key | `golden_aya.py --long`: transformers' own layers one at a time, the prompt in 512-token chunks over one `DynamicCache` (proved exact past the window on a tiny model: chunks 5, 3, 17, 15 vs HF's full forward within 1.8e-6); final norm of every position, logits at 93 rows (every 64th, 4090-4100, the last 8), 10 greedy tokens. 356 s, 2.4 GB peak, 146 MB (gitignored) |
| engine, fp32 CPU (`test_aya_long`) | the engine's tokenizer gives the key's 4,804 ids; final norm worst row 1.3e-5 before position 4096 and 1.6e-5 after (median 1.8e-6); logits 6.1e-6; 9 decode steps 2.5e-6; greedy 10/10. Sensitivity: the same tokens from 4096 on with the window turned off: worst row 4.1 (median 0.45), so the check can tell a window from none |
| INT8 / INT4 on Metal (`test_aya_long_quant`) | INT8 KL vs the fp32 key 0.0005 before 4096, 0.0008 after, decode 0.0009, greedy 10/10; with a bf16 KV cache 0.0006 / 0.0009 / 0.0009, greedy also 10/10. INT4 0.0889 / 0.1057 / 0.1648, greedy 5/10; with bf16 KV 0.0889 / 0.1056 / 0.1649, greedy also 5/10 |
| bf16 KV out to 7.6K tokens (INT4) | fp32 KV vs bf16 KV in lockstep over 7,603 tokens: KL 0.0000 before and after 4096, greedy identical for 32/32 tokens |
| cap | Tiny Aya's registry max_model_len 4096 -> 8192 (its config's; prompt + output, max_tokens stays <= 4096) |
| one decode failure (still unexplained; see part C) | the first run had the CPU and Metal parts in one process while the Mac was swapping hard (9.5 GB of swap, Chrome, VS Code and a VM open; 9.2 GB peak footprint). In it, INT8 with a bf16 KV cache gave garbage decode steps (KL 5.2, greedy 1/10) while its prefill was right (KL 0.0009). Not reproduced then: 3 fresh-process runs, the split Metal suite, and the same one-process sequence with 1.9 GB of swap all give KL 0.0009 and 10/10. Hypothesis then: under memory pressure a GPU dispatch fails without an error and leaves its output buffer's old contents. Part C found the same symptom caused by GPU memory exhaustion, but macOS's unified log, where Metal records every aborted command buffer, has none before 2026-10-06 01:59, so this run's cause is not established. The Metal part runs in its own process and frees each KV state and the MPS cache between runs (8 GB) |

## Porting Tiny Aya, M5 part C: the sliding layers' blocks reused (2026-10-06)

| step | result |
|---|---|
| allocator | `BlockAllocator` knows which blocks are out: a double free, a block it never handed out, -1 or an id out of range raises (before, one block could silently go to two sequences); frees stay LIFO |
| grouped pool | `GroupedKVPool`: layers in groups of gcd(9, 27) = 9 (1 full, 3 sliding); the unit is one 16-token block of one group; one allocator for all groups; `num_blocks` keeps its meaning (blocks of every layer, x4 units). A sequence's full group gets a growing table, each sliding group a ring of R = ceil((W + max_chunk + bs - 2) / bs) units (289 at W 4096, chunks of 512), logical block b in slot b % R, each slot tagged with the block it holds: a read of a block the ring no longer holds raises. What a sequence holds only grows, so the scheduler's up-front reservation still means a prefill never runs out halfway |
| exactness (tiny window-8 model, `test_window`) | 40 positions in chunks <= 3 (rings of 4 blocks, every slot reused 2-3 times, NaN-poisoned when it takes a new block) == HF within 2.1e-6, greedy 10 tokens past it == HF; batched decode of 3 sequences through their rings == each alone (2.6e-6); a chunk the ring cannot hold refused alone and in a pack, nothing moved; 150 random chunks over 5 (window, block, chunk) settings: every read == what was written, units held exact |
| scheduler | admission, never-fit and preemption count pool units (`blocks_for`, headroom of one unit per group per sequence); gauges `kv_unit_bytes`, `kv_bytes_held`. The real scheduler on the tiny model with chunks of 3 and 56 units for 4 requests: 1 preemption, every request's 30 tokens == HF greedy, every unit and state slot back. Qwen3.5: unchanged (`test_server`, `test_prefill`, `test_decision`, `test_paged` pass) |
| KV bytes per sequence (exact: units x unit bytes) | bf16: 72 / 144 / 288 MiB at 1K / 2K / 4K, as before (below the window nothing can be reused); 352 MiB at 6K (432 with one table for every layer), **388 MiB at 8K (576)**: 0.67x, and 3x less than the fp32 single table (1,152 MiB). In 1.5 GiB: three 8K sequences instead of two |
| one buffer for K and V | PyTorch's MPS allocator puts a request of 10-512 MiB that fits no free space in a new 1 GiB heap (later requests share it) and gives one of 512 MiB or more a heap of its own size: K and V as two 432 MiB tensors cost 1,024 MiB (V fits in the heap K made), one 864 MiB buffer costs exactly 864. Every KV cache (contiguous, paged, grouped) now allocates K and V as the two halves of one buffer (`state.kv_buffers`): a cache of 512 MiB or more costs what it holds (Tiny Aya's 768-block pool); a smaller one, such as Qwen3.5's default pool (384 MiB), still shares a 1 GiB heap |
| server pool for Tiny Aya | registry `kv_blocks=768` (864 MiB of bf16 KV: two 8K sequences, rings counted), the default of `create_app` / `serve.py` for this model (Qwen3.5 keeps 1024). The M5 review found the first footprint numbers came from a pool sized for one sequence, not the server's |
| 8K prefill footprint, Metal INT4, 512-token chunks (`bench.py --paged --kv-blocks N --trace`: one prefill in a fresh process) | weights resident 3.0 GB; with the server's 768-block pool, flat at 3,870 MiB for 4 chunks, then one +1 GB step at chunk 5: PyTorch's MPS allocator puts every tensor of 10-512 MiB in a 1 GiB heap, and the first long-context temporary that fits nowhere else creates one (re-measured with the low watermark below: the same, 4,882 MiB, since the process is under 4.0 GiB when the heap is made), **peak 4,880 MiB**, under the 5 GiB bound; with a pool sized for the sequence 4,097-4,103 MiB (its 1 GiB heap absorbs the working set). That is one request in a fresh process. A long-running server also keeps what the allocator cached for earlier requests: several 8K runs in one process (the pool reused) reached 5,569 MiB of GPU memory before the low watermark below, over the 5.33 GiB Metal recommends, and macOS's log holds 15 Metal aborts from that process. With the watermark, the same multi-run case: INT4 4,553 MiB (0.83x), no aborts; INT8 5,609 MiB (1.03x), no aborts while nothing else used the GPU, TTFT 99.6 s vs INT4's 45.8 s. INT8's single 8K prefill grows to a 5,919 MiB footprint with the context (heaps sized to requests), so at 8K on 8 GB it is over the limit by itself: long prompts there want INT4 (the registry comment says so) |
| 8K timings (quiet machine) | TTFT ~49 s for 8,000 tokens (~160 tokens/s): every 512-token chunk widens all INT4 weights to fp32 for the GEMM. Decode at 8K ~6 tok/s on one sequence through `forward` (decode attention runs one threadgroup per query head over 4K-8K keys, and the single-sequence paged path gathers K/V first; the server's `decode_batch` reads the pool in place). Next performance steps: a faster prefill matmul (simdgroup tiles on INT4) and split-K decode attention |
| long key regenerated (M5 review) | the review found four of the long texts began in their files' front matter (en in its preface, es and fr in their tables of contents, fr only in the last ~270 characters, ja in the transcriber's note); `long_texts.py` now starts each at its work's opening sentence (an anchor string; zh by offset) and the key was regenerated: 4,802 tokens. fp32 engine: every position within 3.4e-5 before 4096 and 1.4e-5 after (median 1.9e-6), logits 5.7e-6, decode 2.5e-6, greedy 10/10, window off 3.9 (median 0.41). Metal, with the fix below: INT8 KL 0.0007 before 4096 / 0.0009 after / decode 0.0008, greedy 9/10, with fp32 and with bf16 KV; INT4 0.0800 / 0.1240 / 0.1349 (bf16 KV 0.0801 / 0.1243 / 0.1359), greedy 6/10; bf16 vs fp32 KV over 7,604 tokens KL 0.0000, greedy 32/32 |
| a decode failure with the same symptom, found: GPU memory exhaustion | at 12:52 on 2026-10-06 in a fresh process (INT8, fp32 KV, the 4,802-token key: decode steps KL 7.43, greedy 1/10, the prefill rows right at 0.0007 / 0.0009). At ~13:00 a separate script passed 4 of 4 fp32-KV runs (and 4 bf16); from 13:49, while the lock screen played video (another GPU user), every fp32-KV run failed, each with Metal aborts in macOS's log (9 per full run, one per decode step), and a bf16-KV run at the same 5.21 GiB failed too. GPU memory at decode was 5.21 GiB (`torch.mps.driver_allocated_memory`), against the 5.33 GiB Metal recommends on this 8 GB M2, and in some runs Apple's MPS framework printed Metal's `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` on stderr (the test runner keeps only stdout). In that period, the same computation with the prefill's heap released before decode (4.20 GiB) was right in 3 of 3 runs (KL 0.0008, greedy 8/9), and at 5.21 GiB garbage in every run. Why 5.21: PyTorch 2.14's MPS allocator puts each tensor of 10-512 MiB in a 1 GiB heap unless it is "under memory pressure", which by default starts at 1.4x Metal's recommended size (7.46 GiB, never reached on 8 GB), so weights 3.53 + KV 0.67 + one 1 GiB heap for the prefill's temporaries (measured: any tensor of 12-432 MiB alone costs 1,024 MiB) = 5.21. Why silent: PyTorch 2.14 never reads a command buffer's status, so an aborted buffer's work is simply missing (which writes Metal skipped is not visible; it can report partial completion per encoder) (fixed on PyTorch main after 2.14.0, pytorch/pytorch#198883; llama.cpp had the same symptom, ggml-org/llama.cpp#1881) |
| the fix: a low watermark | `src/backend/__init__.py` sets `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.75` before torch allocates on the GPU (read once; a value already in the environment wins): above 4.0 GiB the allocator sizes heaps to their requests and frees empty ones. 0.75 is where a 1 GiB heap started just under the limit still ends below 0.95x Metal's recommended size (the point where MLX starts freeing its cache). The failing configuration: 4.70 GiB at decode, right (KL 0.0008, greedy 8/9) at 0.75 and at 0.6; in `test_aya_long_quant` the highest run peaks at 4.91-4.92 GiB (INT8, fp32 KV; the allocator's own peak, two runs). Speed: the first INT8 4K comparison (4 alternating process pairs while the lock screen played video) was mixed, TTFT 26.9 / 19.1 / 19.5 / 23.0 s at PyTorch's default vs 55.2 / 19.4 / 33.4 / 22.4 s with the watermark; on a quiet machine, 6 pairs gave TTFT median 22.8 vs 21.9 s and decode 72.5 vs 74.8 ms/token (per pair -3.8 to +5.0 ms): no clear cost, at most a few percent. In one process, below vs above the watermark (a ballast): INT4 decode 33.1 vs 32.3 ms/token and a 2,048-token prefill 8.00 vs 8.01 s (PyTorch's default as the control: 32.9 vs 33.1, 7.93 vs 7.94); INT8 at 4K, TTFT 19.2 vs 18.9 s, decode 72.2 vs 70.1 ms/token. Above the watermark PyTorch also commits a command buffer after each MPSGraph op and frees empty heaps, which is what could cost. Metal logs every aborted command buffer in macOS's unified log, silent ones included (9 per failing run this afternoon, one per decode step; none since the fix), so `test_aya_long_quant` now reads it for its own process, and gates every run's peak GPU memory (the allocator's peak) at 0.95x the recommended size. Still unprotected: another process taking the memory (PyTorch reports nothing until a release with the fix), and INT8 at 8K (above) |
| host-to-GPU copies: a caller hazard (found while hunting it, not its cause) | each decode step copied its token id to the GPU with `non_blocking=True`. On MPS that copy reads the host memory when the GPU reaches it, not when it is issued, and the CPU runs ahead (it queues a step in ~3.4 ms, the GPU runs it in ~30): a caller that reused its ids tensor once `forward()` returned changed what already-issued steps embedded. With the GPU busy, overwriting each id moves steps 2-6 by 17-24 in the logits (step 1, issued to an idle GPU, by 0). A freed temporary is safe: PyTorch 2.14 keeps a non_blocking copy's source storage until the blit is done (Copy.mm, buffer_with_offset_from_tensor), and no engine caller wrote to its tensor, so this was a latent API hazard, not an observed bug; only `forward(ids)` takes a caller's tensor. Fix: `state.to_device` copies from a private clone, and all 8 host-to-device copies in `src/` go through it (the other 7 for one rule, not because they raced); the DeltaNet slot index is built once per batch, not per layer. A blocking copy is also correct, but waits for every queued kernel: one per DeltaNet layer took Qwen3.5 batch-1 decode from 20.2 to 27.5-29.5 ms (separate runs). Interleaved in one process, Qwen3.5-0.8B `decode_batch` at batch 1 / 4 / 8: old copy 21.0 / 24.6 / 32.9 ms, `to_device` 20.4 / 24.6 / 33.0, blocking 21.2 / 25.6 / 34.0. `test_cohere2` 11 overwrites each id behind ~100 ms of queued GPU work (fails without the clone), and checks that no other `non_blocking` copy exists in `src/` |
| the gates that missed it | `test_aya_long_quant` and `test_aya_quant` judged only the bf16-KV run, and the bf16-vs-fp32 change one way, so a broken fp32 run made the change negative and passed (-0.6556). Now every scheme x KV run is gated (the long test's 9 decode steps also on their own), the change both ways, and the long test also gates each run's peak GPU memory (the allocator's own peak) and Metal's aborts (above) |

## Review workflows run

| week | agents | findings → confirmed | notable |
|---|---:|---|---|
| W6 | — | 9 confirmed | top-p boundary, padding ids, stream tail |
| W7–8 | 56 | 18 confirmed | non-atomic OutOfBlocks, benchmark method |
| W9–10 | 48 | 5 (measurement) | cache-hot bandwidth overstated |
| W13 | — | — | chat reasoning rules, missing special tokens, atomic state reservation, long-prompt goldens (commit `4b31841`) |
| W15 | 106 | 50 → 9 confirmed | disconnect, regex, serve defaults, /ready before load |
| batching speed-up | 20 | 1 confirmed | pooled-state scheduler test |
| documentation numbers, round 1 | 140 | 67 → 38 confirmed, all fixed | GPU is 8-core, DeltaNet batch update is 2 dispatches, 2B answer key is bf16 |
| documentation numbers, round 2 | ~90 | 45, fixed | server gap is lockstep prefill + CPU sampling + ~28 ms unprofiled work; variance up to ~20% |
| server gap, round 1 | 14 | 5 → 2 confirmed | a large top_k slowed the whole batch; bf16 packs over 32 tokens differed from the prompt alone |
| server gap, round 2 | 16 | 7 → 5 confirmed (2 duplicates) | seeded generators reused on retry; cancelled prefilling request kept its blocks; weak parity/tie tests |
| server gap, round 3 (window, CPU sampling) | 8 | 3 → 1 confirmed, 1 split (both fixed) | `--batch-wait-ms inf` killed the engine thread; a cancel counted as an arrival |
| server gap documentation numbers | 68 | 32 → 27 confirmed, 3 split, all fixed | 63 not 64 decode steps per wave; prefill cost explanation; "closed most of the gap" → almost half |
| server gap documentation numbers, round 2 | 26 | 30 earlier fixes all hold; 11 new → 8 confirmed, 1 split, all fixed | the "~28 ms" was the 2026-09-28 run's slower steps, not prefill stalls; a breakdown that did not add up |
| server gap documentation numbers, rounds 3–4 | 52 | 20 → 16 confirmed, then 4 → 3 confirmed; all fixed (after `0ee0f08`) | the gap split mixed two runs (now ~3/4 prefill within one run); the 2026-09-28 run was slow only at 4–8 clients; a prefill total included loadgen's warm-up prompts |
| deployment fixes (lock, gauges, log reasons) | 31 | 14 → 2 confirmed (the same defect), fixed | the log fix wrote `cancelled` for stop-string completions |
| removing Qwen2.5 (plan, port, review) | 19 | plan: 5 readers + 1 merge; review: 5 → 2 confirmed, fixed | a pointer to a moved test output; timings cited from an unarchived run (→ `run_tests.py --save`) |
| decision endpoint | 41 | 19 → 9 confirmed + 1 split, fixed | prefix scoring, unchunked prefill, jobs blocking decode, abandoned jobs |
| Tiny Aya research | 10 | 4 reports + 5 checks; 1 report off-task (its checker supplied the facts) | the official config recovered by hash; KV is 144 KiB/token (6x Qwen) in fp32; the old tokenizer would drop all non-digit text |
| Tiny Aya M1 | 8 | 30 → 25 confirmed: 4 deferred to M2 / M4 / M6, 2 documented as rare known differences, the rest fixed | `\b` vs Oniguruma; template errors were 500; `enable_thinking` on Tiny Aya gave an empty reply; the legacy rope key was accepted |
| Tiny Aya M2 | 6 | 11 → 8 confirmed (7 distinct), all fixed; 3 refuted, 2 of them added anyway as cheap checks | no model bug; untied heads accepted; the INT4 policy missed the fused qkv; the length cap only in create_app; test gaps found by 63 planted bugs |
| Tiny Aya M4 | 7 | 9 → 8 confirmed (6 distinct), all fixed; 1 refuted, its wording adopted | the new int8 calibration mode broke the bf16 one (no test runs the script); an interrupted quantize destroyed the earlier .qt; the served-KL mask left out BOS and the reply's first prediction |
