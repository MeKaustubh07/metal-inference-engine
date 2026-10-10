# Design: a from-scratch inference engine for Qwen3.5 on an 8 GB Mac

This document describes how the engine is built, why each decision was made, and how it would change at larger
scale. Measured numbers live in `docs/bench/`; this document cites them.

## 1. Goal and constraints

Serve **Qwen3.5-2B** (hybrid Gated DeltaNet + gated attention, 248k vocabulary) through an OpenAI-compatible HTTP
API on an **Apple M2 with 8 GB of unified memory**, with every model component written by hand and verified
against Hugging Face `transformers`.

| constraint | consequence |
|---|---|
| 8 GB shared by OS, apps, weights, KV cache | 2B bf16 (4 GB of weights) runs but crowds everything else; INT4 (1.4 GB) is the deployment format |
| no CUDA; Metal only; no Xcode | kernels are MSL compiled at runtime by `torch.mps.compile_shader` (a small Objective-C++ runtime in `src/native/` runs the same MSL for study and benchmarks) |
| ~100 GB/s memory bandwidth | decode is bandwidth-bound: the ceiling is `bandwidth / bytes read per token` |
| learning project | PyTorch is used only for tensor storage, device placement and primitive ops; no `torch.nn`, no HF code at runtime |

## 2. Architecture

```
 client ──HTTP/SSE──▶ FastAPI (uvicorn event loop)                      src/server/app.py
                        │  validate · chat template · tokenize · 429 when the queue is full
                        ▼
                      Scheduler (one engine thread owns the GPU)         src/server/scheduler.py
                        │  waiting → admit (KV budget) → packed chunked prefill → batched decode + sampling
                        │  tokens flow back via AsyncSink (call_soon_threadsafe), no thread per request
                        ▼
                      Model: Qwen35Model                                 src/models/
                        │  forward(ids, state)      prefill, chunked DeltaNet (WY form, chunk 64)
                        │  forward_packed(chunks)   several sequences' prompt chunks in one pass
                        │  decode_batch(tokens, states)   B sequences, one pass over the weights
                        ▼
                      Backend protocol (linear, rms_norm, rope, attention, deltanet_decode, …)  src/backend/
                        │  TorchBackend: CPU fp32 oracle / MPS bf16
                        │  MetalBackend: hand-written kernels, bf16 / INT8 / INT4 weights
                        ▼
                      Weights: mmap'd safetensors (bf16 views) or .qt (quantized), lazily paged in
                      State:   HybridState = paged KV (6 attention layers) + DeltaNet S and conv tail (18 layers)
```

### Components

| component | file | responsibility |
|---|---|---|
| loader | `src/weight_loader.py` | parse the safetensors header, mmap the file, hand out zero-copy bf16 tensor views; a sharded checkpoint through its `model.safetensors.index.json` |
| config | `src/config.py` | `Qwen35Config`, `Cohere2Config` (refuses settings the engine does not implement) |
| tokenizer | `src/tokenizer.py` | byte-level BPE from `tokenizer.json` (GPT-2 byte alphabet, merge ranks, special tokens first, NFC if the file asks, every regex Split, BOS from the post-processor); merges as HF tokenizers does them, a heap of pairs by rank and position, n log n per word |
| chat template | `src/chat.py` | Qwen3.5 ChatML incl. thinking-mode rules; or the model's own Jinja template, rendered as transformers renders it (Tiny Aya) |
| model | `src/models/qwen3_5.py` | hybrid layers, fused projections, partial RoPE, output gate, DeltaNet prefill and decode |
| model | `src/models/cohere2.py` | Tiny Aya: all-attention layers, LayerNorm feeding attention and MLP in parallel, interleaved RoPE (by reordering q/k rows at load) on sliding layers only, a 4096-token sliding window on 27 of 36 layers (they read only their window's suffix of the KV cache), logit scale, bf16 KV |
| state | `src/state.py` | contiguous and paged KV caches, block allocator (refuses double frees), `HybridState`; `GroupedKVPool` for Tiny Aya: layer groups as pool units, a per-sequence ring of blocks for the sliding layers (an 8K sequence holds 388 MiB of bf16 KV instead of 576) |
| kernels | `src/kernels/*.metal` | matvec (bf16, INT8, INT4, batched), RMSNorm, LayerNorm (Tiny Aya), RoPE, decode attention, DeltaNet step, SwiGLU |
| quantization | `src/quant.py` | block-32 INT8 / asymmetric INT4, calibrated mixed-precision policy, `.qt` format |
| scheduler | `src/server/scheduler.py` | continuous batching, admission control, preemption, cancellation, drain |
| API | `src/server/app.py` | `/v1/completions`, `/v1/chat/completions` (SSE), `/health`, `/ready`, `/metrics` |

## 3. Request lifecycle

1. **Validate and tokenize.** Pydantic validates on the event loop and rejects bad parameters with 422; the
   tokenizer runs in a worker thread so a long prompt cannot stall other streams; prompts that can never fit
   (`prompt + max_tokens > max_model_len`, or more KV blocks than the pool holds) get 400.
2. **Submit.** The request joins a bounded waiting queue. If the queue is full the client gets **429 with
   Retry-After**: shedding load early is better than accepting work that will time out.
3. **Admit** (engine thread). Requests move waiting → prefilling → running. A request is admitted when there is a
   batch slot, a DeltaNet state slot, and room in the pool for `blocks(prompt) + one block of headroom per
   sequence in flight`; its prompt's blocks are reserved on admission, so a prefill never runs out halfway. The
   headroom stops a newcomer from being preempted on its very first step. When an idle engine is woken by an
   arrival, it keeps collecting arrivals that come within 5 ms of each other (`--batch-wait-ms`, default 5; twice
   that in total at most), so requests that arrive together share one prefill pass. With `--prefix-cache` (Tiny
   Aya), the chat template's fixed preamble is prefilled once at start-up into a sequence that is never freed; a
   chat whose ids start with it borrows those KV units read-only and prefills only the rest (`server/prefix.py`:
   whole blocks only, never written or freed through the borrower, and only while its prompt + max_tokens cannot
   wrap a ring onto them).
4. **Prefill (packed and chunked).** Each loop iteration runs **one packed forward** over at most `prefill_chunk`
   prompt tokens (default 512), taken first come, first served from the prefilling requests. Per-token work
   (norms, projections, MLP) runs once over all their tokens, so several short prompts share one pass over the
   weights; attention and the DeltaNet recurrence run per sequence on its slice. A long prompt is split into
   chunks across iterations, with a decode step between chunks, so running requests keep generating. A request
   whose whole prompt is in its state samples its first token (TTFT).
5. **Decode.** Each loop iteration then runs **one `decode_batch` for every running sequence**. Projections and MLPs
   go through batched matvec kernels that read each weight once per step for the whole batch. Attention is one
   **paged-attention** dispatch per layer: every (head, sequence) reads its keys/values in place through its block
   table. The DeltaNet update is two dispatches per layer for the whole batch (conv step, delta-rule update),
   because every sequence's recurrent state lives in one pooled tensor (`HybridPool`) indexed by sequence slot.
6. **Sample.** The batch's logits are copied to the CPU once (0.5 ms on unified memory), where the repetition
   penalty and a top-(k + 8) candidate selection run once for every request; each request's temperature / top-k /
   top-p / seeded draw then runs on its few dozen candidates, exactly as the single-request sampler would. (On
   the M2, `torch.topk` over the 248k vocabulary is 4× faster on the CPU than on MPS, so the GPU was the wrong
   place for it.) A request that cannot be sampled fails alone.
7. **Stream.** Tokens are decoded incrementally. Text ending in an incomplete UTF-8 character (U+FFFD) is held
   back until the next token completes it. Each piece goes to the handler's asyncio queue and out as an SSE chunk.
8. **Finish** on EOS, `max_tokens`, cancellation (client disconnect), or error. KV blocks go back to the pool
   immediately, so a waiting request can take the slot on the next step.

**Preemption.** If running sequences need more blocks than are free, the newest one is preempted: its blocks are
freed and it goes back to the front of the queue with its generated tokens. When re-admitted it **recomputes**
prompt + generated tokens through the same packed, chunked prefill (vLLM-style recompute). The client sees a
pause, not a gap: the output is identical to an uninterrupted run (`tests/test_server.py` checks this with greedy
decoding; `tests/test_prefill.py` with 4-token prefill chunks on Qwen3.5-0.8B INT4, the hybrid model).

**Decisions (`POST /v1/decide`, `src/decision.py`).** A context, a question and N options in; the probability of
each option and a decision out, for three question types: choice (pick one), boolean (yes / no), score (an ordinal
label). Listing the options in the prompt would give each a different position and let later options see earlier
ones, so the answer could depend on their order. Instead:
1. The context is prefilled once, in `prefill_chunk` pieces.
2. Each option continues from its own **fork** of the context's state (`HybridState.fork`): a copy of the
   attention layers' K/V and of the DeltaNet recurrent state and conv tail. Attention could hide options from each
   other with a mask, but a recurrence has no mask: its state summarizes everything it has read, so isolating the
   options means giving each its own copy. Every fork has the context's length, so every option starts at the same
   position.
3. All options run as segments of one packed forward pass (`packed_hidden`), in a canonical order (sorted by token
   ids), so any permutation of the options produces the identical batch and identical scores, bit for bit. (The
   batched kernels group rows, so a row's rounding can depend on where it lands in the batch.)
4. An option scores as a whole word: log P(its tokens) + log P(the next token cannot continue its last word or
   number). So "1" is not credited with "10", and "No" is not penalized because the model would go on "No, the
   capital is ...". Probabilities are a softmax over the options' per-token scores, normalized in sorted order so
   they too are identical under any permutation.
The work runs on the engine thread as a stepwise job (`Scheduler.run_job`): one context chunk or option group per
loop iteration, with decode steps for running streams in between; a client that disconnects cancels it. Forks are
standalone states outside the KV pool, run in groups that keep them under 256 MB; when only one option fits a group
(long contexts), each option runs on the context's own state, which is then rewound (its length reset, the DeltaNet
state restored from a copy): bit-identical to a fork, without copying the context. The JEV-style "answer token that
sees every option" has no order-invariant equivalent in the DeltaNet layers (a recurrence reads the options in
some order), so options are scored independently.

## 4. Memory budget (Qwen3.5-2B, INT4, 8 GB machine)

| item | size | how |
|---|---:|---|
| weights, INT4 + INT8 policy tensors + INT8 embedding | 1.41 GB | `.qt` file, mmap'd, pages in on first use |
| KV cache per token | 24 KiB | 6 attention layers × 2 KV heads × 256 dims × (K+V) × fp32 |
| KV pool, 1024 blocks × 16 tokens | 384 MiB | 16k tokens across all sequences |
| DeltaNet state per sequence | 19.3 MiB | 18 layers × 16 heads × 128 × 128 fp32 (18.0 MiB) + conv tail 18 × 3 × 6144 fp32 (1.3 MiB) |
| state pool, `max_batch` = 8 slots | 154 MiB | fixed; does not grow with context |

The hybrid architecture is what makes this fit. If all 24 layers were full attention, a token would cost 96 KiB
and a 4k-token context 384 MiB **per sequence**. With 18 of 24 layers as DeltaNet the same context costs 96 MiB + 20
MiB of fixed state. Qwen3.5's KV cache is kept in fp32 to match the reference exactly. Tiny Aya's (36 attention
layers, 144 KiB per token in fp32) is stored in bf16 (`kv_dtype` in the registry): the decode-attention kernels read
bf16 or fp32 K/V and widen each element as they read it, and the cost is measured as a KL change (none measurable
at the answer key's positions; tests/test_aya_quant.py).

## 5. Performance model

Decode reads every weight once per step, so for batch size 1: `tok/s ≤ bandwidth / weight bytes`. For the 2B model
that means ≈ 27 tok/s in bf16 (3.76 GB), ≈ 50 in INT8 (2.00 GB) and ≈ 71 in INT4 (1.41 GB) at 100 GB/s.
Measured (`docs/bench/qwen35.md`, `decode_batch` at B = 1, no sampling): 20.6, 39.2 and 50.1 tok/s, i.e. 71–78%
of the ceiling.

| Qwen3.5-2B INT4, M2 Air | result |
|---|---|
| single-stream decode (`decode_batch` step, no sampling) | 50 tok/s (llama.cpp Q4_1 42–53, MLX-LM 4-bit 70) |
| batch-8 aggregate decode | 116 tok/s (llama.cpp 62, MLX-LM 73) |
| prefill 256 / 1024 tokens | 398 / 450 tok/s (llama.cpp 601 / 452, MLX-LM 499 / 448) |
| server, 8 closed-loop clients | 94 tok/s, TTFT p50 0.84 s, TPOT p50 73 ms; the previous server measured side by side: 77 tok/s, 1.06 s, 92 ms (`docs/bench/serving.md`) |

The levers that follow from this model:
- **Fewer bytes per weight** (quantization). INT4 reads about a third of bf16's bytes. It is dequantized in
  registers inside the matvec kernel, so the full-precision matrix never exists in memory.
- **More tokens per weight read** (batching). `decode_batch` with B sequences reads the weights once for B tokens.
  This needed dedicated kernels (`matvec_{bf16,q8,q4}_rows`): the general quantized path would dequantize the
  whole matrix every step. The first version (one output row per SIMD group) was only 1.3× cheaper than 8
  separate single-row calls at M = 8 (sharing one weight read should allow far more), because each weight still
  cost 8 activation loads. Computing 2 output rows per SIMD group reuses each activation load, which made the
  INT8 head 2.5× faster at M = 8.
- **No per-sequence work in the batched step.** Looping over sequences for attention and the DeltaNet update
  cost 6–10 ms per sequence per step, mostly dispatch overhead from dozens of tiny ops. The paged-attention
  kernel and the pooled DeltaNet state turned that into a handful of dispatches per layer for the whole batch
  (0.8 ms per sequence). The batch-8 step went from 186–189 ms (two profiling runs) to 69 ms.
- **Fewer CPU→GPU round trips.** With the hand-written kernels in place, Qwen2.5 decode sat at ~41 tok/s
  because two hidden synchronizations (`int(positions[0])` in every layer, and a CPU-side embedding lookup) made
  the CPU wait for the GPU every layer. Removing them gave ~73 tok/s (`docs/bench/decode.md`; the starting point,
  plain PyTorch ops on MPS, was ~20).
- **Host copies that neither wait nor read late.** On MPS, a `non_blocking=True` copy reads the host tensor when
  the GPU reaches the transfer, not when it is issued. PyTorch keeps the source alive until then, so a freed temporary
  is safe; but the CPU runs ahead, and a caller that reuses its ids tensor once `forward()` returns changes what an
  already-issued step embeds (shown with the GPU busy). A blocking copy is correct but waits for every queued kernel:
  ~0.3 ms at the start of a step, and in the middle of one the CPU stalls on the GPU (one per DeltaNet layer took
  Qwen3.5 batch-1 decode from 20 to 28-30 ms). `state.to_device` copies from a private clone, without waiting.
- **GPU memory headroom, and failures PyTorch does not report.** PyTorch's MPS allocator puts every tensor of
  10-512 MiB in a 1 GiB heap until the process is "under memory pressure", which by default starts at 1.4x Metal's
  recommended working set: never, on an 8 GB Mac. Tiny Aya INT8 with an fp32 KV cache at 4.8K tokens reached 5.21 of
  the 5.33 GiB Metal recommends. While another process used the GPU too, Metal aborted command buffers for lack of
  memory, and PyTorch 2.14, which never reads a command buffer's status, returned garbage without an error (the same
  run had passed an hour earlier). The backend package sets a low watermark of 0.75x before torch allocates anything,
  so above 4.0 GiB heaps are sized to their requests: 4.70 GiB, decode exact. Cost, INT8 at 4K over 6 alternating
  runs: prefill 22.8 vs 21.9 s, decode 72.5 vs 74.8 ms/token (per pair -3.8 to +5.0 ms), so at most a few percent.
  PyTorch still cannot report an abort, so the long Metal test reads them from macOS's unified log, where Metal
  records each one.

Prefill is compute-bound (`[T, K] × [K, N]` GEMMs), so it uses PyTorch's tuned MPS GEMM. The hand-written tiled
GEMM is 4× slower and is kept only for study. Quantized weights are expanded to fp32 by one kernel pass per 32 MB
row chunk, then multiplied in fp32: on the M2, MPS runs fp32 GEMM at ~2.5 TFLOPS and bf16 at only ~1.4 (no
native bf16 arithmetic before M3), so fp32 is both the faster and the exact choice. The first version expanded
through several full-size fp32 tensor ops and took 2.0 s for a 256-token prompt; now 0.65 s. bf16 weights take
the same route above 32 rows, so a prompt packed with others keeps fp32 activations, as it would alone.

## 6. Key decisions and trade-offs

| decision | chosen | alternative | why |
|---|---|---|---|
| runtime language | Python + PyTorch tensors, MSL kernels | C++ end to end | readable, verifiable against HF in one process; kernels are where speed lives |
| activations | fp32 | bf16 | reference-exact; decode is weight-bandwidth-bound, so activation precision is nearly free |
| KV layout | paged blocks of 16 tokens | contiguous per sequence | no fragmentation, no worst-case reservation; enables admission by block count and preemption |
| preemption | recompute | swap KV to CPU | unified memory gives swapping no bandwidth advantage; recompute is simpler and cheap for short prompts |
| INT4 scheme | asymmetric, block 32, fp16 scale + min | symmetric | on Qwen2.5-0.5B symmetric INT4 cost +28% perplexity; asymmetric plus a calibrated policy recovered most of it (`docs/bench/quant.md`) |
| mixed precision | measured per-tensor KL, threshold 0.005 nats | fixed rules | the sensitive tensors (attention sinks at position 0, a few output projections) are model-specific |
| batching | continuous (join/leave every step) | static batches | a finished sequence's slot is reused next step; no head-of-line blocking |
| batched attention | paged-attention kernel over block tables | gather each sequence's KV, then attend | the gather loop cost ~14 small ops per sequence per layer; the kernel is one dispatch per layer |
| DeltaNet state | one pooled tensor, slot per sequence | a tensor per sequence | the pool lets one dispatch (per kernel) update every sequence in the batch |
| engine threading | one engine thread, asyncio front end | thread per request | the GPU is one device; one owner means no locks around model state |
| API | OpenAI-compatible | custom | existing clients, SDKs and load tools work unchanged |

## 7. Correctness strategy

- **Answer key**: HF `transformers` dumps activations through forward hooks, plus 10 greedy tokens per prompt.
  The bring-up model, Qwen2.5-0.5B, was checked at layer-0 granularity (block ≤ 3.3e-5, logits ≤ 3e-4, greedy ==
  HF); it was removed on 2026-09-30, and its test outputs are in `docs/bench/raw/tests/2026-09-30-with-qwen2.5/`.
  Qwen3.5 (eager attention; 7 prompts including Hindi and a 126-token prompt that crosses the 64-token DeltaNet
  chunk boundary): embedding, every layer output, final norm, logits; fp32 for 0.8B, bf16 for 2B, whose fp32
  weights (~7.5 GB) do not fit.
- **Tolerances are relative to magnitude**. Qwen3.5-0.8B: every layer ≤ 1e-4 relative, logits ≤ 1e-3 absolute,
  top-1 100%, greedy text identical.
- **bf16 and quantized paths are judged on tokens**: KL divergence against the reference and top-1 flips only at
  positions where the top-2 gap is above bf16 noise.
- **Invariants**: cached == uncached; paged == contiguous; batched == sequential (greedy tokens identical, logits
  within 6e-5 in fp32 on the CPU; bit-identical in bf16 on Metal); preempted == uninterrupted; fused kernels ==
  reference ops; batched kernels == per-row kernels.
- `scripts/run_tests.py` runs all 19 suites, reports any that skipped, and exits nonzero on any failure.

## 8. Operations

| concern | mechanism |
|---|---|
| liveness vs readiness | `/health` = process alive; `/ready` = model loaded, engine thread alive, queue has room (route traffic here) |
| overload | bounded queue → 429 + Retry-After; admission by KV blocks; preemption instead of OOM |
| slow or gone clients | a disconnect (streaming, or non-streaming, polled every second) cancels the request at the next step and frees its blocks |
| shutdown | SIGTERM → uvicorn stops accepting and lets open requests finish (up to `--drain-timeout`, 25 s) → the lifespan stops the scheduler (anything left has no client, 2 s) → exit; run containers with a longer stop timeout (`docker run --stop-timeout 30`) |
| metrics | Prometheus: request/token counters, decisions answered and decision jobs turned away, prefill passes and tokens, running/prefilling/waiting/waiting-job/KV-free gauges (read at scrape time, so a request queued behind a step in progress shows), weights locked in RAM, TTFT/TPOT/e2e histograms |
| logs | one JSON line per finished request (id, tokens, finish reason as the client was told it, `cancelled` for a client that left, TTFT, latency) |
| memory residency | after the warm-up the weights are `mlock`ed, so an idle server's next request does not wait for macOS to page them back in (measured 0.25 s warm vs up to 0.81 s after 4 min idle; `--no-lock-weights` turns it off) |
| failure isolation | a failed packed prefill pass fails the requests in that pass; a request that cannot be sampled fails alone; a failed decode step fails every request in flight (running and still prefilling), not the server |

## 9. Deployment

| target | backend | notes |
|---|---|---|
| a Mac (M-series), native | Metal INT4 | `scripts/serve.py` as a launchd service (below); best performance; expose with a reverse proxy or a tunnel (Tailscale / Cloudflare Tunnel) |
| cloud Apple silicon | Metal INT4 | AWS EC2 Mac (mac2.metal = M1, mac2-m2.metal = M2) or Scaleway Apple silicon; same launchd setup |
| Linux container | CPU fp32 (TorchBackend) | `Dockerfile` (1.37 GB image), for Linux hosts and CI with memory to spare. Default Qwen3.5-0.8B in fp32 (~3.5 GB): on this 8 GB Mac under Docker Desktop, even with a 5 GB VM, ready in 51 s and healthy but ~0.1 tok/s, because the weights do not stay resident in the VM (`docs/bench/raw/docker_2026-10-01.md`). The graceful stop was verified with the previous default, Qwen2.5-0.5B (~4 tok/s, since removed). Docker on a Mac cannot reach the Apple GPU |

A container cannot use Metal: Docker Desktop on macOS runs Linux in a VM with no Apple GPU. The GPU deployment is
therefore a native process; its "container" is a pinned venv plus a launchd service (not included in the repo;
a minimal definition, saved as `~/Library/LaunchAgents/com.example.inference-engine.plist` and loaded with
`launchctl load`):

```xml
<plist version="1.0"><dict>
  <key>Label</key><string>com.example.inference-engine</string>
  <key>WorkingDirectory</key><string>/path/to/InferenceEngine</string>
  <key>ProgramArguments</key><array>
    <string>/path/to/InferenceEngine/scripts/.venv/bin/python</string>
    <string>scripts/serve.py</string><string>--host</string><string>127.0.0.1</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ExitTimeOut</key><integer>30</integer>
  <key>StandardErrorPath</key><string>/tmp/inference-engine.log</string>
</dict></plist>
```

launchd restarts the process if it dies (`KeepAlive`) and sends SIGTERM with a 30 s grace period on stop
(`ExitTimeOut`), which covers the server's 25 s drain. A reverse proxy or tunnel in front terminates TLS and
routes to `127.0.0.1:8000`; its health check should use `/ready`. Give the server memory of its own: on unified
memory, a browser, an editor or Docker Desktop's VM on the same 8 GB push an idle server's pages out (the weights are
locked, the KV cache and DeltaNet state pools are not; `docs/bench/serving.md`).

## 10. Scaling beyond one machine (design, not built)

The engine is **stateful**: a sequence's KV blocks and DeltaNet state live in one process. That shapes scale-out:

1. **Replicas behind a load balancer**, routing on `/ready` and least outstanding requests. Each replica is
   independent (one model copy, own KV pool).
2. **Session affinity / prefix-aware routing**: send a conversation's next turn to the replica that holds its prefix.
   With prefix caching (hash each full KV block and share it copy-on-write across sequences), multi-turn chat skips
   re-prefilling the history. For the hybrid model, the DeltaNet state is a fixed-size snapshot per prefix, so a
   prefix cache stores one snapshot per cached boundary.
3. **Disaggregated prefill and decode**: prefill is compute-bound and decode bandwidth-bound, so separate pools
   let each be sized for its bottleneck. The KV and recurrent state move from prefill to decode workers after the
   first token.
4. **Larger models**: tensor parallelism splits each matmul across devices (all-reduce per layer); pipeline
   parallelism splits layers (micro-batches hide the bubbles). Neither pays off on unified-memory Macs.

## 11. Limitations and next steps

- **The server still trails the engine step** (94 vs 116 tok/s at 8 clients; `docs/bench/serving.md`). It was
  77 (re-measured; the published 56 and its "~28 ms per step of unprofiled work" came from a run that was slower
  at 4 and 8 clients, for reasons not established). Measured, about three quarters of the 77-vs-116 gap was prompts
  prefilled one at a time while the admitted requests waited, and a quarter per-step work around the decode step,
  mostly per-request CPU sampling (8.8 ms per step in the profiled run). Packed + chunked prefill, a 5 ms batching
  window and batched sampling closed almost half of the gap
  (77 → 94 of the engine's 116 tok/s). What remains is mostly the packed prefill pass (0.82 s for 8 short prompts,
  ~15% of the time at 8 clients; sampling is ~4%). About a third of that pass (~0.28 s) is attention and the
  DeltaNet recurrence running once per packed sequence: a variable-length prefill kernel is the next step
  (prefill/decode disaggregation is the multi-machine version).
- **After idle and on first use**: the KV cache and DeltaNet state pools (~0.5 GB) are not locked, so an idle server
  can still lose up to ~0.2 s on its next first token, and the first request of each new prompt length pays
  ~0.1–0.2 s of MPS shape compilation. Locking the pools and warming common prompt lengths at start-up fix both.
- **Decisions prefill their context outside the pool**: short contexts fork per option group (grouped under 256 MB);
  long ones score each option on the context's own state and rewind it, so the peak is the context's prefill, not
  copies (Tiny Aya INT4 at 8K: 4.99 GiB, hence its 6,144-token decide cap). Sharing the pinned chat preamble with
  decide's context would save its ~1.1 s prefill per chat-style decision.
- Batch 2 barely beats batch 1: the linears cost 33 vs 17 ms, and the 2-rows kernel is 1.2–2.1× slower than the
  single-row kernel at one row (which is why M = 1 keeps its own kernel).
  Simdgroup-matrix (8×8 hardware tile) kernels, as in MLX and llama.cpp's `mul_mm`, are the next step for both
  small-batch decode and prefill (which today expands weights to fp32 first).
- Single-stream decode trails MLX-LM (50 vs 70 tok/s): mostly bytes (our tied head is INT8, 0.5 GB of the 1.41 GB,
  by a rule carried over from Qwen2.5 and not re-measured on Qwen3.5; MLX's is 4-bit), plus ~300 dispatches per
  step issued from Python.
- Qwen3.5's KV cache is fp32 (Tiny Aya's is bf16); an INT8 KV cache would double the bf16 capacity again.
- Prefix caching covers only the chat template's fixed preamble (opt-in, Tiny Aya); caching any shared prefix
  (multi-turn history, Qwen's DeltaNet snapshots) is item 2 above.
- Speculative decoding (Qwen3.5-0.8B drafting for 2B) would cut per-token latency at batch 1.
- Text only: the checkpoint's vision tower is ignored.
