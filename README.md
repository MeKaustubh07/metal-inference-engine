# Inference engine for Qwen3.5, from scratch, on an 8 GB Mac

An LLM inference engine written from first principles in Python, with PyTorch used for tensors, devices and a
few primitive ops (prefill runs PyTorch's GEMM), plus hand-written Metal kernels for decode. It serves **Qwen3.5-2B** (hybrid Gated DeltaNet + gated attention) in INT4 through
an OpenAI-compatible API with continuous batching, on an Apple M2 with 8 GB of memory.

Everything a model needs is implemented here: safetensors parsing, byte-level BPE, RMSNorm, RoPE, GQA attention,
the chunked and recurrent gated delta rule, paged KV cache, sampling, chat templates, quantization, kernels,
scheduler and server. The model path is checked against Hugging Face `transformers`; the serving path is checked
against the engine's own one-request-at-a-time decoding (batched, preempted and streamed outputs must match it).

- Design and trade-offs: [`docs/design.md`](docs/design.md)
- Benchmarks: [`docs/bench/`](docs/bench/); every measurement in order: [`docs/results-log.md`](docs/results-log.md)

## Results (MacBook Air M2, 8-core GPU, 8 GB)

| Qwen3.5-2B | this engine INT4 (1.41 GB) | llama.cpp Q4_1 | MLX-LM 4-bit |
|---|---:|---:|---:|
| accuracy vs HF bf16 | KL 0.045, 3/190 top-1 flips | — | — |
| single-stream decode (engine step) | 50 tok/s | 42–53 tok/s | 70 tok/s |
| batch-8 aggregate decode (engine step) | **116 tok/s** | 62 tok/s | 73 tok/s |
| prefill, 256 / 1024 tokens | 398 / 450 tok/s | 601 / 452 tok/s | 499 / 448 tok/s |
| through the HTTP server, 8 clients | 94 tok/s (TTFT p50 0.84 s) | — | — |

bf16 on our engine: KL 0.0004 with 0/190 flips; INT8: KL 0.0007 with greedy output identical to HF on all 70 test
tokens. The server went from 77 to 94 tok/s at 8 clients (measured side by side) after profiling showed where the
gap to the engine step was: mostly prompts prefilled one at a time while every running request waited, plus ~9 ms
per step of per-request sampling. Prompts are
now packed into shared, chunked prefill passes, and sampling is batched (on the CPU, where top-k over the 248k
vocabulary is 4× faster than on MPS). What remains is mostly the prefill pass itself (~15% of the time in this
load test). Details and methodology: [`docs/bench/compare.md`](docs/bench/compare.md),
[`docs/bench/qwen35.md`](docs/bench/qwen35.md), [`docs/bench/serving.md`](docs/bench/serving.md).

## What is in it

| layer | highlights |
|---|---|
| models | Qwen3.5-2B (served, INT4) and Qwen3.5-0.8B (same-architecture fp32 reference), 18 Gated DeltaNet + 6 gated-attention layers; Qwen2.5-0.5B was the bring-up model (removed 2026-09-30) |
| correctness | vs HF `transformers`: bring-up (Qwen2.5-0.5B, removed): layer-0 block ≤ 3.3e-5 and logits ≤ 3e-4 (fp32), greedy == HF on 5 prompts; Qwen3.5-0.8B every one of 24 layers ≤ 3.8e-6 relative (fp32), greedy identical on 7 prompts incl. Hindi; Qwen3.5-2B vs HF bf16 (fp32 does not fit in 8 GB): KL 0.0004 / 0.0007 / 0.045 for bf16 / INT8 / INT4, greedy 64 / 70 / 47 of 70 tokens, every divergence at a near-tie |
| state | paged KV cache (block allocator, block tables) + fixed-size DeltaNet recurrent/conv state per sequence |
| kernels (MSL) | bf16/INT8/INT4 matvec (+ batched), paged attention, batched DeltaNet step, RMSNorm, RoPE, fused SwiGLU, INT4/INT8 → fp32 expansion |
| quantization | block-32 INT8, asymmetric INT4 with a calibrated per-tensor mixed-precision policy; `.qt` mmap format |
| serving | continuous batching, packed + chunked prefill, batched sampling, recompute preemption, 429 backpressure, SSE streaming, cancellation, graceful drain; an order-invariant decision endpoint (choice / boolean / score) |
| operations | `/health`, `/ready`, Prometheus `/metrics` (TTFT/TPOT/e2e histograms), JSON request logs, weights locked in RAM (so macOS cannot page out an idle server's weights), Dockerfile |

## Quick start

```bash
python3.14 -m venv scripts/.venv && scripts/.venv/bin/pip install -r requirements-dev.txt
scripts/.venv/bin/hf download Qwen/Qwen3.5-2B --local-dir models/qwen3.5-2b
```

Quantize once (reads one tensor at a time, keeps the quantized output in memory and writes the file at the end;
measured peak 3.5 GB for the 2B; uses the calibrated policy in `configs/quant/`):

```bash
scripts/.venv/bin/python scripts/quantize.py models/qwen3.5-2b/model.safetensors-00001-of-00001.safetensors models/qwen3.5-2b/model.int4.qt --scheme int4 --policy configs/quant/qwen3.5-2b.json --prefix model.language_model.
```

Chat in the terminal:

```bash
scripts/.venv/bin/python scripts/repl.py --model qwen3.5-2b --backend metal-int4 --weights models/qwen3.5-2b/model.int4.qt
```

Serve the OpenAI-compatible API:

```bash
scripts/.venv/bin/python scripts/serve.py --model qwen3.5-2b --backend metal-int4 --weights models/qwen3.5-2b/model.int4.qt
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Explain KV caching in two sentences."}],"stream":true}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8000/v1"`.

Decisions: each option is scored as a continuation of the context from its own copy of the context's state, so the
answer cannot depend on the order the options are listed in (`docs/design.md`, `docs/bench/decision.md`):

```bash
curl http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' -d '{"type":"choice","question":"What is the capital of France?","options":["Lyon","Paris","Nice"]}'
```

`type` is `choice` (pick one), `boolean` (yes / no; no options needed) or `score` (ordinal labels, lowest first).

## Tests

The suites need more than the Quick start: the Qwen3.5-0.8B checkpoint, the 2B INT8 file, the native runtime and
the HF golden references (gitignored; generating them needs `transformers`). Once:

```bash
scripts/.venv/bin/hf download Qwen/Qwen3.5-0.8B --local-dir models/qwen3.5-0.8b
```

```bash
scripts/.venv/bin/python scripts/quantize.py models/qwen3.5-2b/model.safetensors-00001-of-00001.safetensors models/qwen3.5-2b/model.int8.qt --scheme int8 --prefix model.language_model.
```

```bash
scripts/build_native.sh && scripts/.venv/bin/python scripts/golden_qwen35.py && scripts/.venv/bin/python scripts/golden_qwen35.py models/qwen3.5-2b tests/golden_qwen35_2b bf16
```

The Tiny Aya checks need its gated files (accept the terms on the model page, then download a pinned revision; see
[`docs/tiny-aya-plan.md`](docs/tiny-aya-plan.md)) and three local answer keys (the tokenizer's, the model's, and the
long one past the window). Without the files they print SKIP; with
the files but without a key they fail and name the command that makes it:

```bash
scripts/.venv/bin/hf download CohereLabs/tiny-aya-global --revision af89d219b53ed9b13b8a4645f9c8028973510324 --local-dir models/tiny-aya-global
```

```bash
scripts/.venv/bin/python scripts/golden_tokens.py models/tiny-aya-global tests/golden_tokens_aya.json
```

```bash
scripts/.venv/bin/python scripts/golden_aya.py
```

```bash
scripts/.venv/bin/python scripts/golden_aya.py --long
```

(`golden_tokens.py` makes the tokenizer's answer key; `golden_aya.py` the real model's: transformers' own layers one
at a time in fp32, ~3.5 min and ~2.7 GB; `--long`, a 4,802-token key past the 4096-token sliding window, ~6 min and
2.4 GB, from public-domain texts in 8 languages that `scripts/long_texts.py` downloads, pinned by SHA-256, into the
gitignored `models/long_texts/`). The quantized files (~25 s each; INT4 with the calibrated policy):

```bash
scripts/.venv/bin/python scripts/quantize.py models/tiny-aya-global/model.safetensors.index.json models/tiny-aya-global/model.int8.qt --scheme int8
```

```bash
scripts/.venv/bin/python scripts/quantize.py models/tiny-aya-global/model.safetensors.index.json models/tiny-aya-global/model.int4.qt --scheme int4 --policy configs/quant/tiny-aya-global.json
```

Then:

```bash
scripts/.venv/bin/python scripts/run_tests.py
```

`--quick` skips the cache, quantization, Qwen3.5, Tiny Aya model / long / quant, serving, prefill and decision suites
(it runs tokenizer, aya, cohere2, window, sampling, paged, kernels and native). The 19 suites cover the tokenizer (both
models vs HF),
Tiny Aya's files (config, sharded weights, chat template, registry, server routing), the Cohere2 model vs HF on small
random models (every layer, greedy, caches, batching, bf16 KV, the length cap, a decode step reading its token id when called on Metal), the sliding window (exhaustively vs a
float64 oracle, the model vs HF past the window on every path, the sliding layers' per-sequence ring of blocks with every slot reused and poisoned, the scheduler on it with preemption) and the real 3.35B Tiny Aya in fp32
streamed one layer at a time (every layer, logits, decode steps and greedy vs the answer key; 5 languages, code and
chat) and past its 4096-token window on a 4,802-token text in 8 languages (every position, logits, decode, greedy,
and a window-off sensitivity check), Tiny Aya in INT8 and INT4 on Metal vs both keys (KL, top-1 flips, fp32 and bf16
KV, the bf16 KV cache out to 7.6K tokens, GPU memory headroom and Metal's own log of aborted work), sampling (incl. batched == per-request), the hybrid cache
(cached == uncached, KV and DeltaNet byte accounting, MPS fp32/bf16 and Metal), paged state (isolation, running out
of blocks or state slots), kernels, native runtime, quantization, Qwen3.5-0.8B vs HF fp32 (every layer),
Qwen3.5-2B (bf16/INT8/INT4), the server (batched == sequential, preemption, streaming, 429, cancellation, drain)
packed/chunked prefill (packed == alone, chunked == sequential, batching window) and decisions (forked == alone,
shuffled options give bit-identical scores, stepwise jobs between decode steps). `--save docs/bench/raw/tests`
archives each suite's output and a summary with timings: the last full run (247 checks, 449 s) is in
[`docs/bench/raw/tests/`](docs/bench/raw/tests/); the last run with Qwen2.5 is in
[`docs/bench/raw/tests/2026-09-30-with-qwen2.5/`](docs/bench/raw/tests/2026-09-30-with-qwen2.5/).

## Benchmarks

`scripts/bench.py` measures TTFT, TPOT, prefill/decode throughput and batched decode. `scripts/loadgen.py` drives
a running server at several concurrency levels. Results and methodology are in [`docs/bench/`](docs/bench/).

## Deployment

| target | how |
|---|---|
| Mac, native (GPU) | `scripts/serve.py` as a launchd service (definition in `docs/design.md`) behind a reverse proxy or tunnel; Metal INT4 |
| cloud Apple silicon | AWS EC2 Mac or Scaleway Apple silicon, same setup |
| Linux / any Docker host | `docker build -t inference-engine .` then `docker run --stop-timeout 30 -p 8000:8000 -v "$PWD/models:/app/models:ro" inference-engine` (CPU backend, Qwen3.5-0.8B in fp32 by default; for Linux hosts and CI with RAM to spare, not for a Mac) |

Docker on macOS cannot reach the Apple GPU (containers run in a Linux VM), so on a Mac the deployment is the native
process. The CPU image (1.37 GB) is for Linux hosts and CI. On this 8 GB Mac under Docker Desktop, even with a 5 GB VM,
the Qwen3.5-0.8B default was ready in 51 s and passed the healthcheck but decoded at ~0.1 tok/s, because the fp32
weights do not stay resident in the VM (`docs/bench/raw/docker_2026-10-01.md`; natively, the same model on the CPU
does 11.7 tok/s). With the previous default, Qwen2.5-0.5B (since removed), the image was ready in ~12 s, used 2.4 GB,
decoded ~4 tok/s, and `docker stop` drained an in-flight stream before exiting. See
[`docs/design.md`](docs/design.md#9-deployment).

## Layout

```
src/weight_loader.py   safetensors mmap, sharded index  src/state.py        KV caches, block allocator, HybridState
src/tokenizer.py       byte-level BPE                   src/quant.py        INT8/INT4, policy, .qt files
src/chat.py            ChatML, model Jinja templates    src/sampler.py      temperature/top-k/top-p/repetition
src/config.py          Qwen35Config, Cohere2Config
src/models/            Qwen3.5, Cohere2, packed prefill src/engine.py       model registry, load_engine
src/backend/           protocol, torch reference, Metal src/kernels/        *.metal kernels
src/native/            Objective-C++ Metal runtime      src/server/         scheduler, API, metrics
scripts/               goldens, bench, quantize, calibrate, serve, loadgen, repl, run_tests
```

## Tiny Aya (in progress)

The engine is being extended to [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (Cohere2, 3.35B
parameters, 70 languages): plan in [`docs/tiny-aya-plan.md`](docs/tiny-aya-plan.md), results in
[`docs/results-log.md`](docs/results-log.md). Tiny Aya is by Cohere and Cohere Labs. Its weights are not part of this
repository: they are gated on Hugging Face, licensed
[CC-BY-NC 4.0 with an Acceptable Use Addendum](https://cohere.com/cohere-labs-cc-by-nc-license), subject to the
[Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy), and provided as
is, without warranty. This is an independent, non-commercial project, not affiliated with or endorsed by Cohere; sample
outputs quoted in `docs/` are model-generated.

```bibtex
@misc{salamanca2026tinyayabridgingscale,
      title={Tiny Aya: Bridging Scale and Multilingual Depth},
      author={Alejandro R. Salamanca and Diana Abagyan and Daniel D'souza and Ammar Khairi and David Mora and Saurabh Dash and Viraat Aryabumi and Sara Rajaee and Mehrnaz Mofakhami and Ananya Sahu and Thomas Euyang and Brittawnya Prince and Madeline Smith and Hangyu Lin and Acyr Locatelli and Sara Hooker and Tom Kocmi and Aidan Gomez and Ivan Zhang and Phil Blunsom and Nick Frosst and Joelle Pineau and Beyza Ermis and Ahmet Üstün and Julia Kreutzer and Marzieh Fadaee},
      year={2026},
      eprint={2603.11510},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.11510},
}
```

## License

[Apache-2.0](LICENSE). Model weights are not part of this repository; each model keeps its own licence.
