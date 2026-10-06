# Porting Tiny Aya (Cohere2) to this engine: plan

Written 2026-10-04, before any code. Results and changes go to [`results-log.md`](results-log.md) as milestones land.

Every fact below was checked against a primary source; anything not yet confirmed is marked UNVERIFIED. Sources: `transformers` 5.16.1 `models/cohere2/` (local), the official small files already in
`models/tiny-aya-global/` (downloaded 2026-10-02 from commit `af89d219`, hash-matched to the official repo), the tech
report (arXiv 2603.11510), the model cards, the HF API, and the engine's own code and docs.

## 1. Tiny Aya Global in one table

| item | value | source |
|---|---|---|
| class | `Cohere2ForCausalLM` (model_type `cohere2`) | config.json |
| parameters | 3,349,227,520, all BF16 (6.70 GB): embedding 536,870,912 + 36 x 78,120,960 + 2,048 | index total_size = 2 x params; exact recount |
| layers / hidden / FFN | 36 / 2048 / 11008 (SwiGLU, SiLU) | config; paper Table 2 |
| heads | 16 query, 4 KV (GQA, 4 query heads per KV head), head_dim 128 | config; configuration_cohere2.py:93 |
| attention pattern | 3 sliding-window layers then 1 full layer, repeating: full = layers 3, 7, ..., 35 (9 full, 27 sliding) | config `layer_types` |
| sliding window | 4096 (a token sees itself and the 4095 before it) | config; masking_utils.py:98-99 |
| positions | RoPE theta 50000, interleaved (GPT-J) pairs, all 128 dims, **sliding layers only**; full layers have no positional encoding | modeling_cohere2.py:89, 150-155, 229-230 |
| norm | LayerNorm (subtracts the mean), weight only, no bias, eps 1e-5, computed in fp32 | modeling_cohere2.py:96-110 |
| block | parallel: `h + attn(LN(h)) + mlp(LN(h))`, one norm per layer | modeling_cohere2.py:306-318 |
| head | tied to the embedding (no lm_head tensor), logits x `logit_scale` = 1.0 (class default is 0.0625: read it from config) | config; configuration_cohere2.py:69, 81 |
| biases | none | config; modeling :262-264 |
| vocabulary | 262,144 embedding rows; tokenizer defines 261,008 ids, so 1,136 rows have no token (the engine already slices logits to `tokenizer.vocab_size()`; test that it is 261,008) | tokenizer.json |
| tokenizer | byte-level BPE, 260,729 merges, no normalizer, pre-tokenizer = digit split (groups of 3) + GPT-4o regex + ByteLevel; BOS prepended | tokenizer.json |
| stop ids | 3 `<EOS_TOKEN>`, 6 `<|END_OF_TURN_TOKEN|>`, 261001 `<|END_RESPONSE|>` | generation_config lists only 3; tokenizer_config eos is 6 |
| context | config 8192; card and paper say 8K input + 8K output (UNVERIFIED upper bound); GGUF says 500000, ignore | config; Table 2; HF API |
| chat | fixed ~360-token system preamble on every chat (a one-line question renders to 372 tokens) | official chat template |
| sampling (card) | temperature 0.1, top_p 0.95, top_k 50 (HF default) | model card |

Variants: earth, fire and water have byte-identical config.json, index and tokenizer.json to global, so they load
the same way. Base has a different tokenizer file and no chat template. The thinkers and base-32K differ: 181 extra bias tensors
in the thinkers (names hash-verified; values presumed zero), 32K context, and a tokenizer file whose post-processor
appends EOS. Later, if ever.

## 2. New ideas compared with Qwen3.5 (one line each)

- **LayerNorm vs RMSNorm**: RMSNorm only rescales a vector by its size; LayerNorm first subtracts the vector's mean, then rescales.
- **Parallel block**: attention and the MLP both read the same normalised input and both add to the residual, instead of running one after the other.
- **Sliding-window attention**: a layer only looks back a fixed distance (4096 tokens), so its cache can stop growing.
- **NoPE layers**: the 9 full-attention layers get no position signal at all; order is inferred from the causal mask and from the sliding layers.
- **Interleaved vs half-split RoPE**: both rotate pairs of numbers; interleaved pairs neighbours (0,1), (2,3), ...; half-split pairs (0,64), (1,65), ... (Qwen and the engine's kernel).
- **Tied embeddings + logit scale**: one 262k x 2048 matrix is both the input lookup table and the output scorer; the scores are multiplied by a constant (1.0 here).

## 3. Engine gaps (component, today, change, size, risk)

Size: S = under a day; M = a few days; L = a week or more.

| component | engine today | change | size | risk |
|---|---|---|---|---|
| norm | RMSNorm(1+w) only | `layer_norm` op + Metal kernel (copy of rms_norm_f32w plus a mean pass) | S | low |
| parallel block | sequential block in qwen3_5.py | one LN, `a = o_proj(.., residual=h)`, `h = mlp(x, residual=a)` (same add order as HF) | S | low |
| RoPE | half-split kernel | **permute each head's W_q/W_k rows [0,2,..,126,1,3,..,127] at load**: exact, reuses the kernel (verified mathematically); skip RoPE on full layers | S | low |
| attention <= 4096 tokens | causal mask only | nothing: below 4097 tokens the window excludes no key, so sliding = full causal exactly | - | - |
| attention 4097-8192 | no window | window start in prefill mask and in the 6 loop starts of the two decode kernels + new kernel arguments | S-M | med (off-by-one) |
| KV cache | **fp32, one block table shared by all layers** | **144 KiB/token, 6x Qwen3.5-2B** -> bf16 KV (kernel variants + pool dtype) is the priority; window block freeing needs split pools (9 full / 27 sliding) | M / M-L | med |
| tokenizer | uses only the first pre-tokenizer with `findall`, always NFC | **today it would drop all non-digit text (engine matched HF on 4 of 286 research cases)**; every Isolated Split, NFC only when asked, `\b` with Oniguruma's word characters, BOS as a per-call argument (decision options must never get BOS). DONE in M1 | S | low after fix |
| chat template | Qwen ChatML only, hard-coded in src/chat.py | the model's own Jinja template, rendered as transformers renders it (`TemplateChat`, `chat="template"`), so the ~360-token preamble is read from `models/tiny-aya-global/tokenizer_config.json` at load time and never copied into committed code. DONE in M1 | S | low |
| loader | one safetensors file, prefix `model.language_model.` | sharded loader from the index (also in quantize.py); prefix `model.`; `Cohere2Config.from_json` (tie default True, logit_scale from config) | S | low |
| quantization | save_qt keeps output in RAM; calibration loads bf16 on MPS | write .qt incrementally (INT8 output 3.56 GB + a 5 GB shard mmap on 8 GB); calibrate INT4 damage against an all-INT8 baseline | M | med |
| model interface | none (`Engine.model: object`); Qwen leaks in decision, scheduler, goldens | small base class (weight cache, embed, mlp, forward skeleton, pools) + family registry; `cohere2.py` ~200 lines as the readable spec; M2 shortcut: reuse HybridState with 0 linear layers | M | low |
| server | Qwen defaults, `</think>` filter, template errors -> 500 | done in M1: registry entry, stop ids {3, 6, 261001}, contiguous-vocab and stop-id guard at load, 400 on template errors, 400 for `enable_thinking` on a model without a thinking mode, the model's length cap in `create_app`. Left: request defaults from the registry's sampling (M6), a model-level assert (M2; repl/generate/bench bypass the server cap) | S | low |
| decide endpoint | copies the context KV per option | at 144 KiB/token, a 4K fork is ~0.6 GB: group size drops to 1; attention-only models can share a read-only context KV (copy-on-write, already in design.md) | M | med |

## 4. Budgets

**Disk** (~17.7 GB free before the download; 11 GiB after it, measured 2026-10-04 21:50; `models/qwen3.5-2b` uses 7.4 GiB, of which 4.55 GB is its BF16 safetensors):
1. the two BF16 shards: 6.70 GB, pinned to revision `af89d219b53ed9b13b8a4645f9c8028973510324` (DONE)
2. INT8 .qt: ~3.56 GB; INT4 .qt: ~2.33 GB
3. total ~12.6 GB, leaving ~5.1 GB (4.8 GiB): tight, because macOS swap (7 GB allocated now) lives on the same disk and grows during quantization and reference runs
4. options: skip or delete the INT8 .qt after its KL check; load HF only from the local folder (never a second copy in ~/.cache/huggingface); the q8_0 GGUF (3.57 GB) only if llama.cpp is used as a second reference
5. M7 needs the q4_0 (2.03 GB) and q4_k_m (2.14 GB) GGUFs plus a self-converted MLX 4-bit (~1.9 GB): ~6.1 GB, more than what is left. Before M7, delete the INT8 .qt and/or move Qwen3.5-2B's BF16 safetensors (4.55 GB) off the Mac. A second variant (fire) needs its own 6.70 GB download, which cannot sit beside global's shards

**Weights in RAM**: INT4 body 2,812,280,832 x 0.625 B = 1.76 GB (0.5 B data + an fp16 scale and an fp16 min per 32 weights) + INT8 tied head 0.57 GB = **~2.33 GB**, a floor: each group calibration keeps at INT8 adds 0.4375 B/weight (Qwen3.5-2B INT4: 1.41 GB). The head is ~25% of the bytes read per token.

**KV cache** (per token, all 36 layers): fp32 144 KiB, bf16 72 KiB.
- one 4096-token sequence: 576 MiB fp32 / 288 MiB bf16
- one 8192-token sequence with window freeing: 720 MiB fp32 / 360 MiB bf16 (without freeing: 1,152 / 576 MiB)
- 8 sequences x 4096: 4.5 GiB fp32 (does not fit beside the weights) / 2.25 GiB bf16
- the current default pool (1024 x 16 tokens) would be 2.25 GiB in fp32: size it per model

**Decode speed (estimate, UNVERIFIED)**: ~2.33 GB read per token at batch 1 -> ceiling ~43 tok/s at the ~100 GB/s the engine's docs use; Qwen3.5-2B's INT4 path reaches 71% of its ceiling, so expect **~30 tok/s** (Qwen3.5-2B INT4: 50.1 tok/s). Long contexts add KV reads (0.6 GB per step at 4K fp32, possibly more because each query head re-reads its KV head: measure).

## 5. Correctness on 8 GB (the main schedule risk)

HF fp32 Tiny Aya needs 13.4 GB, bf16 6.7 GB; neither fits reliably, and unlike Qwen3.5 there is no small twin.

1. **Tiny random Cohere2 models** (highest value): build small models with HF from `Cohere2Config` (e.g. hidden 64, 8 layers, logit_scale 0.25), save them to scratch, and require engine == HF in fp32 on CPU. Two configs: **tiny-full** (sliding_window at least the longest test sequence, so the window never binds) for M1-M2 and the scheduler invariant suites (batched == sequential, preempted == uninterrupted, packed == alone, forks); **tiny-window** (window 8) for M5: sequences past the window, chunked prefill across the window edge, freed blocks. Seconds per run, no download.
2. **Real-model fp32 answer key, one layer at a time**: build HF's own `Cohere2DecoderLayer`, load its tensors with `safe_open`, run, keep the output, free it; the tied head in row chunks. Peak ~2.5 GB. Pair it with an engine fp32 CPU mode that does not keep widened weights. Compare every layer, the logits and greedy output on multilingual prompts (including Hindi).
3. **INT8 / INT4 on Metal**: KL and top-1 flips against that key, the same way as Qwen3.5-2B.
4. Greedy goldens come from the layer-streamed key in item 2 (re-run the 36 streamed layers per new token, or keep each layer's K/V between steps), stopping on [3, 6, 261001], with an explicit BOS policy. A whole-model HF bf16 `generate()` is optional: try it with `attn_implementation="eager"` under memory monitoring (UNVERIFIED that it completes on 8 GB). Note: HF casts cos/sin to the activation dtype, so compare against fp32 layers.
5. Optional second opinion: llama.cpp on the public q8_0 GGUF, always with an explicit `-c` (its metadata says 500000).

## 6. Milestones (each with an exit test)

| # | milestone | exit test | size |
|---|---|---|---|
| M0 | Apache-2.0 LICENSE; download the shards (pinned revision) | DONE 2026-10-04: SHA-256 of both shards == their HF LFS hashes. Remaining: optional Sigstore check (`model_signing verify`, needs a pip install in a scratch venv); `integrity_check` (truncation only) once the sharded loader exists | S |
| M1 | infrastructure: sharded loader, `Cohere2Config`, tokenizer fix + per-call BOS, the model's own chat template (`TemplateChat`), registry entry | DONE: both tokenizers == HF on 191/191 answer-key cases (Qwen's original 66 unchanged); chat == `apply_chat_template` on 10 conversations x 2, text and ids; `tests/test_aya.py` 40/40 (see results-log.md) | S-M |
| M2 | `Cohere2Model` (fp32, CPU, no window code) on tiny-full; the engine's max_model_len = min(registry, `model.max_positions` = the window), also enforced by the scheduler | DONE: `tests/test_cohere2.py` 21/21 on random models with Tiny Aya's head ratio: every layer and logits within 1.5e-6 of HF, greedy identical, cached == uncached, paged == contiguous, batched and packed == alone, forks, the window guard atomic for fresh, continuing, packed and batched sequences. Registering `cohere2` in `load_engine` waits for a load path that holds the model (M3 / M4) | M |
| M3 | real-model fp32 answer key (layer-streamed) + engine fp32 CPU mode | DONE: `scripts/golden_aya.py` runs transformers' own decoder layers one at a time (bit-identical to the full model on random models; 212 s, 2.7 GB for 9 prompts); `Cohere2Model(stream=True)` vs that key: every layer within 6.5e-6 (9.4e-6 for the worst single token), logits 5.7e-6, chat hidden states 1.5e-5 per token, 81 decode steps 1.4e-5, greedy identical on 9/9 prompts in 5 languages, code and chat (178 s, 2.7 GB) | M |
| M4 | quantization: incremental .qt writer, `quantize()` in row chunks, INT8/INT4, INT8-baseline calibration, Metal `layer_norm`, kernel tests at 16/4 x 128; registered for the quantized Metal backends only | DONE: writer byte-identical to the old one, INT8 3.56 GB / INT4 2.34 GB in ~25 s and ~3 GB; vs the M3 fp32 key (169 positions): INT8 KL 0.0006, 0 flips, greedy 90/90, 19-21 tok/s; INT4 KL 0.105 (0.054 at the 150 positions a server uses: no in-template positions, no decode steps after the model's own stop), 9/116 flips, 29-32 tok/s; an interrupted quantize leaves any earlier .qt as it was. Calibration keeps 1 of 144 tensors INT8: Tiny Aya's INT4 damage is spread over all tensors (top 10 = 36%), so a better 4-bit format, not a bigger policy, would be the next step | M |
| M5 | memory: bf16 KV, then window support to 8192, then window block freeing (split pools) | DONE. Part A (bf16 KV): templated decode-attention kernels read bf16 K/V (bit-identical to the fp32 kernel on the widened cache, a wrong dtype refused), `kv_dtype` per model (Tiny Aya bf16, Qwen fp32); vs the M3 key, INT8 KL change +0.0000, INT4 -0.0014, same flips and greedy tokens, same decode speed (the kernel is not bandwidth-bound: 807 vs 849 us at 4K). Part B1-B5 DONE: windowed attention on every path (ops.attention band mask + key suffix, Metal paged kernel s0, contiguous decode by host slice), exhaustively equal to a float64 oracle with HF's predicate (18,320 cases; W +- 1 caught), the tiny-window model == HF past the window on every path (stateless, 6 chunk schedules, block sizes 4 and 3, greedy 20 tokens, batched, packed, forks) with pre-window blocks NaN-poisoned; Metal RoPE now reads transformers' fp32 frequency table (drift 2e-4 / 4e-4 relative at 4095 / 8191 -> ~1e-7 flat; Qwen3.5-2B numbers unchanged to 4 decimals); in-place attention scores (bit-identical), chunked generate prefill, server cap defaults to the model's. B6/B7 DONE: the real model past the window vs a 4,802-token transformers key in 8 languages (public-domain texts from each work's opening sentence, `scripts/long_texts.py`): fp32 engine every position within 3.4e-5, logits 5.7e-6, decode 2.5e-6, greedy 10/10, window-off sensitivity 3.9; Metal INT8 KL 0.0008 (fp32 and bf16 KV, greedy 9/10), INT4 0.095 (6/10); bf16 vs fp32 KV out to 7.6K tokens KL 0.0000, greedy 32/32; cap 8192. Part C DONE: one allocator over 9-layer units, a per-sequence ring of blocks for the sliding layers (exact vs HF with every slot reused; scheduler accounting in units, preemption exact); an 8K sequence holds 388 MiB of bf16 KV (576 with one table, 1,152 in fp32); one buffer for K and V (the MPS allocator puts a 10-512 MiB tensor in a 1 GiB heap, a larger one in a heap of its own size); Tiny Aya's server pool 768 blocks (864 MiB); an 8K prefill with it peaks at 4,880 MiB with INT4 (`bench.py --trace`), under the 5 GiB bound; INT8 at 8K holds 5.48 GiB, over the 5.33 GiB Metal recommends, so INT8 serves at most 4,096 tokens from a 512-block pool (registry `per_backend`; server and REPL): 4.45 GiB generating at 4K, 4.82 scoring `/v1/decide` options (INT4 decide at 8K: 5.00). A decode failure with B6's symptom, found: GPU memory exhaustion (PyTorch's 1 GiB heaps took INT8 + fp32 KV at 4.8K to 5.21 of the 5.33 GiB Metal recommends; with another GPU user, Metal aborted command buffers and PyTorch 2.14 reports nothing; B6's own run left no abort in macOS's log, so its cause stays open); fixed with a low allocator watermark (4.70 GiB, decode exact; INT8 at 4K over 6 alternating runs, prefill no slower and decode within a few percent), and the long test now gates GPU memory and reads Metal's aborts from macOS's log. Also fixed: a latent `non_blocking` host-copy hazard for callers that reuse their tensors (`state.to_device`), and quantization gates that judged only the bf16-KV run | M-L |
| M6 | serving: request defaults from the registry's sampling, invariant suites, decide-endpoint fork cost (copy-on-write), a heap-based BPE merge (long unspaced text is quadratic today: ~2 min for 64K CJK characters, Qwen too); optional prefix caching of the ~360-token preamble | all server suites green on the tiny model and on INT4 Aya; TTFT with and without the cached preamble | M |
| M7 | benchmarks and results | single-stream and batch-8 tok/s with fixed token counts (llama.cpp ignore-eos: the GGUF's eos is 3, chat turns end on 6/261001), TTFT vs llama.cpp q4_0 / q4_k_m (official GGUFs) and MLX 4-bit (convert yourself: there is no mlx-community 4-bit global); to quote the paper's phone numbers, match its 100-in/100-out MLX workload; per-language INT4 KL; tokens per character | M |

Time estimates in weeks come after M1-M2, when the pace on this model is known.

## 7. Risks (ranked)

1. **Correctness reference** cannot be a whole HF model on 8 GB: the layer-streamed key (M3) must work. Fallback: llama.cpp q8_0 as a looser second reference.
2. **KV memory** is 6x Qwen per token: without bf16 KV, batch 8 at 4K does not fit.
3. **Quantization peak RAM**: the incremental writer and INT8-baseline calibration are new code on an 8 GB machine with ~5 GB of free disk for swap.
4. **Window correctness past 4096** (off-by-one at the edge, freed blocks, forks and preemption with split pools).
5. **Prefill memory past 4K**: an fp32 [Hq, T, S] score tensor per layer (268 MB at 8K with 512-token chunks, several transients) beside the weights and KV pool; measure peak RSS, lower the prefill chunk if needed.
6. **Context ambiguity**: 8192 total vs 8K in + 8K out; cap at 8192 until verified.
7. Tokenizer edge cases: `\b` now follows Oniguruma (ZWNJ, ZWJ, superscripts, fractions; in the goldens). Known, rare, documented: Unicode 17 code points (`regex` has Unicode 17 tables, Oniguruma 16) and, for Qwen's NFC, a few dozen combining marks newer than tokenizers' tables.

## 8. Decisions (2026-10-04)

1. **Variant**: tiny-aya-global first (the default "best balance", official GGUF baseline, most downloads). Fire second (South Asian languages; earth/fire/water share global's config, index and tokenizer, so no new code, but each needs its own 6.70 GB download and quantization, which does not fit beside global's shards on this disk). Thinkers and 32K: not now.
2. **Context target**: 4096 first (exact with no window code), then 8192.
3. **bf16 KV**: yes. It changes the engine's "fp32 KV to match the reference exactly" rule, so its cost is measured as a KL change.
4. **Disk**: keep Qwen3.5-2B's BF16 safetensors for now; revisit before M4/M7 (see the disk budget).
5. **License**: Apache-2.0 for this repository's code.

## 9. Results to aim for

1. Correctness: every layer and the logits against HF at fp32 (layer-streamed), KL for INT8 / INT4 weights and for the bf16 KV cache (bf16 weights, 6.70 GB, do not fit on 8 GB), greedy match on multilingual prompts.
2. Per-language INT4 KL for ~20 languages grouped by web presence, computed on the fly (never store full 262,144-wide fp32 logits: 1 MiB per token). Next to the official q4_0 / q4_k_m GGUFs on the same text and positions, scored against the same fp32 key (or clearly labelled as llama.cpp's own KLD against a llama.cpp reference). Cite Marchisio et al. 2024 (arXiv 2407.03211: automatic metrics, LLM judges and human evaluation; non-Latin scripts hurt most; automatic metrics underestimate what humans see), so present KL as a diagnostic, not a quality verdict. The Tiny Aya report (section 6, judge scores) finds the quantization penalty grows as web presence falls.
3. On the same M2: single-stream tok/s, TTFT and batch-8 aggregate tok/s against llama.cpp and MLX.
4. Concurrent sequences in a fixed RAM budget, with and without freeing sliding-window blocks (frame it as a measurement on 8 GB, not a capability others lack).
5. Tokens per character per language, Tiny Aya vs Qwen3.5 (FLORES-200), turned into characters served per second.

## 10. Licence and access (short)

- Access: the weights are gated on Hugging Face (accept the terms on the model page, automatic approval), then `hf download CohereLabs/tiny-aya-global --revision af89d219b53ed9b13b8a4645f9c8028973510324 --local-dir models/tiny-aya-global`.
- Code, numbers, correctness results and labelled sample outputs can very likely be published without asking: the weights' CC-BY-NC 4.0 + AUP say nothing against evaluations, and the API agreement's benchmarking ban covers Cohere's API products. This is a reading, not an explicit grant (not legal advice). Never use Tiny Aya through Cohere's API for comparisons (that is under the API's no-benchmarking terms).
- Do not publish weights (.qt), and do not commit copies of tokenizer.json, config or the chat template, including its ~360-token preamble (they sit behind the gate under the repo's CC-BY-NC tag): load them from the local folder.
- README: "Tiny Aya by Cohere and Cohere Labs", model link, "CC-BY-NC 4.0 with Acceptable Use Addendum" and AUP links, the paper's BibTeX, "independent, non-commercial; not affiliated with or endorsed by Cohere", no logos. DONE (README section "Tiny Aya").
