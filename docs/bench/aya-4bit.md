# Tiny Aya at 4 bits, per language: this engine vs llama.cpp vs MLX-LM

How far each 4-bit build of Tiny Aya Global (3.35B) moves the next-token distribution from the full-precision model,
language by language, on long text; Qwen3.5-2B in this engine on the same text for comparison. Measured 2026-10-10 on the
MacBook Air M2 (8 GB), one engine in memory at a time. Raw output: [`raw/aya_lang_kl/`](raw/aya_lang_kl/); script:
`scripts/lang_kl.py`.

## Method

- **Text:** two 512-token windows per language (en fr de es hi ar zh ja), cut at 25% and 60% of one public-domain work
  each (`scripts/long_texts.py`: Austen, Hugo, Kafka, Cervantes, Bharatendu Harishchandra, the Nights, Journey to the
  West, Akutagawa), past their famous openings. One work per language, not parallel text.
- **Scored:** positions 256-510 of each window (510 per language, each with 256+ tokens of context), as
  `llama-perplexity --kl-divergence` scores the second half of each chunk.
- **References:** Tiny Aya in fp32 (transformers' own layers on the CPU, one at a time, as `scripts/golden_aya.py`);
  Qwen3.5-2B HF in bf16 (fp32 does not fit in 8 GB).
- **Metric:** KL(reference || build) in nats per token, with llama.cpp's own formula (tokens whose reference log
  probability is above -16). The reference is written as llama.cpp's KL base file, so llama.cpp scores its GGUFs with
  its own tool and this engine and MLX-LM are scored by `lang_kl.py`'s copy of the formula. llama.cpp reads the files
  exactly as `lang_kl.py` does: its base perplexity equals ours to 5 digits (en 21.5669, zh 95.8392). llama.cpp's
  standard error on each per-language mean is 0.002-0.004 for Q4_K_M and 0.004-0.006 for Q4_0.

| build | weights | what is at higher precision |
|---|---:|---|
| this engine INT4 | 2.34 GB | min/max rounding, block 32, fp16 scale + min (5 bits per weight); tied embedding/head and layer 0's `down_proj` INT8 (`configs/quant/tiny-aya-global.json`) |
| llama.cpp Q4_0 (official GGUF) | 2.03 GB | Q4_0 everywhere, tied embedding Q6_K |
| llama.cpp Q4_K_M (official GGUF) | 2.14 GB | Q4_K (256-weight super-blocks, 32-weight sub-blocks, scales searched); tied embedding, and `ffn_down` + `attn_v` in 18 of 36 layers (0-3, every 3rd, 30-35), Q6_K; no imatrix in the file |
| MLX-LM 0.31.3, `convert -q` | 1.88 GB | nothing: affine, group 64, 4 bits on every matrix, the tied embedding/head included |

## Results (KL, nats per token; lower is better)

| build | en | fr | de | es | hi | ar | zh | ja | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny Aya, this engine INT8 | 0.0005 | 0.0004 | 0.0005 | 0.0005 | 0.0004 | 0.0006 | 0.0006 | 0.0006 | 0.0005 |
| this engine INT4, nothing kept INT8 but the embedding | 0.0810 | 0.0662 | 0.0776 | 0.0846 | 0.0731 | 0.0915 | 0.1119 | 0.1115 | 0.0872 |
| this engine INT4 as shipped | 0.0733 | 0.0599 | 0.0707 | 0.0761 | 0.0693 | 0.0858 | 0.1017 | 0.1018 | 0.0798 |
| this engine INT4, top-10 calibration tensors INT8 | 0.0602 | 0.0503 | 0.0570 | 0.0605 | 0.0536 | 0.0730 | 0.0841 | 0.0818 | 0.0651 |
| llama.cpp Q4_0 | 0.0853 | 0.0869 | 0.1004 | 0.0932 | 0.0780 | 0.1117 | 0.1273 | 0.1327 | 0.1020 |
| **llama.cpp Q4_K_M** | **0.0470** | **0.0405** | **0.0436** | **0.0495** | **0.0363** | **0.0576** | **0.0708** | **0.0632** | **0.0511** |
| MLX-LM `convert -q` (4 bits everywhere) | 0.1481 | 0.1377 | 0.1555 | 0.1651 | 0.1007 | 0.1592 | 0.1863 | 0.1760 | 0.1536 |
| MLX-LM, recipe `mixed_4_6` (1.99 GB) | 0.1164 | 0.1165 | 0.1239 | 0.1326 | 0.0808 | 0.1173 | 0.1448 | 0.1408 | 0.1216 |
| MLX-LM, Q4_K_M's bit layout (2.12 GB) | 0.0736 | 0.0701 | 0.0819 | 0.0818 | 0.0615 | 0.0884 | 0.1141 | 0.1095 | 0.0851 |
| Qwen3.5-2B, this engine INT8 | 0.0009 | 0.0007 | 0.0008 | 0.0009 | 0.0007 | 0.0010 | 0.0010 | 0.0010 | 0.0009 |
| Qwen3.5-2B INT4, nothing kept INT8 but the embedding | 0.0679 | 0.0716 | 0.0708 | 0.0690 | 0.0698 | 0.1198 | 0.1004 | 0.1076 | 0.0846 |
| Qwen3.5-2B INT4 as shipped (1.41 GB) | 0.0547 | 0.0565 | 0.0628 | 0.0577 | 0.0509 | 0.0906 | 0.0786 | 0.0832 | 0.0669 |

The MLX-LM recipes were quantized in memory from the HF checkpoint with MLX-LM's own rounding (`lang_kl.py mlx
--recipe`): `mixed_4_6` is MLX-LM's copy of Q4_K_M's layer rule (`v_proj` and `down_proj` at 6 bits in the same layers;
it promotes `lm_head`, which Tiny Aya does not have: its head is the tied embedding, which stays at 4 bits); "Q4_K_M's
bit layout" adds the tied embedding at 6 bits, so it matches the GGUF tensor for tensor. The in-memory default equals the
converted model to the last digit, and this engine's INT4 quantized at load equals the shipped .qt file.

## Reading the results

- **Same text, the two models are close.** With nothing rescued, Tiny Aya's INT4 is 0.087 and Qwen3.5-2B's 0.085
  (+3%). The gap reported in M4 (KL 0.105 vs 0.045) came from short prompts: 6 of Tiny Aya's 169 positions, 5 of them
  after its own end token or inside the chat template where the reference itself is unsure (without them 0.050).
- **Where the damage sits differs.** Calibration puts 83% of Qwen3.5-2B's position-0 damage in 3 tensors (layer 14
  `down_proj`, layer 15 `o_proj`, layer 6 DeltaNet `out_proj`); Tiny Aya has almost none at position 0 (0.002 summed)
  and spreads the rest over 144 tensors (median 0.0003). On long text the rescues cost about the same per tensor:
  Qwen's 5 tensors -21%, Tiny Aya's 1 -8%, its top 10 -25%.
- **The quantizer decides more than the model.** On the same Tiny Aya, MLX-LM's default 4-bit is 3x llama.cpp's
  Q4_K_M (0.154 vs 0.051). On MLX-LM's rounding: Q4_K_M's 36 six-bit tensors take it to 0.122 (+110 MB), the tied head
  at 6 bits to 0.085 (+130 MB), and llama.cpp's K-quants with the same bits per tensor at nearly the same size (2.14 vs
  2.12 GB) to 0.051. That last 1.7x is the quantizer: sub-block size and scale search together (not separated here).
- **This engine's INT4 is not the best 4-bit build.** Q4_K_M beats it in every language from a file 200 MB smaller
  (0.051 vs 0.080), and keeping its 10 most damaged tensors INT8 (0.065) does not catch up. It beats Q4_0 (0.102) and
  MLX-LM's default. The likely gap is the rounding (min/max per block of 32, no search).
- **Languages.** zh and ja are the two worst in every 4-bit build, 1.2-1.6x en per token (3.7-6x per character: Tiny Aya
  spends 0.91 tokens per Chinese character and 0.24 per English one on these texts); hi is never worse than en. But the
  ranking follows how unsure the reference is on each work (reference perplexity en 21.6, hi 20.1, zh 95.8, ja 113.5;
  rank correlation with the KL 0.90-0.98 across builds), so with one work per language, script and text cannot be told
  apart.
- **Agrees with Cohere's report:** it picks Q4_K_M as the best trade-off (an average judge-score drop of 1.4 points vs
  2.1 for Q4_0, Tiny Aya report §6); here Q4_K_M is the best of the eight 4-bit builds in all eight languages.
- **KL is not quality.** Marchisio et al. (2024) found a quantized model's Japanese drop at 1.7% on automatic metrics
  but 16% with human raters; a KL table cannot show that.

## Reproduce

```bash
scripts/.venv/bin/python scripts/lang_kl.py ref aya REF_AYA            # ~2.5 min CPU, 2.1 GB of reference files
scripts/.venv/bin/python scripts/lang_kl.py ours tiny-aya-global metal-int4 models/tiny-aya-global/model.int4.qt REF_AYA aya_int4.json
scripts/.venv/bin/python scripts/lang_kl.py ours tiny-aya-global metal-int4 - REF_AYA aya_top10.json --keep top10
scripts/.venv/bin/python scripts/lang_kl.py llama tiny-aya-global-q4_k_m.gguf REF_AYA llama_q4_k_m    # brew llama.cpp b11146
MLX_PY scripts/lang_kl.py mlx models/tiny-aya-global REF_AYA mlx_q4km.json --recipe q4km_bits         # an MLX-LM 0.31.3 venv
scripts/.venv/bin/python scripts/lang_kl.py ref qwen REF_QWEN           # ~8.5 min CPU (HF bf16)
scripts/.venv/bin/python scripts/lang_kl.py table RESULTS_DIR
```

Limitations: 510 scored positions per language (s.e. 0.002-0.006 on each mean); one work per language; KL against a
bf16 reference for Qwen3.5 and an fp32 one for Tiny Aya (INT8 shows each reference's floor: 0.0009 vs 0.0005); the
tokenizers differ, so per-token KL across the two models is per different units; no accuracy or human evaluation.
