# Order-invariant decisions (`POST /v1/decide`)

A context, a question and N options in; each option's probability and a decision out, for three question types:
**choice** (pick one), **boolean** (yes / no) and **score** (an ordinal label). Code: `src/decision.py`; design:
`docs/design.md` §3; tests: `tests/test_decision.py` (24 checks). Qwen3.5-2B INT4 on the MacBook Air M2 (8 GB).

## How it stays order-invariant

Listing options in a prompt gives each a different position and lets later options read earlier ones, so a model's
answer can depend on the order. Here the context is prefilled once, each option continues from its own **fork** of
the context's state (attention K/V, and the DeltaNet recurrent state, which has no attention mask, so isolation
means a copy), every option starts at the same position, and all options run in one packed pass, packed in a
canonical order. The same idea as JEV-style decision networks, which build a custom attention mask and position ids
for a pure-attention model; on a hybrid model the recurrent layers need the fork. An option scores as a whole word:
its tokens, then a token that cannot continue its last word or number, so "1" is not credited with "10".

| property | result |
|---|---|
| a fork continues exactly like the original; feeding a fork leaves the original untouched | max \|d\| 0.0 (CPU fp32 and Metal INT4) |
| forked scores == each option scored alone as context + option | 5.3e-5 (CPU fp32), 5.7e-6 (Metal INT4) |
| shuffled options: scores, probabilities and embeddings | **bit-identical** (CPU and Metal) |
| shuffled options on the dataset below | **40 / 40** identical decisions and probabilities |
| "What is 7 + 3?" with options 1 / 10 / 7 | picks "10" (P("1" as a word) 0.003, P("10") 0.981) |

## Zero-shot accuracy on a decision dataset

`avbiswas/bev-decision` (the dataset of the JEV livestream), test split: 300 questions, 100 per type, sampled with a
fixed seed, prompts of at most 1,536 tokens (raw: `raw/decision_eval_2026-10-02.md`). No training: the 2B's own
probabilities.

| type | accuracy | baselines |
|---|---:|---|
| choice (2–10 options) | **44.0%** | random 24.1%, always the first option 29.0% |
| boolean | **73.0%** | majority class 70.0% |
| score (5-point) | **33.0%** exact, 68.0% within one; mean error of the expected label 1.07 | always the middle label 25.0% exact, mean error 1.16 |

Well above chance on choice, barely above the majority class on yes/no questions: a 2B model scoring zero-shot is a
floor, not a trained decision model. 0.86 s per question (mean prompt 183 tokens).

## Cost: forking vs re-reading the context per option

(raw: `raw/decision_cost_2026-10-02.md`; both paths prefill in 512-token chunks)

| context tokens | options | forked | naive | speed-up |
|---:|---:|---:|---:|---:|
| 89 | 4 | 0.67 s | 1.81 s | 2.7× |
| 89 | 16 | 1.05 s | 6.71 s | 6.4× |
| 439 | 4 | 1.39 s | 4.65 s | 3.4× |
| 439 | 16 | 1.89 s | 17.75 s | 9.4× |
| 1,279 | 4 | 3.36 s | 12.53 s | 3.7× |
| 1,279 | 16 | 3.81 s | 53.25 s | 14.0× |

The two paths' scores agree to ≤ 1.5e-5. The saving grows with the number of options: the context is read once.

## On the server

Decisions run on the engine thread as stepwise jobs: one context chunk or option group per loop iteration, with a
decode step for running streams in between (a stream generated 8 tokens during a 6-chunk decision in the test).
Queued decisions count toward `/ready` and the `waiting_jobs` gauge; a client that disconnects cancels its job.
Forks are standalone states outside the KV pool, kept under 256 MB at a time.

## Limitations and next steps

- Forks copy the context's K/V only when several options fit a group; long contexts score each option on the
  context's own state and rewind it (bit-identical, no copy; M6 step 6). Reusing the pinned chat preamble for
  decide's context would also save its prefill.
- The JEV-style "answer token that sees every option" has no order-invariant counterpart in the DeltaNet layers;
  options are scored independently.
- Zero-shot only. A trained head over the per-option embeddings (`"embeddings": true`) is future work.
