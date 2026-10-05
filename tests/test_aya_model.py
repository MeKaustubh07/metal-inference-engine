"""Tiny Aya port, M3: the real model (3.35B parameters, fp32 on the CPU, streamed one layer at a time) against the
answer key from transformers' own decoder layers (scripts/golden_aya.py, gitignored): every layer and the logits of
7 raw prompts (5 languages and code), 10 greedy tokens each with the logits of every decode step, and 2 chat prompts
end to end (the engine's tokenizer and chat template, the final hidden state of every position, the logits of the last
8, greedy). Errors are measured per row (each token's worst element over its own largest one) as well as over the whole
tensor, because the first (BOS) token's activations are up to 50x larger than the others' and would hide their errors. All prompts go together: one packed prefill, then one batched
decode step per token, so each weight is widened ~10 times in all. Without the gated files it prints SKIP; with
them but without the answer key it fails and says how to make it."""
import glob
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
from chat import TemplateChat, format_chat
from config import Cohere2Config
from models.cohere2 import Cohere2Model
from tokenizer import Tokenizer
from weight_loader import open_weights

D, G = "models/tiny-aya-global", "tests/golden_aya"
if not os.path.exists(f"{D}/model.safetensors.index.json"):
    print(f"SKIP: needs the gated files in {D}")
    sys.exit(0)
goldens = sorted(glob.glob(f"{G}/*.pt"), key=lambda f: int(os.path.basename(f)[:-3]))
if not goldens:
    print("FAIL: no answer key: run scripts/.venv/bin/python scripts/golden_aya.py")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def rel_rows(a, b):
    """Each row's worst error over that row's largest value; the worst row."""
    return ((a - b).abs().amax(-1) / b.abs().amax(-1)).max().item()


t0 = time.perf_counter()
recs = [torch.load(f) for f in goldens]
tok, chat = Tokenizer(f"{D}/tokenizer.json"), TemplateChat(f"{D}/tokenizer_config.json")
model = Cohere2Model(Cohere2Config.from_json(f"{D}/config.json"), open_weights(f"{D}/model.safetensors.index.json"),
                     stream=True)

# 1. the engine's tokenizer and chat template give the answer key's ids (made with transformers' tokenizer)
def engine_ids(r):
    if r["chat"]:
        return tok.encode(format_chat([{"role": "user", "content": r["text"]}], style=chat))
    return tok.encode(r["text"], add_bos=True)
check(f"ids from the engine's tokenizer and chat template == the answer key's, all {len(recs)} prompts "
      f"({sum(len(r['ids']) for r in recs)} tokens)", all(engine_ids(r) == r["ids"].tolist() for r in recs))

# 2. one packed prefill for all prompts (each weight widened once for all of them), each into its own state: every
# layer of the raw prompts, the logits of all their positions, the last 8 of the chat prompts
states = [model.new_state(len(r["ids"]) + 16) for r in recs]
cap = {}
h = model.packed_hidden([(r["ids"], st) for r, st in zip(recs, states)], capture=cap)
lens = [len(r["ids"]) for r in recs]
caps = {k: v.split(lens) for k, v in cap.items()}
rows = [hj[r["logits_from"]:] for hj, r in zip(h.split(lens), recs)]
logits = model.head(torch.cat(rows)).split([len(x) for x in rows])
KEYS = ["embed"] + [f"l{i}_out" for i in range(model.config.num_hidden_layers)] + ["final_norm"]
raw = [j for j, r in enumerate(recs) if not r["chat"]]
inf = float("inf")
per_layer = {k: max(rel(caps[k][j], recs[j][k]) if k in caps else inf for j in raw) for k in KEYS}
per_row = {k: max(rel_rows(caps[k][j], recs[j][k]) if k in caps else inf for j in raw) for k in KEYS}
logit_err = [(rel(logits[j], recs[j]["logits"]), rel_rows(logits[j], recs[j]["logits"])) for j in raw]
chat = [j for j, r in enumerate(recs) if r["chat"]]
chat_err = [(rel_rows(logits[j], recs[j]["logits"]), rel_rows(caps["final_norm"][j], recs[j]["final_norm"])) for j in chat]
first = [int(lg[-1].argmax()) for lg in logits]
embed_exact = all(torch.equal(caps["embed"][j], recs[j]["embed"]) for j in raw)
del h, cap, caps, rows, logits
worst, worst_row = max(per_layer, key=per_layer.get), max(per_row, key=per_row.get)
check(f"the embedding rows of the {len(raw)} raw prompts are bit-identical to the answer key's", embed_exact)
check(f"all {len(KEYS)} captured tensors (embedding, 36 layers, final norm) == the answer key: worst relative error "
      f"{per_layer[worst]:.1e} ({worst}); worst single token {per_row[worst_row]:.1e} ({worst_row})",
      per_layer[worst] < 3e-5 and per_row[worst_row] < 5e-5)
check(f"logits of every position == the answer key: worst {max(e for e, _ in logit_err):.1e}, worst single token "
      f"{max(r for _, r in logit_err):.1e}", max(e for e, _ in logit_err) < 3e-5 and max(r for _, r in logit_err) < 5e-5)
check(f"chat prompts ({', '.join(str(len(recs[j]['ids'])) for j in chat)} tokens): the final hidden state of every "
      f"position (worst token {max(f for _, f in chat_err):.1e}) and the logits of the last 8 (worst token "
      f"{max(l for l, _ in chat_err):.1e}) == the answer key",
      max(f for _, f in chat_err) < 8e-5 and max(l for l, _ in chat_err) < 5e-5)

# 3. greedy: 9 more tokens for every prompt, one batched decode step for all of them at a time, each step's logits
# compared with the answer key's (the same tokens are fed to both, as long as the greedy tokens agree)
tokens, step_err = [[f] for f in first], 0.0
for k in range(9):
    out = model.decode_batch([t[-1] for t in tokens], states)
    for t, row, r in zip(tokens, out, recs):
        t.append(int(row.argmax()))
        step_err = max(step_err, rel_rows(row[None], r["step_logits"][k][None]))
same = [t == r["greedy"].tolist() for t, r in zip(tokens, recs)]
for t, r, ok in zip(tokens, recs, same):
    print(f"      {'ok  ' if ok else 'DIFF'} {r['text'][:40]!r} -> {tok.decode(t)!r}")
check(f"greedy: 10 tokens identical to the answer key on {sum(same)}/{len(same)} prompts; the logits of all "
      f"{9 * len(recs)} decode steps == the answer key's (worst {step_err:.1e})", all(same) and step_err < 7e-5)
print(f"\n{sum(results)}/{len(results)} passed ({time.perf_counter() - t0:.0f} s)")
sys.exit(0 if all(results) else 1)
