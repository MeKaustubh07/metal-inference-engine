"""Tiny Aya port, M5 B6: the real model past its 4096-token sliding window, fp32 on the CPU (streamed one layer at a
time), against the long answer key from transformers' own decoder layers (scripts/golden_aya.py --long, gitignored):
~4.8K tokens of public-domain text in 8 languages (scripts/long_texts.py). Compared per row (each token's worst element
over its own largest one): the final hidden state of every position, the logits of the key's rows (every 64th, the
window's edge 4090-4100, the last 8), 9 decode steps and 10 greedy tokens. Sensitivity: the same tokens from position
4096 on, with the window turned off, must land far from the key, so the comparison can tell a window from none.
The quantized engine on Metal past the window: tests/test_aya_long_quant.py (a separate process, so neither holds
the other's memory).
Needs the gated files and the key; otherwise SKIP, or FAIL naming the command."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from config import Cohere2Config
from models.cohere2 import Cohere2Model
from tokenizer import Tokenizer
from weight_loader import open_weights

D, KEY = "models/tiny-aya-global", "tests/golden_aya_long/0.pt"
if not os.path.exists(f"{D}/model.safetensors.index.json"):
    print(f"SKIP: needs the gated files in {D}")
    sys.exit(0)
if not os.path.exists(KEY):
    print("FAIL: no long answer key: run scripts/.venv/bin/python scripts/golden_aya.py --long")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def rel_rows(a, b):
    """Each row's worst error over that row's largest value -> per-row errors [rows]."""
    return (a - b).abs().amax(-1) / b.abs().amax(-1)


t0 = time.perf_counter()
key = torch.load(KEY)
ids, n, W = key["ids"], len(key["ids"]), 4096
tok = Tokenizer(f"{D}/tokenizer.json")
cfg = Cohere2Config.from_json(f"{D}/config.json")
model = Cohere2Model(cfg, open_weights(f"{D}/model.safetensors.index.json"), stream=True)

# 1. the engine's tokenizer gives the key's ids for the whole text (made with transformers' tokenizer)
import long_texts
from tokenizers import Tokenizer as HFTokenizer
hf_tok = HFTokenizer.from_file(f"{D}/tokenizer.json")
class HFEnc:
    @staticmethod
    def encode(text):
        return hf_tok.encode(text, add_special_tokens=False).ids
text = long_texts.build(HFEnc, key["tokens_per_language"])
check(f"ids from the engine's tokenizer == the key's for the {n}-token text in {len(key['languages'])} languages "
      f"({', '.join(key['languages'])})", tok.encode(text, add_bos=True) == ids.tolist())

# 2. prefill in 1024-token chunks into one contiguous fp32 state (the window first binds at position 4096, a chunk
# boundary: a copy of the state is kept there for the sensitivity check)
CHUNK = 1024
st, hs, fork = model.new_state(n + 16), [], None
for i in range(0, n, CHUNK):
    if i == W:
        fork = st.fork()
    hs.append(model.packed_hidden([(ids[i:i + CHUNK], st)]))
    print(f"      {time.perf_counter() - t0:5.0f} s  prefill {min(i + CHUNK, n)}/{n}", flush=True)
h = torch.cat(hs)
err = rel_rows(h, key["final_norm"])
before, after = err[:W].max().item(), err[W:].max().item()
check(f"final norm of all {n} positions == the key's: worst row {before:.1e} before position {W}, {after:.1e} from it "
      f"on (median {err.median().item():.1e})", max(before, after) < 1e-4)
logits = model.head(h[key["rows"]])
lerr = rel_rows(logits, key["logits"])
check(f"logits at the key's {len(key['rows'])} rows (every 64th, {W - 6}-{W + 4}, the last 8): worst row "
      f"{lerr.max().item():.1e}", lerr.max().item() < 1e-4)

# 3. 9 decode steps on the reference's own greedy tokens, then 10 free greedy tokens
steps = torch.cat([model.forward(torch.tensor([t]), state=st) for t in key["greedy"][:-1].tolist()])
serr = rel_rows(steps, key["step_logits"]).max().item()
check(f"9 decode steps at positions {n}-{n + 8} == the key's, worst row {serr:.1e}", serr < 1e-4)
first = int(logits[-1].argmax())
check(f"greedy: the first new token == the key's ({first}), and the key's 10 tokens are the argmax at every step",
      first == int(key["greedy"][0]) and torch.equal(steps.argmax(-1), key["greedy"][1:]))

# 4. sensitivity: from position 4096 with the window off, the rows the window changes must be far from the key
real = model._window
model._window = lambda i: None
off = model.packed_hidden([(ids[W:], fork)])
model._window = real
off_err = rel_rows(off, key["final_norm"][W:])
check(f"window off from position {W}: worst row {off_err.max().item():.1e} (median {off_err.median().item():.1e}), "
      f"vs {after:.1e} with it", off_err.max().item() > 100 * after)
print(f"      {time.perf_counter() - t0:.0f} s")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
