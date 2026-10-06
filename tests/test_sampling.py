"""Week 6: sampler behaviour (incl. HF parity at the top-p boundary), streaming, batched sampling."""
import gc
import sys
import torch

sys.path.insert(0, "src")
from chat import format_chat
from engine import load_engine
from generate import generate_stream
from sampler import SamplingParams, sample
from tokenizer import Tokenizer
from transformers.generation.logits_process import TopPLogitsWarper

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

OFF = dict(top_k=0, top_p=1.0, repetition_penalty=1.0)

# 1-2. greedy paths
logits = torch.tensor([1.0, 5.0, 2.0, 4.9])
check("temperature 0 picks argmax", sample(logits, [], SamplingParams(temperature=0, repetition_penalty=1.0)) == 1)
g = torch.Generator().manual_seed(0)
check("top_k=1 always picks argmax", all(sample(logits, [], SamplingParams(temperature=2.0, top_k=1, top_p=1.0, repetition_penalty=1.0), g) == 1 for _ in range(50)))

# 3. top-p kept set equals HF's TopPLogitsWarper, including exact-boundary cases
def kept_ours(lg, p, draws=600):
    gen = torch.Generator().manual_seed(1)
    return {sample(lg, [], SamplingParams(temperature=1.0, top_k=0, top_p=p, repetition_penalty=1.0), gen) for _ in range(draws)}
def kept_hf(lg, p):
    out = TopPLogitsWarper(top_p=p)(torch.zeros(1, 1, dtype=torch.long), lg[None].clone())[0]
    return set(torch.nonzero(out > float("-inf")).flatten().tolist())
cases = [
    (torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])), 0.75),       # ordinary case
    (torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125])), 0.75),    # mass before token 2 equals p exactly
    (torch.log(torch.tensor([0.05, 0.15, 0.5, 0.3])), 0.75),       # same distribution, shuffled order
    (torch.log(torch.tensor([0.9, 0.05, 0.05])), 0.5),             # top token alone exceeds p
]
for lg, p in cases:
    ours, hf = kept_ours(lg, p), kept_hf(lg, p)
    check(f"top_p={p} kept set {sorted(ours)} == HF {sorted(hf)}", ours == hf)
check("top_p tiny still keeps one token", kept_ours(torch.tensor([2.0, 1.0, 0.0]), 1e-6) == {0})

# 4. repetition penalty, both branches (HF: divide positive logits, multiply negative ones)
check("repetition penalty, positive logit: repeated token loses its lead",
      sample(torch.tensor([3.0, 3.2]), [1], SamplingParams(temperature=0, repetition_penalty=1.1)) == 0)
check("repetition penalty, negative logit: repeated token is pushed further down",
      sample(torch.tensor([-3.0, -3.2]), [0], SamplingParams(temperature=0, repetition_penalty=1.1)) == 1)

# 5. reproducibility with ONE generator per run (and the draws actually vary)
lp = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
def draws(seed):
    gen = torch.Generator().manual_seed(seed)
    return [sample(lp, [], SamplingParams(temperature=1.0, **OFF), gen) for _ in range(40)]
a, b = draws(7), draws(7)
check("seeded sampling is reproducible and not constant", a == b and len(set(a)) > 1)

# 6. streaming with a scripted fake model (no weights): multi-byte tail, EOS, length cut, padding ids
tok = Tokenizer("models/qwen3.5-0.8b/tokenizer.json")
EOS = {248046, 248044}
class FakeState:
    def __init__(self): self.length = 0
class Scripted:
    """Follows the model protocol (new_state / forward with a state) and forces a scripted token sequence."""
    def __init__(self, script, vocab=248320, pad_bias=False):
        self.script, self.vocab, self.pad_bias = script, vocab, pad_bias
    def new_state(self, max_len): return FakeState()
    def forward(self, ids, state=None, last_only=False):
        state.length += len(ids)
        step = state.length - self.start
        out = torch.full((1, self.vocab), -1e9)
        nxt = self.script[step] if step < len(self.script) else 248046
        out[0, nxt] = 10.0
        if self.pad_bias:
            out[0, 248300] = 50.0          # a padding row scoring far higher than any real token
        return out
def run(script, max_new, pad_bias=False):
    m = Scripted(script, pad_bias=pad_bias); prompt = tok.encode("hi"); m.start = len(prompt)
    rec = {}
    text = "".join(generate_stream(m, tok, prompt, SamplingParams(temperature=0, repetition_penalty=1.0), max_new, EOS, rec))
    return text, rec
# each check also proves its stream went through the hold-back path (a prefix that decodes to a partial character)
emoji_ids = tok.encode(" Hi 🫠")
held = all(tok.decode(emoji_ids[:k]).endswith("\ufffd") for k in (2, 3))
text, rec = run(emoji_ids, 50)
check(f"stream ending on an emoji is complete: {text!r}", held and text == tok.decode(emoji_ids) and rec["stop"] == "eos")
cut_ids = tok.encode(" é中文🫠")
held = tok.decode(cut_ids[:3]).endswith("\ufffd")                  # the length cut lands inside the emoji
text, rec = run(cut_ids, 3)
check(f"length-cut stream equals decode of generated ids: {text!r}", held and text == tok.decode(rec["ids"]) and rec["stop"] == "length")
text, rec = run(tok.encode(" ok"), 5, pad_bias=True)
check(f"padding ids (>= {tok.vocab_size()}) are never produced", all(i < tok.vocab_size() for i in rec["ids"]) and text == " ok")

# 7. batched sampling (the server's path) == sampling each row alone, token for token, same seeds
from sampler import sample_batch
def compare(device, trials=300):
    rng = torch.Generator().manual_seed(7)
    mismatches = 0
    for t in range(trials):
        B, V = 1 + t % 8, 5000
        logits = torch.randn(B, V, generator=rng) * 4
        if t % 7 == 0:
            logits[0, :50] = logits[0].max()                      # 50-way tie at the top: forces the tie fallback
        if t % 7 == 3:                                             # 30 tokens tied at ranks 16-45: a tie group
            order = torch.argsort(logits[0], descending=True)      # straddling the top-k cut (top_k 20 or 40)
            logits[0, order[15:45]] = logits[0, order[15]].item()
        params = [SamplingParams(temperature=[0.0, 0.7, 1.3][(t + i) % 3], top_k=[20, 1, 0, 40, 300][(t + i) % 5],
                                 top_p=[0.8, 1.0, 0.95][(t + i) % 3], repetition_penalty=[1.0, 1.3][(t + i) % 2])
                  for i in range(B)]
        prev = [torch.randint(0, V, (1 + (t + i) % 30,), generator=rng).tolist() for i in range(B)]
        if t % 3 == 1:                                             # the history holds each row's 25 best tokens, as in
            prev = [p + torch.topk(logits[i], 25).indices.tolist() for i, p in enumerate(prev)]   # real text: the
            # penalty reorders the top, so candidates must be picked by penalized logit, not raw
        want = [sample(logits[i], prev[i], params[i], torch.Generator().manual_seed(t * 10 + i)) for i in range(B)]
        got = sample_batch(logits.to(device), prev, params, [torch.Generator().manual_seed(t * 10 + i) for i in range(B)])
        mismatches += sum(a != b for a, b in zip(want, got))
    return mismatches
for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    m = compare(device)
    check(f"sample_batch == sample row by row on {device} (300 random batches: greedy, penalties on the top tokens, "
          f"top-k off or above the batch cap, all-tie and straddling ties): {m} mismatches", m == 0)

# a row that cannot be sampled fails alone: its seeded batch-mate draws exactly what it would alone
bad = torch.randn(2, 1000)
bad[1] = float("nan")
p = SamplingParams(temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.0)
alone = sample(bad[0], [], p, torch.Generator().manual_seed(5))
got = sample_batch(bad, [[], []], [p, p], [torch.Generator().manual_seed(5), torch.Generator().manual_seed(6)])
check(f"a NaN row yields its error while its seeded batch-mate keeps its own first draw ({got[0]} == {alone})",
      got[0] == alone and isinstance(got[1], Exception))

# one request asking for a huge top-k must not slow the whole batch down (it takes the per-row path)
import time
V = 248077                                                         # the served vocabulary (tokenizer ids)
logits = torch.randn(8, V)
prev = [list(range(0, 3000, 7))] * 8
base = [SamplingParams(temperature=0.7, top_k=20, repetition_penalty=1.1)] * 8
huge = [SamplingParams(temperature=0.7, top_k=247000, repetition_penalty=1.1)] + base[1:]
def timed(params):
    t0 = time.perf_counter()
    sample_batch(logits, prev, params, [torch.Generator().manual_seed(i) for i in range(8)])
    return time.perf_counter() - t0
timed(base)
t_base, t_huge = timed(base), timed(huge)
check(f"a batch-mate with top_k=247000 costs only its own row: {t_base * 1e3:.0f} ms -> {t_huge * 1e3:.0f} ms "
      f"(< base + 150 ms)", t_huge < t_base + 0.15)

# 8. a real streamed chat reply (last, so every model-free check runs first): stops on EOS, stream == decode(ids)
eng = load_engine("qwen3.5-0.8b", "cpu")                           # fp32 CPU: deterministic
ids = eng.tokenizer.encode(format_chat([{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
                                       style=eng.chat_style))
rec = {}
reply = "".join(generate_stream(eng.model, eng.tokenizer, ids, SamplingParams(seed=0, **eng.sampling), 40, eng.eos_ids, rec))
print(f"      reply: {reply!r}  ({len(rec['ids'])} tokens, stop={rec['stop']})")
check("real reply stops on EOS before the token limit", rec["stop"] == "eos" and len(rec["ids"]) < 40)
check("real reply stream == decode(generated ids)", reply == eng.tokenizer.decode(rec["ids"]))
check("real reply mentions Paris", "Paris" in reply)
del eng; gc.collect()

# 9. request sampling (M6): a field the client did not send now takes the model's registry value instead of a default
# hard-coded in the server. For Qwen3.5 the two are equal, so every request must sample exactly as before: the old
# constructor (kept here as the oracle) against the new one, for every subset of the four fields, each at 2 values
import itertools

from engine import MODELS
from server.app import Sampling

OLD_DEFAULTS = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)   # the server's fields until M6
VALUES = dict(temperature=(0.0, 1.3), top_p=(0.5, 1.0), top_k=(0, 40), repetition_penalty=(1.0, 1.2))
for name in ("qwen3.5-0.8b", "qwen3.5-2b"):
    model = SamplingParams(**MODELS[name]["sampling"])
    cases = same = 0
    for k in range(5):
        for fields in itertools.combinations(VALUES, k):
            for vals in itertools.product(*(VALUES[f] for f in fields)):
                sent = dict(zip(fields, vals))
                old = SamplingParams(**{**OLD_DEFAULTS, **sent}, seed=7)
                cases, same = cases + 1, same + (Sampling(**sent, seed=7).sampling(model) == old)
    check(f"{name}: every request samples as before M6 ({same}/{cases} subsets of fields sent)", same == cases == 81)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
