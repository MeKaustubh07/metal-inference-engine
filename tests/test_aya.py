"""Tiny Aya Global (Cohere2), everything before the model itself: Cohere2Config against transformers' config class
(and the configs it must refuse), the sharded weights against the safetensors library (and a corrupt index), the
model's own chat template against apply_chat_template, the registry entry and its guards, the BOS policy where prompts
are tokenized, the server with a stand-in scheduler, and a guard that none of the gated files' content is in the repo.
Needs the gated files in models/tiny-aya-global (see docs/tiny-aya-plan.md); otherwise it prints SKIP."""
import json
import os
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # everything is local: transformers must not query the Hub
import torch
from safetensors import safe_open
from transformers import AutoConfig

sys.path.insert(0, "src")
from config import Cohere2Config
from weight_loader import ShardedSafetensors, open_weights

D = "models/tiny-aya-global"
INDEX = f"{D}/model.safetensors.index.json"
if not os.path.exists(INDEX):
    print(f"SKIP: needs {INDEX} (gated download)")
    sys.exit(0)

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

# 1. the config, field by field against transformers' Cohere2Config (the reference the port is checked against)
cfg, hf = Cohere2Config.from_json(f"{D}/config.json"), AutoConfig.from_pretrained(D)
hf_theta = (getattr(hf, "rope_parameters", None) or {}).get("rope_theta", getattr(hf, "rope_theta", None))
pairs = {"layers": (cfg.num_hidden_layers, hf.num_hidden_layers), "hidden": (cfg.hidden_size, hf.hidden_size),
         "ffn": (cfg.intermediate_size, hf.intermediate_size), "vocab": (cfg.vocab_size, hf.vocab_size),
         "heads": (cfg.num_attention_heads, hf.num_attention_heads),
         "kv heads": (cfg.num_key_value_heads, hf.num_key_value_heads),
         "head_dim": (cfg.head_dim, hf.head_dim), "eps": (cfg.layer_norm_eps, hf.layer_norm_eps),
         "window": (cfg.sliding_window, hf.sliding_window), "layer_types": (cfg.layer_types, list(hf.layer_types)),
         "logit_scale": (cfg.logit_scale, hf.logit_scale), "tied": (cfg.tie_word_embeddings, hf.tie_word_embeddings),
         "rope_theta": (cfg.rope_theta, float(hf_theta))}
check(f"Cohere2Config == transformers on all {len(pairs)} fields", all(a == b for a, b in pairs.values()))
for k, (a, b) in pairs.items():
    if a != b:
        print(f"      {k}: ours {a}, transformers {b}")
full = [i for i in range(cfg.num_hidden_layers) if not cfg.is_sliding(i)]
check("36 layers, 16/4 heads x 128, window 4096, full attention at 3, 7, ..., 35, logit_scale 1.0, tied",
      (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.sliding_window)
      == (36, 16, 4, 128, 4096) and full == list(range(3, 36, 4))
      and cfg.logit_scale == 1.0 and cfg.tie_word_embeddings)

# 2. the traps: settings whose defaults differ, and settings this engine does not implement
raw = json.load(open(f"{D}/config.json"))
with tempfile.TemporaryDirectory() as tmp:
    def variant(**edits):
        c = {k: v for k, v in raw.items() if k not in edits or edits[k] is not None}
        c.update({k: v for k, v in edits.items() if v is not None})
        json.dump(c, open(f"{tmp}/config.json", "w"))
        return f"{tmp}/config.json"
    no_scale = Cohere2Config.from_json(variant(logit_scale=None)).logit_scale
    check(f"no logit_scale in the file -> {no_scale}, as transformers ({AutoConfig.from_pretrained(tmp).logit_scale})",
          no_scale == AutoConfig.from_pretrained(tmp).logit_scale == 0.0625)
    nested = variant(rope_theta=None, rope_parameters={"rope_type": "default", "rope_theta": 50000})
    check("rope_theta nested under rope_parameters (transformers 5 files) is read too",
          Cohere2Config.from_json(nested).rope_theta == 50000.0)
    for edit in ({"use_qk_norm": True}, {"position_embedding_type": "rope"}, {"use_parallel_block": False},
                 {"rope_scaling": {"type": "linear", "factor": 2.0}}, {"head_dim": 256}, {"model_type": "cohere"},
                 {"rope_parameters": {"type": "linear", "factor": 2.0, "rope_theta": 50000}},   # the legacy key
                 {"rope_parameters": {"rope_type": "dynamic", "factor": 2.0}}, {"sliding_window": 0},
                 {"sliding_window": None}, {"use_embedding_sharing": False}, {"tie_word_embeddings": False}):
        try:
            Cohere2Config.from_json(variant(**edit)); check(f"refuses {edit}", False)
        except ValueError:
            check(f"refuses {edit}", True)

# 3. the sharded weights, through the index, against the safetensors library
w = open_weights(INDEX)
names = set(w.tensor_names())
per_layer = ["input_layernorm.weight"] + [f"self_attn.{p}_proj.weight" for p in "qkvo"] + \
            [f"mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
expected = {"model.embed_tokens.weight", "model.norm.weight"} | \
           {f"model.layers.{i}.{n}" for i in range(cfg.num_hidden_layers) for n in per_layer}
check(f"index -> ShardedSafetensors with exactly the {len(expected)} expected tensors (no lm_head: tied)",
      isinstance(w, ShardedSafetensors) and names == expected)
check("integrity: each file accounts for its bytes, and the tensor bytes add up to the index's total_size",
      w.integrity_check())
infos = {n: w.info(n) for n in names}
params = sum(torch.Size(i["shape"]).numel() for i in infos.values())
check(f"{params:,} parameters, all BF16 (the published count is 3,349,227,520)",
      params == 3_349_227_520 and {i["dtype"] for i in infos.values()} == {"BF16"})
weight_map = json.load(open(INDEX))["weight_map"]
by_shard = {}
for n, f in sorted(weight_map.items()):
    by_shard.setdefault(f, n)                                      # the first tensor of each file
same = True
for n in sorted({"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"} | set(by_shard.values())):
    with safe_open(f"{D}/{weight_map[n]}", framework="pt") as f:
        same &= torch.equal(w.get(n), f.get_tensor(n))
check(f"get() == safetensors' own reader, bit for bit, on tensors from all {len(by_shard)} files", same)
try:
    w.get("lm_head.weight"); check("an unknown name raises KeyError", False)
except KeyError:
    check("an unknown name raises KeyError", True)

# 4. a corrupt index (a tensor listed under the wrong file) is refused
with tempfile.TemporaryDirectory() as tmp:
    idx = json.load(open(INDEX))
    files = sorted(set(idx["weight_map"].values()))
    for f in files:
        os.symlink(os.path.abspath(f"{D}/{f}"), f"{tmp}/{f}")
    n = next(n for n, f in idx["weight_map"].items() if f == files[0])
    idx["weight_map"][n] = files[1]
    json.dump(idx, open(f"{tmp}/model.safetensors.index.json", "w"))
    try:
        ShardedSafetensors(f"{tmp}/model.safetensors.index.json")
        check("an index that disagrees with its files is refused", False)
    except ValueError:
        check("an index that disagrees with its files is refused", True)

# 5. the chat template: the model's own Jinja template, rendered by TemplateChat, against apply_chat_template
from transformers import AutoTokenizer
from chat import TemplateChat, format_chat
from tokenizer import Tokenizer

hf_tok, tok = AutoTokenizer.from_pretrained(D), Tokenizer(f"{D}/tokenizer.json")
aya = TemplateChat(f"{D}/tokenizer_config.json")
U, A, S = "user", "assistant", "system"
convs = [[(U, "What is the capital of France?")],
         [(S, "You are terse."), (U, "Hi")],
         [(S, ""), (U, "Hi")],                                                  # an empty system message adds nothing
         [(U, "2+2?"), (A, "4"), (U, "and 3+3?")],
         [(S, "  Be brief. \n"), (U, "  hi \n")],                               # no stripping: whitespace is kept
         [(U, "first"), (S, "late system"), (A, "ok"), (U, "next")],            # the first system message anywhere
         [(S, "Same."), (U, "a"), (A, "b"), (S, "Same."), (U, "c")],            # a repeat of it is skipped
         [(S, "One."), (U, "a"), (A, "b"), (S, "Two."), (U, "c")],              # a different one is its own turn
         [(U, "a"), ("chatbot", "b"), (U, "c")],                                # "chatbot" means assistant
         [(U, "नमस्ते, आप कैसे हैं? 12345"), (A, "मैं ठीक हूँ।\n\n- एक\n- दो"), (U, "مرحبا 你好 🚀")]]
convs = [[{"role": r, "content": c} for r, c in conv] for conv in convs]
same_text = same_ids = True
for conv in convs:
    for gen in (True, False):
        ours = format_chat(conv, add_generation_prompt=gen, style=aya)
        ref = hf_tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=gen)
        ref_ids = hf_tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=gen)
        ref_ids = list(ref_ids["input_ids"] if hasattr(ref_ids, "keys") else ref_ids)
        same_text &= ours == ref
        same_ids &= tok.encode(ours) == ref_ids                     # the template writes BOS itself: add_bos=False
        if ours != ref:
            print("      differs:", [m["role"] for m in conv], "generation prompt:", gen)
check(f"chat text == apply_chat_template on {len(convs)} conversations x 2 (with and without generation prompt)",
      same_text)
check("their token ids are equal too, with a single BOS at the start", same_ids)
one = tok.encode(format_chat(convs[0], style=aya))
print(f"      a one-line question is {len(one)} prompt tokens, starting {one[:3]} (BOS {one.count(tok.bos_id)}x)")
for bad, why in (([(U, "a"), (U, "b")], "two user turns in a row"), ([(A, "a"), (U, "b")], "an assistant turn first")):
    conv = [{"role": r, "content": c} for r, c in bad]
    errors = []
    for fn in (lambda: format_chat(conv, style=aya), lambda: hf_tok.apply_chat_template(conv, tokenize=False)):
        try:
            fn(); errors.append(None)
        except Exception as e:                                       # ours: ValueError; transformers: TemplateError
            errors.append(type(e).__name__)
    check(f"{why}: refused by both (ours {errors[0]}, transformers {errors[1]})",
          errors[0] == "ValueError" and errors[1])
qwen_dir = "models/qwen3.5-0.8b"
if os.path.exists(f"{qwen_dir}/tokenizer_config.json"):            # TemplateChat is general: Qwen's template as well
    from transformers import AutoTokenizer as _AT
    qt, qhf = TemplateChat(f"{qwen_dir}/tokenizer_config.json"), _AT.from_pretrained(qwen_dir)
    conv = convs[3]
    check("TemplateChat on Qwen3.5's own template == apply_chat_template == the hand-written qwen3.5 style",
          all(qt.render(conv, True, th) == qhf.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                                                     enable_thinking=th)
              == format_chat(conv, style="qwen3.5", enable_thinking=th) for th in (False, True)))

# 6. the engine registry entry and the BOS policy at every place a prompt is tokenized
from decision import option_ids, prompt_ids
from engine import MODELS, ROOT, chat_style, load_engine

spec = MODELS["tiny-aya-global"]
stops = {i: tok.id_to_token[i] for i in spec["eos"]}
check(f"vocab: {tok.vocab_size():,} contiguous ids (<= {cfg.vocab_size:,} output rows); every stop id survives the "
      "cut of the logits to the vocab", tok.vocab_size() == max(tok.id_to_token) + 1 == 261_008
      and tok.vocab_size() <= cfg.vocab_size and max(spec["eos"]) < tok.vocab_size())
check(f"registry: index weights, chat from the model's template, no thinking mode, stops {stops}",
      os.path.exists(ROOT / spec["dir"] / spec["weights"])
      and isinstance(chat_style(spec, ROOT / spec["dir"]), TemplateChat) and not spec.get("thinking")
      and list(stops.values()) == ["<EOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"])
check(f"registry: the card's sampling {spec['sampling']}; max_model_len {spec['max_model_len']} <= the sliding window",
      spec["sampling"] == dict(temperature=0.1, top_p=0.95, top_k=50, repetition_penalty=1.0)
      and spec["max_model_len"] <= cfg.sliding_window)
try:
    load_engine("tiny-aya-global"); check("load_engine refuses until the cohere2 model class exists", False)
except NotImplementedError:
    check("load_engine refuses until the cohere2 model class exists", True)
plain = prompt_ids(tok, "", "Is the sky blue?", False, aya)
chatted = prompt_ids(tok, "", "Is the sky blue?", True, aya)
opts = [option_ids(tok, o, c) for o in ("Yes", " No") for c in (True, False)]
check("decide: a raw prompt gets one BOS, a chat prompt one (from its template), an option none",
      plain[0] == 2 and plain.count(2) == 1 and chatted[0] == 2 and chatted.count(2) == 1
      and all(2 not in o for o in opts))
check("a raw prompt's ids == HF's default encode (BOS included)",
      tok.encode("The capital of France is", add_bos=True) == hf_tok("The capital of France is")["input_ids"])

# 7. the server, with a stand-in scheduler (no weights needed): the BOS policy per endpoint, the model's length cap,
# conversations the template refuses, and enable_thinking on a model without a thinking mode
import asyncio

import httpx

import server.app as app_module
from engine import Engine
from server.scheduler import QueueFull


class RecordingScheduler:
    """Takes the scheduler's place: records the ids each request would run, then answers "busy" (HTTP 429)."""
    def __init__(self, engine, metrics, **kw):
        self.max_model_len, self.prefill_chunk, self.lock_error, self.submitted = kw["max_model_len"], 512, None, []

    def submit(self, ids, params, max_new, out=None):
        self.submitted.append(list(ids))
        raise QueueFull()

    def shutdown(self, timeout):
        pass


app_module.Scheduler = RecordingScheduler
eng = Engine("tiny-aya-global", None, tok, set(spec["eos"]), aya, spec["sampling"], spec["max_model_len"])
app = app_module.create_app(eng, max_model_len=8192)
sched = app.state.scheduler

async def post(path, body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(path, json=body)
        return r.status_code, r.text

def chat(*msgs, **extra):
    return asyncio.run(post("/v1/chat/completions", {"messages": [{"role": r, "content": c} for r, c in msgs],
                                                      "max_tokens": 4, **extra}))

check("create_app(max_model_len=8192) gives the scheduler the model's own cap, 4096", sched.max_model_len == 4096)
code, _ = asyncio.run(post("/v1/completions", {"prompt": "Hi", "max_tokens": 4}))
check(f"/v1/completions: raw text gets one BOS ({sched.submitted[-1]})",
      code == 429 and sched.submitted[-1] == tok.encode("Hi", add_bos=True) and sched.submitted[-1].count(2) == 1)
code, _ = chat((U, "Hi"))
check(f"/v1/chat/completions: one BOS, from the template ({len(sched.submitted[-1])} prompt tokens)",
      code == 429 and sched.submitted[-1] == tok.encode(format_chat([{"role": U, "content": "Hi"}], style=aya))
      and sched.submitted[-1].count(2) == 1)
for msgs, why in ((((U, "a"), (U, "b")), "two user turns"), (((S, "s"), (A, "x"), (U, "b")), "an assistant turn first")):
    code, text = chat(*msgs)
    check(f"/v1/chat/completions: {why} -> 400 from the template, not 500", code == 400 and "alternate" in text)
code, text = chat((U, "Hi"), enable_thinking=True)
check("/v1/chat/completions: enable_thinking on a model without a thinking mode -> 400", code == 400 and "thinking" in text)
code, text = asyncio.run(post("/v1/decide", {"type": "boolean", "question": "Is it?", "context": "word " * 5000}))
check("/v1/decide: a context past the model's 4096-token cap -> 400", code == 400 and "exceeds 4096" in text)

# 8. licence guard: no copy of a gated file, and no copy of the template's text (however it is wrapped), in anything
# a commit can contain: the working tree and the staged index
import hashlib
import re
import subprocess

def flat(t):
    return " ".join(t.replace("\\n", " ").replace("\\r", " ").replace('\\"', '"').split())

source = json.load(open(f"{D}/tokenizer_config.json"))["chat_template"][0]["template"]
texts = [hf_tok.apply_chat_template(c, tokenize=False).split("<|SYSTEM_TOKEN|>", 1)[1].split("<|END_OF_TURN_TOKEN|>")[0]
         for c in (convs[0], convs[1])]                                  # the rendered preamble, without and with
texts += re.split(r"\{%.*?%\}|\{\{.*?\}\}|\{#.*?#\}", source, flags=re.S)    # a system turn; the literal text
shingles = {f[i:i + 64] for f in map(flat, texts) for i in range(0, len(f) - 63, 16)}
gated = {hashlib.sha256(open(f"{D}/{n}", "rb").read()).hexdigest() for n in os.listdir(D)   # the small files: the
         if os.path.isfile(f"{D}/{n}") and os.path.getsize(f"{D}/{n}") < 64 << 20}           # shards are 6.7 GB
paths = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True,
                       text=True, check=True).stdout.splitlines()
index = [line.split("\t", 1) for line in subprocess.run(["git", "ls-files", "-s"], capture_output=True, text=True,
                                                          check=True).stdout.splitlines()]
blobs = [meta.split()[1] for meta, _ in index]
staged = subprocess.run(["git", "cat-file", "--batch"], input=("\n".join(blobs) + "\n").encode(), capture_output=True,
                        check=True).stdout
names = [f for f in paths if os.path.isfile(f)] + [f"{path} (staged)" for _, path in index]
contents, i = [open(f, "rb").read() for f in paths if os.path.isfile(f)], 0
while i < len(staged):                                                   # "<sha> blob <size>\n<bytes>\n" per object
    header_end = staged.index(b"\n", i)
    size = int(staged[i:header_end].split()[2])
    contents.append(staged[header_end + 1:header_end + 1 + size])
    i = header_end + 2 + size
def leaks(b):
    text = flat(b.decode("utf-8", errors="ignore"))                     # flatten each file once
    return hashlib.sha256(b).hexdigest() in gated or any(sh in text for sh in shingles)
hits = [names[n] for n, b in enumerate(contents) if leaks(b)]
if hits:
    print("      gated content in:", hits)
check(f"no gated file and none of {len(shingles)} 64-char pieces of the template's text in the {len(paths)} repo "
      f"files or {len(blobs)} staged blobs", not hits and len(shingles) > 20)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
