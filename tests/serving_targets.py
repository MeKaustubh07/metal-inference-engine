"""Serving targets for the server suites (tests/test_server.py, test_prefill.py, test_decision.py), chosen with
--target: the engine they serve and what they may assume about it.

  qwen  Qwen3.5-0.8B in fp32 on the CPU (the default): exact, a real trained model, a thinking mode.
  tiny  a random 8-layer Cohere2 (Tiny Aya's architecture) with an 8-token sliding window, a byte-level BPE trained
        here at test time (BOS from its post-processor, as Tiny Aya's), and a chat template of its own with a preamble
        of 76 tokens: exact (fp32 on the CPU) and fast, but random, so checks of meaning do not apply. Served
        with prefill chunks of 8, its rings hold 2 blocks, so every request past 32 tokens wraps its ring over HTTP.

A check that does not apply to a target prints "N/A" (run_tests.py counts SKIP lines as missing files)."""
import argparse
import gc
import os
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass, field

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from chat import TemplateChat
from engine import Engine, load_engine
from tokenizer import Tokenizer


@dataclass
class Target:
    name: str
    engine: Engine
    exact: bool                       # fp32 on the CPU: batched / packed / chunked runs equal sequential ones
    real: bool                        # a trained model: checks of meaning (Paris, 7 + 3, review scores) apply
    thinking: bool                    # the model has a <think> mode
    logit_tol: float                  # batched vs sequential logits
    logprob_tol: float                # decision scores vs references
    prefill_chunk: int = 512          # what server apps built by the suites use
    step_delay: float = 0.0           # seconds added to every decode step, for tests that need a stream in flight
    keep: list = field(default_factory=list)   # objects that must outlive the target (its temp directory)

    def raw_ids(self, text: str) -> list[int]:
        """A raw prompt's ids as /v1/completions builds them: BOS first, if the model has one."""
        return self.engine.tokenizer.encode(text, add_bos=True)

    def chat_len(self) -> int:
        """Tokens of a one-line chat: suites size their max_model_len from it."""
        from chat import format_chat
        return len(self.engine.tokenizer.encode(format_chat([{"role": "user", "content": "Hi"}],
                                                            style=self.engine.chat_style)))

    def slow_steps(self, sched) -> None:
        """Make each decode step take at least step_delay, so a stream is still running when a test acts on it."""
        if self.step_delay:
            real = sched._decode_step
            def step(*a, **k):
                time.sleep(self.step_delay)
                return real(*a, **k)
            sched._decode_step = step


def na(name: str) -> None:
    print(f"N/A   {name}")


# ---------------------------------------------------------------- tiny: tokenizer, template, model
CORPUS = [
    "The capital of France is Paris, and the capital of Italy is Rome.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "The history of computing is a story of abstraction. Each generation of engineers built tools that hid the "
    "details of the layer below, so the next generation could think in bigger pieces.",
    "Hello world. Inference engines turn trained weights into answers, one token at a time.",
    "What is 2+2? Answer with one number. What is 7 + 3? Answer with the number only.",
    "A server batches requests, streams tokens, and frees memory when a client goes away.",
    "Is Paris the capital of France? Yes. Is Lyon the capital of France? No.",
    "Numbers: 12345 and 3.14159, year 2026. Words, punctuation; quotes 'and' \"marks\".",
]
SPECIALS = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
TEMPLATE = (
    "{{ bos_token }}<|system|>"
    "{% if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}"
    "{% else %}You are a very small model that helps test an inference server. Keep each answer short, stay on the "
    "question, and stop when you are done.{% endif %}<|end|>"
    "{% if not messages | selectattr('role', 'equalto', 'user') | list %}"
    "{{ raise_exception('a conversation needs a user message') }}{% endif %}"
    "{% for m in messages %}{% if m['role'] == 'system' %}{% if not loop.first %}"
    "{{ raise_exception('the system message must come first') }}{% endif %}"
    "{% else %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TINY_SEED = 0
PROMPTS = ["The capital of France is", "def fibonacci(n):",
           "The history of computing is a story of abstraction. Each generation of engineers built tools that",
           "Hello world"]


def train_tokenizer(path: str) -> None:
    """A byte-level BPE with Tiny Aya's shape (a regex Split, then a plain ByteLevel; BOS from a TemplateProcessing
    post-processor), at most 512 ids, the specials first so pad, bos and eos are the model config's 0, 1 and 2."""
    from tokenizers import Regex, Tokenizer as HFTokenizer, decoders, models, pre_tokenizers, processors, trainers
    tk = HFTokenizer(models.BPE())
    gpt2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(gpt2), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=SPECIALS, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS * 4, trainer)
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    tk.save(path)


def tiny_engine(tmp: str, seed: int = TINY_SEED) -> Engine:
    import json

    from tiny_cohere2 import build
    train_tokenizer(f"{tmp}/tokenizer.json")
    json.dump({"chat_template": TEMPLATE, "bos_token": "<bos>", "eos_token": "<eos>"},
              open(f"{tmp}/tokenizer_config.json", "w"))
    tok = Tokenizer(f"{tmp}/tokenizer.json")
    _, make = build(tmp, window=8, seed=seed, vocab_size=512)
    model = make()
    eos = {tok.special["<eos>"], tok.special["<|end|>"]}
    return Engine("tiny-cohere2", model, tok, eos, TemplateChat(f"{tmp}/tokenizer_config.json"),
                  dict(temperature=0.1, top_p=0.95, top_k=50, repetition_penalty=1.0), model.max_positions)


def self_check(eng: Engine) -> list[str]:
    """What the suites need from a random model: greedy continuations that differ between prompts, run 400 tokens
    without an end token, and stream in many pieces. -> the problems found (empty: usable)."""
    from server.scheduler import Request  # noqa: F401  (imports the server package the suites use)
    m, tok, V = eng.model, eng.tokenizer, eng.tokenizer.vocab_size()
    problems, firsts = [], []
    for p in PROMPTS:
        st = m.new_state(512)
        lg = m.forward(torch.tensor(tok.encode(p, add_bos=True)), state=st, last_only=True)[0]
        out = []
        for _ in range(400):
            t = int(lg[:V].argmax())
            if t in eng.eos_ids:
                problems.append(f"{p!r} reaches an end token after {len(out)} tokens")
                break
            out.append(t)
            lg = m.forward(torch.tensor([t]), state=st)[0]
        firsts.append(tuple(out[:20]))
        pieces, shown = 0, ""
        for k in range(1, 25):
            text = tok.decode(out[:k])
            if not text.endswith("�") and len(text) > len(shown):
                pieces, shown = pieces + 1, text
        if pieces < 10:
            problems.append(f"{p!r}: 24 tokens stream in only {pieces} pieces")
    if len(set(firsts)) < len(firsts):
        problems.append("two prompts share their first 20 greedy tokens")
    return problems


_cache: dict[str, Target] = {}
_lock = threading.Lock()


def load_target(name: str) -> Target:
    with _lock:
        if name in _cache:
            return _cache[name]
        if name == "qwen":
            t = Target("qwen3.5-0.8b", load_engine("qwen3.5-0.8b", "cpu"), exact=True, real=True, thinking=True,
                       logit_tol=2e-3, logprob_tol=2e-4)
        elif name == "tiny":
            tmp = tempfile.TemporaryDirectory()
            torch.set_grad_enabled(False)
            eng = tiny_engine(tmp.name)
            if problems := self_check(eng):
                raise RuntimeError(f"the tiny target is unusable with seed {TINY_SEED}: {problems}")
            t = Target("tiny-cohere2", eng, exact=True, real=False, thinking=False, logit_tol=2e-3,
                       logprob_tol=2e-4, prefill_chunk=8, step_delay=0.01, keep=[tmp])
        else:
            raise ValueError(f"unknown target {name!r} (qwen, tiny)")
        _cache[name] = t
        return t


def unload_target(name: str) -> bool:
    """Forget a loaded target and free its model, before a suite loads Qwen on the GPU (the fp32 CPU Qwen is 3.2 GB of
    an 8 GB machine). The caller drops its own references first (the engine, its schedulers). FastAPI keeps every
    endpoint function create_app made in module-level lru_caches (fastapi.dependencies.models), and those functions
    hold the engine, so the caches are cleared here. -> whether the model is gone."""
    with _lock:
        t = _cache.pop(name, None)
    if t is None:
        return True
    model = weakref.ref(t.engine.model)
    del t
    try:
        import fastapi.dependencies.models as fm
        for f in vars(fm).values():
            if callable(getattr(f, "cache_clear", None)):
                f.cache_clear()
    except ImportError:
        pass
    gc.collect()
    return model() is None


def target_from_argv(default: str = "qwen") -> str:
    """--target qwen|tiny (or --target=tiny); anything else on the command line is an error, so a mistyped flag cannot
    fall back to the qwen target and its Metal sections."""
    p = argparse.ArgumentParser(description="a server suite (tests/serving_targets.py)")
    p.add_argument("--target", choices=("qwen", "tiny"), default=default)
    return p.parse_args().target


if __name__ == "__main__":                               # check the tiny target on its own: build, self-check, sizes
    t0 = time.perf_counter()
    t = load_target(os.environ.get("TARGET", "tiny"))
    tok = t.engine.tokenizer
    print(f"{t.name}: vocab {tok.vocab_size()}, BOS {tok.bos_id}, eos {sorted(t.engine.eos_ids)}, one-line chat "
          f"{t.chat_len()} tokens, built and self-checked in {time.perf_counter() - t0:.1f} s")
