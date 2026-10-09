"""Serving targets for the server suites (tests/test_server.py, test_prefill.py, test_decision.py), chosen with
--target: the engine they serve and what they may assume about it.

  qwen  Qwen3.5-0.8B in fp32 on the CPU (the default): exact, a real trained model, a thinking mode.
  tiny  a random 8-layer Cohere2 (Tiny Aya's architecture) with an 8-token sliding window, a byte-level BPE trained
        here at test time (BOS from its post-processor, as Tiny Aya's), and a chat template of its own with a preamble
        of 76 tokens: exact (fp32 on the CPU) and fast, but random, so checks of meaning do not apply. Served
        with prefill chunks of 8, its rings hold 2 blocks, so every request past 32 tokens wraps its ring over HTTP.
  aya-int4, aya-int8  Tiny Aya Global on Metal, INT4 or INT8 weights (models/tiny-aya-global/model.<scheme>.qt) and
        bf16 KV: real but not exact. A batch of 2 or more rows takes the batched INT4/INT8 matvec, which dequantizes
        in another order than the one-row kernel (about 1e-7 relative on one linear, about 1e-2 on the logits after
        36 layers; one sequence alone, paged or contiguous, is bit-identical), so served greedy runs are compared
        token by token under the near-tie rule (Target.agree), and checks of meaning are reported, not gated, except
        Paris and decision 0 (D10). Each suite also gates peak GPU memory and Metal aborts (gpu_gates). SKIP without
        a Metal GPU or the model's files. One target per process: the suites' Qwen Metal sections are N/A beside it.

A check that does not apply to a target prints "N/A" (run_tests.py counts SKIP lines as missing files)."""
import argparse
import gc
import os
import subprocess
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
    started: str = ""                 # local time just before the model loaded: Metal aborts are counted from it
    gate_meaning: bool = True         # checks of meaning gate; False: reported only (a quantized model, D10)

    @property
    def gpu(self) -> bool:
        return self.engine.model.b.device.type == "mps"

    @property
    def tie(self) -> float:
        """A served token other than the reference's pick is a near-tie when the reference ranks it less than this
        below its pick: two runs whose logits differ by at most logit_tol can order such a pair either way (D9)."""
        return 2 * self.logit_tol

    def agree(self, ref, got: list[int], finish: str | None = None, what: str = "") -> bool:
        """Served greedy ids `got` == a sequential reference ref = (tokens, logits rows), which has one more row than
        tokens when it stopped on an end token, for the same max_tokens. Exact targets: equal ids. The others (D9):
        equal ids (a cancelled request may stop short of the reference), or a near-tie at the first difference, where
        the reference ranks the served token (an end token if the request stopped there: finish 'stop') less than
        `tie` below its own pick; later tokens follow different contexts and are not compared. Each near-tie,
        divergence or wrong length is noted on the next check's label (notes())."""
        toks, logs = ref
        if self.exact:
            return list(got) == list(toks)
        end = object()
        r_seq = list(toks) + ([end] if len(logs) > len(toks) else [])
        s_seq = list(got) + ([end] if finish == "stop" else [])
        k = next((i for i, (a, b) in enumerate(zip(r_seq, s_seq)) if a != b), None)
        if k is None:                              # one is a prefix of the other: only a cancel may stop short
            if len(s_seq) == len(r_seq) or finish == "cancelled" and len(s_seq) < len(r_seq):
                return True
            _notes.append(f"; {what + ': ' if what else ''}{len(got)} tokens ({finish}), the reference {len(toks)}")
            return False
        row, eos = logs[k], sorted(self.engine.eos_ids)
        def logit(t):
            return row[eos].max().item() if t is end else row[t].item()
        gap = logit(r_seq[k]) - logit(s_seq[k])
        _notes.append(f"; {what + ': ' if what else ''}{'near-tie' if gap < self.tie else 'diverged'} at token {k} "
                      f"({gap:.1e} below the reference's pick, {'<' if gap < self.tie else '>='} {self.tie:.0e})")
        return gap < self.tie

    def tidy(self) -> None:
        """Between sub-suites on a GPU target: free the apps, schedulers and pools they built (memory is gated)."""
        if self.gpu:
            release_apps()
            gc.collect()
            torch.mps.empty_cache()

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


_notes: list[str] = []


def notes() -> str:
    """The notes of the comparisons since the last call (near-ties, divergences), for a check's label. Empty on exact
    targets, so their output is unchanged."""
    s = "".join(_notes)
    _notes.clear()
    return s


def report(name: str, ok: bool) -> None:
    """A check of meaning on a quantized model (D10): printed, not gated (run_tests.py counts PASS, FAIL, SKIP and the
    not-applicable mark anywhere in a line, so a report says neither)."""
    print(f"INFO  {name}: {'holds' if ok else 'does not hold'} (reported, not gated)")


def agree_selfcheck() -> bool:
    """The near-tie rule on made-up logits over 5 ids, end token 3: what it accepts and what it refuses."""
    from types import SimpleNamespace
    t = Target("made-up", SimpleNamespace(eos_ids={3}), exact=False, real=False, thinking=False, logit_tol=1e-2,
               logprob_tol=0.0)
    def rows(*steps):                              # per step: (the pick, a runner-up, how far below the pick it is)
        out = []
        for top, second, gap in steps:
            r = torch.zeros(5)
            r[top], r[second] = 1.0, 1.0 - gap
            out.append(r)
        return out
    ref = ([1, 2], rows((1, 4, 0.5), (2, 4, 0.01)))
    stops = ([1], rows((1, 4, 0.5), (3, 2, 0.01)))              # the reference ends at step 1 (id 2 close behind)
    ends_close = ([1, 2], rows((1, 3, 0.01), (2, 4, 0.5)))      # the end token 0.01 below at step 0
    ok = [t.agree(ref, [1, 2]), t.agree(ref, [1], "cancelled"), t.agree(ref, [1, 4]), not t.agree(ref, [4, 2]),
          not t.agree(ref, [1, 0]), t.agree(stops, [1], "stop"), t.agree(stops, [1, 2], "length"),
          not t.agree(stops, [1, 4], "length"), t.agree(ends_close, [], "stop"), not t.agree(ref, [], "stop"),
          not t.agree(ref, [1], "length"), not t.agree(ref, [1, 2, 0], "length"),
          not t.agree(ref, [1, 2, 0], "cancelled"),
          not Target("exact", t.engine, True, False, False, 1e-2, 0.0).agree(ref, [1, 4])]
    _notes.clear()
    return all(ok)


class Served:
    """Records the Requests a scheduler is given, so a check can read the ids it generated (HTTP returns text)."""

    def __init__(self, sched):
        self.reqs, submit = [], sched.submit
        def recording(ids, *a, **k):
            req = submit(ids, *a, **k)
            self.reqs.append(req)
            return req
        sched.submit = recording

    def last(self, ids: list[int]):
        """The latest request with these prompt ids."""
        return next(r for r in reversed(self.reqs) if r.prompt_ids == list(ids))


def release_apps() -> None:
    """Drop FastAPI's references to every endpoint create_app made: module-level lru_caches in
    fastapi.dependencies.models keep the endpoint functions, and those hold their scheduler, its pool and the engine."""
    try:
        import fastapi.dependencies.models as fm
    except ImportError:
        return
    for f in vars(fm).values():
        if callable(getattr(f, "cache_clear", None)):
            f.cache_clear()


def gpu_peak() -> int:
    """This process's GPU memory high point: the allocator's peak heap bytes, plus what Metal holds outside them now
    (as tests/test_aya_long_quant.py measures it)."""
    torch.mps.synchronize()
    A = torch.accelerator
    return A.max_memory_reserved() + max(0, torch.mps.driver_allocated_memory() - A.memory_reserved())


ABORT = "Execution of the command buffer was aborted"


def metal_aborts(since: str) -> tuple[bool, str]:
    """Whether Metal aborted none of this process's command buffers since `since` (local time), read from macOS's
    log: PyTorch 2.14 never reads a command buffer's status, so an out-of-memory abort is otherwise silent.
    -> (none aborted, what the log said)"""
    try:
        log = subprocess.run(["/usr/bin/log", "show", "--start", since, "--style", "compact", "--predicate",
                              f'processID == {os.getpid()} AND eventMessage CONTAINS "{ABORT}"'],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "macOS's log not readable: log show timed out after 300 s"
    if log.returncode != 0:                                  # it needs an admin account and no sandbox
        return False, f"macOS's log not readable (log show: {log.returncode}, {log.stderr.strip()[:150]})"
    aborts = [line.split(ABORT)[-1].strip() for line in log.stdout.splitlines() if ABORT in line]
    return not aborts, f"macOS log: {len(aborts)}" + (f", e.g. {aborts[0][:90]}" if aborts else "")


def gpu_gates(t: Target, check) -> None:
    """On a GPU target, two gates over the whole suite: peak GPU memory at most 0.95 x what Metal recommends (where
    MLX starts freeing its cache), and no command buffer aborted by Metal. Nothing on a CPU target. Call it while the
    target is loaded."""
    if not t.gpu:
        return
    peak, rec = gpu_peak(), torch.mps.recommended_max_memory()
    check(f"GPU memory: peak {peak / 2**30:.2f} GiB <= 0.95 x the {rec / 2**30:.2f} GiB Metal recommends",
          peak <= 0.95 * rec)
    ok, what = metal_aborts(t.started)
    check(f"Metal aborted no command buffer of this process ({what})", ok)


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
        elif name in ("aya-int4", "aya-int8"):
            scheme, d = name[4:], "models/tiny-aya-global"
            qt = f"{d}/model.{scheme}.qt"
            need = [qt] + [f"{d}/{f}" for f in ("config.json", "tokenizer.json", "tokenizer_config.json")]
            if not torch.backends.mps.is_available() or not all(map(os.path.exists, need)):
                policy = " --policy configs/quant/tiny-aya-global.json" if scheme == "int4" else ""
                print(f"SKIP: the {name} target needs a Metal GPU, the gated files in {d} and {qt} (scripts/"
                      f"quantize.py {d}/model.safetensors.index.json {qt} --scheme {scheme}{policy})")
                sys.exit(0)
            from quant import QtFile
            if (held := QtFile(qt).scheme) != scheme:
                raise RuntimeError(f"{qt} holds {held} weights, not {scheme}")
            torch.set_grad_enabled(False)
            started = time.strftime("%Y-%m-%d %H:%M:%S")     # before the backend's first GPU work (shader compiles)
            # tolerances: about twice the largest difference measured (2026-10-08): logits 1.4e-2 (batched decode, INT4;
            # INT8 1.1e-2; a pack through the GEMM 9.3e-3), decision scores 6.8e-3 (a 428-token context in 4-token chunks)
            t = Target(f"tiny-aya-global {scheme}", load_engine("tiny-aya-global", f"metal-{scheme}", qt),
                       exact=False, real=True, thinking=False, logit_tol=3e-2, logprob_tol=2e-2, started=started,
                       gate_meaning=False)
        else:
            raise ValueError(f"unknown target {name!r} (qwen, tiny, aya-int4, aya-int8)")
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
    model, gpu = weakref.ref(t.engine.model), t.gpu
    del t
    release_apps()
    gc.collect()
    if gpu:
        torch.mps.empty_cache()
    return model() is None


def target_from_argv(default: str = "qwen") -> str:
    """--target qwen|tiny|aya-int4|aya-int8 (or --target=tiny); anything else on the command line is an error, so a
    mistyped flag cannot fall back to the qwen target and its Metal sections."""
    p = argparse.ArgumentParser(description="a server suite (tests/serving_targets.py)")
    p.add_argument("--target", choices=("qwen", "tiny", "aya-int4", "aya-int8"), default=default)
    return p.parse_args().target


if __name__ == "__main__":                               # check the tiny target on its own: build, self-check, sizes
    t0 = time.perf_counter()
    t = load_target(os.environ.get("TARGET", "tiny"))
    tok = t.engine.tokenizer
    print(f"{t.name}: vocab {tok.vocab_size()}, BOS {tok.bos_id}, eos {sorted(t.engine.eos_ids)}, one-line chat "
          f"{t.chat_len()} tokens, built and self-checked in {time.perf_counter() - t0:.1f} s")
