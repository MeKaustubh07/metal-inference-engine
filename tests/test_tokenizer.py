"""Compare our tokenizer against the official answer keys (scripts/golden_tokens.py) for each model's tokenizer.json:
the ids with and without the post-processor's specials (Tiny Aya's BOS), and the decoded text."""
import json
import os
import sys
import unicodedata

sys.path.insert(0, "src")
from tokenizer import Tokenizer

MODELS = [("Qwen3.5-0.8B", "models/qwen3.5-0.8b", "tests/golden_tokens_qwen35.json"),
          # gitignored: the gated files are needed to make it (scripts/golden_tokens.py models/tiny-aya-global ...)
          ("Tiny Aya Global", "models/tiny-aya-global", "tests/golden_tokens_aya.json")]

all_ok = True
for name, model_dir, golden in MODELS:
    if not os.path.exists(f"{model_dir}/tokenizer.json"):
        print(f"SKIP {name}: needs {model_dir}/tokenizer.json")
        continue
    if not os.path.exists(golden):                     # the model is here but its answer key was never made
        print(f"FAIL {name}: run scripts/.venv/bin/python scripts/golden_tokens.py {model_dir} {golden}")
        all_ok = False
        continue
    tok = Tokenizer(f"{model_dir}/tokenizer.json")
    cases = json.load(open(golden, encoding="utf-8"))
    passed = 0
    for c in cases:
        plain, full = tok.encode(c["text"]), tok.encode(c["text"], add_bos=True)
        text = unicodedata.normalize("NFC", c["text"]) if tok.nfc else c["text"]   # what decoding should give back
        decoded = tok.decode(plain)
        if plain == c["ids_plain"] and full == c["ids"] and decoded == c["decoded"] == text:
            passed += 1
        else:
            print("FAIL", name, repr(c["text"]), "\n  ours:", plain, "\n  gold:", c["ids_plain"],
                  "\n  with BOS ok:", full == c["ids"], " decoded ok:", decoded == c["decoded"], decoded == text)
    all_ok &= passed == len(cases)
    print(f"{'PASS' if passed == len(cases) else 'FAIL'}  {name}: {passed}/{len(cases)} cases match the official "
          f"tokenizer (vocab {tok.vocab_size()}, BOS {tok.bos_id})")

# the 2B ships the same tokenizer files, so the 0.8B's answer keys cover it too
same = all(open(f"models/qwen3.5-2b/{f}", "rb").read() == open(f"models/qwen3.5-0.8b/{f}", "rb").read()
           for f in ("tokenizer.json", "tokenizer_config.json"))
print(f"{'PASS' if same else 'FAIL'}  Qwen3.5-2B tokenizer.json and tokenizer_config.json are byte-equal to the 0.8B's")
all_ok &= same


# ---------------------------------------------------------------- the merge (M6): a heap, as HF tokenizers does it
# The old loop rescanned the whole word for its best pair after every merge: quadratic in the word's length, and an
# unspaced CJK text is one long word (64K characters: 93-128 s). The heap merge must give HF's tokens everywhere,
# today's tokens on every word Tiny Aya merges (its table has no merge ranked before one that makes a part of it),
# and be fast on long unspaced text.
import random
import time
from types import SimpleNamespace

from tokenizers import Tokenizer as HFTokenizer, models as hf_models

FULL = "--full" in sys.argv


def check(name, ok):
    global all_ok
    all_ok &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def old_bpe(merge_rank, chunk):
    """The merge loop before M6, kept here as an oracle: glue every occurrence of the best-ranked pair, repeat."""
    symbols = list(chunk)
    while len(symbols) > 1:
        best = min(((merge_rank[p], p) for p in zip(symbols, symbols[1:]) if p in merge_rank), default=None)
        if best is None:
            break
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best[1]:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def nonmonotone(merge_rank):
    """Merges ranked before any merge that makes one of their parts (a token can have several makers)."""
    made = {}
    for (a, b), r in merge_rank.items():
        made[a + b] = max(made.get(a + b, r), r)
    return sum(any(r < made.get(x, -1) for x in pair) for pair, r in merge_rank.items())


# 1. synthetic tables over "abc", every string up to length 7 (9 with --full): == HF's BPE. HF merges one occurrence
# at a time, so where a merge ranks before the one that makes its part, it can fire between two occurrences of the
# same pair: with (ab, a) ranked before (a, b), "abab" is [aba, b] in HF but [ab, ab] in a loop that glues every
# occurrence at once. The first table is that one; the random ones are mostly shuffled, so not monotone either
rng, mismatch, old_differs, n_str, n_bad_tables = random.Random(0), 0, 0, 0, 0
for t in range(60 if FULL else 24):
    if t == 0:
        tokens, merges = ["a", "b", "c", "ab", "aba"], [("ab", "a"), ("a", "b")]
    else:
        tokens, merges = ["a", "b", "c"], []
        while len(merges) < rng.randint(6, 14):
            x, y = rng.choice(tokens), rng.choice(tokens)
            if (x, y) not in merges and len(x + y) <= 6:
                merges.append((x, y))
                tokens += [x + y] if x + y not in tokens else []
        if t % 4:
            rng.shuffle(merges)
    rank = {m: i for i, m in enumerate(merges)}
    n_bad_tables += nonmonotone(rank) > 0
    hf = HFTokenizer(hf_models.BPE(vocab={tk: i for i, tk in enumerate(tokens)}, merges=merges))
    ours = SimpleNamespace(merge_rank=rank)
    strings = [""]
    for _ in range(9 if FULL else 7):
        strings = [s + c for s in strings for c in "abc"] + strings
    for st in set(strings) - {""}:
        want = hf.encode(st).tokens
        mismatch += Tokenizer._bpe(ours, st) != want
        old_differs += old_bpe(rank, st) != want
        n_str += 1
check(f"merge == HF tokenizers' BPE on {n_str} strings over {t + 1} synthetic tables ({n_bad_tables} not monotone, "
      f"where the old loop differs from HF on {old_differs}): {mismatch} differ", mismatch == 0 and old_differs > 0)

# 2. the real tokenizers on whole texts: ids == HF's; and Tiny Aya's merges == the old loop's, piece by piece
for name, model_dir, golden in MODELS:
    if not os.path.exists(f"{model_dir}/tokenizer.json"):
        print(f"SKIP {name}: needs {model_dir}/tokenizer.json")
        continue
    tok, hf = Tokenizer(f"{model_dir}/tokenizer.json"), HFTokenizer.from_file(f"{model_dir}/tokenizer.json")
    rng = random.Random(1)
    pool = "abcdefg hijk ABC 0123 .,!? 你好世界中文 いっぱいもうです カタカナ 한국어 नमस्ते مرحبا ñéü 😀🚀\n\t"
    texts = [c["text"] for c in json.load(open(golden, encoding="utf-8"))] if os.path.exists(golden) else []
    texts += ["".join(rng.choice(pool) for _ in range(rng.randint(1, 80))) for _ in range(5000)]
    texts += ["".join(chr(rng.randrange(0x4E00, 0x9FA6)) for _ in range(n)) for n in (100, 500, 1000, 4000)]
    for f in sorted(os.listdir("models/long_texts")) if os.path.isdir("models/long_texts") else []:
        texts.append(open(f"models/long_texts/{f}", encoding="utf-8").read()[:20000])
    bad = [t for t in texts if tok.encode(t) != hf.encode(t, add_special_tokens=False).ids]
    nm = nonmonotone(tok.merge_rank)
    first = f"; first difference {bad[0][:30]!r}" if bad else ""
    check(f"{name}: ids == HF tokenizers' on {len(texts)} texts (goldens, 5,000 random, unspaced CJK to 4,000 "
          f"characters, the long texts; {nm} merges out of order){first}", not bad)
    if name == "Tiny Aya Global":                    # the words encode merges: specials peeled off, NFC, pre-split,
        words = set()                                  # mapped to byte characters
        for t in texts:
            for part in tok.special_re.split(t) if tok.special_re else [t]:
                if part and part not in tok.special:
                    part = unicodedata.normalize("NFC", part) if tok.nfc else part
                    words |= {"".join(tok.byte_to_char[b] for b in w.encode("utf-8")) for w in tok._pieces(part)}
        short = {w for w in words if len(w) <= 1500}                  # the old loop is quadratic
        differ = sum(Tokenizer._bpe(tok, w) != old_bpe(tok.merge_rank, w) for w in short)
        check(f"{name}: no merge ranks before one that makes a part of it ({nm}), and the merge == the old loop on "
              f"all {len(short)} distinct words of up to 1,500 characters ({differ} differ; "
              f"{len(words) - len(short)} longer ones checked against HF only)", nm == 0 and differ == 0)

# 3. fast on long unspaced text (a two-stage gate: the 64K run only if the 8K one is quick)
for name, model_dir, _ in MODELS:
    if not os.path.exists(f"{model_dir}/tokenizer.json"):
        continue
    tok, rng = Tokenizer(f"{model_dir}/tokenizer.json"), random.Random(2)
    cjk = "".join(chr(rng.randrange(0x4E00, 0x9FA6)) for _ in range(65536))
    t0 = time.perf_counter(); tok.encode(cjk[:8192]); t8 = time.perf_counter() - t0
    t64 = None
    if t8 < 0.5:
        t0 = time.perf_counter(); tok.encode(cjk); t64 = time.perf_counter() - t0
    check(f"{name}: 8,192 unspaced CJK characters in {t8:.2f} s (< 0.5), 65,536 in "
          f"{f'{t64:.2f} s' if t64 is not None else 'not run'} (< 2)", t8 < 0.5 and t64 is not None and t64 < 2)

sys.exit(0 if all_ok else 1)
