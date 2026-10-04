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
    if not (os.path.exists(f"{model_dir}/tokenizer.json") and os.path.exists(golden)):
        print(f"SKIP {name}: needs {model_dir}/tokenizer.json and {golden}")
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
    print(f"{name}: {passed}/{len(cases)} cases match the official tokenizer (vocab {tok.vocab_size()}, BOS {tok.bos_id})")

# the 2B ships the same tokenizer files, so the 0.8B's answer keys cover it too
same = all(open(f"models/qwen3.5-2b/{f}", "rb").read() == open(f"models/qwen3.5-0.8b/{f}", "rb").read()
           for f in ("tokenizer.json", "tokenizer_config.json"))
print(f"{'PASS' if same else 'FAIL'}  Qwen3.5-2B tokenizer.json and tokenizer_config.json are byte-equal to the 0.8B's")
all_ok &= same
sys.exit(0 if all_ok else 1)
