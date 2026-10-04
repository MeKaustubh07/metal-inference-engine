"""Answer key: tokenize test strings with the official HF tokenizer, write JSON goldens.

Per case: ids (HF's default encode, with the post-processor's specials, e.g. Tiny Aya's BOS), ids_plain (without
them) and decoded (HF's decode of ids_plain).
usage: golden_tokens.py [model_dir] [out.json]      (default: Qwen3.5-0.8B -> tests/golden_tokens_qwen35.json)
       golden_tokens.py models/tiny-aya-global tests/golden_tokens_aya.json   (gitignored: needs the gated files)
"""
import json
import random
import sys

from tokenizers import Tokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3.5-0.8b"
out_path = sys.argv[2] if len(sys.argv) > 2 else "tests/golden_tokens_qwen35.json"
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
cases = [
    "The capital of France is",
    "Hello world",
    "hello   world  ",
    "I'm here, you're there. Don't stop!",
    "Numbers: 12345 and 3.14159, year 2026",
    "Kaustubh builds an inference engine in TypeScript.",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "नमस्ते दुनिया — こんにちは世界 — 🚀🔥",
    "def f(x):\n    return x * 2\n\n\n",
    "   leading spaces and\ttabs\tand trailing   ",
    "",
    # scripts with combining marks (Qwen3.5's pre-tokenizer lists \p{M}; see src/tokenizer.py)
    "தமிழ் மொழி", "สวัสดีครับ", "مَرْحَبًا بِكُمْ", "שָׁלוֹם", "Tiếng Việt có dấu", "éclair", "❤️ 👍🏽",
    "ক্ষমা করুন", "ಕನ್ನಡ ಭಾಷೆ", "ພາສາລາວ", "Ελληνικά ά", "ǟb", "Zalgo z̷̢a̶l̵g̸o", "日本語のテキスト", "한국어 텍스트",
]
random.seed(0)
alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789.,!?'\n-_()[]{}<>|éü你好😀्ािु"
for _ in range(40):
    cases.append("".join(random.choice(alphabet) for _ in range(random.randint(1, 60))))
# digit grouping (Tiny Aya splits digits in threes from the right), other numeral systems, NFD, CRLF, chat specials
cases += [
    "12345", "1234567 apples", "12345abc", "x_12345", "a1b22c333d4444", "year 2026, pop 8,100,000", "price: $1,000,000.00",
    "١٢٣٤٥ عربي", "१२३४५ हिन्दी", "１２３４５", "日本語 123456 テキスト", "🚀🔥 2026", "éclair (NFD)",
    "line1\r\nline2\n\n\tend", "Kiswahili ni lugha", "Ọmọ Yorùbá", "ሰላም ልዑል", "Xin chào thế giới",
    "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi<|END_OF_TURN_TOKEN|>",
]
random.seed(1)
digits = "0123456789٠١٢٣٤٥٦٧٨٩०१२३४५६७८९０１２ ,.ab_\n"
for _ in range(100):
    cases.append("".join(random.choice(digits) for _ in range(random.randint(1, 40))))

out = []
for c in cases:
    plain = tok.encode(c, add_special_tokens=False).ids
    out.append({"text": c, "ids": tok.encode(c).ids, "ids_plain": plain, "decoded": tok.decode(plain, skip_special_tokens=False)})
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(out)} cases to {out_path}")
