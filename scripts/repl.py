"""Terminal chat with the engine: multi-turn, streaming.

usage: repl.py [--model qwen3.5-0.8b|qwen3.5-2b] [--backend cpu|mps|metal|metal-int8|metal-int4] [--weights file.qt]
"""
import argparse
import sys

sys.path.insert(0, "src")
from chat import format_chat
from engine import load_engine
from generate import generate_stream
from sampler import SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--weights", default=None, help="optional pre-quantized .qt file")
    ap.add_argument("--think", action="store_true", help="Qwen3.5: let the model think before answering")
    a = ap.parse_args()
    eng = load_engine(a.model, a.backend, a.weights)
    params = SamplingParams(**eng.sampling)
    history: list[dict] = []
    print(f"Chat with {eng.name} on {eng.model.b.name} (your engine). Empty line or Ctrl-D to quit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        ids = eng.tokenizer.encode(format_chat(history, style=eng.chat_style, enable_thinking=a.think))
        limit = eng.max_model_len                          # the model's, or its backend's (Tiny Aya INT8: 4096)
        if limit and len(ids) + 512 > limit:
            alone = eng.tokenizer.encode(format_chat(history[-1:], style=eng.chat_style, enable_thinking=a.think))
            if len(alone) + 512 > limit:                   # too long even alone: refused, the conversation kept
                print(f"(that message is {len(alone)} tokens; with a reply it passes {limit})")
                history.pop()
                continue
            print(f"(the conversation would pass {limit} tokens with a reply: starting a new one from your message)")
            history, ids = history[-1:], alone
        print("model> ", end="", flush=True)
        reply = ""
        for piece in generate_stream(eng.model, eng.tokenizer, ids, params, max_new_tokens=512, eos_ids=eng.eos_ids):
            print(piece, end="", flush=True)
            reply += piece
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
