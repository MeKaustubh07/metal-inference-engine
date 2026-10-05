"""Quantize a safetensors checkpoint into a .qt file, one tensor at a time. The input is read lazily (memory-mapped);
save_qt writes the header first, then quantizes, writes and drops each tensor in turn, in whole-row chunks of at most
2^24 weights, so peak RAM is about one tensor's output plus the checkpoint's mapped pages. The file appears at <out.qt>
only once it is complete.

usage: quantize.py <model.safetensors | model.safetensors.index.json> <out.qt> --scheme int8|int4
                   [--policy configs/quant/<model>.json] [--prefix P]
"""
import argparse
import sys
import time

sys.path.insert(0, "src")
from quant import load_policy, save_qt
from weight_loader import open_weights


class Filtered:
    """Expose only tensors under `prefix` (e.g. the text model of a multimodal checkpoint)."""

    def __init__(self, src, prefix: str):
        self.src, self.prefix = src, prefix
        if not self.tensor_names():
            raise SystemExit(f"no tensor name starts with {prefix!r} (names look like {src.tensor_names()[0]!r})")

    def tensor_names(self):
        return [n for n in self.src.tensor_names() if n.startswith(self.prefix)]

    def get(self, name):
        return self.src.get(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--scheme", choices=["int8", "int4"], required=True)
    ap.add_argument("--policy", default=None, help="JSON with keep_int8 tensor names (INT4 mixed precision)")
    ap.add_argument("--prefix", default="", help="only keep tensors whose name starts with this")
    a = ap.parse_args()
    t0 = time.perf_counter()
    save_qt(a.out, Filtered(open_weights(a.src), a.prefix), a.scheme, keep_int8=load_policy(a.policy))
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
