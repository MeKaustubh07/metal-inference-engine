"""Run the OpenAI-compatible server.

usage: serve.py [--model qwen3.5-2b] [--backend metal-int4] [--weights models/qwen3.5-2b/model.int4.qt]
                [--host 127.0.0.1] [--port 8000] [--max-batch 8] [--max-waiting 64] [--kv-blocks 1024]
                [--max-model-len N] [--drain-timeout 25] [--prefill-chunk 512] [--batch-wait-ms 5]
                [--no-lock-weights]

With a quantized backend and no --weights, models/<model>/model.<int8|int4>.qt is used when it exists (quantizing
at load time would briefly need the fp32 weights in RAM). The model is warmed up before the port opens, then its
weights are locked in RAM (Metal backends), so that an idle server's weights are not compressed or swapped out;
--no-lock-weights turns that off.
On SIGTERM: stop accepting, let open requests finish for up to --drain-timeout seconds, then exit. Give the
supervisor a longer stop timeout than that (e.g. docker run --stop-timeout 30).
"""
import argparse
import logging
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, "src")
from engine import MODELS, ROOT, load_engine
from server.app import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b", choices=sorted(MODELS))
    ap.add_argument("--backend", default="metal-int4", choices=["cpu", "mps", "metal", "metal-int8", "metal-int4"])
    ap.add_argument("--weights", default=None, help="pre-quantized .qt file (quantized backends only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-waiting", type=int, default=64)
    ap.add_argument("--kv-blocks", type=int, default=1024, help="16-token blocks; 24 KiB/token on Qwen3.5-2B")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="prompt + output tokens per request (default: the model's limit, else 4096; only lowers it)")
    ap.add_argument("--drain-timeout", type=float, default=25.0)
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="prompt tokens per packed prefill pass; longer prompts are split across engine steps")
    ap.add_argument("--batch-wait-ms", type=float, default=5.0,
                    help="an idle engine collects arrivals this close together into one prefill pass (0 = off)")
    ap.add_argument("--no-lock-weights", action="store_true",
                    help="do not mlock the weights (Metal): an idle server's first request then pays to page them in")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not 1 <= a.max_batch <= 32:                         # batched decode kernels handle up to 32 rows (4 x 8)
        ap.error("--max-batch must be between 1 and 32")
    if not 0 <= a.batch_wait_ms <= 1000:
        ap.error("--batch-wait-ms must be between 0 and 1000")
    if a.prefill_chunk < 1:
        ap.error("--prefill-chunk must be at least 1")
    scheme = a.backend.removeprefix("metal-") if a.backend in ("metal-int8", "metal-int4") else None
    if a.weights is None and scheme:
        qt = ROOT / MODELS[a.model]["dir"] / f"model.{scheme}.qt"
        if qt.exists():
            a.weights = str(qt)
    if a.weights and a.weights.endswith(".qt") and not scheme:
        ap.error(f"{a.weights} holds quantized weights: use --backend metal-int8 or metal-int4")
    if a.weights and scheme and f".{scheme}." not in Path(a.weights).name:
        ap.error(f"{a.weights} does not look like an {scheme} file")

    logging.info(f"loading {a.model} on {a.backend} ({a.weights or 'safetensors'})")
    eng = load_engine(a.model, a.backend, a.weights)
    app = create_app(eng, a.max_batch, a.max_waiting, a.kv_blocks, a.max_model_len, prefill_chunk=a.prefill_chunk,
                     batch_wait_ms=a.batch_wait_ms, lock_weights=not a.no_lock_weights)  # warms up before returning
    uvicorn.run(app, host=a.host, port=a.port, log_level="info", timeout_graceful_shutdown=a.drain_timeout)


if __name__ == "__main__":
    main()
