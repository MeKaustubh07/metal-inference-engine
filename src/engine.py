"""One place that knows how to build a ready-to-run engine for each supported model."""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from backend.metal import MetalBackend
from backend.torch_ref import TorchBackend
from config import Qwen35Config
from models.qwen3_5 import Qwen35Model
from quant import QtFile
from tokenizer import Tokenizer
from weight_loader import open_weights

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen3.5-0.8b": dict(dir="models/qwen3.5-0.8b", family="qwen3_5", weights="model.safetensors-00001-of-00001.safetensors",
                         eos=[248046, 248044], chat="qwen3.5", policy="configs/quant/qwen3.5-0.8b.json",
                         sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)),
    "qwen3.5-2b": dict(dir="models/qwen3.5-2b", family="qwen3_5", weights="model.safetensors-00001-of-00001.safetensors",
                       eos=[248046, 248044], chat="qwen3.5", policy="configs/quant/qwen3.5-2b.json",
                       sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)),
}


@dataclass
class Engine:
    name: str
    model: object
    tokenizer: Tokenizer
    eos_ids: set[int]
    chat_style: str
    sampling: dict = field(default_factory=dict)


def make_backend(kind: str, policy: str | None):
    if kind == "cpu":
        return TorchBackend("cpu", torch.float32)
    if kind == "mps":
        return TorchBackend("mps", torch.bfloat16)
    if kind == "metal":
        return MetalBackend()
    if kind in ("metal-int8", "metal-int4"):
        scheme = kind.split("-")[1]
        return MetalBackend(scheme, policy if scheme == "int4" and policy and (ROOT / policy).exists() else None)
    raise ValueError(f"unknown backend {kind}")


def load_engine(name: str, backend: str = "metal", weights_file: str | None = None) -> Engine:
    spec = MODELS[name]
    d = ROOT / spec["dir"]
    be = make_backend(backend, str(ROOT / spec["policy"]))
    weights = QtFile(weights_file) if weights_file else open_weights(str(d / spec["weights"]))
    if spec["family"] != "qwen3_5":
        raise ValueError(f"unknown model family {spec['family']!r}")
    model = Qwen35Model(Qwen35Config.from_json(str(d / "config.json")), weights, be)
    return Engine(name, model, Tokenizer(str(d / "tokenizer.json")), set(spec["eos"]), spec["chat"], spec["sampling"])
