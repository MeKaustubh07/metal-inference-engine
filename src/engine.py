"""One place that knows how to build a ready-to-run engine for each supported model."""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from backend.metal import MetalBackend
from backend.torch_ref import TorchBackend
from chat import TemplateChat
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
    # Tiny Aya Global (Cohere2). chat "template": the model's own Jinja template, read from its tokenizer_config.json.
    # Stops: <EOS_TOKEN>, <|END_OF_TURN_TOKEN|>, <|END_RESPONSE|>. Sampling: the model card's. max_model_len: the
    # sliding window, below which sliding layers equal full causal attention (window support comes later).
    "tiny-aya-global": dict(dir="models/tiny-aya-global", family="cohere2", weights="model.safetensors.index.json",
                            eos=[3, 6, 261001], chat="template", policy="configs/quant/tiny-aya-global.json",
                            sampling=dict(temperature=0.1, top_p=0.95, top_k=50, repetition_penalty=1.0),
                            max_model_len=4096),
}
FAMILIES = {"qwen3_5": (Qwen35Config, Qwen35Model)}         # family -> (config class, model class); cohere2 comes next


@dataclass
class Engine:
    name: str
    model: object
    tokenizer: Tokenizer
    eos_ids: set[int]
    chat_style: str | TemplateChat
    sampling: dict = field(default_factory=dict)
    max_model_len: int | None = None           # a model's hard cap on prompt + generated tokens (None: no cap)


def chat_style(spec: dict, d: Path) -> str | TemplateChat:
    return TemplateChat(str(d / "tokenizer_config.json")) if spec["chat"] == "template" else spec["chat"]


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
    if spec["family"] not in FAMILIES:
        raise NotImplementedError(f"{name}: no model class for family {spec['family']!r} yet")
    config_cls, model_cls = FAMILIES[spec["family"]]
    be = make_backend(backend, str(ROOT / spec["policy"]))
    weights = QtFile(weights_file) if weights_file else open_weights(str(d / spec["weights"]))
    model = model_cls(config_cls.from_json(str(d / "config.json")), weights, be)
    return Engine(name, model, Tokenizer(str(d / "tokenizer.json")), set(spec["eos"]), chat_style(spec, d),
                  spec["sampling"], spec.get("max_model_len"))
