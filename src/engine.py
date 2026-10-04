"""One place that knows how to build a ready-to-run engine for each supported model."""
import logging
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
log = logging.getLogger("engine")

MODELS = {
    "qwen3.5-0.8b": dict(dir="models/qwen3.5-0.8b", family="qwen3_5", weights="model.safetensors-00001-of-00001.safetensors",
                         eos=[248046, 248044], chat="qwen3.5", thinking=True, policy="configs/quant/qwen3.5-0.8b.json",
                         sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)),
    "qwen3.5-2b": dict(dir="models/qwen3.5-2b", family="qwen3_5", weights="model.safetensors-00001-of-00001.safetensors",
                       eos=[248046, 248044], chat="qwen3.5", thinking=True, policy="configs/quant/qwen3.5-2b.json",
                       sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)),
    # Tiny Aya Global (Cohere2). chat "template": the model's own Jinja template, read from its tokenizer_config.json.
    # Stops: <EOS_TOKEN>, <|END_OF_TURN_TOKEN|>, <|END_RESPONSE|>. Sampling: the model card's. max_model_len: the
    # sliding window, below which sliding layers equal full causal attention (window support comes later). No
    # thinking mode. policy: written by scripts/calibrate_quant.py (milestone M4); until then INT4 warns.
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
    thinking: bool = False                     # the model has a reasoning mode (Qwen3.5: <think> ... </think>)


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
        if scheme == "int4" and policy and not (ROOT / policy).exists():
            log.warning(f"no INT4 policy {policy}: every group goes to INT4 (scripts/calibrate_quant.py writes it)")
        return MetalBackend(scheme, policy if scheme == "int4" and policy and (ROOT / policy).exists() else None)
    raise ValueError(f"unknown backend {kind}")


def load_engine(name: str, backend: str = "metal", weights_file: str | None = None) -> Engine:
    spec = MODELS[name]
    d = ROOT / spec["dir"]
    if spec["family"] not in FAMILIES:
        raise NotImplementedError(f"{name}: no model class for family {spec['family']!r} yet")
    config_cls, model_cls = FAMILIES[spec["family"]]
    cfg, tok = config_cls.from_json(str(d / "config.json")), Tokenizer(str(d / "tokenizer.json"))
    # logits are cut to tokenizer.vocab_size() (the head may be padded past it): the ids must be contiguous, and every
    # stop id must survive the cut
    n = tok.vocab_size()
    if max(tok.id_to_token) + 1 != n or n > cfg.vocab_size or max(spec["eos"]) >= n:
        raise ValueError(f"{name}: {n} token ids (max {max(tok.id_to_token)}), {cfg.vocab_size} output rows, "
                         f"stop ids {spec['eos']}")
    be = make_backend(backend, str(ROOT / p) if (p := spec.get("policy")) else None)
    weights = QtFile(weights_file) if weights_file else open_weights(str(d / spec["weights"]))
    model = model_cls(cfg, weights, be)
    caps = [c for c in (spec.get("max_model_len"), getattr(model, "max_positions", None)) if c]
    return Engine(name, model, tok, set(spec["eos"]), chat_style(spec, d), spec["sampling"],
                  min(caps) if caps else None, spec.get("thinking", False))
