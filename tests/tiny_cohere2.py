"""The shared test oracle for Cohere2 (Tiny Aya), used by tests/test_cohere2.py and tests/test_window.py (importing a
test file would run it): small random models built by transformers and loaded by both HF and the engine, HF's own
outputs, and a float64 brute-force attention that applies the sliding window from absolute positions."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Cohere2Config as HFConfig, Cohere2ForCausalLM
from transformers.utils import logging as hf_logging

from config import Cohere2Config
from models.cohere2 import Cohere2Model
from weight_loader import SafetensorsFile

hf_logging.disable_progress_bar()


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def build(tmp: str, window: int, seed: int = 0, eps: float = 1e-5, tied: bool = True, max_pos: int = 8192,
          vocab_size: int = 512):
    """A random 8-layer Cohere2 (sliding, sliding, sliding, full, x2) with 8 query heads per 2 KV heads of 16 dims,
    saved and loaded by HF and by the engine."""
    torch.manual_seed(seed)
    hc = HFConfig(vocab_size=vocab_size, hidden_size=128, intermediate_size=320, num_hidden_layers=8, num_attention_heads=8,
                  num_key_value_heads=2, max_position_embeddings=max_pos, layer_norm_eps=eps, logit_scale=0.25,
                  sliding_window=window, rope_parameters={"rope_type": "default", "rope_theta": 50000.0},
                  tie_word_embeddings=tied, initializer_range=0.3, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    hf = Cohere2ForCausalLM._from_config(hc, attn_implementation="eager").eval()
    for m in hf.modules():                                     # LayerNorm weights start at 1: make them matter
        if type(m).__name__ == "Cohere2LayerNorm":
            m.weight.copy_(1 + 0.3 * torch.randn_like(m.weight))
    hf.save_pretrained(tmp)
    return hf, lambda **kw: Cohere2Model(Cohere2Config.from_json(f"{tmp}/config.json"),
                                         SafetensorsFile(f"{tmp}/model.safetensors"), **kw)


def hf_run(hf, ids):
    """HF logits [T, vocab] and the outputs of the embedding, every layer and the final norm."""
    cap, hooks = {}, []
    m = hf.model
    hooks.append(m.embed_tokens.register_forward_hook(lambda _m, _i, o: cap.__setitem__("embed", o[0])))
    for i, layer in enumerate(m.layers):
        hooks.append(layer.register_forward_hook(
            lambda _m, _i, o, i=i: cap.__setitem__(f"l{i}_out", (o[0] if isinstance(o, tuple) else o)[0])))
    hooks.append(m.norm.register_forward_hook(lambda _m, _i, o: cap.__setitem__("final_norm", o[0])))
    logits = hf(ids[None]).logits[0]
    for h in hooks:
        h.remove()
    return logits, cap


def hf_greedy(hf, ids, n):
    out = ids.tolist()
    for _ in range(n):
        out.append(int(hf(torch.tensor([out])).logits[0, -1].argmax()))
    return out[len(ids):]


def ref_attention(q, k, v, q_pos, k_pos, window=None):
    """Attention in float64, written from absolute positions: the query at q_pos[i] sees the key at k_pos[j] iff
    k_pos[j] <= q_pos[i] and, with a window W, k_pos[j] > q_pos[i] - W (transformers' masking_utils: causal and
    `kv_idx > q_idx - sliding_window`). q [T, Hq, d], k/v [S, Hkv, d] -> [T, Hq, d] (float64)."""
    d, group = q.shape[-1], q.shape[1] // k.shape[1]
    kk, vv = (x.double().repeat_interleave(group, dim=1) for x in (k, v))
    scores = torch.einsum("thd,shd->hts", q.double(), kk) / d ** 0.5
    qp, kp = q_pos.view(-1, 1), k_pos.view(1, -1)
    sees = kp <= qp
    if window is not None:
        sees &= kp > qp - window
    return torch.einsum("hts,shd->thd", torch.softmax(scores.masked_fill(~sees, float("-inf")), -1), vv)
