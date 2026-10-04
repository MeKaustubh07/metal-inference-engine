"""Model recipe cards, read from config.json: Qwen35Config (under text_config) and Cohere2Config (top level)."""
import json
from dataclasses import dataclass


@dataclass
class Qwen35Config:
    """Qwen3.5 text model (hybrid Gated DeltaNet + gated full attention). Fields live under config["text_config"]."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int          # 8 query heads (each with a same-size output gate)
    num_key_value_heads: int          # 2
    head_dim: int                     # 256, explicit (NOT hidden_size // heads)
    rms_norm_eps: float
    rope_theta: float                 # 1e7
    partial_rotary_factor: float      # 0.25 -> only the first 64 of 256 dims rotate
    layer_types: list[str]            # "linear_attention" | "full_attention", one per layer
    linear_conv_kernel_dim: int       # 4
    linear_key_head_dim: int          # 128
    linear_value_head_dim: int        # 128
    linear_num_key_heads: int         # 16
    linear_num_value_heads: int       # 16
    tie_word_embeddings: bool

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_json(cls, path: str) -> "Qwen35Config":
        top = json.load(open(path))
        c = top["text_config"]
        rope = c["rope_parameters"]
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"], rms_norm_eps=c["rms_norm_eps"],
            rope_theta=float(rope["rope_theta"]), partial_rotary_factor=rope["partial_rotary_factor"],
            layer_types=list(c["layer_types"]), linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
            linear_key_head_dim=c["linear_key_head_dim"], linear_value_head_dim=c["linear_value_head_dim"],
            linear_num_key_heads=c["linear_num_key_heads"], linear_num_value_heads=c["linear_num_value_heads"],
            tie_word_embeddings=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
        )


@dataclass
class Cohere2Config:
    """Cohere2 (Tiny Aya): every layer is attention. Sliding-window layers (RoPE) alternate with full layers (no
    positional encoding); one LayerNorm per layer feeds attention and the MLP in parallel; the embedding is tied."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int          # 16 query heads
    num_key_value_heads: int          # 4
    head_dim: int                     # 128 = hidden_size // heads
    layer_norm_eps: float
    rope_theta: float                 # 50000, interleaved pairs, all head_dim dims, sliding layers only
    sliding_window: int               # 4096: a token sees itself and the 4095 before it
    layer_types: list[str]            # "sliding_attention" | "full_attention", one per layer
    logit_scale: float                # logits are multiplied by this (1.0 for Tiny Aya)
    tie_word_embeddings: bool
    max_position_embeddings: int

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    @classmethod
    def from_json(cls, path: str) -> "Cohere2Config":
        c = json.load(open(path))
        # The settings this engine implements. Some change transformers' graph (attention_bias, hidden_act,
        # rope_scaling); the original-Cohere keys (use_parallel_block, rotary_pct, ...) are ignored by transformers,
        # which always builds the Tiny Aya graph. Either way a config asking for anything else is refused here,
        # instead of producing wrong text.
        want = {"model_type": "cohere2", "use_parallel_block": True, "use_qk_norm": False, "attention_bias": False,
                "rotary_pct": 1.0, "position_embedding_type": "rope_gptj", "rope_scaling": None, "hidden_act": "silu",
                "use_gated_activation": True, "use_parallel_embedding": False, "use_embedding_sharing": True}
        bad = {k: c.get(k) for k, v in want.items() if k in c and c[k] != v}
        if c.get("model_type") != "cohere2":
            bad["model_type"] = c.get("model_type")
        heads = c["num_attention_heads"]
        head_dim = c["hidden_size"] // heads              # transformers takes config.json's head_dim when present;
        if c.get("head_dim", head_dim) != head_dim:       # this engine supports only hidden_size // heads
            bad["head_dim"] = c["head_dim"]
        layer_types = list(c["layer_types"])
        if len(layer_types) != c["num_hidden_layers"] or set(layer_types) - {"sliding_attention", "full_attention"}:
            bad["layer_types"] = layer_types
        sw = c.get("sliding_window")
        if not isinstance(sw, int) or isinstance(sw, bool) or sw <= 0:
            bad["sliding_window"] = sw
        # transformers 5 nests rope_theta under rope_parameters (older files keep it at the top level), and reads the
        # scaling type from "rope_type", or else the legacy "type"
        rope = c.get("rope_parameters") or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default" or set(rope) - {"rope_type", "type", "rope_theta"}:
            bad["rope_parameters"] = rope
        if bad:
            raise ValueError(f"unsupported Cohere2 settings: {bad}")
        rope_theta = rope.get("rope_theta", c.get("rope_theta"))
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=heads,
            num_key_value_heads=c["num_key_value_heads"], head_dim=head_dim, layer_norm_eps=c["layer_norm_eps"],
            rope_theta=float(rope_theta), sliding_window=c["sliding_window"], layer_types=layer_types,
            logit_scale=float(c.get("logit_scale", 0.0625)),          # transformers' default when the key is absent
            tie_word_embeddings=c.get("tie_word_embeddings", True),   # Tiny Aya's file never says; the default ties
            max_position_embeddings=c["max_position_embeddings"],
        )
