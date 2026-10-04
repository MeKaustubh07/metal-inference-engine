"""The backend protocol: the fixed set of operations the model is built from.

The model only ever calls these. Swapping the implementation (fp32 CPU reference, bf16 on the GPU,
hand-written Metal kernels, quantized weights) never touches model code.
"""
from typing import Protocol

import torch


class Backend(Protocol):
    name: str
    device: torch.device

    def prepare(self, w: torch.Tensor, name: str | None = None):
        """Move/convert one weight into this backend's resident format (called once per weight).
        `name` lets a backend apply per-tensor policy, e.g. keep the embedding at higher precision."""

    def linear(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None,
               residual: torch.Tensor | None = None) -> torch.Tensor:
        """x [T, K] @ w[N, K].T (+ b) (+ residual) -> [T, N] in fp32. T == 1 is the decode matvec."""

    def embedding(self, table, ids: torch.Tensor) -> torch.Tensor:
        """Rows `ids` of the (possibly quantized) embedding table, as fp32 [T, hidden]."""

    def rms_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """[..., D] -> [..., D]"""

    def rope(self, x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
        """x [T, H, d] rotated by absolute positions [T]."""

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """q [T, Hq, d], k/v [S, Hkv, d] -> [T, Hq, d]; query i sits at position S - T + i."""

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """silu(gate) * up"""

    def swiglu(self, x: torch.Tensor, w_gate_up: torch.Tensor) -> torch.Tensor:
        """w_gate_up = [gate; up] stacked [2F, K]: returns silu(x @ gate.T) * (x @ up.T), [T, F]."""

    # ---- protocol v2, added for Qwen3.5 (the port forced the interface to grow)
    def causal_conv1d(self, x: torch.Tensor, tail: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Depthwise causal conv + SiLU: x [T, C], tail [K-1, C], w [C, K] -> (out [T, C], new tail)."""

    def gated_delta(self, q, k, v, beta, g, S) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated delta rule: q, k [T, H, dk], v [T, H, dv], beta, g [T, H], S [H, dk, dv] -> (o [T, H, dv], S)."""

    def rms_norm_gated(self, x, z, w, eps) -> torch.Tensor:
        """RMSNorm(x) * w * silu(z) over the last dim."""

    def deltanet_decode(self, qkv, z, b, a, state, slot, conv_w, A_log, dt_bias, norm_w, eps, dims) -> torch.Tensor:
        """One token of one Gated DeltaNet layer (conv + delta rule + gated norm); updates state's slot in place."""

    def deltanet_decode_batch(self, qkv, z, b, a, states, slot, conv_w, A_log, dt_bias, norm_w, eps,
                              dims) -> torch.Tensor:
        """The same for B sequences at once (row i belongs to states[i]) -> [B, H*dv]."""

    def paged_attention(self, q, k_pool, v_pool, tables, lens, block_size) -> torch.Tensor:
        """Batched decode attention reading a paged KV pool in place: q [B, Hq, d] -> [B, Hq, d]."""

    # ---- protocol v3, added for Cohere2 (Tiny Aya)
    def layer_norm(self, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        """LayerNorm without bias over the last dim: (x - mean) / sqrt(var + eps) * w."""
