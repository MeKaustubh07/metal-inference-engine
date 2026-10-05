"""Reference backend: the hand-written ops from ops.py, on CPU in fp32 (the oracle) or on the GPU (MPS)."""
import torch

import ops


class TorchBackend:
    """device='cpu', weight_dtype=float32  -> exact reference path used for HF comparisons.
    device='mps', weight_dtype=bfloat16    -> fast path: bf16 weights resident on the GPU, fp32 activations."""

    def __init__(self, device: str = "cpu", weight_dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.name = f"torch-{self.device.type}-{str(weight_dtype).removeprefix('torch.')}"

    def prepare(self, w: torch.Tensor, name: str | None = None) -> torch.Tensor:
        return w.to(device=self.device, dtype=self.weight_dtype)

    def linear(self, x, w, b=None, residual=None):
        y = (x.to(w.dtype) @ w.T).float()
        if b is not None:
            y = y + b.float()
        return y + residual if residual is not None else y

    def embedding(self, table, ids):
        return table[ids.to(table.device)].float()

    def rms_norm(self, x, w, eps):
        return ops.rms_norm(x, w, eps)

    def layer_norm(self, x, w, eps):
        return ops.layer_norm(x, w, eps)

    def rope(self, x, positions, theta):
        return ops.rope(x, positions, theta)

    def attention(self, q, k, v, causal=True, window=None):
        return ops.attention(q, k, v, causal, window)

    def silu_mul(self, gate, up):
        return ops.silu_mul(gate, up)

    def swiglu(self, x, w_gate_up):
        y = self.linear(x, w_gate_up)
        F = w_gate_up.shape[0] // 2
        return self.silu_mul(y[:, :F], y[:, F:])

    # ---- Qwen3.5 (Gated DeltaNet) ops: backend protocol v2
    def causal_conv1d(self, x, tail, w):
        return ops.causal_conv1d(x.float(), tail, w)

    def gated_delta(self, q, k, v, beta, g, S):
        """Chunked form for prefill (matrix products), recurrent form for single-token decode."""
        if q.shape[0] == 1:
            return ops.gated_delta_recurrent(q, k, v, beta, g, S)
        return ops.gated_delta_chunked(q, k, v, beta, g, S)

    def rms_norm_gated(self, x, z, w, eps):
        return ops.rms_norm_gated(x, z, w, eps)

    def deltanet_decode(self, qkv, z, b, a, state, slot, conv_w, A_log, dt_bias, norm_w, eps, dims):
        """One-token Gated DeltaNet step (conv + delta rule + gated norm). Updates state.conv_tail/S[slot] in place.
        dims = (H, dk, dv, key_dim). Returns [1, H*dv]."""
        H, dk, dv, key_dim = dims
        u, state.conv_tail[slot] = self.causal_conv1d(qkv, state.conv_tail[slot], conv_w)
        q = ops.l2norm(u[:, :key_dim].reshape(1, H, dk).float()) * (1.0 / dk ** 0.5)
        k = ops.l2norm(u[:, key_dim:2 * key_dim].reshape(1, H, dk).float())
        v = u[:, 2 * key_dim:].reshape(1, H, dv).float()
        beta = torch.sigmoid(b.float())
        ab = a.float() + dt_bias
        g = -torch.exp(A_log) * torch.where(ab > 20, ab, torch.log1p(torch.exp(ab)))
        o, state.S[slot] = self.gated_delta(q, k, v, beta, g, state.S[slot])
        return self.rms_norm_gated(o.reshape(H, dv), z.reshape(H, dv), norm_w, eps).reshape(1, H * dv)

    def paged_attention(self, q, k_pool, v_pool, tables, lens, block_size, window=None):
        """Batched decode attention over paged KV: q [B, Hq, d]; k_pool/v_pool one layer [blocks, bs, Hkv, d];
        tables [B, max_nb], lens [B]. Reference: gather each sequence's positions, then ordinary attention. With a
        window, only from position lens[i] - window on: table entries below its block are never read."""
        outs = []
        for i in range(q.shape[0]):
            n = int(lens[i]); nb = -(-n // block_size)
            s0 = max(0, n - window) if window else 0
            b0 = s0 // block_size
            rows = slice(s0 - b0 * block_size, n - b0 * block_size)
            K = k_pool[tables[i, b0:nb].long()].flatten(0, 1)[rows]
            V = v_pool[tables[i, b0:nb].long()].flatten(0, 1)[rows]
            outs.append(self.attention(q[i:i + 1], K, V, causal=True, window=window))
        return torch.cat(outs)

    def deltanet_decode_batch(self, qkv, z, b, a, states, slot, conv_w, A_log, dt_bias, norm_w, eps, dims):
        """B sequences, one token each: rows of qkv/z/b/a belong to states[i]. Returns [B, H*dv]."""
        return torch.cat([self.deltanet_decode(qkv[i:i + 1], z[i:i + 1], b[i:i + 1], a[i:i + 1], st, slot, conv_w,
                                               A_log, dt_bias, norm_w, eps, dims) for i, st in enumerate(states)])

    def sync(self) -> None:
        """Wait for queued GPU work (needed for honest timing on MPS)."""
        if self.device.type == "mps":
            torch.mps.synchronize()
