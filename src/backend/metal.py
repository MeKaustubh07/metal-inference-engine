"""Metal backend: hand-written MSL kernels (src/kernels/*.metal) compiled at runtime and dispatched on the GPU.

Decode (one token) runs entirely through custom kernels: matvec, RMSNorm, RoPE, attention, SwiGLU.
Prefill (many tokens) keeps PyTorch's tuned GEMM and attention for the matrix-matrix work.
"""
from pathlib import Path

import torch

from backend.torch_ref import TorchBackend
from quant import QuantTensor, load_policy, quantize, scheme_for

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
TG = 256                                                   # threads per threadgroup for row-parallel kernels
MAX_BATCH = 8                                              # rows one batched matvec dispatch handles (MAX_BATCH in MSL)
KERNEL_ROWS = 4 * MAX_BATCH                                # up to here: batched kernels in groups; beyond: GEMM
DEQUANT_CHUNK = 1 << 23                                    # weights expanded per prefill GEMM piece (32 MB fp32)


def load_library():
    """Concatenate every .metal file and compile it once (runtime compiler; no Xcode needed)."""
    src = "\n".join(p.read_text() for p in sorted(KERNEL_DIR.glob("*.metal")))
    return torch.mps.compile_shader(src)


class MetalBackend(TorchBackend):
    """scheme=None keeps bf16 weights; "int8" / "int4" quantizes every 2-D weight (or accepts pre-quantized
    QuantTensors from a .qt file) and runs decode through the quantized matvec kernels."""

    def __init__(self, scheme: str | None = None, policy: str | None = None):
        super().__init__("mps", torch.bfloat16)
        self.scheme = scheme
        self.keep_int8 = load_policy(policy)              # INT4 mode: tensors measured too sensitive stay INT8
        self.name = f"metal-kernels-{scheme or 'bf16'}"
        self.lib = load_library()
        self._no_bias = torch.zeros(1, dtype=torch.bfloat16, device=self.device)
        self._no_res = torch.zeros(1, device=self.device)
        self._seq0 = torch.zeros(1, dtype=torch.int32, device=self.device)   # a standalone state = sequence 0

    def prepare(self, w, name=None):
        if isinstance(w, QuantTensor):
            return w.to(self.device)
        if self.scheme and w.ndim == 2 and w.shape[1] % 32 == 0:
            return quantize(w, scheme_for(name, self.scheme, self.keep_int8)).to(self.device)
        return super().prepare(w, name)

    def embedding(self, table, ids):
        if isinstance(table, QuantTensor):
            return table.dequantize(rows=ids.to(self.device))
        return super().embedding(table, ids)

    def _groups(self, fn, x, w, b, residual):
        """Rows MAX_BATCH at a time through a batched kernel (each group reads the weights once)."""
        return torch.cat([fn(x[i:i + MAX_BATCH], w, b, None if residual is None else residual[i:i + MAX_BATCH])
                          for i in range(0, x.shape[0], MAX_BATCH)])

    def _qlinear(self, x, w: QuantTensor, b=None, residual=None):
        M = x.shape[0]
        N, K = w.shape
        if MAX_BATCH < M <= KERNEL_ROWS:
            return self._groups(self._qlinear, x, w, b, residual)
        if M > KERNEL_ROWS:                                # prefill: expand to fp32 on the GPU, tuned GEMM
            # in row chunks, so the 248k-row tied head never needs a full-size temporary
            x, step = x.float(), max(1, DEQUANT_CHUNK // K)
            y = torch.empty(M, N, device=self.device)
            for r in range(0, N, step):
                rows = slice(r, min(N, r + step))
                y[:, rows] = x @ self._dequant_f32(w, rows).T
            if b is not None:
                y = y + b.float()
            return y + residual if residual is not None else y
        y = torch.empty(M, N, device=self.device)
        res = residual.float().contiguous() if residual is not None else self._no_res
        args = (y, w.data, w.scales, x.float().contiguous(), b if b is not None else self._no_bias, res,
                K, N, int(b is not None), int(residual is not None))
        if M == 1:                                         # single-sequence decode
            if w.scheme == "int8":
                self.lib.matvec_q8(*args, threads=N * 32, group_size=TG)
            else:
                self.lib.matvec_q4(*args, w.mins, threads=N * 32, group_size=TG)
        else:                                              # batched decode: weights read once for all M rows
            g = -(-N // 2) * 32                            # 2 output rows per SIMD group
            if w.scheme == "int8":
                self.lib.matvec_q8_rows(*args, M, threads=g, group_size=TG)
            else:
                self.lib.matvec_q4_rows(*args, w.mins, M, threads=g, group_size=TG)
        return y

    def _dequant_f32(self, w: QuantTensor, rows: slice) -> torch.Tensor:
        """Rows of a quantized matrix as fp32 [rows, K], one kernel pass."""
        K = w.shape[1]
        n_rows = rows.stop - rows.start
        out = torch.empty(n_rows, K, device=self.device)
        if w.scheme == "int4":
            n = n_rows * K // 8
            self.lib.dequant_q4_f32(out, w.data[rows], w.scales[rows], w.mins[rows], K, n, threads=n, group_size=TG)
        else:
            n = n_rows * K // 4
            self.lib.dequant_q8_f32(out, w.data[rows], w.scales[rows], K, n, threads=n, group_size=TG)
        return out

    def linear(self, x, w, b=None, residual=None):
        if isinstance(w, QuantTensor):
            return self._qlinear(x, w, b, residual)
        M = x.shape[0]
        if w.dtype != torch.bfloat16:
            return super().linear(x, w, b, residual)
        if M > KERNEL_ROWS or (M > 1 and w.shape[1] % 4):
            # prefill: fp32 GEMM on weights widened in 32 MB row chunks. Activations stay fp32 at every size, so a
            # prompt's result does not depend on how many other prompts share its packed prefill (MPS's bf16 GEMM
            # would round them), and fp32 GEMM is the faster one on the M2 (2.5 vs 1.4 TFLOPS)
            x, step = x.float(), max(1, DEQUANT_CHUNK // w.shape[1])
            y = torch.empty(M, w.shape[0], device=self.device)
            for r in range(0, w.shape[0], step):
                y[:, r:r + step] = x @ w[r:r + step].float().T
            if b is not None:
                y = y + b.float()
            return y + residual if residual is not None else y
        if M > MAX_BATCH:
            return self._groups(self.linear, x, w, b, residual)
        N, K = w.shape
        y = torch.empty(M, N, device=self.device)
        res = residual.float().contiguous() if residual is not None else self._no_res
        args = (y, w, x.float().contiguous(), b if b is not None else self._no_bias, res,
                K, N, int(b is not None), int(residual is not None))
        if M == 1:
            self.lib.matvec_bf16(*args, threads=N * 32, group_size=TG)
        else:                                              # batched decode: fp32 activations, weights read once
            self.lib.matvec_bf16_rows(*args, M, threads=-(-N // 2) * 32, group_size=TG)
        return y

    def gemm(self, x, w):
        """Tiled-GEMM kernel for prefill (kept for study/benchmarks; linear() uses PyTorch's tuned GEMM)."""
        x = x.float().contiguous()
        T, K = x.shape
        N = w.shape[0]
        y = torch.empty(T, N, device=self.device)
        gx, gy = -(-N // 16), -(-T // 16)
        self.lib.gemm_bf16(y, x, w, T, N, K, threads=(gx * 16, gy * 16), group_size=(16, 16))
        return y

    def swiglu(self, x, w_gate_up):
        if isinstance(w_gate_up, QuantTensor):
            y = self._qlinear(x, w_gate_up)
            F = w_gate_up.shape[0] // 2
            return self.silu_mul(y[:, :F], y[:, F:])
        if x.shape[0] != 1 or w_gate_up.dtype != torch.bfloat16:
            y = self.linear(x, w_gate_up)                  # batched kernel for small M, GEMM for prefill
            F = w_gate_up.shape[0] // 2
            return self.silu_mul(y[:, :F], y[:, F:])
        F, K = w_gate_up.shape[0] // 2, w_gate_up.shape[1]
        out = torch.empty(1, F, device=self.device)
        self.lib.matvec_swiglu_bf16(out, w_gate_up, x.float().contiguous(), K, F, threads=F * 32, group_size=TG)
        return out

    def rms_norm(self, x, w, eps):
        x = x.float().contiguous()
        out = torch.empty_like(x)
        d = x.shape[-1]
        rows = x.numel() // d
        kernel = self.lib.rms_norm if w.dtype == torch.bfloat16 else self.lib.rms_norm_f32w
        kernel(out, x, w.to(self.device).contiguous(), float(eps), d, threads=rows * TG, group_size=TG)
        return out

    def layer_norm(self, x, w, eps):
        x = x.float().contiguous()
        out = torch.empty_like(x)
        d = x.shape[-1]
        rows = x.numel() // d
        self.lib.layer_norm(out, x, w.to(self.device, torch.float32).contiguous(), float(eps), d,
                            threads=rows * TG, group_size=TG)
        return out

    def rope(self, x, positions, theta):
        x = x.float().contiguous()
        T, H, d = x.shape
        out = torch.empty_like(x)
        n = T * H * (d // 2)
        self.lib.rope(out, x, positions.to(device=self.device, dtype=torch.int32), float(theta), H, d, n,
                      threads=n, group_size=min(TG, n))
        return out

    def attention(self, q, k, v, causal=True):
        if q.shape[0] != 1:
            return super().attention(q, k, v, causal)       # prefill: many queries, masked
        _, Hq, d = q.shape
        S, Hkv, _ = k.shape
        out = torch.empty(1, Hq, d, device=self.device)
        scores = torch.empty(Hq, S, device=self.device)
        self.lib.attention_decode(out, q.float().contiguous(), k.float().contiguous(), v.float().contiguous(),
                                  scores, S, Hkv, Hq // Hkv, d, threads=Hq * TG, group_size=TG)
        return out

    def paged_attention(self, q, k_pool, v_pool, tables, lens, block_size):
        """One dispatch for the whole batch: every (head, sequence) reads its keys/values in place via its block
        table (no per-sequence gather)."""
        B, Hq, d = q.shape
        Hkv = k_pool.shape[2]
        max_len = tables.shape[1] * block_size
        out = torch.empty(B, Hq, d, device=self.device)
        scores = torch.empty(B, Hq, max_len, device=self.device)
        self.lib.paged_attention_decode(out, q.float().contiguous(), k_pool, v_pool, scores, tables, lens,
                                        tables.shape[1], max_len, block_size, Hkv, Hq // Hkv, d,
                                        threads=(Hq * TG, B), group_size=(TG, 1))
        return out

    def silu_mul(self, gate, up):
        gate, up = gate.float().contiguous(), up.float().contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        self.lib.silu_mul(out, gate, up, n, threads=n, group_size=min(TG, n))
        return out

    def deltanet_decode(self, qkv, z, b, a, state, slot, conv_w, A_log, dt_bias, norm_w, eps, dims):
        """Fused Metal path: conv_step + gdn_decode (2 dispatches instead of ~20 tensor ops)."""
        return self.deltanet_decode_batch(qkv, z, b, a, [state], slot, conv_w, A_log, dt_bias, norm_w, eps, dims)

    def deltanet_decode_batch(self, qkv, z, b, a, states, slot, conv_w, A_log, dt_bias, norm_w, eps, dims):
        """All B sequences in 2 dispatches when their states share one HybridPool (the server's case); a lone
        standalone state is treated as a pool of one. Otherwise, one sequence at a time."""
        pool = getattr(states[0], "pool", None)
        if len(states) == 1 and pool is None:
            S, conv, n_linear, seqs = states[0].S, states[0].conv_tail, states[0].S.shape[0], self._seq0
        elif pool is not None and all(getattr(st, "pool", None) is pool for st in states):
            S, conv, n_linear = pool.S, pool.conv, pool.S.shape[1]
            seqs = torch.tensor([st.seq for st in states], dtype=torch.int32).to(self.device, non_blocking=True)
        else:
            return super().deltanet_decode_batch(qkv, z, b, a, states, slot, conv_w, A_log, dt_bias, norm_w, eps, dims)
        H, dk, dv, key_dim = dims
        B, C = qkv.shape
        u = torch.empty(B, C, device=self.device)
        self.lib.conv_step(u, qkv.float().contiguous(), conv, conv_w.contiguous(), C, slot, n_linear, seqs,
                           threads=(C, B), group_size=(TG, 1))
        out = torch.empty(B, H * dv, device=self.device)
        self.lib.gdn_decode(out, u, z.float().contiguous(), b.float().contiguous(), a.float().contiguous(),
                            A_log, dt_bias, norm_w, S, H, dk, slot, float(eps), n_linear, seqs,
                            threads=(H * dv, B), group_size=(dv, 1))
        return out
