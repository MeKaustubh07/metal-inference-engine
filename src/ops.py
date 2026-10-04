"""Hand-written ops on torch primitives. No torch.nn, no torch.nn.functional model ops."""
import math

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: scale each token's vector to unit root-mean-square, then per-channel weight.

    x:      [..., D]   the residual stream (one row per token)
    weight: [D]        learned per-channel volume knobs
    """
    x32 = x.float()                                     # do the math in fp32 for accuracy
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return (x32 / rms) * weight.float()                 # broadcast weight across tokens


def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm without bias (Cohere2 / Tiny Aya): subtract each token's mean, scale to unit variance, then
    per-channel weight. Unlike RMSNorm, the mean is removed first. Same operation order as transformers, in fp32."""
    x32 = x.float()
    centered = x32 - x32.mean(dim=-1, keepdim=True)
    variance = centered.pow(2).mean(dim=-1, keepdim=True)
    return weight.float() * (centered * torch.rsqrt(variance + eps))


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary position embedding: rotate pairs of numbers by a position-dependent angle.

    x:         [T, H, d]  queries or keys, split into H heads of d numbers
    positions: [T]        position of each token in the sequence (0, 1, 2, ...)
    theta:     base frequency from the config (1e7 for Qwen3.5)

    Pairing convention (HF "rotate_half"): number i pairs with number i + d/2.
    """
    d = x.shape[-1]
    half = d // 2
    # One rotation speed per pair: fast for the first pairs, very slow for the last.
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / half))  # [d/2]
    angles = positions.to(x.device).float()[:, None] * freqs[None, :]                          # [T, d/2]
    cos = torch.cos(angles)[:, None, :]                                           # [T, 1, d/2]
    sin = torch.sin(angles)[:, None, :]                                           # broadcast over heads

    x32 = x.float()
    a, b = x32[..., :half], x32[..., half:]         # the two members of every pair
    return torch.cat((a * cos - b * sin,            # standard 2-D rotation of (a, b)
                      a * sin + b * cos), dim=-1)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Turn scores into weights that are positive and sum to 1 along `dim`."""
    x32 = x.float()
    m = x32.max(dim=dim, keepdim=True).values      # subtract the max first: exp() of big numbers overflows
    e = torch.exp(x32 - m)
    return e / e.sum(dim=dim, keepdim=True)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Grouped-query attention.

    q:    [T, Hq, d]   queries for the T new tokens (already rotated by RoPE)
    k, v: [S, Hkv, d]  keys/values for all S tokens seen so far (S == T until the KV cache exists)
    returns [T, Hq, d]
    """
    T, Hq, d = q.shape
    S, Hkv, _ = k.shape
    group = Hq // Hkv                                   # 8 // 2 = 4 query heads share each key/value head
    k = k.float().repeat_interleave(group, dim=1)       # [S, Hq, d]: KV head 0 serves query heads 0-3, head 1 serves 4-7
    v = v.float().repeat_interleave(group, dim=1)

    qh, kh, vh = q.float().transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)   # heads first: [H, tokens, d]
    scores = (qh @ kh.transpose(1, 2)) / math.sqrt(d)   # [Hq, T, S]: every query dotted with every key

    if causal:
        # query i sits at absolute position S - T + i and may only see keys at positions <= that
        future = torch.ones(T, S, dtype=torch.bool, device=q.device).triu(diagonal=S - T + 1)
        scores = scores.masked_fill(future, float("-inf"))   # exp(-inf) = 0 -> zero weight

    weights = softmax(scores, dim=-1)                   # [Hq, T, S]: each row sums to 1
    out = weights @ vh                                  # [Hq, T, d]: weighted mix of values
    return out.transpose(0, 1)                          # back to [T, Hq, d]


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU core: silu(gate) * up, where silu(g) = g * sigmoid(g) = g / (1 + e^-g)."""
    g = gate.float()
    return (g / (1.0 + torch.exp(-g))) * up.float()


# ---------------------------------------------------------------- Qwen3.5 (Gated DeltaNet) ops

def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / ||x|| over the last dim (SUM of squares, fixed eps). Not RMSNorm."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rms_norm_gated(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Gated RMSNorm of the DeltaNet output: normalize x, scale by PLAIN w (not 1 + w), then multiply by silu(z)."""
    x32, z32 = x.float(), z.float()
    y = x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + eps)
    return y * w.float() * (z32 / (1.0 + torch.exp(-z32)))


def causal_conv1d(x: torch.Tensor, tail: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution + SiLU over the time axis.

    x:    [T, C]    new raw inputs (in_proj_qkv output, before the q/k/v split)
    tail: [K-1, C]  the previous K-1 raw inputs (zeros at the start of a sequence)
    w:    [C, K]    per-channel taps; tap K-1 multiplies the current token (cross-correlation, no flip)
    returns (silu(conv) [T, C], new tail [K-1, C])
    """
    K = w.shape[1]
    T = x.shape[0]
    hist = torch.cat([tail.to(x.dtype), x], dim=0)                  # [T + K - 1, C]
    y = sum(hist[j:j + T] * w[:, j].float() for j in range(K))
    return y / (1.0 + torch.exp(-y)), hist[-(K - 1):].clone()


def gated_delta_recurrent(q, k, v, beta, g, S):
    """Gated delta rule, one token at a time (exact; used for decode).

    q, k: [T, H, dk]  (l2-normalized, q already scaled)   v: [T, H, dv]   beta, g: [T, H]   S: [H, dk, dv] fp32
    Per token:  S = exp(g) * S;  S += k (x) (beta * (v - k^T S));  o = q^T S   (reads the UPDATED state)
    returns (o [T, H, dv], S)
    """
    T = q.shape[0]
    out = torch.empty(T, v.shape[1], v.shape[2], dtype=torch.float32, device=q.device)
    for t in range(T):
        S = S * torch.exp(g[t])[:, None, None]
        kv_mem = (S * k[t][:, :, None]).sum(-2)                     # [H, dv] = k^T S
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]                # rank-1 write
        out[t] = (S * q[t][:, :, None]).sum(-2)                     # [H, dv] = q^T S
    return out, S


def gated_delta_chunked(q, k, v, beta, g, S, chunk: int = 64):
    """The same rule computed 64 tokens at a time with matrix products (the WY / UT transform), for prefill.

    Equal to gated_delta_recurrent up to float rounding (~1e-7). Shapes as in gated_delta_recurrent.
    """
    T, H, dk = q.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:                                                        # zero rows: k = beta = g = 0 leave S untouched
        z = lambda t, *rest: torch.cat([t, t.new_zeros(pad, *rest)], 0)
        q, k, v, beta, g = z(q, H, dk), z(k, H, dk), z(v, H, dv), z(beta, H), z(g, H)
    n = q.shape[0] // chunk
    # heads first, then chunks: [H, n, C, d]
    to_hc = lambda t: t.transpose(0, 1).reshape(H, n, chunk, *t.shape[2:])
    q, k, v, beta, g = to_hc(q), to_hc(k), to_hc(v), to_hc(beta), to_hc(g)
    vb, kb = v * beta[..., None], k * beta[..., None]
    G = g.cumsum(-1)                                               # [H, n, C] cumulative log-decay inside a chunk
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device)
    D = (G[..., :, None] - G[..., None, :]).masked_fill(~tri.tril(), float("-inf")).exp()   # decay from j to i (i >= j)
    A = -((kb @ k.transpose(-1, -2)) * D).masked_fill(~tri.tril(-1), 0.0)
    for i in range(1, chunk):                                      # forward substitution: A = (I + A)^-1 - I, row by row
        A[..., i, :i] = A[..., i, :i] + (A[..., i, :, None] * A[..., :, :i]).sum(-2)[..., :i]
    A = A + torch.eye(chunk, device=q.device)
    U = A @ vb                                                     # [H, n, C, dv]
    W = A @ (kb * G.exp()[..., None])                              # [H, n, C, dk]
    out = torch.empty(H, n, chunk, dv, dtype=torch.float32, device=q.device)
    for c in range(n):
        qc, kc, Gc = q[:, c], k[:, c], G[:, c]
        P = (qc @ kc.transpose(-1, -2)) * D[:, c]                  # causal, decayed q.k within the chunk
        vn = U[:, c] - W[:, c] @ S                                 # what the chunk writes, net of the carried state
        out[:, c] = (qc * Gc.exp()[..., None]) @ S + P @ vn
        S = S * Gc[:, -1].exp()[:, None, None] + (kc * (Gc[:, -1:] - Gc).exp()[..., None]).transpose(-1, -2) @ vn
    out = out.reshape(H, n * chunk, dv).transpose(0, 1)[:T]
    return out, S
