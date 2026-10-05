"""Cohere2 (Tiny Aya): every layer is attention, on a pluggable backend.

Follows the verified spec of HF transformers' modeling_cohere2.py (docs/tiny-aya-plan.md). What differs from Qwen3.5:
- no DeltaNet: all 36 layers are attention. Three sliding-window layers (with RoPE) are followed by one full layer
  with no positional encoding at all, repeating;
- one LayerNorm per layer (mean subtracted, plain weight, no bias) feeds attention AND the MLP, in parallel:
  h + attn(LN(h)) + mlp(LN(h));
- RoPE rotates neighbouring numbers (2i, 2i+1) ("interleaved", GPT-J) instead of (i, i + d/2), at theta 50000, over
  all head dims;
- logits are multiplied by logit_scale; no QK-norm, no output gate, no biases; the head is tied to the embedding.

Interleaved RoPE without a new kernel: within each head, the rows of W_q and W_k are reordered at load time so that
numbers 2i and 2i+1 land at i and i + d/2. The half-split rotation then turns exactly the pairs HF turns, at the
same frequencies, and q.k does not change when q and k get the same reordering (cached keys stay reordered).

The sliding window itself is not implemented yet. Below sliding_window positions it excludes no key, so a forward
whose positions stay below it is exact; anything longer is refused rather than silently wrong.

stream=True (the fp32 CPU reference of the real model, which would need 13.4 GB if every widened weight were kept):
each weight is widened when used and dropped after, the embedding rows are read from the stored table, and the tied
head is widened HEAD_ROWS rows at a time. Slower, the same numbers; peak memory ~1 layer instead of the model.

kv_dtype: what the KV cache stores. fp32 (the default) keeps K/V exactly as computed, as HF does with fp32
activations; bf16 (Tiny Aya served on Metal: 72 KiB per token instead of 144) rounds them as they are stored, and every
read widens them back to fp32, so all arithmetic stays fp32. The rounding is a measured deviation (tests/test_aya_quant.py).
"""
import torch

from backend.torch_ref import TorchBackend
from config import Cohere2Config
from models.packing import Segment, advance_all, pack, reserve_all
from quant import concat_rows, select_rows
from state import ContiguousKVCache, HybridPool, HybridState, PagedKVPool, PagedSequence

P = "model."
HEAD_ROWS = 32768                 # stream=True: rows of the tied head widened at a time (256 MB at hidden 2048)


def interleaved_to_half(n_heads: int, d: int) -> torch.Tensor:
    """Row order that moves each head's RoPE pair (2i, 2i+1) to (i, i + d/2): evens first, then odds, per head."""
    per_head = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    return (torch.arange(n_heads)[:, None] * d + per_head[None, :]).flatten()


class Cohere2Model:
    def __init__(self, config: Cohere2Config, weights, backend=None, stream: bool = False,
                 kv_dtype: torch.dtype = torch.float32):
        if kv_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"kv_dtype must be torch.float32 or torch.bfloat16, not {kv_dtype}")
        self.config = config
        self.kv_dtype = kv_dtype
        self.weights = weights
        self.b = backend or TorchBackend()
        self._cache: dict[str, torch.Tensor] = {}
        self.stream = stream
        if stream and (self.b.device.type != "cpu" or getattr(self.b, "weight_dtype", None) != torch.float32):
            raise ValueError("stream=True is the fp32 CPU reference mode")
        names = getattr(weights, "tensor_names", None)
        if names is not None and "lm_head.weight" in names():         # the head is the embedding (tied): a separate
            raise ValueError("this checkpoint has a separate lm_head.weight; Cohere2Model ties the head")  # one is refused

    # ---------------------------------------------------------------- weights
    def _w(self, name: str):
        """A 2-D weight in the backend's resident format (bf16 / quantized / fp32)."""
        if self.stream:
            return self.b.prepare(self.weights.get(P + name), name)
        if name not in self._cache:
            self._cache[name] = self.b.prepare(self.weights.get(P + name), name)
        return self._cache[name]

    def _fused(self, key: str, build):
        if self.stream:
            return self.b.prepare(build(), key)
        if key not in self._cache:
            self._cache[key] = self.b.prepare(build(), key)
        return self._cache[key]

    def _vec(self, name: str) -> torch.Tensor:
        """A norm weight, kept in fp32 on the device."""
        if name not in self._cache:
            self._cache[name] = self.weights.get(P + name).float().to(self.b.device)
        return self._cache[name]

    # ---------------------------------------------------------------- state
    # Every layer keeps a KV cache and nothing else: a HybridState with no linear (DeltaNet) layers, so forks, the
    # paged pool and the scheduler work unchanged.
    def new_state(self, max_len: int, kv=None) -> HybridState:
        c = self.config
        if kv is None:
            kv = ContiguousKVCache(c.num_hidden_layers, c.num_key_value_heads, c.head_dim, max_len,
                                   device=self.b.device, dtype=self.kv_dtype)
        return HybridState(kv, device=self.b.device)

    def new_paged_pool(self, num_blocks: int, block_size: int = 16, max_seqs: int = 8) -> HybridPool:
        c = self.config
        kv = PagedKVPool(c.num_hidden_layers, c.num_key_value_heads, c.head_dim, num_blocks, block_size,
                         device=self.b.device, dtype=self.kv_dtype)
        return HybridPool(kv, max_seqs, 0, 0, 0, 0, 0, 1, device=self.b.device)

    def new_paged_state(self, pool: HybridPool) -> HybridState:
        return pool.new_sequence()

    @property
    def max_positions(self) -> int:
        """The longest sequence this implementation computes exactly (the sliding window, until it is implemented)."""
        return self.config.sliding_window

    def _check_window(self, end: int) -> None:
        if end > self.config.sliding_window:
            raise ValueError(f"{end} positions: past the {self.config.sliding_window}-token sliding window, which "
                             "is not implemented yet")

    # ---------------------------------------------------------------- layers
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.weights.get(P + "embed_tokens.weight") if self.stream else self._w("embed_tokens.weight")
        return self.b.embedding(table, ids.to(self.b.device, non_blocking=True))

    def _qkv(self, i: int):
        """Fused [q ; k ; v] projection; on RoPE (sliding) layers, q and k rows reordered for interleaved RoPE."""
        c, p = self.config, f"layers.{i}.self_attn."

        def build():
            q, k, v = (self.weights.get(P + p + f"{n}_proj.weight") for n in "qkv")
            if c.is_sliding(i):
                q = select_rows(q, interleaved_to_half(c.num_attention_heads, c.head_dim))
                k = select_rows(k, interleaved_to_half(c.num_key_value_heads, c.head_dim))
            return concat_rows([q, k, v])
        return self._fused(p + "qkv.weight", build)

    def _project(self, i: int, x, positions):
        """x [N, hidden] -> q [N, Hq, d], k and v [N, Hkv, d], RoPE applied on sliding layers only."""
        c, b = self.config, self.b
        N, Hq, Hkv, d = x.shape[0], c.num_attention_heads, c.num_key_value_heads, c.head_dim
        y = b.linear(x, self._qkv(i))
        q = y[:, : Hq * d].reshape(N, Hq, d)
        k = y[:, Hq * d:(Hq + Hkv) * d].reshape(N, Hkv, d)
        v = y[:, (Hq + Hkv) * d:].reshape(N, Hkv, d)
        if c.is_sliding(i):                                                   # full layers: no positional encoding
            qk = b.rope(torch.cat([q, k], dim=1).contiguous(), positions, c.rope_theta)
            q, k = qk[:, :Hq], qk[:, Hq:]
        return q, k, v

    def attention(self, i: int, x, positions, segs: list[Segment], residual):
        """x: [N, hidden], the tokens of every segment. Projections run once over all N; attention per segment."""
        q, k, v = self._project(i, x, positions)
        outs = []
        for sg in segs:                                            # each sequence attends to its own history only
            qs, ks, vs = q[sg.lo:sg.hi], k[sg.lo:sg.hi], v[sg.lo:sg.hi]
            if sg.state is not None:
                sg.state.kv.write(i, sg.start, ks, vs)
                ks, vs = sg.state.kv.read(i, sg.start + sg.hi - sg.lo)
            outs.append(self.b.attention(qs.contiguous(), ks, vs, causal=True))          # scale 1/sqrt(head_dim)
        o = (torch.cat(outs) if len(outs) > 1 else outs[0]).reshape(x.shape[0], -1)
        return self.b.linear(o, self._w(f"layers.{i}.self_attn.o_proj.weight"), residual=residual)

    def mlp(self, i: int, x, residual):
        b, p = self.b, f"layers.{i}.mlp."
        w = self._fused(p + "gate_up.weight", lambda: concat_rows(
            [self.weights.get(P + p + "gate_proj.weight"), self.weights.get(P + p + "up_proj.weight")]))
        return b.linear(b.swiglu(x, w), self._w(p + "down_proj.weight"), residual=residual)

    def block(self, i: int, h, positions, segs: list[Segment]):
        x = self.b.layer_norm(h, self._vec(f"layers.{i}.input_layernorm.weight"), self.config.layer_norm_eps)
        a = self.attention(i, x, positions, segs, residual=h)                 # h + attention
        return self.mlp(i, x, residual=a)                                     # (h + attention) + mlp, as in HF

    def _layers(self, ids, positions, segs: list[Segment], capture: dict | None = None):
        """Embedding and all layers over the packed tokens -> hidden states [N, hidden] (before the final norm)."""
        for sg in segs:
            self._check_window(sg.start + sg.hi - sg.lo)
        reserve_all(segs)                            # fail before any layer mutates state (atomic forward)
        h = self.embed(ids)
        if capture is not None:
            capture["embed"] = h
        for i in range(self.config.num_hidden_layers):
            h = self.block(i, h, positions, segs)
            if capture is not None:
                capture[f"l{i}_out"] = h
        advance_all(segs)
        return h

    def _final_norm(self, h):
        return self.b.layer_norm(h, self._vec("norm.weight"), self.config.layer_norm_eps)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden states [N, hidden] -> logits [N, vocab]: the tied head, times logit_scale."""
        if self.stream:
            table = self.weights.get(P + "embed_tokens.weight")
            logits = torch.cat([self.b.linear(h, self.b.prepare(table[r:r + HEAD_ROWS]))
                                for r in range(0, table.shape[0], HEAD_ROWS)], dim=1)
        else:
            logits = self.b.linear(h, self._w("embed_tokens.weight"))
        return logits * self.config.logit_scale if self.config.logit_scale != 1.0 else logits

    def forward(self, ids: torch.Tensor, state: HybridState | None = None, last_only: bool = False,
                capture: dict | None = None) -> torch.Tensor:
        """Token ids [T] -> logits [T, vocab]. With a state, continues from state.length and updates it."""
        start = state.length if state is not None else 0
        positions = torch.arange(start, start + len(ids), device=self.b.device, dtype=torch.int32)
        h = self._layers(ids, positions, [Segment(0, len(ids), state, start)], capture)
        h = self._final_norm(h[-1:] if last_only else h)
        if capture is not None:
            capture["final_norm"] = h
        return self.head(h)

    def forward_packed(self, chunks: list[tuple[torch.Tensor, HybridState]]) -> torch.Tensor:
        """Several sequences' chunks (each continuing its own state) in one pass -> logits of each chunk's last
        token [len(chunks), vocab]. Same result as calling forward on each chunk, with each weight read once."""
        ids, positions, segs = pack(chunks, self.b.device)
        h = self._layers(ids, positions, segs)
        last = torch.tensor([sg.hi - 1 for sg in segs]).to(self.b.device, non_blocking=True)
        return self.head(self._final_norm(h[last]))

    def packed_hidden(self, chunks: list[tuple[torch.Tensor, HybridState]],
                      capture: dict | None = None) -> torch.Tensor:
        """Like forward_packed, but -> the final-norm hidden state of EVERY token [N, hidden], rows in chunk order.
        capture gets the packed embedding, every layer's output and the final norm (rows in chunk order too)."""
        ids, positions, segs = pack(chunks, self.b.device)
        h = self._final_norm(self._layers(ids, positions, segs, capture))
        if capture is not None:
            capture["final_norm"] = h
        return h

    def decode_batch(self, tokens: list[int], states: list) -> torch.Tensor:
        """One decode step for B independent sequences -> logits [B, vocab]. Projections and MLPs run batched (each
        weight read once per step for everyone); attention reads each sequence's own KV history."""
        c, b = self.config, self.b
        B = len(tokens)
        for st in states:
            self._check_window(st.length + 1)
        for st in states:
            st.reserve(st.length + 1)
        starts = [st.length for st in states]
        positions = torch.tensor(starts, device=b.device, dtype=torch.int32)
        kvs = [st.kv for st in states]
        paged = all(isinstance(kv, PagedSequence) and kv.pool is kvs[0].pool for kv in kvs)
        if paged:                                       # one KV write and one attention dispatch per layer for all
            kv_pool = kvs[0].pool
            blocks, offsets, tables, lens = kv_pool.batch_layout(kvs, starts)
        h = self.embed(torch.tensor(tokens))
        for i in range(c.num_hidden_layers):
            x = b.layer_norm(h, self._vec(f"layers.{i}.input_layernorm.weight"), c.layer_norm_eps)
            q, k, v = self._project(i, x, positions)
            if paged:
                kv_pool.write_batch(i, blocks, offsets, k, v)
                o = b.paged_attention(q.contiguous(), kv_pool.k[i], kv_pool.v[i], tables, lens, kv_pool.block_size)
            else:
                outs = []
                for j, st in enumerate(states):
                    st.kv.write(i, starts[j], k[j:j + 1], v[j:j + 1])
                    K, V = st.kv.read(i, starts[j] + 1)
                    outs.append(b.attention(q[j:j + 1].contiguous(), K, V, causal=True))
                o = torch.cat(outs)
            a = b.linear(o.reshape(B, -1), self._w(f"layers.{i}.self_attn.o_proj.weight"), residual=h)
            h = self.mlp(i, x, residual=a)
        for st in states:
            st.advance(1)
        return self.head(self._final_norm(h))
