"""Per-sequence decode state: the KV cache.

Keys and values of past tokens never change, so each layer stores them once and every later
step reads them back instead of recomputing the whole prefix (O(n^3) total work -> O(n^2)).
"""
import torch


class ContiguousKVCache:
    """One preallocated buffer per layer: k/v [max_len, n_kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_len: int,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.max_len = max_len
        self.k = torch.zeros(n_layers, max_len, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0                                   # tokens committed so far

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store k/v [T, Hkv, d] for positions start .. start+T-1."""
        end = start + k.shape[0]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, capacity {self.max_len}")
        self.k[layer, start:end] = k.to(self.k.dtype)
        self.v[layer, start:end] = v.to(self.v.dtype)

    def read(self, layer: int, end: int, lo: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys/values for positions lo .. end-1 (views, no copy). lo > 0: a sliding-window layer's suffix."""
        return self.k[layer, lo:end], self.v[layer, lo:end]

    def advance(self, n: int) -> None:
        self.length += n

    def reserve(self, end: int) -> None:
        """Fail BEFORE any layer runs if positions up to `end` don't fit."""
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, capacity {self.max_len}")

    def bytes_used(self) -> int:
        return 2 * self.k[:, : self.length].numel() * self.k.element_size()

    def fork(self) -> "ContiguousKVCache":
        """An independent copy holding the same positions 0 .. length-1 (same capacity)."""
        n_layers, max_len, h, d = self.k.shape
        other = ContiguousKVCache(n_layers, h, d, max_len, device=self.k.device, dtype=self.k.dtype)
        other.k[:, : self.length] = self.k[:, : self.length]
        other.v[:, : self.length] = self.v[:, : self.length]
        other.length = self.length
        return other


class OutOfBlocks(RuntimeError):
    """The shared pool has no free block left (the scheduler should queue or evict)."""


class BlockAllocator:
    """Hands out fixed-size blocks from a free list, like an OS page allocator."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks - 1, -1, -1))   # pop() returns block 0 first

    def allocate(self) -> int:
        if not self.free_list:
            raise OutOfBlocks(f"all {self.num_blocks} KV blocks are in use")
        return self.free_list.pop()

    def free(self, blocks: list[int]) -> None:
        self.free_list.extend(reversed(blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_list)


class PagedKVPool:
    """One big pool of KV blocks shared by every sequence: k/v [layers, num_blocks, block_size, kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, num_blocks: int, block_size: int = 16,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.block_size = block_size
        self.k = torch.zeros(n_layers, num_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.allocator = BlockAllocator(num_blocks)

    def new_sequence(self) -> "PagedSequence":
        return PagedSequence(self)

    def batch_layout(self, seqs: list["PagedSequence"], starts: list[int]):
        """For a batched decode step (one new token per sequence at position starts[i], blocks already reserved):
        where each new token goes (block, offset) and each sequence's block table / length after it, as device
        tensors, built once per step and shared by every layer."""
        bs, dev = self.block_size, self.k.device
        blocks = torch.tensor([s.block_table[p // bs] for s, p in zip(seqs, starts)])
        offsets = torch.tensor([p % bs for p in starts])
        width = max(len(s.block_table) for s in seqs)
        tables = torch.tensor([s.block_table + [0] * (width - len(s.block_table)) for s in seqs], dtype=torch.int32)
        lens = torch.tensor([p + 1 for p in starts], dtype=torch.int32)
        return tuple(t.to(dev, non_blocking=True) for t in (blocks, offsets, tables, lens))

    def write_batch(self, layer: int, blocks: torch.Tensor, offsets: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """k, v: [B, kv_heads, head_dim], one new token per sequence."""
        self.k[layer, blocks, offsets] = k.to(self.k.dtype)
        self.v[layer, blocks, offsets] = v.to(self.v.dtype)


class PagedSequence:
    """One sequence's view of the pool: a block table mapping logical positions to physical blocks.

    Same interface as ContiguousKVCache (write / read / advance / length), so the model can't tell them apart.
    """

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.block_table: list[int] = []      # logical block i -> physical block id
        self.length = 0

    def _ensure(self, end: int) -> None:
        """Grow the block table to cover `end` positions. All-or-nothing: if the pool can't supply every block
        needed, raise before taking any, so a failed request never holds blocks (no hold-and-wait deadlock)."""
        bs = self.pool.block_size
        need = -(-end // bs) - len(self.block_table)
        if need <= 0:
            return
        if need > self.pool.allocator.num_free:
            raise OutOfBlocks(f"need {need} more KV blocks, only {self.pool.allocator.num_free} free")
        self.block_table.extend(self.pool.allocator.allocate() for _ in range(need))

    def reserve(self, end: int) -> None:
        """Allocate every block needed for positions up to `end` before any layer runs (all-or-nothing)."""
        self._ensure(end)

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[0]
        self._ensure(end)
        pos = torch.arange(start, end, device=self.pool.k.device)
        table = torch.tensor(self.block_table, device=self.pool.k.device)
        blocks, offsets = table[pos // self.pool.block_size], pos % self.pool.block_size
        self.pool.k[layer, blocks, offsets] = k.to(self.pool.k.dtype)
        self.pool.v[layer, blocks, offsets] = v.to(self.pool.v.dtype)

    def read(self, layer: int, end: int, lo: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather positions lo .. end-1 from their blocks into contiguous [end - lo, kv_heads, head_dim]. Only the
        blocks holding them are touched (lo > 0: a sliding-window layer, whose older blocks may be gone)."""
        bs = self.pool.block_size
        b0, nb = lo // bs, -(-end // bs)                      # first block read, ceil(end / block_size)
        table = torch.tensor(self.block_table[b0:nb], device=self.pool.k.device)
        k = self.pool.k[layer, table].flatten(0, 1)[lo - b0 * bs:end - b0 * bs]   # [blocks*bs, H, d] -> the rows
        v = self.pool.v[layer, table].flatten(0, 1)[lo - b0 * bs:end - b0 * bs]
        return k, v

    def advance(self, n: int) -> None:
        self.length += n

    def free(self) -> None:
        """Return this sequence's blocks to the pool (called when the request finishes)."""
        self.pool.allocator.free(self.block_table)
        self.block_table = []
        self.length = 0

    def bytes_used(self) -> int:
        p = self.pool
        return 2 * len(self.block_table) * p.k[0, 0].numel() * p.k.shape[0] * p.k.element_size()


class HybridState:
    """SequenceState for hybrid models (Qwen3.5): each layer asks for the kind of state it needs.

    attention layers -> a KV cache (ContiguousKVCache or PagedSequence), indexed by attention-layer slot
    linear layers    -> a fixed-size recurrent state S [H, dk, dv] (fp32) and the last K-1 raw conv inputs [K-1, C]
    The linear-layer state never grows with the sequence: that is the point of linear attention.
    Standalone states own their S / conv_tail; states from a HybridPool are views of one pool slot (`seq`).
    """

    def __init__(self, kv, n_linear: int = 0, n_heads: int = 0, dk: int = 0, dv: int = 0, conv_dim: int = 0,
                 conv_k: int = 1, device: torch.device | str = "cpu", pool: "HybridPool | None" = None,
                 seq: int | None = None):
        self.kv, self.pool, self.seq = kv, pool, seq
        if pool is not None:
            self.S, self.conv_tail = pool.S[seq], pool.conv[seq]
        else:
            self.S = torch.zeros(n_linear, n_heads, dk, dv, device=device, dtype=torch.float32)
            self.conv_tail = torch.zeros(n_linear, conv_k - 1, conv_dim, device=device, dtype=torch.float32)

    @property
    def length(self) -> int:
        return self.kv.length

    def advance(self, n: int) -> None:
        self.kv.advance(n)

    def reserve(self, end: int) -> None:
        self.kv.reserve(end)

    def free(self) -> None:
        """Release KV blocks and reset the recurrent/conv state so the object can't leak history into a reuse.
        A pooled state gives its slot back (zeroed) and must not be used afterwards."""
        if hasattr(self.kv, "free"):
            self.kv.free()
        if self.pool is not None:
            self.pool.release(self.seq)
            self.pool = None
            self.S = self.conv_tail = None                    # the slot may now belong to another sequence
        elif self.seq is None:
            self.S.zero_()
            self.conv_tail.zero_()

    def bytes_used(self) -> int:
        fixed = (self.S.numel() + self.conv_tail.numel()) * 4
        return self.kv.bytes_used() + fixed

    def fork(self) -> "HybridState":
        """An independent copy of this sequence so far: its KV positions, recurrent state and conv tail. What is
        fed to the copy never reaches the original or other copies (a recurrence has no attention mask: keeping
        continuations apart means giving each its own state). Standalone states only."""
        if self.pool is not None or not isinstance(self.kv, ContiguousKVCache):
            raise ValueError("fork() needs a standalone state with a contiguous KV cache")
        other = HybridState.__new__(HybridState)
        other.kv, other.pool, other.seq = self.kv.fork(), None, None
        other.S, other.conv_tail = self.S.clone(), self.conv_tail.clone()
        return other


class HybridPool:
    """Storage shared by many Qwen3.5 sequences: the paged KV pool for the attention layers, plus one fixed-size
    recurrent/conv state slot per sequence for the DeltaNet layers. With every sequence's DeltaNet state in one
    tensor, a batched decode updates all of them in one kernel dispatch per layer instead of one per sequence."""

    def __init__(self, kv: PagedKVPool, max_seqs: int, n_linear: int, n_heads: int, dk: int, dv: int,
                 conv_dim: int, conv_k: int, device: torch.device | str = "cpu"):
        self.kv = kv
        self.S = torch.zeros(max_seqs, n_linear, n_heads, dk, dv, device=device, dtype=torch.float32)
        self.conv = torch.zeros(max_seqs, n_linear, conv_k - 1, conv_dim, device=device, dtype=torch.float32)
        self.free_seqs = list(range(max_seqs - 1, -1, -1))

    @property
    def allocator(self) -> BlockAllocator:                   # KV block accounting, as on PagedKVPool
        return self.kv.allocator

    @property
    def block_size(self) -> int:
        return self.kv.block_size

    def new_sequence(self) -> HybridState:
        if not self.free_seqs:
            raise OutOfBlocks(f"all {self.S.shape[0]} sequence state slots are in use")
        return HybridState(self.kv.new_sequence(), pool=self, seq=self.free_seqs.pop())

    def release(self, seq: int) -> None:
        self.S[seq].zero_()
        self.conv[seq].zero_()
        self.free_seqs.append(seq)
