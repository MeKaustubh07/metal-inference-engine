"""Per-sequence decode state: the KV cache.

Keys and values of past tokens never change, so each layer stores them once and every later
step reads them back instead of recomputing the whole prefix (O(n^3) total work -> O(n^2)).
"""
import math

import torch


def kv_buffers(shape: tuple, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys and values as the two halves of ONE zeroed buffer [2, *shape]. PyTorch's MPS allocator puts a request of
    10-512 MiB that fits no free space in a new 1 GiB heap (later requests share it), and gives one of 512 MiB or more
    a heap of its own size: K and V of 432 MiB each took 1,024 MiB, one 864 MiB buffer took 864. So a cache of 512 MiB
    or more costs what it holds (a smaller one shares a 1 GiB heap either way). Both halves are contiguous views."""
    kv = torch.zeros(2, *shape, device=device, dtype=dtype)
    return kv[0], kv[1]


def to_device(t: torch.Tensor, device) -> torch.Tensor:
    """A small host tensor (token ids, positions, block tables) on `device`, without waiting for the GPU. On MPS a
    non_blocking copy reads the host memory when the GPU reaches it, not when it is issued. PyTorch 2.14 keeps the
    source's storage alive until then (Copy.mm, buffer_with_offset_from_tensor), so a freed temporary is safe, but a
    caller that writes to its tensor after the call changes what the GPU reads (tests/test_cohere2.py 11). So the copy
    is made from a private clone. A blocking copy is safe too, but waits for every queued kernel: ~0.3 ms at the start
    of a step, a layer's work in the middle of one."""
    if torch.device(device).type != "mps":
        return t.to(device)
    return t.clone().to(device, non_blocking=True)


class ContiguousKVCache:
    """One preallocated buffer per layer: k/v [max_len, n_kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_len: int,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.max_len = max_len
        self.k, self.v = kv_buffers((n_layers, max_len, n_kv_heads, head_dim), device, dtype)
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
    """Hands out fixed-size blocks from a free list, like an OS page allocator. It knows which blocks are out: freeing
    one twice, or one it never handed out, raises instead of letting two sequences share a block."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks - 1, -1, -1))   # pop() returns block 0 first
        self.in_use: set[int] = set()

    def allocate(self) -> int:
        if not self.free_list:
            raise OutOfBlocks(f"all {self.num_blocks} KV blocks are in use")
        b = self.free_list.pop()
        self.in_use.add(b)
        return b

    def free(self, blocks: list[int]) -> None:
        bad = [b for b in blocks if b not in self.in_use]
        if bad or len(set(blocks)) != len(blocks):
            raise ValueError(f"freeing KV blocks that are not in use (or twice): {sorted(set(bad))[:8]}")
        self.in_use.difference_update(blocks)
        self.free_list.extend(reversed(blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_list)


class PagedKVPool:
    """One big pool of KV blocks shared by every sequence: k/v [layers, num_blocks, block_size, kv_heads, head_dim]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, num_blocks: int, block_size: int = 16,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.block_size = block_size
        self.k, self.v = kv_buffers((n_layers, num_blocks, block_size, n_kv_heads, head_dim), device, dtype)
        self.allocator = BlockAllocator(num_blocks)
        self.unit_bytes = 2 * self.k[:, 0].numel() * self.k.element_size()   # one block of every layer, K and V

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
        return tuple(to_device(t, dev) for t in (blocks, offsets, tables, lens))

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


class GroupedKVPool:
    """The paged KV pool for a model whose layers look back different distances (Tiny Aya: 9 full-attention layers
    and 27 with a 4096-token sliding window), so the sliding layers' memory stops growing past the window.

    The layers are split into groups of G = gcd(n_full, n_sliding) layers (Tiny Aya: 1 full group and 3 sliding
    groups of 9); the pool's unit is one block_size-token block of one group's G layers, k/v [G, units, bs, Hkv, d],
    and one allocator hands units to every group. num_blocks keeps its meaning (blocks of every layer):
    units = num_blocks x groups. A sequence's full groups get a growing table of units; each sliding group a ring of
    at most R units, logical block b in ring slot b % R, which is safe for every forward of at most max_chunk tokens:
    R = ceil((W + max_chunk + bs - 2) / bs) holds every block that forward reads or writes. What a sequence holds only
    grows (blocks_for), so reserving a prompt's units up front still means a prefill can never run out halfway.
    poison=True (tests): a ring slot taking a new block is filled with NaN first, so a stale read shows."""

    def __init__(self, layer_windows: list[int | None], n_kv_heads: int, head_dim: int, num_blocks: int,
                 block_size: int = 16, max_chunk: int = 512, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32, poison: bool = False):
        full = [i for i, w in enumerate(layer_windows) if w is None]
        sliding = [i for i, w in enumerate(layer_windows) if w is not None]
        if len({w for w in layer_windows if w is not None}) > 1:
            raise ValueError("one sliding window size per model")
        self.window = layer_windows[sliding[0]] if sliding else None
        G = math.gcd(len(full), len(sliding))
        self.groups = [full[i:i + G] for i in range(0, len(full), G)] + \
            [sliding[i:i + G] for i in range(0, len(sliding), G)]
        self.sliding = [False] * (len(full) // G) + [True] * (len(sliding) // G)
        self.layer_map = {layer: (g, slot) for g, layers in enumerate(self.groups) for slot, layer in enumerate(layers)}
        self.n_groups, self.block_size, self.poison = len(self.groups), block_size, poison
        self.ring = -(-(self.window + max_chunk + block_size - 2) // block_size) if self.window else None
        self.k, self.v = kv_buffers((G, num_blocks * self.n_groups, block_size, n_kv_heads, head_dim), device, dtype)
        self.allocator = BlockAllocator(num_blocks * self.n_groups)
        self.unit_bytes = 2 * self.k[:, 0].numel() * self.k.element_size()
        self.max_step_units = self.n_groups            # one decode step adds at most one unit per group
        self.pad_block = 0                             # table entries that are never read (before a window)

    def blocks_for(self, n: int) -> int:
        """Units a sequence of n positions holds: every block for a full group, at most R for a sliding one."""
        nb = -(-n // self.block_size)
        return sum(min(self.ring, nb) if sl else nb for sl in self.sliding)

    def new_sequence(self) -> "WindowedSequence":
        return WindowedSequence(self)

    def batch_layout(self, seqs: list["WindowedSequence"], starts: list[int]):
        """For a batched decode step (one new token per sequence at position starts[i], units reserved): per group,
        the unit each new token goes to and each sequence's table by absolute block (a sliding group's entries before
        the window point at pad_block and are never read); offsets and lengths are shared. Device tensors:
        blocks [groups, B], offsets [B], tables [groups, B, max_nb], lens [B]."""
        bs, dev, W = self.block_size, self.k.device, self.window
        width = max(-(-(p + 1) // bs) for p in starts)
        blocks = [[s.unit(g, p // bs, write=True) for s, p in zip(seqs, starts)] for g in range(self.n_groups)]
        tables = []
        for g, sl in enumerate(self.sliding):
            rows = []
            for s, p in zip(seqs, starts):
                first = max(0, p + 1 - W) // bs if sl else 0
                row = [self.pad_block] * first + [s.unit(g, b) for b in range(first, -(-(p + 1) // bs))]
                rows.append(row + [self.pad_block] * (width - len(row)))
            tables.append(rows)
        out = (torch.tensor(blocks), torch.tensor([p % bs for p in starts]), torch.tensor(tables, dtype=torch.int32),
               torch.tensor([p + 1 for p in starts], dtype=torch.int32))
        return tuple(to_device(t, dev) for t in out)

    def write_batch(self, slot: int, blocks: torch.Tensor, offsets: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """k, v: [B, kv_heads, head_dim], one new token per sequence, for the layer at `slot` of one group."""
        self.k[slot, blocks, offsets] = k.to(self.k.dtype)
        self.v[slot, blocks, offsets] = v.to(self.v.dtype)


class WindowedSequence:
    """One sequence's view of a GroupedKVPool: a table of units per full group, a ring of units per sliding group,
    each ring slot tagged with the logical block it holds. Same interface as PagedSequence (write / read(lo) /
    reserve / advance / free); block_table is the first group's table (a full group's: one unit per block)."""

    def __init__(self, pool: GroupedKVPool):
        self.pool = pool
        self.tables: list[list[int]] = [[] for _ in range(pool.n_groups)]
        self.tags: list[list[int]] = [[] for _ in range(pool.n_groups)]   # sliding: logical block in each slot
        self.length = 0

    @property
    def block_table(self) -> list[int]:
        return self.tables[0]

    def check_forward(self, start: int, end: int) -> None:
        """Refuse, before anything moves, a forward whose blocks (the first query's window to the last token) would
        not fit in the ring: only a chunk longer than the pool's max_chunk can."""
        p = self.pool
        if p.ring is not None and -(-end // p.block_size) - max(0, start - p.window + 1) // p.block_size > p.ring:
            raise ValueError(f"a forward from position {start} to {end} spans more than the {p.ring}-block ring "
                             "(chunks of at most the pool's max_chunk tokens)")

    def reserve(self, end: int) -> None:
        """Take every unit positions up to `end` need, all or nothing (as PagedSequence.reserve)."""
        p = self.pool
        need = p.blocks_for(end) - sum(map(len, self.tables))
        if need <= 0:
            return
        if need > p.allocator.num_free:
            raise OutOfBlocks(f"need {need} more KV units, only {p.allocator.num_free} free")
        nb = -(-end // p.block_size)
        for g, sl in enumerate(p.sliding):
            for _ in range((min(p.ring, nb) if sl else nb) - len(self.tables[g])):
                self.tables[g].append(p.allocator.allocate())
                self.tags[g].append(-1)

    def unit(self, g: int, b: int, write: bool = False) -> int:
        """The unit holding logical block b of group g. A sliding group's ring slot is retagged when written; a read
        of a block the ring no longer holds raises instead of returning another block's keys."""
        p = self.pool
        if not p.sliding[g]:
            return self.tables[g][b]
        i = b % p.ring
        if write and self.tags[g][i] != b:
            if p.poison:
                p.k[:, self.tables[g][i]] = float("nan")
                p.v[:, self.tables[g][i]] = float("nan")
            self.tags[g][i] = b
        elif self.tags[g][i] != b:
            raise RuntimeError(f"block {b} of group {g} is no longer in the ring (slot {i} holds {self.tags[g][i]})")
        return self.tables[g][i]

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[0]
        self.reserve(end)
        g, slot = self.pool.layer_map[layer]
        bs, dev = self.pool.block_size, self.pool.k.device
        units = torch.tensor([self.unit(g, q // bs, write=True) for q in range(start, end)], device=dev)
        offsets = torch.arange(start, end, device=dev) % bs
        self.pool.k[slot, units, offsets] = k.to(self.pool.k.dtype)
        self.pool.v[slot, units, offsets] = v.to(self.pool.v.dtype)

    def read(self, layer: int, end: int, lo: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions lo .. end-1 of one layer, gathered into [end - lo, kv_heads, head_dim]."""
        g, slot = self.pool.layer_map[layer]
        bs = self.pool.block_size
        b0, nb = lo // bs, -(-end // bs)
        units = torch.tensor([self.unit(g, b) for b in range(b0, nb)], device=self.pool.k.device)
        rows = slice(lo - b0 * bs, end - b0 * bs)
        return self.pool.k[slot, units].flatten(0, 1)[rows], self.pool.v[slot, units].flatten(0, 1)[rows]

    def advance(self, n: int) -> None:
        self.length += n

    def free(self) -> None:
        self.pool.allocator.free([u for t in self.tables for u in t])
        self.tables = [[] for _ in range(self.pool.n_groups)]
        self.tags = [[] for _ in range(self.pool.n_groups)]
        self.length = 0

    def bytes_used(self) -> int:
        return sum(map(len, self.tables)) * self.pool.unit_bytes


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
        self._seq_index: tuple[tuple, torch.Tensor | None] = ((), None)

    @property
    def allocator(self) -> BlockAllocator:                   # KV block accounting, as on PagedKVPool
        return self.kv.allocator

    @property
    def block_size(self) -> int:
        return self.kv.block_size

    def blocks_for(self, n: int) -> int:
        """Pool units a sequence of n positions holds (a GroupedKVPool's own count; else one block per bs tokens)."""
        return self.kv.blocks_for(n) if hasattr(self.kv, "blocks_for") else -(-n // self.kv.block_size)

    @property
    def max_step_units(self) -> int:
        """Units one decode step can add to one sequence (one per group)."""
        return getattr(self.kv, "max_step_units", 1)

    @property
    def unit_bytes(self) -> int:
        return self.kv.unit_bytes

    def seq_index(self, states: list) -> torch.Tensor:
        """The states' slots as a device tensor, for the batched DeltaNet kernels: one copy per batch composition,
        not one per layer."""
        key = tuple(st.seq for st in states)
        if self._seq_index[0] != key:
            self._seq_index = (key, to_device(torch.tensor(key, dtype=torch.int32), self.S.device))
        return self._seq_index[1]

    def new_sequence(self) -> HybridState:
        if not self.free_seqs:
            raise OutOfBlocks(f"all {self.S.shape[0]} sequence state slots are in use")
        return HybridState(self.kv.new_sequence(), pool=self, seq=self.free_seqs.pop())

    def release(self, seq: int) -> None:
        self.S[seq].zero_()
        self.conv[seq].zero_()
        self.free_seqs.append(seq)
