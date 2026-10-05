"""Block-wise symmetric weight quantization (INT8 / INT4) and the .qt file format.

Every `block` consecutive weights along a row share fp16 parameters:
    INT8 (symmetric):  scale = absmax / 127, w ≈ q * scale, q in [-127, 127]            (1 byte per weight)
    INT4 (asymmetric): scale = (max - min) / 15, w ≈ q * scale + min, q in [0, 15], two weights per byte:
                       element 2j in the low nibble, 2j+1 in the high nibble    (0.5 byte per weight + 2 fp16/block)
Asymmetric INT4 uses all 16 levels across each block's real range; symmetric INT4 wasted range on the side a
block doesn't use and cost +28% perplexity on Qwen2.5-0.5B (asymmetric: +1%).
Blocks never cross rows, so stacking quantized matrices row-wise (fused QKV, gate+up) is exact.

Mixed precision: with INT4 the tied embedding / output head stays INT8 (it scores every token; llama.cpp keeps it
at higher precision too).
"""
import json
import mmap
import os
import struct
from dataclasses import dataclass

import torch

BLOCK = 32
CHUNK = 1 << 24                 # weights quantized at a time (64 MB per fp32 temporary): the 262k-row tied embedding
                                # of Tiny Aya would need ~6.5 GB of temporaries in one piece


@dataclass
class QuantTensor:
    scheme: str                 # "int8" | "int4"
    data: torch.Tensor          # int8 [N, K]  or  uint8 [N, K/2] (packed)
    scales: torch.Tensor        # fp16 [N, K / BLOCK]
    shape: tuple[int, int]      # logical [N, K]
    block: int = BLOCK
    mins: torch.Tensor | None = None   # fp16 [N, K / BLOCK], INT4 only

    def to(self, device) -> "QuantTensor":
        mins = self.mins.to(device) if self.mins is not None else None
        return QuantTensor(self.scheme, self.data.to(device), self.scales.to(device), self.shape, self.block, mins)

    @property
    def nbytes(self) -> int:
        extra = self.mins.numel() * 2 if self.mins is not None else 0
        return self.data.numel() * self.data.element_size() + self.scales.numel() * 2 + extra

    def dequantize(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        """fp32 weights (optionally only the given rows, e.g. an embedding lookup)."""
        data = self.data if rows is None else self.data[rows]
        scales = self.scales if rows is None else self.scales[rows]
        if self.scheme == "int8":
            q = data.float()
            return (q.view(*q.shape[:-1], -1, self.block) * scales.float()[..., None]).flatten(-2)
        mins = self.mins if rows is None else self.mins[rows]
        lo, hi = (data & 0x0F).float(), (data >> 4).float()
        q = torch.stack([lo, hi], dim=-1).flatten(-2)                          # interleave back to 2j, 2j+1
        q = q.view(*q.shape[:-1], -1, self.block)
        return (q * scales.float()[..., None] + mins.float()[..., None]).flatten(-2)


def quantize(w: torch.Tensor, scheme: str, block: int = BLOCK) -> QuantTensor:
    """Quantize a [N, K] weight, CHUNK weights (whole rows) at a time. Blocks never cross rows, so the result is
    bit-identical to quantizing the whole tensor in one piece."""
    N, K = w.shape
    if K % block:
        raise ValueError(f"K={K} is not a multiple of the block size {block}")
    if scheme not in ("int8", "int4"):
        raise ValueError(f"unknown scheme {scheme}")
    rows = max(1, CHUNK // K)
    if N <= rows:
        return _quantize_rows(w, scheme, block)
    return concat_rows([_quantize_rows(w[r:r + rows], scheme, block) for r in range(0, N, rows)])


def _quantize_rows(w: torch.Tensor, scheme: str, block: int) -> QuantTensor:
    N, K = w.shape
    x = w.float().view(N, K // block, block)
    if scheme == "int8":
        scales16 = (x.abs().amax(-1) / 127).clamp(min=1e-12).to(torch.float16)
        q = torch.round(x / scales16.float()[..., None])              # divide by the STORED scale
        return QuantTensor(scheme, q.clamp(-127, 127).to(torch.int8).view(N, K), scales16, (N, K), block)
    if scheme == "int4":
        lo, hi = x.amin(-1), x.amax(-1)
        mins16 = lo.to(torch.float16)
        scales16 = ((hi - mins16.float()) / 15).clamp(min=1e-12).to(torch.float16)
        q = torch.round((x - mins16.float()[..., None]) / scales16.float()[..., None]).clamp(0, 15)
        q = q.to(torch.uint8).view(N, K)
        data = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()          # [N, K/2]
        return QuantTensor(scheme, data, scales16, (N, K), block, mins16)
    raise ValueError(f"unknown scheme {scheme}")


_GROUPS = [  # checkpoint component / model fused-tensor suffix -> the fused group it belongs to
    (("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
      "self_attn.qkvg.weight", "self_attn.qkv.weight"), "attn_in"),     # Qwen3.5's qkvg, Cohere2's qkv
    (("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.gate_up.weight"), "mlp_in"),
    (("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight",
      "linear_attn.in_proj_a.weight", "linear_attn.in_proj.weight"), "linear_in"),
]


def policy_group(name: str) -> str:
    """Canonical id for policy matching, independent of model family and checkpoint prefix.

    'model.language_model.layers.3.self_attn.k_proj.weight' (checkpoint) and 'layers.3.self_attn.qkvg.weight'
    (fused) both map to 'layers.3.attn_in', so every part of one fused matrix gets the same scheme (row-stacking
    needs that)."""
    i = name.find("layers.")
    base = name[i:] if i >= 0 else name
    for suffixes, group in _GROUPS:
        for suf in suffixes:
            if base.endswith(suf):
                return base[: -len(suf)] + group
    return base


def scheme_for(name: str | None, scheme: str, keep_int8: frozenset = frozenset()) -> str:
    """Mixed-precision policy for INT4: the tied embedding / output head, and any tensor group the calibration
    (scripts/calibrate_quant.py) measured as too sensitive, stay INT8."""
    if scheme == "int4" and name is not None and (
            "embed_tokens" in name or policy_group(name) in {policy_group(k) for k in keep_int8}):
        return "int8"
    return scheme


def load_policy(path: str | None) -> frozenset:
    """Canonical groups that stay INT8 in INT4 mode (from configs/quant/<model>.json); empty if no file."""
    if not path:
        return frozenset()
    return frozenset(policy_group(n) for n in json.load(open(path))["keep_int8"])


def select_rows(w, idx: torch.Tensor):
    """Rows `idx` of a plain or quantized weight (quantization blocks never cross rows)."""
    if isinstance(w, QuantTensor):
        mins = w.mins[idx] if w.mins is not None else None
        return QuantTensor(w.scheme, w.data[idx], w.scales[idx], (len(idx), w.shape[1]), w.block, mins)
    return w[idx]


def concat_rows(parts: list) -> "QuantTensor | torch.Tensor":
    """Row-wise stack of plain tensors or of QuantTensors with the same scheme."""
    if not isinstance(parts[0], QuantTensor):
        return torch.cat(parts, dim=0)
    if len({p.scheme for p in parts}) != 1:
        raise ValueError(f"cannot stack mixed quantization schemes {[p.scheme for p in parts]}")
    mins = torch.cat([p.mins for p in parts]) if parts[0].mins is not None else None
    return QuantTensor(parts[0].scheme, torch.cat([p.data for p in parts]), torch.cat([p.scales for p in parts]),
                       (sum(p.shape[0] for p in parts), parts[0].shape[1]), parts[0].block, mins)


def should_quantize(name: str, t: torch.Tensor) -> bool:
    """Quantize the big 2-D linear weights (and the tied embedding); keep norms and biases as they are."""
    return t.ndim == 2 and t.shape[1] % BLOCK == 0 and name.endswith(".weight") and "norm" not in name


# ---------------- .qt file: [8-byte LE header length][JSON header][raw bytes] (safetensors-like) -----------

_DT = {torch.int8: "I8", torch.uint8: "U8", torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32"}
_TD = {v: k for k, v in _DT.items()}


def _quantized_parts(name: str, t: torch.Tensor, scheme: str) -> list[tuple[str, torch.dtype, list[int]]]:
    """The tensors a weight becomes in the file, with dtype and shape, known before quantizing it."""
    N, K = t.shape
    parts = [(name + "::q", torch.int8 if scheme == "int8" else torch.uint8, [N, K if scheme == "int8" else K // 2]),
             (name + "::s", torch.float16, [N, K // BLOCK])]
    return parts + ([(name + "::m", torch.float16, [N, K // BLOCK])] if scheme == "int4" else [])


def _bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    return (t.view(torch.uint8) if t.dtype == torch.bfloat16 else t).numpy().tobytes()


def save_qt(path: str, source, scheme: str, log=print, keep_int8: frozenset = frozenset()) -> None:
    """Stream tensors from `source` (anything with tensor_names()/get()) into a quantized .qt file, one tensor in
    memory at a time: every tensor's size follows from its shape and scheme, so the header is written first and
    each tensor is quantized, written and dropped in turn. It goes to a temporary file that replaces `path` only once
    it is complete, so an interrupted run leaves any earlier file as it was."""
    header, offset, plan = {"__metadata__": {"scheme": scheme, "block": str(BLOCK)}}, 0, []
    for name in source.tensor_names():
        t = source.get(name)                                         # a view of the file: nothing is read yet
        if should_quantize(name, t):
            sch = scheme_for(name, scheme, keep_int8)
            for key, dt, shape in _quantized_parts(name, t, sch):
                n = torch.Size(shape).numel() * torch.tensor([], dtype=dt).element_size()
                header[key] = {"dtype": _DT[dt], "shape": shape, "data_offsets": [offset, offset + n]}
                offset += n
            header[name + "::q"] |= {"logical_shape": list(t.shape), "scheme": sch}
            plan.append((name, sch))
        else:
            n = t.numel() * t.element_size()
            header[name] = {"dtype": _DT[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + n]}
            offset += n
            plan.append((name, None))
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)                                      # keep the data section 8-byte aligned
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(hb))); f.write(hb)
            for name, sch in plan:
                t = source.get(name)
                if sch is None:
                    f.write(_bytes(t))
                    continue
                q = quantize(t, sch)
                for part in (q.data, q.scales) + ((q.mins,) if q.mins is not None else ()):
                    f.write(_bytes(part))
                del q
            written = f.tell() - 8 - len(hb)
        if written != offset:
            raise RuntimeError(f"{path}: wrote {written} bytes of tensor data, the header promised {offset}")
        os.replace(tmp, path)
    except BaseException:                                            # Ctrl-C included: no half-written file is left
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log(f"wrote {path}: {offset / 1e9:.2f} GB of tensor data")


class QtFile:
    """Loads a .qt file lazily via mmap; get() returns QuantTensor for quantized weights, tensors otherwise."""

    def __init__(self, path: str):
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        n = struct.unpack("<Q", self._mm[:8])[0]
        self._h = json.loads(self._mm[8:8 + n])
        meta = self._h.pop("__metadata__")
        self.scheme = meta["scheme"]
        self._start = 8 + n
        end = self._start + max((e["data_offsets"][1] for e in self._h.values()), default=0)
        if end > len(self._mm):
            raise ValueError(f"{path}: {len(self._mm)} bytes, but its header needs {end} (an interrupted write?)")

    def _raw(self, key: str) -> torch.Tensor:
        e = self._h[key]
        dt = _TD[e["dtype"]]
        a, b = e["data_offsets"]
        size = torch.tensor([], dtype=dt).element_size()
        return torch.frombuffer(self._mm, dtype=dt, count=(b - a) // size, offset=self._start + a).view(*e["shape"])

    def tensor_names(self) -> list[str]:
        return sorted({k.split("::")[0] for k in self._h})

    def get(self, name: str):
        if name + "::q" in self._h:
            e = self._h[name + "::q"]
            N, K = e["logical_shape"]
            mins = self._raw(name + "::m") if name + "::m" in self._h else None
            return QuantTensor(e["scheme"], self._raw(name + "::q"), self._raw(name + "::s"), (N, K), BLOCK, mins)
        return self._raw(name)
