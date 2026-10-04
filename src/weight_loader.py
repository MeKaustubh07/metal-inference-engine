"""The loading dock: a .safetensors file on disk -> named bf16 tensors in memory, zero-copy.

File layout:  [8-byte little-endian header length][JSON catalog][raw tensor bytes, back-to-back]
A big checkpoint is split into several such files plus model.safetensors.index.json (ShardedSafetensors).
"""
import json
import mmap
import struct
from pathlib import Path

import torch

# safetensors dtype string -> torch dtype (only the ones we expect to meet)
DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


class SafetensorsFile:
    def __init__(self, path: str):
        self._file = open(path, "rb")

        # Memory-map the file: the OS maps its bytes into our address space and
        # pulls pages from the SSD only when first touched. Nothing is read up front.
        # ACCESS_COPY = private copy-on-write mapping: readable, writable in our
        # process only, never written back to disk.
        self._mm = mmap.mmap(
            self._file.fileno(),
            0,
            access=mmap.ACCESS_COPY,
        )

        # First 8 bytes: the catalog's length. '<Q' = little-endian unsigned 64-bit int.
        header_len = struct.unpack("<Q", self._mm[:8])[0]

        # Next header_len bytes: the JSON catalog (name -> dtype, shape, data_offsets).
        header = json.loads(
            self._mm[8 : 8 + header_len]
        )

        # Keep every real tensor; drop the one non-tensor entry.
        self._catalog = {
            name: info
            for name, info in header.items()
            if name != "__metadata__"
        }

        # Where the raw tensor bytes begin; catalog offsets are relative to this point.
        self._data_start = 8 + header_len

    def tensor_names(self) -> list[str]:
        return list(self._catalog)

    def info(self, name: str) -> dict:
        if name not in self._catalog:
            raise KeyError(f"no tensor named {name!r}")

        return self._catalog[name]

    def get(self, name: str) -> torch.Tensor:
        """Return a tensor that VIEWS the file bytes directly. No copy, no dtype conversion."""
        t = self.info(name)

        dtype = DTYPES[t["dtype"]]

        start, end = t["data_offsets"]

        # bytes / bytes-per-element = element count (2 for bf16)
        numel = (
            (end - start)
            // torch.tensor([], dtype=dtype).element_size()
        )

        # Point a tensor at the mmap'd bytes: warehouse door + catalog offset = file position.
        flat = torch.frombuffer(
            self._mm,
            dtype=dtype,
            count=numel,
            offset=self._data_start + start,
        )

        return flat.view(*t["shape"])

    def integrity_check(self) -> bool:
        """The catalog must account for every byte: last end offset == end of file."""
        max_end = max(
            t["data_offsets"][1]
            for t in self._catalog.values()
        )

        return self._data_start + max_end == len(self._mm)


class ShardedSafetensors:
    """Several .safetensors files read as one, through model.safetensors.index.json, whose "weight_map" says which
    file holds each tensor. Same methods as SafetensorsFile, so models and the quantizer do not care which they get."""

    def __init__(self, index_path: str):
        index = json.load(open(index_path))
        folder = Path(index_path).parent
        weight_map: dict[str, str] = index["weight_map"]
        self._shards = {f: SafetensorsFile(str(folder / f)) for f in sorted(set(weight_map.values()))}
        self._where = {name: self._shards[f] for name, f in weight_map.items()}
        self._total_size = index.get("metadata", {}).get("total_size")
        # the index and the files must agree exactly: each file holds the tensors listed for it, and nothing else
        for f, shard in self._shards.items():
            listed = {n for n, g in weight_map.items() if g == f}
            if set(shard.tensor_names()) != listed:
                raise ValueError(f"{f}: its tensors differ from what {Path(index_path).name} lists for it")

    def tensor_names(self) -> list[str]:
        return list(self._where)

    def info(self, name: str) -> dict:
        if name not in self._where:
            raise KeyError(f"no tensor named {name!r}")
        return self._where[name].info(name)

    def get(self, name: str) -> torch.Tensor:
        if name not in self._where:
            raise KeyError(f"no tensor named {name!r}")
        return self._where[name].get(name)

    def integrity_check(self) -> bool:
        """Every file passes its own check, and the tensor bytes add up to the index's total_size."""
        offsets = [f.info(n)["data_offsets"] for f in self._shards.values() for n in f.tensor_names()]
        nbytes = sum(end - start for start, end in offsets)
        return all(f.integrity_check() for f in self._shards.values()) and self._total_size in (None, nbytes)


def open_weights(path: str):
    """A checkpoint: one .safetensors file, or the model.safetensors.index.json of a sharded one."""
    return ShardedSafetensors(path) if path.endswith(".json") else SafetensorsFile(path)
