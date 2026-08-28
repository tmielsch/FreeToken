"""Shared GGUF access helpers: detection, metadata, split-file enumeration, and tensors.

Thin layer over ``gguf.GGUFReader``. Metadata is read into a plain dict keyed by
GGUF field name; tensors are exposed as ``GgufTensor`` records carrying the
*torch* shape, ggml quant type, and a zero-copy ``uint8`` view of the packed bytes.

llama.cpp split GGUFs use ``...-00001-of-00003.gguf`` naming. Passing any shard
resolves the complete sibling set; metadata comes from shard 1 and tensor
enumeration spans every shard. This matters for large Unsloth UD checkpoints.
"""

from __future__ import annotations

import functools
import glob
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)


def is_gguf_path(model_path: str) -> bool:
    """Whether ``model_path`` is a concrete GGUF file."""
    return (
        isinstance(model_path, str)
        and os.path.isfile(model_path)
        and model_path.lower().endswith(".gguf")
    )


def gguf_split_paths(model_path: str) -> tuple[str, ...]:
    """Resolve a GGUF path to its complete llama.cpp split set.

    Non-split GGUFs return a one-element tuple. A split file must have every
    sibling present; failing early avoids silently building a partial model from
    e.g. only ``00001-of-00003``.
    """
    if not is_gguf_path(model_path):
        raise ValueError(f"Not a GGUF file: {model_path}")

    path = os.path.abspath(model_path)
    name = os.path.basename(path)
    match = _SPLIT_RE.match(name)
    if match is None:
        return (path,)

    prefix = match.group("prefix")
    count = int(match.group("count"))
    folder = os.path.dirname(path)
    paths = tuple(
        os.path.join(folder, f"{prefix}-{index:05d}-of-{count:05d}.gguf")
        for index in range(1, count + 1)
    )
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Split GGUF is incomplete: expected {count} shards beside {path}; "
            f"missing {[os.path.basename(p) for p in missing]}"
        )
    return paths


# Canonical name of the metadata-only GGUF that ``convert_checkpoint`` drops into an FTW
# dir built from a bare ``.gguf`` source. A GGUF carries its config AND tokenizer in the
# file's KV section, not sibling files, so a converted checkpoint has nowhere else to read
# them from -- this file is the header + KV bytes verbatim (tensor_count patched to 0, no
# tensor infos, no weight data), letting the FTW dir resolve config/tokenizer the exact
# same way as the original ``.gguf`` file does.
FTW_METADATA_GGUF = "source_metadata.gguf"
# Records whether the source carried an untied ``output.weight`` head (the tensor table
# is stripped from metadata-only gguf files, so the fact travels as a KV).
OUTPUT_WEIGHT_PRESENT_KV = "freetoken.output_weight_present"


def _gguf_in_directory(model_path: str) -> str | None:
    """Pick the unique GGUF model family in a local directory, if there is one."""
    files = sorted(
        p
        for p in glob.glob(os.path.join(model_path, "*.gguf"))
        if os.path.basename(p) != FTW_METADATA_GGUF
    )
    if not files:
        return None

    # A split family is represented by its first shard. Multiple quant families in
    # one directory are ambiguous and must be selected explicitly by file path.
    first_shards = []
    singles = []
    for path in files:
        match = _SPLIT_RE.match(os.path.basename(path))
        if match is None:
            singles.append(path)
        elif int(match.group("index")) == 1:
            first_shards.append(path)

    candidates = singles + first_shards
    if len(candidates) == 1:
        # Validate the sibling set now so every downstream consumer sees the same failure.
        return gguf_split_paths(candidates[0])[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{model_path} contains multiple GGUF model families; pass the desired "
            f".gguf (usually its 00001 split) explicitly: "
            f"{[os.path.basename(p) for p in candidates]}"
        )
    return None


def gguf_config_source(model_path: str) -> str | None:
    """The GGUF file to source config/tokenizer/metadata from, or ``None``.

    A split GGUF resolves to shard 1. A local directory containing exactly one
    GGUF family resolves to that family, which is convenient for Unsloth's
    multi-shard downloads. FTW directories keep using ``source_metadata.gguf``.
    """
    if is_gguf_path(model_path):
        return gguf_split_paths(model_path)[0]
    if isinstance(model_path, str) and os.path.isdir(model_path):
        cand = os.path.join(model_path, FTW_METADATA_GGUF)
        if os.path.isfile(cand):
            return cand
        return _gguf_in_directory(model_path)
    return None


def write_metadata_gguf(source_gguf: str, dest_path: str) -> None:
    """Write a metadata-only GGUF from shard 1 of a source model."""
    import gguf

    source_gguf = gguf_split_paths(source_gguf)[0]
    reader = gguf.GGUFReader(source_gguf)
    assert reader.tensors, f"{source_gguf}: no tensors to bound the KV section"
    kv_end = int(reader.tensors[0].field.offset)
    buf = bytearray(reader.data[:kv_end].tobytes())
    buf[8:16] = b"\x00" * 8  # tensor_count
    key = OUTPUT_WEIGHT_PRESENT_KV.encode()

    # For split GGUFs the untied output tensor may live in another shard.
    present = "output.weight" in gguf_tensor_names(source_gguf)
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", int(gguf.GGUFValueType.BOOL)) + bytes([1 if present else 0])
    struct.pack_into("<Q", buf, 16, struct.unpack_from("<Q", buf, 16)[0] + 1)
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, dest_path)

    check = gguf.GGUFReader(dest_path)
    assert not check.tensors, "metadata gguf still lists tensors after patch"
    src_keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    dst_keys = {k for k in check.fields if not k.startswith("GGUF.")}
    assert dst_keys == src_keys | {OUTPUT_WEIGHT_PRESENT_KV}, (
        f"metadata gguf KV keys differ from source: "
        f"missing {sorted(src_keys - dst_keys)}, "
        f"extra {sorted(dst_keys - src_keys - {OUTPUT_WEIGHT_PRESENT_KV})}"
    )


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # torch order (ggml dims reversed)
    ggml_type: int
    rows: int
    row_bytes: int
    _raw: np.ndarray

    def packed(self) -> torch.Tensor:
        return torch.from_numpy(self._raw)


def _field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.contents()


@functools.cache
def _reader(model_path: str):
    import gguf
    return gguf.GGUFReader(model_path)


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    first = gguf_split_paths(model_path)[0] if is_gguf_path(model_path) else model_path
    reader = _reader(first)
    return {name: field.contents() for name, field in reader.fields.items()}


def gguf_architecture(model_path: str) -> str:
    first = gguf_split_paths(model_path)[0] if is_gguf_path(model_path) else model_path
    arch = _field_value(_reader(first), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {first} has no general.architecture")
    return str(arch)


def _iter_file_tensors(path: str) -> Iterator[GgufTensor]:
    import gguf

    reader = _reader(path)
    for t in reader.tensors:
        ne = [int(s) for s in t.shape]
        torch_shape = tuple(reversed(ne))
        block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
        n_fast = ne[0]
        if n_fast % block != 0:
            raise ValueError(
                f"{t.name}: fastest dim {n_fast} not a multiple of block {block} "
                f"for {t.tensor_type.name}"
            )
        row_bytes = n_fast // block * type_size
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        flat = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
        raw = flat.reshape(rows, row_bytes)
        yield GgufTensor(
            name=t.name,
            shape=torch_shape,
            ggml_type=int(t.tensor_type),
            rows=rows,
            row_bytes=row_bytes,
            _raw=raw,
        )


def iter_gguf_tensors(model_path: str) -> Iterator[GgufTensor]:
    """Yield tensors across all shards of a single/split GGUF model."""
    paths = gguf_split_paths(model_path)
    seen: set[str] = set()
    for path in paths:
        for tensor in _iter_file_tensors(path):
            if tensor.name in seen:
                raise ValueError(
                    f"duplicate GGUF tensor {tensor.name!r} across split shards"
                )
            seen.add(tensor.name)
            yield tensor


def gguf_tensor_names(model_path: str) -> set[str]:
    return {t.name for t in iter_gguf_tensors(model_path)}


__all__ = [
    "is_gguf_path",
    "gguf_split_paths",
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "gguf_config_source",
    "write_metadata_gguf",
    "GgufTensor",
    "load_gguf_metadata",
    "gguf_architecture",
    "iter_gguf_tensors",
    "gguf_tensor_names",
]
