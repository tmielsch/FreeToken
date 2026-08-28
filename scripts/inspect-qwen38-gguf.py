#!/usr/bin/env python3
"""Inspect a Qwen3.8-Flash-Next GGUF without reading model weight payloads.

Designed for large Unsloth Dynamic/UD GGUFs, including llama.cpp split files.
The script intentionally does not import ``freetoken``: it only needs the small
``gguf`` Python package, so it can run before the full FreeToken environment is
installed. ``GGUFReader`` memory-maps the file and this script walks tensor
metadata to report quant types, routed-expert geometry, and (optionally) a compact
architecture/tensor map useful when implementing a new FreeToken GGUF adapter.

Usage:
    python scripts/inspect-qwen38-gguf.py /path/to/model.gguf
    python scripts/inspect-qwen38-gguf.py /path/to/model.gguf --tensor-map
    python scripts/inspect-qwen38-gguf.py /path/to/model-00001-of-00003.gguf
    python scripts/inspect-qwen38-gguf.py /path/to/model-directory/
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path
from typing import Any


_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)

_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.(?P<proj>ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$"
)

_BLOCK_RE = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<suffix>.+)$")
_NUMERIC_COMPONENT_RE = re.compile(r"(?<=\.)\d+(?=\.)")

# These tokenizer fields are small and useful to an adapter. The huge vocabulary,
# score, merge, and token-type arrays are intentionally not dumped.
_SMALL_TOKENIZER_KEYS = {
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
}


def _split_paths(model_path: str) -> tuple[str, ...]:
    path = os.path.abspath(model_path)
    if not os.path.isfile(path) or not path.lower().endswith(".gguf"):
        raise SystemExit(f"Not a GGUF file: {model_path}")

    match = _SPLIT_RE.match(os.path.basename(path))
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
        raise SystemExit(
            f"Split GGUF is incomplete: expected {count} shards; missing "
            f"{[os.path.basename(p) for p in missing]}"
        )
    return paths


def _resolve(path: str) -> str:
    if not os.path.isdir(path):
        return _split_paths(path)[0]

    files = sorted(glob.glob(os.path.join(path, "*.gguf")))
    if not files:
        raise SystemExit(f"No GGUF files found in {path}")

    candidates: list[str] = []
    for file_path in files:
        match = _SPLIT_RE.match(os.path.basename(file_path))
        if match is None or int(match.group("index")) == 1:
            candidates.append(file_path)

    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one GGUF model family in {path}; found "
            f"{[os.path.basename(p) for p in candidates]}"
        )
    return _split_paths(candidates[0])[0]


def _type_name(tensor_type) -> str:
    return getattr(tensor_type, "name", str(tensor_type))


def _field_value(field) -> Any:
    """Return a GGUF field value without making assumptions about gguf-py versions."""
    value = field.contents()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _compact_value(value: Any, *, max_items: int = 16, max_chars: int = 180) -> str:
    """Human-readable metadata value that never dumps tokenizer-sized arrays."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return repr(value.item())
            if value.size <= max_items:
                return repr(value.tolist())
            return f"<ndarray shape={tuple(int(v) for v in value.shape)} dtype={value.dtype}>"
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        if len(value) <= max_items:
            return repr(value)
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, dict):
        if len(value) <= max_items:
            return repr(value)
        return f"<dict len={len(value)}>"
    text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _normalize_tensor_name(name: str) -> str:
    """Collapse numeric path components so layer/part families appear once."""
    return _NUMERIC_COMPONENT_RE.sub("*", name)


def _compress_ints(values) -> str:
    values = sorted(set(int(v) for v in values))
    if not values:
        return "-"
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _architecture_metadata(reader) -> tuple[str, list[tuple[str, str]]]:
    arch_field = reader.fields.get("general.architecture")
    arch = str(_field_value(arch_field)) if arch_field is not None else "<missing>"
    prefixes = ("general.", "split.")
    rows: list[tuple[str, str]] = []
    for name, field in reader.fields.items():
        if (
            name.startswith(prefixes)
            or (arch != "<missing>" and name.startswith(f"{arch}."))
            or name in _SMALL_TOKENIZER_KEYS
        ):
            rows.append((name, _compact_value(_field_value(field))))
    return arch, sorted(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="GGUF file (any split) or directory containing one family")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the compact text report",
    )
    parser.add_argument(
        "--tensor-map",
        action="store_true",
        help=(
            "also print compact architecture metadata, normalized tensor families, and "
            "layer-structure signatures (for GGUF adapter development)"
        ),
    )
    args = parser.parse_args()

    try:
        import gguf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python package 'gguf'. Install the FreeToken dependencies or run "
            "`python -m pip install gguf`, then retry."
        ) from exc

    first = _resolve(args.model)
    shards = _split_paths(first)

    first_reader = gguf.GGUFReader(shards[0], mode="r")
    arch, arch_metadata = _architecture_metadata(first_reader)

    all_types: collections.Counter[str] = collections.Counter()
    expert_layers: dict[int, dict[str, dict]] = collections.defaultdict(dict)
    dense_types: collections.Counter[str] = collections.Counter()
    tensor_count = 0

    # family -> signature(type, ggml shape, torch shape) -> aggregate info
    families: dict[str, dict[tuple, dict[str, Any]]] = collections.defaultdict(dict)
    # layer -> complete list of (suffix, type, ggml shape). This lets us group QSA/GDN/etc.
    # layer structures without hardcoding converter tensor names.
    layer_structures: dict[int, list[tuple[str, str, tuple[int, ...]]]] = collections.defaultdict(list)

    for shard in shards:
        reader = gguf.GGUFReader(shard, mode="r")
        for tensor in reader.tensors:
            tensor_count += 1
            qtype = _type_name(tensor.tensor_type)
            all_types[qtype] += 1
            ne = tuple(int(v) for v in tensor.shape)
            torch_shape = tuple(reversed(ne))

            family = _normalize_tensor_name(tensor.name)
            signature = (qtype, ne, torch_shape)
            aggregate = families[family].setdefault(
                signature,
                {"count": 0, "layers": set(), "examples": []},
            )
            aggregate["count"] += 1
            if len(aggregate["examples"]) < 3:
                aggregate["examples"].append(tensor.name)

            block_match = _BLOCK_RE.match(tensor.name)
            if block_match:
                layer = int(block_match.group("layer"))
                aggregate["layers"].add(layer)
                layer_structures[layer].append((block_match.group("suffix"), qtype, ne))

            match = _EXPERT_RE.match(tensor.name)
            if match:
                layer = int(match.group("layer"))
                proj = match.group("proj").removeprefix("ffn_").removesuffix("_exps")
                block, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
                fast = ne[0]
                if fast % block:
                    raise RuntimeError(
                        f"{tensor.name}: fastest dim {fast} is not divisible by {block} ({qtype})"
                    )
                row_bytes = fast // block * type_size
                rows_per_expert = 1
                for value in ne[1:-1]:
                    rows_per_expert *= value
                expert_layers[layer][proj] = {
                    "type": qtype,
                    "ggml_type": int(tensor.tensor_type),
                    "shape_ggml": list(ne),
                    "shape_torch": list(torch_shape),
                    "row_bytes": row_bytes,
                    "bytes_per_expert": row_bytes * rows_per_expert,
                }
            else:
                dense_types[qtype] += 1

    layer_rows = []
    signatures: collections.Counter[tuple[str, str, str]] = collections.Counter()
    for layer in sorted(expert_layers):
        info = expert_layers[layer]
        missing = [name for name in ("gate", "up", "down") if name not in info]
        if missing:
            raise RuntimeError(f"layer {layer}: missing expert projections {missing}")
        signature = tuple(info[p]["type"] for p in ("gate", "up", "down"))
        signatures[signature] += 1
        layer_rows.append(
            {
                "layer": layer,
                "gate": info["gate"],
                "up": info["up"],
                "down": info["down"],
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family in sorted(families):
        for (qtype, ne, torch_shape), aggregate in sorted(
            families[family].items(), key=lambda item: (item[0][0], item[0][1])
        ):
            family_rows.append(
                {
                    "family": family,
                    "type": qtype,
                    "shape_ggml": list(ne),
                    "shape_torch": list(torch_shape),
                    "count": int(aggregate["count"]),
                    "layers": sorted(aggregate["layers"]),
                    "examples": list(aggregate["examples"]),
                }
            )

    # Group whole decoder layers by their exact set of tensor suffix/type/shape tuples.
    # This exposes e.g. QSA vs GDN layer schedules and special PLE layers with no Qwen4
    # naming assumptions in the scanner itself.
    structure_groups: dict[tuple, list[int]] = collections.defaultdict(list)
    for layer, entries in layer_structures.items():
        structure_groups[tuple(sorted(entries))].append(layer)
    structure_rows: list[dict[str, Any]] = []
    for index, (structure, layers) in enumerate(
        sorted(structure_groups.items(), key=lambda item: min(item[1])), start=1
    ):
        structure_rows.append(
            {
                "id": index,
                "layers": sorted(layers),
                "tensors": [
                    {"suffix": suffix, "type": qtype, "shape_ggml": list(ne)}
                    for suffix, qtype, ne in structure
                ],
            }
        )

    payload = {
        "first_shard": first,
        "shards": list(shards),
        "architecture": arch,
        "tensor_count": tensor_count,
        "all_quant_types": dict(all_types),
        "dense_quant_types": dict(dense_types),
        "expert_signatures": [
            {"gate": sig[0], "up": sig[1], "down": sig[2], "layers": count}
            for sig, count in signatures.items()
        ],
        "expert_layers": layer_rows,
    }
    if args.tensor_map:
        payload["architecture_metadata"] = [
            {"key": key, "value": value} for key, value in arch_metadata
        ]
        payload["tensor_families"] = family_rows
        payload["layer_structure_signatures"] = structure_rows

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"GGUF family: {Path(first).name}")
    print(f"Architecture: {arch}")
    print(f"Shards: {len(shards)} | tensors: {tensor_count}")
    print("All tensor types:", ", ".join(f"{k}={v}" for k, v in sorted(all_types.items())))

    if not args.tensor_map:
        print("\nRouted-expert quant signatures (gate / up / down):")
        for signature, count in signatures.most_common():
            print(
                f"  {signature[0]:>10} / {signature[1]:>10} / {signature[2]:>10} : "
                f"{count} layers"
            )

        print("\nPer-layer routed experts:")
        for row in layer_rows:
            fields = []
            for proj in ("gate", "up", "down"):
                item = row[proj]
                mib = item["bytes_per_expert"] / 2**20
                fields.append(f"{proj}={item['type']} ({mib:.3f} MiB/expert)")
            print(f"  L{row['layer']:02d}: " + " | ".join(fields))
        return

    print("\n=== Architecture metadata (large tokenizer arrays omitted) ===")
    for key, value in arch_metadata:
        print(f"{key} = {value}")

    print("\n=== Routed-expert signatures ===")
    for signature, count in signatures.most_common():
        layers = [
            row["layer"]
            for row in layer_rows
            if tuple(row[p]["type"] for p in ("gate", "up", "down")) == signature
        ]
        print(
            f"{signature[0]} / {signature[1]} / {signature[2]} : "
            f"{count} layers [{_compress_ints(layers)}]"
        )

    print("\n=== Tensor family inventory ===")
    print("family | type | ggml-shape | torch-shape | count | layers")
    for row in family_rows:
        layers = _compress_ints(row["layers"]) if row["layers"] else "-"
        print(
            f"{row['family']} | {row['type']} | {row['shape_ggml']} | "
            f"{row['shape_torch']} | {row['count']} | {layers}"
        )

    print("\n=== Decoder layer structure signatures ===")
    for row in structure_rows:
        print(f"\n[structure {row['id']}] layers={_compress_ints(row['layers'])}")
        for tensor in row["tensors"]:
            print(f"  {tensor['suffix']} | {tensor['type']} | {tensor['shape_ggml']}")


if __name__ == "__main__":
    main()
