#!/usr/bin/env python3
"""Inspect a Qwen3.8-Flash-Next GGUF without reading model weight payloads.

Designed for large Unsloth Dynamic/UD GGUFs, including llama.cpp split files.
The script intentionally does not import ``freetoken``: it only needs the small
``gguf`` Python package, so it can run before the full FreeToken environment is
installed. ``GGUFReader`` memory-maps the file and this script walks tensor
metadata to report quant types and routed-expert geometry.

Usage:
    python scripts/inspect-qwen38-gguf.py /path/to/model.gguf
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


_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)

_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.(?P<proj>ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$"
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="GGUF file (any split) or directory containing one family")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the compact text report",
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

    all_types: collections.Counter[str] = collections.Counter()
    expert_layers: dict[int, dict[str, dict]] = collections.defaultdict(dict)
    dense_types: collections.Counter[str] = collections.Counter()
    tensor_count = 0

    for shard in shards:
        reader = gguf.GGUFReader(shard, mode="r")
        for tensor in reader.tensors:
            tensor_count += 1
            qtype = _type_name(tensor.tensor_type)
            all_types[qtype] += 1
            match = _EXPERT_RE.match(tensor.name)
            if match:
                layer = int(match.group("layer"))
                proj = match.group("proj").removeprefix("ffn_").removesuffix("_exps")
                ne = [int(v) for v in tensor.shape]
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
                    "shape_ggml": ne,
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

    payload = {
        "first_shard": first,
        "shards": list(shards),
        "tensor_count": tensor_count,
        "all_quant_types": dict(all_types),
        "dense_quant_types": dict(dense_types),
        "expert_signatures": [
            {"gate": sig[0], "up": sig[1], "down": sig[2], "layers": count}
            for sig, count in signatures.items()
        ],
        "expert_layers": layer_rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"GGUF family: {Path(first).name}")
    print(f"Shards: {len(shards)} | tensors: {tensor_count}")
    print("All tensor types:", ", ".join(f"{k}={v}" for k, v in sorted(all_types.items())))
    print("\nRouted-expert quant signatures (gate / up / down):")
    for signature, count in signatures.most_common():
        print(f"  {signature[0]:>10} / {signature[1]:>10} / {signature[2]:>10} : {count} layers")

    print("\nPer-layer routed experts:")
    for row in layer_rows:
        fields = []
        for proj in ("gate", "up", "down"):
            item = row[proj]
            mib = item["bytes_per_expert"] / 2**20
            fields.append(f"{proj}={item['type']} ({mib:.3f} MiB/expert)")
        print(f"  L{row['layer']:02d}: " + " | ".join(fields))


if __name__ == "__main__":
    main()
