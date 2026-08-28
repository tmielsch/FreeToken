#!/usr/bin/env python3
"""Inspect a Qwen3.8-Flash-Next GGUF without reading model weight payloads.

Designed for large Unsloth Dynamic/UD split GGUFs. ``gguf.GGUFReader`` mmaps the
files; this script walks tensor metadata and reports the per-layer quant types,
especially the routed MoE experts whose type/row-size determines FreeToken's
expert-cache layout.

Usage:
    python scripts/inspect-qwen38-gguf.py /path/to/*-00001-of-00003.gguf
    python scripts/inspect-qwen38-gguf.py /path/to/UD-Q3_K_XL/
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path


def _bootstrap_repo() -> None:
    root = Path(__file__).resolve().parents[1]
    python_dir = root / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


_bootstrap_repo()

from freetoken.models.gguf.reader import (  # noqa: E402
    gguf_config_source,
    gguf_split_paths,
)


_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.(?P<proj>ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$"
)


def _resolve(path: str) -> str:
    if os.path.isdir(path):
        resolved = gguf_config_source(path)
        if resolved is None:
            raise SystemExit(f"No unique GGUF family found in {path}")
        return resolved
    return gguf_split_paths(path)[0]


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

    import gguf

    first = _resolve(args.model)
    shards = gguf_split_paths(first)

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
                rows = 1
                for value in ne[1:-1]:
                    rows *= value
                # qwen4exp expert tensors are [fast input, output, experts] in ggml order.
                # The final dimension is the expert count, so one expert owns all rows
                # before it. Keep both raw dims and the byte count for easy auditing.
                experts = ne[-1] if len(ne) >= 3 else 1
                total_bytes = int(tensor.n_bytes) if hasattr(tensor, "n_bytes") else row_bytes
                expert_layers[layer][proj] = {
                    "type": qtype,
                    "ggml_type": int(tensor.tensor_type),
                    "shape_ggml": ne,
                    "row_bytes": row_bytes,
                    "bytes_per_expert": total_bytes // experts,
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
