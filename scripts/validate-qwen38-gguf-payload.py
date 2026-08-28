#!/usr/bin/env python3
"""Stream and reconcile Qwen3.8 Flash Next GGUF resident payloads.

This validates the real GGUF -> FreeToken resident-weight mapping one tensor at
a time. Routed experts and the enormous PLE hash table are intentionally excluded.
By default the GDN ssm_out dequantization is replaced with a meta shape stand-in;
pass --cuda-ssm-out to exercise the real GGUF CUDA dequant + V-head un-tiling path.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

import torch

import freetoken.models.qwen3_5_moe.attention as qwen35_attention
import freetoken.models.qwen4_exp.model as qwen4_model
import freetoken.models.qwen4_exp.gguf_weights as gguf_weights
from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.models import create_model
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.qwen4_exp.gguf import parse_gguf_config


class _MetaRotaryStub:
    def __init__(self, *, head_dim: int, rotary_dim: int, is_neox: bool = True):
        self.head_size = int(head_dim)
        self.rotary_dim = int(rotary_dim)
        self.is_neox = bool(is_neox)
        self._cos_sin_cache = torch.empty((0, 0), device="meta")

    def forward(self, *args, **kwargs):
        raise RuntimeError("meta RoPE stub cannot execute")

    def apply_inplace(self, *args, **kwargs):
        raise RuntimeError("meta RoPE stub cannot execute")

    def apply_rope_with_cos_sin_cache_inplace(self, *args, **kwargs):
        raise RuntimeError("meta RoPE stub cannot execute")


def _meta_get_rope(
    *,
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling=None,
    is_neox: bool = True,
):
    del max_position, base, rope_scaling
    return _MetaRotaryStub(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        is_neox=is_neox,
    )


def _build_meta_state(path: str):
    shim = build_gguf_shim(path)
    if shim.model_type != "qwen4exp":
        raise ValueError(f"expected qwen4exp GGUF, got {shim.model_type!r}")
    config = replace(parse_gguf_config(shim), moe_backend="offload")

    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
    elif (tp.rank, tp.size) != (0, 1):
        raise RuntimeError(
            f"payload validation requires TP=1, got rank={tp.rank} size={tp.size}"
        )

    old35 = qwen35_attention.get_rope
    old4 = qwen4_model.get_rope
    qwen35_attention.get_rope = _meta_get_rope
    qwen4_model.get_rope = _meta_get_rope
    try:
        with torch.device("meta"):
            model = create_model(config)
    finally:
        qwen35_attention.get_rope = old35
        qwen4_model.get_rope = old4
    return config, model.state_dict()


def validate(model_path: str, *, cuda_ssm_out: bool) -> None:
    path = os.path.abspath(model_path)
    config, expected = _build_meta_state(path)

    original_dequant = gguf_weights._dequant_2d_to_bf16
    if not cuda_ssm_out:
        def _shape_only_ssm_out(t):
            # _ungroup_v is still exercised, but no CUDA kernel or dense storage is needed.
            return torch.empty(t.shape, dtype=torch.bfloat16, device="meta")
        gguf_weights._dequant_2d_to_bf16 = _shape_only_ssm_out
    elif not torch.cuda.is_available():
        raise RuntimeError("--cuda-ssm-out requested but torch.cuda.is_available() is false")

    seen: set[str] = set()
    count = 0
    dtype_casts = 0
    logical_bytes = 0
    try:
        iterator = gguf_weights.iter_gguf_weights_impl(
            path,
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
        for name, tensor in iterator:
            count += 1
            if name in seen:
                raise RuntimeError(f"duplicate mapped state key: {name}")
            seen.add(name)
            target = expected.get(name)
            if target is None:
                raise RuntimeError(f"mapped key is not present in model state_dict: {name}")
            if tuple(tensor.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"shape mismatch for {name}: loaded {tuple(tensor.shape)}, "
                    f"target {tuple(target.shape)}"
                )
            if tensor.dtype != target.dtype:
                dtype_casts += 1
            logical_bytes += tensor.numel() * tensor.element_size()
            if count % 100 == 0:
                print(f"  reconciled {count}/{len(expected)} state tensors...")
            del tensor
    finally:
        gguf_weights._dequant_2d_to_bf16 = original_dequant

    missing = sorted(set(expected) - seen)
    if missing:
        preview = ", ".join(missing[:12])
        raise RuntimeError(
            f"resident loader produced {len(seen)}/{len(expected)} state keys; "
            f"missing {len(missing)}: {preview}" + (" ..." if len(missing) > 12 else "")
        )

    print("Qwen3.8 Flash Next GGUF resident payload reconciliation: OK")
    print(f"  file: {path}")
    print(f"  state keys reconciled: {count}/{len(expected)}")
    print(f"  logical yielded bytes: {logical_bytes / 2**30:.2f} GiB")
    print(f"  tensors cast by engine materializer: {dtype_casts}")
    print("  routed experts: skipped (host-bank path)")
    print("  PLE hash table: skipped (mmap row-gather path)")
    print(
        "  GDN ssm_out: "
        + ("real CUDA dequant + V-head un-tiling" if cuda_ssm_out else "shape-only; rerun with --cuda-ssm-out")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="GGUF file or first split shard")
    parser.add_argument(
        "--cuda-ssm-out",
        action="store_true",
        help="exercise real GGUF CUDA dequantization for GDN ssm_out tensors",
    )
    args = parser.parse_args()
    validate(args.model, cuda_ssm_out=args.cuda_ssm_out)


if __name__ == "__main__":
    main()
