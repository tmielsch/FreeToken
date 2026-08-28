#!/usr/bin/env python3
"""Header/meta-only smoke validator for Qwen3.8 Flash Next GGUF.

This intentionally does NOT iterate resident weight payloads, load routed expert
banks, or call PLE host-weight loading.  It parses the real GGUF headers and
constructs the complete FreeToken model on the ``meta`` device, which is enough
to validate registry routing, config geometry, per-tensor quant module sizing,
and the state-dict contract without moving tens of GiB through RAM/VRAM.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

# Allow running straight from a source checkout without an editable install.
ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

import torch

from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.models import create_model
from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.gguf.dequant import GGML_NAME, row_bytes
from freetoken.models.qwen4_exp.gguf import (
    _tensor_types_header_only,
    parse_gguf_config,
)


def _shape(state: dict[str, torch.Tensor], key: str) -> tuple[int, ...]:
    if key not in state:
        raise RuntimeError(f"expected state key is missing: {key}")
    return tuple(state[key].shape)


def _expect_shape(
    state: dict[str, torch.Tensor], key: str, expected: tuple[int, ...]
) -> None:
    actual = _shape(state, key)
    if actual != expected:
        raise RuntimeError(f"{key}: shape {actual}, expected {expected}")


def _fmt_type(qtype: int | None) -> str:
    if qtype is None:
        return "<missing>"
    return f"{GGML_NAME.get(qtype, str(qtype))} ({qtype})"


def validate(model_path: str) -> None:
    path = os.path.abspath(model_path)
    shim = build_gguf_shim(path)
    if shim.model_type != "qwen4exp":
        raise ValueError(
            f"expected general.architecture=qwen4exp, got {shim.model_type!r}"
        )
    if shim.architectures != ["Qwen4ExpGGUFForCausalLM"]:
        raise RuntimeError(
            "GGUF registry routing mismatch: " f"{shim.architectures!r}"
        )

    config = parse_gguf_config(shim)
    config = replace(config, moe_backend="offload")
    types = _tensor_types_header_only(path)

    # Standalone scripts do not pass through Engine's distributed bootstrap.
    # The GGUF path is TP=1 only, so seed the same global information that
    # VocabParallelEmbedding / parallel linear layers expect during construction.
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
    elif (tp.rank, tp.size) != (0, 1):
        raise RuntimeError(
            f"Qwen4Exp GGUF smoke validation requires TP=1, got rank={tp.rank} size={tp.size}"
        )

    # Full model construction, but no storage allocation and no payload access.
    with torch.device("meta"):
        model = create_model(config)
    if type(model).__name__ != "Qwen4ExpGGUFForCausalLM":
        raise RuntimeError(
            f"registry constructed {type(model).__name__}, expected Qwen4ExpGGUFForCausalLM"
        )
    state = model.state_dict()

    qsa_ids = tuple(
        layer for layer in range(config.num_layers) if not config.is_linear_layer(layer)
    )
    gdn_ids = tuple(
        layer for layer in range(config.num_layers) if config.is_linear_layer(layer)
    )
    expected_qsa = tuple(range(3, config.num_layers, 4))
    if qsa_ids != expected_qsa:
        raise RuntimeError(f"QSA layers {qsa_ids}, expected {expected_qsa}")

    args = config.qwen4_args
    expert_types = args.gguf_expert_types
    if expert_types is None or len(expert_types) != config.num_layers:
        raise RuntimeError("routed expert qtypes are missing for one or more layers")
    expert_signatures = Counter(expert_types)

    # PLE hash constants/table must be host state, never ordinary model-state tensors.
    forbidden_fragments = (
        "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_vocab_sizes",
        "ple_embedding.ngram_heads_offsets",
        "per_layer_token_embd",
    )
    forbidden = [
        key for key in state if any(fragment in key for fragment in forbidden_fragments)
    ]
    if forbidden:
        raise RuntimeError(f"PLE host-only tensors leaked into state_dict: {forbidden}")

    # Routed expert payloads belong to the host bank/offload cache, not the resident state.
    expert_state = [
        key
        for key in state
        if ".mlp.experts." in key
        and any(token in key for token in ("gate_up_proj", "down_proj"))
    ]
    if expert_state:
        raise RuntimeError(
            "routed expert payloads unexpectedly allocated in resident state: "
            + ", ".join(expert_state[:8])
        )

    H = config.hidden_size
    I = config.moe_intermediate_size
    E = config.num_experts
    embed_type = types.get("token_embd.weight")
    head_type = types.get("output.weight")
    ple_type = types.get("per_layer_token_embd.weight")
    if embed_type is None or head_type is None or ple_type is None:
        raise RuntimeError("embedding/lm-head/PLE qtype is missing from the GGUF header")

    _expect_shape(
        state,
        "model.embed_tokens.qweight",
        (config.vocab_size, row_bytes(H, embed_type)),
    )
    if not config.tie_word_embeddings:
        _expect_shape(
            state,
            "lm_head.qweight",
            (config.vocab_size, row_bytes(H, head_type)),
        )

    # One representative QSA layer: all q/k/v are Q8_0 in the scanned Unsloth file,
    # so gguf_merged_or_plain should materialize one fused packed qweight.
    qsa = qsa_ids[0]
    q_type = types[f"blk.{qsa}.attn_q.weight"]
    k_type = types[f"blk.{qsa}.attn_k.weight"]
    v_type = types[f"blk.{qsa}.attn_v.weight"]
    if len({q_type, k_type, v_type}) == 1:
        q_out = config.num_qo_heads * config.head_dim * 2
        kv_out = config.num_kv_heads * config.head_dim
        _expect_shape(
            state,
            f"model.layers.{qsa}.self_attn.qkv_proj.qweight",
            (q_out + 2 * kv_out, row_bytes(H, q_type)),
        )
    else:
        for part in range(3):
            key = f"model.layers.{qsa}.self_attn.qkv_proj.qweight_{part}"
            if key not in state:
                raise RuntimeError(f"mixed QSA projection is missing {key}")

    # One representative GDN layer: this Unsloth file mixes Q8_0 qkv/gate with
    # F32 beta/alpha, so the merged projection must expose four independent parts.
    gdn = gdn_ids[0]
    gdn_suffixes = (
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_beta.weight",
        "ssm_alpha.weight",
    )
    gdn_types = [types[f"blk.{gdn}.{suffix}"] for suffix in gdn_suffixes]
    if len(set(gdn_types)) == 1:
        key = f"model.layers.{gdn}.linear_attn.in_proj.qweight"
        if key not in state:
            raise RuntimeError(f"uniform GDN projection is missing {key}")
    else:
        for part in range(4):
            key = f"model.layers.{gdn}.linear_attn.in_proj.qweight_{part}"
            if key not in state:
                raise RuntimeError(f"mixed GDN projection is missing {key}")

    # Correctness-first implementation intentionally keeps ssm_out dense because
    # llama.cpp tiles its V-head columns and packed column permutation is unsafe.
    _expect_shape(
        state,
        f"model.layers.{gdn}.linear_attn.out_proj.weight",
        (H, config.linear_attention_group().num_value_heads * config.linear_attention_group().value_head_dim),
    )

    ple_layers = tuple(args.ple_layer_ids)
    if ple_layers:
        ple_layer = ple_layers[0]
        _expect_shape(
            state,
            f"model.layers.{ple_layer}.ple.key_proj.qweight",
            (
                args.hc_count * H,
                row_bytes(args.ple_embed_dim, types[f"blk.{ple_layer}.ple_key.weight"]),
            ),
        )
        _expect_shape(
            state,
            f"model.layers.{ple_layer}.ple.value_proj.qweight",
            (
                H,
                row_bytes(args.ple_embed_dim, types[f"blk.{ple_layer}.ple_value.weight"]),
            ),
        )

    print("Qwen3.8 Flash Next GGUF runtime smoke validation: OK")
    print(f"  file: {path}")
    print(f"  model class: {type(model).__name__}")
    print(f"  layers: {config.num_layers} ({len(gdn_ids)} GDN + {len(qsa_ids)} QSA)")
    print(f"  QSA layers: {','.join(map(str, qsa_ids))}")
    print(f"  experts: {E}, top-k: {config.num_experts_per_tok}")
    print(f"  resident state tensors: {len(state)}")
    print(f"  embedding: {_fmt_type(embed_type)}")
    print(f"  lm_head: {_fmt_type(head_type)}")
    print(
        "  PLE table: "
        f"{_fmt_type(ple_type)}, {row_bytes(args.ple_embed_dim // ((args.ngram_size - 1) * args.heads_per_ngram), ple_type)} packed bytes/head-row"
    )
    print("  routed expert signatures (gate_up/down):")
    for (gate_up, down), count in sorted(
        expert_signatures.items(), key=lambda item: (-item[1], item[0])
    ):
        print(
            f"    {count:2d} layers: {_fmt_type(gate_up)} / {_fmt_type(down)}"
        )
    print("  payload read: none (headers/meta model only)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="GGUF file or first split shard")
    args = parser.parse_args()
    validate(args.model)


if __name__ == "__main__":
    main()
