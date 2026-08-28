"""Native GGUF adapter for Qwen3.8-Flash-Next (``qwen4exp``).

The GGUF metadata contract comes from llama.cpp's qwen4exp converter/model. This
module intentionally treats quantization as a per-tensor property: Unsloth's UD
GGUFs are dynamic/mixed quants, so the file-level label (e.g. UD-Q3_K_XL) must
never be interpreted as one uniform ggml type.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
)

from .args import Qwen4ExpArgs

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


_MISSING = object()


def _meta_get(metadata: dict, key: str, default=_MISSING):
    full = f"qwen4exp.{key}"
    value = metadata.get(full, default)
    if value is _MISSING:
        raise KeyError(f"missing GGUF metadata key {full}")
    return value


def _as_tuple(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    try:
        return tuple(int(v) for v in value.tolist())
    except AttributeError:
        return (int(value),)


def _tensor_types_header_only(model_path: str) -> dict[str, int]:
    """Return ``{tensor_name: ggml_type}`` without touching tensor payload pages.

    ``GGUFReader`` mmaps a file, but tensor names/types live in the tensor-info
    section. In particular this must never call ``GgufTensor.packed()`` or turn
    ``tensor.data`` into a contiguous NumPy array: Qwen3.8 carries a huge PLE
    table and scanning its payload just to discover quant types would be fatal.
    """
    import gguf

    from freetoken.models.gguf.reader import gguf_split_paths

    types: dict[str, int] = {}
    for path in gguf_split_paths(model_path):
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name in types:
                raise ValueError(
                    f"duplicate GGUF tensor {tensor.name!r} across split shards"
                )
            types[tensor.name] = int(tensor.tensor_type)
    return types


def _gguf_geometry(
    model_path: str, num_layers: int
) -> tuple[int | None, tuple[tuple[int, int], ...] | None]:
    """Return embedding type and routed-expert ``(gate_up, down)`` types per layer.

    Published Qwen4Exp GGUFs store routed gate and up as separate tensors. The
    generic FreeToken GGUF expert bank fuses their packed rows, which is only
    valid when both use the same ggml type for that layer; assert this rather
    than silently corrupting a dynamic Unsloth checkpoint.
    """
    types = _tensor_types_header_only(model_path)
    if not types:
        # Metadata-only GGUF (e.g. a future FTW conversion) has no tensor table.
        return None, None

    embed_type = types.get("token_embd.weight")
    expert_types: list[tuple[int, int]] = []
    missing: list[str] = []
    for layer in range(num_layers):
        gate_name = f"blk.{layer}.ffn_gate_exps.weight"
        up_name = f"blk.{layer}.ffn_up_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"
        gate = types.get(gate_name)
        up = types.get(up_name)
        down = types.get(down_name)
        for name, value in ((gate_name, gate), (up_name, up), (down_name, down)):
            if value is None:
                missing.append(name)
        if gate is None or up is None or down is None:
            continue
        if gate != up:
            raise ValueError(
                f"Qwen4Exp GGUF layer {layer} uses different routed gate/up types "
                f"({gate} != {up}); the fused gate_up bank requires one type per layer"
            )
        expert_types.append((gate, down))

    if missing:
        raise ValueError(
            "Qwen4Exp GGUF tensor table is missing routed expert tensors: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    if len(expert_types) != num_layers:
        raise ValueError(
            f"Qwen4Exp GGUF recovered {len(expert_types)} expert layers, "
            f"expected {num_layers}"
        )
    return embed_type, tuple(expert_types)


def _layer_groups(metadata: dict, num_layers: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return ``(linear_ids, qsa_ids, index_compress_ratio)``.

    llama.cpp's Qwen4Exp converter explicitly writes
    ``attention.compress_ratios`` with the QSA ratio on full-attention layers
    and zero on recurrent/GatedDeltaNet layers. Prefer that exact signal; older
    files may instead carry ``attention.recurrent_layers``.
    """
    ratios = metadata.get("qwen4exp.attention.compress_ratios")
    if ratios is not None:
        ratios = _as_tuple(ratios)
        if len(ratios) != num_layers:
            raise ValueError(
                "qwen4exp.attention.compress_ratios length "
                f"{len(ratios)} != block_count {num_layers}"
            )
        qsa_ids = tuple(i for i, ratio in enumerate(ratios) if ratio > 0)
        linear_ids = tuple(i for i, ratio in enumerate(ratios) if ratio == 0)
        nonzero = {ratio for ratio in ratios if ratio > 0}
        if len(nonzero) != 1:
            raise ValueError(
                "Qwen4Exp expects one non-zero QSA compress ratio, "
                f"got {sorted(nonzero)}"
            )
        return linear_ids, qsa_ids, next(iter(nonzero))

    recurrent = metadata.get("qwen4exp.attention.recurrent_layers")
    if recurrent is None:
        raise KeyError(
            "Qwen4Exp GGUF needs qwen4exp.attention.compress_ratios or "
            "qwen4exp.attention.recurrent_layers to recover the hybrid layer pattern"
        )
    recurrent = tuple(bool(v) for v in recurrent)
    if len(recurrent) != num_layers:
        raise ValueError(
            "qwen4exp.attention.recurrent_layers length "
            f"{len(recurrent)} != block_count {num_layers}"
        )
    linear_ids = tuple(i for i, is_recurrent in enumerate(recurrent) if is_recurrent)
    qsa_ids = tuple(i for i, is_recurrent in enumerate(recurrent) if not is_recurrent)
    return linear_ids, qsa_ids, 4


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata
    g = lambda key, default=_MISSING: _meta_get(m, key, default)

    num_layers = int(g("block_count"))
    hidden = int(g("embedding_length"))
    num_q_heads = int(g("attention.head_count"))
    num_kv_heads = int(g("attention.head_count_kv"))
    head_dim = int(g("attention.key_length"))
    rotary_dim = int(g("rope.dimension_count"))
    max_position = int(g("context_length"))
    rope_base = float(g("rope.freq_base"))

    sections = _as_tuple(g("rope.dimension_sections"))
    if len(sections) < 3:
        raise ValueError(f"Qwen4Exp MRoPE needs >=3 sections, got {sections}")
    mrope_section = tuple(sections[:3])
    if sum(mrope_section) * 2 != rotary_dim:
        raise ValueError(
            "Qwen4Exp MRoPE sections do not cover the rotary dimension: "
            f"{mrope_section} vs {rotary_dim}"
        )

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_base,
        scaling=None,
    )

    linear_ids, qsa_ids, index_compress_ratio = _layer_groups(m, num_layers)

    ssm_state = int(g("ssm.state_size"))
    ssm_groups = int(g("ssm.group_count"))
    ssm_value_heads = int(g("ssm.time_step_rank"))
    ssm_conv = int(g("ssm.conv_kernel"))
    ssm_inner = int(g("ssm.inner_size"))
    if ssm_inner != ssm_value_heads * ssm_state:
        raise ValueError(
            "Qwen4Exp GGUF SSM geometry mismatch: "
            f"inner={ssm_inner}, value_heads={ssm_value_heads}, state={ssm_state}"
        )

    indexer_n_heads = int(g("attention.indexer.head_count"))
    indexer_head_dim = int(g("attention.indexer.key_length"))
    indexer_budget = int(g("attention.indexer.top_k"))

    ple_layers_raw = m.get("qwen4exp.ple.layers", ())
    ple_layer_ids = _as_tuple(ple_layers_raw) if ple_layers_raw is not None else ()
    if ple_layer_ids:
        ngram_size = int(g("ple.ngram_size"))
        heads_per_ngram = int(g("ple.heads_per_ngram"))
        ple_head_dim = int(g("embedding_length_per_layer_input"))
        ple_embed_dim = ple_head_dim * (ngram_size - 1) * heads_per_ngram
        ple_conv_kernel = int(g("ple.conv_kernel"))
        eos_token_id = int(g("ple.eos_token_id"))
    else:
        ngram_size = 0
        heads_per_ngram = 0
        ple_embed_dim = 0
        ple_conv_kernel = 0
        eos_token_id = int(
            m.get("tokenizer.ggml.eos_token_id", m.get("tokenizer.ggml.eot_token_id", 0))
        )

    embed_quant, expert_types = _gguf_geometry(shim.model_path, num_layers)
    qwen4_args = Qwen4ExpArgs(
        hc_count=int(g("hyper_connection.count")),
        hc_lowrank=int(g("hyper_connection.low_rank")),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=ple_conv_kernel,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=0,
        split_ngram_parts=1,
        eos_token_id=eos_token_id,
        indexer_n_heads=indexer_n_heads,
        indexer_kv_heads=1,
        indexer_head_dim=indexer_head_dim,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=index_compress_ratio,
        output_gate_type="sigmoid",
        mrope_section=mrope_section,
        mrope_interleaved=True,
        gguf_model_path=shim.model_path,
        gguf_embed_quant=embed_quant,
        gguf_expert_types=expert_types,
    )

    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=ssm_groups,
            num_value_heads=ssm_value_heads,
            key_head_dim=ssm_state,
            value_head_dim=ssm_state,
            conv_kernel_dim=ssm_conv,
            output_gate=True,
        ),
        QSAAttentionGroupConfig(
            name="qsa",
            layer_ids=qsa_ids,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_config=rotary,
            index_num_heads=indexer_n_heads,
            index_num_kv_heads=1,
            index_head_dim=indexer_head_dim,
            index_token_budget=indexer_budget,
            index_compress_ratio=index_compress_ratio,
        ),
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=False,
        model_type="qwen4exp",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="gguf",
        moe_weight_format="gguf",
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        use_qk_norm=True,
        image_token_id=(
            int(m["qwen4exp.ple.image_token_id"])
            if "qwen4exp.ple.image_token_id" in m
            else None
        ),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        requires_naive_cache=True,
        supports_cuda_graph=False,
    )


def gguf_quant_inventory(model_path: str) -> dict[int, int]:
    """Count tensors by native ggml type from headers only."""
    counts = Counter(_tensor_types_header_only(model_path).values())
    return dict(sorted(counts.items()))


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Load resident Qwen4Exp GGUF weights through the filtered payload adapter."""
    from .gguf_weights import iter_gguf_weights_impl

    yield from iter_gguf_weights_impl(
        model_path,
        device,
        include_moe_experts=include_moe_experts,
        include_non_moe=include_non_moe,
    )


def is_gguf_model(config: ModelConfig) -> bool:
    return config.model_type == "qwen4exp" and config.moe_weight_format == "gguf"


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "gguf_quant_inventory",
    "is_gguf_model",
]
