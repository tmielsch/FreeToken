"""Resident-weight adapter for Qwen3.8 Flash Next native GGUF.

This module is intentionally separate from the routed-expert banks and PLE hash
table.  It only materializes tensors after their GGUF name has been accepted, so
the 320M-row ``per_layer_token_embd.weight`` is never accidentally copied while
loading ordinary resident weights.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import torch

from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    GGML_UNQUANTIZED,
    row_bytes,
)
from freetoken.models.gguf.reader import GgufTensor, gguf_split_paths


_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)
_PLE_TABLE = "per_layer_token_embd.weight"

_DTYPE_FOR_GGML = {
    GGML_F32: torch.float32,
    GGML_F16: torch.float16,
    GGML_BF16: torch.bfloat16,
}


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size != 1:
        raise NotImplementedError(
            f"Qwen4Exp GGUF {what} currently supports TP=1 only"
        )


def _selected_tensor(raw) -> GgufTensor:
    """Materialize one already-selected GGUF tensor as native packed bytes."""
    import gguf

    ne = [int(v) for v in raw.shape]
    torch_shape = tuple(reversed(ne))
    block, type_size = gguf.GGML_QUANT_SIZES[raw.tensor_type]
    if ne[0] % block:
        raise ValueError(
            f"{raw.name}: fastest dim {ne[0]} is not divisible by "
            f"{raw.tensor_type.name} block size {block}"
        )
    bytes_per_row = ne[0] // block * type_size
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    # IMPORTANT: this is the first access to raw.data. Callers have already
    # filtered by name, so PLE and routed experts cannot reach this line.
    flat = np.ascontiguousarray(raw.data).reshape(-1).view(np.uint8)
    packed = flat.reshape(rows, bytes_per_row)
    return GgufTensor(
        name=raw.name,
        shape=torch_shape,
        ggml_type=int(raw.tensor_type),
        rows=rows,
        row_bytes=bytes_per_row,
        _raw=packed,
    )


def _iter_selected(
    model_path: str, accept: Callable[[str], bool]
) -> Iterator[GgufTensor]:
    """Iterate payloads, testing ``accept(name)`` before touching tensor.data."""
    import gguf

    seen: set[str] = set()
    for path in gguf_split_paths(model_path):
        reader = gguf.GGUFReader(path)
        for raw in reader.tensors:
            if raw.name in seen:
                raise ValueError(
                    f"duplicate GGUF tensor {raw.name!r} across split shards"
                )
            seen.add(raw.name)
            if not accept(raw.name):
                continue
            yield _selected_tensor(raw)


def _dense_unquantized(t: GgufTensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Interpret an F32/F16/BF16 GGUF tensor without quant/dequant math."""
    if t.ggml_type not in GGML_UNQUANTIZED:
        raise TypeError(
            f"{t.name}: expected unquantized tensor, got "
            f"{GGML_NAME.get(t.ggml_type, t.ggml_type)}"
        )
    value = t.packed().reshape(-1).view(_DTYPE_FOR_GGML[t.ggml_type]).reshape(t.shape)
    return value if dtype is None else value.to(dtype)


def _centered_norm(t: GgufTensor) -> torch.Tensor:
    """Undo llama.cpp's +1 fold for FreeToken norms that apply 1+w at runtime."""
    return _dense_unquantized(t, torch.float32) - 1.0


def _ungroup_v(
    t: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_per_k: int,
    head_dim: int,
) -> torch.Tensor:
    """Inverse of llama.cpp's grouped->tiled linear-attention V-head reorder."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    view = (
        shape[:dim]
        + [num_v_per_k, num_k_heads, head_dim]
        + shape[dim + 1 :]
    )
    out = t.reshape(*view)
    perm = list(range(len(view)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return out.permute(*perm).contiguous().reshape(*shape)


def _ungroup_packed_rows(
    packed: torch.Tensor,
    num_k_heads: int,
    num_v_per_k: int,
    head_dim: int,
) -> torch.Tensor:
    # Reordering whole output rows is safe for every GGUF block quant.
    return _ungroup_v(packed, 0, num_k_heads, num_v_per_k, head_dim)


def _dequant_2d_to_bf16(t: GgufTensor) -> torch.Tensor:
    """Dense fallback for a 2-D quantized tensor that needs a column permutation."""
    if len(t.shape) != 2:
        raise ValueError(f"{t.name}: expected 2-D tensor, got {t.shape}")
    if t.ggml_type in GGML_UNQUANTIZED:
        return _dense_unquantized(t, torch.bfloat16)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{t.name}: column un-tiling requires dense dequantization of "
            f"{GGML_NAME.get(t.ggml_type, t.ggml_type)}, but CUDA is unavailable"
        )
    from freetoken.kernel.gguf import ggml_dequantize

    out_features, in_features = t.shape
    packed = t.packed().reshape(
        out_features, row_bytes(in_features, t.ggml_type)
    ).cuda()
    dense = ggml_dequantize(
        packed,
        t.ggml_type,
        out_features,
        in_features,
        torch.bfloat16,
    )
    return dense.cpu()


def _quant_types(model_path: str) -> dict[str, int]:
    # Header-only helper; importing lazily avoids a module-init cycle with gguf.py.
    from .gguf import _tensor_types_header_only

    return _tensor_types_header_only(model_path)


def _config_for(model_path: str):
    from freetoken.utils import cached_load_hf_config
    from .gguf import parse_gguf_config

    return parse_gguf_config(cached_load_hf_config(model_path))


def _resident_name(name: str, num_layers: int) -> bool:
    """True only for payloads consumed by the ordinary resident-weight iterator."""
    if name == _PLE_TABLE:
        return False
    if name in {
        "token_embd.weight",
        "output.weight",
        "output_hc_down.weight",
        "output_hc_norm.weight",
        "output_hc_up.weight",
    }:
        return True
    if not name.startswith("blk."):
        return False
    _, layer_text, suffix = name.split(".", 2)
    layer = int(layer_text)
    if layer >= num_layers or suffix.startswith("nextn."):
        return False
    if suffix in _EXPERT_SUFFIXES:
        return False
    return True


def iter_gguf_weights_impl(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every resident Qwen4Exp parameter except routed experts and PLE table.

    Quantized projections stay in native GGUF byte layout.  GDN ``ssm_out`` is
    the one correctness-first exception: llama.cpp tiled its V-head *columns*,
    which cannot be undone by permuting packed rows, so it is dequantized to
    BF16 and column-un-tiled.  This costs resident VRAM and is a later
    optimization target, not a correctness shortcut.
    """
    if include_moe_experts:
        raise ValueError(
            "Qwen4Exp GGUF routed experts are loaded by the offload expert-bank path"
        )
    if not include_non_moe:
        return
    _require_tp1("resident weight loading")

    config = _config_for(model_path)
    types = _quant_types(model_path)
    qsa_layers = {
        i for i in range(config.num_layers) if not config.is_linear_layer(i)
    }
    gdn = config.linear_attention_group()
    assert gdn is not None
    k_heads = gdn.num_key_heads
    v_heads = gdn.num_value_heads
    v_per_k = v_heads // k_heads
    v_dim = gdn.value_head_dim
    qk_rows = 2 * gdn.num_key_heads * gdn.key_head_dim
    conv_dim = qk_rows + gdn.num_value_heads * gdn.value_head_dim

    # Fusion buffers are order-independent because GGUF tensor order is not a contract.
    qkv: dict[int, dict[str, torch.Tensor]] = {}
    gdn_in: dict[int, dict[str, torch.Tensor]] = {}
    shared: dict[int, dict[str, torch.Tensor]] = {}
    indexer: dict[int, dict[str, torch.Tensor]] = {}

    def emit_qkv(layer: int):
        slots = qkv.get(layer)
        if slots is None or set(slots) != {"q", "k", "v"}:
            return None
        qs = [
            types[f"blk.{layer}.attn_q.weight"],
            types[f"blk.{layer}.attn_k.weight"],
            types[f"blk.{layer}.attn_v.weight"],
        ]
        values = [slots["q"], slots["k"], slots["v"]]
        del qkv[layer]
        if len(set(qs)) == 1:
            return [(f"model.layers.{layer}.self_attn.qkv_proj.qweight", torch.cat(values, dim=0))]
        return [
            (f"model.layers.{layer}.self_attn.qkv_proj.qweight_{i}", value)
            for i, value in enumerate(values)
        ]

    def emit_shared(layer: int):
        slots = shared.get(layer)
        if slots is None or set(slots) != {"gate", "up"}:
            return None
        qts = [
            types[f"blk.{layer}.ffn_gate_shexp.weight"],
            types[f"blk.{layer}.ffn_up_shexp.weight"],
        ]
        values = [slots["gate"], slots["up"]]
        del shared[layer]
        if len(set(qts)) == 1:
            return [(
                f"model.layers.{layer}.mlp.shared_expert.gate_up_proj.qweight",
                torch.cat(values, dim=0),
            )]
        return [
            (
                f"model.layers.{layer}.mlp.shared_expert.gate_up_proj.qweight_{i}",
                value,
            )
            for i, value in enumerate(values)
        ]

    def emit_indexer(layer: int):
        slots = indexer.get(layer)
        if slots is None or set(slots) != {"q", "k"}:
            return None
        out = torch.cat([slots["q"], slots["k"]], dim=0)
        del indexer[layer]
        return (f"model.layers.{layer}.self_attn.indexer.index_qk_proj.weight", out)

    def emit_gdn(layer: int):
        slots = gdn_in.get(layer)
        if slots is None or set(slots) != {"qkv", "gate", "beta", "alpha"}:
            return None
        suffixes = (
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
        )
        qts = [types[f"blk.{layer}.{suffix}"] for suffix in suffixes]
        values = [slots["qkv"], slots["gate"], slots["beta"], slots["alpha"]]
        del gdn_in[layer]
        if len(set(qts)) == 1:
            return [(f"model.layers.{layer}.linear_attn.in_proj.qweight", torch.cat(values, dim=0))]
        return [
            (f"model.layers.{layer}.linear_attn.in_proj.qweight_{i}", value)
            for i, value in enumerate(values)
        ]

    for t in _iter_selected(
        model_path, lambda name: _resident_name(name, config.num_layers)
    ):
        name = t.name

        # Global weights.
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            if not config.tie_word_embeddings:
                yield "lm_head.qweight", t.packed()
            continue
        if name == "output_hc_norm.weight":
            yield "model.hyper_connection_mixer.hc_norm.weight", _centered_norm(t)
            continue
        if name == "output_hc_down.weight":
            yield "model.hyper_connection_mixer.input_mix_weight_down.qweight", t.packed()
            continue
        if name == "output_hc_up.weight":
            yield "model.hyper_connection_mixer.input_mix_weight_up.qweight", t.packed()
            continue

        if not name.startswith("blk."):
            continue
        _, layer_text, suffix = name.split(".", 2)
        layer = int(layer_text)
        base = f"model.layers.{layer}"

        # Hyper-connections: llama.cpp stored their plus-one norm scales already folded.
        for prefix, attr in (
            ("hc_attn_", "attn_hyper_connection"),
            ("hc_ffn_", "mlp_hyper_connection"),
        ):
            if suffix == prefix + "norm.weight":
                yield f"{base}.{attr}.hc_norm.weight", _centered_norm(t)
                break
            if suffix == prefix + "down.weight":
                yield f"{base}.{attr}.input_mix_weight_down.qweight", t.packed()
                break
            if suffix == prefix + "up.weight":
                yield f"{base}.{attr}.input_mix_weight_up.qweight", t.packed()
                break
            if suffix == prefix + "inject.weight":
                yield f"{base}.{attr}.block_inject_weight.weight", _dense_unquantized(t)
                break
        if suffix.startswith("hc_attn_") or suffix.startswith("hc_ffn_"):
            continue

        # Router + shared expert are present on every layer.
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _dense_unquantized(t)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _dense_unquantized(t).reshape(1, -1)
            continue
        if suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
            continue
        if suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            shared.setdefault(layer, {})[
                "gate" if suffix.startswith("ffn_gate") else "up"
            ] = t.packed()
            ready = emit_shared(layer)
            if ready:
                yield from ready
            continue

        # PLE's small resident tensors (the enormous hash table is explicitly excluded).
        if suffix == "ple_key.weight":
            yield f"{base}.ple.key_proj.qweight", t.packed()
            continue
        if suffix == "ple_value.weight":
            yield f"{base}.ple.value_proj.qweight", t.packed()
            continue
        if suffix == "ple_norm_key.weight":
            yield f"{base}.ple.norm_key.weight", _centered_norm(t)
            continue
        if suffix == "ple_norm_query.weight":
            yield f"{base}.ple.norm_query.weight", _centered_norm(t)
            continue
        if suffix == "ple_norm_conv.weight":
            yield f"{base}.ple.norm_conv.weight", _centered_norm(t)
            continue
        if suffix == "ple_conv1d.weight":
            yield f"{base}.ple.conv1d.weight", _dense_unquantized(t).reshape(
                config.qwen4_args.hc_count * config.hidden_size,
                1,
                config.qwen4_args.ple_conv_kernel_size,
            )
            continue

        if layer in qsa_layers:
            if suffix == "attn_q.weight":
                qkv.setdefault(layer, {})["q"] = t.packed()
            elif suffix == "attn_k.weight":
                qkv.setdefault(layer, {})["k"] = t.packed()
            elif suffix == "attn_v.weight":
                qkv.setdefault(layer, {})["v"] = t.packed()
            elif suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.qweight", t.packed()
            elif suffix == "attn_q_norm.weight":
                yield f"{base}.self_attn.q_norm.weight", _centered_norm(t)
            elif suffix == "attn_k_norm.weight":
                yield f"{base}.self_attn.k_norm.weight", _centered_norm(t)
            elif suffix == "indexer.q_proj.weight":
                indexer.setdefault(layer, {})["q"] = _dense_unquantized(t)
            elif suffix == "indexer.k_proj.weight":
                indexer.setdefault(layer, {})["k"] = _dense_unquantized(t)
            elif suffix == "indexer.q_norm.weight":
                yield f"{base}.self_attn.indexer.q_layernorm.weight", _centered_norm(t)
            elif suffix == "indexer.k_norm.weight":
                yield f"{base}.self_attn.indexer.k_layernorm.weight", _centered_norm(t)
            else:
                raise ValueError(f"unmapped Qwen4Exp QSA GGUF tensor: {name}")

            ready = emit_qkv(layer)
            if ready:
                yield from ready
            idx_ready = emit_indexer(layer)
            if idx_ready:
                yield idx_ready
            continue

        # Gated DeltaNet. Undo llama.cpp's V-head tiling before feeding FreeToken.
        if suffix == "attn_qkv.weight":
            packed = t.packed()
            packed = torch.cat(
                [
                    packed[:qk_rows],
                    _ungroup_packed_rows(
                        packed[qk_rows:], k_heads, v_per_k, v_dim
                    ),
                ],
                dim=0,
            )
            gdn_in.setdefault(layer, {})["qkv"] = packed
        elif suffix == "attn_gate.weight":
            gdn_in.setdefault(layer, {})["gate"] = _ungroup_packed_rows(
                t.packed(), k_heads, v_per_k, v_dim
            )
        elif suffix in ("ssm_beta.weight", "ssm_alpha.weight"):
            gdn_in.setdefault(layer, {})[
                "beta" if suffix.startswith("ssm_beta") else "alpha"
            ] = _ungroup_packed_rows(t.packed(), k_heads, v_per_k, 1)
        elif suffix == "ssm_a":
            stored_a = _ungroup_v(
                _dense_unquantized(t, torch.float32),
                0,
                k_heads,
                v_per_k,
                1,
            )
            if not bool((stored_a < 0).all()):
                raise ValueError(
                    f"{name}: expected llama.cpp A=-exp(A_log), got "
                    f"min={float(stored_a.min())}, max={float(stored_a.max())}"
                )
            yield f"{base}.linear_attn.A_log", torch.log(-stored_a)
        elif suffix == "ssm_dt.bias":
            dt = _ungroup_v(
                _dense_unquantized(t, torch.float32),
                0,
                k_heads,
                v_per_k,
                1,
            )
            yield f"{base}.linear_attn.dt_bias", dt
        elif suffix == "ssm_conv1d.weight":
            conv = _dense_unquantized(t, torch.bfloat16).reshape(
                conv_dim, gdn.conv_kernel_dim
            )
            conv = torch.cat(
                [
                    conv[:qk_rows],
                    _ungroup_v(
                        conv[qk_rows:], 0, k_heads, v_per_k, v_dim
                    ),
                ],
                dim=0,
            )
            yield f"{base}.linear_attn.conv1d.weight", conv.reshape(
                conv_dim, 1, gdn.conv_kernel_dim
            )
        elif suffix == "ssm_norm.weight":
            # This is the one Qwen3Next norm excluded from llama.cpp's +1 fold.
            yield f"{base}.linear_attn.norm.weight", _dense_unquantized(t)
        elif suffix == "ssm_out.weight":
            dense = _dequant_2d_to_bf16(t)
            dense = _ungroup_v(
                dense, 1, k_heads, v_per_k, v_dim
            )
            yield f"{base}.linear_attn.out_proj.weight", dense
        else:
            raise ValueError(f"unmapped Qwen4Exp GDN GGUF tensor: {name}")

        ready = emit_gdn(layer)
        if ready:
            yield from ready

    incomplete = {
        "qkv": sorted(qkv),
        "gdn_in": sorted(gdn_in),
        "shared": sorted(shared),
        "indexer": sorted(indexer),
    }
    incomplete = {k: v for k, v in incomplete.items() if v}
    if incomplete:
        raise RuntimeError(f"incomplete Qwen4Exp GGUF fusion groups: {incomplete}")


def convert_qwen4exp_to_gguf(model, config, *, model_path: str) -> None:
    """Replace resident dense modules with native-GGUF ops before state_dict creation."""
    from freetoken.layers.gguf import (
        GGUFEmbedding,
        GGUFLMHead,
        GGUFLinear,
        gguf_merged_or_plain,
    )

    _require_tp1("module conversion")
    types = _quant_types(model_path)

    def qt(name: str) -> int:
        try:
            return types[name]
        except KeyError as exc:
            raise ValueError(
                f"Qwen4Exp GGUF is missing tensor {name!r}; cannot size packed module"
            ) from exc

    def swap(owner, attr: str, tensor_name: str) -> None:
        old = getattr(owner, attr)
        out_features, in_features = old.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(
                in_features,
                out_features,
                qt(tensor_name),
                has_bias=old.bias is not None,
            ),
        )

    inner = model.model
    embed_type = qt("token_embd.weight")
    embed = GGUFEmbedding(
        config.vocab_size,
        config.hidden_size,
        embed_type,
    )
    inner.embed_tokens = embed

    # Final hyper-connection mixer.
    swap(
        inner.hyper_connection_mixer,
        "input_mix_weight_down",
        "output_hc_down.weight",
    )
    swap(
        inner.hyper_connection_mixer,
        "input_mix_weight_up",
        "output_hc_up.weight",
    )

    gdn = config.linear_attention_group()
    assert gdn is not None
    in_split = [
        2 * gdn.num_key_heads * gdn.key_head_dim
        + gdn.num_value_heads * gdn.value_head_dim,
        gdn.num_value_heads * gdn.value_head_dim,
        gdn.num_value_heads,
        gdn.num_value_heads,
    ]
    qkv_split = [
        config.num_qo_heads * config.head_dim * 2,
        config.num_kv_heads * config.head_dim,
        config.num_kv_heads * config.head_dim,
    ]

    for layer_id, layer in enumerate(inner.layers.op_list):
        # Both hyper-connection blocks.
        for prefix, hc in (
            ("hc_attn", layer.attn_hyper_connection),
            ("hc_ffn", layer.mlp_hyper_connection),
        ):
            swap(hc, "input_mix_weight_down", f"blk.{layer_id}.{prefix}_down.weight")
            swap(hc, "input_mix_weight_up", f"blk.{layer_id}.{prefix}_up.weight")

        # Shared expert (router and shared-expert gate stay dense F32/BF16).
        layer.mlp.shared_expert.gate_up_proj = gguf_merged_or_plain(
            config.hidden_size,
            [
                config.shared_expert_intermediate_size,
                config.shared_expert_intermediate_size,
            ],
            [
                qt(f"blk.{layer_id}.ffn_gate_shexp.weight"),
                qt(f"blk.{layer_id}.ffn_up_shexp.weight"),
            ],
            has_bias=False,
        )
        swap(
            layer.mlp.shared_expert,
            "down_proj",
            f"blk.{layer_id}.ffn_down_shexp.weight",
        )

        if config.is_linear_layer(layer_id):
            layer.linear_attn.in_proj = gguf_merged_or_plain(
                config.hidden_size,
                in_split,
                [
                    qt(f"blk.{layer_id}.attn_qkv.weight"),
                    qt(f"blk.{layer_id}.attn_gate.weight"),
                    qt(f"blk.{layer_id}.ssm_beta.weight"),
                    qt(f"blk.{layer_id}.ssm_alpha.weight"),
                ],
                has_bias=False,
            )
            # out_proj deliberately remains dense for the correctness milestone:
            # its GGUF columns are V-head tiled and are un-tiled after dequant at load.
        else:
            layer.self_attn.qkv_proj = gguf_merged_or_plain(
                config.hidden_size,
                qkv_split,
                [
                    qt(f"blk.{layer_id}.attn_q.weight"),
                    qt(f"blk.{layer_id}.attn_k.weight"),
                    qt(f"blk.{layer_id}.attn_v.weight"),
                ],
                has_bias=False,
            )
            swap(
                layer.self_attn,
                "o_proj",
                f"blk.{layer_id}.attn_output.weight",
            )
            # Indexer q/k projections are BF16 in this Unsloth layout and stay dense.

        if layer.ple is not None:
            swap(layer.ple, "key_proj", f"blk.{layer_id}.ple_key.weight")
            swap(layer.ple, "value_proj", f"blk.{layer_id}.ple_value.weight")

    if config.tie_word_embeddings:
        from freetoken.models.gemma4.gguf import GGUFTiedLMHead

        model.lm_head = GGUFTiedLMHead(embed, embed_type)
    else:
        model.lm_head = GGUFLMHead(
            config.hidden_size,
            config.vocab_size,
            qt("output.weight"),
            has_bias=False,
        )


__all__ = [
    "iter_gguf_weights_impl",
    "convert_qwen4exp_to_gguf",
]
