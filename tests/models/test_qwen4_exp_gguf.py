from __future__ import annotations

from pathlib import Path

import pytest
import torch

from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.gguf.dequant import GGML_IQ3_XXS, GGML_IQ4_NL, GGML_Q8_0
from freetoken.models.gguf.reader import gguf_config_source, gguf_split_paths
from freetoken.models.qwen4_exp.gguf import parse_gguf_config
from freetoken.models.qwen4_exp.gguf_weights import _resident_name, _ungroup_v


def _metadata() -> dict:
    # Geometry of the released Qwen3.8-Flash-Next, expressed through the
    # llama.cpp qwen4exp GGUF metadata contract. The real Unsloth tensor map has
    # QSA on zero-based decoder layers 3,7,11,...,47.
    ratios = [4 if i % 4 == 3 else 0 for i in range(48)]
    return {
        "qwen4exp.block_count": 48,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.context_length": 262144,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.attention.compress_ratios": ratios,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2048,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.inner_size": 6144,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.ple.image_token_id": 248056,
        # llama.cpp stores one PLE head row width; FreeToken reconstructs the
        # total 16-head embedding width (16 * 160 == 2560).
        "qwen4exp.embedding_length_per_layer_input": 160,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
    }


def _fake_geometry(num_layers: int):
    # Representative dominant signature from the user's Unsloth UD-Q3_K_XL:
    # gate/up IQ3_XXS, down IQ4_NL. Header-only parsing is tested separately
    # against actual files; this unit test deliberately needs no multi-GB GGUF.
    return GGML_Q8_0, tuple(
        (GGML_IQ3_XXS, GGML_IQ4_NL) for _ in range(num_layers)
    )


def test_qwen4exp_gguf_config_matches_released_geometry(monkeypatch):
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.gguf._gguf_geometry",
        lambda _path, num_layers: _fake_geometry(num_layers),
    )
    shim = GgufConfigShim(
        architectures=["Qwen4ExpGGUFForCausalLM"],
        model_path="unused.gguf",
        model_type="qwen4exp",
        metadata=_metadata(),
        vocab_size=248320,
        tie_word_embeddings=False,
    )
    config = parse_gguf_config(shim)

    assert config.num_layers == 48
    assert config.hidden_size == 2560
    assert config.num_qo_heads == 24
    assert config.num_kv_heads == 2
    assert config.head_dim == 256
    assert config.num_experts == 512
    assert config.num_experts_per_tok == 10
    assert config.moe_intermediate_size == 640
    assert config.shared_expert_intermediate_size == 640
    assert config.expert_quant == "gguf"
    assert config.qwen4_args.hc_count == 4
    assert config.qwen4_args.hc_lowrank == 320
    assert config.qwen4_args.mrope_section == (11, 11, 10)
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.qwen4_args.ple_embed_dim == 2560
    assert config.qwen4_args.indexer_compress_ratio == 4
    assert config.qwen4_args.gguf_embed_quant == GGML_Q8_0
    assert len(config.qwen4_args.gguf_expert_types) == 48
    assert config.is_linear_layer(0)
    assert config.is_linear_layer(1)
    assert config.is_linear_layer(2)
    assert not config.is_linear_layer(3)
    assert not config.is_linear_layer(47)


def test_qwen4exp_filtered_resident_iterator_never_selects_ple_or_routed_experts():
    assert not _resident_name("per_layer_token_embd.weight", 48)
    for suffix in (
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ):
        assert not _resident_name(f"blk.7.{suffix}", 48)
    assert _resident_name("blk.7.ffn_gate_shexp.weight", 48)
    assert _resident_name("blk.7.attn_q.weight", 48)


def test_qwen4exp_inverse_v_head_tiling_order():
    # GGUF tiled order [R,K,D] -> FreeToken grouped order [K,R,D].
    tiled = torch.arange(6)
    grouped = _ungroup_v(
        tiled,
        dim=0,
        num_k_heads=2,
        num_v_per_k=3,
        head_dim=1,
    )
    assert grouped.tolist() == [0, 2, 4, 1, 3, 5]


def test_split_gguf_resolves_all_unsloth_style_shards(tmp_path: Path):
    names = [
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
    ]
    for name in names:
        (tmp_path / name).touch()

    paths = gguf_split_paths(str(tmp_path / names[1]))
    assert [Path(p).name for p in paths] == names
    assert Path(gguf_config_source(str(tmp_path))).name == names[0]


def test_split_gguf_fails_if_a_shard_is_missing(tmp_path: Path):
    first = tmp_path / "model-00001-of-00003.gguf"
    first.touch()
    (tmp_path / "model-00003-of-00003.gguf").touch()

    with pytest.raises(FileNotFoundError, match="00002-of-00003"):
        gguf_split_paths(str(first))
