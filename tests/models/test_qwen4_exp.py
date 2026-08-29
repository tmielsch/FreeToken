from __future__ import annotations

import pytest
import torch

import freetoken.models.qwen4_exp as qwen4_exp
from freetoken.models.qwen4_exp.weight import _rename, _try_fuse
from freetoken.models.register import get_model_spec


def test_qwen4_exports_nvfp4_loader_hooks():
    assert callable(qwen4_exp.load_nvfp4_expert_sources)
    assert callable(qwen4_exp.load_nvfp4_expert_sources_parallel)


def test_qwen4_exports_gguf_loader_hooks():
    assert callable(qwen4_exp.parse_gguf_config)
    assert callable(qwen4_exp.iter_gguf_weights)
    assert callable(qwen4_exp.gguf_quant_inventory)


def test_qwen4_registry_entry():
    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def test_qwen4_gguf_registry_entry():
    spec = get_model_spec("Qwen4ExpGGUFForCausalLM")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpGGUFForCausalLM"


def test_qwen4_weight_names():
    assert _rename("model.language_model.layers.1.ple.key_proj.weight") == (
        "model.layers.1.ple.key_proj.weight"
    )
    # Served text-only: the vision tower is skipped, not renamed.
    assert _rename("model.visual.blocks.0.attn.qkv.weight") is None
    assert _rename("model.language_model.layers.3.self_attn.indexer.q_layernorm.weight") == (
        "model.layers.3.self_attn.indexer.q_layernorm.weight"
    )


def test_qwen4_projection_fusion_order():
    buffers = {}
    base = "model.layers.3.self_attn."
    parts = [
        ("q_proj.weight", torch.full((2, 3), 1.0)),
        ("k_proj.weight", torch.full((1, 3), 2.0)),
        ("v_proj.weight", torch.full((1, 3), 3.0)),
    ]
    assert _try_fuse(base + parts[0][0], parts[0][1], buffers) == ()
    assert _try_fuse(base + parts[1][0], parts[1][1], buffers) == ()
    name, fused = _try_fuse(base + parts[2][0], parts[2][1], buffers)
    assert name == base + "qkv_proj.weight"
    assert fused[:, 0].tolist() == [1.0, 1.0, 2.0, 3.0]
