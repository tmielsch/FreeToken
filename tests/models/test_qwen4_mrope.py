from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.qwen4_exp.mrope import build_mrope_positions


def test_mrope_position_builder_matches_transformers_reference():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    class Reference:
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = Qwen3VLModel.get_vision_position_ids
        get_rope_index = Qwen3VLModel.get_rope_index

    input_ids = torch.arange(9)
    token_types = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0, 0])
    grid = torch.tensor([[1, 4, 4]])
    actual, delta = build_mrope_positions(input_ids, token_types, grid, 2)
    expected, expected_delta = Reference().get_rope_index(
        input_ids.view(1, -1),
        token_types.view(1, -1),
        image_grid_thw=grid,
    )
    assert torch.equal(actual, expected[:, 0])
    assert delta == int(expected_delta[0, 0])
