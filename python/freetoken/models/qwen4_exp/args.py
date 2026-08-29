from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen4VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    in_channels: int
    hidden_act: str
    deepstack_visual_indexes: tuple[int, ...]


__all__ = ["Qwen4VisionConfig"]
