from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_experts_gguf_mixed(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,
    up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    gate_type: int,
    up_type: int,
    down_type: int,
    intermediate_size: int,
    hidden_size: int,
    activation: str,
) -> torch.Tensor:
    """Routed MoE over native mixed GGUF expert slots.

    ``gate_q``/``up_q``/``down_q`` are flat uint8 slot caches. Each cache slot is
    padded to the largest packed expert of that projection across all layers;
    the CUDA kernel receives that slot stride separately from the current layer's
    native GGML quant type. This preserves Unsloth Dynamic per-layer quants without
    expanding the host experts to BF16.
    """
    if activation != "silu":
        raise NotImplementedError(
            f"mixed GGUF experts currently support silu only, got {activation!r}"
        )

    from freetoken.kernel.gguf_mixed import ggml_moe_a8_vec_strided

    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]

    gate = ggml_moe_a8_vec_strided(
        hidden_states,
        gate_q,
        topk_ids,
        top_k,
        gate_type,
        intermediate_size,
        num_tokens,
        gate_q.shape[1],
    )
    up = ggml_moe_a8_vec_strided(
        hidden_states,
        up_q,
        topk_ids,
        top_k,
        up_type,
        intermediate_size,
        num_tokens,
        up_q.shape[1],
    )
    inter = F.silu(gate) * up

    out = ggml_moe_a8_vec_strided(
        inter,
        down_q,
        topk_ids,
        1,
        down_type,
        hidden_size,
        num_tokens * top_k,
        down_q.shape[1],
    )
    out = out.reshape(num_tokens, top_k, hidden_size)
    out = out * topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype)
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf_mixed"]
