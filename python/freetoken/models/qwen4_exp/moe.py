from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers.moe import make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _t() -> float:
    return time.perf_counter()


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
            return
        # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
        # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
        # rebuild the routed experts with the fp8_block bank layout.
        super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        from freetoken.utils.gputime import timed

        _t0 = _t()
        with timed("router"):
            router_logits = self.gate.forward(hidden_states)
        _t1 = _t()
        with timed("shared"):
            shared = self.shared_expert.forward(hidden_states)
        _t2 = _t()
        with timed("sgate"):
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        _t3 = _t()
        with timed("routed"):
            routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        _t4 = _t()
        with timed("mul"):
            out = shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)
        if os.path.exists(r"D:\temp\opencode\ft_steptime.flag"):
            try:
                from freetoken.utils.logger import init_logger

                init_logger("freetoken.qwen4exp.moe").info(
                    "MOESPLIT router_ms=%.1f shared_ms=%.1f sgate_ms=%.1f routed_ms=%.1f mul_ms=%.1f",
                    (_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3, (_t3 - _t2) * 1e3,
                    (_t4 - _t3) * 1e3, (time.perf_counter() - _t4) * 1e3,
                )
            except Exception:  # pragma: no cover
                pass
        return out


__all__ = ["Qwen4ExpMoE"]
