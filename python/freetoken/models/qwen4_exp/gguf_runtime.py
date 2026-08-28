"""Qwen4Exp GGUF runtime glue built on FreeToken's generic GGUF geometry cache.

The earlier branch-local prototype implemented its own variable-stride cache and
copy path. That is intentionally gone: upstream #199 already teaches the normal
``OffloadMoeCache`` to keep compact heterogeneous host rows and carve decode pools
per row geometry. Qwen4Exp only supplies its per-layer ggml types and expert banks.

This class is deliberately not registered until dense/PLE GGUF weight mapping is
complete, so these hooks cannot accidentally make a partial model serve.
"""

from __future__ import annotations

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache

from .gguf_experts import dummy_gguf_expert_sources, load_gguf_expert_sources
from .model import Qwen4ExpForCausalLM


class Qwen4ExpGGUFForCausalLM(Qwen4ExpForCausalLM):
    """Qwen4Exp with native mixed-type GGUF routed experts."""

    def __init__(self, config):
        from freetoken.moe import is_offload_moe_backend

        if not is_offload_moe_backend(config.moe_backend):
            raise ValueError(
                "Qwen3.8 GGUF currently requires an offload-family MoE backend"
            )
        super().__init__(config)

        types = getattr(config.qwen4_args, "gguf_expert_types", None)
        if not types or len(types) != config.num_moe_layers:
            raise ValueError(
                "Qwen3.8 GGUF routed expert types were not recovered for every layer"
            )

        # The generic OffloadMoELayer's quant_format == "gguf" branch reads these
        # four attributes to dispatch the correct ggml MoE kernel for each layer.
        for layer_id, layer in enumerate(self.model.layers.op_list):
            experts = layer.mlp.experts
            if not isinstance(experts, OffloadMoELayer):
                raise TypeError(
                    f"Qwen3.8 GGUF layer {layer_id}: expected OffloadMoELayer, "
                    f"got {type(experts).__name__}"
                )
            gate_up_type, down_type = types[layer_id]
            experts.gguf_gate_up_type = gate_up_type
            experts.gguf_down_type = down_type
            experts.gguf_gate_up_rows = 2 * config.moe_intermediate_size
            experts.gguf_down_rows = config.hidden_size

    def make_offload_moe_cache(self, engine_config, device):
        """Build the normal #199 heterogeneous-row GGUF cache for Qwen4Exp."""
        if engine_config.moe_backend != "offload" or engine_config.moe_cpu_layers is not None:
            raise NotImplementedError(
                "Qwen3.8 GGUF first milestone supports pure GPU expert offload only; "
                "CPU/hybrid layers come after direct offload validation"
            )

        # Geometry decode pools are independent of the older whole-layer prefill
        # overlap path. Keep prefill synchronous for the first correctness milestone.
        if engine_config.moe_prefill_overlap:
            object.__setattr__(engine_config, "moe_prefill_overlap", False)

        model_config = engine_config.model_config
        sources = (
            dummy_gguf_expert_sources(model_config)
            if engine_config.use_dummy_weight
            else load_gguf_expert_sources(engine_config.model_path, model_config)
        )
        cache = OffloadMoeCache(
            num_layers=model_config.num_moe_layers,
            num_experts=model_config.num_experts,
            cache_size=engine_config.moe_cache_size,
            device=device,
            cache_policy=engine_config.moe_cache_policy,
            prefill_overlap=False,
            prefill_hit_d2d=False,
            quant_format="gguf",
            decode_target="gpu",
            hybrid_max_fetch=engine_config.moe_hybrid_max_fetch,
            geometry_pool_top_k=model_config.num_experts_per_tok,
            geometry_pool_max_batch=max(
                engine_config.max_running_req,
                engine_config.cuda_graph_max_bs or 0,
                1,
            ),
        )
        cache.set_bank_sources(sources)
        cache.set_alphas(None, None)
        return cache

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        if dummy:
            # The inherited PLE object knows how to become a zero-producing dummy
            # without opening safetensors.
            self.model.load_host_weights(model_path, dummy=True)
            return
        raise NotImplementedError(
            "Qwen3.8 GGUF PLE host embedding mapping is not wired yet; "
            "the runtime class stays unregistered until that phase is complete"
        )


__all__ = ["Qwen4ExpGGUFForCausalLM"]
