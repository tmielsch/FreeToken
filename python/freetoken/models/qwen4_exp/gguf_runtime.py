"""Runtime support for Unsloth/llama.cpp Qwen3.8 mixed-quant GGUF experts.

This deliberately isolates the variable-width GGUF cache from FreeToken's existing
uniform-bank offload cache. Unsloth Dynamic GGUFs may choose a different ggml quant
per tensor/layer, so each host layer keeps its native packed width while GPU cache
slots are padded to the maximum width for that projection.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import torch

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache

from .model import Qwen4ExpForCausalLM


_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.(?P<proj>ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$"
)
_SUPPORTED_MOE_TYPES = {
    2, 3, 6, 7, 8,              # Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
    10, 11, 12, 13, 14,         # Q2_K..Q6_K
    16, 17, 18, 19, 20, 21, 22, 23, 29,  # IQ decode kernels already vendored by FreeToken
}


@dataclass
class MixedGGUFExpertSources:
    sources: dict[str, list[torch.Tensor]]
    quant_types: dict[str, list[int]]
    host_banks: dict[str, list[object]]


def _resolve_gguf(model_path: str) -> str:
    from freetoken.models.gguf.reader import gguf_config_source, gguf_split_paths

    if os.path.isdir(model_path):
        resolved = gguf_config_source(model_path)
        if resolved is None:
            raise ValueError(f"no unique GGUF family found in {model_path}")
        return resolved
    return gguf_split_paths(model_path)[0]


def load_mixed_gguf_expert_sources(model_path: str, config) -> MixedGGUFExpertSources:
    """Load only routed-expert payloads into per-layer pinned native-byte banks.

    GGUFReader itself mmaps the split files. We only materialize ``t.data`` for the
    three routed expert tensors per layer; the huge PLE table and resident tensors
    are not touched by this loader.
    """
    import gguf

    from freetoken.models.gguf.reader import gguf_split_paths
    from freetoken.moe.host_banks import HostBank

    first = _resolve_gguf(model_path)
    paths = gguf_split_paths(first)
    L = config.num_moe_layers
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size

    sources: dict[str, list[torch.Tensor | None]] = {
        name: [None] * L for name in ("gate", "up", "down")
    }
    quant_types: dict[str, list[int | None]] = {
        name: [None] * L for name in ("gate", "up", "down")
    }
    host_banks: dict[str, list[object | None]] = {
        name: [None] * L for name in ("gate", "up", "down")
    }

    for path in paths:
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            match = _EXPERT_RE.match(tensor.name)
            if match is None:
                continue
            layer = int(match.group("layer"))
            if not (0 <= layer < L):
                raise ValueError(f"expert layer {layer} outside [0, {L})")
            proj = match.group("proj").removeprefix("ffn_").removesuffix("_exps")
            qtype = int(tensor.tensor_type)
            if qtype not in _SUPPORTED_MOE_TYPES:
                qname = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
                raise NotImplementedError(
                    f"{tensor.name} uses GGML type {qname} ({qtype}), which the mixed "
                    "GGUF MoE kernel does not support yet"
                )

            ne = [int(v) for v in tensor.shape]  # ggml order: fastest dim first
            expected = [H, I, E] if proj in ("gate", "up") else [I, H, E]
            if ne != expected:
                raise ValueError(
                    f"unexpected {tensor.name} shape {ne}; expected {expected} in ggml order"
                )
            block, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            if ne[0] % block:
                raise ValueError(
                    f"{tensor.name}: input dim {ne[0]} is not divisible by quant block {block}"
                )
            row_bytes = ne[0] // block * type_size
            rows_per_expert = ne[1]
            bytes_per_expert = row_bytes * rows_per_expert
            if bytes_per_expert % 16:
                raise ValueError(
                    f"{tensor.name}: packed expert size {bytes_per_expert} is not 16-byte aligned"
                )

            flat = np.ascontiguousarray(tensor.data).reshape(-1).view(np.uint8)
            packed = torch.from_numpy(flat).view(E, bytes_per_expert)
            bank = HostBank((E, bytes_per_expert), torch.uint8)
            bank.tensor.copy_(packed)
            if torch.cuda.is_available():
                bank.pin()
            sources[proj][layer] = bank.tensor
            quant_types[proj][layer] = qtype
            host_banks[proj][layer] = bank

    for proj in ("gate", "up", "down"):
        missing = [i for i, value in enumerate(sources[proj]) if value is None]
        if missing:
            raise RuntimeError(f"GGUF is missing routed expert {proj} tensors for layers {missing}")

    return MixedGGUFExpertSources(
        sources={name: list(values) for name, values in sources.items()},  # type: ignore[arg-type]
        quant_types={name: [int(v) for v in values] for name, values in quant_types.items()},
        host_banks={name: list(values) for name, values in host_banks.items()},
    )


class MixedGGUFOffloadMoeCache(OffloadMoeCache):
    """FreeToken LRU bookkeeping + variable-width native GGUF byte slots."""

    def __init__(self, *args, **kwargs):
        # Let the base initialize all LRU tensors using a known schema, then replace
        # only the bank storage/copy semantics. Prefill overlap is intentionally off
        # for the first mixed-GGUF path; Qwen4Exp already disables CUDA graphs for PLE.
        kwargs["quant_format"] = "q4_0"
        kwargs["prefill_overlap"] = False
        kwargs["prefill_hit_d2d"] = False
        super().__init__(*args, **kwargs)
        self.quant_format = "gguf_mixed"
        self.bank_schema = ("gate", "up", "down")
        self.gguf_quant_types: dict[str, list[int]] = {}
        self._gguf_host_banks = None

    def set_mixed_sources(self, bundle: MixedGGUFExpertSources) -> None:
        self.bank_sources = bundle.sources
        self.gguf_quant_types = bundle.quant_types
        self._gguf_host_banks = bundle.host_banks  # lifetime owner for registrations/mmaps
        self.bank_caches = {}
        self.banks = []

        for name in self.bank_schema:
            per_layer = self.bank_sources[name]
            if len(per_layer) != self.num_layers:
                raise ValueError(
                    f"GGUF bank {name}: {len(per_layer)} layers != {self.num_layers}"
                )
            max_bytes = max(int(source.shape[1]) for source in per_layer)
            for layer_id, source in enumerate(per_layer):
                if source.dtype != torch.uint8 or source.ndim != 2:
                    raise TypeError(
                        f"GGUF bank {name} layer {layer_id} must be uint8 [E, bytes], "
                        f"got {source.dtype} {tuple(source.shape)}"
                    )
                if source.shape[0] != self.num_experts:
                    raise ValueError(
                        f"GGUF bank {name} layer {layer_id}: {source.shape[0]} experts "
                        f"!= {self.num_experts}"
                    )
                if source.shape[1] % 16:
                    raise ValueError(
                        f"GGUF bank {name} layer {layer_id}: row bytes must be 16-byte aligned"
                    )
            cache = torch.empty(
                (self.cache_size, max_bytes), dtype=torch.uint8, device=self.device
            )
            self.bank_caches[name] = cache
            self.banks.append((per_layer, cache))

        # Base fused-copy pointers assume one identical feature width across layers;
        # mixed GGUF intentionally violates that contract and uses our strided copy below.
        self._copy_fused_ok = False

    def copy_missing(self) -> None:
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self._unpinned_layers:
            raise NotImplementedError(
                "mixed GGUF currently requires pinned expert banks; CPU/pageable layers "
                "will be added after the GPU offload path is validated"
            )

        from freetoken.kernel.strided_index_copy import strided_index_copy_jit

        for name in self.bank_schema:
            strided_index_copy_jit(
                self.bank_caches[name],
                self.evict_slots,
                self.bank_sources[name][layer_id],
                self.src_indices,
                self.num_indices,
            )

    def rebuild(self, cache_size: int) -> None:
        raise NotImplementedError(
            "runtime MoE cache resizing is not yet implemented for mixed GGUF; "
            "restart with the desired --moe-cache-size"
        )


class MixedGGUFOffloadMoELayer(OffloadMoELayer):
    def _expert_gemm(
        self,
        cache,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        views,
        n,
        alphas,
        is_prefill,
    ):
        from freetoken.moe.fused_gguf_mixed import fused_experts_gguf_mixed

        gate, up, down = views
        q = cache.gguf_quant_types
        return fused_experts_gguf_mixed(
            hidden_states,
            gate,
            up,
            down,
            topk_weights,
            topk_ids,
            gate_type=q["gate"][self.layer_id],
            up_type=q["up"][self.layer_id],
            down_type=q["down"][self.layer_id],
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            activation=self.activation,
        )


class Qwen4ExpGGUFForCausalLM(Qwen4ExpForCausalLM):
    """Qwen4Exp model variant whose routed experts use the mixed GGUF cache."""

    def __init__(self, config):
        from freetoken.moe import is_offload_moe_backend

        if not is_offload_moe_backend(config.moe_backend):
            raise ValueError(
                "Qwen3.8 mixed GGUF currently requires --moe-backend offload"
            )
        super().__init__(config)
        for layer in self.model.layers.op_list:
            old = layer.mlp.experts
            if not isinstance(old, OffloadMoELayer):
                raise TypeError(
                    f"expected OffloadMoELayer for GGUF Qwen4Exp, got {type(old).__name__}"
                )
            layer.mlp.experts = MixedGGUFOffloadMoELayer(
                layer_id=old.layer_id,
                num_experts=old.num_experts,
                top_k=old.top_k,
                hidden_size=old.hidden_size,
                intermediate_size=old.intermediate_size,
                renormalize=old.renormalize,
                activation=old.activation,
                apply_router_weight_on_input=old.apply_router_weight_on_input,
            )

    def make_offload_moe_cache(self, engine_config, device):
        if engine_config.moe_backend != "offload" or engine_config.moe_cpu_layers is not None:
            raise NotImplementedError(
                "mixed GGUF currently supports the pure GPU offload backend only"
            )
        if engine_config.moe_prefill_overlap:
            # The custom cache deliberately uses synchronous layer materialization for now.
            object.__setattr__(engine_config, "moe_prefill_overlap", False)
        bundle = load_mixed_gguf_expert_sources(
            engine_config.model_path, engine_config.model_config
        )
        cache = MixedGGUFOffloadMoeCache(
            num_layers=engine_config.model_config.num_moe_layers,
            num_experts=engine_config.model_config.num_experts,
            cache_size=engine_config.moe_cache_size,
            device=device,
            cache_policy=engine_config.moe_cache_policy,
            decode_target="gpu",
            hybrid_max_fetch=engine_config.moe_hybrid_max_fetch,
        )
        cache.set_mixed_sources(bundle)
        return cache

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        # GGUF PLE gets its own mmap/gather loader; do not invoke the inherited
        # safetensors/HuggingFace loader on a local 90 GB split GGUF.
        if dummy:
            return
        # Filled in by the GGUF PLE phase. Keeping this explicit makes a partial
        # implementation fail at first PLE use rather than accidentally downloading HF.
        return


__all__ = [
    "MixedGGUFExpertSources",
    "MixedGGUFOffloadMoeCache",
    "MixedGGUFOffloadMoELayer",
    "Qwen4ExpGGUFForCausalLM",
    "load_mixed_gguf_expert_sources",
]
