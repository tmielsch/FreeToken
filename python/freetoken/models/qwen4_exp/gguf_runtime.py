"""Qwen4Exp GGUF runtime glue built on FreeToken's generic GGUF geometry cache.

The earlier branch-local prototype implemented its own variable-stride cache and
copy path. That is intentionally gone: upstream #199 already teaches the normal
``OffloadMoeCache`` to keep compact heterogeneous host rows and carve decode pools
per row geometry. Qwen4Exp only supplies its per-layer ggml types and expert banks.
"""

from __future__ import annotations

from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache

from .gguf_experts import dummy_gguf_expert_sources, load_gguf_expert_sources
from .model import Qwen4ExpForCausalLM


class Qwen4ExpGGUFForCausalLM(Qwen4ExpForCausalLM):
    """Qwen4Exp with native mixed-type GGUF resident weights + routed experts."""

    def __init__(self, config):
        from freetoken.moe import is_offload_moe_backend

        if not is_offload_moe_backend(config.moe_backend):
            raise ValueError(
                "Qwen3.8 GGUF currently requires an offload-family MoE backend"
            )
        super().__init__(config)

        model_path = getattr(config.qwen4_args, "gguf_model_path", None)
        if not model_path:
            raise ValueError("Qwen3.8 GGUF config does not carry its source GGUF path")

        # Swap the resident Q8/Q6/mixed-GDN modules before the engine collects
        # state_dict(), so packed qweight buffers have their exact per-tensor sizes.
        from .gguf_weights import convert_qwen4exp_to_gguf

        convert_qwen4exp_to_gguf(self, config, model_path=model_path)

        # The PLE table is host state rather than an ordinary state_dict tensor:
        # the hash constants ride the resident iterator (GGUF metadata), and the
        # table attach happens in load_host_tables. Record the quant type of the
        # mmap'd table so the backend can validate the checkpoint later.
        from .gguf import _tensor_types_header_only

        tensor_types = _tensor_types_header_only(model_path)
        ple_type = tensor_types.get("per_layer_token_embd.weight")
        if config.qwen4_args.ple_layer_ids and ple_type is None:
            raise ValueError(
                "Qwen3.8 GGUF declares PLE layers but has no per_layer_token_embd.weight"
            )
        self._ple_quant_type = int(ple_type) if ple_type is not None else None

        types = getattr(config.qwen4_args, "gguf_expert_types", None)
        if not types or len(types) != config.num_moe_layers:
            raise ValueError(
                "Qwen3.8 GGUF routed expert types were not recovered for every layer"
            )

        # The generic OffloadMoELayer's quant_format == "gguf" branch reads these
        # attributes to dispatch the correct ggml MoE kernel for each layer.
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

    def load_host_tables(self, engine_config):
        """Attach the PLE n-gram table: pinned GGUF-mmap backend, or zeros for dummy weights.

        Mirrors ``Qwen4ExpForCausalLM.load_host_tables``; the table cannot be a
        pinned resident bank because it is the quantized ``per_layer_token_embd``
        mmap (see ``gguf_ple.GGUFPLETableBackend``). Returns the pinned host bytes
        the engine reserves from its pin budget (0: staging is a transient buffer).
        """
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants
        from .gguf_ple import GGUFPLETableBackend

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        table = GGUFPLETableBackend(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the GGUF readers; keep the mmap alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(table)
        return 0


__all__ = ["Qwen4ExpGGUFForCausalLM"]
