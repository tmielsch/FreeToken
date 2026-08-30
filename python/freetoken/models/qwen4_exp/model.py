"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger, nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

_model_logger = init_logger("freetoken.qwen4exp.model")

# set to a list by Qwen4ExpModel.forward while the hidden-dump flag is active; each layer
# appends its mid-block states (attn_in / attn_out / mlp_in / mlp_out) for the layer harness
_mid_collector: list | None = None


def _prof_on() -> bool:
    return os.path.exists(r"D:\temp\opencode\ft_steptime.flag")


def _hidden_dump_on() -> bool:
    """Layer-dump harness gate (prefill only): dump the per-layer hyper-connection residual
    to ``D:\\temp\\opencode\\ft_hidden_dump\\<input-hash>.npz`` as fp32. See the flag-gated
    LOGIT-capture pattern in engine.py; this one is one-shot per prefill keyed by input ids
    so the warmup prefill cannot poison the harness's dump."""
    return os.path.exists(r"D:\temp\opencode\ft_hidden_dump.flag")


def _hidden_dump_decode_on() -> bool:
    """Decode-stage variant of the layer-dump (prefill-vs-decode invariance harness): when this
    flag is ALSO present, dump the per-layer residual for every decode step, keyed by the
    absolute context position (``batch.positions[-1]``), so a full-prefill dump and an
    incremental-decode dump of the same sequence are directly comparable position-by-position."""
    return os.path.exists(r"D:\temp\opencode\ft_hidden_dump_decode.flag")


def _prefill_barrier() -> bool:
    return os.getenv("FREETOKEN_PREFILL_BARRIER", "1") not in ("0", "false", "no", "off")


def _t() -> float:
    return time.perf_counter()


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )
        self._prof_acc: dict[str, float] | None = None

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        from freetoken.utils.gputime import timed

        global _mid_collector
        prof = _prof_on()
        if prof and self._prof_acc is None:
            self._prof_acc = {"ple": 0.0, "attn": 0.0, "mlp": 0.0, "hc": 0.0}
        acc = self._prof_acc
        t0 = _t() if prof else None
        if self.ple is not None:
            with timed("ple"):
                hidden = hidden + self.ple.forward(hidden, batch)
        t1 = _t() if prof else None
        with timed("hc_attn_mix"):
            block_input, inject = self.attn_hyper_connection.mix(hidden)
        t2 = _t() if prof else None
        if self._is_linear:
            with timed("attn_linear"):
                block_output = self.linear_attn.forward(block_input)
        else:
            with timed("attn_qsa"):
                block_output = self.self_attn.forward(block_input, batch)
        if _mid_collector is not None:
            _mid_collector.append(
                (self._layer_id, "attn_in", block_input.detach().float().cpu())
            )
            _mid_collector.append(
                (self._layer_id, "attn_out", block_output.detach().float().cpu())
            )
        t3 = _t() if prof else None
        with timed("hc_attn_combine"):
            hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        t4 = _t() if prof else None
        with timed("hc_mlp_mix"):
            block_input, inject = self.mlp_hyper_connection.mix(hidden)
        if _mid_collector is not None:
            _mid_collector.append(
                (self._layer_id, "mlp_in", block_input.detach().float().cpu())
            )
        with timed("mlp"):
            blk = self.mlp.forward(block_input)
        if _mid_collector is not None:
            _mid_collector.append(
                (self._layer_id, "mlp_out", blk.detach().float().cpu())
            )
        t5 = _t() if prof else None
        with timed("hc_mlp_combine"):
            out = self.mlp_hyper_connection.combine(hidden, blk, inject)
        t6 = _t() if prof else None
        if acc is not None:
            acc["ple"] += t1 - t0
            acc["attn"] += t3 - t2
            acc["mlp"] += t5 - t4
            acc["hc"] += (t2 - t1) + (t4 - t3) + (t6 - t5)
        return out


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)
        self._ple_host_verified = False

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def stage_ple_decode(self, batch: Batch, device: torch.device) -> None:
        """Host-side PLE staging for a decode forward, run OUTSIDE the graph/model forward.
        Computes the ngram row ids on the host from each request's token history and gathers
        the GGUF rows into the persistent device staging buffer (``GGUFPLETableBackend.stage``).
        The in-forward ``lookup`` then only dequantizes that buffer (capture-safe, no D2H sync).
        On by default (CUDA-graph decode needs it); FREETOKEN_PLE_HOST=0 disables. No-op when
        the batch is not a decode.
        """
        if not self._ple or not batch.is_decode:
            return
        if os.getenv("FREETOKEN_PLE_HOST", "1") in ("0", "false", "no", "off"):
            return
        from .ple import build_ple_metadata, host_decode_ngram_ids

        _emb = self._ple[0].ple_embedding
        meta = build_ple_metadata(batch, self._ple[0].args, device)
        _host_ids = host_decode_ngram_ids(_emb, meta, batch)
        if os.getenv("FREETOKEN_PLE_HOST_VERIFY", "0") == "1" and not self._ple_host_verified:
            # One- shot safety net: the per-step .to("cpu") sync stalls the scheduler loop
            # enough to starve the detokenizer worker's keepalive, so only check the first
            # decode step for host-vs-device id agreement.
            _dev_ids = _emb.row_ids(meta).to("cpu", non_blocking=False)
            self._ple_host_verified = True
            if not torch.equal(_host_ids, _dev_ids):
                _model_logger.warning(
                    "PLE_HOST mismatch dev=%s host=%s",
                    _dev_ids[0].tolist()[:4], _host_ids[0].tolist()[:4],
                )
            else:
                _model_logger.info("PLE_HOST verified: host ngram ids match the device reference.")
        _emb.table.stage(_host_ids, device)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        global _mid_collector
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        hdump = None
        if _hidden_dump_on() and not batch.is_decode:
            # one-shot diagnostics: capture the fp32 residual after every layer, plus the
            # initial embedding stream, keyed by input ids so the warmup prefill is ignored
            hdump = {"key": tuple(input_ids.cpu().tolist()), "states": [], "mid": []}
            _mid_collector = hdump["mid"]
        elif _hidden_dump_decode_on() and batch.is_decode:
            try:
                pos = int(batch.positions[-1].cpu().item())
            except Exception:  # pragma: no cover
                pos = -1
            hdump = {"pos": pos, "states": []}
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            _tbl = self._ple[0].ple_embedding.table
            # The capture-safe decode path stages rows OUTSIDE the forward (stage_ple_decode
            # from the engine, before replay). Any other path (prefill, non-host decode) must
            # invalidate a stale staging so lookups never consume leftovers.
            if not (
                os.getenv("FREETOKEN_PLE_HOST", "1") not in ("0", "false", "no", "off")
                and batch.is_decode
            ):
                _tbl._staged_count = 0
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            if _prefill_barrier() and not batch.is_decode:
                # Shallow-queue barrier: without it the eager prefill enqueues ~340
                # ops per layer (tvm-ffi + triton launches) far ahead of the GPU; the
                # resulting deep driver queue makes every launch block ~0.3-0.7 ms on
                # Windows (measured ~240 ms/layer -> 11.5 s for a 10-token prefill)
                # while the true GPU cost is ~5 ms/layer. FreeToken's graphs only
                # cover decode; the prefill stays eager, so pace it layer-wise.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            hidden = layer.forward(hidden, batch)
            if hdump is not None:
                hdump["states"].append(hidden.detach().float().cpu())
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        from freetoken.utils.gputime import flush as _gputime_flush

        _gputime_flush()
        if _prof_on():
            try:
                acc = {"ple": 0.0, "attn": 0.0, "mlp": 0.0, "hc": 0.0}
                for layer in self.layers.op_list:
                    if layer._prof_acc is not None:
                        for k in acc:
                            acc[k] += layer._prof_acc[k]
                        layer._prof_acc = None
                total = sum(acc.values())
                _model_logger.info(
                    "MODELPROF total_ms=%.1f ple_ms=%.1f attn_ms=%.1f mlp_ms=%.1f hc_ms=%.1f",
                    total * 1e3, acc["ple"] * 1e3, acc["attn"] * 1e3,
                    acc["mlp"] * 1e3, acc["hc"] * 1e3,
                )
            except Exception:  # pragma: no cover
                pass
        if hdump is not None:
            try:
                import hashlib
                import numpy as np

                out_dir = r"D:\temp\opencode\ft_hidden_dump"
                os.makedirs(out_dir, exist_ok=True)
                if "pos" in hdump:
                    fname = f"decode_pos_{hdump['pos']:04d}.npz"
                    arrays = {"pos": np.asarray(hdump["pos"], dtype=np.int32)}
                    for li, st in enumerate(hdump["states"]):
                        arrays[f"layer_{li}"] = st.numpy()
                    np.savez_compressed(os.path.join(out_dir, fname), **arrays)
                    _model_logger.info("HIDDENDUMP decode pos=%d layers=%d", hdump["pos"], len(hdump["states"]))
                else:
                    key = "-".join(str(i) for i in hdump["key"])
                    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
                    arrays = {"input_ids": np.asarray(hdump["key"], dtype=np.int32)}
                    for li, st in enumerate(hdump["states"]):
                        arrays[f"layer_{li}"] = st.numpy()
                    for li, stage, st in hdump["mid"]:
                        arrays[f"{stage}_{li}"] = st.numpy()
                    np.savez_compressed(os.path.join(out_dir, f"{digest}.npz"), **arrays)
                    _model_logger.info("HIDDENDUMP wrote ntok=%d layers=%d key=%s", len(hdump["key"]), len(hdump["states"]), digest)
            except Exception:  # pragma: no cover
                pass
        _mid_collector = None
        return self.hyper_connection_mixer.mix(hidden)[0]


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

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

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))

    def stage_ple_decode(self, batch, device) -> None:
        return self.model.stage_ple_decode(batch, device)


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
