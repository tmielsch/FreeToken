"""Host-mmapped PLE n-gram table for Qwen3.8 Flash Next GGUF checkpoints.

The GGUF stores the complete n-gram hash table as one enormous quantized tensor
(``per_layer_token_embd.weight``).  It must never be materialized or copied as a
whole: ``GGUFReader`` exposes quantized tensor payloads as NumPy views over its
file memmap.  This backend implements ``ple.PLETableBackend`` on top of that
view -- it gathers only the rows requested by the current token batch into a tiny
pinned staging buffer and dequantizes those few rows on the GPU, so the
multi-GiB source stays file-backed.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_UNQUANTIZED,
    row_bytes,
)

_DTYPE_FOR_GGML = {
    GGML_F32: torch.float32,
    GGML_F16: torch.float16,
    GGML_BF16: torch.bfloat16,
}


def _long_tuple(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    try:
        return tuple(int(v) for v in value.tolist())
    except AttributeError:
        return (int(value),)


class GGUFPLETableBackend:
    """``PLETableBackend`` whose packed rows stay in the GGUF file mmap.

    ``lookup(row_ids, out)`` index-selects the packed rows on the CPU (zero-copy
    view, piped through a small pinned staging buffer), then dequantizes on
    ``row_ids.device``.  ``out`` is optional -- giving it reuses the graph buffer
    exactly like ``PinnedUVATable``.  The required-row bound and per-head
    constants come from the same ``qwen4exp.ple.*`` metadata the resident
    iterator feeds ``NGramEmbedding``.
    """

    def __init__(self, model_path: str, args) -> None:
        import gguf

        from .gguf_weights import gguf_ngram_constants
        from freetoken.models.gguf.reader import gguf_config_source, gguf_split_paths

        self.num_rows = 0
        self.head_dim = args.ngram_head_dim
        self.dtype = torch.bfloat16

        source = gguf_config_source(model_path)
        if source is None:
            raise ValueError(f"cannot resolve Qwen4Exp GGUF source from {model_path}")

        multipliers, vocab_sizes, offsets = gguf_ngram_constants(model_path)
        expected_heads = args.num_ngram_heads
        if len(multipliers) != args.ngram_size:
            raise ValueError(
                f"PLE has {len(multipliers)} layer multipliers, expected {args.ngram_size}"
            )
        if len(vocab_sizes) != expected_heads or len(offsets) != expected_heads:
            raise ValueError(
                f"PLE has {len(vocab_sizes)} vocab sizes / {len(offsets)} offsets, "
                f"expected {expected_heads} each"
            )
        required_rows = int(offsets[-1] + vocab_sizes[-1])

        found = None
        readers = []
        quant_type = None
        for path in gguf_split_paths(source):
            reader = gguf.GGUFReader(path, mode="c")
            readers.append(reader)
            for tensor in reader.tensors:
                if tensor.name != "per_layer_token_embd.weight":
                    continue
                if found is not None:
                    raise ValueError(
                        "duplicate per_layer_token_embd.weight across GGUF shards"
                    )
                found = tensor

        if found is None:
            raise RuntimeError(
                "Qwen4Exp GGUF has PLE metadata but no per_layer_token_embd.weight"
            )
        quant_type = int(found.tensor_type)
        self._quant_type = quant_type
        self._row_bytes = row_bytes(self.head_dim, quant_type)

        ne = [int(v) for v in found.shape]  # ggml order: [row_dim, rows]
        if len(ne) != 2 or ne[0] != self.head_dim:
            raise ValueError(
                f"unexpected PLE GGUF shape {ne}; expected [{self.head_dim}, rows]"
            )
        rows = ne[1]
        expected_bytes = rows * self._row_bytes
        if int(found.n_bytes) != expected_bytes:
            raise ValueError(
                f"PLE table byte size {int(found.n_bytes)} != "
                f"{rows} * {self._row_bytes} = {expected_bytes}"
            )

        # ReaderTensor.data is already a view into GGUFReader.data (np.memmap).
        # mode="c" keeps it zero-copy/copy-on-write while giving torch a writable view.
        # Do NOT call np.ascontiguousarray here: that would allocate ~29 GiB.
        data = found.data
        if data.dtype != np.uint8:
            # Published PLE is IQ4_NL; quantized GGUF payloads are byte views.
            # An unquantized future checkpoint can be supported separately.
            raise TypeError(
                f"PLE mmap path expects quantized uint8 payload, got {data.dtype}"
            )
        if not data.flags.c_contiguous:
            raise RuntimeError(
                "PLE GGUF payload is not C-contiguous; refusing a reshape that "
                "could silently materialize the full table"
            )
        flat = data.reshape(-1)
        packed_np = flat.reshape(rows, self._row_bytes)
        if not np.shares_memory(packed_np, data):
            raise RuntimeError(
                "PLE row reshape stopped sharing the GGUF mmap; refusing full-table copy"
            )

        # Zero-copy CPU tensor over the file mmap. It is read-only in practice;
        # lookup only index_selects from it.
        self._packed_rows = torch.from_numpy(packed_np)
        self.num_rows = rows
        # Keep all readers alive so the memmap backing the selected tensor remains valid.
        self._readers = readers

        if self.num_rows < required_rows:
            raise RuntimeError(
                f"PLE table has {self.num_rows} rows, metadata needs {required_rows}"
            )

    def prefetch(self, row_ids: torch.Tensor) -> None:
        # CPU-side gather; staging happens in lookup. Kept for protocol symmetry.
        return None

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        ids = row_ids.reshape(-1)
        if ids.numel() == 0:
            return torch.empty(0, self.head_dim, device=row_ids.device, dtype=self.dtype)
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.num_rows:
            raise IndexError(
                f"PLE row id range [{lo}, {hi}] outside [0, {self.num_rows})"
            )

        # Gather into a tiny pinned staging buffer: 16 rows/token * 90 bytes/row
        # for the current IQ4_NL model. The ~29 GiB source remains file-backed.
        staging = torch.empty(
            (ids.numel(), self._row_bytes),
            dtype=torch.uint8,
            pin_memory=torch.cuda.is_available(),
        )
        torch.index_select(self._packed_rows, 0, ids, out=staging)

        if os.path.exists(r"D:\temp\opencode\ft_debug_logits.flag"):
            try:
                from freetoken.utils.logger import init_logger

                _log = init_logger("freetoken.qwen4exp.ple")
                _nids = ids.numel()
                _head_ids = ids[-3:].tolist() if _nids else []
                _span = (lo, hi)
                _log.info(
                    "GGUF_PLE n=%d ngram_ids_tail=%s span=%s",
                    _nids, str(_head_ids), _span,
                )
            except Exception:  # pragma: no cover
                pass

        if self._quant_type in GGML_UNQUANTIZED:
            raw_dtype = _DTYPE_FOR_GGML[self._quant_type]
            values = staging.view(raw_dtype).to(device=row_ids.device, dtype=self.dtype)
        else:
            if row_ids.device.type != "cuda":
                raise RuntimeError(
                    "Qwen4Exp GGUF PLE quantized rows currently require CUDA dequantization"
                )
            from freetoken.kernel.gguf import ggml_dequantize

            packed = staging.to(
                device=row_ids.device,
                non_blocking=staging.is_pinned(),
            )
            values = ggml_dequantize(
                packed,
                self._quant_type,
                ids.numel(),
                self.head_dim,
                self.dtype,
            )

        values = values.reshape(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out


__all__ = ["GGUFPLETableBackend"]
