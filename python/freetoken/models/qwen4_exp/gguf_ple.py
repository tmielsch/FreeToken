"""Host-mmap PLE embedding for Qwen3.8 Flash Next GGUF.

The GGUF stores the complete n-gram hash table as one enormous quantized tensor
(``per_layer_token_embd.weight``).  It must never be materialized or copied as a
whole.  ``GGUFReader`` exposes quantized tensor payloads as NumPy views over its
file memmap; this adapter keeps that view alive, gathers only the rows requested
by the current token batch, and dequantizes those tiny row batches on the GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from freetoken.models.gguf.dequant import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    GGML_UNQUANTIZED,
    row_bytes,
)

from .model import _HostNGramEmbedding


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


class GGUFHostNGramEmbedding(_HostNGramEmbedding):
    """Drop-in PLE embedding whose hash-table rows stay in the GGUF file mmap."""

    def __init__(self, config, layer_id: int, quant_type: int):
        super().__init__(config, layer_id)

        # In the HF checkpoint these are ordinary tensors and therefore public
        # state_dict entries. llama.cpp moves them into GGUF metadata instead.
        # Remove the public placeholders so BaseOP does not demand nonexistent
        # tensor weights; load_host_weights installs private CPU constants.
        del self.layer_multipliers
        del self.ngram_heads_vocab_sizes
        del self.ngram_heads_offsets

        self._quant_type = int(quant_type)
        self._packed_rows: torch.Tensor | None = None
        self._row_bytes = row_bytes(self.head_dim, self._quant_type)
        self._row_count = 0
        self._readers: list[object] = []

    def load_host_weights(self, model_path: str, *, dummy: bool = False) -> None:
        if dummy:
            self._dummy = True
            return

        import gguf

        from freetoken.models.gguf.reader import (
            gguf_config_source,
            gguf_split_paths,
            load_gguf_metadata,
        )

        source = gguf_config_source(model_path)
        if source is None:
            raise ValueError(f"cannot resolve Qwen4Exp GGUF source from {model_path}")

        metadata = load_gguf_metadata(source)

        def meta(name: str):
            key = f"qwen4exp.ple.{name}"
            if key not in metadata:
                raise KeyError(f"Qwen4Exp GGUF is missing PLE metadata {key}")
            return metadata[key]

        multipliers = torch.tensor(
            _long_tuple(meta("layer_multipliers")), dtype=torch.long
        )
        vocab_sizes = torch.tensor(
            _long_tuple(meta("head_vocab_sizes")), dtype=torch.long
        )
        offsets = torch.tensor(
            _long_tuple(meta("head_offsets")), dtype=torch.long
        )

        expected_heads = self.ngram_heads
        if multipliers.numel() != self.ngram_size:
            raise ValueError(
                f"PLE has {multipliers.numel()} layer multipliers, "
                f"expected {self.ngram_size}"
            )
        if vocab_sizes.numel() != expected_heads or offsets.numel() != expected_heads:
            raise ValueError(
                f"PLE has {vocab_sizes.numel()} vocab sizes / {offsets.numel()} offsets, "
                f"expected {expected_heads} each"
            )
        self._host_constants = (multipliers, vocab_sizes, offsets)

        found = None
        found_reader = None
        readers = []
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
                found_reader = reader

        if found is None:
            raise RuntimeError(
                "Qwen4Exp GGUF has PLE metadata but no per_layer_token_embd.weight"
            )
        if int(found.tensor_type) != self._quant_type:
            raise ValueError(
                "PLE table quant type changed after model construction: "
                f"{GGML_NAME.get(self._quant_type, self._quant_type)} -> "
                f"{getattr(found.tensor_type, 'name', int(found.tensor_type))}"
            )

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
        # forward only index_selects from it.
        self._packed_rows = torch.from_numpy(packed_np)
        self._row_count = rows
        # Keep all readers alive so the memmap backing the selected tensor remains valid.
        self._readers = readers
        assert found_reader is not None

        required_rows = int(offsets[-1] + vocab_sizes[-1])
        if self._row_count < required_rows:
            raise RuntimeError(
                f"PLE table has {self._row_count} rows, metadata needs {required_rows}"
            )

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._dummy:
            from freetoken.core import get_global_ctx

            token_count = get_global_ctx().batch.input_ids.numel()
            return torch.zeros(
                token_count, self.embedding_dim, device=device, dtype=dtype
            )
        if self._packed_rows is None or self._host_constants is None:
            raise RuntimeError("Qwen4Exp GGUF PLE host weights are not loaded")

        ids = self._current_ngram_ids().reshape(-1)
        if ids.numel() == 0:
            return torch.empty(
                0, self.embedding_dim, device=device, dtype=dtype
            )
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self._row_count:
            raise IndexError(
                f"PLE row id range [{lo}, {hi}] outside [0, {self._row_count})"
            )

        # Gather into a tiny pinned staging buffer: 16 rows/token * 90 bytes/row
        # for the current IQ4_NL model. The 28.8 GiB source remains file-backed.
        staging = torch.empty(
            (ids.numel(), self._row_bytes),
            dtype=torch.uint8,
            pin_memory=torch.cuda.is_available(),
        )
        torch.index_select(self._packed_rows, 0, ids, out=staging)

        if self._quant_type in GGML_UNQUANTIZED:
            raw_dtype = _DTYPE_FOR_GGML[self._quant_type]
            values = staging.view(raw_dtype).to(device=device, dtype=dtype)
        else:
            if device.type != "cuda":
                raise RuntimeError(
                    "Qwen4Exp GGUF PLE quantized rows currently require CUDA dequantization"
                )
            from freetoken.kernel.gguf import ggml_dequantize

            packed = staging.to(
                device=device,
                non_blocking=staging.is_pinned(),
            )
            values = ggml_dequantize(
                packed,
                self._quant_type,
                ids.numel(),
                self.head_dim,
                dtype,
            )

        return values.reshape(-1, self.embedding_dim)


__all__ = ["GGUFHostNGramEmbedding"]
