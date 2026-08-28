from __future__ import annotations

from functools import lru_cache

import torch

from .utils import load_jit, make_cpp_args


@lru_cache(maxsize=None)
def _module(*, num_threads: int, num_blocks: int):
    args = make_cpp_args(num_threads, num_blocks)
    return load_jit(
        "strided_index_copy",
        *args,
        cuda_files=["strided_index_copy.cuh"],
        cuda_wrappers=[("launch", f"&StridedIndexCopyKernel<{args}>::run")],
    )


def strided_index_copy_jit(
    dst: torch.Tensor,
    dst_indices: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    num_threads: int = 1024,
    num_blocks: int = 8,
) -> None:
    """Copy indexed rows from a compact source into wider fixed-stride cache slots.

    Both tensors are byte matrices. ``src.shape[1]`` is the native bytes/expert for
    the current GGUF layer/projection; ``dst.shape[1]`` is the maximum slot stride
    reserved for that projection across the model. Only the source-width prefix of
    each destination slot is written.
    """
    assert dst.dtype == src.dtype == torch.uint8
    assert dst.ndim == src.ndim == 2
    assert src.shape[1] <= dst.shape[1]
    assert src.shape[1] % 16 == 0
    assert dst.is_cuda
    module = _module(num_threads=num_threads, num_blocks=num_blocks)
    module.launch(dst, dst_indices, src, src_indices, num_indices)


__all__ = ["strided_index_copy_jit"]
