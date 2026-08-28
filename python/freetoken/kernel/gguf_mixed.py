from __future__ import annotations

import functools
import os
import pathlib
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _host_compiler() -> str | None:
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr"]
    host_cxx = _host_compiler()
    if host_cxx is not None:
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    return load(
        name="freetoken_gguf_mixed_moe_kernels",
        sources=[str(_CSRC / "gguf_moe_strided.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


def ggml_moe_a8_vec_strided(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_stride_bytes: int,
) -> torch.Tensor:
    """GGUF expert GEMV over padded cache slots with a runtime byte stride."""
    return _module().ggml_moe_a8_vec_strided(
        x,
        weight,
        topk_ids,
        top_k,
        quant_type,
        row,
        tokens,
        expert_stride_bytes,
    )


__all__ = ["ggml_moe_a8_vec_strided"]
