"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil

import torch

from ._toolchain import check_nvcc_matches_torch, ensure_windows_msvc_env

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    Linux prefers clang++ and then older GCC versions because very new system GCC
    versions can reject torch headers. Windows deliberately uses the MSVC toolchain
    initialized by ``ensure_windows_msvc_env``; an explicit override remains available
    through ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    if os.name == "nt":
        return None
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

    # A normal PowerShell/cmd session does not inherit Visual Studio's developer
    # environment. PyTorch's JIT path requires cl.exe plus INCLUDE/LIB on Windows,
    # so bootstrap vcvars64 before it asks Ninja to compile anything.
    ensure_windows_msvc_env()
    check_nvcc_matches_torch()

    extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr"]
    extra_cflags: list[str] = []
    if os.name == "nt":
        # CUDA 13.2's CCCL requires the conforming MSVC preprocessor.
        extra_cuda_cflags += ["-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/bigobj"]
        extra_cflags += ["/Zc:preprocessor", "/DNOMINMAX", "/bigobj"]
        # Torch already injects /DNOMINMAX via cpp_extension, but repeat for the JIT's
        # extra_cflags path which is MSVC-direct.
        # Host compiler for the CUDA file may need extra heap for the huge ATen headers.
        extra_cuda_cflags += ["-Xcompiler", "/Zm800"]
        extra_cflags += ["/Zm800"]
        # compiled_autograd.h has a Windows guard that requires USE_CUDA to be
        # defined to avoid a known `if constexpr` + `std` ambiguity on MSVC.
        # Ensure the guard fires during the host compilation of the .cu file.
        extra_cuda_cflags += ["-DUSE_CUDA", "-Xcompiler", "/DUSE_CUDA"]
        extra_cflags += ["/DUSE_CUDA"]
        allocator_header = (
            pathlib.Path(torch.__file__).resolve().parent
            / "include"
            / "c10"
            / "cuda"
            / "CUDACachingAllocator.h"
        )
        if not allocator_header.is_file():
            raise RuntimeError(f"Torch CUDA header not found: {allocator_header}")
        # The Windows-only shadow header uses this absolute path to include the
        # matching installed Torch header after it clears rpcndr.h's `small`.
        extra_cuda_cflags += [
            f'-DFREETOKEN_CUDA_CACHING_ALLOCATOR_HEADER=\\"{allocator_header.as_posix()}\\"'
        ]
    host_cxx = _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    # On Windows, shadow CUDACachingAllocator.h with a tiny wrapper that removes
    # rpcndr.h's `small` macro, then includes the matching installed Torch header.
    include_dirs = [str(_CSRC / "torch_fix"), str(_CSRC)] if os.name == "nt" else [str(_CSRC)]
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=include_dirs,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
    expert_stride_bytes: int = 0,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``.

    ``expert_stride_bytes`` == 0 assumes dense contiguous banks; > 0 reads each
    expert at that fixed byte offset (padded flat banks for mixed-quant models,
    where a layer's real payload occupies the leading bytes of each expert slot).
    """
    return _module().ggml_moe_a8_vec(
        x, weight, topk_ids, top_k, quant_type, row, tokens, expert_stride_bytes
    )


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
