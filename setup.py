from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)

    # CUDA's library layout differs by platform. Linux toolkits conventionally
    # expose lib64/, while the Windows toolkit installs cudart.lib in lib/x64/.
    # Keep lib/ as a final compatibility fallback for non-standard layouts.
    candidates = [
        cuda_home / "lib64",
        cuda_home / "lib" / "x64",
        cuda_home / "lib",
    ]
    library_dirs = [str(path) for path in candidates if path.exists()]
    if not library_dirs:
        raise RuntimeError(
            f"CUDA runtime library directory not found under {cuda_home}; "
            "expected one of lib64, lib/x64, or lib"
        )
    return [str(cuda_home / "include")], library_dirs


def _cxx_args(*, cpu_moe: bool = False) -> list[str]:
    """Compiler flags for setuptools' active native compiler.

    PyTorch's CppExtension does not translate GCC flags for MSVC. Passing -O3,
    -std=c++17, and -pthread through cl.exe merely produces D9002 warnings and
    makes Windows builds unnecessarily fragile. Use native MSVC spellings there;
    the CPU MoE extension does not need a separate pthread flag on Windows.

    CUDA/Torch headers can pull in Windows headers that define ``min`` and
    ``max`` macros. Those corrupt ordinary C++ calls such as ``std::min(...)``
    in cpu_moe_ext.cpp, so suppress them globally for native Windows builds.
    """
    if os.name == "nt":
        return ["/O2", "/std:c++17", "/DNOMINMAX"]
    args = ["-O3", "-std=c++17"]
    if cpu_moe:
        args.append("-pthread")
    return args


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=_cxx_args(),
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=_cxx_args(cpu_moe=True),
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
