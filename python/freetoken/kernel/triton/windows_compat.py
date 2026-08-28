"""Windows compatibility hooks for Triton wheels."""

from __future__ import annotations

import functools
import os


_CUDA_UTILS_EMPTY_INITIALIZERS = (
    "CUlaunchAttribute clusterAttr = {};",
    "CUlaunchAttribute clusterSchedulingAttr = {};",
)


def patch_cuda_utils_source(source: str) -> str:
    """Make Triton 3.7's CUDA helper valid C11 for MSVC.

    Triton compiles this file with ``/std:c11``, where MSVC rejects the C++
    empty initializer syntax. The replacement preserves zero initialization.
    """
    for initializer in _CUDA_UTILS_EMPTY_INITIALIZERS:
        source = source.replace(initializer, initializer.replace("{}", "{0}"))
    return source


def install_windows_triton_driver_patch() -> None:
    """Patch Triton's in-memory CUDA helper compilation on Windows only."""
    if os.name != "nt":
        return

    try:
        from triton.backends.nvidia import driver
    except ImportError:
        return

    if getattr(driver, "_freetoken_windows_cuda_utils_patch", False):
        return

    compile_module = driver.compile_module_from_src

    @functools.wraps(compile_module)
    def compile_module_from_src(*args, **kwargs):
        name = kwargs.get("name") if "name" in kwargs else (args[1] if len(args) > 1 else None)
        if name == "cuda_utils":
            if "src" in kwargs:
                kwargs = dict(kwargs)
                kwargs["src"] = patch_cuda_utils_source(kwargs["src"])
            elif args:
                args = (patch_cuda_utils_source(args[0]), *args[1:])
        return compile_module(*args, **kwargs)

    driver.compile_module_from_src = compile_module_from_src
    driver._freetoken_windows_cuda_utils_patch = True


__all__ = ["install_windows_triton_driver_patch", "patch_cuda_utils_source"]
