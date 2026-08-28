"""CUDA toolchain/torch consistency checks.

Standalone on purpose: setup.py and the kernel-cache build backend load this
file by path, so it must not import the freetoken package.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

ALLOW_MISMATCH_ENV = "FREETOKEN_ALLOW_CUDA_MISMATCH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _nvcc_path() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME:
        return os.path.join(CUDA_HOME, "bin", "nvcc")
    return shutil.which("nvcc")


def nvcc_release(nvcc: str) -> tuple[int, int] | None:
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", proc.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_cuda_major() -> int | None:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    return int(cuda.split(".")[0]) if cuda else None


def _vswhere_path() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("vswhere")
    return Path(found) if found else None


def _vcvars64_path() -> Path | None:
    vswhere = _vswhere_path()
    if vswhere is not None:
        try:
            proc = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            install = proc.stdout.strip().splitlines()
            if install:
                candidate = Path(install[-1]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.is_file():
                    return candidate
        except (OSError, subprocess.CalledProcessError):
            pass

    roots: list[Path] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            roots.append(Path(root) / "Microsoft Visual Studio")
    editions = ("Community", "Professional", "Enterprise", "BuildTools")
    for root in roots:
        for year in ("2022", "2019"):
            for edition in editions:
                candidate = root / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.is_file():
                    return candidate
    return None


@functools.cache
def ensure_windows_msvc_env() -> None:
    """Populate this process with a Visual Studio x64 developer environment."""
    if os.name != "nt" or shutil.which("cl"):
        return

    vcvars = _vcvars64_path()
    if vcvars is None:
        raise RuntimeError(
            "MSVC cl.exe is not on PATH and Visual Studio vcvars64.bat could not be found. "
            "Install the Visual Studio C++ build tools (x64) or run FreeToken from an "
            "x64 Native Tools Command Prompt."
        )

    try:
        proc = subprocess.run(
            ["cmd.exe", "/d", "/c", f'call "{vcvars}" >nul && set'],
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"failed to initialize MSVC environment via {vcvars}") from exc

    for line in proc.stdout.splitlines():
        if not line or line.startswith("=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value

    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

    if shutil.which("cl") is None:
        raise RuntimeError(
            f"initialized {vcvars}, but cl.exe is still not discoverable on PATH"
        )


@functools.cache
def check_nvcc_matches_torch() -> None:
    """Refuse to nvcc-compile kernels across CUDA majors.

    nvcc-built binaries link libcudart.so.<nvcc major>; at runtime only the
    torch wheel's own CUDA runtime is guaranteed to be loadable.
    """
    if os.getenv(ALLOW_MISMATCH_ENV, "").strip().lower() in _TRUE_VALUES:
        return
    torch_major = torch_cuda_major()
    if torch_major is None:
        return
    nvcc = _nvcc_path()
    if nvcc is None:
        return
    release = nvcc_release(nvcc)
    if release is None:
        return
    if release[0] != torch_major:
        import torch

        raise RuntimeError(
            f"nvcc {release[0]}.{release[1]} would build kernels linking "
            f"libcudart.so.{release[0]}, but torch {torch.__version__} ships CUDA "
            f"{torch.version.cuda} (libcudart.so.{torch_major}). Install a CUDA "
            f"{torch_major}.x toolkit, or set {ALLOW_MISMATCH_ENV}=1 to override."
        )
