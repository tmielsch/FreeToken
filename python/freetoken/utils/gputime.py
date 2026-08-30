"""Flag-gated GPU-event region timing for eager diagnosis.

MOESPLIT/MODELPROF/STEPTIM measure HOST enqueue time; on Windows/WDDM the
launch cost and queue waits dominate those numbers, so they never show the
true GPU kernel time. This helper wraps regions in ``torch.cuda.Event``
pairs instead: ``flush()`` (called once per model forward, at the end of the
step) computes the real device-side elapsed time per region name and logs
one aggregated ``GPUSPLIT`` line.

Enable with the file ``D:\\temp\\opencode\\ft_gputime.flag`` and run the
engine in EAGER decode (FREETOKEN_CAPTURE=0); event recording is a no-op
inside CUDA-graph capture scope, so the flag is inert during capture.
"""
from __future__ import annotations

import collections
import contextlib
import os

import torch

FLAG = r"D:\temp\opencode\ft_gputime.flag"
_regions: list["_Region"] = []


def enabled() -> bool:
    return os.path.exists(FLAG)


def _active() -> bool:
    if not enabled() or not torch.cuda.is_available():
        return False
    return not torch.cuda.is_current_stream_capturing()


class _Region:
    def __init__(self, name: str) -> None:
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "_Region":
        self.start.record(torch.cuda.current_stream())
        return self

    def __exit__(self, *exc: object) -> None:
        self.end.record(torch.cuda.current_stream())
        _regions.append(self)


def timed(name: str) -> contextlib.AbstractContextManager:
    """GPU-event timing context for the named region (no-op when disabled)."""
    if not _active():
        return contextlib.nullcontext()
    return _Region(name)


def flush() -> None:
    """Log per-step aggregated GPU milliseconds by region name (one device sync)."""
    global _regions
    if not _regions:
        return
    totals: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    for r in _regions:
        totals[r.name] += r.start.elapsed_time(r.end)
        counts[r.name] += 1
    _regions = []
    from freetoken.utils.logger import init_logger

    parts = " ".join(f"{k}={v:.2f}(x{counts[k]})" for k, v in sorted(totals.items()))
    init_logger("freetoken.utils.gputime").info("GPUSPLIT %s", parts)
