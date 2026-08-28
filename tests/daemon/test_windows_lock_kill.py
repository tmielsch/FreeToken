"""Windows support for the daemon's two POSIX primitives: the single-instance
file lock (msvcrt byte-range lock instead of flock) and the process-tree kill
(taskkill /T instead of killpg). Both must preserve the semantics the
ServeManager relies on: a second acquirer fails fast, release frees the lock,
and a tree kill on Windows is tree-wide (mp-spawned scheduler/tokenizer workers
must not survive as VRAM-holding orphans)."""

from __future__ import annotations

import os

import pytest

from freetoken.daemon import osproc
from freetoken.daemon.pidfile import AlreadyRunning, SingleInstance
from freetoken.daemon.serve_manager import build_serve_command


def test_single_instance_blocks_second_acquirer_and_recovers(tmp_path):
    path = str(tmp_path / "daemon.pid")
    first = SingleInstance(path)
    first.acquire()
    with pytest.raises(AlreadyRunning):
        SingleInstance(path).acquire()
    first.release()
    # closing the fd drops the lock, so a later daemon bootstraps cleanly even
    # after a crash (the lock is never wedged behind a dead pid)
    second = SingleInstance(path)
    second.acquire()
    second.release()


def test_signal_group_windows_uses_taskkill_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(osproc.os, "name", "nt")
    monkeypatch.setattr(osproc.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    osproc.signal_group(4242, 9)
    assert calls, "taskkill must be invoked on Windows"
    argv = calls[0][0][0]
    assert argv[:3] == ["taskkill", "/PID", "4242"]
    assert "/T" in argv and "/F" in argv
    # no console-window flash (the desktop app would otherwise see one per stop)
    assert calls[0][1].get("creationflags") == getattr(osproc.subprocess, "CREATE_NO_WINDOW", 0)


def test_signal_group_windows_survives_taskkill_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("no taskkill")

    monkeypatch.setattr(osproc.os, "name", "nt")
    monkeypatch.setattr(osproc.subprocess, "run", boom)
    osproc.signal_group(4242, 15)  # degraded start/stop must never raise


def test_build_serve_command_swaps_pythonw_for_console_python(monkeypatch, tmp_path):
    # pythonw (= windowless) starves the backend workers' stdio; the serve always
    # needs the console interpreter. Fabricate a fake interpreter pair.
    (tmp_path / "pythonw.exe").write_bytes(b"")
    (tmp_path / "python.exe").write_bytes(b"")
    monkeypatch.setattr(os, "name", "nt")
    argv, _ = build_serve_command(
        "m", 1919, [], python=str(tmp_path / "pythonw.exe"), log_dir=str(tmp_path)
    )
    assert argv[0] == str(tmp_path / "python.exe")

    argv, _ = build_serve_command(
        "m", 1919, [], python=str(tmp_path / "python.exe"), log_dir=str(tmp_path)
    )
    assert argv[0] == str(tmp_path / "python.exe")

    monkeypatch.setattr(os, "name", "posix")  # POSIX: pythonw swap must not fire
    argv, _ = build_serve_command(
        "m", 1919, [], python=str(tmp_path / "pythonw.exe"), log_dir=str(tmp_path)
    )
    assert argv[0] == str(tmp_path / "pythonw.exe")
