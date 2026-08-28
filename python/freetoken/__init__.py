"""FreeToken inference runtime."""

from freetoken.version import __version__

# NOTE: the Windows Triton driver MSVC patch is armed at the top of
# freetoken.kernel.__init__ instead of here: importing the kernel package chain
# here would import torch at bare ``import freetoken`` time, which breaks the
# daemon's torch-free guarantee (tests/daemon/test_daemon_import_safety.py).

__all__ = ["__version__"]
