"""FreeToken inference runtime."""

from freetoken.kernel.triton.windows_compat import install_windows_triton_driver_patch
from freetoken.version import __version__

install_windows_triton_driver_patch()

__all__ = ["__version__"]
