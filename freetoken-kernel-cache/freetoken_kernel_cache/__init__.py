"""AOT kernel cache package.

The compiled SYCL kernel modules (built by ``freetoken.kernel.aot``) are stored
under this package's directory, so the cache lives with the installed
``freetoken-kernel-cache`` distributable and is shared across processes. This
module exposes the directory the rest of the toolchain writes to.
"""
from __future__ import annotations

import os
import pathlib

__version__ = "0.0.0"

# AOT/JIT kernel modules are written here. Overridable with
# FREETOKEN_KERNEL_CACHE_DIR (see freetoken.kernel.aot). The default keeps them
# inside the installed package (writable) so a reinstall doesn't orphan the
# cache, while remaining per-user when the site-packages dir is read-only.
_ENV_OVERRIDE = "FREETOKEN_KERNEL_CACHE_DIR"


def _default_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).parent
    if os.access(here, os.W_OK):
        return here / "kernels"
    return pathlib.Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser() / "freetoken-kernel-cache" / "kernels"


def get_jit_cache_dir() -> str:
    """The directory that holds precompiled (AOT) kernel modules.

    An explicit ``FREETOKEN_KERNEL_CACHE_DIR`` wins; otherwise the package's own
    (writable) ``kernels/`` dir is used, falling back to the user cache dir when
    the install location is read-only.
    """
    override = os.getenv(_ENV_OVERRIDE)
    if override:
        return str(pathlib.Path(override).expanduser())
    return str(_default_dir())
