"""Import boundary for legacy direct calls into the optional AOT package.

New multi-backend operators should use :mod:`sglang.kernels.registry`.  This
module only lets existing direct CPU call sites detect an absent ``sgl_kernel``
package without making backend imports eager in the public kernel namespace.
"""

from __future__ import annotations

import importlib
import platform
import sys
from collections.abc import Callable
from typing import Any


def _is_macos_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() == "arm64"


def get_optional_sgl_kernel(symbol: str) -> Callable[..., Any] | None:
    """Return an optional macOS ``sgl_kernel`` callable when available.

    Apple Silicon builds may omit the CPU AOT package or expose only Metal
    symbols. Other platforms preserve the existing hard dependency so a
    missing or incompatible CPU extension remains visible.

    Args:
        symbol: Name exported from the top-level ``sgl_kernel`` module.

    Returns:
        The callable when it exists, otherwise ``None``.
    """
    try:
        module = importlib.import_module("sgl_kernel")
    except ModuleNotFoundError as error:
        if error.name == "sgl_kernel" and _is_macos_arm64():
            return None
        raise

    try:
        kernel = getattr(module, symbol)
    except AttributeError:
        if _is_macos_arm64():
            return None
        raise

    if not callable(kernel):
        raise TypeError(f"sgl_kernel.{symbol} is not callable")
    return kernel
