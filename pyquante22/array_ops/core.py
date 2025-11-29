# array_ops/core.py
from typing import Protocol


class ArrayBackend(Protocol):
    """Minimal interface every backend must support."""

    def init(self, data, dtype=None):
        ...

    def zeros(self, shape, dtype=None):
        ...

    def sum(self, x, axis=None, keepdims: bool = False):
        ...


_backend = None  # will hold the active backend instance


def set_backend(name: str) -> None:
    """Select which backend to use: 'numpy' or 'torch'."""
    global _backend

    if name == "numpy":
        from .numpy import backend
    elif name == "torch":
        from .torch import backend
    else:
        raise ValueError(f"unknown backend {name!r}, expected 'numpy' or 'torch'")

    _backend = backend


def get_backend() -> ArrayBackend:
    """Return the active backend (must call set_backend() first)."""
    if _backend is None:
        raise RuntimeError(
            "array_ops backend not set. Call set_backend('numpy') or set_backend('torch') first."
        )
    return _backend
