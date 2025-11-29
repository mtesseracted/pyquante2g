# from .core import *
# array_ops/__init__.py
"""
Minimal unified array ops over multiple backends.

Usage:
    import array_ops as arr

    arr.set_backend("numpy")   # or "torch"
    x = arr.init([1, 2, 3])
    z = arr.zeros((2, 3))
    s = arr.sum(x, axis=0)
"""

from .core import ArrayBackend, get_backend, set_backend

__all__ = ["ArrayBackend", "get_backend", "set_backend"]


def __getattr__(name: str):
    """
    Any unknown attribute on `array_ops` is looked up on the active backend.

    So arr.init, arr.zeros, arr.sum, etc. resolve to backend.init / backend.zeros / backend.sum.
    """
    backend = get_backend()
    return getattr(backend, name)
