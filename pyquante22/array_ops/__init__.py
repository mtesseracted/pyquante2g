# array_ops/__init__.py
"""
Unified array ops over multiple backends.
Two ways to use it:

1) Global-style (one default instance):

    import array_ops as arr
    arr.set_backend("numpy")   # or "torch"
    x = arr.init([1, 2, 3])

2) Multiple independent instances:

    from array_ops import ArrayOps
    ops1 = ArrayOps("numpy")
    ops2 = ArrayOps("torch")
    x = ops1.init([1, 2, 3])
    y = ops2.init([1, 2, 3])
"""

from .core import ArrayBackend, ArrayOps

# Default, module-level instance (for the global-style API)
_default_ops = ArrayOps("numpy")


def get_default_ops() -> ArrayOps:
    return _default_ops


__all__ = ["ArrayBackend", "ArrayOps", "set_backend", "get_default_ops"]


def __getattr__(name: str):
    """
    Forward arr.init / arr.zeros / arr.sum / ... to the default instance.
    So `import array_ops as arr; arr.init(...)` still works.
    """
    return getattr(_default_ops, name)
