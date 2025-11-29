# array_ops/__init__.py
"""
Minimal unified array ops over multiple backends.

Two ways to use it:

1) Global-style (one default instance):

    import array_ops as arr

    arr.set_backend("numpy")   # or "torch"
    x = arr.init([1, 2, 3])
    s = arr.sum(x, axis=0)

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


def set_backend(name: str) -> None:
    _default_ops.set_backend(name)


def get_default_ops() -> ArrayOps:
    return _default_ops


__all__ = ["ArrayBackend", "ArrayOps", "set_backend", "get_default_ops"]


def __getattr__(name: str):
    """
    Forward arr.init / arr.zeros / arr.sum / ... to the default instance.

    So `import array_ops as arr; arr.init(...)` still works.
    """
    return getattr(_default_ops, name)
