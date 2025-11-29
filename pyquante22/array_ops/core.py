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


def _load_backend(name: str) -> ArrayBackend:
    """Internal helper: import a backend by name."""
    if name == "numpy":
        from .numpy import backend
    elif name == "torch":
        from .torch import backend
    else:
        raise ValueError(f"unknown backend {name!r}, expected 'numpy' or 'torch'")
    return backend


class ArrayOps:
    """
    A handle that owns its own backend instance.

    Example:
        from array_ops import ArrayOps

        ops1 = ArrayOps("numpy")
        ops2 = ArrayOps("torch")

        x = ops1.init([1, 2, 3])
        y = ops2.init([1, 2, 3])
    """

    def __init__(self, backend):
        self.set_backend(backend)

    def set_backend(self, backend) -> None:
        """backend can be a string name or an ArrayBackend instance."""
        if isinstance(backend, str):
            self._backend = _load_backend(backend)
        else:
            # assume it already implements ArrayBackend
            self._backend = backend

    @property
    def backend(self) -> ArrayBackend:
        return self._backend

    @property
    def backend_name(self) -> str:
        # optional convenience
        return getattr(self._backend, "name", "<unknown>")

    def __getattr__(self, name: str):
        """
        Forward attribute lookups (init, zeros, sum, ...) to the backend.

        So ops.init(...) => ops.backend.init(...)
        """
        return getattr(self._backend, name)
