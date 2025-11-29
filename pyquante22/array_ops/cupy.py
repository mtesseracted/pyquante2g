# array_ops/cupy.py

import cupy as cp
from .core import ArrayBackend


class CupyBackend(ArrayBackend):
    name = "cupy"
    xp = cp
    ArrayType = cp.ndarray

    def __init__(self, device=None):
        super().__init__(device=device)

    def _dev_ctx(self):
        # device can be None, int, or something you define; here we accept None or int / str
        if self.device is None:
            return cp.cuda.Device()     # current device
        return cp.cuda.Device(self.device)

    def asarray(self, data, dtype=None):
        with self._dev_ctx():
            return cp.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        with self._dev_ctx():
            return cp.zeros(shape, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        with self._dev_ctx():
            if stop is None:
                return cp.arange(start, dtype=dtype)
            return cp.arange(start, stop, step, dtype=dtype)

    def stack(self, arrays, axis=0):
        with self._dev_ctx():
            return cp.stack(list(arrays), axis=axis)

    def tolist(self, x):
        # CuPy array → NumPy array → list
        return cp.asnumpy(x).tolist()

    # combinations: you can either override to stay on GPU, or
    # just use the default base-class Python implementation.


backend_cls = CupyBackend
