# array_ops/numpy.py
import numpy as np

from .core import ArrayBackend


class NumpyBackend(ArrayBackend):
    name = "numpy"
    array_type = np.ndarray

    def init(self, data, dtype=None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def sum(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.sum(x, axis=axis, keepdims=keepdims)


backend = NumpyBackend()
