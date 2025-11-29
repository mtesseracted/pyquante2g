# array_ops/numpy.py
import numpy as np
from itertools import combinations as _it_combinations

from .core import ArrayBackend


class NumpyBackend(ArrayBackend):
    name = "numpy"
    array_type = np.ndarray

    # --- core creation / reduction ---

    def init(self, data, dtype=None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def sum(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.sum(x, axis=axis, keepdims=keepdims)

    # --- elementwise / math ---

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def pow(self, x, y) -> np.ndarray:
        return np.power(x, y)

    def exp(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def floor(self, x: np.ndarray) -> np.ndarray:
        return np.floor(x)

    # --- creation / shape ops ---

    def arange(self, start, stop=None, step=1, dtype=None) -> np.ndarray:
        # uses NumPy semantics for None
        return np.arange(start, stop, step, dtype=dtype)

    def stack(self, arrays, axis=0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    # --- norms / reductions ---

    def norm(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    def amin(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.amin(x, axis=axis, keepdims=keepdims)

    def amax(self, x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
        return np.amax(x, axis=axis, keepdims=keepdims)

    # --- conversion / combinatorics ---

    def tolist(self, x: np.ndarray):
        return x.tolist()

    def combinations(self, x: np.ndarray, r: int) -> np.ndarray:
        """
        All r-combinations (no replacement) of a 1D array.

        Returns an array of shape (n_combinations, r).
        """
        x = np.asarray(x).ravel()
        combs = list(_it_combinations(x, r))
        if not combs:
            return np.empty((0, r), dtype=x.dtype)
        return np.array(combs, dtype=x.dtype)


backend = NumpyBackend()
