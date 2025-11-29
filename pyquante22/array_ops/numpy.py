# array_ops/numpy.py

import numpy as np
from .core import ArrayBackend


class NumpyBackend(ArrayBackend):
    name = "numpy"
    xp = np
    ArrayType = np.ndarray

    def __init__(self, device=None, 
                 default_dtype="float64", int_dtype="int64",):
        super().__init__(device=device, 
                         default_dtype=default_dtype, int_dtype=int_dtype)


backend_cls = NumpyBackend
