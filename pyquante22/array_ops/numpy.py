# array_ops/numpy.py
import numpy as np
from .core import ArrayBackend
from math import lgamma
from pyquante22.utils.math import igamma


class NumpyBackend(ArrayBackend):
    name = "numpy"
    xp = np
    ArrayType = np.ndarray

    def __init__(self, device=None, 
                 default_dtype="float64", int_dtype="int64",):
        super().__init__(device=device, 
                         default_dtype=default_dtype, int_dtype=int_dtype)

    def lgamma(self, x):
        return np.vectorize(lgamma)(x)

    def gamma_inc(self, a, x):
        return np.vectorize(igamma)(a, x)

    def conv_full_batch(self, a, b):
        return np.array([
            np.convolve(sig, ker, mode='full') 
            for sig, ker in zip(a, b)
        ])

#end class NumpyBackend(ArrayBackend):


backend_cls = NumpyBackend
