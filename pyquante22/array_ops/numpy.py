# array_ops/numpy.py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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

    def reciprocal(self, x):
        return np.reciprocal(x)

    def flip(self, x, axis=None):
        return np.flip(x, axis=axis)

    def conv_full_batch(self, a, b):
        return np.array([
            np.convolve(sig, ker, mode='full') 
            for sig, ker in zip(a, b)
        ])

    def segment_sum(self, x, lengths, axis=0):
        lengths = np.asarray(lengths, dtype=np.int64)
        if axis != 0:
            x = np.moveaxis(x, axis, 0)

        stops = np.cumsum(lengths)
        starts = np.concatenate(([0], stops[:-1]))
        out = np.stack([x[s0:s1].sum(axis=0) for s0, s1 in zip(starts, stops)], axis=0)

        if axis != 0:
            out = np.moveaxis(out, 0, axis)
        return out

    def conv1d(self, x, weight, groups=1, padding=0):
        if x.ndim != 3 or weight.ndim != 3:
            raise ValueError("conv1d expects x and weight to be rank-3 arrays")

        if isinstance(padding, int):
            pad_l = pad_r = padding
        else:
            pad_l, pad_r = padding

        xpad = np.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
        batch, cin, n = xpad.shape
        cout, cwg, kw = weight.shape
        if cin % groups != 0 or cout % groups != 0:
            raise ValueError("conv1d groups must divide input and output channels")

        cin_g = cin // groups
        cout_g = cout // groups
        if cwg != cin_g:
            raise ValueError("conv1d weight second dimension mismatch")

        windows = sliding_window_view(xpad, kw, axis=-1)
        out_w = windows.shape[-2]
        windows = windows.reshape(batch, groups, cin_g, out_w, kw)
        weight = weight.reshape(groups, cout_g, cin_g, kw)
        out = np.einsum("bgiok,gcik->bgco", windows, weight)
        return out.reshape(batch, cout, out_w)

    def scatter_add(self, x, axis, index, src):
        axis = axis % x.ndim
        out = x.copy()
        coords = np.indices(index.shape)
        dest = []
        for dim in range(out.ndim):
            if dim == axis:
                dest.append(index)
            else:
                dest.append(coords[dim])
        np.add.at(out, tuple(dest), src)
        return out

    def scatter_add_inplace(self, x, axis, index, src):
        axis = axis % x.ndim
        coords = np.indices(index.shape)
        dest = []
        for dim in range(x.ndim):
            if dim == axis:
                dest.append(index)
            else:
                dest.append(coords[dim])
        np.add.at(x, tuple(dest), src)
        return x

    def index_add_inplace(self, x, axis, index, src):
        axis = axis % x.ndim
        index = np.asarray(index, dtype=np.intp)
        if index.ndim != 1:
            raise ValueError("index_add_inplace expects a 1D index array")
        if src.shape[axis] != index.shape[0]:
            raise ValueError("src axis length must match index length")

        for i, dst in enumerate(index):
            dst_sel = [slice(None)] * x.ndim
            src_sel = [slice(None)] * src.ndim
            dst_sel[axis] = dst
            src_sel[axis] = i
            x[tuple(dst_sel)] += src[tuple(src_sel)]
        return x

#end class NumpyBackend(ArrayBackend):


backend_cls = NumpyBackend
