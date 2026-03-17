# array_ops/cupy.py

import cupy as cp
from cupyx.scipy import special as cpspecial

from .core import ArrayBackend


class CupyBackend(ArrayBackend):
    name = "cupy"
    xp = cp
    ArrayType = cp.ndarray

    def __init__(self, device=None, default_dtype="float64", int_dtype="int64"):
        super().__init__(
            device=device,
            default_dtype=default_dtype,
            int_dtype=int_dtype,
        )

    def _device_id(self):
        if self.device is None:
            return cp.cuda.Device().id
        if isinstance(self.device, str):
            if self.device.startswith("cuda:"):
                return int(self.device.split(":", 1)[1])
            return int(self.device)
        return int(self.device)

    def _dev_ctx(self):
        return cp.cuda.Device(self._device_id())

    def asarray(self, data, dtype=None):
        dtype = dtype if dtype is not None else self.default_dtype
        with self._dev_ctx():
            return cp.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        dtype = dtype if dtype is not None else self.default_dtype
        with self._dev_ctx():
            return cp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        dtype = dtype if dtype is not None else self.default_dtype
        with self._dev_ctx():
            return cp.ones(shape, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        if dtype is None:
            float_args = any(isinstance(arg, float) for arg in (start, stop, step))
            dtype = self.default_dtype if float_args else self.int_dtype
        with self._dev_ctx():
            if stop is None:
                return cp.arange(start, dtype=dtype)
            return cp.arange(start, stop, step, dtype=dtype)

    def stack(self, arrays, axis=0):
        with self._dev_ctx():
            return cp.stack(list(arrays), axis=axis)

    def concat(self, arrays, axis=0):
        with self._dev_ctx():
            return cp.concatenate(list(arrays), axis=axis)

    def tolist(self, x):
        return cp.asnumpy(x).tolist()

    def lgamma(self, x):
        return cpspecial.gammaln(x)

    def gamma_inc(self, a, x):
        return cp.exp(cpspecial.gammaln(a)) * cpspecial.gammainc(a, x)

    def reciprocal(self, x):
        return cp.reciprocal(x)

    def flip(self, x, axis=None):
        return cp.flip(x, axis=axis)

    def conv_full_batch(self, a, b):
        with self._dev_ctx():
            return cp.stack(
                [cp.convolve(sig, ker, mode="full") for sig, ker in zip(a, b)],
                axis=0,
            )

    def segment_sum(self, x, lengths, axis=0):
        lengths = cp.asarray(lengths, dtype=cp.int64)
        if axis != 0:
            x = cp.moveaxis(x, axis, 0)

        stops = cp.cumsum(lengths)
        starts = cp.concatenate((cp.asarray([0], dtype=cp.int64), stops[:-1]))
        out = cp.stack(
            [x[int(s0):int(s1)].sum(axis=0) for s0, s1 in zip(starts, stops)],
            axis=0,
        )

        if axis != 0:
            out = cp.moveaxis(out, 0, axis)
        return out

    def conv1d(self, x, weight, groups=1, padding=0):
        if x.ndim != 3 or weight.ndim != 3:
            raise ValueError("conv1d expects x and weight to be rank-3 arrays")

        if isinstance(padding, int):
            pad_l = pad_r = padding
        else:
            pad_l, pad_r = padding

        xpad = cp.pad(x, ((0, 0), (0, 0), (pad_l, pad_r)))
        batch, cin, n = xpad.shape
        cout, cwg, kw = weight.shape
        if cin % groups != 0 or cout % groups != 0:
            raise ValueError("conv1d groups must divide input and output channels")

        cin_g = cin // groups
        cout_g = cout // groups
        if cwg != cin_g:
            raise ValueError("conv1d weight second dimension mismatch")

        out_w = n - kw + 1
        out = cp.zeros((batch, cout, out_w), dtype=cp.result_type(x, weight))
        for group in range(groups):
            xg = xpad[:, group * cin_g:(group + 1) * cin_g, :]
            wg = weight[group * cout_g:(group + 1) * cout_g]
            for out_idx in range(cout_g):
                for pos in range(out_w):
                    patch = xg[:, :, pos:pos + kw]
                    out[:, group * cout_g + out_idx, pos] = cp.sum(
                        patch * wg[out_idx][None, :, :],
                        axis=(1, 2),
                    )
        return out

    def slice_add(self, x, axis, start, src):
        return super().slice_add(x, axis, start, src)

    def scatter_add(self, x, axis, index, src):
        axis = axis % x.ndim
        out = x.copy()
        coords = cp.indices(index.shape)
        dest = []
        for dim in range(out.ndim):
            if dim == axis:
                dest.append(index)
            else:
                dest.append(coords[dim])
        cp.add.at(out, tuple(dest), src)
        return out

    def scatter_add_inplace(self, x, axis, index, src):
        axis = axis % x.ndim
        coords = cp.indices(index.shape)
        dest = []
        for dim in range(x.ndim):
            if dim == axis:
                dest.append(index)
            else:
                dest.append(coords[dim])
        cp.add.at(x, tuple(dest), src)
        return x

    def index_add_inplace(self, x, axis, index, src):
        axis = axis % x.ndim
        index = cp.asarray(index, dtype=cp.int64)
        if index.ndim != 1:
            raise ValueError("index_add_inplace expects a 1D index array")
        if src.shape[axis] != index.shape[0]:
            raise ValueError("src axis length must match index length")

        for i, dst in enumerate(index.tolist()):
            dst_sel = [slice(None)] * x.ndim
            src_sel = [slice(None)] * src.ndim
            dst_sel[axis] = dst
            src_sel[axis] = i
            x[tuple(dst_sel)] += src[tuple(src_sel)]
        return x

    def eigh(self, x):
        return cp.linalg.eigh(x)

    def eye(self, n, **kwargs):
        with self._dev_ctx():
            return cp.eye(n, **kwargs)


backend_cls = CupyBackend
