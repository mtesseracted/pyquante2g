import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from jax.scipy import special as jspecial

from .core import ArrayBackend


class JaxBackend(ArrayBackend):
    name = "jax"
    xp = jnp
    ArrayType = getattr(jax, "Array", type(jnp.asarray(0)))

    def __init__(self, device=None, default_dtype="float64", int_dtype="int64"):
        super().__init__(
            device=device,
            default_dtype=default_dtype,
            int_dtype=int_dtype,
        )

    def _device_obj(self):
        if self.device is None:
            return None
        if hasattr(self.device, "platform") and hasattr(self.device, "id"):
            return self.device
        if isinstance(self.device, str):
            platform, sep, index = self.device.partition(":")
            matches = [dev for dev in jax.devices() if dev.platform == platform]
            if not matches:
                raise ValueError(f"No JAX devices found for platform {platform!r}")
            if not sep:
                return matches[0]
            dev_id = int(index)
            for dev in matches:
                if getattr(dev, "id", None) == dev_id:
                    return dev
            raise ValueError(f"No JAX device found for {self.device!r}")
        if isinstance(self.device, int):
            return jax.devices()[self.device]
        return self.device

    def _put(self, x):
        device = self._device_obj()
        if device is None:
            return x
        return jax.device_put(x, device=device)

    def asarray(self, data, dtype=None):
        dtype = dtype if dtype is not None else self.default_dtype
        return self._put(jnp.asarray(data, dtype=dtype))

    def zeros(self, shape, dtype=None):
        dtype = dtype if dtype is not None else self.default_dtype
        return self._put(jnp.zeros(shape, dtype=dtype))

    def ones(self, shape, dtype=None):
        dtype = dtype if dtype is not None else self.default_dtype
        return self._put(jnp.ones(shape, dtype=dtype))

    def arange(self, start, stop=None, step=1, dtype=None):
        if dtype is None:
            float_args = any(isinstance(arg, float) for arg in (start, stop, step))
            dtype = self.default_dtype if float_args else self.int_dtype
        if stop is None:
            return self._put(jnp.arange(start, dtype=dtype))
        return self._put(jnp.arange(start, stop, step, dtype=dtype))

    def stack(self, arrays, axis=0):
        return self._put(jnp.stack(list(arrays), axis=axis))

    def concat(self, arrays, axis=0):
        return self._put(jnp.concatenate(list(arrays), axis=axis))

    def tolist(self, x):
        return jax.device_get(x).tolist()

    def lgamma(self, x):
        return jspecial.gammaln(x)

    def gamma_inc(self, a, x):
        return jnp.exp(jspecial.gammaln(a)) * jspecial.gammainc(a, x)

    def reciprocal(self, x):
        return jnp.reciprocal(x)

    def flip(self, x, axis=None):
        return jnp.flip(x, axis=axis)

    def conv_full_batch(self, a, b):
        return self._put(
            jnp.stack(
                [jnp.convolve(sig, ker, mode="full") for sig, ker in zip(a, b)],
                axis=0,
            )
        )

    def segment_sum(self, x, lengths, axis=0):
        lengths = [int(v) for v in jax.device_get(lengths)]
        if axis != 0:
            x = jnp.moveaxis(x, axis, 0)

        starts = []
        stops = []
        total = 0
        for length in lengths:
            starts.append(total)
            total += length
            stops.append(total)

        out = jnp.stack(
            [jnp.sum(x[s0:s1], axis=0) for s0, s1 in zip(starts, stops)],
            axis=0,
        )

        if axis != 0:
            out = jnp.moveaxis(out, 0, axis)
        return self._put(out)

    def conv1d(self, x, weight, groups=1, padding=0):
        if x.ndim != 3 or weight.ndim != 3:
            raise ValueError("conv1d expects x and weight to be rank-3 arrays")

        if isinstance(padding, int):
            pad_l = pad_r = padding
        else:
            pad_l, pad_r = padding

        cin = x.shape[1]
        cout, cwg, _ = weight.shape
        if cin % groups != 0 or cout % groups != 0:
            raise ValueError("conv1d groups must divide input and output channels")
        if cwg != cin // groups:
            raise ValueError("conv1d weight second dimension mismatch")

        return self._put(
            lax.conv_general_dilated(
                lhs=x,
                rhs=weight,
                window_strides=(1,),
                padding=((pad_l, pad_r),),
                dimension_numbers=("NCH", "OIH", "NCH"),
                feature_group_count=groups,
            )
        )

    def slice_add(self, x, axis, start, src):
        axis = axis % x.ndim
        start = int(start)
        stop = start + src.shape[axis]
        if start < 0 or stop > x.shape[axis]:
            raise ValueError("slice_add source does not fit in destination")

        dst_sel = [slice(None)] * x.ndim
        dst_sel[axis] = slice(start, stop)
        return x.at[tuple(dst_sel)].add(src)

    def scatter_add(self, x, axis, index, src):
        axis = axis % x.ndim
        index = jnp.asarray(index, dtype=jnp.int64)
        coords = jnp.indices(index.shape, dtype=jnp.int64)
        dest = []
        for dim in range(x.ndim):
            if dim == axis:
                dest.append(index)
            else:
                dest.append(coords[dim])
        return x.at[tuple(dest)].add(src)

    def scatter_add_inplace(self, x, axis, index, src):
        return self.scatter_add(x, axis, index, src)

    def index_add_inplace(self, x, axis, index, src):
        axis = axis % x.ndim
        index = jnp.asarray(index, dtype=jnp.int64)
        if index.ndim != 1:
            raise ValueError("index_add_inplace expects a 1D index array")
        if src.shape[axis] != index.shape[0]:
            raise ValueError("src axis length must match index length")

        coords = jnp.indices(src.shape, dtype=jnp.int64)
        index_shape = [1] * src.ndim
        index_shape[axis] = index.shape[0]
        index_grid = jnp.broadcast_to(index.reshape(index_shape), src.shape)

        dest = []
        for dim in range(x.ndim):
            if dim == axis:
                dest.append(index_grid)
            else:
                dest.append(coords[dim])
        return x.at[tuple(dest)].add(src)

    def eigh(self, x):
        return jnp.linalg.eigh(x)

    def eye(self, n, **kwargs):
        return self._put(jnp.eye(n, **kwargs))


backend_cls = JaxBackend
