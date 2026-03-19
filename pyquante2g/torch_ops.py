import torch
import torch.nn.functional as F


class TorchOps:
    xp = torch
    name = "torch"
    ArrayType = torch.Tensor

    def __init__(
        self,
        backend_or_device="cpu",
        device=None,
        default_dtype="float64",
        int_dtype="int64",
    ):
        if backend_or_device in ("torch", "pytorch"):
            resolved_device = device
        elif device is None:
            resolved_device = backend_or_device
        else:
            resolved_device = device

        self.device = "cpu" if resolved_device is None else resolved_device
        self.default_dtype = self._resolve_dtype(default_dtype)
        self.int_dtype = self._resolve_dtype(int_dtype)

    def _resolve_dtype(self, dtype):
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        return dtype

    def set_device(self, device):
        self.device = device

    def set_dtype(self, default_dtype=None, int_dtype=None):
        if default_dtype is not None:
            self.default_dtype = self._resolve_dtype(default_dtype)
        if int_dtype is not None:
            self.int_dtype = self._resolve_dtype(int_dtype)

    def asarray(self, data, dtype=None):
        dtype = self.default_dtype if dtype is None else dtype
        tensor = torch.as_tensor(data, dtype=dtype)
        return tensor.to(self.device) if self.device is not None else tensor

    def init(self, data, dtype=None):
        return self.asarray(data, dtype=dtype)

    def arr(self, data, dtype=None):
        return self.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        dtype = self.default_dtype if dtype is None else dtype
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        dtype = self.default_dtype if dtype is None else dtype
        return torch.ones(shape, dtype=dtype, device=self.device)

    def arange(self, start, stop=None, step=1, dtype=None):
        if dtype is None:
            uses_float = any(isinstance(v, float) for v in (start, stop, step))
            dtype = self.default_dtype if uses_float else self.int_dtype
        kwargs = {"dtype": dtype}
        if self.device is not None:
            kwargs["device"] = self.device
        if stop is None:
            return torch.arange(start, **kwargs)
        return torch.arange(start, stop, step, **kwargs)

    def tolist(self, x):
        return x.detach().cpu().tolist()

    def norm(self, x, axis=None, keepdims=False):
        return torch.linalg.norm(x, dim=axis, keepdim=keepdims)

    def _reduce(self, fn, x, axis=None, keepdims=False):
        if axis is None:
            return fn(x)
        if isinstance(axis, (tuple, list)):
            dims = [dim % x.ndim for dim in axis]
            if keepdims:
                for dim in sorted(dims):
                    x = fn(x, dim=dim, keepdim=True)
                return x
            for dim in sorted(dims, reverse=True):
                x = fn(x, dim=dim)
            return x
        return fn(x, dim=axis, keepdim=keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce(torch.sum, x, axis=axis, keepdims=keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return self._reduce(torch.prod, x, axis=axis, keepdims=keepdims)

    def amax(self, x, axis=None, keepdims=False):
        return self._reduce(torch.amax, x, axis=axis, keepdims=keepdims)

    def amin(self, x, axis=None, keepdims=False):
        return self._reduce(torch.amin, x, axis=axis, keepdims=keepdims)

    def any(self, x, axis=None, keepdims=False):
        return self._reduce(torch.any, x, axis=axis, keepdims=keepdims)

    def all(self, x, axis=None, keepdims=False):
        return self._reduce(torch.all, x, axis=axis, keepdims=keepdims)

    def stack(self, arrays, axis=0):
        return torch.stack(list(arrays), dim=axis)

    def concat(self, arrays, axis=0):
        return torch.concat(list(arrays), dim=axis)

    def combinations(self, x, r):
        return torch.combinations(x.reshape(-1), r=r)

    def unique(self, x, axis=None, **kwargs):
        if axis is not None:
            kwargs["dim"] = axis
        return torch.unique(x, **kwargs)

    def gamma_inc(self, a, x):
        return torch.exp(torch.lgamma(a)) * torch.igamma(a, x)

    def reciprocal(self, x):
        return torch.reciprocal(x)

    def conv_full_batch(self, a, b):
        return torch.conv_transpose1d(
            a[None, :, :],
            b[:, None, :],
            groups=a.shape[0],
        ).squeeze(0)

    def tile(self, x, n):
        dims = tuple(n) if hasattr(n, "__len__") else (n,)
        return torch.tile(x, dims=dims)

    def transpose(self, x, axes=None):
        if axes is None:
            if x.ndim != 2:
                raise ValueError("transpose must specify axes for ndim>2")
            return torch.transpose(x, 0, 1)
        return torch.permute(x, dims=axes)

    def take_along_axis(self, x, indices, axis=None):
        return torch.take_along_dim(x, indices, dim=axis)

    def slice_add(self, x, axis, start, src):
        axis = axis % x.ndim
        start = int(start)
        stop = start + src.shape[axis]
        if start < 0 or stop > x.shape[axis]:
            raise ValueError("slice_add source does not fit in destination")
        dst_sel = [slice(None)] * x.ndim
        dst_sel[axis] = slice(start, stop)
        x[tuple(dst_sel)] += src
        return x

    def scatter_add(self, x, axis, index, src):
        out = x.clone()
        return out.scatter_add_(dim=axis, index=index, src=src)

    def scatter_add_inplace(self, x, axis, index, src):
        return x.scatter_add_(dim=axis, index=index, src=src)

    def index_add_inplace(self, x, axis, index, src):
        return x.index_add_(dim=axis, index=index, source=src)

    def pad(self, x, pad_width, mode="constant", constant_values=0):
        if isinstance(pad_width, int):
            pad_width = ((pad_width, pad_width),) * x.ndim
        elif len(pad_width) == 2 and not hasattr(pad_width[0], "__len__"):
            if x.ndim != 1:
                raise ValueError("scalar axis pair pad_width is only valid for 1D inputs")
            pad_width = (tuple(pad_width),)
        elif len(pad_width) != x.ndim:
            raise ValueError("pad_width must provide one (before, after) pair per axis")

        torch_pad = []
        for before, after in reversed(pad_width):
            torch_pad.extend([before, after])
        return F.pad(x, pad=tuple(torch_pad), mode=mode, value=constant_values)

    def cumprod(self, x, axis=None):
        if axis is None:
            return torch.cumprod(x.reshape(-1), dim=0)
        return torch.cumprod(x, dim=axis)

    def flip(self, x, axis=None):
        if axis is None:
            dims = list(range(x.ndim))
        elif hasattr(axis, "__len__"):
            dims = list(axis)
        else:
            dims = [axis]
        return torch.flip(x, dims=dims)

    def segment_sum(self, x, lengths, axis=0):
        lengths_t = torch.as_tensor(lengths, device=x.device, dtype=torch.int64)
        x0 = x.movedim(axis, 0)
        flat = x0.reshape(x0.shape[0], -1)
        seg_id = torch.repeat_interleave(
            torch.arange(lengths_t.numel(), device=x.device, dtype=torch.int64),
            lengths_t,
        )
        out = torch.zeros(
            (lengths_t.numel(), flat.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
        out.index_add_(0, seg_id, flat)
        out = out.reshape((lengths_t.numel(),) + x0.shape[1:])
        return out.movedim(0, axis)

    def segment_weighted_sum(self, x, lengths, coeff, axis=0):
        x0 = x.movedim(axis, 0).reshape(x.shape[axis], -1)
        c0 = coeff.reshape(-1, 1).to(dtype=x0.dtype, device=x0.device)
        lengths_t = torch.as_tensor(lengths, device=x0.device, dtype=torch.int64)
        seg_id = torch.repeat_interleave(
            torch.arange(lengths_t.numel(), device=x0.device, dtype=torch.int64),
            lengths_t,
        )
        out = torch.zeros(
            (lengths_t.numel(), x0.shape[1]),
            device=x0.device,
            dtype=x0.dtype,
        )
        out.index_add_(0, seg_id, x0 * c0)
        out_shape = list(x.shape)
        out_shape[axis] = lengths_t.numel()
        out = out.reshape([out_shape[axis]] + [s for i, s in enumerate(out_shape) if i != axis])
        return out.movedim(0, axis)

    def eigh(self, x):
        return torch.linalg.eigh(x)

    def eye(self, n, dtype=None, device=None):
        dtype = self.default_dtype if dtype is None else dtype
        device = self.device if device is None else device
        return torch.eye(n, dtype=dtype, device=device)

    def reshape(self, x, shape):
        return torch.reshape(x, shape)

    def broadcast_to(self, x, shape):
        return torch.broadcast_to(x, shape)

    def conv1d(self, x, weight, **kwargs):
        return F.conv1d(x, weight, **kwargs)

    def allclose(self, a, b, **kwargs):
        return torch.allclose(a, b, **kwargs)

    def broadcast_shapes(self, *shapes):
        return torch.broadcast_shapes(*shapes)

    def finfo(self, dtype):
        return torch.finfo(dtype)

    def iinfo(self, dtype):
        return torch.iinfo(dtype)

    def __getattr__(self, name):
        try:
            return getattr(torch, name)
        except AttributeError as exc:
            raise AttributeError(f"TorchOps has no attribute {name}") from exc


_default_ops = TorchOps()


def get_default_ops() -> TorchOps:
    return _default_ops


def set_default_ops(device=None, default_dtype=None, int_dtype=None) -> TorchOps:
    if device is not None:
        _default_ops.set_device(device)
    if default_dtype is not None or int_dtype is not None:
        _default_ops.set_dtype(default_dtype=default_dtype, int_dtype=int_dtype)
    return _default_ops


__all__ = ["TorchOps", "get_default_ops", "set_default_ops"]


def __getattr__(name: str):
    return getattr(_default_ops, name)
