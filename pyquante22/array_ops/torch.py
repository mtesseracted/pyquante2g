# array_ops/torch.py

import torch
import torch.nn.functional as F
from .core import ArrayBackend


class TorchBackend(ArrayBackend):
    xp = torch
    name = "torch"
    ArrayType = torch.Tensor

    def __init__(self, device="cpu", 
                 default_dtype="float64", int_dtype="int64",):
        if device is None:
            device = "cpu"
        super().__init__(device=device, 
                         default_dtype=default_dtype, int_dtype=int_dtype)

    # Creation / conversion should honor self.device

    def asarray(self, data, dtype=None):
        dtype = dtype if dtype else self.default_dtype
        t = torch.as_tensor(data, dtype=dtype)
        if self.device is not None:
            return t.to(self.device)
        return t


    def zeros(self, shape, dtype=None):
        dtype = dtype if dtype else self.default_dtype
        return torch.zeros(shape, dtype=dtype, device=self.device)


    def ones(self, shape, dtype=None):
        dtype = dtype if dtype else self.default_dtype
        return torch.ones(shape, dtype=dtype, device=self.device)


    def arange(self, start, stop=None, step=1, dtype=None):
        if dtype is None:
            float_args = isinstance(start, float) or isinstance(stop, float) or isinstance(step, float)
            dtype = self.default_dtype if float_args else self.int_dtype
        if stop is None:
            if self.device is not None:
                return torch.arange(start, dtype=dtype, device=self.device)
            return torch.arange(start, dtype=dtype)
        if self.device is not None:
            return torch.arange(start, stop, step, dtype=dtype, device=self.device)
        return torch.arange(start, stop, step, dtype=dtype)

    def stack(self, arrays, axis=0):
        # arrays should already be on the right device if created via this backend
        return torch.stack(arrays, dim=axis)

    def tolist(self, x):
        # Bring to CPU for safety
        return x.detach().cpu().tolist()

    def combinations(self, x, r):
        # torch.combinations keeps the input device
        x_flat = x.view(-1)
        return torch.combinations(x_flat, r=r)

    def unique(self, x, **kwargs):
        if 'axis' in kwargs:
            kwargs['dim'] = kwargs.pop('axis')
        return torch.unique(x, **kwargs)

    def gamma_inc(self, a, x):
        p = torch.igamma(a, x) # regularized
        ga = torch.exp(torch.lgamma(a))
        return ga * p

    def reciprocal(self, x):
        return torch.reciprocal(x)

    # def conv_transpose1d(self, *args, **kwargs):
    #     return torch.conv_transpose1d(*args, **kwargs)
    def conv_full_batch(self, a, b): # @TODO: needs to be in array_ops
        """ batch convolve a with b (full size output)
        a is (batch, n), b is (batch, m)
        output is (batch, n+m-1)
        """
        return torch.conv_transpose1d( # tranpose1d does "full" mode
            a[None,:,:], b[:,None,:], groups=a.shape[0]
            # a, b, groups=a.shape[0]
        ).squeeze(0)

    def tile(self, x, n):
        if hasattr(n, '__len__'):
            return torch.tile(x, dims=n)
        return torch.tile(x, dims=(n,))

    def transpose(self, x, axes=None):
        if axes is None:
            if len(x.shape) == 2:
                return torch.transpose(x, 0, 1)
            else:
                raise ValueError("transpose must specify axes for ndim>2")
        else:
            return torch.permute(x, dims=axes)

    def take_along_axis(self, x, indices, axis=None):
        return torch.take_along_dim(x, indices, dim=axis)

    def slice_add(self, x, axis, start, src):
        return super().slice_add(x, axis, start, src)

    def scatter_add_inplace(self, x, axis, index, src):
        # note: torch.scatter_reduce is more general and can 
        # do any ufunc, not just add
        return x.scatter_add_(dim=axis, index=index, src=src)

    def index_add_inplace(self, x, axis, index, src):
        return x.index_add_(dim=axis, index=index, source=src)

    def pad(self, x, pad_width, mode="constant", constant_values=0):
        # NumPy-style pad widths: ((before_0, after_0), ..., (before_n, after_n))
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
            return torch.cumprod(x.view(-1), dim=0)
        return torch.cumprod(x, dim=axis)

    def flip(self, x, axis=None):
        if axis is not None:
            if hasattr(axis, "__len__"):
                axis = list(axis)
            else:
                axis = [axis]
        return torch.flip(x, dims=axis)

    def segment_sum(self, x, lengths, axis=0):
        lengths = torch.as_tensor(lengths, device=x.device, dtype=torch.int64)
        if axis == 0:
            return torch.segment_reduce(x, lengths=lengths, axis=0, reduce="sum")
        x2 = x.movedim(axis, 0)
        y2 = torch.segment_reduce(x2, lengths=lengths, axis=0, reduce="sum")
        return y2.movedim(0, axis)
        # tf equivalent: tf.math.segment_sum with a moveaxis for axis!=0
        # jax equivalent: jax.ops.segment_sum with a moveaxis for axis!=0
        #  - might need to repeat lengths like in scatter_add, also set indices_are_sorted=True
        # numpy equivalent: add.reduceat, and ids are cumsum(lengths)


    # def segment_weighted_sum(self, x, lengths, coeff, axis=0):
    #     x0 = x.movedim(axis, 0).reshape(x.shape[axis], -1)
    #     c0 = coeff.reshape(-1, 1).to(dtype=x0.dtype, device=x0.device)

    #     lengths_t = torch.as_tensor(lengths, device=x0.device, dtype=torch.int64)
    #     seg_id = torch.repeat_interleave(
    #         torch.arange(lengths_t.numel(), device=x0.device),
    #         lengths_t
    #     )

    #     y0 = torch.zeros((lengths_t.numel(), x0.shape[1]), device=x0.device, dtype=x0.dtype)
    #     y0.index_add_(0, seg_id, x0 * c0)

    #     out_shape = list(x.shape)
    #     out_shape[axis] = lengths_t.numel()
    #     y = y0.reshape([out_shape[axis]] + [s for i, s in enumerate(out_shape) if i != axis]).movedim(0, axis)
    #     return y
    def segment_weighted_sum(self, x, lengths, coeff, axis=0):
        # Move axis to front: (L, R)
        x0 = x.movedim(axis, 0).reshape(x.shape[axis], -1)  # (L, R)
        c0 = coeff.reshape(-1, 1).to(dtype=x0.dtype, device=x0.device)  # (L, 1)

        lengths_t = torch.as_tensor(lengths, device=x0.device, dtype=torch.int64)  # (nseg,)
        seg_id = torch.repeat_interleave(
            torch.arange(lengths_t.numel(), device=x0.device),
            lengths_t
        )  # (L,)

        y0 = torch.zeros((lengths_t.numel(), x0.shape[1]), device=x0.device, dtype=x0.dtype)
        y0.index_add_(0, seg_id, x0 * c0)  # (nseg, R)

        # Reshape back and restore axis position
        out_shape = list(x.shape)
        out_shape[axis] = lengths_t.numel()
        y = y0.reshape([out_shape[axis]] + [s for i, s in enumerate(out_shape) if i != axis]).movedim(0, axis)
        return y


    def eigh(self, x):
        return torch.linalg.eigh(x)

    def eye(self, n, device=None, **kwargs):
        if device is None:
            device = self.device
        return torch.eye(n, device=device, **kwargs)

#end class TorchBackend(ArrayBackend):

backend_cls = TorchBackend
