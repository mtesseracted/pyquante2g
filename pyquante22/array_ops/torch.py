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
        if self.device is not None:
            return torch.zeros(shape, dtype=dtype, device=self.device)
        return torch.zeros(shape, dtype=dtype)

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
        return torch.stack(list(arrays), dim=axis)

    def tolist(self, x):
        # Bring to CPU for safety
        return x.detach().cpu().tolist()

    def combinations(self, x, r):
        # torch.combinations keeps the input device
        x_flat = x.view(-1)
        return torch.combinations(x_flat, r=r)

    def gamma_inc(self, a, x):
        p = torch.igamma(a, x) # regularized
        ga = torch.exp(torch.lgamma(a))
        return ga * p

    def conv_transpose1d(self, *args, **kwargs):
        return torch.conv_transpose1d(*args, **kwargs)

    def tile(self, x, n):
        return torch.tile(x, dims=(n,))
#end class TorchBackend(ArrayBackend):

backend_cls = TorchBackend
