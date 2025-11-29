# array_ops/torch.py
import torch

from .core import ArrayBackend


class TorchBackend(ArrayBackend):
    name = "torch"
    array_type = torch.Tensor

    def init(self, data, dtype=None) -> torch.Tensor:
        return torch.as_tensor(data, dtype=dtype)

    def zeros(self, shape, dtype=None) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    def sum(
        self,
        x: torch.Tensor,
        axis=None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        """
        NumPy-style sum(x, axis=..., keepdims=...) mapped to torch.sum.
        """
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis, keepdim=keepdims)


backend = TorchBackend()

