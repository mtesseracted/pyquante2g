# array_ops/torch.py
import torch

from .core import ArrayBackend


class TorchBackend(ArrayBackend):
    name = "torch"
    array_type = torch.Tensor

    # --- core creation / reduction ---

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
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis, keepdim=keepdims)

    # --- elementwise / math ---

    def sqrt(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    def pow(self, x, y) -> torch.Tensor:
        return torch.pow(x, y)

    def exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def floor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)

    # --- creation / shape ops ---

    # array_ops/torch.py

    def arange(self, start, stop=None, step=1, dtype=None) -> torch.Tensor:
        if stop is None:
            return torch.arange(0, start, step, dtype=dtype)
        return torch.arange(start, stop, step, dtype=dtype)

    def stack(self, arrays, axis=0) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    # --- norms / reductions ---

    def norm(
        self,
        x: torch.Tensor,
        axis=None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if axis is None:
            return torch.linalg.norm(x)
        return torch.linalg.norm(x, dim=axis, keepdim=keepdims)

    def amin(
        self,
        x: torch.Tensor,
        axis=None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if axis is None:
            return torch.amin(x)
        return torch.amin(x, dim=axis, keepdim=keepdims)

    def amax(
        self,
        x: torch.Tensor,
        axis=None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if axis is None:
            return torch.amax(x)
        return torch.amax(x, dim=axis, keepdim=keepdims)

    # --- conversion / combinatorics ---

    def tolist(self, x: torch.Tensor):
        return x.detach().cpu().tolist()

    def combinations(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """
        All r-combinations (no replacement) of a 1D tensor.

        Returns a tensor of shape (n_combinations, r).
        """
        x_flat = x.view(-1)
        return torch.combinations(x_flat, r=r)


backend = TorchBackend()
