from __future__ import annotations
from typing import Optional, Tuple, Union, Any

import torch

IndexLike = Union[int, torch.Tensor]


class SymmetricArray4d:
    """
    4D array T[i,j,k,l] with symmetries:
      (i <-> j), (k <-> l), and (ij) <-> (kl)

    Storage: fully preallocated packed-unique vector `data` of length n_uniq
    (dense over uniques, minimal over full 4D).

    Supports:
      - T[i,j,k,l] get/set via quad_index
      - torch.sum(T) (fast for full sum, falls back to dense for dim != None)
      - torch.einsum(..., T, ...) (falls back to dense for correctness)
    """

    def __init__(
        self,
        n: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        fill_value: float = 0.0,
        requires_grad: bool = False,
        precompute_multiplicity: bool = True,  # used for fast torch.sum(T)
    ):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)
        self.n_pair = self.n * (self.n + 1) // 2
        self.n_uniq = self.n_pair * (self.n_pair + 1) // 2

        self.data = torch.full(
            (self.n_uniq,),
            fill_value,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        self._multiplicity: Optional[torch.Tensor] = None
        if precompute_multiplicity:
            self._multiplicity = self._build_multiplicity(self.device)

    # ---------- symmetry packing ----------
    @staticmethod
    def _pair_index(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        a = torch.maximum(i, j)
        b = torch.minimum(i, j)
        return a * (a + 1) // 2 + b

    @classmethod
    def _quad_index(cls, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        p = cls._pair_index(i, j)
        q = cls._pair_index(k, l)
        hi = torch.maximum(p, q)
        lo = torch.minimum(p, q)
        return hi * (hi + 1) // 2 + lo

    def quad_index(self, i: IndexLike, j: IndexLike, k: IndexLike, l: IndexLike) -> torch.Tensor:
        dev = self.device
        i = torch.as_tensor(i, dtype=torch.long, device=dev)
        j = torch.as_tensor(j, dtype=torch.long, device=dev)
        k = torch.as_tensor(k, dtype=torch.long, device=dev)
        l = torch.as_tensor(l, dtype=torch.long, device=dev)
        return self._quad_index(i, j, k, l)

    # ---------- tensor-like properties ----------
    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.n, self.n, self.n, self.n)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, "
                f"device={self.device}, n_uniq={self.n_uniq})")

    # ---------- indexing ----------
    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 4:
            raise TypeError("Indexing must be a 4-tuple: T[i,j,k,l]")
        if any(isinstance(x, (slice, type(Ellipsis))) for x in key):
            raise NotImplementedError("Slice/ellipsis indexing not implemented for packed storage.")

        i, j, k, l = key
        idx = self.quad_index(i, j, k, l)

        # Optional bounds checks (comment out for max speed)
        for name, t in zip(("i", "j", "k", "l"), (i, j, k, l)):
            tt = torch.as_tensor(t)
            if (tt < 0).any() or (tt >= self.n).any():
                raise IndexError(f"Index {name} out of bounds for n={self.n}")

        return self.data[idx]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 4:
            raise TypeError("Assignment must be a 4-tuple: T[i,j,k,l] = value")
        if any(isinstance(x, (slice, type(Ellipsis))) for x in key):
            raise NotImplementedError("Slice/ellipsis assignment not implemented for packed storage.")

        i, j, k, l = key
        idx = self.quad_index(i, j, k, l)
        v = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self.data[idx] = v

    # ---------- conversions / utils ----------
    def clone(self) -> "SymmetricArray4d":
        out = SymmetricArray4d(self.n, dtype=self.dtype, device=self.device, fill_value=0.0)
        out.data = self.data.clone()
        out._multiplicity = self._multiplicity.clone() if self._multiplicity is not None else None
        return out

    def to(self, *args, **kwargs) -> "SymmetricArray4d":
        out = SymmetricArray4d(self.n, dtype=self.dtype, device=self.device, fill_value=0.0)
        out.data = self.data.to(*args, **kwargs)
        if self._multiplicity is not None:
            out._multiplicity = self._multiplicity.to(out.data.device)
        return out

    def requires_grad_(self, requires_grad: bool = True) -> "SymmetricArray4d":
        self.data.requires_grad_(requires_grad)
        return self

    def detach(self) -> "SymmetricArray4d":
        out = SymmetricArray4d(self.n, dtype=self.dtype, device=self.device, fill_value=0.0)
        out.data = self.data.detach()
        out._multiplicity = self._multiplicity  # ok to share; it's constant
        return out

    def as_packed(self) -> torch.Tensor:
        """The packed-unique storage tensor (length n_uniq)."""
        return self.data

    def to_dense_4d(self) -> torch.Tensor:
        """
        Materialize full (n,n,n,n). O(n^4) memory/time — only for small n.
        """
        n = self.n
        ii = torch.arange(n, device=self.device, dtype=torch.long)
        i, j, k, l = torch.meshgrid(ii, ii, ii, ii, indexing="ij")
        idx = self._quad_index(i, j, k, l)
        return self.data[idx]

    # ---------- fast full sum using multiplicities ----------
    @staticmethod
    def _tri_unrank(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of t = hi*(hi+1)/2 + lo with hi>=lo>=0.
        Returns (hi, lo). Works for x within float64-exact integer range.
        """
        x = x.to(torch.long)
        # hi = floor((sqrt(8x+1)-1)/2)
        hi = torch.floor((torch.sqrt((8 * x).to(torch.float64) + 1.0) - 1.0) * 0.5).to(torch.long)
        lo = x - hi * (hi + 1) // 2
        return hi, lo

    @staticmethod
    def _pair_unrank(p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of pair_index(a,b) = a(a+1)/2 + b with a>=b>=0.
        Returns (a,b).
        """
        return SymmetricArray4d._tri_unrank(p)

    def _build_multiplicity(self, device: torch.device) -> torch.Tensor:
        """
        multiplicity[idx] = number of distinct (i,j,k,l) tuples in the orbit of the symmetry group
        (swap ij, swap kl, swap pairs), i.e., 8 / stabilizer_size.

        This enables fast torch.sum(T) without densifying.
        """
        idx = torch.arange(self.n_uniq, device=device, dtype=torch.long)
        p, q = self._tri_unrank(idx)     # p>=q are pair-indices
        a, b = self._pair_unrank(p)      # a>=b
        c, d = self._pair_unrank(q)      # c>=d

        s_ij = torch.where(a == b, 2, 1)     # swap(i,j) stabilizer
        s_kl = torch.where(c == d, 2, 1)     # swap(k,l) stabilizer
        s_pq = torch.where(p == q, 2, 1)     # swap pairs stabilizer

        stab = s_ij * s_kl * s_pq
        mult = (8 // stab).to(torch.long)
        return mult

    def _ensure_multiplicity(self):
        if self._multiplicity is None or self._multiplicity.device != self.device:
            self._multiplicity = self._build_multiplicity(self.device)

    def sum(self, dim=None, keepdim: bool = False):
        """
        Mirror Tensor.sum API.
        Fast path only for full sum (dim=None).
        For dim != None, we fall back to dense 4D (correct but expensive).
        """
        if dim is None:
            self._ensure_multiplicity()
            out = (self.data * self._multiplicity.to(self.data.dtype)).sum()
            if keepdim:
                # keepdim=True with dim=None returns scalar in torch; match that behavior.
                return out
            return out

        # Correctness-first fallback for partial reductions
        dense = self.to_dense_4d()
        return dense.sum(dim=dim, keepdim=keepdim)

    # ---------- torch interop: sum + einsum ----------
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # torch.sum(T, ...)
        if func is torch.sum:
            (x, *rest) = args
            if isinstance(x, SymmetricArray4d):
                dim = kwargs.get("dim", None) if rest == [] else (rest[0] if len(rest) >= 1 else kwargs.get("dim", None))
                keepdim = kwargs.get("keepdim", False) if len(rest) < 2 else rest[1]
                return x.sum(dim=dim, keepdim=keepdim)

        # torch.einsum(equation, *operands)
        if func is torch.einsum:
            equation = args[0]
            operands = list(args[1:])
            operands = [op.to_dense_4d() if isinstance(op, SymmetricArray4d) else op for op in operands]
            return torch.einsum(equation, *operands, **kwargs)

        return NotImplemented


# ---- quick sanity ----
if __name__ == "__main__":
    T = SymmetricArray4d(6, dtype=torch.float64, fill_value=0.0, precompute_multiplicity=True)
    T[2, 1, 4, 0] = 3.14

    # symmetry indexing
    print(T[1, 2, 0, 4])  # 3.14

    # torch.sum(T) works (fast full sum)
    print(torch.sum(T))

    # einsum works (densifies)
    A = torch.randn(6, 6, dtype=torch.float64)
    # Example: contract last two indices with a matrix twice (just a demo)
    out = torch.einsum("ijkl,km,ln->ijmn", T, A, A)
    print(out.shape)  # (6,6,6,6)
    debug_var = 1

