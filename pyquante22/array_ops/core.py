# array_ops/core.py
from itertools import combinations as _combinations


# ops that need to be implemented but have the same function signature (mostly)
# across all backends, split them so they're easier to test
math_ops_unary = [
    "sqrt", "square",
    "exp", "log",
    "sin", "cos", "atan",
    "floor", "ceil", 
    "lgamma",
]
math_ops_binary = [
    "pow", 
    "gamma_inc", # not regularized
]
reduction_ops = [
    "sum", "prod",
    "amax", "amin",
]
# creation_ops = ["arange",]
manipulation_ops = [
    "reshape", "view",
    "stack", "where",
    "broadcast_to", "tile",
]
basic_ops = sum([
    math_ops_unary, 
    math_ops_binary,
    reduction_ops,
    # creation_ops,
    manipulation_ops,
], []) # concat lists


class ArrayBackend:
    """
    Base class specifying the full array API and default implementations.

    Subclasses MUST set:
        - self.xp   (the underlying array namespace, e.g. numpy, torch, cupy)
        - self.name (string, e.g. 'numpy', 'torch', 'cupy')

    Subclasses MAY override methods to handle `self.device` in a
    backend-specific way.
    """

    xp = None           # imoprted backend module
    name = "base"       # backend module name
    # ArrayType: type     # type for arrays

    def __init__(self, device=None, 
                 default_dtype=None, int_dtype=None,
    ):
        # Backend-specific interpretation of this is up to subclasses.
        self.device = device
        self.default_dtype = self._resolve_dtype(default_dtype)
        self.int_dtype = self._resolve_dtype(int_dtype)

    def _resolve_dtype(self, dtype):
        if isinstance(dtype, str):
            return getattr(self.xp, dtype)
        return dtype

    # ---------- conversion / creation ----------

    def tolist(self, x):
        return x.tolist()

    def asarray(self, data, dtype=None):
        if dtype is None:
            dtype = self.default_dtype
        return self.xp.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.default_dtype
        return self.xp.zeros(shape, dtype=dtype)

    def arange(self, start, stop=None, step=1, dtype=None):
        if dtype is None:
            dtype = self.default_dtype
        if stop is None:
            return self.xp.arange(start, dtype=dtype)
        return self.xp.arange(start, stop, step, dtype=dtype)

    # short-hands
    def init(self, data, dtype=None):
        return self.asarray(data, dtype=dtype)
    def arr(self, data, dtype=None):
        return self.asarray(data, dtype=dtype)
    # ---------- reductions / norms ----------

    def norm(self, x, axis=None, keepdims=False):
        return self.xp.linalg.norm(x, axis=axis, keepdims=keepdims)

    # ---------- combinatorics ----------

    def combinations(self, x, r):
        """
        Default: build combinations on Python side, then convert back.
        Backends can override to stay on-device / be faster.
        """
        seq = self.tolist(x)
        combs = list(_combinations(seq, r))
        return self.asarray(combs)
#end class ArrayBackend:

## add the basic ops to ArrayBackend here
def _make_op(name):
    def op(self, *args, **kwargs):
        return getattr(self.xp, name)(*args, **kwargs)
    op.__name__ = name
    return op

for _name in basic_ops:
    setattr(ArrayBackend, _name, _make_op(_name))


# ---------- loader + multi-instance facade ----------

def _load_backend(name,):
    name = name.lower()
    if name == "numpy":
        from . import numpy as m
    elif name in ("torch", "pytorch"):
        from . import torch as m
    elif name == "cupy":
        from . import cupy as m
    else:
        raise ValueError(f"Unknown backend {name!r}")
    return m.backend_cls()


class ArrayOps:
    """
    Per-instance facade that forwards every op to a chosen backend.

    backend can be:
      - a string: 'numpy', 'torch', 'cupy'
      - an already-constructed backend instance

    device: device where arrays will reside in memory
    default_dtype: dtype for floats
    int_dtype: dtype for ints
    """
    def __init__(self, backend, device=None, 
                 default_dtype=None, int_dtype=None):

        self.set_backend(backend, device=device, 
                         default_dtype=default_dtype, int_dtype=int_dtype)

    def set_backend(self, backend, device=None, 
                    default_dtype=None, int_dtype=None):
        if isinstance(backend, str):
            self._backend = _load_backend(backend)
        elif isinstance(backend, ArrayBackend):
            self._backend = backend
        else:
            raise ValueError(f"Uknown backend type: {type(backend)}")

        if device:
            self.set_device(device)
        if default_dtype or int_dtype:
            self.set_dtype(default_dtype, int_dtype=int_dtype)
    #end def set_backend()

    def set_device(self, device):
        self._backend.device = device

    def set_dtype(self, dtype=None, int_dtype=None):
        if dtype:
            self._backend.default_dtype = self._backend._resolve_dtype(dtype)
        if int_dtype:
            self._backend.int_dtype= self._backend._resolve_dtype(int_dtype)

    @property
    def backend(self):
        return self._backend

    @property
    def backend_name(self):
        return getattr(self._backend, "name", "<unknown>")

    @property
    def device(self):
        return getattr(self._backend, "device", None)

    @property
    def ArrayType(self):
        return self.backend.ArrayType

    def __getattr__(self, name):
        # Forward all ops (init, sqrt, sum, arange, combinations, …)
        try: # @TODO this probably causes a performance penalty, not clean
            # but maybe add the error string as the default val in getattr...
            return getattr(self._backend, name)
        except AttributeError:
            raise AttributeError(f"The {self.backend.name} backend doesn't have {name}" + \
                                 "\ntry adding to basic_ops in array_ops/core.py")
#end class ArrayOps:
