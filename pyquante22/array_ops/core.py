# array_ops/core.py
from itertools import combinations as _combinations


convert_data_ops = ["arr", "init", "asarray"]
tolist_ops = ["tolist"]
shape_create_ops = ["zeros", "ones"]
range_create_ops = ["arange"]
norm_ops = ["norm"]
combinatoric_ops = ["combinations"]
conv_batch_ops = ["conv_full_batch"]

unary_positive_ops = ["sqrt", "log", "lgamma", "reciprocal"]
unary_array_ops = [
    "square",
    "exp",
    "sin",
    "cos",
    "atan",
    "abs",
    "sign",
    "round",
    "floor",
    "ceil",
]
unary_axis_ops = ["cumprod"]
predicate_array_ops = ["isfinite"]

binary_positive_ops = [
    "pow",
    "gamma_inc",  # not regularized
]
binary_array_ops = ["maximum", "minimum"]

numeric_reduction_ops = ["sum", "prod", "amax", "amin"]
logical_reduction_ops = ["any", "all"]

eye_ops = ["eye"]
reshape_ops = ["reshape",] # "view"]
broadcast_ops = ["broadcast_to"]
transpose_ops = ["transpose"]
tile_ops = ["tile"]
stack_ops = ["stack", "concat"]
where_ops = ["where"]
take_ops = ["take_along_axis"]
slice_ops = ["slice_add"]
scatter_ops = ["scatter_add", "scatter_add_inplace", "index_add_inplace"]
pad_ops = ["pad"]
flip_ops = ["flip"]
unique_ops = ["unique"]
einsum_ops = ["einsum"]
conv1d_ops = ["conv1d"]
segment_ops = ["segment_sum"]
diag_ops = ["diag"]
comparison_ops = ["allclose"]
broadcast_shape_ops = ["broadcast_shapes"]
dtype_info_ops = ["finfo", "iinfo"]

OPS_BY_SIGNATURE = {
    "convert_data": convert_data_ops,
    "tolist": tolist_ops,
    "shape_create": shape_create_ops,
    "range_create": range_create_ops,
    "norm": norm_ops,
    "combinations": combinatoric_ops,
    "conv_batch": conv_batch_ops,
    "unary_positive": unary_positive_ops,
    "unary_array": unary_array_ops,
    "unary_axis": unary_axis_ops,
    "predicate_array": predicate_array_ops,
    "binary_positive": binary_positive_ops,
    "binary_array": binary_array_ops,
    "numeric_reduce": numeric_reduction_ops,
    "logical_reduce": logical_reduction_ops,
    "eye": eye_ops,
    "reshape": reshape_ops,
    "broadcast": broadcast_ops,
    "transpose": transpose_ops,
    "tile": tile_ops,
    "stack": stack_ops,
    "where": where_ops,
    "take": take_ops,
    "slice": slice_ops,
    "scatter": scatter_ops,
    "pad": pad_ops,
    "flip": flip_ops,
    "unique": unique_ops,
    "einsum": einsum_ops,
    "conv1d": conv1d_ops,
    "segment": segment_ops,
    "diag": diag_ops,
    "comparison": comparison_ops,
    "broadcast_shape": broadcast_shape_ops,
    "dtype_info": dtype_info_ops,
}
OP_SIGNATURES = {
    op_name: sig_name
    for sig_name, ops in OPS_BY_SIGNATURE.items()
    for op_name in ops
}

# Backward-compatible coarse groups used by older tests.
math_ops_unary = unary_positive_ops + unary_array_ops + unary_axis_ops
math_ops_binary = binary_positive_ops + binary_array_ops
reduction_ops = numeric_reduction_ops
creation_ops = eye_ops
manipulation_ops = (
    reshape_ops
    + broadcast_ops
    + transpose_ops
    + tile_ops
    + stack_ops
    + where_ops
    + take_ops
    + slice_ops
    + scatter_ops
    + pad_ops
    + flip_ops
)
misc_ops = (
    unique_ops
    + einsum_ops
    + conv1d_ops
    + segment_ops
    + diag_ops
    + predicate_array_ops
    + logical_reduction_ops
    + comparison_ops
    + broadcast_shape_ops
    + dtype_info_ops
)
basic_ops = sum(
    [
        math_ops_unary,
        math_ops_binary,
        reduction_ops,
        creation_ops,
        manipulation_ops,
        misc_ops,
    ],
    [],
)  # concat lists
testable_ops = sum(OPS_BY_SIGNATURE.values(), [])


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

    def ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.default_dtype
        return self.xp.ones(shape, dtype=dtype) # BUG? no device

    def arange(self, start, stop=None, step=1, dtype=None):
        # infer dtype from args if not given
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
        return self.asarray(combs, dtype=x.dtype)

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

    # ---------- other ----------

    def conv_full_batch(self, a, b):
        raise NotImplementedError("conv_full_batch not implemented for backend " + self.name)   

#end class ArrayBackend:

## add the basic ops to ArrayBackend here
def _make_op(name):
    def op(self, *args, **kwargs):
        return getattr(self.xp, name)(*args, **kwargs)
    op.__name__ = name
    return op

for _name in basic_ops:
    if hasattr(ArrayBackend, _name):
        continue
    setattr(ArrayBackend, _name, _make_op(_name))


# ---------- loader + multi-instance facade ----------

def _load_backend(name,):
    name = name.lower()
    if name == "numpy":
        from . import numpy as m
    elif name == "jax":
        from . import jax as m
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
