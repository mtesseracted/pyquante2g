import pytest
import pyquante22.array_ops as arr
import importlib
from tests.array_ops.conftest import backend, ftype, INSTALLED_BACKENDS, FLOAT_TYPES


def test_backend_init(backend, ftype):
    m = importlib.import_module(backend)
    mdtype = getattr(m, ftype, None)
    if mdtype is None:
        pytest.skip(f"No {ftype} in {backend}")
    arr.set_backend(backend)
    arr.set_dtype(ftype)
    assert arr.default_dtype is mdtype, f"{backend}@{ftype}"


def test_multiple_instances():
    # create multiple instances simulatneously
    aopss = {}
    for backend in INSTALLED_BACKENDS:
        m = importlib.import_module(backend)
        for ftype in FLOAT_TYPES:
            mdtype = getattr(m, ftype, None)
            if mdtype is None: continue
            aopss[(backend,ftype)] =arr.ArrayOps(backend, default_dtype=ftype)

    # now check that those instances initialized correctly
    # while all existing concurrently
    for backend in INSTALLED_BACKENDS:
        m = importlib.import_module(backend)
        for ftype in FLOAT_TYPES:
            mdtype = getattr(m, ftype, None)
            if mdtype is None: continue
            assert aopss[(backend,ftype)].name == backend
            assert aopss[(backend,ftype)].default_dtype is mdtype
#end def test_multiple_instances():


def test_array_ops_basic(backend):
    arr.set_backend(backend)

    a = arr.arr([0,1])
    assert isinstance(a, arr.ArrayType)
    assert a.shape == (2,)

    a = arr.zeros(4)
    assert a.shape == (4,)
    assert arr.sum(a) == arr.arr(0)

    a = arr.zeros((2,3))
    assert a.shape == (2,3)
    assert arr.sum(a) == arr.arr(0)

    a = arr.arange(2)
    assert a.shape == (2,)
    assert arr.sum(a) == arr.arr(1)
#end def test_array_ops_np_basic():


def test_array_ops_unary(backend):
    arr.set_backend(backend)
    ops = arr.core.math_ops_unary
    n = arr.arange(2.) + arr.arr(1.)
    for op in ops:
        fn = getattr(arr, op)(n,)
        assert isinstance(fn, arr.ArrayType) and \
            (fn.dtype == arr.default_dtype), f"{op} not callable"

def test_array_ops_binary(backend):
    arr.set_backend(backend)
    ops = arr.core.math_ops_binary
    n = arr.arange(2.) + arr.arr(1.)
    for op in ops:
        fn = getattr(arr, op)(n,n)
        assert isinstance(fn, arr.ArrayType) and \
            (fn.dtype == arr.default_dtype), f"{op} not callable"


def test_array_ops_reduction(backend):
    arr.set_backend(backend)
    ops = arr.core.reduction_ops
    n = arr.arange(2.) + arr.arr(1.)
    for op in ops:
        fn = getattr(arr, op)(n)
        assert (fn.dtype == arr.default_dtype), f"{op} not callable"


def test_array_ops_manipulation(backend):
    arr.set_backend(backend)
    ops = arr.core.reduction_ops
    for op in ops: # they have different function signatures, can't call
        assert callable(getattr(arr, op)), f"{op} not callable"


def test_array_ops_mixture(backend):
    arr.set_backend(backend)
    arr.set_dtype("float64")

    a = arr.arange(4.0)
    assert all(a == arr.sqrt(arr.pow(a,2)))
    assert all(arr.floor(a + 0.1) == a)
    assert arr.exp(a[0]) == 1
    b = arr.stack([a,a])
    assert b.shape == (2,len(a))
    assert arr.amax(a) == arr.arr(len(a)-1)
    assert arr.amin(a) == arr.arr(0)
    # this norm() test only works for float64 default_dtype
    assert arr.norm(a)**2 == arr.arr(14), f"{arr.norm(a)**2 - arr.arr(14)}, {a.dtype}"
    assert arr.tolist(a) == list(range(4))
    cs = arr.combinations(a, 2)
    assert cs.shape == (6,2)
#end def test_array_ops_np_ops():
 

# def test_conv_full_batch(backend):
#     arr.set_backend(backend)
#     arr.set_dtype("float64")
#     # from cart_gto calc_prod_binoms we want:
#     # conv_full_batch to have the same behavior
#     # from cart_gto triple_conv_dot behavior we want:
