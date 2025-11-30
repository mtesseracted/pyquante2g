import pytest
import pyquante22.array_ops as arr
import importlib

@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    return request.param

@pytest.fixture(params=["float32", "float64"])
def ftype(request):
    return request.param

def test_backend_init(backend, ftype):
    ilib = importlib.import_module(backend)
    arr.set_backend(backend)
    arr.set_dtype(ftype)
    print(f"{getattr(ilib, ftype)}")
    assert arr.default_dtype is getattr(ilib, ftype), f"{backend}@{ftype}"

def test_multiple_instances():
    nplib = importlib.import_module("numpy")
    thlib = importlib.import_module("torch")
    np64 = arr.ArrayOps("numpy", default_dtype="float64")
    np32 = arr.ArrayOps("numpy", default_dtype="float32")
    th64 = arr.ArrayOps("torch", default_dtype="float64")
    th32 = arr.ArrayOps("torch", default_dtype="float32")
    assert (np64.name == "numpy") and (np64.default_dtype is getattr(nplib, "float64"))
    assert (np32.name == "numpy") and (np32.default_dtype is getattr(nplib, "float32"))
    assert (th64.name == "torch") and (th64.default_dtype is getattr(thlib, "float64"))
    assert (th32.name == "torch") and (th32.default_dtype is getattr(thlib, "float32"))


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
 
# @TODO: get on a gpu machine and test 'device' is done correctly