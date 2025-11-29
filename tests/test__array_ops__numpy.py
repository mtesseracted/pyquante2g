import pyquante22.array_ops as arr
import numpy as np


def test_array_ops_np_basic():
    arr.set_backend("numpy")
    a = arr.init([0,1])
    assert isinstance(a, np.ndarray)
    assert a.shape == (2,)

    a = arr.zeros(4)
    assert a.shape == (4,)
    assert arr.sum(a) == 0

    a = arr.zeros((2,3))
    assert a.shape == (2,3)
    assert arr.sum(a) == 0

    a = arr.arange(2)
    assert a.shape == (2,)
    assert arr.sum(a) == 1
#end def test_array_ops_np_basic():


def test_array_ops_np_ops():
    arr.set_backend("numpy")

    a = arr.arange(4)
    assert all(a == arr.sqrt(arr.pow(a,2)))
    assert all(arr.floor(a + 0.1) == a)
    assert arr.exp(a[0]) == 1
    b = arr.stack([a,a])
    assert b.shape == (2,len(a))
    assert arr.amax(a) == len(a)-1
    assert arr.amin(a) == 0
    assert arr.norm(a)**2 == 14
    assert arr.tolist(a) == list(range(4))
    cs = arr.combinations(a, 2)
    assert cs.shape == (6,2)
#end def test_array_ops_np_ops():
    

def test_array_ops_np_unary():
    arr.set_backend("numpy")
    ops = arr.core.math_ops_unary
    n = arr.arange(2.) + arr.init(1.)
    for op in ops:
        f = getattr(arr, op)
        assert isinstance(f(n), np.ndarray), f"{op} not callable"

def test_array_ops_np_binary():
    arr.set_backend("numpy")
    ops = arr.core.math_ops_unary
    n = arr.arange(2.) + arr.init(1.)
    for op in ops:
        f = getattr(arr, op)
        assert isinstance(f(n,n), np.ndarray), f"{op} not callable"

def test_array_ops_np_reduction():
    arr.set_backend("numpy")
    ops = arr.core.reduction_ops
    n = arr.arange(2.) + arr.init(1.)
    for op in ops:
        f = getattr(arr, op)
        # assert callable(f), f"{op} not callable"
        # @TODO need to set default dtype
        assert isinstance(f(n), np.float64), f"{op} not callable"

# def test_array_ops_np_manipulation():
#     arr.set_backend("numpy")
#     ops = arr.core.creation_ops
#     for op in ops:
#         f = getattr(arr, op)
#         assert isinstance(f(2), np.ndarray), f"{op} not callable"


