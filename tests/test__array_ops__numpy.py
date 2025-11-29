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
    