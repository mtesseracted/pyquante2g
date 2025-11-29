import pyquante22.array_ops as arr
from typing import Protocol

def test_core():
    assert issubclass(arr.ArrayBackend, Protocol)
    assert isinstance(arr.get_default_ops(), arr.ArrayOps)
    arr.set_backend("numpy")
    assert arr.name == "numpy"
    ops_instance = arr.ArrayOps("numpy")
    assert ops_instance.name == "numpy"
