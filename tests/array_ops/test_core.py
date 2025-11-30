import pyquante22.array_ops as arr

def test_core():
    assert isinstance(arr.get_default_ops(), arr.ArrayOps)
    arr.set_backend("numpy")
    assert arr.name == "numpy"
    ops_instance = arr.ArrayOps("numpy")
    assert ops_instance.name == "numpy"
