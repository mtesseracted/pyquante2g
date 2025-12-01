import pytest
import pyquante22.array_ops as arr
from tests.array_ops.conftest import (
    INSTALLED_GPU_BACKENDS, AVAILABLE_GPUS, GPU_BACKENDS, backend_gpu)

pytestmark = pytest.mark.skipif(
    not INSTALLED_GPU_BACKENDS,
    reason=f"No GPU backends found ({'/'.join(GPU_BACKENDS.values())})",
)
pytestmark = pytest.mark.skipif(
    not AVAILABLE_GPUS,
    reason=f"No GPUs found with {'/'.join(INSTALLED_GPU_BACKENDS) or 'any backend'}",
)


def test_inits(backend_gpu):
    backend, dev = backend_gpu
    arr.set_backend(backend, device=dev)
    assert arr.name == backend
    assert arr.device == dev


def test_init_devs(backend_gpu):
    backend, dev = backend_gpu
    arr.set_backend(backend, device=dev)
    a = arr.arr([0,1])
    assert str(a.device) == dev # @TODO need to make a method that abstacts this
    assert a.dtype == arr.default_dtype
    a = arr.arange(4)
    assert str(a.device) == dev 
    assert a.dtype == arr.int_dtype
    a = arr.arange(4.0)
    assert str(a.device) == dev 
    assert a.dtype == arr.default_dtype
    a = arr.zeros((2,3))
    assert a.shape == (2,3)
    assert str(a.device) == dev
    assert a.dtype == arr.default_dtype
    a = arr.zeros((2,3), dtype=arr.int_dtype)
    assert a.dtype == arr.int_dtype
#end def test_init_devs(backend_gpu):

def test_array_ops_unary_gpu(backend_gpu):
    backend, dev = backend_gpu
    arr.set_backend(backend, device=dev)
    ops = arr.core.math_ops_unary
    n = arr.arange(2.) + arr.arr(1.)
    for op in ops:
        fn = getattr(arr, op)(n,)
        assert isinstance(fn, arr.ArrayType) and \
            (str(fn.device) == dev), f"{op} not callable"

