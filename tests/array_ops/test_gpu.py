import pytest
import pyquante22.array_ops as arr
from tests.array_ops.conftest import (
    INSTALLED_GPU_BACKENDS, AVAILABLE_GPUS, GPU_BACKENDS, backend_gpu)
from tests.array_ops.helpers import get_array_device_label

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not INSTALLED_GPU_BACKENDS,
        reason=f"No GPU backends found ({'/'.join(GPU_BACKENDS.values())})",
    ),
    pytest.mark.skipif(
        not AVAILABLE_GPUS,
        reason=f"No GPUs found with {'/'.join(INSTALLED_GPU_BACKENDS) or 'any backend'}",
    ),
]


def test_inits(backend_gpu):
    backend, dev = backend_gpu
    arr.set_backend(backend, device=dev)
    assert arr.name == backend
    assert arr.device == dev


def test_init_devs(backend_gpu):
    backend, dev = backend_gpu
    arr.set_backend(backend, device=dev)
    a = arr.arr([0,1])
    assert get_array_device_label(a) == dev
    assert a.dtype == arr.default_dtype
    a = arr.arange(4)
    assert get_array_device_label(a) == dev
    assert a.dtype == arr.int_dtype
    a = arr.arange(4.0)
    assert get_array_device_label(a) == dev
    assert a.dtype == arr.default_dtype
    a = arr.zeros((2,3))
    assert a.shape == (2,3)
    assert get_array_device_label(a) == dev
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
            (get_array_device_label(fn) == dev), f"{op} not callable"


def test_array_ops_binary_gpu(backend_gpu):
    backend, dev = backend_gpu
    arr.set_backend(backend, device=dev)
    ops = arr.core.math_ops_binary
    n = arr.arange(2.) + arr.arr(1.)
    for op in ops:
        fn = getattr(arr, op)(n,n)
        assert isinstance(fn, arr.ArrayType) and \
            (get_array_device_label(fn) == dev), f"{op} not callable"
