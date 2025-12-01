import pytest
import importlib.util
from tests.array_ops.helpers import get_gpu0


def _module_is_available(modname: str) -> bool:
    """Return True if module importable without actually importing it."""
    return importlib.util.find_spec(modname) is not None

_bits = [16, 32, 64, 128]
FLOAT_TYPES = [f"float{b}" for b in _bits]
@pytest.fixture(scope="function", params=FLOAT_TYPES)
def ftype(request):
    return request.param

INT_TYPES = [f"int{b}" for b in _bits]
@pytest.fixture(scope="function", params=INT_TYPES)
def itype(request):
    return request.param

# Map nicknames to module names 
CPU_BACKENDS = {"numpy": "numpy",}
GPU_BACKENDS = {
    "jax": "jax",
    "cupy": "cupy",
    "torch": "torch",
    "tensorflow": "tensorflow",
}
ALL_BACKENDS = {**CPU_BACKENDS, **GPU_BACKENDS}

INSTALLED_CPU_BACKENDS = [name for name, mod in CPU_BACKENDS.items() 
                        if _module_is_available(mod)]
INSTALLED_GPU_BACKENDS = [name for name, mod in GPU_BACKENDS.items() 
                        if _module_is_available(mod)]
INSTALLED_BACKENDS = [*INSTALLED_CPU_BACKENDS, *INSTALLED_GPU_BACKENDS]

_AVAILABLE_GPUS = [get_gpu0(backend) for backend in INSTALLED_GPU_BACKENDS]
AVAILABLE_GPUS = {name:gpu 
    for name,gpu in zip(INSTALLED_GPU_BACKENDS, _AVAILABLE_GPUS) if gpu}

@pytest.fixture(scope="function", params=INSTALLED_CPU_BACKENDS)
def backend_cpu(request):
    return request.param

@pytest.fixture(scope="function", params=INSTALLED_BACKENDS)
def backend(request):
    return request.param

@pytest.fixture(scope="function", 
    params=list(AVAILABLE_GPUS.items()), ids=lambda p: p[0])
def backend_gpu(request):
    return request.param

