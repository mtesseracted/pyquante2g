import pytest
import pyquante22.array_ops as arr
from tests.array_ops.conftest import backend, ftype, itype
import importlib


def test_core(backend):
    assert isinstance(arr.get_default_ops(), arr.ArrayOps)
    assert arr.get_default_ops() is arr._default_ops
    assert hasattr(arr, "set_backend")
    arr.set_backend(backend)
    assert arr.name == backend


def test_instance(backend):
    ops_instance = arr.ArrayOps(backend)
    assert ops_instance.name == backend


def test_invalid():
    with pytest.raises(ValueError):
        arr.set_backend("notreal")
    with pytest.raises(ValueError):
        arr.ArrayOps("notreal")


def test_default_dtype(backend, ftype):
    m = importlib.import_module(backend)
    mdtype = getattr(m, ftype, None)
    if mdtype is None:
        pytest.skip(f"No {ftype} in {backend}")
    opi = arr.ArrayOps(backend, default_dtype=ftype)
    assert opi.default_dtype is mdtype


def test_int_dtype(backend, itype,):
    m = importlib.import_module(backend)
    mdtype = getattr(m, itype, None)
    if mdtype is None:
        pytest.skip(f"No {ftype} in {backend}")
    opi = arr.ArrayOps(backend, int_dtype=itype)
    assert opi.int_dtype is mdtype
