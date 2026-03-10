import pytest
import pyquante22.array_ops as arr
import importlib
from tests.array_ops.conftest import backend, ftype, INSTALLED_BACKENDS, FLOAT_TYPES
from tests.array_ops.optest_helpers import (
    assert_smoke_output,
    call_op,
    make_ops,
)


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
    ops = make_ops(backend)

    a = ops.arr([0,1])
    assert isinstance(a, ops.ArrayType)
    assert a.shape == (2,)

    a = ops.zeros(4)
    assert a.shape == (4,)
    assert ops.sum(a) == ops.arr(0)

    a = ops.zeros((2,3))
    assert a.shape == (2,3)
    assert ops.sum(a) == ops.arr(0)

    a = ops.arange(2)
    assert a.shape == (2,)
    assert ops.sum(a) == ops.arr(1)
#end def test_array_ops_np_basic():


@pytest.mark.parametrize("op_name", arr.core.testable_ops)
def test_array_ops_smoke(backend, op_name):
    ops = make_ops(backend)
    out = call_op(ops, op_name)

    assert_smoke_output(op_name, out, ops)


def test_array_ops_mixture(backend):
    ops = make_ops(backend)

    a = ops.arange(4.0)
    assert all(a == ops.sqrt(ops.pow(a,2)))
    assert all(ops.floor(a + 0.1) == a)
    assert ops.exp(a[0]) == 1
    b = ops.stack([a,a])
    assert b.shape == (2,len(a))
    assert ops.amax(a) == ops.arr(len(a)-1)
    assert ops.amin(a) == ops.arr(0)
    # this norm() test only works for float64 default_dtype
    assert ops.norm(a)**2 == ops.arr(14), f"{ops.norm(a)**2 - ops.arr(14)}, {a.dtype}"
    assert ops.tolist(a) == list(range(4))
    cs = ops.combinations(a, 2)
    assert cs.shape == (6,2)
#end def test_array_ops_np_ops():
 

# def test_conv_full_batch(backend):
#     arr.set_backend(backend)
#     arr.set_dtype("float64")
#     # from cart_gto calc_prod_binoms we want:
#     # conv_full_batch to have the same behavior
#     # from cart_gto triple_conv_dot behavior we want:
