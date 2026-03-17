import numpy as np

import pyquante22.array_ops as arr


EPS = np.finfo(np.float64).eps


def make_ops(backend_name):
    return arr.ArrayOps(
        backend_name,
        device=None,
        default_dtype="float64",
        int_dtype="int64",
    )


def available_testable_ops(ops):
    return [
        name
        for name in arr.core.testable_ops
        if callable(getattr(ops, name, None))
    ]


def _sample_inputs(ops):
    return {
        "x": ops.arr([[1.25, -2.5], [3.75, -4.5]]),
        "xpos": ops.arr([[0.5, 1.5], [2.5, 3.5]]),
        "y": ops.arr([[0.25, 1.0], [1.5, 2.0]]),
        "xb": ops.arr([[True, False], [False, True]]),
        "xi": ops.arr([0, 1, 2, 3], dtype=ops.int_dtype),
    }


def call_spec_for_signature(sig_name, ops):
    samples = _sample_inputs(ops)
    x = samples["x"]
    xpos = samples["xpos"]
    y = samples["y"]
    xb = samples["xb"]
    xi = samples["xi"]

    if sig_name == "convert_data":
        return ([[1.25, -2.5], [3.75, -4.5]],), {}
    if sig_name == "tolist":
        return (x,), {}
    if sig_name == "shape_create":
        return ((2, 3),), {}
    if sig_name == "range_create":
        return (0, 6, 2), {}
    if sig_name == "norm":
        return (x,), {}
    if sig_name == "combinations":
        return (xi, 2), {}
    if sig_name == "conv_batch":
        a = ops.arr([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        b = ops.arr([[1.0, 0.5], [1.0, -1.0]])
        return (a, b), {}
    if sig_name == "unary_positive":
        return (xpos,), {}
    if sig_name == "unary_array":
        return (x,), {}
    if sig_name == "unary_axis":
        return (x,), {}
    if sig_name == "predicate_array":
        return (x,), {}
    if sig_name == "binary_positive":
        return (xpos, y), {}
    if sig_name == "binary_array":
        return (x, y), {}
    if sig_name == "numeric_reduce":
        return (x,), {}
    if sig_name == "logical_reduce":
        return (ops.arr([[True, False], [True, True]]),), {}
    if sig_name == "eye":
        return (3,), {}
    if sig_name == "reshape":
        return (ops.arange(6.0), (2, 3)), {}
    if sig_name == "broadcast":
        return (ops.arr([1.0, 2.0]), (2, 2)), {}
    if sig_name == "transpose":
        return (ops.arr([[1.0, 2.0], [3.0, 4.0]]),), {}
    if sig_name == "tile":
        return (ops.arr([1.0, 2.0]), (2,)), {}
    if sig_name == "stack":
        arrays = [ops.arr([1.0, 2.0]), ops.arr([3.0, 4.0])]
        return (arrays,), {}
    if sig_name == "where":
        return (xb > 0, xpos, y), {}
    if sig_name == "take":
        a = ops.arr([[10.0, 30.0, 20.0], [60.0, 40.0, 50.0]])
        idx = ops.arr([[2, 1, 0], [0, 2, 1]], dtype=ops.int_dtype)
        return (a, idx), {"axis": 1}
    if sig_name == "slice":
        a = ops.zeros((3, 4))
        src = ops.arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return (a, 1, 1, src), {}
    if sig_name == "scatter":
        a = ops.zeros((3, 2))
        idx = ops.arr([[0, 1], [1, 0], [2, 1]], dtype=ops.int_dtype)
        src = ops.arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return (a, 0, idx, src), {}
    if sig_name == "pad":
        return (ops.arr([[1.0, 2.0], [3.0, 4.0]]), ((1, 1), (1, 1))), {}
    if sig_name == "flip":
        return (ops.arr([[1.0, 2.0], [3.0, 4.0]]),), {"axis": 0}
    if sig_name == "unique":
        return (ops.arr([3, 1, 3, 2, 1], dtype=ops.int_dtype),), {}
    if sig_name == "einsum":
        a = ops.arr([[1.0, 2.0], [3.0, 4.0]])
        b = ops.arr([[5.0, 6.0], [7.0, 8.0]])
        return ("ij,jk->ik", a, b), {}
    if sig_name == "conv1d":
        a = ops.arr([[[1.0, 2.0, 3.0, 4.0]]])
        b = ops.arr([[[1.0, -1.0]]])
        return (a, b), {}
    if sig_name == "segment":
        a = ops.arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return (a, [1, 2]), {"axis": 0}
    if sig_name == "diag":
        return (ops.arr([1.0, 2.0, 3.0]),), {}
    if sig_name == "comparison":
        a = ops.arr([1.0, 2.0])
        b = ops.arr([1.0, 2.0 + 1e-15])
        return (a, b), {}
    if sig_name == "broadcast_shape":
        return ((2, 1), (1, 3)), {}
    if sig_name == "dtype_info":
        return None

    raise KeyError(f"Unknown signature family {sig_name}")


def call_spec_for_op(op_name, ops):
    sig_name = arr.core.OP_SIGNATURES[op_name]
    if sig_name == "dtype_info":
        if op_name == "finfo":
            return (ops.default_dtype,), {}
        if op_name == "iinfo":
            return (ops.int_dtype,), {}
    if op_name == "index_add_inplace":
        a = ops.zeros((3, 4))
        idx = ops.arr([0, 2], dtype=ops.int_dtype)
        src = ops.arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return (a, 1, idx, src), {}
    if op_name == "gamma_inc":
        return (
            _sample_inputs(ops)["xpos"],
            ops.arr([[0.0, 0.25], [1.0, 2.0]]),
        ), {}
    return call_spec_for_signature(sig_name, ops)


def call_op(ops, op_name):
    args, kwargs = call_spec_for_op(op_name, ops)
    return getattr(ops, op_name)(*args, **kwargs)


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        x = x.get()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def is_info_obj(x):
    return hasattr(x, "min") and hasattr(x, "max") and hasattr(x, "bits")


def assert_smoke_output(op_name, out, ops):
    if is_info_obj(out):
        assert out.bits > 0, op_name
        return
    if isinstance(out, ops.ArrayType):
        assert out.numel() > 0 if hasattr(out, "numel") else out.size > 0, op_name
        return
    passed = out is not None
    if not passed:
        debug_var = 1
    assert out is not None, op_name


def assert_same_output(op_name, ref_out, test_out):
    if is_info_obj(ref_out) and is_info_obj(test_out):
        for attr in ("bits", "min", "max"):
            assert getattr(ref_out, attr) == getattr(test_out, attr), op_name
        if hasattr(ref_out, "eps") and hasattr(test_out, "eps"):
            assert np.isclose(
                ref_out.eps, test_out.eps, rtol=32 * EPS, atol=32 * EPS
            ), op_name
        if hasattr(ref_out, "tiny") and hasattr(test_out, "tiny"):
            assert np.isclose(
                ref_out.tiny, test_out.tiny, rtol=32 * EPS, atol=32 * EPS
            ), op_name
        return

    ref_np = to_numpy(ref_out)
    test_np = to_numpy(test_out)
    assert ref_np.shape == test_np.shape, op_name

    if ref_np.dtype.kind in "iu?":
        assert np.array_equal(ref_np, test_np), op_name
        return

    assert np.allclose(
        ref_np,
        test_np,
        rtol=32 * EPS,
        atol=32 * EPS,
        equal_nan=True,
    ), op_name
