import numpy as np
import pytest

from pyquante2g.basis.cart_gto import CartGtoPrim as CGP
from tests.torch_test_utils import make_torch_ops


RTOL = 1e-12
ATOL = 1e-14


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        x = x.get()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


@pytest.fixture
def ao():
    return make_torch_ops()


def _make_prim(ao, amoms, cens, alphas):
    return CGP(
        ao.arr(amoms, dtype=ao.int_dtype),
        ao.arr(cens, dtype=ao.default_dtype),
        ao.arr(alphas, dtype=ao.default_dtype),
        ao=ao,
    )


def _split_prim(g):
    ao = g.ao
    half_alpha = 0.5 * g.alpha
    gs = CGP(
        ao.zeros(g.amom.shape, dtype=ao.int_dtype),
        g.cen,
        half_alpha,
        ao=ao,
        co=g.co,
        gm=g.gm,
    )
    gl = CGP(
        g.amom,
        g.cen,
        half_alpha,
        ao=ao,
        co=g.co,
        gm=g.gm,
    )
    return gl, gs


def _diag_last2(x):
    idx = np.arange(x.shape[-1])
    return x[..., idx, idx]


def _diag_first2(x):
    idx = np.arange(x.shape[0])
    return x[idx, idx, ...]


def test_eri3_vs_eri4_split(ao):
    ga = _make_prim(
        ao,
        [[1, 0, 0], [0, 1, 1]],
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]],
        [0.8, 1.1],
    )
    gb = _make_prim(
        ao,
        [[0, 0, 1], [1, 1, 0]],
        [[-0.3, 0.1, 0.0], [0.4, 0.2, -0.2]],
        [1.4, 0.9],
    )
    gc = _make_prim(
        ao,
        [[2, 0, 0], [0, 2, 1]],
        [[0.1, -0.2, 0.5], [-0.4, 0.3, 0.1]],
        [0.7, 1.3],
    )

    direct = _to_numpy((ga * gb).calc_eri3(gc))
    gcl, gcs = _split_prim(gc)
    lifted = _to_numpy((ga * gb).calc_eri(gcl * gcs))
    ref = _diag_last2(lifted)
    assert np.allclose(direct, ref, rtol=RTOL, atol=ATOL)


def test_eri2_vs_eri4_split(ao):
    ga = _make_prim(
        ao,
        [[1, 0, 1], [0, 2, 0]],
        [[0.0, 0.1, -0.1], [0.3, -0.4, 0.2]],
        [0.6, 1.5],
    )
    gc = _make_prim(
        ao,
        [[0, 1, 0], [2, 0, 1]],
        [[-0.2, 0.2, 0.3], [0.5, 0.0, -0.3]],
        [1.2, 0.75],
    )

    direct = _to_numpy(ga.calc_eri2(gc))
    gal, gas = _split_prim(ga)
    gcl, gcs = _split_prim(gc)
    lifted = _to_numpy((gal * gas).calc_eri(gcl * gcs))
    ref = _diag_last2(_diag_first2(lifted))
    assert np.allclose(direct, ref, rtol=RTOL, atol=ATOL)


def test_prim_nai_vs_pair_nai_split(ao):
    ga = _make_prim(
        ao,
        [[1, 1, 0], [0, 0, 2]],
        [[0.0, 0.0, 0.0], [0.25, -0.15, 0.35]],
        [0.9, 1.25],
    )
    rcen = ao.arr(
        [[0.1, -0.2, 0.4], [-0.3, 0.2, 0.1]],
        dtype=ao.default_dtype,
    )

    direct = _to_numpy(ga.calc_nai(rcen))
    gal, gas = _split_prim(ga)
    lifted = _to_numpy((gal * gas).calc_nai(rcen))
    ref = _diag_first2(lifted)
    assert np.allclose(direct, ref, rtol=RTOL, atol=ATOL)


def test_eri3_vs_eri4_split_batch(ao):
    ga = _make_prim(
        ao,
        [[0, 0, 0], [1, 0, 1], [0, 2, 0]],
        [[0.0, 0.1, -0.1], [0.3, -0.2, 0.4], [-0.4, 0.2, 0.1]],
        [0.9, 1.2, 0.7],
    )
    gb = _make_prim(
        ao,
        [[0, 1, 0], [1, 1, 0], [2, 0, 0]],
        [[-0.3, 0.0, 0.2], [0.2, 0.4, -0.1], [0.1, -0.5, 0.3]],
        [1.1, 0.8, 1.4],
    )
    gc = _make_prim(
        ao,
        [[0, 0, 0], [1, 0, 0], [0, 1, 1]],
        [[0.15, -0.25, 0.35]] * 3,
        [0.95] * 3,
    )

    direct = _to_numpy((ga * gb).calc_eri3(gc))
    gcl, gcs = _split_prim(gc)
    lifted = _to_numpy((ga * gb).calc_eri(gcl * gcs))
    ref = _diag_last2(lifted)
    assert np.allclose(direct, ref, rtol=RTOL, atol=ATOL)


def test_eri2_vs_eri4_split_batch(ao):
    ga = _make_prim(
        ao,
        [[0, 0, 0], [1, 1, 0], [0, 0, 2]],
        [[0.1, -0.2, 0.0], [0.3, 0.2, -0.4], [-0.2, 0.4, 0.1]],
        [0.75, 1.05, 1.35],
    )
    gc = _make_prim(
        ao,
        [[0, 0, 0], [2, 0, 0], [0, 1, 1]],
        [[-0.1, 0.25, 0.2]] * 3,
        [1.1] * 3,
    )

    direct = _to_numpy(ga.calc_eri2(gc))
    gal, gas = _split_prim(ga)
    gcl, gcs = _split_prim(gc)
    lifted = _to_numpy((gal * gas).calc_eri(gcl * gcs))
    ref = _diag_last2(_diag_first2(lifted))
    assert np.allclose(direct, ref, rtol=RTOL, atol=ATOL)


def test_prim_nai_vs_pair_nai_split_batch(ao):
    ga = _make_prim(
        ao,
        [[0, 0, 0], [1, 0, 1], [0, 2, 0]],
        [[0.0, 0.0, 0.0], [0.15, -0.2, 0.25], [-0.3, 0.1, 0.4]],
        [0.85, 1.15, 0.95],
    )
    rcen = ao.arr(
        [[0.1, -0.2, 0.4], [-0.3, 0.2, 0.1], [0.25, 0.15, -0.35]],
        dtype=ao.default_dtype,
    )

    direct = _to_numpy(ga.calc_nai(rcen))
    gal, gas = _split_prim(ga)
    lifted = _to_numpy((gal * gas).calc_nai(rcen))
    ref = _diag_first2(lifted)
    assert np.allclose(direct, ref, rtol=RTOL, atol=ATOL)
