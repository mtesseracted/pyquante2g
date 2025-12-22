# test if we can combine the int testing funcs into one
import pytest, sys, itertools
from dataclasses import dataclass
import numpy as np
import pyquante22.array_ops as arr
from pyquante22.ref_ints.utils import GTO
from pyquante22.basis.cart_gto import CartGtoPrim as CGP, amax
from pyquante22.basis.cart_gto import CartGtoPrimPair as CGPP
from pyquante22.basis.cart_gto_utils import find_sum_triplets
import pyquante22.arb_math as arb

EPS = sys.float_info.epsilon
backend = "torch"
ao = arr.ArrayOps(
    backend, device="cpu", default_dtype="float64", int_dtype="int64"
)


def am_mark(x, med=3, slo=7):  # mark ang.mom. cases
    if x >= slo:
        return pytest.param(x, marks=(pytest.mark.slo, pytest.mark.med))
    if x >= med:
        return pytest.param(x, marks=pytest.mark.med)
    return x


@pytest.fixture(params=[am_mark(x) for x in range(20 + 1)])
def am_case(request):
    return request.param


def to_np(a):  # to numpy
    if hasattr(a, "shape"):
        return np.asarray(a.tolist())
    return np.asarray(a.item())


def gto2arb(gto):
    l = arb.arr_init(gto.l.tolist())
    r = arb.arr_init(gto.r.tolist())
    a = arb.arb(gto.a.item())
    return [a, l, r]


def ove_ref(*args):
    """Overlap reference, args is one CartGtoPrimPair"""
    ab = args[0]  # one CGPP
    a = GTO(l=to_np(ab.ga.amom), r=to_np(ab.ga.cen), a=to_np(ab.ga.alpha[0]))
    b = GTO(l=to_np(ab.gb.amom), r=to_np(ab.gb.cen), a=to_np(ab.gb.alpha[0]))
    fargs = sum([gto2arb(g) for g in [a, b]], [])
    return arb.overlap_tho66(*fargs)


def kin_ref(*args):
    """Overlap reference, args is one CartGtoPrimPair"""
    ab = args[0]  # one CGPP
    a = GTO(l=to_np(ab.ga.amom), r=to_np(ab.ga.cen), a=to_np(ab.ga.alpha[0]))
    b = GTO(l=to_np(ab.gb.amom), r=to_np(ab.gb.cen), a=to_np(ab.gb.alpha[0]))
    fargs = sum([gto2arb(g) for g in [a, b]], [])
    return arb.kinetic_tho66(*fargs)


def eri_ref(*args):
    """ERI reference, args is two CartGtoPrimPair"""
    ab, cd = args  # two CGPPs
    a = GTO(l=to_np(ab.ga.amom), r=to_np(ab.ga.cen), a=to_np(ab.ga.alpha[0]))
    b = GTO(l=to_np(ab.gb.amom), r=to_np(ab.gb.cen), a=to_np(ab.gb.alpha[0]))
    c = GTO(l=to_np(cd.ga.amom), r=to_np(cd.ga.cen), a=to_np(cd.ga.alpha[0]))
    d = GTO(l=to_np(cd.gb.amom), r=to_np(cd.gb.cen), a=to_np(cd.gb.alpha[0]))
    fargs = sum([gto2arb(g) for g in [a, b, c, d]], [])
    return arb.coulomb_rep4(*fargs)


@dataclass
class GTOTestCase:
    name: str
    arity: int
    arg_iter: callable  # yields tuples of GTOs
    compute: callable  # (args) -> (ans, ref)
    rtol: float
    atol: float


@pytest.fixture(params=["overlap", "kinetic", "eri"])
def gto_test_case(request):
    if request.param == "overlap":
        return GTOTestCase(
            name="overlap",
            arity=2,
            # arg_iter=lambda gtos: itertools.product(gtos, repeat=2),
            arg_iter=lambda gs: ((g, gs[0]) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_overlap().item(),
                float(ove_ref(args[0] * args[1])),
            ),
            rtol=16 * EPS,
            atol=EPS / 64,
        )

    if request.param == "kinetic":
        return GTOTestCase(
            name="kinetic",
            arity=2,
            arg_iter=lambda gs: ((g, gs[0]) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_kinetic().item(),
                float(kin_ref(args[0] * args[1])),
            ),
            rtol=32 * EPS,
            atol=EPS / 64,
        )

    if request.param == "eri":
        return GTOTestCase(
            name="eri",
            arity=4,
            # arg_iter=lambda gtos: itertools.islice(
            #     itertools.product(gtos, repeat=4), 1000),
            arg_iter=lambda gs: ((g, *[gs[0]] * 3) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_eri(args[2] * args[3]).item(),
                float(eri_ref(args[0] * args[1], args[2] * args[3])),
            ),
            rtol=64 * EPS,  # not ideal
            atol=EPS / 64,
        )


@pytest.fixture
def gtos_setup(am_case):
    ra = ao.arr([0.125] * 3)
    rb = ao.arr([0.0625] * 3)
    rc = ao.arr([0.0, 0.5, -0.25])
    rd = ao.arr([10.0, -3.5, -1.25])
    rs = [ra, rb, rc, rd]
    # aas = [0.125, 0.25, 0.5, 1.0,]# 1.5, 2.0]
    aas = [1.0, 1.5, 2.0]
    aas = [ao.arr([v] * 3) for v in aas]
    aone = ao.arr([1] * 3)
    am_s = ao.arr([0] * 3, dtype=ao.int_dtype)
    ret = [
        CGP(ao.arr(am, dtype=ao.int_dtype), rs[ii % len(rs)], aas[ii % len(aas)])
        for ii, am in enumerate(find_sum_triplets(am_case))
    ]
    return [CGP(am_s, ra, aone)] + ret


def test_gto_int_vs_ref(gtos_setup, gto_test_case):
    case = gto_test_case
    for args in case.arg_iter(gtos_setup):
        try:
            ans, gt = case.compute(args)
        except:
            debug_var = 1
        diff = ans - gt
        perr = diff / gt if gt != 0.0 else diff
        passed = np.isclose(ans, gt, rtol=case.rtol, atol=case.atol)
        if not passed:
            debug_var = 1
        assert passed, (
            f"[{case.name}] failed for args={args}, "
            f"%err={perr:.4e}, gt={gt:.4e}, ans={ans:.4e}"
        )
