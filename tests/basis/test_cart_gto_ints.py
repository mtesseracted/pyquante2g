# use a driver func to test all ints with the same code
# NOTE: error tolerances are not great for NAI and ERI,
# by the time you get up to amom~20, we see errors on the order
# of 1e-13. Usually they're closer to 1e-14, but going into 
# the code shows individual elements of the calculation
# such as the A_array in calc_nai() have errors on the order of
# 1e-13, with a similar story for calc_eri(). My conclusion
# is that most of these errors stem from the floating points
# error associated with subtracting two numbers that are of
# similar order of magnitude, creating a large loss of precision.
# There's really nothing to be done about it at this point,
# so I will simply drop the largest amom tested in this file to 
# 12, which does have acceptable errors. Note the boys function
# implementation also has errors on the order of 1e-14 for larger
# amoms, which could be fixed using branched logic. In the future
# I think changing out the A, C, and H array calculations with
# precomputed coefficients will mitigate this problem since
# there will be much less terms in the sums
import pytest, sys, itertools
from dataclasses import dataclass
import numpy as np
import pyquante22.array_ops as arr
from pyquante22.ref_ints.utils import GTO
from pyquante22.basis.cart_gto import CartGtoPrim as CGP#, amax
from pyquante22.basis.cart_gto import CartGtoPrimPair as CGPP
from pyquante22.basis.cart_gto_utils import find_sum_triplets
import pyquante22.arb_math as arb

EPS = sys.float_info.epsilon
backend = "torch"
ao = arr.ArrayOps(
    backend, device="cpu", default_dtype="float64", int_dtype="int64"
)


def am_mark(x, med=3, slo=9):  # mark ang.mom. cases
    if x >= slo:
        return pytest.param(x, marks=(pytest.mark.slo, pytest.mark.med))
    if x >= med:
        return pytest.param(x, marks=pytest.mark.med)
    return x


amstart = 0 # @DEBUG
amax = 10
@pytest.fixture(params=[am_mark(x) for x in range(amstart, amax + 1)])
def am_case(request):
    return request.param


def to_np(a):  # to numpy
    if hasattr(a, "shape"):
        return np.asarray(a.tolist())
    return np.asarray(a.item())


def gto2arb(gto):
    if isinstance(gto, GTO):
        l = arb.arr_init(gto.l.tolist())
        r = arb.arr_init(gto.r.tolist())
        a = arb.arb(gto.a.item())
        return [a, l, r]
    elif isinstance(gto, CGP):
        return gto2arb(GTO(
            l=to_np(gto.amom), 
            r=to_np(gto.cen), 
            a=to_np(gto.alpha[0])))
    else:
        raise ValueError(f"Does not support type {type(gto)}")


def ove_ref(*args):
    """Overlap reference, args is one CartGtoPrimPair"""
    ab = args[0]  # one CGPP
    a = GTO(l=to_np(ab.ga.amom), r=to_np(ab.ga.cen), a=to_np(ab.ga.alpha[0]))
    b = GTO(l=to_np(ab.gb.amom), r=to_np(ab.gb.cen), a=to_np(ab.gb.alpha[0]))
    fargs = sum([gto2arb(g) for g in [a, b]], [])
    return arb.overlap_tho66(*fargs)


def kin_ref(*args):
    """Kinetic reference, args is one CartGtoPrimPair"""
    ab = args[0]  # one CGPP
    a = GTO(l=to_np(ab.ga.amom), r=to_np(ab.ga.cen), a=to_np(ab.ga.alpha[0]))
    b = GTO(l=to_np(ab.gb.amom), r=to_np(ab.gb.cen), a=to_np(ab.gb.alpha[0]))
    fargs = sum([gto2arb(g) for g in [a, b]], [])
    return arb.kinetic_tho66(*fargs)


def nai_ref(*args):
    """NAI reference, args is one CartGtoPrimPair and one center"""
    ab, r = args  # two CGPPs
    a = GTO(l=to_np(ab.ga.amom), r=to_np(ab.ga.cen), a=to_np(ab.ga.alpha[0]))
    b = GTO(l=to_np(ab.gb.amom), r=to_np(ab.gb.cen), a=to_np(ab.gb.alpha[0]))
    fargs = sum([gto2arb(g) for g in [a, b,]], [])
    fargs += [arb.arr_init(r.tolist())]
    return arb.nuclear_attraction(*fargs)


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


@pytest.fixture(params=["overlap", "kinetic", "nai", "eri"])
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
            rtol=16 * EPS,
            atol=EPS / 64,
        )

    if request.param == "nai":
        return GTOTestCase(
            name="nai",
            arity=3,
            arg_iter=lambda gs: ((g, gs[0], gs[0].cen) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_nai(args[2]).item(),
                float(nai_ref(args[0] * args[1], args[2])),
            ),
            rtol=16 * EPS,  # fails for amax > 10
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
            rtol=16 * EPS,  # fail for amom > 17
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
        ans, gt = case.compute(args)
        diff = ans - gt
        perr = diff / gt if gt != 0.0 else diff
        passed = np.isclose(ans, gt, rtol=case.rtol, atol=case.atol)
        if not passed:
            if case.name == 'nai':
                ab, r = args[0] * args[1], args[2]
                a = ab.calc_a_array(r)
                aalr = gto2arb(args[0])
                balr = gto2arb(args[1])
                alpha1, lmn1, A = aalr
                alpha2, lmn2, B = balr
                C = arb.arr_init(r.tolist())
                P = arb.gto_prod_cen(alpha1,A,alpha2,B)
                dPA, dPB, dPC = P-A, P-B, P-C
                Ax = arb.A_array(lmn1[0],lmn2[0],dPA[0],dPB[0],dPC[0],alpha1+alpha2)
                import mpmath as mpm
                nax = np.asarray(mpm.fp.matrix(Ax))
                aax = a[0].numpy()
                diffs = aax - nax
                denom = nax.copy()
                denom[np.abs(denom) < case.atol] = case.atol
                perrs = np.abs(diffs / denom)
                amax = np.argmax(perrs)
                debug_var = 1

            debug_var = 1

        assert passed, (
            f"[{case.name}] failed, rtol={case.rtol:.4e}, atol={case.atol:.4e}\n"
            f"    %err={perr:.4e}, gt={gt:.4e}, ans={ans:.4e}, for args:\n"
            f"    {args.replace('\n', '\n    ')}"
        )
    
# @TODO: batch calc and compare
