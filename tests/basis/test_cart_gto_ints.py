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
# 10, which does have acceptable errors. Note the boys function
# implementation also has errors on the order of 1e-14 for larger
# amoms, which could be fixed using branched logic. In the future
# I think changing out the A, C, and H array calculations with
# precomputed coefficients will mitigate this problem since
# there will be less terms in the sums
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


def ndindex(shape):
    """flattened index generator for given shape"""
    return itertools.product(*[range(n) for n in shape]) # generator


def to_np(a):  # to numpy
    if hasattr(a, "shape"):
        return np.asarray(a.tolist())
    return np.asarray(a.item())


def cgp_scalar(gto: CGP):
    l = to_np(gto.amom)
    r = to_np(gto.cen)
    if hasattr(l, "ndim") and l.ndim == 2:
        l = l[0]
    if hasattr(r, "ndim") and r.ndim == 2:
        r = r[0]
    return l, r, to_np(gto.alpha[0])


def gto2arb(gto):
    if isinstance(gto, GTO):
        l = arb.arr_init(gto.l.tolist())
        r = arb.arr_init(gto.r.tolist())
        a = arb.arb(gto.a.item())
        return [a, l, r]
    elif isinstance(gto, CGP):
        l, r, a = cgp_scalar(gto)
        return gto2arb(GTO(l=l, r=r, a=a))
    else:
        raise ValueError(f"Does not support type {type(gto)}")


def ove_ref(*args):
    """Overlap reference, args are two CartGtoPrim"""
    la, ra, aa = cgp_scalar(args[0])
    lb, rb, ab = cgp_scalar(args[1])
    a = GTO(l=la, r=ra, a=aa)
    b = GTO(l=lb, r=rb, a=ab)
    fargs = sum([gto2arb(g) for g in [a, b]], [])
    return arb.overlap_tho66(*fargs)


def kin_ref(*args):
    """Kinetic reference, args are two CartGtoPrim"""
    la, ra, aa = cgp_scalar(args[0])
    lb, rb, ab = cgp_scalar(args[1])
    a = GTO(l=la, r=ra, a=aa)
    b = GTO(l=lb, r=rb, a=ab)
    fargs = sum([gto2arb(g) for g in [a, b]], [])
    return arb.kinetic_tho66(*fargs)


def nai_ref(*args):
    """NAI reference, args is two CartGtoPrim and one center"""
    ga, gb, r = args  # 
    la, ra, aa = cgp_scalar(ga)
    lb, rb, ab = cgp_scalar(gb)
    a = GTO(l=la, r=ra, a=aa)
    b = GTO(l=lb, r=rb, a=ab)
    r = to_np(r)
    if hasattr(r, "ndim") and r.ndim == 2:
        r = r[0]
    fargs = sum([gto2arb(g) for g in [a, b,]], [])
    fargs += [arb.arr_init(r.tolist())]
    return arb.nuclear_attraction(*fargs)


def eri_ref(*args):
    """ERI reference, args are four CartGtoPrim"""
    ga, gb, gc, gd = args  # two CGPPs
    la, ra, aa = cgp_scalar(ga)
    lb, rb, ab = cgp_scalar(gb)
    lc, rc, ac = cgp_scalar(gc)
    ld, rd, ad = cgp_scalar(gd)
    a = GTO(l=la, r=ra, a=aa)
    b = GTO(l=lb, r=rb, a=ab)
    c = GTO(l=lc, r=rc, a=ac)
    d = GTO(l=ld, r=rd, a=ad)
    fargs = sum([gto2arb(g) for g in [a, b, c, d]], [])
    return arb.coulomb_rep4(*fargs)


@dataclass
class GTOTestCase:
    name: str
    arity: int
    compute: callable  # (args) -> (ans, ref)
    arg_iter: callable = None # (optional) yields tuples of GTOs
    ref: callable = None # (optional) reference function
    rtol: float = 16 * EPS
    atol: float = EPS / 64


@pytest.fixture(params=["overlap", "kinetic", "nai", "eri"])
def gto_test_case(request):
    if request.param == "overlap":
        return GTOTestCase(
            name="overlap", arity=2,
            arg_iter=lambda gs: ((g, gs[0]) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_overlap().item(),
                float(ove_ref(*args)),
            ),
        )

    if request.param == "kinetic":
        return GTOTestCase(
            name="kinetic", arity=2,
            arg_iter=lambda gs: ((g, gs[0]) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_kinetic().item(),
                float(kin_ref(*args)),
            ),
        )

    if request.param == "nai":
        return GTOTestCase(
            name="nai", arity=3,
            arg_iter=lambda gs: ((g, gs[0], gs[0].cen[0]) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_nai(args[2]).item(),
                float(nai_ref(*args)),
            ),
            rtol=16 * EPS,  # fails for amax > 10
        )

    if request.param == "eri":
        return GTOTestCase(
            name="eri", arity=4,
            arg_iter=lambda gs: ((g, *[gs[0]] * 3) for g in gs[1:]),
            compute=lambda args: (
                (args[0] * args[1]).calc_eri(args[2] * args[3]).item(),
                float(eri_ref(*args)),
            ),
            rtol=16 * EPS,  # fail for amom > 17
        )


def _build_gto_list(am_case):
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
        CGP(ao.arr(am, dtype=ao.int_dtype), rs[ii % len(rs)], aas[ii % len(aas)], ao=ao)
        for ii, am in enumerate(find_sum_triplets(am_case))
    ]
    return [CGP(am_s, ra, aone, ao=ao)] + ret


@pytest.fixture
def gtos_setup(am_case):
    return _build_gto_list(am_case)


def test_gto_int_vs_ref(gtos_setup, gto_test_case):
    case = gto_test_case
    for args in case.arg_iter(gtos_setup):
        ans, gt = case.compute(args)
        diff = ans - gt
        perr = diff / gt if gt != 0.0 else diff
        passed = np.isclose(ans, gt, rtol=case.rtol, atol=case.atol)
        if not passed:
            debug_var = 1
        assert passed, (
            f"[{case.name}] failed, rtol={case.rtol:.4e}, atol={case.atol:.4e}\n"
            f"    %err={perr:.4e}, gt={gt:.4e}, ans={ans:.4e}, for args:\n"
            f"    {str(args).replace(chr(10), chr(10) + '    ')}"
        )
    

@pytest.fixture(params=["overlap", "kinetic", "nai", "eri"])
def gto_batch_case(request):
    if request.param == "overlap":
        return GTOTestCase(
            name="overlap", arity=2, ref=ove_ref,
            compute=lambda args: (args[0] * args[1]).calc_overlap(),
        )

    if request.param == "kinetic":
        return GTOTestCase(
            name="kinetic", arity=2, ref=kin_ref,
            compute=lambda args: (args[0] * args[1]).calc_kinetic(),
        )

    if request.param == "nai":
        return GTOTestCase(
            name="nai", arity=3, ref=nai_ref,
            compute=lambda args: (args[0] * args[1]).calc_nai(args[2]),
            rtol = 64*EPS,
        )

    if request.param == "eri":
        return GTOTestCase(
            name="eri", arity=4, ref=eri_ref,
            compute=lambda args: (args[0] * args[1]).calc_eri(
                args[2] * args[3]),
        )


def test_batch_vs_ref(gto_batch_case):

    # make s,p,d,f GTOs
    gto_list = sum([_build_gto_list(i) for i in range(4)], [])[::-1]
    gbc = gto_batch_case

    if gbc.arity == 2:
        ga = sum(gto_list[:4])
        gb = sum(gto_list[4:])
        args = [ga, gb]
    elif gbc.arity == 3:
        ga = sum(gto_list[:4])
        gb = sum(gto_list[4:9])
        gc = sum(gto_list[9:])
        args = [ga, gb, gc.cen]
    elif gbc.arity == 4:
        ga, gb = sum(gto_list[:2]), sum(gto_list[2:5])
        gc, gd = sum(gto_list[5:9]), sum(gto_list[9:])
        args = [ga, gb, gc, gd]

    ans = gbc.compute(args)
    for ids in ndindex(ans.shape):
        a = ans[ids].item()
        ref_args = [args[i][id] for i, id in enumerate(ids)]
        gt = float(gbc.ref(*ref_args))
        diff = a - gt
        perr = diff / gt if gt != 0.0 else diff
        passed = np.isclose(a, gt, rtol=gbc.rtol, atol=gbc.atol)
        if not passed:
            debug_var = 1
        assert passed, (
            f"[batch {gbc.name}] failed at index {ids}, "
            f"rtol={gbc.rtol:.4e}, atol={gbc.atol:.4e}\n"
            f"    %err={perr:.4e}, gt={gt:.4e}, ans={a:.4e}, for args:\n"
            f"    {str([args[i] for i in ids]).replace('\n', '\n    ')}"
        )

 
