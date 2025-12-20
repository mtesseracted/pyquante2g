import pytest, sys
import numpy as np
from pyquante22.basis.cart_gto import CartGtoPrim as CGP, amax
from pyquante22.basis.cart_gto import CartGtoPrimPair as CGPP
import pyquante22.array_ops as arr


EPS = sys.float_info.epsilon
backend = "torch"
ao = arr.ArrayOps(backend, device="cpu", 
                default_dtype="float64", int_dtype="int64")


@pytest.fixture(params=list(range(amax+1)))
def am_case(request):
    return request.param


def tonp(a): # to numpy
    if hasattr(a, "shape"):
        return np.asarray(a.tolist())
    return np.asarray(a.item())

def eri_ref(*args, method="arb_tho66"):
    from pyquante22.ref_ints.utils import GTO
    ab, cd = args # two CGPPs
    a = GTO(l=tonp(ab.ga.amom), r=tonp(ab.ga.cen), a=tonp(ab.ga.alpha[0]))
    b = GTO(l=tonp(ab.gb.amom), r=tonp(ab.gb.cen), a=tonp(ab.gb.alpha[0]))
    ab = cd
    c = GTO(l=tonp(ab.ga.amom), r=tonp(ab.ga.cen), a=tonp(ab.ga.alpha[0]))
    d = GTO(l=tonp(ab.gb.amom), r=tonp(ab.gb.cen), a=tonp(ab.gb.alpha[0]))

    if method == "ref_tho66":
        from pyquante22.ref_ints.eri import eri_tho66 as f
        # def eri_tho66(ga:GTO, gb:GTO, gc:GTO, gd:GTO) -> float:
        fargs = a, b, c, d

    elif method == "arb_tho66":
        import pyquante22.arb_math as arb
        from pyquante22.arb_math import coulomb_rep4 as f
        def gto2arb(gto):
                l = arb.arr_init(gto.l.tolist())
                r = arb.arr_init(gto.r.tolist())
                a = arb.arb(gto.a.item())
                return [a, l, r]
        fargs = sum([gto2arb(g) for g in [a,b,c,d]], [])

    else:
        raise NotImplementedError("only tho66 supported currently")

    return f(*fargs)
    

def ove_ref(*args, method="tho66"):
    if method == "tho66":
        pass
    

def test_eri_s():
    rtol, atol = 16*EPS, EPS/16
    eri_fac = 2 * np.pi**2.5
    aa = ao.arr([1.0,]*3)
    ab = ao.arr([2.0 ** -(6/5),]*3) # gives 1*eri_fac for (ss|ss)
    ra = ao.arr([0.,]*3)
    am0 = ao.arr([0,0,0], dtype=ao.int_dtype)
    gsa = CGP(am0, ra, aa)
    gsb = CGP(am0, ra, ab)
    gpa = CGPP(gsa, gsa) # or syntactic sugar: gsa*gsa
    gpb = CGPP(gsb, gsb)

    gta = eri_fac / 8 #eri_ref(gpa, gpa).item()
    ansa = gpa.calc_eri(gpa).item()
    assert np.isclose(ansa, gta, rtol=rtol, atol=0), f"err @ a=1"
    gtb = eri_fac # eri_ref(gpb, gpb).item()
    ansb = gpb.calc_eri(gpb)
    assert np.isclose(ansb, gtb, rtol=rtol, atol=0), f"err @ a=2^(-6/5)"
#end def test_eri_s():

@pytest.mark.med
def test_eri_vs_ref2(am_case):
    # compare eri funcs vs arb for differnt ang.moms. (with
    # find_sum_triplets), different centers, and different 
    # exponents
    from pyquante22.basis.cart_gto_utils import find_sum_triplets

    rtol, atol = 16*EPS, EPS**2 # 3.5527e-15, 2**-48 & 2**-(2*52)
    ra = ao.arr([0.125,]*3)     # 1/8
    rb = ao.arr([0.0625,]*3)    # 1/16
    rc = ao.arr([0.0,0.5,-0.25])
    rd = ao.arr([10.0,-3.5,-1.25])
    rs = [ra, rb, rc, rd]
    aas = [0.125, 1.0, 1.5, 2.0]
    aas = [ao.arr([v,]*3) for v in aas]
    s0 = ao.arr((0,)*3, dtype=ao.int_dtype)
    gto0 = CGP(s0, ra, aas[1])

    trips = find_sum_triplets(am_case) # unique triplets
    for ii, am in enumerate(trips):
        am = ao.arr(am, dtype=ao.int_dtype)
        gto1 = CGP(am, rs[ii%len(rs)], aas[ii%len(aas)])
        gpa, gpb = gto1*gto0, gto0*gto0
        gt = float(eri_ref(gpa, gpb))
        ans = gpa.calc_eri(gpb).item()
        diff = ans - gt
        perr = diff / gt if gt != 0.0 else diff
        passed = np.isclose(ans, gt, rtol=rtol, atol=atol)
        if not passed:
            debug_var = 1
        assert passed, \
            f"failed for am={am}" + \
            f", %err={perr:.4e}, gt={gt:.4e}, ans={ans:.4e}"
    #end for ii, am in enumerate(trips):
#end def test_eri_vs_arb2(am_case):


@pytest.mark.med
def test_batch_eri_vs_ref2():
    # compare eri funcs vs arb for differnt ang.moms. (with
    # find_sum_triplets), different centers, and different 
    # exponents using batch calc for cart_gto.calc_eri
    from pyquante22.basis.cart_gto_utils import find_sum_triplets

    rtol, atol = 16*EPS, EPS**2 # 3.5527e-15, 2**-48 & 2**-(2*52)
    ra = ao.arr([0.125,]*3)     # 1/8
    rb = ao.arr([0.0625,]*3)    # 1/16
    rc = ao.arr([0.0,0.5,-0.25])
    rd = ao.arr([10.0,-3.5,-1.25])
    rs = [ra, rb, rc, rd]
    aas = [0.125, 1.0, 1.5, 2.0]
    aas = [ao.arr([v,]*3) for v in aas]
    s0 = ao.arr((0,)*3, dtype=ao.int_dtype)
    gto0 = CGP(s0, ra, aas[1])
    gto1s_list = []

    # create the CartGtoPrim instances we want to batch
    for am_case in range(amax):
        trips = find_sum_triplets(am_case) # unique triplets
        for ii, am in enumerate(trips):
            am = ao.arr(am, dtype=ao.int_dtype)
            gto1s_list.append(CGP(am, rs[ii%len(rs)], aas[ii%len(aas)]))

    # create the batch from the list
    gto0s = sum([gto0 for _ in range(len(gto1s_list))])
    gto1s = sum(gto1s_list)
    gpas = gto0s * gto0s
    gpbs = gto1s * gto0s
    anss = gpbs.calc_eri(gpas) # run batch calculation

    # compare batch calculation one-by-one with ref calc
    bcount = len(gto0s) // gto0s.nd
    assert bcount == len(anss)
    for ii in range(bcount):
        gto1 = gto1s_list[ii]
        gpa, gpb = gto1*gto0, gto0*gto0
        gt = float(eri_ref(gpa, gpb))
        ans = anss[ii] #gpa.calc_eri(gpb).item()
        diff = ans - gt
        perr = diff / gt if gt != 0.0 else diff
        passed = np.isclose(ans, gt, rtol=rtol, atol=atol)
        if not passed:
            debug_var = 1
        assert passed, \
            f"failed for am={am}" + \
            f", %err={perr:.4e}, gt={gt:.4e}, ans={ans:.4e}"
    #end for ii, am in enumerate(trips):
#end def test_eri_vs_arb2(am_case):


