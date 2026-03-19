import pytest
import sys
from pyquante2g.ref_ints.nai import nai_funcs
from pyquante2g.ref_ints.utils import GTO, amax
import numpy as np
from math import isclose as py_isclose

EPS = sys.float_info.epsilon # 2.2204-16, 2**-52
def isclose(a, b, rel_tol=16*EPS, abs_tol=0.0):
    return py_isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

@pytest.fixture(params=nai_funcs)
def nai(request):
    return request.param

# @pytest.fixture(params=list(range(amax+1)))
@pytest.fixture(params=list(range(8+1)))
def am_case(request):
    return request.param


def test_nai_s0(nai):
    # test for known solution with s-type at center=0
    ra0 = np.asarray([0,]*3)
    am000 = np.asarray([0,0,0])
    g0 = GTO(a=1, l=am000, r=ra0)
    g1 = GTO(a=0.5, l=am000, r=ra0)
    ans0 = nai(g0, g0, ra0).item()
    assert np.isclose(ans0, -np.pi), f"fail at {nai.__name__}"
    ans1 = nai(g1, g1, ra0).item()
    assert np.isclose(ans1, -2*np.pi), f"fail at {nai.__name__}"


@pytest.mark.med
@pytest.mark.arb
def test_nai_vs_arb1(nai, am_case):
    # compare nai func vs arb at r=0
    import pyquante2g.arb_math as arb
    rtol = 16*EPS                    # 3.5527e-15, 2**-48
    ra = np.asarray([0,]*3)
    s0 = np.asarray([0,0,0])

    def gto2arb(gto):
        l = arb.arr_init(gto.l.tolist())
        r = arb.arr_init(gto.r.tolist())
        a = arb.arb(gto.a)
        return a, l, r

    xyz = 3 if am_case < 6 else 1
    for idim in range(xyz):
        am = np.asarray((0, 0, 0))
        am[idim] = am_case
        gto1 = GTO(l=s0, r=ra, a=1.0)
        gto2 = GTO(l=am, r=ra, a=0.25*(am_case+1)) 
        ga1, ga2 = gto2arb(gto1), gto2arb(gto2)
        expected = float(arb.nuclear_attraction(
            *ga1, *ga2, arb.arr_init(ra.tolist())))
        result = nai(gto1, gto2, ra).item()
        diff = result - expected
        perr = diff / result if result != 0.0 else diff
        assert isclose(result, expected, rel_tol=rtol), \
            f"failed for am={am}, nai={nai.__name__}" + \
            f", %err={perr:.4e}, val={result:.4e}"
    #end for idim in range(xyz):
#end def test_nai_vs_arb1(nai, am_case):


@pytest.mark.med
@pytest.mark.arb
def test_nai_vs_arb2(nai, am_case):
    # compare nai funcs vs arb for differnt ang.moms. (with
    # find_sum_triplets), different centers, and different 
    # exponents
    import pyquante2g.arb_math as arb
    from pyquante2g.basis.cart_gto_utils import find_sum_triplets

    def gto2arb(gto):
        l = arb.arr_init(gto.l.tolist())
        r = arb.arr_init(gto.r.tolist())
        a = arb.arb(gto.a)
        return a, l, r

    rtol, atol = 64*EPS, EPS**2 # 2**-46 & 2**-(2*52)
    ra = np.asarray([0.125,]*3)     # 1/8
    rb = np.asarray([0.0625,]*3)    # 1/16
    rc = np.asarray([0.0,0.5,-0.25])
    rd = np.asarray([10.0,-3.5,-1.25])
    rs = [ra, rb, rc, rd]
    aas = [0.125, 0.25, 0.5, 1.0]
    s0 = np.asarray((0,)*3)
    gto0 = GTO(l=s0, r=ra, a=1.0)
    ga0 = gto2arb(gto0)

    trips = find_sum_triplets(am_case) # unique triplets
    for ii, am in enumerate(trips):
        am = np.asarray(am)
        gto1 = GTO(l=am, r=rs[ii%len(rs)], a=aas[ii%len(aas)])
        rn = rs[(ii+1)%len(rs)]
        ga1 = gto2arb(gto1)
        expected = float(arb.nuclear_attraction(
            *ga1, *ga0, arb.arr_init(rn.tolist())))
        result = nai(gto1, gto0, rn).item()
        diff = result - expected
        perr = diff / result if result != 0.0 else diff
        assert isclose(result, expected, rel_tol=rtol, abs_tol=atol), \
            f"failed for am={am}, nai={nai.__name__}" + \
            f", %err={perr:.4e}, val={result:.4e}"
    #end for ii, am in enumerate(trips):
#end def test_eri_vs_arb2(am_case):


