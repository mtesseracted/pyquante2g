import pytest
import sys
import numpy as np
from pyquante22.ref_ints.utils import GTO, amax
from pyquante22.ref_ints.eri import eri_funcs
from math import isclose as py_isclose

EPS = sys.float_info.epsilon # 2.2204-16, 2**-52
def isclose(a, b, rel_tol=16*EPS, abs_tol=0.0):
    return py_isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

@pytest.fixture(params=list(range(amax+1)))
def am_case(request):
    return request.param


def test_eri_s():
    eri_fac = 2 * np.pi**2.5
    a1 = 1.0
    a2 = 2.0 ** -(6/5) # gives 1*eri_fac for (ss|ss)
    ra = np.asarray([0.,]*3)
    s0 = np.asarray((0,)*3)
    gto1 = GTO(l=s0, r=ra, a=a1)
    gto2 = GTO(l=s0, r=ra, a=a2)

    for eri in eri_funcs:
        result = eri(*[gto1,]*4).item()
        assert isclose(result, eri_fac/8,)
        result = eri(*[gto2,]*4).item()
        assert isclose(result, eri_fac,)


def test_eri_even_odd():
    # test evens and odds cancel out at same center
    ra0 = np.asarray([1.0,]*3)
    am000 = np.asarray([0,0,0])
    g0 = GTO(l=am000, r=ra0, a=1.0)
    am_odds = [
        np.asarray([1,0,0]),
        np.asarray([0,1,0]),
        np.asarray([0,0,1]),
        np.asarray([0,0,9]),
        np.asarray([0,0,amax - (amax%2 == 0)]),
    ]
    am_evens = [
        np.asarray([0,0,0]),
        np.asarray([2,0,0]),
        np.asarray([0,2,0]),
        np.asarray([0,0,2]),
        np.asarray([0,0,8]),
        np.asarray([0,0,amax - amax%2]),
    ]
    for amo in am_odds:
        for ame in am_evens:
            godd = GTO(l=amo, r=ra0, a=1.0)
            geve = GTO(l=ame, r=ra0, a=1.0)
            for eri in eri_funcs:
                result = eri(godd, g0, geve, g0)
                assert result == 0.0
#end def test_eri_even_odd():


@pytest.mark.med
@pytest.mark.arb
def test_eri_vs_arb1(am_case):
    # compare eri funcs vs arb for 1 dim ((am,0,0),s|(am,0,0),s)
    # and test each dimension separately for am<6
    import pyquante22.arb_math as arb
    rtol = 16*EPS                    # 3.5527e-15, 2**-48
    ra = np.asarray([0.,]*3)
    rb = np.asarray([0.125,]*3)
    # rb = np.asarray([0.0,]*3)
    s0 = np.asarray((0,)*3)

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
        # make larger amom have larger exponent to avoid huge 
        # ERI values since these GTOs are unnormalized
        gto2 = GTO(l=am, r=rb, a=0.25*(am_case+1)) 
        ga1, ga2 = gto2arb(gto1), gto2arb(gto2)
        expected = float(
            arb.coulomb_rep4(*ga1, *ga1, *ga2, *ga2,))

        for eri in eri_funcs:
            result = eri(gto1, gto1, gto2, gto2).item()
            diff = result - expected
            perr = diff / result if result != 0.0 else diff
            assert isclose(result, expected, rel_tol=rtol), \
                f"failed for am={am}, eri={eri.__name__}" + \
                f", %err={perr:.4e}, val={result:.4e}"
    #end for idim in range(xyz):
#end def test_eri_vs_arb1():


@pytest.mark.med
@pytest.mark.arb
def test_eri_vs_arb2(am_case):
    # compare eri funcs vs arb for differnt ang.moms. (with
    # find_sum_triplets), different centers, and different 
    # exponents
    import pyquante22.arb_math as arb
    from pyquante22.basis.cart_gto_utils import find_sum_triplets

    def gto2arb(gto):
        l = arb.arr_init(gto.l.tolist())
        r = arb.arr_init(gto.r.tolist())
        a = arb.arb(gto.a)
        return a, l, r

    rtol, atol = 16*EPS, EPS**2 # 3.5527e-15, 2**-48 & 2**-(2*52)
    ra = np.asarray([0.125,]*3)     # 1/8
    rb = np.asarray([0.0625,]*3)    # 1/16
    rc = np.asarray([0.0,0.5,-0.25])
    rd = np.asarray([10.0,-3.5,-1.25])
    rs = [ra, rb, rc, rd]
    aas = [0.125, 1.0, 1.5, 2.0]
    s0 = np.asarray((0,)*3)
    gto0 = GTO(l=s0, r=ra, a=1.0)
    ga0 = gto2arb(gto0)

    trips = find_sum_triplets(am_case) # unique triplets
    for ii, am in enumerate(trips):
        am = np.asarray(am)
        gto1 = GTO(l=am, r=rs[ii%len(rs)], a=aas[ii%len(aas)])
        ga1 = gto2arb(gto1)
        expected = float(arb.coulomb_rep4(*ga1, *ga0, *ga0, *ga0))
        for eri in eri_funcs:
            result = eri(gto1, gto0, gto0, gto0).item()
            diff = result - expected
            perr = diff / result if result != 0.0 else diff
            assert isclose(result, expected, rel_tol=rtol, abs_tol=atol), \
                f"failed for am={am}, eri={eri.__name__}" + \
                f", %err={perr:.4e}, val={result:.4e}"
    #end for ii, am in enumerate(trips):
#end def test_eri_vs_arb2(am_case):


