import pytest
import sys
import numpy as np
from pyquante22.ref_ints.utils import GTO, amax
from pyquante22.ref_ints.overlap import overlap_funcs
from math import isclose as py_isclose

EPS = sys.float_info.epsilon # 2.2204-16, 2**-52
def isclose(a, b, rel_tol=16*EPS, abs_tol=0.0):
    return py_isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

@pytest.fixture(params=list(range(amax+1)))
def am_case(request):
    return request.param


def GTOs2aarrll(ga, gb):
    """ helper for converting from GTO to 
    the arg func signature in ref_ints
    @TODO: change to use GTO to be consisten with eri.py
    """
    vals = [ga.arl, gb.arl]
    return sum([list(row) for row in zip(*vals)], [])


def test_overlap_s():
    sfac = np.pi**1.5
    a1 = 0.5
    ra = np.asarray([0.,]*3)
    s0 = np.asarray((0,)*3)
    gto1 = GTO(l=s0, r=ra, a=a1)
    for ov in overlap_funcs:
        result = ov(*GTOs2aarrll(gto1, gto1))
        assert isclose(result, sfac)


def test_overlap_even_odd():

    ra0 = np.asarray([1.0,]*3)
    am000 = np.asarray([0,0,0])
    g0 = GTO(l=am000, r=ra0, a=1.0)
    am_odds = [
        np.asarray([1,0,0]),
        np.asarray([0,1,0]),
        np.asarray([0,0,1]),
        np.asarray([0,0,9]),
        # np.asarray([0,0,amax - (amax%2 == 0)]),
    ]
    am_evens = [
        np.asarray([0,0,0]),
        np.asarray([2,0,0]),
        np.asarray([0,2,0]),
        np.asarray([0,0,2]),
        np.asarray([0,0,8]),
        # np.asarray([0,0,amax - amax%2]),
    ]
    for amo in am_odds:
        for ame in am_evens:
            godd = GTO(l=amo, r=ra0, a=1.0)
            geve = GTO(l=ame, r=ra0, a=1.0)
            for ov in overlap_funcs:
                result = ov(*GTOs2aarrll(godd, geve))
                assert result == 0.0
#end def test_overlap_even_odd():




if False:
    import pyquante22.arb_math as arb
    def gto2arb(gto):
        l = arb.arr_init(gto.l.tolist())
        r = arb.arr_init(gto.r.tolist())
        a = arb.arb(gto.a)
        return a, l, r



