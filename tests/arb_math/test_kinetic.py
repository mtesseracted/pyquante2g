import pytest
import pyquante22.arb_math as arb
import mpmath as mpm

pytestmark = pytest.mark.arb   # applies to all tests in this file

# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


def test_kin_s(digits_case):
    arb.set_dps(digits_case)
    sfac = arb.pi**1.5
    a1 = arb.arb(0.5)
    ra = arb.arr_init([0.,]*3)
    s0 = arb.arr_init((0,)*3)
    g0 = [a1, s0, ra]
    ans0 = arb.kinetic_tho66(*g0, *g0) # alr
    expected = arb.arb(3/4) * sfac
#end def test_overlap_even_odd
    assert mpm.almosteq(ans0, expected), \
        f"GT: {float(expected):.4f} Ans: {float(ans0):.4f} Diff:{float(ans0-sfac):.4e}"


def test_kinetic_even_odd(digits_case):
    # test evens and odds cancel out at center=0
    arb.set_dps(digits_case)
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    g0 = [arb.aone, am000, ra0]
    am_odds = [
        arb.arr_init([1,0,0]),
        arb.arr_init([0,1,0]),
        arb.arr_init([0,0,1]),
        arb.arr_init([0,0,9]),
    ]
    am_evens = [
        arb.arr_init([0,0,0]),
        arb.arr_init([2,0,0]),
        arb.arr_init([0,2,0]),
        arb.arr_init([0,0,2]),
        arb.arr_init([0,0,8]),
    ]
    for amo in am_odds:
        for ame in am_evens:
            godd = [arb.aone, amo, ra0]
            geve = [arb.aone, ame, ra0]
            ans = arb.kinetic_tho66(*godd, *geve,)
            assert ans == arb.azero
#end def test_kinetic_even_odd
