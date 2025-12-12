import pytest
import pyquante22.arb_math as arb
import mpmath as mpm

pytestmark = pytest.mark.arb   # applies to all tests in this file

# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


def test_nai_s(digits_case):
    # test for known solution with s-type at center=0
    arb.set_dps(digits_case)
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    g0 = [arb.aone, am000, ra0]
    ans0 = arb.nuclear_attraction(*sum([g0,]*2, []), ra0)
    assert mpm.almosteq(ans0, -arb.pi)
    g1 = [arb.ahalf, am000, ra0]
    ans1 = arb.nuclear_attraction(*sum([g1,]*2, []), ra0)
    assert mpm.almosteq(ans1, -2*arb.pi)


