import pytest
import pyquante22.arb_math as arb
import mpmath as mpm

# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


def test_norm2(digits_case):
    arb.set_dps(digits_case)
    x = arb.arr_init(list(range(4)))
    assert arb.norm2(x) == arb.arb(14)
    x = arb.arr_init([arb.pi,]*3)
    assert mpm.almosteq(arb.norm2(x), 3*arb.pi**2)


def test_binom2(digits_case):
    arb.set_dps(digits_case)
    assert arb.binom2(0,0) == arb.arb(1)
    assert arb.binom2(1,0) == arb.arb(1)
    assert arb.binom2(1,1) == arb.arb(0) # invalids are 0, same a binom()
    assert arb.binom2(1,2) == arb.arb(0)
    assert arb.binom2(2,1) == arb.arb(2)
    assert arb.binom2(4,2) == arb.arb(12)
    

def test_fboys(digits_case):
    arb.set_dps(digits_case)
    # Gamma(0.5,0,x) == sqrt(pi) * erf(sqrt(x))
    # boys(0,x) = 0.5*Gamma(0.5,0,x) / sqrt(x)
    #   = 0.5 * sqrt(pi) * erf(sqrt(x)) / sqrt(x)
    assert mpm.almosteq(
        arb.fboys(0, arb.pi),
        0.5 * mpm.erf(mpm.sqrt(arb.pi))
    )
    assert mpm.almosteq(
        arb.fboys(0, 1),
        0.5 * arb.sqrt(arb.pi) * mpm.erf(1)
    )
    # boys(m,0) = 0.5/(m+0.5) = 1/(2m+1)
    assert mpm.almosteq(arb.fboys(2, 0), arb.arb(0.2)) 
    # limiting case
    assert arb.fboys(0, 0) == arb.arb(1)
