import pytest
import pyquante22.arb_math as arb
import mpmath as mpm

# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


def fact_exact(n:int): # use python's arbitrary int
    if n < 2:
        return 1
    return n * fact_exact(n-1)

def binom2_exact(n:int, k:int): # use python's arbitrary int
    if 2*k > n:
        return 0
    return fact_exact(n) // (fact_exact(k) * fact_exact(n-2*k))

def test_norm2(digits_case):
    arb.set_dps(digits_case)
    x = arb.arr_init(list(range(4)))
    assert arb.norm2(x) == arb.arb(14)
    x = arb.arr_init([arb.pi,]*3)
    assert mpm.almosteq(arb.norm2(x), 3*arb.pi**2)


def test_binom2_base(digits_case):
    arb.set_dps(digits_case)
    assert arb.binom2(0,0) == arb.arb(1)
    assert arb.binom2(1,0) == arb.arb(1)
    assert arb.binom2(1,1) == arb.arb(0) # invalids are 0, same a binom()
    assert arb.binom2(1,2) == arb.arb(0)
    assert arb.binom2(2,1) == arb.arb(2)
    assert arb.binom2(4,2) == arb.arb(12)

def test_binom2_large(digits_case):
    arb.set_dps(digits_case)
    test_nks = [
        [4, 1],
        [4, 2],
        [4, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [20, 0],
        [20, 1],
        [20, 2],
        [20, 4],
        [20, 8],
        [20, 9],
        [20, 10],
        [20, 11],
        [40, 0],
        [40, 1],
        [40, 2],
        [40, 4],
        [40, 8],
        [40, 16],
        [40, 19],
        [40, 20],
        [40, 21],
    ]
    for n,k in test_nks:
        gt = str(binom2_exact(n,k))
        if len(gt) >= digits_case: continue # not enough precision
        assert arb.binom2(n,k) == arb.arb(gt), f"@({n},{k})"
#end def test_binom2_large(digits_case):


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
