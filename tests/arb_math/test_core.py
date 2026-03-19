import pytest
import pyquante2g.arb_math as arb
import mpmath as mpm
from fractions import Fraction

pytestmark = pytest.mark.arb   # applies to all tests in this file

# 256 digits of pi from `bc -l <<< "scale=256; 4*a(1)"`
@pytest.fixture
def pi256():
    return (
        "3.141592653589793238462643383279502884197169399375105820974944592307"
        "81640628620899862803482534211706798214808651328230664709384460955058"
        "22317253594081284811174502841027019385211055596446229489549303819644"
        "288109756659334461284756482337867831652712019091456484")

# 256 digits of e from `bc -l <<< "scale=256; e(1)"`
@pytest.fixture
def e256():
    return (
        "2.718281828459045235360287471352662497757247093699959574966967627724"
        "07663035354759457138217852516642742746639193200305992181741359662904"
        "35729003342952605956307381323286279434907632338298807531952510190115"
        "738341879307021540891499348841675092447614606680822648")

# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param

def fact_exact(n:int): # use python's arbitrary int
    if n < 2:
        return 1
    return n * fact_exact(n-1)

def binom_exact(n:int, k:int): # use python's arbitrary int
    if k > n:
        return 0
    return fact_exact(n) // (fact_exact(k) * fact_exact(n-k))


def test_frac_approx():
    f = arb.frac_approx(1/2)
    assert f.numerator == 1 and f.denominator == 2
    f = arb.frac_approx(1/3)
    assert f.numerator == 1 and f.denominator == 3
    f = arb.frac_approx(1/5)
    assert f.numerator == 1 and f.denominator == 5
    f = arb.frac_approx(1.1)
    assert f.numerator == 11 and f.denominator == 10


def test_arange():
    assert arb.arange(0) == range(0)
    assert arb.arange(1) == range(1)
    assert arb.arange(2) == range(2)
    assert arb.arange(1,7,) == range(1,7,)
    assert arb.arange(1,7,3) == range(1,7,3)
    assert arb.arange(7,1,-3) == range(7,1,-3)


def test_arb(digits_case): # test arb() correctly handles values that are inexact in binary
    arb.set_dps(digits_case)
    assert arb.arb(0.1) == arb.arb("0.1") # 1/10 is infinite in binary
    assert arb.arb(-0.1) == arb.arb("-0.1") # 
    assert arb.arb(1.1) == arb.arb("1.1") # 
    assert arb.arb(1.1) == arb.arb(arb.arb(1.1)) # recast should be allowed
    # should we support upcasting an mpf after changing dps similarly?


def test_arr_init(digits_case):
    arb.set_dps(digits_case)
    vals_str = ["0", "1/2", "1/3", "1/5", "-1"]
    vals = [float(Fraction(v)) for v in vals_str]
    arr0 = arb.arr_init(vals_str)       # convert from string
    arr1 = arb.arr_init(vals)           # convert from inexact floats
    assert arr0[3] == arb.arb(0.2)      # test element access
    assert arr1[3] == arb.arb(0.2)
    assert all([a0 == a1 for a0, a1 in zip(arr0,arr1)]) # test list comprehension
    

def test_dps(digits_case, pi256): # test set_dps() propogates to constants
    arb.set_dps(digits_case)
    api = arb.pi
    pigt = arb.arb(pi256)
    assert float(api - pigt) == 0.0


def test_nint(digits_case, pi256):
    arb.set_dps(digits_case)
    assert arb.nint(arb.pi) == arb.nint(arb.arb(pi256))
    assert arb.nint(arb.pi) == arb.arb(3)
    assert arb.nint(3.51) == arb.arb(4)
    assert arb.nint(3.49) == arb.arb(3)
    assert arb.nint(-1.1) == arb.arb(-1)
    assert arb.nint(0.0001) == arb.arb(0)


def test_sqrt(digits_case, pi256):
    arb.set_dps(digits_case)
    assert arb.sqrt(arb.arb(4)) == arb.arb(2)
    assert arb.sqrt(arb.arb(169)) == arb.arb(13)
    assert arb.sqrt(arb.arb(pi256)**2) == arb.arb(pi256)


def test_exp(digits_case, e256):
    arb.set_dps(digits_case)
    # test complex? => not built into core, initialized with mpm.mpc(x,y) instead of mpm.mpf(x)
    assert arb.exp(arb.arb(0)) == arb.arb(1)
    assert arb.exp(arb.arb(1)) == arb.arb(e256)


def test_lgamma(digits_case, pi256):
    arb.set_dps(digits_case)
    assert mpm.almosteq(
        arb.exp(arb.lgamma(arb.arb(0.5))),
        arb.sqrt(arb.arb(pi256))
    ) # Gamma(1/2) == sqrt(pi)
    assert mpm.almosteq(
        arb.exp(arb.lgamma(arb.arb(4))),
        arb.arb(6)
     ) # 3!
    

def test_gammainc(digits_case,):
    arb.set_dps(digits_case)
    assert mpm.almosteq(
        arb.gammainc(arb.arb(0.5)),
        arb.sqrt(arb.pi)
    )
    # at s=0.5, lower==upper at x=erfinv(0.5)^2
    # at s=1, lower==upper at x=log(2)
    l2 = mpm.log(arb.arb(2))
    assert mpm.almosteq(
        arb.gammainc(1, 0, l2),
        arb.gammainc(1, l2, None),
    )
    assert mpm.almosteq(
        arb.gammainc(1, 0, l2),
        arb.arb(0.5)
    )
    a = arb.gammainc(2, 0, 1) + arb.gammainc(2, 1, 2)
    b = arb.gammainc(2, 0, 2)
    assert mpm.almosteq(a, b) # Gamma(x,a,b) + Gamma(x,b,c) == Gamma(x,a,c)


def test_binom_base(digits_case):
    arb.set_dps(digits_case)
    assert arb.binom(1,0) == arb.arb(1)
    assert arb.binom(0,1) == arb.arb(0)
    assert arb.binom(1,1) == arb.arb(1)
    assert arb.binom(4,4) == arb.arb(1)
    assert arb.binom(5,3) == arb.arb(10)

def test_binom_large(digits_case):
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
        [20, 19],
        [20, 20],
        [20, 21],
        [40, 0],
        [40, 1],
        [40, 2],
        [40, 4],
        [40, 8],
        [40, 16],
        [40, 19],
        [40, 20],
        [40, 21],
        [40, 39],
        [40, 40],
        [40, 41],
    ]
    for n,k in test_nks:
        gt = str(binom_exact(n,k))
        if len(gt) >= digits_case: continue # not enough precision
        assert arb.binom(n,k) == arb.arb(gt), f"@{n},{k}"
#end def test_binom_large(digits_case):

    