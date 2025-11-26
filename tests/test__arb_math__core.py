import pytest
import pyquante22.arb_math as arb
import mpmath as mpm

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

# digits of precision to test
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


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


def test_arb_approx(digits_case):
    arb.set_dps(digits_case)
    assert arb.arb(0.1) == arb.arb("0.1") # 1/10 is infinite in binary
    arb.set_dps(digits_case)
    assert arb.arb(1.1) == arb.arb("1.1") # 1/10 is infinite in binary
    

def test_dps(digits_case, pi256):
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


def test_loggamma(digits_case, pi256):
    arb.set_dps(digits_case)
    assert mpm.almosteq(
        arb.exp(arb.loggamma(arb.arb(0.5))),
        arb.sqrt(arb.arb(pi256))
    ) # Gamma(1/2) == sqrt(pi)
    assert mpm.almosteq(
        arb.exp(arb.loggamma(arb.arb(4))),
        arb.arb(6)
     ) # 3!
    

def test_gammainc(digits_case, pi256):
    arb.set_dps(digits_case)
    # assert arb.gammainc(arb.arb(0.5)) == arb.sqrt(arb.arb(pi256))
    assert mpm.almosteq(
        arb.gammainc(arb.arb(0.5)),
        arb.sqrt(arb.pi)
    )
    a = arb.gammainc(2, 0, 1) + arb.gammainc(2, 1, 2)
    b = arb.gammainc(2, 0, 2)
    assert mpm.almosteq(a, b)


def test_binomial(digits_case):
    arb.set_dps(digits_case)
    assert arb.binomial(1,0) == arb.arb(1)
    assert arb.binomial(0,1) == arb.arb(0)
    assert arb.binomial(1,1) == arb.arb(1)
    assert arb.binomial(4,4) == arb.arb(1)
    assert arb.binomial(5,3) == arb.arb(10)
    

