import sys
import pytest
import pyquante22.utils.math as math
from math import factorial as py_factorial
from math import isclose as py_isclose
from math import pi, sqrt

EPS = sys.float_info.epsilon
def isclose(a, b, rel_tol=16*EPS, abs_tol=0.0):
    return py_isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

numbers = [0, 1, 2, 3, 4, 7, 13, 20, 50]
facts = [1, 1, 2, 6, 24]
# @pytest.fixture(params=list(zip(numbers, facts)), ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.fixture(params=[(n, py_factorial(n)) for n in numbers],)
def fact_pairs(request):
    return request.param

def test_fact_exact(fact_pairs):
    n, f = fact_pairs
    assert math.fact_exact(n) == f


def test_fact_exact2():
    assert math.fact_exact(0, dn=2) == 1
    assert math.fact_exact(1, dn=2) == 1
    assert math.fact_exact(2, dn=2) == 2
    assert math.fact_exact(3, dn=2) == 3
    assert math.fact_exact(4, dn=2) == 4*2
    assert math.fact_exact(5, dn=2) == 5*3
    assert math.fact_exact(6, dn=2) == 6*4*2
    assert math.fact_exact(7, dn=2) == 7*5*3
    assert math.fact_exact(8, dn=2) == 8*6*4*2


def test_bionom_exact():
    assert math.binom_exact(0,0) == 1
    assert math.binom_exact(1,0) == 1
    assert math.binom_exact(1,1) == 1
    assert math.binom_exact(1,2) == 0
    assert math.binom_exact(2,1) == 2
    assert math.binom_exact(4,0) == 1
    assert math.binom_exact(4,1) == 4
    assert math.binom_exact(4,2) == 6
    assert math.binom_exact(4,3) == 4
    assert math.binom_exact(4,4) == 1
    assert math.binom_exact(4,5) == 0


def test_binom2_exact():
    assert math.binom2k_exact(0,0) == 1
    assert math.binom2k_exact(1,0) == 1
    assert math.binom2k_exact(1,1) == 0 # invalids are 0, same a binom()
    assert math.binom2k_exact(1,2) == 0
    assert math.binom2k_exact(2,1) == 2
    assert math.binom2k_exact(4,0) == 1
    assert math.binom2k_exact(4,1) == 12
    assert math.binom2k_exact(4,2) == 12
    assert math.binom2k_exact(4,3) == 0
    assert math.binom2k_exact(4,4) == 0
    assert math.binom2k_exact(4,5) == 0


def test_gamma_half_int():
    # go up tp 80 for 4*amax; amax=20
    sqpi = sqrt(pi)
    for n in range(80):
        expected = math.fact_exact(2*n -1, dn=2) / (2**n)
        assert isclose(math.gamma_half_int(n), expected)


@pytest.mark.arb
def test_igamma():
    import pyquante22.arb_math as am
    test_values = [
        (0.5, 1e-12), (0.5, 1e-20),
        (0.5, 0.0), (0.5, 0.1), (0.5, 0.5), (0.5, 1.0), (0.5, 4.0),
        (1.5, 0.0), (1.5, 0.5), (1.5, 1.0), (1.5, 2.0), (1.5, 4.0),
        (2.5, 1e-12), (2.5, 0.1), (2.5, 10.1), (2.5, 20.5),
        (5.0, 0.1), (5.0, 2.0), (5.0, 6.0), (5.0, 8.0), (5.0, 10.0),
        (10.0, 0.1), (10.0, 1.0), (10.0, 11.0), (10.0, 50.0),
        (10.5, 0.1), (10.5, 1.0), (10.5, 11.0), (10.5, 50.0),
        (20.5, 0.1), (20.5, 1.0), (20.5, 11.0), (20.5, 50.0),
        (40.5, 0.1), (40.5, 1.0), (40.5, 11.0), (40.5, 50.0),
    ]
    for a, x in test_values:
        expected = float(am.gammainc(a, 0, x, regularized=False))
        result = math.igamma(a, x)
        perr = (result - expected) / expected if expected != 0 else result
        amax = max(abs(result), abs(expected))
        passed = isclose(result, expected,rel_tol=16*EPS, abs_tol=EPS**2)
        assert passed, f"Failed for a={a}, x={x}, %err={perr:.2e}, amax={amax:.2e}"  


@pytest.mark.arb
def test_log_igamma():
    # go up tp 80 for 4*amax; amax=20
    import pyquante22.arb_math as am
    test_values = [
        (20, 1e-12), (20, 1e-20), (80, 1e-12), (80, 1e-20),
        (20.5, 0.1), (20.5, 1.0), (20.5, 11.0), (20.5, 50.0),
        (40.5, 0.1), (40.5, 1.0), (40.5, 11.0), (40.5, 50.0),
        (80.5, 0.1), (80.5, 1.0), (80.5, 1.1), (80.5, 11.0),
        (80.5, 20), (80.5, 50), (80.5, 90), (80.5, 160),
    ]
    for a, x in test_values:
        expected = float(am.log(am.gammainc(a, 0, x, regularized=False)))
        result = math.log_igamma(a, x)
        perr = (result - expected) / expected if expected != 0 else result
        amax = max(abs(result), abs(expected))
        passed = isclose(result, expected,rel_tol=16*EPS, abs_tol=EPS**2)
        assert passed, f"Failed for a={a}, x={x}, %err={perr:.2e}, amax={amax:.2e}"  


# @pytest.mark.skip(reason="Temporarily skip fboys tests")
test_values = [
    (0., 1e-20), (0., 1e-16), (0., 1e-14), (0., 1e-12), (0., 1e-10),
    (0., 1e-8), (0., 1e-6), (0., 1e-4), (0., 1e-2),
    (0.0, 0.0), (0.0, 0.1), (0.0, 0.0), (0.0, 1.0), (0.0, 4.0),
    (1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 4.0),
    (2.0, 1e-12), (2.0, 0.1), (2.0, 10.1), (2.0, 20.5),
    (5.0, 0.1), (5.0, 2.0), (5.0, 6.0), (5.0, 8.0), (5.0, 10.0),
    (10.0, 0.1), (10.0, 1.0), (10.0, 11.0), (10.0, 50.0),
    (20.0, 0.1), (20.0, 1.0), (20.0, 11.0), (20.0, 20.0),
    (30.0, 0.1), (30.0, 1.0), (30.0, 11.0), (30.0, 20.0),
    (30.0, 11.0), (30.0, 1.0), (30.0, 11.0), (30.0, 20.0),
    (39.0, 1e-16), (39.0, 1e-14), (39.0, 1e-12), (39.0, 1e-10),
    (40.0, 0.1), (40.0, 1.0), (40.0, 11.0), (40.0, 20.0), 
    (41.0, 0.1), (41.0, 1.0), (41.0, 11.0), (41.0, 20.0),
    (41.0, 0.9), (41.0, 1.1), (41.0, 1.01), (41.0, 0.99),
    (50.0, 0.1), (50.0, 1.0), (50.0, 1.1), (50.0, 11.0), (50, 20),
    (60.0, 0.1), (60.0, 1.0), (60.0, 1.1), (60.0, 11.0), (60, 20),
    (70.0, 0.1), (70.0, 1.0), (70.0, 1.1), (70.0, 11.0), (70, 20),
    (80, 0), (80, 1e-16), (80, 1e-14), (80, 1e-12), (80, 1e-10), 
    (80, 1e-8), (80, 1e-6), (80, 1e-4), (80, 1e-2),
    (80.0, 0.1), (80.0, 1.0), (80.0, 1.1), (80.0, 11.0), 
    (80, 20), (80, 50), (80, 90), (80, 160),
]
@pytest.fixture(params=test_values, ids=lambda p: f"m={p[0]},x={p[1]}")
def fboys_pair(request):
    return request.param


@pytest.mark.arb
def test_fboys(fboys_pair):
    # go up tp 80 for 4*amax; amax=20
    # densely sample around piecewise boundaries, (m=m, x<1e-15), (m=40, x=1)
    import pyquante22.arb_math as am
    a, x = fboys_pair
    # can't get as accurate for large m and x: (m>20,x=m*0.9)
    rel_tol = 16*EPS if a < 20 else 64*EPS 
    expected = float(am.fboys(a, x))
    result = math.fboys(a, x)
    perr = (result - expected) / expected if expected != 0 else result
    amax = max(abs(result), abs(expected))
    passed = isclose(result, expected,rel_tol=rel_tol, abs_tol=EPS**2)
    assert passed, f"Failed for a={a}, x={x}, %err={perr:.2e}, amax={amax:.2e}"  
