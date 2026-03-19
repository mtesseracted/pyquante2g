import pytest
import pyquante2g.arb_math as arb
import mpmath as mpm

pytestmark = pytest.mark.arb   # applies to all tests in this file

# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


def test_nai_s0(digits_case):
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


def test_nai_sr(digits_case):
    # test for known solution with s-type at various radii
    # For rho(r) = exp(-a*r^2)
    # V(r) = (pi/a)^(3/2) * erf(sqrt(a)*r)/r
    arb.set_dps(digits_case)
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    alpha = arb.aone
    gam = 2*alpha
    g0 = [alpha, am000, ra0]

    # limiting case at x=0
    expected = 2 * arb.pi / gam
    result = -arb.nuclear_attraction(*g0, *g0, ra0)
    perr = (result - expected) / expected
    assert mpm.almosteq(result, expected), f"@{float(0):.2f} %err = {float(perr):.2e}"

    xs = mpm.linspace(0, 5, 51)[1:] # skip 0
    c = arb.pow(arb.pi/(2*alpha), arb.arb(3/2))
    for x in xs:
        expected = c * arb.erf(arb.sqrt(gam)*x) / x
        ra1 = arb.arr_init([x,0,0])
        result = -arb.nuclear_attraction(*g0, *g0, ra1)
        perr = (result - expected) / expected
        assert mpm.almosteq(result, expected), f"@{float(x):.2f} %err = {float(perr):.2e}"
#end def test_nai_sr(digits_case):


def test_nai_dr(digits_case):
    # test for known solution with d-type at various radii
    # For rho(r) = r^2 exp(-a*r^2)
    # V(r) = 3*sqrt(pi)/(8*a^(5/2))*erf(sqrt(a)*r)/r - exp(-a*r^2)/(4*a^2)
    # we can get this with NAIs by summing NAI(di,s|r); di in [dx2, dy2, dz2]
    arb.set_dps(digits_case)
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    am200 = arb.arr_init([2,0,0])
    am020 = arb.arr_init([0,2,0])
    am002 = arb.arr_init([0,0,2])
    alpha = arb.aone
    gam = 2 * alpha
    g0 = [alpha, am000, ra0]
    gz = [alpha, am002, ra0]
    gy = [alpha, am020, ra0]
    gx = [alpha, am200, ra0]
    gds = [gx, gy, gz]

    # limiting case at x=0
    expected = 2 * arb.pi * arb.pow(gam, -2)
    result = -sum([arb.nuclear_attraction(*gd, *g0, ra0) for gd in gds])
    perr = (result - expected) / expected
    assert mpm.almosteq(result, expected), f"@{float(0):.2f} %err = {float(perr):.2e}"

    xs = mpm.linspace(0, 5, 51)[1:] # skip 0
    c0 = 3 * arb.sqrt(arb.pi)/ (8*arb.pow(gam,5/2))
    c1 = -arb.arb(0.25) / arb.pow(gam, 2)
    for x in xs:
        expected  = c0 * arb.erf(arb.sqrt(gam)*x) / x
        expected += c1 * arb.exp(-gam*x*x)
        expected *= 4 * arb.pi
        ra1 = arb.arr_init([x,0,0])
        result = -sum([arb.nuclear_attraction(*gd, *g0, ra1) for gd in gds])
        perr = (result - expected) / expected
        assert mpm.almosteq(result, expected), f"@{float(x):.2f} %err = {float(perr):.2e}"
#end def test_nai_dr(digits_case):
