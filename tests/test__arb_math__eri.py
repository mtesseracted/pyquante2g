import pytest
import pyquante22.arb_math as arb
import mpmath as mpm


# decimal digits of precision to test (approx because mpmath converts to bits)
@pytest.fixture(params=[8, 16, 32, 64, 128])
def digits_case(request):
    return request.param


def test_coloumb_rep4_s(digits_case):
    # test for known solution with s-type at center=0
    arb.set_dps(digits_case)
    eri_fac = arb.arb(2) * (arb.pi**arb.arb(2.5))
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    g0 = [arb.aone, am000, ra0]
    ans0 = arb.coulomb_rep4(*sum([g0,]*4, []))
    assert mpm.almosteq(ans0, arb.arb(1/8)*eri_fac)
    g1 = [arb.arb(2)**-arb.arb(6/5), am000, ra0]
    ans1 = arb.coulomb_rep4(*sum([g1,]*4, []))
    assert mpm.almosteq(ans1, arb.arb(1/1)*eri_fac)


def test_coloumb_rep4_even_odd(digits_case):
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
            ans = arb.coulomb_rep4(*godd, *g0, *geve, *g0)
            assert ans == arb.azero
#end def test_coloumb_rep4_even_odd(digits_case):


def test_coloumb_rep4_cen0(digits_case):
    # test for when all centers are same, so fboys is 1/(2n+1)
    arb.set_dps(digits_case)
    ra0 = arb.arr_init([0,]*3)
    am000 = arb.arr_init([0,0,0])
    am100 = arb.arr_init([1,0,0])
    gs = [arb.ahalf, am000, ra0]
    gp = [arb.ahalf, am100, ra0]
    eri_fac = arb.arb(2) * (arb.pi**arb.arb(2.5))
    eri_fac1 = eri_fac / arb.sqrt(arb.arb(2)) # gamma1=gamma2=1

    cxs = [ # Cx for (ns|ss); n=range(7) (up to h-type)
        ['1/1'],
        ['0/1', '0/1'],
        ['1/2', '-1/4', '0/1'],
        ['0/1', '0/1', '0/1', '0/1'],
        ['3/4', '-3/4', '3/16', '0/1', '0/1'],
        ['0/1', '0/1', '0/1', '0/1', '0/1', '0/1'],
        ['15/8', '-45/16', '45/32', '-15/64', '0/1', '0/1', '0/1'],
    ]
    for ii in range(len(cxs)):
        gn = [arb.ahalf, arb.arr_init([ii,0,0]), ra0]
        cx = arb.arr_init(cxs[ii])
        fbs = arb.arr_init([1/(2*m+1) for m in range(len(cx))])
        gt = (cx.T * fbs)[0] * eri_fac1
        ans = arb.coulomb_rep4(*gn, *gs, *gs, *gs)
        assert mpm.almosteq(gt, ans)

    cxp = [ # Cx for (ns|ss); n=range(8) (up to j-type)
        ['0/1', '0/1'],
        ['0/1', '1/4', '0/1'],
        ['0/1', '0/1', '0/1', '0/1'],
        ['0/1', '3/8', '-3/16', '0/1', '0/1'],
        ['0/1', '0/1', '0/1', '0/1', '0/1', '0/1'],
        ['0/1', '15/16', '-15/16', '15/64', '0/1', '0/1', '0/1'],
        ['0/1', '0/1', '0/1', '0/1', '0/1', '0/1', '0/1', '0/1'],
        ['0/1', '105/32', '-315/64', '315/128', '-105/256', '0/1', 
         '0/1', '0/1', '0/1'],
    ]
    for ii in range(len(cxp)):
        gn = [arb.ahalf, arb.arr_init([ii,0,0]), ra0]
        cx = arb.arr_init(cxp[ii])
        fbs = arb.arr_init([1/(2*m+1) for m in range(len(cx))])
        gt = (cx.T * fbs)[0] * eri_fac1
        ans = arb.coulomb_rep4(*gn, *gs, *gp, *gs)
        assert mpm.almosteq(gt, ans)
#end def test_coloumb_rep4_cen0(digits_case):
