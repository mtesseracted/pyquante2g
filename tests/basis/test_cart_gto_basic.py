import pytest, sys
import numpy as np
from pyquante22.basis.cart_gto import CartGtoPrim as CGP
from pyquante22.basis.cart_gto import CartGtoPrimPair as CGPP
import pyquante22.array_ops as arr


EPS = sys.float_info.epsilon
backend = "torch"
ao = arr.ArrayOps(
    backend, device="cpu", default_dtype="float64", int_dtype="int64"
)


def pad_ragged(ragged):  # stack a ragged list by padding to the max size
    max_len = max(len(row) for row in ragged)
    padded = [row + [0] * (max_len - len(row)) for row in ragged]
    return np.array(padded)


# @TODO add pytest.fixture of backends to test over, that will ensure
# the convolution is correct before adding an explicit test
# it can only do one right now, because it's hardcoded at the top of cart_gto.py
def test_add():
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    amp = ao.arr([1, 0, 0], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    aa = ao.arr([1.0] * 3)
    gs = CGP(ams, r0, aa)
    assert len(gs) == 3
    gs2 = gs + gs
    assert len(gs2) == 6


# @pytest.mark.arb # @DEBUG
@pytest.mark.med
def test_calc_binoms():
    from pyquante22.ref_ints.utils import binomial_prefactor

    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    am1 = ao.arr([3, 1, 2], dtype=ao.int_dtype)
    am2 = ao.arr([4, 2, 2], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    rc = ao.arr([0.625, 0.125, 0.375])
    rd = ao.arr([4.0, 1.0, 2.0])
    am_list = [ams, am1, am2]
    r_list = [r0, rb, rc, rd]
    aa = ao.arr([0.25] * 3)
    ab = ao.arr([0.5] * 3)

    def gt_bps_arb():
        pass

    # import pyquante22.arb_math as arb
    # def gt_bps_arb(ia, ib, xpa, xpb):
    #     imax = ia+ib+1
    #     ret = [None,]*imax
    #     for ii in range(imax):
    #         ret[ii] = float(arb.binom_pre(ii,ia, ib,xpa,xpb))
    #         # conversion to arb.arb makes it an unfair comparison, because
    #         # it infers a higher precision, e.g. 0.1 would be exact for many
    #         # more digits,
    #         # ret[ii] = float(arb.binom_pre(arb.arb(ii),
    #         #     arb.arb(ia), arb.arb(ib),
    #         #     arb.arb(xpa), arb.arb(xpb)))
    #     return ret

    def gt_bps_ref(ia, ib, xpa, xpb):
        imax = ia + ib + 1
        ret = [None] * imax
        for ii in range(imax):
            ret[ii] = binomial_prefactor(ii, ia, ib, xpa, xpb)
        return ret

    def gt_bps_pair(gpp, method='ref'):
        f = gt_bps_arb if method == 'arb' else gt_bps_ref
        pa = gpp.pab - gpp.ga.cen
        pb = gpp.pab - gpp.gb.cen
        la, lb = gpp.ga.amom, gpp.gb.amom
        rets = [None] * len(gpp)
        for ii in range(len(gpp)):
            rets[ii] = f(
                la[ii].item(), lb[ii].item(), pa[ii].item(), pb[ii].item()
            )
        return rets

    for r1 in r_list:
        for a1 in am_list:
            for r2 in r_list:
                for a2 in am_list:
                    lmax = sum(a1) + sum(a2)
                    rtol = (
                        min(lmax + 1, 16) * 16 * EPS
                    )  # loosen tolerances for high amom
                    atol = min(lmax + 1, 16) / 16 * EPS
                    g1 = CGP(a1, r1, aa)
                    g2 = CGP(a2, r2, ab)
                    g12 = CGPP(g1, g2)
                    bps = np.asarray(g12.calc_prod_binoms().tolist())
                    bpgt = pad_ragged(gt_bps_pair(g12, method='ref'))
                    # bpgt = pad_ragged(gt_bps_pair(g12, method='arb'))
                    passed = np.allclose(bps, bpgt, rtol=rtol, atol=atol)
                    perrs = bps - bpgt
                    mm = np.abs(perrs) > 0
                    perrs[mm] = perrs[mm] / bpgt[mm]
                    assert (
                        passed
                    ), f"fail @ {str(g12)} max(%err): {np.abs(perrs).max():.4e}"


# end def test_calc_binoms():


def test_triple_conv_sum():
    from pyquante22.basis.cart_gto import triple_conv_dot
    batch_sz = 11
    ax = np.random.rand(batch_sz, batch_sz-3)
    ay = np.random.rand(batch_sz, batch_sz-0)
    az = np.random.rand(batch_sz, batch_sz+3)
    fxyz = np.random.rand(batch_sz, ax.shape[1]+ay.shape[1]+az.shape[1])

    # explicit formula we're trying to match
    def triple_sum_dot(ax, ay, az, fxyz):
        totals = np.zeros(ax.shape[0])
        for ii in range(ax.shape[1]):
            for jj in range(ay.shape[1]):
                for kk in range(az.shape[1]):
                    totals += ax[:,ii] * ay[:,jj] * az[:,kk] * fxyz[:,ii+jj+kk]
        return totals

    nargs = [ax, ay, az, fxyz]
    aargs = [ao.arr(a) for a in nargs]
    ans = np.asarray(ao.tolist(triple_conv_dot(*aargs)))
    gt = triple_sum_dot(*nargs)
    perrs = (ans - gt) / gt
    pmax = np.abs(perrs).max()
    assert np.allclose(ans, gt, rtol=16*EPS, atol=EPS/64), \
        f"worst %err: {pmax:.4e}"
#end def test_triple_conv_sum():


def test_calc_h_array():
    from pyquante22.ref_ints.eri import H_array

    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    am1 = ao.arr([3, 1, 2], dtype=ao.int_dtype)
    am2 = ao.arr([4, 2, 2], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    rc = ao.arr([0.625, 0.125, 0.375])
    rd = ao.arr([4.0, 1.0, 2.0])
    am_list = [ams, am1, am2]
    r_list = [r0, rb, rc, rd]
    aa = ao.arr([0.25] * 3)
    ab = ao.arr([0.5] * 3)

    def gt_h_array(gpp):
        pa = gpp.pab - gpp.ga.cen
        pb = gpp.pab - gpp.gb.cen
        la, lb = gpp.ga.amom, gpp.gb.amom
        gab = gpp.gamma
        rets = [None] * len(gpp)
        for ii in range(len(gpp)):
            rets[ii] = H_array(
                la[ii].item(),
                lb[ii].item(),
                pa[ii].item(),
                pb[ii].item(),
                gab[ii].item(),
            ).tolist()
        return rets

    for r1 in r_list:
        for a1 in am_list:
            for r2 in r_list:
                for a2 in am_list:
                    lmax = sum(a1) + sum(a2)
                    rtol = (
                        min(lmax + 1, 16) * 16 * EPS
                    )  # loosen tolerances for high amom
                    atol = min(lmax + 1, 16) / 16 * EPS
                    g1 = CGP(a1, r1, aa)
                    g2 = CGP(a2, r2, ab)
                    g12 = CGPP(g1, g2)
                    hab = np.asarray(g12.calc_h_array().tolist())
                    hgt = pad_ragged(gt_h_array(g12))
                    passed = np.allclose(hab, hgt, rtol=rtol, atol=atol)
                    perrs = hab - hgt
                    mm = np.abs(perrs) > 0
                    perrs[mm] = perrs[mm] / hgt[mm]
                    assert (
                        passed
                    ), f"fail @ {str(g12)} max(%err): {np.abs(perrs).max():.4e}"
    # end for r1 in r_list:
# end def test_calc_h_array():


def test_norm_basic():
    alpha = 1.0 # power of an s-type orbital
    gamma = 2*alpha # orbital squared doubles exponent
    # gamma = 1/(2*sigma^2) # relation of gamma to normal distro's sigma
    sigma = 1.0 / np.sqrt(2*gamma)
    n1d = np.pow(2*np.pi*sigma**2, -1/4) # sqrt again for half of the norm
    n3d = n1d**3

    ams = ao.arr([0,]*3, dtype=ao.int_dtype)
    r0 = ao.arr([0,]*3)
    aa = ao.arr([1,]*3)
    ans = CGP(ams, r0, aa).norm
    assert np.isclose(ans, n3d, rtol=16*EPS, atol=0)


def test_norm_vs_overlap():
    rtol, atol = 16*EPS, 0.0
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    am1 = ao.arr([3, 1, 2], dtype=ao.int_dtype)
    am2 = ao.arr([4, 2, 2], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    rc = ao.arr([0.625, 0.125, 0.375])
    rd = ao.arr([4.0, 1.0, 2.0])
    am_list = [ams, am1, am2]
    r_list = [r0, rb, rc, rd]
    aa = ao.arr([0.25] * 3)

    for r1 in r_list:
        for a1 in am_list:
            g1 = CGP(a1, r1, aa)
            overlap = (g1*g1).calc_overlap()
            oovesq = 1 / ao.sqrt(overlap) # should never get div 0
            norm = g1.norm
            passed = np.isclose(oovesq, norm, rtol=rtol, atol=atol)
            perr = (oovesq - norm) / norm
            assert passed, f"fail @ {str(g1)} %err: {perr:.4e}"
# end def test_norm_vs_overlap():

