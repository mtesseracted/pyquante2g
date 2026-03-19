import os
from pathlib import Path
import pytest, sys
import numpy as np
from pyquante2g.basis.cart_gto import CartGtoPrim as CGP
from pyquante2g.basis.cart_gto import CartGtoPrimPair as CGPP
from tests.torch_test_utils import TORCH_FLOAT_DTYPES, make_torch_ops


ALLCLOSE_LOG_PATH = Path(
    os.environ.get(
        "PYQUANTE22_TEST_CART_GTO_BASIC_LOG",
        "./tmp/pyquante2g_test_cart_gto_basic_allclose.log",
    )
)


FLOAT_DTYPES = TORCH_FLOAT_DTYPES


def make_ao_params():
    return [pytest.param(dtype, id=f"torch:{dtype}") for dtype in FLOAT_DTYPES]


@pytest.fixture(scope="module", params=make_ao_params())
def ao(request):
    return make_torch_ops(default_dtype=request.param)


@pytest.fixture(scope="module")
def eps(ao):
    if "float32" in str(ao.default_dtype):
        return np.finfo(np.float32).eps
    return np.finfo(np.float64).eps


@pytest.fixture(scope="module", autouse=True)
def reset_allclose_log():
    ALLCLOSE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALLCLOSE_LOG_PATH.write_text("", encoding="ascii")


def pad_ragged(ragged):  # stack a ragged list by padding to the max size
    max_len = max(len(row) for row in ragged)
    padded = [row + [0] * (max_len - len(row)) for row in ragged]
    return np.array(padded)


def to_numpy(ao, x):
    return np.asarray(ao.tolist(x))


def to_host_array(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        x = x.get()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def safe_abs_rel_errors(pred, gt, rtol, atol):
    pred = to_host_array(pred)
    gt = to_host_array(gt)
    abs_diff = np.abs(pred - gt)
    abs_gt = np.abs(gt)
    if rtol > 0.0:
        denom = abs_gt + float(atol) / float(rtol)
    else:
        denom = np.maximum(abs_gt, float(atol))
    errs = abs_diff / denom
    return errs.reshape(-1)


def fmt_setting(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float):
        return f"{value:.4e}"
    return repr(value).replace(" ", "")


def allclose_settings(ao, **settings):
    vals = {"torch": ao.name, **settings}
    return ",".join(f"{key}={fmt_setting(val)}" for key, val in vals.items())


def log_allclose_summary(test_name, label, settings, pred, gt, rtol=0.0, atol=0.0):
    errs = safe_abs_rel_errors(pred, gt, rtol=rtol, atol=atol)
    if errs.size == 0:
        summary = np.full(5, np.nan, dtype=np.float64)
    else:
        summary = np.quantile(errs, [0.0, 0.25, 0.5, 0.75, 1.0])
    summary_txt = ",".join(f"{val:.4e}" for val in summary)
    with ALLCLOSE_LOG_PATH.open("a", encoding="ascii") as fh:
        fh.write(f"{test_name}#{label}#{settings}@{summary_txt}\n")


# @TODO add a dedicated fixture for dtype coverage that will ensure
# the convolution is correct before adding an explicit test
# it can only do one right now, because it's hardcoded at the top of cart_gto.py
def test_add(ao, eps):
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    amp = ao.arr([1, 0, 0], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    aa = ao.arr([1.0] * 2)
    gs = CGP.create_contraction(ams, r0, aa, ao=ao)
    assert len(gs) == 2
    gs2 = gs + gs
    assert len(gs2) == 4
    gs3 = gs2 + CGP.create_contraction(amp, rb, aa, ao=ao)
    assert len(gs3) == 6


def test_mul(ao, eps):
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    amp = ao.arr([1, 0, 0], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    aa = ao.arr([1.0] * 3)
    ga = CGP.create_contraction(ams, r0, aa, ao=ao)
    ga = ga + ga
    gb = CGP.create_contraction(amp, rb, aa, ao=ao)
    gmul = ga * gb
    gab = CGPP(ga, gb)
    assert gab.shape == gmul.shape
    assert ao.abs(gab.pab - gmul.pab).max().item() == 0.0


def test_calc_eri_self_fallback_mixed_prims(ao, eps):
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    ra = ao.arr([0.0, 0.0, 0.0])
    rb = ao.arr([0.5, 0.0, 0.0])

    # Na=1, Nb>1 (variation along b should trigger fallback)
    ga = CGP.create_contraction(ams, ra, ao.arr([0.8]), ao=ao)
    gb = CGP.create_contraction(ams, rb, ao.arr([0.2, 0.6]), ao=ao)
    gpp = CGPP(ga, gb)
    pab_var = ao.abs(gpp.pab - gpp.pab[0, 0]).max().item()
    assert pab_var > 0.0
    eri_self = to_numpy(ao, gpp.calc_eri_self())
    eri_gen = to_numpy(ao, gpp.calc_eri(gpp))
    log_allclose_summary(
        "test_calc_eri_self_fallback_mixed_prims",
        "nb_varies",
        allclose_settings(ao, ra=ra, rb=rb, ga_alpha=[0.8], gb_alpha=[0.2, 0.6]),
        eri_self,
        eri_gen,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(eri_self, eri_gen, rtol=1e-10, atol=1e-12)

    # Swapped: Na>1, Nb=1 (variation along a should trigger fallback)
    ga = CGP.create_contraction(ams, ra, ao.arr([0.2, 0.6]), ao=ao)
    gb = CGP.create_contraction(ams, rb, ao.arr([0.8]), ao=ao)
    gpp = CGPP(ga, gb)
    pab_var = ao.abs(gpp.pab - gpp.pab[0, 0]).max().item()
    assert pab_var > 0.0
    eri_self = to_numpy(ao, gpp.calc_eri_self())
    eri_gen = to_numpy(ao, gpp.calc_eri(gpp))
    log_allclose_summary(
        "test_calc_eri_self_fallback_mixed_prims",
        "na_varies",
        allclose_settings(ao, ra=ra, rb=rb, ga_alpha=[0.2, 0.6], gb_alpha=[0.8]),
        eri_self,
        eri_gen,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(eri_self, eri_gen, rtol=1e-10, atol=1e-12)


def test_str(ao, eps):
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    aa = ao.arr([1.0] * 1)
    gto = CGP.create_contraction(ams, r0, aa, ao=ao)
    s = str(gto)
    assert "GTO" in s, f"Unexpected str({type(gto)}), was the method updated?"

    gp = CGPP(gto, gto)
    s = str(gp)
    assert "PairGTO" in s, f"Unexpected str({type(gp)}), was the method updated?"


def test_repr(ao, eps):
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    aa = ao.arr([1.0] * 3)
    gto = CGP.create_contraction(ams, r0, aa, ao=ao)
    s = repr(gto)
    assert "CartGtoPrim" in s, f"Unexpected repr({type(gto)}), was the method updated?"

    gp = CGPP(gto, gto)
    s = repr(gp)
    assert "A: CartGtoPrim" in s, f"Unexpected repr({type(gp)}), was the method updated?"


# @pytest.mark.arb # @DEBUG
@pytest.mark.med
def test_calc_binoms(ao, eps):
    from pyquante2g.ref_ints.utils import binomial_prefactor

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
        raise NotImplementedError()

    # import pyquante2g.arb_math as arb
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
        pa = gpp.pab - gpp.ga.cen[:, None, :]
        pb = gpp.pab - gpp.gb.cen[None, :, :]
        la, lb = gpp.ga.amom, gpp.gb.amom
        Na, Nb, nd = pa.shape
        ragged = []
        max_len = 0
        for ia in range(Na):
            for ib in range(Nb):
                for d in range(nd):
                    la_id = int(la[ia, d].item())
                    lb_id = int(lb[ib, d].item())
                    pa_id = float(pa[ia, ib, d].item())
                    pb_id = float(pb[ia, ib, d].item())
                    vals = f(la_id, lb_id, pa_id, pb_id)
                    max_len = max(max_len, len(vals))
                    ragged.append(vals)
        padded = pad_ragged(ragged)
        return padded.reshape(Na, Nb, nd, max_len)

    for r1 in r_list:
        for a1 in am_list:
            for r2 in r_list:
                for a2 in am_list:
                    lmax = int(ao.sum(a1).item()) + int(ao.sum(a2).item())
                    rtol = (
                        min(lmax + 1, 16) * 16 * eps
                    )  # loosen tolerances for high amom
                    atol = min(lmax + 1, 16) * eps
                    g1 = CGP.create_contraction(a1, r1, aa, ao=ao)
                    g2 = CGP.create_contraction(a2, r2, ab, ao=ao)
                    g12 = CGPP(g1, g2)
                    bps = np.asarray(g12.calc_prod_binoms().tolist())
                    bpgt = gt_bps_pair(g12, method='ref')
                    # bpgt = pad_ragged(gt_bps_pair(g12, method='arb'))
                    ls = bpgt.shape[-1]
                    settings = allclose_settings(
                        ao,
                        r1=r1,
                        a1=a1,
                        r2=r2,
                        a2=a2,
                        rtol=rtol,
                        atol=atol,
                    )
                    log_allclose_summary(
                        "test_calc_binoms",
                        "prod_binoms_main",
                        settings,
                        bps[..., :ls],
                        bpgt,
                        rtol=rtol,
                        atol=atol,
                    )
                    passed = np.allclose(bps[..., :ls], bpgt, rtol=rtol, atol=atol)
                    log_allclose_summary(
                        "test_calc_binoms",
                        "prod_binoms_tail_zero",
                        settings,
                        bps[..., ls:],
                        np.zeros_like(bps[..., ls:]),
                        rtol=rtol,
                        atol=atol,
                    )
                    assert np.allclose(bps[..., ls:], 0, rtol=rtol, atol=atol)

                    if not passed:
                        diff = bps[...,:ls] - bpgt
                        tol = atol + rtol * np.abs(bpgt)
                        fail_mask = np.abs(diff) > tol
                        perrs = np.abs(diff[fail_mask] / bpgt[fail_mask])
                        debug_var = 1

                    assert (
                        passed
                    ), f"fail @ {str(g12)} max(%err): {np.abs(perrs).max():.4e}"


# end def test_calc_binoms():


def test_triple_conv_sum(ao, eps):
    from pyquante2g.basis.cart_gto import GtoMath
    gm = GtoMath(ao)
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
    ans = np.asarray(ao.tolist(gm.triple_conv_dot(*aargs)))
    gt = triple_sum_dot(*nargs)
    perrs = (ans - gt) / gt
    pmax = np.abs(perrs).max()
    log_allclose_summary(
        "test_triple_conv_sum",
        "triple_conv_dot",
        allclose_settings(ao, batch_sz=batch_sz),
        ans,
        gt,
        rtol=16 * eps,
        atol=eps / 64,
    )
    assert np.allclose(ans, gt, rtol=16*eps, atol=eps/64), \
        f"worst %err: {pmax:.4e}"
#end def test_triple_conv_sum():


def test_calc_h_array(ao, eps):
    from pyquante2g.ref_ints.eri import H_array

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
        pa = gpp.pab - gpp.ga.cen[:, None, :]
        pb = gpp.pab - gpp.gb.cen[None, :, :]
        la, lb = gpp.ga.amom, gpp.gb.amom
        gab = gpp.gamma
        Na, Nb, nd = pa.shape
        ragged = []
        max_len = 0
        for ia in range(Na):
            for ib in range(Nb):
                for d in range(nd):
                    vals = H_array(
                        int(la[ia, d].item()),
                        int(lb[ib, d].item()),
                        float(pa[ia, ib, d].item()),
                        float(pb[ia, ib, d].item()),
                        float(gab[ia, ib].item()),
                    ).tolist()
                    max_len = max(max_len, len(vals))
                    ragged.append(vals)
        padded = pad_ragged(ragged)
        return padded.reshape(Na, Nb, nd, max_len)

    for r1 in r_list:
        for a1 in am_list:
            for r2 in r_list:
                for a2 in am_list:
                    lmax = int(ao.sum(a1).item()) + int(ao.sum(a2).item())
                    rtol = (
                        min(lmax + 1, 16) * 16 * eps
                    )  # loosen tolerances for high amom
                    atol = min(lmax + 1, 16) * eps
                    g1 = CGP.create_contraction(a1, r1, aa, ao=ao)
                    g2 = CGP.create_contraction(a2, r2, ab, ao=ao)
                    g12 = CGPP(g1, g2)
                    hab = np.asarray(g12.calc_h_array().tolist())
                    hgt = gt_h_array(g12)
                    ls = hgt.shape[-1]
                    settings = allclose_settings(
                        ao,
                        r1=r1,
                        a1=a1,
                        r2=r2,
                        a2=a2,
                        rtol=rtol,
                        atol=atol,
                    )
                    log_allclose_summary(
                        "test_calc_h_array",
                        "h_array_main",
                        settings,
                        hab[..., :ls],
                        hgt,
                        rtol=rtol,
                        atol=atol,
                    )
                    passed = np.allclose(hab[..., :ls], hgt, rtol=rtol, atol=atol)
                    log_allclose_summary(
                        "test_calc_h_array",
                        "h_array_tail_zero",
                        settings,
                        hab[..., ls:],
                        np.zeros_like(hab[..., ls:]),
                        rtol=rtol,
                        atol=atol,
                    )
                    assert np.allclose(hab[..., ls:], 0, rtol=rtol, atol=atol)

                    if not passed:
                        diff = hab[...,:ls] - hgt
                        tol = atol + rtol * np.abs(hgt)
                        fail_mask = np.abs(diff) > tol
                        perrs = np.abs(diff[fail_mask] / hgt[fail_mask])
                        debug_var = 1

                    assert (
                        passed
                    ), f"fail @ {str(g12)} max(%err): {np.abs(perrs).max():.4e}"
    # end for r1 in r_list:
# end def test_calc_h_array():


def test_norm_basic(ao, eps):
    alpha = 1.0 # power of an s-type orbital
    gamma = 2*alpha # orbital squared doubles exponent
    # gamma = 1/(2*sigma^2) # relation of gamma to normal distro's sigma
    sigma = 1.0 / np.sqrt(2*gamma)
    n1d = np.pow(2*np.pi*sigma**2, -1/4) # sqrt again for half of the norm
    n3d = n1d**3

    ams = ao.arr([0,]*3, dtype=ao.int_dtype)
    r0 = ao.arr([0,]*3)
    aa = ao.arr([1,]*1)
    ans = CGP(ams, r0, aa, ao=ao).norm
    assert np.isclose(ans.item(), n3d, rtol=16*eps, atol=0)


def test_norm_vs_overlap(ao, eps):
    rtol, atol = 16*eps, 0.0
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    am1 = ao.arr([3, 1, 2], dtype=ao.int_dtype)
    am2 = ao.arr([4, 2, 2], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    rc = ao.arr([0.625, 0.125, 0.375])
    rd = ao.arr([4.0, 1.0, 2.0])
    am_list = [ams, am1, am2]
    r_list = [r0, rb, rc, rd]
    aa = ao.arr([0.25] * 1)

    for r1 in r_list:
        for a1 in am_list:
            g1 = CGP(a1, r1, aa, ao=ao)
            overlap = (g1*g1).calc_overlap()
            oovesq = (1 / ao.sqrt(overlap)).item()  # should never get div 0
            norm = g1.norm.item()
            passed = np.isclose(oovesq, norm, rtol=rtol, atol=atol)
            perr = (oovesq - norm) / norm
            assert passed, f"fail @ {str(g1)} %err: {perr:.4e}"
# end def test_norm_vs_overlap():


def test_eri_self(ao, eps):
    if "float32" in str(ao.default_dtype):
        rtol = 256 * eps
    else:
        rtol = 16 * eps
    atol = 0.0
    am0 = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    am1 = ao.arr([3, 1, 1], dtype=ao.int_dtype)
    am2 = ao.arr([4, 2, 2], dtype=ao.int_dtype)
    am3 = ao.arr([1, 3, 3], dtype=ao.int_dtype)
    ra = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, -0.125])
    am_list = [am0, am1, am2, am3]
    aa_list = [ao.arr([v]) for v in [0.5, 1.0, 2.0]]

    # test same center and exponent for a and b respectively
    ga_list = [CGP.create_contraction(am, ra, aa_list[0], ao=ao) for am in am_list]
    gb_list = [CGP.create_contraction(am, rb, aa_list[1], ao=ao) for am in am_list]
    ga, gb = sum(ga_list), sum(gb_list)
    gab = CGPP(ga, gb)
    assert ao.abs(gab.pab - gab.pab[0]).max().item() == 0, "cannot test eri_self with non-uniform pab"

    ans = gab.calc_eri_self()
    gt = gab.calc_eri(gab)
    imxdfiff = ao.xp.argmax(ao.abs(ans - gt)).item()
    amx, gmx = ans.flatten()[imxdfiff].item(), gt.flatten()[imxdfiff].item()
    perr = (amx - gmx) / gmx if gmx != 0.0 else amx - gmx
    passed = np.isclose(amx, gmx, rtol=rtol, atol=atol)
    assert passed, f"fail eri_self %err: {perr:.4e}"

    # test same center for both a and b, different exponents
    ga_list = [CGP.create_contraction(am, ra, aa, ao=ao) for am in am_list for aa in aa_list]
    gb_list = [CGP.create_contraction(am, ra, aa, ao=ao) for am in am_list[::-1] for aa in aa_list[1:]]
    ga, gb = sum(ga_list), sum(gb_list)
    gab = CGPP(ga, gb)
    assert ao.abs(gab.pab - gab.pab[0]).max().item() == 0, "cannot test eri_self with non-uniform pab"

    ans = gab.calc_eri_self()
    gt = gab.calc_eri(gab)
    imxdfiff = ao.xp.argmax(ao.abs(ans - gt)).item()
    amx, gmx = ans.flatten()[imxdfiff].item(), gt.flatten()[imxdfiff].item()
    perr = (amx - gmx) / gmx if gmx != 0.0 else amx - gmx
    passed = np.isclose(amx, gmx, rtol=rtol, atol=atol)
    assert passed, f"fail eri_self %err: {perr:.4e}"
#end def test_eri_self():
