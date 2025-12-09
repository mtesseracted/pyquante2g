from dataclasses import dataclass
import numpy as np
# from pyquante22.utils.math import binom_exact, binom2_exact, fboys
from pyquante22.utils.math import fboys
from pyquante22.ref_ints.utils import GTO, arange_max, binoms, binom2s
from pyquante22.ref_ints.utils import norm2, gaussian_product_center, binomial_prefactor


def h_term(ii, rr, l0, l1, a, b, g):
    return binom2s[ii,rr] * binomial_prefactor(ii,l0,l1,a,b,) * np.pow(4*g,rr-ii)

def H_array(l0, l1, a, b, g):
    lmax = l0 + l1
    h_array = np.zeros(lmax+1)
    for ia in range(lmax+1):
        for ir in range(ia//2 + 1):
            ii = ia - 2*ir
            h_array[ii] += h_term(ia, ir, l0, l1, a, b, g)
    return h_array

def c_term(l, m, u, hl, hm, px,):
    # d can be factored sin l+m-u == ii
    lm2u = l + m - 2*u
    pre = (-1)**(m+u) * px ** lm2u #* d ** -(l+m-u)
    return hl * hm * binom2s[l+m,u] * pre

def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):
    delta = 0.25 * (1/g1 + 1/g2)
    l01mx, l23mx = l0 + l1, l2 + l3
    c_array = np.zeros(l01mx+l23mx+1)
    dpows = delta ** -np.arange(c_array.shape[0])
    hls = H_array(l0, l1, px0, px1, g1)
    hms = H_array(l2, l3, qx2, qx3, g2)
    for il in range(l01mx+1): #
        for im in range(l23mx+1):
            umx = (il+im) // 2
            for iu in range(umx+1):
                ii = il + im - iu
                c_array[ii] += c_term(il, im, iu, hls[il], hms[im], qpx)
                
    c_array *= dpows
    return c_array
#end def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):


def eri_tho66(ga:GTO, gb:GTO, gc:GTO, gd:GTO) -> float:
    """ implements the version in THO66 Eq. 3.3 - 3.5 (with typo in equation 3.5 fixed)
    """
    la, ma, na = ga.l
    lb, mb, nb = gb.l
    lc, mc, nc = gc.l
    ld, md, nd = gd.l

    xa, ya, za = ga.r
    xb, yb, zb = gb.r
    xc, yc, zc = gc.r
    xd, yd, zd = gd.r

    alphaa = ga.a
    alphab = gb.a
    alphac = gc.a
    alphad = gd.a

    rab2 = norm2(ga.r - gb.r)
    rcd2 = norm2(gc.r - gd.r)
    xyzp = gaussian_product_center(alphaa, ga.r, alphab, gb.r)
    xp, yp, zp = xyzp
    xyzq = gaussian_product_center(alphac, gc.r, alphad, gd.r)
    xq, yq, zq = xyzq
    rpq2 = norm2(xyzp - xyzq)
    gamma1 = alphaa + alphab
    gamma2 = alphac + alphad
    delta = 0.25 * (1.0 / gamma1 + 1.0 / gamma2)

    Cx = C_array(la, lb, lc, ld, xp - xa, xp - xb, xq - xc, xq - xd, xq - xp, gamma1, gamma2)
    Cy = C_array(ma, mb, mc, md, yp - ya, yp - yb, yq - yc, yq - yd, yq - yp, gamma1, gamma2)
    Cz = C_array(na, nb, nc, nd, zp - za, zp - zb, zq - zc, zq - zd, zq - zp, gamma1, gamma2)

    imx = la + lb + lc + ld
    jmx = ma + mb + mc + md
    kmx = na + nb + nc + nd
    fbx = 0.25 * rpq2 / delta
    boys = np.asarray([fboys(ii, fbx) for ii in range(imx + jmx + kmx + 1)])

    total = 0.0
    for I in range(imx + 1):
        for J in range(jmx + 1):
            for K in range(kmx + 1):
                total += Cx[I] * Cy[J] * Cz[K] * boys[I + J + K]

    return (
        2 * np.pi ** 2.5
        / (gamma1 * gamma2 * np.sqrt(gamma1 + gamma2))
        * np.exp(-alphaa * alphab * rab2 / gamma1)
        * np.exp(-alphac * alphad * rcd2 / gamma2)
        * total
    )
#end def coulomb_rep4(

eri_funcs = [eri_tho66,]

if __name__ == "__main__":

    from pennylane.qchem.integrals import electron_repulsion as pleri
    def eri_pl(eri_args):
        arr_args = np.concat([np.atleast_1d(f) 
            for arg in eri_args for f in [*arg]]).reshape(4,-1)
        ls, rs, alphas = arr_args[:,:3], arr_args[:,3:6], arr_args[:,6]
        return pleri(*ls.astype(int), *rs, *alphas).item()


    ra = np.asarray([0.,]*3)
    rb = np.asarray([0.125,]*3)
    s0 = (0,)*3
    px = (1, 0, 0) 
    dx = (2, 0, 0) 
    a4 = (4, 0, 0) 
    gto1 = GTO(l=np.asarray(s0), r=ra, a=1.0)
    gto2 = GTO(l=np.asarray(px), r=rb, a=1.25)
    gto3 = GTO(l=np.asarray(dx), r=rb, a=1.5)
    gto4 = GTO(l=np.asarray(a4), r=rb, a=2.0)

    eri_args_list = [
        [gto1, gto1, gto1, gto1],
        [gto1, gto1, gto2, gto2],
        [gto1, gto1, gto3, gto3],
        [gto1, gto2, gto2, gto3],
        [gto1, gto4, gto3, gto4],
    ]

    for ii, eri_args in enumerate(eri_args_list):
        val = eri_tho66(*eri_args).item()
        val_pl = eri_pl(eri_args)
        diff = val - val_pl
        perr = diff / val if val != 0.0 else diff
        print(f"{ii}: {perr:.4e}")


    debug_var = 1