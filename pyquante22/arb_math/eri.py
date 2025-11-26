from .core import arb, arange, nint, sqrt, exp
from .core import azero, ahalf, aone, pi
from .functions import norm2_arb, binom_arb, binom2_arb, fboys_arb


def binom_pre_arb(s, ia, ib, xpa, xpb):
    total = arb(0)
    for t in arange(s-ia, ib+aone):
        total += binom_arb(ia,s-t) * binom_arb(ib,t) * \
            (xpa ** (ia-s+t)) * (xpb ** (ib-t))
    return total


def h_term_arb(ii, rr, l0, l1, a, b, g):
    return binom2_arb(ii,rr) * binom_pre_arb(ii,l0,l1,a,b) * (4*g)**(rr-ii)

def H_array_arb(l0, l1, a, b, g):
    lmax = l0 + l1
    h_array = [arb(0) for _ in arange(lmax+1)]
    for ia in arange(lmax+1):
        for ir in arange(int(nint(ia))//2+1):
            ii = int(ia - 2*ir)
            h_array[ii] += h_term_arb(ia, ir, l0, l1, a, b, g)
    return h_array

def c_term(l, m, u, hl, hm, px):
    ret = binom2_arb(l+m, u) * arb(-1) ** (m+u)
    ret *= hl * hm * px ** (l+m-2*u)
    return ret

def C_array_arb(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):
    l01mx, l23mx = l0 + l1, l2 + l3
    c_array = [arb(0) for _ in arange(l01mx+l23mx+1)]
    hls = H_array_arb(l0, l1, px0, px1, g1)
    hms = H_array_arb(l2, l3, qx2, qx3, g2)
    for il in arange(l01mx+1): #
        for im in arange(l23mx+1):
            umx = (il+im) // 2
            for iu in range(umx+1):
                ii = il + im - iu
                c_array[ii] += c_term(il, im, iu, hls[il], hms[im], qpx)
                
    delta = arb(0.25)/g1 + arb(0.25)/g2
    for ii in arange(l01mx+l23mx+1):
        c_array[ii] *= delta ** arb(-ii)
    return c_array
#end def C_array_arb()


def gto_prod_cen_arb(aa, ra, ab, rb):
    return (aa*ra + ab*rb) / (aa+ab)


def coulomb_rep4_arb(
    alphaa, lmna, xyza,
    alphab, lmnb, xyzb,
    alphac, lmnc, xyzc,
    alphad, lmnd, xyzd,):
    """ implements the version in THO Eq. 3.3 - 3.5 (with typo in equation 3.5 fixed)
    """

    la,ma,na = lmna
    lb,mb,nb = lmnb
    lc,mc,nc = lmnc
    ld,md,nd = lmnd
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd

    rab2 = norm2_arb(xyza-xyzb)
    rcd2 = norm2_arb(xyzc-xyzd)
    xyzp = gto_prod_cen_arb(alphaa,xyza,alphab,xyzb)
    xp,yp,zp = xyzp
    xyzq = gto_prod_cen_arb(alphac,xyzc,alphad,xyzd)
    xq,yq,zq = xyzq
    rpq2 = norm2_arb(xyzp-xyzq)
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = arb(0.25)/gamma1 + arb(0.25)/gamma2

    Cx = C_array_arb(la, lb, lc, ld, xp-xa, xp-xb, xq-xc, xq-xd, xq-xp, gamma1, gamma2)
    Cy = C_array_arb(ma, mb, mc, md, yp-ya, yp-yb, yq-yc, yq-yd, yq-yp, gamma1, gamma2)
    Cz = C_array_arb(na, nb, nc, nd, zp-za, zp-zb, zq-zc, zq-zd, zq-zp, gamma1, gamma2)

    # # @DEBUG
    # Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    # By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    # Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
    # mxdiff = np.abs(np.concat([Cx-Bx,Cy-By,Cz-Bz])).max()
    # if mxdiff > 1.0e-15:
    #     raise ValueError()

    imx = int(la+lb+lc+ld)
    jmx = int(ma+mb+mc+md)
    kmx = int(na+nb+nc+nd)
    fbx = arb(0.25) * rpq2  / delta
    # boys = np.asarray([fboys(ii, fbx) for ii in range(imx+jmx+kmx+1)])
    boys = [fboys_arb(ii, fbx) for ii in arange(imx+jmx+kmx+1)]

    total = arb(0)
    for I in range(imx+1):
        for J in range(jmx+1):
            for K in range(kmx+1):
                tmp = Cx[I]*Cy[J]*Cz[K]*boys[I+J+K]
                total += tmp

    return arb(2)*(pi**arb(2.5))/(gamma1*gamma2*sqrt(gamma1+gamma2)) \
           *exp(-alphaa*alphab*rab2/gamma1) \
           *exp(-alphac*alphad*rcd2/gamma2)*total
#end def coulomb_rep4(


# tmp = H_array_arb(arb(2), arb(2), arb(1), arb(2), arb(2))
# tm2 = H_array_arb(arb(1), arb(1), arb(1), arb(1), arb(2))
debug_var = 1

if __name__ == "__main__":

    debug_var = 1

