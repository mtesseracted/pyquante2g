# THO'66 formulation of ERI
try:
    from . import core
    from . import functions as f
except ImportError:
    import core # direct run file
    import functions as f


def binom_pre(s, ia, ib, xpa, xpb):
    total = core.arb(0)
    for t in core.arange(s-ia, ib+core.aone):
        total += core.binom(ia,s-t) * core.binom(ib,t) * \
            (xpa ** (ia-s+t)) * (xpb ** (ib-t))
    return total


def h_term(ii, rr, l0, l1, a, b, g):
    return f.binom2(ii,rr) * binom_pre(ii,l0,l1,a,b) * (4*g)**(rr-ii)

def H_array(l0, l1, a, b, g):
    lmax = l0 + l1
    h_array = [core.arb(0) for _ in core.arange(lmax+1)]
    for ia in core.arange(lmax+1):
        for ir in core.arange(int(core.nint(ia))//2+1):
            ii = int(ia - 2*ir)
            h_array[ii] += h_term(ia, ir, l0, l1, a, b, g)
    return h_array


def c_term(l, m, u, hl, hm, px):
    ret = f.binom2(l+m, u) * core.arb(-1) ** (m+u)
    ret *= hl * hm * px ** (l+m-2*u)
    return ret

def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):
    l01mx, l23mx = l0 + l1, l2 + l3
    c_array = [core.arb(0) for _ in core.arange(l01mx+l23mx+1)]
    hls = H_array(l0, l1, px0, px1, g1)
    hms = H_array(l2, l3, qx2, qx3, g2)
    for il in core.arange(l01mx+1): #
        for im in core.arange(l23mx+1):
            umx = (il+im) // 2
            for iu in range(umx+1):
                ii = il + im - iu
                c_array[ii] += c_term(il, im, iu, hls[il], hms[im], qpx)
                
    delta = core.arb(0.25)/g1 + core.arb(0.25)/g2
    for ii in core.arange(l01mx+l23mx+1):
        c_array[ii] *= delta ** core.arb(-ii)
    return c_array
#end def C_array()


def gto_prod_cen(aa, ra, ab, rb):
    return (aa*ra + ab*rb) / (aa+ab)


def coulomb_rep4(
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

    rab2 = f.norm2(xyza-xyzb)
    rcd2 = f.norm2(xyzc-xyzd)
    xyzp = gto_prod_cen(alphaa,xyza,alphab,xyzb)
    xp,yp,zp = xyzp
    xyzq = gto_prod_cen(alphac,xyzc,alphad,xyzd)
    xq,yq,zq = xyzq
    rpq2 = f.norm2(xyzp-xyzq)
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = core.arb(0.25)/gamma1 + core.arb(0.25)/gamma2

    Cx = C_array(la, lb, lc, ld, xp-xa, xp-xb, xq-xc, xq-xd, xq-xp, gamma1, gamma2)
    Cy = C_array(ma, mb, mc, md, yp-ya, yp-yb, yq-yc, yq-yd, yq-yp, gamma1, gamma2)
    Cz = C_array(na, nb, nc, nd, zp-za, zp-zb, zq-zc, zq-zd, zq-zp, gamma1, gamma2)

    imx = int(la+lb+lc+ld)
    jmx = int(ma+mb+mc+md)
    kmx = int(na+nb+nc+nd)
    fbx = core.arb(0.25) * rpq2  / delta
    boys = [f.fboys(ii, fbx) for ii in core.arange(imx+jmx+kmx+1)]

    total = core.arb(0)
    for I in range(imx+1):
        for J in range(jmx+1):
            for K in range(kmx+1):
                tmp = Cx[I]*Cy[J]*Cz[K]*boys[I+J+K]
                total += tmp

    return core.arb(2)*(core.pi**core.arb(2.5))/(gamma1*gamma2*core.sqrt(gamma1+gamma2)) \
           *core.exp(-alphaa*alphab*rab2/gamma1) \
           *core.exp(-alphac*alphad*rcd2/gamma2)*total
#end def coulomb_rep4(


if __name__ == "__main__":
    eri_fac = core.arb(2) * (core.pi**core.arb(2.5))
    ra0 = core.arr_init([0,]*3)
    am000 = core.arr_init([0,0,0])
    g0 = [core.aone, am000, ra0]
    # ans0 = coulomb_rep4(*sum([g0,]*4, []))
    # assert core.mpm.almosteq(ans0, core.arb(1/8)*pre)
    # g1 = [core.arb(2)**-core.arb(6/5), am000, ra0]
    # ans1 = coulomb_rep4(*sum([g1,]*4, []))
    # assert core.mpm.almosteq(ans1, core.arb(1/1)*pre)

    am100 = core.arr_init([2,0,0])
    # g0 = [core.aone, am100, ra0]
    # ans0 = coulomb_rep4(*sum([g0,]*4, []))
    # # assert core.mpm.almosteq(ans0, core.arb(1/8)*pre)
    g0 = [core.arb(2)**-core.arb(6/5), am000, ra0]
    g1 = [core.arb(2)**-core.arb(6/5), am100, ra0]
    g0 = [core.arb(0.5), am000, ra0]
    g1 = [core.arb(0.5), am100, ra0]
    # ans1 = coulomb_rep4(*g1, *g0, *g1, *g0)
    # assert core.mpm.almosteq(ans1, core.arb(1/1)*pre)

    # C_array's for various amoms with gamma1=gamma2=1 and all centers=0
    # s p d f g h j
    # 0 1 2 3 4 5 6
    # (ss|ss) : [1]
    # (ps|ss) : [0] # odd * even
    # (ds|ss) : [1/2, -1/4, 0]
    # (gs|ss) : [3/4, -3/4, 3/16, 0, 0]
    # (js|ss) : [15/8, -45/16, 45/32, -15/64, 0, 0, 0]
    # (ps|ps) : [0, 1/4, 0]
    # (ds|ds) : [1/4, -1/4, 3/16, 0, 0]
    # (fs|fs) : [0, 9/16, -9/16, 15/64, 0, 0, 0]
    # (pp|pp) : [1/4, -1/4, 3/16, 0, 0]

    eri_sfac = 1/core.sqrt(core.arb(2)) # for gamma1=gamma2=1 (alpha=0.5)

    g0 = [core.ahalf, am000, ra0]
    g1 = [core.ahalf, core.arr_init([2,0,0]), ra0]
    cx = core.arr_init([1/2, -1/4, 0])
    import numpy as np
    fbs = core.arr_init(1/(2*np.arange(len(cx))+1))
    gt = (cx.T * fbs)[0] * eri_fac * eri_sfac
    ans = coulomb_rep4(*g1, *g0, *g0, *g0)

    print("Cxs for (ns|ss)")
    cxs = []
    for ii in range(8):
        cx = C_array(ii, 0, 0, 0, *[0,]*5, 1, 1)
        cxf = [core.frac_approx(float(c)) for c in cx]
        cxstr = [f"{f.numerator}/{f.denominator}" for f in cxf]
        cxs.append(cxstr)
        print(f"{cxstr},")

    
    print("Cxs for (ns|ps)")
    cxp = []
    for ii in range(8):
        cx = C_array(ii, 0, 1, 0, *[0,]*5, 1, 1)
        cxf = [core.frac_approx(float(c)) for c in cx]
        cxstr = [f"{f.numerator}/{f.denominator}" for f in cxf]
        cxp.append(cxstr)
        print(f"{cxstr},")
 
    debug_var = 1

