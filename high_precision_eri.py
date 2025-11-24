import sys
import mpmath as mpm
from fractions import Fraction
import math
from typing import Tuple, Optional, Union, Iterable, List
import torch as th

th.set_default_dtype(th.float64)

mpm.mp.dps = 40 # decimal precision, rougly double float64
# mpm.mp.prec = 53 # float64 mantissa
meps = mpm.mpf(10)**mpm.mpf(-mpm.mp.dps)
Number = Union[float, int]

def frac_approx(n:Number, eps=1e-12):
    return Fraction(n).limit_denominator(int(1.0/eps))

def num2arb(n, eps=10*sys.float_info.epsilon):
    """ convert a number to arbitrary precision """
    if isinstance(n, (float, int)):
        n = frac_approx(n, eps=eps)
    return mpm.mpf(n.numerator) / mpm.mpf(n.denominator)

arb = lambda x: num2arb(x)

gt = arb(0.2)
pred = mpm.mpf(0.2)
pdiff = (pred-gt)/gt

azero = arb(0)
aone = arb(1)
ahalf = arb(0.5)
pi = 4 * mpm.atan(aone)

def norm2_arb(x):
    """ x should be an mpmath.matrix column matrix """
    return (x.T * x)[0]


def binom_arb(n, i):
    return mpm.binomial(n, i)

def binom2_arb(n, i, round=True):
    ans = mpm.exp(mpm.loggamma(n+aone) - mpm.loggamma(i+aone) - mpm.loggamma(n-2*i+aone))
    if round: ans = mpm.nint(ans)
    return ans

def fboys_arb(m, x):
    m2 = m + ahalf
    if x == azero:
        return ahalf / m2
    return ahalf * mpm.gammainc(m2, 0, x, regularized=False) * x ** -m2

def arange(*args):
    return range(*[int(mpm.nint(a)) for a in args])


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
        for ir in arange(int(mpm.nint(ia))//2+1):
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

    return arb(2)*(pi**arb(2.5))/(gamma1*gamma2*mpm.sqrt(gamma1+gamma2)) \
           *mpm.exp(-alphaa*alphab*rab2/gamma1) \
           *mpm.exp(-alphac*alphad*rcd2/gamma2)*total
#end def coulomb_rep4(


# tmp = H_array_arb(arb(2), arb(2), arb(1), arb(2), arb(2))
# tm2 = H_array_arb(arb(1), arb(1), arb(1), arb(1), arb(2))
debug_var = 1


if __name__ == "__main__":


    arr_init = lambda x: mpm.matrix([arb(xx) for xx in x])
    ra0 = arr_init([0,]*3)
    ra1 = arr_init([1,]*3)
    ra2 = arr_init([0,1,2])
    ra3 = arr_init([0.5,1.5,2.5])
    am000 = arr_init([0,0,0])
    am012 = arr_init([0,1,2])
    am123 = arr_init([1,2,3])
    am222 = arr_init([1,2,3])
    am333 = arr_init([1,2,3])
    am454 = arr_init([4,5,5])
    am100 = arr_init([1,0,0])
    am200 = arr_init([2,0,0])
    am300 = arr_init([3,0,0])
    am400 = arr_init([4,0,0])
    am500 = arr_init([5,0,0])
    am600 = arr_init([6,0,0])
    am700 = arr_init([7,0,0])

    gto_raw_data = [
        (am000, ra0, aone),
        (am100, ra0, aone),
        (am200, ra0, aone),
        (am300, ra0, aone),
        (am400, ra0, aone),
        (am500, ra0, aone),
        (am600, ra0, aone),
        (am700, ra0, aone),
        # (am000, ra1, arb(0.5)),
        # (am012, ra3, aone),
        # (am123, ra0, arb(1.0)),
        # (am222, ra0, arb(1.0)),
        # (am333, ra0, arb(1.0)),
        # (am500, ra0, arb(1.0)),
        # (am600, ra0, arb(1.0)),
        # (am700, ra0, arb(1.0)),
    ]
    # test_data_stacks = list(zip(*gto_raw_data))

    arb_eris = []
    for ii in range(len(gto_raw_data)):
        ama, ra, aa = gto_raw_data[ii]
        tmp = coulomb_rep4_arb(
            aa, ama, ra, aa, ama, ra, 
            aa, ama, ra, aa, ama, ra, 
        )
        arb_eris.append(tmp)

    pt000 = 4.373354581906216
    pt_eris = [
        4.373354581906216,
        0.223223306784796,
        0.116421448586877,
        0.174303015654784,
        0.518826506672436,
        2.572626579561370,
        19.146158300106190,
        199.622241091182843,
    ]
    print("="*40 + " PyTorch ERI % Errs " + "="*40)
    for ii in range(len(gto_raw_data)):
        ama, ra, aa = gto_raw_data[ii]
        lmn = ','.join([str(int(i)) for i in list(ama)])
        pt = mpm.mpf(pt_eris[ii])
        gt = arb_eris[ii]
        pdiff = float((pt - gt) / gt)
        print(f"({lmn}) %: {pdiff:.4e}")


    pq2_eris = [
        4.373354581906216,
        0.223223304680411,
        0.116421444773967,
        0.174302996519949,
        0.518826262216948,
        2.572622774697928,
        19.146353262665023,
        199.648849774657776,
    ]
    print("="*40 + " PyQuante2 ERI % Errs " + "="*40)
    for ii in range(len(gto_raw_data)):
        ama, ra, aa = gto_raw_data[ii]
        lmn = ','.join([str(int(i)) for i in list(ama)])
        pt = mpm.mpf(pq2_eris[ii])
        gt = arb_eris[ii]
        pdiff = float((pt - gt) / gt)
        print(f"({lmn}) %: {pdiff:.4e}")








    debug_var = 1



    


    

    