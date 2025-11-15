import time 
import numpy as np
import torch as th


def pow_safe(b, e):
    if np.abs(b) < np.finfo(np.float32).eps:
        return 0.0
    else:
        return np.pow(b, e)

def norm2(x, axis=None):
    return (x*x).sum(axis=axis)

def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n-1)

def binom(n, i): # slow but correct
    ni, ii = int(n), int(i)
    return factorial(n) // (factorial(ii) * factorial(ni-ii))

def fact_ratio2(a,b): 
    return factorial(a)// (factorial(b) * factorial(a-2*b))

def binomial_prefactor(s,ia,ib,xpa,xpb):
    """
    The integral prefactor containing the binomial coefficients from Augspurger and Dykstra.
    >>> binomial_prefactor(0,0,0,0,0)
    1
    """
    total= 0
    for t in range(s+1):
        if s-ia <= t <= ib:
            # total +=  binomial(ia,s-t)*binomial(ib,t)* \
            total +=  binom(ia,s-t)*binom(ib,t)* \
                     np.pow(xpa,ia-s+t)*np.pow(xpb,ib-t)
    return total



def binomial_prefactors_all(l0, l1, x0, x1):
    lmax = l0 + l1
    lrng = np.arange(lmax+1)
    totals = np.zeros(lmax+1)
    ss = np.arange(lmax + 1)
    binom0s = np.asarray([binom(l0, l) for l in lrng[:l0+1]])
    binom1s = np.asarray([binom(l1, l) for l in lrng[:l1+1]])
    pow0s = x0 ** lrng[:l0+1]
    pow1s = x1 ** lrng[:l1+1]
    for s in ss:
        lo, hi = max(0, s-l0), min(s, l1)
        ts = lrng[lo:hi+1]
        totals[s] = np.sum(
            binom0s[s-ts] * binom1s[ts] * \
            pow0s[l0-s+ts] * pow1s[l1-ts]
        )
    return totals
 

def gaussian_product_center(alpha1,A,alpha2,B):
    """ the center of the product of two cartesian gaussians """
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)


def B0(i,r,g): return fact_ratio2(i,r)*pow(4*g,r-i)
def fB(i,l1,l2,P,A,B,r,g): return binomial_prefactor(i,l1,l2,P-A,P-B)*B0(i,r,g)


def gamma_inc(a, x):
    p = th.igamma(a, x) # regularized
    ga = th.exp(th.lgamma(a))
    return ga * p

def fboys(m, x):
    mt, xt = th.as_tensor(m+0.5), th.as_tensor(x)
    if xt > th.finfo(th.float32).eps:
        xm = xt.clamp(th.finfo(th.float64).eps, None) 
        #  this formulation doesn't have correct limiting behavior at x=0, so clamp
        return (0.5 *  th.pow(xm, -mt) * gamma_inc(mt, xm)).item()
    else:
        return (0.5/mt).item()


def B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,Px,Ax,Bx,Qx,Cx,Dx,gamma1,gamma2,delta):
    "THO eq. 2.22"
    val= fB(i1,l1,l2,Px,Ax,Bx,r1,gamma1) \
           *pow(-1,i2)*fB(i2,l3,l4,Qx,Cx,Dx,r2,gamma2) \
           *pow(-1,u)*fact_ratio2(i1+i2-2*(r1+r2),u) \
           *pow(Qx-Px,i1+i2-2*(r1+r2)-2*u) \
           /pow(delta,i1+i2-2*(r1+r2)-u)
    return val


def B_array(l1,l2,l3,l4,p,a,b,q,c,d,g1,g2,delta):
    Imax = l1+l2+l3+l4+1
    B = [0]*Imax
    for i1 in range(l1+l2+1):
        for i2 in range(l3+l4+1):
            for r1 in range(i1//2+1):
                for r2 in range(i2//2+1):
                    for u in range((i1+i2)//2-r1-r2+1):
                        I = i1+i2-2*(r1+r2)-u
                        B[I] = B[I] + B_term(i1,i2,r1,r2,u,l1,l2,l3,l4,
                                             p,a,b,q,c,d,g1,g2,delta)
    return B
 

def coulomb_rep_wrap( # wrapper for no norm
    alphaa, lmna, xyza,
    alphab, lmnb, xyzb,
    alphac, lmnc, xyzc,
    alphad, lmnd, xyzd,):
    norm = 1.0
    return coulomb_repulsion(
        xyza,norm,lmna,alphaa,
        xyzb,norm,lmnb,alphab,
        xyzc,norm,lmnc,alphac,
        xyzd,norm,lmnd,alphad)

def coulomb_repulsion(xyza,norma,lmna,alphaa,
                      xyzb,normb,lmnb,alphab,
                      xyzc,normc,lmnc,alphac,
                      xyzd,normd,lmnd,alphad):
    """
    Return the coulomb repulsion between four primitive gaussians a,b,c,d with the given origin
,    x,y,z, normalization constants norm, angular momena l,m,n, and exponent alpha.
    >>> p1 = array((0.,0.,0.),'d')
    >>> p2 = array((0.,0.,1.),'d')
    >>> lmn = (0,0,0)
    >>> isclose(coulomb_repulsion(p1,1.,lmn,1.,p1,1.,lmn,1.,p1,1.,lmn,1.,p1,1.,lmn,1.),4.373355)
    True
    >>> isclose(coulomb_repulsion(p1,1.,lmn,1.,p1,1.,lmn,1.,p2,1.,lmn,1.,p2,1.,lmn,1.),3.266127)
    True
    >>> isclose(coulomb_repulsion(p1,1.,lmn,1.,p2,1.,lmn,1.,p1,1.,lmn,1.,p2,1.,lmn,1.),1.6088672)
    True
    """
    la,ma,na = lmna
    lb,mb,nb = lmnb
    lc,mc,nc = lmnc
    ld,md,nd = lmnd
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd

    rab2 = norm2(xyza-xyzb)
    rcd2 = norm2(xyzc-xyzd)
    xyzp = gaussian_product_center(alphaa,xyza,alphab,xyzb)
    xp,yp,zp = xyzp
    xyzq = gaussian_product_center(alphac,xyzc,alphad,xyzd)
    xq,yq,zq = xyzq
    rpq2 = norm2(xyzp-xyzq)
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = 0.25*(1/gamma1+1/gamma2)

    Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)

    sum = 0.
    for I in range(la+lb+lc+ld+1):
        for J in range(ma+mb+mc+md+1):
            for K in range(na+nb+nc+nd+1):
                tmp = Bx[I]*By[J]*Bz[K]*fboys(I+J+K,0.25*rpq2/delta)
                if not np.isfinite(tmp):
                    debug_var = 1
                sum += tmp

    return 2*pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
           *np.exp(-alphaa*alphab*rab2/gamma1) \
           *np.exp(-alphac*alphad*rcd2/gamma2)*sum*norma*normb*normc*normd
#end def coulomb_repulsion(xyza,norma,lmna,alphaa,


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

    rab2 = norm2(xyza-xyzb)
    rcd2 = norm2(xyzc-xyzd)
    xyzp = gaussian_product_center(alphaa,xyza,alphab,xyzb)
    xp,yp,zp = xyzp
    xyzq = gaussian_product_center(alphac,xyzc,alphad,xyzd)
    xq,yq,zq = xyzq
    rpq2 = norm2(xyzp-xyzq)
    gamma1 = alphaa+alphab
    gamma2 = alphac+alphad
    delta = 0.25*(1/gamma1+1/gamma2)

    Cx = C_array(la, lb, lc, ld, xp-xa, xp-xb, xq-xc, xq-xd, xq-xp, gamma1, gamma2)
    Cy = C_array(ma, mb, mc, md, yp-ya, yp-yb, yq-yc, yq-yd, yq-yp, gamma1, gamma2)
    Cz = C_array(na, nb, nc, nd, zp-za, zp-zb, zq-zc, zq-zd, zq-zp, gamma1, gamma2)

    # # @DEBUG
    # Bx = B_array(la,lb,lc,ld,xp,xa,xb,xq,xc,xd,gamma1,gamma2,delta)
    # By = B_array(ma,mb,mc,md,yp,ya,yb,yq,yc,yd,gamma1,gamma2,delta)
    # Bz = B_array(na,nb,nc,nd,zp,za,zb,zq,zc,zd,gamma1,gamma2,delta)
    # mxdiff = np.abs(np.concat([Cx-Bx,Cy-By,Cz-Bz])).max()
    # if mxdiff > 1.0e-15:
    #     raise ValueError()

    imx = la+lb+lc+ld
    jmx = ma+mb+mc+md
    kmx = na+nb+nc+nd
    fbx = 0.25 * rpq2  / delta
    boys = np.asarray([fboys(ii, fbx) for ii in range(imx+jmx+kmx+1)])

    total = 0.0
    for I in range(la+lb+lc+ld+1):
        for J in range(ma+mb+mc+md+1):
            for K in range(na+nb+nc+nd+1):
                tmp = Cx[I]*Cy[J]*Cz[K]*boys[I+J+K]
                # if not np.isfinite(tmp):
                #     raise ValueError()
                total += tmp

    return 2*pow(np.pi,2.5)/(gamma1*gamma2*np.sqrt(gamma1+gamma2)) \
           *np.exp(-alphaa*alphab*rab2/gamma1) \
           *np.exp(-alphac*alphad*rcd2/gamma2)*total
#end def coulomb_rep4(


def h_term(ii, rr, l0, l1, a, b, g):
    facts = factorial(ii) // (factorial(rr) * factorial(ii-2*rr))
    return facts * binomial_prefactor(ii,l0,l1,a,b,) * np.pow(4*g,rr-ii)

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
    facts = factorial(l+m) // (factorial(u) * factorial(lm2u))
    pre = (-1)**(m+u) * px ** lm2u #* d ** -(l+m-u)
    return hl * hm * facts * pre

def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):
    delta = 0.25 * (1/g1 + 1/g2)
    l01mx, l23mx = l0 + l1, l2 + l3
    c_array = np.zeros(l01mx+l23mx+1)
    dpows = delta ** -np.arange(c_array.shape[0])
    hls = H_array(l0, l1, px0, px1, g1)
    hms = H_array(l2, l3, qx2, qx3, g2)
    for il in range(l01mx+1): # wrong!, indices are over l01mx
        for im in range(l23mx+1):
            umx = (il+im) // 2
            for iu in range(umx+1):
                ii = il + im - iu
                c_array[ii] += c_term(il, im, iu, hls[il], hms[im], qpx)
                
    c_array *= dpows
    return c_array
#end def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):



if __name__ == "__main__":

    arr_init = lambda x, dtype=float: np.asarray(x, dtype=dtype)
    s0 = arr_init([0,]*3, dtype=np.int32)
    px = arr_init([1, 0, 0], dtype=np.int32)
    dxy = arr_init([1, 1, 0], dtype=np.int32)
    fxyz = arr_init([1, 1, 1], dtype=np.int32)
    am222 = arr_init([2,]*3, dtype=np.int32)
    am121 = arr_init([1, 2, 1], dtype=np.int32)
    am131 = arr_init([1, 3, 1], dtype=np.int32)
    am444 = arr_init([4, 4, 4], dtype=np.int32)
    am333 = arr_init([3, 3, 3], dtype=np.int32)
    am500 = arr_init([5, 0, 0], dtype=np.int32)
    ra = arr_init([0,]*3)
    rb = arr_init([0.1,]*3)
    fone = 1.0

    test_data_list = [
        (fone, s0, ra, fone, s0, ra),
        (fone, s0, ra, fone, s0, rb),
        (fone, s0, ra, 2*fone, fxyz, ra),
        (fone, fxyz, ra, 2*fone, s0, rb),
        (fone, fxyz, ra, 2*fone, px, rb),
        (fone, fxyz, ra, 2*fone, fxyz, rb),
        (fone, am121, ra, 2*fone, am333, rb),
        (fone, am131, ra, 2*fone, am131, rb),
        (fone, am131, ra, 2*fone, s0, rb),
        (fone, am222, ra, 2*fone, am333, rb),
        (fone, am444, ra, 2*fone, am333, rb),
        (fone, am500, ra, 2*fone, am333, rb),
    ]

    test_data_stacks = list(zip(*test_data_list))

    alpha0s = np.stack(test_data_stacks[0])
    lmn0s = np.stack(test_data_stacks[1])
    r0s = np.stack(test_data_stacks[2])

    alpha1s = np.stack(test_data_stacks[3])
    lmn1s = np.stack(test_data_stacks[4])
    r1s = np.stack(test_data_stacks[5])

    time_loops = 10

    for idx in range(1, len(test_data_list)):
        t0 = time.time()
        for _ in range(time_loops):
            pred = coulomb_rep4(*test_data_list[idx], *test_data_list[idx])
        t1 = time.time()
        for _ in range(time_loops):
            gt = coulomb_rep_wrap(*test_data_list[idx], *test_data_list[idx])
        t2 = time.time()
        # print(f"({idx:02d}) :: {gt:12.8e}  {pred:12.8e}")

        ta = t1 - t0
        tb = t2 - t1
        tr = tb / ta
        pdiff = (pred - gt) / gt
        ll, mm = test_data_list[idx][1].sum(), test_data_list[idx][4].sum()
        print(f"[{ll},{mm}]\t{ta:.4f}\t{tb:.4f}\t{tr:.2f}\t{pdiff:.4e}")

    # l0, l1, a, b = 3, 2, 1.0, 2.0
    # gt = np.asarray([binomial_prefactor(s, l0, l1, a, b) for s in range(l0+l1+1)])


    debug_var = 1

    # @ end of day on 09Nov25, after finding typos in the H formulas
    # from THO Eq. 3.5, which is a simpler form of the B_terms pyquante2 uses
    # I haven't been able to find the form of H_array / C_array that 
    # matches B_array. At this point I believe I need to download another
    # program like psi4 and calculate some ERIs that way to make sure 
    # the pyquante2 implementation is correct


debug_var = 1