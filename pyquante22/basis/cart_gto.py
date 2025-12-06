# import os, time
from hashlib import sha256
import pyquante22.array_ops as arr
import pyquante22.utils.math as me # math exact
# from typing import Type


# get an instance of ArrayOps
ao = arr.ArrayOps("torch", device="cpu", 
                  default_dtype="float64", int_dtype="int64")

ArrType = ao.ArrayType
one = ao.arr(1)
two = ao.arr(2)
zero = ao.arr(0)
half = ao.arr(0.5)
pi = ao.arr(4) * ao.atan(one)
# _parity = ao.arr([1,-1]) # (-1)^t = _parity[t%2]

amax = 20
arange_max = ao.arange(6*amax+1) # 2 for pairs, 3 for 3d space
# facts = ao.arr([me.fact_exact(a) for a in range(2*amax+1)])
binoms = ao.arr([[me.binom_exact(n,k) for k in range(2*amax+1)] for n in range(2*amax+1)])
binom2s = ao.arr([[me.binom2_exact(n,k) for k in range(1*amax+1)] for n in range(2*amax+1)])
# ghalfs = ao.arr([me.gamma_half_int_exact(n) for n in range(2*amax+1)])
def dfact2m1o2(n):
    """ double factorial of 2n-1 over 2**n """
    return ao.exp(ao.lgamma(n + 0.5*one)) / ao.sqrt(pi)
ghalfs = dfact2m1o2(arange_max[:2*amax+1].to(ao.default_dtype))


def conv_full_batch(a, b):
    """ batch convolve a with b (full size output)
    a is (batch, n), b is (batch, m)
    output is (batch, n+m-1)
    """
    return ao.conv_transpose1d( # tranpose1d does "full" mode
        a[None,:,:], b[:,None,:], groups=a.shape[0]
    ).squeeze(0)


def sign_parity(t): # more efficient than pow(-1, t)?
    # alternative: p = [1,-1]; return p[t%2]
    return one - 2 * (t%2)

def fboys_ch(m, x): # stable @ x=0
    m2 = m + 0.5*one
    ret = 0.5 *  ao.pow(x, -m2) * ao.gamma_inc(m2, x)
    mask = x < ao.backend.xp.finfo(x.dtype).eps
    ret[:] = ao.where(mask, 0.5/m2, ret) # @TODO: optimize for broadcasting with m and x being different sizes
    # ret[mask.expand(-1,ret.shape[1])] = 0.5/m2.expand(ret.shape[0], -1)
    return ret


class CartGtoPrim():
    """ Cartesian GTO Primitive """
    def __init__(self, amom:ArrType, cen:ArrType, alpha:ArrType, nd:int=3):
        if nd != 3:
            raise NotImplementedError("Only 3d GTOs currently implemented")
        self.nd = nd # spatial dimension
        self.amom = amom # int (batch*nd,)
        self.cen = cen # float (batch*nd,)
        self.alpha = alpha # float (batch*nd,) currently just repeated
        self.lmax = self.amom.max()

    def __len__(self):
        return len(self.amom)

    def __mul__(self, other):
        return CartGtoPrimPair(self, other)

    def __add__(self, other):
        """ create a larger batch (overload __and__? ga & gb)"""
        return CartGtoPrim(
            ao.concat([self.amom, other.amom]),
            ao.concat([self.cen, other.cen]),
            ao.concat([self.alpha, other.alpha]),
        )

    def __repr__(self):
        ret = f"{type(self).__name__} (Lx,Ly,Lz) (Rx,Ry,Rz) Exp"
        for ii in range(len(self)//self.nd):
            i0, i1 = ii*self.nd, (ii+1)*self.nd
            ll = " ".join(f"{v.item():2d}" for v in self.amom[i0:i1])
            xx = " ".join(f"{v.item():12.4f}" for v in self.cen[i0:i1])
            ee = f"{self.alpha[i0].item():12.4e}"
            ret += f"\n{ll}  {xx}  {ee}"
        return ret

    def hash(self, digits=8):
        return sha256(repr(self).encode('utf-8')).hexdigest()[:digits]

    def __str__(self):
        if len(self) <= self.nd:
            ams = ",".join([f"{v}" for v in self.amom.tolist()])
            rxs = ",".join([f"{v:.2f}" for v in self.cen.tolist()])
            aas = ",".join([f"{v:.2e}" for v in self.alpha[::self.nd].tolist()])
        else:
            def summarize_pairs(keys, counts, fmt="{k}x{c}", limit=4):
                s = ",".join(fmt.format(k=k, c=c) for k, c in zip(keys[:limit], counts[:limit]))
                return s + ("..." if len(keys) > limit else "")
            # amom summary
            aas, acs = ao.unique(self.amom.reshape(-1, self.nd).sum(axis=1), return_counts=True)
            ams = summarize_pairs(ao.tolist(aas), ao.tolist(acs))
            # cen summary
            rus, rcs = ao.unique(self.cen.reshape(-1, self.nd), axis=0, return_counts=True)
            rxs = summarize_pairs(['A','B','C','D'], ao.tolist(rcs))
            if len(rus) == 1: 
                rxs = ",".join(f"{v:.2f}" for v in ao.tolist(rus[0]))
            # alpha summary
            aas = ao.unique(self.alpha[::self.nd])
            aas = ao.tolist(aas)[0] if len(aas) == 1 else \
                f"{ao.amin(self.alpha)}...{ao.amax(self.alpha)}"
        return f"GTO[{len(self)//self.nd}]_{{{ams}}}@({rxs};{aas})[{self.hash()}]"
    #end def __str__(self):
    
       
    def calc_binoms(self, pab:ArrType, offset=None) -> ArrType:
        """ compute the binomial expansion coefficients given a center
        offset: (3,) will change the amom for the calculation
        """
        if offset is not None:
            bsz = len(self)//len(offset) # error check?
            amom = self.amom + ao.tile(offset, bsz)
            lmax = amom.max()
        else:
            amom = self.amom
            lmax = self.lmax
        irng = arange_max[:lmax+1]
        pows = amom[:,None] - irng[None,:]
        pows[pows<0] = zero # @TODO optimize with relu
        ret = ao.pow((pab-self.cen)[:,None], pows) * \
            binoms[amom[:,None], irng[None,:]]
        mask = (amom < 0)[:,None] # zero out negative amom (for kinetic)
        ret[ao.broadcast_to(mask, (-1, ret.shape[1]))] = zero
        return ret
    #end def calc_binoms()

#end class CartGtoPrim():


CGP = CartGtoPrim
class CartGtoPrimPair():
    """ Pairs of CartGtoPrim for batch'd eval of integrals
    """
    def __init__(self, ga:CGP, gb:CGP):
        if len(ga) != len(gb):
            raise ValueError("::ERR::(CartGtoPrimPair) pairs must be same batch size")
        self.ga = ga
        self.gb = gb
        self.gamma = ga.alpha + gb.alpha            # 
        self.pab = self.calc_product_centers()      # combined center, P_{AB}
        self.binom_pres = self.calc_prod_binoms()   # binomial prefactors
        # self.hab = self.calc_h_array()

    def __len__(self):
        return len(self.ga)

    def __repr__(self):
        rga = "A: " + repr(self.ga)
        rgb = "B: " + repr(self.gb)
        return f"{rga}\n{rgb}"

    def __str__(self):
        pre = f"GTO[{len(self)//self.ga.nd}]_"
        sga = str(self.ga).replace(pre, "")
        sgb = str(self.gb).replace(pre, "")
        return f"PairGTO[{len(self)}]{sga}//{sgb}"

    def hash(self, digits=8):
        return sha256(repr(self).encode('utf-8')).hexdigest()[:digits]

    def calc_product_centers(self) -> ArrType:
        ga, gb = self.ga, self.gb
        return (ga.alpha*ga.cen + gb.alpha*gb.cen) / self.gamma

    
    def calc_prod_binoms(self) -> ArrType:
        """ calculate binomial coefficents for the product of GTOs
        """
        # treat xyz as batch
        self.abinoms = self.ga.calc_binoms(self.pab)
        self.bbinoms = self.gb.calc_binoms(self.pab)
        return conv_full_batch(self.abinoms, self.bbinoms)


    def calc_overlap1d(self, offset=None):
        if offset is not None:
            # @TODO optimization: only calculate the evens instead of slicing
            bbinoms = self.gb.calc_binoms(self.pab, offset=offset)
            bps = conv_full_batch(self.abinoms, bbinoms)[:,::2]
        else:
            bps = self.binom_pres[:,::2] # only take even factors
        irng = arange_max[:bps.shape[1]][None,:]
        totals = bps * ghalfs[irng] * ao.pow(self.gamma[:,None], -irng)
        return ao.sum(totals, axis=1)


    def calc_overlap(self, nd=3, offset=None):
        """ calculate the overlap of the pair
        nd: spatial dimensions flattened into the batch, may not work currently if nd != 3
        offset: optional ang. mom. offset, must be (nd,)
        return: overlaps of (len(self)//nd,)
        """
        if len(self)%nd != 0:
            raise ValueError("::ERR::(calc_overlap) batch size must be a multiple of 3")
        bsz = (len(self) // nd, nd) # batch size (shape tuple)
        ga, gb = self.ga, self.gb
        ret = ao.sqrt(pi/self.gamma) * \
            ao.exp(-ga.alpha*gb.alpha*ao.square(ga.cen-gb.cen)/self.gamma)
        ret = ret.view(bsz) # ss overlap term, split into spatial dims
        ret *= self.calc_overlap1d(offset=offset).view(bsz) # 1D overlaps
        return ao.prod(ret, axis=1)
    #end def calc_overlap(self, nd=3, offset=None):


    def calc_kinetic(self, nd=3):
        """ calculate the kinetic operator expectation for the pair (THO Eq. 2.14)
        nd: spatial dimensions flattened into the batch, may not work currently if nd != 3
        return: values of (len(self)//nd,)
        """
        # @TODO optimization:
        # the 'offset' feature in calc_overlap() is minimal extra alloc's,
        # but we're still recalculating the same values for the dimensions
        # that aren't being changed. We could either store the intermediates
        # of calc_overlap1d, or pass an offset list and calc the whole offset
        # list. Would need to return as a list or extra dimension, or we could
        # pass an additional set of weights for collapsing the sum...
        ga, gb = self.ga, self.gb
        bsz = (len(self)//nd, nd)
        amb = gb.amom.view(bsz)
        ab = gb.alpha[::nd]
        offsets = 2 * ao.eye(nd, dtype=ga.amom.dtype)

        term0 = ab * (2*ao.sum(amb, axis=1) + 3) * \
            self.calc_overlap()

        term1  = self.calc_overlap(offset=offsets[0])
        term1 += self.calc_overlap(offset=offsets[1])
        term1 += self.calc_overlap(offset=offsets[2])
        term1 *= -2 * ao.square(ab)

        l2 = gb.amom[0::nd]
        term2  = l2 * (l2-1) * self.calc_overlap(offset=-offsets[0])
        l2 = gb.amom[1::nd]
        term2 += l2 * (l2-1) * self.calc_overlap(offset=-offsets[1])
        l2 = gb.amom[2::nd]
        term2 += l2 * (l2-1) * self.calc_overlap(offset=-offsets[2])
        term2 *= -half

        return term0 + term1 + term2
    #end def calc_kinetic(gpair, nd=3):



def calc_kinetic2(gpair, nd=3):
    """ formulas are hard coded to nd=3 """
    ga, gb = gpair.ga, gpair.gb
    bsz = (len(ga)//nd, nd)
    amb = gb.amom.view(bsz)
    ab = gb.alpha[::nd]
    term0 = ab * (2*ao.sum(amb, axis=1) + 3) * \
        gpair.calc_overlap()

    offsets = 2 * ao.eye(nd, dtype=gpair.ga.amom.dtype)
    term1  = gpair.calc_overlap(offset=offsets[0])
    term1 += gpair.calc_overlap(offset=offsets[1])
    term1 += gpair.calc_overlap(offset=offsets[2])
    term1 *= -2 * ao.square(ab)

    l2 = gb.amom[0::nd]
    term2  = l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[0])
    l2 = gb.amom[1::nd]
    term2 += l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[1])
    l2 = gb.amom[2::nd]
    term2 += l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[2])
    term2 *= -half

    return term0 + term1 + term2


debug_var = 1
if __name__ == "__main__":
    n = 2
    ams = ao.arr([0,0,0], dtype=ao.int_dtype)
    amx = ao.arr([-2,0,0], dtype=ao.int_dtype)
    amp = ao.arr([1,0,0], dtype=ao.int_dtype)
    amn = ao.arr([n,0,0], dtype=ao.int_dtype)
    r0 =  ao.arr([0,0,0])
    rb =  ao.arr([1,2,0.5])
    aa =  ao.arr([1.0,]*3)
    gs = CartGtoPrim(ams, r0, aa)
    gx = CartGtoPrim(amx, r0, aa)
    gsb = CartGtoPrim(ams, rb, aa)
    gp = CartGtoPrim(amp, r0, aa)
    gn = CartGtoPrim(amn, r0, aa)

    mys = str(gx)

    gns = CartGtoPrimPair(gn, gs)
    tmp = gns.calc_overlap()


    gxs = CartGtoPrimPair(gx, gs)
    tmp = gxs.calc_overlap()

    gss = CartGtoPrimPair(gs, gs)
    gssb = CartGtoPrimPair(gs, gsb)
    gps = CartGtoPrimPair(gp, gs)

    ssgt = ao.pow(pi/2, 3/2)
    tmp = gss.calc_overlap()

    # tk = calc_kinetic(gss)

    import numpy as np
    # from pyquante22.basis.pq2_one import overlap1d as p2ov1
    from pyquante22.basis.pq2_one import overlap as p2ov
    from pyquante22.basis.one_ints_pyquante2 import gto_overlap_3d as ov3d
    from pennylane.qchem.integrals import gaussian_overlap as plov
    tol = 5.e-15

    nr0 = np.asarray([0.,]*3)
    nrx = np.asarray([0.,1.,2.])
    nry = np.asarray([-0.1,1.2,1.75])
    na0 = np.asarray([0,0,0])
    na2 = np.asarray([2,0,0])
    nax = np.asarray([2,3,5])
    nay = np.asarray([4,1,3])

    a = p2ov(1.0, na0, nr0, 1.0, na0, nr0)
    b = ov3d(1.0, 1.0, nr0, nr0, na0, na0)
    assert np.abs((a-b)/a) < tol

    a = p2ov(1.0, na2, nr0, 1.0, na0, nr0)
    b = ov3d(1.0, 1.0, nr0, nr0, na2, na0)
    assert np.abs((a-b)/a) < tol

    a = p2ov(1.0, na2, nrx, 2.0, na0, nry)
    b = ov3d(1.0, 2.0, nrx, nry, na2, na0)
    assert np.abs((a-b)/a) < tol

    a = p2ov(1.0, nax, nrx, 2.0, nay, nry) # 
    b = ov3d(1.0, 2.0, nrx, nry, nax, nay)
    c = plov(nax, nay, nrx, nry, 1.0, 2.0)

    arx = ao.arr(nrx)
    ary = ao.arr(nry)
    aax = ao.arr(nax, dtype=ao.int_dtype)
    aay = ao.arr(nay, dtype=ao.int_dtype)
    gx = CartGtoPrim(aax, arx, ao.arr([1.0,]*3))
    gy = CartGtoPrim(aay, ary, ao.arr([2.0,]*3))
    gxy = CartGtoPrimPair(gx, gy)
    gxx = CartGtoPrimPair(gx, gx)
    d = gxy.calc_overlap().item()
    assert np.abs((a-b)/a) < tol
    assert np.abs((a-c)/a) < tol
    assert np.abs((a-d)/a) < tol

    from pyquante22.basis.pq2_one import kinetic as p2t
    from pyquante22.basis.one_ints_pyquante2 import gto_kinetic_3d as t3d
    from pennylane.qchem.integrals import gaussian_kinetic as plt

    a = p2t(1.0, nax, nrx, 2.0, nay, nry) # 
    b = t3d(1.0, 2.0, nrx, nry, nax, nay)
    c = plt(nax, nay, nrx, nry, 1.0, 2.0)
    d = gxy.calc_kinetic().item()
    assert np.abs((a-b)/a) < tol
    assert np.abs((a-c)/a) < tol
    assert np.abs((a-d)/a) < tol

    # e = calc_kinetic2(gxy).item()

    # @TODO: batch into more than 1 3d primitive and try
    # add tests
    ga = gx + gy
    gb = gx + gx
    gaa = ga * ga
    gab = ga * gb
    tmp = CartGtoPrimPair(ga, ga)
    sga = str(ga)


    debug_var = 1
