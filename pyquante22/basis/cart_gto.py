# import os, time
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
    def __init__(self, amom:ArrType, cen:ArrType, alpha:ArrType):
        self.amom = amom # int (3,)
        self.cen = cen # float (3,)
        self.alpha = alpha # float
        self.lmax = self.amom.max()

    def __len__(self):
        return len(self.amom)

    def calc_product_centers(self, other) -> ArrType:
        gamma = self.alpha + other.alpha
        return (self.alpha*self.cen + other.alpha*other.cen) / gamma
 
    def calc_binoms(self, pab:ArrType) -> ArrType:
        """ precompute the binomial expansion coefficients given a new center
        """
        self.lmax = self.amom.max() # recompute for safety (amom changed in kinetic)
        irng = arange_max[:self.lmax+1]
        pows = self.amom[:,None] - irng[None,:]
        pows[pows<0] = zero # @TODO optimize with relu
        ret = ao.pow((pab-self.cen)[:,None], pows) * \
            binoms[self.amom[:,None], irng[None,:]]
        mask = (self.amom < 0)[:,None] # zero out negative amom (for kinetic)
        ret[ao.broadcast_to(mask, (-1, ret.shape[1]))] = zero
        # ret[:] = ao.where(
        return ret

    def change_amom(self, delta, d, nd=3):
        am = self.amom.clone()
        am[d::nd] += delta
        return CartGtoPrim(am, self.cen, self.alpha)

#end class CartGtoPrim():


CGP = CartGtoPrim
class CartGtoPrimPair():
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

    def calc_product_centers(self) -> ArrType:
        ga, gb = self.ga, self.gb
        return (ga.alpha*ga.cen + gb.alpha*gb.cen) / self.gamma
    
    def calc_prod_binoms(self) -> ArrType:
        """ calculate binomial coefficents for the product of GTOs
        """
        # treat xyz as batch
        abinoms = self.ga.calc_binoms(self.pab)
        bbinoms = self.gb.calc_binoms(self.pab)
        ret = ao.conv_transpose1d( # tranpose1d does "full" mode
            abinoms[None,:,:], bbinoms[:,None,:],
            groups=abinoms.shape[0]
        ).squeeze(0)
        return ret

    def calc_overlap1d(self):
        bps = self.binom_pres[:,::2] # only take even factors
        irng = arange_max[:bps.shape[1]][None,:]
        totals = bps * ghalfs[irng] * ao.pow(self.gamma[:,None], -irng)
        return ao.sum(totals, axis=1)

    def calc_overlap(self, nd=3):
        if len(self)%nd != 0:
            raise ValueError("::ERR::(calc_overlap) batch size must be a multiple of 3")
        bsz = (len(self) // nd, nd) # batch size (shape tuple)
        ga, gb = self.ga, self.gb
        ret = ao.sqrt(pi/self.gamma) * \
            ao.exp(-ga.alpha*gb.alpha*ao.square(ga.cen-gb.cen)/self.gamma)
        ret = ret.view(bsz) # ss overlap term, split into spatial dims
        ret *= self.calc_overlap1d().view(bsz) # 1D overlaps
        return ao.prod(ret, axis=1)

    def calc_kinetic(self, nd=3):
        # to use the THO Eqn. 2.14 we need overlaps with amom vals < 0 to
        # = 0. The current implementation returns them as s-type (amom=0)
        # the pyquante2 implemenation only works up to -2, and error for < -2
        # hard-coded formula for nd=3
        bsz = (len(self) // nd, nd) # batch size (shape tuple)


        pass

#end class CartGtoPrimPair

def calc_kinetic(gpair, nd=3):
    """ formulas are hard coded to nd=3 """
    ga, gb = gpair.ga, gpair.gb
    bsz = (len(ga)//nd, nd)
    amb = gb.amom.view(bsz)
    ab = gb.alpha[::nd]
    term0 = ab * (2*ao.sum(amb, axis=1) + 3) * \
        gpair.calc_overlap()

    amb_orig = gb.amom.clone()
    # clone the gb.amom so it doesn't mess up other references (possibly in gpair.ga)
    gb_orig = gb
    gpair.gb = CartGtoPrim(gb.amom.clone(), gb.cen, gb.alpha) # clone amom to new ref
    gb = gpair.gb

    gb.amom[:] += ao.tile(ao.arr([2,0,0], dtype=gb.amom.dtype), bsz[0])
    gpair.binom_pres = gpair.calc_prod_binoms()
    term1 = gpair.calc_overlap()
    gb.amom[:] += ao.tile(ao.arr([-2,2,0], dtype=gb.amom.dtype), bsz[0])
    gpair.binom_pres = gpair.calc_prod_binoms()
    term1 += gpair.calc_overlap()
    gb.amom[:] += ao.tile(ao.arr([0,-2,2], dtype=gb.amom.dtype), bsz[0])
    gpair.binom_pres = gpair.calc_prod_binoms()
    term1 += gpair.calc_overlap()
    term1 *= -2 * ao.pow(ab, 2)

    gb.amom[:] += ao.tile(ao.arr([-2,0,-2], dtype=gb.amom.dtype), bsz[0])
    l2 = gb.amom[0::nd]+2
    gpair.binom_pres = gpair.calc_prod_binoms()
    term2 = l2 * (l2-1) * gpair.calc_overlap()
    gb.amom[:] += ao.tile(ao.arr([2,-2,0], dtype=gb.amom.dtype), bsz[0])
    l2 = gb.amom[1::nd]+2
    gpair.binom_pres = gpair.calc_prod_binoms()
    term2 += l2 * (l2-1) * gpair.calc_overlap()
    gb.amom[:] += ao.tile(ao.arr([0,2,-2], dtype=gb.amom.dtype), bsz[0])
    l2 = gb.amom[2::nd]+2
    gpair.binom_pres = gpair.calc_prod_binoms()
    term2 += l2 * (l2-1) * gpair.calc_overlap()
    term2 *= -half

    # rest gpair memory ref
    gpair.gb = gb_orig
    return term0 + term1 + term2
#end def calc_kinetic(gpair, nd=3):

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
    d = calc_kinetic(gxy).item()
    assert np.abs((a-b)/a) < tol
    assert np.abs((a-c)/a) < tol
    assert np.abs((a-d)/a) < tol

    debug_var = 1
