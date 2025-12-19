from dataclasses import dataclass, replace
import numpy as np
from pyquante22.utils.math import binom_exact, binom2_exact, gamma_half_int

@dataclass
class GTO:
    l: np.ndarray   # int, (3,) Angular momentum
    r: np.ndarray   # float, (3,) Center
    a: float        # float, Exponent

    def __getitem__(self, key): # allows indexing and upacking
        if key == 0:
            return self.l
        elif key == 1:
            return self.r
        elif key == 2:
            return self.a
        raise IndexError("GTO only has 3 fields")

    def clone(self, **kwargs):
        return replace(self, **kwargs)

    # below are all the possible return orders for intercompatabilty
    @property
    def lra(self): # used by pennylane(interleaved)@(overlap,kinetic,nai,eri)
        return self.l, self.r, self.a
    
    @property
    def lar(self):
        return self.l, self.a, self.r
    
    @property
    def alr(self): # used by pyquante2@(overlap,kinetic,nai),
        return self.a, self.l, self.r

    @property
    def arl(self): # used by: HH12(interleaved)@overlap,
        return self.a, self.r, self.l

    @property
    def ral(self):
        return self.r, self.a, self.l
    
    @property
    def rla(self): # used by: gbasis@all, pyquante2@eri,
        return self.r, self.l, self.a
#end class GTO:


amax = 20  # max exponent for precomputed factorials
arange_max = np.arange(6*amax+1) # 2 for pairs, 3 for 3d space
binoms = np.asarray([[binom_exact(n,k) for k in range(2*amax+1)] for n in range(2*amax+1)])
binom2s = np.asarray([[binom2_exact(n,k) for k in range(1*amax+1)] for n in range(2*amax+1)])
ghalfs = np.asarray(gamma_half_int(n) for n in arange_max[:2*amax+1])   

def norm2(a):
    return np.dot(a,a)

# def gaussian_product_center(alpha1,A,alpha2,B):
#     """ the center of the product of two cartesian gaussians """
#     return (alpha1*A+alpha2*B)/(alpha1+alpha2)

# def gaussian_product_center_fma(alpha1, A, alpha2, B):
#     p = alpha1 + alpha2
#     w = alpha2 / p
#     return np.fma(w, (B - A), A)  # A + w*(B-A) with one rounding

def gaussian_product_center(alpha1, A, alpha2, B):
    p = alpha1 + alpha2 # scale by p to keep ratios well-behaved
    a1, a2 = alpha1 / p, alpha2 / p
    return a1 * A + a2 * B


def binomial_prefactor(s,ia,ib,xpa,xpb):
    """ The integral prefactor containing the binomial coefficients
    """
    total= 0.0
    # for t in range(s+1):
    #     if s-ia <= t <= ib:
    for t in range(s-ia, ib+1):
        total +=  binoms[ia,s-t]*binoms[ib,t]* \
                    np.pow(xpa,ia-s+t)*np.pow(xpb,ib-t)
    return total







