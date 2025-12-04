import os, time
import pyquante22.array_ops as arr
import pyquante22.utils.math as me # math exact
from typing import Type


# get an instance of ArrayOps
ao = arr.ArrayOps("torch", device="cpu", 
                  default_dtype="float64", int_dtype="int64")

one = ao.arr(1)
two = ao.arr(2)
zero = ao.arr(0)
pi = ao.arr(4) * ao.atan(one)

amax = 20
facts = ao.arr([me.fact_exact(a) for a in range(2*amax+1)])
binoms = ao.arr([[me.binom_exact(n,k) for k in range(2*amax+1)] for n in range(2*amax+1)])
binom2s = ao.arr([[me.binom2_exact(n,k) for k in range(1*amax+1)] for n in range(2*amax+1)])
ghalfs = ao.arr([me.gamma_half_int_exact(n) for n in range(2*amax+1)])


class CartGto():
    """ Cartesian GTO (contracted) 
    CODING: the product center is dependent on alpha, so you can't just 
    precompute P_AB with another CartGTO for all alphas. If K is the contraction
    length (assume they're equal), there are K^2 product centers and sets
    of binomial coefficients. Aha! This is the reason for horizontal recurrence
    relations! If you put all the amom into one and other is zero, the double
    sum over K becomes a single sum over 2K!
    """
    def __init__(self, amom, cen, alpha):
        self.amom = amom
        self.cen = cen
        self.alpha = alpha
        self.lmax = ao.amax(self.amom)

    def __len__(self):
        # do we do number of GTOs or contraction length?
        return len(self.amom)





class CartGtoShellPair():
    """ represents a pair ket of contracted cartesian GTOs, |ab) 
    CODING: the problem is we can calcaulate in batch, as shown in 
    CartGtoPair under gto_eri_stability.py, but if we instead imagine 
    a 'sparse' format where we only store the center 3-vec, amom 3-vec,
    exponents n-vec, and contraction coefficients n-vec, it's now ragged
    for different contraction lengths. Also! a shell is all amom's with
    the same total, so now it's (l+1)(l+2)/2 of those, also variable.
    """
    def __init__(self):
        pass
