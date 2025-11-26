import numpy as np
import sympy as sp
from itertools import combinations_with_replacement

def pairs(n):
    pass


def fact(n, d=1):
    if n < 2:
        return 1
    return n * fact(n - d, d=d)

def binom(n, k):
    assert k <= n, "binom(n,k) only for k<=n"
    return fact(n) // (fact(k) * fact(n-k))


def binom_prefac(s, ia, ib, xa, xb): 
    # numerical impelmentation
    ret = 0
    lo, hi = max(0, s-ia), min(s, ib)
    for t in range(lo, hi+1):
        ret += binom(ia, s-t) * binom(ib, t) * \
            xa ** (ia-s+t) * xb ** (ib-t)
    return ret

s, ia, ib, t = sp.symbols('s ia ib t', integer=True)
xa, xb = sp.symbols('xa xb')

bp_sum = sp.Sum(
    sp.binomial(ia, s-t) * sp.binomial(ib, t) * \
    xa**(ia-s+t) * xb**(ib-t),
    (t, sp.Max(0, s-ia), sp.Min(s, ib))
)

def binom_prefac_form(sn, ian, ibn):
    return sp.simplify(bp_sum.subs({s:sn, ia:ian, ib:ibn}).doit())


amax = 2
for ii in range(amax+1): # ia
    for jj in range(ii+1): # ib
        for ss in range(ii+jj+1): # s
            bpf_sij = binom_prefac_form(ss, ii, jj)
            print(f"({ii},{jj})[{ss}] :: {bpf_sij}")
        print('-'*20)

debug_var = 1

