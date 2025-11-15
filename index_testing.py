import os, time
from collections import defaultdict
from math import floor
import numpy as np
from collections import defaultdict
import math

rngmx = 9
time_loops = 20

def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n-1)
all_facts = np.asarray([factorial(i) for i in range(2*rngmx+2)])

def func1(ia, ra):
    k = ia - 2*ra
    # facts = factorial(ia) // (factorial(ra) * factorial(k))
    facts = all_facts[ia] // (all_facts[ra] * all_facts[k])
    return (ia + 0.2*np.sqrt(ra) + 1) * 3**k * facts

def func2(iab, u):
    k = iab - 2*u
    # facts = factorial(iab) // (factorial(u) * factorial(k))
    facts = all_facts[iab] // (all_facts[u] * all_facts[k])
    return 2**k * facts


for l01mx in range(rngmx+1):
    for l23mx in range(l01mx+1):

        umx = (l01mx+l23mx) // 2

        totals = np.zeros(l01mx+l23mx+1)
        t0 = time.time()
        for _ in range(time_loops):
            totals[:] = 0.0
            for ia in range(l01mx+1):
                for ra in range(int(np.floor(ia/2))+1):
                    aterm = func1(ia, ra)
                    # if aterm == 0.0: continue
                    for ib in range(l23mx+1):
                        for rb in range(int(np.floor(ib/2))+1):
                            bterm = func1(ib, rb)
                            # if bterm == 0.0: continue
                            for u in range(int(np.floor((ia+ib)/2-(ra+rb)))+1):
                                iia = ia - 2*ra
                                iib = ib - 2*rb
                                ii = iia + iib - u
                                # totals[ii] += aterm * bterm * func2(iia+iib, u)
                                totals[ii] += func1(ia,ra) * func1(ib,rb) * func2(iia+iib, u)
            #end for ia in range(l01mx+1):
        #end for _ in range(time_loops):
        t1 = time.time()


        # Precompute contributions for each iia from (ia, ra)
        A = np.zeros(l01mx + 1, dtype=float)
        B = np.zeros(l23mx + 1, dtype=float)
        totals2 = np.zeros(l01mx + l23mx + 1, dtype=float)

        t2 = time.time()
        for _ in range(time_loops):
            A[:] = 0.0
            for ia in range(l01mx + 1):
                max_ra = ia // 2
                for ra in range(max_ra + 1):
                    iia = ia - 2 * ra
                    A[iia] += func1(ia, ra)

            # Precompute contributions for each iib from (ib, rb)
            B[:] = 0.0
            for ib in range(l23mx + 1):
                max_rb = ib // 2
                for rb in range(max_rb + 1):
                    iib = ib - 2 * rb
                    B[iib] += func1(ib, rb)

            # Now only a triple sum over (iia, iib, u)
            totals2[:] = 0
            for iia in range(l01mx + 1):
                term_a = A[iia]
                # if term_a == 0.0: continue
                for iib in range(l23mx + 1):
                    term_ab = term_a * B[iib]
                    # if term_ab == 0.0: continue
                    s = iia + iib
                    umax = s // 2
                    for u in range(umax + 1):
                        ii = s - u
                        totals2[ii] += term_ab * func2(s, u)
            #end for iia in range(l01mx + 1):
        #end for _ in range(time_loops):
        t3 = time.time()

        told = (t1-t0)
        tnew = (t3-t2)
        trat = told/tnew
        pdiffs = (totals2 - totals) / totals
        pdmx = np.abs(pdiffs).max()
        if pdmx > 1.0e-14:
            print(f"::WARN:: % diff exceeded 1e-14")
        print(f"({l01mx},{l23mx}) :: {t1-t0:.4f}\t{t3-t2:.4f} (s)\t{trat:.2f}\t{pdmx:.2e}")

    #end for l23mx in range(l01mx+1):
#end for l01mx in range(rngmx+1):

debug_var = 1
