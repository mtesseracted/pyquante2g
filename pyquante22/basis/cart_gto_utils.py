import numpy as np
from itertools import permutations


am2sym = ['S', 'P', 'D']
ABCs = [chr(ord('A') + ii) for ii in range(26)]
am2sym += [l for l in ABCs[ABCs.index('F') :] if l not in (am2sym + ['J'])]
sym2am = dict(zip(am2sym, range(len(am2sym))))


def cartPow2fullSym(am):
    if len(am) != 3:
        raise ValueError("Only valid for 3D momentums")
    j = sum(am)
    ret = am2sym[j].lower() if j < len(am2sym) else '@'
    ret += ''.join(
        [
            (c + ('' if a == 1 else str(a)))
            for c, a in zip(['x', 'y', 'z'], am)
            if a != 0
        ]
    )
    return ret


# pow2fullsym = { # deprecated, call cartPow2fullSym
#     (0,0,0): 's',
#     (1,0,0): 'px',
#     (0,1,0): 'py',
#     (0,0,1): 'pz',
#     (2,0,0): 'dx2',
#     (0,2,0): 'dy2',
#     (0,0,2): 'dz2',
#     (1,1,0): 'dxy',
#     (0,1,1): 'dyz',
#     (1,0,1): 'dxz',
# }


def find_sum_triplets(n):
    """find the uniq triplets that sum to n (0's valid)"""

    def find_sums_betw(m, n):
        # doublets betw m and n inclusive that sum to n
        lo = n // 2
        a = np.arange(max(n - m, 0), lo + 1)
        return np.stack([n - a, a]).T

    nn = int(np.floor((n**2 + 6 * n + 12) / 12))  # triplet count
    ret = np.full((nn, 3), -1)

    ii, lo = 0, int(np.ceil(n / 3))
    for ihi in range(n, lo - 1, -1):
        b = find_sums_betw(ihi, n - ihi)
        ret[ii : ii + len(b), 0] = ihi
        ret[ii : ii + len(b), 1:3] = b
        ii += len(b)

    return ret


# end def find_sum_triplets(n):


def gen_cartesian_momentums(n):
    trips = find_sum_triplets(n)
    nn = ((n + 1) * (n + 2)) // 2
    ret = np.full((nn, 3), -1)

    ii = 0
    for t in trips:
        ts = sorted(set(permutations(t)), reverse=True)
        ret[ii : ii + len(ts), :] = ts
        ii += len(ts)

    return ret


# end def find_cartesian_momentums(n):
