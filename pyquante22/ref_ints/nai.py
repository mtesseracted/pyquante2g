import numpy as np
from itertools import product
from pyquante22.utils.math import fboys
from pyquante22.ref_ints.utils import GTO

# implement the pennylane (T. Helgaker and P. Tayler 1995)[HT95] nuclear attraction integral
# https://doi.org/10.1142/9789812832115_0001
# https://github.com/PennyLaneAI/pennylane/blob/c2b66bd228c054d02574ce5c51a765c0df3fe45c/pennylane/qchem/integrals.py


def expansion(la, lb, ra, rb, alpha, beta, t):
    """
    https://github.com/PennyLaneAI/pennylane/blob/c2b66bd228c054d02574ce5c51a765c0df3fe45c/pennylane/qchem/integrals.py#L168
    """
    p = alpha + beta
    q = np.asarray(alpha * beta / p)
    r = ra - rb

    if la == lb == t == 0:
        return np.exp(-q * r**2)

    if t < 0 or t > (la + lb):
        return 0.0

    if lb == 0:
        return (
            (1 / (2 * p)) * expansion(la - 1, lb, ra, rb, alpha, beta, t - 1)
            - (q * r / alpha) * expansion(la - 1, lb, ra, rb, alpha, beta, t)
            + (t + 1) * expansion(la - 1, lb, ra, rb, alpha, beta, t + 1)
        )

    return (
        (1 / (2 * p)) * expansion(la, lb - 1, ra, rb, alpha, beta, t - 1)
        + (q * r / beta) * expansion(la, lb - 1, ra, rb, alpha, beta, t)
        + (t + 1) * expansion(la, lb - 1, ra, rb, alpha, beta, t + 1)
    )


def _hermite_coulomb(t, u, v, n, p, dr):
    """
    https://github.com/PennyLaneAI/pennylane/blob/c2b66bd228c054d02574ce5c51a765c0df3fe45c/pennylane/qchem/integrals.py#L770
    """
    x, y, z = dr[0:3]
    T = p * (dr**2).sum(axis=0)
    r = 0

    if t == u == v == 0:
        return ((-2 * p) ** n) * fboys(n, T)

    if t == u == 0:
        if v > 1:
            r = r + (v - 1) * _hermite_coulomb(t, u, v - 2, n + 1, p, dr)
        r = r + z * _hermite_coulomb(t, u, v - 1, n + 1, p, dr)
        return r

    if t == 0:
        if u > 1:
            r = r + (u - 1) * _hermite_coulomb(t, u - 2, v, n + 1, p, dr)
        r = r + y * _hermite_coulomb(t, u - 1, v, n + 1, p, dr)
        return r

    if t > 1:
        r = r + (t - 1) * _hermite_coulomb(t - 2, u, v, n + 1, p, dr)
    r = r + x * _hermite_coulomb(t - 1, u, v, n + 1, p, dr)
    return r



def _nuclear_attraction(la, lb, ra, rb, alpha, beta, r):
    """
    https://github.com/PennyLaneAI/pennylane/blob/c2b66bd228c054d02574ce5c51a765c0df3fe45c/pennylane/qchem/integrals.py#L834
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb
    p = alpha + beta
    # rgp = (alpha * ra[:, np.newaxis, np.newaxis] + beta * rb[:, np.newaxis, np.newaxis]) / (
    #     alpha + beta)
    # dr = rgp - r[:, np.newaxis, np.newaxis]
    # turn off vectorization because fboys() doesn't work for arrays
    rgp = (alpha * ra[:] + beta * rb[:]) / (alpha + beta)
    dr = rgp - r[:]

    a = 0.0
    for t, u, v in product(*[range(l) for l in [l1 + l2 + 1, m1 + m2 + 1, n1 + n2 + 1]]):
        a = a + expansion(l1, l2, ra[0], rb[0], alpha, beta, t) * expansion(
            m1, m2, ra[1], rb[1], alpha, beta, u
        ) * expansion(n1, n2, ra[2], rb[2], alpha, beta, v) * _hermite_coulomb(t, u, v, 0, p, dr)
    a *= -2 * np.pi / p # negative assumes charges are opposite
    return a


def nai_ht95(ga:GTO, gb:GTO, R):
    return _nuclear_attraction(ga.l, gb.l, ga.r, gb.r, ga.a, gb.a, R)

nai_funcs = [nai_ht95,]


if __name__ == "__main__":
    s0 = np.asarray([0,]*3)
    ra = np.asarray([0.,]*3)
    g0 = GTO(s0, ra, 1.0)
    ans = nai_ht95(g0, g0, ra)
    assert np.isclose(ans, np.pi)
    g0 = GTO(s0, ra, 0.5)
    ans = nai_ht95(g0, g0, ra)
    assert np.isclose(ans, 2*np.pi)
    debug_var = 1