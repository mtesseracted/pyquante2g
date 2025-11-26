# from .core import nint, binomial, core.loggamma, gammainc, exp
# from .core import azero, core.ahalf, core.aone, pi
# from core import core.aone, core.ahalf, azero, pi
# import arb_math.core as core
from . import core  # NOT "from .arb_math import pi"


def norm2_arb(x):
    """ x should be an mpmath.matrix column matrix """
    return (x.T * x)[0]


def binom_arb(n, i):
    return core.binomial(n, i)

def binom2_arb(n, i, round=True):
    tmp = core.loggamma(n+core.aone)
    tmp -= (core.loggamma(i+core.aone) + core.loggamma(n-2*i+core.aone))
    ans = core.exp(tmp)
    if round: ans = core.nint(ans)
    return ans

def fboys_arb(m, x):
    m2 = m + core.ahalf
    if x == core.azero:
        return core.ahalf / m2
    return core.ahalf * core.gammainc(m2, 0, x, regularized=False) * x ** -m2



if __name__ == "__main__":
    # # 256 digits of pi from `bc -l <<< "scale=256; 4*a(1)"`
    # pi256  = "3.141592653589793238462643383279502884197169399375105820974944592307"
    # pi256 += "81640628620899862803482534211706798214808651328230664709384460955058"
    # pi256 += "22317253594081284811174502841027019385211055596446229489549303819644"
    # pi256 += "288109756659334461284756482337867831652712019091456484"


    # for dps in [10, 20, 80, 128]:
    #     set_dps(dps)
    #     pigt = mpm.mpf(pi256)
    #     err = pi - pigt
    #     print(f"{dps:4d}\t{float(err):.4e}")
    debug_var = 1
