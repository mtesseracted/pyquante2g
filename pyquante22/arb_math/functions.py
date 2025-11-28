# expose or implement more math functions
try:
    from . import core
except ImportError:
    import core # direct run file


def norm2(x):
    """ x should be an mpmath.matrix column matrix """
    return (x.T * x)[0]


def binom2(n, i, round=True):
    """ n! / (i! (n-2i)!) ; if 2i>n return 0, same as binom(n,n+1) """
    if n - 2*i < core.azero: return core.azero
    if i == core.azero: return core.aone
    tmp = core.lgamma(n+core.aone)
    tmp -= (core.lgamma(i+core.aone) + core.lgamma(n-2*i+core.aone))
    ans = core.exp(tmp)
    if round: ans = core.nint(ans)
    return ans


def fboys(m, x):
    """ boys function """
    m2 = m + core.ahalf
    if x == core.azero: # limiting case, switch to check < eps?
        return core.ahalf / m2
    return core.ahalf * core.gammainc(m2, 0, x, regularized=False) * x ** -m2



if __name__ == "__main__":
    debug_var = 1
