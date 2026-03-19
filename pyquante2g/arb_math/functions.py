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

def binom22(i,r,u, round=True):
    """ i! / (r! u! (i-2(r+u))!) 
    if all inputs are int, the result should be int so round can correct errors
    proof: multiply out b(i,r)b(i-r,r)b(i-2r,u)b(i-2r-u,u) where b(n,k) is biomial
    and you get binom22() times r!u!, so clearly int.
    """
    if 2*(r+u) > i: return core.azero
    tmp = core.lgamma(i+core.aone)
    tmp -= core.lgamma(r+core.aone) + core.lgamma(u+core.aone) + \
        core.lgamma(i-2*(r+u)+core.aone)
    ans = core.exp(tmp)
    if round: ans = core.nint(ans)
    return ans


def gamma_half_int(n):
    """ compute Gamma(n + 1/2) /sqrt(pi) == (2n-1)!! / 2^n """
    return core.gamma(n + core.ahalf) / core.sqrt(core.pi)


# def binomial_prefactor(s,ia,ib,xpa,xpb):
#     """ The integral prefactor containing the binomial coefficients
#     """
#     total= 0
#     for t in core.arange(s-ia, ib+1):
#         total +=  core.binom(ia,s-t) * core.binoms(ib,t)* \
#                     (xpa **(ia-s+t)) * (xpb **(ib-t))
#     return total
# def gaussian_product_center(alpha1,A,alpha2,B):
#     return (alpha1*A+alpha2*B)/(alpha1+alpha2)


def gto_prod_cen(aa, ra, ab, rb):
    return (aa*ra + ab*rb) / (aa+ab)


def binom_pre(s, ia, ib, xpa, xpb):
    total = core.arb(0)
    for t in core.arange(s-ia, ib+core.aone):
        total += core.binom(ia,s-t) * core.binom(ib,t) * \
            (xpa ** (ia-s+t)) * (xpb ** (ib-t))
    return total


def fboys(m, x):
    """ boys function """
    m2 = m + core.ahalf
    if x == core.azero: # limiting case, switch to check < eps?
        return core.ahalf / m2
    return core.ahalf * core.gammainc(m2, 0, x, regularized=False) * x ** -m2



if __name__ == "__main__":
    debug_var = 1
