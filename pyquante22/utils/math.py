# Math functions using python's native math.
# Use python's arbitrary int math to do factorial, binomials, etc.

def fact_exact(n:int, dn=1): # use python's arbitrary int
    """ n! (or n!! if dn=2) """
    if n < 2:
        return 1
    return n * fact_exact(n-dn, dn=dn)

def binom_exact(n:int, k:int): # use python's arbitrary int
    """ n! / (k! (n-k)!)"""
    if k > n:
        return 0
    return fact_exact(n) // (fact_exact(k) * fact_exact(n-k))

def binom2_exact(n:int, k:int): # use python's arbitrary int
    """ n! / (k! (n-2k)!)"""
    if 2*k > n:
        return 0
    return fact_exact(n) // (fact_exact(k) * fact_exact(n-2*k))

def gamma_half_int_exact(n): # use python's arbitrary int
    """ (2n-1)!! / 2**n = Gamma(n+1/2) / sqrt(pi) """
    raise NotImplementedError()
    # previously named dfact2m1o2
    # return fact_exact(2*n-1,dn=2) // (2**n) # @DEBUG: not an integer!
    # switch to float div or use divmod for more precision?
    # Gamma[2+1/2] / Sqrt[Pi] -  ((2*2-1)!! / 2^2)

debug_var = 1

# import torch as th

# dtype = th.float64
# exp = th.exp
# fone = th.as_tensor(1, dtype=dtype)


# def binom_ch(n, i, round=True):
#     # binom(n,i) = exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
#     # binom(n,k) == 0.0 for k > n
#     ret = exp(th.lgamma(n+fone) - (th.lgamma(i+fone) + th.lgamma(n-i+fone)))
#     if round: ret = th.round(ret)
#     return ret.to(dtype)
    
# def binom2_ch(n, i, round=True):
#     ret = exp(th.lgamma(n+fone) - (th.lgamma(i+fone) + th.lgamma(n-2*i+fone)))
#     if round: ret = th.round(ret)
#     return ret.to(dtype)


