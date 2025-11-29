import torch as th

dtype = th.float64
exp = th.exp
fone = th.as_tensor(1, dtype=dtype)


def binom_ch(n, i, round=True):
    # binom(n,i) = exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
    # binom(n,k) == 0.0 for k > n
    ret = exp(th.lgamma(n+fone) - (th.lgamma(i+fone) + th.lgamma(n-i+fone)))
    if round: ret = th.round(ret)
    return ret.to(dtype)
    
def binom2_ch(n, i, round=True):
    ret = exp(th.lgamma(n+fone) - (th.lgamma(i+fone) + th.lgamma(n-2*i+fone)))
    if round: ret = th.round(ret)
    return ret.to(dtype)

debug_var = 1
