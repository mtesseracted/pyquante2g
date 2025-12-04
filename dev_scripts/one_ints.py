import os, time
import pyquante22.array_ops as arr
import pyquante22.utils.math as me # math exact


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


def overlap1d_ch2(l0s, l1s, px0s, px1s, gammas):
    """ assume same number of batches for 0s and 1s, and we need to calc the overlaps
    between them, so return is the same len as the ls, pdiffs, and gammas
    (PySEC pdist2 logic style for the distance func)
    """

    typ, dev = gammas.dtype, gammas.device
    bipres = binomial_prefactors_ch_all(l0s, l1s, px0s, px1s)
    irng = ao.arange(one * bipres.shape[1], device=dev)
    totals = bipres * dfact2m1o2(irng)[None,:] * ao.pow(gammas[:,None], -irng[None,:])
    return ao.sum(totals, axis=1)




def overlap_ch(alpha0s, lmn0s, r0s, alpha1s, lmn1s, r1s):
    """
    """
    # lmx1, lmx2 = lmn0s.max(), lmn1s.max()
    # lmx = ch.max(lmx1, lmx2)
    # lrng = ch.arange(lmx+1, device=device, dtype=dtype)
    # binom_coeffs = binom_ch(lmx, lrng) # 

    r12sqs = ao.pow(ao.norm(r0s-r1s, axis=1), two)
    gammas = alpha0s + alpha1s
    pcens = (alpha0s[:,None]*r0s + alpha1s[:,None]*r1s) / gammas[:,None]
    pdiff0s = pcens - r0s
    pdiff1s = pcens - r1s
    ret = ao.pow(pi/gammas, ao.arr(3/2)) * ao.exp(-(alpha0s*alpha1s)*r12sqs/gammas)
    ret *= overlap1d_ch2(lmn0s[:,0], lmn1s[:,0], pdiff0s[:,0], pdiff1s[:,0], gammas)
    ret *= overlap1d_ch2(lmn0s[:,1], lmn1s[:,1], pdiff0s[:,1], pdiff1s[:,1], gammas)
    ret *= overlap1d_ch2(lmn0s[:,2], lmn1s[:,2], pdiff0s[:,2], pdiff1s[:,2], gammas)
    return ret




debug_var = 1
