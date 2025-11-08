import os, time
import numpy as np
import torch as ch
import torch.nn.functional as F

dtype=ch.float64
device='cpu'

# funcs
pow = ch.pow
exp = ch.exp
floor = ch.floor
arr_init = lambda x, dtype=dtype, device=device: ch.as_tensor(x, dtype=dtype, device=device)
arr_sum = lambda x, axis=None: ch.sum(x, dim=axis)
arange = ch.arange
# constants
pi = arr_init(ch.pi)
sqpi = ch.sqrt(pi)
fone = arr_init(1.0)
ione = arr_init(1, dtype=ch.int32)
fzero = arr_init(0.0)
izero = arr_init(0, dtype=ch.int32)


def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n-1)

def fact2(n):
    if n < 2:
        return ione
    return n * fact2(n-2)

def norm2(x, axis=None):
    return (x*x).sum(dim=axis)

def binom(n, i): # slow but correct
    ni, ii = int(n), int(i)
    return factorial(n) // (factorial(ii) * factorial(ni-ii))

def binom_ch(n, i):
    # binom(n,i) = exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
    return exp(ch.lgamma(n+ione) - ch.lgamma(i + 1) - ch.lgamma(n-i+ione))

def coeffs_ch(n, m, a, b, device=device, dtype=device):
    # Build A[i] = C(n,i) a^(n-i), i=0..n
    i = ch.arange(n + 1, device=device, dtype=dtype)
    A = binom_ch(n, i) * (ch.tensor(a, dtype=dtype, device=device) ** (n - i))
    # Build B[j] = C(m,j) b^(m-j), j=0..m
    j = ch.arange(m + 1, device=device, dtype=dtype)
    B = binom_ch(m, j) * (ch.tensor(b, dtype=dtype, device=device) ** (m - j))

    # Full polynomial convolution via conv1d (which computes cross-correlation),
    # so flip the kernel and pad by len(B)-1.
    A_ = A.view(1, 1, -1)
    K  = ch.flip(B, dims=[0]).view(1, 1, -1)
    pad = (B.numel() - 1, B.numel() - 1)  # left,right padding for "full"
    # conv1d with symmetric padding to emulate full conv; take center slice
    # Easier: pad only left = len(B)-1 to get length n+m+1 directly
    out = F.conv1d(A_, K, padding=B.numel() - 1)
    # Slice to exact full length
    full_len = n + m + 1
    C = out.view(-1)[0:full_len]
    return C  # shape (n+m+1,)


def coeff_of_xj_ch(n, m, a, b, j, device=device, dtype=dtype):
    if not (0 <= j <= n + m):
        return ch.tensor(0, dtype=dtype, device=device)
    C = coeffs_ch(n, m, a, b, device=device, dtype=dtype)
    return C[j]


def gaussian_product_center(alpha1,A,alpha2,B):
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)

def overlap(alpha1,lmn1,A,alpha2,lmn2,B):
    """
    Full form of the overlap integral. Taken from THO eq. 2.12
    """
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    rab2 = norm2(A-B)
    gamma = alpha1+alpha2
    P = gaussian_product_center(alpha1,A,alpha2,B)

    pre = pow(pi/gamma,1.5)*exp(-alpha1*alpha2*rab2/gamma)

    wx = overlap1d(l1,l2,P[0]-A[0],P[0]-B[0],gamma)
    wy = overlap1d(m1,m2,P[1]-A[1],P[1]-B[1],gamma)
    wz = overlap1d(n1,n2,P[2]-A[2],P[2]-B[2],gamma)
    return pre*wx*wy*wz

def overlap1d(l1,l2,PAx,PBx,gamma):
    """ """
    total = 0
    for i in ch.arange(1+int(floor(0.5*(l1+l2))), device=device):
        total += binomial_prefactor(2*i,l1,l2,PAx,PBx)* \
                 fact2(2*i-1)/pow(2*gamma,i)
    return total

def binomial_prefactor(s,ia,ib,xpa,xpb):
    """ """
    total= 0
    # @TODO: optimization: only loop range(s-ia, ib+1) instead of conditional
    for t in ch.arange(s+1, device=device):
        if s-ia <= t <= ib:
            # total +=  binom_ch(ia,s-t)*binom_ch(ib,t)* \
            total +=  binom(ia,s-t)*binom(ib,t)* \
                     pow(xpa,ia-s+t)*pow(xpb,ib-t)
    return total


def overlap_ch(alpha0s, lmn0s, r0s, alpha1s, lmn1s, r1s):
    """
    """
    # lmx1, lmx2 = lmn0s.max(), lmn1s.max()
    # lmx = ch.max(lmx1, lmx2)
    # lrng = ch.arange(lmx+1, device=device, dtype=dtype)
    # binom_coeffs = binom_ch(lmx, lrng) # 

    r12sqs = norm2(r0s-r1s, axis=1)
    gammas = alpha0s + alpha1s
    pcens = (alpha0s[:,None]*r0s + alpha1s[:,None]*r1s) / gammas[:,None]
    pdiff0s = pcens - r0s
    pdiff1s = pcens - r1s
    ret = pow(pi/gammas, 3/2) * exp(-(alpha0s*alpha1s)*r12sqs/gammas)
    ret *= overlap1d_ch2(lmn0s[:,0], lmn1s[:,0], pdiff0s[:,0], pdiff1s[:,0], gammas)
    ret *= overlap1d_ch2(lmn0s[:,1], lmn1s[:,1], pdiff0s[:,1], pdiff1s[:,1], gammas)
    ret *= overlap1d_ch2(lmn0s[:,2], lmn1s[:,2], pdiff0s[:,2], pdiff1s[:,2], gammas)
    return ret


def dfact2m1o2(n):
    """ double factorial of 2n-1 over 2**n """
    return ch.exp(ch.lgamma(n + 0.5*fone)) / sqpi

def overlap1d_ch2(l0s, l1s, px0s, px1s, gammas):
    """ assume same numbers of 0s and 1s, and we need to calc the overlaps
    between them, so return is the same len as the ls, pdiffs, and gammas
    (PySEC pdist2 logic style for the distance func)
    """

    typ, dev = gammas.dtype, gammas.device
    bipres = binomial_prefactors_ch_all(l0s, l1s, px0s, px1s)
    irng = ch.arange(ione * bipres.shape[1], device=dev)
    totals = bipres * dfact2m1o2(irng)[None,:] * pow(gammas[:,None], -irng[None,:])
    return arr_sum(totals, axis=1)



def overlap1d_ch(l0s, l1s, px0s, px1s, gammas):
    """ assume same numbers of 0s and 1s, and we need to calc the overlaps
    between them, so return is the same len as the ls, pdiffs, and gammas
    (PySEC pdist2 logic style for the distance func)
    """

    typ, dev = gammas.dtype, gammas.device
    # l0mx, l1mx = l0s.max(), l1s.max()
    # lmx = ch.max(l0mx, l1mx)
    summx = (l0s + l1s).max()
    ilim = int(floor(0.5 * summx)) * ione # cast to tensor
    # irng = ch.arange(ilim + 1, device=l0s.device)
    # binom_coeffs = binom_ch(2*ilim, irng)
    # xp0s = pdiff0s ** (l0mx - irng[:l0mx]) # this is a matrix... could decrease by just doing uniques but still 2d, then the convolution will make larger again, too much

    totals = ch.zeros_like(gammas)
    for i in ch.arange(1+ilim, device=dev):
        tmp = binomial_prefactors_ch(2*i, l0s, l1s, px0s, px1s)
        totals += tmp * dfact2m1o2(i) * pow(gammas, -i)
        # totals += tmp * fact2(2*i-1) / pow(2*gammas, i)
    return totals


def binomial_prefactors_ch(ll, l0s, l1s, px0s, px1s, opt=False):
    krng = ch.arange(ll+1, device=ll.device)
    mask = ((ll-l0s)[:,None] <= krng[None,:]) & (krng[None,:] <= l1s[:,None])

    # # subset to only the necessary k's in k-range
    if opt:
        kmsk = mask.any(dim=0)
        krng = krng[kmsk]
        mask = ((ll-l0s)[:,None] <= krng[None,:]) & (krng[None,:] <= l1s[:,None])

    tmps = binom_ch(l0s[:,None], (ll-krng)[None,:]) * \
           binom_ch(l1s[:,None], krng[None,:]) * \
           pow(px0s[:,None], (l0s[:,None] - ll[None,None] + krng[None,:])) * \
           pow(px1s[:,None], (l1s[:,None] - krng[None,:]))
    
    tmps[~mask] = fzero
    ret = tmps.sum(dim=1)
    return ret


def binomial_prefactors_ch_all(l0s, l1s, px0s, px1s):
    """
    Vectorized over all ll in range(ilim+1),
    where ilim = floor(0.5 * max(l0s + l1s)).

    Inputs are batched 1D tensors (same length B):
      l0s, l1s : integer tensors [B]
      px0s, px1s : float tensors [B]

    Returns:
      ret : [B, ilim] with ret[:, ll] = binomial_prefactors_ch(ll, ...)
    """
    device = l0s.device
    summx = (l0s + l1s).max()
    ilim = int(ch.floor(0.5 * summx).item())

    # Axes: L for ll (0..ilim), K for k (0..ilim)
    L = 2 * ch.arange(ilim + 1, device=device)        # [L]
    K = ch.arange(ilim + 1, device=device)        # [K]
    LL = L.view(1, -1, 1)                         # [1, L, 1]
    KK = K.view(1, 1, -1)                         # [1, 1, K]

    # Broadcast batch to [B, 1, 1] then to [B, L, K]
    l0 = l0s.view(-1, 1, 1)                       # [B,1,1]
    l1 = l1s.view(-1, 1, 1)                       # [B,1,1]
    px0 = px0s.view(-1, 1, 1)                     # [B,1,1]
    px1 = px1s.view(-1, 1, 1)                     # [B,1,1]

    # Valid region: 0 <= k <= ll, (ll - l0) <= k <= l1
    mask = (KK <= LL) & ((LL - l0) <= KK) & (KK <= l1)   # [B,L,K]

    # Terms (broadcast to [B,L,K])
    term = (
        binom_ch(l0, LL - KK) *
        binom_ch(l1, KK) *
        (px0 ** (l0 - LL + KK)) *
        (px1 ** (l1 - KK))
    )

    # Zero out invalid (avoid any spurious values from binom/powers off-domain)
    term = ch.where(mask, term, ch.zeros_like(term))

    # Sum over k → output [B, L]
    ret = term.sum(dim=-1)
    return ret #, ilim

import torch as ch



if __name__ == "__main__":

    s0 = arr_init([0,]*3, dtype=ch.int32)
    px = arr_init([1, 0, 0], dtype=ch.int32)
    dxy = arr_init([1, 1, 0], dtype=ch.int32)
    fxyz = arr_init([1, 1, 1], dtype=ch.int32)
    am222 = arr_init([2,]*3, dtype=ch.int32)
    am131 = arr_init([1, 3, 1], dtype=ch.int32)
    am444 = arr_init([4, 4, 4], dtype=ch.int32)
    am333 = arr_init([3, 3, 3], dtype=ch.int32)
    am500 = arr_init([5, 0, 0], dtype=ch.int32)
    ra = arr_init([0,]*3)
    rb = arr_init([1,]*3)

    # ans = overlap(fone, s0, ra, fone, s0, ra)
    # assert ch.isclose(ans, pow(pi/2, 3/2))

    test_data_list = [
        (fone, s0, ra, fone, s0, ra),
        (fone, s0, ra, fone, s0, rb),
        (fone, s0, ra, 2*fone, s0, ra),
        (fone, s0, ra, fone, px, ra),
        (fone, s0, ra, fone, px, rb),
        (fone, s0, ra, 2*fone, px, ra),
        (fone, s0, ra, fone, dxy, ra),
        (fone, s0, ra, fone, dxy, rb),
        (fone, s0, ra, 2*fone, dxy, ra),
        (fone, s0, ra, fone, fxyz, ra),
        (fone, s0, ra, fone, fxyz, rb),
        (fone, s0, ra, 2*fone, fxyz, ra),
        (fone, fxyz, ra, 2*fone, s0, rb),
        (fone, fxyz, ra, 2*fone, px, rb),
        (fone, fxyz, ra, 2*fone, dxy, rb),
        (fone, fxyz, ra, 2*fone, fxyz, rb),
        (fone, s0, ra, 2*fone, am222, rb),
        (fone, am222, ra, 2*fone, am222, rb),
        (fone, s0, ra, 2*fone, am222, rb),
        (fone, s0, ra, 2*fone, am131, rb),
        (fone, am131, ra, 2*fone, am131, rb),
        (fone, am131, ra, 2*fone, s0, rb),
        (fone, am222, ra, 2*fone, am333, rb),
        (fone, am444, ra, 2*fone, am333, rb),
        (fone, am500, ra, 2*fone, am333, rb),
    ]

    test_data_stacks = list(zip(*test_data_list))

    alpha0s = ch.stack(test_data_stacks[0])
    lmn0s = ch.stack(test_data_stacks[1])
    r0s = ch.stack(test_data_stacks[2])

    alpha1s = ch.stack(test_data_stacks[3])
    lmn1s = ch.stack(test_data_stacks[4])
    r1s = ch.stack(test_data_stacks[5])

    ans_ch = overlap_ch(alpha0s, lmn0s, r0s, alpha1s, lmn1s, r1s)
    for idx in range(len(test_data_list)):
        ans0 = overlap(*test_data_list[idx])
        ans1 = ans_ch[idx]
        perr = (ans0 - ans1) / ans0
        print(f"{idx:02d}\t{ans0.numpy()}\t{ans1.numpy()}\t{perr}")


    # shorthands for debugging
    l0s, m0s, n0s = lmn0s[:,0], lmn0s[:,1], lmn0s[:,2]
    l1s, m1s, n1s = lmn1s[:,0], lmn1s[:,1], lmn1s[:,2]
    r12sqs = norm2(r0s-r1s, axis=1)
    gammas = alpha0s + alpha1s
    pcens = (alpha0s[:,None]*r0s + alpha1s[:,None]*r1s) / gammas[:,None]
    pdiff0s = pcens - r0s
    pdiff1s = pcens - r1s
    px0s, px1s = pdiff0s[:,0], pdiff1s[:,1]

    debug_var = 1


    if False:
        imax = 6
        aas = [[] for _ in range(imax)]
        bbs = [[] for _ in range(imax)]
        for i in range(imax):
            for r in range(int(floor(ione*i/2)+1)):
                for u in range(int(floor((ione*i-2*r)/2)+1)):
                    ii = i - 2*r - u
                    istr=f"{ii}={i}-2*{r}-{u}"
                    aas[i].append(istr)
                    bbs[ii].append([i,r,u])
                    print(istr)
            print("="*10)
        # bbs = np.asarray(bbs) # ragged


    summx = (l0s + l1s).max()
    ilim = int(ch.floor(0.5 * summx).item())
    tmps = []
    for i in ch.arange(1+ilim, device=device):
        tmps += [binomial_prefactors_ch(2*i, l0s, l1s, px0s, px1s, opt=False),]
    bp_ans = ch.stack(tmps).T

    bps = binomial_prefactors_ch_all(l0s, l1s, px0s, px1s)



 



    debug_var = 1
