import os, time
import numpy as np
import torch as ch
import torch.nn.functional as F

dtype=ch.float64
ftyp = dtype
ityp = ch.int64
device='cpu'

# funcs
pow = ch.pow
exp = ch.exp
lgamma = ch.lgamma
floor = ch.floor
arr_init = lambda x, dtype=dtype, device=device: ch.as_tensor(x, dtype=dtype, device=device)
# constants
pi = arr_init(ch.pi)
sqpi = ch.sqrt(pi)
fone = arr_init(1.0)
ione = arr_init(1, dtype=ityp)
fzero = arr_init(0.0)
izero = arr_init(0, dtype=ityp)


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

def fact_ch(n):
    return exp(ch.lgamma(n+ione))


def gamma_inc(a, x):
    p = ch.igamma(a, x) # regularized
    ga = exp(lgamma(a))
    return ga * p

def fboys(m, x):
    mt, xt = arr_init(m+0.5), arr_init(x)
    xm = xt.clamp(ch.finfo(x.dtype).eps, None) 
    #  this formulation doesn't have correct limiting behavior at x=0, so clamp
    return 0.5 *  pow(xm, -mt) * gamma_inc(mt, xm)

def fboys_ch(m, x): # stable @ x=0
    m2 = m + 0.5*fone
    ret = 0.5 *  pow(x, -m2) * gamma_inc(m2, x)
    mask = x < ch.finfo(x.dtype).eps
    ret[:] = ch.where(mask, 0.5/m2, ret)
    return ret

def binomial_prefactor(s,ia,ib,xpa,xpb):
    """
    The integral prefactor containing the binomial coefficients from Augspurger and Dykstra.
    >>> binomial_prefactor(0,0,0,0,0)
    1
    """
    total= 0
    for t in range(s+1):
        if s-ia <= t <= ib:
            # total +=  binomial(ia,s-t)*binomial(ib,t)* \
            total +=  binom(ia,s-t)*binom(ib,t)* \
                     np.pow(xpa,ia-s+t)*np.pow(xpb,ib-t)
    return total


def gaussian_product_center(alpha1,A,alpha2,B):
    """ the center of the product of two cartesian gaussians """
    return (alpha1*A+alpha2*B)/(alpha1+alpha2)


def nuclear_attraction(alpha1,lmn1,A,alpha2,lmn2,B,C):
    """ Full form of the nuclear attraction integral """
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    gamma = alpha1+alpha2

    P = gaussian_product_center(alpha1,A,alpha2,B)
    rab2 = norm2(A-B)
    rcp2 = norm2(C-P)

    dPA = P-A
    dPB = P-B
    dPC = P-C

    Ax = A_array(l1,l2,dPA[0],dPB[0],dPC[0],gamma)
    Ay = A_array(m1,m2,dPA[1],dPB[1],dPC[1],gamma)
    Az = A_array(n1,n2,dPA[2],dPB[2],dPC[2],gamma)

    total = 0.
    for I in range(l1+l2+1):
        for J in range(m1+m2+1):
            for K in range(n1+n2+1):
                # total += Ax[I]*Ay[J]*Az[K]*Fgamma(I+J+K,rcp2*gamma)
                tmp = Ax[I]*Ay[J]*Az[K]*fboys(I+J+K,rcp2*gamma)
                if not np.isfinite(tmp):
                    debug_var = 1
                total += tmp
                
    val= -2*pi/gamma*exp(-alpha1*alpha2*rab2/gamma)*total
    return val

def A_term(i,r,u,l1,l2,PAx,PBx,CPx,gamma):
    """ THO eq. 2.18 """
    return np.pow(-1,i)*binomial_prefactor(i,l1,l2,PAx,PBx)*\
           np.pow(-1,u)*factorial(i)*np.pow(CPx,i-2*r-2*u)*\
           np.pow(0.25/gamma,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u)

def A_array(l1,l2,PA,PB,CP,g):
    """ THO eq. 2.18 and 3.1 """
    Imax = l1+l2+1
    A = [0]*Imax
    for i in range(Imax):
        for r in range(int(np.floor(i/2)+1)):
            for u in range(int(np.floor((i-2*r)/2)+1)):
                I = i-2*r-u
                A[I] = A[I] + A_term(i,r,u,l1,l2,PA,PB,CP,g)
    return A


def gaussian_product_center_ch(alpha0s,r0s,alpha1s,r1s):
    """ the center of the product of two cartesian gaussians """
    gammas = alpha0s + alpha1s
    return (alpha0s[:,None]*r0s[:,:] + alpha1s[:,None]*r1s[:,:]) / gammas[:,None]



def sum_ax_ay_az_f_conv(ax: ch.Tensor, ay: ch.Tensor, az: ch.Tensor, f: ch.Tensor) -> ch.Tensor:
    B, nx = ax.shape
    _, ny = ay.shape
    _, nz = az.shape

    # Full length of 3-way convolution
    L = nx + ny + nz - 2
    m = min(f.size(1), L)

    # First conv: ax * ay
    x1 = ax.unsqueeze(0)                      # (1,B,nx)
    w1 = ay.flip(-1).unsqueeze(1)             # (B,1,ny)  (flip for true convolution)
    c1 = F.conv1d(x1, w1, groups=B, padding=ny-1) #.squeeze(0)  # (B, nx+ny-1)

    # Second conv: (ax*ay) * az
    # x2 = c1.unsqueeze(1)                      # (B,1,nx+ny-1)
    w2 = az.flip(-1).unsqueeze(1)             # (B,1,nz)
    c  = F.conv1d(c1, w2, groups=B, padding=nz-1).squeeze(0)  # (B, nx+ny+nz-2)

    # Weighted sum with f
    return (c[:, :m] * f[:, :m]).sum(dim=1)   # (B,)



def nuclear_attraction_ch(alpha0s,lmn0s,r0s,alpha1s,lmn1s,r1s,Rns):
    """ Full form of the nuclear attraction integral """

    gammas = alpha0s + alpha1s
    # l0s, m0s, n0s = lmn0s[:,0], lmn0s[:,1], lmn0s[:,2]
    # l1s, m1s, n1s = lmn1s[:,0], lmn1s[:,1], lmn1s[:,2]
    # # px0s, px1s = pdiff0s[:,0], pdiff1s[:,1]

    Ps = gaussian_product_center_ch(alpha0s,r0s,alpha1s,r1s)
    r01sqs = norm2(r0s-r1s, axis=1)
    rnpsqs = norm2(Rns-Ps, axis=1)
    dP0, dP1, dPn = Ps-r0s, Ps-r1s, Ps-Rns

    Ax = A_array_ch(lmn0s[:,0], lmn1s[:,0], dP0[:,0], dP1[:,0], dPn[:,0], gammas)
    Ay = A_array_ch(lmn0s[:,1], lmn1s[:,1], dP0[:,1], dP1[:,1], dPn[:,1], gammas)
    Az = A_array_ch(lmn0s[:,2], lmn1s[:,2], dP0[:,2], dP1[:,2], dPn[:,2], gammas)

    totals = ch.zeros_like(gammas)

    lmn_max = (lmn0s + lmn1s).sum(dim=1).max()
    ijks = ch.arange(lmn_max+ione, dtype=lmn0s.dtype, device=lmn0s.device)
    fboys_ijk = fboys_ch(ijks[None,:], (rnpsqs*gammas)[:,None]) # [B,ijks]

    #@TODO: optimize by get Ay and Az in flipped form, then dont need
    # to flip them in sum_ax_ay_az_f_conv()
    totals = sum_ax_ay_az_f_conv(Ax, Ay, Az, fboys_ijk)
    totals *= -2 * pi / gammas * exp(-alpha0s*alpha1s*r01sqs/gammas)
    return totals
    # total = 0.
    # for I in range(l1+l2+1):
    #     for J in range(m1+m2+1):
    #         for K in range(n1+n2+1):
    #             # total += Ax[I]*Ay[J]*Az[K]*Fgamma(I+J+K,rcp2*gamma)
    #             tmp = Ax[I]*Ay[J]*Az[K]*fboys(I+J+K,rcp2*gamma)
    #             if not np.isfinite(tmp):
    #                 debug_var = 1
    #             total += tmp
                
    # val= -2*pi/gamma*exp(-alpha1*alpha2*rab2/gamma)*total
    # return val



def binomial_prefactors_ch_all(l0s, l1s, px0s, px1s):
    """ all valid binomial prefactors, batched
    NOTE!: for nuclear attraction we need all, we need a way to specify
    which ones we need
    """
    device = l0s.device
    # summx = (l0s + l1s).max()
    # ilim = int(ch.floor(0.5 * summx).item())
    ilim = (l0s + l1s).max()

    # Axes: L for ll (0..ilim), K for k (0..ilim)
    L = ch.arange(ilim + 1, device=device)        # [L]
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



def A_term_ch(i,r,u,l1,l2,PAx,PBx,CPx,gamma):
    """ THO eq. 2.18 """
    return pow(-1,i)*binomial_prefactor(i,l1,l2,PAx,PBx)*\
           pow(-1,u)*factorial(i)*pow(CPx,i-2*r-2*u)*\
           pow(0.25/gamma,r+u)/factorial(r)/factorial(u)/factorial(i-2*r-2*u)


def sign_parity_ch(t): # more efficient than pow(-1, t)
    return 1 - 2 * (t%2)

def A_array_ch(l0s,l1s,px0s,px1s,pns,gs, flipi=False):
    """ THO eq. 2.18 and 3.1 
    flipi : flip the I index of the returned tensor (for efficient convolution)
    """
    # return should be [B, imax]
    # local shapes are [B, imx, rmx, umx]
    typ, dev = gs.dtype, gs.device
    l01s = (l0s + l1s)[:,None,None,None]
    imx = l01s.max()
    rmx = floor(imx/2).to(imx.dtype)
    umx = rmx #floor((imx-2*rmx)/2) * ione
    irng = ch.arange(imx+1, device=dev)[None,:,None,None]
    rrng = ch.arange(rmx+1, device=dev)[None,None,:,None]
    urng = ch.arange(umx+1, device=dev)[None,None,None,:]
    nrurng = -(rrng + urng)

    bipres = binomial_prefactors_ch_all(l0s, l1s, px0s, px1s) # [B, imx]
    bipres = bipres[:,:,None,None]
    gpows = pow(4*gs[:,None,None,None], nrurng)
    cpows = pow(pns[:,None,None,None], irng + 2*nrurng)

    iparity = sign_parity_ch(irng)
    uparity = sign_parity_ch(urng)
    ifacts = fact_ch(irng)
    rfacts = fone / fact_ch(rrng)
    ufacts = fone / fact_ch(urng)
    irufacts = fone / fact_ch(irng + 2*nrurng)

    terms = iparity * bipres * uparity * ifacts * cpows * \
        gpows * rfacts * ufacts * irufacts

    # now I need to mask and then do the weird sum
    imask = (rrng <= floor(irng/2)) & (urng <= floor((irng-2*rrng)/2))
    # terms[:] = ch.where(mask, terms, fzero)
    # terms[~mask] = fzero.to(dtype=typ, device=dev)

    sum_idx = irng - 2*rrng - urng
    sum_idx[~imask] = imx+1 # dummy index 
    sum_idx_batch = sum_idx.view(-1).expand(gs.shape[0], -1) # expand to batch
    terms_flat = terms.view(gs.shape[0], -1)
    ret = ch.zeros((gs.shape[0], imx+2), dtype=typ, device=dev)
    ret.scatter_add_(dim=1, index=sum_idx_batch, src=terms_flat)


    # ii = -1
    # a = ch.where(sum_idx[0] == 0)
    # b = (arr_init([ii,]*a[0].shape[0], dtype=ityp), *a)
    # tmp = terms[b]
    # l0, l1 = l0s[ii].item(), l1s[ii].item()
    # x0, x1 = px0s[ii].item(), px1s[ii].item()
    # g, xn = gammas[ii].item(), dPn[ii,0].item()
    # gt = np.stack(A_array(l0, l1, x0, x1, xn, g))
    # # 840, 822, 630, 6
    # a210 = A_term(2,1,0, l0, l1, x0, x1, xn, g)


    ret = ret[:, :-1] # get rid of the dummy sum
    mask = (l01s < irng)[:,:,0,0] # invalids, squeeze
    ret[:] = ch.where(mask, fzero, ret)
    return ret

    # for i in range(Imax): # Imax = l1+l2+1
    #     for r in range(int(np.floor(i/2)+1)):
    #         for u in range(int(np.floor((i-2*r)/2)+1)):
    #             I = i-2*r-u
 


if __name__ == "__main__":
    # ans = (pi/2)**(3/2)
    # assert np.isclose(ans, overlap(alp, s0, ra, alp, s0, ra))
    # >>> isclose(nuclear_attraction(1,(0,0,0),array((0,0,0),'d'),1,(0,0,0),array((0,0,0),'d'),array((0,0,0),'d')),-3.141593)

    # ans = - pi
    # nai0 = nuclear_attraction(fone, s0, ra, fone, s0, ra, ra)
    # assert np.isclose(ans, nai0)

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

    test_data_list = [
        (fone, s0, ra, fone, s0, ra),
        (fone, s0, ra, fone, s0, rb),
        (fone, s0, ra, 2*fone, fxyz, ra),
        (fone, fxyz, ra, 2*fone, s0, rb),
        (fone, fxyz, ra, 2*fone, px, rb),
        (fone, fxyz, ra, 2*fone, fxyz, rb),
        (fone, am222, ra, 2*fone, am222, rb),
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

    Rns = 0.5*(r0s + r1s)

    Ps = gaussian_product_center_ch(alpha0s,r0s,alpha1s,r1s)
    r01sqs = norm2(r0s-r1s, axis=1)
    rnpsqs = norm2(Rns-Ps)
    dP0, dP1, dPn = Ps-r0s, Ps-r1s, Ps-Rns

    # shorthands for debugging
    l0s, m0s, n0s = lmn0s[:,0], lmn0s[:,1], lmn0s[:,2]
    l1s, m1s, n1s = lmn1s[:,0], lmn1s[:,1], lmn1s[:,2]
    r12sqs = norm2(r0s-r1s, axis=1)
    gammas = alpha0s + alpha1s
    pcens = (alpha0s[:,None]*r0s + alpha1s[:,None]*r1s) / gammas[:,None]
    px0s, px1s, pxns = dP0[:,0], dP1[:,1], dPn[:,0]

    # ansx = A_array_ch(l0s, l1s, px0s, px1s, dPn[:,0], gammas)
    ansx = A_array_ch(lmn0s[:,0], lmn1s[:,0], dP0[:,0], dP1[:,0], dPn[:,0], gammas)
    ansy = A_array_ch(lmn0s[:,1], lmn1s[:,1], dP0[:,1], dP1[:,1], dPn[:,1], gammas)
    ansz = A_array_ch(lmn0s[:,2], lmn1s[:,2], dP0[:,2], dP1[:,2], dPn[:,2], gammas)
    preds = [ansx, ansy, ansz]

    print("Relative errors in xyz for A_arrays")
    for idx in range(len(test_data_list)):
        perrs = [None]*3
        for ix in range(3):
            gt_ans = np.stack(A_array(
                lmn0s[idx,ix], lmn1s[idx,ix], dP0[idx,ix], 
                dP1[idx,ix], dPn[idx,ix], gammas[idx]))
            diff = gt_ans - preds[ix][idx,:len(gt_ans)].numpy()
            perrs[ix] = np.max(diff / np.clip(np.abs(gt_ans), 1.0e-12, None))

        print(f"({idx}) :: " + " ".join("{:.2e}".format(float(p.squeeze())) for p in perrs))
    print()


    ans_ch = nuclear_attraction_ch(alpha0s, lmn0s, r0s, alpha1s, lmn1s, r1s, Rns)

    print("Relative errors for NAIs")
    for idx in range(len(test_data_list)):
        gt = nuclear_attraction(*test_data_list[idx], Rns[idx,:])
        perr = (ans_ch[idx].item() - gt) / max(np.abs(gt), 1.0e-12)
        print(f"({idx}) :: {perr:.2e}")



    debug_var = 1
