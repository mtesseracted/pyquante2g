
import time, sys
import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

th.set_default_dtype(th.float64)
th.manual_seed(0)
fzero, fone = th.as_tensor(0.0), th.as_tensor(1.0)

def func_ch(m, x):
    return th.pow(x, m) + th.pow(x, 0.5*m)


def calc_gt(cx, cy, cz, fxs):
    gt = th.zeros(cx.size(0))
    for ii in range(cx.shape[1]):
        for jj in range(cy.shape[1]):
            for kk in range(cz.shape[1]):
                gt += cx[:,ii]*cy[:,jj]*cz[:,kk]*fxs[:,ii+jj+kk]
    return gt


def calc_conv1(ax, ay, az, f, preflip=False):
    """ preflip is if ay and az are already flipped along axis=1 """
    B, nx = ax.shape
    _, ny = ay.shape
    _, nz = az.shape

    # Full length of 3-way convolution
    L = nx + ny + nz - 2
    m = min(f.size(1), L)

    # First conv: ax * ay
    w1 = ay if preflip else ay.flip(-1)             # (B,1,ny)  (flip for true convolution)
    c1 = F.conv1d(ax.unsqueeze(0), w1.unsqueeze(1), 
                  groups=B, padding=ny-1).squeeze(0)  # (B, nx+ny-1)

    # Second conv: (ax*ay) * az
    w2 = az if preflip else az.flip(-1) # (B,1,nz)
    c  = F.conv1d(c1.unsqueeze(0), w2.unsqueeze(1), 
                  groups=B, padding=nz-1).squeeze(0)[:,:m]  # (B, nx+ny+nz-2)

    # Weighted sum with f
    return th.einsum('ij,ij->i', c[:,:m], f[:,:m])


def calc_conv2_orig(ax, ay, az, f, preflip=False):
    B, nx = ax.shape
    ny, nz, nf = ay.size(1), az.size(1), f.size(1)

    # 1) Fuse small kernels: k = ay * az  (true convolution => flip az)
    w = az if preflip else az.flip(-1)
    k = F.conv1d(ay.unsqueeze(0), w.unsqueeze(1), 
                 groups=ay.size(0), padding=nz-1).squeeze(0)   # (B,ny+nz-1)

    # 2) Correlate f with k, but only need nx samples (i = 0..nx-1)
    nk = k.size(1)
    need = max(0, nx + nk - 1 - nf)    # pad so output length == nx
    pad_l = min(need, nk - 1)
    pad_r = need - pad_l
    c = F.conv1d(
        F.pad(f, (pad_l, pad_r)).unsqueeze(0), 
        k.unsqueeze(1), groups=f.size(0)
    ).squeeze(0)[:,:nx] # (B,nx)

    # 3) Weight by ax and reduce
    return th.einsum('ij,ij->i', ax, c)


def calc_conv2(ax, ay, az, f, preflip: bool=False):
    # Ensure nice memory layout for conv1d
    # ax = ax.contiguous(); ay = ay.contiguous(); az = az.contiguous(); f = f.contiguous()

    B, nx = ax.shape
    ny, nz, nf = ay.size(1), az.size(1), f.size(1)

    # ---- k = ay * az (true convolution) ----
    # input: (1, B, ny), weight: (B, 1, nz), groups=B
    w1 = (az if preflip else az.flip(-1)).unsqueeze(1)     # (B,1,nz)
    x1 = ay.unsqueeze(0)                                   # (1,B,ny)
    k  = F.conv1d(x1, w1, groups=B, padding=nz-1).squeeze(0)  # (B, ny+nz-1)
    nk = k.size(1)

    # ---- correlate f with k but compute only first nx outputs ----
    need_len  = nx + nk - 1                 # f coverage needed for first nx outputs
    take      = min(nf, need_len)
    right_pad = need_len - take             # only pad on the right if short

    x2 = f[:, :take].unsqueeze(0)           # (1,B,take)
    if right_pad:
        x2 = F.pad(x2, (0, right_pad))      # asymmetric pad (left=0, right)
    w2 = k.unsqueeze(1)                      # (B,1,nk)
    c  = F.conv1d(x2, w2, groups=B).squeeze(0)   # (B, nx) exactly

    # ---- final reduction (single-kernel dot) ----
    return th.linalg.vecdot(ax, c, dim=1)   # (B,)


# def calc_conv2_from_flat(a_flat: th.Tensor, f: th.Tensor, preflip: bool=False):
#     """
#     a_flat: (B*3, L) laid out as [ax_b, ay_b, az_b, ax_{b+1}, ay_{b+1}, az_{b+1}, ...]
#     f:      (B, 3*L - 2)
#     """
#     B  = f.size(0)
#     L  = a_flat.size(1)

#     # Make contiguous (3,B,L) so ax, ay, az are contiguous (B,L)
#     a3 = a_flat.view(B, 3, L).transpose(0, 1).contiguous()
#     ax, ay, az = a3[0], a3[1], a3[2]   # each (B, L), contiguous

#     return calc_conv2(ax, ay, az, f, preflip=preflip)


def per_batch_conv1d(x: th.Tensor, w: th.Tensor, pad=(0, 0),
                     flip_kernel: bool = False) -> th.Tensor:
    """
    x: (B, Lx), w: (B, Kw)  -> returns (B, Lout)
    Correlation by default; set flip_kernel=True for true convolution.
    """
    B, Lx = x.shape
    x_ = x.unsqueeze(0)                               # (1, B, Lx)  (batch-as-channels)
    if pad != (0, 0):
        x_ = F.pad(x_, pad)                           # asymmetric pad on-the-fly
    w_ = (w.flip(-1) if flip_kernel else w).unsqueeze(1)  # (B, 1, Kw)
    y = F.conv1d(x_, w_, groups=B)                    # (1, B, Lout)
    return y.squeeze(0)                               # (B, Lout)

def calc_conv3(ax, ay, az, f, preflip=False):
    B, nx = ax.shape
    ny, nz, nf = ay.size(1), az.size(1), f.size(1)

    # 1) Fuse small kernels: k = ay * az (true convolution)
    k = per_batch_conv1d(ay, az, pad=(nz-1, nz-1), flip_kernel=not preflip)   # (B, ny+nz-1)
    nk = k.size(1)

    # 2) Only compute the first nx correlation outputs:
    # they access f indices in [0, nx+nk-2], i.e. need length (nx+nk-1).
    need_len = nx + nk - 1
    take = min(nf, need_len)
    f_sub = f[:, :take]
    right_pad = need_len - take      # 0 if nf >= need_len, else the shortfall

    # Correlate and get exactly nx outputs (no extra slicing/padding):
    c = per_batch_conv1d(f_sub, k, pad=(0, right_pad))    # (B, nx)

    # 3) Weight by ax and reduce
    # return (ax * c).sum(dim=1)
    return th.einsum('ij,ij->i', ax, c)



bsz, asz = round(2**10), 21
cx = th.rand(bsz, asz-2)
cx[:,2] = fzero
cy = th.rand(bsz, asz)
cy[:,-2:] = fzero
cy[:,2] = fzero
cz = th.rand(bsz, asz+3)
cz[:,2] = fzero
cyf, czf = cy.flip(-1), cz.flip(-1)

xs = th.rand(bsz)
xs[((0,5),)] = fzero
ijkmax = cx.shape[1] + cy.shape[1] + cz.shape[1]
fxs = func_ch(th.arange(ijkmax-2)[None,:], xs[:,None])

# ax, ay, az, f = cx, cy, cz, fxs
t0 = time.time()
gt = calc_gt(cx, cy, cz, fxs)
t1 = time.time()
an1 = calc_conv1(cx, cy, cz, fxs)
t2 = time.time()
an2 = calc_conv2(cx, cy, cz, fxs)
t3 = time.time()
an3 = calc_conv3(cx, cy, cz, fxs)
t4 = time.time()

den = th.clamp(th.abs(gt), th.finfo(gt.dtype).eps, None)
pd1 = abs((an1-gt)/den).max().item()
pd2 = abs((an2-gt)/den).max().item()
pd3 = abs((an3-gt)/den).max().item()
pstr  = f"Initial validity check, gt({t1-t0:.2f}s)"
pstr += f" conv1({t2-t1:.2f}s)@ {pd1:.2e}"
pstr += f" conv2({t3-t2:.2f}s)@ {pd2:.2e}"
pstr += f" conv3({t4-t3:.2f}s)@ {pd3:.2e}"
print(pstr)


def fibonacci(n):
    ret = [1,1]
    for ii in range(1,n):
        ret.append(ret[ii]+ret[ii-1])
        if ret[ii+1] >= n:
            break
    return ret[1:]

records = []
for bpow in th.arange(18, 21):
    for asz in th.as_tensor(fibonacci(89)[4:]):
        bsz = int(2**bpow)
        cx = th.rand(bsz, asz)
        cy = th.rand(bsz, asz)
        cz = th.rand(bsz, asz)
        # cyf, czf = cy.flip(-1), cz.flip(-1) # no accuracy check, no need
        xs = th.rand(bsz)
        ijkmax = cx.shape[1] + cy.shape[1] + cz.shape[1]
        fxs = func_ch(th.arange(ijkmax-2)[None,:], xs[:,None])
        t1 = time.time()
        an1 = calc_conv1(cx, cy, cz, fxs, preflip=False)
        t2 = time.time()
        an2 = calc_conv2(cx, cy, cz, fxs, preflip=False)
        t3 = time.time()
        an3 = calc_conv3(cx, cy, cz, fxs, preflip=False)
        t4 = time.time()
        print(f"2^{bpow:2d}\t{asz}\t{t2-t1:.2f}\t{t3-t2:.2f}\t{t4-t3:.2f}"); sys.stdout.flush()
        records.append([bpow, asz, t2-t1, t3-t2])

records = np.stack(records)

debug_var = 1
