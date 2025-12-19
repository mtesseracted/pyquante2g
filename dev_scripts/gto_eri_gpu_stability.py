import time
import torch as th
import torch.nn.functional as F
import numpy as np # debugging

from pq2_eri import coulomb_rep4

dtype=th.float64
device='cpu'
ammax = 20
ftyp = dtype
ityp = th.int64
ArrType = th.Tensor
# th.set_default_dtype(dtype)

# funcs
pow = th.pow
exp = th.exp
floor = th.floor
arr_init = lambda x, dtype=dtype, device=device: \
    th.as_tensor(x, dtype=dtype, device=device)
arr_sum = lambda x, **kwargs: th.sum(x, **kwargs)
arange = lambda n: th.arange(n, device=device)
# constants
pi = arr_init(th.pi)
sqpi = th.sqrt(pi)
fone = arr_init(1.0)
ione = arr_init(1, dtype=ityp)
fzero = arr_init(0.0)
izero = arr_init(0, dtype=ityp)
o4 = 0.25 * fone
eri_fac = (2*fone) * pow(pi, 2.5*fone)
# re-useable arrays
irngmx = arange(6*ammax+1*ione) # 2 for pairs, 3 for 3d in ERI

def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n-1)

# amax = 50
# bn2s = th.zeros((amax, amax), )


def sign_parity_ch(t): # more efficient than pow(-1, t)?
    # alternative: p = [1,-1]; return p[t%2]
    return ione - 2 * (t%2)

def binom_ch(n, i, round=True):
    # binom(n,i) = exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
    # binom(n,k) == 0.0 for k > n
    ret = exp(th.lgamma(n+fone) - (th.lgamma(i+fone) + th.lgamma(n-i+fone)))
    if round: ret = th.round(ret)
    return ret.to(dtype)
    
def binom2_ch(n, i, round=True):
    # tmp1 = th.lgamma(n+ione)
    # tmp2 = th.lgamma(i+ione)
    # tmp3 = th.lgamma(n-2*i+ione)
    # note: th.lgamma(int64) makes a float32
    ret = exp(th.lgamma(n+fone) - (th.lgamma(i+fone) + th.lgamma(n-2*i+fone)))
    if round: ret = th.round(ret)
    return ret.to(dtype)

def gamma_inc(a, x):
    at = a.to(dtype)
    p = th.igamma(at, x) # regularized
    ga = exp(th.lgamma(at))
    return ga * p

def fboys_ch(m, x): # stable @ x=0
    m2 = m + 0.5*fone
    ret = 0.5 *  pow(x, -m2) * gamma_inc(m2, x)
    mask = x < th.finfo(x.dtype).eps
    ret[:] = th.where(mask, 0.5/m2, ret) # @TODO: optimize for broadcasting with m and x being different sizes
    # ret[mask.expand(-1,ret.shape[1])] = 0.5/m2.expand(ret.shape[0], -1)
    return ret


class CartGtoPrim():
    """ Cartesian GTO Primitive """
    def __init__(self, amom:ArrType, cen:ArrType, alpha:ArrType):
        self.amom = amom # int (3,)
        self.cen = cen # float (3,)
        self.alpha = alpha # float
        self.lmax = self.amom.max()
        # self.binoms = self.calc_binoms() # These are wrong!

    def __len__(self):
        return len(self.amom)

    def calc_product_centers(self, other) -> ArrType:
        gamma = self.alpha + other.alpha
        return (self.alpha*self.cen + other.alpha*other.cen) / gamma
 
    def calc_binoms(self, pab:ArrType) -> ArrType:
        """ precompute the binomial expansion coefficients
        """
        irng = irngmx[:self.lmax+1]
        pows = self.amom[:,None] - irng[None,:]
        # pows = th.where(pows > 0, pows, fzero)
        # @TODO possible optmization: if we invert sign of pows, could apply relu then reinvert
        pows[pows<0] = fzero.to(pows.dtype) # @TODO optimize relu
        ret = th.pow((pab-self.cen)[:,None], pows) * \
            binom_ch(self.amom[:,None], irng[None,:])
        return ret

#end class CartGtoPrim():


CGP = CartGtoPrim
class CartGtoPrimPair():
    def __init__(self, ga:CGP, gb:CGP):
        self.ga = ga
        self.gb = gb
        self.gamma = ga.alpha + gb.alpha
        self.pab = self.calc_product_centers()
        self.hab = self.calc_h_array()
    
    def __len__(self):
        return len(self.ga)

    def calc_product_centers(self) -> ArrType:
        ga, gb = self.ga, self.gb
        return (ga.alpha*ga.cen + gb.alpha*gb.cen) / self.gamma
    
    def swap(self):
        tmp = self.ga
        self.ga = self.gb
        self.gb = tmp

    def calc_prod_binoms(self) -> ArrType:
        """ calculate binomial coefficents for the product of GTOs
        """
        # treat xyz as batch
        abinoms = self.ga.calc_binoms(self.pab)
        bbinoms = self.gb.calc_binoms(self.pab)
        ret = F.conv_transpose1d( # tranpose1d does "full" mode
            abinoms[None,:,:], bbinoms[:,None,:],
            groups=abinoms.shape[0]
        ).squeeze(0)
        return ret


    def calc_h_array(self) -> ArrType:
        """ v2 is fancier masking logic """
        # bsz = len(self)
        llmax = (self.ga.amom + self.gb.amom).max().item()
        full_sz = (len(self), llmax+1, llmax//2+1)

        bps = self.calc_prod_binoms()[:,:full_sz[1],None] # (3,llmax+1,1)
        g4s = (4*(self.gamma)).view(-1,1,1).expand(full_sz)

        irng = irngmx[:full_sz[1]][:,None]
        rrng = irngmx[:full_sz[2]][None,:]
        irmask = rrng <= irng//2
        iflat = irng.expand(full_sz[1:])[irmask]
        rflat = rrng.expand(full_sz[1:])[irmask]
        iiflat = iflat - 2*rflat

        h_vals_flat = bps.expand(full_sz)[:,irmask] # irng > amom == 0 from bps
        h_vals_flat *= th.pow(g4s[:,irmask], (rflat-iflat)[None,:])
        h_vals_flat *= binom2_ch(iflat, rflat)
        ret = th.zeros(full_sz[:2], dtype=dtype, device=device)
        ret.index_add_(1, iiflat, h_vals_flat)
        # @DEBUG
        # ii, rr = iflat[47], rflat[47]
        # tmp = binom2_ch(ii, rr)
        # gt = arb.binom2(ii.item(), rr.item())
        # @DEBUG
        return ret
    #end def calc_h_array(self, other):


    def calc_c_array(self, other:"CartGtoPrimPair") -> ArrType:
        # ga, gb, gc, gd = self.ga, self.gb, other.ga, other.gb
        hab, hcd = self.hab, other.hab
        # hab = th.as_tensor(self.calc_h_array_gt()) # @DEBUG
        # hcd = th.as_tensor(other.calc_h_array_gt()) # @DEBUG
        qp = other.pab - self.pab
        deltas = 0.25 * (fone/self.gamma + fone/other.gamma)
        abmax, cdmax = hab.shape[1]-1, hcd.shape[1]-1
        lmax = abmax + cdmax
        # full_sz = (len(self), abmax+1, cdmax+1, lmax//2+1)

        abrng = irngmx[:abmax+1][:,None,None]
        cdrng = irngmx[:cdmax+1][None,:,None]
        urng = irngmx[:lmax//2+1][None,None,:]
        iirng = abrng + cdrng - urng            # (L,M,U)
        iurng = iirng - urng
        iumask = iurng >= 0

        iiflat = iirng[iumask]
        iuflat = iurng[iumask]
        uflat = urng.expand(*iumask.shape[:2], -1)[iumask]
        facts = binom2_ch(iiflat+uflat, uflat) # float32
        parity = sign_parity_ch(cdrng+urng).expand(iumask.shape[0], -1, -1)[iumask]

        c_vals_flat = th.pow(qp[:,None], iuflat[None,:])
        c_vals_flat *= (hab[:,:,None]*hcd[:,None,:])[:,:,:,None].expand(-1,-1,-1,iumask.shape[-1])[:,iumask]
        c_vals_flat *= (facts * parity)[None,:]

        ret = th.zeros((len(self), lmax+2), dtype=dtype, device=device)
        ret.index_add_(1, iiflat, c_vals_flat)
        tmp3 = th.pow(deltas[:,None], -irngmx[:lmax+2])
        ret *= tmp3
        return ret
    #end def calc_c_array(self, other:"CartGtoPrimPair") -> ArrType:


    @staticmethod
    def triple_conv_dot(ax:ArrType, ay:ArrType, az:ArrType, fxyz:ArrType, 
        preflip:bool=False) -> ArrType:
        """ ans[:] = sum_{i,j,k} ax[:,i] ay[:,j] az[:,k] fxyz[:,i+j+k]
        preflip indicates whether az has been flipped along axis=1
        """
        B, nx = ax.shape
        nz, nf = az.shape[1], fxyz.shape[1]

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

        x2 = fxyz[:, :take].unsqueeze(0)           # (1,B,take)
        if right_pad:
            x2 = F.pad(x2, (0, right_pad))      # asymmetric pad (left=0, right)
        w2 = k.unsqueeze(1)                      # (B,1,nk)
        c  = F.conv1d(x2, w2, groups=B).squeeze(0)   # (B, nx) exactly

        # ---- final reduction (single-kernel dot) ----
        return th.linalg.vecdot(ax, c, dim=1)   # (B,)
    #end def triple_conv_dot()


    def coulomb_rep(self, other) -> ArrType:
        if len(self) != len(other):
            raise ValueError("::ERR::(coulomb_rep) Batch size mismatch")
        if len(self)%3 != 0:
            raise ValueError("::ERR::(coulomb_rep) Batch size must be a multiple of 3")

        gab, gcd = self.gamma[::3], other.gamma[::3]
        bsz = len(self) // 3
        tmp = th.empty(bsz)
        c_array = self.calc_c_array(other).view(bsz,3,-1).transpose(0,1).contiguous()
        delta = o4 / gab + o4 / gcd
        qp2 = th.pow(self.pab-other.pab, 2*fone).reshape(-1,3).sum(axis=1)
        fbx = o4 * qp2 / delta
        ijkmax = 3 * c_array.shape[-1]
        boys = fboys_ch(irngmx[:ijkmax+1][None,:], fbx[:,None])
        ret = self.triple_conv_dot(c_array[0], c_array[1], c_array[2], boys)
        del fbx, boys, 

        aa, ab = self.ga.alpha[::3], self.gb.alpha[::3]
        ac, ad = other.ga.alpha[::3], other.gb.alpha[::3]
        ra, rb = self.ga.cen.view(bsz,3), self.gb.cen.view(bsz,3)
        rc, rd = other.ga.cen.view(bsz,3), other.gb.cen.view(bsz,3)
        rab2 = th.sum(th.square(ra-rb), dim=1)
        rcd2 = th.sum(th.square(rc-rd), dim=1)

        ret *= eri_fac / (gab*gcd*th.sqrt(gab+gcd))
        ret *= exp(-aa*ab*rab2/gab)
        ret *= exp(-ac*ad*rcd2/gcd)
        return ret
    #end def coulomb_rep(self, other) -> ArrType:
#end class CartGtoPrimPair():

tmp = binom2_ch(arr_init(12), arr_init(5))

import pyquante22.arb_math as arb

# s p d f g h j
# 0 1 2 3 4 5 6
n = 12
ams = arr_init([0,0,0], dtype=th.int64)
amp = arr_init([1,0,0], dtype=th.int64)
amn = arr_init([n,0,0], dtype=th.int64)
r0 = arr_init([0,0,0])
rb = arr_init([1,2,0.5])
aa = arr_init([0.5,]*3)
gs = CartGtoPrim(ams, r0, aa)
gsb = CartGtoPrim(ams, rb, aa)
gp = CartGtoPrim(amp, r0, aa)
gn = CartGtoPrim(amn, r0, aa)
gns = CartGtoPrimPair(gn, gs)
gss = CartGtoPrimPair(gs, gs)
gssb = CartGtoPrimPair(gs, gsb)
gps = CartGtoPrimPair(gp, gs)

hpt = gns.hab[0]
hgt = arr_init([float(f) for f in arb.H_array(n, 0, 0, 0, 1)])

hff = [arb.frac_approx(f.item()) for f in hgt]
hfs = [f"{f.numerator}/{f.denominator}" for f in hff]


pff = [arb.frac_approx(f.item()) for f in hpt]
pfs = [f"{f.numerator}/{f.denominator}" for f in pff]


ans = gns.coulomb_rep(gssb)

gt = arb.coulomb_rep4(
    arb.arb(0.5), arb.arr_init([n,0,0]), arb.arr_init([0,]*3),
    arb.arb(0.5), arb.arr_init([0,0,0]), arb.arr_init([0,]*3),
    arb.arb(0.5), arb.arr_init([0,0,0]), arb.arr_init([0,]*3),
    arb.arb(0.5), arb.arr_init([0,0,0]), arb.arr_init([1,2,0.5]),
)

gtf = float(gt)
pdiff = (ans.item() - gtf) / gtf


debug_var = 1
