import time
import torch as th
import torch.nn.functional as F
import numpy as np # debugging

from pq2_eri import coulomb_rep4

# can we precompute terms for a basis func, e.g. the binomial terms
# needed in binomial_prefactors so that when you compute them
# later it's all dot products and einsums, no new computation

dtype=th.float64
device='cpu'
ammax = 20
ftyp = dtype
ityp = th.int64
ArrType = th.Tensor

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


def factorial(n, round=True):
    nt = th.as_tensor(n, dtype=dtype)
    v = exp(th.lgamma(nt+ione))
    if round: v = th.round(v)
    return type(n)(v.item())

def binom(n, i): # slow but correct
    ni, ii = int(n), int(i)
    return factorial(n) // (factorial(ii) * factorial(ni-ii))


def binom_ch(n, i, round=True):
    # binom(n,i) = exp(lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1))
    # binom(n,k) == 0.0 for k > n
    ret = exp(th.lgamma(n+ione) - th.lgamma(i+ione) - th.lgamma(n-i+ione))
    if round: ret = th.round(ret)
    return ret.to(dtype)
    
def binom2_ch(n, i, round=True):
    ret = exp(th.lgamma(n+ione) - th.lgamma(i+ione) - th.lgamma(n-2*i+ione))
    if round: ret = th.round(ret)
    return ret.to(dtype)


def gamma_inc(a, x):
    p = th.igamma(a, x) # regularized
    ga = exp(th.lgamma(a))
    return ga * p

def fboys_ch(m, x): # stable @ x=0
    m2 = m + 0.5*fone
    ret = 0.5 *  pow(x, -m2) * gamma_inc(m2, x)
    mask = x < th.finfo(x.dtype).eps
    ret[:] = th.where(mask, 0.5/m2, ret) # @TODO: optimize for broadcasting with m and x being different sizes
    # ret[mask.expand(-1,ret.shape[1])] = 0.5/m2.expand(ret.shape[0], -1)
    return ret



def binomial_prefactor(s,ia,ib,xpa,xpb):
    """ The integral prefactor containing the binomial coefficients from Augspurger and Dykstra.
    """
    total= 0
    for t in range(s+1):
        if s-ia <= t <= ib:
            # total +=  binomial(ia,s-t)*binomial(ib,t)* \
            total +=  binom(ia,s-t)*binom(ib,t)* \
                     np.pow(xpa,ia-s+t)*np.pow(xpb,ib-t)
    return total


def h_term(ii, rr, l0, l1, a, b, g):
    # facts = factorial(ii) // (factorial(rr) * factorial(ii-2*rr))
    facts = binom2_ch(arr_init(ii), arr_init(rr)).item()
    return facts * binomial_prefactor(ii,l0,l1,a,b,) * np.pow(4*g,rr-ii)

def H_array(l0, l1, a, b, g):
    lmax = l0 + l1
    h_array = np.zeros(lmax+1)
    for ia in range(lmax+1):
        for ir in range(ia//2 + 1):
            ii = ia - 2*ir
            h_array[ii] += h_term(ia, ir, l0, l1, a, b, g)
    return h_array

def H_array_full(l0, l1, a, b, g):
    lmax = l0 + l1
    h_array = np.zeros((lmax+1, lmax//2+1))
    for ia in range(lmax+1):
        for ir in range(ia//2 + 1):
            # ii = ia - 2*ir
            h_array[ia,ir] += h_term(ia, ir, l0, l1, a, b, g)
    return h_array



def c_term(l, m, u, hl, hm, px,):
    # d can be factored sin l+m-u == ii
    lm2u = l + m - 2*u
    if lm2u < 0:
        debug_var = 1
    facts = factorial(l+m) // (factorial(u) * factorial(lm2u))
    pre = (-1)**(m+u) * px ** lm2u #* d ** -(l+m-u)
    return hl * hm * facts * pre

def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):
    delta = 0.25 * (1/g1 + 1/g2)
    l01mx, l23mx = l0 + l1, l2 + l3
    c_array = np.zeros(l01mx+l23mx+1)
    dpows = delta ** -np.arange(c_array.shape[0])
    hls = H_array(l0, l1, px0, px1, g1)
    hms = H_array(l2, l3, qx2, qx3, g2)
    for il in range(l01mx+1): # wrong!, indices are over l01mx
        for im in range(l23mx+1):
            umx = (il+im) // 2
            for iu in range(umx+1):
                ii = il + im - iu
                c_array[ii] += c_term(il, im, iu, hls[il], hms[im], qpx)
                
    c_array *= dpows
    return c_array
#end def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):




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

    def calc_prod_binoms(self, other:"CartGtoPrim"):
        """ calculate binomial coefficents for the product of GTOs
        output is self.lmax+other.lmax+1, 
        not the ideal of (self.amom+other.amom).max()
        """
        # treat xyz as batch
        pab = self.calc_product_centers(other)
        abinoms = self.calc_binoms(pab)
        bbinoms = other.calc_binoms(pab)
        return F.conv_transpose1d( # tranpose1d does "full" mode
            abinoms[None,:,:], bbinoms[:,None,:],
            groups=abinoms.shape[0]
        ).squeeze(0)

    def calc_h_array_gt(self, other:"CartGtoPrim") -> ArrType:
        # BUG! H_array if from p-cen, not cen, look
        # at C_array
        bsz = self.amom.shape[0]
        g4s = (4*(self.alpha + other.alpha)).view(-1,1,1).expand(bsz,1,1) # handles scalar or batch
        gts = [H_array(self.amom[i].item(), other.amom[i].item(),
            self.cen[i].item(), other.cen[i].item(), 0.25*g4s[i].item())
            for i in range(self.amom.shape[0])]
        szmx = max(len(gt) for gt in gts)
        ret = np.stack([np.pad(gt, (0,szmx-len(gt))) for gt in gts])
        return ret
        

    def calc_h_array_v2(self, other:"CartGtoPrim") -> ArrType:
        """ v2 is fancier masking logic """
        bsz = self.amom.shape[0]
        llmax = (self.amom + other.amom).max().item()
        full_sz = (bsz, llmax+1, llmax//2+1)

        bps = self.calc_prod_binoms(other)[:,:full_sz[1],None] # (3,llmax+1,1)
        g4s = (4*(self.alpha + other.alpha)).view(-1,1,1).expand(full_sz)

        irng = irngmx[:full_sz[1]][:,None]
        rrng = irngmx[:full_sz[2]][None,:]
        irmask = rrng <= irng//2
        iflat = irng.expand(full_sz[1:])[irmask]
        rflat = rrng.expand(full_sz[1:])[irmask]
        iiflat = iflat - 2*rflat

        # @DEBUG
        # amask = irng[None,:,0] > (self.amom+other.amom)[:,None]
        # if th.any(bps[amask] != 0):
        #     debug_var = 1
        #     raise ValueError("wtf bro")

        h_vals_flat = bps.expand(full_sz)[:,irmask]
        h_vals_flat *= th.pow(g4s[:,irmask], (rflat-iflat)[None,:])
        h_vals_flat *= binom2_ch(iflat, rflat)
        # the places where irng > amom should already be 0 from bps
        ret = th.zeros(full_sz[:2], dtype=dtype, device=device)
        ret.index_add_(1, iiflat, h_vals_flat)
        return ret
    #end def calc_h_array_v2(self, other):


    def calc_h_array(self, other) -> ArrType:
        llmax = (self.amom + other.amom).max().item()
        bps = self.calc_prod_binoms(other)[:,:llmax+1,None] # (3,llmax+1,1)
        g4s = (4*(self.alpha + other.alpha)).view(-1, 1, 1) #,1,1)#.expand(bsz,llmax+1,llmax//2+1)

        irng = irngmx[:llmax+1][None,:,None]
        rrng = irngmx[:llmax//2+1][None,None,:]
        imask = rrng > irng//2
        iirng = irng - 2*rrng
        # tmp = F.relu(iirng)
        iirng[imask] = 0 #llmax + 1

        h_vals = (binom2_ch(irng, rrng) * th.pow(g4s, rrng-irng)) * bps
        # vmask = irng <= (self.amom + other.amom)[:,None,None]
        # h_vals[~vmask.expand(-1, -1, h_vals.shape[2])] = fzero # already zero from bps

        ret = th.zeros((len(self), llmax+1), dtype=dtype, device=device)
        ret.index_add_(1, iirng.flatten(), h_vals.reshape(ret.shape[0], -1))
        return ret

#end class CartGtoPrim():


def sign_parity_ch(t): # more efficient than pow(-1, t)?
    # alternative: p = [1,-1]; return p[t%2]
    return ione - 2 * (t%2)

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

    def calc_h_array_gt(self, full=False) -> ArrType:
        # BUG! H_array if from p-cen, not cen, look
        # at C_array
        pa = self.pab - self.ga.cen
        pb = self.pab - self.gb.cen
        if full:
            ret = [H_array_full(self.ga.amom[i].item(), self.gb.amom[i].item(),
                pa[i].item(), pb[i].item(), self.gamma[i].item())
                for i in range(len(self))]
        else:
            gts = [H_array(self.ga.amom[i].item(), self.gb.amom[i].item(),
                pa[i].item(), pb[i].item(), self.gamma[i].item())
                for i in range(len(self))]
            szmx = max(len(gt) for gt in gts)
            ret = np.stack([np.pad(gt, (0,szmx-len(gt))) for gt in gts])
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
        return ret
    #end def calc_h_array(self, other):


    def calc_c_array_gt(self, other):
        # def C_array(l0, l1, l2, l3, px0, px1, qx2, qx3, qpx, g1, g2):
        gts = [None,]*len(self)
        pxa = self.pab - self.ga.cen
        pxb = self.pab - self.gb.cen
        qxc = other.pab - other.ga.cen
        qxd = other.pab - other.gb.cen
        qp = other.pab - self.pab
        # delta = 0.25 / (fone/self.gamma + fone/other.gamma)
        for ii in range(len(self)):
            gts[ii] = C_array(
                self.ga.amom[ii].item(),self.gb.amom[ii].item(),
                other.ga.amom[ii].item(),other.gb.amom[ii].item(),
                pxa[ii].item(), pxb[ii].item(), qxc[ii].item(), qxd[ii].item(),
                qp[ii].item(), self.gamma[ii].item(), other.gamma[ii].item())
        szmx = max(len(gt) for gt in gts)
        ret = np.stack([np.pad(gt, (0,szmx-len(gt))) for gt in gts])
        return ret

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
#end class CartGtoPrimPair():


def sum_ax_ay_az_f_conv(ax, ay, az, f) -> ArrType:
    B, nx = ax.shape
    _, ny = ay.shape
    _, nz = az.shape

    # Full length of 3-way convolution
    L = nx + ny + nz - 2
    m = min(f.size(1), L)

    # First conv: ax * ay
    x1 = ax.unsqueeze(1)                      # (B,1,nx)
    w1 = ay.flip(-1).unsqueeze(1)             # (B,1,ny)  (flip for true convolution)
    c1 = F.conv1d(x1, w1, groups=B, padding=ny-1).squeeze(1)  # (B, nx+ny-1)

    # Second conv: (ax*ay) * az
    x2 = c1.unsqueeze(1)                      # (B,1,nx+ny-1)
    w2 = az.flip(-1).unsqueeze(1)             # (B,1,nz)
    c  = F.conv1d(x2, w2, groups=B, padding=nz-1).squeeze(1)  # (B, nx+ny+nz-2)

    # Weighted sum with f
    return (c[:, :m] * f[:, :m]).sum(dim=1)   # (B,)





if __name__ == "__main__":

    ra0 = arr_init([0,]*3)
    ra1 = arr_init([1,]*3)
    ra2 = arr_init([0,1,2])
    ra3 = arr_init([0.5,1.5,2.5])
    am000 = arr_init([0,0,0], dtype=ityp)
    am012 = arr_init([0,1,2], dtype=ityp)
    am123 = arr_init([1,2,3], dtype=ityp)
    am222 = arr_init([1,2,3], dtype=ityp)
    am333 = arr_init([1,2,3], dtype=ityp)
    am454 = arr_init([4,5,5], dtype=ityp)
    am100 = arr_init([1,0,0], dtype=ityp)
    am200 = arr_init([2,0,0], dtype=ityp)
    am300 = arr_init([3,0,0], dtype=ityp)
    am400 = arr_init([4,0,0], dtype=ityp)
    am500 = arr_init([5,0,0], dtype=ityp)
    am600 = arr_init([6,0,0], dtype=ityp)
    am700 = arr_init([7,0,0], dtype=ityp)

    gto_init_data = [
        (am000, ra0, fone),
        # (am000, ra1, 0.5*fone),
        # (am012, ra3, fone),
        # (am123, ra0, 2.2*fone),
        # (am222, ra0, 2.2*fone),
        # (am333, ra0, 2.2*fone),
        # (am454, ra0, fone),
        # (am454, ra3, 0.1*fone),
        # (am454, ra2, 0.2*fone),
        (am100, ra0, fone),
        (am200, ra0, fone),
        (am300, ra0, fone),
        (am400, ra0, fone),
        (am500, ra0, fone),
        (am600, ra0, fone),
        (am700, ra0, fone),
    ]
    gto_list = [CartGtoPrim(*args) for args in gto_init_data]

    eps = th.finfo(dtype).eps
    if False:
        for ii in range(len(gto_list)):
            gi = gto_list[ii]
            for jj in range(ii+1):
                gj = gto_list[jj]
                gt = gi.calc_h_array_gt(gj)
                a1 = gi.calc_h_array(gj)
                a2 = gi.calc_h_array_v2(gj)
                tgt = F.pad(th.from_numpy(gt), (0, a1.shape[1]-gt.shape[1]))

                diffs1 = (a1 - tgt) / th.clamp(tgt, eps, None)
                diffs2 = (a2 - tgt) / th.clamp(tgt, eps, None)
                mx1, mx2 = diffs1.abs().max(), diffs2.abs().max()
                if mx1 != mx2:
                    print(f"({ii},{jj}) :: {mx1:.2e}\t{mx2:.2e}")
                    # raise ValueError("oops")
                if max(mx1, mx2) > 1.0e-14:
                    raise ValueError("oopsy")
            

    ams = th.concat([gid[0] for gid in gto_init_data])
    r0s = th.concat([gid[1] for gid in gto_init_data])
    als = th.concat([gid[2].repeat(3) for gid in gto_init_data])

    reps = 100
    for reps in 10**np.arange(0,1): #[100, 500, 1000, 2000, 10000]:
        mm, rr, aa = ams.repeat(reps), r0s.repeat(reps), als.repeat(reps)
        m2, r2, a2 = mm.flip(dims=[0]), rr.flip(dims=[0]), aa.flip(dims=[0])
        a = CartGtoPrim(mm, rr, aa)
        b = CartGtoPrim(m2, r2, a2)

        mmn, rrn, aan = mm[:ams.shape[0]].numpy(), rr[:r0s.shape[0]].numpy(), aa[:ams.shape[0]].numpy()
        m2n, r2n, a2n = mmn[::-1], rrn[::-1], aan[::-1]
        gts = np.zeros((mmn.shape[0], np.amax(mmn+m2n)+1))
        ta = time.time()
        for ii in range(gts.shape[0]):
            tmp = H_array(mmn[ii], m2n[ii], rrn[ii], r2n[ii], aan[ii]+a2n[ii])
            gts[ii,:len(tmp)] = tmp
        tb = time.time()
        gts = np.tile(gts, (reps, 1))

        t0 = time.time()
        tmp = a.calc_h_array(b)
        t1 = time.time()
        tm2 = a.calc_h_array_v2(b)
        t2 = time.time()

        diffs1 = (tmp.numpy() - gts) / np.clip(gts, eps, None)
        diffs2 = (tm2.numpy() - gts) / np.clip(gts, eps, None)
        mx1, mx2 = diffs1.max(), diffs2.max()
        print(f"{mm.shape[0]:8d}\t{tb-ta:8.2f}\t{t1-t0:8.2f}\t{t2-t1:8.2f}\t{mx1:.2e}\t{mx2:.2e}")
        debug_var = 1
        

    gpair = CartGtoPrimPair(a, a)
    ctmp = gpair.calc_c_array(gpair)
    cgt = th.as_tensor(gpair.calc_c_array_gt(gpair))
    cgt = F.pad(cgt, (0, ctmp.shape[1]-cgt.shape[1]))


    result = gpair.coulomb_rep(gpair)

    gp1, gp2 = gpair, gpair
    gts = []
    tmps = []
    for ip in range(len(gp1)//3):
        aa, ama, ra  = gp1.ga.alpha[3*ip], gp1.ga.amom[3*ip:3*ip+3], gp1.ga.cen[3*ip:3*ip+3]
        ab, amb, rb  = gp1.gb.alpha[3*ip], gp1.gb.amom[3*ip:3*ip+3], gp1.gb.cen[3*ip:3*ip+3]
        ac, amc, rc  = gp2.ga.alpha[3*ip], gp2.ga.amom[3*ip:3*ip+3], gp2.ga.cen[3*ip:3*ip+3]
        ad, amd, rd  = gp2.gb.alpha[3*ip], gp2.gb.amom[3*ip:3*ip+3], gp2.gb.cen[3*ip:3*ip+3]
        gt = coulomb_rep4(aa.numpy(), ama.numpy(), ra.numpy(), 
                          ab.numpy(), amb.numpy(), rb.numpy(),
                          ac.numpy(), amc.numpy(), rc.numpy(),
                          ad.numpy(), amd.numpy(), rd.numpy(),)

        gxa = CartGtoPrim(ama, ra, aa.tile(3))
        gxb = CartGtoPrim(amb, rb, ab.tile(3))
        gya = CartGtoPrim(amc, rc, ac.tile(3))
        gyb = CartGtoPrim(amd, rd, ad.tile(3))
        gx = CartGtoPrimPair(gxa, gxb)
        gy = CartGtoPrimPair(gya, gyb)
        tmps.append(gx.coulomb_rep(gy))
        gts.append(gt)

    tmps = th.stack(tmps).squeeze()
    tgt = arr_init(np.stack(gts))

    pdiffs = (tmps-tgt) / th.clamp(th.abs(tgt), 1.0e-15, None)

    # that that ctmp is slightly larger than cgt @ [0] says this is probably
    # a similar masking issue as we found when making calc_h_array

    # penny lane answers (all self)
    # pl_am500_a22 = 0.00013492516817825066 # am=(5,0,0), alpha=2.2
    # pl_am700_a22 = 0.0004469882930051651  # am=(7,0,0), alpha=2.2

    for r in result:
        print(f"{r.item():.15f}")

    debug_var = 1
