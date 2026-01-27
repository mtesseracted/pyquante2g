##############################################################################
# This version does "batched" primitives in the first dim of CartGtoPrim
# The batch dimension hold primitives, so it's more of a contracted GTO
# Then CartGtoPrimPair takes two CartGtoPrim instances, and the output
# of pair calculations is a 2d "batch" of all possible combinations. 
# It will return the full combinations, making it the caller's responsiblity
# to know which primitive indices belong to which basis functions for
# collapsing into inter-basis function calculations
##############################################################################
import os, time
from hashlib import sha256
import pyquante22.utils.math as pm  # python math for combinatorials

if True:
    import pyquante22.array_ops as arr

    # get an instance of ArrayOps
    ao = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )
    ao.conv1d = ao.xp.nn.functional.conv1d  # @DEBUG
else:
    import pyquante22.array_ops as ao

    ao.conv1d = ao.conv_full_batch  # nope

ArrType = ao.ArrayType
one = ao.arr(1)
two = ao.arr(2)
zero = ao.arr(0)
half = ao.arr(0.5)
pi = ao.arr(4) * ao.atan(one)
sqpi = ao.sqrt(pi)
eri_fac = (2 * one) * pow(pi, 2.5 * one)
_parity = ao.arr([1, -1])  # (-1)^t = _parity[t%2]

amax = 20  # s-type to y-type, highest I've seen is i-type in def2-universal.gbs
arange_max = ao.arange(6 * amax + 1)  # 2 for pairs, 3 for 3d space
# facts = ao.arr([pm.fact_exact(a) for a in range(2*amax+1)])
# fact2s = ao.arr([pm.fact_exact(n,dn=2) for n in range(2*amax+1)])
# dfact2n1s = ao.arr([pm.fact_exact(2*n-1,dn=2) for n in range(2*amax+1)]) # (2n-1)!!, double factorial of 2n-1
ghalfs = ao.arr([pm.gamma_half_int(n) for n in range(2*amax+1)]) # more exact than exp(lgamma)
binoms = ao.arr(
    [
        [pm.binom_exact(n, k) for k in range(2 * amax + 1)]
        for n in range(2 * amax + 1)
    ]
)
binom2s = ao.arr(
    [
        [pm.binom2k_exact(n, k) for k in range(1 * amax + 1)]
        for n in range(2 * amax + 1)
    ]
)
binom2ns = ao.arr(
    [
        [pm.binom2n_exact(n, k) for k in range(1 * amax + 1)]
        for n in range(1 * amax + 1)
    ]
)

## @DEBUG
import torch
def check_finite(name, x):
    ok = torch.isfinite(x).all().item()
    print(name, "finite:", ok, "dtype:", x.dtype,
          "min:", x.min().item(), "max:", x.max().item())
    if not ok:
        bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:10]
        print("  first bad idx:", bad)

def lmu_indices(i, L, M):
    """return (l,m,u) triplets where i==l+m-u
    u in [0,(L+M)//2] and l+m-2u >= 0
    """
    U = (L+M)//2
    lrng = arange_max[:L+1]
    mrng = arange_max[:M+1]
    urng = arange_max[:U+1]
    iirng = lrng[:,None,None] + mrng[None,:,None] - urng[None,None,:]
    iurng = iirng - urng[None,None,:]
    iumask = (iurng >= 0) & (iirng == i)
    idtuple = torch.nonzero(iumask, as_tuple=True) 
    # return torch.stack(idtuple, dim=1)
    return idtuple
## @DEBUG

def binom22(i,r,u):
    """ i! / (r! u! (i-2(r+u))!) """
    # this one is a function using the form factored into 
    # binomials so we're not storing a 3d array, which would 
    # be 141kB in float64 with amax=20 (i=[0,2*mx+1], r,u=[0,mx+1])
    t = r + u
    return binoms[i,2*t] * binom2ns[t,r]


def fboys0(m2, x):
    """series expansion around x=0"""
    # expansion around x=0: (1 / (2*m + 1)) - (x / (2*m + 3)) + (x**2 / 6*(2*m + 5))
    # (0.5 / m2) - (0.5*x / (m2+1)) + (0.5*x*x / (m2+2))/6
    return 0.5 / m2 - 0.5 * x / (m2 + one)

def fboys(m, x):  # stable @ x=0
    """boys function"""
    m2 = m + 0.5 * one
    ret = 0.5 * pow(x, -m2) * ao.gamma_inc(m2, x)
    mask = x < ao.xp.finfo(x.dtype).eps
    mask = ao.broadcast_to(mask, ret.shape)
    # ret[:] = ao.where(mask, 0.5/m2, ret) # @TODO: optimize for broadcasting with m and x being different sizes
    ret[mask] = ao.broadcast_to(0.5 / m2, ret.shape)[mask]
    return ret


def triple_conv_dot(
    ax: ArrType, ay: ArrType, az: ArrType, fxyz: ArrType, preflip: bool = False
) -> ArrType:
    """ans[:] = sum_{i,j,k} ax[:,i] ay[:,j] az[:,k] fxyz[:,i+j+k]
    preflip indicates whether az has been flipped along axis=1
    """
    B, nx = ax.shape
    nz, nf = az.shape[1], fxyz.shape[1]

    # ---- k = ay * az (true convolution) ----
    # input: (1, B, ny), weight: (B, 1, nz), groups=B
    w1 = (az if preflip else az.flip(-1)).unsqueeze(1)  # (B,1,nz)
    x1 = ay.unsqueeze(0)  # (1,B,ny)
    k = ao.conv1d(x1, w1, groups=B, padding=nz - 1).squeeze(0)  # (B, ny+nz-1)
    # w1 = az if preflip else az.flip(-1)
    # k  = ao.conv_full_batch(ay, w1) # (B,ny+nz-1)
    nk = k.size(1)

    # ---- correlate f with k but compute only first nx outputs ----
    need_len = nx + nk - 1  # f coverage needed for first nx outputs
    take = min(nf, need_len)
    right_pad = need_len - take  # only pad on the right if short

    x2 = fxyz[:, :take].unsqueeze(0)  # (1,B,take)
    if right_pad:
        x2 = ao.pad(x2, (0, right_pad))  # asymmetric pad (left=0, right)
    w2 = k.unsqueeze(1)  # (B,1,nk)
    c = ao.conv1d(x2, w2, groups=B).squeeze(0)  # (B, nx) exactly

    # c  = ao.conv_full_batch(fxyz[:,:take], k)

    # ---- final reduction (single-kernel dot) ----
    # return th.linalg.vecdot(ax, c, dim=1)   # (B,)
    return ao.einsum('ij,ij->i', ax, c)
# end def triple_conv_dot()


class CartGtoPrim:
    """Cartesian GTO Primitive (batched)"""
    # legacy: when batch and spatial dim were flattened into one

    def __init__(self, amom: ArrType, cen: ArrType, alpha: ArrType, nd: int = 3):
        if nd != 3:
            raise NotImplementedError("Only 3d GTOs currently implemented")
        self.nd = nd
        amom = self._cast_batch(amom, nd=self.nd, name="amom") # support legacy
        cen  = self._cast_batch(cen,  nd=self.nd, name="cen")
        B = amom.shape[0]
        if cen.shape[0] != B:
            raise ValueError("amom and cen must have the same batch size")

        # alpha: require (B,) but allow (B,1) or legacy (B*nd,)
        if getattr(alpha, "ndim", 0) == 0: # scalar
            alpha = ao.arr([alpha])
        if getattr(alpha, "ndim", None) == 2 and alpha.shape[1] == 1: # (B,1)
            alpha = alpha.reshape(-1)
        if getattr(alpha, "ndim", None) == 1 and len(alpha) == B * nd: # support legacy
            alpha = alpha.reshape(B, nd)[:, 0]

        self.amom = amom
        self.cen = cen
        self.alpha = alpha
        self.lmax = self.amom.max().item()
        self.norm = self.calc_norm()

    @staticmethod
    def _cast_batch(x, nd: int, name: str):
        if getattr(x, "ndim", None) == 1:  # allow legacy flattened (B*nd,) input
            if len(x) % nd != 0:
                raise ValueError(f"{name} length must be a multiple of nd")
            return x.reshape(-1, nd)

        if getattr(x, "ndim", None) != 2 or x.shape[1] != nd:
            raise ValueError(f"{name} must be shape (B,{nd}) or (B*{nd},)")

        return x

    # def __getitem__(self, idx):
    #     return self.amom[idx], self.cen[idx], self.alpha[idx]
    def __getitem__(self, idx):
        return CartGtoPrim(self.amom[idx], self.cen[idx], self.alpha[idx])

    def __len__(self):
        return int(self.amom.shape[0])

    def __mul__(self, other):
        return CartGtoPrimPair(self, other)

    def __add__(self, other):
        if self.nd != other.nd:
            raise ValueError("nd mismatch")
        return CartGtoPrim(
            ao.concat([self.amom, other.amom], axis=0),
            ao.concat([self.cen, other.cen], axis=0),
            ao.concat([self.alpha, other.alpha], axis=0),
            nd=self.nd,
        )

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __repr__(self):
        ret = f"{type(self).__name__} (Lx,Ly,Lz) (Rx,Ry,Rz) Exp"
        for i in range(len(self)):
            ll = " ".join(f"{int(v):2d}" for v in ao.tolist(self.amom[i]))
            xx = " ".join(f"{float(v):12.4f}" for v in ao.tolist(self.cen[i]))
            ee = f"{float(self.alpha[i]):12.4e}"
            ret += f"\n{ll}  {xx}  {ee}"
        return ret

    def hash(self, digits=8):
        return sha256(repr(self).encode("utf-8")).hexdigest()[:digits]

    def __str__(self):
        if len(self) == 1:
            ams = ",".join([f"{int(v)}" for v in ao.tolist(self.amom[0])])
            rxs = ",".join([f"{float(v):.2f}" for v in ao.tolist(self.cen[0])])
            aas = f"{float(self.alpha[0]):.2e}"
        else:

            def summarize_pairs(keys, counts, fmt="{k}x{c}", limit=4):
                s = ",".join(fmt.format(k=k, c=c) 
                    for k, c in zip(keys[:limit], counts[:limit]))
                return s + ("..." if len(keys) > limit else "")

            # amom summary (by total angular momentum Lx+Ly+Lz)
            Ls, Lc = ao.unique(self.amom.sum(axis=1), return_counts=True)
            ams = summarize_pairs(ao.tolist(Ls), ao.tolist(Lc))

            # center summary
            rus, rcs = ao.unique(self.cen, axis=0, return_counts=True)
            if len(rus) == 1:
                rxs = ",".join(f"{float(v):.2f}" for v in ao.tolist(rus[0]))
            else:
                rxs = summarize_pairs(["R"] * len(rcs), ao.tolist(rcs), fmt="{k}{c}")

            # alpha summary
            aas_u = ao.unique(self.alpha)
            if len(aas_u) == 1:
                aas = f"{float(ao.tolist(aas_u)[0]):.2e}"
            else:
                aas = f"{float(ao.amin(self.alpha)):.2e}...{float(ao.amax(self.alpha)):.2e}"

        return f"GTO[{len(self)}]_{{{ams}}}@<{rxs}>;{aas}[{self.hash()}]"
    #end def __str__(self):


    def calc_norm(self) -> ArrType:
        """Normalize basis function. From THO eq. 2.2, with alpha shared across xyz."""
        # norm2_1d shape: (B,nd)
        norm2_1d = ao.pow(2 * self.alpha[:, None], self.amom + half)
        norm2_1d /= ghalfs[self.amom]
        norm2 = ao.prod(norm2_1d, axis=1)  # (B,)
        return ao.sqrt(norm2)


    def calc_binoms(self, pab: ArrType, offset=None) -> ArrType:
        """Compute binomial expansion coeffs given a center
        param: pab: (B,nd) or (B,C,nd)
        param: offset: (nd,) optional; change to amom (for kinetic)
        returns: (B,nd,Lmax+1) or (B,C,nd,Lmax+1)
        """
        B, nd = len(self), self.nd
        # if pab.ndim not in (2, 3) or pab.shape[0] != B or pab.shape[-1] != nd:
        #     raise ValueError(f"pab must be shape (B,{nd}) or (B,C,{nd}) with B={B}")
        keepC = (pab.ndim == 3)
        pab  = pab if keepC else pab[:, None, :]     # (B,C,nd)

        if offset is not None:
            amom = (self.amom + offset[None,:])[:,None,:] # (B,1,nd)
            lmax = max(0, amom.max().item())
        else:
            amom = self.amom[:,None,:] # (B,1,nd)
            lmax = self.lmax

        irng = arange_max[:lmax + 1]  # (L,)
        pows = amom[..., None] - irng[None, None, None, :]  # (B,C,nd,L)
        pows[pows < 0] = zero  # @TODO optimize with relu
        ret = binoms[amom[..., None], irng[None, None, None, :]]  * \
            ao.pow((pab - self.cen[:, None, :])[..., None], pows) # (B,C,nd,L)
        mask = (amom < 0)[..., None]  # (B,1,nd,1)
        ret[ao.broadcast_to(mask, ret.shape)] = zero
        return ret if keepC else ret[:,0,:,:]
    #end def calc_binoms(self, pab: ArrType, offset=None) -> ArrType:

#end class CartGtoPrim:

CGP = CartGtoPrim # shorthand

class CartGtoPrimPair:
    """Pairs of CartGtoPrim for batch'd eval of integrals
    """

    def __init__(self, ga: CGP, gb: CGP):
        if ga.nd != gb.nd:
            raise ValueError("::ERR::(CartGtoPrimPair) pairs must be same spatial dim")
        self.nd = ga.nd
        self.ga = ga
        self.gb = gb
        self.gamma = ga.alpha[:, None] + gb.alpha[None, :]  # (Na,Nb) exponent sums
        self.pab = self.calc_product_centers()              # (Na,Nb,3) product centers P_AB
        self.binom_pres = self.calc_prod_binoms()           # (Na,Nb,3,Lp)
        self.hab = self.calc_h_array()                      # (Na,Nb,3,Lh)

    def __len__(self):
        return int(len(self.ga) * len(self.gb))

    @property
    def shape(self):
        return len(self.ga), len(self.gb)

    def __getitem__(self, idx):
        return CartGtoPrimPair(self.ga[idx[0]], self.gb[idx[1]])

    def __repr__(self):
        rga = "A: " + repr(self.ga)
        rgb = "B: " + repr(self.gb)
        return f"{rga}\n{rgb}"

    def __str__(self):
        return f"PairGTO[{len(self.ga)}x{len(self.gb)}]{str(self.ga)}//{str(self.gb)}"

    def __or__(self, other):
        # syntactic sugar, ERI: (a*b|c*d), NAI: (a*b|R)
        if isinstance(other, ArrType):
            return self.calc_nai(other)  # implemented later
        if isinstance(other, type(self)):
            return self.calc_eri(other)  # implemented later
        raise NotImplementedError(f"No integral implemented for {type(other)}")

    def hash(self, digits=8):
        return sha256(repr(self).encode("utf-8")).hexdigest()[:digits]

    def calc_product_centers(self) -> ArrType:
        """GTO product centers"""
        ga, gb = self.ga, self.gb
        # numerically stable form: P = (aa/g)*A + (bb/g)*B
        aa = ga.alpha[:, None] / self.gamma   # (Na,Nb)
        bb = gb.alpha[None, :] / self.gamma   # (Na,Nb)
        ra = ga.cen[:, None, :]               # (Na,1,3)
        rb = gb.cen[None, :, :]               # (1,Nb,3)
        return (aa[..., None] * ra) + (bb[..., None] * rb)  # (Na,Nb,3)


    def calc_prod_binoms(self) -> ArrType:
        """GTO product binomial prefactors, returns: (Na, Nb, nd, Lp)"""
        ga, gb, nd = self.ga, self.gb, self.nd
        Na, Nb, P = len(ga), len(gb), self.pab              # P:(Na,Nb,nd)
        self.abinoms = ga.calc_binoms(P)                    # (Na,Nb,nd,La)
        btmp = gb.calc_binoms(ao.transpose(P,(1, 0, 2)))    # (Nb,Na,nd,Lb)
        self.bbinoms = ao.transpose(btmp,(1, 0, 2, 3))      # (Na,Nb,nd,Lb)
        G = Na * Nb * nd
        pres = ao.conv_full_batch(self.abinoms.reshape(G, -1), self.bbinoms.reshape(G, -1))
        return pres.reshape(Na, Nb, nd, -1)


    def calc_a_array(self, rcen:ArrType) -> ArrType:
        """THO eq. 2.18 and 3.1"""
        # assume rcen is shape (C,3)
        bps = self.binom_pres[...,None,None]        # (Na,Nb,1,nd,lmax,1,1)
        g4s = (4*self.gamma)[:,:,None,None,None,None] # (Na,Nb,1,1,1,1)

        dPn = self.pab[:,:,None,:] - rcen[None,None,:,:]
        lmax = bps.shape[3] - 1 # (Na,Nb,nd,lmax)
        rumax = lmax // 2
        full_sz = (len(self.ga), len(self.gb), self.ga.nd, 
                   lmax + 1, rumax + 1, rumax + 1)
 
        irng = arange_max[: full_sz[3]][:,None,None]
        rrng = arange_max[: full_sz[4]][None,:,None]
        urng = arange_max[: full_sz[5]][None,None,:]
        nrurng = -(rrng + urng)

        gpows = ao.pow(g4s, nrurng)
        cpows = ao.pow(dPn[:,None,None,None], irng + 2*nrurng)


        if False:
            abmom = self.ga.amom + self.gb.amom
            llmax = abmom.max().item()
            full_sz = (len(self), llmax + 1, llmax // 2 + 1, llmax // 2 + 1)
            # local shapes are [B, imx, rmx, umx], (batch, i, r, u)
            irng = arange_max[: full_sz[1]][None,:,None,None]
            rrng = arange_max[: full_sz[2]][None,None,:,None]
            urng = arange_max[: full_sz[3]][None,None,None,:]
            nrurng = -(rrng + urng)

            bipres = self.calc_prod_binoms()[:,:,None,None]
            gpows = ao.pow(4*self.gamma[:,None,None,None], nrurng)
            cpows = ao.pow(dPn[:,None,None,None], irng + 2*nrurng)

            iuparity = _parity[(irng+urng)%2]
            terms = iuparity * bipres * cpows * gpows * \
                binom22(irng,rrng,urng)

            # now I need to mask and then do the weird sum
            imask = (rrng <= ao.floor(irng/2)) & \
                (urng <= ao.floor((irng-2*rrng)/2))

            sum_idx = irng - 2*rrng - urng
            sum_idx[~imask] = full_sz[1] # dummy index @TODO check if always 0 and optimize this out
            terms_flat = terms.reshape(full_sz[0], -1)
            sum_idx_batch = ao.broadcast_to(sum_idx.reshape(-1), (full_sz[0], -1))
            ret = ao.zeros((full_sz[0], full_sz[1]+1))
            ret.scatter_add_(dim=1, index=sum_idx_batch, src=terms_flat)

            ret = ret[:, :-1] # get rid of the dummy sum
            mask = (abmom < irng)[:,:,0,0] # invalids, squeeze
            ret[:] = ao.where(mask, ao.arr(0), ret)
            return ret




    def calc_h_array(self) -> ArrType:
        """THO Eqn X.X"""

        bps = self.binom_pres[...,None] # (Na,Nb,nd,lmax)
        lmax = bps.shape[3] - 1
        full_sz = (len(self.ga), len(self.gb), self.ga.nd, lmax + 1, lmax // 2 + 1)
        g4s = ao.broadcast_to((4*self.gamma)[:,:,None,None,None], full_sz)

        irng = arange_max[: full_sz[3]][:, None]
        rrng = arange_max[: full_sz[4]][None, :]
        irmask = rrng <= irng // 2
        iflat = ao.broadcast_to(irng, full_sz[3:])[irmask]
        rflat = ao.broadcast_to(rrng, full_sz[3:])[irmask]
        iiflat = iflat - 2 * rflat

        h_vals_flat = ao.broadcast_to(bps, full_sz)[...,irmask]  # irng > amom == 0 from bps
        h_vals_flat *= ao.pow(g4s[...,irmask], (rflat - iflat)[None,None,None,:])
        h_vals_flat *= binom2s[iflat, rflat][None,None,None,:]

        ret = ao.zeros(full_sz[:-1])
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), h_vals_flat.shape)
        ret.scatter_add_(dim=-1, index=sum_ids, src=h_vals_flat)
        return ret
    # end def calc_h_array(self):


    # def debug_c_calc(self, other):
    #     Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
    #     N1, N2 = Na*Nb, Nc*Nd
    #     B = N1 * N2
    #     hab, hcd = self.hab, other.hab
    #     hab = hab.reshape(-1, *hab.shape[2:]) # (Na*Nb,nd,L)
    #     hcd = hcd.reshape(-1, *hcd.shape[2:]) # (Nc*Nd,nd,M)

    #     gab, gcd = self.gamma.reshape(-1), other.gamma.reshape(-1) # (Na*Nb), (Nc*Nd)
    #     deltas = (0.25 * (one / gab[:,None] + one / gcd[None,:])) #.reshape(-1)

    #     qp = other.pab[None,None,:,:,:] - self.pab[:,:,None,None,:] 
    #     qp = qp.reshape(N1, N2, self.nd) # (Na,Nb,Nc,Nd,nd) -> (B*nd)

    #     check_finite("Hab", hab)
    #     check_finite("Hcd", hcd)
    #     check_finite("b2",  binom2s)
    #     check_finite("qp",  qp)
    #     check_finite("delta", deltas)
    #     print("delta min abs:", deltas.abs().min().item())
    #     print("qp max abs:", qp.abs().max().item())

    #     cgt = self.calc_c_array(other)
    #     from pyquante22.basis.c_array_funcs_v2 import compute_methods

    #     for id in range(self.nd):
    #         h1i = hab[:,id,:]
    #         h2i = hcd[:,id,:]
    #         qpi = qp[:,:,id]

    #         # L, M = h1i.shape[-1]-1, h2i.shape[-1]-1
    #         # iid = 1
    #         # li, mi, ui = lmu_indices(iid, L, M)
    #         # h12i = h1i[:,None,li] 
    #         # h12i *= h2i[None,:,mi] * ao.pow(-one, mi+ui)[None,None,:]
    #         # iterms = h12i * binom2s[li+mi,ui][None,None,:]
    #         # iterms *= ao.pow(qpi[:,:,None], (li+mi-2*ui)[None,None,:])
    #         # iterms *= ao.pow(deltas[:,:,None], -(li+mi-ui)[None,None,:])
    #         # iterm = ao.sum(iterms, axis=-1)

    #         # beta = qpi/deltas
    #         cgti = cgt[:,:,:,:,id,:]

    #         for method in compute_methods:
    #             tmp = method(h1i,h2i,binom2s,qpi,deltas)
    #             cmi = tmp.reshape(Na,Nb,Nc,Nd,-1)
    #             lmin = min(cgti.shape[-1], cmi.shape[-1])
    #             assert ao.abs(cgti[...,lmin:]).sum().item() == 0
    #             assert ao.abs(cmi[...,lmin:]).sum().item() == 0
    #             if ao.xp.any(~ao.xp.isfinite(cmi[...,])):
    #                 print(f"NaNs in {method.__name__}")
    #             else:
    #                 gt = cgti[...,:lmin]
    #                 diffs = cmi[...,:lmin] - gt
    #                 denoms = ao.where(ao.abs(gt)>0, gt, one)
    #                 perrs = diffs/denoms
    #                 mxerr = ao.abs(perrs).max().item()
    #                 print(f"{method.__name__} %err: {mxerr:.4e}")
    #                 if mxerr > 1e-10:
    #                     import torch
    #                     flat_idx = torch.argmax(ao.abs(perrs))  # global argmax over all elements
    #                     idx = tuple(torch.unravel_index(flat_idx, perrs.shape))

    #                     debug_var = 1

    #             debug_var = 1
    #         debug_var = 1
    #     debug_var = 1

    # flatten all batch and spatial dims into one for ease of broadcasting
    # then re-use the logic from cart_gto.py. It will be slightly different
    # because this supports other being different batch shapes than self
    # so we will pre-combine certain quantities to make them the combo batch size
    # B = Na * Nb * Nc * Nd
    # this method has terrible scaling, we're creating the full size (large ndim)
    # intermediate objects. Instead we need to go back and re-derive how
    # to create C_array from einsums of lower-dimensional objects, such as hab, etc.
    # will probably need to create some new intermediate objects, but
    # that will also clean up the code if they're functionalized

    # rederivation: we already saw this before when we did the arb_math
    # calculation, but the delta powers can be factored out because
    # they're all to the index power, and only need to range up
    # to L+M, which is smaller than L*M. the qp terms are similar,
    # but the extra optimization is that theyre to the L+M-2u, which is just
    # (L+M-u) - u, so we can factor the same as delta for the L+M-u part
    # and then we only need all the combination for the urng
    # (which is unclear how that will work with scatter_add)

    # another optimization: the whole add terms where L+M-u = i at
    # index i can be precomputed as a grid of indices that we
    # can then reuse? It is 3d, so maybe too large
    # it probably has a generating function though we could derive

    def calc_c_array(self, other) -> ArrType:
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        B = Na * Nb * Nc * Nd

        # if hasattr(self, "deltas"):
        #     deltas = self.deltas
        # else:
        gab, gcd = self.gamma.reshape(-1), other.gamma.reshape(-1) # (Na*Nb), (Nc*Nd)
        deltas = (0.25 * (one / gab[:,None] + one / gcd[None,:])).reshape(-1)

        hab, hcd = self.hab, other.hab
        hab = hab.reshape(-1, *hab.shape[2:]) # (Na*Nb,nd,L)
        hcd = hcd.reshape(-1, *hcd.shape[2:]) # (Nc*Nd,nd,M)
        habcd = hab[:,None,:,:,None] * hcd[None,:,:,None,:] # (Na*Nb,Nc*Nd,nd,L,M)
        habcd = habcd.reshape(-1, hab.shape[-1], hcd.shape[-1]) # (B*nd, L, M)

        # qp = other.pab[:,:,None,None,:] - self.pab[None,None,:,:,:]
        qp = other.pab[None,None,:,:,:] - self.pab[:,:,None,None,:] 
        qp = qp.reshape(-1) # (Na,Nb,Nc,Nd,nd) -> (B*nd)
        abmax, cdmax = habcd.shape[1] - 1, habcd.shape[2] - 1
        lmax = abmax + cdmax
        L, M, U = abmax+1, cdmax+1, lmax//2 + 1

        abrng = arange_max[: abmax + 1][:, None, None] # (L,1,1)
        cdrng = arange_max[: cdmax + 1][None, :, None] # (1,M,1)
        urng = arange_max[: lmax // 2 + 1][None, None, :] # (1,1,U)
        iirng = abrng + cdrng - urng  # L+M-u, (L,M,U)
        iurng = iirng - urng # L+M-2u, (L,M,U)
        iumask = iurng >= 0 # 

        iiflat = iirng[iumask]
        iuflat = iurng[iumask]
        uflat = ao.broadcast_to(urng, (*iumask.shape[:2], urng.shape[-1]))[iumask]
        facts = binom2s[iiflat + uflat, uflat]  # (L,M,U)
        # parity = _parity((cdrng+urng)%2).expand(iumask.shape[0],-1,-1)[iumask]
        parity = ao.broadcast_to(
            _parity[(cdrng + urng) % 2],
            (iumask.shape[0], cdrng.shape[1], urng.shape[2]),
        )[iumask]

        c_vals_flat = ao.pow(qp[:,None], iuflat[None,:])
        c_vals_flat *= ao.broadcast_to(
            habcd[:, :, :, None], #(B*nd,L,M,1)
            (*habcd.shape, iumask.shape[-1]),
            # (*hab.shape[:2], hcd.shape[1], iumask.shape[-1]),
        )[:, iumask]
        c_vals_flat *= (facts * parity)[None, :] # (1,L,M,U)

        ret = ao.zeros((B*self.nd, lmax + 2))
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), c_vals_flat.shape)
        ret.scatter_add_(dim=1, index=sum_ids, src=c_vals_flat)
        d2 = ao.broadcast_to(deltas[:,None], (B,self.nd)).reshape(-1)
        ret *= ao.pow(d2[:, None], -arange_max[: lmax + 2])
        ret = ret.reshape(Na,Nb,Nc,Nd,self.nd, lmax+2)
        return ret
    # end def calc_c_array(self, other:"CartGtoPrimPair") -> ArrType:


    def calc_c_array_v2(self, other) -> ArrType:
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        nd = self.nd
        Nab, Ncd = Na * Nb, Nc * Nd
        B = Nab * Ncd

        if hasattr(self, "deltas"):
            deltas = self.deltas
        else:
            gab = self.gamma.reshape(-1)          # (Nab,)
            gcd = other.gamma.reshape(-1)         # (Ncd,)
            deltas = (0.25 * (one / gab[:, None] + one / gcd[None, :])).reshape(-1)  # (B,)

        hab = self.hab.reshape(-1, *self.hab.shape[2:])    # (Nab, nd, L)
        hcd = other.hab.reshape(-1, *other.hab.shape[2:])  # (Ncd, nd, M)

        L, M = hab.shape[-1], hcd.shape[-1]
        lmax = (L - 1) + (M - 1)
        U = lmax // 2 + 1

        qp = (other.pab[None, None, :, :, :] - self.pab[:, :, None, None, :]).reshape(-1)  # (B*nd,)
        N = qp.shape[0]

        qp_pows = ao.ones((N, lmax + 1))
        if lmax:
            qp_pows[:, 1] = qp
            for k in range(2, lmax + 1):
                qp_pows[:, k] = qp_pows[:, k - 1] * qp

        inv = one / ao.broadcast_to(deltas[:, None], (B, nd)).reshape(-1)  # (B*nd,)
        inv_pows = ao.ones((N, lmax + 2))
        if lmax + 1:
            inv_pows[:, 1] = inv
            for k in range(2, lmax + 2):
                inv_pows[:, k] = inv_pows[:, k - 1] * inv

        # (l,m) anti-diagonals using arange_max slices (assumed large enough)
        lm = []
        for s in range(lmax + 1):
            l0 = max(0, s - (M - 1))
            l1 = min(L - 1, s)
            l = arange_max[l0 : l1 + 1]
            m = (s - l)
            lm.append((l, m))

        ret = ao.zeros((N, lmax + 2))

        for u in range(U):
            for s in range(2 * u, lmax + 1):
                l, m = lm[s]
                w = _parity[(m + u) % 2]  # (P,)

                diag = (hab[:, :, l][:, None, :, :] * hcd[:, :, m][None, :, :, :] * w[None, None, None, :]).sum(-1)
                ret[:, s - u] += diag.reshape(-1) * binom2s[s, u] * qp_pows[:, s - 2 * u]

        ret *= inv_pows
        return ret.reshape(Na, Nb, Nc, Nd, nd, lmax + 2)


    def calc_c_array_v3(self, other) -> ArrType:
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        nd = self.nd
        Nab, Ncd = Na * Nb, Nc * Nd
        B = Nab * Ncd

        if hasattr(self, "deltas"):
            deltas = self.deltas
        else:
            gab = self.gamma.reshape(-1)          # (Nab,)
            gcd = other.gamma.reshape(-1)         # (Ncd,)
            deltas = (0.25 * (one / gab[:, None] + one / gcd[None, :])).reshape(-1)  # (B,)

        hab = self.hab.reshape(-1, *self.hab.shape[2:])      # (Nab, nd, L)
        hcd = other.hab.reshape(-1, *other.hab.shape[2:])    # (Ncd, nd, M)
        L, M = hab.shape[-1], hcd.shape[-1]
        lmax = (L - 1) + (M - 1)
        S = lmax + 1
        U = lmax // 2 + 1

        qp = (other.pab[None, None, :, :, :] - self.pab[:, :, None, None, :]).reshape(-1)  # (B*nd,)
        N = qp.shape[0]

        # qp^k for k=0..lmax
        srng = arange_max[:S]
        qp_pows = ao.pow(qp[:, None], srng[None, :])  # (N, S)

        # parity-folded hcd variants (u%2 selects which one)
        mrng = arange_max[:M]
        hcd0 = hcd * _parity[(mrng % 2)][None, None, :]
        hcd1 = hcd * _parity[((mrng + 1) % 2)][None, None, :]

        # anti-diagonal gather map: m = s - l
        lrng = arange_max[:L]
        m_map = (srng[:, None] - lrng[None, :]) # (S, L)
        mmask = (m_map >= 0) & (m_map < M)
        m_idx = ao.where(mmask, m_map, 0) # (S, L)

        # diag convs: (Nab,Ncd,nd,S) -> (N,S) after flatten
        hab_e = hab[:, None, :, None, :]                           # (Nab,1,nd,1,L)
        mm_e = mmask[None, None, None, :, :]                       # (1,1,1,S,L)

        diag0 = (hab_e * hcd0[:, :, m_idx][None, :, :, :, :] * mm_e).sum(-1).reshape(-1, S)  # (N,S)
        diag1 = (hab_e * hcd1[:, :, m_idx][None, :, :, :, :] * mm_e).sum(-1).reshape(-1, S)  # (N,S)

        # build all (s,u) contributions in one shot, then scatter to i=s-u
        urng = arange_max[:U]
        upar = (urng % 2) # (U,)
        diag = ao.transpose(ao.stack([diag0, diag1], axis=0)[upar],(1, 2, 0))  # (N,S,U)

        s2 = srng[:, None]
        u2 = urng[None, :]
        e = (s2 - 2 * u2) # (S,U)
        i = (s2 - u2) # (S,U)
        emask = e >= 0
        e_idx = ao.where(emask, e, 0)

        qp_e = qp_pows[:, e_idx] * emask[None, :, :]                # (N,S,U)
        facts = binom2s[s2, u2]                                     # (S,U)
        vals = diag * facts[None, :, :] * qp_e                      # (N,S,U)

        emflat = emask.reshape(-1)
        isel = i.reshape(-1)[emflat] # (T,)
        vsel = vals.reshape(N, -1)[:, emflat]                                 # (N,T)

        ret = ao.zeros((N, lmax + 2))
        ret.scatter_add_(dim=1, index=ao.broadcast_to(isel[None, :], vsel.shape), src=vsel)

        d2 = ao.broadcast_to(deltas[:, None], (B, nd)).reshape(-1)
        ret *= ao.pow((one / d2)[:, None], arange_max[: lmax + 2][None, :])

        return ret.reshape(Na, Nb, Nc, Nd, nd, lmax + 2)


    def setup_c_array(self, other):
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        N1, N2 = Na*Nb, Nc*Nd
        B = N1 * N2
        hab, hcd = self.hab, other.hab
        hab = hab.reshape(-1, *hab.shape[2:]) # (Na*Nb,nd,L)
        hcd = hcd.reshape(-1, *hcd.shape[2:]) # (Nc*Nd,nd,M)
        hab = ao.transpose(hab, [1, 0, 2]) # (nd,N1,L)
        hcd = ao.transpose(hcd, [1, 0, 2]) # (nd,N2,M)

        gab, gcd = self.gamma.reshape(-1), other.gamma.reshape(-1) # (Na*Nb), (Nc*Nd)
        deltas = (0.25 * (one / gab[:,None] + one / gcd[None,:])) #.reshape(-1)

        qp = other.pab[None,None,:,:,:] - self.pab[:,:,None,None,:] 
        qp = qp.reshape(N1, N2, self.nd) # (N1,N2,nd)
        qp = ao.transpose(qp, [2,0,1]) # (nd,N1,N2)
        return hab, hcd, qp, deltas

    def calc_c_array_v4(self, other):
        from pyquante22.basis.c_array_funcs_v2 import compute_chatgpt2
        h1, h2, qp, delta = self.setup_c_array(other)

        cis = [None,] * self.nd
        for id in range(self.nd):
            h1i, h2i, qpi = h1[id], h2[id], qp[id]
            cis[id] = compute_chatgpt2(h1i, h2i, qpi, delta, binom2s)

        return cis


# #end class CartGtoPrimPair:


if __name__ == "__main__":
    from pyquante22.basis.cart_gto_orig import CartGtoPrim as CGP0
    from pyquante22.basis.cart_gto_orig import CartGtoPrimPair as CPP0

    def tolegacy(g):
        if isinstance(g, CartGtoPrim):
            aas, ams, rrs = g.alpha, g.amom, g.cen
            ams = ams.reshape(-1)
            rrs = rrs.reshape(-1)
            aas = ao.tile(aas.reshape(-1,1), 3).reshape(-1)
            return CGP0(ams, rrs, aas)
        elif isinstance(g, CartGtoPrimPair):
            return tolegacy(g.ga) * tolegacy(g.gb)
        raise NotImplementedError

    n = 1
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    amx = ao.arr([-2, 0, 0], dtype=ao.int_dtype)
    amp = ao.arr([1, 0, 0], dtype=ao.int_dtype)
    amn = ao.arr([n, n+1, n+3], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    aa = ao.arr([1.0] * 3)
    gn = CartGtoPrim(amn, r0, aa)
    gs = CartGtoPrim(ams, r0, aa)
    gx = CartGtoPrim(amx, r0, aa)
    gsb = CartGtoPrim(ams, rb, aa)
    gp = CartGtoPrim(amp, r0, aa)
    gnb = CartGtoPrim(amn, rb, aa)
    gmb = CartGtoPrim(amn+1, rb, aa)
    gob = CartGtoPrim(amn+2, rb, aa)
    mys1 = str(gx)

    gpp = CartGtoPrimPair(gn, gnb)
    gl = tolegacy(gpp)

    diffs = gpp.binom_pres.squeeze() - gl.binom_pres
    mxdiff = ao.amax(ao.abs(diffs))
    assert mxdiff == zero

    gas = [gs,  gsb, gnb, gnb, gnb]
    gbs = [gp,  gn,  gnb, gmb, gob]
    gpp = CartGtoPrimPair(sum(gas), sum(gbs))
    gqq = CartGtoPrimPair(sum(gbs[:-1]), sum(gas[:-2]))
    # gl = tolegacy(gpp) # binom pres are the diagonals?

    
    for ib, gb in enumerate(gbs):
        gpb = CartGtoPrimPair(sum(gas), sum([gb,]*len(gas)))
        gl = tolegacy(gpb)

        # check binom_pres
        bpl = gl.binom_pres
        ls = bpl.shape[1]
        bps = gpp.binom_pres[:,ib,:,:ls]
        extra = gpp.binom_pres[:,ib,:,ls:]
        assert zero == ao.sum(extra)
        diffs = bps.reshape(-1) - bpl.reshape(-1)
        mxdiff = ao.amax(ao.abs(diffs))
        assert mxdiff == zero, f"fail bps @ {str(gb)}"

        # check h_array
        hl = gl.hab
        ls = hl.shape[1]
        hs = gpp.hab[:,ib,:,:ls]
        extra = gpp.hab[:,ib,:,ls:]
        assert zero == ao.sum(extra)
        diffs = hs.reshape(-1) - hl.reshape(-1)
        mxdiff = ao.amax(ao.abs(diffs))
        assert mxdiff == zero, f"fail hab @ {str(gb)}"


    import itertools
    def ndindex(shape):
        return itertools.product(*[range(n) for n in shape]) # generator

    loops = 100
    g1 = CartGtoPrimPair(sum(gas*1), sum(gbs[:-1]*2))
    g2 = CartGtoPrimPair(sum(gas[:-2]*2), sum(gbs[:-1] + gas[:3]))


    gaslong = gas*10
    gbslong = gbs*10
    times = []
    for ii in range(1,7):
        gab = sum(gaslong[:ii]) * sum(gbslong[:ii])
        gcd = sum(gbslong[:ii]) * sum(gaslong[:ii])
        t0 = time.time()
        for _ in range(loops):
            ans = gab.calc_c_array_v4(gcd)
        t1 = time.time()
        print(f"time for {loops} loops with npf={ii:02d}: {t1-t0:.2f}s")
        times.append([ii,t1-t0])
    
    # tmp = g1.debug_c_calc(g2)
    # tmp = ((gn+gsb)*gmb).setup_c_calc(gs*(gn+gp+gs))
    # tmp = g1[-1,-1].setup_c_calc(g2[-1,-1])
    # tmp = g1[0,0].setup_c_calc(g2[0,0])
    t0 = time.time()
    for _ in range(loops):
        cpp = g1.calc_c_array(g2)
    t1 = time.time()
    print(f"Original: {t1-t0:.4f}s")

    t0 = time.time()
    for _ in range(loops):
        cpp = g1.calc_c_array_v2(g2)
    t1 = time.time()
    print(f"v2: {t1-t0:.4f}s")

    t0 = time.time()
    for _ in range(loops):
        cpp = g1.calc_c_array_v3(g2)
    t1 = time.time()
    print(f"v3: {t1-t0:.4f}s")

    t0 = time.time()
    for _ in range(loops):
        cpp = g1.calc_c_array_v4(g2)
    t1 = time.time()
    print(f"v4: {t1-t0:.4f}s")

    cpp = gpp.calc_c_array_v3(gqq)
    for ia, ib in ndindex(cpp.shape[0:2]):
        gab = gpp[ia, ib]
        gabl = tolegacy(gab)
        for ic, id in ndindex(cpp.shape[2:4]):
            gcd = gqq[ic, id]
            gcdl = tolegacy(gcd)
            cl = gabl.calc_c_array(gcdl)
            ls = cl.shape[1]
            cij = cpp[ia,ib,ic,id,:,:ls]
            extra = cpp[ia,ib,ic,id,:,ls:]
            assert zero == ao.sum(ao.abs(extra))
            diffs = cij.reshape(-1) - cl.reshape(-1)
            mxdiff = ao.amax(ao.abs(diffs))
            assert mxdiff == zero, f"fail Cij @ ({ia},{ib},{ic},{id})"


            debug_var = 1




    debug_var = 1

    # def calc_binoms(self, pab, offset=None):
    #     """Compute binomial expansion coeffs given a center.
    #     pab:    (B,nd) or (B,C,nd)
    #     offset: (nd,) optional; shifts angular momentum (used in kinetic)
    #     returns: (B,nd,L) or (B,C,nd,L) where L = lmax+1
    #     ret[..., i] = binom(l, i) * (pab-cen)^(l-i)
    #     """
    #     ao, co, gm = self.ao, self.co, self.gm
    #     keepC = (pab.ndim == 3)
    #     pab3  = pab if keepC else pab[:, None, :]          # (B,C,nd)
    #     B, C, nd = pab3.shape
    #     assert (B == len(self)) and (self.nd == nd), f"P_ab size mismatch"

    #     if offset is None: # l: (B,nd), apply offset if present
    #         l = self.amom
    #         lmax = self.lmax
    #     else:
    #         l = self.amom + offset[None,:]
    #         lmax = int(max(0, l.max().item()))

    #     L = lmax + 1
    #     irng = co.arange_max[:L]                            # (L,)
    #     xG = (pab3 - self.cen[:, None, :]).reshape(-1)      # (G,)
    #     lG = ao.broadcast_to(l[:, None, :], (B, C, nd)).reshape(-1) # (G,)
    #     xpow = gm.pow_ladder(xG, L) # xG[g]^k, k=0..L-1;    # (G,L)
    #     l_safe = lG #ao.where(lG < 0, co.izero, lG)             # (G,)
    #     exp   = l_safe[:, None] - irng[None, :]             # (G,L)
    #     valid = (lG[:, None] >= 0) & (exp >= 0)             # (G,L)
    #     exp[exp<0] = co.izero

    #     retG = ao.take_along_axis(xpow, exp, axis=1)
    #     retG *= co.binoms[l_safe[:, None], irng[None, :]]
    #     retG = ao.where(valid, retG, co.zero)
    #     ret = retG.reshape(B, C, nd, L)
    #     return ret if keepC else ret[:, 0, :, :]
