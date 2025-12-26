# import os, time
from hashlib import sha256
import pyquante22.utils.math as me  # math exact for combinatorials

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
eri_fac = (2 * one) * pow(pi, 2.5 * one)
_parity = ao.arr([1, -1])  # (-1)^t = _parity[t%2]

amax = 20  # s-type to y-type, highest I've seen is i-type in def2-universal.gbs
arange_max = ao.arange(6 * amax + 1)  # 2 for pairs, 3 for 3d space
# facts = ao.arr([me.fact_exact(a) for a in range(2*amax+1)])
binoms = ao.arr(
    [
        [me.binom_exact(n, k) for k in range(2 * amax + 1)]
        for n in range(2 * amax + 1)
    ]
)
binom2s = ao.arr(
    [
        [me.binom2_exact(n, k) for k in range(1 * amax + 1)]
        for n in range(2 * amax + 1)
    ]
)
binom22s = ao.arr( # this is too big, it's 141kB for amax=20
    [
        [[me.binom22_exact(i, r, u) for u in range(1 * amax + 1)]
        for r in range(1*amax+1)]
        for i in range(2 * amax + 1)
    ]
)
# we could calculate it from the binom's formula (see bion22_exact docstring)
# but that formula has bare factorials, saying it's probably got
# round off error, meaning we need to precompute once, then same problem...
# other solution is to reformulat the NAI math to not need it, which
# must be possible since ERI doesn't need it and it has more centers


# ghalfs = ao.arr([me.gamma_half_int_exact(n) for n in range(2*amax+1)])
def dfact2m1o2(n):
    """double factorial of 2n-1 over 2**n"""
    return ao.exp(ao.lgamma(n + 0.5 * one)) / ao.sqrt(pi)


ghalfs = dfact2m1o2(ao.arr(arange_max[: 2 * amax + 1]))  # cast to default_dtype


def fboys(m, x):  # stable @ x=0
    # expansion around x=0: (1 / (2*m + 1)) - (x / (2*m + 3)) + (x**2 / 6*(2*m + 5))
    m2 = m + 0.5 * one
    ret = 0.5 * pow(x, -m2) * ao.gamma_inc(m2, x)
    mask = x < ao.xp.finfo(x.dtype).eps
    mask = ao.broadcast_to(mask, ret.shape)
    # ret[:] = ao.where(mask, 0.5/m2, ret) # @TODO: optimize for broadcasting with m and x being different sizes
    ret[mask] = ao.broadcast_to(0.5 / m2, ret.shape)[mask]
    return ret


def ndindex(shape):
    import itertools
    return itertools.product(*[range(n) for n in shape]) # generator


def fboys_arb(m, x): # @DEBUG with arbitrary precision math
    import pyquante22.arb_math as arb
    full_sz = tuple(max(a,b) for a,b in zip(m.shape,x.shape))
    mb, xb = ao.broadcast_to(m, full_sz), ao.broadcast_to(x, full_sz)
    ret = ao.zeros(full_sz)
    for idx in ndindex(full_sz):
        ret[idx] = ao.arr(float(arb.fboys(mb[idx].item(), xb[idx].item())))
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

def triple_sum_dot(ax, ay, az, fxyz):
    # for debugging
    totals = ao.zeros(ax.shape[0])
    for ii in range(ax.shape[1]):
        for jj in range(ay.shape[1]):
            for kk in range(az.shape[1]):
                totals += ax[:,ii] * ay[:,jj] * az[:,kk] * fxyz[:,ii+jj+kk]
    return totals


class CartGtoPrim:
    """Cartesian GTO Primitive"""

    def __init__(self, amom: ArrType, cen: ArrType, alpha: ArrType, nd: int = 3):
        if nd != 3:
            raise NotImplementedError("Only 3d GTOs currently implemented")
        self.nd = nd  # spatial dimension
        self.amom = amom  # int (batch*nd,)
        self.cen = cen  # float (batch*nd,)
        self.alpha = alpha  # float (batch*nd,) currently just repeated
        self.lmax = self.amom.max()

    def __len__(self):
        return len(self.amom)

    def __mul__(self, other):
        return CartGtoPrimPair(self, other)

    def __add__(self, other):
        """create a larger batch (overload __and__? ga & gb)"""
        return CartGtoPrim(
            ao.concat([self.amom, other.amom]),
            ao.concat([self.cen, other.cen]),
            ao.concat([self.alpha, other.alpha]),
        )

    def __radd__(self, other):  # handle sum([list_of_gtos])
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __repr__(self):
        ret = f"{type(self).__name__} (Lx,Ly,Lz) (Rx,Ry,Rz) Exp"
        for ii in range(len(self) // self.nd):
            i0, i1 = ii * self.nd, (ii + 1) * self.nd
            ll = " ".join(f"{v.item():2d}" for v in self.amom[i0:i1])
            xx = " ".join(f"{v.item():12.4f}" for v in self.cen[i0:i1])
            ee = f"{self.alpha[i0].item():12.4e}"
            ret += f"\n{ll}  {xx}  {ee}"
        return ret

    def hash(self, digits=8):
        return sha256(repr(self).encode('utf-8')).hexdigest()[:digits]

    def __str__(self):
        if len(self) <= self.nd:
            ams = ",".join([f"{v}" for v in self.amom.tolist()])
            rxs = ",".join([f"{v:.2f}" for v in self.cen.tolist()])
            aas = ",".join([f"{v:.2e}" for v in self.alpha[:: self.nd].tolist()])
        else:

            def summarize_pairs(keys, counts, fmt="{k}x{c}", limit=4):
                s = ",".join(
                    fmt.format(k=k, c=c)
                    for k, c in zip(keys[:limit], counts[:limit])
                )
                return s + ("..." if len(keys) > limit else "")

            # amom summary
            aas, acs = ao.unique(
                self.amom.reshape(-1, self.nd).sum(axis=1), return_counts=True
            )
            ams = summarize_pairs(ao.tolist(aas), ao.tolist(acs))
            # cen summary
            rus, rcs = ao.unique(
                self.cen.reshape(-1, self.nd), axis=0, return_counts=True
            )
            rxs = summarize_pairs(['A', 'B', 'C', 'D'], ao.tolist(rcs))
            if len(rus) == 1:
                rxs = ",".join(f"{v:.2f}" for v in ao.tolist(rus[0]))
            # alpha summary
            aas = ao.unique(self.alpha[:: self.nd])
            aas = (
                ao.tolist(aas)[0]
                if len(aas) == 1
                else f"{ao.amin(self.alpha)}...{ao.amax(self.alpha)}"
            )
        return f"GTO[{len(self)//self.nd}]_{{{ams}}}@({rxs};{aas})[{self.hash()}]"

    # end def __str__(self):

    def __getitem__(self, idx):  # lra ordering
        id0 = idx * self.nd
        id1 = id0 + self.nd
        ids = slice(id0, id1)
        ret = self.amom[ids], self.cen[ids], self.alpha[ids]
        # if tolist:
        #     ret = [ao.tolist(r) for r in ret]
        #     ret[2] = ret[2][0] # only take 1 alpha if tolist
        return ret

    def calc_binoms(self, pab: ArrType, offset=None) -> ArrType:
        """compute the binomial expansion coefficients given a center
        offset: (3,) will change the amom for the calculation
        """
        if offset is not None:
            bsz = len(self) // len(offset)  # error check?
            amom = self.amom + ao.tile(offset, bsz)
            lmax = amom.max()
        else:
            amom = self.amom
            lmax = self.lmax
        irng = arange_max[: lmax + 1]
        pows = amom[:, None] - irng[None, :]
        pows[pows < 0] = zero  # @TODO optimize with relu
        ret = (
            ao.pow((pab - self.cen)[:, None], pows)
            * binoms[amom[:, None], irng[None, :]]
        )
        mask = (amom < 0)[:, None]  # zero out negative amom (for kinetic)
        ret[ao.broadcast_to(mask, ret.shape)] = zero
        return ret

    # end def calc_binoms()


# end class CartGtoPrim():


CGP = CartGtoPrim


class CartGtoPrimPair:
    """Pairs of CartGtoPrim for batch'd eval of integrals"""

    def __init__(self, ga: CGP, gb: CGP):
        if len(ga) != len(gb):
            raise ValueError(
                "::ERR::(CartGtoPrimPair) pairs must be same batch size"
            )
        if ga.nd != gb.nd:
            raise ValueError(
                "::ERR::(CartGtoPrimPair) pairs must be same spatial dim"
            )
        self.nd = ga.nd
        self.ga = ga
        self.gb = gb
        self.gamma = ga.alpha + gb.alpha  #
        self.pab = self.calc_product_centers()  # combined center, P_{AB}
        self.binom_pres = self.calc_prod_binoms()  # binomial prefactors
        self.hab = self.calc_h_array()

    def __len__(self):
        return len(self.ga)

    def __repr__(self):
        rga = "A: " + repr(self.ga)
        rgb = "B: " + repr(self.gb)
        return f"{rga}\n{rgb}"

    def __str__(self):
        pre = f"GTO[{len(self)//self.ga.nd}]_"
        sga = str(self.ga).replace(pre, "")
        sgb = str(self.gb).replace(pre, "")
        return f"PairGTO[{len(self)}]{sga}//{sgb}"

    def __or__(self, other):
        # override for pretty ERI: (a*b|c*d)
        # NAI: (a*b|R)
        if isinstance(other, ArrType):
            raise NotImplementedError()
        if isinstance(other, type(self)):
            return self.calc_eri(other)

    def hash(self, digits=8):
        return sha256(repr(self).encode('utf-8')).hexdigest()[:digits]

    def calc_product_centers(self) -> ArrType:
        ga, gb = self.ga, self.gb
        # return (ga.alpha*ga.cen + gb.alpha*gb.cen) / self.gamma
        aa = ga.alpha / self.gamma  # scale for numerical stability
        bb = gb.alpha / self.gamma
        return aa * ga.cen + bb * gb.cen

    def calc_prod_binoms(self) -> ArrType:
        """calculate binomial coefficents for the product of GTOs"""
        # treat xyz as batch
        self.abinoms = self.ga.calc_binoms(self.pab)
        self.bbinoms = self.gb.calc_binoms(self.pab)
        return ao.conv_full_batch(self.abinoms, self.bbinoms)

    def calc_overlap1d(self, offset=None):
        if offset is not None:
            # @TODO optimization: only calculate the evens instead of slicing
            bbinoms = self.gb.calc_binoms(self.pab, offset=offset)
            bps = ao.conv_full_batch(self.abinoms, bbinoms)[:, ::2]
        else:
            bps = self.binom_pres[:, ::2]  # only take even factors
        irng = arange_max[: bps.shape[1]][None, :]
        totals = bps * ghalfs[irng] * ao.pow(self.gamma[:, None], -irng)
        return ao.sum(totals, axis=1)

    def calc_overlap(self, offset=None):
        """calculate the overlap of the pair
        nd: spatial dimensions flattened into the batch, may not work currently if nd != 3
        offset: optional ang. mom. offset, must be (nd,)
        return: overlaps of (len(self)//nd,)
        """
        if len(self) % self.nd != 0:
            raise ValueError(
                "::ERR::(calc_overlap) batch size must be a multiple of 3"
            )
        bsz = (len(self) // self.nd, self.nd)  # batch size (shape tuple)
        ga, gb = self.ga, self.gb
        ret = ao.sqrt(pi / self.gamma) * ao.exp(
            -ga.alpha * gb.alpha * ao.square(ga.cen - gb.cen) / self.gamma
        )
        ret = ret.reshape(bsz)  # ss overlap term, split into spatial dims
        ret *= self.calc_overlap1d(offset=offset).reshape(bsz)  # 1D overlaps
        return ao.prod(ret, axis=1)

    # end def calc_overlap(self, nd=3, offset=None):

    def calc_kinetic(self):
        """calculate the kinetic operator expectation for the pair (THO Eq. 2.14)
        logic is hardcoded for nd=3
        return: values of (len(self)//nd,)
        """
        # @TODO optimization:
        # the 'offset' feature in calc_overlap() is minimal extra alloc's,
        # but we're still recalculating the same values for the dimensions
        # that aren't being changed. We could either store the intermediates
        # of calc_overlap1d, or pass an offset list and calc the whole offset
        # list. Would need to return as a list or extra dimension, or we could
        # pass an additional set of weights for collapsing the sum...
        nd = self.nd
        ga, gb = self.ga, self.gb
        bsz = (len(self) // nd, nd)
        amb = gb.amom.reshape(bsz)
        ab = gb.alpha[::nd]
        offsets = 2 * ao.eye(nd, dtype=ga.amom.dtype)

        term0 = ab * (2 * ao.sum(amb, axis=1) + 3) * self.calc_overlap()

        term1 = self.calc_overlap(offset=offsets[0])
        term1 += self.calc_overlap(offset=offsets[1])
        term1 += self.calc_overlap(offset=offsets[2])
        term1 *= -2 * ao.square(ab)

        l2 = gb.amom[0::nd]
        term2 = l2 * (l2 - 1) * self.calc_overlap(offset=-offsets[0])
        l2 = gb.amom[1::nd]
        term2 += l2 * (l2 - 1) * self.calc_overlap(offset=-offsets[1])
        l2 = gb.amom[2::nd]
        term2 += l2 * (l2 - 1) * self.calc_overlap(offset=-offsets[2])
        term2 *= -half

        return term0 + term1 + term2
    # end def calc_kinetic(gpair, nd=3):


    def calc_a_array(self, rcen:ArrType) -> ArrType:
        """ THO eq. 2.18 and 3.1 
        flipi : flip the I index of the returned tensor (for efficient convolution)
        """
        dPn = self.pab - rcen
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
            binom22s[irng,rrng,urng]

        # now I need to mask and then do the weird sum
        imask = (rrng <= ao.floor(irng/2)) & \
            (urng <= ao.floor((irng-2*rrng)/2))

        sum_idx = irng - 2*rrng - urng
        sum_idx[~imask] = full_sz[1] # dummy index @TODO check if always 0 and optimize this out
        # sum_idx_batch = sum_idx.view(-1).expand(gs.shape[0], -1) # expand to batch
        terms_flat = terms.reshape(full_sz[0], -1)
        sum_idx_batch = ao.broadcast_to(sum_idx.reshape(-1), (full_sz[0], -1))
        ret = ao.zeros((full_sz[0], full_sz[1]+1))
        ret.scatter_add_(dim=1, index=sum_idx_batch, src=terms_flat)

        ret = ret[:, :-1] # get rid of the dummy sum
        mask = (abmom < irng)[:,:,0,0] # invalids, squeeze
        ret[:] = ao.where(mask, ao.arr(0), ret)
        return ret


    def calc_nai(self, rcen: ArrType) -> ArrType:
        if len(self) != len(rcen):
            raise ValueError("::ERR::(calc_nai) Batch size mismatch")

        nd = self.nd
        bsz = len(self) // nd
        ra, rb = self.ga.cen.reshape(bsz, nd), self.gb.cen.reshape(bsz, nd)
        a_array = self.calc_a_array(rcen).reshape(bsz, nd, -1).transpose(0, 1)

        aa, ab, gab = self.ga.alpha[::nd], self.gb.alpha[::nd], self.gamma[::nd]
        rab2 = ao.sum(ao.square(ra - rb), axis=1) # @TODO: reused in ERI, make a self attr
        rnp2 = ao.sum(ao.square((rcen - self.pab).reshape(bsz, nd)), axis=1)

        fbx = gab * rnp2
        ijkmax = 3 * a_array.shape[-1]  # @TODO get actual max needed
        boys = fboys(arange_max[: ijkmax + 1][None, :], fbx[:, None])
        ret = triple_conv_dot(a_array[0], a_array[1], a_array[2], boys)
        ret *= -2*pi/gab * ao.exp(-aa*ab*rab2/gab)
        return ret
    #end def calc_nai(self, rcen: ArrType) -> ArrType:


    def calc_h_array(self) -> ArrType:
        """v2 is fancier masking logic"""
        # bsz = len(self)
        llmax = (self.ga.amom + self.gb.amom).max().item()
        full_sz = (len(self), llmax + 1, llmax // 2 + 1)

        bps = self.calc_prod_binoms()[:, : full_sz[1], None]  # (3,llmax+1,1)
        g4s = ao.broadcast_to((4 * (self.gamma)).reshape(-1, 1, 1), full_sz)

        irng = arange_max[: full_sz[1]][:, None]
        rrng = arange_max[: full_sz[2]][None, :]
        irmask = rrng <= irng // 2
        # iflat = irng.expand(full_sz[1:])[irmask]
        # rflat = rrng.expand(full_sz[1:])[irmask]
        iflat = ao.broadcast_to(irng, full_sz[1:])[irmask]
        rflat = ao.broadcast_to(rrng, full_sz[1:])[irmask]
        iiflat = iflat - 2 * rflat

        # h_vals_flat = bps.expand(full_sz)[:,irmask] # irng > amom == 0 from bps
        h_vals_flat = ao.broadcast_to(bps, full_sz)[
            :, irmask
        ]  # irng > amom == 0 from bps
        h_vals_flat *= ao.pow(g4s[:, irmask], (rflat - iflat)[None, :])
        h_vals_flat *= binom2s[iflat, rflat]
        ret = ao.zeros(full_sz[:2])
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), h_vals_flat.shape)
        ret.scatter_add_(dim=1, index=sum_ids, src=h_vals_flat)
        # ret.index_add_(1, iiflat, h_vals_flat)
        return ret

    # end def calc_h_array(self, other):

    def calc_c_array(self, other: "CartGtoPrimPair") -> ArrType:
        # ga, gb, gc, gd = self.ga, self.gb, other.ga, other.gb
        hab, hcd = self.hab, other.hab
        qp = other.pab - self.pab
        if hasattr(self, "deltas"):
            deltas = self.deltas
        else:
            deltas = 0.25 * (one / self.gamma + one / other.gamma)
        abmax, cdmax = hab.shape[1] - 1, hcd.shape[1] - 1
        lmax = abmax + cdmax

        abrng = arange_max[: abmax + 1][:, None, None]
        cdrng = arange_max[: cdmax + 1][None, :, None]
        urng = arange_max[: lmax // 2 + 1][None, None, :]
        iirng = abrng + cdrng - urng  # (L,M,U)
        iurng = iirng - urng
        iumask = iurng >= 0

        iiflat = iirng[iumask]
        iuflat = iurng[iumask]
        # uflat = urng.expand(*iumask.shape[:2],-1)[iumask]
        uflat = ao.broadcast_to(urng, (*iumask.shape[:2], urng.shape[-1]))[iumask]
        facts = binom2s[iiflat + uflat, uflat]  #
        # parity = _parity((cdrng+urng)%2).expand(iumask.shape[0],-1,-1)[iumask]
        parity = ao.broadcast_to(
            _parity[(cdrng + urng) % 2],
            (iumask.shape[0], cdrng.shape[1], urng.shape[2]),
        )[iumask]

        c_vals_flat = ao.pow(qp[:, None], iuflat[None, :])
        # c_vals_flat *= ((hab[:,:,None]*hcd[:,None,:])[:,:,:,None]
        #     .expand(-1,-1,-1,iumask.shape[-1])[:,iumask])
        c_vals_flat *= ao.broadcast_to(
            (hab[:, :, None] * hcd[:, None, :])[:, :, :, None],
            (*hab.shape[:2], hcd.shape[1], iumask.shape[-1]),
        )[:, iumask]
        c_vals_flat *= (facts * parity)[None, :]

        ret = ao.zeros((len(self), lmax + 2))
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), c_vals_flat.shape)
        ret.scatter_add_(dim=1, index=sum_ids, src=c_vals_flat)
        # ret.index_add_(1, iiflat, c_vals_flat)
        tmp3 = ao.pow(deltas[:, None], -arange_max[: lmax + 2])
        ret *= tmp3
        return ret

    # end def calc_c_array(self, other:"CartGtoPrimPair") -> ArrType:

    def calc_eri(self, other) -> ArrType:
        if len(self) != len(other):
            raise ValueError("::ERR::(calc_eri) Batch size mismatch")

        nd = self.nd
        bsz = len(self) // nd
        gab, gcd = self.gamma[:], other.gamma[:]
        # tmp = ao.empty(bsz)
        # self.deltas = 0.25 * (one/self.gamma + one/other.gamma)
        self.deltas = 0.25 * (one / gab + one / gcd)
        c_array = (
            self.calc_c_array(other).view(bsz, nd, -1).transpose(0, 1)
        )  # .contiguous()

        # delta = o4 / gab + o4 / gcd
        qp2 = ao.pow(self.pab - other.pab, 2 * one).reshape(-1, nd).sum(axis=1)
        fbx = 0.25 * qp2 / self.deltas[::nd]
        ijkmax = 3 * c_array.shape[-1]  # @TODO get actual max needed
        boys = fboys(arange_max[: ijkmax + 1][None, :], fbx[:, None])
        ret = triple_conv_dot(c_array[0], c_array[1], c_array[2], boys)
        del fbx, boys

        aa, ab = self.ga.alpha[::nd], self.gb.alpha[::nd]
        ac, ad = other.ga.alpha[::nd], other.gb.alpha[::nd]
        ra, rb = self.ga.cen.view(bsz, nd), self.gb.cen.view(bsz, nd)
        rc, rd = other.ga.cen.view(bsz, nd), other.gb.cen.view(bsz, nd)
        rab2 = ao.sum(ao.square(ra - rb), dim=1) # @TODO: reused in NAI, make a self attr
        rcd2 = ao.sum(ao.square(rc - rd), dim=1)

        gab, gcd = gab[::nd], gcd[::nd]
        ret *= eri_fac / (gab * gcd * ao.sqrt(gab + gcd))
        ret *= ao.exp(-aa * ab * rab2 / gab)
        ret *= ao.exp(-ac * ad * rcd2 / gcd)
        return ret

    # end def calc_eri(self, other) -> ArrType:


# end class CartGtoPrimPair():


# def calc_kinetic2(gpair, nd=3):
#     """ formulas are hard coded to nd=3 """
#     ga, gb = gpair.ga, gpair.gb
#     bsz = (len(ga)//nd, nd)
#     amb = gb.amom.view(bsz)
#     ab = gb.alpha[::nd]
#     term0 = ab * (2*ao.sum(amb, axis=1) + 3) * \
#         gpair.calc_overlap()

#     offsets = 2 * ao.eye(nd, dtype=gpair.ga.amom.dtype)
#     term1  = gpair.calc_overlap(offset=offsets[0])
#     term1 += gpair.calc_overlap(offset=offsets[1])
#     term1 += gpair.calc_overlap(offset=offsets[2])
#     term1 *= -2 * ao.square(ab)

#     l2 = gb.amom[0::nd]
#     term2  = l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[0])
#     l2 = gb.amom[1::nd]
#     term2 += l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[1])
#     l2 = gb.amom[2::nd]
#     term2 += l2 * (l2-1) * gpair.calc_overlap(offset=-offsets[2])
#     term2 *= -half

#     return term0 + term1 + term2


debug_var = 1
if __name__ == "__main__":
    n = 2
    ams = ao.arr([0, 0, 0], dtype=ao.int_dtype)
    amx = ao.arr([-2, 0, 0], dtype=ao.int_dtype)
    amp = ao.arr([1, 0, 0], dtype=ao.int_dtype)
    amn = ao.arr([n, 0, 0], dtype=ao.int_dtype)
    r0 = ao.arr([0, 0, 0])
    rb = ao.arr([0.125, 0.25, 0.5])
    aa = ao.arr([1.0] * 3)
    gs = CartGtoPrim(ams, r0, aa)
    gx = CartGtoPrim(amx, r0, aa)
    gsb = CartGtoPrim(ams, rb, aa)
    gp = CartGtoPrim(amp, r0, aa)
    gn = CartGtoPrim(amn, r0, aa)
    gnb = CartGtoPrim(amn, rb, aa)
    mys1 = str(gx)

    gns = CartGtoPrimPair(gn, gs)
    tmp = (gnb*gsb).calc_nai(rb)
    tmp = gns.calc_overlap()
    mys2 = str(gn)

    from pyquante22.arb_math import (
        overlap_tho66 as aove,
        kinetic_tho66 as akin,
        nuclear_attraction as anai,
        coulomb_rep4 as aeri,
    )
    import pyquante22.arb_math as arb
    import mpmath as mpm

    def cgp2arb(gto: CartGtoPrim, id=0):
        l, r, a = gto[id]
        l = arb.arr_init(l.tolist())
        r = arb.arr_init(r.tolist())
        a = arb.arb(a[0].item())
        return a, l, r

    def cgpp2arb(gpp: CartGtoPrimPair, id=0):
        alr1 = cgp2arb(gpp.ga, id=id)
        alr2 = cgp2arb(gpp.gb, id=id)
        return *alr1, *alr2

    tmp = aove(*[*cgp2arb(gs)] * 2)
    assert mpm.almosteq(tmp, (arb.pi / 2) ** (3 / 2))  # (pi/2)**(3/2)
    tmp = akin(*[*cgp2arb(gs)] * 2)
    assert mpm.almosteq(tmp, 3 / 2 * (arb.pi / 2) ** (3 / 2))  # 3/2*(pi/2)**(3/2)
    tmp = anai(*[*cgp2arb(gs)] * 2, arb.arr_init(r0.tolist()))
    assert mpm.almosteq(tmp, -arb.pi)  # -pi
    tmp = aeri(*[*cgp2arb(gs)] * 4)
    assert mpm.almosteq(tmp, arb.pow(arb.pi, 5 / 2) / 4)  # pi**(5/2) / 4

    gxs = CartGtoPrimPair(gx, gs)
    tmp = gxs.calc_overlap()
    assert tmp == 0
    gt = aove(*cgpp2arb(gxs))
    assert float(gt) == tmp.item()

    gss = CartGtoPrimPair(gs, gs)
    gssb = CartGtoPrimPair(gs, gsb)
    gps = CartGtoPrimPair(gp, gs)

    expected = aeri(*cgpp2arb(gn * gnb), *cgpp2arb(gn * gsb))
    # result = (gn*gnb|gn*gsb)

    tmp = gn * gnb

    ans = gss.calc_nai(gs.cen)
    ans = (gn*gnb).calc_nai(gnb.cen)

    # tmp.binom_pres
    # 1/65536,0,-1/128,0,1
    # tensor([[ 1.5259e-05,  0.0000e+00, -7.8125e-03,  0.0000e+00,  1.0000e+00],
    # tmp.hab
    # 12161/65536,0,191/8192,1/4096
    # tensor([[1.8556e-01, 0.0000e+00, 2.3315e-02, 0.0000e+00, 2.4414e-04],

    import numpy as np
    from pyquante22.ref_ints.overlap import overlap1d_hh12 as p2ov1
    from pyquante22.ref_ints.overlap import overlap_hh12 as p2ov

    # from pyquante22.basis.pq2_one import overlap1d as p2ov1
    # from pyquante22.basis.pq2_one import overlap as p2ov
    # from pyquante22.basis.one_ints_pyquante2 import gto_overlap_3d as ov3d
    from pennylane.qchem.integrals import gaussian_overlap as plov

    # from pyquante22.basis.pq2_one import kinetic as p2t
    # from pyquante22.basis.one_ints_pyquante2 import gto_kinetic_3d as t3d
    from pennylane.qchem.integrals import gaussian_kinetic as plt

    tol = 1.0e-14

    if False:
        nr0 = np.asarray([0.0] * 3)
        nrx = np.asarray([0.0, 1.0, 2.0])
        nry = np.asarray([-0.1, 1.2, 1.75])
        na0 = np.asarray([0, 0, 0])
        na2 = np.asarray([2, 0, 0])
        nax = np.asarray([2, 3, 5])
        nay = np.asarray([4, 1, 3])

        a = p2ov(1.0, na0, nr0, 1.0, na0, nr0)
        b = ov3d(1.0, 1.0, nr0, nr0, na0, na0)
        assert np.abs((a - b) / a) < tol

        a = p2ov(1.0, na2, nr0, 1.0, na0, nr0)
        b = ov3d(1.0, 1.0, nr0, nr0, na2, na0)
        assert np.abs((a - b) / a) < tol

        a = p2ov(1.0, na2, nrx, 2.0, na0, nry)
        b = ov3d(1.0, 2.0, nrx, nry, na2, na0)
        assert np.abs((a - b) / a) < tol

        a = p2ov(1.0, nax, nrx, 2.0, nay, nry)  #
        b = ov3d(1.0, 2.0, nrx, nry, nax, nay)
        c = plov(nax, nay, nrx, nry, 1.0, 2.0)

        arx = ao.arr(nrx)
        ary = ao.arr(nry)
        aax = ao.arr(nax, dtype=ao.int_dtype)
        aay = ao.arr(nay, dtype=ao.int_dtype)
        gx = CartGtoPrim(aax, arx, ao.arr([1.0] * 3))
        gy = CartGtoPrim(aay, ary, ao.arr([2.0] * 3))
        gxy = CartGtoPrimPair(gx, gy)
        gxx = CartGtoPrimPair(gx, gx)
        d = gxy.calc_overlap().item()
        assert np.abs((a - b) / a) < tol
        assert np.abs((a - c) / a) < tol
        assert np.abs((a - d) / a) < tol

        a = p2t(1.0, nax, nrx, 2.0, nay, nry)  #
        b = t3d(1.0, 2.0, nrx, nry, nax, nay)
        c = plt(nax, nay, nrx, nry, 1.0, 2.0)
        d = gxy.calc_kinetic().item()
        assert np.abs((a - b) / a) < tol
        assert np.abs((a - c) / a) < tol
        assert np.abs((a - d) / a) < tol

        # e = calc_kinetic2(gxy).item()

        # @TODO: batch into more than 1 3d primitive and try
        # add tests
        ga = gx + gy
        gb = gx + gx
        gaa = ga * ga
        gab = ga * gb
        tmp = CartGtoPrimPair(ga, ga)
        sga = str(ga)

    debug_var = 1
