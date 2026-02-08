from hashlib import sha256

import pyquante22.array_ops as arr
from pyquante22.basis.cart_gto_math import CachedOps, GtoMath


class CartGtoPrim:
    """Cartesian GTO Primitive (batched)
    """

    def __init__(self, amom, cen, alpha, coeffs=None,
                 nd=3, ao=None, co=None, gm=None, precompute=True):
        """
        :param amom: (B,nd) int, angular momentum
        :param cen: (B,nd) float, center
        :param alpha: (B,) float, Gaussian exponent
        :param nd: int, spatial dim count (must be 3)
        :param ao: ArrayOps instance
        :param co: CachedOps instance
        :param precompute: bool, whether to calc norm
        """
        if nd != 3:
            raise NotImplementedError("Only 3d GTOs currently implemented")
        self.ao = ao or arr.get_default_ops()
        self.nd = nd

        AT = self.ao.ArrayType
        assert (isinstance(amom, AT) and isinstance(cen, AT) and isinstance(alpha, AT)), \
            f"amom, cen, alpha must be {AT} types"

        # support getitem with 1 item still batch
        amom = self._cast_batch(amom, nd=self.nd, name="amom") 
        cen  = self._cast_batch(cen,  nd=self.nd, name="cen")
        B = amom.shape[0]
        if cen.shape[0] != B:
            raise ValueError("amom and cen must have the same batch size")

        # alpha: require (B,) but allow (B,1) or legacy (B*nd,)
        if getattr(alpha, "ndim", 0) == 0: # scalar
            alpha = self.ao.arr([alpha])
        if getattr(alpha, "ndim", None) == 2 and alpha.shape[1] == 1: # (B,1)
            alpha = alpha.reshape(-1)
        if getattr(alpha, "ndim", None) == 1 and len(alpha) == B * nd: # support legacy
            alpha = alpha.reshape(B, nd)[:, 0]

        self.amom = amom
        self.cen = cen
        self.alpha = alpha
        self.lmax = self.amom.max().item()
        self.co = co or CachedOps(self.ao, amax=max(self.lmax+2, 10))
        self.gm = gm or GtoMath(self.ao)
        self.coeffs = coeffs
        self.norm = self.calc_norm() if precompute else None
        if precompute and self.coeffs is not None:
            assert isinstance(self.coeffs, AT), f"coeffs must be {AT} type"
            if getattr(coeffs, "ndim", 0) == 0: # scalar
                coeffs = self.ao.arr([coeffs])
            self.coeffs = self.coeffs * self.norm


    @staticmethod
    def _cast_batch(x, nd: int, name: str):
        if getattr(x, "ndim", None) == 1:  # allow legacy flattened (B*nd,) input
            if len(x) % nd != 0:
                raise ValueError(f"{name} length must be a multiple of nd")
            return x.reshape(-1, nd)

        if getattr(x, "ndim", None) != 2 or x.shape[1] != nd:
            raise ValueError(f"{name} must be shape (B,{nd}) or (B*{nd},)")

        return x

    # def __getitem__(self, idx): # returns the same type
    #     return CartGtoPrim(self.amom[idx], self.cen[idx], self.alpha[idx])
    def __getitem__(self, idx): # returns the same type
        coeffs = None if self.coeffs is None else self.coeffs[idx]
        ret = CartGtoPrim(self.amom[idx], self.cen[idx],self.alpha[idx],
            coeffs=coeffs, nd=self.nd, ao=self.ao, co=self.co, gm=self.gm,
            precompute=False,)
        if self.norm is not None: ret.norm = self.norm[idx:idx+1]
        if self.coeffs is not None: ret.coeffs = self.coeffs[idx:idx+1]
        return ret

    def __len__(self):
        return int(self.amom.shape[0])

    def __mul__(self, other):
        return CartGtoPrimPair(self, other)

    def __add__(self, other):
        if self.nd != other.nd:
            raise ValueError("nd mismatch")
        ret = CartGtoPrim( 
            self.ao.concat([self.amom, other.amom], axis=0),
            self.ao.concat([self.cen, other.cen], axis=0),
            self.ao.concat([self.alpha, other.alpha], axis=0),
            nd=self.nd,
            ao=self.ao,
            co=self.co,
            gm=self.gm,
            precompute=False,
        )
        if self.coeffs is not None and other.coeffs is not None:
            ret.coeffs = self.ao.concat([self.coeffs, other.coeffs], axis=0)
        if self.norm is not None and other.norm is not None:
            ret.norm = self.ao.concat([self.norm, other.norm], axis=0)
        return ret

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __repr__(self):
        ret = f"{type(self).__name__} (Lx,Ly,Lz) (Rx,Ry,Rz) Exp"
        for i in range(len(self)):
            ll = " ".join(f"{int(v):2d}" 
                          for v in self.ao.tolist(self.amom[i]))
            xx = " ".join(f"{float(v):12.4f}" 
                          for v in self.ao.tolist(self.cen[i]))
            ee = f"{float(self.alpha[i]):12.4e}"
            ret += f"\n{ll}  {xx}  {ee}"
        return ret

    def hash(self, digits=8):
        return sha256(repr(self).encode("utf-8")).hexdigest()[:digits]

    def __str__(self):
        ao = self.ao
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


    def calc_norm(self):
        """Normalize basis function. From THO eq. 2.2, with alpha shared across xyz."""
        ao, co = self.ao, self.co
        norm2_1d = ao.pow(2 * self.alpha[:, None], 
                          self.amom + co.half)
        norm2_1d /= co.ghalfs[self.amom]
        norm2 = ao.prod(norm2_1d, axis=1)  # (B,)
        return ao.sqrt(norm2)


    def calc_binoms(self, pab, offset=None):
        """Compute binomial expansion coeffs given a center
        param: pab: (B,nd) or (B,C,nd)
        param: offset: (nd,) optional; change to amom (for kinetic)
        returns: (B,nd,Lmax+1) or (B,C,nd,Lmax+1)
        """
        ao, co = self.ao, self.co
        keepC = (pab.ndim == 3)
        pab  = pab if keepC else pab[:, None, :]     # (B,C,nd)

        if offset is not None:
            amom = (self.amom + offset[None,:])[:,None,:] # (B,1,nd)
            lmax = max(0, amom.max().item())
        else:
            amom = self.amom[:,None,:] # (B,1,nd)
            lmax = self.lmax

        irng = co.arange_max[:lmax + 1]  # (L,)
        pows = amom[..., None] - irng[None, None, None, :]  # (B,C,nd,L)

        pows[pows < 0] = co.izero  # 
        ret = co.binoms[amom[..., None], irng[None, None, None, :]]  * \
            ao.pow((pab - self.cen[:, None, :])[..., None], pows) # (B,C,nd,L)
        mask = (amom < 0)[..., None]  # (B,1,nd,1)
        ret = ao.where(mask, co.zero, ret)
        return ret if keepC else ret[:,0,:,:]
    #end def calc_binoms(self, pab: ArrType, offset=None) -> ArrType:

#end class CartGtoPrim:


CGP = CartGtoPrim # shorthand


class CartGtoPrimPair:
    """Pairs of CartGtoPrim for batch'd eval of integrals
    """

    def __init__(self, ga: CGP, gb: CGP, precompute=True):
        if ga.nd != gb.nd:
            raise ValueError("::ERR::(CartGtoPrimPair) pairs must be same spatial dim")
        self.nd = ga.nd
        self.ga = ga
        self.gb = gb
        self.ao = ga.ao
        assert self.ao is gb.ao, "CartGtoPrims must have same ArrayOps backend"
        self.co = ga.co # assume no amax issues

        if precompute:
            # @TODO precompute eri prefactors
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
        if other is self:
            return self.calc_eri_self()
        if isinstance(other, type(self)):
            return self.calc_eri(other)  #
        if hasattr(other, "shape"):
            return self.calc_nai(other)  # 
        raise NotImplementedError(f"No integral implemented for {type(other)}")


    def hash(self, digits=8):
        return sha256(repr(self).encode("utf-8")).hexdigest()[:digits]


    def calc_product_centers(self):
        """GTO product centers"""
        ga, gb = self.ga, self.gb
        # numerically stable form: P = (aa/g)*A + (bb/g)*B
        aa = ga.alpha[:, None] / self.gamma   # (Na,Nb)
        bb = gb.alpha[None, :] / self.gamma   # (Na,Nb)
        ra = ga.cen[:, None, :]               # (Na,1,3)
        rb = gb.cen[None, :, :]               # (1,Nb,3)
        return (aa[..., None] * ra) + (bb[..., None] * rb)  # (Na,Nb,3)


    def calc_prod_binoms(self):
        """GTO product binomial prefactors, returns: (Na, Nb, nd, Lp)"""
        ao, co = self.ao, self.co
        ga, gb, nd = self.ga, self.gb, self.nd
        Na, Nb, P = len(ga), len(gb), self.pab              # P:(Na,Nb,nd)
        self.abinoms = ga.calc_binoms(P)                    # (Na,Nb,nd,La)
        btmp = gb.calc_binoms(ao.transpose(P,(1, 0, 2)))    # (Nb,Na,nd,Lb)
        self.bbinoms = ao.transpose(btmp,(1, 0, 2, 3))      # (Na,Nb,nd,Lb)
        G = Na * Nb * nd
        pres = ao.conv_full_batch(self.abinoms.reshape(G, -1), self.bbinoms.reshape(G, -1))
        return pres.reshape(Na, Nb, nd, -1)

    
    def calc_overlap1d(self, offset=None):
        ao, co = self.ao, self.co
        if offset is not None:
            # @TODO optimization: only calculate the evens instead of slicing
            Na, Nb, P = len(self.ga), len(self.gb), self.pab              # P:(Na,Nb,nd)
            G = Na * Nb * self.nd
            otmp = self.gb.calc_binoms(ao.transpose(self.pab,(1, 0, 2)),
                offset=offset)    # (Nb,Na,nd,Lb)
            obinoms = ao.transpose(otmp,(1, 0, 2, 3))      # (Na,Nb,nd,Lb)
            ret = ao.conv_full_batch(self.abinoms.reshape(G, -1), 
                                     obinoms.reshape(G, -1))
            ret = ret.reshape(Na, Nb, self.nd, -1)[..., ::2]
        else:
            #@TODO: implement copy for ArrayOps (+0 is a hack)
            ret = self.binom_pres[..., ::2] + co.zero # only take even factors, force copy with +0
        irng = co.arange_max[: ret.shape[-1]][None,None,:]
        # pows = ao.pow(self.gamma[:,:,None], irng)
        # ret *= (co.ghalfs[irng] / pows)[:,:,None,:] # add spatial dim
        pows = ao.pow(self.gamma[:,:,None], -irng) # interstingly this is more accurate
        ret *= (co.ghalfs[irng] * pows)[:,:,None,:] # add spatial dim
        return ao.sum(ret, axis=3)


    def calc_overlap(self, offset=None):
        """calculate the overlap of the pair
        nd: spatial dimensions flattened into the batch, may not work currently if nd != 3
        offset: optional ang. mom. offset, must be (nd,)
        return: overlaps of (len(self)//nd,)
        """
        ao, co = self.ao, self.co
        ga, gb = self.ga, self.gb
        invgam = co.one / self.gamma # (Na,Nb)
        ret = ao.exp(ao.square(ga.cen[:,None,:] - gb.cen[None,:,:]) * 
            (-ga.alpha[:,None]*gb.alpha[None,:]*invgam)[:,:,None])
        ret *= ao.sqrt(invgam)[:,:,None]
        ret *= self.calc_overlap1d(offset=offset)  # 1D overlaps
        return ao.prod(ret, axis=2)
    # end def calc_overlap(self, nd=3, offset=None):


    def calc_kinetic(self):
        """calculate the kinetic operator expectation for the pair (THO Eq. 2.14)
        logic is hardcoded for nd=3
        return: values of (len(self)//nd,)
        """
        ao, co = self.ao, self.co
        ga, gb = self.ga, self.gb
        ab, amb = gb.alpha, gb.amom
        offsets = 2 * ao.eye(self.nd, dtype=ga.amom.dtype)

        two, three = 2*co.one, 3*co.one
        fac0 = ab * (two * ao.sum(amb,axis=1) + three)
        term0 = self.calc_overlap()
        term0 *= fac0[None,:]

        term1 = self.calc_overlap(offset=offsets[0])
        term1 += self.calc_overlap(offset=offsets[1])
        term1 += self.calc_overlap(offset=offsets[2])
        term1 *= (-two * ao.square(ab))[None,:]

        if gb.lmax > 1: # 
            l2 = gb.amom[:, 0][None,:] # (1,Nb)
            term2 = (l2 * (l2 - 1)) * self.calc_overlap(offset=-offsets[0])
            l2 = gb.amom[:, 1][None,:]
            term2 += (l2 * (l2 - 1)) * self.calc_overlap(offset=-offsets[1])
            l2 = gb.amom[:, 2][None,:]
            term2 += (l2 * (l2 - 1)) * self.calc_overlap(offset=-offsets[2])
            term2 *= -co.half

            return term0 + term1 + term2

        else:
            return term0 + term1
    # end def calc_kinetic(gpair, nd=3):



    def calc_h_array(self):
        """THO Eqn 3.5 (typo fixed)"""
        ao, co = self.ao, self.co
        bps = self.binom_pres[...,None] # (Na,Nb,nd,lmax)
        lmax = bps.shape[3] - 1
        full_sz = (len(self.ga), len(self.gb), self.ga.nd, lmax + 1, lmax // 2 + 1)
        g4s = ao.broadcast_to((4*self.gamma)[:,:,None,None,None], full_sz)

        irng = co.arange_max[: full_sz[3]][:, None]
        rrng = co.arange_max[: full_sz[4]][None, :]
        irmask = rrng <= irng // 2
        iflat = ao.broadcast_to(irng, full_sz[3:])[irmask]
        rflat = ao.broadcast_to(rrng, full_sz[3:])[irmask]
        iiflat = iflat - 2 * rflat

        h_vals_flat = ao.broadcast_to(bps, full_sz)[...,irmask]  # irng > amom == 0 from bps
        h_vals_flat *= ao.pow(g4s[...,irmask], (rflat - iflat)[None,None,None,:])
        h_vals_flat *= co.binom2s[iflat, rflat][None,None,None,:]

        ret = ao.zeros(full_sz[:-1])
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), h_vals_flat.shape)
        # ret.scatter_add_(dim=-1, index=sum_ids, src=h_vals_flat)
        ao.scatter_add_inplace(ret, -1, sum_ids, h_vals_flat)
        return ret
    # end def calc_h_array(self):

    def calc_a_array(self, rcen):
        """THO eq. 2.18 and 3.1 (batched).
        rcen: (C,nd) or (nd,)
        returns: (Na,Nb,C,nd,L)
        """
        # @TODO refactor to avoid large intermediate arrays
        ao, co = self.ao, self.co
        nd = self.nd

        if getattr(rcen, "ndim", None) == 1:
            if rcen.shape[0] != nd:
                raise ValueError("rcen must be shape (nd,) or (C,nd)")
            rcen = rcen[None, :]
        if getattr(rcen, "ndim", None) != 2 or rcen.shape[1] != nd:
            raise ValueError("rcen must be shape (nd,) or (C,nd)")

        Na, Nb = len(self.ga), len(self.gb)
        C = rcen.shape[0]

        # dPn: (Na,Nb,C,nd)
        dPn = self.pab[:, :, None, :] - rcen[None, None, :, :]

        # flatten all batch and spatial dims to reuse orig logic
        bps = self.binom_pres  # (Na,Nb,nd,L)
        llmax = bps.shape[3] - 1
        rumax = llmax // 2
        B = Na * Nb * C * nd

        bipres = ao.broadcast_to(
            bps[:, :, None, :, :], (Na, Nb, C, nd, llmax + 1)
        ).reshape(B, llmax + 1)
        gamma = ao.broadcast_to(
            self.gamma[:, :, None, None], (Na, Nb, C, nd)
        ).reshape(B)
        dPn_flat = dPn.reshape(B)

        irng = co.arange_max[: llmax + 1][None, :, None, None]
        rrng = co.arange_max[: rumax + 1][None, None, :, None]
        urng = co.arange_max[: rumax + 1][None, None, None, :]
        nrurng = -(rrng + urng)

        gpows = ao.pow((4 * gamma)[:, None, None, None], nrurng) # LARGE!
        cpows = ao.pow(dPn_flat[:, None, None, None], irng + 2 * nrurng) # LARGE!

        iuparity = co.parity[(irng + urng) % 2]
        terms = iuparity * bipres[:, :, None, None] * cpows * gpows * \
            co.binom22(irng, rrng, urng)

        imask = (rrng <= irng // 2) & (urng <= (irng - 2 * rrng) // 2)
        sum_idx = irng - 2 * rrng - urng
        sum_idx = ao.where(imask, sum_idx, co.arange_max[llmax + 1])

        terms_flat = terms.reshape(B, -1)
        sum_idx_batch = ao.broadcast_to(sum_idx.reshape(-1), (B, -1))
        ret = ao.zeros((B, llmax + 2))
        ao.scatter_add_inplace(ret, 1, sum_idx_batch, terms_flat)
        ret = ret[:, :-1]
        return ret.reshape(Na, Nb, C, nd, llmax + 1)


    def calc_nai(self, rcen):
        """Nuclear attraction integral (a*b|R), batched over R.
        rcen: (C,nd) or (nd,)
        returns: (Na,Nb,C)
        """
        ao, co = self.ao, self.co
        nd = self.nd

        if getattr(rcen, "ndim", None) == 1:
            if rcen.shape[0] != nd:
                raise ValueError("rcen must be shape (nd,) or (C,nd)")
            rcen = rcen[None, :]
        if getattr(rcen, "ndim", None) != 2 or rcen.shape[1] != nd:
            raise ValueError("rcen must be shape (nd,) or (C,nd)")

        Na, Nb = len(self.ga), len(self.gb)
        C = rcen.shape[0]

        a_array = self.calc_a_array(rcen)  # (Na,Nb,C,nd,L)
        B = Na * Nb * C
        a_flat = ao.transpose(a_array.reshape(B, nd, -1), (1, 0, 2))  # (nd,B,L)

        dpc = self.pab[:, :, None, :] - rcen[None, None, :, :]  # (Na,Nb,C,nd)
        rnp2 = ao.sum(ao.square(dpc), axis=3)  # (Na,Nb,C)
        fbx = self.gamma[:, :, None] * rnp2   # (Na,Nb,C)

        ijkmax = 3 * a_array.shape[-1]  # @TODO get actual max needed
        boys = self.ga.gm.fboys(co.arange_max[: ijkmax + 1][None, :], fbx.reshape(B, 1))
        ret = self.ga.gm.triple_conv_dot(a_flat[0], a_flat[1], a_flat[2], boys)
        ret = ret.reshape(Na, Nb, C)

        aa = self.ga.alpha[:, None]
        ab = self.gb.alpha[None, :]
        rab2 = ao.sum(ao.square(self.ga.cen[:, None, :] - self.gb.cen[None, :, :]), axis=2)
        ret *= (-2 * co.pi / self.gamma)[:, :, None]
        ret *= ao.exp(-aa * ab * rab2 / self.gamma)[:, :, None]
        return ret


    def calc_ci(self, Hab, Hcd, qp, delta, b2):
        """C-array kernel per dimension (THO Eq. 3.11)
        Hab: (N1, L+1); Hcd: (N2, M+1); qp, delta: (N1, N2)
        b2: (>=S+1, >=U+1)
        returns: (N1, N2, S+1)
        """
        ao, co = self.ao, self.co

        N1, Lp1 = Hab.shape
        N2, Mp1 = Hcd.shape
        L, M = Lp1 - 1, Mp1 - 1
        S, U = L + M, (L + M) // 2

        m = co.arange_max[:Mp1]
        sign_m = co.parity[m % 2]
        Hcd_signed = Hcd * sign_m[None, :]

        # ---- Convolution trick (full length)
        Hab_1c = Hab[:, None, :]               # (N1,1,L+1)
        w = ao.flip(Hcd_signed, -1)[:, None, :]    # (N2,1,M+1)
        C = ao.conv1d(Hab_1c, w, padding=M)    # (N1,N2,S+1)

        qp_is_zero = (qp == 0)

        # ---- (qp/delta)^i for i=0..S  (delta assumed nonzero)
        base = qp / delta
        ones = ao.ones(base.shape)[..., None]                     # (N1,N2,1)
        base_stack = ao.broadcast_to(base[..., None], (N1, N2, S))  # (N1,N2,S)
        qp_over_delta_pows = ao.cumprod(ao.concat([ones, base_stack], axis=-1), axis=-1)

        # ---- qp^{-u} safely: define qp^{-u}=0 for u>0 when qp==0
        qp_inv = ao.where(qp != 0, qp.reciprocal(), ao.zeros(qp.shape))
        ones_u = ao.ones(qp_inv.shape)[..., None]                 # (N1,N2,1)
        inv_stack = ao.broadcast_to(qp_inv[..., None], (N1, N2, U))          # (N1,N2,U)
        qp_inv_pows = ao.cumprod(ao.concat([ones_u, inv_stack], axis=-1), axis=-1)

        # ---- Main accumulation (factored form)
        ans = ao.zeros((N1, N2, S + 1))#, device=Hab.device, dtype=Hab.dtype)

        for u in range(U + 1):
            if 2 * u > S:
                break
            n = S + 1 - 2 * u
            sign_u = -1.0 if (u & 1) else 1.0

            C_s = C[:, :, 2 * u : 2 * u + n]                    # s = 2u..S
            pow_i = qp_over_delta_pows[:, :, u : u + n]         # i = u..S-u
            inv_u = qp_inv_pows[:, :, u][:, :, None]            # qp^{-u}
            b_su = b2[2 * u : 2 * u + n, u].view(1, 1, n)

            ans[:, :, u : u + n].add_(sign_u * C_s * b_su * inv_u * pow_i)

        # ---- Fix missing qp==0 cancellation terms (k==0 => s=2u, i=u)
        umax0 = min(U, S // 2)
        if umax0 >= 1:
            uvals = co.arange_max[1:umax0 + 1]  
            sidx = 2 * uvals
            sign_u = (1 - 2 * (uvals & 1)).to(Hab.dtype)

            inv_delta = delta.reciprocal()
            invd_stack = inv_delta[..., None].expand(N1, N2, S)
            inv_delta_pows = ao.cumprod(
                ao.concat([ao.ones(inv_delta.shape)[..., None], invd_stack], axis=-1),
                axis=-1,
            )

            C_diag = C[:, :, sidx]
            b_diag = b2[sidx, uvals].to(Hab.dtype).view(1, 1, -1)
            invd_u = inv_delta_pows[:, :, uvals]

            corr = (sign_u.view(1, 1, -1) * C_diag * b_diag * invd_u)
            corr = corr * qp_is_zero[...,None]

            ans.index_add_(2, uvals, corr)

        return ans


    def calc_c_array(self, other):
        ao, co = self.ao, self.co
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        N1, N2 = Na*Nb, Nc*Nd
        B = N1 * N2
        hab, hcd = self.hab, other.hab
        hab = hab.reshape(-1, *hab.shape[2:]) # (Na*Nb,nd,L)
        hcd = hcd.reshape(-1, *hcd.shape[2:]) # (Nc*Nd,nd,M)
        hab = ao.transpose(hab, [1, 0, 2]) # (nd,N1,L)
        hcd = ao.transpose(hcd, [1, 0, 2]) # (nd,N2,M)

        gab, gcd = self.gamma.reshape(-1), other.gamma.reshape(-1) # (Na*Nb), (Nc*Nd)
        deltas = (0.25 * (co.one / gab[:,None] + co.one / gcd[None,:])) #.reshape(-1)

        qp = other.pab[None,None,:,:,:] - self.pab[:,:,None,None,:] 
        qp = qp.reshape(N1, N2, self.nd) # (N1,N2,nd)
        qp = ao.transpose(qp, [2,0,1]) # (nd,N1,N2)

        cis = [None,] * self.nd
        for id in range(self.nd):
            h1i, h2i, qpi = hab[id], hcd[id], qp[id]
            cis[id] = self.calc_ci(h1i, h2i, qpi, deltas, co.binom2s)

        return cis # (nd,) list of (Na*Nb,Nc*Nd,L+M)

    def calc_c_self(self):
        """calc C array for self interaction"""
        # ans[n_1,n_2,i] =
        # \frac{b2[2i,i]}{\delta[n_1,n_2]^i}
        # \sum_{l=\max(0,\,2i-M)}^{\min(L,\,2i)}
        # Hab[n_1,l]\;Hab[n_2,2i-l]\;(-1)^{\,i-l}.

        # since the boys reduces to 1/(2n+1) maybe just do this here?
        # need to check if it factors appropriately first

        ao, co = self.ao, self.co
        Na, Nb = len(self.ga), len(self.gb)
        N1 = Na*Nb
        hab = self.hab
        hab = hab.reshape(-1, *hab.shape[2:]) # (Na*Nb,nd,L)
        hab = ao.transpose(hab, [1, 0, 2]) # (nd,N1,L)
        gab = self.gamma.reshape(-1) # (Na*Nb)
        deltas = (0.25 * co.one) / gab #.reshape(-1)
        deltas = deltas[:,None] + deltas[None,:]

        cis = [None,] * self.nd
        for id in range(self.nd):
            cis[id] = self.calc_ci_self(hab[id], deltas, co.binom2s)

        return cis # (nd,) list of (Na*Nb,Nc*Nd,L+M)


    def calc_ci_self(self, Hab, delta, b2):
        """C-array kernel per dimension for self interaction (Hcd == Hab, qp == 0).
        """
        ao, co = self.ao, self.co

        N1, Lp1 = Hab.shape
        L = Lp1 - 1
        S = 2 * L

        # ---- sign into Hcd (Hcd == Hab): Hcd_signed[m] = (-1)^m * Hab[m]
        m = co.arange_max[:Lp1]
        sign_m = co.parity[m % 2]
        Hcd_signed = Hab * sign_m[None, :]

        # ---- Toeplitz-like windows only for even s = 2u
        Hab_1c = Hab[:, None, :]  # (N1,1,L+1)
        w = ao.flip(Hcd_signed, -1)[:, None, :]  # convolution kernel
        C_full = ao.conv1d(Hab_1c, w, padding=L)  # (N1,N1,S+1)
        C_even = C_full[:, :, 0 : S + 1 : 2]  # (N1,N1,U+1)

        # ---- 1/delta^i for i=0..U  (delta assumed nonzero)
        inv_delta = delta.reciprocal()
        ones = ao.ones(inv_delta.shape)[..., None]  # (N1,N1,1)
        inv_stack = ao.broadcast_to(inv_delta[..., None], (N1, N1, L))
        inv_delta_pows = ao.cumprod(ao.concat([ones, inv_stack], axis=-1), axis=-1)
        # ---- scale by (-1)^i * b2[2i,i]
        i = co.arange_max[: Lp1]

        sign_i = co.parity[i % 2]
        b_diag = b2[2 * i, i]
        scale = sign_i[None, None, :] * b_diag[None, None, :]
        core = C_even * scale * inv_delta_pows  # (N1,N1,U+1)
        return core

    
    def calc_eri(self, other):
        """Electron repulsion integrals (a*b|c*d) in 4D batch form"""
        ao, co = self.ao, self.co
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        nd = self.nd

        gab, gcd = self.gamma, other.gamma  # (Na,Nb), (Nc,Nd)
        deltas = 0.25 * (
            (co.one / gab)[:, :, None, None] + (co.one / gcd)[None, None, :, :]
        )  # (Na,Nb,Nc,Nd)

        # build C array and flatten spatial dim for triple_conv_dot
        self.deltas = deltas
        c_array = self.calc_c_array(other)  # (nd,) List of (Na*Nb,Nc*Nd,L+M)
        del self.deltas

        B = Na * Nb * Nc * Nd
        # c_flat = ao.transpose(c_array.reshape(B, nd, -1), (1, 0, 2))  # (nd,B,L)
        c_flat = [ci.reshape(B, -1) for ci in c_array]  # list of (B,L+M)

        qp2 = ao.sum(
            ao.square(self.pab[:, :, None, None, :] - other.pab[None, None, :, :, :]),
            axis=4,
        )  # (Na,Nb,Nc,Nd)
        fbx = 0.25 * qp2 / deltas  # (Na,Nb,Nc,Nd)

        ijkmax = c_flat[0].shape[-1] + c_flat[1].shape[-1] + c_flat[2].shape[-1]
        boys = self.ga.gm.fboys(co.arange_max[: ijkmax + 1][None, :], fbx.reshape(B, 1))

        ret = self.ga.gm.triple_conv_dot(c_flat[0], c_flat[1], c_flat[2], boys)
        ret = ret.reshape(Na, Nb, Nc, Nd)

        # prefactors
        aa = self.ga.alpha[:, None]
        ab = self.gb.alpha[None, :]
        ac = other.ga.alpha[:, None]
        ad = other.gb.alpha[None, :]

        ra = self.ga.cen[:, None, :]
        rb = self.gb.cen[None, :, :]
        rc = other.ga.cen[:, None, :]
        rd = other.gb.cen[None, :, :]
        rab2 = ao.sum(ao.square(ra - rb), axis=2)  # (Na,Nb)
        rcd2 = ao.sum(ao.square(rc - rd), axis=2)  # (Nc,Nd)

        pref = co.eri_fac / (
            gab[:, :, None, None] * gcd[None, None, :, :]
            * ao.sqrt(gab[:, :, None, None] + gcd[None, None, :, :])
        )
        ret *= pref
        ret *= ao.exp(-aa * ab * rab2 / gab)[:, :, None, None]
        ret *= ao.exp(-ac * ad * rcd2 / gcd)[None, None, :, :]
        return ret
    # end def calc_eri(self, other):


    def calc_eri_self(self):
        """Electron repulsion integrals (a*b|a*b) using calc_c_self"""
        # Only works in batch-mode if all GTOs pairs have the same P_ab
        # which means it only works for GTOs is ga and gb having the same
        # center (P_ab = R_a = R_b) OR all primitives in ga have the same 
        # center and exponent, and similarly for gb.
        # Falls back to the general case if the condition is not met.
        ao, co = self.ao, self.co
        if ao.abs(self.pab - self.pab[0]).max().item() != 0: # check validity
            return self.calc_eri(self)  # fallback to general case

        Na, Nb = len(self.ga), len(self.gb)
        c_array = self.calc_c_self()  # (nd,) list of (Na*Nb,Na*Nb,L+1)
        B = Na * Nb * Na * Nb
        c_flat = [ci.reshape(B, -1) for ci in c_array]

        # boys reduces to 1/(2n+1) here
        ijkmax = c_flat[0].shape[-1] + c_flat[1].shape[-1] + c_flat[2].shape[-1]
        boys = co.half / (co.arange_max[: ijkmax + 1] + co.half)
        boys = ao.broadcast_to(boys[None,:], (B, ijkmax+1))

        ret = self.ga.gm.triple_conv_dot(c_flat[0], c_flat[1], c_flat[2], boys)
        ret = ret.reshape(Na, Nb, Na, Nb)

        # prefactors
        aa = self.ga.alpha[:, None]
        ab = self.gb.alpha[None, :]
        rab2 = ao.sum(ao.square(self.ga.cen[:, None, :] - self.gb.cen[None, :, :]), axis=2)

        gab = self.gamma  # (Na,Nb)
        pref = co.eri_fac / (
            gab[:, :, None, None] * gab[None, None, :, :]
            * ao.sqrt(gab[:, :, None, None] + gab[None, None, :, :])
        )
        ret *= pref
        exp_ab = ao.exp(-aa * ab * rab2 / gab)
        ret *= exp_ab[:, :, None, None] 
        ret *= exp_ab[None, None, :, :]
        return ret



if __name__ == "__main__":
    from pyquante22.basis.cart_gto_orig import CartGtoPrim as CGP0
    from pyquante22.basis.cart_gto_orig import CartGtoPrimPair as CPP0
    import itertools
    import time

    def ndindex(shape):
        return itertools.product(*[range(n) for n in shape]) # generator

    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )
    coi = CachedOps(aoi, amax=20)
    # ao.conv1d = ao.xp.nn.functional.conv1d  # @DEBUG

    def tolegacy(g):
        if isinstance(g, CartGtoPrim):
            aas, ams, rrs = g.alpha, g.amom, g.cen
            ams = ams.reshape(-1)
            rrs = rrs.reshape(-1)
            aas = aoi.tile(aas.reshape(-1,1), 3).reshape(-1)
            return CGP0(ams, rrs, aas)
        elif isinstance(g, CartGtoPrimPair):
            return tolegacy(g.ga) * tolegacy(g.gb)
        raise NotImplementedError

    n = 1
    ams = aoi.arr([0, 0, 0], dtype=aoi.int_dtype)
    amx = aoi.arr([-2, 0, 0], dtype=aoi.int_dtype)
    amp = aoi.arr([1, 0, 0], dtype=aoi.int_dtype)
    amn = aoi.arr([n, n+1, n+3], dtype=aoi.int_dtype)
    r0 = aoi.arr([0, 0, 0])
    rb = aoi.arr([0.125, 0.25, 0.5])
    rc = aoi.arr([-0.625, 0.125, 0.25])
    aa = aoi.arr([1.0])
    ab = aoi.arr([2.0])
    gn =  CartGtoPrim(amn, r0, aa, ao=aoi, co=coi)
    gs =  CartGtoPrim(ams, r0, aa, ao=aoi, co=coi)
    gx =  CartGtoPrim(amx, r0, aa, ao=aoi, co=coi)
    gsb = CartGtoPrim(ams, rb, ab, ao=aoi, co=coi)
    gsc = CartGtoPrim(ams, rc, aa, ao=aoi, co=coi)
    gp =  CartGtoPrim(amp, r0, aa, ao=aoi, co=coi)
    gna = CartGtoPrim(amn, r0, aa, ao=aoi, co=coi)
    gnb = CartGtoPrim(amn, rb, ab, ao=aoi, co=coi)
    gnc = CartGtoPrim(amn, rc, ab, ao=aoi, co=coi)
    gma = CartGtoPrim(amn+1, r0, aa, ao=aoi, co=coi)
    gmb = CartGtoPrim(amn+1, rb, ab, ao=aoi, co=coi)
    gmc = CartGtoPrim(amn+1, rc, aa, ao=aoi, co=coi)
    goa = CartGtoPrim(amn+2, r0, aa, ao=aoi, co=coi)
    gob = CartGtoPrim(amn+2, rb, ab, ao=aoi, co=coi)
    gp_list = [gn, gs, gsb, gp, gnb, gmb, gob]

    gp1 = sum([gnb, gsb, gmb, gob])
    gp2 = sum([gsb, gnb, gob, gmb])
    gp12 = gp1 * gp2

    loops = 100
    t0 = time.time()
    for _ in range(loops):
        cas = gp12.calc_c_array(gp12)
    t1 = time.time()
    print(f"orig method took {t1-t0:.4f}s for {loops} loops")

    t0 = time.time()
    for _ in range(loops):
        cbs = gp12.calc_c_self()
    t1 = time.time()
    print(f"new method took {t1-t0:.4f}s for {loops} loops")

    for i in range(len(cas)):
        minl = cbs[i].shape[-1]
        extra = cas[i][:,:,minl:]
        if extra.nelement() > 0:
            assert aoi.abs(extra).max() == 0, f"nonzero extra in c array {i}"
        assert aoi.abs(cas[i][:,:,:minl] - cbs[i]).max() < 1e-16, "c array self mismatch"


    # gp1 = sum([gna, gs, gmc, goa]) # needs check_valid in calc_eri_self
    gp1 = sum([gnb, gsb, gmb, gob])
    gp2 = sum([gsb, gnb, gob, gmb])
    gp12 = gp1 * gp2

    loops = 100
    t0 = time.time()
    for _ in range(loops):
        eas = gp12.calc_eri(gp12)
    t1 = time.time()
    print(f"orig method took {t1-t0:.4f}s for {loops} loops")

    t0 = time.time()
    for _ in range(loops):
        ebs = gp12.calc_eri_self()
    t1 = time.time()
    print(f"new method took {t1-t0:.4f}s for {loops} loops")

    assert aoi.abs(eas - ebs).max() < 1e-16, "eri self mismatch"


    # loops = 500
    # t0 = time.time()
    # for _ in range(loops):
    #     gab = sum(gp_list) * sum(gp_list[-2::-1])
    # t1 = time.time()
    # print(f"{loops} inits took {t1-t0:.4f}s")


    # one body ops test:
    gab = sum(gp_list) * sum(gp_list[-2::-1])
    methods = ["calc_overlap1d", "calc_overlap", "calc_kinetic", "calc_h_array"]
    results = [getattr(gab, f)() for f in methods]
    results = [getattr(gab, f)() for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for ii, f in enumerate(methods):
            gt = getattr(gabl, f)()
            ans = results[ii][i,j]
            gshap, ashap = gt.shape, ans.shape
            if gt.nelement() < 2:
                assert gt.item() == ans.item()
                continue
            assert len(gshap) == len(ashap), "ndim mismatch"
            min_shape = tuple(min(a, b) for a, b in zip(gt.shape, ans.shape))
            overlap_slices = tuple(slice(0, m) for m in min_shape)
            gt2 = gt[overlap_slices]
            an2 = ans[overlap_slices]
            assert aoi.sum(aoi.abs(an2-gt2)) == 0.0, f"mismatch in {f}"
            mask = aoi.ones(min_shape, dtype=bool)
            if min_shape != gt.shape: # check extras
                gt[overlap_slices] = 0
                assert aoi.abs(aoi.sum(gt)) == 0, f"gt extras not 0 {f}"
            if min_shape != ans.shape: # check extras
                ans[overlap_slices] = 0
                assert aoi.abs(aoi.sum(ans)) == 0, f"ans extras not 0 {f}"

            debug_var = 1


    # two body ops test:
    gp4 = sum(gp_list[:4])
    gp3 = sum(gp_list[4:7])
    gp2 = sum(gp_list[:-3:-1])
    gp1 = sum(gp_list[-2:-3:-1])
    gab, gcd = gp2*gp3, gp1*gp4
    rcens = gcd.pab.reshape(-1, gcd.nd)


    methods = ["calc_a_array", "calc_nai",]
    results = [getattr(gab, f)(rcens) for f in methods]
    results = [getattr(gab, f)(rcens) for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for k in range(rcens.shape[0]):
            for ii, f in enumerate(methods):
                gt = getattr(gabl, f)(rcens[k])
                ans = results[ii][i,j,k]
                gshap, ashap = gt.shape, ans.shape
                if gt.nelement() < 2:
                    assert gt.item() == ans.item()
                    continue
                assert len(gshap) == len(ashap), "ndim mismatch"
                min_shape = tuple(min(a, b) for a, b in zip(gt.shape, ans.shape))
                overlap_slices = tuple(slice(0, m) for m in min_shape)
                gt2 = gt[overlap_slices]
                an2 = ans[overlap_slices]
                assert aoi.sum(aoi.abs(an2-gt2)) == 0.0, f"mismatch in {f}"
                mask = aoi.ones(min_shape, dtype=bool)
                if min_shape != gt.shape: # check extras
                    gt[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(gt)) == 0, f"gt extras not 0 {f}"
                if min_shape != ans.shape: # check extras
                    ans[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(ans)) == 0, f"ans extras not 0 {f}"

                debug_var = 1


    methods = ["calc_c_array", "calc_eri",]
    results = [getattr(gab, f)(gcd) for f in methods]
    results = [getattr(gab, f)(gcd) for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for k, l in ndindex(gcd.shape):
            gcdl = tolegacy(gcd[k,l])

            for ii, f in enumerate(methods):
                gt = getattr(gabl, f)(gcdl)
                if f == "calc_c_array": # handle list output
                    stack = aoi.stack(results[ii], axis=0)
                    ans = stack.reshape(gab.nd, *gab.shape, *gcd.shape, -1)[:,i,j,k,l]
                else:
                    ans = results[ii][i,j,k,l]
                if gt.nelement() < 2:
                    assert gt.item() == ans.item()
                    continue

                gshap, ashap = gt.shape, ans.shape
                assert len(gshap) == len(ashap), "ndim mismatch"

                min_shape = tuple(min(a, b) for a, b in zip(gt.shape, ans.shape))
                overlap_slices = tuple(slice(0, m) for m in min_shape)
                gt2 = gt[overlap_slices]
                an2 = ans[overlap_slices]
                assert aoi.sum(aoi.abs(an2-gt2)) == 0.0, f"mismatch in {f}"


                mask = aoi.ones(min_shape, dtype=bool)
                if min_shape != gt.shape: # check extras
                    gt[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(gt)) == 0, f"gt extras not 0 {f}"
                if min_shape != ans.shape: # check extras
                    ans[overlap_slices] = 0
                    assert aoi.abs(aoi.sum(ans)) == 0, f"ans extras not 0 {f}"

                debug_var = 1

    debug_var = 2

    # CartGtoPrim (Lx,Ly,Lz) (Rx,Ry,Rz) Exp
    # 4  2  0        0.0000       0.5000      -0.2500    2.0000e+00, 
    # CartGtoPrim (Lx,Ly,Lz) (Rx,Ry,Rz) Exp
    # 0  0  0        0.1250       0.1250       0.1250    1.0000e+00
    ama = aoi.arr([4, 2, 0], dtype=aoi.int_dtype)
    amb = aoi.arr([0, 0, 0], dtype=aoi.int_dtype)
    ra = aoi.arr([0, 0.5, -0.25])
    rb = aoi.arr([0.125, 0.125, 0.125])
    aa = aoi.arr([2.0])
    ab = aoi.arr([1.0])
    ga = CartGtoPrim(ama, ra, aa, ao=aoi, co=coi)
    gb = CartGtoPrim(amb, rb, ab, ao=aoi, co=coi)
    gab = CartGtoPrimPair(ga, gb)
    ans = gab.calc_kinetic().item()
    gabl = tolegacy(gab)
    ref = gabl.calc_kinetic().item()

    import numpy as np
    import pyquante22.arb_math as arb
    from pyquante22.ref_ints.utils import GTO


    def to_np(a):  # to numpy
        if hasattr(a, "shape"):
            return np.asarray(a.tolist())
        return np.asarray(a.item())

    def cgp_scalar(gto: CGP):
        l = to_np(gto.amom)
        r = to_np(gto.cen)
        if hasattr(l, "ndim") and l.ndim == 2:
            l = l[0]
        if hasattr(r, "ndim") and r.ndim == 2:
            r = r[0]
        return l, r, to_np(gto.alpha[0])


    def gto2arb(gto):
        if isinstance(gto, GTO):
            l = arb.arr_init(gto.l.tolist())
            r = arb.arr_init(gto.r.tolist())
            a = arb.arb(gto.a.item())
            return [a, l, r]
        elif isinstance(gto, CGP):
            l, r, a = cgp_scalar(gto)
            return gto2arb(GTO(l=l, r=r, a=a))
        else:
            raise ValueError(f"Does not support type {type(gto)}")

    def kin_ref(*args):
        """Kinetic reference, args is one CartGtoPrimPair"""
        ab = args[0]  # one CGPP
        la, ra, aa = cgp_scalar(ab.ga)
        lb, rb, abeta = cgp_scalar(ab.gb)
        a = GTO(l=la, r=ra, a=aa)
        b = GTO(l=lb, r=rb, a=abeta)
        fargs = sum([gto2arb(g) for g in [a, b]], [])
        return arb.kinetic_tho66(*fargs)

    gt = float(kin_ref(gab))

    err1 = (ans - ref) /ref
    err2 = (ref - gt) / gt
    err3 = (ans - gt) / gt
    print(f"{coi.amax} {err1:.4e} {err2:.4e} {err3:.4e}")

    debug_var = 1

 
