# import os, time
from hashlib import sha256

#local imports
import pyquante22.utils.math as pm  # python math for combinatorials
import pyquante22.array_ops as arr


class CachedOps():
    """pre-computed arrays of values"""

    def __init__(self, ao=None, amax=10):
        self.ao = ao or arr.get_default_ops()
        self.amax = amax
        self.ArrType = self.ao.ArrayType
        self._create_constants()
        self._create_combinatorials()


    def _create_constants(self):
        ao, amax = self.ao, self.amax
        one = ao.arr(1)
        self.izero = ao.arr(0, dtype=ao.int_dtype)
        self.ione = ao.arr(1, dtype=ao.int_dtype)
        self.zero = ao.arr(0)
        self.half = ao.arr(0.5)
        self.one = one
        self.two = ao.arr(2)
        self.pi = ao.arr(4) * ao.atan(one)
        self.sqpi = ao.sqrt(self.pi)
        self.eri_fac = (2 * one) * pow(self.pi, 2.5 * one)
        self.parity = ao.arr([1, -1])  # (-1)^t = _parity[t%2]
        self.arange_max = ao.arange(6 * amax + 1)  # 2 for pairs, 3 for 3d space

    def _create_combinatorials(self):
        ao, amax = self.ao, self.amax
        self.ghalfs = ao.arr(  # Gamma(n+1/2)
            [pm.gamma_half_int(n) for n in range(2*amax+1)]
        ) 
        self.binoms = ao.arr( # n!/(k!(n-k)!)
            [
                [pm.binom_exact(n, k) for k in range(2 * amax + 2)]
                for n in range(2 * amax + 2)
            ]
        )
        self.binom2s = ao.arr( # n!/(k!(n-2k)!)
            [
                [pm.binom2k_exact(n, k) for k in range(1 * amax + 2)]
                for n in range(4 * amax + 1)
            ]
        )
        self.binom2ns = ao.arr( # (2n)!/(k!(n-k)!)
            [
                [pm.binom2n_exact(n, k) for k in range(1 * amax + 2)]
                for n in range(2 * amax + 2)
            ]
        )

    def binom22(self, i, r, u):
        """ i! / (r! u! (i-2(r+u))!) """
        # this one is a function using the form factored into 
        # binomials so we're not storing a 3d array, which would 
        # be 141kB in float64 with amax=20 (i=[0,2*mx+1], r,u=[0,mx+1])
        t = r + u
        return self.binoms[i,2*t] * self.binom2ns[t,r]
#end class CachedOps():


class GtoMath():

    def __init__(self, ao=None):
        self.ao = ao or arr.get_default_ops()
        self.ArrType = self.ao.ArrayType


    def pow_ladder(self, x_G, L: int):
        """Build xpow where xpow[g,k] = x_G[g]^k for k=0..L-1.
        return: xpow: (G,L)
        """
        ao = self.ao
        assert L >= 1, "L must be >= 1"
        ones = ao.ones((x_G.shape[0], 1), dtype=x_G.dtype)  # (G,1)
        if L == 1:
            return ones

        xrep = ao.broadcast_to(x_G[:, None], (x_G.shape[0], L - 1))  # (G,L-1)
        tail = ao.cumprod(xrep, axis=1)                              # (G,L-1) = x^1..x^(L-1)
        return ao.concat([ones, tail], axis=1)                       # (G,L)


    def fboys(self, m, x,):  # stable @ x=0
        """boyf.ArrType, ay: ArrType, az: ArrType, fxyz: ArrType, preflip: bool = False
    # ) -> ArrTypes function"""
        ao = self.ao
        m2 = m + 0.5 * ao.arr(1)
        ret = 0.5 * pow(x, -m2) * ao.gamma_inc(m2, x)
        mask = x < ao.xp.finfo(x.dtype).eps
        mask = ao.broadcast_to(mask, ret.shape)
        # ret[:] = ao.where(mask, 0.5/m2, ret) # @TODO: optimize for broadcasting with m and x being different sizes
        ret[mask] = ao.broadcast_to(0.5 / m2, ret.shape)[mask]
        return ret


    def triple_conv_dot(self, ax, ay, az, fxyz, preflip=False):
        """ans[:] = sum_{i,j,k} ax[:,i] ay[:,j] az[:,k] fxyz[:,i+j+k]
        preflip indicates whether az has been flipped along axis=1
        """
        #     ax: self.ArrType, ay: ArrType, az: ArrType, fxyz: ArrType, preflip: bool = False
        # ) -> ArrType:
        B, nx = ax.shape
        nz, nf = az.shape[1], fxyz.shape[1]

        # ---- k = ay * az (true convolution) ----
        # input: (1, B, ny), weight: (B, 1, nz), groups=B
        w1 = (az if preflip else az.flip(-1)).unsqueeze(1)  # (B,1,nz)
        x1 = ay.unsqueeze(0)  # (1,B,ny)
        k = self.ao.conv1d(x1, w1, groups=B, padding=nz - 1).squeeze(0)  # (B, ny+nz-1)
        # w1 = az if preflip else az.flip(-1)
        # k  = ao.conv_full_batch(ay, w1) # (B,ny+nz-1)
        nk = k.size(1)

        # ---- correlate f with k but compute only first nx outputs ----
        need_len = nx + nk - 1  # f coverage needed for first nx outputs
        take = min(nf, need_len)
        right_pad = need_len - take  # only pad on the right if short

        x2 = fxyz[:, :take].unsqueeze(0)  # (1,B,take)
        if right_pad:
            x2 = self.ao.pad(x2, (0, right_pad))  # asymmetric pad (left=0, right)
        w2 = k.unsqueeze(1)  # (B,1,nk)
        c = self.ao.conv1d(x2, w2, groups=B).squeeze(0)  # (B, nx) exactly

        # c  = ao.conv_full_batch(fxyz[:,:take], k)

        # ---- final reduction (single-kernel dot) ----
        # return th.linalg.vecdot(ax, c, dim=1)   # (B,)
        return self.ao.einsum('ij,ij->i', ax, c)
    # end def triple_conv_dot()
#end class GtoMath():


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
            coeffs=coeffs, nd=self.nd, ao=self.ao, co=self.co, 
            precompute=False,)
        if self.norm is not None: ret.norm = self.norm[idx]
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
            precompute=False,
        )
        ret.norm = ret.calc_norm()
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
        ao = self.ao
        norm2_1d = ao.pow(2 * self.alpha[:, None], 
                          self.amom + self.co.half)
        norm2_1d /= self.co.ghalfs[self.amom]
        norm2 = ao.prod(norm2_1d, axis=1)  # (B,)
        return ao.sqrt(norm2)


    def calc_binoms(self, pab, offset=None):
        """Compute binomial expansion coeffs given a center
        param: pab: (B,nd) or (B,C,nd)
        param: offset: (nd,) optional; change to amom (for kinetic)
        returns: (B,nd,Lmax+1) or (B,C,nd,Lmax+1)
        """
        keepC = (pab.ndim == 3)
        pab  = pab if keepC else pab[:, None, :]     # (B,C,nd)

        if offset is not None:
            amom = (self.amom + offset[None,:])[:,None,:] # (B,1,nd)
            lmax = max(0, amom.max().item())
        else:
            amom = self.amom[:,None,:] # (B,1,nd)
            lmax = self.lmax

        irng = self.co.arange_max[:lmax + 1]  # (L,)
        pows = amom[..., None] - irng[None, None, None, :]  # (B,C,nd,L)

        pows[pows < 0] = self.co.izero  # 
        ret = self.co.binoms[amom[..., None], irng[None, None, None, :]]  * \
            self.ao.pow((pab - self.cen[:, None, :])[..., None], pows) # (B,C,nd,L)
        mask = (amom < 0)[..., None]  # (B,1,nd,1)
        ret = self.ao.where(mask, self.co.zero, ret)
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
        self.co = ga.co # assume no amax issues

        if precompute:
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
        if hasattr(other, "shape"):
            return self.calc_nai(other)  # implemented later
        if isinstance(other, type(self)):
            return self.calc_eri(other)  # implemented later
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
        ao = self.ao
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
        pows = ao.pow(self.gamma[:,:,None], irng)
        ret *= (co.ghalfs[irng] / pows)[:,:,None,:] # add spatial dim
        return ao.sum(ret, axis=3)


    def calc_overlap(self, offset=None):
        """calculate the overlap of the pair
        nd: spatial dimensions flattened into the batch, may not work currently if nd != 3
        offset: optional ang. mom. offset, must be (nd,)
        return: overlaps of (len(self)//nd,)
        """
        ao, ga, gb = self.ao, self.ga, self.gb
        invgam = self.co.one / self.gamma # (Na,Nb)
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
        ao, ga, gb = self.ao, self.ga, self.gb
        ab, amb = gb.alpha, gb.amom
        offsets = 2 * ao.eye(self.nd, dtype=ga.amom.dtype)

        two, three = 2*self.co.one, 3*self.co.one
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
            term2 *= -self.co.half

            return term0 + term1 + term2

        else:
            return term0 + term1
    # end def calc_kinetic(gpair, nd=3):



    def calc_h_array(self):
        """THO Eqn 3.5 (typo fixed)"""
        ao = self.ao
        bps = self.binom_pres[...,None] # (Na,Nb,nd,lmax)
        lmax = bps.shape[3] - 1
        full_sz = (len(self.ga), len(self.gb), self.ga.nd, lmax + 1, lmax // 2 + 1)
        g4s = ao.broadcast_to((4*self.gamma)[:,:,None,None,None], full_sz)

        irng = self.co.arange_max[: full_sz[3]][:, None]
        rrng = self.co.arange_max[: full_sz[4]][None, :]
        irmask = rrng <= irng // 2
        iflat = ao.broadcast_to(irng, full_sz[3:])[irmask]
        rflat = ao.broadcast_to(rrng, full_sz[3:])[irmask]
        iiflat = iflat - 2 * rflat

        h_vals_flat = ao.broadcast_to(bps, full_sz)[...,irmask]  # irng > amom == 0 from bps
        h_vals_flat *= ao.pow(g4s[...,irmask], (rflat - iflat)[None,None,None,:])
        h_vals_flat *= self.co.binom2s[iflat, rflat][None,None,None,:]

        ret = ao.zeros(full_sz[:-1])
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), h_vals_flat.shape)
        # ret.scatter_add_(dim=-1, index=sum_ids, src=h_vals_flat)
        ao.scatter_add_inplace(ret, -1, sum_ids, h_vals_flat)
        return ret
    # end def calc_h_array(self):


    def calc_c_array(self, other):
        """THO Eqn 3.4, batched over independent (a,b,c,d) indices.
        returns: (Na,Nb,Nc,Nd,nd,L) where L = lmax+2
        """
        ao, co = self.ao, self.co
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        B = Na * Nb * Nc * Nd

        if hasattr(self, "deltas"):
            deltas = self.deltas
        else:
            gab, gcd = self.gamma, other.gamma  # (Na,Nb), (Nc,Nd)
            deltas = 0.25 * (
                (co.one / gab)[:, :, None, None] + (co.one / gcd)[None, None, :, :]
            )  # (Na,Nb,Nc,Nd)

        hab = self.hab.reshape(-1, *self.hab.shape[2:])      # (Na*Nb,nd,L)
        hcd = other.hab.reshape(-1, *other.hab.shape[2:])    # (Nc*Nd,nd,M)
        habcd = hab[:, None, :, :, None] * hcd[None, :, :, None, :]  # (Na*Nb,Nc*Nd,nd,L,M)
        habcd = habcd.reshape(-1, hab.shape[-1], hcd.shape[-1])      # (B*nd,L,M)

        qp = other.pab[None, None, :, :, :] - self.pab[:, :, None, None, :]  # (Na,Nb,Nc,Nd,nd)
        qp = qp.reshape(-1)  # (B*nd,)

        abmax, cdmax = habcd.shape[1] - 1, habcd.shape[2] - 1
        lmax = abmax + cdmax

        abrng = co.arange_max[: abmax + 1][:, None, None]
        cdrng = co.arange_max[: cdmax + 1][None, :, None]
        urng = co.arange_max[: lmax // 2 + 1][None, None, :]
        iirng = abrng + cdrng - urng  # (L,M,U)
        iurng = iirng - urng
        iumask = iurng >= 0

        iiflat = iirng[iumask]
        iuflat = iurng[iumask]
        uflat = ao.broadcast_to(urng, (*iumask.shape[:2], urng.shape[-1]))[iumask]
        facts = co.binom2s[iiflat + uflat, uflat]
        parity = ao.broadcast_to(
            co.parity[(cdrng + urng) % 2],
            (iumask.shape[0], cdrng.shape[1], urng.shape[2]),
        )[iumask]

        c_vals_flat = ao.pow(qp[:, None], iuflat[None, :])
        c_vals_flat *= ao.broadcast_to(
            habcd[:, :, :, None],
            (*habcd.shape, iumask.shape[-1]),
        )[:, iumask]
        c_vals_flat *= (facts * parity)[None, :]

        ret = ao.zeros((B * self.nd, lmax + 2))
        sum_ids = ao.broadcast_to(iiflat.reshape(1, -1), c_vals_flat.shape)
        ao.scatter_add_inplace(ret, 1, sum_ids, c_vals_flat)

        d2 = ao.broadcast_to(deltas[..., None], (Na, Nb, Nc, Nd, self.nd)).reshape(-1)
        ret *= ao.pow(d2[:, None], -co.arange_max[: lmax + 2][None, :])
        return ret.reshape(Na, Nb, Nc, Nd, self.nd, lmax + 2)
    # end def calc_c_array(self, other):


    def calc_eri(self, other):
        """Electron repulsion integrals (a*b|c*d) in 4D batch form."""
        ao, co = self.ao, self.co
        Na, Nb, Nc, Nd = len(self.ga), len(self.gb), len(other.ga), len(other.gb)
        nd = self.nd

        gab, gcd = self.gamma, other.gamma  # (Na,Nb), (Nc,Nd)
        deltas = 0.25 * (
            (co.one / gab)[:, :, None, None] + (co.one / gcd)[None, None, :, :]
        )  # (Na,Nb,Nc,Nd)

        # build C array and flatten spatial dim for triple_conv_dot
        self.deltas = deltas
        c_array = self.calc_c_array(other)  # (Na,Nb,Nc,Nd,nd,L)
        del self.deltas

        B = Na * Nb * Nc * Nd
        c_flat = ao.transpose(c_array.reshape(B, nd, -1), (1, 0, 2))  # (nd,B,L)

        qp2 = ao.sum(
            ao.square(self.pab[:, :, None, None, :] - other.pab[None, None, :, :, :]),
            axis=4,
        )  # (Na,Nb,Nc,Nd)
        fbx = 0.25 * qp2 / deltas  # (Na,Nb,Nc,Nd)

        ijkmax = 3 * c_array.shape[-1]  # @TODO get actual max needed
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
            gab[:, :, None, None]
            * gcd[None, None, :, :]
            * ao.sqrt(gab[:, :, None, None] + gcd[None, None, :, :])
        )
        ret *= pref
        ret *= ao.exp(-aa * ab * rab2 / gab)[:, :, None, None]
        ret *= ao.exp(-ac * ad * rcd2 / gcd)[None, None, :, :]
        return ret
    # end def calc_eri(self, other):



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
    coi = CachedOps(aoi, amax=10)
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
    aa = aoi.arr([1.0] * 3)
    gn =  CartGtoPrim(amn, r0, aa, ao=aoi)
    gs =  CartGtoPrim(ams, r0, aa, ao=aoi)
    gx =  CartGtoPrim(amx, r0, aa, ao=aoi)
    gsb = CartGtoPrim(ams, rb, aa, ao=aoi)
    gp =  CartGtoPrim(amp, r0, aa, ao=aoi)
    gnb = CartGtoPrim(amn, rb, aa, ao=aoi)
    gmb = CartGtoPrim(amn+1, rb, aa, ao=aoi)
    gob = CartGtoPrim(amn+2, rb, aa, ao=aoi)
    gp_list = [gn, gs, gsb, gp, gnb, gmb, gob]

    # one body ops to test:
    # overlap1d, overlap, kinetic, calc_h_array
    loops = 500
    t0 = time.time()
    for _ in range(loops):
        gab = sum(gp_list) * sum(gp_list[-2::-1])
    t1 = time.time()
    print(f"{loops} inits took {t1-t0:.4f}s")


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



    # two body ops to test:
    # calc_a_array, calc_c_array, calc_nai, calc_eri
    gp4 = sum(gp_list[:4])
    gp3 = sum(gp_list[4:7])
    gp2 = sum(gp_list[:-3:-1])
    gp1 = sum(gp_list[-2:-3:-1])
    gab, gcd = gp2*gp3, gp1*gp4

    methods = ["calc_c_array", "calc_eri",]# "calc_nai", "calc_h_array"]
    results = [getattr(gab, f)(gcd) for f in methods]
    results = [getattr(gab, f)(gcd) for f in methods] # call twice to make sure data isn't corrupted

    for i,j in ndindex(gab.shape):
        gabl = tolegacy(gab[i,j])
        for k, l in ndindex(gcd.shape):
            gcdl = tolegacy(gcd[k,l])

            for ii, f in enumerate(methods):
                gt = getattr(gabl, f)(gcdl)
                ans = results[ii][i,j,k,l]
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



    debug_var = 2
    if False:

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




        g1 = gn + gsb
        gp1 = g1 * gmb
        tmp = gp1.calc_overlap1d()
        tmp = gp1.calc_overlap()

        tmp = (gs*gs).calc_kinetic().item()
        gsl = tolegacy(gs)
        gt = (gsl*gsl).calc_kinetic().item()
        assert tmp == gt, "kinetic error"


        debug_var = 1
