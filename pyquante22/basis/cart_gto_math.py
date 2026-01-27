import pyquante22.utils.math as pm  # python math for combinatorials
import pyquante22.array_ops as arr


class CachedOps:
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
            [pm.gamma_half_int(n) for n in range(2 * amax + 2)]
        )
        self.binoms = ao.arr(  # n!/(k!(n-k)!)
            [
                [pm.binom_exact(n, k) for k in range(4 * amax + 2)]
                for n in range(4 * amax + 2)
            ]
        )
        self.binom2s = ao.arr(  # n!/(k!(n-2k)!)
            [
                [pm.binom2k_exact(n, k) for k in range(4 * amax + 2)]
                for n in range(4 * amax + 2)
            ]
        )
        self.binom2ns = ao.arr(  # (2n)!/(k!(n-k)!)
            [
                [pm.binom2n_exact(n, k) for k in range(2 * amax + 2)]
                for n in range(2 * amax + 2)
            ]
        )

    def binom22(self, i, r, u):
        """ i! / (r! u! (i-2(r+u))!) """
        # this one is a function using the form factored into
        # binomials so we're not storing a 3d array, which would
        # be 141kB in float64 with amax=20 (i=[0,2*mx+1], r,u=[0,mx+1])
        t = r + u
        return self.binoms[i, 2 * t] * self.binom2ns[t, r]


class GtoMath:
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

    def fboys(self, m, x):  # stable @ x=0
        """boys function"""
        ao = self.ao
        m2 = m + 0.5 * ao.arr(1)
        ret = 0.5 * pow(x, -m2) * ao.gamma_inc(m2, x)
        mask = x < ao.xp.finfo(x.dtype).eps
        mask = ao.broadcast_to(mask, ret.shape)
        ret[mask] = ao.broadcast_to(0.5 / m2, ret.shape)[mask]
        return ret

    def triple_conv_dot(self, ax, ay, az, fxyz, preflip=False):
        """ans[:] = sum_{i,j,k} ax[:,i] ay[:,j] az[:,k] fxyz[:,i+j+k]
        preflip indicates whether az has been flipped along axis=1
        """
        B, nx = ax.shape
        nz, nf = az.shape[1], fxyz.shape[1]

        # ---- k = ay * az (true convolution) ----
        # input: (1, B, ny), weight: (B, 1, nz), groups=B
        w1 = (az if preflip else az.flip(-1)).unsqueeze(1)  # (B,1,nz)
        x1 = ay.unsqueeze(0)  # (1,B,ny)
        k = self.ao.conv1d(x1, w1, groups=B, padding=nz - 1).squeeze(0)  # (B, ny+nz-1)
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

        return self.ao.einsum("ij,ij->i", ax, c)
