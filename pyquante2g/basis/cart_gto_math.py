import pyquante2g.utils.math as pm  # python math for combinatorials
import pyquante2g.torch_ops as arr
import torch



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

#end class CachedOps:


class GtoMath:
    """Numerical math functions needed for GTO calculations
    NB: all the specific boys functions expect args shapes of m: (1,M); x: (B,1)
    """
    def __init__(self, ao=None):
        self.ao = ao or arr.get_default_ops()
        self.ArrType = self.ao.ArrayType
        self.one = self.ao.arr(1)

        if 'float32' in str(self.ao.default_dtype):
            self.fboys = self.fboys_float32
        elif 'float64' in str(self.ao.default_dtype):
            self.fboys = self.fboys_float64
        else:
            self.fboys = self.fboys_igamma 


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



    def triple_conv_dot(self, ax, ay, az, fxyz, preflip=False):
        """ans[:] = sum_{i,j,k} ax[:,i] ay[:,j] az[:,k] fxyz[:,i+j+k]
        preflip indicates whether az has been flipped along axis=1
        """
        ao = self.ao
        B, nx = ax.shape
        ny = ay.shape[1]
        nz, nf = az.shape[1], fxyz.shape[1]

        # ---- k = ay * az (true convolution) ----
        # input: (1, B, ny), weight: (B, 1, nz), groups=B
        w1 = ao.reshape(az if preflip else ao.flip(az, axis=-1), (B, 1, nz))
        x1 = ao.reshape(ay, (1, B, ny))
        k = ao.conv1d(x1, w1, groups=B, padding=nz - 1)[0]  # (B, ny+nz-1)
        nk = k.shape[1]

        # ---- correlate f with k but compute only first nx outputs ----
        need_len = nx + nk - 1  # f coverage needed for first nx outputs
        take = min(nf, need_len)
        right_pad = need_len - take  # only pad on the right if short

        x2 = ao.reshape(fxyz[:, :take], (1, B, take))
        if right_pad:
            x2 = ao.pad(x2, ((0, 0), (0, 0), (0, right_pad)))
        w2 = ao.reshape(k, (B, 1, nk))
        c = ao.conv1d(x2, w2, groups=B)[0]  # (B, nx) exactly

        return ao.einsum("ij,ij->i", ax, c)


    def fboys_float32(self, m, x):  # 
        """boys function for float32 piecewise splits"""
        ao = self.ao
        xsw, msw = 0.5, 26
        row_small = x[:, 0] < xsw
        col_large = m[0, :] > msw
        series_vals = self.fboys_series(m, x, max_terms=8)
        large_vals = self.fboys_large_m(m, x, max_terms=12)
        igamma_vals = self.fboys_igamma(m, x)
        rest_vals = ao.where(col_large[None, :], large_vals, igamma_vals)
        return ao.where(row_small[:, None], series_vals, rest_vals)


    def fboys_float64(self, m, x):  # 
        """boys function for float64 piecewise splits"""
        ao = self.ao
        xsw = 0.1
        row_small = x[:, 0] < xsw
        series_vals = self.fboys_series(m, x, max_terms=8)
        # series_vals = self.fboys_series8_unrolled(m, x,)
        igamma_vals = self.fboys_igamma(m, x)
        return ao.where(row_small[:, None], series_vals, igamma_vals)


    def fboys_igamma(self, m, x):  # unstable @ x=0 and large m
        """boys function using incomplete gamma func"""
        m2 = m + 0.5
        # ret = 0.5 * pow(x, -m2) * self.ao.gamma_inc(m2, x)
        ret = 0.5 / (x ** m2) * self.ao.gamma_inc(m2, x)
        return ret


    def fboys_ligamma(self, m, x): # unstable @ x=0
        """boys func with log-incomplete gamma"""
        ao = self.ao
        m2 = m + 0.5
        lg = ao.log(ao.gamma_inc(m2, x)) # hacky! should use dedicated log_igamma  
        return 0.5 *ao.exp(lg - m2*ao.log(x))
 
    
    def fboys_series(self, m, x, max_terms=8): # unstable for x >= 1
        """Taylor series expansion at x=0"""
        ao = self.ao
        a = m + 0.5
        term = ao.ones(x.shape) # n = 0 term
        s = term / a

        for k in range(1, max_terms): #
            term *= -x / k
            s += term / (a+k)
        return 0.5 * s


    def fboys_series8_unrolled(self, m, x):
        """
        8-term x=0 series for Boys function with no loop.
        Equivalent to fboys_series(m, x, max_terms=8).
        """
        a = m + 0.5
        z = -x
        term = 1.0 / a
        s = term

        m2 = a + 1.0
        term = term * z * a / m2
        s = s + term

        m2 = m2 + 1.0
        term = term * z * (m2 - 1.0) / (2.0 * m2)
        s = s + term

        m2 = m2 + 1.0
        term = term * z * (m2 - 1.0) / (3.0 * m2)
        s = s + term

        m2 = m2 + 1.0
        term = term * z * (m2 - 1.0) / (4.0 * m2)
        s = s + term

        m2 = m2 + 1.0
        term = term * z * (m2 - 1.0) / (5.0 * m2)
        s = s + term

        m2 = m2 + 1.0
        term = term * z * (m2 - 1.0) / (6.0 * m2)
        s = s + term

        m2 = m2 + 1.0
        term = term * z * (m2 - 1.0) / (7.0 * m2)
        s = s + term
        return 0.5 * s


    def fboys_series_xa(self, m, x, x0, fx0_terms):
        """Taylor series around x0 using precomputed fx0_terms[n] = F_n(x0)."""
        ao = self.ao
        shp = x.shape
        xx = x.reshape(-1)
        idx = ao.arr(m.reshape(-1), dtype=ao.int_dtype)
        mmax = int(ao.amax(idx).item())
        idx = ao.minimum(ao.maximum(idx, ao.arr(0, dtype=ao.int_dtype)), ao.arr(mmax, dtype=ao.int_dtype))

        f0 = ao.arr(fx0_terms, dtype=xx.dtype)
        max_terms = int(f0.shape[0]) - mmax
        if max_terms < 1:
            raise ValueError(f"fx0_terms needs at least {mmax + 1} entries for max m={mmax}; got {int(f0.shape[0])}")
        need = mmax + max_terms
        c = f0[:need][idx[:, None] + ao.arange(max_terms, dtype=ao.int_dtype)[None, :]]

        s = c[:, -1]
        dx = xx - ao.arr(float(x0), dtype=xx.dtype)
        for i in range(max_terms - 2, -1, -1):
            s = c[:, i] - dx * s / ao.arr(i + 1.0, dtype=dx.dtype)
        return s.reshape(shp)


    def fboys_m0(self, x): # unstable near x=0
        """boys func for m=0 (not stable near x=0)"""
        ao, xp = self.ao, self.ao.xp
        sqpi = ao.sqrt(ao.arr(4) * ao.atan(ao.arr(1)))
        sqx = ao.sqrt(x)
        return sqpi / (2*sqx) * xp.erf(sqx)


    def fboys_upward(self, m, T):
        """upward recursion from m=0"""
        ao = self.ao
        shp = T.shape
        t = T.reshape(-1)
        idx = ao.arr(m.reshape(-1), dtype=ao.int_dtype)

        mmax = int(ao.amax(idx).item())
        idx = ao.minimum(
            ao.maximum(idx, ao.arr(0, dtype=ao.int_dtype)),
            ao.arr(mmax, dtype=ao.int_dtype),
        )

        f0 = self.fboys_m0(t)
        expmt, inv2t = ao.exp(-t), 0.5 / t

        rows = [f0]
        for k in range(mmax):
            rows.append(((2.0 * k + 1.0) * rows[-1] - expmt) * inv2t)

        F = ao.stack(rows, axis=0)
        col = ao.arange(t.shape[0], dtype=ao.int_dtype)
        return F[idx, col].reshape(shp)


    def fboys_miller(self, m, T, m_margin=2, t_margin=2):
        """Miller downward recursion """
        ao = self.ao
        shp = T.shape
        t = T.reshape(-1)
        idx = ao.arr(m.reshape(-1), dtype=ao.int_dtype)

        mmax = int(ao.amax(idx).item())
        M = max(mmax + m_margin, int(ao.amax(t).item()) + t_margin)
        idx = ao.minimum(
            ao.maximum(idx, ao.arr(0, dtype=ao.int_dtype)),
            ao.arr(mmax, dtype=ao.int_dtype),
        )

        f0 = 0.5 * ao.sqrt(ao.arr(ao.xp.pi, dtype=t.dtype)) * ao.xp.erf(ao.sqrt(t)) / ao.sqrt(t)
        e = ao.exp(-t)
        two_t = 2.0 * t
        n = t.shape[0]

        fk = ao.ones((n,), dtype=t.dtype)
        for k in range(M - 1, mmax, -1): # recurse down from the margin
            fk = (two_t * fk + e) / (2.0 * k + 1.0)

        rows = []
        for k in range(mmax, -1, -1):
            fk = (two_t * fk + e) / (2.0 * k + 1.0)
            rows.append(fk)

        F = ao.stack(list(reversed(rows)), axis=0)
        F *= (f0 / F[0])[None, :]
        col = ao.arange(n, dtype=ao.int_dtype)
        return F[idx, col].reshape(shp)

   
    def fboys_large_m(self, m, x, max_terms=12):
        r"""asymptotic large-m expansion
        f_m(x) = e^{-x} / (2m+1) \sum_k=0 x^k / \prod_{j=1}^k (m+1/2+j)
        """
        ao = self.ao
        m2 = m + 0.5
        ret = 0.5 * ao.exp(-x) / m2

        s = 1
        term, m3 = 1, m2 + 0
        for k in range(1, max_terms):
            m3 += 1
            term *= x / m3
            s += term

        return ret * s

#end class GtoMath:


if __name__ == "__main__":
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import pyquante2g.arb_math as am

    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def add_segmented_xticks(ax, x_vals, subsets):
        # Add x ticks at the first/last values of each concatenated segment.
        xv = _to_numpy(x_vals)
        ticks, starts, ends = [], set(), set()
        i0 = 0
        for s in subsets:
            n = int(_to_numpy(s).shape[0])
            if n <= 0: continue
            seg = np.rint(np.linspace(i0, i0 + n - 1, 4)).astype(int)
            ticks.extend(seg.tolist())
            starts.add(int(seg[0]))
            ends.add(int(seg[-1]))
            i0 += n

        if not ticks:
            ax.set_xticks([])
            return

        ticks = sorted(set(ticks))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [("" if i in ends else f"{xv[i]:g}") for i in ticks], rotation=45, ha="center"
        )

        lines = ax.get_xticklines()
        labels = ax.get_xticklabels()
        for i, pos in enumerate(ticks):
            if pos in starts:
                for line in lines[2 * i : 2 * i + 2]:
                    line.set_markersize(8.0)
                    line.set_markeredgewidth(1.5)
            labels[i].set_fontsize(8 if pos in starts else 8)


    def eval_error_grids(funcs, A, X, gt, EPS, loops=1, print_times=True):
        gt_tiny = gt < EPS
        denom = aoi.where(gt_tiny, EPS, gt)
        func_names = [f.__name__ for f in funcs]
        error_grids, raw_error_grids, eval_grids, run_times = [], [], [], []
        for ii, f in enumerate(funcs):
            t0 = time.perf_counter()
            for _ in range(loops):
                f_vals = f(A, X)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
            if print_times:
                print(f"{func_names[ii]:40s} :: {t1 - t0:.2f}s")

            perrs = aoi.abs(f_vals - gt) / denom
            raw_np = _to_numpy(perrs)
            raw_error_grids.append(raw_np)
            eval_grids.append(_to_numpy(f_vals))

            pnp = raw_np.copy()
            isf = np.isfinite(pnp)
            if not np.all(isf):
                print(f"::WARN:: {func_names[ii]} had NaNs!")
                pnp[~isf] = 1.0
            pnp = np.clip(pnp, EPS, 1)
            pnp[~isf] = 10 ** 2
            error_grids.append(pnp)
        return func_names, error_grids, raw_error_grids, eval_grids, run_times


    dtypes = ["float16", "float32", "float64"]
    aoc = arr.TorchOps(
        "torch", device="cpu", default_dtype=dtypes[2], int_dtype="int64"
    )
    aoi = aoc
    gm = GtoMath(aoi)
    EPS = aoi.xp.finfo(aoi.default_dtype).eps

    xs1 = 10.0 ** aoi.arange(-11, -2, 1.0)
    xs2 = aoi.arange(0.2, 1.0, 0.1)
    xs3 = aoi.arange(1.0, 3, 0.2)
    xs4 = aoi.arange(3, 9, 0.5)
    xs5 = aoi.arange(9, 18, 1.0)
    xs6 = aoi.arange(18, 80, 5)
    xs7 = aoi.arange(80, 141, 10)
    xsubsets = [xs1, xs2, xs3, xs4, xs5, xs6, xs7]

    x_vals = aoi.concat(xsubsets)
    # x_vals = aoi.arange(201).to(aoi.default_dtype)
    nrng = aoi.arange(64)
    X, A = aoi.xp.meshgrid(x_vals, nrng.to(x_vals), indexing="ij")


    ## use the arbitrary precision calculator for ground truth
    am.set_prec(16) # 
    exact_vec = np.vectorize(am.fboys)
    f_exact_old = exact_vec(A, X).astype(np.float64)
    print(f"Finding precision needed for exact cacluations with arbitrary precision on the numerical grid")
    print(f"precision | diff_count | nz_diff_max | nz_diff_median | nz_diff_mean")
    for prec in range(24, 1024, 8):
        am.set_prec(prec) # 
        exact_vec = np.vectorize(am.fboys)
        f_exact = exact_vec(A, X).astype(np.float64)
        if np.all(f_exact == f_exact_old):
            print(f"Exact match found between prec={prec} and previous setting")
            break
        debug_var = 1
        diffs = f_exact_old - f_exact
        nzids = np.where(diffs != 0)
        diff_count = len(nzids[0])
        nzdiffs = np.abs(diffs[nzids])
        print(f"{prec:d}\t{diff_count}\t{nzdiffs.max():.4e}\t{np.median(nzdiffs):.4e}\t{nzdiffs.mean():.4e}")
        f_exact_old = f_exact

    gt = aoi.arr(f_exact) 
    assert aoi.sum(gt < 0).item() == 0, f"This should always be non-negative" # 0 possible from rounding
    assert aoi.sum(gt > 1+EPS).item() == 0, f"Should be bounded by 1.0"


    if False: # series expansion around a point, use arb_math for the coefficients
        series_x0p5_terms = 16
        series_x0p5 = 0.1
        max_m_eval = int(_to_numpy(nrng).max())
        f1_exact_x0p5 = np.array(
            [float(am.fboys(mm, series_x0p5)) for mm in range(max_m_eval + series_x0p5_terms)],
            dtype=np.float64,
        )
        def fboys_series_x0p5(m, x):
            return gm.fboys_series_xa(
                m,
                x,
                x0=series_x0p5,
                fx0_terms=f1_exact_x0p5,
            )

    # eval funcs and their errors
    boys_funcs = [
        gm.fboys_series,
        # fboys_series_x0p5,
        # gm.fboys_miller,
        # gm.fboys_upward, 
        gm.fboys_igamma, 
        # gm.fboys_ligamma,
        gm.fboys_large_m,
        # gm.fboys, # test overloading is working
    ]
    time_loops = int(10**3)
    mb, xb = nrng[None,:].to(x_vals), x_vals[:,None]
    func_names, error_grids, raw_error_grids, eval_grids, run_times = eval_error_grids(
        boys_funcs, mb, xb, gt, EPS, loops=time_loops, print_times=True
    )
        

    all_grids = [] + error_grids
    err_best = np.minimum.reduce(error_grids)
    all_grids += [err_best]
    func_names += ["minimum error"]
    print(f"Min Err over all funcs gives max: {err_best.max():.4e} mean: {err_best.mean():.4e} median: {np.median(err_best):.4e}")

    err_stack = np.stack(error_grids, axis=0)
    winner = np.argmin(err_stack, axis=0)
    all_grids += [winner]
    func_names += ["lowest err func id"]


    nplots = len(all_grids)
    nrows = max(1, int(np.floor(np.sqrt(nplots))))
    ncols = int(np.ceil(nplots / nrows))
    fig, aax = plt.subplots(ncols=ncols, nrows=nrows)
    for iax, ax in enumerate(aax.flat):
        if iax >= len(all_grids): 
            fig.delaxes(ax)
            continue

        is_last = iax == (len(all_grids) - 1)
        img = all_grids[iax].T

        if is_last:
            img_np = _to_numpy(img)
            uniq = np.unique(img_np)
            # Remap values to contiguous indices for a discrete colorbar.
            img_idx = np.searchsorted(uniq, img_np)
            cmap = plt.get_cmap("tab20", len(uniq))
            bounds = np.arange(-0.5, len(uniq) + 0.5, 1)
            norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(
                img_idx, origin="lower", aspect=1.0, cmap=cmap, norm=norm, interpolation="none",
            )
            tick_locs = np.arange(len(uniq))
            cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
            cbar.ax.set_yticklabels([f"{int(u)}" for u in uniq])
        else:
            im = ax.imshow(img, origin="lower", aspect=1.0, cmap="cubehelix",
                interpolation="none")
            img_np = np.asarray(img)
            img_pos = img_np[img_np > 0]
            vmin = img_pos.min() if img_pos.size else 1e-12
            im.set_norm(LogNorm(vmin=EPS, vmax=100)) #img_np.max()))
            cbar = fig.colorbar(im, ax=ax)

        add_segmented_xticks(ax, x_vals, xsubsets)
        ax_title = f"{iax}) {func_names[iax]}"
        if iax < len(run_times): ax_title += f" ({run_times[iax]:.2f}s)"
        ax.set_title(ax_title)

    fig.suptitle(f"Errors for dtype {aoi.default_dtype}")
    plt.show()


    # search piecewise switch points
    if False:
        search = {}
        for msw in range(1): #range(18, 31):
            for xsw in [0.1, 0.2, 0.5, 1.0]:
                funcs = [gm.fboys_series8, gm.fboys_igamma, gm.fboys_large_m]
                pw_error_grid = np.full_like(error_grids[0], 10**4)
                mask0 = X < xsw
                pw_error_grid[mask0] = error_grids[0][mask0]
                # mask2 = A > msw
                # pw_error_grid[mask2] = error_grids[2][mask2]
                mask1 = ~mask0 # & ~mask2
                pw_error_grid[mask1] = error_grids[1][mask1]
                search[(xsw, msw)] = pw_error_grid.max()
                print(f"({xsw}, {msw}) : {pw_error_grid.max():.4e}  {pw_error_grid.mean():.4e}  {np.median(pw_error_grid):.4e}")

    if aoi.xp.finfo(aoi.default_dtype).dtype == 'float32':
        xsw, msw = 0.5, 26
        funcs = [gm.fboys_series, gm.fboys_igamma, gm.fboys_large_m]
        _, pw_grids, _, _, _ = eval_error_grids(
            funcs, mb, xb, gt, EPS, loops=1, print_times=False
        )
        pw_error_grid = np.full_like(pw_grids[0], 10**4)
        mask0 = _to_numpy(X < xsw)
        pw_error_grid[mask0] = pw_grids[0][mask0]
        mask2 = _to_numpy(A > msw)
        pw_error_grid[mask2] = pw_grids[2][mask2]
        mask1 = ~mask0 & ~mask2
        pw_error_grid[mask1] = pw_grids[1][mask1]
        print(f"({xsw}, {msw}) : {pw_error_grid.max():.4e}  {pw_error_grid.mean():.4e}  {np.median(pw_error_grid):.4e}")

    elif aoi.xp.finfo(aoi.default_dtype).dtype == 'float64':
        xsw = 0.1
        funcs = [gm.fboys_series, gm.fboys_igamma,]
        _, pw_grids, _, _, _ = eval_error_grids(
            funcs, A, X, gt, EPS, loops=1, print_times=False
        )
        pw_error_grid = np.full_like(pw_grids[0], 10**4)
        mask0 = _to_numpy(X < xsw)
        pw_error_grid[mask0] = pw_grids[0][mask0]
        mask1 = ~mask0
        pw_error_grid[mask1] = pw_grids[1][mask1]
        print(f"({xsw}) : {pw_error_grid.max():.4e}  {pw_error_grid.mean():.4e}  {np.median(pw_error_grid):.4e}")

    else:
        raise NotImplementedError()

    fig, ax = plt.subplots()
    im = ax.imshow(pw_error_grid.T, origin="lower", aspect=1.0, cmap="cubehelix",
        interpolation="none")
    img_np = np.asarray(pw_error_grid)
    img_pos = img_np[img_np > 0]
    vmin = img_pos.min() if img_pos.size else 1e-12
    add_segmented_xticks(ax, x_vals, xsubsets)
    im.set_norm(LogNorm(vmin=EPS, vmax=10**4)) #img_np.max()))
    cbar = fig.colorbar(im, ax=ax)
    plt.show()

    
    debug_var = 1
