# Math functions using python's native math.
# Use python's arbitrary int to do factorial, binomials, etc.
from functools import lru_cache
from math import pi, sqrt, log, exp, gamma, lgamma, log1p, erfc


@lru_cache(maxsize=None)
def fact_exact(n:int, dn=1): # use python's arbitrary int
    """ n! (or n!! if dn=2) """
    if n < 2:
        return 1
    return n * fact_exact(n-dn, dn=dn)


@lru_cache(maxsize=None)
def binom_exact(n:int, k:int): # use python's arbitrary int
    """ n! / (k! (n-k)!)"""
    if k > n:
        return 0
    return fact_exact(n) // (fact_exact(k) * fact_exact(n-k))


@lru_cache(maxsize=None)
def binom2k_exact(n:int, k:int): # use python's arbitrary int
    """ n! / (k! (n-2k)!)"""
    if 2*k > n:
        return 0
    return fact_exact(n) // (fact_exact(k) * fact_exact(n-2*k))


@lru_cache(maxsize=None)
def binom2n_exact(n:int, k:int):
    """ (2n)! / (k! (n-k)!) """
    if k > n:
        return 0
    return fact_exact(2*n) // (fact_exact(k) * fact_exact(n-k))


@lru_cache(maxsize=None)
def binom22_exact(i:int, r:int, u:int):
    """ i! / (r! u! (i-2(r+u))!) 
    if all inputs are int, the result should be int
    proof: multiply out b(i,r)b(i-r,r)b(i-2r,u)b(i-2r-u,u) where b(n,k) is biomial
    and you get binom22() times r!u!, so clearly int.
    """
    if 2*(r+u) > i: return 0
    return fact_exact(i) // (
        fact_exact(r) * fact_exact(u) * fact_exact(i-2*(r+u))
    )


@lru_cache(maxsize=None)
def gamma_half_int(n): # 
    """sqrt(pi) * (2n-1)!! / 2**n = Gamma(n+1/2)"""
    return gamma(n + 0.5) #/ sqrt(pi)


def _lower_gamma_small_x(a, x, eps=1e-16, max_iter=10000):
    r"""Series for \gamma(a,x) valid for small x."""
    if x == 0.0:
        return 0.0
    # term_0 = x^a / a
    term = exp(a * log(x)) / a
    s = term
    n = 0
    while n < max_iter:
        n += 1
        # term_n / term_{n-1} = -x * (a+n-1) / (n * (a+n))
        term *= -x * (a + n - 1.0) / (n * (a + n))
        s += term
        if abs(term) < abs(s) * eps:
            break
    return s
#end def _lower_gamma_small_x(a, x, eps=1e-16, max_iter=10000):


def _lower_gamma_series(a, x, eps=1e-16, max_iter=1000):
    r"""Series for lower incomplete gamma \gamma(a,x).
    \gamma(a, x) = e^{-x} x^a \sum_{n=0}^{\infty} \frac{x^n}{a (a+1)\cdots(a+n)}
    """
    if x == 0.0: return 0.0
    ap = a
    term = 1.0 / a
    s = term
    for _ in range(1, max_iter + 1):
        ap += 1.0
        term *= x / ap
        s += term
        if abs(term) < abs(s) * eps: break
    return s * exp(-x + a * log(x))
#end def _lower_gamma_series(a, x, eps=1e-16, max_iter=1000):


def _upper_gamma_contfrac(a, x, eps=1e-16, max_iter=1000):
    r"""Continued fraction for upper incomplete gamma \Gamma(a,x).
    Uses a Lentz-type algorithm such that \Gamma(a, x) = e^{-x} x^a h,
    """
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN: d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h * exp(-x + a * log(x))
#end def _upper_gamma_contfrac(a, x, eps=1e-16, max_iter=1000):


def igamma(a, x):
    r"""\gamma(a,x)=\int_0^x t^{a-1}e^{-t}dt for a>0,x>=0."""
    if a <= 0.0: raise ValueError("Parameter 'a' must be positive.")
    if x < 0.0: raise ValueError("Parameter 'x' must be non-negative.")
    if x == 0.0: return 0.0
    if x < 1.0: # small x series needed for large a
        return _lower_gamma_small_x(a, x)
    elif x < a + 1.0: # compute via series
        return _lower_gamma_series(a, x)
    else: # Gamma - upper gamma via continued fraction
        return gamma(a) - _upper_gamma_contfrac(a, x)
#end def igamma(a, x):


def log_igamma(a, x, eps=1e-16, max_iter=10000):
    r"""log(\gamma(a,x)) for a>0, x>=0."""
    if a <= 0.0: raise ValueError("Parameter 'a' must be positive.")
    if x < 0.0: raise ValueError("Parameter 'x' must be non-negative.")
    if x == 0.0: return float("-inf")
    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        s = term
        for _ in range(1, max_iter + 1):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * eps: break
        if s <= 0.0: return float("-inf")
        return log(s) - x + a * log(x)
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN: d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    gln = lgamma(a)
    diff = -x + a * log(x) + log(h) - gln
    if diff < -50.0: return gln
    if diff > 0.0: diff = 0.0
    return gln + log1p(-exp(diff))
#end def log_igamma(a, x, eps=1e-16, max_iter=10000):


def _fboys_ser(a, x, eps=1e-16, max_terms=100):
    r"""
    Compute 0.5 * x^{-a} * gamma(a, x) for small x in [1e-10, 1e-2]
    using the series:
        0.5 * x^{-a} * gamma(a,x) = 0.5 * sum_{n=0}^\infty (-1)^n x^n / (n!(a+n)).
    """
    term = 0.5 / a  # n = 0 term: 0.5 / a
    if x < eps:    # limit as x -> 0 is 0.5 / a
        return term

    s = term
    for n in range(max_terms): # calc with a stable upwards recurrence
        # t_{n+1} = t_n * [ -x * (a+n) / ((n+1)(a+n+1)) ]
        term *= -x * (a + n) / ((n + 1.0) * (a + n + 1.0))
        s += term
        if abs(term) < eps * abs(s):
            break
    return s


def fboys_tested(m, x, zero_tol=1e-8, large_m=20):
    r""" Boys function F_m(x).
    F_m(x) = \int_0^1 t^{2m} e^{-x t^2} dt
           = 0.5 * x^{-(m+0.5)} * \gamma(m + 0.5, x)
    """
    lo = exp(-x)
    m2 = m + 0.5
    upper = 0.5 / m2            # upper limit
    lower = exp(-x) * upper     # lower limit
    if (x < zero_tol):
        # return (0.5 / m2) - (x / (2*m + 3)) + (x**2 / 6*(2*m + 5))
        return _fboys_ser(m2, x) # heavy handed
    if m2 >= large_m: # igamma may be innacurate
        if x < 1:
            return _fboys_ser(m2, x)
        return 0.5 * exp(log_igamma(m2, x) - m2*log(x))
    try:
        return 0.5 *  (x ** -m2) * igamma(m2, x)
    except OverflowError:
        debug_var = 1


def fboys(m, x, series_x=0.5, xoff=1.5, log_m=1):
    r""" Boys function F_m(x).
    F_m(x) = \int_0^1 t^{2m} e^{-x t^2} dt
           = 0.5 * x^{-(m+0.5)} * \gamma(m + 0.5, x)
    """
    m2 = m + 0.5
    if x < series_x: # found from testing
        return _fboys_ser(m2, x) # 
    if (m > log_m) and (m > (x-xoff)): # found from testing
        return 0.5 * exp(log_igamma(m2, x) - m2*log(x))
    return 0.5 *  (x ** -m2) * igamma(m2, x)



def P_edgeworth(a, x):
    # regularized lower incomplete gamma P(a,x) using Edgeworth around x ~ a
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if x < 0.0:
        raise ValueError("x must be non-negative.")

    tau = (x - a) / sqrt(a)
    phi = exp(-0.5 * tau * tau) / (sqrt(2.0 * 3.141592653589793))
    Phi = 0.5 * erfc(-tau / sqrt(2.0))

    t2 = tau * tau
    t3 = t2 * tau
    t5 = t3 * t2

    c1 = (t2 - 1.0) / (3.0 * sqrt(a))
    c2 = (t3 - 3.0 * tau) / (4.0 * a)
    c3 = (t5 - 10.0 * t3 + 15.0 * tau) / (18.0 * a)

    return Phi + phi * (c1 + c2 + c3)

def f_scaled_edgeworth(a, x):
    # f(a,x) = 0.5 * x^(-a) * gamma(a,x) using the central Edgeworth expansion
    if x == 0.0:
        return 0.5 / a  # limit

    P = P_edgeworth(a, x)
    if P <= 0.0:
        return 0.0  # or handle underflow/log version
    return 0.5 * exp(lgamma(a) - a * log(x) + log(P))



def _f_integrand(u, a, x):
    if u == 0.0:
        u = 1e-16
    return (u ** (a - 1.0)) * exp(-x * u)

def _simpson(fa, fm, fb, a, b):
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0

def _adaptive_simpson(f, a, b, fa, fm, fb, S, eps, depth, a_param, x_param):
    if depth <= 0:
        return S
    c = 0.5 * (a + b)
    ld = 0.5 * (a + c)
    rd = 0.5 * (c + b)
    f_ld = f(ld, a_param, x_param)
    f_rd = f(rd, a_param, x_param)
    S_left  = _simpson(fa,  f_ld, fm, a, c)
    S_right = _simpson(fm,  f_rd, fb, c, b)
    S2 = S_left + S_right
    if abs(S2 - S) < 15.0 * eps:
        return S2 + (S2 - S) / 15.0  # Richardson refinement
    return (_adaptive_simpson(f, a, c, fa, f_ld, fm, S_left,  eps / 2.0, depth - 1, a_param, x_param) +
            _adaptive_simpson(f, c, b, fm, f_rd, fb, S_right, eps / 2.0, depth - 1, a_param, x_param))

def f_quad(a, x, eps=1e-15):
    """Compute f(a,x) = 0.5 * x^{-a} * gamma(a,x) via integral on [0,1]."""
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if x < 0.0:
        raise ValueError("x must be non-negative.")
    if x == 0.0:
        # limit x -> 0: gamma(a,0)=0, but your scaled f -> 0.5/a
        return 0.5 / a

    fa = _f_integrand(0.0, a, x)
    fb = _f_integrand(1.0, a, x)
    m  = 0.5
    fm = _f_integrand(m, a, x)
    S0 = _simpson(fa, fm, fb, 0.0, 1.0)
    return 0.5 * _adaptive_simpson(_f_integrand, 0.0, 1.0, fa, fm, fb, S0, eps, 15, a, x)



if __name__ == "__main__":
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from scipy.ndimage import binary_opening, binary_closing
    import pyquante2g.arb_math as am

    def five_number_summary(data):
        """ Compute min, 25th percentile, median, 75th percentile, max """
        q75, q50, q25 = np.percentile(data, [75, 50, 25])
        return np.stack([np.amin(data), q25, q50, q75, np.amax(data)])

    def fboys0(m,x): 
        return _fboys_ser(m+0.5,x)
        # return fboys(m,x)
        # return f_quad(m+0.5,x)
    def fboys1(m,x): 
        if x == 0.0:
            return 0.5 / (m+0.5)
        return 0.5 *  (x ** -(m+0.5)) * igamma(m+0.5, x)
    def fboys2(m,x): 
        if x == 0.0:
            return 0.5 / (m+0.5)
        return 0.5 *exp(log_igamma(m+0.5, x) - (m+0.5)*log(x))
    funcs = [fboys0, fboys1, fboys2,]
    func_names = ["series@x=0", "igamma", "log_igamma",]
    # func_names = ["fquad", "igamma", "log_igamma",]


    a_vals = np.arange(0, 31, 0.2)
    x_vals = np.arange(0, 31, 0.2)
    A, X = np.meshgrid(a_vals, x_vals, indexing="ij")
    exact_vec = np.vectorize(am.fboys)
    E_exact = exact_vec(A, X).astype(np.float64)
    abs_exact = np.abs(E_exact)
    abs_fboys = np.abs(np.vectorize(fboys)(A, X))

    error_grids = []
    for ii, f in enumerate(funcs):
        f_vec = np.vectorize(f)
        E_f = f_vec(A, X)
        diff = np.abs(E_f - E_exact)
        rel_err = np.where(abs_exact > 0, diff / abs_exact, diff)  # fallback to abs error if exact=0
        error_grids.append(rel_err)

    
    err_best = np.minimum.reduce(error_grids)
    err_best[abs_exact < sys.float_info.epsilon] = 0.0  # ignore errors where exact is ~0
    all_grids = error_grids + [err_best]
    func_names += ["minimum error"]
    # all_grids[0] = abs_exact
    # func_names[0] = "|F_m(x)|"
    debug_var = 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    xt = x_vals[::max(1, len(x_vals)//20)]
    yt = a_vals[::max(1, len(a_vals)//20)]

    for ax, err, title in zip(axes, all_grids, func_names):
        err = np.clip(err, None, 1e0)
        v = err[np.isfinite(err) & (err > 0)]
        emin, emax = v.min(), v.max()
        kmin, kmax = int(np.floor(np.log10(emin))), int(np.ceil(np.log10(emax)))
        bounds = 10.0 ** np.arange(kmin, kmax + 0.5, 0.25)   # 2 bins/decade
        n_bins = len(bounds) - 1

        base = plt.get_cmap("tab20") if abs(kmax-kmin) <= 10 else plt.get_cmap("tab20")
        cmap = colors.ListedColormap(np.resize(base.colors, (n_bins, 3)))
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(err, origin="lower", aspect="auto",
            extent=[x_vals[0], x_vals[-1], a_vals[0], a_vals[-1]],
            norm=norm, cmap=cmap,)
        ax.set_title(title)
        ax.set_xticks(xt); ax.set_yticks(yt)
        ax.set_xlabel("x")
        if ax in (axes[0], axes[2]): ax.set_ylabel("a")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("relative error")
        cbar.set_ticks(10.0 ** np.arange(kmin, kmax + 1, 1))
        cbar.set_ticklabels([f"1e{k}" for k in range(kmin, kmax + 1)])

    plt.tight_layout()
    plt.show()

    debug_var = 1

    # imshow regions of best performance by function
    abs_errs = [np.abs(g) for g in error_grids]
    err_stack = np.stack(abs_errs, axis=0)          # (n_funcs, ny, nx)
    winner = np.argmin(err_stack, axis=0)           # (ny, nx), values in [0, n_funcs-1]
    clean = winner.copy()
    n_funcs = err_stack.shape[0]
    cmap = colors.ListedColormap(["tab:blue", "tab:orange", "tab:green"][:n_funcs])
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(clean, origin="lower", interpolation="none",
        cmap=cmap, vmin=-0.5, vmax=n_funcs - 0.5,)
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_funcs))
    cbar.ax.set_yticklabels(func_names[:len(error_grids)])
    cbar.set_label("Function with lowest |error|")
    ax.set_xlabel("x index")
    ax.set_ylabel("a index")
    ax.set_title("Best approximation by grid point")
    # ax.set_xticks(np.arange(len(x_vals))[::6])
    ax.set_xticks(np.linspace(0, len(x_vals)-1, len(xt)))
    ax.set_xticklabels([f"{x:.1f}" for x in xt])
    # ax.set_yticks(np.arange(len(a_vals)))
    ax.set_yticks(np.linspace(0, len(a_vals)-1, len(yt)))
    ax.set_yticklabels([f"{a:.1f}" for a in yt])
    plt.tight_layout()
    plt.show()


    max_min = 1.0
    med_min = 1.0
    for sx in [0.1, 0.2, 0.4, 0.5]:
        for xo in [1.0, 1.3, 1.4, 1.5, 1.6, 2.0]:
            for lm in [0.5, 1.0, 1.5, 2.0, 5.0]:
                fb = np.vectorize(lambda m, x: fboys(m, x, series_x=sx, xoff=xo, log_m=lm))
                fbax = fb(A,X)
                diff = np.abs(fbax - E_exact)
                rel_err = np.where(abs_exact > 0, diff / abs_exact, diff)  # fallback to abs error if exact=0
                stats = five_number_summary(rel_err)
                if stats[-1] <= max_min and stats[2] <= med_min:
                    med_min = stats[2]
                    max_min = stats[-1]
                    min_arg = (sx, xo, lm)
                sstr = "  ".join(f"{v:.4e}" for v in stats)
                print(f"{sx} {xo} {lm}: {sstr}")
    print(f"Best min/max at series_x={min_arg[0]}, xoff={min_arg[1]}, log_m={min_arg[2]}")

    debug_var = 1
