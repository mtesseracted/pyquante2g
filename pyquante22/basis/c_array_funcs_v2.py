import torch
import torch.nn.functional as F

# idea gleaned from ai answers:
# K size is order L*M*U
# S size is order L + M
# creating a meshgrid of LMU is size K
# so we can create Hab and Hcd intermediates
# that are expanded to meshgrid in LMU, that
# are "only" size (N1,K) and (N2,K), which sould be < (N1,N2,S)
# having a meshgrid opens up the possibility of einsum
# especially for frozen values of [:,:,i] output

def compute_ref(Hab, Hcd, b2, qp, delta):
    r"""slow but correct reference
    ret[n1,n2,i] = sum_{l,m,u}^{l+m-u==i}
        Hab[n1,l] * Hcd[n2,m] * (-1)^(m+u) * b2[l+m,u] * qp^(l+m-2u) / delta^(l+m-u)
    constraints: l+m-2u >= 0
    """

    # Use float64 for reference stability (optional but strongly recommended)
    work_dtype = torch.float64
    Hab = Hab.to(work_dtype)
    Hcd = Hcd.to(work_dtype)
    b2  = b2.to(work_dtype)
    qp  = qp.to(work_dtype)
    delta = delta.to(work_dtype)

    N1, Lp1 = Hab.shape
    N2, Mp1 = Hcd.shape
    L = Lp1 - 1
    M = Mp1 - 1
    S = L + M
    U = (L + M) // 2

    b2 = b2[: S + 1, : U + 1]  # only needed slice

    ret = torch.zeros((N1, N2, S + 1), device=Hab.device, dtype=work_dtype)

    qp_is_zero = (qp == 0)

    for l in range(L + 1):
        Hab_l = Hab[:, l].view(N1, 1)          # (N1,1)
        for m in range(M + 1):
            Hcd_m = Hcd[:, m].view(1, N2)      # (1,N2)
            base_nm = Hab_l * Hcd_m            # (N1,N2)

            # If base_nm == 0, the term is exactly 0 regardless of qp/delta powers.
            base_zero = (base_nm == 0)

            s = l + m
            umax = min(U, s // 2)
            sign_m = -1.0 if (m & 1) else 1.0

            for u in range(umax + 1):
                i = s - u
                k = s - 2*u   # >= 0 by constraint
                sign_u = -1.0 if (u & 1) else 1.0
                coeff = (sign_m * sign_u) * b2[s, u]  # scalar

                # qp^k with qp==0 handled (k never negative)
                if k == 0:
                    qp_pow = torch.ones_like(qp)
                else:
                    qp_pow = torch.where(qp_is_zero, torch.zeros_like(qp), qp.pow(k))

                # tmp = torch.pow(qp, k)
                # assert torch.all(tmp == qp_pow), "qp_pow mismatch"

                # delta^i (delta assumed >0, finite); i >= 0
                if i == 0:
                    frac = qp_pow
                else:
                    frac = qp_pow / delta.pow(i)

                # Avoid 0 * inf -> NaN by forcing frac=0 wherever base_nm==0
                # tmp = base_nm * (coeff * frac)
                frac = torch.where(base_zero, torch.zeros_like(frac), frac)
                # tmp2 = base_nm * (coeff * frac)
                # if not torch.all(tmp==tmp2): #"habcd mask changed something"
                #     debug_var = 1

                ret[:, :, i].add_(base_nm * (coeff * frac))

    # cast back to original dtype if you want
    return ret.to(Hab.dtype)


def compute_ref2(Hab, Hcd, b2, qp, delta):
    r"""slow but correct reference, with rho^i applied AFTER accumulation

    Start from the factored form:
      term = Hab_l * Hcd_m * (-1)^(m+u) * b2[s,u] * rho^i * qp_safe^(-u)
    where:
      s = l+m,  i = s-u,  rho = qp/delta,  qp_safe = (qp==0 ? 1 : qp)

    Because the output is grouped by i, we accumulate:
      acc[:,:,i] += Hab_l*Hcd_m * (-1)^(m+u) * b2[s,u] * qp_safe^(-u)
    and then apply:
      ret[:,:,i] = acc[:,:,i] * rho^i
    """
    orig_dtype = Hab.dtype
    device = Hab.device

    # reference stability
    work_dtype = torch.float64
    Hab = Hab.to(device=device, dtype=work_dtype)
    Hcd = Hcd.to(device=device, dtype=work_dtype)
    b2  = b2.to(device=device, dtype=work_dtype)
    qp  = qp.to(device=device, dtype=work_dtype)
    delta = delta.to(device=device, dtype=work_dtype)

    N1, Lp1 = Hab.shape
    N2, Mp1 = Hcd.shape
    L = Lp1 - 1
    M = Mp1 - 1
    S = L + M
    U = (L + M) // 2

    b2 = b2[: S + 1, : U + 1]
    # qp_safe trick: map zeros -> 1 so qp_safe**(-u) is always finite
    qp_safe = torch.where(qp == 0, torch.ones_like(qp), qp)
    # Precompute qp_safe^(-u) for u=0..U  => (N1,N2,U+1)
    u_exp = torch.arange(U + 1, device=device, dtype=torch.int64)
    qp_neg_pows = qp_safe.unsqueeze(-1).pow(-u_exp)
    # Accumulate WITHOUT rho^i
    acc = torch.zeros((N1, N2, S + 1), device=device, dtype=work_dtype)

    for l in range(L + 1):
        Hab_l = Hab[:, l].view(N1, 1)          # (N1,1)
        for m in range(M + 1):
            Hcd_m = Hcd[:, m].view(1, N2)      # (1,N2)
            base = Hab_l * Hcd_m               # (N1,N2)

            s = l + m
            umax = min(U, s // 2)              # enforces l+m-2u >= 0
            sign_m = -1.0 if (m & 1) else 1.0

            for u in range(umax + 1):
                i = s - u
                sign_u = -1.0 if (u & 1) else 1.0
                coeff = (sign_m * sign_u) * b2[s, u]  # scalar

                # exactly the "non-rho" part
                acc[:, :, i].add_(base * (coeff * qp_neg_pows[..., u]))

    # Apply rho^i AFTER the loops
    rho = qp / delta
    i_exp = torch.arange(S + 1, device=device, dtype=torch.int64)
    rho_pows = rho.unsqueeze(-1).pow(i_exp)  # (N1,N2,S+1)

    ret = acc * rho_pows
    return ret.to(orig_dtype)


def compute_ref3(Hab, Hcd, b2, qp, delta):
    r"""Reference with rho^i applied after loops, correct for qp==0.

    Original:
      ret[:,:,i] += Hab_l*Hcd_m * (-1)^(m+u) * b2[s,u] * qp^(s-2u) / delta^i
      s=l+m, i=s-u, k=s-2u>=0.

    We accumulate acc[:,:,i] for the non-rho part, then apply rho^i where rho=qp/delta.
    The naive factorization loses terms when qp==0 and k==0 (i.e., s==2u).
    We add those back via a separate acc_k0, computed once per (l,m) when s is even.
    """
    orig_dtype = Hab.dtype
    device = Hab.device

    work_dtype = torch.float64
    Hab = Hab.to(device=device, dtype=work_dtype)
    Hcd = Hcd.to(device=device, dtype=work_dtype)
    b2  = b2.to(device=device, dtype=work_dtype)
    qp  = qp.to(device=device, dtype=work_dtype)
    delta = delta.to(device=device, dtype=work_dtype)

    N1, Lp1 = Hab.shape
    N2, Mp1 = Hcd.shape
    L = Lp1 - 1
    M = Mp1 - 1
    S = L + M
    U = (L + M) // 2

    b2 = b2[: S + 1, : U + 1]

    qp_is_zero = (qp == 0)

    # qp_safe only used inside qp^{-u}; safe finite values everywhere
    qp_safe = torch.where(qp_is_zero, torch.ones_like(qp), qp)

    # Precompute qp_safe^{-u} for u=0..U  -> (N1,N2,U+1)
    u_exp = torch.arange(U + 1, device=device, dtype=torch.int64)
    qp_neg_pows = qp_safe.unsqueeze(-1).pow(-u_exp)

    acc = torch.zeros((N1, N2, S + 1), device=device, dtype=work_dtype)
    acc_k0 = torch.zeros_like(acc)  # stores coefficients for k==0 terms

    for l in range(L + 1):
        Hab_l = Hab[:, l].view(N1, 1)
        for m in range(M + 1):
            Hcd_m = Hcd[:, m].view(1, N2)
            base = Hab_l * Hcd_m  # (N1,N2)

            s = l + m
            umax = min(U, s // 2)  # enforces k>=0
            sign_m = -1.0 if (m & 1) else 1.0

            # Main accumulation (no k==0 special-casing here)
            for u in range(umax + 1):
                i = s - u
                sign_u = -1.0 if (u & 1) else 1.0
                coeff = (sign_m * sign_u) * b2[s, u]  # scalar
                acc[:, :, i].add_(base * (coeff * qp_neg_pows[..., u]))

            # Add the k==0 contribution once (if it exists):
            # k==0 <=> s == 2*u0, so u0 = s//2 and i = s-u0 = u0
            if (s & 1) == 0:  # s even
                u0 = s // 2
                if u0 <= U:
                    sign_u0 = -1.0 if (u0 & 1) else 1.0
                    coeff0 = (sign_m * sign_u0) * b2[s, u0]  # scalar
                    acc_k0[:, :, u0].add_(base * coeff0)

    # Apply rho^i after loops
    rho = qp / delta
    i_exp = torch.arange(S + 1, device=device, dtype=torch.int64)
    rho_pows = rho.unsqueeze(-1).pow(i_exp)  # (N1,N2,S+1)
    ret = acc * rho_pows

    # Add back missing k==0 terms for qp==0: (qp^0 / delta^i) = 1/delta^i
    delta_pows = delta.unsqueeze(-1).pow(i_exp)  # (N1,N2,S+1)
    corr = acc_k0 / delta_pows
    corr[..., 0] = 0.0  # i=0 already correct (only u=0 contributes)
    ret = ret + qp_is_zero.unsqueeze(-1).to(work_dtype) * corr

    return ret.to(orig_dtype)


def compute_chatgpt(Hab, Hcd, b2, qp, delta):
    """
    Hab:   (N1, L+1)  [l = 0..L]
    Hcd:   (N2, M+1)  [m = 0..M]
    b2:    (>=S+1, >=U+1) but may be larger than needed
    qp:    (N1, N2)   allowed to be 0
    delta: (N1, N2)
    returns:
      ans: (N1, N2, S+1) where i = 0..S
    """
    device = Hab.device
    dtype  = Hab.dtype

    N1, Lp1 = Hab.shape
    N2, Mp1 = Hcd.shape
    L = Lp1 - 1
    M = Mp1 - 1
    S = L + M
    U = (L + M) // 2

    # ---- slice b2 to only what we need: (S+1, U+1)
    b2 = b2[:S+1, :U+1].to(device=device, dtype=dtype)

    # ---- sign into Hcd: Hcd_signed[m] = (-1)^m * Hcd[m]
    m = torch.arange(Mp1, device=device)
    sign_m = (1 - 2*(m & 1)).to(dtype)  # +1,-1,+1,-1,...
    Hcd_signed = Hcd * sign_m[None, :]

    # ---- Build Toeplitz-like windows for convolution:
    # windows_rev[n2, s, l] = Hcd_signed[n2, s-l] (out of range -> 0)
    # Size: (N2, S+1, L+1) which is <= output since L+1 < N1 (given).
    Hpad = F.pad(Hcd_signed, (L, L))                    # (N2, Mp1 + 2L)
    windows = Hpad.unfold(dimension=-1, size=Lp1, step=1)  # (N2, S+1, L+1)
    windows_rev = windows.flip(-1)                      # reverse within window

    # ---- C[n1,n2,s] = sum_l Hab[n1,l] * windows_rev[n2,s,l]
    # Output size: (N1, N2, S+1) == final output size.
    C = torch.einsum("nl, msl -> nms", Hab, windows_rev)

    # ---- Precompute (qp/delta)^i for i=0..S safely, size == output
    # base = qp/delta, but avoid division-by-zero explosions if desired.
    # If delta==0 and qp!=0, your math is singular; here we set base=0 there.
    base = torch.where(delta != 0, qp / delta, torch.zeros_like(qp))
    ones = torch.ones_like(base)[..., None]             # (N1,N2,1)
    base_stack = base[..., None].expand(N1, N2, S)      # (N1,N2,S)
    qp_over_delta_pows = torch.cat([ones, base_stack], dim=-1).cumprod(dim=-1)
    # qp_over_delta_pows[..., i] = (qp/delta)^i

    # ---- Precompute qp^{-u} safely for u=0..U:
    # define qp^{-u}=0 for u>0 when qp==0, and qp^0=1.
    qp_inv = torch.where(qp != 0, qp.reciprocal(), torch.zeros_like(qp))
    ones_u = torch.ones_like(qp_inv)[..., None]         # (N1,N2,1)
    inv_stack = qp_inv[..., None].expand(N1, N2, U)     # (N1,N2,U)
    qp_inv_pows = torch.cat([ones_u, inv_stack], dim=-1).cumprod(dim=-1)
    # qp_inv_pows[..., u] = qp^{-u} with safe qp==0 behavior

    # ---- Accumulate ans with only a loop over u (U+1 iterations)
    ans = torch.zeros((N1, N2, S+1), device=device, dtype=dtype)

    for u in range(U + 1):
        # valid i: u..S-u  (length = S+1-2u)
        if 2*u > S: break
        n = S + 1 - 2*u
        sign_u = -1.0 if (u & 1) else 1.0

        # s = i+u runs from 2u..S
        C_s   = C[:, :, 2*u : 2*u + n]                 # (N1,N2,n)
        pow_i = qp_over_delta_pows[:, :, u : u + n]    # (N1,N2,n)
        inv_u = qp_inv_pows[:, :, u][:, :, None]       # (N1,N2,1)
        b_su  = b2[2*u : 2*u + n, u].view(1, 1, n)     # (1,1,n)

        ans[:, :, u : u + n].add_(sign_u * C_s * b_su * inv_u * pow_i)

    return ans


def compute_perplexity(Hab, Hcd, b2, qp, delta):
    """
    Hab:   (N1, L)
    Hcd:   (N2, M)
    b2:    at least (L+M+1, (L+M)//2 + 1)
    qp:    (N1, N2)
    delta: (N1, N2)

    Constraints:
      0 <= l < L, 0 <= m < M, 0 <= u <= (L+M)//2
      l + m - u = i
      l + m - 2u >= 0

    ans[n1,n2,i] = (qp/delta)^i * sum_{l,m,u} Hab[n1,l] * Hcd[n2,m]
                   * (-1)^(m+u) * b2[l+m,u] * qp[n1,n2]^(-u)
                   over triples satisfying constraints.
    """
    device = Hab.device
    dtype = Hab.dtype

    N1, L = Hab.shape
    N2, M = Hcd.shape
    Umax = (L + M) // 2
    Imax = L + M

    ans = torch.zeros(N1, N2, Imax + 1, device=device, dtype=dtype)

    # Small epsilon to avoid division by zero
    eps = torch.finfo(dtype).eps

    # Safe qp and delta for exponentiation / division
    # Clamp magnitudes and avoid 0 in denominators
    qp_safe = qp.clone()
    qp_safe_abs = qp_safe.abs().clamp(min=eps)
    qp_safe = torch.sign(qp_safe) * qp_safe_abs

    delta_safe = delta.clone()
    delta_safe_abs = delta_safe.abs().clamp(min=eps)
    delta_safe = torch.sign(delta_safe) * delta_safe_abs

    qp_over_delta = qp_safe / delta_safe       # safe ratio

    qp_nonzero = qp != 0

    for i in range(Imax + 1):
        l = torch.arange(L, device=device)
        u = torch.arange(Umax + 1, device=device)

        Lg, Ug = torch.meshgrid(l, u, indexing='ij')    # (L, Umax+1)

        m = i + Ug - Lg

        # Basic constraints
        valid = (m >= 0) & (m < M)
        # Extra constraint: l + m - 2u >= 0
        valid = valid & (Lg + m - 2 * Ug >= 0)

        if not valid.any():
            continue

        Lg = Lg[valid]          # (K_i,)
        Ug = Ug[valid]          # (K_i,)
        m = m[valid]            # (K_i,)
        K_i = Lg.numel()

        Hab_l = Hab[:, Lg]                      # (N1, K_i)
        Hcd_m = Hcd[:, m]                       # (N2, K_i)
        b2_lm_u = b2[Lg + m, Ug]                # (K_i,)

        sign_mu = ((m + Ug) % 2 * -2 + 1).to(dtype)   # (K_i,)

        Hab_term = Hab_l[:, None, :]            # (N1, 1, K_i)
        Hcd_term = Hcd_m[None, :, :]            # (1, N2, K_i)

        core = Hab_term * Hcd_term              # (N1, N2, K_i)
        core = core * sign_mu                   # (N1, N2, K_i)
        core = core * b2_lm_u                   # (N1, N2, K_i)

        # (qp/delta)^i, safe: i >= 0 so no NaN from zero base
        if i == 0:
            qp_over_delta_i = torch.ones(N1, N2, K_i, device=device, dtype=dtype)
        else:
            qp_over_delta_i = (qp_over_delta.clamp(min=-1e6, max=1e6) ** i)[..., None]

        # qp^{-u}
        Ug_float = Ug.to(dtype=dtype)

        # Start from zeros
        qp_neg_u = torch.zeros(N1, N2, K_i, device=device, dtype=dtype)

        # Case 1: qp != 0, safe negative powers
        if qp_nonzero.any():
            base_qp = qp_safe[..., None]                       # (N1, N2, 1)
            # Clamp magnitude to avoid overflow in large |u|
            base_qp_clamped = base_qp.clamp(min=-1e6, max=1e6)
            qp_neg_u_nonzero = base_qp_clamped ** (-Ug_float)  # (N1, N2, K_i)
            mask_nonzero = qp_nonzero[..., None]
            qp_neg_u = torch.where(mask_nonzero, qp_neg_u_nonzero, qp_neg_u)

        # Case 2: qp == 0; define qp^{-u} = 1 if u == 0 else 0
        if (~qp_nonzero).any():
            mask_zero = (~qp_nonzero)[..., None]
            u_is_zero = (Ug == 0).to(dtype)[None, None, :]     # (1,1,K_i)
            qp_neg_u = torch.where(mask_zero, u_is_zero, qp_neg_u)

        contrib = core * qp_over_delta_i * qp_neg_u           # (N1, N2, K_i)

        # Replace any remaining NaNs/Infs in contrib defensively
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)

        ans[:, :, i] = contrib.sum(dim=-1)

    return ans


def compute_chatgpt2(Hab, Hcd, qp, delta, b2):
    """
    Hab:   (N1, L+1)
    Hcd:   (N2, M+1)
    qp:    (N1, N2)   allowed to be 0
    delta: (N1, N2)   assumed nonzero
    b2:    (>=S+1, >=U+1)
    returns:
      ans: (N1, N2, S+1) where i = 0..S
    """
    device = Hab.device
    dtype  = Hab.dtype

    N1, Lp1 = Hab.shape
    N2, Mp1 = Hcd.shape
    L = Lp1 - 1
    M = Mp1 - 1
    S = L + M
    U = (L + M) // 2

    b2 = b2[:S+1, :U+1].to(device=device, dtype=dtype)

    # ---- sign into Hcd: Hcd_signed[m] = (-1)^m * Hcd[m]
    m = torch.arange(Mp1, device=device, dtype=torch.int64)
    sign_m = (1 - 2*(m & 1)).to(dtype)  # +1,-1,+1,-1,...
    Hcd_signed = Hcd * sign_m[None, :]

    # ---- Toeplitz-like windows for convolution
    Hpad = F.pad(Hcd_signed, (L, L))                         # (N2, Mp1 + 2L)
    windows = Hpad.unfold(dimension=-1, size=Lp1, step=1)    # (N2, S+1, L+1)
    windows_rev = windows.flip(-1)

    # ---- C[n1,n2,s] = sum_l Hab[n1,l] * windows_rev[n2,s,l]
    C = torch.einsum("nl, msl -> nms", Hab, windows_rev)     # (N1,N2,S+1)

    qp_is_zero = (qp == 0)

    # ---- (qp/delta)^i for i=0..S  (delta assumed nonzero)
    base = qp / delta
    ones = torch.ones_like(base)[..., None]                  # (N1,N2,1)
    base_stack = base[..., None].expand(N1, N2, S)           # (N1,N2,S)
    qp_over_delta_pows = torch.cat([ones, base_stack], dim=-1).cumprod(dim=-1)
    # qp_over_delta_pows[..., i] = (qp/delta)^i

    # ---- qp^{-u} safely: define qp^{-u}=0 for u>0 when qp==0
    qp_inv = torch.where(qp != 0, qp.reciprocal(), torch.zeros_like(qp))
    ones_u = torch.ones_like(qp_inv)[..., None]              # (N1,N2,1)
    inv_stack = qp_inv[..., None].expand(N1, N2, U)          # (N1,N2,U)
    qp_inv_pows = torch.cat([ones_u, inv_stack], dim=-1).cumprod(dim=-1)
    # qp_inv_pows[..., u] = qp^{-u} (safe)

    # ---- Main accumulation (factored form)
    ans = torch.zeros((N1, N2, S+1), device=device, dtype=dtype)

    for u in range(U + 1):
        if 2*u > S:
            break
        n = S + 1 - 2*u
        sign_u = -1.0 if (u & 1) else 1.0

        C_s   = C[:, :, 2*u : 2*u + n]                      # s = 2u..S
        pow_i = qp_over_delta_pows[:, :, u : u + n]         # i = u..S-u
        inv_u = qp_inv_pows[:, :, u][:, :, None]            # qp^{-u}
        b_su  = b2[2*u : 2*u + n, u].view(1, 1, n)

        ans[:, :, u : u + n].add_(sign_u * C_s * b_su * inv_u * pow_i)

    # ---- Fix the missing qp==0 cancellation terms (k==0 => s=2u, i=u)
    # For qp==0, the true factor at k==0 is: qp^0 / delta^u = 1 / delta^u
    umax0 = min(U, S // 2)
    if umax0 >= 1:
        uvals = torch.arange(1, umax0 + 1, device=device, dtype=torch.int64)   # 1..umax0
        sidx  = 2 * uvals                                                      # s=2u
        sign_u = (1 - 2*(uvals & 1)).to(dtype)                                 # (-1)^u

        # inv_delta^i for i=0..S
        inv_delta = delta.reciprocal()
        invd_stack = inv_delta[..., None].expand(N1, N2, S)                    # (N1,N2,S)
        inv_delta_pows = torch.cat([torch.ones_like(inv_delta)[..., None], invd_stack], dim=-1).cumprod(dim=-1)
        # inv_delta_pows[..., i] = 1/delta^i

        # correction[:,:,u] = (-1)^u * C[:,:,2u] * b2[2u,u] * 1/delta^u   (only when qp==0)
        C_diag = C[:, :, sidx]                                                # (N1,N2,nu)
        b_diag = b2[sidx, uvals].to(dtype).view(1, 1, -1)                      # (1,1,nu)
        invd_u = inv_delta_pows[:, :, uvals]                                   # (N1,N2,nu)

        corr = (sign_u.view(1, 1, -1) * C_diag * b_diag * invd_u)
        corr = corr * qp_is_zero.to(dtype).unsqueeze(-1)

        # add into ans at i = u
        ans.index_add_(2, uvals, corr)

    return ans


def compute_c_self(Hab, delta, b2):
    """
    Specialized version for Hcd == Hab and qp == 0.

    Hab:   (N, L+1)
    delta: (N, N)   assumed nonzero
    b2:    (>=2L+1, >=L+1)
    returns:
      ans: (N, N, 2L+1) where i = 0..2L
      (Only i=0..L can be nonzero under qp==0.)
    """
    device = Hab.device
    dtype = Hab.dtype

    N, Lp1 = Hab.shape
    L = Lp1 - 1
    S = 2 * L
    U = L

    b2 = b2[:S + 1, :U + 1].to(device=device, dtype=dtype)

    # ---- sign into Hcd (Hcd == Hab): Hcd_signed[m] = (-1)^m * Hab[m]
    m = torch.arange(Lp1, device=device, dtype=torch.int64)
    sign_m = (1 - 2 * (m & 1)).to(dtype)  # +1,-1,+1,-1,...
    Hcd_signed = Hab * sign_m[None, :]

    # ---- Toeplitz-like windows only for even s = 2u
    # step=2 yields windows for s = 0,2,4,...,2L
    Hpad = F.pad(Hcd_signed, (L, L))                          # (N, L+1 + 2L)
    windows_even = Hpad.unfold(dimension=-1, size=Lp1, step=2)  # (N, L+1, L+1)
    windows_even = windows_even.flip(-1)

    # ---- C_even[n1,n2,u] = sum_l Hab[n1,l] * Hcd_signed[n2,2u-l]
    C_even = torch.einsum("nl, msl -> nms", Hab, windows_even)  # (N,N,U+1)

    # ---- 1/delta^i for i=0..U  (delta assumed nonzero)
    inv_delta = delta.reciprocal()
    ones = torch.ones_like(inv_delta)[..., None]               # (N,N,1)
    inv_stack = inv_delta[..., None].expand(N, N, U)           # (N,N,U)
    inv_delta_pows = torch.cat([ones, inv_stack], dim=-1).cumprod(dim=-1)

    # ---- b2[2i,i] and (-1)^i factors
    i = torch.arange(U + 1, device=device, dtype=torch.int64)
    b_diag = b2[2 * i, i]                                      # (U+1,)
    sign_i = (1 - 2 * (i & 1)).to(dtype)                       # (-1)^i

    scale = sign_i.view(1, 1, -1) * b_diag.view(1, 1, -1)
    core = C_even * scale * inv_delta_pows                    # (N,N,U+1)

    ans = torch.zeros((N, N, S + 1), device=device, dtype=dtype)
    ans[:, :, :U + 1] = core
    return ans



compute_methods = [
    compute_ref3,
    compute_chatgpt2,
    # compute_perplexity,
    # compute_sonnet,
]

