import torch
import torch.nn.functional as F

def errs(arr):
    err = torch.any(~torch.isfinite(arr)).item()
    if err:
        debug_var = 1



def compute_perplexity(Hab, Hcd, b2, qp, beta):
    """ perplexity (pro?) solution
    Compute:
        ans[n1,n2,i] = beta^i * sum_{l,m,u}^{l+m-u=i}
                           Hab[n1,l] * Hcd[n2,m] * (-1)^(m+u) * b2[l+m,u] / qp[n1,n2]^u

    Shapes:
        Hab: (N1, L)
        Hcd: (N2, M)
        b2:  (L+M+1, Umax+1), where Umax = (L+M)//2
        qp:  (N1, N2)
        beta: scalar
    Output:
        ans: (N1, N2, Imax+1), with Imax = L + M
    """
    device = Hab.device
    dtype = Hab.dtype
    N1, L = Hab.shape
    N2, M = Hcd.shape
    Umax = (L + M) // 2
    Imax = L + M

    # 1. Index ranges
    l_idx = torch.arange(L, device=device)
    m_idx = torch.arange(M, device=device)
    u_idx = torch.arange(Umax + 1, device=device)

    # 2. Build (L,M,U) grids
    l3 = l_idx.view(L, 1, 1)           # (L,1,1)
    m3 = m_idx.view(1, M, 1)           # (1,M,1)
    u3 = u_idx.view(1, 1, -1)          # (1,1,U)

    lm = l3 + m3                       # (L,M,1)
    i_lmu = lm - u3                    # (L,M,U)

    # valid triples where 0 <= i <= Imax
    valid_lmu = (i_lmu >= 0) & (i_lmu <= Imax)   # (L,M,U)

    # 3. b2[l+m,u] and (-1)^(m+u)
    b2_lmu = b2[lm, u3]                # (L,M,U)

    sign_m = (-1.0) ** m_idx           # (M,)
    sign_u = (-1.0) ** u_idx           # (U,)
    sign_lmu = sign_m.view(1, M, 1) * sign_u.view(1, 1, -1)  # (L,M,U)

    coeff_lmu = b2_lmu * sign_lmu      # (L,M,U)
    coeff_lmu = coeff_lmu * valid_lmu  # zero invalid (l,m,u)

    # 4. Contract over l, m with Hab and Hcd to get S[n1,n2,u]
    # Hab: (N1,L), expand to (N1,L,1)
    Hab_exp = Hab.unsqueeze(-1)        # (N1,L,1)

    # Step 1: sum over l -> tmp[n1,m,u]
    # coeff_lmu: (L,M,U)
    tmp = torch.einsum('nlu,lmu->nmu', Hab_exp, coeff_lmu)   # (N1,M,U)

    # Step 2: sum over m with Hcd[n2,m]
    # Hcd: (N2,M)
    S = torch.einsum('nmu,pm->npu', tmp, Hcd)                # (N1,N2,U)

    # 5. Divide by qp^u
    u_pows = u_idx.to(dtype).view(1, 1, -1)                  # (1,1,U)
    qp_pows = qp.unsqueeze(-1) ** u_pows                     # (N1,N2,U)
    T = S / qp_pows                                          # (N1,N2,U)

    # 6. Map u to i using i = l+m-u via a precomputed (U,I) mask
    # Collapse (L,M) to see which (u,i) combos exist
    i_flat = i_lmu.view(-1, Umax + 1)                        # (L*M,U)
    valid_flat = valid_lmu.view(-1, Umax + 1)                # (L*M,U)

    exist_ui = torch.zeros(Umax + 1, Imax + 1,
                           dtype=torch.bool, device=device)  # (U,I)

    for u_ in range(Umax + 1):
        i_u = i_flat[:, u_]
        v_u = valid_flat[:, u_]
        i_u = i_u[v_u]                                      # all i for this u
        if i_u.numel() > 0:
            exist_ui[u_, i_u.unique()] = True

    # 7. Accumulate over u into ans[..., i]
    ans = torch.zeros(N1, N2, Imax + 1, device=device, dtype=dtype)

    for u_ in range(Umax + 1):
        mask_i = exist_ui[u_]                               # (Imax+1,)
        if not mask_i.any():
            continue
        i_list = mask_i.nonzero(as_tuple=False).view(-1)    # (K,)
        contrib = T[..., u_].unsqueeze(-1)                  # (N1,N2,1)
        ans.index_add_(2, i_list, contrib.expand(N1, N2, i_list.numel()))

    # 8. Multiply by beta^i
    # beta_pows = beta ** torch.arange(Imax + 1, device=device, dtype=dtype)  # (Imax+1,)
    # ans = ans * beta_pows.view(1, 1, -1)
    i_idx = torch.arange(Imax + 1, device=device, dtype=dtype)   # (Imax+1,)
    beta_pows = beta.unsqueeze(-1) ** i_idx.view(1, 1, -1)       # (N1,N2,Imax+1)
    ans = ans * beta_pows

    return ans


# def compute_sonnet(Hab, Hcd, b2, qp, beta):
#     """ claude sonnet 4.5
#     Fully vectorized with single loop over u only.
#     Intermediate size bounded by output size.
#     """
#     N1, L_plus_1 = Hab.shape
#     N2, M_plus_1 = Hcd.shape
#     L = L_plus_1 - 1
#     M = M_plus_1 - 1
#     max_i = L + M
#     u_max = (L + M) // 2
    
#     device = Hab.device
#     dtype = Hab.dtype
    
#     # Precompute signs
#     m_idx = torch.arange(M + 1, device=device)
#     sign_m = (-1.0) ** m_idx  # (M+1,)
#     Hcd_signed = Hcd * sign_m.unsqueeze(0)  # (N2, M+1)
    
#     # Precompute: A[n1, s] = sum_{l} Hab[n1,l] where l contributes to sum_lm=s
#     # Similarly: C[n2, s] = sum_{m} Hcd_signed[n2,m] where m contributes to sum_lm=s
#     # But l+m=s, so we need convolution!
    
#     # Key insight: sum over l,m with l+m=s of Hab[n1,l]*Hcd[n2,m] is a convolution!
#     # conv[n1,n2,s] = sum_l Hab[n1,l] * Hcd[n2,s-l]
    
#     # Compute convolution along last axis: (N1, L+1) conv (N2, M+1) -> (N1, N2, L+M+1)
#     # Use FFT-based convolution or direct
    
#     # Direct convolution building (L+M+1) terms:
#     # For each s in [0, L+M], conv[s] = sum_{l=max(0,s-M)}^{min(L,s)} Hab[:,l] * Hcd[:,s-l]
    
#     # Build conv tensor: (N1, N2, L+M+1)
#     # This matches output size, so it's acceptable
    
#     conv = torch.zeros(N1, N2, max_i + 1, device=device, dtype=dtype)
    
#     for s in range(max_i + 1):
#         l_min = max(0, s - M)
#         l_max = min(L, s)
#         l_range = torch.arange(l_min, l_max + 1, device=device)
#         m_range = s - l_range
        
#         # Hab[:, l_range]: (N1, num_l)
#         # Hcd_signed[:, m_range]: (N2, num_l)
#         # Want: sum_l Hab[n1,l] * Hcd[n2,m] = Hab[:, l_range] @ Hcd_signed[:, m_range].T summed correctly
#         # Actually: for each l, it's outer product, then sum
#         # = sum_l (Hab[:,l].unsqueeze(1) * Hcd_signed[:,s-l].unsqueeze(0))
#         # = Hab[:, l_range] @ Hcd_signed[:, m_range].T -- NO, this gives (N1, N2) but sums wrong indices
        
#         # Correct: einsum 'nl,nm->nm' summing over l? No...
#         # For each l: Hab[n1,l] * Hcd_signed[n2, s-l] contributes to conv[n1,n2,s]
#         # So: conv[:,:,s] = sum_l Hab[:,l] outer Hcd_signed[:,s-l]
#         #                 = sum over l of (N1,1)*(1,N2) = sum of (N1,N2)
        
#         # Vectorized: (Hab[:, l_range].unsqueeze(2) * Hcd_signed[:, m_range].unsqueeze(0)).sum(dim=1)
#         # Shapes: (N1, num_l, 1) * (1, num_l, N2) -- wait, need to align
        
#         # Hab[:, l_range]: (N1, num_l) -> (N1, num_l, 1)
#         # Hcd_signed[:, m_range]: (N2, num_l) -> (1, num_l, N2)
#         # Product: (N1, num_l, N2), sum over dim=1 -> (N1, N2)
        
#         Hab_s = Hab[:, l_range]  # (N1, num_l)
#         Hcd_s = Hcd_signed[:, m_range]  # (N2, num_l)
        
#         conv[:, :, s] = torch.einsum('nl,ml->nm', Hab_s, Hcd_s)
    
#     # Now conv[n1,n2,s] = sum_{l+m=s} Hab[n1,l] * (-1)^m * Hcd[n2,m]
    
#     # Original sum: ans[n1,n2,i] = beta^i * sum_{s,u: s-u=i} conv[n1,n2,s] * (-1)^u * b2[s,u] / qp[n1,n2]^u
#     #             = beta^i * sum_{u} conv[n1,n2,i+u] * (-1)^u * b2[i+u,u] / qp[n1,n2]^u
    
#     # For each i: sum over u from 0 to min(u_max, L+M-i)
    
#     ans = torch.zeros(N1, N2, max_i + 1, device=device, dtype=dtype)
    
#     # Single loop over u
#     for u in range(u_max + 1):
#         sign_u = (-1.0) ** u
#         qp_pow_u = qp ** u  # (N1, N2)
        
#         # For this u, contributions go to i where i+u <= L+M, i.e., i <= L+M-u
#         max_i_for_u = max_i - u
        
#         # i ranges from 0 to max_i_for_u
#         # s = i + u ranges from u to max_i
        
#         i_range = torch.arange(max_i_for_u + 1, device=device)
#         s_range = i_range + u  # (num_i,)
        
#         # b2[s, u] for s in s_range
#         b2_vals = b2[s_range, u]  # (num_i,)
        
#         # beta^i for i in i_range
#         # beta_pow_i = beta ** i_range  # (num_i,)
        
#         # Coefficient: beta^i * (-1)^u * b2[i+u, u]
#         # coeff = beta_pow_i * sign_u * b2_vals  # (num_i,)
#         coeff = sign_u * b2_vals  # (num_i,)
        
#         # conv[:, :, s_range]: (N1, N2, num_i)
#         conv_slice = conv[:, :, s_range]  # (N1, N2, num_i)
        
#         # Contribution: conv_slice * coeff / qp^u
#         # (N1, N2, num_i) * (num_i,) / (N1, N2, 1)
#         contrib = conv_slice * coeff.view(1, 1, -1) / qp_pow_u.unsqueeze(-1)
        
#         ans[:, :, :max_i_for_u + 1] += contrib
    
#     return ans

def compute_code44(Hab, Hcd, b2, qp, beta):
    """ some app building bs from claude """
    N1, L_plus_1 = Hab.shape
    N2, M_plus_1 = Hcd.shape
    L = L_plus_1 - 1
    M = M_plus_1 - 1
    max_i = L + M
    u_max = (L + M) // 2
    
    device = Hab.device
    dtype = Hab.dtype
    
    m_idx = torch.arange(M + 1, device=device)
    sign_m = (-1.0) ** m_idx
    Hcd_signed = Hcd * sign_m.unsqueeze(0)
    
    conv = torch.zeros(N1, N2, max_i + 1, device=device, dtype=dtype)
    
    for s in range(max_i + 1):
        l_min = max(0, s - M)
        l_max = min(L, s)
        l_range = torch.arange(l_min, l_max + 1, device=device)
        m_range = s - l_range
        
        Hab_s = Hab[:, l_range]
        Hcd_s = Hcd_signed[:, m_range]
        conv[:, :, s] = torch.einsum('nl,ml->nm', Hab_s, Hcd_s)
    
    ans = torch.zeros(N1, N2, max_i + 1, device=device, dtype=dtype)
    
    for u in range(u_max + 1):
        sign_u = (-1.0) ** u
        qp_pow_u = qp ** u  # (N1, N2)
        max_i_for_u = max_i - u
        
        i_range = torch.arange(max_i_for_u + 1, device=device)
        s_range = i_range + u
        
        b2_vals = b2[s_range, u]  # (num_i,)
        coeff = sign_u * b2_vals  # (num_i,)
        
        conv_slice = conv[:, :, s_range]  # (N1, N2, num_i)
        contrib = conv_slice * coeff.view(1, 1, -1) / qp_pow_u.unsqueeze(-1)
        
        ans[:, :, :max_i_for_u + 1] += contrib
    
    # Apply beta^i at the end
    i_all = torch.arange(max_i + 1, device=device)
    beta_pow = beta.unsqueeze(-1) ** i_all.view(1, 1, -1)  # (N1, N2, max_i+1)
    ans = ans * beta_pow
    
    return ans 


def compute_sonnet(Hab, Hcd, b2, qp, beta):
    """ claude sonnet 4.5
    Efficiently compute summation with no Python loops and controlled memory:
    ans[n1,n2,i] = beta^i * sum_{l,m,u}^{l+m-u=i} 
                   Hab[n1,l] * Hcd[n2,m] * (-1)^(m+u) * b2[l+m,u] / qp[n1,n2]^u
    
    Strategy: Create meshgrid once, use vectorized operations with einsum.
    Memory: O(N1*N2*max_i + L*M*max_u) - no larger than output.
    
    Args:
        Hab: (N1, L+1) tensor
        Hcd: (N2, M+1) tensor  
        b2: (L+M+1, (L+M)//2+1) tensor (or larger)
        qp: (N1, N2) tensor
        beta: scalar or (N1, N2) tensor
    
    Returns:
        ans: (N1, N2, L+M+1) tensor
    """
    N1, L_plus_1 = Hab.shape
    N2, M_plus_1 = Hcd.shape
    L = L_plus_1 - 1
    M = M_plus_1 - 1
    max_i = L + M
    max_u = (L + M) // 2
    
    device = Hab.device
    dtype = Hab.dtype
    
    # Create meshgrid ONCE for (l, m, u)
    l_vals = torch.arange(L + 1, device=device)
    m_vals = torch.arange(M + 1, device=device)
    u_vals = torch.arange(max_u + 1, device=device)
    
    l_grid, m_grid, u_grid = torch.meshgrid(l_vals, m_vals, u_vals, indexing='ij')
    # Shape: (L+1, M+1, max_u+1)
    
    # Compute i = l + m - u for all combinations
    i_grid = l_grid + m_grid - u_grid  # (L+1, M+1, max_u+1)
    
    # Precompute spatial terms (independent of N1, N2)
    # Term: (-1)^(m+u) * b2[l+m, u]
    signs = (-1.0) ** (m_grid + u_grid)  # (L+1, M+1, max_u+1)
    s_grid = l_grid + m_grid  # (L+1, M+1, max_u+1)
    b2_selected = b2[s_grid, u_grid]  # (L+1, M+1, max_u+1)
    
    spatial_term = signs * b2_selected  # (L+1, M+1, max_u+1)
    
    # Precompute qp^(-u) for all u values
    # Shape: (N1, N2, max_u+1)
    # qp_inv_powers = qp[:, :, None] ** (-u_vals[None, None, :])
    qp_pows = torch.pow(qp[:,:,None], u_vals[None,None,:])
    qpmask = torch.where(qp == 0)
    qp_inv_pows = torch.reciprocal(qp_pows)
    qp_inv_pows[*qpmask, 0] = torch.as_tensor(1, dtype=qp.dtype)
    qp_inv_pows[*qpmask, 1:] = torch.as_tensor(0, dtype=qp.dtype)
    errs(qp_inv_pows)
    qp_inv_powers = qp_inv_pows
   
    # Initialize output
    ans = torch.zeros(N1, N2, max_i + 1, device=device, dtype=dtype)
    
    # Process all i values simultaneously using advanced indexing
    # Key insight: Use einsum to contract over (l,m,u) dimensions efficiently
    
    # For each unique i value, we want to sum contributions where l+m-u=i
    # We'll use a different approach: one-hot encode i values and use matmul
    
    # Create one-hot encoding for i values
    # one_hot[l,m,u,i] = 1 if l+m-u==i else 0
    i_grid_flat = i_grid.reshape(-1)  # (K,) where K = (L+1)*(M+1)*(max_u+1)
    valid_mask = (i_grid_flat >= 0) & (i_grid_flat <= max_i)
    
    # Use masked operations
    i_valid = i_grid_flat[valid_mask]  # (K_valid,)
    
    # Create index mapping for scatter
    # We need to map each (l,m,u) to its contribution to i
    
    # Reshape spatial_term and create multi-hot encoding
    spatial_flat = spatial_term.reshape(-1)[valid_mask]  # (K_valid,)
    
    # Get l, m, u indices for valid entries
    l_flat = l_grid.reshape(-1)[valid_mask]  # (K_valid,)
    m_flat = m_grid.reshape(-1)[valid_mask]  # (K_valid,)
    u_flat = u_grid.reshape(-1)[valid_mask]  # (K_valid,)
    
    # Compute contributions using einsum
    # Hab[n1, l] * Hcd[n2, m] * spatial_term[l,m,u] * qp_inv[n1,n2,u]
    
    # Extract needed slices
    Hab_slice = Hab[:, l_flat]  # (N1, K_valid)
    Hcd_slice = Hcd[:, m_flat]  # (N2, K_valid)
    qp_inv_slice = qp_inv_powers[:, :, u_flat]  # (N1, N2, K_valid)
    spatial_slice = spatial_flat[None, None, :]  # (1, 1, K_valid)
    
    # Compute product: (N1, K_valid) * (N2, K_valid) * (N1, N2, K_valid) * (1, 1, K_valid)
    # Result: (N1, N2, K_valid)
    contributions = Hab_slice[:, None, :] * Hcd_slice[None, :, :] * qp_inv_slice * spatial_slice
    
    # Scatter add to output using i indices
    i_indices = i_valid[None, None, :].expand(N1, N2, -1)
    ans.scatter_add_(2, i_indices, contributions)
    
    # Apply beta^i
    i_range = torch.arange(max_i + 1, device=device, dtype=dtype)
    if isinstance(beta, torch.Tensor):
        beta_powers = beta[:, :, None] ** i_range[None, None, :]
    else:
        beta_powers = beta ** i_range[None, None, :]
    
    ans = ans * beta_powers
    
    return ans


def compute_chatgpt(Hab, Hcd, b2, qp, beta):
    """ chatgpt 5.2
    Computes:
      ans[n1,n2,i] = beta^i * sum_{l=0..L} sum_{m=0..M} sum_{u=0..U, l+m-u=i}
                        Hab[n1,l] * Hcd[n2,m] * (-1)^(m+u) * b2[l+m,u] / qp[n1,n2]^u
    where U=(L+M)//2 and i in [0..L+M].

    Intermediates never exceed the output size (N1,N2,S), where S=L+M+1.
    """

    device, dtype = Hab.device, Hab.dtype
    N1, Lp1 = Hab.shape
    N2, Mp1 = Hcd.shape
    L = Lp1 - 1
    M = Mp1 - 1
    S = L + M + 1
    U = (L + M) // 2

    # --- incorporate (-1)^m into Hcd to move sign out of convolution
    m = torch.arange(Mp1, device=device)
    sign_m = (1 - 2 * (m & 1)).to(dtype)          # +1,-1,+1,...
    Hcd_signed = Hcd * sign_m.view(1, -1)

    # If you REALLY meant Hcd[n2,m]^2, use:
    # Hcd_signed = (Hcd * Hcd) * sign_m.view(1, -1)

    # --- full convolution over s=l+m, producing T[n1,n2,s] of shape (N1,N2,S)
    # conv1d computes correlation, so flip kernel to get convolution
    x = Hab.unsqueeze(1)                           # (N1,1,L+1)
    w = Hcd_signed.unsqueeze(1).flip(-1)           # (N2,1,M+1)
    T = F.conv1d(x, w, padding=M)                  # (N1,N2,S)
    errs(T)

    # --- precompute qp^{-u} for u=0..U as (N1,N2,U+1), which is <= output size
    qpmask = qp != 0
    qp_inv = torch.where(qpmask, qp.reciprocal(), torch.ones_like(qp))
    ones = torch.ones_like(qp_inv)[..., None]
    qp_inv_pows = torch.cat([ones, qp_inv[..., None].expand(N1, N2, U)], dim=-1).cumprod(-1)
    qp_inv_pows = torch.where(qpmask[..., None], qp_inv_pows, torch.zeros_like(qp_inv_pows))
    qp_inv_pows[..., 0] = 1.0


    # qp_inv = qp.reciprocal()                       # (N1,N2)
    # # stack [1, qp_inv, qp_inv, ...] then cumprod -> [1, qp_inv^1, ..., qp_inv^U]
    # ones = torch.ones_like(qp_inv)[..., None]      # (N1,N2,1)
    # qp_inv_stack = qp_inv[..., None].expand(N1, N2, U)  # (N1,N2,U)
    # qp_inv_pows = torch.cat([ones, qp_inv_stack], dim=-1).cumprod(dim=-1)  # (N1,N2,U+1)

    # # qp_inv_pows[torch.isinf(qp_inv_pows)] = torch.as_tensor(0, dtype=qp.dtype) # @DEBUG
    # qp_zeros = torch.where(qp == 0.0)
    # # qp_inv_pows[qp_zeros,0] = torch.as_tensor(1, dtype=qp.dtype)
    # qp_inv_pows[*qp_zeros,1:] = torch.as_tensor(0, dtype=qp.dtype)

    errs(qp_inv_pows)

    # --- precompute u-dependent b2 weights into a tiny (U+1,S) table
    # weight_u[i] = (-1)^u * b2[i+u, u] for i=0..S-1-u, else 0
    b2_need = b2[:S, :U+1].to(dtype=dtype, device=device)
    weights = Hab.new_zeros((U + 1, S))            # small
    for u in range(U + 1):
        sign_u = 1.0 if (u % 2 == 0) else -1.0
        weights[u, :S - u] = sign_u * b2_need[u:, u]

    # --- accumulate ans_pre without beta^i, using a reusable buffer (size == output)
    out = Hab.new_zeros((N1, N2, S))
    buf = Hab.new_empty((N1, N2, S))               # reuse; never bigger than out
    errs(out)

    for u in range(U + 1):
        len_i = S - u

        # buf[:,:,:len_i] = T[:,:,u:] * qp^{-u}  (broadcast qp^{-u} over i)
        torch.mul(T[:, :, u:], qp_inv_pows[:, :, u].unsqueeze(-1), out=buf[:, :, :len_i])

        # out[:,:,:len_i] += buf[:,:,:len_i] * weights[u,:len_i]
        out[:, :, :len_i].addcmul_(
            buf[:, :, :len_i],
            weights[u, :len_i].view(1, 1, len_i)
        )

    errs(out)

        # ---- beta^(i) per (n1,n2): build beta_pows with cumprod (size == output)
    # beta_pows[...,0]=1; beta_pows[...,i]=beta^i
    beta_ = beta.to(dtype=dtype, device=device)
    beta_pows = Hab.new_empty((N1, N2, S))
    beta_pows[..., 0] = 1
    if S > 1:
        beta_pows[..., 1:] = beta_.unsqueeze(-1).expand(N1, N2, S - 1)
        beta_pows = beta_pows.cumprod(dim=-1)

    out.mul_(beta_pows)
    errs(out)
    return out


compute_methods = [
    compute_perplexity,
    compute_code44,
    compute_sonnet,
    compute_chatgpt,
]