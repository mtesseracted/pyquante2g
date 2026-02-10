# restricted Hatree-Fock 
import os, time

debug = True

def density_rms(d_new, d_old, ao):
    diff = d_new - d_old
    denom = diff.reshape(-1).shape[0]
    return ao.sqrt(ao.sum(diff * diff) / denom)


def orthogonalizer(s_mat, ao):
    evals, evecs = ao.eigh(s_mat)
    neg_count = (evals <= 0).sum()
    if hasattr(neg_count, "item"):
        neg_count = neg_count.item()
    if neg_count:
        raise ValueError("Overlap matrix has non-positive eigenvalues")
    return (evecs / ao.sqrt(evals)) @ ao.transpose(evecs)


def guess_density_from_core(hcore, s_mat, nocc, ao):
    """Build a closed-shell density matrix from the core Hamiltonian."""
    s_inv_sqrt = orthogonalizer(s_mat, ao)

    f_prime = s_inv_sqrt @ hcore @ s_inv_sqrt
    _, c_prime = ao.eigh(f_prime)
    coeffs = s_inv_sqrt @ c_prime

    c_occ = coeffs[:, :nocc]
    density = 2.0 * (c_occ @ ao.transpose(c_occ))
    return density



def build_fock_matrix(hcore, eri, density, ao, restriced=True):
    """Build the Fock matrix: F = H + 2J - K."""
    j_mat = ao.einsum("kl,ijkl->ij", density, eri)
    k_mat = ao.einsum("kl,ikjl->ij", density, eri)
    if restriced:
        return hcore + j_mat - 0.5 * k_mat
    return hcore + (2.0 * j_mat) - k_mat


def solve_fock(fock, s_inv_sqrt, nocc, ao):
    xp = ao.xp
    f_prime = s_inv_sqrt @ fock @ s_inv_sqrt
    eps, c_prime = xp.linalg.eigh(f_prime)
    coeffs = s_inv_sqrt @ c_prime
    c_occ = coeffs[:, :nocc]
    density = 2.0 * (c_occ @ ao.transpose(c_occ))
    return density, eps, coeffs


def build_fock_jk(hcore, eri, density, ao):
    """
    Build RHF Fock from a *spin-summed* closed-shell density D = 2*Cocc*Cocc^T.

    Conventions (chemist's ERI): eri[p,q,r,s] = (pq|rs)

      J_pq = sum_rs D_rs (pq|rs)
      K_pq = sum_rs D_rs (pr|qs)

    For spin-summed D, the RHF Fock is:
      F = H + J - 1/2 K

    Returns: (F, J, K, G) where G = J - 1/2 K
    """
    J = ao.einsum("rs,pqrs->pq", density, eri)
    K = ao.einsum("rs,prqs->pq", density, eri)
    G = J - 0.5 * K
    F = hcore + G
    return F, J, K, G


def build_jk_batch(density, integral_engine, batch_size=64):
    ie, ao = integral_engine, integral_engine.ao
    basis = ie.basis
    ranges = basis.shell_range

    if len(ie.shell_pairs) == 0:
        ie.fill_screened_overlap_pairs()

    j_mat = ao.zeros((basis.nbf,) * 2, dtype=density.dtype)
    k_mat = ao.zeros((basis.nbf,) * 2, dtype=density.dtype)

    # Build J and K shell-pair row by shell-pair row using calc_eri_batch.
    # Each call returns ERI quartets for one pair-row in canonical pair order.
    for ids in sorted(ie.shell_pairs.keys()):
        i, j = ids
        ri, rj = ranges[i], ranges[j]
        j_block = ao.zeros((basis.shells[i].nbf, basis.shells[j].nbf), dtype=density.dtype)

        # Hoist density slices used repeatedly in K contractions.
        d_j_blocks = [density[rj, ranges[s]] for s in range(basis.nsh)]
        d_i_blocks = None
        if i != j:
            d_i_blocks = [density[ri, ranges[s]] for s in range(basis.nsh)]

        eri_row = ie.calc_eri_batch(ids, batch_size=batch_size, only_triu=False)
        for qids, eri_block in eri_row.items():
            if qids[:2] == ids:
                k, l = qids[2], qids[3]
                eri_ijkl = eri_block
            else:
                # eri_block is (kl|ij) in this branch; transpose to (ij|kl).
                k, l = qids[0], qids[1]
                eri_ijkl = ao.transpose(eri_block, (2, 3, 0, 1))

            rk, rl = ranges[k], ranges[l]

            d_kl = density[rk, rl]
            # calc_eri_batch iterates unique shell pairs (k <= l). To match
            # dense AO contractions, use (ij|kl) = (ij|lk) and fold in D_lk^T.
            if k != l:
                d_lk = density[rl, rk]
                d_kl_sym = d_kl + ao.transpose(d_lk, (1, 0))
                j_block = j_block + ao.einsum("kl,ijkl->ij", d_kl_sym, eri_ijkl)
            else:
                j_block = j_block + ao.einsum("kl,ijkl->ij", d_kl, eri_ijkl)

            # Build K from the same (ij|kl) block:
            #   K_ik = sum_jl D_jl (ij|kl)
            d_jl = d_j_blocks[l]
            k_mat[ri, rk] = k_mat[ri, rk] + ao.einsum("jl,ijkl->ik", d_jl, eri_ijkl)

            # Add swapped (l, k) contribution when k != l.
            if k != l:
                d_jk = d_j_blocks[k]
                eri_ijlk = ao.transpose(eri_ijkl, (0, 1, 3, 2))
                k_mat[ri, rl] = k_mat[ri, rl] + ao.einsum("jk,ijlk->il", d_jk, eri_ijlk)

            # If i != j, also add the swapped (j, i) outer-pair contributions.
            if i != j:
                d_il = d_i_blocks[l]
                eri_jikl = ao.transpose(eri_ijkl, (1, 0, 2, 3))
                k_mat[rj, rk] = k_mat[rj, rk] + ao.einsum("il,jikl->jk", d_il, eri_jikl)

                if k != l:
                    d_ik = d_i_blocks[k]
                    eri_jilk = ao.transpose(eri_jikl, (0, 1, 3, 2))
                    k_mat[rj, rl] = k_mat[rj, rl] + ao.einsum("ik,jilk->jl", d_ik, eri_jilk)

        j_mat[ri, rj] = j_block
        if i != j:
            j_mat[rj, ri] = ao.transpose(j_block, (1, 0))

    return j_mat, k_mat


def build_jk_batch_block(density, integral_engine, batch_size=64):
    ie, ao = integral_engine, integral_engine.ao
    basis = ie.basis
    ranges = basis.shell_range
    j_mat = ao.zeros((basis.nbf,) * 2, dtype=density.dtype)
    k_mat = ao.zeros((basis.nbf,) * 2, dtype=density.dtype)

    for ids in sorted(ie.shell_pairs.keys()):
        i, j = ids
        ri, rj = ranges[i], ranges[j]
        j_block = ao.zeros((basis.shells[i].nbf, basis.shells[j].nbf))

        for kds, eris, k_bounds, l_bounds in ie.calc_eri_batch_block(ids, batch_size=batch_size):
            if len(kds) == 0:
                continue

            # kids = ao.concat([ao.arange(*lims) for lims in k_bounds.values()])
            # lids = ao.concat([ao.arange(*lims) for lims in l_bounds.values()])
            kids = ao.concat([ao.arange(ranges[k].start, ranges[k].stop) for k in k_bounds.keys()])
            lids = ao.concat([ao.arange(ranges[k].start, ranges[k].stop) for k in l_bounds.keys()])

            d_kl = density[kids[:, None], lids[None, :]]
            j_block += ao.einsum("kl,ijkl->ij", d_kl, eris)

            # need to add the lk symm term while avoiding the l==k double-counting
            mask = lids[:, None] != kids[None, :]
            d_lk = density[lids[:, None], kids[None, :]] * mask
            j_block += ao.einsum("lk,ijkl->ij", d_lk, eris)

            # d_jl = density[rj, lids]
            # k_block = ao.einsum("jl,ijkl->ik", d_jl, eris)
            # k_mat[ri, kids] += k_block
            # need to add in a swapped symmetry, not sure which...

            debug_var = 1
        j_mat[ri, rj] = j_block
        if i != j:
            j_mat[rj, ri] = ao.transpose(j_block, (1, 0))

    return j_mat, k_mat


def scf_rhf(
    integral_engine,
    batch_size=64,
    max_iter=50,
    e_conv=1e-8,
    d_conv=1e-6,
    verbose=True,
):
    ie, ao = integral_engine, integral_engine.ao
    t0 = time.perf_counter()
    s_mat, t_mat, v_mat = ie.one_e_mats.values()
    hcore = t_mat + v_mat
    eri4d = ie.build_eri_4d()

    nocc = ie.basis.molecule.nclosed()
    s_inv_sqrt = orthogonalizer(s_mat, ao)
    density = guess_density_from_core(hcore, s_mat, nocc, ao)
    enuc = ie.basis.molecule.nuclear_repulsion()

    e_last = None
    fock = None
    eps = None
    coeffs = None

    # ---- SCF ----
    for it in range(1, max_iter + 1):
        # Build Fock from the *current* density
        if batch_size is not None and batch_size > 0:
            j_mat, k_mat = build_jk_batch(density, ie, batch_size=batch_size)
            g_mat = j_mat - 0.5 * k_mat
            fock = hcore + g_mat

        else:
            fock, j_mat, k_mat, g_mat = build_fock_jk(hcore, eri4d, density, ao)

        if debug:
            jtmp, ktmp = build_jk_batch(density, ie, batch_size=batch_size)
            fock, j_mat, k_mat, g_mat = build_fock_jk(hcore, eri4d, density, ao)
            diff = jtmp - j_mat
            mask = j_mat > 0
            perrs = diff[mask] / j_mat[mask]
            jpmax = ao.abs(perrs).max()
            assert jpmax < 1e-14, "jtmp mismatch"
            diff = ktmp - k_mat
            mask = k_mat > 0
            perrs = diff[mask] / k_mat[mask]
            kpmax = ao.abs(perrs).max()
            assert kpmax < 1e-14, "ktmp mismatch"


            jtmp, ktmp = build_jk_batch_block(density, ie, batch_size=batch_size)
            diff = jtmp - j_mat
            mask = j_mat > 0
            perrs = diff[mask] / j_mat[mask]
            jpmax = ao.abs(perrs).max()
            diff = ktmp - k_mat
            mask = k_mat > 0
            perrs = diff[mask] / k_mat[mask]
            kpmax = ao.abs(perrs).max()
            assert jpmax < 1e-14, "jtmp mismatch"
            assert kpmax < 1e-14, "ktmp mismatch"


        # Compute a CONSISTENT energy for this same density (no extra Fock build)
        # E_elec = Tr[D H] + 1/2 Tr[D G], where G = J - 1/2 K for spin-summed D
        e_elec = ao.sum(density * hcore) + 0.5 * ao.sum(density * g_mat)
        e_tot = e_elec + enuc

        # Solve Roothaan equations to get the updated density
        d_new, eps, coeffs = solve_fock(fock, s_inv_sqrt, nocc, ao)

        # Convergence metrics: delta-E between successive "current-density" energies,
        # and RMS density change
        e_tot_val = float(e_tot.item())
        rms_val = float(density_rms(d_new, density, ao))

        if e_last is None:
            de_val = 0.0
        else:
            de_val = e_tot_val - e_last

        if verbose:
            print(
                f"SCF {it:3d} E = {e_tot_val:.12f} dE = {de_val:.3e} dRMS = {rms_val:.3e}"
            )

        if e_last is not None and abs(de_val) < e_conv and rms_val < d_conv:
            scf_time = time.perf_counter() - t0
            if verbose:
                print(f"SCF time {scf_time:.3f}s")
            return {
                "converged": True,
                "energy": e_tot_val,
                "density": d_new,
                "fock": fock,
                "coeffs": coeffs,
                "orb_energies": eps,
                "iterations": it,
                "scf_time": scf_time,
            }

        density = d_new
        e_last = e_tot_val

    scf_time = time.perf_counter() - t0
    if verbose:
        print(f"SCF time {scf_time:.3f}s")
    return {
        "converged": False,
        "energy": e_last,
        "density": density,
        "fock": fock,
        "coeffs": coeffs,
        "orb_energies": eps,
        "iterations": max_iter,
        "scf_time": scf_time,
    }


if __name__ == "__main__":

    import numpy as np
    np.set_printoptions(precision=2, suppress=False, linewidth=200)
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.patheffects as pe

    from pyquante22.atoms.molecule import Molecule
    from pyquante22.basis.basisset import BasisSet
    from pyquante22.basis.integrals import IntegralEngine
    import pyquante22.array_ops as arr

    def nsymm(n):
        return n * (n + 1) // 2

    def nsymm8(n):
        return nsymm(nsymm(n))

    def five_number_summary(data):
        """Compute min, 25th percentile, median, 75th percentile, max"""
        q75, q50, q25 = np.percentile(data, [75, 50, 25])
        return np.stack([np.amin(data), q25, q50, q75, np.amax(data)])

    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )

    xyzs = "H 0 0 0\nF 1.4 0 0"
    m1 = Molecule(xyzs, ao=aoi)
    # b1 = BasisSet(m1, name='sto-3g')
    # m1 = Molecule("./samples/glycine.sdf", ao=aoi)
    print(f"Using molecule: {str(m1)}")
    b1 = BasisSet(m1, name='sto-3g')
    # b1 = BasisSet(m1, name='aug-cc-pvqz')
    print(
        f"with basis: {str(b1)} full J is {b1.nbf**4}, uniqs is {nsymm8(b1.nbf)}"
    )

    t0 = time.perf_counter()
    ie1 = IntegralEngine(b1)
    t1 = time.perf_counter()
    print(f"Time to do init integral engine: {t1-t0:.2f}s")

    if False:
        t0 = time.perf_counter()
        ie1.build_one_electron_matrices()
        t1 = time.perf_counter()
        print(f"Time to do build 1e mats: {t1-t0:.2f}s")

        t0 = time.perf_counter()
        ie1.build_eri_sparse()
        t1 = time.perf_counter()
        print(f"Time to do build sparse eri: {t1-t0:.2f}s")

        scf_result = scf_rhf(ie1, batch_size=3)
        print(
            f"SCF converged {scf_result['converged']} "
            f"in {scf_result['iterations']} iterations, "
            f"E = {scf_result['energy']:.12f}"
        )

        debug_var = 1

    def isprime(n):
        if n <= 2:
            return True
        lim = int(n**0.5) + 1
        for i in range(2, lim):
            if n % i == 0:
                return False
        return True

    primes = [n for n in range(2, 100) if isprime(n)]
    def prime_facs(n):
        n = int(round(float(n)))
        if n == 1:
            return [1]
        facs = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                facs.append(p)
                n //= p
        if n > 1:
            facs.append(n)
        return facs


    screen = ie1._eri_screen_setup()
    kds_list = [kds for kds, _ in ie1._iter_screened_k_pairs((0,0), screen, only_triu=False)]
    kds_iter = list(ie1._iter_eri_kds_batches(kds_list, batch_size=2))

    ao_mat = aoi.ones((b1.nbf, b1.nbf))
    sh_mat = aoi.ones((b1.nsh, b1.nsh))

    for ibatch, kds_batch in enumerate(kds_iter):
        kds, kids, lids = kds_batch
        for k,l in kds:
            sh_mat[k, l] *= primes[ibatch]
        ao_mat[kids, lids] *= primes[ibatch]
    # sh_mat[0,0] *= 5 # debug

    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    sh_np = _to_numpy(sh_mat)
    ao_np = _to_numpy(ao_mat)
    uniq_vals = np.unique(np.concatenate([sh_np.ravel(), ao_np.ravel()]))
    if uniq_vals.size == 1:
        boundaries = np.array([uniq_vals[0] - 0.5, uniq_vals[0] + 0.5])
    else:
        mids = (uniq_vals[:-1] + uniq_vals[1:]) / 2.0
        first = uniq_vals[0] - (mids[0] - uniq_vals[0])
        last = uniq_vals[-1] + (uniq_vals[-1] - mids[-1])
        boundaries = np.concatenate([[first], mids, [last]])

    cmap = plt.get_cmap("gist_ncar", uniq_vals.size)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes[0].matshow(sh_np, cmap=cmap, norm=norm, interpolation="none")
    axes[0].set_title("sh_mat")
    im1 = axes[1].matshow(ao_np, cmap=cmap, norm=norm, interpolation="none")
    axes[1].set_title("ao_mat")
    cbar = fig.colorbar(im1, ax=axes, ticks=uniq_vals)
    cbar.ax.set_yticklabels([str(v) for v in uniq_vals])

    def _annotate_mat(ax, mat):
        nrows, ncols = mat.shape
        max_dim = max(nrows, ncols)
        if max_dim <= 10:
            fontsize = 8
        elif max_dim <= 20:
            fontsize = 6
        else:
            fontsize = 4
        for i in range(nrows):
            for j in range(ncols):
                val, label = mat[i,j], ""
                if val > 1:
                    label = ",".join(str(primes.index(f)) for f in prime_facs(val))

                if ',' in label:
                    ax.text(j, i, label, ha="center", va="center", color="red", 
                        fontsize=round(1.4*fontsize),
                        path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal(),],
                    )
                else:
                    ax.text(j, i, label, ha="center", va="center", fontsize=fontsize, color="black",)

    _annotate_mat(axes[0], sh_np)
    _annotate_mat(axes[1], ao_np)
    plt.show()
    





    debug_var = 1

    
