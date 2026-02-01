import os
import time
import itertools
import pyquante22.array_ops as arr


def nsymm(n):
    return n * (n + 1) // 2


def nsymm8(n):
    return nsymm(nsymm(n))


def _shell_coeffs(shell, gto, ao):
    if gto.coeffs is None:
        raise ValueError("Shell is missing contraction coefficients")
    return ao.reshape(gto.coeffs, (shell.nbf, shell.npf))


def _contract_shell_pair(prim_vals, shell_i, shell_j, coeffs_i, coeffs_j, ao):
    prim = ao.reshape(
        prim_vals,
        (shell_i.nbf, shell_i.npf, shell_j.nbf, shell_j.npf),
    )
    return ao.einsum("ip,jq,ipjq->ij", coeffs_i, coeffs_j, prim)


def _fill_swap_symm(array, ids, vals, ao):
    """fill (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)"""
    array[ids] = vals
    id2 = (ids[1], ids[0], ids[2], ids[3])
    array[id2] = ao.transpose(vals, (1, 0, 2, 3))
    id2 = (ids[0], ids[1], ids[3], ids[2])
    array[id2] = ao.transpose(vals, (0, 1, 3, 2))
    id2 = (ids[1], ids[0], ids[3], ids[2])
    array[id2] = ao.transpose(vals, (1, 0, 3, 2))


def _fill_eri_symm(array, ids, vals, ao):
    """fill (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk), repeat for (kl|ij)"""
    _fill_swap_symm(array, ids, vals, ao)
    id2 = (ids[2], ids[3], ids[0], ids[1])
    val2 = ao.transpose(vals, (2, 3, 0, 1))
    _fill_swap_symm(array, id2, val2, ao)


def build_one_electron_matrices(basis):
    """Build overlap, kinetic, and nuclear attraction matrices for a BasisSet."""
    ao = basis.molecule.ao
    nbf = basis.nbf
    s_mat = ao.zeros((nbf, nbf))
    t_mat = ao.zeros((nbf, nbf))
    v_mat = ao.zeros((nbf, nbf))

    shell_bfs = [shell.bfs for shell in basis.shells]
    shell_coeffs = [
        _shell_coeffs(shell, gto, ao) for shell, gto in zip(basis.shells, shell_bfs)
    ]

    atoms = basis.molecule.atoms
    if atoms:
        rcen = ao.stack([atom.r for atom in atoms])
        charges = ao.arr([atom.Z for atom in atoms])
    else:
        rcen = None
        charges = None

    for i, shell_i in enumerate(basis.shells):
        gi = shell_bfs[i]
        ci = shell_coeffs[i]
        ri = basis.shell_range[i]

        for j in range(i + 1):
            shell_j = basis.shells[j]
            gj = shell_bfs[j]
            cj = shell_coeffs[j]
            rj = basis.shell_range[j]

            gpair = gi * gj

            ov_prim = gpair.calc_overlap()
            kin_prim = gpair.calc_kinetic()

            ov_block = _contract_shell_pair(ov_prim, shell_i, shell_j, ci, cj, ao)
            kin_block = _contract_shell_pair(kin_prim, shell_i, shell_j, ci, cj, ao)

            if atoms:
                nai_prim = gpair.calc_nai(rcen)
                nai_weighted = ao.einsum("ijc,c->ij", nai_prim, charges)
                nai_block = _contract_shell_pair(
                    nai_weighted, shell_i, shell_j, ci, cj, ao
                )
            else:
                nai_block = ao.zeros((shell_i.nbf, shell_j.nbf))

            s_mat[ri, rj] = ov_block
            t_mat[ri, rj] = kin_block
            v_mat[ri, rj] = nai_block

            if i != j:
                s_mat[rj, ri] = ao.transpose(ov_block, (1, 0))
                t_mat[rj, ri] = ao.transpose(kin_block, (1, 0))
                v_mat[rj, ri] = ao.transpose(nai_block, (1, 0))

    return s_mat, t_mat, v_mat


def build_eri_tensor(basis):
    """Build the full 4D electron repulsion integral tensor."""
    ao = basis.molecule.ao
    nbf = basis.nbf
    eri = ao.zeros((nbf, nbf, nbf, nbf))

    shells = basis.shells
    ranges = basis.shell_range
    nsh = len(shells)

    pair_list = []
    for i in range(nsh):
        for j in range(i + 1):
            pair_list.append((i, j))

    for p, (i, j) in enumerate(pair_list):
        sp_ij = shells[i] * shells[j]
        for q in range(p + 1):
            k, l = pair_list[q]
            sp_kl = shells[k] * shells[l]
            block = sp_ij | sp_kl  # (ni,nj,nk,nl)
            ids = (ranges[i], ranges[j], ranges[k], ranges[l])
            _fill_eri_symm(eri, ids, block, ao)

    return eri


def _to_scalar(x, ao):
    if hasattr(x, "item"):
        return x.item()
    return float(ao.tolist(x))


def _density_rms(d_new, d_old, ao):
    diff = d_new - d_old
    denom = diff.reshape(-1).shape[0]
    return ao.sqrt(ao.sum(diff * diff) / denom)


def orthogonalizer(s_mat, ao):
    xp = ao.xp
    evals, evecs = xp.linalg.eigh(s_mat)
    neg_count = (evals <= 0).sum()
    if hasattr(neg_count, "item"):
        neg_count = neg_count.item()
    if neg_count:
        raise ValueError("Overlap matrix has non-positive eigenvalues")
    return (evecs / xp.sqrt(evals)) @ ao.transpose(evecs)


def guess_density_from_core(hcore, s_mat, nocc, ao):
    """Build a closed-shell density matrix from the core Hamiltonian."""
    xp = ao.xp
    s_inv_sqrt = orthogonalizer(s_mat, ao)

    f_prime = s_inv_sqrt @ hcore @ s_inv_sqrt
    _, c_prime = xp.linalg.eigh(f_prime)
    coeffs = s_inv_sqrt @ c_prime

    c_occ = coeffs[:, :nocc]
    density = 2.0 * (c_occ @ ao.transpose(c_occ))
    return density


def build_fock_matrix(hcore, eri, density, ao=None):
    """Build the Fock matrix: F = H + 2J - K."""
    ao = ao or arr.get_default_ops()
    j_mat = ao.einsum("kl,ijkl->ij", density, eri)
    k_mat = ao.einsum("kl,ikjl->ij", density, eri)
    # return hcore + (2.0 * j_mat) - k_mat
    return hcore + j_mat - 0.5 * k_mat


def solve_fock(fock, s_inv_sqrt, nocc, ao):
    xp = ao.xp
    f_prime = s_inv_sqrt @ fock @ s_inv_sqrt
    eps, c_prime = xp.linalg.eigh(f_prime)
    coeffs = s_inv_sqrt @ c_prime
    c_occ = coeffs[:, :nocc]
    density = 2.0 * (c_occ @ ao.transpose(c_occ))
    return density, eps, coeffs


def scf_rhf(
    basis,
    s_mat=None,
    t_mat=None,
    v_mat=None,
    eri=None,
    max_iter=50,
    e_conv=1e-8,
    d_conv=1e-6,
    verbose=True,
):
    ao = basis.molecule.ao
    t0 = time.perf_counter()
    if s_mat is None or t_mat is None or v_mat is None:
        s_mat, t_mat, v_mat = build_one_electron_matrices(basis)
    hcore = t_mat + v_mat
    if eri is None:
        eri = build_eri_tensor(basis)

    nocc = basis.molecule.nclosed()
    s_inv_sqrt = orthogonalizer(s_mat, ao)
    density = guess_density_from_core(hcore, s_mat, nocc, ao)
    enuc = basis.molecule.nuclear_repulsion()

    e_last = None
    fock = None
    eps = None
    coeffs = None

    for it in range(1, max_iter + 1):
        fock = build_fock_matrix(hcore, eri, density, ao=ao)
        d_new, eps, coeffs = solve_fock(fock, s_inv_sqrt, nocc, ao)

        e_elec = 0.5 * ao.sum(d_new * (hcore + fock))
        e_tot = e_elec + enuc
        e_tot_val = _to_scalar(e_tot, ao)
        rms_val = _to_scalar(_density_rms(d_new, density, ao), ao)

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


def build_fock_jk(hcore, eri, density, ao=None):
    """
    Build RHF Fock from a *spin-summed* closed-shell density D = 2*Cocc*Cocc^T.

    Conventions (chemist's ERI): eri[p,q,r,s] = (pq|rs)

      J_pq = sum_rs D_rs (pq|rs)
      K_pq = sum_rs D_rs (pr|qs)

    For spin-summed D, the RHF Fock is:
      F = H + J - 1/2 K

    Returns: (F, J, K, G) where G = J - 1/2 K
    """
    ao = ao or arr.get_default_ops()
    J = ao.einsum("rs,pqrs->pq", density, eri)
    K = ao.einsum("rs,prqs->pq", density, eri)
    G = J - 0.5 * K
    F = hcore + G
    return F, J, K, G


def scf_rhf_v2(
    basis,
    s_mat=None,
    t_mat=None,
    v_mat=None,
    eri=None,
    max_iter=50,
    e_conv=1e-8,
    d_conv=1e-6,
    verbose=True,
):
    ao = basis.molecule.ao
    t0 = time.perf_counter()

    # ---- integrals ----
    if s_mat is None or t_mat is None or v_mat is None:
        s_mat, t_mat, v_mat = build_one_electron_matrices(basis)
    hcore = t_mat + v_mat

    if eri is None:
        eri = build_eri_tensor(basis)

    # ---- setup ----
    nocc = basis.molecule.nclosed()
    s_inv_sqrt = orthogonalizer(s_mat, ao)
    density = guess_density_from_core(hcore, s_mat, nocc, ao)
    enuc = basis.molecule.nuclear_repulsion()

    e_last = None
    fock = None
    eps = None
    coeffs = None

    # ---- SCF ----
    for it in range(1, max_iter + 1):
        # Build Fock from the *current* density
        fock, j_mat, k_mat, g_mat = build_fock_jk(hcore, eri, density, ao=ao)

        # Compute a CONSISTENT energy for this same density (no extra Fock build)
        # E_elec = Tr[D H] + 1/2 Tr[D G], where G = J - 1/2 K for spin-summed D
        e_elec = ao.sum(density * hcore) + 0.5 * ao.sum(density * g_mat)
        e_tot = e_elec + enuc

        # Solve Roothaan equations to get the updated density
        d_new, eps, coeffs = solve_fock(fock, s_inv_sqrt, nocc, ao)

        # Convergence metrics: delta-E between successive "current-density" energies,
        # and RMS density change
        e_tot_val = _to_scalar(e_tot, ao)
        rms_val = _to_scalar(_density_rms(d_new, density, ao), ao)

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
    import time
    import numpy as np
    from pyquante22.atoms.molecule import Molecule
    from pyquante22.utils.file_utils import read_xyz_lines
    from pyquante22.basis.basisset import BasisSet

    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )

    # xyzs = "F 1 0 0\nH -1 0 0"
    xyzs = "H 0 0 0\nF 1.4 0 0" # bohr
    # xyzs = "H 0 0 0\nH 0.735 0 0" # angstrom
    m1 = Molecule(xyzs, ao=aoi)
    # b1 = BasisSet(m1, name='sto-3g')
    # m1 = Molecule("./samples/glycine.sdf", ao=aoi)
    print(f"Using molecule: {str(m1)}")
    b1 = BasisSet(m1, name='sto-3g')
    # b1 = BasisSet(m1, name='aug-cc-pvqz')
    print(f"with basis: {str(b1)} full J is {b1.nbf**4}, uniqs is {nsymm8(b1.nbf)}")

    t_mats = time.perf_counter()
    s_mat, t_mat, v_mat = build_one_electron_matrices(b1)
    print(f"Built one-electron matrices in {time.perf_counter() - t_mats:.3f}s")
    print(f"S shape {tuple(s_mat.shape)} T shape {tuple(t_mat.shape)} V shape {tuple(v_mat.shape)}")


    np.set_printoptions(precision=10, suppress=True, linewidth=200)
    S, T, V = s_mat.numpy(), t_mat.numpy(), v_mat.numpy()
    print("AO overlap S:\n", S, "\n")
    print("AO kinetic T:\n", T, "\n")
    print("AO nuclear attraction V:\n", V, "\n")
    print("AO core Hamiltonian H = T + V:\n", T+V, "\n")


    t_eri = time.perf_counter()
    eri = build_eri_tensor(b1)
    print(f"Built ERI tensor in {time.perf_counter() - t_eri:.3f}s")
    print(f"ERI shape {tuple(eri.shape)}")

    # J = eri.numpy()
    # print(f"AO ERI tensor shape: {J.shape}")
    # print("AO ERI (ij|kl) full tensor:\n", J, "\n")

    hcore = t_mat + v_mat
    nocc = m1.nclosed()
    density = guess_density_from_core(hcore, s_mat, nocc, aoi)
    fock = build_fock_matrix(hcore, eri, density, ao=aoi)
    print(f"Fock shape {tuple(fock.shape)}")

    scf_result = scf_rhf_v2(
        b1,
        s_mat=s_mat,
        t_mat=t_mat,
        v_mat=v_mat,
        eri=eri,
        max_iter=50,
        e_conv=1e-8,
        d_conv=1e-6,
        verbose=True,
    )
    print(
        f"SCF converged {scf_result['converged']} "
        f"in {scf_result['iterations']} iterations, "
        f"E = {scf_result['energy']:.12f}"
    )

    debug_var = 1
