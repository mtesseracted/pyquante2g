import gc
import time
import numpy as np
import pyquante2g.torch_ops as arr
from pyquante2g.atoms.molecule import load_sample_molecule
from pyquante2g.basis.basisset import BasisSet
from pyquante2g.basis.integral_engines import (
    IntegralEngineSparse,
    guess_density_from_core,
    orthogonalizer,
)


def estimate_energy_from_initial_guess(ie, mol, hcore, s_mat, batch_size=int(2**16)):
    ao = ie.ao
    density = guess_density_from_core(hcore, s_mat, mol.nclosed(), ao)
    j_mat, k_mat = ie.build_jk_batch(density, batch_size=batch_size)
    fock = hcore + j_mat - 0.5 * k_mat
    energy = 0.5 * ao.einsum("ij,ij->", density, hcore + fock)
    energy = energy + mol.nuclear_repulsion()
    return energy, density, fock


def density_from_fock(fock, s_inv_sqrt, nocc, ao):
    f_prime = s_inv_sqrt @ fock @ s_inv_sqrt
    _, c_prime = ao.eigh(f_prime)
    coeffs = s_inv_sqrt @ c_prime
    c_occ = coeffs[:, :nocc]
    return 2.0 * (c_occ @ ao.transpose(c_occ))


def solve_closed_shell_hf(
    ie,
    mol,
    hcore,
    s_mat,
    batch_size=int(2**16),
    max_iter=50,
    e_tol=1e-10,
    d_tol=1e-10,
):
    ao = ie.ao
    torch = ao.xp
    use_cuda = str(ao.device).startswith("cuda")
    nocc = mol.nclosed()
    energy_prev = None
    converged = False

    with torch.no_grad():
        density = guess_density_from_core(hcore, s_mat, nocc, ao)
        s_inv_sqrt = orthogonalizer(s_mat, ao)

        for iteration in range(1, max_iter + 1):
            j_mat, k_mat = ie.build_jk_batch(density, batch_size=batch_size)
            fock = hcore + j_mat - 0.5 * k_mat
            energy = 0.5 * ao.einsum("ij,ij->", density, hcore + fock)
            energy = energy + mol.nuclear_repulsion()
            new_density = density_from_fock(fock, s_inv_sqrt, nocc, ao)

            de = None if energy_prev is None else abs(energy.item() - energy_prev)
            dd = ao.norm(new_density - density).item()
            density = new_density.detach()
            energy_prev = energy.item()

            del j_mat, k_mat, fock, energy, new_density
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

            if de is not None and de < e_tol and dd < d_tol:
                converged = True
                break

    if not converged:
        raise RuntimeError("HF did not converge")

    density0 = density.detach()
    s_inv_sqrt = orthogonalizer(s_mat, ao)
    j_mat, k_mat = ie.build_jk_batch(density0, batch_size=batch_size)
    fock = hcore + j_mat - 0.5 * k_mat
    density = density_from_fock(fock, s_inv_sqrt, nocc, ao)
    del j_mat, k_mat, fock

    j_mat, k_mat = ie.build_jk_batch(density, batch_size=batch_size)
    fock = hcore + j_mat - 0.5 * k_mat
    energy = 0.5 * ao.einsum("ij,ij->", density, hcore + fock)
    energy = energy + mol.nuclear_repulsion()
    return energy, density, fock, iteration


def main():
    tstart = time.time()
    sample_name = "Water"
    basis_name = "6-31gs"
    stretch_fac = 1.0
    batch_size = int(2**15)
    ao = arr.TorchOps(
        "torch", device="cuda:0", default_dtype="float64", int_dtype="int64"
    )
    torch = ao.xp
    use_cuda = str(ao.device).startswith("cuda")

    def cleanup_cuda():
        gc.collect()
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def build_case():
        mol = load_sample_molecule(sample_name, ao)
        for atom in mol.atoms:
            atom.r = atom.r * stretch_fac
            atom.r.requires_grad_()
        basis = BasisSet(mol, name=basis_name)
        ie = IntegralEngineSparse(basis)
        s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
        hcore = t_mat + v_mat
        return mol, ie, s_mat, hcore

    def get_forces(term, mol, retain_graph):
        grads = torch.autograd.grad(
            term,
            [atom.r for atom in mol.atoms],
            retain_graph=retain_graph,
        )
        return [np.asarray((-grad.detach()).cpu(), dtype=float) for grad in grads]

    def print_breakdown(label, mol, energy, density, fock, hcore, s_mat, niter=None):
        g_mat = fock - hcore
        e_nuc = mol.nuclear_repulsion()
        e_hcore = 0.5 * ao.einsum("ij,ij->", density, hcore)
        e_fock = 0.5 * ao.einsum("ij,ij->", density, fock)
        e_g = 0.5 * ao.einsum("ij,ij->", density, g_mat)
        e_total = e_nuc + e_hcore + e_fock

        forces_nuc = get_forces(e_nuc, mol, retain_graph=True)
        forces_hcore = get_forces(e_hcore, mol, retain_graph=True)
        forces_fock = get_forces(e_fock, mol, retain_graph=True)
        forces_g = get_forces(e_g, mol, retain_graph=True)
        forces_total = get_forces(e_total, mol, retain_graph=False)

        if niter is None:
            print(label)
        else:
            print(f"{label} ({niter} iterations)")
        print(f"E = {energy.detach().item():.12f}")
        print(f"E_nuc = {e_nuc.detach().item():.12f}")
        print(f"E_hcore = {e_hcore.detach().item():.12f}")
        print(f"E_fock = {e_fock.detach().item():.12f}")
        print(f"E_G = {e_g.detach().item():.12f}")
        print(f"trace(D S) = {ao.einsum('ij,ij->', density, s_mat).detach().item():.12f}")
        print(f"||F|| = {ao.norm(fock).detach().item():.12f}")

        for ii, atom in enumerate(mol.atoms):
            pos = atom.r.detach()
            print(
                f"atom {ii:2d} {atom.symbol():>2s} "
                f"({float(pos[0]):.4g}, {float(pos[1]):.4g}, {float(pos[2]):.4g})"
            )
            print(
                f"  F_total = [{forces_total[ii][0]: .8f}, {forces_total[ii][1]: .8f}, {forces_total[ii][2]: .8f}]"
            )
            print(
                f"  F_nuc   = [{forces_nuc[ii][0]: .8f}, {forces_nuc[ii][1]: .8f}, {forces_nuc[ii][2]: .8f}]"
            )
            print(
                f"  F_hcore = [{forces_hcore[ii][0]: .8f}, {forces_hcore[ii][1]: .8f}, {forces_hcore[ii][2]: .8f}]"
            )
            print(
                f"  F_fock  = [{forces_fock[ii][0]: .8f}, {forces_fock[ii][1]: .8f}, {forces_fock[ii][2]: .8f}]"
            )
            print(
                f"  F_G     = [{forces_g[ii][0]: .8f}, {forces_g[ii][1]: .8f}, {forces_g[ii][2]: .8f}]"
            )

    print(f"Running for stretch factor: {stretch_fac}")
    print(f"sample={sample_name} basis={basis_name}")

    mol, ie, s_mat, hcore = build_case()
    energy, density, fock = estimate_energy_from_initial_guess(
        ie, mol, hcore, s_mat, batch_size=batch_size
    )
    print_breakdown("Initial guess", mol, energy, density, fock, hcore, s_mat)
    del mol, ie, s_mat, hcore, energy, density, fock
    cleanup_cuda()

    mol, ie, s_mat, hcore = build_case()
    energy, density, fock, niter = solve_closed_shell_hf(
        ie, mol, hcore, s_mat, batch_size=batch_size
    )
    print_breakdown("Converged HF", mol, energy, density, fock, hcore, s_mat, niter=niter)
    del mol, ie, s_mat, hcore, energy, density, fock, niter
    cleanup_cuda()

    tend = time.time()
    print(f"Total time: {tend - tstart:.2f} seconds")


if __name__ == "__main__":
    main()
