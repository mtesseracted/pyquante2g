import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import pyquante2g.torch_ops as arr
from pyquante2g.atoms.molecule import load_sample_molecule
from pyquante2g.basis.basisset import BasisSet
from pyquante2g.basis.integral_engines import (
    IntegralEngineSparse,
    IntegralEngineBase,
    guess_density_from_core,
    orthogonalizer,
)


def estimate_energy_from_initial_guess(ie, mol, hcore, s_mat, batch_size=int(2**16)):
    ao = ie.ao
    density = guess_density_from_core(hcore, s_mat, mol.nclosed(), ao)
    j_mat, k_mat = ie.build_jk_batch(density, batch_size=batch_size)
    fock = hcore + j_mat - 0.5 * k_mat
    e_elec = 0.5 * ao.einsum("ij,ij->", density, hcore + fock)
    e_tot = e_elec + mol.nuclear_repulsion()
    return e_tot, density, fock


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
    basis_name = "sto-3g"
    # basis_name = "6-31gs"
    # basis_name = "aug-cc-pVDZ"
    batch_size = int(2**15)
    stretch_facs = np.arange(0.5, 1.5, 0.1)
    plot_path = f"examples/autodiff_coords_scan__{sample_name}__{basis_name}.png"
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

    def build_case(stretch_fac):
        mol = load_sample_molecule(sample_name, ao)
        for atom in mol.atoms:
            atom.r = atom.r * stretch_fac
            atom.r.requires_grad_()
        basis = BasisSet(mol, name=basis_name)
        # ie = IntegralEngineSparse(basis)
        ie = IntegralEngineBase(basis)
        s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
        hcore = t_mat + v_mat
        return mol, ie, s_mat, hcore

    def print_result(label, energy, density, fock, mol, s_mat, niter=None):
        if niter is None:
            print(label)
        else:
            print(f"{label} ({niter} iterations)")
        print(f"E = {energy.detach().item():.12f}")
        print(f"trace(D S) = {ao.einsum('ij,ij->', density, s_mat).detach().item():.12f}")
        print(f"||F|| = {ao.norm(fock).detach().item():.12f}")
        forces = []
        for ii, atom in enumerate(mol.atoms):
            pos = atom.r.detach()
            force = -atom.r.grad.detach()
            force_cpu = np.asarray(force.cpu(), dtype=float)
            forces.append(force_cpu)
            print(
                f"atom {ii:2d} {atom.symbol():>2s} "
                f"({float(pos[0]):.4g}, {float(pos[1]):.4g}, {float(pos[2]):.4g}) F=-dE/dR = "
                f"[{float(force_cpu[0]): .8f}, {float(force_cpu[1]): .8f}, {float(force_cpu[2]): .8f}]"
            )
        return forces

    guess_energies = []
    hf_energies = []
    guess_forces = []
    hf_forces = []

    for stretch_fac in stretch_facs:
        t0 = time.time()
        print(f"Running for stretch factor: {stretch_fac}")
        print(f"sample={sample_name} basis={basis_name}")

        mol, ie, s_mat, hcore = build_case(stretch_fac)
        energy, density, fock = estimate_energy_from_initial_guess(
            ie, mol, hcore, s_mat, batch_size=batch_size
        )
        energy.backward()
        guess_energies.append(energy.detach().item())
        guess_forces.append(print_result("Initial guess", energy, density, fock, mol, s_mat))
        del mol, ie, s_mat, hcore, energy, density, fock
        cleanup_cuda()

        mol, ie, s_mat, hcore = build_case(stretch_fac)
        energy, density, fock, niter = solve_closed_shell_hf(
            ie, mol, hcore, s_mat, batch_size=batch_size
        )
        energy.backward()
        hf_energies.append(energy.detach().item())
        hf_forces.append(print_result("Converged HF", energy, density, fock, mol, s_mat, niter=niter))
        del mol, ie, s_mat, hcore, energy, density, fock, niter
        cleanup_cuda()
        t1 = time.time()
        print(f" -- Time for stretch factor {stretch_fac:.2f}: {t1 - t0:.2f} seconds")
        print('-' * 80)

    tend = time.time()
    print('=' * 80)
    print(f"Total time: {tend - tstart:.2f} seconds")
    print('=' * 80)

    stretch_facs = np.asarray(stretch_facs, dtype=float)
    guess_energies = np.asarray(guess_energies, dtype=float)
    hf_energies = np.asarray(hf_energies, dtype=float)
    guess_fz0 = np.asarray([forces[0][2] for forces in guess_forces], dtype=float)
    hf_fz0 = np.asarray([forces[0][2] for forces in hf_forces], dtype=float)
    guess_fz1 = np.asarray([forces[1][2] for forces in guess_forces], dtype=float)
    hf_fz1 = np.asarray([forces[1][2] for forces in hf_forces], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(stretch_facs, guess_energies, "o-", label="Initial guess")
    axes[0].plot(stretch_facs, hf_energies, "s-", label="Converged HF")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].plot(stretch_facs, guess_fz0, "o-", label="Initial guess")
    axes[1].plot(stretch_facs, hf_fz0, "s-", label="Converged HF")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Atom 0 Fz")
    axes[1].legend()

    axes[2].plot(stretch_facs, guess_fz1, "o-", label="Initial guess")
    axes[2].plot(stretch_facs, hf_fz1, "s-", label="Converged HF")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Atom 1 Fz")
    axes[2].set_xlabel("stretch_fac")
    axes[2].legend()

    fig.suptitle(f"{sample_name} / {basis_name}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.show()
    plt.close(fig)
    debug_var = 1


if __name__ == "__main__":
    main()
