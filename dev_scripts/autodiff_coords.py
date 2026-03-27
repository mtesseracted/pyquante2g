import pyquante2g.torch_ops as arr
from pyquante2g.atoms.molecule import load_sample_molecule
from pyquante2g.basis.basisset import BasisSet
from pyquante2g.basis.integral_engines import (
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
    nocc = mol.nclosed()
    density = guess_density_from_core(hcore, s_mat, nocc, ao)
    s_inv_sqrt = orthogonalizer(s_mat, ao)
    energy_prev = None

    for iteration in range(1, max_iter + 1):
        j_mat, k_mat = ie.build_jk_batch(density, batch_size=batch_size)
        fock = hcore + j_mat - 0.5 * k_mat
        energy = 0.5 * ao.einsum("ij,ij->", density, hcore + fock)
        energy = energy + mol.nuclear_repulsion()
        new_density = density_from_fock(fock, s_inv_sqrt, nocc, ao)

        if energy_prev is not None:
            de = abs(float((energy - energy_prev).detach()))
            dd = float(ao.norm(new_density - density).detach())
            if de < e_tol and dd < d_tol:
                density = new_density
                break

        density = new_density
        energy_prev = energy
    else:
        raise RuntimeError("HF did not converge")

    j_mat, k_mat = ie.build_jk_batch(density, batch_size=batch_size)
    fock = hcore + j_mat - 0.5 * k_mat
    energy = 0.5 * ao.einsum("ij,ij->", density, hcore + fock)
    energy = energy + mol.nuclear_repulsion()
    return energy, density, fock, iteration


def build_case(sample_name, basis_name, stretch_fac, ao):
    mol = load_sample_molecule(sample_name, ao)
    for atom in mol.atoms:
        atom.r = atom.r * stretch_fac
        atom.r.requires_grad_()
    basis = BasisSet(mol, name=basis_name)
    ie = IntegralEngineBase(basis)
    s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
    hcore = t_mat + v_mat
    return mol, ie, s_mat, hcore


def print_result(label, energy, density, fock, mol, s_mat, ao, niter=None):
    if niter is None:
        print(label)
    else:
        print(f"{label} ({niter} iterations)")
    print(f"E = {float(energy.detach()):.12f}")
    print(f"trace(D S) = {float(ao.einsum('ij,ij->', density, s_mat).detach()):.12f}")
    print(f"||F|| = {float(ao.norm(fock).detach()):.12f}")
    for ii, atom in enumerate(mol.atoms):
        pos = atom.r.detach()
        grad = atom.r.grad.detach()
        print(
            f"atom {ii:2d} {atom.symbol():>2s} "
            f"({float(pos[0]):.4g}, {float(pos[1]):.4g}, {float(pos[2]):.4g}) dE/dR = "
            f"[{float(grad[0]): .8f}, {float(grad[1]): .8f}, {float(grad[2]): .8f}]"
        )


def main():
    sample_name = "Hydrogen"
    basis_name = "sto-3g"
    batch_size = int(2**16)
    ao = arr.TorchOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )

    for stretch_fac in [0.75, 1.0, 1.5]:
        print(f"Running for stretch factor: {stretch_fac}")
        print(f"sample={sample_name} basis={basis_name}")

        mol, ie, s_mat, hcore = build_case(sample_name, basis_name, stretch_fac, ao)
        energy, density, fock = estimate_energy_from_initial_guess(
            ie, mol, hcore, s_mat, batch_size=batch_size
        )
        energy.backward()
        print_result("Initial guess", energy, density, fock, mol, s_mat, ao)

        mol, ie, s_mat, hcore = build_case(sample_name, basis_name, stretch_fac, ao)
        energy, density, fock, niter = solve_closed_shell_hf(
            ie, mol, hcore, s_mat, batch_size=batch_size
        )
        energy.backward()
        print_result("Converged HF", energy, density, fock, mol, s_mat, ao, niter=niter)


if __name__ == "__main__":
    main()
