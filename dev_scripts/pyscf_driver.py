"""
Run PySCF RHF on a molecule loaded from samples.py.
"""
from importlib import resources
import importlib.util
import os

import numpy as np
import pyquante22.array_ops as arr
from pyquante22.atoms.molecule import Molecule, load_sample_molecule
from pyquante22.basis.cart_gto_utils import sym2am
from pyquante22.utils.file_utils import find_closest_file_match, load_gbs_file

try:
    from pyscf import gto, scf
    has_pyscf = True
except ImportError:
    has_pyscf = False


SAMPLE_NAME = "Propane"  # substring match against samples.py names
BASIS_NAME = "sto-3g"


def build_pyscf_mol(mol, basis_name):
    atoms = [
        (atom.symbol(), tuple(float(x) for x in mol.ao.tolist(atom.r)))
        for atom in mol.atoms
    ]
    basis_dir = resources.files("pyquante22.basis").joinpath("cart_gto_data")
    basis_file = find_closest_file_match(basis_dir, f"{basis_name}.gbs")
    if basis_file is None:
        raise ValueError(f"Cannot find {basis_name}.gbs in {basis_dir}")
    raw_basis = load_gbs_file(basis_file)
    elem_syms = sorted({atom.symbol() for atom in mol.atoms})
    basis_data = {}
    for sym in elem_syms:
        if sym not in raw_basis:
            raise ValueError(f"No basis data for element {sym} in {basis_file}")
        shells = []
        for am_sym, prims in raw_basis[sym]:
            lval = sym2am[am_sym.upper()]
            shell = [lval]
            for expn, coeff in prims:
                shell.append([float(expn), float(coeff)])
            shells.append(shell)
        basis_data[sym] = shells
    pmol = gto.Mole()
    pmol.atom = atoms
    pmol.unit = "Bohr"
    pmol.basis = basis_data
    pmol.charge = mol.charge
    pmol.spin = mol.multiplicity - 1
    pmol.cart = True
    pmol.build()
    return pmol


def main():
    if not has_pyscf:
        print("PySCF not available. Install it to run this: pip install pyscf")
        return

    outdir = "dev_scripts/tmp"
    os.makedirs(outdir, exist_ok=True)
    ao = arr.ArrayOps("numpy", default_dtype="float64", int_dtype="int64")

    sample_molecules = ["Hydrogen", "Water", "Methane", "Propane"]
    sample_bases = ["sto-3g", "aug-cc-pvdz"]
    sample_list = [(sample_bases[0], mol) for mol in sample_molecules]
    sample_list += [(sample_bases[1], mol) for mol in sample_molecules[:2]]

    for basis_name, sample_name in sample_list:
        print(f"\n=== Sample: {sample_name}, Basis: {basis_name} ===")
        mol = load_sample_molecule(sample_name, ao=ao)
        print(f"Using sample molecule: {mol}")

        pmol = build_pyscf_mol(mol, basis_name)
        if pmol.spin != 0:
            print(
                f"Warning: spin={pmol.spin} (multiplicity={mol.multiplicity}). "
                "RHF is closed-shell; consider ROHF/UHF for open-shell."
            )

        mf = scf.RHF(pmol)
        energy = mf.kernel()
        print(f"RHF/{basis_name} energy: {energy:.12f}")
        sample_tag = sample_name.replace(" ", "_")
        sample_tag += f"_{basis_name.replace('-', '')}"

        S = pmol.intor("int1e_ovlp_cart")
        T = pmol.intor("int1e_kin_cart")
        V = pmol.intor("int1e_nuc_cart")
        H = T + V
        eri_cart = pmol.intor("int2e_cart")

        ao_label_path = os.path.join(outdir, f"ao_labels_{sample_tag}.txt")
        with open(ao_label_path, 'w') as ofd:
            for ao_label in pmol.ao_labels():
                ofd.write(ao_label+'\n')
        np.save(os.path.join(outdir, f"S_{sample_tag}.npy"), S)
        np.save(os.path.join(outdir, f"H_{sample_tag}.npy"), H)
        np.save(os.path.join(outdir, f"ERI_{sample_tag}.npy"), eri_cart)


if __name__ == "__main__":
    main()
