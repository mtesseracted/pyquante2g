"""
Compare SCF results with PySCF to find energy discrepancy
"""
import numpy as np
from pyquante22.atoms.molecule import Molecule
from pyquante22.basis.basisset import BasisSet
import pyquante22.array_ops as arr

# Try PySCF comparison
try:
    from pyscf import gto, scf
    has_pyscf = True
except ImportError:
    has_pyscf = False

# Setup for H2 molecule
aoi = arr.ArrayOps("torch", device="cpu", default_dtype="float64")
xyzs = "H 0 0 0\nH 1.4 0 0"  # bohr
m1 = Molecule(xyzs, ao=aoi)
b1 = BasisSet(m1, name='sto-3g')

from pyquante22.basis.hf_driver import (
    build_one_electron_matrices,
    build_eri_tensor,
    scf_rhf,
)

# Build matrices
s_mat, t_mat, v_mat = build_one_electron_matrices(b1)
hcore = t_mat + v_mat
eri = build_eri_tensor(b1)

# Run your SCF
print("=" * 60)
print("YOUR SCF RESULTS:")
print("=" * 60)
scf_result = scf_rhf(
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
your_energy = scf_result['energy']
your_density = scf_result['density']
print(f"\nFinal energy: {your_energy:.12f}")

if has_pyscf:
    print("\n" + "=" * 60)
    print("PYSCF RESULTS:")
    print("=" * 60)
    
    # Create PySCF molecule
    mol = gto.Mole()
    mol.atom = [['H', (0, 0, 0)], ['H', (1.4, 0, 0)]]
    mol.basis = 'sto-3g'
    mol.unit = 'bohr'
    mol.build()
    
    # Run PySCF
    mf = scf.RHF(mol)
    pyscf_energy = mf.kernel()
    
    print(f"\nPySCF energy: {pyscf_energy:.12f}")
    print(f"\nEnergy difference: {your_energy - pyscf_energy:.12f}")
    
    # Compare density matrices
    print("\n" + "=" * 60)
    print("DENSITY MATRIX COMPARISON:")
    print("=" * 60)
    your_d = your_density if hasattr(your_density, 'numpy') else your_density
    if hasattr(your_d, 'numpy'):
        your_d = your_d.numpy()
    pyscf_d = mf.make_rdm1()
    
    print("\nYour density:\n", your_d)
    print("\nPySCF density:\n", pyscf_d)
    print("\nDensity difference:\n", your_d - pyscf_d)
    print(f"Max density difference: {np.max(np.abs(your_d - pyscf_d)):.2e}")
    
    # Compare orbital energies
    print("\n" + "=" * 60)
    print("ORBITAL ENERGIES COMPARISON:")
    print("=" * 60)
    print(f"Your orbital energies: {scf_result['orb_energies']}")
    print(f"PySCF orbital energies: {mf.mo_energy}")
    
else:
    print("\nPySCF not available. Install it to compare: pip install pyscf")
