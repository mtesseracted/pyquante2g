from pathlib import Path

import numpy as np
import pytest

import pyquante22.array_ops as arr
from pyquante22.atoms.molecule import load_sample_molecule
from pyquante22.basis.basisset import BasisSet
from pyquante22.basis.cart_gto_utils import cartPow2fullSym, sym2am
from pyquante22.basis.integral_engines import IntegralEngineBase
from pyquante22.utils.file_utils import find_closest_file_match, load_gbs_file

try:
    from pyscf import gto
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


CASES = [
    ("sto-3g", "Hydrogen"),
    ("sto-3g", "Water"),
    ("sto-3g", "Methane"),
    ("sto-3g", "Propane"),
    ("aug-cc-pvdz", "Hydrogen"),
    ("aug-cc-pvdz", "Water"),
]


def _case_id(case, markers=()):
    basis, name = case
    suffix = ""
    if markers:
        suffix = f"({'+'.join(markers)})"
    return f"{basis}_{name}{suffix}"


def _one_e_params():
    params = []
    for case in CASES:
        basis, _ = case
        if basis.startswith("sto"):
            params.append(pytest.param(case, id=_case_id(case)))
        else:
            params.append(
                pytest.param(
                    case,
                    id=_case_id(case, markers=("med",)),
                    marks=pytest.mark.med,
                )
            )
    return params


def _eri_params():
    params = []
    for case in CASES:
        basis, name = case
        marks = [pytest.mark.med]
        marker_labels = ["med"]
        if (not basis.startswith("sto")) and (name != "Hydrogen"):
            marks.append(pytest.mark.slo)
            marker_labels.append("slo")
        params.append(
            pytest.param(case, id=_case_id(case, markers=marker_labels), marks=marks)
        )
    return params


@pytest.fixture(params=_one_e_params())
def one_elec_case(request):
    return request.param


@pytest.fixture(params=_eri_params())
def eri_case(request):
    return request.param


def _build_pyscf_mol(mol, basis_name):
    atoms = [
        (atom.symbol(), tuple(float(x) for x in mol.ao.tolist(atom.r)))
        for atom in mol.atoms
    ]
    basis_dir = Path(__file__).resolve().parents[2] / "pyquante22" / "basis" / "cart_gto_data"
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


def _assert_tensor_matches(name, tensor, ref_tensor, RTOL, ATOL):
    a = _to_numpy(tensor)
    b = ref_tensor
    if np.allclose(a, b, rtol=RTOL, atol=ATOL):
        return

    amask = np.abs(a) > ATOL
    bmask = np.abs(b) > ATOL
    azero_diffs = np.where((amask & ~bmask))
    bzero_diffs = np.where((bmask & ~amask))

    diffs = a - b
    denom = np.where(np.abs(b) > 0, b, 1.0)
    pfull = diffs / denom
    pfull[~bmask] = 0.0
    idmax = np.abs(pfull).argmax().item()
    idfull = tuple(int(i) for i in np.unravel_index(idmax, b.shape))
    pmax = pfull[idfull].item()
    amax = a[idfull].item()
    bmax = b[idfull].item()

    details = [
        f"{name} mismatch: max%err: {pmax:.4e} @{idfull}, a: {amax:.4e}, b: {bmax:.4e}",
    ]
    if len(azero_diffs[0]) > 0:
        details.append(
            f"{name} has {len(azero_diffs[0])} elements nonzero in tensor but zero in reference."
        )
    if len(bzero_diffs[0]) > 0:
        details.append(
            f"{name} has {len(bzero_diffs[0])} elements zero in tensor but nonzero in reference."
        )

    assert False, "\n".join(details)


def _assert_ao_ordering_matches(basis, pmol):
    ref_ao_labels = pmol.ao_labels()
    bfs = basis.to_prims()
    assert len(bfs) == len(ref_ao_labels), "Mismatch in len of AO labels"
    for ii in range(len(bfs)):
        bfa, bfb = str(bfs[ii]), ref_ao_labels[ii]
        ama = cartPow2fullSym(bfs[ii].amom[0].tolist(), style="repeat")
        amb = bfb.split()[-1][1:]
        assert ama == amb, f"AO ordering mismatch, ours: {bfa}  // ref: {bfb}"


def _build_case(sample_case):
    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )
    basis_name, sample_name = sample_case
    mol = load_sample_molecule(sample_name, aoi)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")

    pmol = _build_pyscf_mol(mol, basis_name)
    _assert_ao_ordering_matches(basis, pmol)
    ie = IntegralEngineBase(basis)
    return basis, pmol, ie


def test_one_elec_vs_pyscf(one_elec_case):
    """Check S and H against PySCF reference data."""
    if not HAS_PYSCF:
        pytest.skip("PySCF not available. Install it to run this test.")

    RTOL = 1e-12
    ATOL = 1e-14
    _, pmol, ie = _build_case(one_elec_case)
    s, t, v = ie.build_one_electron_matrices()
    h = t + v

    s_ref = pmol.intor("int1e_ovlp_cart")
    t_ref = pmol.intor("int1e_kin_cart")
    v_ref = pmol.intor("int1e_nuc_cart")
    h_ref = t_ref + v_ref

    sn = 1.0 / np.sqrt(np.diag(s_ref))
    s_ref = np.einsum("i,ij,j->ij", sn, s_ref, sn)
    h_ref = np.einsum("i,ij,j->ij", sn, h_ref, sn)

    _assert_tensor_matches("S", s, s_ref, RTOL, ATOL)
    _assert_tensor_matches("H", h, h_ref, RTOL, ATOL)


def test_eri_vs_pyscf(eri_case):
    """Check ERI against PySCF reference data."""
    if not HAS_PYSCF:
        pytest.skip("PySCF not available. Install it to run this test.")

    RTOL = 1e-10
    ATOL = 1e-12
    _, pmol, ie = _build_case(eri_case)
    eri4d = ie.build_eri_4d_sh()

    s_ref = pmol.intor("int1e_ovlp_cart")
    sn = 1.0 / np.sqrt(np.diag(s_ref))
    eri_ref = pmol.intor("int2e_cart")
    eri_ref = np.einsum("ijkl,i,j,k,l->ijkl", eri_ref, *(sn,) * 4)

    _assert_tensor_matches("ERI", eri4d, eri_ref, RTOL, ATOL)
