import numpy as np
import pytest

from pyquante2g.atoms.molecule import load_sample_molecule
from pyquante2g.basis.basisset import BasisSet
from pyquante2g.basis.integral_engines import IntegralEngineBase, guess_density_from_core
from tests.torch_test_utils import make_torch_ops


@pytest.fixture
def ao():
    return make_torch_ops()


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        x = x.get()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def test_j_batch_matches_dense(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineBase(basis)

    s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
    density = guess_density_from_core(t_mat + v_mat, s_mat, mol.nclosed(), ao)

    j_batch = ie.build_j_batch(density, batch_size=128)
    j_dense = ie.build_j_dense(density)
    assert np.allclose(_to_numpy(j_batch), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)

    j_cached = ie.build_j_batch(density, batch_size=128)
    assert np.allclose(_to_numpy(j_cached), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)


def test_k_batch_matches_dense(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineBase(basis)

    s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
    density = guess_density_from_core(t_mat + v_mat, s_mat, mol.nclosed(), ao)

    k_batch = ie.build_k_batch(density, batch_size=128)
    k_dense = ie.build_k_dense(density)
    assert np.allclose(_to_numpy(k_batch), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)

    k_cached = ie.build_k_batch(density, batch_size=128)
    assert np.allclose(_to_numpy(k_cached), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)


def test_jk_batch_matches_dense(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineBase(basis)

    s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
    density = guess_density_from_core(t_mat + v_mat, s_mat, mol.nclosed(), ao)

    j_batch, k_batch = ie.build_jk_batch(density, batch_size=128)
    j_dense = ie.build_j_dense(density)
    k_dense = ie.build_k_dense(density)
    assert np.allclose(_to_numpy(j_batch), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(k_batch), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)

    j_cached, k_cached = ie.build_jk_batch(density, batch_size=128)
    assert np.allclose(_to_numpy(j_cached), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(k_cached), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)
