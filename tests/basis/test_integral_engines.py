import numpy as np
import pytest

from pyquante2g.atoms.molecule import load_sample_molecule
from pyquante2g.basis.basisset import BasisSet
from pyquante2g.basis.integral_engines import (
    IntegralEngineBase,
    IntegralEngineSparse,
    guess_density_from_core,
    ndindex_unique_8fold,
)
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


def test_sparse_batch_iterator_covers_unique_shell_quartets(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineSparse(basis)

    seen = []
    for sh_batch, sh_slices, ao_slices in ie.iter_eri_unique_batch(batch_size=16):
        assert len(sh_slices) == 4
        assert len(ao_slices) == 4
        seen.extend(sh_batch)

    assert seen == list(ndindex_unique_8fold(basis.nsh))


def test_sparse_packed_storage_is_1d(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineSparse(basis)

    s_symm, t_symm, v_symm = ie.build_one_electron_matrices_symm(store=False)
    eri_symm = ie.build_eri_sparse_sh(store=False)

    npair = basis.nbf * (basis.nbf + 1) // 2
    neri = npair * (npair + 1) // 2

    assert s_symm.ndim == 1 and s_symm.shape[0] == npair
    assert t_symm.ndim == 1 and t_symm.shape[0] == npair
    assert v_symm.ndim == 1 and v_symm.shape[0] == npair
    assert eri_symm.ndim == 1 and eri_symm.shape[0] == neri


def test_sparse_one_electron_matrices_match_base(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")

    ie_base = IntegralEngineBase(basis)
    ie_sparse = IntegralEngineSparse(basis)

    s_base, t_base, v_base = ie_base.build_one_electron_matrices()
    s_sparse, t_sparse, v_sparse = ie_sparse.build_one_electron_matrices()

    assert np.allclose(_to_numpy(s_sparse), _to_numpy(s_base), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(t_sparse), _to_numpy(t_base), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(v_sparse), _to_numpy(v_base), rtol=1e-10, atol=1e-12)


def test_shell_pair_packed_access(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineBase(basis)

    ie._build_shell_pairs()

    assert ie._get_shell_pair(0, 1) is ie._get_shell_pair(1, 0)

    pair_ids = ie._pair_pack_idx(slice(0, 2), slice(1, 3), basis.nsh)
    pairs = ie._get_shell_pair(slice(0, 2), slice(1, 3))
    assert tuple(pair_ids.shape) == (2, 2)
    assert pairs[0][0] is ie._get_shell_pair(0, 1)
    assert pairs[1][1] is ie._get_shell_pair(1, 2)

    ii = ao.arr([0, 1, 2], dtype=ao.int_dtype)
    jj = ao.arr([1, 1, 2], dtype=ao.int_dtype)
    pairs = ie._get_shell_pair(ii, jj)
    refs = [(0, 1), (1, 1), (2, 2)]
    assert len(pairs) == len(refs)
    for pair, (i, j) in zip(pairs, refs):
        assert pair is ie._get_shell_pair(i, j)


def test_sparse_eri_matches_dense(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")

    ie_base = IntegralEngineBase(basis)
    ie_sparse_sh = IntegralEngineSparse(basis)
    ie_sparse_batch = IntegralEngineSparse(basis)

    eri_base = ie_base.build_eri_4d_sh()
    eri_sparse_sh = ie_sparse_sh.build_eri_4d_sh()
    eri_sparse_batch = ie_sparse_batch.build_eri_4d_batch(batch_size=128)

    assert np.allclose(_to_numpy(eri_sparse_sh), _to_numpy(eri_base), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(eri_sparse_batch), _to_numpy(eri_base), rtol=1e-10, atol=1e-12)


def test_sparse_jk_matches_dense(ao):
    basis_name, sample_name = "sto-3g", "Water"
    mol = load_sample_molecule(sample_name, ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie_dense = IntegralEngineBase(basis)
    ie_sparse = IntegralEngineSparse(basis)

    s_mat, t_mat, v_mat = ie_dense.build_one_electron_matrices()
    density = guess_density_from_core(t_mat + v_mat, s_mat, mol.nclosed(), ao)

    j_dense = ie_dense.build_j_dense(density)
    k_dense = ie_dense.build_k_dense(density)
    j_sparse = ie_sparse.build_j_batch(density, batch_size=128)
    k_sparse = ie_sparse.build_k_batch(density, batch_size=128)
    j_sparse_dense = ie_sparse.build_j_dense(density)
    k_sparse_dense = ie_sparse.build_k_dense(density)
    j_sparse_jk, k_sparse_jk = ie_sparse.build_jk_batch(density, batch_size=128)

    assert np.allclose(_to_numpy(j_sparse), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(k_sparse), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(j_sparse_dense), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(k_sparse_dense), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(j_sparse_jk), _to_numpy(j_dense), rtol=1e-10, atol=1e-12)
    assert np.allclose(_to_numpy(k_sparse_jk), _to_numpy(k_dense), rtol=1e-10, atol=1e-12)
