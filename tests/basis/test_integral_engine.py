import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import pyquante22.array_ops as arr
from pyquante22.atoms.molecule import Molecule
from pyquante22.basis.basisset import BasisSet
from pyquante22.basis.integrals import IntegralEngine, ndindex_unique

pytestmark = pytest.mark.skip("deprecated")

EPS = sys.float_info.epsilon
rtol = 64 * EPS # 2**-46 = 1.4e-14
atol = EPS / 64 # 2**-58 = 3.5e-18


def allclose(a, b, rel_tol=rtol, abs_tol=atol):
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return np.allclose(a_np, b_np, rtol=rel_tol, atol=abs_tol)


def _load_samples_module():
    samples_path = Path(__file__).resolve().parents[2] / "samples" / "samples.py"
    spec = importlib.util.spec_from_file_location("samples.samples", samples_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ao():
    return arr.ArrayOps(
        "torch", device=None, default_dtype="float64", int_dtype="int64"
    )


@pytest.fixture(scope="module")
def samples_module():
    return _load_samples_module()


@pytest.fixture(
    params=["h_raw", "h2_raw", "h2o_raw", "ch4_raw"],
    ids=["H", "H2", "H2O", "CH4"],
)
def molecule(request, ao, samples_module):
    raw = getattr(samples_module, request.param)
    return Molecule(raw, ao=ao)


@pytest.fixture(params=["sto-3g"], ids=["sto-3g"])
def basis_name(request):
    return request.param


@pytest.fixture
def basis_set(molecule, basis_name):
    return BasisSet(molecule, name=basis_name)


def test_shell_pair_eri_max_matches_full(basis_set):
    ie = IntegralEngine(basis_set)
    ie.fill_screened_eri_pairs()

    ao = ie.ao
    basis = ie.basis
    pair_max = ao.zeros((basis.nsh, basis.nsh))
    for sii, sji in ndindex_unique(basis.nsh, 2):
        sij = ie.shell_pairs.get(
            (sii, sji), basis.shells[sii] * basis.shells[sji]
        )
        bf_vals = sij | sij
        max_val = ao.abs(bf_vals).max()
        pair_max[sii, sji] = max_val
        pair_max[sji, sii] = max_val
        del bf_vals

    passed = allclose(
        pair_max,
        ie.shell_pair_eri_max,
        rel_tol=rtol,
        abs_tol=float(ie.eri_screen_tol),
    )
    if not passed:
        assert passed # debug stop


def test_sparse_eri_matches_4d_reference(basis_set):
    ie = IntegralEngine(basis_set)
    eri_dict = ie.build_eri_sparse()

    nlim = int(2000**0.5) # equivalent 4d size of a 2k matrix
    if ie.basis.nbf >= nlim:
        pytest.skip("nbf too large for 4d reference comparison")

    eri4d = ie.build_eri_4d(eri_dict)
    eri_ref = ie.build_eri_4d_ref()

    passed = allclose(eri4d, eri_ref, rel_tol=rtol, abs_tol=atol)
    if not passed:
        assert passed # debug stop


def test_calc_eri_batch_matches_sparse(basis_set):
    ie = IntegralEngine(basis_set)
    eri_dict = ie.build_eri_sparse()

    eri_dict2 = {}
    for ids in ndindex_unique(ie.basis.nsh, 2):
        eri_dict2.update(ie.calc_eri_batch(ids, only_triu=True))

    assert set(eri_dict.keys()) == set(eri_dict2.keys())

    nlim = int(2000**0.5) # equivalent 4d size of a 2k matrix
    if ie.basis.nbf >= nlim:
        pytest.skip("nbf too large for 4d batch comparison")

    eri4d = ie.build_eri_4d(eri_dict)
    eri4d2 = ie.build_eri_4d(eri_dict2)
    passed = allclose(eri4d2, eri4d, rel_tol=rtol, abs_tol=atol)

    if not passed:
        ao, xp = ie.ao, ie.ao.xp
        diffs = eri4d - eri4d2
        mask = ao.abs(diffs) > 0
        perrs = diffs[mask] / eri4d2[mask]
        abs_perrs = ao.abs(perrs)
        pmax = abs_perrs.max()
        idmax = abs_perrs.argmax()
        if hasattr(idmax, "item"):
            idmax = idmax.item()
        idxs = ao.where(mask)
        idmsk = tuple(int(ax[idmax]) for ax in idxs)
        val1, val2 = eri4d[idmsk], eri4d2[idmsk]
        assert passed, f"max err at idxs {idmsk}, perr={pmax:.4e}, val1={val1:.4e}, val2={val2:.4e}"
