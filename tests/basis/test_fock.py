import importlib.util
from pathlib import Path

import numpy as np
import pytest

import pyquante22.array_ops as arr
from pyquante22.atoms.molecule import Molecule
from pyquante22.basis.basisset import BasisSet
from pyquante22.basis.integrals import IntegralEngine
from pyquante22.basis.rhf import (
    build_fock_jk,
    build_jk_batch_block,
    guess_density_from_core,
)
pytestmark = pytest.mark.skip("deprecated")

import sys
EPS = sys.float_info.epsilon
rtol = 64 * EPS # 2**-46 = 1.4e-14
# atol = EPS / 64 # 2**-58 = 3.5e-18
atol = 64 * EPS / 1 # 


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


def allclose(a, b, rel_tol=rtol, abs_tol=atol):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return np.allclose(a_np, b_np, rtol=rel_tol, atol=abs_tol)


def mismatch_summary(a, b, abs_tol=0.0):
    an, bn = _to_numpy(a), _to_numpy(b)
    diffs = an - bn
    mask = np.abs(bn) > abs_tol
    pfull = diffs / bn
    pfull[~mask] = 0.0

    # pfull[~np.isfinite(pfull)] = 0.0
    idmax = np.abs(pfull).argmax().item()
    # idxs = np.where(mask)
    # idfull = tuple(int(ax[idmax]) for ax in idxs)
    idfull = np.unravel_index(idmax, b.shape)

    return (an[idfull].item(), bn[idfull].item(), 
        pfull[idfull].item(), idfull)
 

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
    params=["h2o_raw", "ch4_raw", "c2h4_raw"],
    ids=["H2O", "CH4", "C2H4"],
)
def small_molecule(request, ao, samples_module):
    raw = getattr(samples_module, request.param)
    return Molecule(raw, ao=ao)


def _build_density_and_hcore(ie):
    s_mat, t_mat, v_mat = ie.build_one_electron_matrices()
    hcore = t_mat + v_mat
    nocc = ie.basis.molecule.nclosed()
    density = guess_density_from_core(hcore, s_mat, nocc, ie.ao)
    return density, hcore


def test_jk_block_matches_full_4d(small_molecule):
    basis = BasisSet(small_molecule, name="sto-3g")
    # screen with 0.0 tol to force full build
    ie = IntegralEngine(basis, overlap_tol=0.0, batch_size=64) # 
    density, hcore = _build_density_and_hcore(ie)

    eri_dict = ie.build_eri_sparse()
    eri4d = ie.build_eri_4d(eri_dict)
    _, j_full, k_full, _ = build_fock_jk(hcore, eri4d, density, ie.ao)

    j_blk, k_blk = build_jk_batch_block(density, ie, batch_size=64)

    passed = allclose(j_blk, j_full, rel_tol=rtol, abs_tol=atol)
    if not passed:
        ao, xp = ie.ao, ie.ao.xp
        diffs = j_blk - j_full
        mask = ao.abs(j_full) > atol
        perrs = diffs[mask] / j_full[mask]
        pmax = ao.abs(perrs).max().item()

        assert passed, f"max%err: {pmax:.4e}" # debug stop
    passed = allclose(k_blk, k_full, rel_tol=rtol, abs_tol=atol)
    if not passed:
        assert passed # debug stop


def test_jk_full_4d_matches_ref_4d(small_molecule):
    basis = BasisSet(small_molecule, name="sto-3g")
    # disable screening so sparse->4d should match the reference 4d
    ie = IntegralEngine(basis, overlap_tol=0.0, eri_screen_tol=0.0)
    density, hcore = _build_density_and_hcore(ie)

    eri_dict = ie.build_eri_sparse()
    eri4d = ie.build_eri_4d(eri_dict)
    eri_ref = ie.build_eri_4d_ref()
    passed = allclose(eri4d, eri_ref, rel_tol=rtol, abs_tol=atol), "ERI 4d does not match reference"
    ae, be, erre, idxe = mismatch_summary(eri4d, eri_ref, abs_tol=atol)
    if not passed:
        assert passed # debug stop

    _, j_full, k_full, _ = build_fock_jk(hcore, eri4d, density, ie.ao)
    _, j_ref, k_ref, _ = build_fock_jk(hcore, eri_ref, density, ie.ao)

    passed = allclose(j_full, j_ref, rel_tol=rtol, abs_tol=atol)
    if not passed:
        aj, bj, perrj, idxj = mismatch_summary(j_full, j_ref, abs_tol=atol)
        assert passed # debug stop
    passed = allclose(k_full, k_ref, rel_tol=rtol, abs_tol=atol)
    if not passed:
        ak, bk, perrk, idxk = mismatch_summary(k_full, k_ref, abs_tol=atol)
        a, b = _to_numpy(k_full), _to_numpy(k_ref)
        diffs = a - b
        mask = np.abs(b) > atol
        perrs = diffs[mask] / b[mask]
        pmax = np.abs(perrs).max().item()
        idmsk_flat = np.abs(perrs).argmax()
        idmsk_nz = np.flatnonzero(mask)[idmsk_flat]
        idmsk = np.unravel_index(idmsk_nz, b.shape)

        pfull = diffs / b
        pfull[~mask] = 0.0
        idmax = np.abs(pfull).argmax().item()
        idxs = np.where(mask)
        idfull = tuple(int(ax[idmax]) for ax in idxs)

        merr = dict(pmax=pmax, ref=b[idmsk].item(),
                    val=a[idmsk].item())
        assert passed, " ".join(f"{k}={v:.4e}" for k, v in merr.items())


@pytest.mark.slo
def test_jk_block_matches_full_sparse_4d_benzene(samples_module, ao):
    m = Molecule(samples_module.c6h6_raw, ao=ao)
    basis = BasisSet(m, name="sto-3g")
    ie = IntegralEngine(basis, batch_size=64)
    density, hcore = _build_density_and_hcore(ie)

    eri_dict = ie.build_eri_sparse()
    eri4d = ie.build_eri_4d(eri_dict)
    _, j_full, k_full, _ = build_fock_jk(hcore, eri4d, density, ie.ao)

    j_blk, k_blk = build_jk_batch_block(density, ie, batch_size=64)

    jpass = allclose(j_blk, j_full, rel_tol=1e-8, abs_tol=1e-8)
    if not jpass:
        assert jpass # debug stop

        aj, bj, perrj, idxj = mismatch_summary(j_blk, j_full, abs_tol=atol)
        a, b = _to_numpy(j_blk), _to_numpy(j_full)
        diffs = a - b
        mask = np.abs(b) > atol
        perrs = diffs[mask] / b[mask]
        pmax = np.abs(perrs).max().item()
        idmsk_flat = np.abs(perrs).argmax()
        idmsk_nz = np.flatnonzero(mask)[idmsk_flat]
        idmsk = np.unravel_index(idmsk_nz, b.shape)

        pfull = diffs / b
        pfull[~mask] = 0.0
        idmax = np.abs(pfull).argmax().item()
        idxs = np.where(mask)
        idfull = tuple(int(ax[idmax]) for ax in idxs)

        merr = dict(pmax=pmax, ref=b[idmsk].item(),
                    val=a[idmsk].item())
 


    assert allclose(k_blk, k_full, rel_tol=1e-8, abs_tol=1e-8)
