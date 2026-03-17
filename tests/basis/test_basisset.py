import numpy as np
import pytest

import pyquante22.array_ops as arr
from pyquante22.atoms.molecule import load_sample_molecule
from pyquante22.basis.basisset import (
    BasisSet,
    CartGtoContPair,
    CartShell,
    CartShellBatch,
)
from pyquante22.basis.cart_gto import CartGtoPrim
from tests.array_ops.conftest import INSTALLED_BACKENDS
import itertools


SAMPLE_MOLECULES = ["Hydrogen", "LiH", "Water"]
SAMPLE_BASES = ["sto-3g", "6-31g", "aug-cc-pvdz"]


@pytest.fixture(params=INSTALLED_BACKENDS)
def ao(request):
    return arr.ArrayOps(
        request.param,
        device=None,
        default_dtype="float64",
        int_dtype="int64",
    )


def _assert_basic_methods(obj):
    s = str(obj)
    r = repr(obj)
    assert isinstance(s, str) and s
    assert isinstance(r, str) and r
    if hasattr(type(obj), "__len__"):
        assert isinstance(len(obj), int)
        assert len(obj) >= 0


def _make_shells(ao):
    sh_s = CartShell(
        amtot=0,
        cen=ao.arr([0.0, 0.0, 0.0]),
        alphas=ao.arr([0.6, 1.2]),
        coeffs=ao.arr([0.3, 0.7]),
        ao=ao,
    )
    sh_p = CartShell(
        amtot=1,
        cen=ao.arr([1.0, 0.0, 0.0]),
        alphas=ao.arr([0.5, 1.1, 2.2]),
        coeffs=ao.arr([0.5, 0.3, 0.2]),
        ao=ao,
    )
    return sh_s, sh_p


def test_cont_pair_smoke(ao):
    ga = CartGtoPrim.create_contraction(
        ao.arr([0, 0, 0], dtype=ao.int_dtype),
        ao.arr([0.0, 0.0, 0.0]),
        ao.arr([0.6, 1.2]),
        coeffs=ao.arr([0.3, 0.7]),
        ao=ao,
    )
    gb = CartGtoPrim.create_contraction(
        ao.arr([1, 0, 0], dtype=ao.int_dtype),
        ao.arr([1.0, 0.0, 0.0]),
        ao.arr([0.5, 1.1]),
        coeffs=ao.arr([0.4, 0.6]),
        ao=ao,
    )
    pair = CartGtoContPair(ga, gb)
    _assert_basic_methods(pair)
    assert pair.gab.shape == (len(ga), len(gb))


def test_shell_smoke(ao):
    sh_s, _ = _make_shells(ao)
    _assert_basic_methods(sh_s)

    prims = sh_s.to_prims()
    bfs = sh_s.bfs
    first = sh_s[0]
    assert len(prims) == sh_s.nbf
    assert len(bfs) == len(sh_s)
    assert isinstance(first, CartGtoPrim)


def test_shell_batch_bfs(ao):
    sh_s, sh_p = _make_shells(ao)

    batch = CartShellBatch([sh_s, sh_p])
    _assert_basic_methods(batch)
    bfs = batch.bfs

    actual = np.asarray(ao.tolist(bfs.alpha))
    expected = np.array([0.6, 1.2] + [0.5, 1.1, 2.2] * sh_p.nbf)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)
    assert np.all(actual > 0.0)


def test_shell_pair_smoke(ao):
    sh_s, sh_p = _make_shells(ao)
    pair = sh_s * sh_p
    _assert_basic_methods(pair)

    bfs = pair.bfs
    assert len(pair) == len(sh_s) * len(sh_p)
    assert bfs.shape == (len(sh_s.bfs), len(sh_p.bfs))


@pytest.mark.parametrize(
    "molecule_name,basis_name",
    itertools.product(SAMPLE_MOLECULES, SAMPLE_BASES),
    ids=[f"{m}-{b}" for m, b in itertools.product(SAMPLE_MOLECULES, SAMPLE_BASES)],
)
def test_basisset_init(
    ao, molecule_name, basis_name,
):
    mol = load_sample_molecule(molecule_name, ao=ao)
    basis = BasisSet(mol, name=basis_name)

    assert basis.nsh == len(basis.shells)
    assert len(basis.shells) == len(basis.shell_range)
    assert basis.nbf > 0
    assert basis.size == sum(len(s) for s in basis.shells)
    assert basis.shell_range[-1].stop == basis.nbf

    to_prim = basis.to_prims()
    to_prim_count = len(to_prim)
    bfs_count = len(basis.bfs)
    assert to_prim_count == basis.nbf
    assert sum(len(bp.alpha) for bp in to_prim) == basis.size
    assert bfs_count == basis.size

    alphas = np.asarray(ao.tolist(basis.bfs.alpha))
    assert np.all(alphas > 0.0)


def test_basisset_smoke(ao):
    mol = load_sample_molecule("Water", ao=ao)
    basis = BasisSet(mol, name="sto-3g")
    _assert_basic_methods(basis)

    first = basis[0]
    first_two = basis[:2]
    bfs = basis.bfs
    prims = basis.to_prims()
    assert isinstance(first, CartGtoPrim)
    assert len(first_two) == 2
    assert len(bfs) == basis.size
    assert len(prims) == basis.nbf


@pytest.mark.parametrize("basis_name", ["not-a-real-basis"])
def test_basisset_bad_name(ao, basis_name):
    mol = load_sample_molecule("Hydrogen", ao=ao)
    with pytest.raises(ValueError, match="Cannot find the .* basis set"):
        BasisSet(mol, name=basis_name)


def test_jax_shell_pair_eri_smoke():
    pytest.importorskip("jax")

    ao = arr.ArrayOps(
        "jax",
        device=None,
        default_dtype="float64",
        int_dtype="int64",
    )
    mol = load_sample_molecule("Hydrogen", ao=ao)
    basis = BasisSet(mol, name="sto-3g")
    sh = basis.shells[0]

    eri = (sh * sh) | (sh * sh)
    eri_np = np.asarray(ao.tolist(eri))

    assert eri_np.shape == (1, 1, 1, 1)
    assert np.all(np.isfinite(eri_np))
