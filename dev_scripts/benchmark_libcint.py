"""Benchmark pyquante22 integrals against libcint through the PySCF interface.

This tries to stay as close to a 1:1 comparison as possible:
- both sides use the same molecule geometry and basis data from this repo
- both sides use Cartesian shells with PySCF ordering
- AO ordering is verified before timing
- timing excludes molecule/basis construction and focuses on integral builds

Because PySCF/libcint returns unnormalized Cartesian AOs while pyquante22
returns normalized contracted AOs, the report includes two libcint timings:
- `libcint raw`: just the `pmol.intor(...)` calls
- `libcint matched`: raw integrals plus the normalization needed to match
  pyquante22's output convention
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from importlib import resources
from typing import Callable

import numpy as np

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


@dataclass
class BenchmarkStats:
    times: list[float]

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times)

    @property
    def median(self) -> float:
        return statistics.median(self.times)


@dataclass
class TensorDiff:
    max_abs: float
    max_rel: float
    idx: tuple[int, ...]
    ours: float
    ref: float


@dataclass
class BenchmarkDefaults:
    cases: list[tuple[str, str]]
    backend: str = "numpy"
    device: str | None = None
    tasks: tuple[str, ...] = ("one", "eri")
    eri_method: str = "sh"
    batch_size: int = 2**20
    warmups: int = 1
    repeats: int = 3
    rtol: float = 1e-10
    atol: float = 1e-12


def parse_args(defaults: BenchmarkDefaults) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="BASIS:MOLECULE",
        help="Benchmark case, e.g. sto-3g:Water. May be passed multiple times.",
    )
    parser.add_argument(
        "--backend",
        default=defaults.backend,
        help="pyquante22 ArrayOps backend for the engine benchmark. Default: numpy",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="ArrayOps device, e.g. cpu or cuda:0. Default: backend default",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("one", "eri"),
        default=defaults.tasks,
        help="Which integral groups to benchmark. Default: one eri",
    )
    parser.add_argument(
        "--eri-method",
        choices=("sh", "batch", "bf"),
        default=defaults.eri_method,
        help="pyquante22 ERI builder to benchmark. Default: sh",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Batch size for --eri-method batch. Default: 1048576",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=defaults.warmups,
        help="Warmup runs before timing. Default: 1",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=defaults.repeats,
        help="Timed repetitions per benchmark. Default: 3",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=defaults.rtol,
        help="Relative tolerance for accuracy checks. Default: 1e-10",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=defaults.atol,
        help="Absolute tolerance for accuracy checks. Default: 1e-12",
    )
    return parser.parse_args()


def parse_cases(
    case_args: list[str], default_cases: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    if not case_args:
        return list(default_cases)

    cases = []
    for raw in case_args:
        if ":" not in raw:
            raise ValueError(f"Invalid --case '{raw}'. Expected BASIS:MOLECULE")
        basis_name, sample_name = raw.split(":", 1)
        basis_name = basis_name.strip()
        sample_name = sample_name.strip()
        if not basis_name or not sample_name:
            raise ValueError(f"Invalid --case '{raw}'. Expected BASIS:MOLECULE")
        cases.append((basis_name, sample_name))
    return cases


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def build_pyscf_mol(mol, basis_name: str):
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
            shell = [sym2am[am_sym.upper()]]
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


def assert_ao_ordering_matches(basis: BasisSet, pmol) -> None:
    ref_ao_labels = pmol.ao_labels()
    bfs = basis.to_prims()
    if len(bfs) != len(ref_ao_labels):
        raise ValueError(
            f"AO count mismatch: pyquante22={len(bfs)} pyscf={len(ref_ao_labels)}"
        )
    for ii, (bf, label) in enumerate(zip(bfs, ref_ao_labels)):
        ours = cartPow2fullSym(bf.amom[0].tolist(), style="repeat")
        ref = label.split()[-1][1:]
        if ours != ref:
            raise ValueError(
                f"AO ordering mismatch at {ii}: pyquante22={ours} pyscf={ref}"
            )


def normalize_one_electron(s_raw, t_raw, v_raw):
    sn = 1.0 / np.sqrt(np.diag(s_raw))
    s_ref = np.einsum("i,ij,j->ij", sn, s_raw, sn)
    t_ref = np.einsum("i,ij,j->ij", sn, t_raw, sn)
    v_ref = np.einsum("i,ij,j->ij", sn, v_raw, sn)
    return s_ref, t_ref, v_ref


def normalize_eri(eri_raw, sn):
    eri = eri_raw
    eri = eri * sn[:, None, None, None]
    eri = eri * sn[None, :, None, None]
    eri = eri * sn[None, None, :, None]
    eri = eri * sn[None, None, None, :]
    return eri


def summarize_diff(ours, ref, atol: float) -> TensorDiff:
    a = _to_numpy(ours)
    b = np.asarray(ref)
    diff = np.abs(a - b)
    idx_abs = tuple(int(i) for i in np.unravel_index(np.argmax(diff), diff.shape))
    max_abs = float(diff[idx_abs])
    rel = diff / np.maximum(np.abs(b), atol)
    max_rel = float(rel.max())
    return TensorDiff(
        max_abs=max_abs,
        max_rel=max_rel,
        idx=idx_abs,
        ours=float(a[idx_abs]),
        ref=float(b[idx_abs]),
    )


def benchmark_call(fn: Callable[[], object], warmups: int, repeats: int) -> tuple[BenchmarkStats, object]:
    for _ in range(warmups):
        fn()

    times = []
    result = None
    for _ in range(repeats):
        gc.collect()
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()
        try:
            t0 = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - t0
        finally:
            if gc_enabled:
                gc.enable()
        times.append(elapsed)
    return BenchmarkStats(times=times), result


def print_stats(label: str, stats: BenchmarkStats) -> None:
    print(
        f"  {label:<18} best={stats.best:9.4f}s "
        f"median={stats.median:9.4f}s mean={stats.mean:9.4f}s"
    )


def print_diff(label: str, diff: TensorDiff, ok: bool) -> None:
    verdict = "ok" if ok else "mismatch"
    print(
        f"  {label:<18} {verdict:<8} "
        f"max_abs={diff.max_abs:10.3e} max_rel={diff.max_rel:10.3e} "
        f"@{diff.idx} ours={diff.ours: .6e} ref={diff.ref: .6e}"
    )


def run_one_electron_case(ie: IntegralEngineBase, pmol, args: argparse.Namespace) -> None:
    print(" one-electron")

    engine_stats, (s_ours, t_ours, v_ours) = benchmark_call(
        lambda: ie.build_one_electron_matrices(store=False),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    libcint_raw_stats, (s_raw, t_raw, v_raw) = benchmark_call(
        lambda: (
            pmol.intor("int1e_ovlp_cart"),
            pmol.intor("int1e_kin_cart"),
            pmol.intor("int1e_nuc_cart"),
        ),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    libcint_matched_stats, (s_ref, t_ref, v_ref) = benchmark_call(
        lambda: normalize_one_electron(
            pmol.intor("int1e_ovlp_cart"),
            pmol.intor("int1e_kin_cart"),
            pmol.intor("int1e_nuc_cart"),
        ),
        warmups=args.warmups,
        repeats=args.repeats,
    )

    print_stats("pyquante22", engine_stats)
    print_stats("libcint raw", libcint_raw_stats)
    print_stats("libcint matched", libcint_matched_stats)
    print(
        f"  speedup vs raw      x{libcint_raw_stats.median / engine_stats.median:8.3f}"
    )
    print(
        f"  speedup vs matched  x{libcint_matched_stats.median / engine_stats.median:8.3f}"
    )

    h_ours = _to_numpy(t_ours) + _to_numpy(v_ours)
    h_ref = t_ref + v_ref
    s_diff = summarize_diff(s_ours, s_ref, args.atol)
    t_diff = summarize_diff(t_ours, t_ref, args.atol)
    v_diff = summarize_diff(v_ours, v_ref, args.atol)
    h_diff = summarize_diff(h_ours, h_ref, args.atol)

    print_diff("S", s_diff, np.allclose(_to_numpy(s_ours), s_ref, rtol=args.rtol, atol=args.atol))
    print_diff("T", t_diff, np.allclose(_to_numpy(t_ours), t_ref, rtol=args.rtol, atol=args.atol))
    print_diff("V", v_diff, np.allclose(_to_numpy(v_ours), v_ref, rtol=args.rtol, atol=args.atol))
    print_diff("H=T+V", h_diff, np.allclose(h_ours, h_ref, rtol=args.rtol, atol=args.atol))


def build_eri_runner(ie: IntegralEngineBase, args: argparse.Namespace) -> Callable[[], object]:
    if args.eri_method == "sh":
        return lambda: ie.build_eri_4d_sh(store=False)
    if args.eri_method == "bf":
        return lambda: ie.build_eri_4d_bf(store=False)
    if args.eri_method == "batch":
        return lambda: ie.build_eri_4d_batch(batch_size=args.batch_size, store=False)
    raise ValueError(f"Unsupported ERI method: {args.eri_method}")


def run_eri_case(ie: IntegralEngineBase, pmol, args: argparse.Namespace) -> None:
    print(f" eri ({args.eri_method})")
    eri_runner = build_eri_runner(ie, args)

    engine_stats, eri_ours = benchmark_call(
        eri_runner,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    libcint_raw_stats, eri_raw = benchmark_call(
        lambda: pmol.intor("int2e_cart"),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    s_raw = pmol.intor("int1e_ovlp_cart")
    sn = 1.0 / np.sqrt(np.diag(s_raw))
    libcint_matched_stats, eri_ref = benchmark_call(
        lambda: normalize_eri(pmol.intor("int2e_cart"), sn),
        warmups=args.warmups,
        repeats=args.repeats,
    )

    print_stats("pyquante22", engine_stats)
    print_stats("libcint raw", libcint_raw_stats)
    print_stats("libcint matched", libcint_matched_stats)
    print(
        f"  speedup vs raw      x{libcint_raw_stats.median / engine_stats.median:8.3f}"
    )
    print(
        f"  speedup vs matched  x{libcint_matched_stats.median / engine_stats.median:8.3f}"
    )

    eri_diff = summarize_diff(eri_ours, eri_ref, args.atol)
    print_diff(
        "ERI",
        eri_diff,
        np.allclose(_to_numpy(eri_ours), eri_ref, rtol=args.rtol, atol=args.atol),
    )


def run_case(case: tuple[str, str], args: argparse.Namespace) -> None:
    basis_name, sample_name = case
    ao = arr.ArrayOps(
        args.backend,
        device=args.device,
        default_dtype="float64",
        int_dtype="int64",
    )

    mol = load_sample_molecule(sample_name, ao=ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    ie = IntegralEngineBase(basis)
    pmol = build_pyscf_mol(mol, basis_name)
    assert_ao_ordering_matches(basis, pmol)

    print()
    print(f"=== {sample_name} / {basis_name} ===")
    print(
        f" nbf={basis.nbf} nsh={basis.nsh} prims={basis.size} "
        f"charge={mol.charge} multiplicity={mol.multiplicity}"
    )

    if "one" in args.tasks:
        run_one_electron_case(ie, pmol, args)
    if "eri" in args.tasks:
        run_eri_case(ie, pmol, args)


def main(defaults: BenchmarkDefaults) -> int:
    args = parse_args(defaults)
    cases = parse_cases(args.case, defaults.cases)

    if not HAS_PYSCF:
        print("PySCF is not available. Install it to benchmark against libcint.")
        print("Suggested command: python3 -m pip install pyscf")
        return 1

    for case in cases:
        run_case(case, args)
    return 0


if __name__ == "__main__":
    defaults = BenchmarkDefaults(
        cases=[
            # ("sto-3g", "Hydrogen"),
            ("sto-3g", "Water"),
            # ("sto-3g", "Methane"),
            ("aug-cc-pvdz", "Water"),
        ],
        backend="torch",
        device="cuda:0",
        tasks=("one", "eri"),
        eri_method="sh",
        batch_size=2**20,
        warmups=1,
        repeats=3,
        rtol=1e-10,
        atol=1e-12,
    )
    raise SystemExit(main(defaults))
