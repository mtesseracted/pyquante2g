"""Benchmark direct shell/block integral calls against libcint via PySCF.

This intentionally avoids dense matrix/tensor builds. It compares individual
shell-pair and shell-quartet evaluations so the benchmark stays focused on the
integral kernels you actually have implemented.
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

from pyquante2g import TorchOps
from pyquante2g.atoms.molecule import load_sample_molecule
from pyquante2g.basis.basisset import BasisSet
from pyquante2g.basis.cart_gto_utils import cartPow2fullSym, sym2am
from pyquante2g.utils.file_utils import find_closest_file_match, load_gbs_file

try:
    from pyscf import gto

    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


ONE_E_INTORS = {
    "overlap": "int1e_ovlp_cart",
    "kinetic": "int1e_kin_cart",
    "nuclear": "int1e_nuc_cart",
}


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
    device: str | None = None
    default_dtype: str = "float64"
    int_dtype: str = "int64"
    integrals: tuple[str, ...] = ("overlap", "kinetic", "nuclear", "eri")
    shell_pairs: list[tuple[int, int]] | None = None
    shell_quartets: list[tuple[int, int, int, int]] | None = None
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
        "--device",
        default=defaults.device,
        help="Torch device, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--dtype",
        default=defaults.default_dtype,
        choices=("float32", "float64"),
        help="Floating-point dtype for pyquante2g tensors",
    )
    parser.add_argument(
        "--int-dtype",
        default=defaults.int_dtype,
        choices=("int32", "int64"),
        help="Integer dtype for pyquante2g tensors",
    )
    parser.add_argument(
        "--integrals",
        nargs="+",
        choices=("overlap", "kinetic", "nuclear", "eri"),
        default=defaults.integrals,
        help="Integral kernels to benchmark",
    )
    parser.add_argument(
        "--shell-pair",
        action="append",
        default=[],
        metavar="I,J",
        help="Shell pair for one-electron blocks, may be passed multiple times",
    )
    parser.add_argument(
        "--shell-quartet",
        action="append",
        default=[],
        metavar="I,J,K,L",
        help="Shell quartet for ERI blocks, may be passed multiple times",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=defaults.warmups,
        help="Warmup runs before timing",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=defaults.repeats,
        help="Timed repetitions per benchmark",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=defaults.rtol,
        help="Relative tolerance for accuracy checks",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=defaults.atol,
        help="Absolute tolerance for accuracy checks",
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


def parse_index_tuple(raw: str, size: int) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != size:
        raise ValueError(f"Expected {size} comma-separated indices, got '{raw}'")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid integer tuple '{raw}'") from exc


def choose_shell_pairs(
    cli_values: list[str], default_values: list[tuple[int, int]] | None
) -> list[tuple[int, int]]:
    if cli_values:
        return [parse_index_tuple(raw, 2) for raw in cli_values]
    return list(default_values or [])


def choose_shell_quartets(
    cli_values: list[str], default_values: list[tuple[int, int, int, int]] | None
) -> list[tuple[int, int, int, int]]:
    if cli_values:
        return [parse_index_tuple(raw, 4) for raw in cli_values]
    return list(default_values or [])


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
    basis_dir = resources.files("pyquante2g.basis").joinpath("cart_gto_data")
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
            f"AO count mismatch: pyquante2g={len(bfs)} pyscf={len(ref_ao_labels)}"
        )
    for ii, (bf, label) in enumerate(zip(bfs, ref_ao_labels)):
        ours = cartPow2fullSym(bf.amom[0].tolist(), style="repeat")
        ref = label.split()[-1][1:]
        if ours != ref:
            raise ValueError(
                f"AO ordering mismatch at {ii}: pyquante2g={ours} pyscf={ref}"
            )


def benchmark_call(
    fn: Callable[[], object], warmups: int, repeats: int
) -> tuple[BenchmarkStats, object]:
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
        f"  {label:<12} best={stats.best:9.4f}s "
        f"median={stats.median:9.4f}s mean={stats.mean:9.4f}s"
    )


def print_diff(label: str, diff: TensorDiff, ok: bool) -> None:
    verdict = "ok" if ok else "mismatch"
    print(
        f"  {label:<12} {verdict:<8} "
        f"max_abs={diff.max_abs:10.3e} max_rel={diff.max_rel:10.3e} "
        f"@{diff.idx} ours={diff.ours: .6e} ref={diff.ref: .6e}"
    )


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


def resolve_shell_indices(idxs: tuple[int, ...], nsh: int) -> tuple[int, ...]:
    resolved = []
    for idx in idxs:
        if idx < 0:
            idx += nsh
        if idx < 0 or idx >= nsh:
            raise IndexError(f"Shell index {idx} out of range for nsh={nsh}")
        resolved.append(idx)
    return tuple(resolved)


def get_shell_norms(pmol, sh_idx: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    if sh_idx not in cache:
        block = np.asarray(pmol.intor_by_shell("int1e_ovlp_cart", (sh_idx, sh_idx)))
        cache[sh_idx] = 1.0 / np.sqrt(np.diag(block))
    return cache[sh_idx]


def normalize_one_electron_block(block, norm_i: np.ndarray, norm_j: np.ndarray):
    return np.einsum("i,ij,j->ij", norm_i, np.asarray(block), norm_j)


def normalize_eri_block(
    block,
    norm_i: np.ndarray,
    norm_j: np.ndarray,
    norm_k: np.ndarray,
    norm_l: np.ndarray,
):
    eri = np.asarray(block)
    eri = eri * norm_i[:, None, None, None]
    eri = eri * norm_j[None, :, None, None]
    eri = eri * norm_k[None, None, :, None]
    eri = eri * norm_l[None, None, None, :]
    return eri


def calc_shell_pair_block(shell_i, shell_j, integral: str, rcen=None, charges=None):
    sp = shell_i * shell_j
    ao = sp.ao
    gp = sp.bfs

    ci = gp.ga.coeffs.reshape(shell_i.nbf, shell_i.npf)
    cj = gp.gb.coeffs.reshape(shell_j.nbf, shell_j.npf)

    if integral == "overlap":
        prim_vals = gp.calc_overlap()
    elif integral == "kinetic":
        prim_vals = gp.calc_kinetic()
    elif integral == "nuclear":
        prim_vals = gp.calc_nai(rcen)
        prim_vals = ao.einsum("ijc,c->ij", prim_vals, charges)
    else:
        raise ValueError(f"Unsupported one-electron integral '{integral}'")

    prim_vals = prim_vals.reshape(shell_i.nbf, shell_i.npf, shell_j.nbf, shell_j.npf)
    return ao.einsum("iajb,ia,jb->ij", prim_vals, ci, cj)


def calc_shell_quartet_block(shell_i, shell_j, shell_k, shell_l):
    return (shell_i * shell_j) | (shell_k * shell_l)


def run_one_electron_benchmark(
    basis: BasisSet,
    pmol,
    args: argparse.Namespace,
    shell_pair: tuple[int, int],
    integral: str,
    rcen,
    charges,
    norm_cache: dict[int, np.ndarray],
) -> None:
    i, j = resolve_shell_indices(shell_pair, basis.nsh)
    shell_i = basis.shells[i]
    shell_j = basis.shells[j]

    print(f" {integral} shells={i,j}")
    engine_stats, ours = benchmark_call(
        lambda: calc_shell_pair_block(shell_i, shell_j, integral, rcen=rcen, charges=charges),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    libcint_stats, ref_raw = benchmark_call(
        lambda: pmol.intor_by_shell(ONE_E_INTORS[integral], (i, j)),
        warmups=args.warmups,
        repeats=args.repeats,
    )

    norm_i = get_shell_norms(pmol, i, norm_cache)
    norm_j = get_shell_norms(pmol, j, norm_cache)
    ref = normalize_one_electron_block(ref_raw, norm_i, norm_j)

    print_stats("pyquante2g", engine_stats)
    print_stats("libcint", libcint_stats)
    print(f"  speedup      x{libcint_stats.median / engine_stats.median:8.3f}")

    diff = summarize_diff(ours, ref, args.atol)
    print_diff(
        integral,
        diff,
        np.allclose(_to_numpy(ours), ref, rtol=args.rtol, atol=args.atol),
    )


def run_eri_benchmark(
    basis: BasisSet,
    pmol,
    args: argparse.Namespace,
    shell_quartet: tuple[int, int, int, int],
    norm_cache: dict[int, np.ndarray],
) -> None:
    i, j, k, l = resolve_shell_indices(shell_quartet, basis.nsh)
    shell_i = basis.shells[i]
    shell_j = basis.shells[j]
    shell_k = basis.shells[k]
    shell_l = basis.shells[l]

    print(f" eri shells={i,j,k,l}")
    engine_stats, ours = benchmark_call(
        lambda: calc_shell_quartet_block(shell_i, shell_j, shell_k, shell_l),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    libcint_stats, ref_raw = benchmark_call(
        lambda: pmol.intor_by_shell("int2e_cart", (i, j, k, l)),
        warmups=args.warmups,
        repeats=args.repeats,
    )

    ref = normalize_eri_block(
        ref_raw,
        get_shell_norms(pmol, i, norm_cache),
        get_shell_norms(pmol, j, norm_cache),
        get_shell_norms(pmol, k, norm_cache),
        get_shell_norms(pmol, l, norm_cache),
    )

    print_stats("pyquante2g", engine_stats)
    print_stats("libcint", libcint_stats)
    print(f"  speedup      x{libcint_stats.median / engine_stats.median:8.3f}")

    diff = summarize_diff(ours, ref, args.atol)
    print_diff(
        "eri",
        diff,
        np.allclose(_to_numpy(ours), ref, rtol=args.rtol, atol=args.atol),
    )


def run_case(
    case: tuple[str, str],
    args: argparse.Namespace,
    shell_pairs: list[tuple[int, int]],
    shell_quartets: list[tuple[int, int, int, int]],
) -> None:
    basis_name, sample_name = case
    ao = TorchOps(
        device=args.device,
        default_dtype=args.dtype,
        int_dtype=args.int_dtype,
    )

    mol = load_sample_molecule(sample_name, ao=ao)
    basis = BasisSet(mol, name=basis_name, cart_order="pyscf")
    pmol = build_pyscf_mol(mol, basis_name)
    assert_ao_ordering_matches(basis, pmol)

    norm_cache: dict[int, np.ndarray] = {}
    rcen = ao.stack([atom.r for atom in mol.atoms])
    charges = ao.arr([atom.Z for atom in mol.atoms])

    print()
    print(f"=== {sample_name} / {basis_name} ===")
    print(
        f" nbf={basis.nbf} nsh={basis.nsh} dtype={args.dtype} "
        f"int_dtype={args.int_dtype} device={ao.device}"
    )

    for integral in args.integrals:
        if integral == "eri":
            for shell_quartet in shell_quartets:
                run_eri_benchmark(basis, pmol, args, shell_quartet, norm_cache)
            continue

        for shell_pair in shell_pairs:
            run_one_electron_benchmark(
                basis,
                pmol,
                args,
                shell_pair,
                integral,
                rcen,
                charges,
                norm_cache,
            )


def main(defaults: BenchmarkDefaults) -> int:
    args = parse_args(defaults)
    cases = parse_cases(args.case, defaults.cases)
    shell_pairs = choose_shell_pairs(args.shell_pair, defaults.shell_pairs)
    shell_quartets = choose_shell_quartets(args.shell_quartet, defaults.shell_quartets)

    if not HAS_PYSCF:
        print("PySCF is not available. Install it to benchmark against libcint.")
        print("Suggested command: python3 -m pip install pyscf")
        return 1

    if any(name in args.integrals for name in ("overlap", "kinetic", "nuclear")) and not shell_pairs:
        raise ValueError("No shell pairs configured for one-electron benchmarks")
    if "eri" in args.integrals and not shell_quartets:
        raise ValueError("No shell quartets configured for ERI benchmarks")

    for case in cases:
        run_case(case, args, shell_pairs, shell_quartets)
    return 0


if __name__ == "__main__":
    defaults = BenchmarkDefaults(
        cases=[
            ("aug-cc-pVQZ", "Cyclobutadiene"),
        ],
        device="cuda:0",
        default_dtype="float64",
        int_dtype="int64",
        integrals=("eri",),
        shell_pairs=[],
        shell_quartets=[
            # aug-cc-pVQZ carbon shells are
            # S,S,S,S,S,S,P,P,P,P,P,D,D,D,D,F,F,F,G,G
            # Cyclobutadiene starts with 4 carbon atoms, so these are G shells
            # on four different atoms.
            (18, 38, 58, 78),
            (19, 39, 59, 79),
        ],
        warmups=1,
        repeats=3,
        rtol=1e-10,
        atol=1e-12,
    )
    raise SystemExit(main(defaults))
