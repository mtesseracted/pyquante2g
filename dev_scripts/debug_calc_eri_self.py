#!/usr/bin/env python3
"""
Compare build_eri_4d_bf vs build_eri_4d_sh and report mismatched indices.
"""
import argparse
import numpy as np

import pyquante22.array_ops as arr
from pyquante22.atoms.molecule import load_sample_molecule
from pyquante22.basis.basisset import BasisSet
from pyquante22.basis.integral_engines import IntegralEngineBase


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def compare_eri_builds(basis, rtol=1e-10, atol=1e-12):
    ie = IntegralEngineBase(basis)
    eri_bf = _to_numpy(ie.build_eri_4d_bf())
    eri_sh = _to_numpy(ie.build_eri_4d_sh())

    diffs = eri_sh - eri_bf
    thresh = atol + rtol * np.abs(eri_bf)
    mask = np.abs(diffs) > thresh
    if not np.any(mask):
        print("No mismatches found.")
        return

    aeri = np.abs(eri_bf)
    denom = np.where(aeri > atol, aeri, 1.0)
    rels = np.abs(diffs) / aeri

    idxs = np.argwhere(mask)
    print(f"Found {len(idxs)} mismatches:")
    for idx in idxs:
        idx = tuple(int(i) for i in idx)
        a = eri_sh[idx]
        b = eri_bf[idx]
        rel = (a - b) / (b if b != 0 else 1.0)
        print(f"  idx {idx}: sh={a:.6e} bf={b:.6e} rel={rel:.6e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="Hydrogen")
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument("--backend", default="torch", choices=("numpy", "torch"))
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    ao = arr.ArrayOps(args.backend, default_dtype="float64", int_dtype="int64")
    mol = load_sample_molecule(args.sample, ao=ao)
    basis = BasisSet(mol, name=args.basis)

    for ii, bf in enumerate(basis.to_prims()):
        print(f"{ii:02d} ::  {str(bf)}")

    compare_eri_builds(basis, rtol=args.rtol, atol=args.atol)



if __name__ == "__main__":
    main()
