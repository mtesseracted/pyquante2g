#!/usr/bin/env python3
"""Investigate backend disagreement in calc_prod_binoms/calc_h_array.

This script reproduces one test_cart_gto_basic.py case and compares:
1. backend intermediates across two backends
2. backend convolution output vs a host-side reference convolution
3. backend h-array output vs a host-side reference scatter accumulation

The goal is to answer whether disagreement shows up at the operation-order
steps (`conv_full_batch`, scatter-add in `calc_h_array`) or earlier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys

import numpy as np

import pyquante22.array_ops as arr
from pyquante22.basis.cart_gto import CartGtoPrim as CGP
from pyquante22.basis.cart_gto import CartGtoPrimPair as CGPP


def parse_vec(text: str, dtype=float):
    return [dtype(tok) for tok in text.split(",") if tok]


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "get"):
        x = x.get()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def safe_rel(diff, ref, atol):
    denom = np.maximum(np.abs(ref), atol)
    return np.abs(diff) / denom


def allclose_scaled(diff, ref, rtol, atol):
    return np.abs(diff) / (atol + rtol * np.abs(ref))


def topk_indices(metric, k=5):
    flat = metric.reshape(-1)
    if flat.size == 0:
        return []
    k = min(k, flat.size)
    idxs = np.argpartition(flat, -k)[-k:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    return [tuple(int(i) for i in np.unravel_index(idx, metric.shape)) for idx in idxs]


def print_metric(name, test, ref, rtol, atol, topk=5):
    diff = test - ref
    abs_diff = np.abs(diff)
    rel = safe_rel(diff, ref, atol)
    scaled = allclose_scaled(diff, ref, rtol, atol)
    print(f"\n[{name}]")
    print(f"  shape={test.shape}")
    print(f"  max_abs={abs_diff.max():.4e}")
    print(f"  max_rel={rel.max():.4e}")
    print(f"  max_allclose_scaled={scaled.max():.4e}")
    print(f"  allclose={np.allclose(test, ref, rtol=rtol, atol=atol)}")
    for idx in topk_indices(scaled, k=topk):
        print(
            "  "
            f"idx={idx} test={test[idx]:.4e} ref={ref[idx]:.4e} "
            f"abs={abs_diff[idx]:.4e} rel={rel[idx]:.4e} scaled={scaled[idx]:.4e}"
        )


def host_conv_full_batch(a, b):
    return np.stack(
        [np.convolve(sig, ker, mode="full") for sig, ker in zip(a, b)],
        axis=0,
    )


def host_scatter_add_last(x, index, src):
    coords = np.indices(index.shape)
    dest = []
    for dim in range(x.ndim):
        if dim == x.ndim - 1:
            dest.append(index)
        else:
            dest.append(coords[dim])
    np.add.at(x, tuple(dest), src)
    return x


def host_calc_h_array(binom_pres, gamma, binom2s):
    bps = binom_pres[..., None]
    lmax = bps.shape[3] - 1
    full_sz = (
        binom_pres.shape[0],
        binom_pres.shape[1],
        binom_pres.shape[2],
        lmax + 1,
        lmax // 2 + 1,
    )
    g4s = np.broadcast_to((4.0 * gamma)[:, :, None, None, None], full_sz)

    irng = np.arange(full_sz[3])[:, None]
    rrng = np.arange(full_sz[4])[None, :]
    irmask = rrng <= irng // 2
    iflat = np.broadcast_to(irng, full_sz[3:])[irmask]
    rflat = np.broadcast_to(rrng, full_sz[3:])[irmask]
    iiflat = iflat - 2 * rflat

    h_vals_flat = np.broadcast_to(bps, full_sz)[..., irmask]
    h_vals_flat = h_vals_flat * np.power(g4s[..., irmask], (rflat - iflat)[None, None, None, :])
    h_vals_flat = h_vals_flat * binom2s[iflat, rflat][None, None, None, :]

    ret = np.zeros(full_sz[:-1], dtype=np.result_type(binom_pres, gamma, np.float64))
    sum_ids = np.broadcast_to(iiflat.reshape(1, -1), h_vals_flat.shape)
    return host_scatter_add_last(ret, sum_ids, h_vals_flat)


@dataclass
class PairData:
    backend: str
    pair: CGPP
    gamma: np.ndarray
    pab: np.ndarray
    abinoms: np.ndarray
    bbinoms: np.ndarray
    binom_pres: np.ndarray
    hab: np.ndarray
    binom2s: np.ndarray


def make_pair_data(backend, a1, r1, a2, r2, aa, ab):
    ao = arr.ArrayOps(backend, device=None, default_dtype="float64", int_dtype="int64")
    g1 = CGP.create_contraction(
        ao.arr(a1, dtype=ao.int_dtype),
        ao.arr(r1, dtype=ao.default_dtype),
        ao.arr(aa, dtype=ao.default_dtype),
        ao=ao,
    )
    g2 = CGP.create_contraction(
        ao.arr(a2, dtype=ao.int_dtype),
        ao.arr(r2, dtype=ao.default_dtype),
        ao.arr(ab, dtype=ao.default_dtype),
        ao=ao,
    )
    pair = CGPP(g1, g2)
    return PairData(
        backend=backend,
        pair=pair,
        gamma=to_numpy(pair.gamma),
        pab=to_numpy(pair.pab),
        abinoms=to_numpy(pair.abinoms),
        bbinoms=to_numpy(pair.bbinoms),
        binom_pres=to_numpy(pair.binom_pres),
        hab=to_numpy(pair.hab),
        binom2s=to_numpy(pair.co.binom2s),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-backend", default="numpy")
    parser.add_argument("--test-backend", default="cupy")
    parser.add_argument("--compare-backend", default="torch")
    parser.add_argument("--a1", default="0,0,0", help="Angular momentum for ga")
    parser.add_argument("--r1", default="0.0,0.0,0.0", help="Center for ga")
    parser.add_argument("--a2", default="3,1,2", help="Angular momentum for gb")
    parser.add_argument("--r2", default="0.125,0.25,0.5", help="Center for gb")
    parser.add_argument("--aa", default="0.25,0.25,0.25", help="Exponents for ga contraction")
    parser.add_argument("--ab", default="0.5,0.5,0.5", help="Exponents for gb contraction")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    a1 = parse_vec(args.a1, int)
    r1 = parse_vec(args.r1, float)
    a2 = parse_vec(args.a2, int)
    r2 = parse_vec(args.r2, float)
    aa = parse_vec(args.aa, float)
    ab = parse_vec(args.ab, float)

    lmax = sum(a1) + sum(a2)
    rtol = min(lmax + 1, 16) * 32 * sys.float_info.epsilon
    atol = min(lmax + 1, 16) * sys.float_info.epsilon

    try:
        ref = make_pair_data(args.ref_backend, a1, r1, a2, r2, aa, ab)
        test = make_pair_data(args.test_backend, a1, r1, a2, r2, aa, ab)
    except Exception as exc:
        print(f"Failed to initialize backends {args.ref_backend}/{args.test_backend}: {exc}")
        return 1

    extra = None
    if args.compare_backend:
        try:
            extra = make_pair_data(args.compare_backend, a1, r1, a2, r2, aa, ab)
        except Exception as exc:
            print(f"Skipping compare backend {args.compare_backend}: {exc}")

    print("Case:")
    print(f"  ref_backend={args.ref_backend}")
    print(f"  test_backend={args.test_backend}")
    if extra is not None:
        print(f"  compare_backend={args.compare_backend}")
    print(f"  a1={a1} r1={r1} a2={a2} r2={r2} aa={aa} ab={ab}")
    print(f"  rtol={rtol:.4e} atol={atol:.4e}")

    print_metric("pab", test.pab, ref.pab, rtol, atol, topk=args.topk)
    print_metric("abinoms", test.abinoms, ref.abinoms, rtol, atol, topk=args.topk)
    print_metric("bbinoms", test.bbinoms, ref.bbinoms, rtol, atol, topk=args.topk)

    g = np.prod(test.abinoms.shape[:3])
    test_conv_host = host_conv_full_batch(test.abinoms.reshape(g, -1), test.bbinoms.reshape(g, -1))
    test_conv_host = test_conv_host.reshape(test.binom_pres.shape)
    ref_conv_host = host_conv_full_batch(ref.abinoms.reshape(g, -1), ref.bbinoms.reshape(g, -1))
    ref_conv_host = ref_conv_host.reshape(ref.binom_pres.shape)

    print_metric("binom_pres_test_vs_ref", test.binom_pres, ref.binom_pres, rtol, atol, topk=args.topk)
    print_metric(
        "binom_pres_test_vs_hostconv(test inputs)",
        test.binom_pres,
        test_conv_host,
        rtol,
        atol,
        topk=args.topk,
    )
    print_metric(
        "hostconv(test inputs)_vs_hostconv(ref inputs)",
        test_conv_host,
        ref_conv_host,
        rtol,
        atol,
        topk=args.topk,
    )

    h_rtol = min(lmax + 1, 16) * 64 * sys.float_info.epsilon
    h_atol = min(lmax + 1, 16) * sys.float_info.epsilon

    test_h_host = host_calc_h_array(test.binom_pres, test.gamma, test.binom2s)
    test_h_host_from_hostconv = host_calc_h_array(test_conv_host, test.gamma, test.binom2s)
    ref_h_host = host_calc_h_array(ref.binom_pres, ref.gamma, ref.binom2s)

    print_metric("hab_test_vs_ref", test.hab, ref.hab, h_rtol, h_atol, topk=args.topk)
    print_metric(
        "hab_test_vs_hosth(test binom_pres)",
        test.hab,
        test_h_host,
        h_rtol,
        h_atol,
        topk=args.topk,
    )
    print_metric(
        "hosth(test binom_pres)_vs_hosth(test hostconv)",
        test_h_host,
        test_h_host_from_hostconv,
        h_rtol,
        h_atol,
        topk=args.topk,
    )
    print_metric(
        "hosth(test hostconv)_vs_hosth(ref binom_pres)",
        test_h_host_from_hostconv,
        ref_h_host,
        h_rtol,
        h_atol,
        topk=args.topk,
    )

    if extra is not None:
        print_metric(
            f"binom_pres_{args.compare_backend}_vs_{args.ref_backend}",
            extra.binom_pres,
            ref.binom_pres,
            rtol,
            atol,
            topk=args.topk,
        )
        print_metric(
            f"hab_{args.compare_backend}_vs_{args.ref_backend}",
            extra.hab,
            ref.hab,
            h_rtol,
            h_atol,
            topk=args.topk,
        )

    print("\nInterpretation:")
    print("  If binom_pres_test_vs_hostconv is large, the disagreement starts in conv_full_batch.")
    print("  If that is small but hab_test_vs_hosth is large, the disagreement starts in calc_h_array accumulation.")
    print("  If abinoms/bbinoms already differ, the disagreement starts before the reductions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
