import numpy as np

import pyquante22.array_ops as arr
from pyquante22.basis.basisset import CartShell
from pyquante22.basis.cart_gto_utils import cartPow2fullSym, gen_cartesian_momentums
from pyquante22.basis.integrals import IntegralEngine


def _labels_for_l(ang_mom):
    ams = gen_cartesian_momentums(ang_mom)
    return [cartPow2fullSym(tuple(am)) for am in ams]


def _print_matrix(name, labels, mat):
    mat_np = np.array(mat)
    print(f"{name}:")
    header = " " * 6 + " ".join(f"{lbl:>12s}" for lbl in labels)
    print(header)
    for i, lbl in enumerate(labels):
        row = " ".join(f"{mat_np[i, j]:12.4e}" for j in range(mat_np.shape[1]))
        print(f"{lbl:>4s} {row}")
    print()


def _shell_overlap_and_eri(ao, sh_a, sh_b):
    sij = sh_a * sh_b
    ove_prims = sij.bfs.calc_overlap()
    coeffs = sij.bfs.ga.coeffs, sij.bfs.gb.coeffs
    prim_lens = sh_a.npf, sh_b.npf
    ove = IntegralEngine._contract_strides_2d(
        ove_prims, prim_lens, coeffs=coeffs, ao=ao
    )
    ove_max = float(ao.amax(ao.abs(ove)))

    eri = sij | sij
    eri_diag = ao.zeros((sh_a.nbf, sh_b.nbf))
    for i in range(sh_a.nbf):
        for j in range(sh_b.nbf):
            eri_diag[i, j] = eri[i, j, i, j]
    eri_diag_max = float(ao.amax(ao.abs(eri_diag)))

    return ove, ove_max, eri_diag, eri_diag_max


def _scan_demo_case(ao, alphas, r_vals, ratios, eri_min=1e-3):
    best = None
    for r in r_vals:
        cen_a = ao.arr([0.0, 0.0, 0.0])
        cen_b = ao.arr([0.0, r, 0.0])
        for ratio in ratios:
            coeffs = ao.arr([1.0, -ratio])
            sh_a = CartShell(1, cen_a, alphas, coeffs, ao=ao)
            sh_b = CartShell(1, cen_b, alphas, coeffs, ao=ao)
            ove, ove_max, eri_diag, eri_diag_max = _shell_overlap_and_eri(
                ao, sh_a, sh_b
            )
            if eri_diag_max < eri_min:
                continue
            score = ove_max / eri_diag_max
            if best is None or score < best["score"]:
                best = {
                    "r": float(r),
                    "ratio": float(ratio),
                    "coeffs": coeffs,
                    "ove": ove,
                    "ove_max": ove_max,
                    "eri_diag": eri_diag,
                    "eri_diag_max": eri_diag_max,
                    "score": score,
                }
    return best


def main():
    ao = arr.ArrayOps("torch", default_dtype="float64", int_dtype="int64")
    labels = _labels_for_l(1)

    # Two-primitive contracted p shells with opposite-sign coefficients.
    # Radial cancellation can make overlap tiny even when (ij|ij) is not.
    alphas = ao.arr([8.0, 0.5])
    r_vals = np.linspace(0.8, 2.4, 9)  # center separation in bohr
    ratios = np.linspace(0.2, 2.5, 117)

    best = _scan_demo_case(ao, alphas, r_vals, ratios, eri_min=1e-3)
    if best is None:
        print("No suitable case found. Try expanding r_vals or ratios.")
        return

    print("Demo case: contracted p shells with sign-canceling overlaps")
    print(f"Separation R = {best['r']:.3f} bohr (A at origin, B on +y)")
    print(f"Primitive exponents: {np.array(alphas)}")
    print(f"Contraction coeffs: {np.array(best['coeffs'])}")
    print(f"Max |overlap| across shell components: {best['ove_max']:.4e}")
    print(f"Max |(ij|ij)| across shell components: {best['eri_diag_max']:.4e}")
    print(f"Overlap/ERI ratio: {best['score']:.4e}")
    print()

    _print_matrix("Overlap matrix S_ij", labels, best["ove"])
    _print_matrix("(ij|ij) diagonal", labels, best["eri_diag"])


if __name__ == "__main__":
    main()
