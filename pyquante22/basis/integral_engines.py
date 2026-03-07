import os
import time
from math import floor, ceil
import itertools
from dataclasses import dataclass
import pyquante22.array_ops as arr
from pyquante22.basis.basisset import (
    BasisSet,
    CartGtoContPair,
    CartShellBatch,
    CartShellPair,
)
from pyquante22.basis.cart_gto_utils import cartPow2fullSym

def ndindex(shape):
    """n-dim index generator"""
    return itertools.product(*[range(n) for n in shape])  # generator


def ndindex_unique(n, ndim):
    """n-dim equivalent generator of triu indices"""
    return itertools.combinations_with_replacement(range(n), ndim)


def ndindex_unique_8fold(n):
    """
    Unique indices for an 8-fold symmetric 4d tensor (triu of triu)
    symmetry: (ijkl) = (jikl) = (ijlk) = (jilk); (ijkl) = (klij)
    ordering: i <= j; k <= l; (i, j) <= (k, l) lexicographically
    """
    pairs = list(ndindex_unique(n, 2))
    for (i, j), (k, l) in itertools.combinations_with_replacement(pairs, 2):
        yield i, j, k, l


class IntegralEngineBase:
    """ dense build of one and two-electron integrals
    implements reference, benchmarking, and basic functionality
    """
    def __init__(self, basis):
        self.basis = basis
        self.ao = basis.ao
        self.one_e_mats = {}
        self.two_e_mats = {}
        self.eri_4d = None


    def build_one_electron_matrices(self, store=True):
        bfs = self.basis.to_prims()
        ao, nbf = self.ao, self.basis.nbf
        s_mat = ao.zeros((nbf, nbf))
        t_mat = ao.zeros((nbf, nbf))
        v_mat = ao.zeros((nbf, nbf))

        atoms = self.basis.molecule.atoms
        if atoms:
            rcen = ao.stack([atom.r for atom in atoms])
            charges = ao.arr([atom.Z for atom in atoms])
        else:
            raise NotImplementedError("No logic for no atoms")

        for ii in range(nbf):
            gi = bfs[ii]
            for jj in range(ii, nbf):
                gj = bfs[jj]
                gij = gi * gj
                cij = gi.coeffs, gj.coeffs

                prim_vals = gij.calc_overlap()
                bf_val = ao.einsum('ij,i,j', prim_vals, *cij)
                s_mat[ii,jj] = bf_val
                s_mat[jj,ii] = bf_val  # symmetric

                prim_vals = gij.calc_kinetic()
                bf_val = ao.einsum('ij,i,j', prim_vals, *cij)
                t_mat[ii,jj] = bf_val
                t_mat[jj,ii] = bf_val  # symmetric

                prim_vals = gij.calc_nai(rcen)
                prim_vals = ao.einsum("ijc,c->ij", prim_vals, charges)
                bf_val = ao.einsum('ij,i,j', prim_vals, *cij)
                v_mat[ii,jj] = bf_val
                v_mat[jj,ii] = bf_val  # symmetric

        if store:
            self.one_e_mats['S'] = s_mat
            self.one_e_mats['T'] = t_mat
            self.one_e_mats['V'] = v_mat
        return s_mat, t_mat, v_mat
    #end def build_one_electron_matrices(self):


    def build_eri_4d_bf(self, store=True):
        """build in per-basis func loop form with 4-fold symmetry"""
        nbf = self.basis.nbf
        bfs = self.basis.to_prims()
        eri4d = self.ao.zeros((self.basis.nbf,) * 4)
        for ii in range(nbf):
            gi = bfs[ii]
            for jj in range(ii, nbf):
                gj = bfs[jj]
                gij = gi * gj
                cij = gi.coeffs, gj.coeffs
                for kk in range(nbf):
                    gk = bfs[kk]
                    for ll in range(kk, nbf):
                        gl = bfs[ll]
                        gkl = gk * gl
                        cijkl = cij + (gk.coeffs, gl.coeffs)
                        prim_vals = gij | gkl
                        val = self.ao.einsum('ijkl,i,j,k,l', prim_vals, *cijkl)
                        eri4d[ii, jj, kk, ll] = val
                        eri4d[jj, ii, kk, ll] = val
                        eri4d[ii, jj, ll, kk] = val
                        eri4d[jj, ii, ll, kk] = val

        if store:
            self.eri_4d = eri4d
        return eri4d
    #def build_eri_4d_bf(self):


    def build_eri_4d_sh(self, store=True):
        """build in per-shell loop form with 8-fold symmetry """
        basis = self.basis
        nsh, nbf, ranges = basis.nsh, basis.nbf, basis.shell_range
        eri4d = self.ao.zeros((nbf,) * 4)

        shell_pairs = {}
        for ij in ndindex_unique(nsh, 2):
            i, j = ij
            shell_pairs[ij] = basis.shells[i] * basis.shells[j]

        for ijkl in ndindex_unique_8fold(nsh):
            ij, kl = ijkl[:2], ijkl[2:]
            sij = shell_pairs[ij]
            skl = shell_pairs[kl]
            vals = sij | skl
            rngs = (
                ranges[ij[0]], ranges[ij[1]],
                ranges[kl[0]], ranges[kl[1]],
            )
            self._fill_eri_symm(eri4d, rngs, vals, self.ao)

        if store:
            self.eri_4d = eri4d
        return eri4d
    #end def build_eri_4d_sh(self):


    def build_eri_4d_batch(self, batch_size=64, store=True):
        """build dense eri with batch"""
        basis = self.basis
        nbf = self.basis.nbf
        bfs = self.basis.to_prims()
        eri4d = self.ao.zeros((self.basis.nbf,) * 4)
        for sh_slices, ao_slices in self.iter_eri_batch(batch_size=batch_size):
            ii, jj, kk, ll = sh_slices
            si = CartShellBatch(basis.shells[ii])
            sj = CartShellBatch(basis.shells[jj])
            sk = CartShellBatch(basis.shells[kk])
            sl = CartShellBatch(basis.shells[ll])
            sij = si * sj
            skl = sk * sl
            vals =  sij | skl
            try:
                eri4d[ao_slices] = vals
            except Exception as e:
                debug_var = 1

        if store:
            self.eri_4d = eri4d
        return eri4d


    def iter_eri_batch(self, batch_size=64):
        """Yield ((shell_slices), (ao_slices)) in row-major batching order."""
        basis = self.basis
        nbf = basis.nbf
        nsh = basis.nsh
        shells = basis.shells
        ranges = basis.shell_range
        sh_nbf = [sh.nbf for sh in shells]
        max_sh_nbf = max(sh_nbf) if sh_nbf else 0

        if batch_size is None or batch_size <= 0:
            batch_size = max_sh_nbf**4

        if max_sh_nbf > 0 and batch_size < max_sh_nbf**4:
            print(f"::WARN:: batch_size ({batch_size}) smaller than largest shell quartet ({max_sh_nbf**4})")

        full = slice(0, nsh)

        def _iter_shell_slices(max_ao):
            if max_ao <= 0:
                max_ao = 1
            start = 0
            acc = 0
            for idx, sz in enumerate(sh_nbf):
                if acc == 0 and sz > max_ao:
                    yield slice(idx, idx + 1)
                    start = idx + 1
                    acc = 0
                    continue
                if acc > 0 and acc + sz > max_ao:
                    yield slice(start, idx)
                    start = idx
                    acc = 0
                acc += sz
            if acc > 0:
                yield slice(start, nsh)

        def _ao_slice_from_shell_slice(sh_slice):
            s0 = 0 if sh_slice.start is None else sh_slice.start
            s1 = nsh if sh_slice.stop is None else sh_slice.stop
            if s1 <= s0:
                return slice(0, 0)
            ao_start = ranges[s0].start
            ao_stop = ranges[s1 - 1].stop
            return slice(ao_start, ao_stop)

        def _yield_pair(si, sj, sk, sl):
            ao_i = _ao_slice_from_shell_slice(si)
            ao_j = _ao_slice_from_shell_slice(sj)
            ao_k = _ao_slice_from_shell_slice(sk)
            ao_l = _ao_slice_from_shell_slice(sl)
            yield (si, sj, sk, sl), (ao_i, ao_j, ao_k, ao_l)

        # Try full (j,k,l) rows first if they fit for the largest shell.
        max_i_ao = batch_size // (nbf**3)
        if max_i_ao >= max_sh_nbf and nsh > 0:
            for si in _iter_shell_slices(max_i_ao):
                yield from _yield_pair(si, full, full, full)
            return

        # Otherwise, split j (and deeper if needed) per i shell.
        for i in range(nsh):
            ni = sh_nbf[i]
            max_j_ao = batch_size // (ni * (nbf**2))
            if max_j_ao >= max_sh_nbf:
                for sj in _iter_shell_slices(max_j_ao):
                    yield from _yield_pair(slice(i, i + 1), sj, full, full)
                continue

            for j in range(nsh):
                nj = sh_nbf[j]
                max_k_ao = batch_size // (ni * nj * nbf)
                if max_k_ao >= max_sh_nbf:
                    for sk in _iter_shell_slices(max_k_ao):
                        yield from _yield_pair(
                            slice(i, i + 1), slice(j, j + 1), sk, full
                        )
                    continue

                for k in range(nsh):
                    nk = sh_nbf[k]
                    max_l_ao = batch_size // (ni * nj * nk)
                    for sl in _iter_shell_slices(max_l_ao):
                        yield from _yield_pair(
                            slice(i, i + 1),
                            slice(j, j + 1),
                            slice(k, k + 1),
                            sl,
                        )



    # def build_j_dense(self, density, eri=None):
    #     if eri is None:
    #         if self.eri_4d is None:
    #             eri = self.build_eri_4d_sh()
    #         else:
    #             eri = self.eri_4d

    #     return self.ao.einsum('ijkl,kl', eri, density)

    # def build_j_batch(self, density, batch_size=64):
    #     pass

    @staticmethod
    def _fill_swap_symm(array, ids, vals, ao):
        """fill (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)"""
        array[ids] = vals
        id2 = (ids[1], ids[0], ids[2], ids[3])
        array[id2] = ao.transpose(vals, (1, 0, 2, 3))
        id2 = (ids[0], ids[1], ids[3], ids[2])
        array[id2] = ao.transpose(vals, (0, 1, 3, 2))
        id2 = (ids[1], ids[0], ids[3], ids[2])
        array[id2] = ao.transpose(vals, (1, 0, 3, 2))

    @classmethod
    def _fill_eri_symm(cls, array, ids, vals, ao):
        """fill (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk), repeat for (kl|ij)"""
        cls._fill_swap_symm(array, ids, vals, ao)
        id2 = (ids[2], ids[3], ids[0], ids[1])
        val2 = ao.transpose(vals, (2, 3, 0, 1))
        cls._fill_swap_symm(array, id2, val2, ao)

#end class IntegralEngineBase:


# class IntegralEngineSparse(IntegralEngineBase):
#     """Sparse build/storage of ERIs (no screening).
#     Stores unique (ij|kl) blocks with i<=j, k<=l, (ij)<=(kl).
#     The storage is designed to support efficient J builds without
#     materializing the full 4D tensor.
#     """
#     def __init__(self, basis):
#         super().__init__(basis)
#         self.shell_pairs = None
#         self.eri_sparse = {}


# class ScreenedIntegralEngine(IntegralEngineSparse):
#     """Sparse ERI engine with screening."""
#     def __init__(self, basis, eri_screen_tol=1e-12):
#         super().__init__(basis)
#         self.eri_screen_tol = float(eri_screen_tol)
#         self.shell_pair_eri_max = None




## helper functions for testing 
def orthogonalizer(s_mat, ao):
    evals, evecs = ao.eigh(s_mat)
    neg_count = (evals <= 0).sum()
    if hasattr(neg_count, "item"):
        neg_count = neg_count.item()
    if neg_count:
        raise ValueError("Overlap matrix has non-positive eigenvalues")
    return (evecs / ao.sqrt(evals)) @ ao.transpose(evecs)


def guess_density_from_core(hcore, s_mat, nocc, ao):
    """Build a closed-shell density matrix from the core Hamiltonian."""
    s_inv_sqrt = orthogonalizer(s_mat, ao)

    f_prime = s_inv_sqrt @ hcore @ s_inv_sqrt
    _, c_prime = ao.eigh(f_prime)
    coeffs = s_inv_sqrt @ c_prime

    c_occ = coeffs[:, :nocc]
    density = 2.0 * (c_occ @ ao.transpose(c_occ))
    return density


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def mismatch_summary(name, tensor, ref_tensor, RTOL=1e-14, ATOL=1e-16):
    a = _to_numpy(tensor)
    b = ref_tensor
    if np.allclose(a, b, rtol=RTOL, atol=ATOL):
        return None

    amask = np.abs(a) > ATOL
    bmask = np.abs(b) > ATOL
    azero_diffs = np.where((amask & ~bmask))
    bzero_diffs = np.where((bmask & ~amask))

    diffs = a - b
    denom = np.where(np.abs(b) > 0, b, 1.0)
    pfull = diffs / denom
    pfull[~bmask] = 0.0
    idmax = np.abs(pfull).argmax().item()
    idfull = tuple(int(i) for i in np.unravel_index(idmax, b.shape))
    pmax = pfull[idfull].item()
    amax = a[idfull].item()
    bmax = b[idfull].item()

    details = [
        f"{name} mismatch: max%err: {pmax:.4e} @{idfull}, a: {amax:.4e}, b: {bmax:.4e}",
    ]
    if len(azero_diffs[0]) > 0:
        details.append(
            f"{name} has {len(azero_diffs[0])} elements nonzero in tensor but zero in reference."
        )
    if len(bzero_diffs[0]) > 0:
        details.append(
            f"{name} has {len(bzero_diffs[0])} elements zero in tensor but nonzero in reference."
        )

    return "\n".join(details)




if __name__ == "__main__":    # simple smoke test
    import numpy as np
    from pyquante22.atoms.molecule import Molecule, load_sample_molecule
    from pyquante22.basis.basisset import BasisSet

    np.set_printoptions(precision=2, suppress=False, linewidth=200)
    aoc = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )
    aog = arr.ArrayOps(
        "torch", device="cuda:0", default_dtype="float64", int_dtype="int64"
    )

    sample_molecules = ["Hydrogen", "LiH", "Water", "Methane",]# "Carbon Monoxide", "Carbon Dioxide", "Propane"]
    sample_bases = ["sto-3g", "6-31g", "aug-cc-pvdz"]
    sample_list = [(sample_bases[0], mol) for mol in sample_molecules[:3]]
    # sample_list += [(sample_bases[1], mol) for mol in sample_molecules[1:]]
    # sample_list += [(sample_bases[2], mol) for mol in sample_molecules[:1]]

    tensor_names = ['S', 'H', 'ERI']
    ref_tensor_path = "dev_scripts/tmp"
    rtol, atol = 1e-14, 1e-14

    sample_nbfs, sample_nels = [], []

    for basis_name, sample_name in sample_list:
        # if basis_name != "aug-cc-pvdz" or sample_name != "Water":
        #     continue

        sample_tag = sample_name.replace(" ", "_")
        sample_tag += f"_{basis_name.replace('-', '')}"

        t0 = time.perf_counter()
        print(f"\n=== Sample: {sample_name}, Basis: {basis_name} ===")
        mol = load_sample_molecule(sample_name, aoc)
        basis = BasisSet(mol, name=basis_name)
        print(f"Using sample molecule: {mol} // {basis}")
        sample_nbfs.append(basis.nbf)
        sample_nels.append(mol.nel())

        t0 = time.perf_counter()
        ie = IntegralEngineBase(basis)
        t1 = time.perf_counter()
        s, t, v = ie.build_one_electron_matrices()
        h = t + v
        t2 = time.perf_counter()
        dens0 = guess_density_from_core(h, s, mol.nclosed(), aoc)
        t3 = time.perf_counter()
        print(f"IE init: {t1-t0:.2f}s 1Es: {t2-t1:.2f}s D0: {t3-t2:.2f}")

        t0 = time.perf_counter()
        eri4da = ie.build_eri_4d_batch(batch_size=2**20)
        t1 = time.perf_counter()
        print(f"Time to batch dense 4d eri: {t1-t0:.2f}s")

        # t0 = time.perf_counter()
        # eri4db = ie.build_eri_4d_sh()
        # t1 = time.perf_counter()
        # # j0b = aoc.einsum('ijkl,kl', eri4db, dens0)
        # # t2 = time.perf_counter()
        # print(f"Time to build dense 4d eri: {t1-t0:.2f}s")

        # details = mismatch_summary("eri", eri4da, eri4db)
        # if details: 
        #     print(details)

    debug_var = 1
