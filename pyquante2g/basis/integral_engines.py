import os
import time
from math import floor, ceil
import itertools
from dataclasses import dataclass
import torch
import pyquante2g.torch_ops as arr
from pyquante2g.basis.basisset import (
    BasisSet,
    CartGtoContPair,
    CartShellBatch,
    CartShellPair,
)
from pyquante2g.basis.cart_gto_utils import cartPow2fullSym

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
    """ dense build of one and two-electron integrals for reference
    and basic functionality. 
    """
    def __init__(self, basis):
        self.basis = basis
        self.ao = basis.ao
        self.one_e_mats = {}
        self.two_e_mats = {}
        self.eri_4d = None
        self.shell_pairs = None


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
                bf_val = ao.einsum('ijc,i,j,c', prim_vals, *cij, charges)
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
        self._build_shell_pairs()

        for ijkl in ndindex_unique_8fold(nsh):
            ij, kl = ijkl[:2], ijkl[2:]
            sij = self._get_shell_pair(*ij)
            skl = self._get_shell_pair(*kl)
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


    def build_eri_4d_batch(self, batch_size=int(2*10), store=True):
        """build dense eri with batching"""
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
            eri4d[ao_slices] = vals

        if store:
            self.eri_4d = eri4d
        return eri4d


    def iter_eri_batch(self, batch_size=int(2**10)):
        """Yield ((shell_slices), (ao_slices)) in row-major batching order."""
        # batching scheme tries to fill as many shells as possible, starting
        # with the last axis
        basis = self.basis
        nbf = basis.nbf
        nsh = basis.nsh
        shells = basis.shells
        ranges = basis.shell_range
        sh_nbfs = [sh.nbf for sh in shells]
        max_sh_nbf = max(sh_nbfs) if sh_nbfs else 0

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
            for idx, sz in enumerate(sh_nbfs):
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
            ni = sh_nbfs[i]
            max_j_ao = batch_size // (ni * (nbf**2))
            if max_j_ao >= max_sh_nbf:
                for sj in _iter_shell_slices(max_j_ao):
                    yield from _yield_pair(slice(i, i + 1), sj, full, full)
                continue

            for j in range(nsh):
                nj = sh_nbfs[j]
                max_k_ao = batch_size // (ni * nj * nbf)
                if max_k_ao >= max_sh_nbf:
                    for sk in _iter_shell_slices(max_k_ao):
                        yield from _yield_pair(
                            slice(i, i + 1), slice(j, j + 1), sk, full
                        )
                    continue

                for k in range(nsh):
                    nk = sh_nbfs[k]
                    max_l_ao = batch_size // (ni * nj * nk)
                    for sl in _iter_shell_slices(max_l_ao):
                        yield from _yield_pair(
                            slice(i, i + 1),
                            slice(j, j + 1),
                            slice(k, k + 1),
                            sl,
                        )
    #end def iter_eri_batch(self, batch_size=int(2**10)):


    def build_j_dense(self, density, eri=None):
        if eri is None:
            if self.eri_4d is None:
                eri = self.build_eri_4d_sh()
            else:
                eri = self.eri_4d

        return self.ao.einsum('ijkl,kl', eri, density)


    def build_k_dense(self, density, eri=None):
        if eri is None:
            if self.eri_4d is None:
                eri = self.build_eri_4d_sh()
            else:
                eri = self.eri_4d

        eri_prqs = self.ao.transpose(eri, (0, 2, 1, 3))
        return self.build_j_dense(density, eri=eri_prqs)


    def _iter_eri_blocks(self, batch_size):
        basis = self.basis
        eri4d = self.eri_4d

        for sh_slices, ao_slices in self.iter_eri_batch(batch_size=batch_size):
            if eri4d is None:
                ii, jj, kk, ll = sh_slices
                si = CartShellBatch(basis.shells[ii])
                sj = CartShellBatch(basis.shells[jj])
                sk = CartShellBatch(basis.shells[kk])
                sl = CartShellBatch(basis.shells[ll])
                eri_block = (si * sj) | (sk * sl)
            else:
                eri_block = eri4d[ao_slices]

            yield ao_slices, eri_block


    def build_j_batch(self, density, batch_size=int(2**10)):
        basis = self.basis
        ao = self.ao
        j_mat = ao.zeros((basis.nbf, basis.nbf), dtype=density.dtype)

        for ao_slices, eri_block in self._iter_eri_blocks(batch_size=batch_size):
            ai, aj, ak, al = ao_slices
            d_block = density[ak, al]
            j_block = ao.einsum("ijkl,kl->ij", eri_block, d_block)
            j_mat[ai, aj] = j_mat[ai, aj] + j_block

        return j_mat


    def build_k_batch(self, density, batch_size=int(2**10)):
        basis = self.basis
        ao = self.ao
        k_mat = ao.zeros((basis.nbf, basis.nbf), dtype=density.dtype)

        for ao_slices, eri_block in self._iter_eri_blocks(batch_size=batch_size):
            ai, aj, ak, al = ao_slices
            eri_prqs = ao.transpose(eri_block, (0, 2, 1, 3))
            d_block = density[aj, al]
            k_block = ao.einsum("ikjl,jl->ik", eri_prqs, d_block)
            k_mat[ai, ak] = k_mat[ai, ak] + k_block

        return k_mat


    def build_jk_batch(self, density, batch_size=int(2**10)):
        basis = self.basis
        ao = self.ao
        j_mat = ao.zeros((basis.nbf, basis.nbf), dtype=density.dtype)
        k_mat = ao.zeros((basis.nbf, basis.nbf), dtype=density.dtype)

        for ao_slices, eri_block in self._iter_eri_blocks(batch_size=batch_size):
            ai, aj, ak, al = ao_slices

            d_kl = density[ak, al]
            j_block = ao.einsum("ijkl,kl->ij", eri_block, d_kl)
            j_mat[ai, aj] = j_mat[ai, aj] + j_block

            eri_prqs = ao.transpose(eri_block, (0, 2, 1, 3))
            d_jl = density[aj, al]
            k_block = ao.einsum("ikjl,jl->ik", eri_prqs, d_jl)
            k_mat[ai, ak] = k_mat[ai, ak] + k_block

        return j_mat, k_mat

    @staticmethod
    def _nsymm(n):
        return n * (n + 1) // 2

    def _pair_pack_idx(self, i, j, n):
        ao = self.ao

        def _as_index(idx):
            if isinstance(idx, slice):
                start, stop, step = idx.indices(n)
                return ao.arange(start, stop, step, dtype=ao.int_dtype), True
            if isinstance(idx, range):
                idx = list(idx)
            if isinstance(idx, (list, tuple)):
                idx = ao.arr(idx, dtype=ao.int_dtype)
            elif isinstance(idx, ao.ArrayType):
                idx = idx.to(dtype=ao.int_dtype)
            return idx, False

        i, i_is_slice = _as_index(i)
        j, j_is_slice = _as_index(j)

        if isinstance(i, int) and isinstance(j, int):
            if i > j:
                i, j = j, i
            return i * n - (i * (i - 1)) // 2 + (j - i)

        if not isinstance(i, ao.ArrayType):
            i = ao.arr(i, dtype=ao.int_dtype)
        if not isinstance(j, ao.ArrayType):
            j = ao.arr(j, dtype=ao.int_dtype)

        if i_is_slice or j_is_slice:
            i_scalar = i.ndim == 0
            j_scalar = j.ndim == 0
            if i_scalar:
                i = i.reshape(1)
            else:
                i = i.reshape(-1)
            if j_scalar:
                j = j.reshape(1)
            else:
                j = j.reshape(-1)
            i = i.reshape(-1, 1)
            j = j.reshape(1, -1)
            lo = torch.minimum(i, j)
            hi = torch.maximum(i, j)
            ids = lo * n - (lo * (lo - 1)) // 2 + (hi - lo)
            if i_scalar:
                ids = ids[0]
            if j_scalar:
                ids = ids[:, 0]
            return ids

        i, j = torch.broadcast_tensors(i, j)
        lo = torch.minimum(i, j)
        hi = torch.maximum(i, j)
        return lo * n - (lo * (lo - 1)) // 2 + (hi - lo)

    def _build_shell_pairs(self):
        if self.shell_pairs is not None:
            return self.shell_pairs

        basis = self.basis
        self.shell_pairs = [
            basis.shells[i] * basis.shells[j]
            for i, j in ndindex_unique(basis.nsh, 2)
        ]
        return self.shell_pairs

    def _get_shell_pair(self, i, j):
        shell_pairs = self._build_shell_pairs()
        ids = self._pair_pack_idx(i, j, self.basis.nsh)

        if isinstance(ids, int):
            return shell_pairs[ids]
        if ids.ndim == 0:
            return shell_pairs[int(ids.item())]

        shape = tuple(int(dim) for dim in ids.shape)
        pairs = [shell_pairs[idx] for idx in self.ao.tolist(ids.reshape(-1))]

        def _reshape(vals, dims):
            if len(dims) == 1:
                return vals
            stride = 1
            for dim in dims[1:]:
                stride *= dim
            return [
                _reshape(vals[ii * stride:(ii + 1) * stride], dims[1:])
                for ii in range(dims[0])
            ]

        return _reshape(pairs, shape)


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


class IntegralEngineSparse(IntegralEngineBase):
    """Sparse build/storage of ERIs (no screening).
    Stores unique (ij|kl) blocks with i<=j, k<=l, (ij)<=(kl).
    The storage is designed to support efficient J builds without
    materializing the full 4D tensor.
    """
    def __init__(self, basis):
        super().__init__(basis)
        self.eri_sparse = None
        self.eri_sparse_ids = {}


    def build_one_electron_matrices_symm(self, store=True):
        """build and store in packed triu form"""
        bfs = self.basis.to_prims()
        ao, nbf = self.ao, self.basis.nbf
        npair = (nbf * (nbf + 1)) // 2
        s_mat = ao.zeros((npair,))
        t_mat = ao.zeros((npair,))
        v_mat = ao.zeros((npair,))

        atoms = self.basis.molecule.atoms
        if atoms:
            rcen = ao.stack([atom.r for atom in atoms])
            charges = ao.arr([atom.Z for atom in atoms])
        else:
            raise NotImplementedError("No logic for no atoms")

        for pp, (ii, jj) in enumerate(ndindex_unique(nbf, 2)):
            gi = bfs[ii]
            gj = bfs[jj]
            gij = gi * gj
            cij = gi.coeffs, gj.coeffs

            prim_vals = gij.calc_overlap()
            s_mat[pp] = ao.einsum('ij,i,j', prim_vals, *cij)

            prim_vals = gij.calc_kinetic()
            t_mat[pp] = ao.einsum('ij,i,j', prim_vals, *cij)

            prim_vals = gij.calc_nai(rcen)
            v_mat[pp] = ao.einsum('ijc,i,j,c', prim_vals, *cij, charges)

        if store:
            self.one_e_mats['S_symm'] = s_mat
            self.one_e_mats['T_symm'] = t_mat
            self.one_e_mats['V_symm'] = v_mat
        return s_mat, t_mat, v_mat


    def build_one_electron_matrices(self, store=True):
        """build in packed triu and expand to full"""
        ao, nbf = self.ao, self.basis.nbf
        if not('S_symm' in self.one_e_mats and 
               'T_symm' in self.one_e_mats and 
               'V_symm' in self.one_e_mats
        ):
            s_symm, t_symm, v_symm = self.build_one_electron_matrices_symm(store=store)
        else:
            s_symm = self.one_e_mats['S_symm']
            t_symm = self.one_e_mats['T_symm']
            v_symm = self.one_e_mats['V_symm']

        s_mat = ao.zeros((nbf, nbf))
        t_mat = ao.zeros((nbf, nbf))
        v_mat = ao.zeros((nbf, nbf))

        for pp, (ii, jj) in enumerate(ndindex_unique(nbf, 2)):
            s_val = s_symm[pp]
            t_val = t_symm[pp]
            v_val = v_symm[pp]
            s_mat[ii, jj] = s_val
            s_mat[jj, ii] = s_val
            t_mat[ii, jj] = t_val
            t_mat[jj, ii] = t_val
            v_mat[ii, jj] = v_val
            v_mat[jj, ii] = v_val

        if store:
            self.one_e_mats['S'] = s_mat
            self.one_e_mats['T'] = t_mat
            self.one_e_mats['V'] = v_mat
        return s_mat, t_mat, v_mat

    def build_eri_4d_bf(self, store=True):
        return self.build_eri_4d_sh(store=store)

    def _eri_sparse_size(self):
        npair = self._nsymm(self.basis.nbf)
        return self._nsymm(npair)

    def _eri_pack_ids(self, ao_slices):
        key = self._ids_key(ao_slices)
        if key in self.eri_sparse_ids:
            return self.eri_sparse_ids[key]

        ao = self.ao
        nbf = self.basis.nbf
        npair = self._nsymm(nbf)
        ai, aj, ak, al = ao_slices

        ii = ao.arange(ai.start, ai.stop, dtype=ao.int_dtype).reshape(-1, 1, 1, 1)
        jj = ao.arange(aj.start, aj.stop, dtype=ao.int_dtype).reshape(1, -1, 1, 1)
        kk = ao.arange(ak.start, ak.stop, dtype=ao.int_dtype).reshape(1, 1, -1, 1)
        ll = ao.arange(al.start, al.stop, dtype=ao.int_dtype).reshape(1, 1, 1, -1)

        ij_lo = ao.minimum(ii, jj)
        ij_hi = ao.maximum(ii, jj)
        kl_lo = ao.minimum(kk, ll)
        kl_hi = ao.maximum(kk, ll)
        ij_ids = self._pair_pack_idx(ij_lo, ij_hi, nbf)
        kl_ids = self._pair_pack_idx(kl_lo, kl_hi, nbf)

        pq_lo = ao.minimum(ij_ids, kl_ids)
        pq_hi = ao.maximum(ij_ids, kl_ids)
        ids = self._pair_pack_idx(pq_lo, pq_hi, npair)
        self.eri_sparse_ids[key] = ids
        return ids

    def build_eri_sparse_sh(self, store=True):
        basis = self.basis
        self._build_shell_pairs()
        eri_sparse = self.ao.zeros((self._eri_sparse_size(),))

        for ijkl in ndindex_unique_8fold(basis.nsh):
            ij, kl = ijkl[:2], ijkl[2:]
            vals = self._get_shell_pair(*ij) | self._get_shell_pair(*kl)
            ao_slices = tuple(basis.shell_range[idx] for idx in ijkl)
            ids = self._eri_pack_ids(ao_slices)
            eri_sparse[ids.reshape(-1)] = vals.reshape(-1)

        if store:
            self.eri_sparse = eri_sparse
            self.two_e_mats['ERI_symm'] = eri_sparse
        return eri_sparse

    def _expand_eri_sparse(self, eri_sparse, batch_size=int(2**10)):
        basis = self.basis
        eri4d = self.ao.zeros((basis.nbf,) * 4)

        for _, ao_slices in self.iter_eri_batch(batch_size=batch_size):
            ids = self._eri_pack_ids(ao_slices)
            eri4d[ao_slices] = eri_sparse[ids.reshape(-1)].reshape(ids.shape)

        return eri4d

    def build_eri_4d_sh(self, store=True):
        eri_sparse = self.build_eri_sparse_sh(store=store)
        eri4d = self._expand_eri_sparse(eri_sparse)

        if store:
            self.eri_4d = eri4d
        return eri4d

    def iter_eri_unique_batch(self, batch_size=int(2**10)):
        basis = self.basis
        nsh = basis.nsh
        ranges = basis.shell_range
        sh_nbf = [sh.nbf for sh in basis.shells]
        max_sh_nbf = max(sh_nbf) if sh_nbf else 0
        max_block_size = max_sh_nbf**4

        if batch_size is None or batch_size <= 0:
            batch_size = basis.nbf**4

        if max_block_size > 0 and batch_size < max_block_size:
            print(f"::WARN:: batch_size ({batch_size}) smaller than largest shell quartet ({max_block_size})")

        def _bounds_to_slices(bounds):
            sh_slices = tuple(slice(lo, hi) for lo, hi in bounds)
            ao_slices = []
            for lo, hi in bounds:
                ao_slices.append(slice(ranges[lo].start, ranges[hi - 1].stop))
            return sh_slices, tuple(ao_slices)

        def _bounds_size(bounds):
            _, ao_slices = _bounds_to_slices(bounds)
            size = 1
            for sl in ao_slices:
                size *= sl.stop - sl.start
            return size

        sh_batch = []
        bounds = None
        for ijkl in ndindex_unique_8fold(nsh):
            new_bounds = []
            for axis, idx in enumerate(ijkl):
                if bounds is None:
                    new_bounds.append((idx, idx + 1))
                else:
                    lo, hi = bounds[axis]
                    new_bounds.append((min(lo, idx), max(hi, idx + 1)))
            new_bounds = tuple(new_bounds)

            if sh_batch and _bounds_size(new_bounds) > batch_size:
                sh_slices, ao_slices = _bounds_to_slices(bounds)
                yield sh_batch, sh_slices, ao_slices
                sh_batch = []
                bounds = None
                new_bounds = tuple((idx, idx + 1) for idx in ijkl)

            sh_batch.append(ijkl)
            bounds = new_bounds

        if sh_batch:
            sh_slices, ao_slices = _bounds_to_slices(bounds)
            yield sh_batch, sh_slices, ao_slices

    def build_eri_sparse_batch(self, batch_size=int(2**10), store=True):
        basis = self.basis
        eri_sparse = self.ao.zeros((self._eri_sparse_size(),))

        for _, sh_slices, ao_slices in self.iter_eri_unique_batch(batch_size=batch_size):
            ii, jj, kk, ll = sh_slices
            si = CartShellBatch(basis.shells[ii])
            sj = CartShellBatch(basis.shells[jj])
            sk = CartShellBatch(basis.shells[kk])
            sl = CartShellBatch(basis.shells[ll])
            vals = (si * sj) | (sk * sl)
            ids = self._eri_pack_ids(ao_slices)
            eri_sparse[ids.reshape(-1)] = vals.reshape(-1)

        if store:
            self.eri_sparse = eri_sparse
            self.two_e_mats['ERI_symm'] = eri_sparse
        return eri_sparse

    def build_eri_4d_batch(self, batch_size=int(2*10), store=True):
        eri_sparse = self.build_eri_sparse_batch(batch_size=batch_size, store=store)
        eri4d = self._expand_eri_sparse(eri_sparse, batch_size=batch_size)

        if store:
            self.eri_4d = eri4d
        return eri4d

    @staticmethod
    def _ids_key(ids):
        return tuple(
            (idx.start, idx.stop, idx.step) if isinstance(idx, slice) else idx
            for idx in ids
        )

    def _iter_eri_blocks(self, batch_size):
        basis = self.basis

        for sh_slices, ao_slices in self.iter_eri_batch(batch_size=batch_size):
            if self.eri_sparse is None:
                ii, jj, kk, ll = sh_slices
                si = CartShellBatch(basis.shells[ii])
                sj = CartShellBatch(basis.shells[jj])
                sk = CartShellBatch(basis.shells[kk])
                sl = CartShellBatch(basis.shells[ll])
                eri_block = (si * sj) | (sk * sl)
            else:
                ids = self._eri_pack_ids(ao_slices)
                eri_block = self.eri_sparse[ids.reshape(-1)].reshape(ids.shape)

            yield ao_slices, eri_block

    def build_j_dense(self, density, eri=None):
        if eri is not None:
            return super().build_j_dense(density, eri=eri)
        return self.build_j_batch(density, batch_size=None)

    def build_k_dense(self, density, eri=None):
        if eri is not None:
            return super().build_k_dense(density, eri=eri)
        return self.build_k_batch(density, batch_size=None)

    def build_j_batch(self, density, batch_size=int(2**10)):
        return super().build_j_batch(density, batch_size=batch_size)

    def build_k_batch(self, density, batch_size=int(2**10)):
        return super().build_k_batch(density, batch_size=batch_size)

    def build_jk_batch(self, density, batch_size=int(2**10)):
        return super().build_jk_batch(density, batch_size=batch_size)


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
    b = _to_numpy(ref_tensor)
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
    from pyquante2g.atoms.molecule import Molecule, load_sample_molecule
    from pyquante2g.basis.basisset import BasisSet

    np.set_printoptions(precision=2, suppress=False, linewidth=200)
    aoc = arr.TorchOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )
    aog = arr.TorchOps(
        "torch", device="cuda:0", default_dtype="float64", int_dtype="int64"
    )
    ao_dict = dict(cpu=aoc, gpu=aog)

    # sample_molecules = ["Hydrogen", "LiH", "Water", "Methane",]# "Carbon Monoxide", "Carbon Dioxide", "Propane"]
    sample_molecules = ["Hydrogen", "LiH", "Water", "Methane", "Carbon Dioxide",] 
    sample_bases = ["sto-3g", "6-31g", "aug-cc-pvdz"]
    sample_list = []
    sample_list += [(sample_bases[0], mol) for mol in sample_molecules[:3]]
    sample_list += [(sample_bases[1], mol) for mol in sample_molecules[3:]]
    sample_list += [(sample_bases[2], mol) for mol in sample_molecules[:2]]

    tensor_names = ['S', 'H', 'ERI']
    ref_tensor_path = "dev_scripts/tmp"
    rtol, atol = 1e-14, 1e-14
    batch_size = int(2**16)

    sample_nbfs, sample_nels = [], []
    max_nbf = int(10000 ** 0.5)

    for basis_name, sample_name in sample_list:
        # if basis_name != "aug-cc-pvdz" or sample_name != "Water":
        #     continue
        sample_tag = sample_name.replace(" ", "_")
        sample_tag += f"_{basis_name.replace('-', '')}"
        print(f"\n=== Sample: {sample_name}, Basis: {basis_name} ===")

        for device, aoi in ao_dict.items():
            t0 = time.perf_counter()
            mol = load_sample_molecule(sample_name, aoi)
            basis = BasisSet(mol, name=basis_name)
            if device == 'cpu': print(f"Using sample molecule: {mol} // {basis}")
            if basis.nbf > max_nbf:
                print(f"skipping because basis is too large (>{max_nbf})")
                continue
            if device == 'cpu': continue # @debug
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
            print(f"{device} // IE init: {t1-t0:.2f}s 1Es: {t2-t1:.2f}s D0: {t3-t2:.2f}")

            t0 = time.perf_counter()
            j1, k1 = ie.build_jk_batch(dens0, batch_size=batch_size)
            t1 = time.perf_counter()
            print(f"{device} // Time to batch build J/K: {t1-t0:.2f}s")

            t0 = time.perf_counter()
            eri4da = ie.build_eri_4d_batch(batch_size=batch_size)
            t1 = time.perf_counter()
            print(f"{device} // Time to batch dense 4d eri: {t1-t0:.2f}s")

            t0 = time.perf_counter()
            ie2 = IntegralEngineSparse(basis)
            t1 = time.perf_counter()
            s2, t2, v2 = ie2.build_one_electron_matrices()
            h2 = t2 + v2
            t2 = time.perf_counter()
            t3 = time.perf_counter()
            print(f"{device} // IE2 init: {t1-t0:.2f}s 1Es: {t2-t1:.2f}s D0: {t3-t2:.2f}")

            t0 = time.perf_counter()
            eri4d2_symm = ie2.build_eri_sparse_batch(batch_size=batch_size)
            t1 = time.perf_counter()
            eri4d2 = ie2._expand_eri_sparse(eri4d2_symm, batch_size=batch_size)
            t2 = time.perf_counter()
            ie2.eri_4d = eri4d2
            print(f"{device} // Time to batch sparse 4d eri: {t1-t0:.2f}s")
            print(f"{device} // Time to expand sparse 4d eri: {t2-t1:.2f}s")
            print(f"{device} // Time to batch+expand sparse 4d eri: {t2-t0:.2f}s")

            assert torch.allclose(eri4d2, eri4da), "mismatch in packed symm and dense builds"

            # t0 = time.perf_counter()
            # eri4db = ie.build_eri_4d_sh()
            # t1 = time.perf_counter()
            # j0b = aoc.einsum('ijkl,kl', eri4db, dens0)
            # t2 = time.perf_counter()
            # print(f"{device} // Time to build dense 4d eri: {t1-t0:.2f}s")

            # details = mismatch_summary("eri", eri4da, eri4db)
            # if details: 
            #     print(details)
            debug_var = 1 

    debug_var = 1
