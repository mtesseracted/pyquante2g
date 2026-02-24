import os
import time
import itertools
from dataclasses import dataclass
import pyquante22.array_ops as arr
from pyquante22.basis.basisset import (
    BasisSet,
    CartGtoContPair,
    CartShellBatch,
    CartShellPair,
)

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
    for (i, j), (k, l) in ndindex_unique(pairs, 2):
        yield i, j, k, l


@dataclass
class EriScreenContext:
    pair_max: object  # matrix of ERI maxs between shell i and j
    pairs_sorted: list  # shell pairs, sorted by pair_max values
    pair_pos: dict  # shell pair position in pairs_sorted
    single_tol: object  # screening tolerance of a single pair


class IntegralEngine:
    """batch compute integral engine
    should work like DataLoader, you give it the 'dataset', which is a basis.
    Then you request a set of matrices and it will yield them back to you in batches
    # NOTE: overlap_tol screening is not a rigorous bound!
    """

    def __init__(
        self,
        basis,
        overlap_tol=0.0,
        eri_screen_tol=1e-14,
        max_mem=None,
        precompute=True,
        batch_size=64,
    ):
        self.basis = basis
        self.ao = basis.ao
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.overlap_tol = self.ao.arr(overlap_tol)
        self.eri_screen_tol = self.ao.arr(eri_screen_tol)
        self.shell_pairs = (
            {}
        )  # sparse rep of shell pairs, discards non-overlapping shells
        self.shell_pair_ove_max = None  # overlap max between shells, assigns 1.0 if shells share a center
        self.shell_pair_eri_max = None  # eri max of shells with themselves
        self.eri_screen_min = 0.0
        self.pair_order = None
        self.pairs_sorted = None
        self.pair_pos = None
        self.one_e_mats = {}
        self.eri_sparse = {}
        self.kds_batches_cache = None
        self.kds_batches_batch_size = None
        # item values are (CartShellPair, max_overlap), max_overlap is always 1 if the Pair share the same center

        if precompute:  # fill shell_pairs and pair-order/sorted caches
            self.fill_screened_overlap_pairs()
            self._ensure_pair_order()
            self._prepare_screen_state()
            self._build_kds_batches_cache()

    # end def __init__(self, basis, overlap_tol=1e-12, max_mem=None, precompute=True):

    def _invalidate_after_overlap_change(self):
        """Invalidate all caches that depend on the screened shell-pair set."""
        self.one_e_mats = {}
        self.eri_sparse = {}
        self.shell_pair_ove_max = None
        self.shell_pair_eri_max = None
        self.eri_screen_min = 0.0
        self.pairs_sorted = None
        self.pair_pos = None
        self._invalidate_kds_batches_cache()

    def _invalidate_kds_batches_cache(self):
        self.kds_batches_cache = None
        self.kds_batches_batch_size = None

    def fill_screened_overlap_pairs(self):

        ao, basis = self.ao, self.basis
        self.shell_pairs = {}
        self._invalidate_after_overlap_change()
        pair_max = ao.zeros((basis.nsh, basis.nsh))

        if self.overlap_tol <= 0.0:  # no screening, just fill all pairs
            for sii, sji in ndindex_unique(basis.nsh, 2):
                sij = basis.shells[sii] * basis.shells[sji]
                self.shell_pairs[(sii, sji)] = sij
                pair_max[sii, sji] = pair_max[sji, sii] = ao.arr(1.0)
            self.shell_pair_ove_max = pair_max
            return

        for sii, sji in ndindex_unique(basis.nsh, 2):
            sij = basis.shells[sii] * basis.shells[sji]
            if (
                ao.sum(ao.square(sij.si.cen - sij.sj.cen)).item() == 0
            ):  # same center
                self.shell_pairs[(sii, sji)] = sij
                pair_max[sii, sji] = pair_max[sji, sii] = ao.arr(1.0)
                continue

            ove_prims = sij.bfs.calc_overlap()
            coeffs = sij.bfs.ga.coeffs, sij.bfs.gb.coeffs
            prim_lens = sij.si.npf, sij.sj.npf
            ove = self._contract_strides_2d(
                ove_prims, prim_lens, coeffs=coeffs, ao=ao
            )
            ove_max = ao.abs(ove).max()
            if ove_max > self.overlap_tol:
                self.shell_pairs[(sii, sji)] = sij
                pair_max[sii, sji] = pair_max[sji, sii] = ove_max

        self.shell_pair_ove_max = pair_max

    # end def build_screened_overlap_pairs(self):

    def fill_screened_eri_pairs(self):
        ao, basis = self.ao, self.basis
        pair_max = ao.zeros((basis.nsh, basis.nsh))

        if len(self.shell_pairs) > 0:  # only iterate the screened pairs
            for ids, sij in self.shell_pairs.items():
                bf_vals = sij | sij
                max_val = ao.abs(bf_vals).max()
                # if max_val > self.eri_screen_tol: # might as well store it
                pair_max[ids[0], ids[1]] = max_val
                pair_max[ids[1], ids[0]] = max_val

                del bf_vals

        else:  # iterate all pairs
            raise NotImplementedError("@TODO feature")

        self.shell_pair_eri_max = pair_max
        self.eri_screen_min = pair_max[pair_max > 0].min().item()
        self.eri_screen_max = pair_max.max().item()
        self.pairs_sorted = None
        self.pair_pos = None
        self._invalidate_kds_batches_cache()

    def _ensure_pair_order(self):
        if self.pair_order is None:
            self.pair_order = {
                ids: idx
                for idx, ids in enumerate(ndindex_unique(self.basis.nsh, 2))
            }
        return self.pair_order

    def _prepare_screen_state(self):
        """Ensure screened ERI state is computed and cached."""
        if len(self.shell_pairs) == 0:
            self.fill_screened_overlap_pairs()
        if self.shell_pair_eri_max is None:
            self.fill_screened_eri_pairs()

        if self.pairs_sorted is None or self.pair_pos is None:
            self.pairs_sorted = sorted(
                list(self.shell_pairs.items()),
                key=lambda x: self.shell_pair_eri_max[x[0]],
                reverse=True,
            )
            self.pair_pos = {
                pair_ids: ii for ii, (pair_ids, _) in enumerate(self.pairs_sorted)
            }

        return self.pairs_sorted, self.pair_pos

    def _build_kds_batches_cache(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        screen = self._eri_screen_setup()
        cache = {}
        for ids in sorted(self.shell_pairs.keys()):
            kds_list = list(self._iter_screened_k_pairs(
                ids, screen, only_triu=False))
            cache[ids] = list(
                self._iter_kds_batches(kds_list, batch_size=batch_size)
            )
        self.kds_batches_cache = cache
        self.kds_batches_batch_size = batch_size
        return cache

    def build_one_electron_matrices(self):
        ao, basis, nbf = self.ao, self.basis, self.basis.nbf
        s_mat = ao.zeros((nbf, nbf))
        t_mat = ao.zeros((nbf, nbf))
        v_mat = ao.zeros((nbf, nbf))

        atoms = basis.molecule.atoms
        if atoms:
            rcen = ao.stack([atom.r for atom in atoms])
            charges = ao.arr([atom.Z for atom in atoms])
        else:
            raise NotImplementedError("No logic for no atoms")

        if len(self.shell_pairs) > 0:  # only iterate the screened pairs

            for ids, sij in self.shell_pairs.items():
                coeffs = sij.bfs.ga.coeffs, sij.bfs.gb.coeffs
                prim_lens = sij.si.npf, sij.sj.npf
                bf_ids = basis.shell_range[ids[0]], basis.shell_range[ids[1]]
                bf_rid = bf_ids[1], bf_ids[0]  # reveresed ids

                prim_vals = sij.bfs.calc_overlap()
                bf_vals = self._contract_strides_2d(
                    prim_vals, prim_lens, coeffs=coeffs, ao=ao
                )
                del prim_vals
                s_mat[bf_ids] = bf_vals
                s_mat[bf_rid] = bf_vals.T

                prim_vals = sij.bfs.calc_kinetic()
                bf_vals = self._contract_strides_2d(
                    prim_vals, prim_lens, coeffs=coeffs, ao=ao
                )
                del prim_vals
                t_mat[bf_ids] = bf_vals
                t_mat[bf_rid] = bf_vals.T

                prim_vals = sij.bfs.calc_nai(rcen)
                prim_vals = ao.einsum("ijc,c->ij", prim_vals, charges)
                bf_vals = self._contract_strides_2d(
                    prim_vals, prim_lens, coeffs=coeffs, ao=ao
                )
                del prim_vals
                v_mat[bf_ids] = bf_vals
                v_mat[bf_rid] = bf_vals.T

        else:  # iterate all pairs
            raise NotImplementedError("@TODO feature")
            # for sii, sji in ndindex((basis.nsh, basis.nsh)):

        self.one_e_mats['S'] = s_mat
        self.one_e_mats['T'] = t_mat
        self.one_e_mats['V'] = v_mat
        return s_mat, t_mat, v_mat

    # end def build_one_electron_matrices(self):

    def build_eri_sparse(self):
        """build a dict of pairs of pairs that holds the ERIs"""
        eri_dict = {}
        screen = self._eri_screen_setup()
        for ids, sij in screen.pairs_sorted:
            for kds in self._iter_screened_k_pairs(ids, screen, only_triu=True):
                skl = self.shell_pairs[kds]
                if self.pair_order[ids] >= self.pair_order[kds]:
                    eri_dict[ids + kds] = sij | skl
                else:
                    eri_dict[kds + ids] = skl | sij

        self.eri_sparse = eri_dict
        return eri_dict

    def _eri_screen_setup(self):
        """Build shared data needed for screened ERI pair loops."""
        self._ensure_pair_order()
        pairs_sorted, pair_pos = self._prepare_screen_state()
        return EriScreenContext(
            pair_max=self.shell_pair_eri_max,
            pairs_sorted=pairs_sorted,
            pair_pos=pair_pos,
            # single_tol=self.eri_screen_tol / self.eri_screen_min,
            single_tol=self.eri_screen_tol / self.eri_screen_max,
        )

    def _iter_screened_k_pairs(self, ids, screen, only_triu):
        """Yield screened k-pairs for a fixed i-pair using the sparse ERI rules."""
        if ids not in screen.pair_pos:
            return

        ibound = screen.pair_max[ids]
        if ibound < screen.single_tol:
            return

        start = screen.pair_pos[ids] if only_triu else 0
        ktol = self.eri_screen_tol / ibound
        for kk in range(start, len(screen.pairs_sorted)):
            kds, skl = screen.pairs_sorted[kk]
            if screen.pair_max[kds] < ktol:
                break
            yield kds

    def build_eri_4d(self, eri_dict=None):
        if eri_dict is None:
            if len(self.eri_sparse) > 0:
                eri_dict = self.eri_sparse
            else:
                eri_dict = self.build_eri_sparse()

        ranges = self.basis.shell_range
        eri4d = self.ao.zeros((self.basis.nbf,) * 4)

        pair_list = list(ndindex_unique(self.basis.nsh, 2))
        for p, ids in enumerate(pair_list):
            for q in range(p + 1):
                kds = pair_list[q]
                ikds = ids + kds
                if ikds not in eri_dict:
                    continue
                block = eri_dict[ikds]
                rngs = (
                    ranges[ids[0]],
                    ranges[ids[1]],
                    ranges[kds[0]],
                    ranges[kds[1]],
                )
                self._fill_eri_symm(eri4d, rngs, block, self.ao)

        return eri4d

    def build_eri_4d_ref(self):
        """build in full loop form for error-checking"""
        # still exploit 4-fold symmetry though
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

        return eri4d

    @staticmethod
    def _contract_strides_4d(prim_vals, axis_strides, coeffs=None, ao=None):
        """4d strided contraction with optional coefficients"""
        ao = ao or arr.get_default_ops()
        assert isinstance(prim_vals, ao.ArrayType), "prim_vals must be ArrayType"
        assert all(
            [s % a == 0 for s, a in zip(prim_vals.shape, axis_strides)]
        ), "axis_strides must divide eris.shape"
        angm_lens = [s // a for s, a in zip(prim_vals.shape, axis_strides)]
        new_shape = sum(list(zip(angm_lens, axis_strides)), ())
        vals_tmp = prim_vals.reshape(new_shape)
        if coeffs is None:
            vals_cont = ao.sum(vals_tmp, axis=(1, 3, 5, 7))
        else:
            assert all(
                [len(c) == es for c, es in zip(coeffs, prim_vals.shape)]
            ), "prim_vals + coeffs shape mismatch"
            coeffs = [c.reshape(-1, pl) for c, pl in zip(coeffs, axis_strides)]
            vals_cont = ao.einsum(
                'iajbkcld,ia,jb,kc,ld->ijkl',
                vals_tmp,
                coeffs[0],
                coeffs[1],
                coeffs[2],
                coeffs[3],
            )
        return vals_cont

    # end def eri_contraction_pure(eris, axis_strides, coeffs, ao=None):

    @staticmethod
    def _contract_strides_2d(prim_vals, axis_strides, coeffs=None, ao=None):
        """2d strided contraction with optional coefficients"""
        ao = ao or arr.get_default_ops()
        assert isinstance(prim_vals, ao.ArrayType), "prim_vals must be ArrayType"
        assert all(
            [s % a == 0 for s, a in zip(prim_vals.shape, axis_strides)]
        ), "axis_strides must divide eris.shape"
        angm_lens = [s // a for s, a in zip(prim_vals.shape, axis_strides)]
        new_shape = sum(list(zip(angm_lens, axis_strides)), ())
        vals_tmp = prim_vals.reshape(new_shape)
        if coeffs is None:
            vals_cont = ao.sum(vals_tmp, axis=(1, 3))
        else:
            assert all(
                [len(c) == es for c, es in zip(coeffs, prim_vals.shape)]
            ), "prim_vals + coeffs shape mismatch"
            coeffs = [c.reshape(-1, pl) for c, pl in zip(coeffs, axis_strides)]
            vals_cont = ao.einsum(
                'iajb,ia,jb->ij', vals_tmp, coeffs[0], coeffs[1]
            )
        return vals_cont

    # end def eri_contraction_pure(eris, axis_strides, coeffs, ao=None):

    @staticmethod
    def _contract(x, axis_counts, coeffs=None, ao=None):
        """generalized contraction with ragged packing (non-uniform stride)
        x: inter-primitive values with ragged packing along each primitive axis
        axis_counts: list of lists; axis_counts[d] has n_contracted[d] segment lengths
        coeffs: list of 1D arrays; coeffs[d] length == eris.shape[d]
                    coefficients aligned with the primitive packing of that axis
        returns: contracted array
        """
        ao = ao or arr.get_default_ops()
        assert isinstance(x, ao.ArrayType), "eris must be ao.ArrayType"
        assert len(axis_counts) == x.ndim
        if coeffs is not None:
            assert len(coeffs) == x.ndim
        for d in range(x.ndim):
            assert sum(axis_counts[d]) == x.shape[d]
            assert (
                coeffs[d].shape[0] == x.shape[d] if coeffs is not None else True
            )

        dim_ratios = [(d, x.shape[d] // len(axis_counts[d])) for d in range(4)]
        dim_ratios.sort(key=lambda t: t[1], reverse=True)

        for dim, _ in dim_ratios:
            if coeffs is not None:
                c = coeffs[dim]  # (axis_len,)
                shape = [1] * x.ndim
                shape[dim] = c.shape[0]
                x = x * c.reshape(shape)  # broadcast, no giant coeff tensor
            if x.shape[dim] != len(axis_counts[dim]):  # no need to contract
                x = ao.segment_sum(x, axis_counts[dim], axis=dim)

        return x

    # end def _contract

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

    def calc_eri_batch(self, pair_id, batch_size=3, only_triu=False):
        ids = pair_id
        screen = self._eri_screen_setup()
        if ids not in screen.pair_pos:
            return {}

        sij = self.shell_pairs[ids]
        kds_list = list(self._iter_screened_k_pairs(
            ids, screen, only_triu=only_triu
        ))

        kds_sort = sorted(kds_list)
        ret = {}
        batch_count = (len(kds_sort) + batch_size - 1) // batch_size
        for ibatch in range(batch_count):
            # @TODO: need smarter batching that optimizes common ks and
            # minimizes uneccessary cross terms
            ks = slice(ibatch * batch_size, (ibatch + 1) * batch_size)
            kds = kds_sort[ks]
            ks = list(set(kl[0] for kl in kds))
            ls = list(set(kl[1] for kl in kds))
            gk = CartShellBatch([self.basis.shells[k] for k in ks])
            gl = CartShellBatch([self.basis.shells[k] for k in ls])
            skl = CartShellPair(gk, gl)
            eris = sij | skl  # batched calculation

            k_hi = list(itertools.accumulate(sh.nbf for sh in gk.shells))
            l_hi = list(itertools.accumulate(sh.nbf for sh in gl.shells))
            k_lo = [0] + k_hi[:-1]
            l_lo = [0] + l_hi[:-1]
            k_bounds = {k: (lo, hi) for k, lo, hi in zip(ks, k_lo, k_hi)}
            l_bounds = {l: (lo, hi) for l, lo, hi in zip(ls, l_lo, l_hi)}

            for k, l in kds:
                klo, khi = k_bounds[k]
                llo, lhi = l_bounds[l]
                block = eris[:, :, klo:khi, llo:lhi]
                if self.pair_order[ids] >= self.pair_order[(k, l)]:
                    ret[ids + (k, l)] = block
                else:
                    ret[(k, l) + ids] = self.ao.transpose(block, (2, 3, 0, 1))

        return ret

    # end def calc_eri_batch()

    def _iter_eri_kds_batches(self, kds_list, batch_size=4):
        """Yield (kds, kids, lids) for each batch.

        Groups by k and packs multiple k blocks when the total pair
        count stays within batch_size. kids is a 1D AO-basis index
        array. lids is either a slice if contiguous or a 1D index
        array otherwise.
        """
        if batch_size is not None and batch_size <= 0:
            batch_size = None

        # Group kds by k, then sort k and l to favor contiguous blocks.
        k_to_ls = {}
        for k, l in kds_list:
            k_to_ls.setdefault(k, set()).add(l)

        ranges = self.basis.shell_range
        ao = self.ao
        blocks = []
        for k in sorted(k_to_ls):
            ls_full = sorted(k_to_ls[k])
            if batch_size is None:
                blocks.append((k, ls_full))
            else:
                for ii in range(0, len(ls_full), batch_size):
                    ls_chunk = ls_full[ii : ii + batch_size]
                    blocks.append((k, ls_chunk))

        bb = 0
        while bb < len(blocks):
            cur_kds = []
            cur_ks = set()
            cur_ls = set()
            cur_size = 0

            while bb < len(blocks):
                k, ls = blocks[bb]
                add_size = len(ls)
                if batch_size is not None and cur_size > 0 and cur_size + add_size > batch_size:
                    break
                cur_kds.extend((k, l) for l in ls)
                cur_ks.add(k)
                cur_ls.update(ls)
                cur_size += add_size
                bb += 1

            ks = sorted(cur_ks)
            ls = sorted(cur_ls)
            kids = ao.concat(
                [ao.arange(ranges[k].start, ranges[k].stop) for k in ks]
            )
            l_ranges = [ranges[l] for l in ls]
            is_contig = all(
                l_ranges[ii].stop == l_ranges[ii + 1].start
                for ii in range(len(l_ranges) - 1)
            )
            if is_contig:
                lids = slice(l_ranges[0].start, l_ranges[-1].stop)
            else:
                lids = ao.concat([ao.arange(r.start, r.stop)
                    for r in l_ranges])
            yield cur_kds, kids, lids


    def _iter_kds_batches(self, kds_list, batch_size=4):
        """Yield (kds, kids, lids) batches with non-overlapping 2D slices.

        Groups by k and splits a k when its L-side AO count exceeds batch_size.
        kids is a 1D AO-basis index array. lids is either a slice if contiguous
        or a 1D index array otherwise.
        """
        if batch_size is not None and batch_size <= 0:
            batch_size = None

        k_to_ls = {}
        for k, l in kds_list:
            k_to_ls.setdefault(k, set()).add(l)

        ranges = self.basis.shell_range
        ao = self.ao
        def _ao_len(s):
            r = ranges[s]
            return r.stop - r.start

        blocks = []
        for k in sorted(k_to_ls):
            ls_full = sorted(k_to_ls[k])
            if batch_size is None:
                size = sum(_ao_len(l) for l in ls_full)
                blocks.append((k, ls_full, size))
            else:
                cur = []
                cur_size = 0
                for l in ls_full:
                    l_size = _ao_len(l)
                    if cur and cur_size + l_size > batch_size:
                        blocks.append((k, cur, cur_size))
                        cur = [l]
                        cur_size = l_size
                    else:
                        cur.append(l)
                        cur_size += l_size
                if cur:
                    blocks.append((k, cur, cur_size))

        bb = 0
        while bb < len(blocks):
            cur_ks = set()
            cur_ls = set()
            cur_kds = []
            cur_size = 0

            while bb < len(blocks):
                k, ls_k, add_size = blocks[bb]
                if batch_size is not None and cur_size > 0 and cur_size + add_size > batch_size:
                    break
                cur_ks.add(k)
                cur_ls.update(ls_k)
                cur_kds.extend((k, l) for l in ls_k)
                cur_size += add_size
                bb += 1

            ks = sorted(cur_ks)
            k_ranges = [ranges[k] for k in ks]
            is_k_contig = all(
                k_ranges[ii].stop == k_ranges[ii + 1].start
                for ii in range(len(k_ranges) - 1)
            )
            if is_k_contig:
                kids = slice(k_ranges[0].start, k_ranges[-1].stop)
            else:
                kids = ao.concat([ao.arange(r.start, r.stop) for r in k_ranges])
            ls = sorted(cur_ls)
            l_ranges = [ranges[l] for l in ls]
            is_contig = all(
                l_ranges[ii].stop == l_ranges[ii + 1].start
                for ii in range(len(l_ranges) - 1)
            )
            if is_contig:
                lids = slice(l_ranges[0].start, l_ranges[-1].stop)
            else:
                lids = ao.concat([ao.arange(r.start, r.stop) for r in l_ranges])
            yield cur_kds, kids, lids


    def calc_eri_batch_block(self, pair_id, batch_size=None):
        """Yield batched ERI blocks for a fixed (i,j) pair.

        Batches are grouped to keep as many pairs with the same k index
        together as possible. This avoids returning overlapping (k,l)
        sections across batches.

        Each yielded item is:
            (kds, eris, k_bounds, l_bounds, kids, lids)
        where:
            kds      = list of (k,l) shell pairs included in this batch
            eris     = 4D array with shape (ni, nj, sum_nbk, sum_nbl) in (ij|kl) order
            k_bounds = dict {k: (lo, hi)} indices into the third axis of eris
            l_bounds = dict {l: (lo, hi)} indices into the fourth axis of eris
            kids/lids = AO indices as slice when contiguous, else 1D index array
        """
        ids = pair_id
        screen = self._eri_screen_setup()
        if ids not in screen.pair_pos:
            return

        sij = self.shell_pairs[ids]
        if (
            self.kds_batches_cache is not None
            and batch_size == self.kds_batches_batch_size
            and ids in self.kds_batches_cache
        ):
            kds_iter = self.kds_batches_cache[ids]
        else:
            kds_list = list(self._iter_screened_k_pairs(
                ids, screen, only_triu=False
            ))
            kds_iter = self._iter_kds_batches(
                kds_list, batch_size=batch_size
            )

        for kds, kids, lids in kds_iter:
            ks = list(dict.fromkeys(kl[0] for kl in kds))
            ls = list(dict.fromkeys(kl[1] for kl in kds))
            gk = CartShellBatch([self.basis.shells[k] for k in ks])
            gl = CartShellBatch([self.basis.shells[k] for k in ls])
            skl = CartShellPair(gk, gl)
            eris = sij | skl  # batched calculation in (ij|kl) order

            yield eris, kds, kids, lids

    # end def calc_eri_batch_block()


# end class IntegralEngine():


if __name__ == "__main__":
    import os, time
    import numpy as np
    from pyquante22.atoms.molecule import Molecule

    np.set_printoptions(precision=1, suppress=False, linewidth=200)

    def nsymm(n):
        return n * (n + 1) // 2

    def nsymm8(n):
        return nsymm(nsymm(n))

    def five_number_summary(data):
        """Compute min, 25th percentile, median, 75th percentile, max"""
        q75, q50, q25 = np.percentile(data, [75, 50, 25])
        return np.stack([np.amin(data), q25, q50, q75, np.amax(data)])

    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )

    xyzs = "H 0 0 0\nF 1.4 0 0"
    m1 = Molecule(xyzs, ao=aoi)
    # b1 = BasisSet(m1, name='sto-3g')
    # m1 = Molecule("./samples/glycine.sdf", ao=aoi)
    print(f"Using molecule: {str(m1)}")
    b1 = BasisSet(m1, name='sto-3g')
    # b1 = BasisSet(m1, name='aug-cc-pvqz')
    print(
        f"with basis: {str(b1)} full J is {b1.nbf**4}, uniqs is {nsymm8(b1.nbf)}"
    )

    s_ids = [i for i, s in enumerate(b1.shells) if str(s).startswith('S')]
    p_ids = [i for i, s in enumerate(b1.shells) if str(s).startswith('P')]
    d_ids = [i for i, s in enumerate(b1.shells) if str(s).startswith('D')]
    f_ids = [i for i, s in enumerate(b1.shells) if str(s).startswith('F')]

    sii = s_ids[0]
    sji = p_ids[0] if len(p_ids) > 0 else s_ids[1]
    ski = d_ids[0] if len(d_ids) > 0 else s_ids[-2]
    sli = s_ids[-1]
    si, sj, sk, sl = (
        b1.shells[sii],
        b1.shells[sji],
        b1.shells[ski],
        b1.shells[sli],
    )
    gi, gj, gk, gl = si.bfs, sj.bfs, sk.bfs, sl.bfs
    gij, gkl = gi * gj, gk * gl

    t0 = time.perf_counter()
    ie1 = IntegralEngine(b1)
    t1 = time.perf_counter()
    print(f"Time to do overlap screen pairs: {t1-t0:.2f}s")

    t0 = time.perf_counter()
    S, T, V = ie1.build_one_electron_matrices()
    t1 = time.perf_counter()
    print(f"Time to do build 1e integrals: {t1-t0:.2f}s")

    t0 = time.perf_counter()
    ie1.fill_screened_eri_pairs()
    t1 = time.perf_counter()
    print(f"Time to do eri screen pairs: {t1-t0:.2f}s")

    # explicity build shell_pair_eri_max without overlap screening
    t0 = time.perf_counter()
    pair_max = aoi.zeros((b1.nsh, b1.nsh))
    for sii, sji in ndindex_unique(b1.nsh, 2):
        sij = ie1.shell_pairs.get((sii, sji), b1.shells[sii] * b1.shells[sji])
        bf_vals = sij | sij
        max_val = aoi.abs(bf_vals).max()
        pair_max[sii, sji] = max_val
        pair_max[sji, sii] = max_val
        del bf_vals
    t1 = time.perf_counter()
    print(f"Time to do full eri screen pairs: {t1-t0:.2f}s")
    assert aoi.abs(pair_max - ie1.shell_pair_eri_max).max() < ie1.eri_screen_tol

    t0 = time.perf_counter()
    eri_dict = ie1.build_eri_sparse()
    t1 = time.perf_counter()
    print(f"Time to build sparse eri dict: {t1-t0:.2f}s")
    print(f"Eri sparse contains {len(eri_dict)} blocks")
    block_lens = np.empty(len(eri_dict), dtype=int)
    for ii, (ijkl, vals) in enumerate(eri_dict.items()):
        block_lens[ii] = vals.nelement()
    tot, mu, std = np.sum(block_lens), np.mean(block_lens), np.std(block_lens)
    five_summ = [round(v) for v in five_number_summary(block_lens)]
    print(f"Eri sparse contains {tot} eris in all blocks")
    print(f"len_avg: {mu:.1f} +- {std:.1f} // 5summ: {five_summ}")
    uniqs, counts = np.unique(block_lens, return_counts=True)
    big_num = max(uniqs.max(), counts.max())
    digits = int(np.ceil(np.log10(big_num)))
    print(f"length: " + '  '.join(f"{v:{digits}d}" for v in uniqs))
    print(f"counts: " + '  '.join(f"{v:{digits}d}" for v in counts))

    nlim = int(2000**0.5)
    if b1.nbf < nlim:
        t0 = time.perf_counter()
        eri4d = ie1.build_eri_4d(eri_dict)
        t1 = time.perf_counter()
        print(f"Time to blow up to 4d: {t1-t0:.2f}s")

        t0 = time.perf_counter()
        eri_ref = ie1.build_eri_4d_ref()
        t1 = time.perf_counter()
        print(f"Time to build in 4d: {t1-t0:.2f}s")

        diffs = eri4d - eri_ref
        mask = aoi.abs(eri_ref) > 0
        perrs = diffs[mask] / eri_ref[mask]
        assert (
            aoi.abs(perrs).max().item() < 5e-15
        ), f"mismatch in sparse eri building"

    # cg0 = b1.bfs[0]
    bfs = b1.to_prims()
    for i, bf in enumerate(bfs):
        print(f"BF[{i}]: {str(bf)}")

    cp00 = CartGtoContPair(bfs[0], bfs[0])
    cp01 = CartGtoContPair(bfs[0], bfs[1])
    cp10 = CartGtoContPair(bfs[1], bfs[0])
    cp05 = CartGtoContPair(bfs[0], bfs[5])

    ove01 = cp01.calc_overlap()
    ove05 = cp05.calc_overlap()
    eri0001 = cp00 | cp01
    eri0010 = cp00 | cp01
    eri0005 = cp00 | cp05

    # @TODO: batch screening and one elec ints

    # @TODO: batch ERIs
    # ver1: good for direct_sum scf style
    #   so it will compute the (ij|kl) for a fixed (ij| and concat all the relevant |kl)
    #   sub-step: need to support adding Pairs so it doesn't recompute hab
    # ver2: create a global |kl) object, then index into that so there's no recalculation of H_array
    # ver3: sweep through the molecule (maybe build a tree for traversal), as you sweep you grow/shrink
    #   the |kl) object to what is locally relevant and calculate all the (ij| from members of |kl)
    #   in this version you return an iterator to the user so you define the traversal order

    # cleanups: change the basis.bfs and basis.to_prims() naming convention, it's backwards
    # NB: we're getting to the point where we're using syntactic sugar and calling
    # high level functions, if the AI can't code it correctly that's a point against us.
    # it should start obeying the KISS principle now, and if the AI can't code the solution
    # we're definitely not KISS

    # tmp = ie1.calc_eri_batch((2,3))
    t0 = time.perf_counter()
    eri_dict2 = {}
    for ids in ndindex_unique(b1.nsh, 2):
        eri_dict2.update(ie1.calc_eri_batch(ids, only_triu=True))
    t1 = time.perf_counter()
    print(f"Time to batch build sparse eri dict: {t1-t0:.2f}s")

    assert set(eri_dict.keys()) == set(
        eri_dict2.keys()
    ), "calc_eri_batch(only_triu) make different quartets"

    eri4d2 = ie1.build_eri_4d(eri_dict2)
    diffs = eri4d2 - eri4d
    mask = aoi.abs(eri4d) > 0
    perrs = diffs[mask] / eri4d[mask]
    pmax = aoi.abs(perrs).max().item()
    assert (
        pmax < 1e-15
    ), f"relative error of eri batch build and sparse build disagree max%err: {pmax:.2e}"

    debug_var = 1
