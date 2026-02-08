import os
import time
import itertools
import pyquante22.array_ops as arr
from pyquante22.basis.basisset import (
    BasisSet, CartGtoContPair, CartShellBatch,
    CartShellPair,
)

from pyquante22.atoms.molecule import Molecule



def ndindex(shape):
    """n-dim index generator"""
    return itertools.product(*[range(n) for n in shape]) # generator


def ndindex_unique(n, ndim):
    """n-dim equivalent generator of triu indices"""
    return itertools.combinations_with_replacement(range(n), ndim)

def ndindex_unique_8fold(n):
    """
    Unique indices for an 8-fold symmetric 4d tensor. 
    symmetry: (ijkl) = (jikl) = (ijlk) = (jilk); (ijkl) = (klij)
    ordering: i <= j; k <= l; (i, j) <= (k, l) lexicographically
    """
    pairs = list(itertools.combinations_with_replacement(range(n), 2))
    for (i, j), (k, l) in itertools.combinations_with_replacement(pairs, 2):
        yield i, j, k, l


class IntegralEngine():
    """batch compute integral engine
    should work like DataLoader, you give it the 'dataset', which is a basis.
    Then you request a set of matrices and it will yield them back to you in batches
    """
    def __init__(self, basis, overlap_tol=1e-12, eri_screen_tol=None, max_mem=None, precompute=True):
        self.basis = basis
        self.ao = basis.ao
        self.max_mem = max_mem
        self.overlap_tol = self.ao.arr(overlap_tol)
        self.eri_screen_tol = self.overlap_tol if eri_screen_tol is None else self.ao.arr(eri_screen_tol)
        self.shell_pairs = {} # sparse rep of shell pairs, discards non-overlapping shells
        self.shell_pair_ove_max = None # overlap max between shells, assigns 1.0 if shells share a center
        self.shell_pair_eri_max = None # eri max of shells with themselves
        self.eri_screen_min = 0.0
        # item values are (CartShellPair, max_overlap), max_overlap is always 1 if the Pair share the same center

        if precompute: # fill shell_pairs
            self.fill_screened_overlap_pairs()
            self.pair_order ={ # canonical pair order for 4d (triu with pairs of pairs)
                ids: idx for idx, ids in enumerate(ndindex_unique(self.basis.nsh, 2))
            }
    #end def __init__(self, basis, overlap_tol=1e-12, max_mem=None, precompute=True):


    def fill_screened_overlap_pairs(self):

        ao, basis = self.ao, self.basis
        pair_max = ao.zeros((basis.nsh, basis.nsh))

        for sii, sji in ndindex_unique(basis.nsh, 2):
            sij = basis.shells[sii] * basis.shells[sji]
            if ao.sum(ao.square(sij.si.cen - sij.sj.cen)).item() == 0: # same center
                self.shell_pairs[(sii,sji)] = sij
                pair_max[sii,sji] = pair_max[sji,sii] = ao.arr(1.0)
                continue

            ove_prims = sij.bfs.calc_overlap()
            coeffs = sij.bfs.ga.coeffs, sij.bfs.gb.coeffs
            prim_lens = sij.si.npf, sij.sj.npf
            ove = self._contract_strides_2d(ove_prims, prim_lens, coeffs=coeffs, ao=ao)
            ove_max = ao.abs(ove).max()
            if ove_max > self.overlap_tol:
                self.shell_pairs[(sii,sji)] = sij
                pair_max[sii,sji] = pair_max[sji,sii] = ove_max

        self.shell_pair_ove_max = pair_max
    #end def build_screened_overlap_pairs(self):


    def fill_screened_eri_pairs(self):
        ao, basis = self.ao, self.basis
        pair_max = ao.zeros((basis.nsh, basis.nsh))

        if len(self.shell_pairs) > 0: # only iterate the screened pairs
            for ids, sij in self.shell_pairs.items():
                bf_vals = sij | sij
                max_val = ao.abs(bf_vals).max()
                pair_max[ids[0],ids[1]] = max_val
                pair_max[ids[1],ids[0]] = max_val

                del bf_vals

        else: # iterate all pairs
            raise NotImplementedError("@TODO feature")

        self.shell_pair_eri_max = pair_max
        self.eri_screen_min = pair_max[pair_max > 0].min().item()


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

        if len(self.shell_pairs) > 0: # only iterate the screened pairs

            for ids, sij in self.shell_pairs.items():
                coeffs = sij.bfs.ga.coeffs, sij.bfs.gb.coeffs
                prim_lens = sij.si.npf, sij.sj.npf
                bf_ids = basis.shell_range[ids[0]], basis.shell_range[ids[1]]
                bf_rid = bf_ids[1], bf_ids[0] # reveresed ids

                prim_vals = sij.bfs.calc_overlap()
                bf_vals = self._contract_strides_2d(prim_vals, prim_lens, coeffs=coeffs, ao=ao)
                del prim_vals
                s_mat[bf_ids] = bf_vals
                s_mat[bf_rid] = bf_vals.T


                prim_vals = sij.bfs.calc_kinetic()
                bf_vals = self._contract_strides_2d(prim_vals, prim_lens, coeffs=coeffs, ao=ao)
                del prim_vals
                t_mat[bf_ids] = bf_vals
                t_mat[bf_rid] = bf_vals.T

                prim_vals = sij.bfs.calc_nai(rcen)
                prim_vals = ao.einsum("ijc,c->ij", prim_vals, charges)
                bf_vals = self._contract_strides_2d(prim_vals, prim_lens, coeffs=coeffs, ao=ao)
                del prim_vals
                v_mat[bf_ids] = bf_vals
                v_mat[bf_rid] = bf_vals.T

        else: # iterate all pairs
            raise NotImplementedError("@TODO feature")
            # for sii, sji in ndindex((basis.nsh, basis.nsh)):

        return s_mat, t_mat, v_mat
    #end def build_one_electron_matrices(self):


    def build_eri_sparse(self):
        """build a dict of pairs of pairs that holds the ERIs"""
        ao = self.ao
        if self.shell_pair_eri_max is None:
            self.fill_screened_eri_pairs()

        eri_dict = {}
        pairs_list = list(self.shell_pairs.items())
        pair_max = self.shell_pair_eri_max
        pairs_sorted = sorted(pairs_list, key=lambda x: pair_max[x[0]], reverse=True)
        single_tol = self.eri_screen_tol / self.eri_screen_min

        for ii, (ids, sij) in enumerate(pairs_sorted):
            ibound = self.shell_pair_eri_max[ids]
            if ibound < single_tol:
                break

            ktol = self.eri_screen_tol / ibound
            for kk in range(ii, len(pairs_sorted)):
                kds, skl = pairs_sorted[kk]
                kbound = pair_max[kds]
                if kbound < ktol:
                    break

                if self.pair_order[ids] >= self.pair_order[kds]:
                    eri_dict[ids+kds] = sij | skl
                else:
                    eri_dict[kds+ids] = skl | sij

        return eri_dict


    def build_eri_4d(self, eri_dict=None):
        if eri_dict is None:
            eri_dict = self.build_eri_sparse()

        ranges = self.basis.shell_range
        eri4d = self.ao.zeros((self.basis.nbf,)*4)

        pair_list = list(ndindex_unique(self.basis.nsh, 2))
        for p, ids in enumerate(pair_list):
            for q in range(p + 1):
                kds = pair_list[q]
                block = eri_dict[ids+kds]
                rngs = ranges[ids[0]], ranges[ids[1]], ranges[kds[0]], ranges[kds[1]]
                self._fill_eri_symm(eri4d, rngs, block, self.ao)

        return eri4d


    def build_eri_4d_ref(self, eri_dict=None):
        """build in full loop form for error-checking"""
        # still exploit 4-fold symmetry though
        nbf = self.basis.nbf
        bfs = self.basis.to_prims()
        eri4d = self.ao.zeros((self.basis.nbf,)*4)
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
                        eri4d[ii,jj,kk,ll] = val
                        eri4d[jj,ii,kk,ll] = val
                        eri4d[ii,jj,ll,kk] = val
                        eri4d[jj,ii,ll,kk] = val

        return eri4d

    @staticmethod
    def _contract_strides_4d(prim_vals, axis_strides, coeffs=None, ao=None):
        """4d strided contraction with optional coefficients"""
        ao = ao or arr.get_default_ops()
        assert isinstance(prim_vals, ao.ArrayType), "prim_vals must be ArrayType"
        assert all([s%a == 0 for s,a in zip(prim_vals.shape, axis_strides)]), \
            "axis_strides must divide eris.shape"
        angm_lens = [s//a for s,a in zip(prim_vals.shape, axis_strides)]
        new_shape = sum(list(zip(angm_lens, axis_strides)), ())
        vals_tmp = prim_vals.reshape(new_shape)
        if coeffs is None:
            vals_cont = ao.sum(vals_tmp, axis=(1,3,5,7))
        else:
            assert all([len(c) == es for c, es in zip(coeffs, prim_vals.shape)]), \
                "prim_vals + coeffs shape mismatch"
            coeffs  = [c.reshape(-1, pl) for c,pl in zip(coeffs, axis_strides)]
            vals_cont = ao.einsum('iajbkcld,ia,jb,kc,ld->ijkl', vals_tmp,
                            coeffs[0], coeffs[1], coeffs[2], coeffs[3])
        return vals_cont
    # end def eri_contraction_pure(eris, axis_strides, coeffs, ao=None):


    @staticmethod
    def _contract_strides_2d(prim_vals, axis_strides, coeffs=None, ao=None):
        """2d strided contraction with optional coefficients"""
        ao = ao or arr.get_default_ops()
        assert isinstance(prim_vals, ao.ArrayType), "prim_vals must be ArrayType"
        assert all([s%a == 0 for s,a in zip(prim_vals.shape, axis_strides)]), \
            "axis_strides must divide eris.shape"
        angm_lens = [s//a for s,a in zip(prim_vals.shape, axis_strides)]
        new_shape = sum(list(zip(angm_lens, axis_strides)), ())
        vals_tmp = prim_vals.reshape(new_shape)
        if coeffs is None:
            vals_cont = ao.sum(vals_tmp, axis=(1,3,))
        else:
            assert all([len(c) == es for c, es in zip(coeffs, prim_vals.shape)]), \
                "prim_vals + coeffs shape mismatch"
            coeffs  = [c.reshape(-1, pl) for c,pl in zip(coeffs, axis_strides)]
            vals_cont = ao.einsum('iajb,ia,jb->ij', vals_tmp, coeffs[0], coeffs[1])
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
            assert coeffs[d].shape[0] == x.shape[d] if coeffs is not None else True

        dim_ratios = [(d, x.shape[d] // len(axis_counts[d])) for d in range(4)]
        dim_ratios.sort(key=lambda t: t[1], reverse=True)

        for dim, _ in dim_ratios:
            if coeffs is not None:
                c = coeffs[dim]                         # (axis_len,)
                shape = [1] * x.ndim
                shape[dim] = c.shape[0]
                x = x * c.reshape(shape)                # broadcast, no giant coeff tensor
            if x.shape[dim] != len(axis_counts[dim]):   # no need to contract
                x = ao.segment_sum(x, axis_counts[dim], axis=dim)

        return x
    #end def _contract

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


    def calc_eri_batch(self, pair_id, only_triu=False):
        ids = pair_id
        if ids not in self.shell_pairs:
            return {}

        ibound = self.shell_pair_eri_max[ids]
        if ibound < (self.eri_screen_tol / self.eri_screen_min):
            return {}

        sij = self.shell_pairs[ids]
        pairs_list = list(self.shell_pairs.items())
        pair_max = self.shell_pair_eri_max
        pairs_sorted = sorted(pairs_list, key=lambda x: pair_max[x[0]], reverse=True)
        sorted_pos = {pair_ids: i for i, (pair_ids, _) in enumerate(pairs_sorted)}
        ii = sorted_pos[ids] if only_triu else 0

        # skl_list = []
        kds_list = []
        ktol = self.eri_screen_tol / ibound
        for kk in range(ii, len(pairs_sorted)):
            kds, skl = pairs_sorted[kk]
            kbound = pair_max[kds]
            if kbound < ktol:
                break

            # skl_list.append(skl)
            kds_list.append(kds)

        debug_var = 1
        kds_sort = sorted(kds_list)
        batch_size = 3
        ret = {}
        batch_count = len(kds_sort) // batch_size
        if len(kds_sort) % batch_size != 0: batch_count += 1

        for ibatch in range(batch_count):
            ks = slice(ibatch*batch_size, (ibatch+1)*batch_size)
            kds = kds_sort[ks]
            ks = list(set(kl[0] for kl in kds))
            ls = list(set(kl[1] for kl in kds))
            gk = CartShellBatch([self.basis.shells[k] for k in ks])
            gl = CartShellBatch([self.basis.shells[k] for k in ls])
            skl = CartShellPair(gk, gl)
            eris = sij | skl # batched calculation

            cnt = 0
            kszs = [cnt := cnt+sh.nbf for sh in gk.shells] # cumsum
            cnt = 0
            lszs = [cnt := cnt+sh.nbf for sh in gl.shells]

            kmap = {k:i for i,k in enumerate(ks)}
            lmap = {k:i for i,k in enumerate(ls)}
            for k,l in kds:
                ek, el = kmap[k], lmap[l]
                klo = 0 if ek == 0 else kszs[ek-1]
                khi = kszs[ek] if ek < len(kszs) else None
                llo = 0 if el == 0 else lszs[el-1]
                lhi = lszs[el] if el < len(lszs) else None
                if self.pair_order[ids] >= self.pair_order[(k,l)]:
                    ret[ids+(k,l)] = eris[:,:,klo:khi,llo:lhi]
                else:
                    ret[(k,l)+ids] = self.ao.transpose(
                        eris[:,:,klo:khi,llo:lhi],
                        (2,3,0,1))

            debug_var = 1
        return ret
        # what we need to do instead is batch over the l dim, which we can do
        # with the existing machinery and not compute extra terms. We can
        # also batch k and l at once and go ahead and compute the missing
        # cross terms, which is fine, or fill them in with zeros so we don't
        # actually compute that pair from scratch.
        # we still need to do an efficient __add__ implementation for ShellPair
        # or CartPair where it concats the hab terms instead of recomputing.
        # We can add the machinery there to fill in missing cross terms
        # with the computed hab or zeros

        # skl = sum(skl_list) # @TODO, handle sum and ERI over ragged strides
        # bfk_list = [skl.si.bfs for skl in skl_list]
        # bfl_list = [skl.sj.bfs for skl in skl_list]
        # gk = CartShellBatch([skl.si for skl in skl_list])
        # gl = CartShellBatch([skl.sj for skl in skl_list])
        # skl = CartShellPair(gk, gl)
        # ret = sij | skl
        # return ret


#end class IntegralEngine():



    
if __name__ == "__main__":
    import os, time
    import numpy as np
    np.set_printoptions(precision=1, suppress=False, linewidth=200)

    def nsymm(n):
        return n * (n+1) // 2
    def nsymm8(n):
        return nsymm(nsymm(n))
    def five_number_summary(data):
        """ Compute min, 25th percentile, median, 75th percentile, max """
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
    print(f"with basis: {str(b1)} full J is {b1.nbf**4}, uniqs is {nsymm8(b1.nbf)}")

    s_ids = [i for i,s in enumerate(b1.shells) if str(s).startswith('S')] 
    p_ids = [i for i,s in enumerate(b1.shells) if str(s).startswith('P')] 
    d_ids = [i for i,s in enumerate(b1.shells) if str(s).startswith('D')] 
    f_ids = [i for i,s in enumerate(b1.shells) if str(s).startswith('F')] 

    sii = s_ids[0]
    sji = p_ids[0] if len(p_ids) > 0 else s_ids[1]
    ski = d_ids[0] if len(d_ids) > 0 else s_ids[-2]
    sli = s_ids[-1]
    si, sj, sk, sl = b1.shells[sii], b1.shells[sji], b1.shells[ski], b1.shells[sli]
    gi, gj, gk, gl = si.bfs, sj.bfs, sk.bfs, sl.bfs
    gij, gkl = gi*gj, gk*gl

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
        sij = ie1.shell_pairs.get((sii,sji), b1.shells[sii]*b1.shells[sji])
        bf_vals = sij | sij
        max_val = aoi.abs(bf_vals).max()
        pair_max[sii,sji] = max_val
        pair_max[sji,sii] = max_val
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
        assert aoi.abs(perrs).max().item() < 5e-15, f"mismatch in sparse eri building"

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

    #@TODO: batch ERIs 
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

    assert set(eri_dict.keys()) == set(eri_dict2.keys()), "calc_eri_batch(only_triu) make different quartets"

    eri4d2 = ie1.build_eri_4d(eri_dict2)
    diffs = eri4d2 - eri4d
    mask = aoi.abs(eri4d) > 0
    perrs = diffs[mask] / eri4d[mask]
    pmax = aoi.abs(perrs).max().item()
    assert pmax < 1e-15, f"relative error of eri batch build and sparse build disagree max%err: {pmax:.2e}"


    debug_var = 1
