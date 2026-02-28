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


    def build_one_electron_matrices(self, renormalize=False):
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

        if renormalize: # contraction norm
            s_diag = ao.sqrt(ao.diag(s_mat))
            s_mat = ao.einsum('i,j,ij->ij', s_diag, s_diag, s_mat)
            t_mat = ao.einsum('i,j,ij->ij', s_diag, s_diag, t_mat)
            v_mat = ao.einsum('i,j,ij->ij', s_diag, s_diag, v_mat)

        self.one_e_mats['S'] = s_mat
        self.one_e_mats['T'] = t_mat
        self.one_e_mats['V'] = v_mat
        return s_mat, t_mat, v_mat
    #end def build_one_electron_matrices(self):


    def build_eri_4d_bf(self):
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

        return eri4d
    #def build_eri_4d_bf(self):


    def build_eri_4d_sh(self):
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

        return eri4d
    #end def build_eri_4d_sh(self):

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
    """ sparse build of one and two-electron integrals
    """
    def __init__(self, basis):
        super().__init__(basis)


# class IntegralEngineSparse():


class ScreenedIntegralEngine(IntegralEngineSparse):
    """ screened build of two-electron integrals
    """
    def __init__(self, basis, eri_screen_tol=1e-12):
        super().__init__(basis)
        self.eri_screen_tol = eri_screen_tol



if __name__ == "__main__":    # simple smoke test
    import numpy as np
    from pyquante22.atoms.molecule import Molecule, load_sample_molecule
    from pyquante22.basis.basisset import BasisSet
    np.set_printoptions(precision=2, suppress=False, linewidth=200)

    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )

    sample_molecules = ["Hydrogen", "Water", "Methane", "Propane"]
    sample_bases = ["sto-3g", "aug-cc-pvdz"]
    sample_list = [(sample_bases[0], mol) for mol in sample_molecules]
    sample_list += [(sample_bases[1], mol) for mol in sample_molecules[:2]]

    tensor_names = ['S', 'H', 'ERI']
    ref_tensor_path = "dev_scripts/tmp"
    rtol, atol = 1e-14, 1e-14

    for basis_name, sample_name in sample_list:
        # if basis_name != "aug-cc-pvdz" or sample_name != "Water":
        #     continue

        sample_tag = sample_name.replace(" ", "_")
        sample_tag += f"_{basis_name.replace('-', '')}"

        t0 = time.perf_counter()
        print(f"\n=== Sample: {sample_name}, Basis: {basis_name} ===")
        mol = load_sample_molecule(sample_name, aoi)
        basis = BasisSet(mol, name=basis_name)
        print(f"Using sample molecule: {mol} // {basis}")

        ie = IntegralEngineBase(basis)
        s, t, v = ie.build_one_electron_matrices()
        h = t + v
        my_tensors = {'S': s, 'H': h}
        eri4d = ie.build_eri_4d_sh()
        my_tensors = {'S': s, 'H': h, 'ERI': eri4d}
        t1 = time.perf_counter()
        print(f"Calcualted integrals in {t1-t0:.2f}s")

        ref_ao_label_path = os.path.join(ref_tensor_path, f"ao_labels_{sample_tag}.txt")
        with open(ref_ao_label_path, 'r') as ifd:
            ref_ao_labels = ifd.read().splitlines()

        bfs = basis.to_prims()
        assert len(bfs) == len(ref_ao_labels), "Mismatch in len of AO labels"

        for ii in range(len(bfs)):
            bfa, bfb = str(bfs[ii]), ref_ao_labels[ii]
            ama = cartPow2fullSym(bfs[ii].amom[0].tolist(), style='repeat')
            amb = bfb.split()[-1][1:]
            assert ama == amb, f"AO ordering mismatch, ours: {bfa}  // ref: {bfb}"
            # print(f"{ii:02d} :: {amb} // {ama}")

        name = 'S'
        sa = _to_numpy(my_tensors[name])
        sb = np.load(os.path.join(ref_tensor_path, f"{name}_{sample_tag}.npy"))
        sn = 1/np.sqrt(np.diag(sb))

        for name in my_tensors.keys():
            tensor = _to_numpy(my_tensors[name])
            ref_tensor = np.load(os.path.join(ref_tensor_path, f"{name}_{sample_tag}.npy"))
            if ref_tensor.ndim == 2:
                ref_orig = ref_tensor
                ref_tensor = np.einsum('i,ij,j->ij', sn, ref_tensor, sn)
            elif ref_tensor.ndim == 4:
                ref_orig = ref_tensor
                ref_tensor = np.einsum('ijkl,i,j,k,l->ijkl', ref_tensor, *(sn,)*4)
            
            debug_var = 1

            if np.allclose(tensor, ref_tensor, rtol=rtol, atol=atol):
                print(f"{name} matches reference.")
            else:
                a, b = tensor, ref_tensor
                amask = np.abs(a) > atol
                bmask = np.abs(b) > atol
                azero_diffs = np.where((amask & ~bmask))
                bzero_diffs = np.where((bmask & ~amask))
                if len(azero_diffs[0]) > 0:
                    print(f"{name} has {len(azero_diffs[0])} elements nonzero in tensor but zero in reference.")
                if len(bzero_diffs[0]) > 0:
                    print(f"{name} has {len(bzero_diffs[0])} elements zero in tensor but nonzero in reference.")

                diffs = a - b
                denom = np.where(np.abs(b) > 0, b, 1.0)
                pfull = diffs / denom
                pfull[~bmask] = 0.0
                idmax = np.abs(pfull).argmax().item()
                idfull = tuple(int(i) for i in np.unravel_index(idmax, b.shape))
                pmax = pfull[idfull].item()
                amax = a[idfull].item()
                bmax = b[idfull].item()
                print(f"{name} mismach: max%err: {pmax:.4e} @{idfull}, a: {amax:.4e}, b: {bmax:.4e}")

                debug_var = 1
        
        nbf, nsh, ranges = basis.nbf, basis.nsh, basis.shell_range
            

    debug_var = 1
