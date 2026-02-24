import os
import itertools
import math
import numpy as np
from importlib import resources
from pyquante22.utils.elements import element_symbols
from pyquante22.utils.file_utils import find_closest_file_match, load_gbs_file
from pyquante22.basis.cart_gto_utils import am2sym, sym2am, gen_cartesian_momentums
from pyquante22.basis.cart_gto import CartGtoPrim, CartGtoPrimPair
import pyquante22.array_ops as arr

# example basis download link from BSE
# http://www.basissetexchange.org/api/basis/sto-3g/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
# http://www.basissetexchange.org/api/basis/cc-pvdz/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
# http://www.basissetexchange.org/basis/cc-pvdz/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36


def _gto_norm(l, expnt):
    """Primitive radial normalization for g=r^l exp(-alpha r^2)."""
    n, alpha = l * 2 + 2, 2 * expnt
    n1 = 0.5 * (n + 1)
    # \int_0^\infty x^n exp(-alpha x^2) dx
    gauss_int = math.gamma(n1) / (2.0 * (alpha ** n1))
    return 1.0 / math.sqrt(gauss_int)


def _normalize_contracted_coeffs(l, exps, coeffs):
    """Normalize contraction coefficients (PySCF-style).

    Applies the post-contraction normalization factor after primitive norms.
    """
    exps = np.asarray(exps, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    norms = np.array([_gto_norm(l, e) for e in exps], dtype=float)
    cs = coeffs * norms
    ee = exps.reshape(-1, 1) + exps.reshape(1, -1)
    pref = math.gamma(l + 1.5) / 2.0
    ee = pref * np.power(ee, -(l + 1.5))
    val = float(cs @ ee @ cs)
    s1 = 1.0 / math.sqrt(val)
    return coeffs * s1


class CartGtoCont():
    """Contracted Cartesian GTO basis function"""
    def __init__(self, amom, cen, alphas, coeffs, ao=None):
        self.amom = amom
        self.cen = cen
        self.alphas = alphas
        self.coeffs = coeffs
        self.ao = ao or arr.get_default_ops()
        self.bfs_ = None

    def __len__(self):
        return len(self.alphas)

    def __repr__(self):
        ll = ",".join(f"{int(v):d}" for v in self.amom)
        xx = ",".join(f"{float(v):.4f}" for v in self.cen)
        cnt = len(self)
        return f"({ll})@<{xx}>#{cnt}"

    def __getitem__(self, idx):
        return CartGtoPrim(self.amom, self.cen, self.alphas[idx], self.coeffs[idx])

    def to_prim(self):
        """stack the basis functions into a batch of CartGtoPrim
        """
        if self.bfs_ is None:
            sz = len(self) # batch dim
            amoms = self.ao.tile(self.amom, (sz,1)) # repeat for primitives
            cens = self.ao.tile(self.cen, (sz,1)) # repeat for primitives
            self.bfs_ = CartGtoPrim(amom=amoms, cen=cens, alpha=self.alphas, 
                            coeffs=self.coeffs, ao=self.ao) #
        return self.bfs_
#end class ContractedCartGto():


class CartGtoContPair():
    """make this auto-handle contractions and coefficients"""
    def __init__(self, ga, gb):
        # ga, gb are CartGtoPrims
        self.gab = ga * gb
        self.cab = ga.coeffs, gb.coeffs
        self.ao = ga.ao

    def __str__(self):
        return str(self.gab)
    def __repr__(self):
        return "CartGtoContPair:\n" + repr(self.gab)

    def calc_overlap(self):
        prim_vals = self.gab.calc_overlap()
        return self.ao.einsum('ij,i,j', prim_vals, *self.cab)

    def calc_kinetic(self):
        prim_vals = self.gab.calc_kinetic()
        return self.ao.einsum('ij,i,j', prim_vals, *self.cab)

    def calc_nai(self, rcen):
        prim_vals = self.gab.calc_kinetic(rcen)
        return self.ao.einsum('ijk,i,j->k', prim_vals, *self.cab)

    def calc_eri(self, other):
        prim_vals = self.gab | other.gab
        return self.ao.einsum('ijkl,i,j,k,l', prim_vals, *self.cab, *other.cab)

    def __or__(self, other):
        return self.calc_eri(other)
 

class CartShell():
    """Cartesian Shell"""
    def __init__(self, amtot, cen, alphas, coeffs, ao=None):
        self.amtot = amtot
        self.cen = cen
        self.alphas = alphas
        self.coeffs = coeffs
        self.ao = ao or arr.get_default_ops()
        self.bfs_ = None

    def __len__(self): # nbf * npf (all possible primitives)
        return self.nbf * self.npf

    @property
    def nbf(self): # num basis funcs
        return ((self.amtot + 1) * (self.amtot + 2)) // 2

    @property
    def npf(self): # num primitive funcs (per basis func)
        return len(self.alphas)

    def __mul__(self, other):
        return CartShellPair(self, other)

    def __repr__(self): # @TODO: make this more descriptive, this should be __str__
        scen = ','.join(f"{float(v):.2f}" for v in self.cen)
        return f"{am2sym[self.amtot]}-shell@<{scen}>#{self.nbf}({self.npf})"
    
    def __getitem__(self, idx): # not efficient, just a helper
        return self.bfs[idx]

    def to_prims(self):
        am_list = gen_cartesian_momentums(self.amtot)
        ao = self.ao
        prims = []
        for am in am_list:
            ams = ao.arr(am, dtype=ao.int_dtype).reshape(1,3)
            amoms = ao.tile(ams, (1, self.npf)).reshape(-1,3)
            cens = ao.tile(self.cen, (self.npf, 1))
            prims.append(CartGtoPrim(amom=amoms, cen=cens, 
                alpha=self.alphas, coeffs=self.coeffs, ao=ao))
        return prims
         
    @property
    def bfs(self):
        """all primitives in this shell as one CartGtoPrim"""
        if self.bfs_ is None:
            sz = len(self) # nbf * npf
            am_list = gen_cartesian_momentums(self.amtot)
            amoms = self.ao.arr(am_list, dtype=self.ao.int_dtype)
            # amom is the slow index, necessary for later contraction
            amoms = self.ao.tile(amoms, (1,self.npf)).reshape(-1,3) 
            cens = self.ao.tile(self.cen, (sz,1)) # repeat for prims x amoms
            alphas = self.ao.tile(self.alphas, (self.nbf,1)).reshape(-1)
            coeffs = self.ao.tile(self.coeffs, (self.nbf,1)).reshape(-1)
            self.bfs_ = CartGtoPrim(amom=amoms, cen=cens, alpha=alphas, 
                                coeffs=coeffs, ao=self.ao)
        return self.bfs_
#end class CartShell():


class CartShellBatch():
    """Cartesian Shell batch"""
    def __init__(self, shells=[]):
        self.shells = shells # List of CartShell
        self.ao = shells[0].ao if len(shells) > 0 else arr.get_default_ops()

    def __len__(self): # nbf * npf (all possible primitives)
        return sum([len(s) for s in self.shells])

    @property
    def nsh(self): # num basis funcs
        return len(self.shells)

    def __mul__(self, other):
        return CartShellPair(self, other)

    def __repr__(self): # 
        return ' + '.join(s for s in self.shells)
    
    def __getitem__(self, idx): # not efficient, just a helper
        return self.bfs[idx]

    def __str__(self):
        return " ".join([str(sh) for sh in self.shells])
    def __repr__(self):
        return "\n".join([repr(sh) for sh in self.shells])

    @property
    def bfs(self):
        """all primitives in this shell as one CartGtoPrim"""
        if all(sh.bfs_ is None for sh in self.shells): # none instantiated, build from scratch
            ao = self.ao
            sz = len(self)
            amoms = ao.zeros((sz, 3), dtype=ao.int_dtype)
            cens = ao.zeros((sz, 3))
            alphas = ao.zeros((sz,))
            coeffs = ao.zeros((sz,))
            icnt = 0
            am_tots = set([sh.amtot for sh in self.shells])
            am_lists = {amt:gen_cartesian_momentums(amt) for amt in am_tots}
            for sh in self.shells:
                sz = len(sh)
                ams = ao.arr(am_lists[sh.amtot], dtype=ao.int_dtype)
                amoms[icnt:icnt+sz] = ao.tile(ams, (1,sh.npf)).reshape(-1,3)
                cens[icnt:icnt+sz] = ao.tile(sh.cen, (sz, 1))
                alphas[icnt:icnt+sz] = ao.tile(sh.alphas, (sh.nbf,1)).reshape(-1)
                coeffs[icnt:icnt+sz] = ao.tile(sh.coeffs, (sh.nbf,1)).reshape(-1)

            return CartGtoPrim(amom=amoms, cen=cens, alpha=alphas, 
                                coeffs=coeffs, ao=self.ao)

        # some are instantiated, lazy instantiate all, concat with CartGtoPrim.__add__
        return sum([sh.bfs for sh in self.shells])

#end class CartShell():




class CartShellPair():
    """Pair of Cartesian shells"""
    def __init__(self, shell1, shell2): # shells are CartShell or CartShellBatch
        self.si = shell1
        self.sj = shell2
        self.ao = shell1.ao
        assert self.ao is shell2.ao, "shells must have same ArrayOps backend"
        self.bfs_ = None

    def __len__(self):
        return len(self.si) * len(self.sj)

    @property
    def shape(self):
        return 

    def __repr__(self):
        stra = str(self.si)
        strb = str(self.sj)
        return f"ShellPair({stra} X {strb})"

    @property
    def bfs(self):
        """all primitive pairs in this shell pair as one CartGtoPrimPair"""
        if self.bfs_ is None:
            gi, gj = self.si.bfs, self.sj.bfs
            self.bfs_ = CartGtoPrimPair(gi, gj)
        return self.bfs_


    def __or__(self, other):
        return self.calc_eri(other)


    def calc_eri(self, other):
        if isinstance(self.si, CartShell) and \
            isinstance(self.sj, CartShell) and \
            isinstance(other.si, CartShell) and \
            isinstance(other.sj, CartShell):
            return self.calc_eri_pure(other)

        return self.calc_eri_ragged(other)


    def calc_eri_pure(self, other):
        eri_prims = self.bfs | other.bfs # OR attempts calc_eri_self first
        si, sj, sk, sl = self.si, self.sj, other.si, other.sj
        prim_lens = (si.npf, sj.npf, sk.npf, sl.npf)
        gi, gj, gk, gl = self.bfs.ga, self.bfs.gb, other.bfs.ga, other.bfs.gb
        coeffs = gi.coeffs, gj.coeffs, gk.coeffs, gl.coeffs
        # coeffs = si.coeffs, sj.coeffs, sk.coeffs, sl.coeffs
        # we could handle either the gi.coeffs (repeated) or the si.coeffs (one per pim)
        # but using si.coeffs wouldn't have the norms pre-applied
        eris = self.eri_contraction_pure(eri_prims, prim_lens, 
                                         coeffs=coeffs, ao=self.si.ao)

        return eris


    def calc_eri_ragged(self, other):
        """General ERI contraction that supports CartShellBatch on any axis."""
        eri_prims = self.bfs | other.bfs
        gi, gj, gk, gl = self.bfs.ga, self.bfs.gb, other.bfs.ga, other.bfs.gb
        coeffs = gi.coeffs, gj.coeffs, gk.coeffs, gl.coeffs
        axis_counts = (
            self._axis_counts(self.si),
            self._axis_counts(self.sj),
            self._axis_counts(other.si),
            self._axis_counts(other.sj),
        )
        return self.eri_contraction_ragged(
            eri_prims,
            axis_counts=axis_counts,
            coeffs=coeffs,
            ao=self.si.ao,
        )


    @staticmethod
    def eri_contraction_pure(eris, axis_strides, coeffs=None, ao=None):
        """Contraction of ERIs from primitive to contracted basis
        works when all contracted basis have same number of primitives per axis
        eri: 4d array of primitive ERIs
        axis_strides: num primitives per axis
        coeffs: (optional) list of contraction coeffs per axis
        return: 4d array of contracted ERIs
        """
        ao = ao or arr.get_default_ops()
        assert isinstance(eris, ao.ArrayType), "eris must be ArrayType"
        assert all([s%a == 0 for s,a in zip(eris.shape, axis_strides)]), \
            "axis_strides must divide eris.shape"
        angm_lens = [s//a for s,a in zip(eris.shape, axis_strides)]
        new_shape = sum(list(zip(angm_lens, axis_strides)), ())
        eri_tmp = eris.reshape(new_shape)
        if coeffs is None:
            eri_cont = ao.sum(eri_tmp, axis=(1,3,5,7))
        else:
            assert all([len(c) == es for c, es in zip(coeffs, eris.shape)]), \
                "eri coeffs shape mismatch"
            coeffs  = [c.reshape(-1, pl) for c,pl in zip(coeffs, axis_strides)]
            eri_cont = ao.einsum('iajbkcld,ia,jb,kc,ld->ijkl', eri_tmp,
                            coeffs[0], coeffs[1], coeffs[2], coeffs[3])
        return eri_cont
    # end def eri_contraction_pure(eris, axis_strides, coeffs, ao=None):


    @staticmethod
    def _axis_counts(shell):
        """Primitive segment lengths per contracted basis function."""
        if isinstance(shell, CartShell):
            return [shell.npf] * shell.nbf
        if isinstance(shell, CartShellBatch):
            counts = []
            for sh in shell.shells:
                counts.extend([sh.npf] * sh.nbf)
            return counts
        raise TypeError(f"Unsupported shell type: {type(shell)}")


    @staticmethod
    def _segment_sum_fallback(x, lengths, axis, ao):
        """Backend-agnostic segment sum using slicing + stack."""
        if len(lengths) == 0:
            raise ValueError("lengths cannot be empty")
        outs = []
        start = 0
        for seg in lengths:
            stop = start + int(seg)
            sl = [slice(None)] * x.ndim
            sl[axis] = slice(start, stop)
            outs.append(ao.sum(x[tuple(sl)], axis=axis))
            start = stop
        return ao.stack(outs, axis=axis)


    @classmethod
    def eri_contraction_ragged(cls, x, axis_counts, coeffs=None, ao=None):
        """Contraction of ERIs with non-uniform primitive strides per axis."""
        ao = ao or arr.get_default_ops()
        assert isinstance(x, ao.ArrayType), "eris must be ArrayType"
        assert len(axis_counts) == x.ndim
        if coeffs is not None:
            assert len(coeffs) == x.ndim
        for d in range(x.ndim):
            assert sum(axis_counts[d]) == x.shape[d]
            if coeffs is not None:
                assert coeffs[d].shape[0] == x.shape[d]

        dim_ratios = [(d, x.shape[d] // len(axis_counts[d])) for d in range(x.ndim)]
        dim_ratios.sort(key=lambda t: t[1], reverse=True)

        for dim, _ in dim_ratios:
            if coeffs is not None:
                c = coeffs[dim]
                shape = [1] * x.ndim
                shape[dim] = c.shape[0]
                x = x * c.reshape(shape)
            if x.shape[dim] != len(axis_counts[dim]):
                try:
                    x = ao.segment_sum(x, axis_counts[dim], axis=dim)
                except Exception:
                    x = cls._segment_sum_fallback(x, axis_counts[dim], axis=dim, ao=ao)

        return x


#end class CartShellPair():

class BasisSet():
    def __init__(self, molecule, name='sto-3g', contraction_norm=True):
        self.molecule = molecule
        # self.bfs = []
        self.shells = []
        self.shell_range = []
        self.ao = molecule.ao
        ao = self.ao

        basis_dir = resources.files("pyquante22.basis").joinpath("cart_gto_data")
        basis_file = find_closest_file_match(basis_dir, f"{name}.gbs")
        if basis_file is not None:
            basis_data = load_gbs_file(basis_file)
        else:
            # @TODO: try to download from the basis set exchange
            raise ValueError(f"Cannot find the {name} basis set in {basis_dir}")

        self.name = os.path.basename(basis_file)[:-4]
        if self.name.lower() != name.lower():
            print(f"::WARN:: exact match for {name} not found, using {self.name} instead")
        
        self.nbf = 0
        for atom in molecule.atoms:
            cen = ao.arr(atom.r)
            for sym, prims in basis_data[element_symbols[atom.atno]]:
                amtot = sym2am[sym]
                prims = np.asarray(prims, dtype=float)
                alphas = ao.arr(prims[:, 0])
                if contraction_norm:
                    coeffs = ao.arr(_normalize_contracted_coeffs(
                        amtot, prims[:, 0], prims[:, 1]
                    ))
                else:
                    coeffs = ao.arr(prims[:,1])
                curr_shell = CartShell(amtot, cen, alphas ,coeffs, ao=ao)
                self.shells.append(curr_shell)
                self.shell_range.append(
                    slice(self.nbf, self.nbf + curr_shell.nbf))
                self.nbf += curr_shell.nbf
    #end def __init__(self, molecule, name='sto-3g', **kwargs):

    def __getitem__(self,idx):
        # inefficient, only use for debugging
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.nbf)
            return [self[i] for i in range(start, stop, step)]

        if not isinstance(idx, (int, np.integer)):
            raise TypeError("BasisSet index must be int or slice")

        if idx < 0:
            idx += self.nbf
        if idx < 0 or idx >= self.nbf:
            raise IndexError("BasisSet index out of range")

        for sh, rng in zip(self.shells, self.shell_range):
            if rng.start <= idx < rng.stop:
                local_idx = idx - rng.start
                return sh.bfs[local_idx]

        raise RuntimeError("Failed to map basis index to a shell")

    @property
    def bfs(self): # inefficient, only use for debugging
        return sum([sh.bfs for sh in self.shells])

    def to_prims(self):
        return sum([sh.to_prims() for sh in self.shells], [])
    def __repr__(self): 
        return f"{self.name} for {self.molecule.hill_formula()} #{self.nsh}({self.nbf})[{self.size}]"
    def __len__(self): return self.nbf
    @property # number of shells
    def nsh(self): return len(self.shells)
    @property # all possible primitives
    def size(self): return sum(len(s) for s in self.shells)
    # def shell_bfs(self, idx):
    #     return self.bfs[self.shell_range[idx]]
    # def atom_symbols(self,geo):
    #     return [bf.atom_symbol(geo) for bf in self.bfs]
#end class BasisSet():
        

# step 1: find all prim eris between contracted basis ijkl
# step 2: find all prims between shell pairs ijkl

def eri_contraction(eris, axis_counts, ao=None):
    """Contraction of ERIs from primitive to contracted basis
    works for the general case of not all contracted basis having same
    number of primitives in an axis
    eri: 4d array of primitive ERIs
    axis_counts: list per axis (4,) of list of primitive sizes per basis func
    return: 4d array of contracted ERIs
    """
    ao = ao or arr.get_default_ops()
    assert isinstance(eris, ao.ArrayType), "eris must be ao.ArrayType"
    assert len(axis_counts) == eris.ndim
    assert all([sum(c) == s for s,c in zip(eris.shape, axis_counts)]), \
        "axis_counts sums must match eris.shape"

    x = eris
    dim_ratios = [(i,s//len(c)) for i,(c,s) in enumerate(zip(axis_counts, eris.shape)) 
                  if s != len(c)] # skip dims that don't need contraction
    dim_ratios.sort(key=lambda t: t[1], reverse=True) # largest first for efficiency
    for dim, ratio in dim_ratios:
        x = ao.segment_sum(x, axis_counts[dim], axis=dim)
    return x
# end def eri_contraction(eris, axis_counts, coeffs, ao=None):


def eri_contraction_weighted(eris, axis_counts, coeffs=None, ao=None):
    """
    eris: primitive ERIs with ragged packing along each primitive axis
    axis_counts: list of 4 lists; axis_counts[d] has n_contracted[d] segment lengths
    coeffs: list of 4 1D arrays; coeffs[d] length == eris.shape[d]
                coefficients aligned with the primitive packing of that axis
    returns: contracted ERIs with shape (nA, nB, nC, nD)
    """
    ao = ao or arr.get_default_ops()
    assert isinstance(eris, ao.ArrayType), "eris must be ao.ArrayType"
    x = eris

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
            c = coeffs[dim]             # (axis_len,)
            shape = [1] * x.ndim
            shape[dim] = c.shape[0]
            x = x * c.reshape(shape)                     # broadcast, no giant coeff tensor
        if x.shape[dim] != len(axis_counts[dim]):
            x = ao.segment_sum(x, axis_counts[dim], axis=dim)

    return x
#end def eri_contraction_weighted(eris, axis_counts, coeffs=None, ao=None):



if __name__ == "__main__":
    import time
    from pyquante22.atoms.molecule import Molecule
    from pyquante22.utils.file_utils import read_xyz_lines

    def nsymm(n):
        return n * (n+1) // 2
    def nsymm8(n):
        return nsymm(nsymm(n))

    aoi = arr.ArrayOps(
        "torch", device="cpu", default_dtype="float64", int_dtype="int64"
    )

    xyzs = "H 0 0 0\nF 1.4 0 0"
    # xyzs = "H 1 0 0\nH -1 0 0"
    # m1 = Molecule(xyzs)
    # b1 = BasisSet(m1, name='sto-3g')
    m1 = Molecule("./samples/glycine.sdf", ao=aoi)
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

    eri_prims = gij|gkl
    prim_lens = (si.npf, sj.npf, sk.npf, sl.npf)
    angm_lens = (si.nbf, sj.nbf, sk.nbf, sl.nbf)
    new_shape = sum(list(zip(angm_lens, prim_lens)), ())
    eri_tmp = eri_prims.reshape(new_shape)
    eri_cont = aoi.sum(eri_tmp, axis=(1,3,5,7))
    # eri2 = aoi.einsum('iajbkcld,a,b,c,d->ijkl', eri_prims,
    #                   si.coeffs, sj.coeffs, sk.coeffs, sl.coeffs)

    eri_contraction_pure = CartShellPair.eri_contraction_pure
    tmp = eri_contraction_pure(eri_prims, prim_lens, ao=aoi)

    axis_counts = [aoi.arr([npf,]*n, dtype=aoi.int_dtype) 
                   for npf,n in zip(prim_lens, angm_lens)]
    tmp2 = eri_contraction(eri_prims, axis_counts, ao=aoi)

    assert aoi.abs((tmp-tmp2)/tmp).max().item() < 5e-15, f"mismatch in eri contraction"

    sij, skl = si * sj, sk * sl
    tmp3 = sij | skl


    eri_coeffs = eri_prims  * si.bfs.coeffs[:,None,None,None] \
                            * sj.bfs.coeffs[None,:,None,None] \
                            * sk.bfs.coeffs[None,None,:,None] \
                            * sl.bfs.coeffs[None,None,None,:]

    coeffs = [si.bfs.coeffs, sj.bfs.coeffs, 
              sk.bfs.coeffs, sl.bfs.coeffs]
    # coeffs = [c.reshape(-1,pl) for c,pl in zip(coeffs,prim_lens)]
    tmp4 = eri_contraction_pure(eri_prims, prim_lens, coeffs=coeffs, ao=aoi)
    tmp5 = eri_contraction_pure(eri_coeffs, prim_lens, ao=aoi)

    assert aoi.abs((tmp4-tmp3)/tmp3).max().item() < 5e-15, f"mismatch in eri contraction2"
    assert aoi.abs((tmp5-tmp3)/tmp3).max().item() < 5e-15, f"mismatch in eri contraction2"

    tmp6 = eri_contraction(eri_coeffs, axis_counts, ao=aoi)
    tmp7 = eri_contraction_weighted(eri_prims, axis_counts, coeffs=coeffs, ao=aoi)

    assert aoi.abs((tmp6-tmp3)/tmp3).max().item() < 5e-15, f"mismatch in eri contraction2"
    assert aoi.abs((tmp7-tmp3)/tmp3).max().item() < 5e-15, f"mismatch in eri contraction2"

    t0 = time.time()
    Bj = build_j2_blocks(b1)
    t1 = time.time()
    print(f"Built screening J in {t1-t0:.2f}s")


    def five_number_summary(data):
        """ Compute min, 25th percentile, median, 75th percentile, max """
        q75, q50, q25 = np.percentile(data, [75, 50, 25])
        return np.stack([np.amin(data), q25, q50, q75, np.amax(data)])

    nsh = len(b1.shells)
    ti,tj = np.triu_indices(nsh)
    Bjt = Bj[tj,ti].numpy()
    assert len(Bjt) == (nsh*(nsh+1)//2)

    for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
        mask = Bjt > tol
        count = mask.sum()
        print(f"{tol:.2e} {count} {count/len(Bjt):.4f}")
