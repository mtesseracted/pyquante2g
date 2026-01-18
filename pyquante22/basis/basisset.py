import os
import itertools
import numpy as np
from importlib import resources
from pyquante22.utils.elements import element_symbols
from pyquante22.basis.cart_gto_utils import am2sym, sym2am, gen_cartesian_momentums
from pyquante22.basis.cart_gto import CartGtoPrim, CartGtoPrimPair

# example basis download link from BSE
# http://www.basissetexchange.org/api/basis/sto-3g/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
# http://www.basissetexchange.org/api/basis/cc-pvdz/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
# http://www.basissetexchange.org/basis/cc-pvdz/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36


def find_closest_file_match(path, filename):
    """find the file that is the closest match to filename in path"""
    files = os.listdir(path)
    if filename in files: # exact match
        return os.path.join(path, filename)

    def remove_chars(word, remove):
        return "".join(ch for ch in word if ch not in set(remove))

    # try with all lowercase and removed special chars
    remove = ' _.-'
    ext = '.' + filename.split('.')[-1]
    name2 = remove_chars(filename.lower().replace(ext,''), remove)
    files2 = [remove_chars(f.lower().replace(ext,''), remove) for f in files]
    for icut in range(4): # try cutting up to 3 letters off the end
        name3 = name2[:-icut] if icut > 0 else name2
        if name3 in files2:
            return os.path.join(path, files[files2.index(name2)])

    return None



def load_gbs_file(filename):
    """ returns a dict, with elements symbols as keys,
    with entries being a list of tuples with (amom, gto_data)
    where gto_data is an array of size (num_primitives, 2),
    and [:,0] are the primitive exponents and [:,1] are the primitive coeffs
    """
    with open(filename, 'r') as ifd:
        lines = ifd.readlines()

    def strip_non_alpha(s):
        return ''.join([ss for ss in s if ss.isalpha()])

    basis_dict = {}
    curr_element = ''
    curr_gtos = []
    icnt = 0
    while icnt < len(lines):
        line = lines[icnt]
        icnt += 1
        ls = line.strip()
        if ls == '' or ls[0] in ('#', '!'): continue
        if ls == '****':
            if len(curr_gtos) > 0 and curr_element != '':
                basis_dict[curr_element] = curr_gtos
            curr_element = ''
            curr_gtos = []
            continue
        elif 'ecp' in line.lower():
            print(f"::WARN:: ECP data found in {filename}, returning early")
            return basis_dict

        try:
            lss = ls.split()
            if curr_element == '':
                curr_element = strip_non_alpha(lss[0])
                ecp = int(lss[1]) if len(lss) > 1 else 0
                if ecp != 0:
                    raise ValueError("Only ECP=0 supported currently")
            elif lss[0].isalpha(): # orbital type
                am = strip_non_alpha(lss[0].upper())
                fac = float(lss[2]) if len(lss) > 2 else 1.0
                gcnt = int(lss[1])
                gto_lines = lines[icnt:icnt+gcnt]
                icnt += gcnt
                str_data = [[ll.upper().replace('D','E') for ll in l.split()] 
                            for l in gto_lines]
                gto_data = np.asarray(str_data, dtype=float)

                for ij, j in enumerate(am):
                    exps = gto_data[:, 0]
                    coeffs = gto_data[:, ij+1] * fac
                    curr_gtos.append((j, np.stack([exps, coeffs], axis=1)))
            else:
                raise ValueError("::ERR::(load_gbs_file) unknown line type or malformed file")
        except Exception as e:
            print("::ERR::(load_gbs_file) unhandled Exception, re-rasing")
            raise e
    #end while

    # possibly didn't have last '****'
    if curr_element != '' and curr_element not in basis_dict:
        basis_dict[curr_element] = curr_gtos

    return basis_dict
#end def load_gbs_file(filename):


def batch_expand(array, new_shape, flatten=True):
    """ expand the batch dimension of a flat array
    batch expand an array, assume it was 1D start, with spatial
    param: new_shape: multi-dim shape
    param: faltten: whether to flatten the return
    ! DOESNT WORK
    """
    if len(array.shape) != 1:
        raise ValueError("Only works for 1D arrays")
    rshape = [1 for _ in new_shape]
    rshape[new_shape.index(-1)] = len(array)

    array = ao.broadcast_to(array.reshape(rshape), new_shape)
    if flatten:
        array = array.reshape(-1)
    return array


class ContractedCartGto():
    def __init__(self, amom, cen, alphas, coeffs):
        self.amom = amom
        self.cen = cen
        self.alphas = alphas
        self.coeffs = coeffs

    def __len__(self):
        return len(self.alphas)

    def __repr__(self):
        ll = ",".join(f"{int(v):d}" for v in self.amom)
        xx = ",".join(f"{float(v):.4f}" for v in self.cen)
        cnt = len(self)
        return f"({ll})@<{xx}>#{cnt}"

    def __getitem__(self, idx):
        args = [self.amom, self.cen, ao.tile(self.alphas[idx],3)]
        return CartGtoPrim(*args)

    def to_prim(self, new_shape=None):
        """stack the basis functions into a batch of CartGtoPrim

        new_shape: unpacked shape (multi-dim), expands the batch dim
        """
        sz = len(self) # batch dim
        amoms = ao.tile(self.amom, sz) # repeat for primitives
        cens = ao.tile(self.cen, sz)
        alphas = ao.tile(self.alphas.reshape(-1,1), 3).reshape(-1) # 3d

        if new_shape is not None:
            rshape = [1 for _ in new_shape] + [3,]
            new_shape = new_shape + [3,]
            # amoms = batch_expand(amoms, new_shape)
            # cens = batch_expand(cens, new_shape)
            # alphas = batch_expand(alphas, new_shape)
            rshape[new_shape.index(-1)] = sz
            amoms = ao.broadcast_to(amoms.reshape(rshape), 
                new_shape).reshape(-1)
            cens = ao.broadcast_to(cens.reshape(rshape), 
                new_shape).reshape(-1)
            alphas = ao.broadcast_to(alphas.reshape(rshape), 
                new_shape).reshape(-1)

        return CartGtoPrim(amoms, cens, alphas)
#end class ContractedCartGto():
 

class CartShell():
    """Cartesian shell"""
    def __init__(self, amtot, cen, alphas, coeffs):
        self.amtot = amtot
        self.cen = cen
        self.alphas = alphas
        self.coeffs = coeffs

    def __len__(self): # nbf * npf (all possible primitives)
        return self.nbf * self.npf

    @property
    def nbf(self): # num basis funcs
        return ((self.amtot + 1) * (self.amtot + 2)) // 2

    @property
    def npf(self): # num primitive funcs (per basis func)
        return len(self.alphas)

    def __repr__(self):
        scen = ','.join(f"{float(v):.2f}" for v in self.cen)
        return f"{am2sym[self.amtot]}-shell @<{scen}> #{self.nbf}({self.npf})"

    @property
    def bfs(self):
        bfs = []
        r, a, c = self.cen, self.alphas, self.coeffs
        for amom in gen_cartesian_momentums(self.amtot):
            am = ao.arr(amom, dtype=ao.int_dtype)
            bfs.append(ContractedCartGto(am, r, a, c))
        return bfs
#end class CartShell():


class BasisSet():
    def __init__(self, molecule, name='sto-3g', **kwargs):
        self.molecule = molecule
        self.bfs = []
        self.shells = []
        self.shell_range = []

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
                alphas, coeffs = ao.arr(prims.T)
                amtot = sym2am[sym]
                curr_shell = CartShell(amtot, cen, alphas ,coeffs)
                self.shells.append(curr_shell)
                self.shell_range.append(
                    slice(self.nbf, self.nbf + curr_shell.nbf))
                self.nbf += curr_shell.nbf
    #end def __init__(self, molecule, name='sto-3g', **kwargs):

    def __getitem__(self,i): 
        # return self.bfs.__getitem__(i)
        raise NotImplementedError()

    def __repr__(self): 
        return f"{self.name} for {self.molecule.hill_formula()} #{self.nsh}({self.nbf})[{self.size}]"
    def __len__(self): return self.nbf
    @property # number of shells
    def nsh(self): return len(self.shells)
    @property # all possible primitives
    def size(self): return sum(len(s) for s in self.shells)
    def shell_bfs(self, idx):
        return self.bfs[self.shell_range[idx]]
    def atom_symbols(self,geo):
        return [bf.atom_symbol(geo) for bf in self.bfs]
#end class BasisSet():
        

import pyquante22.array_ops as arr
# get an instance of ArrayOps
ao = arr.ArrayOps(
    "torch", device="cpu", default_dtype="float64", int_dtype="int64"
)

def ndindex(shape):
    return itertools.product(*[range(n) for n in shape]) # generator

def eri_shell(shi:CartShell, shj:CartShell, shk:CartShell, shl:CartShell):
    bfis = shi.bfs
    bfjs = shj.bfs
    bfks = shk.bfs
    bfls = shl.bfs
    ret = ao.zeros((len(bfis), len(bfjs), len(bfks), len(bfls)))
    for idx in ndindex(ret.shape):
        bfi = bfis[idx[0]]
        bfj = bfjs[idx[1]]
        bfk = bfks[idx[2]]
        bfl = bfls[idx[3]]
        ret[idx] = eri_bf(bfi, bfj, bfk, bfl)

    return ret


def eri_bf(bfi, bfj, bfk, bfl): # type ContractedCartGto
    # unpack bfs into a CartGtoPrims
    bfs = [bfi, bfj, bfk, bfl]
    full_sz = [len(bf) for bf in bfs]
    cgps, coeffs = [None,]*4, [None,]*4

    for ibf, bf in enumerate(bfs):

        new_sz = [-1 if i == ibf else full_sz[i] for i in range(4)]
        cgps[ibf] = bf.to_prim(new_sz)

        new_sz = [-1 if i == ibf else full_sz[i] for i in range(4)]
        bcast = [full_sz[i] if ibf == i else 1 for i in range(4)]
        coeffs[ibf] = ao.broadcast_to(bf.coeffs.reshape(bcast), new_sz).reshape(-1)

    ab = cgps[0] * cgps[1] # 
    cd = cgps[2] * cgps[3]
    prim_eris = ab.calc_eri(cd)
    cs = coeffs[0] * coeffs[1] * coeffs[2] * coeffs[3]
    na, nb = ab.ga.norm, ab.gb.norm
    nc, nd = cd.ga.norm, cd.gb.norm
    ans = ao.einsum('i,i,i,i,i,i', na, nb, nc, nd, cs, prim_eris)
    return ans


def eri_bf_loop(bfi, bfj, bfk, bfl): # type ContractedCartGto
    # loop version for debugging
    eris = ao.zeros((len(bfi), len(bfj), len(bfk), len(bfl)))
    cs = ao.zeros((len(bfi), len(bfj), len(bfk), len(bfl)))
    ns = ao.zeros((len(bfi), len(bfj), len(bfk), len(bfl)))
    for idx in ndindex(eris.shape):
        bi = bfi[idx[0]]
        bj = bfj[idx[1]]
        bk = bfk[idx[2]]
        bl = bfl[idx[3]]
        eris[idx] = (bi*bj).calc_eri(bk*bl)
        cs[idx] = bfi.coeffs[idx[0]] * bfj.coeffs[idx[1]] * \
            bfk.coeffs[idx[2]] * bfl.coeffs[idx[3]]
        ns[idx] = bi.norm * bj.norm * bk.norm * bl.norm

    return ao.sum(cs * ns * eris)

def build_j2_blocks(basis:BasisSet, cutoff_tol=1e-10):
    # build the 2d J array max_{ij} (ij|ij) for each ij shell combo
    # @TODO: build an octree or barnes-hut model of the moleucle,
    # then only consider the interaction between atoms that are
    # within a radius in sparse form (is order linear time)
    nsh = len(basis.shells)
    B = ao.zeros((nsh,nsh))
    for i in range(len(basis.shells)):
        for j in range(i+1):
            bs = eri_shell(
                basis.shells[i], basis.shells[j],
                basis.shells[i], basis.shells[j],)
            B[i,j] = ao.amax(bs)

    bmax = ao.amax(B)
    debug_var = 1
    return B

    



def build_jk_blocks(basis:BasisSet):
    nbf = basis.nbf
    # Dictionary-of-blocks storage (block sparse)
    J_blocks = {}  # key: (i,j,k,l) -> ndarray (ni,nj,nk,nl)
    K_blocks = {}  # key: (i,k,j,l) -> ndarray (ni,nk,nj,nl) (permute for exchange)
    # J = ao.zeros((nbf,)*4)
    # K = ao.zeros((nbf,)*4)

    pair_list = []
    for i in range(len(basis.shells)):
        for j in range(i+1):  # i >= j
            pair_list.append((i,j,1.0))

    for p, (i, j, Bij) in enumerate(pair_list):
        for q in range(p+1):         # ensure (ij) >= (kl) in this pair_list ordering
            k, l, Bkl = pair_list[q]
            # G has shape (ni, nj, nk, nl)
            G = eri_shell(
                basis.shells[i], basis.shells[j],
                basis.shells[k], basis.shells[l],)
            
            J_blocks[(i, j, k, l)] = G
            K_blocks[(i, k, j, l)] = ao.transpose(G, axes=(0, 2, 1, 3))  # shape (ni, nk, nj, nl)

    return J_blocks, K_blocks


def fill_swap_symm(array, ids, vals):
    """fill (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)"""
    array[ids] = vals
    id2 = ids[1], ids[0], ids[2], ids[3]
    array[id2] = ao.transpose(vals, (1,0,2,3))
    id2 = ids[0], ids[1], ids[3], ids[2]
    array[id2] = ao.transpose(vals, (0,1,3,2))
    id2 = ids[1], ids[0], ids[3], ids[2]
    array[id2] = ao.transpose(vals, (1,0,3,2))

def fill_eri_symm(array, ids, vals):
    """fill (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk), repeat for (kl|ij)"""
    fill_swap_symm(array, ids, vals) # fill index swaps
    id2 = ids[2], ids[3], ids[0], ids[1]
    val2 = ao.transpose(vals, (2,3,0,1))
    fill_swap_symm(array, id2, val2) # fill pair swaps

def inflate_4d(block, shape, ranges):
    """inflate a 4d sparse block form (dict of blocks) into full size"""
    ret = ao.zeros(shape)
    for ijkl, vals in block.items():
        ids = tuple([ranges[i] for i in ijkl])
        fill_eri_symm(ret, ids, vals)
    return ret


if __name__ == "__main__":
    import time
    from pyquante22.atoms.molecule import Molecule
    from pyquante22.utils.file_utils import read_xyz_lines

    def nsymm(n):
        return n * (n+1) // 2
    def nsymm8(n):
        return nsymm(nsymm(n))

    xyzs = "F 1 0 0\nH -1 0 0"
    # xyzs = "H 1 0 0\nH -1 0 0"
    # m1 = Molecule(xyzs)
    # b1 = BasisSet(m1, name='sto-3g')
    m1 = Molecule("./samples/glycine.sdf")
    print(f"Using molecule: {str(m1)}")
    b1 = BasisSet(m1, name='sto-3g')
    print(f"with basis: {str(b1)} full J is {b1.nbf**4}, uniqs is {nsymm8(b1.nbf)}")

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


    t0 = time.time()
    Jb, Kb = build_jk_blocks(b1)
    t1 = time.time()
    print(f"Built JK blocks (nbf={b1.nbf}) in {t1-t0:.2f}s")
    if b1.nbf < 20:
        t0 = time.time()
        J = inflate_4d(Jb, [b1.nbf,]*4, b1.shell_range)
        t1 = time.time()
        print(f"Inflated J to 4d (nbf={b1.nbf}) in {t1-t0:.2f}s")
    debug_var = 1

    # gt = 0.7746059439198978 # from pyscf
    # ans = J[0,0,0,0].item()

    if False:
        basis_dir = 'basis/cart_gto_data'
        basis_files = [f for f in os.listdir(basis_dir) if f.endswith('.gbs')]

        file = 'sto-3g_h_he.gbs'
        tmp = load_gbs_file(os.path.join(basis_dir, file))

        file = '6-31g.gbs'
        tmp = load_gbs_file(os.path.join(basis_dir, file))

        file = 'mg3t.gbs'
        tmp = load_gbs_file(os.path.join(basis_dir, file))

        file = 'mg3.gbs'
        tmp = load_gbs_file(os.path.join(basis_dir, file))

        file = 'def2-tzvp.gbs'
        tmp = load_gbs_file(os.path.join(basis_dir, file))

    debug_var = 1

    # the problem: the current functionality needs repmats with 
    # exact batch matching between a,b,c, and d! quadruple stupid!
    # the way I wrote pdist2 before was to use advanced indexing
    # to get repeats when iterating over pairs to calc distance.
    # so in order to reproduce that we need to support advanced
    # indexing on ContrCartBasis and/or CartShell
    # - note: in the PySEC code I use a meshgrid to generate indices,
    # for this project I want to make it an iterator instead of
    # actually allocating all indices into memory at once
    # - also, we should make basis func pairs |ab), and reuse them
    # when possible
