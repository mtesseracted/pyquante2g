import numpy as np
# import basis.data as data
# from .atoms.elements import symbol

# example basis download link from BSE
# http://www.basissetexchange.org/api/basis/sto-3g/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
# http://www.basissetexchange.org/api/basis/cc-pvdz/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
# http://www.basissetexchange.org/basis/cc-pvdz/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36

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
            print(f":: WARN :: ECP data found in {filename}, returning early")
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
                debug_var = 1
                raise ValueError("load_gbs_file :: dive into the debugger")
        except:
            debug_var = 1
            raise ValueError("load_gbs_file :: dive into the debugger")


    #end while

    # possibly didn't have last '****'
    if curr_element != '' and curr_element not in basis_dict:
        basis_dict[curr_element] = curr_gtos

    return basis_dict
#end def load_gbs_file(filename):






class BasisSet():
    def __init__(self, molecule, name='sto3d', **kwargs):
        self.bfs = []
        self.shells = []
        self.name = name.lower()
        # basis_data = data.basis[self.name]

        for atom in molecule.atoms:
            pass

# lstr = """\
#       0.1873113696D+02       0.3349460434D-01
#       0.2825394365D+01       0.2347269535D+00
#       0.6401216923D+00       0.8137573261D+00"""


import os
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

d31gebug_var = 1
