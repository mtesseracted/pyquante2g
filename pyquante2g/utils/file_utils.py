import os
import numpy as np # casts to array in load_gbs_file


def find_closest_file_match(path, filename, removes=' _.-'):
    """find the file that is the closest match to filename in path"""
    files = os.listdir(path)
    if filename in files: # exact match
        return os.path.join(path, filename)

    def remove_chars(word, removes=removes):
        remove_set = set(removes)
        return "".join(ch for ch in word if ch not in remove_set)

    # try with all lowercase and removed special chars
    ext = '.' + filename.split('.')[-1]
    name2 = remove_chars(filename.lower().replace(ext,''))
    files2 = [remove_chars(f.lower().replace(ext,'')) for f in files]
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


def parseline(line,format):
    """\
    Given a line (a string actually) and a short string telling
    how to format it, return a list of python objects that result.

    The format string maps words (as split by line.split()) into
    python code:
    x   ->    Nothing; skip this word
    s   ->    Return this word as a string
    i   ->    Return this word as an int
    d   ->    Return this word as an int
    f   ->    Return this word as a float

    Basic parsing of strings:
    >>> parseline('Hello, World','ss')
    ['Hello,', 'World']

    You can use 'x' to skip a record; you also don't have to parse
    every record:
    >>> parseline('1 2 3 4','xdd')
    [2, 3]

    >>> parseline('C1   0.0  0.0 0.0','sfff')
    ['C1', 0.0, 0.0, 0.0]
    """
    xlat = {'x':None,'s':str,'f':float,'d':int,'i':int}
    result = []
    words = line.split()
    for i in range(len(format)):
        f = format[i]
        try:
            trans = xlat.get(f,None)
            if trans: result.append(trans(words[i]))
        except Exception as err:
            print(f":: ERR :: i={i}, t={trans}, words = {words}")
    if len(result) == 0: return None
    if len(result) == 1: return result[0]
    return result
#end def parseline(line,format):


def read_xyz(fname):
    """ read .xyz format
    """
    f = open(fname)
    line = f.readline()
    nat = int(line.strip())
    comment = f.readline()
    lines = [f.readline() for i in range(nat)]
    return read_xyz_lines(lines)

def read_xyz_lines(lines):
    from pyquante2g.utils.elements import sym2no
    atuples = []
    for line in lines:
        if line.strip() == '': continue
        sym,x,y,z = parseline(line,'sfff')
        atno = sym2no[sym]
        atuples.append((atno,x,y,z))
    return atuples

def read_mol(fname):
    """ read .moG file
    @TODO: need to infer hydrogens with rdkit
    """
    f = open(fname)
    for _ in range(3): # 3 comments
        comment = f.readline()
    line = f.readline()
    nat = int(line.split()[0])
    lines = [f.readline() for i in range(nat)]
    return read_mol_atom_lines(lines)

def read_mol_atom_lines(lines):
    from pyquante2g.utils.elements import sym2no # 
    atuples = []
    for line in lines:
        x,y,z, sym = parseline(line,'fffs')
        atno = sym2no[sym]
        atuples.append((atno,x,y,z))
    return atuples


if __name__ == "__main__":
    import os
    from samples.samples import raw_samples
    # from molecule import Molecule # circular import

    for raw_samp in raw_samples:
        atoms = read_xyz_lines(raw_samp.pop('xyz').splitlines())
        # mol = Molecule(atoms, **raw_samp)
        # print(f"{mol}")

    samp_files = [f for f in os.listdir('samples') if f.endswith(('.xyz', '.mol', '.sdf'))]
    for f in samp_files:
        if f.startswith('alum'):
            debug_var = 1
        # mol = Molecule.from_file(os.path.join('samples', f))
        # mol.name = f
        # print(f"{mol}")


    debug_var = 1
