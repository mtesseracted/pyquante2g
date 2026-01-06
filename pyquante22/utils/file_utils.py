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
    from pyquante22.utils.elements import sym2no
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
    from pyquante22.utils.elements import sym2no # 
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
