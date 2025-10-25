import os
import numpy as np
from collections import Counter

# local imports
from elements import element_symbols, element_masses, element_radii, element_colors
from utils.file_utils import read_xyz, read_mol, read_xyz_lines

# shorthands for array / linear alg ops so they can be changed easily
# try:
#     import torch as ch
#     arr_dtype = ch.float64
#     arr_init = ch.as_tensor
#     arr_stack = ch.stack
#     arange = ch.arange
#     la_norm = ch.linalg.norm
#     arr_min = lambda arr, axis=None: ch.amin(arr, dim=axis)
#     arr_max = lambda arr, axis=None: ch.amax(arr, dim=axis)
#     numpy = lambda arr: arr if isinstance(arr, np.ndarray) else arr.numpy()
#     arr2list = lambda arr: numpy(arr).tolist()
#     combos = ch.combinations
# except:
if True:
    from itertools import combinations
    arr_dtype = float
    arr_init = np.asarray
    arr_stack = np.stack
    arange = np.arange
    la_norm = np.linalg.norm
    arr_min = lambda arr, axis=None: np.amin(arr, axis=axis)
    arr_max = lambda arr, axis=None: np.amax(arr, axis=axis)
    numpy = lambda arr: arr if isinstance(arr, np.ndarray) else arr.numpy()
    arr2list = lambda arr: numpy(arr).tolist()
    combos = combinations


class Atom():
    """ Stores position info for an atom and has wrappers for properties 
    xyz position is assumed to be in bohr
    """

    def __init__(self, atno, x, y, z, **kwargs):
        self.atno = int(round(atno))
        self.Z = atno
        self.r = arr_init([x,y,z], dtype=arr_dtype)
        return

    def atuple(self): return (self.atno,*self.r.tolist())

    def __repr__(self): return "%d %s %12.6f %12.6f %12.6f" % (
        self.atno,element_symbols[self.atno],*arr2list(self.r))
    
    def __getitem__(self, i): return self.r[i]

    def mass(self): return element_masses[self.atno]
    def radius(self): return element_radii[self.atno]
    def symbol(self): return element_symbols[self.atno]
    def color(self): return element_colors[self.atno]
    def distance(self,other): return la_norm(self.r-other.r)
#end class Atom():


class Molecule():
    """ 
    """
    _default_charge = 0
    _default_dist_units = 'bohr'
    _bohr2ang = 0.529177249  # Conversion of length from bohr to angstrom
    _ang2bohr = 1.0 / _bohr2ang

    def __init__(self, data,**kwargs):
        if isinstance(data, list):
            self._init_from_atoms(data, **kwargs)
        elif isinstance(data, str):
            self._init_from_file(data, **kwargs)
        elif isinstance(data, dict):
            self._init_from_dict(data, **kwargs)
        else:
            raise TypeError("Unsupported init type")

    def _init_from_atoms(self,atomlist=[],**kwargs):
        self.atoms = []
        self.charge = int(kwargs.get('charge', self._default_charge))
        self.multiplicity = kwargs.get('multiplicity')
        self.name = kwargs.get('name','pyquante22 molecule')

        self.units = kwargs.get('units', self._default_dist_units).lower()
        assert self.units[:4] in ['bohr','angs']

        if atomlist:
            for atuple in atomlist:
                atmp = Atom(*atuple)
                if not self.units == 'bohr':
                    atmp.r *= self._ang2bohr
                self.atoms.append(atmp)
        if self.multiplicity is None:
            self.set_multiplicity()
    #end def _init_from_atoms(self,atomlist=[],**kwargs):

    def _init_from_dict(self, dic):

        xyzs = dic.get('xyz', [])
        if isinstance(xyzs, str):
            xyzs = xyzs.splitlines()
        xyzs = [s for s in xyzs if s.strip()]
        if len(xyzs) < 1:
            atoms = []
        else:
            atoms = read_xyz_lines(xyzs)

        self._init_from_atoms(atoms, **{k:v for k,v in dic.items() if k != 'xyz'})


    def _init_from_file(self, fname, format=None):
        if format == None:
            format = fname.split('.')[-1]

        if format == 'xyz':
            atuples = read_xyz(fname)
        elif format in ('mol', 'sdf'):
            atuples = read_mol(fname)
        else:
            raise ValueError(f"Unrecognized file format: {fname}")

        self._init_from_atoms(atuples, units='angstrom', name=os.path.basename(fname))

    def __getitem__(self,i): return self.atoms.__getitem__(i)
    def __len__(self): return self.atoms.__len__()

    def nel(self):
        """ Number of electrons of the molecule """
        return sum(atom.atno for atom in self) - self.charge
    
    def nocc(self): return sum(divmod(self.nel(),2))
    def nclosed(self):
        nc,ierr = divmod(self.nel()-self.nopen(),2)
        assert ierr == 0, "Error in molecule, multiplicity, nclosed, nopen"
        return nc
    def nopen(self): return self.multiplicity-1
    def nup(self): return self.nclosed()+self.nopen()
    def ndown(self): return self.nclosed()
    def mass(self): return sum(at.mass() for at in self.atoms)
    def com(self): return sum(at.mass()*at.r for at in self.atoms)/self.mass()

    def set_multiplicity(self):
        if self.nel() % 2:
            self.multiplicity = 2
        else:
            self.multiplicity = 1
        return

    def hill_formula(self):
        elements = [element_symbols[at.atno] for at in self.atoms]
        counts = {el:str(cnt) if cnt != 1 else '' 
                  for el, cnt in Counter(elements).items()}
        order = [] # Hill system ordering: C (then H if C), then alphabetical
        if "C" in counts:
            order.append("C")
            if "H" in counts: order.append("H")
            order += sorted(e for e in counts if e not in ("C", "H"))
        else:
            order = sorted(counts)

        return "".join(f"{el}{counts[el]}" for el in order)

    def __repr__(self):
        mstr  = f"Name: {self.name} Formula: {self.hill_formula()}"
        mstr += f" Charge: {self.charge} Multiplicity: {self.multiplicity}"
        return mstr

    def zero_center(self):
        r_com = self.com()
        for at in self.atoms:
            at.r = at.r - r_com
        return self

    def atom_coords(self):
        return arr_stack([at.r for at in self.atoms])
    def atom_symbols(self):
        return [at.symbol() for at in self.atoms]
    def atom_colors(self):
        return [at.color() for at in self.atoms]


    def bbox(self, padding=2.,):
        atom_coords = self.atom_coords()
        mins = arr_min(atom_coords, axis=0) - padding
        maxs = arr_max(atom_coords, axis=0) + padding
        return arr_stack([mins, maxs]).T

    def approx_bonds(self,scale=1.1):
        """ find bonds for inter-atom distance < sum of Van der Walls radii """
        bonds = []
        for i,ati in enumerate(self.atoms):
            for j in range(i):
                atj = self.atoms[j]
                if ati.distance(atj) < scale*(ati.radius()+atj.radius()):
                    bonds.append((i,j))
        return bonds

    def nuclear_repulsion(self):
        ats = self.atoms
        if len(ats) < 2: return arr_init(0.0)
        return arr_stack([
            ats[ii].Z * ats[jj].Z / ats[ii].distance(ats[jj]) 
            for ii, jj in combos(arange(len(self)), 2)
        ]).sum()

    def xyz(self,title=None,fobj=None):
        """ output in xyz format, http://en.wikipedia.org/wiki/XYZ_file_format """
        if title is None:
            title = self.name
        lines = ["%d" % len(self.atoms),"%s" % title]
        for atom in self.atoms:
            fstr = "{:4d} " + " ".join("{:12.6f}")
            axyz = arr2list(atom.r * self._bohr2ang)
            lines.append(fstr.format(element_symbols[atom.atno], *axyz))
        record = "\n".join(lines)
        if fobj:
            fobj.write(record+'\n')
        else:
            print(record)
        return


    def _repr_html_(self, tablehead=True):
        import xml.etree.ElementTree as ET
        from xml.dom.minidom import parseString

        def html_row(table, atom, i=0):
            """ Add a row in a table for an html representation of an atom """
            tr = ET.SubElement(table,"tr")
            xyz = arr2list(atom.r)
            vals = [str(i+1), str(atom.atno), *[f"{r:.6f}" for r in xyz]]
            for val in vals:
                td = ET.SubElement(tr,"td")
                td.text = val
            return

        html_root = ET.Element("html") # Create the HTML root
        top = ET.Element("p")         # Create the content
        h2 = ET.SubElement(top, "h2")
        h2.text = self.name
        p = ET.SubElement(top, "p")
        p.text = "Formula = %s, Charge = %d, Multiplicity = %d" % (
            self.hill_formula(), self.charge, self.multiplicity
        )

        table = ET.SubElement(top, "table")
        if tablehead:
            tr = ET.SubElement(table, "tr")
            for item in ["#", "Atno", "Symbol", "x", "y", "z"]:
                th = ET.SubElement(tr, "th")
                th.text = item

        for i, atom in enumerate(self.atoms):
            html_row(table, atom, i)

        html_root.append(top) # Append to HTML root and return string
        html_str = ET.tostring(html_root, encoding='unicode')
        return parseString(html_str).toprettyxml(indent='  ')
    #end def _repr_html_(self, tablehead=True):

#end class Molecule():


if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from samples.samples import raw_samples, taxol_raw, raw_drugs
    from utils.file_utils import read_xyz_lines
    from utils.plotting_utils import plot_molecule_3d, plot_molecule_3d_scatter

    samp_files = [os.path.join('samples',f) 
                  for  f in os.listdir('samples') 
                  if f.endswith(('.xyz', '.mol', '.sdf'))]
    molecules = [Molecule(samp) for samp in raw_samples + samp_files]

    for mol in molecules:
        print(f"{mol}")
        print(f"\tNN: {mol.nuclear_repulsion():.8f}")
        if len(mol) == 2 and mol.name == 'Hydrogen':
            assert np.isclose(mol.nuclear_repulsion(), 0.722356)

        if len(mol) > 2 and len(mol) < 100:
            atom_coords = numpy(mol.atom_coords())
            atom_names, atom_colors = mol.atom_symbols(), mol.atom_colors()
            bonds = mol.approx_bonds(scale=1.10)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_molecule_3d_scatter(ax, atom_coords, atom_symbols=atom_names, atom_colors=atom_colors, bonds=bonds)
            bb = numpy(mol.bbox(padding=0.0))
            mx_rng, bb_cens = 0.55 * np.max(np.ptp(bb, axis=1)), np.mean(bb, axis=1)
            lims = bb_cens[:,None] + np.stack([-mx_rng, mx_rng])[None,:]
            ax.set_xlim(lims[0]), ax.set_ylim(lims[1]), ax.set_zlim(lims[2]) # equal 3d aspect
            ax.set_title(f"{mol.name} ({mol.hill_formula()})")
            plt.show()
            html_str = mol._repr_html_()
            debug_var = 1
        debug_var = 1


    debug_var = 1
