import numpy as np
import basis.data as data

# example basis download link from BSE
# http://www.basissetexchange.org/api/basis/sto-3g/format/gaussian94/?version=1&elements=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54


class BasisSet():
    def __init__(self, molecule, name='sto3d', **kwargs):
        self.bfs = []
        self.shells = []
        self.name = name.lower()
        basis_data = data.basis[self.name]

        for atom in molecule.atoms:

