# GPU accelerated version of PyQuante2

## Usage

Can be installed with `pip install .` from the repo. 

The two main helper classes are `Molecule` and `BasisSet`. A helper is defined in atoms/molecule.py::load_sample_molecule() for loading molecules defined in the samples/samples.py file. The `Molecule` class can load .xyz files. The `BasisSet` class takes an instance of a `Molecule`, and a basis set name, which expects the .gbs file to exist for in pyquante2g/basis/cart_gto_data. An instance of `BasisSet` can be fed to one of the integral engines, which currently has a full dense and an 8-fold symmetry implementations (`IntegralEngineBase` and `IntegralEngineSparse`). All arrays are stored as pytorch tensors, and are therefore fully backpropable. There is an example in examples/autodiff_coords.py that runs restricted Hartree-Fock and then uses pytorchs's autodiff to get the derivative of the energy with respect to nuclear coordinates.


## Code structure
- arb_math: arbitrary precision math
    - core: interface to mpmath and basic funcs
    - functions: funcs needed for ERI
    - eri: implementation of THO's ERI
    - overlap: overlaps and kinetic
    - nai: nuclear attraction integral
- torch_ops: torch-native tensor helpers used by the vectorized code
- atoms: for representing molecules
    - elmements: atomisitic info
    - molecule: classes for atoms and building molecules
- basis: basis set manipulations
    - cart_gto: cart. gto primitives and pairs, calc's integrals
    - cart_gto_data: basis set data files
    - basisset: 
    - cart_gto_utils: util funcs, e.g. generating ang. mom. tuples
    - data: hard-coded basis sets in python data structs
    - one_ints
- utils: utility functions
    - file_utils: 
    - plotting_utils:
- ref_ints: non-batch reference implementation of integrals (numpy)
    - eri: implementation of THO's ERI
    - overlap: overlaps and kinetic
    - nai: nuclear attraction integral

## References
- [Hiroshi Taketa, Sigeru Huzinaga, Kiyoshi O-ohata, 1966 (THO66)][tho66]
- [T. Helgaker and P. Tayler 1995 (HT95)][ht95]
- [Minhhuy Hô and Julio-Manuel Hernández-Pérez, 2012 (HH12)][hh12]

[tho66]: https://doi.org/10.1143/JPSJ.21.2313
[ht95]: https://doi.org/10.1142/9789812832115_0001
[hh12]: https://doi.org/doi:10.3888/tmj.14-3

