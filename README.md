# Vector accelerated version of PyQuante2

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


## @TODOs
- ~~add necessary funcs to run cart_gto tests~~
- need to abstract the tensor.device method at least for tests
- ~~make cart_gto.py have alpha values that aren't full nd size~~
- ~~make analysis scripts for accuracies of: fboys, methods acc.~~
~~and time (e.g. HT95 vs THO66 for NAI)~~
- after analysis, clean up the __main__ sections of those files
- look at whether JIT works cleanly with the torch helper layer and the
integral kernels
- ~~add tests for Molecule class, test NN energy, and marble test all init methods~~
- ~~need a dedicated 2-center 3-GTO ERI for doing RI and Cholesky, which we will~~
    ~~want later so we can efficiently calculate higher-dim ERIs~~
- Add an SCF class that can do RHF and UHF
- Analysis scripts:
    - Speed of some test calculations vs PySCF on cpu (& gpu ideally)
    - Speed and accuracy of our ERIs vs Libint and Libcint (arb_math reference)

### For v0 publication:
- Finish the RHF example: 12Mar26, some examples there, need to clean up
- Make an example that uses backprop to do something (back prop to atom positions?)
- add benchmarks
    - benchmark vs a straight integral engine




