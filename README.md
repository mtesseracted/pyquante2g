# Vector accelerated version of PyQuante2

## Code structure
- arb_math: arbitrary precision math
    - core: interface to mpmath and basic funcs
    - functions: funcs needed for ERI
    - eri: implementation of THO's ERI
    - overlap: overlaps and kinetic
    - nai: nuclear attraction integral
- array_ops: interface for the vectorized backend
    - core: abstractions for interfacing with backends
    <!-- - cupy: -->
    <!-- - jax: -->
    - numpy:
    <!-- - tensorflow: -->
    - torch:
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

@TODO: need to abstract the tensor.device method at least for tests