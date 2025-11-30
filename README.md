# Vector accelerated version of PyQuante2

## Code structure
- arb_math: arbitrary precision math, including ERIs
    - core: interface to mpmath and basic funcs
    - functions: funcs needed for ERI
    - eri: implementation of THO's ERI
- array_ops: interface for the vectorized backend
    - core: abstractions for interfacing with backends
    - cupy:
    <!-- - jax: -->
    - numpy:
    <!-- - tensorflow: -->
    - torch:
- atoms: for representing molecules
    - elmements: atomisitic info
    - molecule: classes for building atoms
- basis: basis set manipulations
    - cart_gto_data: basis set data files
    - basisset:
    - cart_gto_utils:
    - data: hard-coded basis sets in python data structs
    - one_ints
- utils: utility functions
    - file_utils: 
    - plotting_utils:
