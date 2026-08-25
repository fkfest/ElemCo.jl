# Hartree-Fock

```@meta
CurrentModule = ElemCo.HF
```

This module implements closed- and open-shell Hartree-Fock, both
density-fitted (`dfhf`/`dfuhf`, via the `@dfhf` macro) and exact non-DF
from AO integrals (`hf`/`uhf`, via the `@hf`/`@uhf` macros), sharing a common SCF loop with a pluggable
Fock builder. Here's an example of computing density-fitted HF with the
`@dfhf` macro:

```julia
using ElemCo

# Define the molecule
geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-mpfit")

# Compute DF-HF
@dfhf
```

This code defines a water molecule, computes DF-HF using the cc-pVDZ
basis set, and calculates the DF-HF energy.

## Exported functions and types

```@autodocs
Modules = [HF]
Private = false
```

## Internal functions
```@autodocs
Modules = [HF]
Public = false
```  
