# Density-fitted Hartree-Fock

```@meta
CurrentModule = ElemCo.DFHF
```

The density-fitted Hartree-Fock (DF-HF) method is a method for computing
the Hartree-Fock energy using density fitting. The DF-HF method is
implemented in ElemCo.jl using the `@dfhf` macro. Here's an example of
how you can use this macro:

```julia
using ElemCo

# Define the molecule
geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"

basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")

# Compute DF-HF
@dfhf
```

This code defines a water molecule, computes DF-HF using the cc-pVDZ
basis set, and calculates the DF-HF energy.

## Exported functions and types

```@autodocs
Modules = [DFHF]
Private = false
```

## Internal functions
```@autodocs
Modules = [DFHF]
Public = false
```  
