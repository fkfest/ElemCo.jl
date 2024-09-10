# Basis set

```@meta
CurrentModule = ElemCo.BasisSets
```

The basis set is defined as a dictionary, where the keys are
the types of the basis sets, and the values are the basis set:

```julia
basis = Dict("ao"=>"cc-pVTZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvtz-mpfit")
```

The basis set dictionary contains three keys: `ao`, `jkfit`, and
`mpfit`. The `ao` key contains the basis set for the AO integrals, the
`jkfit` key contains the basis set for the density fitting integrals in the Hartree-Fock calculations,
and the `mpfit` key contains the fitting basis set for the correlated calculations.

Alternatively, you can define the basis set using a string that defines the AO basis. In this case, the `jkfit` and `mpfit` basis names will be generated automatically. Here's an example of how you can define the basis set using a string:

```julia
basis = "cc-pVDZ"
```

Common acronyms are also supported for the basis set names, e.g., `cc-pVDZ` can be written as `vdz`, and
`aug-cc-pVTZ` can be written as `avtz`.

```@docs
BasisSets
```

## Exported functions and types

```@autodocs
Modules = [BasisSets]
Private = false
Order = [:type, :function]
```

## Internal functions and types

```@autodocs
Modules = [BasisSets]
Public = false
Order = [:type, :function]
```
