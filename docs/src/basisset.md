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

Additionally to the default basis, it is possible to specify basis sets for each element/center:

```julia
basis = Dict("ao"=>"avdz; O=avtz; H1=vdz; H2=vtz",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvtz-mpfit")
```

In this case, the basis set for the AO integrals is `avdz` for all elements, except for oxygen (irrespective of the name in the geometry, i.e., `"O"`, `"O1"`, etc), which uses `avtz`, and hydrogen, which uses `vdz` and `vtz` for the centers named `"H1"` and `"H2"`, respectively.

Moreover, it is possible to define custom basis sets for each element/center using the Molpro format:

```julia
basis = Dict("ao"=>"cc-pVTZ; o=avdz;
             h={! hydrogen
                s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000
                c, 1.4, 0.0196850, 0.1379770, 0.4781480, 0.5012400
                c, 4.4, 1.0000000
                p, H , 0.7270000
                c, 1.1, 1.0000000}",
             "jkfit"=>"vtz-jkfit",
             "mpfit"=>"avtz-mpfit")
```

In this case, the basis set for the AO integrals is `cc-pVTZ` for all elements, except for oxygen, which uses `avdz`, and hydrogen, which uses a custom basis set. The Fock density fitting basis set is `vtz-jkfit`, and the fitting basis set for the correlated calculations is `avtz-mpfit`.

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
