# ElemCo.jl Documentation


ElemCo.jl is a Julia package for computing electronic structure properties of molecules and materials. It provides a set of tools for performing quantum chemical calculations, including Hartree-Fock and post-HF methods.

## Installation

You can install ElemCo.jl using the Julia package manager:

```julia
julia> using Pkg
julia> Pkg.add("ElemCo")
```

For a development version of ElemCo.jl, clone the repository and create an alias to set the project to the `ElemCo.jl` directory,

```
alias jlm='julia --project=<path_to_ElemCo.jl>'
```

Now the command `jlm` can be used to start the calculations,

```
jlm input.jl
```

## Usage

### Computing density-fitted Hartree-Fock and Coupled Cluster methods

To compute density-fitted Hartree-Fock (DF-HF) using ElemCo.jl, you can use the [`@dfhf`](@ref) macro. In order to run post-HF calculations, the integrals have to be transformed to the MO basis (using the [`@dfints`](@ref) macro), and
the coupled cluster calculations can be performed using [`@cc`](@ref) macro. 
The `@dfints` macro is optional, `@cc` macro will automatically call `@dfints` if it has not been called before.
Here's an example of how you can use these macros:

```julia
using ElemCo

# Print input to the output file
@print_input
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
# Calculate MO integrals (optional)
@dfints
# Run CCSD(T) calculation
@cc ccsd(t)
```

This code defines a water molecule, computes DF-HF using the cc-pVDZ basis set, calculates integrals using density fitting (mp2fit basis) and computes CCSD(T) energy.

### Setting options

To set options ([`ElemCo.ECInfos.Options`](@ref)) for the DF-HF, CC, etc calculations, you can use the [`@set`](@ref) macro. Here's an example of how you can use this macro:

```julia
# Set the maximum number of iterations to 10
@set scf maxit=10

# Compute DF-HF using the new options
@dfhf
```

This code sets the maximum number of iterations for the SCF procedure to 10 using the [`@set`](@ref) macro, and then computes DF-HF using the new options using the [`@dfhf`](@ref) macro.

## Documentation

[Equations](./assets/equations.pdf) for the methods implemented in ElemCo.jl.