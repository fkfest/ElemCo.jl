# ElemCo.jl Documentation


`ElemCo.jl` is a Julia package for computing electronic structure properties of molecules and materials. It provides a set of tools for performing quantum chemical calculations, including Hartree-Fock and post-HF methods.

## Installation

You can install `ElemCo.jl` using the Julia package manager:

```julia
julia> using Pkg
julia> Pkg.add("ElemCo")
```

For a development version of `ElemCo.jl`, clone the [ElemCo.jl-devel](https://github.com/fkfest/ElemCo.jl-devel) repository and create an alias to set the project to the `ElemCo.jl` directory,

```bash
alias jlm='julia --project=<path_to_ElemCo.jl>'
```

Now the command `jlm` can be used to start the calculations,

```bash
jlm input.jl
```

## Usage

### Input file

The input file is a Julia script that contains the calculation details. The script should start with the following lines,

```julia
using ElemCo
@print_input
```

The `@print_input` macro prints the input file to the standard output. The calculation details are specified using the macros provided by `ElemCo.jl`.

### Macros

The following macros are available in `ElemCo.jl` (see [the documentation for more details and macros](@ref list_of_macros)),

- [`@dfhf`](@ref) - Performs a density-fitted Hartree-Fock calculation.
- [`@cc`](@ref)` <method>` - Performs a coupled cluster calculation.
- [`@dfcc`](@ref)` <method>` - Performs a coupled cluster calculation using density fitting.
- [`@set`](@ref)` <option> <setting>` - Sets the options([`ElemCo.ECInfos.Options`](@ref)) for the calculation.

etc.

Default scratch directory path on Windows is the first environment variable found in the ordered list `TMP`, `TEMP`, `USERPROFILE`.
On all other operating systems `TMPDIR`, `TMP`, `TEMP`, and `TEMPDIR`. If none of these are found, the path `/tmp` is used.
Default scratch folder name is `elemcojlscr`.

Variable names `fcidump`, `geometry` and `basis` are reserved for the file name of FCIDUMP, geometry specification and basis sets, respectively.

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
             "mpfit"=>"cc-pvdz-rifit")

# Compute DF-HF 
@dfhf
# Calculate MO integrals (optional)
@dfints
# Run CCSD(T) calculation
@cc ccsd(t)
```

This code defines a water molecule, computes DF-HF using the cc-pVDZ basis set, calculates integrals using density fitting (`mpfit` basis) and computes CCSD(T) energy.

### Setting options

To set options ([`ElemCo.ECInfos.Options`](@ref)) for the DF-HF, CC, etc calculations, you can use the [`@set`](@ref) macro. Here's an example of how you can use this macro:

```julia
# Set the maximum number of iterations to 10
@set scf maxit=10

# Compute DF-HF using the new options
@dfhf
```

This code sets the maximum number of iterations for the SCF procedure to 10 using the [`@set`](@ref) macro, and then computes DF-HF using the new options using the [`@dfhf`](@ref) macro.

### Using AVX2 instructions on AMD "Zen" machines

MKL tends to be rather slow on AMD "Zen" machines (stand 2024).
To use AVX2 instructions in MKL on AMD "Zen" machines, you can slightly modify the `mkl` libraries by running the [`ElemCo.amdmkl`](@ref) function,
which will replace two symbolic links with compiled libraries that enforce the AVX2 instructions,

```julia
using ElemCo
ElemCo.amdmkl()
```

Note: this function has to be called in a separate script (separate Julia session) before running the calculations, i.e., your workflow can look like this:

```bash
> julia -e 'using ElemCo; ElemCo.amdmkl()'
> julia input.jl
```

One can revert the changes by running the function with the argument `true`,

```julia
using ElemCo
ElemCo.amdmkl(true)
```

## Documentation

[Equations](./assets/equations.pdf) for the methods implemented in ElemCo.jl.
