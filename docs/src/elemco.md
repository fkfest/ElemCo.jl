# Calculations

```@meta
CurrentModule = ElemCo
```

```@docs
ElemCo
```

The `ElemCo` module contains the main macros and functions for running electronic structure calculations. The methods are contained in various submodules and are described in the following sections.

## Setting Options

Options can be set using the [`@set`](@ref) macro. The preferred way to set options is to use **local options** within the calculation macro itself. This ensures that options are applied only to the specific calculation and are automatically restored afterwards.

### Local Options (Recommended)

All calculation macros accept an optional `begin...end` block as the last argument to set options locally:

```julia
@cc ccsd begin
  @set wf charge=-1 ms2=1
  @set cc maxit=100 thr=1.e-12
end
```

This is equivalent to:

```julia
@cc ccsd begin
  wf(charge=-1, ms2=1)
  cc(maxit=100, thr=1.e-12)
end
```

The options are **automatically restored** to their previous values after the calculation completes, even if an error occurs. This makes it safe and convenient to run multiple calculations with different settings:

```julia
# Run neutral molecule
@dfhf
@cc ccsd

# Run anion with tighter convergence
@dfhf begin
  @set wf charge=-1
end
@cc ccsd begin
  @set wf charge=-1
  @set cc thr=1.e-12
end

# Options are now back to defaults
```

### Global Options

Options can also be set globally using the [`@set`](@ref) macro outside of calculation blocks:

```julia
@set cc maxit=100
@cc ccsd  # Uses maxit=100
@cc dcsd  # Also uses maxit=100
```

Global options persist until explicitly changed or reset with [`@reset`](@ref).

## Reserved Variables

Various macros are defined and exported to simplify running calculations. The macros use several reserved variable names. The following table lists the reserved variable names and their meanings.

----------------------
| Variable | Meaning |
|:--------:|:--------|
| `EC::ECInfo` | A global information object containing options, molecular system description, integrals and orbital spaces information, see [`ElemCo.ECInfo`](@ref). |
| `geometry::String` | Molecular coordinates, either in the `xyz` format or the file containing the xyz coordinates, see [`ElemCo.MSystems`](@ref). |
| `basis::Union{Dict,String}` | Basis set information, see [`ElemCo.MSystems`](@ref) |
| `fcidump::String` | File containing the integrals in the FCIDUMP format, see [`ElemCo.FciDumps`](@ref). |

The driver routines and macros return energies as ordered descriptive dictionaries [`ElemCo.ODDict`](@ref). The last energy is always the total energy (can be accessed using `last_energy(energies)`). The following table lists the keys and their meanings.

----------------------
| Key | Meaning |
|:---:|:--------|
| `E` | Total energy |
| `Ec` | Correlation energy |
| `HF` | Hartree-Fock energy |
| `MP2` | MP2 energy |
| `CCSD` | CCSD energy |
| `DCSD` | DCSD energy |
| `SING2D-DCSD` | singlet 2D-DCSD energy |
| `TRIP2D-DCSD` | triplet 2D-DCSD energy |
| etc. ||

One can print the keys of the returned `ODDict` to see all the available keys:

```julia
julia> println(keys(energies))
```

or display the complete dictionary together with the descriptions as

```julia
julia> display(energies)
```

The values and the descriptions can be accessed using the keys as

```julia
julia> energies["E"] # Total energy
julia> energies("E") # Description of the total energy
```

## [Macros](@id list_of_macros)

```@autodocs
Modules = [ElemCo]
Private = false
Order = [:macro]
```

## Exported functions

```@autodocs
Modules = [ElemCo]
Private = false
Order = [:function]
```

## Internal functions
```@autodocs
Modules = [ElemCo]
Public = false
Order = [:function]
```
