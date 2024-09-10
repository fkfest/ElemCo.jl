# Calculations

```@meta
CurrentModule = ElemCo
```

```@docs
ElemCo
```

The `ElemCo` module contains the main macros and functions for running electronic structure calculations. The methods are contained in various submodules and are described in the following sections.

Various macros are defined and exported to simplify running calculations. The macros use several reserved variable names. The following table lists the reserved variable names and their meanings.

----------------------
| Variable | Meaning |
|:--------:|:--------|
| `EC::ECInfo` | A global information object containing options, molecular system description, integrals and orbital spaces information, see [`ElemCo.ECInfo`](@ref). |
| `geometry::String` | Molecular coordinates, either in the `xyz` format or the file containing the xyz coordinates, see [`ElemCo.MSystem`](@ref). |
| `basis::Dict` | Basis set information, see [`ElemCo.MSystem`](@ref) |
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
