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
| `fcidump::String` | File containing the integrals in the FCIDUMP format, see [`ElemCo.FciDump`](@ref). |


## Macros

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
