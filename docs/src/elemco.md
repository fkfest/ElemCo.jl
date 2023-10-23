# Calculations

```@meta
CurrentModule = ElemCo
```

```@docs
ElemCo
```

The `ElemCo` module contains functions for calculating electron-correlation methods. 

Various macros are defined to simplify running calculations. The macros use several reserved variable names. The following table lists the reserved variable names and their meanings.

----------------------
| Variable | Meaning |
|:--------:|:--------|
| `EC` | A global information object containing options, molecular system description, integrals and orbital spaces information |
| `geometry` | Molecular coordinates, either in the `xyz` format (see [`ElemCo.MSystem`](@ref)) or the file containing the xyz coordinates. |
| `basis` | Basis set information (see [`ElemCo.MSystem`](@ref)) |
| `fcidump` | File containing the integrals in the FCIDUMP format (see [`ElemCo.FciDump`](@ref)) |


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
