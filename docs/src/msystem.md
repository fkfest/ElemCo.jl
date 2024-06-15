# Molecular system

```@meta
CurrentModule = ElemCo.MSystem
```

```@docs
MSystem
```

The molecular system is the core of the simulation. It contains all the
information about the molecule, including the geometry and basis sets.
The molecular system is an instance of `FlexibleSystem` from the `AtomsBase.jl` package, and the basis set information is stored in `:basis` field of the molecular system and each atom.
The molecular system is defined using the `parse_geometry` function:

```julia
parse_geometry(geometry, basis)
```

where `geometry` is a string containing the molecular geometry in the
[XYZ format](https://en.wikipedia.org/wiki/XYZ_file_format) (or a xyz-file), and `basis` is
a dictionary containing the basis set information (or a string defining the AO basis).

## Geometry

The geometry of the molecule is defined using the `geometry` argument of
the `parse_geometry` function. The geometry is defined in the
[XYZ format](https://en.wikipedia.org/wiki/XYZ_file_format). Here's an
example of how you can define the geometry of a water molecule:

```julia
geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
```

The first line of the geometry string contains the units of the
coordinates. The supported units are `bohr` and `angstrom` (default is `bohr`). If the first line contains the number of atoms (as in the standard XYZ format), then the next line is skipped, and the default units are `angstrom`. The
coordinates of the atoms are specified in the following lines. Each line
contains the atomic symbol and the coordinates of the atom. The
coordinates are separated by spaces or tabs.

## Basis set

see [Basis set](basisset.md)

The basis set is defined using the `basis` argument of the `parse_geometry`
function.

## Exported functions and types

```@autodocs
Modules = [MSystem]
Private = false
Order = [:type, :function]
```

## Internal functions and types

```@autodocs
Modules = [MSystem]
Public = false
Order = [:type, :function]
```
