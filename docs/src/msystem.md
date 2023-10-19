# Molecular system

```@meta
CurrentModule = ElemCo.MSystem
```

```@docs
MSystem
```

The molecular system is the core of the simulation. It contains all the
information about the molecule, including the geometry and basis sets
The molecular system is defined using the `MSys` function:

```julia
MSys(geometry, basis)
```

where `geometry` is a string containing the molecular geometry in the
[XYZ format](https://en.wikipedia.org/wiki/XYZ_file_format), and `basis` is
a dictionary containing the basis set information.

## Geometry

The geometry of the molecule is defined using the `geometry` argument of
the `MSys` function. The geometry is defined in the
[XYZ format](https://en.wikipedia.org/wiki/XYZ_file_format). Here's an
example of how you can define the geometry of a water molecule:

```julia
geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
```

The first line of the geometry string contains the units of the
coordinates. The supported units are `bohr` and `angstrom`. The
coordinates of the atoms are specified in the following lines. Each line
contains the atomic symbol and the coordinates of the atom. The
coordinates are separated by spaces or tabs.

## Basis set

The basis set is defined using the `basis` argument of the `MSys`
function. The basis set is defined as a dictionary, where the keys are
the names of the basis sets, and the values are the basis set
definitions. Here's an example of how you can define the basis set for a
water molecule:

```julia
basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mp2fit"=>"cc-pvdz-rifit")
```

The basis set dictionary contains three keys: `ao`, `jkfit`, and
`mp2fit`. The `ao` key contains the basis set for the AO integrals, the
`jkfit` key contains the basis set for the density fitting integrals in the Hartree-Fock calculations,
and the `mp2fit` key contains the fitting basis set for the correlated calculations.

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
