# Orbital Localization

```@docs
ElemCo.OrbLocalization
```

ElemCo provides three localization schemes for occupied orbitals:

- `"ibo"`: Intrinsic Bond Orbitals (default)
- `"pm"`: Pipek-Mezey localization with Mulliken charges
- `"boys"`: Foster-Boys localization

Virtual orbitals can optionally be localized as orthogonal projected atomic orbitals (OPAOs).

For fragment-tagged dumps and PiOS-style region selection, see [Orbital Regions](region.md).

## Basic usage

```julia
@dfhf
@localize

@set loc method="pm"
@localize

@set loc method="boys" virtual=false
@localize
```

By default, `@localize` localizes occupied orbitals and also builds OPAOs for the virtual space. Set `@set loc virtual=false` to localize only the occupied orbitals.

## Minimal basis for SAD and IAO construction

The SAD starting guess and the IAO construction used by IBO localization both use a minimal basis.

- `scf.minao` controls the minimal basis for the SAD guess
- `loc.minao` controls the minimal basis for IAO construction in localization

If these options are left empty, ElemCo first looks for a `"minao"` entry in the basis dictionary and otherwise falls back to the built-in `minao` basis.

```julia
basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvdz-jkfit",
             "mpfit"=>"cc-pvdz-mpfit",
             "minao"=>"ano-rcc-mb")
```

To override the basis dictionary for a specific workflow, set the option explicitly:

```julia
@set scf minao="minao"
@set loc minao="minao"
```

## Exported functions

```@autodocs
Modules = [ElemCo.OrbLocalization]
Private = false
Order = [:function]
```

## Internal functions
```@autodocs
Modules = [ElemCo.OrbLocalization]
Public = false
Order = [:function]
```
