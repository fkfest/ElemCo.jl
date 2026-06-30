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

## Orthogonal PAOs (virtual space)

OPAOs are built by projecting the AO basis onto the virtual space and orthogonalizing the
resulting projected atomic orbitals. To keep the OPAOs atom-centered (local) while removing
the linear dependencies, the
projected-PAO overlap is handled in three steps, driven by a single Hermitian eigendecomposition:
its numerical rank is determined from a *relative* eigenvalue threshold, exactly that many of the
most independent (atom-centered) PAOs are selected by a rank-revealing column-pivoted QR of the
retained eigenvectors, and these are orthogonalized with a symmetric Löwdin transformation
(followed by one refinement step). Redundant PAOs are dropped, and the kept OPAOs span the
virtual space without redundancies.

The relative threshold is `loc.opaofac * scf.redthr` (with `loc.opaofac` default `3`):
eigenvectors of the PAO overlap with eigenvalue below `opaofac * scf.redthr * λmax` are treated
as redundant. Tying the threshold to the AO basis redundancy threshold `scf.redthr` keeps the
two consistent — only directions that are (near-)redundant by the same standard the basis uses
are removed, while small-but-real directions (e.g. the virtual residual of a frozen core AO) are
kept. Larger `opaofac` prunes more aggressively. The same option governs the fragment OPAOs
built by [`@region`](region.md).

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
