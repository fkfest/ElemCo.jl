# Orbital Regions

The `@region` workflow builds a fragment-oriented orbital dump from an existing wavefunction dump. It reuses the shared localization machinery from `ElemCo.OrbLocalization`, but applies fragment-specific selection, tagging, and optional pi-space construction from `ElemCo.OrbRegion`.

## Basic usage

```julia
@dfhf

@region [1, 2]

@region [:O1, :H1] begin
  @set region mode=:exclusive occ_charge_thr=0.25 atom_charge_thr=0.15
end
```

The centers can be given as atom indices or center labels. Orbitals are read from `wf.start` when set, otherwise from `wf.dump`, and the tagged result is written to `wf.store` when set, otherwise back to the active dump.

Center selection can also be driven directly from options:

- `region.inclusive_centers`: atom indices that should always be treated as inclusive fragment centers.
- `region.exclusive_centers`: atom indices that should always be treated as exclusive fragment centers.

(The `@region [...]` macro argument also accepts center labels, but the `region.*_centers`
options expect atom indices, e.g. `region.inclusive_centers=[1]`.)

When `@region [...]` is called with an explicit center list, those centers are merged into the active selection implied by `region.mode`:

- `region.mode = :inclusive`: the macro argument is added to `region.inclusive_centers`.
- `region.mode = :exclusive`: the macro argument is added to `region.exclusive_centers`.

This also means the macro argument can be omitted entirely when the centers are already configured through `region.inclusive_centers` and `region.exclusive_centers`.

## Selection model

`@region` localizes the occupied space with IBOs and then selects fragment orbitals from atom-resolved IAO charges.

- `region.mode = :inclusive`: keep an occupied orbital if at least one selected center carries a large charge on it.
- `region.mode = :exclusive`: keep an occupied orbital only if all of its large charges remain on the selected centers.
- `region.inclusive_centers`: additional atom indices that participate in the inclusive selection regardless of how `@region` is called.
- `region.exclusive_centers`: additional atom indices that participate in the exclusive selection regardless of how `@region` is called.
- `region.occ_charge_thr`: threshold used to decide whether an occupied orbital belongs to the requested fragment.
- `region.virtual = :complement` (default): build fragment virtuals by projecting the IAOs on the fragment-support atoms into the virtual space, then augment that antibonding-like complement with support-atom OPAOs.
- `region.virtual = :support_opao`: keep the legacy support-atom OPAO construction directly.
- `region.atom_charge_thr`: threshold used to add atoms to the support used for fragment virtual construction. In the default `:complement` mode this uses the accumulated fragment charge over all selected occupied IBOs.
- `region.pao_centers`: additional atom indices whose PAOs are added to the fragment virtual space. These centers are always included (regardless of `atom_charge_thr`) and let you extend the virtual space manually. They apply to all virtual-space constructions: the OPAO/complement ones and `region.pi=:both`, where the PAO OPAOs are appended to the π-projector virtuals (orthogonalized against them).

The fragment occupied orbitals are always tagged as `Inactive` and are placed at the Fermi
level (just below the virtual space). The frozen core *and* the non-selected environment
occupied orbitals are tagged as `Core`, forming a contiguous block below the fragment.
Fragment virtuals are tagged as `Virtual`; the remaining (non-selected) virtuals are tagged
as `Deleted`.

## Downstream usage

With the default `wf.core = :auto` and `wf.freeze_nvirt = -1`, a subsequent correlated
calculation (`@dfmp2`, `@dfcc`, `@cc`, `@fci`, …) reads these classes from the dump and
automatically restricts the active space to the region: `Core` orbitals are frozen and
`Deleted` virtuals are dropped, so only the `Inactive`/`Virtual` fragment is correlated.

```julia
@dfhf
@region [:C1, :C2, :C3, :C4]   # writes the region dump back to wf.dump
@dfmp2                          # correlates only the region; environment is frozen as core
```

The user can override the dump's prescription at any time:

- `@set wf core=:none` (or any explicit `:none`/`:small`/`:large`) or `@set wf freeze_nocc=N`
  selects the frozen core manually instead of using the dump's `Core` orbitals;
- `@set wf freeze_nvirt=N` freezes exactly `N` highest virtuals instead of dropping the dump's
  `Deleted` virtuals.

For an ordinary (non-region) dump, `wf.core = :auto` reproduces the standard `:large`
frozen-core behavior, since such dumps tag only the chemical core as `Core`.

## Pi-space modes

`region.pi` controls whether the fragment is built from the default IBO / complement-plus-OPAO workflow or from the pi-projector workflow.

- `:none`: standard region selection from localized occupied orbitals, antibonding-like fragment-complement virtuals, and support-atom OPAO augmentation.
- `:occupied`: occupied pi orbitals are selected by the pi projector, while the virtual fragment is still defined by support-atom OPAOs.
- `:both`: both occupied and virtual fragment spaces are selected with the pi projector.

The pi workflow builds one local `p'_z` target orbital per selected atom from valence p-type IAOs and a bond-aware local plane. The occupied or virtual block is then rotated by diagonalizing the corresponding projector-overlap matrix.

The PiOS papers also emphasize two practical controls that are now available here:

- `region.pi_electrons`: override the automatic chemistry-based π-electron count when the default connectivity heuristic is not appropriate.
- `region.pi_occupied`: keep only a restricted-reference subset of the highest occupied constructed π orbitals.
- `region.pi_virtual`: keep only a restricted-reference subset of the lowest virtual constructed π orbitals when `region.pi = :both`.

Example: keep only the frontier HOMO/LUMO pair from a four-center π system.

```julia
@region [:C1, :C2, :C3, :C4] begin
  @set region pi=:both pi_occupied=1 pi_virtual=1
end

@region begin
  @set region mode=:exclusive inclusive_centers=[2] exclusive_centers=[1]
end
```

## Pseudo-canonicalization

Set `region.pseudo = true` to pseudo-canonicalize the selected fragment occupied and fragment virtual blocks after the region orbitals have been assembled.

```julia
@region [:C1, :C2, :C3, :C4] begin
  @set region pi=:both pseudo=true
end
```

This diagonalizes the occupied-occupied and virtual-virtual Fock subblocks in the selected fragment basis.

## Notes And Limitations

- unrestricted references are processed separately in alpha and beta space
- pi electron counting is currently implemented for main-group p-block centers only
- `region.pi_electrons` can be used when the automatic chemistry-based π-electron count is not appropriate
- `region.pi_occupied` and `region.pi_virtual` currently apply only to restricted references
- if bonded-neighbor information is insufficient to define a local plane, nearby atoms are used as a fallback

## API

```@docs
ElemCo.OrbRegion
ElemCo.region_orbitals
```

```@autodocs
Modules = [ElemCo.OrbRegion]
Private = false
Order = [:function]
```
