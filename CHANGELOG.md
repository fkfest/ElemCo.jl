# Release notes

## Version [v0.13.0] - 2024.07.09

### Breaking

* `DIIS.perform` has been changed to `DIIS.perform!` in order to allow to read the vectors and residuals as `Vector{}`.
* The signature of `newmmap` function has changed (the type specification is now the last argument and defaults to `Float64`.
* The `FciDump` module has been renamed to `FciDumps`.
* The `FDump` type has been changed to `FDump{N}` with N=3 (for triangular storage of 2-electron integrals) or 4. The logical variable `triang` has been removed (there is a function `is_triang(::FDump)` now). Aliases `TFDump = FDump{3}` and `QFDump = FDump{4}` have been introduced. 
* The `ECInfo` type now accepts only `FDump{3}`. The `FDump{4}` objects have to be transformed first (the transformation functions are not implemented yet).
* The triangular functions have been moved to a separate file `utensors.jl`, part of the `QMTensors` module. `uppertriangular` function has been renamed to `uppertriangular_index`.
* The driver functions and macros now return energies in an ordered descriptive dictionary `OutDict=ODDict{String,Float64}`. Use `last_energy` function to access the last energy (or `last` to access the whole entry including the key and the description).

### Changed

* Save the memory using in Hessian matrix calculation in dfmcscf function.
* `dfdump` stores the MO integrals internally in mmaped files.
* The header of the `FDump` is now stored in a type-stable structure `FDumpHeader`.

### Added

* Export of molden files (`@export_molden`). At the moment the orbital energies and occupations are not exported.
* Add dfmcscf part in documentation
* CCSDT and DC-CCSDT closed-shell implementations generated with Quantwo.
* `QMTensors.SpinMatrix` struct for one-electron matrices (e.g., MO coefficients)
* An ordered descriptive dictionary for energy outputs (`ODDict`) has been implemented. Each key-value entry can have a description.
* `DIIS.perform!` now accepts a tuple of functions to calculate cusomized dot-products (e.g., involving contravariants etc).

## Version [v0.12.0] - 2024.05.28

### Breaking

* the `mp2fit` (`rifit`) basis sets have been renamed to `mpfit`. 
* `ERI_?e?c` routines have been renamed to `eri_?e?idx`.

### Changed

* use SVD in DIIS.
* increase number of iterations in 2D-CCSD IAS test.
* interface to `libcint_jll` has been implemented. The basis set library is added (in Molpro format), and basis sets are parsed to a `BasisSet` object. `GaussianBasis.jl` dependency is removed.

### Added

* Expand README
* `amdmkl()` function to speed up MKL on AMD machines.
* CROP-DIIS option (JCTC 11, 1518 (2015)) which is less sensitive to the DIIS dimension. To activate, set `diis` option `crop=true`, the DIIS dimension can be changed using `maxcrop` (default is 3).
* An option `print_init` is added to the `@print_input` macro (default is `false`). If set to `true`, the `ElemCo.jl` info is printed again (useful if the output is redirected in julia to a file).
* A simple DMRG routine is added based on `ITensors` (adapted from `ITensorChemistry.jl`).
* A Molpro interface to import matrop matrices (orbitals or overlap).

### Fixed

* Get rid of error message from git if .git is not available (e.g., in the case of the released version).
* Sort orblist, which fixes issues if user occupations are not provided in a sorted list.
* Fix amplitudes before Hylleraas energy calculation for FR-CC, which will properly report the energy in a (2,2) (single iteration) calculation.

## Version [v0.11.1] - 2024.04.12

### Changed

* Remove `ArgParse` dependency and set `[compat]` section in `Project.toml`.

## Version [v0.11.0] - 2024.04.12

### Breaking

* `EC.ms` (previously of type `MSys`) in `ECInfo` is renamed to `EC.system` (of type `AbstractSystem`).
* `ECdriver` routine is moved to `CCDriver` module and renamed to `ccdriver`. The `fcidump` keyword-argument is now empty by default. It doesn't accept list of methods anymore, only one method at a time. 
* The driver routines and macros return energies as `NamedTuple`.
* The SVD methods have to be called now as `SVD-<methodname>`, e.g., `svd-dcsd`.
* The `@svdcc` macro is renamed to `@dfcc` macro and calls the `dfccdriver` routine, which is intended as a driver routine for all DF-based correlation methods (i.e., methods which don't use the `EC.fd` integrals).

### Changed

* Renamed function `active_orbitals` to `oss_active_orbitals`.
* Renamed function `calc_ccsd_resid` to `calc_cc_resid`.
* `ECdriver` and `oss_active_orbitals` now return named tuples.
* Improved documentation of occupation strings syntax.
* Switched to `Atom` and `FlexibleSystem` from `AtomsBase` as the internal representation of the molecular system. The basis set is stored for each atom as `:basis` property (as `Dict{String,String}`, e.g., `system[1][:basis]["ao"]`). One can also set `:basis` property for the whole system. 
* Renamed macro `@opt` to `@set`. `@opt` is now an alias of `@set`.

### Added

* The automatically generated `UCCSDT` and `UDC-CCSDT` methods have been added to the docs.
* SCS-MP2, SCS-CCSD and SCS-DCSD

## Version [v0.10.0] - 2024.02.21

### Breaking

* Cholesky threshold `thr` is used for integral decomposition only. Threshold for elimination of redundancies is now called `thred`.
* Files for amplitudes and multipliers are now called `..._1`, `..._2`,... for singles, doubles, etc.

### Changed

* Option `ignore_error` is moved from ECInfo structure to `wf` options.

### Added

* `UCCSD(T)`, `ΛUCCSD(T)`, `ΛUDCSD` have been implemented.
* Pseudo-canonicalization of the FCIDUMP file (instead of full SCF calculation).
* Generated `UCCSDT` and `UDC-CCSDT` methods.
* Macro `@print_input` to print the source of the input file to the output.

### Fixed

* Fix dressing of a three-internal integral (which slightly affected the energy of CCSD/DCSD with `use_kext=false`).

## Version [v0.9.0] - 2024.01.20

### Added

* Add various methods (`DF-[U]HF`, `BO-[U]HF`, `[U/R]CCSD`, `[U/R]DCSD`, `SVD-DCSD`, `SVD-DC-CCSDT`, `CCSD(T)`, `ΛCCSD(T)`, `ΛDCSD`...).
* Setup macros, options etc.
* ...
