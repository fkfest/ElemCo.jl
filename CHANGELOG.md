# Release notes

## Unreleased

### Breaking

### Changed

* use SVD in DIIS
* Save the memory using in Hessian matrix caclulation in dfmcscf function
* use SVD in DIIS.
* increase number of iterations in 2D-CCSD IAS test.

### Added

* Expand README
* `amdmkl()` function to speed up MKL on AMD machines.
* Add dfmcscf part in documentation
* CROP-DIIS option (JCTC 11, 1518 (2015)) which is less sensitive to the DIIS dimension. To activate, set `diis` option `crop=true`, the DIIS dimension can be changed using `maxcrop` (default is 3).

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
