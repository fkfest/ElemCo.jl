# Release notes

## Unreleased

### Breaking

### Changed

### Added

* The automatically generated `UCCSDT` and `UDC-CCSDT` methods have been added to the docs.

### Fixed

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
