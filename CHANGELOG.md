# Release notes

## Unreleased

### Breaking

### Changed

### Added

* `UCCSD(T)`, `ΛUCCSD(T)`, `ΛUDCSD` have been implemented.
* Pseudo-canonicalization of the FCIDUMP file (instead of full SCF calculation).
* generated `UCCSDT` and `UDC-CCSDT` methods.
* macro `@print_input` to print the source of the input file to the output.

### Fixed

* Fix dressing of a three-internal integral (which could have slightly affected the energy of CCSD/DCSD).

## Version [v0.9.0] - 2024.01.20

### Added

* Add various methods (`DF-[U]HF`, `BO-[U]HF`, `[U/R]CCSD`, `[U/R]DCSD`, `SVD-DCSD`, `SVD-DC-CCSDT`, `CCSD(T)`, `ΛCCSD(T)`, `ΛDCSD`...).
* Setup macros, options etc.
* ...
