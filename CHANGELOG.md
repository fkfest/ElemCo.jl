# Release notes

## Unreleased Version

### Breaking

### Changed

* The 4-index 2-electron integral transformation (`transform_int2!`/`transform_int2_Q!`, used for RHF→UHF FCIDUMP conversion and AO→MO derivation) is now fully BLAS-3: the triangular-output transform's final fold, previously a BLAS-1/2 scalar-axpy + rank-1 outer-product loop, is now a GEMM (still exploiting the `r≤s` packing). The output-index block size now adapts to the available memory (see below) so the streaming scratch stays bounded for large orbital counts (previously it eagerly reserved a fixed multiple of `nin³`), while using a single pass over the integrals when memory is ample.
* SVD-DC methods now localize orbitals before amplitude decomposition by default (`cc.localize=true`), and LLAMA now uses `ampsvd_pivotol_mode=:maxdim` by default for more robust rank selection in difficult cases such as ghost-atom calculations.
* The minimal basis used for SAD guesses and IAO construction now defaults to MINAO. It can be overridden explicitly with `scf.minao` and `loc.minao`, or via a `"minao"` entry in the basis dictionary.
* Batching of 4-index integrals is implemented in `kext` routines. This should substantially reduce the memory demand of closed-shell and open-shell CCSD/DCSD.
* The `kext`-type contractions are implemented for unrestricted full-triples methods (UCCSDT and UDC-CCSDT).
* `@savewf` accepts orbital coefficients as matrix or tuple of matrices.
* Use LLAMA for svd decompositions.
* Some speedups in (T) and integrals calculations.
* The test suite was migrated to the TestItems framework (`@testitem`), enabling
  per-test run/debug in the VS Code Test Explorer and parallel execution. Each
  test file in `test/` is wrapped in an `@testitem` tagged by category
  (`:fcidump`, `:cc`, `:df`, `:complex`, …, plus `:quick` or `:long`). Run quick
  tests with `Pkg.test()`, or all tests with `Pkg.test(test_args=["all"])`. Set
  `ELEMCO_TEST_NWORKERS=N` to run test items in parallel across `N` worker
  processes. Subsets can be selected from the REPL by passing tags and/or item
  names, e.g. `Pkg.test(test_args=["df"])` or `Pkg.test(test_args=["h2o","complex"])`;
  the old group names map to tags (`FCIDUMP`→`:fcidump`, `DF`→`:df`, …). The same
  tag/name filtering is also available in the VS Code Test Explorer.

### Added

* Memory-management utilities and options. `available_memory()` estimates how much memory may reasonably be used for a large scratch allocation as a fraction (default `0.8`) of the currently available memory — the minimum of the node's free memory (`≈ MemAvailable`, so reclaimable page cache counts as free), any enforced **cgroup** budget (`cgroup_memory_available`, covering SLURM `ConstrainRAMSpace`, containers and k8s — cgroup v1 and v2), and any **SLURM** per-job limit (`slurm_memory_limit` from `SLURM_MEM_PER_NODE`/`SLURM_MEM_PER_CPU`) — after a `GC.gc(true)` so Julia's own freed memory is reflected. `available_memory(EC)` honors a user budget. New `mem` options: `@set mem budget=<GB>` sets an explicit budget (`≤0` = automatic, the default) and `@set mem fraction=<f>` tunes the fraction used in automatic mode. Currently used to size the blocked 4-index integral transformation (so fat/constrained nodes get the right number of passes over the integrals); intended for broader memory management across the package.
* **Non-DF (exact) AO integrals and AO-direct coupled cluster** (issue #290). Exact four-index AO electron-repulsion integrals can now be generated and used without density fitting. `@ints` writes the AO integrals to scratch files (`ao_int2` triangular ERIs, `S_AA`, `h_AA`); `@hf`/`@uhf` build HF/UHF from them (the AO Fock is assembled by streaming the triangular integrals one slab at a time in `O(nao²)` memory, exploiting the `r≤s` packing — the full `nao⁴` integral tensor is never formed), and closed- **and open-shell** MP2/CCSD/DCSD run **AO-direct**: the frozen core is folded into an effective one-electron Hamiltonian and both the T1-dressed integrals and the `kext` 4-external term are contracted straight from the AO integrals each iteration (occ-early one pass, no full MO integral set is ever formed). Higher closed-shell methods now run AO-direct as well, off a persisted half-transformed integral store (see below); only unrestricted higher methods, FCI, and linearly-dependent/deleted-orbital cases still retransform the AO integrals to a transient MO dump per call. The `EC.fd` FCIDUMP object is now strictly MO-only. Set `@set int df=false` to default new calculations to the exact-AO route; `@set int ao_direct=false` forces the derived-MO-dump route for the cheap methods (e.g. for basis sets with linearly-dependent orbitals, which the AO-direct path does not yet handle and which fall back to the derived dump automatically).
* **Persisted ± (plus/minus) supermatrix store for the exact AO integrals** (`PMStore` module, on by default via `@set int ao_pm=true`). The two ±-symmetrized integral combinations `Vs/Va[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ ± ⟨νμ|ρσ⟩` are symmetric (real) / Hermitian (complex) matrices over the packed pair space; only their lower **block**-triangle is stored as dense σ-aligned column panels (`ao_pm_s`/`ao_pm_a`, ≈ **half the disk** of the joint `ao_int2` — which is never created: the ± fold is fused into the ERI generation, each shell-aligned ket-column block is assembled in a bounded RAM slab, folded and written directly as panels). All AO-direct consumers run on the store: the `kext` 4-external term as zero-copy panel GEMMs at **halved flops and streaming** (measured 1.1–2.2×, growing with the number of correlated occupied orbitals — the ± combinations are built *once* instead of every iteration, which is why this wins where the on-the-fly `cc.use_pm_kext` variant lost), the T1-dressing occ-early sweeps and the AO-HF/frozen-core Fock builders at halved integral streaming, and the open-shell αβ `kext` via an explicit ±-fold of the αβ density. Joint-format consumers (the AO→MO transform for CCSD(T)/Λ/EOM/FCI) reconstruct `ao_int2` transiently and delete it afterwards. Complex integrals are supported throughout (pair-internal swaps are conjugation-free ± signs; the block mirror conjugates). `@set int ao_pm=false` keeps the previous joint-only flow.
* **AO-direct higher methods via a persisted half-transformed integral store.** The one-index→occupied half-transform `⟨iν|ρσ⟩ = Σ_μ ⟨μν|ρσ⟩ C_occ[μ,i]` is the universal downstream intermediate for the correlated methods, and its bra-occupied transform is T1-independent — so it is built **once** per orbital set (per spin) from the ± store and streamed with single-threaded `pread` + `posix_fadvise(WILLNEED)` (measured ≈4.9 GB/s, at half the disk of a full `n⁴/2` store), replacing per-iteration re-streaming of the whole integral set. Off this store every **closed-shell** higher method now runs **AO-direct** — with no transient MO dump: standalone **MP2/UMP2/RMP2**, **CCSD(T)/DCSD(T)**, the **Lagrange (Λ) equations** and correlated properties (**Λ-CCSD/Λ-DCSD**, **ΛCCSD(T)/ΛDCSD(T)**, 1-RDM/dipole), and **EOM-CCSD** excitation energies. Each MO integral block a method needs (they all carry ≥1 occupied index) is a cheap GEMM-only transform of the stored half-transform, written mmapped in the consumer's index order; the Λ 4-external term reuses the ± `kext`, and the Λ singles term is the v,o block of a generalized (2J−K) Fock reusing the streaming ± Fock builder. All match the derived-MO-dump reference to `≤1e-7…1e-13`. Unrestricted higher methods and linearly-dependent/deleted-orbital cases still fall back to the derived MO dump; the open-shell dressing already streams the same store per spin. Complex integrals are supported for the directly-transformed blocks; blocks that would need a bra↔ket Hermiticity swap are real-only (pending complex AO integrals).
* Orbital localization support with Intrinsic Bond Orbitals (IBO), Pipek-Mezey, and Foster-Boys localization for occupied orbitals, plus optional orthogonal projected atomic orbitals (OPAOs) for virtual orbitals.
* **Complex-valued calculations**: Systematic support for `ComplexF64` integrals and amplitudes throughout the codebase. When a complex FCIDUMP is loaded, `ECInfo{ComplexF64}` propagates the element type through solvers, tensor tools, CC methods, EOM, and interfaces. Key changes:
  - `TFDump`/`QFDump` type aliases are now parametric (`FDump{T,3}`/`FDump{T,4}`)
  - DIIS and Davidson solvers are parametric (`Diis{T}`, `Davidson{T}`) with correct Hermitian symmetry
  - Tensor load/save/mmap defaults use `ec_eltype(EC)` instead of `Float64`
  - All CC methods (CCSD, DCSD, (T), Λ-CCSD(T), SVD-DC) propagate element type through amplitudes, residuals, and energy accumulators
  - EOM-CCSD/EOM-DCSD trial vectors and Hamiltonian matrices use `ec_eltype(EC)`
  - Biorthogonal HF (`left_from_right_rotations`) uses `transpose` instead of `adjoint` — a correctness fix for complex orbitals (i.e, the resulting left coefficients are *complex conjugate* of the actual left coefficients. This might change if it turns out to be too confusing.)
  - FCI module: `Symmetric` → `Hermitian` in P-space diagonalization; removed restrictive `Float64` return annotations
  - TREXIO interface: amplitude/determinant write functions accept `AbstractArray{<:Number}`
* EOM-UCCSD/EOM-UDCSD and EOM-RCCSD/EOM-RDCSD (restricted to singlet excitations) methods have been implemented.
* A new factorization of `kext` contractions for closed-shell CCSD/DCSD using a symmetric/antisymmetric representation is implemented (can be activated by setting `@set cc use_pm_kext=true`). This algorithm has two times less FLOPs than the standard implementation, however, it can be less efficient because of cache-unfriendly access in the construction of the intermediates (which is parallelized using `Threads.@threads` and should scale well with the number of threads). The standard implementation is still used by default, but the new one can be activated for testing and benchmarking purposes.
* FCI and CIPHI work for non-Hermitian Hamiltonians.

### Fixed

* Fix a situation when an fcidump was deleted immediately after creation if dummy atoms are present. Now the fcidump is deleted in the `@dummy` macro directly and not in `@setupEC`.
* Fix `freeze_nvirt` option (which was apparently not working at all before).
* Fix redundancy detection in the orthogonal-PAO (OPAO) construction used by `@localize` and `@region`. The projected-PAO overlap was orthogonalized with an *absolute* (ALPACA) rank threshold that failed to detect the near-linear-dependencies that appear with diffuse/augmented basis sets (e.g. aug-cc-pVDZ), leaving numerically amplified junk orbitals in the (active) virtual space. The OPAOs are now built by detecting the rank from a *relative* eigenvalue threshold of the PAO overlap, selecting that many atom-centered PAOs by a rank-revealing column-pivoted QR of the retained eigenvectors, and orthogonalizing them with a symmetric Löwdin transformation (plus one refinement step that restores machine-precision orthonormality even for amplified low-presence directions) — removing redundancies while keeping the OPAOs local. The threshold is tied to the AO basis redundancy threshold: `relthr = loc.opaofac * scf.redthr` (new `loc.opaofac` option, default `3`). This keeps the two consistent and basis-adaptive — only directions that are (near-)redundant by the same standard the basis uses are dropped, while small-but-real directions above `scf.redthr` (e.g. the virtual residual of a frozen core AO, ~`1e-7·λmax`) are kept as genuine degrees of freedom. Because this conservative cut sits below the real (even diffuse) virtual directions, a diffuse basis with full PAO support (e.g. `@region` with all atoms supporting the virtuals) now recovers the complete virtual space instead of dropping real (diffuse) virtuals.

## Version [v0.15.0] - 2026.02.05

### Breaking

* The fallback basis sets are not used by default anymore. Set `@set int use_fallback_basis=true`
  to enable them.
* In `@write_ints`, the `tol` argument is a keyword argument now (default is `-1.0`). 
* `wf.orb` and `wf.left` options are deprecated. The orbitals are now always written to and read from the trexio dump file `wf.dump`. Use `@loadwf` to load the orbitals from the dump file.
* The `@transform_ints` macro now automatically uses biorthogonal transformations for BO orbitals.
* The `@mtensor`, `@mview` macros are moved from `TensorTools` to a new `MTensorOperations` module (still reexported by `TensorTools`).
* The function `get_spaceblocks` has been moved from `TensorTools` to `Utils` module.

### Changed

* The fallback basis sets are now defined for `ao` (`def2-tzvppd`), `jkfit` (`aug-def2-universal-jkfit`), and `mpfit` (`def2-tzvppd-mpfit`) basis sets.
* Functions for `H` and `He` are copied from [aug]-cc-pVXZ basis sets to [aug]-p[w]CVXZ basis sets. Functions for Li and Be are copied from [aug]-cc-pCVXZ basis sets to [aug]-pwCVXZ basis sets.
* `CCDriver` module is renamed to `Drivers`. 
* Function `transform_fcidump` has been renamed to `transform_fcidump!`.
* (T) methods automatically use pseudo-canonical transformation if the Fock matrix is not diagonal in the occupied and virtual subspaces.

### Added

* FCI and CIPHI (selected CI) methods.
* Local options for macros: All calculation macros (`@cc`, `@dfcc`, `@dfhf`, `@dfuhf`, `@dfmcscf`, `@dfmp2`, `@fci`, `@ciphi`, `@bohf`, `@bouhf`) now accept an optional `begin...end` block to set options locally for that specific call. Options are automatically restored after the call completes. This is the recommended way to set options for individual calculations.
* Automatic augmentation of basis sets by additional diffuse or steep functions.
* Functions to get all elements available in a given basis set (`get_available_elements4basis`) and to output the basis set for a given list of elements in the molpro format (`output_basis`).
* Augmented basis sets for jkfit vXz-jkfit and def2-universal-jkfit basis sets.
* A keyword argument `format` in `@write_ints` macro to write integrals to npy files (if `format=:npy`) or to ascii file (if `format=:ascii`).
* DF-HF and DF-UF orbitals are stored in trexio dump file.
* A check for changes of the molecular geometry/basis/fcidump is performed in every macro-command call. If a change is detected, the integrals are set to be recalculated or reloaded from the fcidump file. With this, the user doesn't need to worry about calling `@ECinit` after changing the geometry/basis/fcidump.
* A test for dummy atoms is added. At the moment, if dummy atoms are detected, the integrals are recalculated. In the future, once we have AO-FDump support, the integrals can be reused.
* Macros `@loadwf` and `@savewf` to load and save orbitals (etc) from/to trexio dump files. `@copywf` to copy trexio dump files (e.g., to make a local backup).
* Wavefunction store/start functionality for coupled cluster and selected CI methods. Use `@set wf store="filename.h5"` to store the wavefunction (amplitudes for CC, determinants/coefficients for CIPHI) to a TREXIO file. Use `@set wf start="filename.h5"` to restart a calculation from a previously stored wavefunction. Multi-state CIPHI calculations store each state in separate files (e.g., `filename_state2.h5`).
* `pt2_only` option for CIPHI calculations (`@set ciphi pt2_only=true`) to skip variational iterations and compute only the PT2 correction using stored determinants.

### Fixed

* A simple sanity check of the fitting basis sets is performed (by checking whether it's 
  an AO basis set). The error message can be turned to a warning by setting `@set int check_fit_basis=false`.
* Fix normalization of biorthogonal orbitals to a balanced normalization (i.e., the norms of left and right orbitals are equal).
* Integral transformation now should use much less memory and be faster.
* Fix incompatibility with julia 1.13 (replace call of an internal Base function with a simple custom implementation).
* Fix UCCSD and UDCSD for the case of no beta electrons ([#272]).

## Version [v0.14.1] - 2025.07.03

### Changed

* update libcint to version 6. 
* reduce allocations in the integral calculation routines.

### Added

* a simple XML based interface to Molpro.

### Fixed

* Fix Mac and Windows compatibility issues related to the case insensitivity of the file system. The capital letters in the file names on scratch are now converted to lower case plus a character `ß` (e.g., `oVoO` becomes `ovßoß`).

## Version [v0.14.0] - 2025.04.16

### Breaking

* the definition of `ampsvdtol` threshold for SVD methods has been changed. Now it corresponds to the threshold for the density matrix (i.e., square of the previous definition).
* `verbosity` has been moved from `ECInfo` to `Options.print.time`.
* increase versions of dependencies: julia>1.9 
* The core-entry is now required for (non-npy) FCIDUMP files in order to check whether the file is complete.
* DMRG and AtomsBase interface have been moved to extensions. In order to run DMRG, the `ITensors` package has to be installed and loaded; and in order to use `AtomsBase` interface, the `AtomsBase`, `Unitful` and `UnitfulAtomic` packages have to be installed and loaded. 

### Changed

* faster closed-shell and unrestricted CCSDT and DC-CCSDT implementations.
* ANO-RCC-MB basis is now used as the minimal AO basis for the SAD orbital starting guess.
* memory buffers are now handled by functions and types in `Buffers` module.
* the precompilation is disabled for development versions.
* Buffers is moved to a separate package Buffers.jl.
* reduce memory demand in df-hf.
* remove `IterativeSolvers` dependency.
* `jkfit` basis now falls back to `def2-universal-jkfit` if not found (e.g. for avXz basis sets).

### Added

* a macro `@dummy` has been added to set some atoms to dummy atoms. 
* a `neuralize` function to trick `Base.mightalias` in `TensorOperations` for reshaped-buffer arrays.
* `Buffer` and `ThreadsBuffer` types and `alloc!`, `drop!`, `reset!`, `reshape_buf!` functions.
* `BasisBatcher` structure to calculate 3-index integrals in batches.
* `@dfmp2` to calculate df-mp2 energy without storing integrals.
* `@freeze_orbs` also accepts now a string of indices using the +/- or :/; syntax ([#186])

### Fixed

* Improve parsing method names in macros. Now the parser is not confused by multiple dashes in the name and, e.g., `@cc svd-dc-ccsdt` is evaluated correctly.
* SAD orbital guess for Li and Be has been fixed.
* molden export functionality has been fixed.
* thread-safe handling of buffers using ThreadsBuffer.
* if some of the npy files are not found, the integrals are read from the fcidump file ([#250])

## Version [v0.13.1] - 2024.07.11

### Added

* Store MP2 amplitudes in `cc_amplitudes_` files in `@cc mp2` calculations.

### Fixed

* Fix `maxit=0` case for `cc` calculations. 

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
