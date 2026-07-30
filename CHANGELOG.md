# Release notes

## Unreleased Version

### Breaking

### Changed

* **Integrals are owned by whoever creates them** (behavioral change). Density-fitted MO integrals depend on the orbitals, but a set generated implicitly by `@cc`/`@fci`/`@ciphi` used to persist for the rest of the session and be silently reused — including after the orbitals had changed (a second `@dfhf`, `@localize`, an orbital-optimized method). Only one special case was patched (`wf.dump=""`/`dump4core_only` restarts forced a regeneration). Now a driver that has to create the integrals itself builds them from the current correlation orbitals and deletes them when it returns, so a stale set cannot be reused; integrals created explicitly with `@dfints` (or read from an FCIDUMP) persist and are the user's to refresh. `setup_fcidump_if_needed!` and its restart special case are gone. Scripts that relied on one implicit generation serving several `@cc` calls now repeat that generation (cheap next to the CC iterations) — insert `@dfints` to keep the previous caching. Unchanged: `wf.dump=""` and `wf.dump4core_only` remain reference-orbital selectors for restarts, independent of where the integrals come from. Every correlated run now prints one line saying which integrals it uses (exact AO direct / transient MO dump / per-run DF / fcidump).
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

* **`@moints` — MO integrals from the exact (non-DF) AO integrals.** The non-DF counterpart of `@dfints`: it transforms the exact AO integrals to the MO basis of the current orbitals and keeps the resulting dump in `EC.fd` for the rest of the session (generating the AO integrals first if they are not on file yet, as `@ints` would). Previously the exact-AO route could only produce a *transient* MO dump, derived and discarded inside a driver run, so `@write_ints` had nothing to write and repeated runs each retransformed. Like the density-fitted dump the result covers the active space (frozen core folded into `int0`/`int1`, deleted/frozen virtuals dropped) and follows the "whoever creates them, owns them" rule — it persists and is the user's to refresh when the orbitals change. Note that AO-direct methods (MP2/CCSD/DCSD and friends) are *faster* without it, since they never form the MO integrals at all.
* **libcint `CINTOpt` optimizer for the exact-AO 4-index generation.** The 2-electron libcint wrappers passed a NULL optimizer, so every quartet call recomputed the shell-pair setup data. The generation sweeps (`pm_integrals!` via `calc_2e4idx_tri_blockwise!`, incl. the Schwarz-bounds pass) now allocate a per-basis `CIntOpt` once, share the read-only handle across all threads, and free it deterministically after the sweep. Honest measured gain: **1.06-1.07x** on the ERI kernel across cc-pVDZ/TZ/QZ water dimers -- far below the ~2x sometimes quoted for `CINTOpt`, because for generally-contracted correlation-consistent sets the quartet arithmetic dominates the pair-data setup the optimizer caches (segmented/low-angular-momentum bases would gain more). Values shift only at the ulp level (~1e-15, libcint-internal summation paths); pinned by a unit test at 1e-13.
* **Exact-AO generation computes only the stored block-triangle** (Hermiticity row cut). The ± supermatrix store keeps the lower block-triangle of `Vs/Va[tri(mu nu), tri(rho sigma)]` (the upper part is the conj-Hermitian mirror reconstructed at read time), but the generation used to compute every bra pair and let the writer discard the mirror rows. Now the ERI kernel skips the bra shell pairs that lie entirely below a block's row floor (`max(p,q) >= first(sigma-block)` is the exact kept-pair predicate; blocks are shell-aligned, so the shell-level skip is exact, not conservative) and the ± fold is restricted to the kept rows. The store is BIT-identical; roughly half the libcint quartets and fold flops disappear on every system, compact ones included (unlike screening). Measured (min of 3, 8 threads): water dimer cc-pVQZ (nao=230) 33.1 s -> 19.2 s (1.72x), 1 water cc-pVTZ 1.36x, 4-water chain cc-pVTZ 1.50x unscreened / 1.23x on top of screening (sub-multiplicative: screening already removes far quartets). Small systems are block-granularity-limited (few sigma-blocks -> coarse floors, kept fraction 0.60 at nao=58 vs the 0.51 ideal at nao=230). `pm_integrals!(...; rowcut=false)` restores the full computation (testing).
* **Cauchy-Schwarz prescreening of the exact AO integrals** (`int.screen`, default `1e-12`). The four-index generation previously computed every shell quartet. It now precomputes the shell-pair bounds `Q[P,R] = sqrt(max|(pr|pr)|)` (`nshell^2/2` diagonal quartets, i.e. free next to the `nshell^4` generation) and skips any quartet with `Q[P,R]*Q[Q,S] < int.screen`, storing it as an exact zero. Set `@set int screen=0` to compute every quartet. The gain is entirely a function of how extended the system is - a single compact molecule has no negligible quartet at this threshold and is unaffected (measured 1.02x, i.e. the screening overhead is nil), while for separated fragments it grows quickly: two waters 15 bohr apart 1.50x, a four-water chain (nao=232) **2.37x**, a six-water chain **2.59x**, with 50-88% of quartets skipped. This is the usual `N^4 -> N^2` asymptotic effect, so it matters most exactly where the integrals are expensive.
* `@ints`/`@hf` now announce the exact-AO integral generation (basis-function count and the resulting +/- store size) and report the time taken, instead of appearing to hang silently on large cases.
* Memory-management utilities and options. `available_memory()` estimates how much memory may reasonably be used for a large scratch allocation as a fraction (default `0.8`) of the currently available memory — the minimum of the node's free memory (`≈ MemAvailable`, so reclaimable page cache counts as free), any enforced **cgroup** budget (`cgroup_memory_available`, covering SLURM `ConstrainRAMSpace`, containers and k8s — cgroup v1 and v2), and any **SLURM** per-job limit (`slurm_memory_limit` from `SLURM_MEM_PER_NODE`/`SLURM_MEM_PER_CPU`) — after a `GC.gc(true)` so Julia's own freed memory is reflected. `available_memory(EC)` honors a user budget. New `mem` options: `@set mem budget=<GB>` sets an explicit budget (`≤0` = automatic, the default) and `@set mem fraction=<f>` tunes the fraction used in automatic mode. Currently used to size the blocked 4-index integral transformation (so fat/constrained nodes get the right number of passes over the integrals); intended for broader memory management across the package.
* **Non-DF (exact) AO integrals and AO-direct coupled cluster** (issue #290). Exact four-index AO electron-repulsion integrals can now be generated and used without density fitting. `@ints` writes the AO integrals to scratch files (the ± supermatrix store, `S_AA`, `h_AA`); `@hf`/`@uhf` build HF/UHF from them (the AO Fock is assembled by streaming the triangular integrals one slab at a time in `O(nao²)` memory, exploiting the `r≤s` packing — the full `nao⁴` integral tensor is never formed), and closed- **and open-shell** MP2/CCSD/DCSD run **AO-direct**: the frozen core is folded into an effective one-electron Hamiltonian and both the T1-dressed integrals and the `kext` 4-external term are contracted straight from the AO integrals each iteration (occ-early one pass, no full MO integral set is ever formed). The higher methods now run AO-direct as well, off a persisted half-transformed integral store (see below); only FCI still retransforms the AO integrals to a transient MO dump per call. The `EC.fd` FCIDUMP object is now strictly MO-only. Set `@set int df=false` to default new calculations to the exact-AO route; `@set int ao_direct=false` forces the derived-MO-dump route.
* **Persisted ± (plus/minus) supermatrix store for the exact AO integrals** (`PMStore` module). The two ±-symmetrized integral combinations `Vs/Va[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ ± ⟨νμ|ρσ⟩` are symmetric (real) / Hermitian (complex) matrices over the packed pair space; only their lower **block**-triangle is stored as dense σ-aligned column panels (`ao_pm_s`/`ao_pm_a`, ≈ `n⁴/4`). The ± fold is fused into the ERI generation — each shell-aligned ket-column block is assembled in a bounded RAM slab, folded and written directly as panels — so no jointly packed `n⁴/2` array is formed at any point. **This is now the only representation of the exact AO integrals**: the previously alternative jointly packed `ao_int2` file and every `if joint … else ± …` branch behind it are gone, leaving exactly two integral paths — an MO FCIDUMP (`EC.fd`, triangular) or the AO ± store. The `int.ao_pm` and `cc.use_pm_kext` options were removed with it.
  * The ± split is what makes **both** index pairs triangular at once: `⟨pq|rs⟩ ≠ ⟨qp|rs⟩`, so the bra pair cannot be packed on its own — only the joint symmetry `⟨pq|rs⟩ = ⟨qp|sr⟩` exists, and separating it into a symmetric and an antisymmetric supermatrix turns that one joint symmetry into two independent triangular packings (which is also why the stored object can be `n⁴/4` rather than `n⁴/2`). The exact flop ratio against the MO contraction, `2no/((n+1)(o+1))`, is the product of three effects: the bra pair packed `n² → n(n+1)/2` (×1.99 at `n=232`), two GEMMs instead of one (÷2), and the occupied pair packed `o² → o(o+1)/2` (×1.90 at `o=20`) — the last available only because the ± form supplies both `(pq)` orders, so that `K2[q,p,j,i] = K2[p,q,i,j]` can be used.
  * Measured **2.15–2.35× on the kext** against the MO path at (H₂O)₄/cc-pVTZ (nao 232, 20 correlated occupied), with the same 2× on disk. The advantage is governed by the number of **correlated occupied** orbitals, not by the basis size: it needs a right-hand side (`ntri_oo = nocc(nocc+1)/2`) wide enough for the panel GEMM to reach peak, so at `nocc ≲ 8` the ± kext is no faster (the GEMM then runs at ~300–370 of ~715 GFLOP/s) even for a large basis — benchmarks below `nocc ≈ 12` are not representative.
  * All AO consumers run on the store: the `kext` 4-external term as zero-copy panel GEMMs, the T1-dressing occ-early sweeps and the AO-HF/frozen-core Fock builders at halved integral streaming, the open-shell αβ `kext` via an explicit ±-fold of the αβ density, and the AO→MO transform for the derived dump (`pm_transform`, straight from the panels). Complex integrals are supported throughout (pair-internal swaps are conjugation-free ± signs; the block mirror conjugates).
  * Three per-iteration costs were removed: the store handle is **kept open across CC iterations** instead of being re-mapped every iteration (re-mapping re-faulted the whole store through the page table — ~0.25 s per call for a 6 GB store, 21–26 % of the kext contraction); the ± fold and the four-quadrant scatter around the panel GEMMs are **fused and threaded** (bit-identical output, kext contraction 1.07–1.08 × and its non-GEMM share 11.5 % → 1.7 %, per-call allocation down 21 % / 40 % for the closed- and open-shell variants); and the `rs` blocking of the *MO-path* 4-external contraction now grows with the width of the result instead of being fixed at 128, which streamed the result array `nrs/128` times (1.30–1.35× for ≥ 17 correlated occupied orbitals; narrower cases keep the previous, bit-identical blocking).
* **AO-direct higher methods via a persisted half-transformed integral store.** The one-index→occupied half-transform `⟨iν|ρσ⟩ = Σ_μ ⟨μν|ρσ⟩ C_occ[μ,i]` is the universal downstream intermediate for the correlated methods, and its bra-occupied transform is T1-independent — so it is built **once** per orbital set (per spin) from the ± store and streamed with single-threaded `pread` + `posix_fadvise(WILLNEED)` (measured ≈4.9 GB/s, at half the disk of a full `n⁴/2` store), replacing per-iteration re-streaming of the whole integral set. Off this store the higher methods now run **AO-direct** — with no transient MO dump: standalone **MP2/UMP2/RMP2**; **CCSD(T)/DCSD(T)** and **UCCSD(T)/UDCSD(T)**; the **Lagrange (Λ) equations** and correlated properties, closed-shell (**Λ-CCSD/Λ-DCSD**, **ΛCCSD(T)/ΛDCSD(T)**) and unrestricted (**ΛUCCSD/ΛUDCSD**, **ΛUCCSD(T)/ΛUDCSD(T)**), including the 1-RDM/dipole; and **EOM-CCSD/EOM-UCCSD** excitation energies. Each MO integral block a method needs (they all carry ≥1 occupied index) is a cheap GEMM-only transform of the stored half-transform, written mmapped in the consumer's index order — a block whose occupied index sits on a ket is reached by particle-exchange/Hermiticity relabelings, and mixed-spin blocks simply put the other spin's coefficients on the free slots (the βα-looking reads of the unrestricted (T) resolve to the same five αβ blocks). The Λ 4-external term reuses the ± `kext` (`pm_K2!`/`pm_K2ab!`) by folding Λ2 with the dressed virtual bra, and the Λ general-orbital singles term — the virtual-occupied block of a generalized Fock, `2J−K` closed-shell, `J(D^α+D^β)−K(D^σ)` unrestricted — is read straight off the half-transformed store (`ht_jk_columns!`) instead of building the `nao×nao` Fock and keeping one block of it: the store's bra index *is* the occupied index this term carries, and its two roles supply both orderings with the virtual on the same ket-2 slot (the exchange via the conjugation-free particle symmetry `⟨mq|pe⟩ = ⟨qm|ep⟩`). One sequential pass reading each stored element once, `2·nocc·nao³` multiply-adds against the previous `nao⁴`, measured **9.4× (nao 114) to 61× (nao 264)** per call and 11.6× on a whole unrestricted Λ iteration; no `nocc·norb³` general-orbital block is ever formed. All match the derived-MO-dump reference to `≤1e-8…1e-14`. Basis sets with **linearly-dependent (deleted) orbitals** are handled too: the redundant orbitals are frozen out of the correlation treatment by the usual `freeze_orbitals!` machinery and the AO←MO map keeps only the active columns, so no special casing is needed (verified to machine precision against the derived dump for CCSD, CCSD(T), Λ-CCSD, EOM-CCSD, UCCSD and ΛUCCSD on a rank-deficient basis). The **doubles-only** methods run AO-direct as well — **CCD/DCD** and the quasi-variational **QV-CCD/QV-DCD** need only the bare blocks that the T1 dressing already yields for an empty T1 — and so do their **orbital-optimized** variants **OQV-CCD/OQV-DCD**: instead of re-transforming an MO integral dump every macro-iteration, the orbital rotation is folded into the MO coefficients and the integral blocks are rebuilt straight from the AO integrals in the rotated basis, with the general-orbital block of the orbital gradient (`⟨μν|ρi⟩`) kept in the **AO** basis — the gradient only ever contracts its three general indices with the virtual rotation, so it is indifferent to whether they are MO or AO. The half-transformed store is rebuilt for the rotated occupied orbitals each macro-iteration: its bra *is* the occupied space, so unlike the T1 dressing (which leaves the occupied bra invariant) it cannot be built once. The **Brueckner** variants **BQV-CCD/BQV-DCD** run AO-direct through the same machinery: they differ from the orbital-optimized ones only in what drives the rotation — the singles residual, extracted from the same H·T2 call, instead of the orbital gradient — which needs no integral block the AO path does not already build. FCI is the only remaining method that still derives a transient MO dump. Complex integrals are supported for the directly-transformed blocks; blocks that would need a bra↔ket Hermiticity swap are real-only (pending complex AO integrals).
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
* FCI and CIPHI work for non-Hermitian Hamiltonians.

### Fixed

* **`scf.maxit=0` crashed instead of building the Fock matrix.** With no iterations the SCF loop body never ran, so the Fock matrix the loops return was never assigned and every zero-iteration run died with `UndefVarError: fock` (all five loops: closed-shell, open-shell, positron DF-HF, BO-HF and BO-UHF). `scf.maxit=0` now means "do not iterate", not "do nothing": one pass builds the Fock matrix, the energy and the residual for the given orbitals and stops *before* the orbitals are updated, leaving them exactly as they came in. The orbital energies stored for such a run are the Fock expectation values `⟨p|F|p⟩` instead of zeros — with no update there are no eigenvalues to report, and the orbitals need not be canonical. Fed a converged set of orbitals (`wf.start`), a zero-iteration run reproduces the converged energy and orbital energies exactly, which is what makes it useful: it is one Fock build for orbitals you already have.
* **Converged HF orbitals are now canonical, so post-HF methods no longer pseudo-canonicalize.** An SCF iteration builds the Fock matrix, tests convergence and breaks, so the orbitals the loop left behind diagonalized the (DIIS-extrapolated) Fock matrix of the *previous* iteration — the final one kept occ-occ/virt-virt off-diagonal elements of the order of the remaining orbital gradient. Since `scf.thr` is compared against the *square* of that gradient's norm (`sum(abs2, S·D·F − F·D·S)`), the default `1e-10` leaves a gradient of ~`1e-5` and off-diagonal elements of ~`1e-6` — right at `cc.fock_diag_thr`, so `(T)` was tripped into a pseudo-canonicalization on ordinary converged HF orbitals (measured for water/cc-pVDZ: `1.1e-6` against a `1e-6` threshold, and the off-diagonality tracks `sqrt(scf.thr)` exactly). The loops now canonicalize within the occupied and within the virtual space once at convergence: those rotations leave the density — and therefore the energy, and the Fock matrix itself — exactly invariant, so this costs two small diagonalizations, cannot perturb the converged solution, and drives the off-diagonal elements to machine precision (`1.1e-6 → 2e-14`). Diagonalizing the *full* Fock matrix would not work, and is not what this does: its occ-virt mixing changes the density and re-seeds off-diagonal elements of the same order it removes (measured `1.1e-6 → 0.8e-6`). Applied only on the converged exit, since a `maxit`-exhausted loop leaves the orbitals one update ahead of the Fock matrix; linearly-dependent orbitals are excluded so their sentinel energies survive. As a side effect the stored orbital energies are now the eigenvalues of the Fock matrix that is stored with them, rather than of the previous iteration's.
* **`@write_ints` writes the actual electron count.** `wf.charge` is applied to whatever the integral source says (`nelec`/FCIDUMP/neutral system) at space-setup time, so an in-memory dump built by `@dfints`/`@moints` carries the *pre-charge* `NELEC` — and that count used to be exported verbatim. An FCIDUMP written from a charged calculation therefore described the neutral system, and reading it back gave the wrong number of electrons unless the reader happened to set the same `charge` again. `@write_ints` now applies `wf.charge` to the `NELEC` it writes (and `MS2` to the resulting parity, following the same rule `setup_space_fd!` uses), so the exported file is a self-contained description of the system; reading needs no `charge`, and setting one still ionizes relative to the file, as documented. `write_fcidump` gained the `charge` keyword this uses (default `0`, i.e. the dump's own `NELEC`; `fd` itself is never modified). Deliberately *not* applied to the `int.fcidump` output of `@dfints`: that file is parked on disk to be read back by the same session, where `wf.charge` still describes the molecule (the correlated-property path rebuilds the space from the system via `restore_system_space!`), so it keeps the count `setup_space_fd!` applies the charge to.
* `@write_ints` accepts a variable as the file name. The file argument was interpolated into the macro's expression without escaping, so it was resolved in the `ElemCo` module rather than in the caller's scope and `f = "FCIDUMP"; @write_ints f` failed with `UndefVarError: f not defined in ElemCo` (only string literals worked).
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
