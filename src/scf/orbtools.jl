"""
    OrbTools

A collection of tools for working with orbitals
""" 
module OrbTools
using LinearAlgebra, Printf

using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.MSystems
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions

export guess_orb, guess_pos_orb, load_orbitals, load_rotations, load_left_right_rotations
export orbital_energies, load_positron_orbitals 
export show_orbitals
export rotate_orbs, rotate_orbs!, normalize_phase!
export try_load_starting_orbitals
export left_from_right_rotations, project_onto_basis
export canonical_orthogonalization, select_lowdin_orth, opao_relthr, eigen_orth, n_redundant_orbitals
export orbital_classes_with_deleted, n_deleted_orbitals, freeze_orbitals!

"""
    REDUNDANT_ORBITAL_ENERGY

  Sentinel orbital energy assigned to the linearly-dependent (redundant) orbitals that
  the (DF-)HF projects out (see `eigen_orth`). It is far above any physical
  orbital energy and is used to identify these orbitals in the wavefunction dump,
  distinguishing them from ordinary frozen/deleted virtuals.
"""
const REDUNDANT_ORBITAL_ENERGY = 1.0e20

"""
    REDUNDANT_ENERGY_THR

  Energy cutoff used to identify the linearly-dependent (redundant) orbitals from the
  wavefunction dump: an orbital whose stored energy exceeds this value carries the
  [`REDUNDANT_ORBITAL_ENERGY`](@ref) sentinel (it is far above any physical orbital energy).
"""
const REDUNDANT_ENERGY_THR = 1.0e10

"""
    canonical_orthogonalization(sao::AbstractMatrix, redthr; verbose=false)

  Construct the canonical orthogonalization transformation from the AO overlap
  matrix `sao`. Eigenvectors of `sao` whose eigenvalues are below `redthr` are
  considered linearly dependent (redundant) and are removed from the orbital basis.
  This makes the SCF robust for redundant basis sets (e.g. Cartesian basis sets with
  6 instead of 5 `d` functions), where `sao` is (near-)singular.

  Return `(X, Xredundant)` where `X` (`nAO × nMO`) fulfills `X' sao X = I` and is used
  to solve the SCF equations in an orthonormal basis, and `Xredundant`
  (`nAO × nredundant`) spans the removed near-null space. The latter is kept only to
  pad the MO coefficient matrices to square shape (`nAO × nAO`).
"""
function canonical_orthogonalization(sao::AbstractMatrix, redthr; verbose=false)
  sval, svec = eigen(Hermitian(Array(sao)))
  keep = sval .> redthr
  nredundant = count(!, keep)
  if nredundant > 0 && verbose
    println("Redundant basis set: removed $nredundant of $(length(sval)) linearly-dependent ",
            "orbital(s) with AO overlap eigenvalue < $redthr")
    println("Smallest kept overlap eigenvalue: ", minimum(sval[keep]))
  end
  X = svec[:, keep] * Diagonal(inv.(sqrt.(sval[keep])))
  Xredundant = svec[:, .!keep]
  return X, Xredundant
end

"""
    select_lowdin_orth(S::AbstractMatrix; relthr=3e-8, verbose=false)

  Locality-preserving orthogonalizer of a Hermitian metric `S` (typically a projected
  atomic-orbital (PAO) overlap `S_PAO = C_PAO' S C_PAO`), used to build orthogonal PAOs
  (OPAOs) for the virtual space without redundancies and without delocalizing them.

  Three decoupled steps, each using the right tool:
  1. **Rank** `r` from the eigenvalues of `S` with a *relative* threshold: directions with
     eigenvalue `> relthr * λmax` are kept. With `relthr` tied to the basis redundancy
     threshold (`relthr = loc.opaofac * scf.redthr`, a conservative `~1e-8`) only directions
     that are (near-)redundant by the same standard the basis uses are dropped; small-but-real
     directions (e.g. the virtual residual of a frozen core AO, which sits at `~1e-7 λmax`,
     well above `redthr`) are kept as genuine degrees of freedom.
  2. **Selection** of exactly `r` *actual* (atom-centered) PAOs via the top-`r` pivots
     of a pivoted Cholesky of `S` — the `r` most linearly-independent columns. Pinning
     the count to `r` from step 1 avoids relying on a fragile pivot threshold.
  3. **Symmetric Löwdin** orthogonalization `S_sub^{-1/2}` of the selected (clean,
     well-conditioned) `r×r` block, which keeps each OPAO as close as possible to its
     parent PAO (maximally local; cf. [`canonical_orthogonalization`](@ref), which
     instead rotates into overlap eigenvectors and delocalizes). Löwdin handles the
     amplification of low-presence directions gracefully (unlike ALPACA's `sqrtinvchol`).

  Returns `M` (`size(S,1) × r`) with `M' S M = I`, a drop-in for
  `sqrtinvchol(S; max_rank=...)` in `C_OPAO = C_PAO * M`. `relthr` is the OPAO redundancy
  threshold, [`opao_relthr`](@ref)`(EC) = loc.opaofac * scf.redthr`.
"""
function select_lowdin_orth(S::AbstractMatrix; relthr::Real=3e-8, verbose::Bool=false)
  n = size(S, 1)
  A = Hermitian(Array(S))
  n == 0 && return similar(A, n, 0)
  λ = eigvals(A)                                # ascending real eigenvalues (no eigenvectors)
  λmax = λ[end]
  λmax <= 0 && return similar(A, n, 0)
  r = count(>(relthr * λmax), λ)
  r == 0 && return similar(A, n, 0)
  # No gap/cliff heuristic is needed: because `relthr` is tied to the conservative basis
  # redundancy threshold (`opaofac * scf.redthr`, ~1e-8), the cut sits below the real (even
  # diffuse) virtual directions but above genuine near-/exact redundancies. Extending the rank
  # down a smooth tail would instead risk pulling in sub-`redthr`, heavily amplified junk.
  # select the r most independent (atom-centered) columns via pivoted Cholesky
  keep = sort(cholesky(A, RowMaximum(), check=false).p[1:r])
  # symmetric Löwdin on the clean r×r block (sub-block taken directly from A)
  Fs = eigen(Hermitian(A[keep, keep]))
  Msub = Fs.vectors * Diagonal(inv.(sqrt.(Fs.values))) * Fs.vectors'
  M = zeros(eltype(Msub), n, r)
  M[keep, :] = Msub
  # One Löwdin refinement: when a kept low-presence direction is amplified (~1/√λ, e.g. the
  # virtual residual of a frozen core AO at ~1e-7·λmax) the r×r block is ill-conditioned, so
  # `M' A M` is off identity by ~√cond·ε (≈1e-10). That residual metric is ≈ I, so a second
  # symmetric orthonormalization is well-conditioned and restores machine-precision
  # `M' S_PAO M = I` without changing the rank or the (atom-centered) support of M.
  Fg = eigen(Hermitian(M' * A * M))
  M = M * (Fg.vectors * Diagonal(inv.(sqrt.(Fg.values))) * Fg.vectors')
  verbose && r < n && println("select_lowdin_orth: dropped $(n - r) redundant PAO(s) (relthr=$relthr)")
  return M
end

"""
    opao_relthr(EC::ECInfo)

  Effective relative redundancy threshold for the OPAO (orthogonalized-PAO) construction in
  [`select_lowdin_orth`](@ref): [`loc.opaofac`](@ref ECInfos.LocOptions) × [`scf.redthr`](@ref
  ECInfos.ScfOptions). Tying it to the AO basis redundancy threshold keeps the two consistent —
  only directions that are (near-)redundant by the *same* standard the basis uses are dropped —
  and basis-adaptive: lowering `scf.redthr` (a more complete basis) automatically loosens the
  PAO pruning.
"""
opao_relthr(EC::ECInfo) = EC.options.loc.opaofac * EC.options.scf.redthr

"""
    n_redundant_orbitals(EC::ECInfo)

  Return the number of linearly-dependent (redundant) orbitals of the AO basis set,
  i.e. the number of eigenvalues of the AO overlap matrix below `scf.redthr`.

  These orbitals are projected out by the (DF-)HF (see [`canonical_orthogonalization`](@ref))
  and parked as the highest (unoccupied) orbitals. Post-HF methods freeze them out exactly
  like [`freeze_nvirt`](@ref ECInfos.WfOptions), so they do not enter the correlation
  treatment. Returns `0` if no molecular system is set up (e.g. for a plain FCIDump).
"""
function n_redundant_orbitals(EC::ECInfo)
  isempty(EC.system) && return 0
  sao = overlap(generate_basis(EC, "ao"))
  _, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr)
  return size(Xredundant, 2)
end

"""
    orbital_classes_with_deleted(occ, norb, ndeleted)

  Build a vector of TREXIO orbital classes of length `norb` for storing in the
  wavefunction dump: `"Inactive"` for the occupied orbitals `occ`, `"Virtual"` for the
  remaining orbitals, and `"Deleted"` for the last `ndeleted` orbitals (the
  linearly-dependent orbitals projected out by the (DF-)HF, see
  [`canonical_orthogonalization`](@ref)).
"""
function orbital_classes_with_deleted(occ, norb, ndeleted)
  classes = fill("Virtual", norb)
  classes[occ] .= "Inactive"
  if ndeleted > 0
    classes[norb-ndeleted+1:norb] .= "Deleted"
  end
  return classes
end

"""
    n_deleted_orbitals(EC::ECInfo; MO="mo")

  Number of orbitals marked `"Deleted"` in the wavefunction dump, i.e. the
  linearly-dependent orbitals that the preceding (DF-)HF projected out. Post-HF methods
  freeze these out of the correlation treatment exactly like
  [`freeze_nvirt`](@ref ECInfos.WfOptions).

  The stored count is the authoritative number actually used by the HF. It is
  cross-checked against the present redundancy of the AO basis
  ([`n_redundant_orbitals`](@ref)) and a warning is issued if they disagree (e.g. when
  `scf.redthr` was changed between the HF and the correlation step). If the dump
  contains no class/energy information, the recomputed redundancy is used as a fallback.

  Redundant orbitals are identified as those both marked `"Deleted"` and parked at the
  sentinel energy `REDUNDANT_ORBITAL_ENERGY`; this distinguishes them from ordinary
  frozen virtuals, which are also stored as `"Deleted"` but keep their physical orbital
  energy.
"""
function n_deleted_orbitals(EC::ECInfo; MO="mo")
  nredund = n_redundant_orbitals(EC)
  classa, classb = fetch_orbital_classes(EC; MO=MO)
  ea, eb = fetch_orbital_energies(EC, MO)
  if isempty(classa) || isempty(ea) || length(ea) != length(classa)
    # no usable class/energy information stored (e.g. older dump or imported orbitals)
    return nredund
  end
  # any energy this large is the redundant-orbital sentinel, far above physical orbitals
  thr = REDUNDANT_ENERGY_THR
  count_redundant(cls, en) = count(i -> cls[i] == "Deleted" && en[i] > thr, eachindex(cls))
  ndel = count_redundant(classa, ea)
  if !isempty(classb) && length(eb) == length(classb)
    ndelb = count_redundant(classb, eb)
    ndelb == ndel || @warn "Number of deleted α ($ndel) and β ($ndelb) orbitals in the dump differ; using α."
  end
  if ndel != nredund
    @warn "Number of deleted orbitals stored in the dump ($ndel) does not match the redundancy of the current AO basis ($nredund). Using the stored count; check that scf.redthr is consistent with the HF run."
  end
  return ndel
end

"""
    _try_orbital_classes_energies(EC::ECInfo; MO="mo")

  Best-effort fetch of `(classa, classb, ea, eb)` from the dump, returning empty vectors when
  no class/energy information is available (e.g. an FCIDUMP-only run).
"""
function _try_orbital_classes_energies(EC::ECInfo; MO="mo")
  try
    classa, classb = fetch_orbital_classes(EC; MO=MO)
    ea, eb = fetch_orbital_energies(EC, MO)
    return classa, classb, ea, eb
  catch
    return String[], String[], Float64[], Float64[]
  end
end

"""
    _remove_orbitals_spin!(EC::ECInfo, occ_a, occ_b, virt_a, virt_b)

  Remove the given (spin-resolved) occupied and virtual orbital indices from the active
  subspaces and rebuild the derived spin spaces (`d`/`s`/`S`), mirroring `setup_space!`.
"""
function _remove_orbitals_spin!(EC::ECInfo, occ_a, occ_b, virt_a, virt_b)
  SP = EC.space
  setdiff!(SP['o'], occ_a); setdiff!(SP['O'], occ_b)
  setdiff!(SP['v'], virt_a); setdiff!(SP['V'], virt_b)
  if haskey(SP, 'a')
    for s in (occ_a, occ_b, virt_a, virt_b)
      setdiff!(SP['a'], s)
    end
  end
  SP['d'] = intersect(SP['o'], SP['O'])
  SP['s'] = setdiff(SP['o'], SP['d'])
  SP['S'] = setdiff(SP['O'], SP['d'])
  return
end

"""
    freeze_orbitals!(EC::ECInfo; MO="mo", redundant=true, verbose=true) -> (; occ_a, occ_b, virt_a, virt_b)

  Apply the full frozen-core, redundant- and deleted-virtual selection for a correlated
  calculation, honoring the orbital classes stored in the dump (e.g. from
  [`@region`](@ref ElemCo.@region)) while letting the user override them:

  - Occupied core: if `wf.freeze_nocc ≥ 0`, freeze that many lowest orbitals; otherwise if
    `wf.core == :auto`, freeze the orbitals tagged `"Core"` in the dump (falling back to the
    `:large` chemical core when the dump carries no class information); otherwise freeze the
    chemical core selected by `wf.core`.
  - Virtuals: the linearly-dependent (redundant) orbitals and the (e.g. `@region`) deleted
    virtuals are both stored as class `"Deleted"`. If `wf.freeze_nvirt < 0` (auto), every
    `"Deleted"` orbital is frozen by its *actual* index (never by top index), so a region's
    active virtuals are never frozen by mistake; with no class info the recomputed AO redundancy
    ([`n_redundant_orbitals`](@ref)) is frozen as the highest virtuals. If `wf.freeze_nvirt ≥ 0`,
    exactly that many highest virtuals are frozen (the user decides), with a warning when this
    leaves redundant orbitals in the correlation space.

  Pass `redundant=false` when the redundant orbitals were already excluded (e.g. a 3-index
  FCIDUMP). Returns the number of occupied/virtual orbitals removed, resolved by spin.
"""
function freeze_orbitals!(EC::ECInfo; MO="mo", redundant::Bool=true, verbose=true)
  SP = EC.space
  no_a0, nv_a0 = length(SP['o']), length(SP['v'])
  no_b0, nv_b0 = length(SP['O']), length(SP['V'])
  core = EC.options.wf.core
  freeze_nocc = EC.options.wf.freeze_nocc
  freeze_nvirt = EC.options.wf.freeze_nvirt
  classa, classb, ea, eb = _try_orbital_classes_energies(EC; MO=MO)
  have_classes = !isempty(classa) && length(ea) == length(classa)
  has_beta = !isempty(classb) && length(eb) == length(classb)

  # ---- occupied frozen core ----
  if freeze_nocc >= 0
    freeze_core!(EC, :large, freeze_nocc; verbose=verbose)                  # explicit count
  elseif core == :auto && have_classes && any(==("Core"), classa)
    core_a = findall(==("Core"), classa)
    core_b = has_beta ? findall(==("Core"), classb) : core_a
    _remove_orbitals_spin!(EC, core_a, core_b, Int[], Int[])
    if verbose
      println("Freezing ", length(core_a), " core orbital(s) from the dump's stored classes")
      println()
    end
  else
    freeze_core!(EC, core == :auto ? :large : core, -1; verbose=verbose)    # chemical core
  end

  # ---- virtual freezing ----
  # The linearly-dependent (redundant) orbitals and the (e.g. @region) deleted virtuals are both
  # stored as class "Deleted". Number of redundant ones, for the override warning below.
  nredundant = !redundant ? 0 :
    have_classes ? count(i -> classa[i] == "Deleted" && ea[i] > REDUNDANT_ENERGY_THR, eachindex(classa)) :
    n_redundant_orbitals(EC)
  if freeze_nvirt < 0
    # auto: freeze every "Deleted" orbital (redundant + deleted) by its actual index, so a
    # region's active virtuals are never frozen by mistake
    if redundant && have_classes
      del_a = findall(==("Deleted"), classa)
      del_b = has_beta ? findall(==("Deleted"), classb) : del_a
      if !isempty(del_a) || !isempty(del_b)
        _remove_orbitals_spin!(EC, Int[], Int[], del_a, del_b)
        verbose && println("Freezing ", length(del_a), " deleted (incl. linearly-dependent) virtual orbital(s)\n")
      end
    elseif redundant && nredundant > 0
      # no class/energy info (e.g. imported orbitals): freeze the recomputed redundancy (highest virtuals)
      freeze_nvirt!(EC, nredundant; verbose=false)
      verbose && println("Freezing ", nredundant, " deleted (linearly-dependent) orbital(s) for the correlation treatment\n")
    end
  else
    # explicit: freeze exactly freeze_nvirt highest virtuals — the user decides
    freeze_nvirt < nredundant &&
      @warn "freeze_nvirt ($freeze_nvirt) is smaller than the number of redundant orbitals ($nredundant); $(nredundant - freeze_nvirt) linearly-dependent orbital(s) remain in the correlation space."
    freeze_nvirt!(EC, freeze_nvirt; verbose=verbose)
  end

  return (; occ_a = no_a0 - length(SP['o']), occ_b = no_b0 - length(SP['O']),
            virt_a = nv_a0 - length(SP['v']), virt_b = nv_b0 - length(SP['V']))
end

"""
    eigen_orth(fock::AbstractMatrix, X::AbstractMatrix, Xredundant::AbstractMatrix; large=1.0e6)

  Solve the generalized eigenvalue problem `fock C = sao C ϵ` in the orthonormal basis
  defined by the canonical orthogonalization `X` (see [`canonical_orthogonalization`](@ref)),
  i.e. diagonalize `X' fock X` and back-transform. The redundant directions `Xredundant`
  are appended as virtual orbitals with energy `large`, so that they remain unoccupied
  and the returned coefficient matrix stays square (`nAO × nAO`).

  Return `(ϵ, cMO)` (orbital energies and coefficients).
"""
function eigen_orth(fock::AbstractMatrix, X::AbstractMatrix, Xredundant::AbstractMatrix; large=REDUNDANT_ORBITAL_ENERGY)
  fock_orth = X' * Hermitian(fock) * X
  ϵ, C = eigen(Hermitian(fock_orth))
  cMO = X * C
  if size(Xredundant, 2) > 0
    cMO = hcat(cMO, Xredundant)
    ϵ = vcat(ϵ, fill(convert(eltype(ϵ), large), size(Xredundant, 2)))
  end
  return ϵ, cMO
end

"""
    guess_hcore(EC::ECInfo)

  Guess MO coefficients from core Hamiltonian.
"""
function guess_hcore(EC::ECInfo)
  hsmall = load(EC, "h_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  X, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr)
  ϵ, cMO = eigen_orth(hsmall, X, Xredundant)
  return SpinMatrix(cMO)
end

"""
    guess_pos_hcore(EC::ECInfo)

  Guess MO coefficients for positron from core Hamiltonian.
"""
function guess_pos_hcore(EC::ECInfo)
  hsmall = load(EC, "h_positron_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  X, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr)
  ϵ, cMO = eigen_orth(hsmall, X, Xredundant)
  return SpinMatrix(cMO)
end
  
"""
    guess_sad(EC::ECInfo)
  
  Guess MO coefficients from atomic densities.
"""
function guess_sad(EC::ECInfo)
  bminao = generate_minao_basis(EC, EC.options.scf.minao)
  bao = generate_basis(EC, "ao")
  smin2ao = overlap(bminao, bao)
  smin = overlap(bminao)
  eldist = electron_distribution(EC.system, bminao)
  sao = load(EC, "S_AA", Val(2))
  denao = smin2ao' * diagm(eldist./diag(smin)) * smin2ao
  X, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr)
  eigs, cMO = eigen_orth(-denao, X, Xredundant)
  return SpinMatrix(cMO)
end

function guess_gwh(EC::ECInfo)
  error("not implemented yet")
  return SpinMatrix()
end

"""
    guess_positron(EC::ECInfo)

  Initialize positron MO coefficients as zeroes.
"""
function guess_positron(EC::ECInfo)
  hsmall = load(EC, "h_positron_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  X, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr)
  ϵ, cMO = eigen_orth(hsmall, X, Xredundant)
  return SpinMatrix(cMO)
end

"""
    guess_orb(EC::ECInfo, guess::Symbol)

  Calculate starting guess for MO coefficients.
  Type of initial guess for MO coefficients is given by `guess`.

  See [`ScfOptions.guess`](@ref ECInfos.ScfOptions) for possible values.
"""
function guess_orb(EC::ECInfo, guess::Symbol)
  if guess == :HCORE || guess == :hcore
    return guess_hcore(EC)
  elseif guess == :SAD || guess == :sad
    return guess_sad(EC)
  elseif guess == :GWH || guess == :gwh
    return guess_gwh(EC)
  elseif guess == :ORB || guess == :orb
    return load_orbitals(EC)
  else
    error("unknown guess type")
    return SpinMatrix()
  end
end

"""
  guess_pos_orb(EC::ECInfo, guess::Symbol)

  Calculate starting guess for MO positron coefficients.
  Type of initial guess for MO coefficients is given by `guess`.

  See [`ScfOptions.guess`](@ref ECInfos.ScfOptions) for possible values.

"""
function guess_pos_orb(EC::ECInfo, guess::Symbol)
  if guess == :HCORE || guess == :hcore
    return guess_pos_hcore(EC)
  elseif guess == :ORB || guess == :orb
    return load_positron_orbitals(EC)
  else
    error("unknown guess type")
    return SpinMatrix()
  end
end

"""
    load_orbitals(EC::ECInfo; start::Bool=false)

  Load (last) orbitals from file [`WfOptions.dump`](@ref ECInfos.WfOptions). 

  If `start=true`, load from `wf.start` instead.
  If the basis has changed, the orbitals will be projected onto the new basis.
  Returns `::SpinMatrix`. 
"""
function load_orbitals(EC::ECInfo; start::Bool=false)
  cMO, type, basis = fetch_orbitals(EC; start=start)
  current_basis = generate_basis(EC, "ao")
  return project_onto_basis(cMO, basis, current_basis; check=true, redthr=EC.options.scf.redthr)
end

"""
    try_load_starting_orbitals(EC::ECInfo) -> (SpinMatrix, Bool)

  Try to load starting orbitals from `wf.start` file.

  If `wf.start` is set, load and project orbitals from that file.
  
  Returns `(cMO, loaded)` where `loaded` indicates if orbitals were successfully loaded.
  If no orbitals are available, returns `(SpinMatrix{Float64}(), false)`.
"""
function try_load_starting_orbitals(EC::ECInfo)
  if EC.options.wf.start == "" 
    return SpinMatrix{Float64}(), false
  end
  if has_dumpfile(EC; start=true)
    println("Loading starting orbitals from ", EC.options.wf.start, " ...")
    cMO = load_orbitals(EC; start=true)
    return cMO, true
  end
  println("Warning: Start file ", EC.options.wf.start, " not found.")
  return SpinMatrix{Float64}(), false
end

"""
    left_from_right_rotations(cMOr::SpinMatrix)

  Calculate left biorthogonal rotation coefficients from right BO coefficients.
"""
function left_from_right_rotations(cMOr::SpinMatrix{T}) where {T}
  if is_restricted(cMOr)
    cMOl = SpinMatrix(transpose(inv(cMOr[1])))
    restrict!(cMOl)
  else
    cMOl = SpinMatrix{T}()
    for ispin = 1:2
      cMOl[ispin] = transpose(inv(cMOr[ispin]))
    end
  end
  return cMOl
end

"""
    load_rotations(EC::ECInfo)

  Load (last) orbital rotations from file [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Returns `::SpinMatrix`. 
"""
function load_rotations(EC::ECInfo)
  cRot, type = fetch_rotations(EC)
  if !is_rotation(type)
    error("Dump file does not contain orbital rotations")
  end
  return cRot
end

"""
    load_left_right_rotations(EC::ECInfo) -> (left::SpinMatrix, right::SpinMatrix)

  Load (last) left and right orbital rotations from file [`WfOptions.dump`](@ref ECInfos.WfOptions).

  If the type of the rotations does not contain the word `biorthogonal`, 
  the same rotation is returned for left and right (can be checked with `===`).
"""
function load_left_right_rotations(EC::ECInfo)
  cRot, type = fetch_rotations(EC)
  if !is_rotation(type)
    error("Dump file does not contain orbital rotations")
  end
  if is_biorthogonal(type)
    cRotL = left_from_right_rotations(cRot)
    return cRotL, cRot
  else
    return cRot, cRot
  end
end

"""
    load_positron_orbitals(EC::ECInfo)

  Load (last) positron orbitals from file [`WfOptions.dump`](@ref ECInfos.WfOptions).
  
  Returns `::SpinMatrix`. 
"""
function load_positron_orbitals(EC::ECInfo)
  cMO, type, basis = fetch_orbitals(EC; MO="po")
  current_basis = generate_basis(EC, "ao")
  return project_onto_basis(cMO, basis, current_basis; check=true, redthr=EC.options.scf.redthr)
end

"""
    orbital_energies(EC::ECInfo, spincase::Symbol=:α)

  Return orbital energies for a given `spincase`∈{`:α`,`:β`}.
"""
function orbital_energies(EC::ECInfo, spincase::Symbol=:α)
  if spincase == :α
    eps = load1idx(EC, "e_m")
    ϵo = eps[EC.space['o']]
    ϵv = eps[EC.space['v']]
  elseif spincase == :β
    eps = load1idx(EC, "e_M")
    ϵo = eps[EC.space['O']]
    ϵv = eps[EC.space['V']]
  elseif spincase == :p
    eps = load1idx(EC, "e_p")
    ϵo = eps[EC.space['p']]
    ϵv = eps[EC.space['e']]
  else
    error("orbital_energies: unknown spin case: $spincase")
  end
  return ϵo, ϵv
end

"""
    rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)

  Rotate orbitals `orb1` and `orb2` from [`WfOptions.dump`](@ref ECInfos.WfOptions) 
  by `angle` degrees. For unrestricted orbitals, `spin` can be `:α` or `:β`.
"""
function rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)
  cMO, descr = fetch_rotations(EC)
  basis = BasisSet()
  if !is_rotation(descr)
    cMO, descr, basis = fetch_orbitals(EC)
  end
  if is_restricted(cMO)
    cMOrot = cMO[1]
  else
    cMOrot = cMO[spin]
  end
  rotate_orbs!(cMOrot, orb1, orb2, angle)
  descr *= " rot$(orb1)&$(orb2)by$(angle)"
  if isempty(basis)
    dump_rotations(EC, cMO; type=descr)
  else
    dump_orbitals(EC, cMO; basis=basis, type=descr)
  end
end

"""
    rotate_orbs!(cMO::Matrix, orb1, orb2, angle=90)

  Rotate orbitals `orb1` and `orb2` from `cMO` by `angle` degrees.

  `cMO` is a matrix of MO coefficients.
"""
function rotate_orbs!(cMO::Matrix, orb1, orb2, angle=90)
  if orb1 > size(cMO,2) || orb2 > size(cMO,2)
    error("orbital index out of range")
  end
  if orb1 == orb2
    error("orbital indices must be different")
  end
  cMO[:,[orb1,orb2]] = cMO[:,[orb1,orb2]] * [cosd(angle) -sind(angle); sind(angle) cosd(angle)]
end

"""
    show_orbitals(EC::ECInfo, range=nothing)

  Print the MO coefficients in [`WfOptions.dump`](@ref ECInfos.WfOptions) 
  with respect to the atomic orbitals.
  
  `range` is a range of molecular orbitals to be printed.
"""
function show_orbitals(EC::ECInfo, range=nothing)
  cMO, descr, basis = fetch_orbitals(EC)
  if isnothing(range)
    range = 1:size(cMO, 2)
  end
  println(range," orbitals from $descr")
  if is_restricted(cMO)
    show_orbitals(EC, cMO[1], basis, range)
  else
    println("Alpha orbitals:")
    show_orbitals(EC, cMO[1], basis, range)
    println("Beta orbitals:")
    show_orbitals(EC, cMO[2], basis, range)
  end
end

"""
    show_orbitals(EC::ECInfo, cMO::Matrix, basis::BasisSet, range=1:size(cMO,2)

  Print the MO coefficients in `cMO` with respect to the atomic orbitals in `basis`.

  `range` is a range of molecular orbitals to be printed.
"""
function show_orbitals(EC::ECInfo, cMO::Matrix, basis::BasisSet, range=1:size(cMO,2))
  aos = ao_list(basis)
  nao = length(aos)
  nmo = size(cMO,2)
  nlargest = EC.options.wf.print_nlargest
  thr = EC.options.wf.print_thr
  @assert size(cMO,1) == nao "Wrong number of atomic orbitals in cMO: $(size(cMO,1)) vs $(nao)"
  for imo in range
    @assert imo in 1:size(cMO,2) "Wrong range of orbitals: $(range). Number of orbitals: $(nmo)"
    print("$imo: ")
    # get nlargest coefficients (round to 4 digits to avoid numerical noise)
    idx = argmaxN(cMO[:,imo], nlargest, by=x->round(abs(x),digits=4))
    for iao in idx
      if abs(cMO[iao,imo]) < thr
        continue
      end
      @printf("% .3f", cMO[iao,imo])
      print("(")
      print_ao(aos[iao], basis)
      print(") ")
    end
    println()
  end
end

"""
    normalize_phase!(cMO)

  Normalize the phase of the MO coefficients in `cMO`.

  The phase is chosen such that the first largest coefficient is positive.
"""
function normalize_phase!(cMO)
  nmo = size(cMO,2)
  for imo in 1:nmo
    maxao = argmaxN(cMO[:,imo], 1, by=x->round(abs(x),digits=4))[1]
    c = cMO[maxao,imo]
    if eltype(cMO) <: Complex
      # rotate phase so that the largest coefficient is real and positive
      phase = c / abs(c)
      cMO[:,imo] ./= phase
    elseif c < 0
      cMO[:,imo] .= -cMO[:,imo]
    end
  end
end

"""
    project_onto_basis(cMO::SpinMatrix, old_basis::BasisSet, new_basis::BasisSet; check=false, redthr=1.0e-8)

  Project the MO coefficients onto a new basis.

  The projector ``S_{new}^{-1} S_{new,old}`` uses the Moore-Penrose pseudo-inverse of the
  new-basis overlap, obtained from the canonical orthogonalization
  (see [`canonical_orthogonalization`](@ref)): with `X' S_{new} X = I` after dropping
  overlap eigenvalues below `redthr`, ``S_{new}^{-1} = X X'``. This handles redundant /
  linearly-dependent bases (singular ``S_{new}``) and reduces to the ordinary inverse
  when the basis is not redundant.

If `check` is true, the function will check whether the projection is needed and return the same
array `cMO` if it is not (i.e., it can be checked with `===`).
"""
function project_onto_basis(cMO::SpinMatrix, old_basis::BasisSet, new_basis::BasisSet; check=false, redthr=1.0e-8)
  SAO = overlap(new_basis)
  X, = canonical_orthogonalization(SAO, redthr)  # X X' is the pseudo-inverse of SAO
  proj = (X * X') * overlap(new_basis, old_basis)
  if check && SAO*proj ≈ SAO
    return cMO
  end
  if is_restricted(cMO)
    cMO_new = proj * cMO[1]
    return SpinMatrix(cMO_new)
  else
    cMO_newα = proj * cMO[1]
    cMO_newβ = proj * cMO[2]
    return SpinMatrix(cMO_newα, cMO_newβ)
  end
end

end #module
