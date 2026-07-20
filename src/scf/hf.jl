"""
Closed- and open-shell Hartree-Fock, both density-fitted (`dfhf`/`dfuhf`) and
exact non-DF AO (`hf`), sharing a common SCF loop with a pluggable Fock builder.
"""
module HF
using LinearAlgebra
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.Integrals
using ..ElemCo.MSystems
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.OrbTools
using ..ElemCo.IntegralTools
using ..ElemCo.PMStore
using ..ElemCo.FciDumps
using ..ElemCo.FockFactory
using ..ElemCo.Properties
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, dfhf_positron, dfuhf, hf, uhf

"""
    scf_thren(EC)

  Energy-convergence threshold for the SCF: the user-set `scf.thren` if non-negative,
  otherwise a default derived from the residual threshold (`sqrt(scf.thr)*0.1`).
"""
function scf_thren(EC::ECInfo)
  thren = EC.options.scf.thren
  return thren < 0.0 ? sqrt(EC.options.scf.thr)*0.1 : thren
end

"""
    canonical_orth_for_occ(EC, sao; open_shell)

  Canonical orthogonalization of the AO overlap `sao` (handling linearly-dependent basis
  sets) together with the guard that enough orbitals survive to host the occupied space.
  Returns `(Xorth, Xredundant)`.
"""
function canonical_orth_for_occ(EC::ECInfo, sao; open_shell::Bool)
  SP = EC.space
  Xorth, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr; verbose=true)
  nocc = open_shell ? max(length(SP['o']), length(SP['O'])) : length(SP['o'])
  @assert size(Xorth, 2) ≥ nocc "Too many linearly-dependent orbitals removed: only $(size(Xorth,2)) orbitals left for $nocc occupied. Lower scf.redthr."
  return Xorth, Xredundant
end

"""
    redundant_orbital_classes(EC, Xredundant)

  Closed-shell orbital classes (with deleted/redundant orbitals flagged) for
  `dump_orbitals`, or `nothing` when no orbitals were removed.
"""
function redundant_orbital_classes(EC::ECInfo, Xredundant)
  nredund = size(Xredundant, 2)
  nredund > 0 || return nothing
  SP = EC.space
  return orbital_classes_with_deleted(SP['o'], length(SP[':']), nredund)
end

"""
    redundant_orbital_classes_uhf(EC, Xredundant)

  Open-shell analogue of [`redundant_orbital_classes`](@ref): a per-spin `(α, β)` tuple of
  orbital classes, or `nothing` when no orbitals were removed.
"""
function redundant_orbital_classes_uhf(EC::ECInfo, Xredundant)
  nredund = size(Xredundant, 2)
  nredund > 0 || return nothing
  SP = EC.space
  norb = length(SP[':'])
  return (orbital_classes_with_deleted(SP['o'], norb, nredund),
          orbital_classes_with_deleted(SP['O'], norb, nredund))
end

"""
    starting_orbitals(EC)

  Starting orbitals for the DF SCF (`dfhf`/`dfuhf`): previously saved orbitals if present,
  otherwise the configured `scf.guess`. Always returns a `SpinMatrix`; the caller reduces it
  to the closed-shell α block or unrestricts it as needed.
"""
function starting_orbitals(EC::ECInfo)
  cMO, loaded = try_load_starting_orbitals(EC)
  if !loaded
    cMO = guess_orb(EC, EC.options.scf.guess)
  end
  return cMO
end

"""
    scf_closed_shell!(EC, cMO, sao, hsmall, Enuc, fockbuilder, solver)

  Shared closed-shell SCF iteration loop, parameterized by a pluggable Fock builder
  and eigensolver. Used by both the density-fitted HF (`dfhf`) and the exact
  non-density-fitted HF (`hf`).

  - `cMO` (`nAO × nMO`) holds the orbital coefficients and is updated *in place*.
  - `sao` is the AO overlap (`I` for already-orthonormal MO integrals).
  - `hsmall` is the 1-e (core) Hamiltonian, `Enuc` the constant energy offset.
  - `fockbuilder(cMO) -> fock` builds the Fock matrix from the current orbitals.
  - `solver(fock) -> (ϵ, cMO_new)` solves the (generalized) eigenvalue problem.

  Returns `(EHF, ϵ, fock)` (the converged Fock matrix is returned for orbital dumps).
"""
function scf_closed_shell!(EC::ECInfo{T}, cMO, sao, hsmall, Enuc, fockbuilder, solver) where {T}
  SP = EC.space
  norb = length(SP[':'])
  thren = scf_thren(EC)
  diis = Diis(EC)
  ϵ = zeros(real(T), norb)
  EHF = zero(real(T))
  previousEHF = zero(real(T))
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  t1 = time_ns()
  local fock
  for it=1:EC.options.scf.maxit
    cMO2 = cMO[:,SP['o']]
    fock = fockbuilder(cMO)
    t1 = print_time(EC, t1, "generate Fock matrix", 2)
    fhsmall = fock + hsmall
    @mtensor efhsmall = (conj(cMO2[p,i])*fhsmall[p,q])*cMO2[q,i]
    EHF = real(efhsmall) + Enuc
    ΔE = EHF - previousEHF
    previousEHF = EHF
    den2 = cMO2*cMO2'
    sdf = sao*den2*fock
    Δfock = sdf - sdf'
    var = sum(abs2,Δfock)
    output_iteration(it, var, time_ns() - t0, EHF, ΔE)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    perform!(diis, [fock], [Δfock])
    t1 = print_time(EC, t1, "DIIS", 2)
    ϵ_new, cMO_new = solver(fock)
    ϵ .= ϵ_new
    cMO .= cMO_new
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
  end
  return EHF, ϵ, fock
end

"""
    scf_open_shell!(EC, cMO::SpinMatrix, sao, h1a, h1b, Enuc, fockbuilder, solver)

  Shared unrestricted (open-shell) SCF loop with a pluggable Fock builder and solver,
  the open-shell analogue of [`scf_closed_shell!`](@ref). `cMO` (α/β coefficients) is
  updated in place. `fockbuilder(cMO)` returns the spin Fock pair (indexable `[1]`/`[2]`,
  e.g. a `SpinMatrix`); `solver(fock_spin)` returns `(ϵ_spin, cMO_spin)`. `h1a`/`h1b` are
  the α/β core Hamiltonians (equal for UHF). Convergence is driven by the metric
  residual `S·D·F − F·D·S` per spin. Returns `(EHF, ϵ, fock)` with `ϵ` an `[ϵα, ϵβ]` vector
  and `fock` the converged spin Fock pair.
"""
function scf_open_shell!(EC::ECInfo{T}, cMO::SpinMatrix, sao, h1a, h1b, Enuc, fockbuilder, solver) where {T}
  SP = EC.space
  norb = length(SP[':'])
  thren = scf_thren(EC)
  spocc = ('o', 'O')
  h1 = (h1a, h1b)
  diis = Diis(EC)
  ϵ = [zeros(real(T), norb), zeros(real(T), norb)]
  EHF = zero(real(T))
  previousEHF = zero(real(T))
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  t1 = time_ns()
  local fock
  for it=1:EC.options.scf.maxit
    fock = fockbuilder(cMO)
    t1 = print_time(EC, t1, "generate Fock matrix", 2)
    efhsmall = [zero(real(T)), zero(real(T))]
    Δfock = [zeros(T, norb, norb), zeros(T, norb, norb)]
    var = zero(real(T))
    for ispin = 1:2
      cMOo = cMO[ispin][:, SP[spocc[ispin]]]
      den = cMOo * cMOo'                       # Hermitian density C·Cᵀ (complex: C·C†)
      fhsmall = fock[ispin] + h1[ispin]
      @mtensor efh = 0.5 * (conj(cMOo[p,i]) * fhsmall[p,q]) * cMOo[q,i]
      efhsmall[ispin] = real(efh)
      sdf = sao*den*fock[ispin]
      Δfock[ispin] = sdf - sdf'                # S·D·F − F·D·S
      var += sum(abs2, Δfock[ispin])
    end
    EHF = efhsmall[1] + efhsmall[2] + Enuc
    ΔE = EHF - previousEHF
    previousEHF = EHF
    output_iteration(it, var, time_ns() - t0, EHF, ΔE)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    perform!(diis, fock, Δfock)
    t1 = print_time(EC, t1, "DIIS", 2)
    for ispin = 1:2
      ϵ[ispin], cMO[ispin] = solver(fock[ispin])
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
  end
  return EHF, ϵ, fock
end

"""
    dfhf(EC::ECInfo)

  Perform closed-shell DF-HF calculation.
  If `EC.fd.df3idx` is set, uses pre-existing 3-index integrals (mmL) in MO basis (S=I).
  Returns the energy as the `HF` key in `OutDict`.
"""
function dfhf(EC::ECInfo{T}) where T
  use_df3idx = EC.fd.df3idx
  t1 = time_ns()
  if use_df3idx
    print_info("DF-HF (3-index)")
    setup_space_fd!(EC)
  else
    print_info("DF-HF")
    setup_space_system!(EC)
  end
  SP = EC.space
  norb = length(SP[':'])
  @assert SP['o'] == SP['O'] "DF-HF only for closed-shell"
  direct = false
  local sao, hsmall, mmLfile, mmL, bao, bfit, Xorth, Xredundant
  local fock
  Enuc = zero(real(T))
  if use_df3idx
    hsmall = EC.fd.int1
    Enuc = EC.fd.int0
    mmLfile, mmL = mmap3idx(EC, "mmL")
    cMO = Matrix{T}(I, norb, norb)
    sao = Matrix{T}(I, norb, norb)
  else
    @assert T == Float64 "DF-HF with 3-index integrals only implemented for real case"
    direct = EC.options.scf.direct
    Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
    if direct
      bao = generate_basis(EC, "ao")
      bfit = generate_basis(EC, "jkfit")
    end
    t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
    cMO_sm = starting_orbitals(EC)
    @assert is_restricted(cMO_sm) "DF-HF only implemented for closed-shell"
    cMO = cMO_sm.α
    t1 = print_time(EC, t1, "guess orbitals", 2)
    hsmall = load(EC, "h_AA", Val(2))
    sao = load(EC, "S_AA", Val(2))
    Xorth, Xredundant = canonical_orth_for_occ(EC, sao; open_shell=false)
  end
  fockbuilder = if use_df3idx
    cMO -> gen_df3idx_fock(EC, hsmall, mmL, cMO[:,SP['o']])
  elseif direct
    cMO -> gen_dffock(EC, cMO, bao, bfit)
  else
    cMO -> gen_dffock(EC, cMO)
  end
  solver = use_df3idx ? (fock -> eigen(Hermitian(fock))) : (fock -> eigen_orth(fock, Xorth, Xredundant))
  EHF, ϵ, fock = scf_closed_shell!(EC, cMO, sao, hsmall, Enuc, fockbuilder, solver)
  normalize_phase!(cMO)
  if use_df3idx
    close(mmLfile)
    transform_3idx!(EC, "mmL", cMO)
    EC.fd.int1 = cMO' * hsmall * cMO
    t1 = print_time(EC, t1, "transform integrals", 0)
  end
  occupations = [2*ones(length(SP['o'])); zeros(length(SP['v']))]
  dipole = nothing
  if use_df3idx
    println("WARNING: DF-HF dipole moments are unavailable for pretransformed 3-index integrals.")
  else
    # restricted 1-RDM convention: α holds the total (spin-summed) density, so use the
    # doubly-occupied `occupations` (not a per-spin density) for the dipole RDM.
    dipole = calc_dipole_moment(EC, SpinMatrix(cMO), SpinMatrix(Diagonal(occupations)); basis=direct ? bao : nothing)
    if !isnothing(dipole)
      output_dipole("DF-HF", dipole)
    end
  end
  println("DF-HF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  if use_df3idx
    # store the MO-basis Fock (same basis the rotation is relative to) for post-processing
    dump_rotations(EC, SpinMatrix(cMO); type="DF-HF", energies=ϵ, occupations=occupations, fock=SpinMatrix(fock))
  else
    classes = redundant_orbital_classes(EC, Xredundant)
    # persist the converged AO Fock so non-canonical post-processing (e.g. region.pseudo) can use it
    dump_orbitals(EC, SpinMatrix(cMO); type="DF-HF", energies=ϵ, occupations=occupations,
                  classes=classes, fock=SpinMatrix(fock))
  end
  energies = OutDict("HF"=>(EHF, "closed-shell DF-HF energy"), "E"=>(EHF, "closed-shell DF-HF energy"))
  return isnothing(dipole) ? energies : add_dipole_entries(energies, "DF-HF", dipole)
end

"""
    hf(EC::ECInfo)

  Perform a closed-shell Hartree-Fock calculation directly from exact (non-density-fitted)
  AO integrals stored as scratch files (`"ao_int2"`/`"S_AA"`/`"h_AA"`, generated on demand
  via [`ensure_ao_integrals!`](@ref)); electron count etc. come from the molecular system.

  The SCF is solved in the non-orthogonal AO basis using the overlap `S_AA`:
  canonical orthogonalization handles linearly-dependent basis sets, the Fock matrix is
  built exactly from the memory-mapped 4-index AO integrals (`gen_fock`), and the metric
  residual `S·D·F − F·D·S` drives convergence (shared loop [`scf_closed_shell!`](@ref)).

  Returns the energy as the `HF` key in `OutDict`. The converged MO coefficients are
  written to the wavefunction dump for subsequent (AO→MO) correlation steps.
"""
function hf(EC::ECInfo{T}) where {T}
  ensure_ao_integrals!(EC; method="@hf", alternative="@bohf")
  t1 = time_ns()
  print_info("HF")
  setup_space_system!(EC)
  SP = EC.space
  @assert SP['o'] == SP['O'] "hf only for closed-shell"
  sao = load2idx(EC, "S_AA")
  hsmall = load2idx(EC, "h_AA")
  Enuc = nuclear_repulsion(EC.system)
  Xorth, Xredundant = canonical_orth_for_occ(EC, sao; open_shell=false)
  cMO_sm = starting_orbitals(EC)
  @assert is_restricted(cMO_sm) "hf only for closed-shell"
  cMO = cMO_sm.α
  t1 = print_time(EC, t1, "guess orbitals", 2)
  # integral source: the persisted ± supermatrix store (half the streaming per iteration)
  # when present, else the joint triangular mmap — gen_fock dispatches on the handle type
  use_pm = pm_exists(EC)
  ints = use_pm ? open_pm_store(EC) : nothing
  aofile = nothing
  if !use_pm
    aofile, ints = mmap3idx(EC, "ao_int2")
  end
  fockbuilder = cMO -> gen_fock(EC, ints, hsmall, cMO, cMO)
  solver = fock -> eigen_orth(fock, Xorth, Xredundant)
  EHF, ϵ, fock = scf_closed_shell!(EC, cMO, sao, hsmall, Enuc, fockbuilder, solver)
  use_pm ? close_pm_store!(EC, ints) : close(aofile)
  normalize_phase!(cMO)
  occupations = [2*ones(length(SP['o'])); zeros(length(SP['v']))]
  classes = redundant_orbital_classes(EC, Xredundant)
  dump_orbitals(EC, SpinMatrix(cMO); type="HF", energies=ϵ, occupations=occupations, classes=classes, fock=SpinMatrix(fock))
  println("HF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  return OutDict("HF"=>(EHF, "closed-shell HF energy"),
                 "E"=>(EHF, "closed-shell HF energy"))
end

"""
    uhf(EC::ECInfo)

  Perform exact (non-density-fitted) unrestricted Hartree-Fock from AO integrals stored
  as scratch files (`"ao_int2"`/`"S_AA"`/`"h_AA"`, generated on demand via
  [`ensure_ao_integrals!`](@ref)). Uses the shared
  open-shell loop [`scf_open_shell!`](@ref) with a UHF Fock builder over the AO integrals (`gen_ufock`)
  and canonical orthogonalization for linear-dependence handling. Returns the energy as
  the `UHF` and `HF` keys in `OutDict`.
"""
function uhf(EC::ECInfo{T}) where {T}
  ensure_ao_integrals!(EC; method="@uhf", alternative="@bouhf")
  t1 = time_ns()
  print_info("UHF")
  setup_space_system!(EC)
  SP = EC.space
  sao = load2idx(EC, "S_AA")
  hsmall = load2idx(EC, "h_AA")   # AO core Hamiltonian (same for α and β)
  Enuc = nuclear_repulsion(EC.system)
  Xorth, Xredundant = canonical_orth_for_occ(EC, sao; open_shell=true)
  # core-Hamiltonian guess (same orbitals for α/β; open shells differ via occupations)
  cMO = starting_orbitals(EC)
  unrestrict!(cMO)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  # integral source: ± supermatrix store when present, else the joint mmap (see `hf`)
  use_pm = pm_exists(EC)
  ints = use_pm ? open_pm_store(EC) : nothing
  aofile = nothing
  if !use_pm
    aofile, ints = mmap3idx(EC, "ao_int2")
  end
  fockbuilder = cMO -> gen_ufock(EC, ints, hsmall, cMO, cMO)
  solver = fock -> eigen_orth(fock, Xorth, Xredundant)
  EHF, ϵ, fock = scf_open_shell!(EC, cMO, sao, hsmall, hsmall, Enuc, fockbuilder, solver)
  use_pm ? close_pm_store!(EC, ints) : close(aofile)
  for ispin = 1:2
    normalize_phase!(cMO[ispin])
  end
  occupationsa = [ones(length(SP['o'])); zeros(length(SP['v']))]
  occupationsb = [ones(length(SP['O'])); zeros(length(SP['V']))]
  classes = redundant_orbital_classes_uhf(EC, Xredundant)
  dump_orbitals(EC, cMO; type="UHF", energies=ϵ, occupations=(occupationsa, occupationsb), classes=classes, fock=fock)
  println("UHF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  return OutDict("UHF"=>(EHF, "UHF energy"),
                 "HF"=>(EHF, "UHF energy"),
                 "E"=>(EHF, "UHF energy"))
end

"""
    dfhf_positron(EC::ECInfo)

  Perform closed-shell DF-HF calculation with positron.
  Returns the energy as the `HF` key in `OutDict`.
"""
function dfhf_positron(EC::ECInfo)
  t1 = time_ns()
  print_info("Positron DF-HF")
  setup_space_system!(EC)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = scf_thren(EC)
  direct = EC.options.scf.direct
  guess = EC.options.scf.guess
  guess_pos = EC.options.scf.guess_pos
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  if direct
    bao = generate_basis(EC, "ao")
    bfit = generate_basis(EC, "jkfit")
  end
  t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
  cMO = guess_orb(EC, guess)
  cPO = guess_pos_orb(EC, guess_pos)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  @assert is_restricted(cMO) "Positron DF-HF only implemented for closed-electron-shell"
  cMO = cMO.α
  cPO = cPO.α
  ϵ = zeros(norb)
  ε_pos = zeros(norb)
  local fock
  hsmall = load(EC, "h_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  Xorth, Xredundant = canonical_orth_for_occ(EC, sao; open_shell=false)
  # display(sao)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    eden = gen_density_matrix(EC, cMO, cMO, SP['o'])
    pden = gen_density_matrix(EC, cPO, cPO, [1])
    if direct
      fock, fock_pos, Jp = gen_dffock(EC, cMO, cPO, bao, bfit)
    else
      fock, fock_pos, Jp = gen_dffock(EC, cMO, cPO)
    end
    fhsmall = fock + hsmall + Jp
    t1 = print_time(EC, t1, "generate DF-Fock matrices for e and e+", 2)
    @mtensor E_el = eden[p,q] * fhsmall[p,q]
    @mtensor E_pos = pden[p,q] * fock_pos[p,q]
    EHF = E_el + E_pos + Enuc
    ΔE = EHF - previousEHF
    previousEHF = EHF
    Δfock = sao*eden'*fock - fock*eden'*sao
    Δfock_pos = sao*pden'*fock_pos - fock_pos*pden'*sao
    var = sum(abs2,Δfock) + sum(abs2,Δfock_pos)
    output_iteration(it, var, time_ns() - t0, EHF, ΔE)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    perform!(diis, [fock, fock_pos], [Δfock, Δfock_pos])
    t1 = print_time(EC, t1, "DIIS", 2)
    # solve in the (canonically) orthonormalized basis to handle redundant basis sets
    ϵ_new, cMO_new = eigen_orth(fock, Xorth, Xredundant)
    ε_new_pos, cPO_new = eigen_orth(fock_pos, Xorth, Xredundant)
    ϵ .= ϵ_new
    ε_pos .= ε_new_pos
    cMO .= cMO_new
    cPO .= cPO_new
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    # display(ϵ)
  end
  normalize_phase!(cMO)
  normalize_phase!(cPO)
  println("DF-HF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  open_dump(EC, "w") do io
    occupations = [2*ones(length(SP['o'])); zeros(length(SP['v']))]
    classes = redundant_orbital_classes(EC, Xredundant)
    dump_orbitals(io, EC, SpinMatrix(cMO); type="DF-HF", energies=ϵ, occupations=occupations, classes=classes, fock=SpinMatrix(fock), MO="mo")
    occupations = [1.0; zeros(length(SP['m'])-1)]
    dump_orbitals(io, EC, SpinMatrix(cPO); type="DF-HF positron", energies=ε_pos, occupations=occupations, MO="po")
  end
  return OutDict("HF"=>(EHF, "closed-shell DF-HF+ energy"), "E"=>(EHF, "closed-shell DF-HF+ energy"))
end

"""
    dfuhf(EC::ECInfo)

  Perform DF-UHF calculation.
  If `EC.fd.df3idx` is set, uses pre-existing 3-index integrals (mmL/MML) in MO basis (S=I).
  Returns the energy as the `UHF` and `HF` keys in `OutDict`.
"""
function dfuhf(EC::ECInfo{T}) where T
  use_df3idx = EC.fd.df3idx
  t1 = time_ns()
  if use_df3idx
    print_info("DF-UHF (3-index)")
    setup_space_fd!(EC)
  else
    print_info("DF-UHF")
    setup_space_system!(EC)
  end
  SP = EC.space
  norb = length(SP[':'])
  direct = false
  local sao, hsmall, h1a, h1b, mmLfile, mmL, MMLfile, MML, bao, bfit, Xorth, Xredundant
  local fock
  has_MML = false
  Enuc = zero(real(T))
  if use_df3idx
    h1a = EC.fd.uhf ? EC.fd.int1a : EC.fd.int1
    h1b = EC.fd.uhf ? EC.fd.int1b : EC.fd.int1
    Enuc = EC.fd.int0
    mmLfile, mmL = mmap3idx(EC, "mmL")
    has_MML = file_exists(EC, "MML")
    if has_MML
      MMLfile, MML = mmap3idx(EC, "MML")
    else
      MMLfile = mmLfile
      MML = mmL
    end
    cMO = SpinMatrix(Matrix{T}(I, norb, norb), Matrix{T}(I, norb, norb))
    sao = Matrix{T}(I, norb, norb)
  else
    @assert T == Float64 "DF-UHF with 3-index integrals only implemented for real case"
    direct = EC.options.scf.direct
    Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
    if direct
      bao = generate_basis(EC, "ao")
      bfit = generate_basis(EC, "jkfit")
    end
    t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
    cMO = starting_orbitals(EC)
    unrestrict!(cMO)
    t1 = print_time(EC, t1, "guess orbitals", 2)
    hsmall = load2idx(EC, "h_AA")
    sao = load2idx(EC, "S_AA")
    Xorth, Xredundant = canonical_orth_for_occ(EC, sao; open_shell=true)
  end
  # DF-UHF Fock builder + per-spin solver for the shared open-shell loop. The df3idx
  # path uses the MO metric (S=I, plain `eigen`) with the pretransformed 3-index Fock;
  # the integral-direct/jkfit path uses the AO metric (`eigen_orth`). Both reduce to
  # the same arithmetic the inline loop used (`den` is symmetric ⇒ `den'==den`, and
  # `sao==I` collapses the metric residual to `den·F − F·den`).
  if use_df3idx
    fockbuilder = cMO -> gen_df3idx_fock(EC, h1a, h1b, mmL, MML, cMO[1][:, SP['o']], cMO[2][:, SP['O']])
    solver = fock -> eigen(Hermitian(fock))
  else
    h1a = h1b = hsmall
    fockbuilder = direct ? (cMO -> gen_dffock(EC, cMO, bao, bfit)) : (cMO -> gen_dffock(EC, cMO))
    solver = fock -> eigen_orth(fock, Xorth, Xredundant)
  end
  EHF, ϵ, fock = scf_open_shell!(EC, cMO, sao, h1a, h1b, Enuc, fockbuilder, solver)
  for ispin = 1:2
    normalize_phase!(cMO[ispin])
  end
  if use_df3idx
    close(mmLfile)
    if has_MML
      close(MMLfile)
    else
      copy_file!(EC, "mmL", "MML") # if MML not exists, copy mmL to MML to simplify transform_3idx!
    end
    transform_3idx!(EC, "mmL", cMO[1])
    transform_3idx!(EC, "MML", cMO[2])
    EC.fd.int1a = cMO[1]' * h1a * cMO[1]
    EC.fd.int1b = cMO[2]' * h1b * cMO[2]
    if !EC.fd.uhf
      # make UHF-type
      EC.fd.int1 = zeros(T, 0, 0)
      EC.fd.uhf = true
      EC.fd.head["IUHF"] = [1]
    end
    t1 = print_time(EC, t1, "transform integrals", 2)
  end
  occupationsa = [ones(length(SP['o'])); zeros(length(SP['v']))]
  occupationsb = [ones(length(SP['O'])); zeros(length(SP['V']))]
  dipole = nothing
  if use_df3idx
    println("WARNING: DF-UHF dipole moments are unavailable for pretransformed 3-index integrals.")
  else
    rdm = SpinMatrix(Diagonal(occupationsa), Diagonal(occupationsb))
    dipole = calc_dipole_moment(EC, cMO, rdm; basis=direct ? bao : nothing)
    if !isnothing(dipole)
      output_dipole("DF-UHF", dipole)
    end
  end
  println("DF-UHF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  if use_df3idx
    # store the MO-basis Fock (same basis the rotation is relative to) for post-processing
    dump_rotations(EC, cMO; type="DF-UHF", energies=ϵ, occupations=(occupationsa, occupationsb), fock=fock)
  else
    classes = redundant_orbital_classes_uhf(EC, Xredundant)
    # persist the converged AO Fock so non-canonical post-processing (e.g. region.pseudo) can use it
    dump_orbitals(EC, cMO; type="DF-UHF", energies=ϵ, occupations=(occupationsa, occupationsb),
                  classes=classes, fock=fock)
  end
  energies = OutDict("UHF"=>(EHF,"DF-UHF energy"), "HF"=>(EHF,"DF-UHF energy"), "E"=>(EHF,"DF-UHF energy"))
  return isnothing(dipole) ? energies : add_dipole_entries(energies, "DF-UHF", dipole)
end

end #module
