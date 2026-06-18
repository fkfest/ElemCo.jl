module DFHF
using LinearAlgebra
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.Integrals
using ..ElemCo.MSystems
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.FockFactory
using ..ElemCo.Properties
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, dfhf_positron, dfuhf

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
  diis = Diis(EC)
  thren = EC.options.scf.thren
  if thren < 0.0
    thren = sqrt(EC.options.scf.thr)*0.1
  end
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
    guess = EC.options.scf.guess
    Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
    if direct
      bao = generate_basis(EC, "ao")
      bfit = generate_basis(EC, "jkfit")
    end
    t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
    cMO_sm, loaded = try_load_starting_orbitals(EC)
    if !loaded
      cMO_sm = guess_orb(EC, guess)
    end
    t1 = print_time(EC, t1, "guess orbitals", 2)
    @assert is_restricted(cMO_sm) "DF-HF only implemented for closed-shell"
    cMO = cMO_sm.α
    hsmall = load(EC, "h_AA", Val(2))
    sao = load(EC, "S_AA", Val(2))
    Xorth, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr; verbose=true)
    @assert size(Xorth, 2) ≥ length(SP['o']) "Too many linearly-dependent orbitals removed: only $(size(Xorth,2)) orbitals left for $(length(SP['o'])) occupied. Lower scf.redthr."
  end
  ϵ = zeros(real(T), norb)
  EHF = zero(real(T))
  previousEHF = zero(real(T))
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    cMO2 = cMO[:,SP['o']]
    if use_df3idx
      fock = gen_df3idx_fock(EC, hsmall, mmL, cMO2)
    elseif direct
      fock = gen_dffock(EC, cMO, bao, bfit)
    else
      fock = gen_dffock(EC, cMO)
    end
    t1 = print_time(EC, t1, "generate DF-Fock matrix", 2)
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
    if use_df3idx
      ϵ_new, cMO_new = eigen(Hermitian(fock))
    else
      ϵ_new, cMO_new = eigen_orth(fock, Xorth, Xredundant)
    end
    ϵ .= ϵ_new
    cMO .= cMO_new
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
  end
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
    nredund = size(Xredundant, 2)
    classes = nredund > 0 ? orbital_classes_with_deleted(SP['o'], norb, nredund) : nothing
    # persist the converged AO Fock so non-canonical post-processing (e.g. region.pseudo) can use it
    dump_orbitals(EC, SpinMatrix(cMO); type="DF-HF", energies=ϵ, occupations=occupations,
                  classes=classes, fock=SpinMatrix(fock))
  end
  energies = OutDict("HF"=>(EHF, "closed-shell DF-HF energy"), "E"=>(EHF, "closed-shell DF-HF energy"))
  return isnothing(dipole) ? energies : add_dipole_entries(energies, "DF-HF", dipole)
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
  thren = EC.options.scf.thren
  if thren < 0.0
    thren = sqrt(EC.options.scf.thr)*0.1
  end
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
  Xorth, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr; verbose=true)
  @assert size(Xorth, 2) ≥ length(SP['o']) "Too many linearly-dependent orbitals removed: only $(size(Xorth,2)) orbitals left for $(length(SP['o'])) occupied. Lower scf.redthr."
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
    nredund = size(Xredundant, 2)
    classes = nredund > 0 ? orbital_classes_with_deleted(SP['o'], norb, nredund) : nothing
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
  diis = Diis(EC)
  thren = EC.options.scf.thren
  if thren < 0.0
    thren = sqrt(EC.options.scf.thr)*0.1
  end
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
    guess = EC.options.scf.guess
    Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
    if direct
      bao = generate_basis(EC, "ao")
      bfit = generate_basis(EC, "jkfit")
    end
    t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
    cMO, loaded = try_load_starting_orbitals(EC)
    if !loaded
      cMO = guess_orb(EC, guess)
    end
    t1 = print_time(EC, t1, "guess orbitals", 2)
    unrestrict!(cMO)
    hsmall = load2idx(EC, "h_AA")
    sao = load2idx(EC, "S_AA")
    Xorth, Xredundant = canonical_orthogonalization(sao, EC.options.scf.redthr; verbose=true)
    @assert size(Xorth, 2) ≥ max(length(SP['o']), length(SP['O'])) "Too many linearly-dependent orbitals removed: only $(size(Xorth,2)) orbitals left. Lower scf.redthr."
  end
  ϵ = [zeros(real(T), norb), zeros(real(T), norb)]
  EHF = zero(real(T))
  previousEHF = zero(real(T))
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    if use_df3idx
      fock = gen_df3idx_fock(EC, h1a, h1b, mmL, MML, cMO[1][:, SP['o']], cMO[2][:, SP['O']])
    elseif direct
      fock = gen_dffock(EC, cMO, bao, bfit)
    else
      fock = gen_dffock(EC, cMO)
    end
    t1 = print_time(EC, t1, "generate DF-Fock matrix", 2)
    efhsmall = [zero(real(T)), zero(real(T))]
    Δfock = [zeros(T, norb, norb), zeros(T, norb, norb)]
    var = zero(real(T))
    for (ispin, sp) = enumerate(['o', 'O'])
      if use_df3idx
        h1s = ispin == 1 ? h1a : h1b
        cMO_occ = cMO[ispin][:, SP[sp]]
        den = cMO_occ * cMO_occ'
        fhsmall = fock[ispin] + h1s
        @mtensor efh = 0.5 * (den[q,p] * fhsmall[p,q])
        efhsmall[ispin] = real(efh)
        Δfock[ispin] = den * fock[ispin] - fock[ispin] * den
      else
        den = gen_density_matrix(EC, cMO[ispin], cMO[ispin], SP[sp])
        fhsmall = fock[ispin] + hsmall
        @mtensor efh = 0.5 * (den[p,q] * fhsmall[p,q])
        efhsmall[ispin] = efh
        Δfock[ispin] = sao*den'*fock[ispin] - fock[ispin]*den'*sao
      end
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
      if use_df3idx
        ϵ[ispin], cMO[ispin] = eigen(Hermitian(fock[ispin]))
      else
        ϵ[ispin], cMO[ispin] = eigen_orth(fock[ispin], Xorth, Xredundant)
      end
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
  end
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
    nredund = size(Xredundant, 2)
    classes = nredund > 0 ? (orbital_classes_with_deleted(SP['o'], norb, nredund),
                             orbital_classes_with_deleted(SP['O'], norb, nredund)) : nothing
    # persist the converged AO Fock so non-canonical post-processing (e.g. region.pseudo) can use it
    dump_orbitals(EC, cMO; type="DF-UHF", energies=ϵ, occupations=(occupationsa, occupationsb),
                  classes=classes, fock=fock)
  end
  energies = OutDict("UHF"=>(EHF,"DF-UHF energy"), "HF"=>(EHF,"DF-UHF energy"), "E"=>(EHF,"DF-UHF energy"))
  return isnothing(dipole) ? energies : add_dipole_entries(energies, "DF-UHF", dipole)
end

end #module
