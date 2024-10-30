module DFHF
using LinearAlgebra, TensorOperations
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.Integrals
using ..ElemCo.MSystem
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.FockFactory
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, dfhf_positron, dfuhf

"""
    dfhf(EC::ECInfo)

  Perform closed-shell DF-HF calculation.
  Returns the energy as the `HF` key in `OutDict`.
"""
function dfhf(EC::ECInfo)
  t1 = time_ns()
  print_info("DF-HF")
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
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  if direct
    bao = generate_basis(EC, "ao")
    bfit = generate_basis(EC, "jkfit")
  end
  t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
  cMO = guess_orb(EC, guess)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  @assert is_restricted(cMO) "DF-HF only implemented for closed-shell"
  cMO = cMO.α
  ϵ = zeros(norb)
  hsmall = load(EC, "h_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  # display(sao)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    if direct
      fock = gen_dffock(EC, cMO, bao, bfit)
    else
      fock = gen_dffock(EC, cMO)
    end
    t1 = print_time(EC, t1, "generate DF-Fock matrix", 2)
    cMO2 = cMO[:,SP['o']]
    fhsmall = fock + hsmall
    @tensoropt efhsmall = cMO2[p,i]*fhsmall[p,q]*cMO2[q,i]
    EHF = efhsmall + Enuc
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
    # use Hermitian to ensure real eigenvalues and normalized orbitals
    ϵ_new, cMO_new = eigen(Hermitian(fock),Hermitian(sao))
    ϵ .= ϵ_new
    cMO .= cMO_new
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    # display(ϵ)
  end
  normalize_phase!(cMO)
  println("DF-HF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  save!(EC, EC.options.wf.orb, cMO, description="DFHF orbitals")
  return OutDict("HF"=>(EHF, "closed-shell DF-HF energy"), "E"=>(EHF, "closed-shell DF-HF energy"))
end

"""
    dfhf_positron(EC::ECInfo)

  Perform closed-shell DF-HF calculation.
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
  if direct
    error("Exiting function dfhf_positron: 'direct' option is enabled with positron.")
  end
  guess = EC.options.scf.guess
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  if direct
    bao = generate_basis(EC, "ao")
    bfit = generate_basis(EC, "jkfit")
  end
  t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
  cMO, cPO = guess_orb(EC, guess)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  @assert is_restricted(cMO) "DF-HF only implemented for closed-shell"
  cMO = cMO.α
  cPO = cPO.α
  ϵ = zeros(norb)
  ε_pos = zeros(norb)
  hsmall = load(EC, "h_AA", Val(2))
  hsmall_pos = load(EC, "h_AA_positron", Val(2))
  sao = load(EC, "S_AA", Val(2))
  # display(sao)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    eden = 2.0 * gen_density_matrix(EC, cMO, cMO, SP['o'])
    pden = gen_density_matrix(EC, cPO, cPO, [1])
    if direct
      fock = gen_dffock(EC, cMO, bao, bfit)
    else
      J, Jp, K = gen_dffock(EC, eden, pden)
    end
    fock = hsmall + 0.5 * (2 * J - K) - Jp
    fock_pos = hsmall_pos - J
    t1 = print_time(EC, t1, "generate DF-Fock matrices for e and e+", 2)
    @tensoropt E1body = eden[p,q] * hsmall[p,q]
    @tensoropt E1body += pden[p,q] * hsmall_pos[p,q]
    @tensoropt Ecoul = 0.5 * J[p,q] * eden[p,q]
    @tensoropt Ecoul -= pden[p,q] * J[p,q]
    @tensoropt Eex = -0.25 * eden[p,q] * K[p,q]
    EHF = E1body + Ecoul + Eex + Enuc
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
    perform!(diis, [fock], [Δfock])
    #perform!(diis, [fock_pos], [Δfock_pos])
    t1 = print_time(EC, t1, "DIIS", 2)
    # use Hermitian to ensure real eigenvalues and normalized orbitals
    ϵ_new, cMO_new = eigen(Hermitian(fock),Hermitian(sao))
    ε_new_pos, cPO_new = eigen(Hermitian(fock_pos),Hermitian(sao))
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
  save!(EC, EC.options.wf.orb, cMO, description="DFHF orbitals")
  save!(EC, EC.options.wf.orb_pos, cPO, description="DFHF positron orbitals")
  return OutDict("HF"=>(EHF, "closed-shell DF-HF+ energy"), "E"=>(EHF, "closed-shell DF-HF+ energy"))
end

"""
    dfuhf(EC::ECInfo)

  Perform DF-UHF calculation.
  Returns the energy as the `UHF` and `HF` keys in `OutDict`.
"""
function dfuhf(EC::ECInfo)
  t1 = time_ns()
  print_info("DF-UHF")
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
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  if direct
    bao = generate_basis(EC, "ao")
    bfit = generate_basis(EC, "jkfit")
  end
  t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
  cMO = guess_orb(EC, guess)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  unrestrict!(cMO)
  ϵ = [zeros(norb), zeros(norb)] 
  hsmall = load2idx(EC, "h_AA")
  sao = load2idx(EC, "S_AA")
  # display(sao)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush_output()
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    if direct
      fock = gen_dffock(EC, cMO, bao, bfit)
    else
      fock = gen_dffock(EC, cMO)
    end
    t1 = print_time(EC, t1, "generate DF-Fock matrix", 2)
    efhsmall = Float64[0.0, 0.0]
    Δfock = Matrix{Float64}[zeros(norb,norb), zeros(norb,norb)]
    var = 0.0
    for (ispin, sp) = enumerate(['o', 'O'])
      den = gen_density_matrix(EC, cMO[ispin], cMO[ispin], SP[sp])
      fhsmall = fock[ispin] + hsmall
      @tensoropt efh = 0.5 * (den[p,q] * fhsmall[p,q])
      efhsmall[ispin] = efh
      Δfock[ispin] = sao*den'*fock[ispin] - fock[ispin]*den'*sao
      var += sum(abs2,Δfock[ispin])
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
      # use Hermitian to ensure real eigenvalues and normalized orbitals
      ϵ[ispin], cMO[ispin] = eigen(Hermitian(fock[ispin]), Hermitian(sao))
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    # display(ϵ)
  end
  for ispin = 1:2
    normalize_phase!(cMO[ispin])
  end
  println("DF-UHF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  save!(EC, EC.options.wf.orb, cMO..., description="DFUHF orbitals")
  return OutDict("UHF"=>(EHF,"DF-UHF energy"), "HF"=>(EHF,"DF-UHF energy"), "E"=>(EHF,"DF-UHF energy"))
end

end #module
