module DFHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.FockFactory
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, dfuhf

"""
    dfhf(EC::ECInfo)

  Perform closed-shell DF-HF calculation.
"""
function dfhf(EC::ECInfo)
  t1 = time_ns()
  print_info("DF-HF")
  setup_space_ms!(EC)
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
    bao = generate_basis(EC.ms, "ao")
    bfit = generate_basis(EC.ms, "jkfit")
  end
  t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
  cMO = guess_orb(EC,guess)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  @assert !is_unrestricted_MO(cMO) "DF-HF only implemented for closed-shell"
  ϵ = zeros(norb)
  hsmall = load(EC, "h_AA")
  sao = load(EC, "S_AA")
  # display(sao)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush(stdout)
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    if direct
      fock = gen_dffock(EC,cMO,bao,bfit)
    else
      fock = gen_dffock(EC,cMO)
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
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    flush(stdout)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    fock, = perform(diis,[fock],[Δfock])
    t1 = print_time(EC, t1, "DIIS", 2)
    # use Hermitian to ensure real eigenvalues and normalized orbitals
    ϵ,cMO = eigen(Hermitian(fock),Hermitian(sao))
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    # display(ϵ)
  end
  println("DF-HF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  save!(EC, EC.options.wf.orb, cMO, description="DFHF orbitals")
  return EHF
end

"""
    dfuhf(EC::ECInfo)

  Perform DF-UHF calculation.
"""
function dfuhf(EC::ECInfo)
  t1 = time_ns()
  print_info("DF-UHF")
  setup_space_ms!(EC)
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
    bao = generate_basis(EC.ms, "ao")
    bfit = generate_basis(EC.ms, "jkfit")
  end
  t1 = print_time(EC, t1, "generate AO-DF integrals", 2)
  cMO = guess_orb(EC,guess)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  if !is_unrestricted_MO(cMO)
    cMO = Any[cMO, cMO]
  end
  ϵ = [zeros(norb), zeros(norb)] 
  hsmall = load(EC, "h_AA")
  sao = load(EC, "S_AA")
  # display(sao)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush(stdout)
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    if direct
      fock = gen_dffock(EC,cMO,bao,bfit)
    else
      fock = gen_dffock(EC,cMO)
    end
    t1 = print_time(EC, t1, "generate DF-Fock matrix", 2)
    efhsmall = Any[0.0, 0.0]
    Δfock = Any[zeros(norb,norb), zeros(norb,norb)]
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
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    flush(stdout)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    fock = perform(diis, fock, Δfock)
    t1 = print_time(EC, t1, "DIIS", 2)
    for ispin = 1:2
      # use Hermitian to ensure real eigenvalues and normalized orbitals
      ϵ[ispin], cMO[ispin] = eigen(Hermitian(fock[ispin]), Hermitian(sao))
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    # display(ϵ)
  end
  println("DF-UHF energy: ", EHF)
  draw_endline()
  delete_temporary_files!(EC)
  save!(EC, EC.options.wf.orb, cMO..., description="DFUHF orbitals")
  return EHF
end

end #module
