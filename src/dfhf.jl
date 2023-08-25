module DFHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, generate_integrals 

""" 
    dffock(EC::ECInfo, cMO, bao, bfit)

  Compute closed-shell DF-HF Fock matrix (integral direct).
"""
function dffock(EC,cMO,bao,bfit)
  pqP = ERI_2e3c(bao,bfit)
  PL = load(EC,"C_PL")
  hsmall = load(EC,"h_AA")
  # println(size(Ppq))
  occ2 = intersect(EC.space['o'],EC.space['O'])
  @assert length(setdiff(EC.space['o'],occ2)) == 0 "Closed-shell only!"
  @assert length(setdiff(EC.space['O'],occ2)) == 0 "Closed-shell only!"
  CMO2 = cMO[:,occ2]
  @tensoropt begin 
    pjP[p,j,P] := pqP[p,q,P] * CMO2[q,j]
    cpjL[p,j,L] := pjP[p,j,P] * PL[P,L]
    cL[L] := cpjL[p,j,L] * CMO2[p,j]
    fock[p,q] := hsmall[p,q] - cpjL[p,j,L]*cpjL[q,j,L]
  end
  @tensoropt begin
    cP[P] := cL[L] * PL[P,L]
    fock[p,q] += 2.0*cP[P]*pqP[p,q,P]
  end
  return fock
end

"""
    dffock(EC::ECInfo, cMO)

  Compute closed-shell DF-HF Fock matrix 
  (using precalculated Cholesky-decomposed integrals).
"""
function dffock(EC,cMO)
  occ2 = intersect(EC.space['o'],EC.space['O'])
  @assert length(setdiff(EC.space['o'],occ2)) == 0 "Closed-shell only!"
  @assert length(setdiff(EC.space['O'],occ2)) == 0 "Closed-shell only!"
  CMO2 = cMO[:,occ2]
  pqL = load(EC,"AAL")
  hsmall = load(EC,"h_AA")
  @tensoropt pjL[p,j,L] := pqL[p,q,L] * CMO2[q,j]

  @tensoropt L[L] := pjL[p,j,L] * CMO2[p,j]
  @tensoropt fock[p,q] := hsmall[p,q] - pjL[p,j,L]*pjL[q,j,L]
  @tensoropt fock[p,q] += 2.0*L[L]*pqL[p,q,L]
  return fock
end

"""
    dfhf(EC::ECInfo; direct=false, guess=:SAD)

  Perform closed-shell DF-HF calculation.
"""
function dfhf(EC::ECInfo; direct=false, guess=:SAD)
  println("DF-HF")
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=!direct)
  if direct
    bao = generate_basis(EC.ms, "ao")
    bfit = generate_basis(EC.ms, "jkfit")
  end
  cMO = guess_orb(EC,guess)
  ϵ = zeros(size(cMO,1))
  hsmall = load(EC,"h_AA")
  sao = load(EC,"S_AA")
  # display(sao)
  SP = EC.space
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    if direct
      fock = dffock(EC,cMO,bao,bfit)
    else
      fock = dffock(EC,cMO)
    end
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
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    fock, = perform(diis,[fock],[Δfock])
    # use Hermitian to ensure real eigenvalues and normalized orbitals
    ϵ,cMO = eigen(Hermitian(fock),Hermitian(sao))
    # display(ϵ)
  end
  println("DF-HF energy: ", EHF)
  delete_temporary_files(EC)
  return ϵ, cMO
end

end #module
