module DFHF
using LinearAlgebra, TensorOperations, Printf
using ..ECInfos
using ..MSystem
using ..ECInts
using ..DIIS
using ..TensorTools

export dfhf

""" integral direct df-hf """ 
function dffock(EC,cMO,bao,bfit)
  pqP = ERI_2e3c(bao,bfit)
  PL = load(EC,"PL")
  hsmall = load(EC,"hsmall")
  # println(size(Ppq))
  occ2 = intersect(EC.space['o'],EC.space['O'])
  occ1o = setdiff(EC.space['o'],occ2)
  occ1O = setdiff(EC.space['O'],occ2)
  CMO2 = cMO[:,occ2]
  @tensoropt begin 
    pjP[p,j,P] := pqP[p,q,P] * CMO2[q,j]
    cpjL[p,j,L] := pjP[p,j,P] * PL[P,L]
    cL[L] := cpjL[p,j,L] * CMO2[p,j]
    fock[p,q] := hsmall[p,q] - cpjL[p,j,L]*cpjL[q,j,L]
  end
  if length(occ1o) > 0
    CMO1o = cMO[:,occ1o]
    @tensoropt begin
      pjP[p,j,P] := Ppq[p,q,P] * CMO1o[q,j]
      cpjL[p,j,L] := pjP[p,j,P] * PL[P,L]
      cL[L] += 0.5*cpjL[p,j,L] * CMO1o[p,j]
      fock[p,q] -= cpjL[p,j,L]*cpjL[q,j,L]
    end
  end
  if length(occ1O) > 0
    # CMO1O = cMO[:,occ1O]
    error("beta single occ not there yet")
  end
  @tensoropt begin
    cP[P] := cL[L] * PL[P,L]
    fock[p,q] += 2.0*cP[P]*pqP[p,q,P]
  end
  return fock
end

""" cholesky decomposed integrals df-hf """
function dffock(EC,cMO)
  occ2 = intersect(EC.space['o'],EC.space['O'])
  occ1o = setdiff(EC.space['o'],occ2)
  occ1O = setdiff(EC.space['O'],occ2)
  CMO2 = cMO[:,occ2]
  pqL = load(EC,"munuL")
  hsmall = load(EC,"hsmall")
  @tensoropt pjL[p,j,L] := pqL[p,q,L] * CMO2[q,j]

  @tensoropt L[L] := pjL[p,j,L] * CMO2[p,j]
  @tensoropt fock[p,q] := hsmall[p,q] - pjL[p,j,L]*pjL[q,j,L]
  if length(occ1o) > 0
    CMO1o = cMO[:,occ1o]
    @tensoropt pjL[p,j,L] := pqL[p,q,L] * CMO1o[q,j]
    @tensoropt L[L] += 0.5*pjL[p,j,L] * CMO1o[p,j]
    @tensoropt fock[p,q] -= pjL[p,j,L]*pjL[q,j,L]
  end
  if length(occ1O) > 0
    # CMO1O = cMO[:,occ1O]
    error("beta single occ not there yet")
  end
  @tensoropt fock[p,q] += 2.0*L[L]*pqL[p,q,L]
  return fock
end

function generate_basis(ms::MSys)
  # TODO: use element-specific basis!
  aobasis = lowercase(ms.atoms[1].basis["ao"].name)
  jkfit = lowercase(ms.atoms[1].basis["jkfit"].name)
  bao = BasisSet(aobasis,genxyz(ms,bohr=false))
  bfit = BasisSet(jkfit,genxyz(ms,bohr=false))
  return bao,bfit
end

function generate_integrals(ms::MSys, EC::ECInfo; save3idx = true)
  bao,bfit = generate_basis(ms)
  save(EC,"sao",overlap(bao))
  save(EC,"hsmall",kinetic(bao) + nuclear(bao))
  PQ = ERI_2e2c(bfit)
  CPQ=cholesky(PQ, RowMaximum(), check = false, tol = EC.choltol)
  if CPQ.rank < size(PQ,1)
    redund = size(PQ,1) - CPQ.rank
    println("$redund DF vectors removed using Cholesky decomposition")
  end
  # (P|Q)^-1 = (P|Q)^-1 L ((P|Q)^-1 L)† = M M†
  # (P|Q) = L L†
  # LL† M = L
  Lp=CPQ.L[invperm(CPQ.p),1:CPQ.rank]
  M = CPQ \ Lp
  CPQ = nothing
  Lp = nothing
  if save3idx
    pqP = ERI_2e3c(bao,bfit)
    @tensoropt pqL[p,q,L] := pqP[p,q,P] * M[P,L]
    save(EC,"munuL",pqL)
  else
    save(EC,"PL",M)
  end
  return ECInts.nuclear_repulsion(bao)
end

function dfhf(ms::MSys, EC::ECInfo; direct = false)
  diis = Diis(EC.scr)
  thren = sqrt(EC.thr)*0.1
  Enuc = generate_integrals(ms, EC; save3idx=!direct)
  if direct
    bao,bfit = generate_basis(ms)
  end
  hsmall = load(EC,"hsmall")
  sao = load(EC,"sao")
  ϵ,cMO = eigen(Hermitian(hsmall),Hermitian(sao))
  # display(ϵ)
  SP = EC.space
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  t0 = time_ns()
  for it=1:EC.maxit
    if direct
      fock = dffock(EC,cMO,bao,bfit)
    else
      fock = dffock(EC,cMO)
    end
    cMO2 = cMO[:,SP['o']]
    fhsmall = fock + hsmall
    @tensoropt efhsmall = scalar(cMO2[p,i]*fhsmall[p,q]*cMO2[q,i])
    EHF = efhsmall + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    den2 = cMO2*cMO2'
    sdf = sao*den2*fock 
    Δfock = sdf - sdf'
    var = sum(abs2,Δfock)
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    if abs(ΔE) < thren && var < EC.thr
      break
    end
    fock, = perform(diis,[fock],[Δfock])
    # use Hermitian to ensure real eigenvalues and normalized orbitals
    ϵ,cMO = eigen(Hermitian(fock),Hermitian(sao))
    # display(ϵ)
  end
end




end #module