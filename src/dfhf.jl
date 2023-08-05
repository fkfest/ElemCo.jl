module DFHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, GuessType, GUESS_HCORE, GUESS_SAD

""" 
    dffock(EC::ECInfo, cMO, bao, bfit)

  Compute closed-shell DF-HF Fock matrix (integral direct).
"""
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

"""
    dffock(EC::ECInfo, cMO)

  Compute closed-shell DF-HF Fock matrix 
  (using precalculated Cholesky-decomposed integrals).
"""
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

"""
    generate_basis(ms::MSys)

  Generate basis sets for AO and JK fitting.
"""
function generate_basis(ms::MSys)
  # TODO: use element-specific basis!
  aobasis = lowercase(ms.atoms[1].basis["ao"].name)
  jkfit = lowercase(ms.atoms[1].basis["jkfit"].name)
  bao = BasisSet(aobasis,genxyz(ms,bohr=false))
  bfit = BasisSet(jkfit,genxyz(ms,bohr=false))
  return bao,bfit
end

"""
    generate_integrals(EC::ECInfo; save3idx = true)

  Generate integrals for DF-HF.
  If save3idx is true, save Cholesky-decomposed 3-index integrals, 
  otherwise save pseudo-square-root-inverse Cholesky decomposition.
"""
function generate_integrals(EC::ECInfo; save3idx = true)
  bao,bfit = generate_basis(EC.ms)
  save(EC,"sao",overlap(bao))
  save(EC,"hsmall",kinetic(bao) + nuclear(bao))
  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thr, verbose = true)
  if save3idx
    pqP = ERI_2e3c(bao,bfit)
    @tensoropt pqL[p,q,L] := pqP[p,q,P] * M[P,L]
    save(EC,"munuL",pqL)
  else
    save(EC,"PL",M)
  end
  return nuclear_repulsion(EC.ms)
end

@enum GuessType GUESS_HCORE GUESS_SAD GUESS_GWH GUESS_ORB

"""
    guess_hcore(EC::ECInfo)

  Guess MO coefficients from core Hamiltonian.
"""
function guess_hcore(EC::ECInfo)
  hsmall = load(EC,"hsmall")
  sao = load(EC,"sao")
  ϵ,cMO = eigen(Hermitian(hsmall),Hermitian(sao))
  return cMO
end
  
"""
    guess_sad(EC::ECInfo)
  
  Guess MO coefficients from atomic densities.
"""
function guess_sad(EC::ECInfo)
  # minao = "ano-rcc-mb"
  minao = "ano-r0"
  # minao = "sto-6g"
  bminao = BasisSet(minao,genxyz(EC.ms,bohr=false))
  bao,bfit = generate_basis(EC.ms)
  smin2ao = overlap(bminao,bao)
  eldist = electron_distribution(EC.ms,minao)
  saoinv = invchol(Hermitian(load(EC,"sao")))
  # display(eldist)
  denao = saoinv * smin2ao' * diagm(eldist) * smin2ao * saoinv
  # dc = nc
  n,cMO = eigen(Hermitian(-denao))
  # display(n)
  return cMO
end

function guess_gwh(EC::ECInfo)
  error("not implemented yet")
end

"""
    guess_orb(EC::ECInfo, guess::GuessType)

  Calculate starting guess for MO coefficients.
"""
function guess_orb(EC::ECInfo, guess::GuessType)
  if guess == GUESS_HCORE
    return guess_hcore(EC)
  elseif guess == GUESS_SAD
    return guess_sad(EC)
  elseif guess == GUESS_GWH
    return guess_gwh(EC)
  elseif guess == GUESS_ORB
    return load(EC,"cMO")
  else
    error("unknown guess type")
  end
end

"""
    dfhf(EC::ECInfo; direct = false, guess = GUESS_SAD)

  Perform closed-shell DF-HF calculation.
"""
function dfhf(EC::ECInfo; direct = false, guess = GUESS_SAD)
  println("DF-HF")
  diis = Diis(EC.scr)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = generate_integrals(EC; save3idx=!direct)
  if direct
    bao,bfit = generate_basis(EC.ms)
  end
  cMO = guess_orb(EC,guess)
  ϵ = zeros(size(cMO,1))
  hsmall = load(EC,"hsmall")
  sao = load(EC,"sao")
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
  return ϵ, cMO
end




end #module
