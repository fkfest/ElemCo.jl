module DFHF
using LinearAlgebra, TensorOperations, Printf
using ..ECInfos
using ..MSystem
using ..ECInts
using ..DIIS

export dfhf

function dffock(EC,cMO,CPQ,hsmall,bao,bfit)
  Ppq = permutedims(ERI_2e3c(bao,bfit),[3,1,2])
  # println(size(Ppq))
  occ2 = intersect(EC.space['o'],EC.space['O'])
  occ1o = setdiff(EC.space['o'],occ2)
  occ1O = setdiff(EC.space['O'],occ2)
  CMO2 = cMO[:,occ2]
  @tensoropt Ppj[P,p,j] := Ppq[P,p,q] * CMO2[q,j]

  cPpj = reshape(CPQ \ reshape(Ppj,size(Ppj,1),:),size(Ppj))
  @tensoropt cP[P] := cPpj[P,p,j] * CMO2[p,j]
  @tensoropt fock[p,q] := hsmall[p,q] - cPpj[P,p,j]*Ppj[P,q,j]
  if length(occ1o) > 0
    CMO1o = cMO[:,occ1o]
    @tensoropt Ppj[P,p,j] := Ppq[P,p,q] * CMO1o[q,j]
    cPpj = reshape(CPQ \ reshape(Ppj,size(Ppj,1),:),size(Ppj))
    @tensoropt cP[P] += 0.5*cPpj[P,p,j] * CMO1o[p,j]
    @tensoropt fock[p,q] -= cPpj[P,p,j]*Ppj[P,q,j]
  end
  if length(occ1O) > 0
    # CMO1O = cMO[:,occ1O]
    error("beta single occ not there yet")
  end
  @tensoropt fock[p,q] += 2.0*cP[P]*Ppq[P,p,q]
  return fock
end

function dfhf(ms::MSys, EC::ECInfo)
  diis = Diis(EC.scr)
  # TODO: use element-specific basis!
  aobasis = lowercase(ms.atoms[1].basis["ao"].name)
  jkfit = lowercase(ms.atoms[1].basis["jkfit"].name)
  bao = BasisSet(aobasis,genxyz(ms,bohr=false))
  bfit = BasisSet(jkfit,genxyz(ms,bohr=false))
  Enuc = ECInts.nuclear_repulsion(bao)
  sao = overlap(bao)
  # display(sao)
  hsmall = kinetic(bao) + nuclear(bao)
  ϵ,cMO = eigen(Hermitian(hsmall),Hermitian(sao))
  # display(ϵ)
  PQ = ERI_2e2c(bfit)
  CPQ = cholesky(PQ, RowMaximum(), check = false)
  PQ = nothing
  SP = EC.space
  previousEHF = 0.0
  for it=1:50
    fock = dffock(EC,cMO,CPQ,hsmall,bao,bfit)
    cMO2 = cMO[:,SP['o']]
    fhsmall = fock + hsmall
    @tensoropt efhsmall = scalar(cMO2[p,i]*fhsmall[p,q]*cMO2[q,i])
    EHF = efhsmall + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    fmo = cMO'*fock*cMO
    # use Hermitian to ensure real eigenvalues and normalized orbitals
    ϵ,cMO = eigen(Hermitian(fock),Hermitian(sao))
    # display(ϵ)
    Δfock = fmo - diagm(ϵ)
    # cMO, = perform(diis,[cMO],[Δfock])
    @printf "%3i %12.8f %12.8f %10.2e \n" it EHF ΔE sum(abs2,Δfock)
    if abs(ΔE) < EC.thr
      break
    end
  end
end




end #module