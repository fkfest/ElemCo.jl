module DFHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.FockFactory
using ..ElemCo.DIIS
using ..ElemCo.TensorTools

export dfhf, generate_integrals 

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
      fock = gen_dffock(EC,cMO,bao,bfit)
    else
      fock = gen_dffock(EC,cMO)
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
