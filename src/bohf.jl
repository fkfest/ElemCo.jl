""" bi-orthogonal Hartree-Fock method
    (using a similarity-transformed FciDump)
"""
module BOHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.FciDump
using ..ElemCo.Focks
using ..ElemCo.DIIS

export bohf, bouhf

""" bo-hf on fcidump"""
function bohf(EC::ECInfo)
  println("Bi-orthogonal Hartree-Fock")
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC.scr)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  cMOl = Matrix(I, norb, norb)
  cMOr = Matrix(I, norb, norb)
  ϵ = zeros(norb)
  hsmall = integ1(EC.fd,SCα)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    fock = gen_fock(EC,cMOl,cMOr)
    cMOl2 = cMOl[:,SP['o']]
    cMOr2 = cMOr[:,SP['o']]
    fhsmall = fock + hsmall
    @tensoropt efhsmall = scalar(cMOl2[p,i]*fhsmall[p,q]*cMOr2[q,i])
    EHF = efhsmall + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    den2 = cMOl2*cMOr2'
    Δfock = den2'*fock - fock*den2'
    var = sum(abs2,Δfock)
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    fock, = perform(diis,[fock],[Δfock])
    ϵ,cMOr = eigen(fock)
    cMOl = (inv(cMOr))'
    # display(ϵ)
  end
  println("BO-HF energy: ", EHF)
  return EHF, ϵ, cMOl, cMOr
end

""" bo-uhf on fcidump"""
function bouhf(EC::ECInfo)
  println("Bi-orthogonal unrestricted Hartree-Fock")
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC.scr)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  # 1: alpha, 2: beta (cMOs can become complex(?))
  cMOl = Any[Matrix(I, norb, norb), Matrix(I, norb, norb)]
  cMOr = deepcopy(cMOl)
  ϵ = Any[zeros(norb), zeros(norb)]
  hsmall = [integ1(EC.fd,SCα), integ1(EC.fd,SCβ)]
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    fock = gen_ufock(EC,cMOl,cMOr)
    efhsmall = [0.0, 0.0]
    Δfock = Any[zeros(norb,norb), zeros(norb,norb)]
    var = 0.0
    for (ispin, sp) = enumerate(['o', 'O'])
      cMOlo = cMOl[ispin][:,SP[sp]]
      cMOro = cMOr[ispin][:,SP[sp]]
      fhsmall = fock[ispin] + hsmall[ispin]
      @tensoropt efh = 0.5*scalar(cMOlo[p,i]*fhsmall[p,q]*cMOro[q,i])
      efhsmall[ispin] = efh
      den = cMOlo*cMOro'
      Δfock[ispin] = den'*fock[ispin] - fock[ispin]*den'
      var += sum(abs2,Δfock[ispin])
    end
    EHF = efhsmall[1] + efhsmall[2] + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    fock = perform(diis,fock,Δfock)
    for ispin = 1:2
      ϵ[ispin],cMOr[ispin] = eigen(fock[ispin])
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
    # display(ϵ)
  end
  println("BO-UHF energy: ", EHF)
  return EHF, ϵ, cMOl, cMOr
end




end # module BOHF