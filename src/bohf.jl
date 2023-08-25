""" bi-orthogonal Hartree-Fock method
    (using a similarity-transformed FciDump)
"""
module BOHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.FciDump
using ..ElemCo.FockFactory
using ..ElemCo.DIIS

export bohf, bouhf

""" 
    bohf(EC::ECInfo)

  Perform BO-HF using integrals from fcidump EC.fd.
"""
function bohf(EC::ECInfo)
  println("Bi-orthogonal Hartree-Fock")
  flush(stdout)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  cMOl = Matrix{Float64}(I, norb, norb)
  cMOr = Matrix{Float64}(I, norb, norb)
  ϵ = zeros(norb)
  hsmall = integ1(EC.fd,SCα)
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush(stdout)
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    fock = gen_fock(EC,cMOl,cMOr)
    den = gen_density_matrix(EC, cMOl, cMOr, SP['o'])
    fhsmall = fock + hsmall
    @tensoropt efhsmall = den[p,q]*fhsmall[p,q]
    EHF = efhsmall + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    Δfock = den'*fock - fock*den'
    var = sum(abs2,Δfock)
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    flush(stdout)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    fock, = perform(diis,[fock],[Δfock])
    ϵ,cMOr = eigen(fock)
    cMOl = (inv(cMOr))'
    # display(ϵ)
  end
  # check MOs to be real
  rotate_eigenvectors_to_real!(cMOr,ϵ)
  #cMOr_real = real.(cMOr)    
  #if sum(abs2,cMOr) - sum(abs2,cMOr_real) > EC.options.scf.imagtol
    #println("Large imaginary part in orbital coefficients neglected!")
    #println("Difference between squared norms:",sum(abs2,cMOr)-sum(abs2,cMOr_real))
  #end
  #cMOr = cMOr_real
  cMOl = (inv(cMOr))'
  println("BO-HF energy: ", EHF)
  flush(stdout)
  delete_temporary_files(EC)
  return EHF, ϵ, cMOl, cMOr
end

""" 
    bouhf(EC::ECInfo)

  Perform BO-UHF using integrals from fcidump EC.fd.
"""
function bouhf(EC::ECInfo)
  println("Bi-orthogonal unrestricted Hartree-Fock")
  flush(stdout)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  # 1: alpha, 2: beta (cMOs can become complex(?))
  cMOl = Any[Matrix{Float64}(I, norb, norb), Matrix{Float64}(I, norb, norb)]
  cMOr = deepcopy(cMOl)
  ϵ = Any[zeros(norb), zeros(norb)]
  hsmall = [integ1(EC.fd,SCα), integ1(EC.fd,SCβ)]
  EHF = 0.0
  previousEHF = 0.0
  println("Iter     Energy      DE          Res         Time")
  flush(stdout)
  t0 = time_ns()
  for it=1:EC.options.scf.maxit
    fock = gen_ufock(EC,cMOl,cMOr)
    efhsmall = Any[0.0, 0.0]
    Δfock = Any[zeros(norb,norb), zeros(norb,norb)]
    var = 0.0
    for (ispin, sp) = enumerate(['o', 'O'])
      den = gen_density_matrix(EC, cMOl[ispin], cMOr[ispin], SP[sp])
      fhsmall = fock[ispin] + hsmall[ispin]
      @tensoropt efh = 0.5*den[p,q]*fhsmall[p,q]
      efhsmall[ispin] = efh
      Δfock[ispin] = den'*fock[ispin] - fock[ispin]*den'
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
    fock = perform(diis,fock,Δfock)
    for ispin = 1:2
      ϵ[ispin],cMOr[ispin] = eigen(fock[ispin])
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
    # display(ϵ)
  end
  # check MOs to be real
  for ispin = 1:2
    rotate_eigenvectors_to_real!(cMOr[ispin],ϵ[ispin])
    #cMOr_real = real.(cMOr[ispin])    
    #if sum(abs2,cMOr[ispin]) - sum(abs2,cMOr_real) > EC.options.scf.imagtol
      #println("Large imaginary part in orbital coefficients neglected!")
      #println("Difference between squared norms:",sum(abs2,cMOr[ispin])-sum(abs2,cMOr_real))
    #end
    #cMOr[ispin] = cMOr_real
    cMOl[ispin] = (inv(cMOr[ispin]))'
  end
  println("BO-UHF energy: ", EHF)
  flush(stdout)
  delete_temporary_files(EC)
  return EHF, ϵ, cMOl, cMOr
end




end # module BOHF
