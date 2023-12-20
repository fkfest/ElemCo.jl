""" bi-orthogonal Hartree-Fock method
    (using a similarity-transformed FciDump)
"""
module BOHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.FciDump
using ..ElemCo.FockFactory
using ..ElemCo.DIIS

export bohf, bouhf
export guess_boorb

"""
    guess_boorb(EC::ECInfo, guess::Symbol, uhf=false)

  Calculate starting guess for BO-MO coefficients (left and right).
  Type of initial guess for MO coefficients is given by `guess`.
  `uhf` indicates whether the calculation is restricted or unrestricted.

  See [`ScfOptions.guess`](@ref ECInfos.ScfOptions) for possible values.
  (Note: `:SAD`` is not possible here and will be replaced by identity matrix!)
"""
function guess_boorb(EC::ECInfo, guess::Symbol, uhf=false)
  if EC.fd.uhf
    @assert uhf
  end
  if guess == :HCORE || guess == :hcore
    cMOr = guess_bo_hcore(EC, uhf)
  elseif guess == :I || guess == :i || guess == :IDENTITY || guess == :identity
    cMOr = guess_bo_identity(EC, uhf)
  elseif guess == :SAD || guess == :sad
    println("Warning: SAD guess not possible for BO-HF, using identity matrix instead!")
    cMOr = guess_bo_identity(EC, uhf)
  elseif guess == :GWH || guess == :gwh
    cMOr = guess_bo_gwh(EC, uhf)
  elseif guess == :ORB || guess == :orb
    cMOr = load(EC,EC.options.wf.orb)
  else
    error("Unknown guess for MO coefficients: ", guess)
  end
  if uhf
    cMOl = Any[0.0, 0.0]
    for ispin = 1:2
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
  else
    cMOl = (inv(cMOr))'
  end
  return cMOl, cMOr
end

"""
    guess_bo_hcore(EC::ECInfo, uhf)

  Guess BO-MO coefficients (right) from core Hamiltonian.
"""
function guess_bo_hcore(EC::ECInfo, uhf)
  if uhf
    spins = [:α, :β]
    if !EC.fd.uhf
      spins = [:α, :α]
    end
    CMOr_final = Any[0.0, 0.0]
  else
    spins = [:α]
  end
  isp = 1
  for spin in spins
    hsmall = integ1(EC.fd, spin)
    ϵ,cMOr = eigen(hsmall)
    rotate_eigenvectors_to_real!(cMOr,ϵ)
    if uhf
      CMOr_final[isp] = cMOr
    else
      CMOr_final = cMOr
    end
    isp += 1
  end
  return CMOr_final
end

"""
    guess_bo_identity(EC::ECInfo, uhf)

  Guess BO-MO coefficients (right) from identity matrix.
"""
function guess_bo_identity(EC::ECInfo, uhf)
  norb = length(EC.space[':'])
  if uhf
    return Any[Matrix{Float64}(I, norb, norb), Matrix{Float64}(I, norb, norb)]
  else
    return Matrix{Float64}(I, norb, norb)
  end
end

function guess_bo_gwh(EC::ECInfo, uhf)
  error("not implemented yet")
end

""" 
    bohf(EC::ECInfo)

  Perform BO-HF using integrals from fcidump EC.fd.
"""
function bohf(EC::ECInfo)
  print_info("Bi-orthogonal Hartree-Fock")
  setup_space_fd!(EC)
  flush(stdout)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  cMOl, cMOr = guess_boorb(EC, EC.options.scf.guess, false)
  ϵ = zeros(norb)
  hsmall = integ1(EC.fd,:α)
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
  cMOr = real.(cMOr)
  cMOl = (inv(cMOr))'
  println("BO-HF energy: ", EHF)
  flush(stdout)
  delete_temporary_files!(EC)
  save!(EC, EC.options.wf.orb, cMOr, description="BOHF right orbitals")
  save!(EC, EC.options.wf.orb*EC.options.wf.left, cMOl, description="BOHF left orbitals")
  return EHF
end

""" 
    bouhf(EC::ECInfo)

  Perform BO-UHF using integrals from fcidump EC.fd.
"""
function bouhf(EC::ECInfo)
  print_info("Bi-orthogonal unrestricted Hartree-Fock")
  setup_space_fd!(EC)
  flush(stdout)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  # 1: alpha, 2: beta (cMOs can become complex(?))
  cMOl, cMOr = guess_boorb(EC, EC.options.scf.guess, true)
  ϵ = Any[zeros(norb), zeros(norb)]
  hsmall = [integ1(EC.fd,:α), integ1(EC.fd,:β)]
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
      @tensoropt efh = 0.5 * den[p,q] * fhsmall[p,q]
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
    fock = perform(diis, fock, Δfock)
    for ispin = 1:2
      ϵ[ispin],cMOr[ispin] = eigen(fock[ispin])
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
    # display(ϵ)
  end
  # check MOs to be real
  for ispin = 1:2
    rotate_eigenvectors_to_real!(cMOr[ispin],ϵ[ispin])
    cMOr[ispin] = real.(cMOr[ispin])
    cMOl[ispin] = (inv(cMOr[ispin]))'
  end
  println("BO-UHF energy: ", EHF)
  flush(stdout)
  delete_temporary_files!(EC)
  save!(EC, EC.options.wf.orb, cMOr..., description="BOHF right orbitals")
  save!(EC, EC.options.wf.orb*EC.options.wf.left, cMOl..., description="BOHF left orbitals")
  return EHF
end




end # module BOHF
