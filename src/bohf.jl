""" bi-orthogonal Hartree-Fock method
    (using a similarity-transformed FciDump)
"""
module BOHF
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.Constants
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.FciDump
using ..ElemCo.OrbTools
using ..ElemCo.FockFactory
using ..ElemCo.DIIS

export bohf, bouhf
export guess_boorb

"""
    left_from_right(cMOr)

  Calculate left BO-MO coefficients from right BO-MO coefficients.
"""
function left_from_right(cMOr)
  if is_unrestricted_MO(cMOr)
    cMOl = AbstractArray[[], []]
    for ispin = 1:2
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
  else
    cMOl = (inv(cMOr))'
  end
  return cMOl
end

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
  cMOl = left_from_right(cMOr)
  cMOl, cMOr = heatup(EC, cMOl, cMOr, EC.options.scf.temperature_guess) 
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
    CMOr_final = AbstractArray[[], []]
  else
    spins = [:α]
  end
  isp = 1
  for spin in spins
    hsmall = integ1(EC.fd, spin)
    ϵ,cMOr = eigen(hsmall)
    rotate_eigenvectors_to_real!(cMOr,ϵ)
    if uhf
      CMOr_final[isp] = real.(cMOr)
    else
      CMOr_final = real.(cMOr)
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
    return AbstractArray[Matrix{Float64}(I, norb, norb), Matrix{Float64}(I, norb, norb)]
  else
    return Matrix{Float64}(I, norb, norb)
  end
end

function guess_bo_gwh(EC::ECInfo, uhf)
  error("not implemented yet")
end

"""
    heatup(EC::ECInfo, cMOl, cMOr, temperature)

  Heat up BO-MO coefficients to `temperature` according to Fermi-Dirac.
  
  Returns new BO-MO coefficients `cMOl, cMOr`
"""
function heatup(EC::ECInfo, cMOl, cMOr, temperature)
  if temperature < 1.e-10
    return cMOl, cMOr
  end
  println("Heating up starting guess to ", temperature, " K")
  if is_unrestricted_MO(cMOr)
    return unrestricted_heatup(EC, cMOl, cMOr, temperature)
  else
    return closed_shell_heatup(EC, cMOl, cMOr, temperature)
  end
end

"""
    closed_shell_heatup(EC::ECInfo, cMOl, cMOr, temperature)

  Heat up closed-shell BO-MO coefficients to `temperature` according to Fermi-Dirac.
"""
function closed_shell_heatup(EC::ECInfo, cMOl, cMOr, temperature)
  fock = gen_fock(EC, cMOl, cMOr)
  ϵ,cMOr = eigen(fock)
  rotate_eigenvectors_to_real!(cMOr, ϵ)
  cMOr = real.(cMOr)
  nocc = n_occ_orbs(EC)
  nelec = 2*nocc
  den4temp = density4temperature(EC, ϵ, cMOr, nocc, nelec, temperature)
  fock = gen_fock(EC, den4temp)
  ϵ,cMOr = eigen(fock)
  rotate_eigenvectors_to_real!(cMOr, ϵ)
  cMOr = real.(cMOr)
  cMOl = left_from_right(cMOr)
  return cMOl, cMOr
end

function unrestricted_heatup(EC::ECInfo, cMOl, cMOr, temperature)
  SP = EC.space
  fock = gen_ufock(EC, cMOl, cMOr)
  ϵ = AbstractArray[[], []]
  den4temp = AbstractArray[[], []]
  cMOr_out = AbstractArray[[], []]
  cMOl_out = AbstractArray[[], []]
  for (ispin, sp) = enumerate(['o', 'O'])
    ϵ[ispin],cMOr_out[ispin] = eigen(fock[ispin])
    rotate_eigenvectors_to_real!(cMOr_out[ispin], ϵ[ispin])
    cMOr_out[ispin] = real.(cMOr_out[ispin])
    nocc = length(SP[sp])
    nelec = nocc
    den4temp[ispin] = density4temperature(EC, ϵ[ispin], cMOr_out[ispin], nocc, nelec, temperature)
  end
  fock = gen_ufock(EC, den4temp)
  for (ispin, sp) = enumerate(['o', 'O'])
    ϵ[ispin],cMOr_out[ispin] = eigen(fock[ispin])
    rotate_eigenvectors_to_real!(cMOr_out[ispin], ϵ[ispin])
    cMOr_out[ispin] = real.(cMOr_out[ispin])
    cMOl_out[ispin] = left_from_right(cMOr_out[ispin])
  end
  return cMOl_out, cMOr_out
end

"""
    density4temperature(EC::ECInfo, ϵ, cMOr, nocc, nelec, temperature)

  Calculate density matrix for `temperature` according to Fermi-Dirac.
"""
function density4temperature(EC::ECInfo, ϵ, cMOr, nocc, nelec, temperature)
  cMOl = (inv(cMOr))'
  fermi = (ϵ[nocc] + ϵ[nocc+1])/2
  function occfun(eps) 
    if eps < fermi
      return 1/(1+exp((eps-fermi)*Constants.HARTREE2K/temperature))
    else
      ex = exp(-(eps-fermi)*Constants.HARTREE2K/temperature)
      return ex/(1+ex) 
    end
  end
  occupation = occfun.(ϵ)
  occupation .*= nelec / sum(occupation)
  println("occupation: ", occupation[occupation .> 0.0])
  return gen_frac_density_matrix(EC, cMOl, cMOr, occupation)
end


""" 
    bohf(EC::ECInfo)

  Perform BO-HF using integrals from fcidump EC.fd.
"""
function bohf(EC::ECInfo)
  t1 = time_ns()
  pseudo = EC.options.scf.pseudo
  if pseudo
    print_info("Bi-orthogonal pseudo-canonicalization")
  else
    print_info("Bi-orthogonal Hartree-Fock")
  end
  setup_space_fd!(EC)
  flush(stdout)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  cMOl, cMOr = guess_boorb(EC, EC.options.scf.guess, false)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  ϵ = zeros(norb)
  hsmall = integ1(EC.fd,:α)
  EHF = 0.0
  previousEHF = 0.0
  if pseudo
    println("   Energy       Res         Time")
    maxit = 1
  else
    println("Iter     Energy      DE          Res         Time")
    maxit = EC.options.scf.maxit
  end
  flush(stdout)
  t0 = time_ns()
  for it=1:maxit
    fock = gen_fock(EC, cMOl, cMOr)
    t1 = print_time(EC, t1, "generate Fock matrix", 2)
    den = gen_density_matrix(EC, cMOl, cMOr, SP['o'])
    fhsmall = fock + hsmall
    @tensoropt efhsmall = den[p,q]*fhsmall[p,q]
    EHF = efhsmall + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    Δfock = den'*fock - fock*den'
    var = sum(abs2,Δfock)
    tt = (time_ns() - t0)/10^9
    if pseudo
      @printf "%12.8f %10.2e %8.2f \n" EHF var tt
    else
      @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    end
    flush(stdout)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    if pseudo
      occ = SP['o']
      vir = SP['v']
      ϵ = zeros(Complex{Float64}, norb)
      cMOr = zeros(Complex{Float64}, norb, norb)
      ϵ[occ],cMOr[occ,occ] = eigen(fock[occ,occ])
      ϵ[vir],cMOr[vir,vir] = eigen(fock[vir,vir])
    else
      fock, = perform(diis,[fock],[Δfock])
      t1 = print_time(EC, t1, "DIIS", 2)
      ϵ,cMOr = eigen(fock)
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
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
  t1 = time_ns()
  pseudo = EC.options.scf.pseudo
  if pseudo
    print_info("Bi-orthogonal unrestricted pseudo-canonicalization")
  else
    print_info("Bi-orthogonal unrestricted Hartree-Fock")
  end
  setup_space_fd!(EC)
  flush(stdout)
  SP = EC.space
  norb = length(SP[':'])
  diis = Diis(EC)
  thren = sqrt(EC.options.scf.thr)*0.1
  Enuc = EC.fd.int0
  # 1: alpha, 2: beta (cMOs can become complex(?))
  cMOl, cMOr = guess_boorb(EC, EC.options.scf.guess, true)
  t1 = print_time(EC, t1, "guess orbitals", 2)
  ϵ = AbstractArray[zeros(norb), zeros(norb)]
  hsmall = [integ1(EC.fd,:α), integ1(EC.fd,:β)]
  EHF = 0.0
  previousEHF = 0.0
  if pseudo
    println("   Energy       Res         Time")
    maxit = 1
  else
    println("Iter     Energy      DE          Res         Time")
    maxit = EC.options.scf.maxit
  end
  flush(stdout)
  t0 = time_ns()
  for it=1:maxit
    fock = gen_ufock(EC, cMOl, cMOr)
    t1 = print_time(EC, t1, "generate Fock matrix", 2)
    efhsmall = Number[0.0, 0.0]
    Δfock = AbstractArray[zeros(norb,norb), zeros(norb,norb)]
    var = 0.0
    for (ispin, sp) = enumerate(['o', 'O'])
      den = gen_density_matrix(EC, cMOl[ispin], cMOr[ispin], SP[sp])
      fhsmall = fock[ispin] + hsmall[ispin]
      @tensoropt efh = 0.5 * (den[p,q] * fhsmall[p,q])
      efhsmall[ispin] = efh
      Δfock[ispin] = den'*fock[ispin] - fock[ispin]*den'
      var += sum(abs2,Δfock[ispin])
    end
    EHF = efhsmall[1] + efhsmall[2] + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    tt = (time_ns() - t0)/10^9
    if pseudo
      @printf "%12.8f %10.2e %8.2f \n" EHF var tt
    else
      @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it EHF ΔE var tt
    end
    flush(stdout)
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    if !pseudo
      fock = perform(diis, fock, Δfock)
      t1 = print_time(EC, t1, "DIIS", 2)
    end
    for (ispin, ov) = enumerate(["ov", "OV"])
      if pseudo
        occ = SP[ov[1]]
        vir = SP[ov[2]]
        ϵ[ispin] = zeros(Complex{Float64}, norb)
        cMOr[ispin] = zeros(Complex{Float64}, norb, norb)
        ϵ[ispin][occ],cMOr[ispin][occ,occ] = eigen(fock[ispin][occ,occ])
        ϵ[ispin][vir],cMOr[ispin][vir,vir] = eigen(fock[ispin][vir,vir])
      else
        ϵ[ispin],cMOr[ispin] = eigen(fock[ispin])
      end
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
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
