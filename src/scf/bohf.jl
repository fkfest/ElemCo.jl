""" bi-orthogonal Hartree-Fock method
    (using a similarity-transformed FciDump)
"""
module BOHF
using LinearAlgebra
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.Constants
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.FciDumps
using ..ElemCo.OrbTools
using ..ElemCo.FockFactory
using ..ElemCo.DIIS
using ..ElemCo.Wavefunctions

export bohf, bouhf
export guess_boorb

# Rotate eigenvectors to real only for real-valued calculations.
# For genuinely complex integrals (T<:Complex), keep eigenvectors as-is.
_maybe_rotate_real(::ECInfo{T}, evecs, evals) where T =
  T <: Complex ? (evecs, evals) : rotate_eigenvectors_to_real(evecs, evals)

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
    cMOr = load_rotations(EC)
  else
    error("Unknown guess for MO coefficients: ", guess)
  end
  cMOl = left_from_right_rotations(cMOr)
  cMOl, cMOr = heatup(EC, cMOl, cMOr, EC.options.scf.temperature_guess) 
  return cMOl, cMOr
end

"""
    guess_bo_hcore(EC::ECInfo, uhf)

  Guess BO-MO coefficients (right) from core Hamiltonian.
"""
function guess_bo_hcore(EC::ECInfo{T}, uhf) where T
  cMOr_final = SpinMatrix{T}()
  if uhf
    spins = [:α, :β]
    if !EC.fd.uhf
      spins = [:α, :α]
    end
  else
    spins = [:α]
  end
  isp = 1
  for spin in spins
    hsmall = integ1(EC.fd, spin)
    ϵ, cMOr = eigen(hsmall)
    cMOr_final[isp], ϵ = _maybe_rotate_real(EC, cMOr, ϵ)
    isp += 1
  end
  if !uhf
    restrict!(cMOr_final)
  end
  return cMOr_final
end

"""
    guess_bo_identity(EC::ECInfo, uhf)

  Guess BO-MO coefficients (right) from identity matrix.
"""
function guess_bo_identity(EC::ECInfo{T}, uhf) where T
  norb = length(EC.space[':'])
  if uhf
    return SpinMatrix(Matrix{T}(I, norb, norb), Matrix{T}(I, norb, norb))
  else
    return SpinMatrix(Matrix{T}(I, norb, norb))
  end
end

function guess_bo_gwh(EC::ECInfo, uhf)
  error("not implemented yet")
  return SpinMatrix()
end

"""
    heatup(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix, temperature)

  Heat up BO-MO coefficients to `temperature` according to Fermi-Dirac.
  
  Returns new BO-MO coefficients `cMOl::SpinMatrix, cMOr::SpinMatrix`
"""
function heatup(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix, temperature)
  if temperature < 1.e-10
    return cMOl, cMOr
  end
  println("Heating up starting guess to ", temperature, " K")
  if is_restricted(cMOr)
    return closed_shell_heatup(EC, cMOl, cMOr, temperature)
  else
    return unrestricted_heatup(EC, cMOl, cMOr, temperature)
  end
end

"""
    closed_shell_heatup(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix, temperature)

  Heat up closed-shell BO-MO coefficients to `temperature` according to Fermi-Dirac.
"""
function closed_shell_heatup(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix, temperature)
  fock = gen_fock(EC, cMOl[1], cMOr[1])
  ϵ, cMOr_new = eigen(fock)
  cMOr[1], ϵ = _maybe_rotate_real(EC, cMOr_new, ϵ)
  nocc = n_occ_orbs(EC)
  nelec = 2*nocc
  den4temp = density4temperature(EC, ϵ, cMOr[1], nocc, nelec, temperature)
  fock = gen_fock(EC, den4temp)
  ϵ, cMOr_new = eigen(fock)
  cMOr[1], ϵ = _maybe_rotate_real(EC, cMOr_new, ϵ)
  cMOl = left_from_right_rotations(cMOr)
  return cMOl, cMOr
end

"""
    unrestricted_heatup(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix, temperature)

  Heat up unrestricted BO-MO coefficients to `temperature` according to Fermi-Dirac.
"""
function unrestricted_heatup(EC::ECInfo{T}, cMOl::SpinMatrix, cMOr::SpinMatrix, temperature) where T
  SP = EC.space
  fock = gen_ufock(EC, cMOl, cMOr)
  den4temp = SpinMatrix{T}()
  cMOr_out = SpinMatrix{T}()
  cMOl_out = SpinMatrix{T}()
  for (ispin, sp) = enumerate(['o', 'O'])
    ϵ, cMOr_new = eigen(fock[ispin])
    cMOr_out[ispin], ϵ = _maybe_rotate_real(EC, cMOr_new, ϵ)
    nocc = length(SP[sp])
    nelec = nocc
    den4temp[ispin] = density4temperature(EC, ϵ, cMOr_out[ispin], nocc, nelec, temperature)
  end
  fock = gen_ufock(EC, den4temp)
  for (ispin, sp) = enumerate(['o', 'O'])
    ϵ, cMOr_new = eigen(fock[ispin])
    cMOr_out[ispin], ϵ = _maybe_rotate_real(EC, cMOr_new, ϵ)
    cMOl_out[ispin] = transpose(inv(cMOr_out[ispin]))
  end
  return cMOl_out, cMOr_out
end

"""
    density4temperature(EC::ECInfo, ϵ, cMOr, nocc, nelec, temperature)

  Calculate density matrix for `temperature` according to Fermi-Dirac.
"""
function density4temperature(EC::ECInfo, ϵ, cMOr, nocc, nelec, temperature)
  cMOl = transpose(inv(cMOr))
  ϵ_real = real.(ϵ)
  fermi = (ϵ_real[nocc] + ϵ_real[nocc+1])/2
  function occfun(eps) 
    if eps < fermi
      return 1/(1+exp((eps-fermi)*Constants.HARTREE2K/temperature))
    else
      ex = exp(-(eps-fermi)*Constants.HARTREE2K/temperature)
      return ex/(1+ex) 
    end
  end
  occupation = occfun.(ϵ_real)
  occupation .*= nelec / sum(occupation)
  println("occupation: ", occupation[occupation .> 0.0])
  return gen_frac_density_matrix(EC, cMOl, cMOr, occupation)
end


""" 
    bohf(EC::ECInfo)

  Perform BO-HF using integrals from fcidump EC.fd.
"""
function bohf(EC::ECInfo{T}) where T
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
  ϵ = zeros(T, norb)
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
    fock = gen_fock(EC, cMOl[1], cMOr[1])
    t1 = print_time(EC, t1, "generate Fock matrix", 2)
    den = gen_density_matrix(EC, cMOl[1], cMOr[1], SP['o'])
    fhsmall = fock + hsmall
    @mtensor efhsmall = den[p,q]*fhsmall[p,q]
    EHF = efhsmall + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    Δfock = transpose(den)*fock - fock*transpose(den)
    var = sum(abs2, Δfock)
    if pseudo
      output_E_var(EHF, var, time_ns() - t0)
    else
      output_iteration(it, var, time_ns() - t0, EHF, ΔE)
    end
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    if pseudo
      occ = SP['o']
      vir = SP['v']
      ϵ_occ, cMOr_occ = eigen(fock[occ,occ])
      cMOr_occ, ϵ_occ = _maybe_rotate_real(EC, cMOr_occ, ϵ_occ)
      println("eigenvalues occupied: ", ϵ_occ)
      ϵ_vir, cMOr_vir = eigen(fock[vir,vir])
      cMOr_vir, ϵ_vir = _maybe_rotate_real(EC, cMOr_vir, ϵ_vir)
      ϵ_new = zeros(T, norb)
      cMOr_new = zeros(T, norb, norb)
      ϵ_new[occ] .= ϵ_occ
      ϵ_new[vir] .= ϵ_vir
      cMOr_new[occ,occ] .= cMOr_occ
      cMOr_new[vir,vir] .= cMOr_vir
    else
      perform!(diis, [fock], [Δfock])
      t1 = print_time(EC, t1, "DIIS", 2)
      ϵ_new, cMOr_new = eigen(fock)
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    cMOr[1], ϵ = _maybe_rotate_real(EC, cMOr_new, ϵ_new)
    cMOr[1], cMOl[1] = balance_norms!(cMOr[1])
    restrict!(cMOr)
    restrict!(cMOl)
    # display(ϵ)
  end
  println("BO-HF energy: ", EHF)
  flush(stdout)
  delete_temporary_files!(EC)
  dump_rotations(EC, cMOr; type="BO-HF", energies=ϵ, biorthogonal=true)
  return OutDict("HF"=>(EHF, "closed-shell BO-HF energy"), "E"=>(EHF, "closed-shell BO-HF energy"))
end

""" 
    bouhf(EC::ECInfo)

  Perform BO-UHF using integrals from fcidump EC.fd.
"""
function bouhf(EC::ECInfo{T}) where T
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
  ϵ = [zeros(T, norb), zeros(T, norb)]
  hsmall = [integ1(EC.fd,:α), integ1(EC.fd,:β)]
  efhsmall = zeros(T, 2)
  Δfock = [zeros(T, norb, norb), zeros(T, norb, norb)]
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
    var = 0.0
    for (ispin, sp) = enumerate(['o', 'O'])
      den = gen_density_matrix(EC, cMOl[ispin], cMOr[ispin], SP[sp])
      fhsmall = fock[ispin] + hsmall[ispin]
      @mtensor efh = 0.5 * (den[p,q] * fhsmall[p,q])
      efhsmall[ispin] = efh
      Δfock[ispin] = transpose(den)*fock[ispin] - fock[ispin]*transpose(den)
      var += sum(abs2,Δfock[ispin])
    end
    EHF = efhsmall[1] + efhsmall[2] + Enuc
    ΔE = EHF - previousEHF 
    previousEHF = EHF
    if pseudo
      output_E_var(EHF, var, time_ns() - t0)
    else
      output_iteration(it, var, time_ns() - t0, EHF, ΔE)
    end
    if abs(ΔE) < thren && var < EC.options.scf.thr
      break
    end
    t1 = print_time(EC, t1, "HF residual", 2)
    if !pseudo
      perform!(diis, fock, Δfock)
      t1 = print_time(EC, t1, "DIIS", 2)
    end
    for (ispin, ov) = enumerate(["ov", "OV"])
      if pseudo
        occ = SP[ov[1]]
        vir = SP[ov[2]]
        ϵ_occ, cMOr_occ = eigen(fock[ispin][occ,occ])
        cMOr_occ, ϵ_occ = _maybe_rotate_real(EC, cMOr_occ, ϵ_occ)
        ϵ_vir, cMOr_vir = eigen(fock[ispin][vir,vir])
        cMOr_vir, ϵ_vir = _maybe_rotate_real(EC, cMOr_vir, ϵ_vir)
        ϵ_new = zeros(T, norb)
        cMOr_new = zeros(T, norb, norb)
        ϵ_new[occ] .= ϵ_occ
        ϵ_new[vir] .= ϵ_vir
        cMOr_new[occ,occ] .= cMOr_occ
        cMOr_new[vir,vir] .= cMOr_vir
      else
        ϵ_new, cMOr_new = eigen(fock[ispin])
      end
      cMOr[ispin], ϵ[ispin] = _maybe_rotate_real(EC, cMOr_new, ϵ_new)
      cMOr[ispin], cMOl[ispin] = balance_norms!(cMOr[ispin])
    end
    t1 = print_time(EC, t1, "diagonalize Fock matrix", 2)
    # display(ϵ)
  end
  println("BO-UHF energy: ", EHF)
  flush(stdout)
  delete_temporary_files!(EC)
  dump_rotations(EC, cMOr; type="BO-UHF", energies=ϵ, biorthogonal=true)
  return OutDict("UHF"=>(EHF, "BO-UHF energy"), "HF"=>(EHF, "BO-UHF energy"), "E"=>(EHF, "BO-UHF energy"))
end

end # module BOHF
