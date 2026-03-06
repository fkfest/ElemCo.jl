"""
     module EOM

This module contains the implementation of the EOM (Equation of Motion) methods for
computing the excited states. 
"""
module EOM
using LinearAlgebra
using Buffers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.DavidsonSolver
using ..ElemCo.CoupledCluster
using ..ElemCo.CCTools
using ..ElemCo.OrbTools
using ..ElemCo.Outputs
using ..ElemCo.DecompTools
using ..ElemCo.LaplaceQuadrature

export calc_eom
export calc_df_eom
export calc_svd_eom

include("cis.jl")
include("cis(d).jl")
include("svd_dcsd.jl")

function calc_eom(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  print_info(method_name(method; main=false))

  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 2
    error("only implemented upto doubles")
  end
  unrestricted = is_unrestricted(method) || has_prefix(method, "R")
  if unrestricted
    if highest_full_exc == 2
      cis_method = has_prefix(method, "R") ? ECMethod("EOM-RCCS") : ECMethod("EOM-UCCS")
      print_info2(method_name(cis_method; main=false))
      omegas = eom_u_iterations(EC, cis_method)
      print_info2(method_name(method; main=false))
      omegas = eom_u_iterations2(EC, method)
    else
      print_info2(method_name(method; main=false))
      omegas = eom_u_iterations(EC, method)
    end
  else
    if highest_full_exc == 2
      print_info2("EOM-CCS")
      omegas = eom_iterations(EC, ECMethod("EOM-CCS"))
      print_info2(method_name(method; main=false))
      omegas = eom_iterations2(EC, method)
    else
      print_info2(method_name(method; main=false))
      omegas = eom_iterations(EC, method)
    end
  end
  energies = OutDict()
  mname = method_name(method; main=false)
  for (st, en) in enumerate(omegas)
    println("State $st: Excitation energy = $en")
    energies["ω$st"] = (en, "$mname excitation energy for state $st")
  end
  return energies
end

function eom_iterations(EC::ECInfo{T}, method::ECMethod) where T
  t0 = time_ns()
  nstates = EC.options.eom.nstates
  shift = EC.options.eom.shift
  dav = Davidson(EC, nstates; hermitian=true)
  # first guess for U1
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  states = [1:nstates;]
  energies = zeros(T, nstates)
  U1 = zeros(T, nvirt, nocc)
  V1 = zeros(T, nvirt, nocc)
  Vecs = (U1,)
  custom_dots = (calc_cs_singles_dot,)
  # HOMO-LUMO guess
  en_guess, vec_guess = cis_homo_lumo_guess(EC, nstates)
  nv_guess = size(vec_guess, 1)
  no_guess = size(vec_guess, 2)
  for st in states
    U1 .= 0.0
    U1[1:nv_guess,end-no_guess+1:end] = vec_guess[:,:,st]
    add_trial_vector!(dav, (U1,), st, custom_dots)
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, (U1,), st)
      V1 .= cis_HU1(EC, U1)
      add_product_vector!(dav, (V1,), st, custom_dots)
    end
    energies = perform!(dav)
    if do_refresh(dav, length(states))
      refresh!(dav, Vecs, custom_dots)
      output_iteration(it, -1.0, time_ns() - t0, energies...)
      states = [1:nstates;]
      continue
    end
    states2do = Int[]
    maxNormR = 0.0
    for st in 1:nstates
      get_residual!(dav, (V1,), st)
      NormR1 = calc_singles_norm(V1)
      NormR = NormR1
      maxNormR = max(maxNormR, NormR)
      converged = NormR < EC.options.eom.thr
      output_state(st, NormR, energies[st]; converged=converged)
      if !converged
        U1 .= new_singles_trial(EC, V1, energies[st], shift)
        add_trial_vector!(dav, (U1,), st, custom_dots)
        push!(states2do, st)
      end
    end
    output_iteration(it, maxNormR, time_ns() - t0, energies...)
    if isempty(states2do)
      println("Converged")
      break
    end
    states = states2do
  end
  # store the final eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", excitation_level)
  if mainfilename != ""
    for st in 1:nstates
      filename = mainfilename*"_$excitation_level"*"^$st"
      println("Saving $descr for state $st to $filename")
      get_eigenvector!(dav, (U1,), st)
      print_main_singles(U1, EC.options.print.ncoeff; info="State $st")
      save!(EC, filename, U1, description=descr)
    end
  end
  return energies
end

function eom_iterations2(EC::ECInfo{T}, method::ECMethod) where T
  t0 = time_ns()
  dc = (method.theory[1:2] == "DC")
  calc_intermediates4Jacobian(EC, method)
  nstates = EC.options.eom.nstates
  dav = Davidson(EC, nstates; hermitian=false)
  # first guess for U1 from CIS
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  states = [1:nstates;]
  energies = zeros(T, nstates)
  U1 = zeros(T, nvirt, nocc)
  U2 = zeros(T, nvirt, nvirt, nocc, nocc)
  Vecs = (U1, U2)
  custom_dots = (calc_cs_singles_dot, calc_cs_doubles_dot)
  # custom_dots = (calc_contra_cs_singles_dot, calc_contra_cs_doubles_dot)
  # load the CIS eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", excitation_level, false)
  if mainfilename != ""
    for st in 1:nstates
      filename = mainfilename*"_$excitation_level"*"^$st"
      if file_exists(EC, filename)
        println("Reading $descr from file $filename")
        load!(EC, filename, U1)
        add_trial_vector!(dav, Vecs, st, custom_dots)
      else
        error("File $filename not found, cannot read CIS eigenvector")
      end
    end
  else
    error("No file found for CIS eigenvectors")
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, Vecs, st)
      V1, V2 = calc_ccsd_vector_times_Jacobian(EC, Vecs...; dc=dc, with_rhs=false)
      add_product_vector!(dav, (V1,V2), st, custom_dots)
    end
    energies = perform!(dav)
    if do_refresh(dav, length(states))
      refresh!(dav, Vecs, custom_dots)
      output_iteration(it, -1.0, time_ns() - t0, energies...)
      states = [1:nstates;]
      continue
    end
    states2do = Int[]
    maxNormR = 0.0
    for st in 1:nstates
      get_residual!(dav, Vecs, st)
      NormR1 = calc_singles_norm(Vecs[1])
      NormR2 = calc_doubles_norm(Vecs[2])
      NormR = NormR1 + NormR2
      maxNormR = max(maxNormR, NormR)
      converged = NormR < EC.options.eom.thr
      output_state(st, NormR, energies[st]; converged=converged)
      if !converged
        new_trial_vector!(EC, Vecs, energies[st])
        add_trial_vector!(dav, Vecs, st, custom_dots)
        push!(states2do, st)
      end
    end
    output_iteration(it, maxNormR, time_ns() - t0, energies...)
    if isempty(states2do)
      println("Converged")
      break
    end
    states = states2do
  end
  # store the final eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", 1)
  if mainfilename != ""
    for st in 1:nstates
      get_eigenvector!(dav, Vecs, st)
      print_main_singles(Vecs[1], EC.options.print.ncoeff; info="State $st")
      for excitation_level in 1:2
        mainfilename, descr = save_or_start_file(EC, "X", excitation_level)
        filename = mainfilename*"_$excitation_level"*"^$st"
        println("Saving $descr for state $st to $filename")
        save!(EC, filename, Vecs[excitation_level], description=descr)
      end
    end
  end
  return energies
end

"""
    eom_u_iterations(EC::ECInfo, method::ECMethod)

  Perform unrestricted EOM-CIS iterations using the Davidson solver.
"""
function eom_u_iterations(EC::ECInfo{T}, method::ECMethod) where T
  t0 = time_ns()
  restrict = has_prefix(method, "R")
  nstates = EC.options.eom.nstates
  shift = EC.options.eom.shift
  dav = Davidson(EC, nstates; hermitian=true)
  # first guess for U1a, U1b
  nocca = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  nvirta = n_virt_orbs(EC)
  nvirtb = n_virtb_orbs(EC)
  states = [1:nstates;]
  energies = zeros(T, nstates)
  U1a = zeros(T, nvirta, nocca)
  U1b = zeros(T, nvirtb, noccb)
  V1a = zeros(T, nvirta, nocca)
  V1b = zeros(T, nvirtb, noccb)
  Vecs = (U1a, U1b)
  custom_dots = (calc_u_singles_dot, calc_u_singles_dot)
  # HOMO-LUMO guess
  # for restricted: request more states and filter to singlet excitations
  nstates_guess = restrict ? 10*nstates : nstates
  en_guess, veca_guess, vecb_guess = ucis_homo_lumo_guess(EC, nstates_guess)
  nva_guess = size(veca_guess, 1)
  noa_guess = size(veca_guess, 2)
  nvb_guess = size(vecb_guess, 1)
  nob_guess = size(vecb_guess, 2)
  nst = 0
  for iguess in axes(veca_guess, 3)
    U1a .= 0.0
    U1b .= 0.0
    U1a[1:nva_guess,end-noa_guess+1:end] = veca_guess[:,:,iguess]
    U1b[1:nvb_guess,end-nob_guess+1:end] = vecb_guess[:,:,iguess]
    if restrict
      spin_project!(EC, U1a, U1b)
      vnorm = sqrt(calc_singles_norm(U1a, U1b))
      if vnorm < 1.e-6
        continue  # skip triplet/spin-flip state
      end
      U1a ./= vnorm
      U1b ./= vnorm
    end
    nst += 1
    add_trial_vector!(dav, (U1a, U1b), nst, custom_dots)
    nst >= nstates && break
  end
  if restrict && nst < nstates
    @warn "Only found $nst singlet CIS guess states out of $nstates requested"
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, (U1a, U1b), st)
      V1a .= ucis_HU1(EC, U1a, U1b, :α)
      V1b .= ucis_HU1(EC, U1b, U1a, :β)
      if restrict
        spin_project!(EC, V1a, V1b)
      end
      add_product_vector!(dav, (V1a, V1b), st, custom_dots)
    end
    energies = perform!(dav)
    if do_refresh(dav, length(states))
      refresh!(dav, Vecs, custom_dots)
      output_iteration(it, -1.0, time_ns() - t0, energies...)
      states = [1:nstates;]
      continue
    end
    states2do = Int[]
    maxNormR = 0.0
    for st in 1:nstates
      get_residual!(dav, (V1a, V1b), st)
      if restrict
        spin_project!(EC, V1a, V1b)
      end
      NormR = calc_singles_norm(V1a, V1b)
      maxNormR = max(maxNormR, NormR)
      converged = NormR < EC.options.eom.thr
      output_state(st, NormR, energies[st]; converged=converged)
      if !converged
        new_singles_trial!(EC, U1a, U1b, V1a, V1b, energies[st], shift)
        if restrict
          spin_project!(EC, U1a, U1b)
        end
        add_trial_vector!(dav, (U1a, U1b), st, custom_dots)
        push!(states2do, st)
      end
    end
    output_iteration(it, maxNormR, time_ns() - t0, energies...)
    if isempty(states2do)
      println("Converged")
      break
    end
    states = states2do
  end
  # store the final eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", excitation_level)
  if mainfilename != ""
    for st in 1:nstates
      filename = mainfilename*"_$excitation_level"*"^$st"
      println("Saving $descr for state $st to $filename")
      get_eigenvector!(dav, (U1a, U1b), st)
      if restrict
        spin_project!(EC, U1a, U1b)
      end
      print_main_singles(U1a, EC.options.print.ncoeff; info="State $st α")
      print_main_singles(U1b, EC.options.print.ncoeff; info="State $st β")
      save!(EC, filename, U1a, U1b, description=descr)
    end
  end
  return energies
end

"""
    eom_u_iterations2(EC::ECInfo, method::ECMethod)

  Perform unrestricted EOM-CCSD/DCSD iterations using the Davidson solver.
"""
function eom_u_iterations2(EC::ECInfo{T}, method::ECMethod) where T
  t0 = time_ns()
  dc = (method.theory[1:2] == "DC")
  restrict = has_prefix(method, "R")
  calc_intermediates4Jacobian(EC, method)
  nstates = EC.options.eom.nstates
  dav = Davidson(EC, nstates; hermitian=false)
  # first guess for U1a, U1b from CIS
  nocca = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  nvirta = n_virt_orbs(EC)
  nvirtb = n_virtb_orbs(EC)
  states = [1:nstates;]
  energies = zeros(T, nstates)
  U1a = zeros(T, nvirta, nocca)
  U1b = zeros(T, nvirtb, noccb)
  U2a = zeros(T, nvirta, nvirta, nocca, nocca)
  U2b = zeros(T, nvirtb, nvirtb, noccb, noccb)
  U2ab = zeros(T, nvirta, nvirtb, nocca, noccb)
  Vecs = (U1a, U1b, U2a, U2b, U2ab)
  custom_dots = (calc_u_singles_dot, calc_u_singles_dot,
                 calc_samespin_doubles_dot, calc_samespin_doubles_dot, calc_ab_doubles_dot)
  # load the CIS eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", excitation_level, false)
  if mainfilename != ""
    for st in 1:nstates
      filename = mainfilename*"_$excitation_level"*"^$st"
      if file_exists(EC, filename)
        println("Reading $descr from file $filename")
        load!(EC, filename, U1a, U1b)
        add_trial_vector!(dav, Vecs, st, custom_dots)
      else
        error("File $filename not found, cannot read CIS eigenvector")
      end
    end
  else
    error("No file found for CIS eigenvectors")
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, Vecs, st)
      V1a, V1b, V2a, V2b, V2ab = calc_ccsd_vector_times_Jacobian(EC, Vecs...; dc=dc, with_rhs=false)
      if restrict
        spin_project!(EC, V1a, V1b, V2a, V2b, V2ab)
      end
      add_product_vector!(dav, (V1a, V1b, V2a, V2b, V2ab), st, custom_dots)
    end
    energies = perform!(dav)
    if do_refresh(dav, length(states))
      refresh!(dav, Vecs, custom_dots)
      output_iteration(it, -1.0, time_ns() - t0, energies...)
      states = [1:nstates;]
      continue
    end
    states2do = Int[]
    maxNormR = 0.0
    for st in 1:nstates
      get_residual!(dav, Vecs, st)
      if restrict
        spin_project!(EC, Vecs...)
      end
      NormR1 = calc_singles_norm(Vecs[1], Vecs[2])
      NormR2 = calc_doubles_norm(Vecs[3], Vecs[4], Vecs[5])
      NormR = NormR1 + NormR2
      maxNormR = max(maxNormR, NormR)
      converged = NormR < EC.options.eom.thr
      output_state(st, NormR, energies[st]; converged=converged)
      if !converged
        new_u_trial_vector!(EC, Vecs, energies[st])
        if restrict
          spin_project!(EC, Vecs...)
        end
        add_trial_vector!(dav, Vecs, st, custom_dots)
        push!(states2do, st)
      end
    end
    output_iteration(it, maxNormR, time_ns() - t0, energies...)
    if isempty(states2do)
      println("Converged")
      break
    end
    states = states2do
  end
  # store the final eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", 1)
  if mainfilename != ""
    for st in 1:nstates
      get_eigenvector!(dav, Vecs, st)
      if restrict
        spin_project!(EC, Vecs...)
      end
      print_main_singles(Vecs[1], EC.options.print.ncoeff; info="State $st α")
      print_main_singles(Vecs[2], EC.options.print.ncoeff; info="State $st β")
      for excitation_level in 1:2
        mainfilename, descr = save_or_start_file(EC, "X", excitation_level)
        filename = mainfilename*"_$excitation_level"*"^$st"
        println("Saving $descr for state $st to $filename")
        if excitation_level == 1
          save!(EC, filename, Vecs[1], Vecs[2], description=descr)
        else
          save!(EC, filename, Vecs[3], Vecs[4], Vecs[5], description=descr)
        end
      end
    end
  end
  return energies
end

"""
    ucis_HU1(EC::ECInfo, U1, U1os, spin)

  Compute H times U1 for unrestricted CIS.
  `U1` are the singles of `spin`∈{`:α`,`:β`},
  `U1os` are the opposite-spin singles.
"""
function ucis_HU1(EC::ECInfo, U1, U1os, spin)
  SP = EC.space
  isα = (spin == :α)
  o4s = space4spin('o', isα)
  v4s = space4spin('v', isα)
  m4s = space4spin('m', isα)
  o4os = space4spin('o', !isα)
  v4os = space4spin('v', !isα)

  f_mm = load2idx(EC, "f_"*m4s*m4s)
  f_oo = f_mm[SP[o4s], SP[o4s]]
  f_vv = f_mm[SP[v4s], SP[v4s]]

  @mtensor V1[a,i] := f_vv[a,c] * U1[c,i] - f_oo[k,i] * U1[a,k]
  # same-spin integrals
  int2 = ints2(EC, v4s*o4s*o4s*v4s)
  @mtensor V1[a,i] += int2[a,k,i,c] * U1[c,k]
  int2 = ints2(EC, v4s*o4s*v4s*o4s)
  @mtensor V1[a,i] -= int2[a,k,c,i] * U1[c,k]
  # opposite-spin integrals
  if isα
    int2 = ints2(EC, v4s*o4os*o4s*v4os)
    @mtensor V1[a,i] += int2[a,K,i,C] * U1os[C,K]
  else
    int2 = ints2(EC, o4os*v4s*v4os*o4s)
    @mtensor V1[A,I] += int2[k,A,c,I] * U1os[c,k]
  end
  return V1
end

"""
    ucis_homo_lumo_guess(EC, nstates)

  Generate an unrestricted CIS starting guess for `nstates` 
  by preparing a CIS matrix around the HOMO-LUMO and diagonalizing it.
"""
function ucis_homo_lumo_guess(EC::ECInfo{T}, nstates) where T
  SP = EC.space
  # number of open-shell orbitals in alpha and in beta
  nsa = length(SP['s'])
  nsb = length(SP['S'])
  noa = min(max(nstates + nsa, 5), n_occ_orbs(EC))
  nob = min(max(nstates + nsb, 5), n_occb_orbs(EC)) 
  nva = min(max(nstates + nsb, EC.options.davidson.maxdav), n_virt_orbs(EC))
  nvb = min(max(nstates + nsa, EC.options.davidson.maxdav), n_virtb_orbs(EC))
  spoa = SP['o'][end-noa+1:end]
  spva = SP['v'][1:nva]
  spob = SP['O'][end-nob+1:end]
  spvb = SP['V'][1:nvb]
  dim_a = nva * noa
  dim_b = nvb * nob
  dim = dim_a + dim_b
  HH = zeros(T, dim, dim)
  # α-α block
  f_mm = load2idx(EC, "f_mm")
  f_ooa = f_mm[spoa, spoa]
  f_vva = f_mm[spva, spva]
  HHaa = zeros(T, nva, noa, nva, noa)
  for i = 1:noa
    for j = 1:noa
      HHaa[:,i,:,j] = f_vva .- f_ooa[i,j]
    end
  end
  int2 = ints2(EC, spva, spoa, spoa, spva, :α)
  HHaa .+= permutedims(int2, (1,3,4,2))
  int2 = ints2(EC, spva, spoa, spva, spoa, :α)
  HHaa .-= permutedims(int2, (1,4,3,2))
  HH[1:dim_a, 1:dim_a] = reshape(HHaa, (dim_a, dim_a))
  # β-β block
  f_MM = load2idx(EC, "f_MM")
  f_oob = f_MM[spob, spob]
  f_vvb = f_MM[spvb, spvb]
  HHbb = zeros(T, nvb, nob, nvb, nob)
  for i = 1:nob
    for j = 1:nob
      HHbb[:,i,:,j] = f_vvb .- f_oob[i,j]
    end
  end
  int2 = ints2(EC, spvb, spob, spob, spvb, :β)
  HHbb .+= permutedims(int2, (1,3,4,2))
  int2 = ints2(EC, spvb, spob, spvb, spob, :β)
  HHbb .-= permutedims(int2, (1,4,3,2))
  HH[dim_a+1:dim, dim_a+1:dim] = reshape(HHbb, (dim_b, dim_b))
  # α-β coupling block (Coulomb only, no exchange between different spins)
  int2 = ints2(EC, spva, spob, spoa, spvb, :αβ)
  HHab = permutedims(int2, (1,3,4,2))
  HH[1:dim_a, dim_a+1:dim] = reshape(HHab, (dim_a, dim_b))
  int2 = ints2(EC, spoa, spvb, spva, spob, :αβ)
  HHba = permutedims(int2, (2,4,3,1))
  HH[dim_a+1:dim, 1:dim_a] = reshape(HHba, (dim_b, dim_a))
  vals, vecs = eigen(Hermitian(HH))
  vec_a = reshape(vecs[1:dim_a, 1:nstates], (nva, noa, nstates))
  vec_b = reshape(vecs[dim_a+1:dim, 1:nstates], (nvb, nob, nstates))
  return vals[1:nstates], vec_a, vec_b
end

"""
    new_singles_trial!(EC::ECInfo, U1a, U1b, R1a, R1b, omega, shift)

  Calculate new unrestricted singles trial vectors.
"""
function new_singles_trial!(EC::ECInfo, U1a, U1b, R1a, R1b, omega, shift)
  U1a .= new_u_singles_trial(EC, R1a, omega, shift, :α)
  U1b .= new_u_singles_trial(EC, R1b, omega, shift, :β)
  vnorm = sqrt(calc_singles_norm(U1a, U1b))
  U1a ./= vnorm
  U1b ./= vnorm
  return (U1a, U1b)
end

"""
    new_u_singles_trial(EC, R1a, R1b, omega, shift, spin)

  Calculate new unrestricted singles trial vector for `spin`.
"""
function new_u_singles_trial(EC, R1, omega, shift, spin)
  ϵo, ϵv = orbital_energies(EC, spin)
  U1 = deepcopy(R1)
  omega -= shift
  for I ∈ CartesianIndices(U1)
    a,i = Tuple(I)
    U1[I] /= -(ϵv[a] - ϵo[i] - omega)
  end
  return U1
end

"""
    new_u_trial_vector!(EC, vecs, omega)

  Calculate new unrestricted trial vector using residuals from `vecs`. 

  `vecs` is a tuple `(U1a, U1b, U2a, U2b, U2ab)`.
  Updates the vectors in place.
"""
function new_u_trial_vector!(EC, vecs, omega)
  ϵoa, ϵva = orbital_energies(EC)
  ϵob, ϵvb = orbital_energies(EC, :β)
  shift = EC.options.eom.shift
  # singles α
  new_singles_trial!(vecs[1], ϵoa, ϵva, omega, shift)
  # singles β
  new_singles_trial!(vecs[2], ϵob, ϵvb, omega, shift)
  NormU = calc_singles_norm(vecs[1], vecs[2])
  if length(vecs) > 2
    # doubles αα
    new_doubles_trial!(vecs[3], ϵoa, ϵva, omega, shift)
    # doubles ββ
    new_doubles_trial!(vecs[4], ϵob, ϵvb, omega, shift)
    # doubles αβ
    new_u_doubles_trial!(vecs[5], ϵoa, ϵva, ϵob, ϵvb, omega, shift)
    NormU += calc_doubles_norm(vecs[3], vecs[4], vecs[5])
  end
  for i in eachindex(vecs)
    vecs[i] ./= sqrt(NormU)
  end
end

"""
    new_u_doubles_trial!(Vec2, ϵoa, ϵva, ϵob, ϵvb, omega, shift)

  Calculate new αβ doubles trial vector.

  Updates the vector in place.
"""
function new_u_doubles_trial!(Vec2, ϵoa, ϵva, ϵob, ϵvb, omega, shift)
  omega -= shift
  for I ∈ CartesianIndices(Vec2)
    a,b,i,j = Tuple(I)
    Vec2[I] /= -(ϵva[a] + ϵvb[b] - ϵoa[i] - ϵob[j] - omega)
  end
  return Vec2
end

"""
    cis_homo_lumo_guess(EC, nstates)

  generate a CIS starting guess for nstates 
  by preparing a CIS matrix around the HOMO-LUMO and diagonalizing it
"""
function cis_homo_lumo_guess(EC::ECInfo{T}, nstates) where T
  noa = max(nstates, 5)
  nva = max(nstates, EC.options.davidson.maxdav)
  SP = EC.space
  noa = min(noa, length(SP['o']))
  nva = min(nva, length(SP['v']))
  spo = SP['o'][end-noa+1:end]
  spv = SP['v'][1:nva]
  f_mm = load2idx(EC, "f_mm")
  f_oo = f_mm[spo, spo]
  f_vv = f_mm[spv, spv]
  HH = zeros(T, nva, noa, nva, noa)
  for i = 1:noa
    for j = 1:noa
      HH[:,i,:,j] = f_vv .- f_oo[i,j]
    end
  end
  int2 = ints2(EC, spv, spo, spo, spv, :α) 
  HH .+= 2 * permutedims(int2, (1,3,4,2))
  int2 = ints2(EC, spv, spo, spv, spo, :α) 
  HH .-= permutedims(int2, (1,4,3,2))
  vals, vecs = eigen(Hermitian(reshape(HH, (nva*noa, nva*noa))))
  return vals[1:nstates], reshape(vecs[:,1:nstates], (nva, noa, nstates))
end

"""
    new_trial_vector!(EC, vecs, omega)

  Calculate new trial vector using residuals from `vecs`. 

  Updates the vector in place.
"""
function new_trial_vector!(EC, vecs, omega)
  ϵo, ϵv = orbital_energies(EC)
  shift = EC.options.eom.shift
  new_singles_trial!(vecs[1], ϵo, ϵv, omega, shift)
  NormU = calc_singles_norm(vecs[1])
  if length(vecs) > 1
    new_doubles_trial!(vecs[2], ϵo, ϵv, omega, shift)
    NormU += calc_doubles_norm(vecs[2])
  end
  for i in eachindex(vecs)
    vecs[i] ./= sqrt(NormU)
  end
end

"""
    new_singles_trial!(Vec1, ϵo, ϵv, omega, shift)

  Calculate new singles trial vector.

  Updates the vector in place.
"""
function new_singles_trial!(Vec1, ϵo, ϵv, omega, shift)
  omega -= shift
  for I ∈ CartesianIndices(Vec1)
    a,i = Tuple(I)
    Vec1[I] /= -(ϵv[a] - ϵo[i] - omega)
  end
  return Vec1
end

"""
    new_doubles_trial!(Vec2, ϵo, ϵv, omega, shift)

  Calculate new doubles trial vector.

  Updates the vector in place.
"""
function new_doubles_trial!(Vec2, ϵo, ϵv, omega, shift)
  omega -= shift
  for I ∈ CartesianIndices(Vec2)
    a,b,i,j = Tuple(I)
    Vec2[I] /= -(ϵv[a] + ϵv[b] - ϵo[i] - ϵo[j] - omega)
  end
  return Vec2
end

"""
    new_singles_trial(EC, R1, omega, shift)

  Calculate new singles trial vector.
"""
function new_singles_trial(EC, R1, omega, shift)
  ϵo, ϵv = orbital_energies(EC)
  U1 = deepcopy(R1)
  omega -= shift
  for I ∈ CartesianIndices(U1)
    a,i = Tuple(I)
    U1[I] /= -(ϵv[a] - ϵo[i] - omega)
  end
  vnorm = norm(U1)
  U1 ./= vnorm
  return U1
end


function calc_df_eom(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  print_info(method_name(method))

  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 1
    error("only implemented upto singles")
  end
  if is_unrestricted(method) || has_prefix(method, "R")
    error("open-shell not implemented")
  end

  energies = df_eom_iterations(EC, method)
end


function df_eom_iterations(EC::ECInfo{T}, method::ECMethod) where T
  t0 = time_ns()
  nstates = EC.options.eom.nstates
  shift = EC.options.eom.shift
  dav = Davidson(EC, nstates; hermitian=false)
  # first guess for U1
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  states = [1:nstates;]
  energies = zeros(T, nstates)
  U1 = zeros(T, nvirt, nocc)
  V1 = zeros(T, nvirt, nocc)
  Vecs = (U1,)
  custom_dots = (calc_cs_singles_dot,)
  # HOMO-LUMO guess
  en_guess, vec_guess = df_cis_homo_lumo_guess(EC, nstates)
  nv_guess = size(vec_guess, 1)
  no_guess = size(vec_guess, 2)
  for st in states
    U1 .= 0.0
    U1[1:nv_guess,end-no_guess+1:end] = vec_guess[:,:,st]
    add_trial_vector!(dav, Vecs, st, custom_dots)
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, Vecs, st)
      V1 .= df_cis_HU1(EC, U1)
      add_product_vector!(dav, (V1,), st, custom_dots)
    end
    energies = perform!(dav)
    if do_refresh(dav, length(states))
      refresh!(dav, Vecs, custom_dots)
      output_iteration(it, -1.0, time_ns() - t0, energies...)
      states = [1:nstates;]
      continue
    end
    states2do = Int[]
    maxNormR = 0.0
    for st in 1:nstates
      get_residual!(dav, (V1,), st)
      NormR1 = calc_singles_norm(V1)
      NormR = NormR1
      maxNormR = max(maxNormR, NormR)
      converged = NormR < EC.options.eom.thr
      output_state(st, NormR, energies[st]; converged=converged)
      if !converged
        U1 .= df_new_singles_trial(EC, V1, energies[st], shift)
        add_trial_vector!(dav, Vecs, st, custom_dots)
        push!(states2do, st)
      end
      #println(string(st) * "U1")
      #save!(EC, string(st) * "U1", U1)
      #display(U1)
    end
    output_iteration(it, maxNormR, time_ns() - t0, energies...)
    if isempty(states2do)
      println("Converged")
      break
    end
    states = states2do
  end
  # store the final eigenvectors
  excitation_level = 1
  mainfilename, descr = save_or_start_file(EC, "X", excitation_level)
  if mainfilename != ""
    for st in 1:nstates
      filename = mainfilename*"_$excitation_level"*"^$st"
      println("Saving $descr for state $st to $filename")
      get_eigenvector!(dav, (U1,), st)
      save!(EC, filename, U1, description=descr)
    end
  end
  #save!(EC, "U1", U1)
  return energies
end

"""
    df_cis_homo_lumo_guess(EC, nstates)

  generate a CIS starting guess for nstates 
  by preparing a CIS matrix around the HOMO-LUMO and diagonalizing it
"""
function df_cis_homo_lumo_guess(EC::ECInfo{T}, nstates) where T
  noa = max(nstates, 5)
  nva = max(nstates, EC.options.davidson.maxdav)
  SP = EC.space
  noa = min(noa, length(SP['o']))
  nva = min(nva, length(SP['v']))
  spo = SP['o'][end-noa+1:end]
  spv = SP['v'][1:nva]
  f_mm = load2idx(EC, "f_mm")
  f_oo = f_mm[spo, spo]
  f_vv = f_mm[spv, spv]
  HH = zeros(T, nva, noa, nva, noa)
  for i = 1:noa
    for j = 1:noa
      HH[:,i,:,j] = f_vv .- f_oo[i,j]
    end
  end
  
  mmLfile, mmL = mmap3idx(EC, "mmL")
  ovL = mmL[spo,spv,:]
  vvL = mmL[spv,spv,:]
  voL = mmL[spv,spo,:]
  ooL = mmL[spo,spo,:]
  
  @mtensor HH[a,i,b,j] += 2 * ovL[j,b,L] * voL[a,i,L]
  @mtensor HH[a,i,b,j] -= vvL[a,b,L] * ooL[j,i,L]
    
  close(mmLfile)  

  vals, vecs = eigen(Hermitian(reshape(HH, (nva*noa, nva*noa))))
  return vals[1:nstates], reshape(vecs[:,1:nstates], (nva, noa, nstates))
end

"""
    df_new_singles_trial(EC, R1, omega, shift)

  Calculate new singles trial vector with density fitting.
"""
function df_new_singles_trial(EC, R1, omega, shift)
  ϵo, ϵv = orbital_energies(EC)
  U1 = deepcopy(R1)
  omega -= shift
  for I ∈ CartesianIndices(U1)
    a,i = Tuple(I)
    U1[I] /= -(ϵv[a] - ϵo[i] - omega)
  end
  vnorm = norm(U1)
  U1 ./= vnorm
  return U1
end

end # module EOM
