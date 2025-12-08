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
  print_info(method_name(method))

  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 2
    error("only implemented upto doubles")
  end
  if is_unrestricted(method) || has_prefix(method, "R")
    error("open-shell not implemented")
  end
  if highest_full_exc == 2
    energies = eom_iterations(EC, ECMethod("EOM-CCS"))
    energies = eom_iterations2(EC, method)
  else
    energies = eom_iterations(EC, method)
  end

end

function eom_iterations(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  nstates = EC.options.eom.nstates
  shift = EC.options.eom.shift
  dav = Davidson(EC, nstates; hermitian=true)
  # first guess for U1
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  states = [1:nstates;]
  energies = zeros(nstates)
  U1 = zeros(nvirt, nocc)
  V1 = zeros(nvirt, nocc)
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
      save!(EC, filename, U1, description=descr)
    end
  end
  return energies
end

function eom_iterations2(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  dc = (method.theory[1:2] == "DC")
  calc_intermediates4Jacobian(EC, method)
  nstates = EC.options.eom.nstates
  dav = Davidson(EC, nstates; hermitian=false)
  # first guess for U1 from CIS
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  states = [1:nstates;]
  energies = zeros(nstates)
  U1 = zeros(nvirt, nocc)
  U2 = zeros(nvirt, nvirt, nocc, nocc)
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
    cis_homo_lumo_guess(EC, nstates)

  generate a CIS starting guess for nstates 
  by preparing a CIS matrix around the HOMO-LUMO and diagonalizing it
"""
function cis_homo_lumo_guess(EC, nstates)
  noa = nstates
  nva = max(nstates, EC.options.davidson.maxdav)
  SP = EC.space
  noa = min(noa, length(SP['o']))
  nva = min(nva, length(SP['v']))
  spo = SP['o'][end-noa+1:end]
  spv = SP['v'][1:nva]
  f_mm = load2idx(EC, "f_mm")
  f_oo = f_mm[spo, spo]
  f_vv = f_mm[spv, spv]
  HH = zeros(nva, noa, nva, noa)
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


function df_eom_iterations(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  nstates = EC.options.eom.nstates
  shift = EC.options.eom.shift
  dav = Davidson(EC, nstates; hermitian=false)
  # first guess for U1
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  states = [1:nstates;]
  energies = zeros(nstates)
  U1 = zeros(nvirt, nocc)
  V1 = zeros(nvirt, nocc)
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
function df_cis_homo_lumo_guess(EC, nstates)
  noa = nstates
  nva = max(nstates, EC.options.davidson.maxdav)
  SP = EC.space
  noa = min(noa, length(SP['o']))
  nva = min(nva, length(SP['v']))
  spo = SP['o'][end-noa+1:end]
  spv = SP['v'][1:nva]
  f_mm = load2idx(EC, "f_mm")
  f_oo = f_mm[spo, spo]
  f_vv = f_mm[spv, spv]
  HH = zeros(nva, noa, nva, noa)
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
