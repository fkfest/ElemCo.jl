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
using ..ElemCo.CCTools
using ..ElemCo.OrbTools
using ..ElemCo.Outputs

export calc_eom
export calc_df_eom

include("cis.jl")

function calc_eom(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  print_info(method_name(method))

  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 1
    error("only implemented upto singles")
  end
  if is_unrestricted(method) || has_prefix(method, "R")
    error("open-shell not implemented")
  end

  energies = eom_iterations(EC, method)

end

function eom_iterations(EC::ECInfo, method::ECMethod)
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
  # HOMO-LUMO guess
  en_guess, vec_guess = cis_homo_lumo_guess(EC, nstates)
  nv_guess = size(vec_guess, 1)
  no_guess = size(vec_guess, 2)
  for st in states
    U1 .= 0.0
    U1[1:nv_guess,end-no_guess+1:end] = vec_guess[:,:,st]
    add_trial_vector!(dav, (U1,), st)
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, (U1,), st)
      V1 .= cis_HU1(EC, U1)
      add_product_vector!(dav, (V1,), st)
    end
    energies = perform!(dav)
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
        add_trial_vector!(dav, (U1,), st)
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
  # HOMO-LUMO guess
  en_guess, vec_guess = df_cis_homo_lumo_guess(EC, nstates)
  nv_guess = size(vec_guess, 1)
  no_guess = size(vec_guess, 2)
  for st in states
    U1 .= 0.0
    U1[1:nv_guess,end-no_guess+1:end] = vec_guess[:,:,st]
    add_trial_vector!(dav, (U1,), st)
  end
  println("Iter    Energy    Res       Time")
  for it in 1:EC.options.eom.maxit
    t1 = time_ns()
    for st in states
      get_current_trial_vector!(dav, (U1,), st)
      V1 .= df_cis_HU1(EC, U1)
      add_product_vector!(dav, (V1,), st)
    end
    energies = perform!(dav)
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
        add_trial_vector!(dav, (U1,), st)
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
