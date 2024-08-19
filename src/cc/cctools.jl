"""
    CCTools

A collection of tools for working with coupled cluster theory.
"""
module CCTools
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.FockFactory
using ..ElemCo.OrbTools

export calc_fock_matrix, calc_HF_energy
export calc_singles_energy_using_dfock
export update_singles, update_doubles, update_singles!, update_doubles!, update_triples!, update_deco_doubles, update_deco_triples
export calc_singles_norm, calc_doubles_norm, calc_triples_norm, calc_contra_singles_norm, calc_contra_doubles_norm, calc_deco_doubles_norm, calc_deco_triples_norm
export read_starting_guess4amplitudes, save_current_singles, save_current_doubles
export transform_amplitudes2lagrange_multipliers!
export try2save_amps!, try2start_amps, try2save_singles!, try2save_doubles!, try2start_singles, try2start_doubles
export contra2covariant
export spin_project!, spin_project_amplitudes
export clean_cs_triples!
export calc_cs_singles_dot, calc_u_singles_dot
export calc_cs_doubles_dot, calc_samespin_doubles_dot, calc_ab_doubles_dot
export calc_cs_triples_dot, calc_samespin_triples_dot, calc_mixedspin_triples_dot
export triples_4ext!

""" 
    calc_fock_matrix(EC::ECInfo, closed_shell)

  Calculate fock matrix from FCIDump
"""
function calc_fock_matrix(EC::ECInfo, closed_shell)
  t1 = time_ns()
  if closed_shell
    fock = gen_fock(EC)
    save!(EC, "f_mm", fock)
    save!(EC, "f_MM", fock)
    eps = diag(fock)
    println("Occupied orbital energies: ", eps[EC.space['o']])
    save!(EC, "e_m", eps)
    save!(EC, "e_M", eps)
  else
    fock = gen_fock(EC, :α)
    eps = diag(fock)
    println("Occupied α orbital energies: ", eps[EC.space['o']])
    save!(EC, "f_mm", fock)
    save!(EC, "e_m", eps)
    fock = gen_fock(EC, :β)
    eps = diag(fock)
    println("Occupied β orbital energies: ", eps[EC.space['O']])
    save!(EC,"f_MM", fock)
    save!(EC,"e_M", eps)
  end
  t1 = print_time(EC,t1,"fock matrix",1)
end

""" 
    calc_HF_energy(EC::ECInfo, closed_shell)

  Calculate HF energy from FCIDump and EC info. 
"""
function calc_HF_energy(EC::ECInfo, closed_shell)
  SP = EC.space
  if closed_shell
    ϵo = load1idx(EC,"e_m")[SP['o']]
    EHF = sum(ϵo) + sum(diag(ints1(EC,"oo"))) + EC.fd.int0
  else
    ϵo = load1idx(EC,"e_m")[SP['o']]
    ϵob = load1idx(EC,"e_M")[SP['O']]
    EHF = 0.5*(sum(ϵo)+sum(ϵob) + sum(diag(ints1(EC, "oo"))) + sum(diag(ints1(EC, "OO")))) + EC.fd.int0
  end
  return EHF
end

"""
    spin_project!(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab)

  Spin-project singles and doubles amplitudes/residuals.

  Only possible for high-spin states.
"""
function spin_project!(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab)
  SP = EC.space
  @assert length(SP['S']) == 0 " Spin-projection only possible for high-spin states!"
  soa = subspace_in_space(SP['s'], SP['o'])
  svb = subspace_in_space(SP['s'], SP['V'])
  @assert length(soa) == length(svb)
  doa = setdiff(1:length(SP['o']), soa)
  @assert length(doa) == length(SP['O'])
  dvb =setdiff(1:length(SP['V']), svb)
  @assert length(dvb) == length(SP['v'])

  # calc closed-shell part of spin-restricted T2
  # ``T^{ij}_{ab} = \frac{1}{6} ( ^{αα}T^{ij}_{ab} + ^{ββ}T^{ij}_{ab} + 2 ^{αβ}T^{ij}_{ab} + ^{αβ}T^{ij}_{ba} +2 ^{αβ}T^{ji}_{ba} + ^{αβ}T^{ji}_{ab})``
  T2abc = T2ab[:,dvb,doa,:]
  @tensoropt T2ab[:,dvb,doa,:][a,b,i,j] = (1/6) * (T2a[:,:,doa,doa][a,b,i,j] + T2b[dvb,dvb,:,:][a,b,i,j] + 2*T2abc[a,b,i,j] + T2abc[b,a,i,j] + 2*T2abc[b,a,j,i] + T2abc[a,b,j,i])
  T2abc = nothing
  # calc ``T^{ij}_{at} = \frac{1}{3} ( ^{ββ}T^{ij}_{at} + 2 ^{αβ}T^{ij}_{at} + ^{αβ}T^{ji}_{at})``
  Tvsdd = T2ab[:,svb,doa,:]
  @tensoropt T2ab[:,svb,doa,:][a,t,i,j] = (1/3) * (T2b[dvb,svb,:,:][a,t,i,j] + 2*Tvsdd[a,t,i,j] + Tvsdd[a,t,j,i])
  Tvsdd = nothing
  # calc ``T^{tj}_{ab} = \frac{1}{3} ( ^{αα}T^{tj}_{ab} + 2 ^{αβ}T^{tj}_{ab} + ^{αβ}T^{tj}_{ba})``
  Tvvsd = T2ab[:,dvb,soa,:]
  @tensoropt T2ab[:,dvb,soa,:][a,b,t,j] = (1/3) * (T2a[:,:,soa,doa][a,b,t,j] + 2*Tvvsd[a,b,t,j] + Tvvsd[b,a,t,j])
  Tvvsd = nothing

  if length(T1b) > 0
    ms2 = length(soa)
    @tensoropt T1add[a,i] := T2ab[:,svb,soa,:][a,t,t,i]
    T1c = (1/(2+ms2))*(T1b[dvb,:] - T1a[:,doa] - T1add)
    for i in 1:length(doa)
      T2ab[:,svb,soa,i] .+= T1c[:,i]
    end
    @tensoropt T1add[a,i] = 0.5*T2ab[:,svb,soa,:][a,t,t,i]
    T1 = 0.5 * (T1a[:,doa] + T1b[dvb,:])
    T1a[:,doa] .= T1 - T1add
    T1b[dvb,:] .= T1 + T1add
  end
  @tensoropt T2a[:,:,:,doa][a,b,i,j] = T2ab[:,dvb,:,:][a,b,i,j] - T2ab[:,dvb,:,:][b,a,i,j]
  @tensoropt T2a[:,:,doa,soa][a,b,i,j] = T2a[:,:,soa,doa][b,a,j,i]
  @tensoropt T2b[dvb,:,:,:][a,b,i,j] = T2ab[:,:,doa,:][a,b,i,j] - T2ab[:,:,doa,:][a,b,j,i]
  @tensoropt T2b[svb,dvb,:,:][a,b,i,j] = T2b[dvb,svb,:,:][b,a,j,i]
end

"""
    spin_project_amplitudes(EC::ECInfo, with_singles=true)

  Spin-project singles (if with_singles) and doubles amplitudes 
  from files `"T_vo"`, `"T_VO"`, `"T_vvoo"`,
  `"T_VVOO"` and `"T_vVoO"`.
"""
function spin_project_amplitudes(EC::ECInfo, with_singles=true)
  if with_singles
    T1a = load2idx(EC, "T_vo")
    T1b = load2idx(EC, "T_VO")
  else
    T1a = T1b = zeros(0, 0)
  end
  T2a = load4idx(EC, "T_vvoo")
  T2b = load4idx(EC, "T_VVOO")
  T2ab = load4idx(EC, "T_vVoO")
  spin_project!(EC, T1a, T1b, T2a, T2b, T2ab)
  if with_singles
    save!(EC, "T_vo", T1a)
    save!(EC, "T_VO", T1b)
  end
  save!(EC, "T_vvoo", T2a)
  save!(EC, "T_VVOO", T2b)
  save!(EC, "T_vVoO", T2ab)
end

"""
    calc_singles_energy_using_dfock(EC::ECInfo, T1; fock_only=false)

  Calculate coupled-cluster closed-shell singles energy 
  using dressed fock matrix.

  if `fock_only` is true, the energy will be calculated using only non-dressed fock matrix.
  Returns total energy, SS, OS, and Openshell (0.0) contributions
  as `OutDict` with keys (`E`, `ESS`, `EOS`, `EO`).
"""
function calc_singles_energy_using_dfock(EC::ECInfo, T1; fock_only=false)
  SP = EC.space
  ET1 = 0.0
  if length(T1) > 0
    if fock_only
      ET1SS = ET1OS = ET1 = 0.0
    else
      if !file_exists(EC, "dfc_ov") || !file_exists(EC, "dfe_ov")
        error("Files dfc_ov and dfe_ov are required in calc_singles_energy_using_dfock!")
      end
      dfockc_ov = load2idx(EC, "dfc_ov")
      dfocke_ov = load2idx(EC, "dfe_ov")
      @tensoropt begin
        ET1d = T1[a,i] * dfockc_ov[i,a] 
        ET1ex = T1[a,i] * dfocke_ov[i,a]
      end
      ET1SS = ET1d - ET1ex
      ET1OS = ET1d
      ET1 = ET1SS + ET1OS
    end
    fov = load2idx(EC,"f_mm")[SP['o'],SP['v']] 
    @tensoropt ET1 += 2.0*(fov[i,a] * T1[a,i])
  end
  return OutDict("E"=>ET1, "ESS"=>ET1SS, "EOS"=>ET1OS, "EO"=>0.0)
end


"""
    update_singles(R1, ϵo, ϵv, shift)

  Calculate update for singles amplitudes.
"""
function update_singles(R1, ϵo, ϵv, shift)
  ΔT1 = deepcopy(R1)
  for I ∈ CartesianIndices(ΔT1)
    a,i = Tuple(I)
    ΔT1[I] /= -(ϵv[a] - ϵo[i] + shift)
  end
  return ΔT1
end

"""
    update_singles(EC::ECInfo, R1; spincase::Symbol=:α, use_shift=true)

  Calculate update for singles amplitudes for a given `spincase`∈{`:α`,`:β`}.
"""
function update_singles(EC::ECInfo, R1; spincase::Symbol=:α, use_shift=true)
  shift = use_shift ? EC.options.cc.shifts : 0.0
  if spincase == :α
    ϵo, ϵv = orbital_energies(EC)
    return update_singles(R1, ϵo, ϵv, shift)
  else
    ϵob, ϵvb = orbital_energies(EC, :β)
    return update_singles(R1, ϵob, ϵvb, shift)
  end
end

"""
    update_doubles(R2, ϵo1, ϵv1, ϵo2, ϵv2, shift)

  Calculate update for doubles amplitudes.
"""
function update_doubles(R2, ϵo1, ϵv1, ϵo2, ϵv2, shift, antisymmetrize=false)
  ΔT2 = deepcopy(R2)
  if antisymmetrize
    ΔT2 -= permutedims(R2,(1,2,4,3))
  end
  for I ∈ CartesianIndices(ΔT2)
    a,b,i,j = Tuple(I)
    ΔT2[I] /= -(ϵv1[a] + ϵv2[b] - ϵo1[i] - ϵo2[j] + shift)
  end
  return ΔT2
end

"""
    update_doubles(EC::ECInfo, R2; spincase::Symbol=:α, antisymmetrize=false, use_shift=true)

  Calculate update for doubles amplitudes for a given `spincase`∈{`:α`,`:β`,`:αβ`}.
"""
function update_doubles(EC::ECInfo, R2; spincase::Symbol=:α, antisymmetrize=false, use_shift=true)
  shift = use_shift ? EC.options.cc.shiftp : 0.0
  if spincase == :α
    ϵo, ϵv = orbital_energies(EC)
    return update_doubles(R2, ϵo, ϵv, ϵo, ϵv, shift, antisymmetrize)
  elseif spincase == :β
    ϵob, ϵvb = orbital_energies(EC, :β)
    return update_doubles(R2, ϵob, ϵvb, ϵob, ϵvb, shift, antisymmetrize)
  else
    ϵo, ϵv = orbital_energies(EC)
    ϵob, ϵvb = orbital_energies(EC, :β)
    return update_doubles(R2, ϵo, ϵv, ϵob, ϵvb, shift, antisymmetrize)
  end
end

"""
    update_singles!(EC::ECInfo, T1, R1)

  Update singles amplitudes in `T1` with `R1`.
"""
function update_singles!(EC::ECInfo, T1, R1)
  T1 .+= update_singles(EC, R1)
end

"""
    update_singles!(EC::ECInfo, T1a, T1b, R1a, R1b)

  Update singles amplitudes in `T1a`, `T1b` with `R1a`, `R1b`.
"""
function update_singles!(EC::ECInfo, T1a, T1b, R1a, R1b)
  T1a .+= update_singles(EC, R1a)
  T1b .+= update_singles(EC, R1b; spincase=:β)
end

"""
    update_doubles!(EC::ECInfo, T2, R2)

  Update doubles amplitudes in `T2` with `R2`.
"""
function update_doubles!(EC::ECInfo, T2, R2)
  T2 .+= update_doubles(EC, R2)
end

"""
    update_doubles!(EC::ECInfo, T2a, T2b, T2ab, R2a, R2b, R2ab)

  Update doubles amplitudes in `T2a`, `T2b`, `T2ab` with `R2a`, `R2b`, `R2ab`.
"""
function update_doubles!(EC::ECInfo, T2a, T2b, T2ab, R2a, R2b, R2ab)
  T2a .+= update_doubles(EC, R2a)
  T2b .+= update_doubles(EC, R2b; spincase=:β)
  T2ab .+= update_doubles(EC, R2ab; spincase=:αβ)
end

"""
    update_triples!(EC::ECInfo, T3a, T3b, T3aab, T3abb, R3a, R3b, R3aab, R3abb)

  Update triples amplitudes in `T3a`, `T3b`, `T3aab` and `T3abb` with `R3a`, `R3b`, `R3aab` and `R3abb`.
"""
function update_triples!(EC::ECInfo, T3a, T3b, T3aab, T3abb, R3a, R3b, R3aab, R3abb)
  T3a .+= update_triples(EC, R3a; spincase=:α)
  T3b .+= update_triples(EC, R3b; spincase=:β)
  T3aab .+= update_triples(EC, R3aab; spincase=:ααβ)
  T3abb .+= update_triples(EC, R3abb; spincase=:αββ)
end

"""
    update_triples!(EC::ECInfo, T3, R3)

  Update triples amplitudes in `T3`, with `R3`.
"""
function update_triples!(EC::ECInfo, T3, R3)
  T3 .+= update_triples(EC, R3)
end

"""
    update_triples(EC::ECInfo, R3; spincase::Symbol=:α, antisymmetrize=false, use_shift=true)

  Calculate update for triples amplitudes for a given `spincase`∈{`:α`,`:β`,`:ααβ`,`:αββ`}.
"""
function update_triples(EC::ECInfo, R3; spincase::Symbol=:α, use_shift=true)
  shift = use_shift ? EC.options.cc.shiftp : 0.0
  if spincase == :α
    ϵo, ϵv = orbital_energies(EC)
    return update_triples(R3, ϵo, ϵv, ϵo, ϵv, ϵo, ϵv, shift)
  elseif spincase == :β
    ϵob, ϵvb = orbital_energies(EC, :β)
    return update_triples(R3, ϵob, ϵvb, ϵob, ϵvb, ϵob, ϵvb, shift)
  elseif spincase == :ααβ
    ϵo, ϵv = orbital_energies(EC)
    ϵob, ϵvb = orbital_energies(EC, :β)
    return update_triples(R3, ϵo, ϵv, ϵo, ϵv, ϵob, ϵvb, shift)
  elseif spincase == :αββ
    ϵo, ϵv = orbital_energies(EC)
    ϵob, ϵvb = orbital_energies(EC, :β)
    return update_triples(R3, ϵo, ϵv, ϵob, ϵvb, ϵob, ϵvb, shift)
  else
    error("Unexpected spin case $spincase.")
  end
end

"""
    update_triples(R3, ϵo1, ϵv1, ϵo2, ϵv2, ϵo3, ϵv3, shift)

  Calculate update for triples amplitudes.
"""
function update_triples(R3, ϵo1, ϵv1, ϵo2, ϵv2, ϵo3, ϵv3, shift)
  ΔT3 = deepcopy(R3)
  for I ∈ CartesianIndices(ΔT3)
    a,b,c,i,j,k = Tuple(I)
    ΔT3[I] /= -(ϵv1[a] + ϵv2[b] + ϵv3[c] - ϵo1[i] - ϵo2[j] - ϵo3[k] + shift)
  end
  return ΔT3
end


"""
    update_deco_doubles(EC, R2; use_shift=true)

  Update decomposed doubles amplitudes.
  
  If `R2` is ``R^{ij}_{ab}``, the update is calculated using
  `update_doubles(EC, R2, use_shift=use_shift)`.
"""
function update_deco_doubles(EC, R2; use_shift=true)
  if ndims(R2) == 4
    return update_doubles(EC, R2; use_shift)
  else
    shift = use_shift ? EC.options.cc.shiftp : 0.0
    ΔT2 = deepcopy(R2)
    ϵX = load1idx(EC,"e_X")
    for I ∈ CartesianIndices(ΔT2)
      X,Y = Tuple(I)
      ΔT2[I] /= -(ϵX[X] + ϵX[Y] + shift)
    end
    return ΔT2
  end
end

"""
    update_deco_triples(EC, R3, use_shift=true)

  Update decomposed triples amplitudes.

  Note that the sign of the residual is opposite
  to the usual definition of the triples residual
  and therefore the update is calculated using 
  a positive denominator...
"""
function update_deco_triples(EC, R3, use_shift=true)
  shift = use_shift ? EC.options.cc.shiftt : 0.0
  ΔT3 = deepcopy(R3)
  ϵX = load1idx(EC,"e_X")
  for I ∈ CartesianIndices(ΔT3)
    X,Y,Z = Tuple(I)
    ΔT3[I] /= (ϵX[X] + ϵX[Y] + ϵX[Z] + shift)
  end
  return ΔT3
end

"""
    calc_singles_norm(T1)

  Calculate squared norm of closed-shell singles amplitudes.
"""
function calc_singles_norm(T1)
  @tensor NormT1 = 2.0*T1[a,i]*T1[a,i]
  return NormT1
end

"""
    calc_contra_singles_norm(T1)

  Calculate squared norm of closed-shell contravariant singles amplitudes.
"""
function calc_contra_singles_norm(T1)
  @tensor NormT1 = 0.5*T1[a,i]*T1[a,i]
  return NormT1
end

"""
    calc_singles_norm(T1a, T1b)

  Calculate squared norm of unrestricted singles amplitudes.
"""
function calc_singles_norm(T1a, T1b)
  @tensor begin
    NormT1 = T1a[a,i]*T1a[a,i]
    NormT1 += T1b[a,i]*T1b[a,i]
  end
  return NormT1
end

"""
    calc_contra_singles_norm(T1a, T1b)

  Calculate squared norm of unrestricted singles amplitudes 
  (same as `calc_singles_norm(T1a, T1b)`).
"""
function calc_contra_singles_norm(T1a, T1b)
  return calc_singles_norm(T1a, T1b)
end

"""
    calc_cs_singles_dot(T1, T1_)

  Calculate dot product of closed-shell singles amplitudes.
"""
function calc_cs_singles_dot(T1::Matrix{Float64}, T1_::Matrix{Float64})
  @tensor DotT1 = 2.0*T1[a,i]*T1_[a,i]
  return DotT1::Float64
end
calc_cs_singles_dot(T1, T1_) = error("calc_cs_singles_dot: T1 and T1_ must be matrices!")

"""
    calc_u_singles_dot(T1, T1_)

  Calculate dot of unrestricted singles amplitudes.
"""
function calc_u_singles_dot(T1::Matrix{Float64}, T1_::Matrix{Float64})
  @tensor begin
    DotT1 = T1[a,i]*T1_[a,i]
  end
  return DotT1::Float64
end
calc_u_singles_dot(T1, T1_) = error("calc_u_singles_dot: T1 and T1_ must be matrices!")

"""
    calc_doubles_norm(T2)

  Calculate squared norm of closed-shell doubles amplitudes.
"""
function calc_doubles_norm(T2)
  @tensoropt NormT2 = (2.0*T2[a,b,i,j] - T2[b,a,i,j])*T2[a,b,i,j]
  return NormT2
end

"""
    calc_contra_doubles_norm(T2)

  Calculate squared norm of closed-shell contravariant doubles amplitudes.
"""
function calc_contra_doubles_norm(T2)
  @tensoropt NormT2 = (2.0*T2[a,b,i,j] + T2[b,a,i,j])*T2[a,b,i,j]
  return NormT2/3.0
end

"""
    calc_doubles_norm(T2a, T2b, T2ab)

  Calculate squared norm of unrestricted doubles amplitudes.
"""
function calc_doubles_norm(T2a, T2b, T2ab)
  @tensoropt begin
    NormT2 = 0.25*(T2a[a,b,i,j]*T2a[a,b,i,j])
    NormT2 += 0.25*(T2b[a,b,i,j]*T2b[a,b,i,j])
    NormT2 += T2ab[a,b,i,j]*T2ab[a,b,i,j]
  end
  return NormT2
end

"""
    calc_cs_doubles_dot(T2, T2_)

  Calculate dot of closed-shell doubles amplitudes.
"""
function calc_cs_doubles_dot(T2::Array{Float64,4}, T2_::Array{Float64,4})
  @tensoropt DotT2 = (2.0*T2[a,b,i,j] - T2[b,a,i,j])*T2_[a,b,i,j]
  return DotT2::Float64
end
calc_cs_doubles_dot(T2, T2_) = error("calc_cs_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_samespin_doubles_dot(T2, T2_)

  Calculate dot of unrestricted same-spin doubles amplitudes.
"""
function calc_samespin_doubles_dot(T2::Array{Float64,4}, T2_::Array{Float64,4})
  @tensoropt begin
    DotT2 = 0.25*(T2[a,b,i,j]*T2_[a,b,i,j])
  end
  return DotT2::Float64
end
calc_samespin_doubles_dot(T2, T2_) = error("calc_samespin_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_ab_doubles_dot(T2, T2_)

  Calculate dot of unrestricted αβ doubles amplitudes.
"""
function calc_ab_doubles_dot(T2::Array{Float64,4}, T2_::Array{Float64,4})
  @tensoropt begin
    DotT2 = T2[a,b,i,j]*T2_[a,b,i,j]
  end
  return DotT2::Float64
end
calc_ab_doubles_dot(T2, T2_) = error("calc_ab_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_triples_norm(T3aaa, T3bbb, T3abb, T3aab)

  Calculate squared norm of unrestricted triples amplitudes.
"""
function calc_triples_norm(T3aaa, T3bbb, T3abb, T3aab)
  @tensoropt begin
    NormT3 = (1/36)*(T3aaa[a,b,c,i,j,k]*T3aaa[a,b,c,i,j,k])
    NormT3 += (1/36)*(T3bbb[a,b,c,i,j,k]*T3bbb[a,b,c,i,j,k])
    NormT3 += 0.25*(T3abb[a,b,c,i,j,k]*T3abb[a,b,c,i,j,k])
    NormT3 += 0.25*(T3aab[a,b,c,i,j,k]*T3aab[a,b,c,i,j,k])
  end
  return NormT3
end

"""
    calc_triples_norm(T3)

  Calculate squared norm of triples amplitudes.
"""
function calc_triples_norm(T3)
  NormT3 = 0.0
  nocc = size(T3, 6)
  for k = 1:nocc 
    for j = 1:k
      prefac = (j == k) ? 1.0 : 2.0
      for i = 1:j
        fac = prefac 
        if i == j 
          if j == k
            continue
          end 
          fac = 1.0
        end
        T3_ijk = @view T3[:,:,:,i,j,k]
        @tensoropt begin
          NormT3_ = 4*(T3_ijk[a,b,c]*T3_ijk[a,b,c])
          NormT3_ -= 2*(T3_ijk[a,b,c]*T3_ijk[a,c,b])
          NormT3_ -= 2*(T3_ijk[a,b,c]*T3_ijk[c,b,a])
          NormT3_ -= 2*(T3_ijk[a,b,c]*T3_ijk[b,a,c])
          NormT3_ += (T3_ijk[a,b,c]*T3_ijk[c,a,b])
          NormT3_ += (T3_ijk[a,b,c]*T3_ijk[b,c,a])
        end
        NormT3 += fac*NormT3_
      end
    end
  end
  return NormT3
end

"""
    calc_cs_triples_dot(T3, T3_)

  Calculate dot of closed-shell triples amplitudes.
"""
function calc_cs_triples_dot(T3::Array{Float64,6}, T3_::Array{Float64,6})
  DotT3 = 0.0
  nocc = size(T3, 6)
  for k = 1:nocc 
    for j = 1:k
      prefac = (j == k) ? 1.0 : 2.0
      for i = 1:j
        fac = prefac 
        if i == j 
          if j == k
            continue
          end 
          fac = 1.0
        end
        T3_ijk = @view T3[:,:,:,i,j,k]
        T3_ijk_ = @view T3_[:,:,:,i,j,k]
        @tensoropt begin
          DotT3_ = 4*(T3_ijk[a,b,c]*T3_ijk_[a,b,c])
          DotT3_ -= 2*(T3_ijk[a,b,c]*T3_ijk_[a,c,b])
          DotT3_ -= 2*(T3_ijk[a,b,c]*T3_ijk_[c,b,a])
          DotT3_ -= 2*(T3_ijk[a,b,c]*T3_ijk_[b,a,c])
          DotT3_ += (T3_ijk[a,b,c]*T3_ijk_[c,a,b])
          DotT3_ += (T3_ijk[a,b,c]*T3_ijk_[b,c,a])
        end
        DotT3 += fac*DotT3_
      end
    end
  end
  return DotT3::Float64
end
calc_cs_triples_dot(T3, T3_) = error("calc_cs_triples_dot not implemented for this type of T3")

"""
    calc_samespin_triples_dot(T3, T3_)

  Calculate dot of unrestricted same-spin triples amplitudes.
"""
function calc_samespin_triples_dot(T3::Array{Float64,6}, T3_::Array{Float64,6})
  @tensoropt begin
    DotT3 = (1/36)*(T3[a,b,c,i,j,k]*T3_[a,b,c,i,j,k])
  end
  return DotT3::Float64
end
calc_samespin_triples_dot(T3, T3_) = error("calc_samespin_triples_dot not implemented for this type of T3")

"""
    calc_mixedspin_triples_dot(T3, T3_)

  Calculate dot of unrestricted mixed-spin triples amplitudes.
"""
function calc_mixedspin_triples_dot(T3::Array{Float64,6}, T3_::Array{Float64,6})
  @tensoropt begin
    DotT3 = 0.25*(T3[a,b,c,i,j,k]*T3_[a,b,c,i,j,k])
  end
  return DotT3::Float64
end
calc_mixedspin_triples_dot(T3, T3_) = error("calc_mixedspin_triples_dot not implemented for this type of T3")

"""
    clean_cs_triples!(T3)

  Clean closed-shell triples amplitudes by setting ``T^{iii}_{abc} = T^{ijk}_{aaa} = 0``.
"""
function clean_cs_triples!(T3)
  nocc = size(T3, 6)
  diagindx = [CartesianIndex(i,i,i) for i in 1:nocc]
  T3[:,:,:,diagindx] .= 0.0
  nvirt = size(T3, 3)
  diagindx = [CartesianIndex(i,i,i) for i in 1:nvirt]
  T3[diagindx,:,:,:] .= 0.0
  return T3
end
"""
    calc_contra_doubles_norm(T2a, T2b, T2ab)

  Calculate squared norm of unrestricted doubles amplitudes
  (the same as `calc_doubles_norm`)
"""
function calc_contra_doubles_norm(T2a, T2b, T2ab)
  return calc_doubles_norm(T2a, T2b, T2ab)
end

"""
    calc_deco_doubles_norm(T2, tT2=Float64[])

  Calculate squared norm of doubles (for decomposed doubles: without contravariant!)
  T2 are decomposed doubles amplitudes `T2[X,Y]`=``T_{XY}`` or
  full doubles amplitudes `T2[a,b,i,j]`=``T^{ij}_{ab}``. 
  
  If the contravariant amplitude `tT2` is provided, 
  the norm will be calculated as ``T_{XY} T̃_{XY}``.
"""
function calc_deco_doubles_norm(T2, tT2=Float64[])
  if ndims(T2) == 4
    normT2 = calc_doubles_norm(T2)
  else
    if length(tT2) > 0
      @tensoropt normT2 = T2[X,Y] * tT2[X,Y]
    else
      @tensoropt normT2 = T2[X,Y] * T2[X,Y]
    end
  end
  return normT2
end

"""
    calc_deco_triples_norm(T3)

  Calculate a *simple* norm of triples (without contravariant!)
"""
function calc_deco_triples_norm(T3)
  @tensoropt NormT3 = T3[X,Y,Z] * T3[X,Y,Z]
  return NormT3
end

"""
    save_or_start_file(EC::ECInfo, type, excitation_level, save=true)

  Return filename and description for saving or starting amplitudes/lagrange multipliers.

  `type` is either `"T"` for amplitudes or `"LM"` for Lagrange multipliers.
  `excitation_level` is the excitation level of the amplitudes (1, 2 etc.)
  If `save` is true, the filename for saving is returned, otherwise the filename for starting.
"""
function save_or_start_file(EC::ECInfo, type, excitation_level, save=true)
  mainfilename = descr = ""
  descr = ["singles", "doubles", "triples", "quadruples"][excitation_level]
  if type == "T"
    descr *= " amplitudes"
    mainfilename = save ? EC.options.cc.save : EC.options.cc.start
  elseif type == "LM"
    descr *= " Lagrange multipliers"
    mainfilename = save ? EC.options.cc.save_lm : EC.options.cc.start_lm
  else
    error("unknown type $type")
  end
  return mainfilename, descr
end

"""
    try2save_amps!(EC::ECInfo, ::Val{excitation_level}, amps...; type="T")

  Save amplitudes (type="T") or Lagrange multipliers (type="LM") 
  to file `EC.options.cc.save[_lm]*"_excitation_level"`.
"""
function try2save_amps!(EC::ECInfo, ::Val{excitation_level}, amps...; type="T") where excitation_level
  mainfilename, descr = save_or_start_file(EC, type, excitation_level)
  if mainfilename != ""
    filename = mainfilename*"_$excitation_level"
    println("Save $descr to file $filename")
    save!(EC, filename, amps..., description=descr)
  end
end

"""
    try2start_amps(EC::ECInfo, ::Val{excitation_level}; type="T")

  Read amplitudes (type="T") or Lagrange multipliers (type="LM") 
  from file `EC.options.cc.start[_lm]*"_excitation_level"`.
"""
function try2start_amps(EC::ECInfo, ::Val{excitation_level}; type="T") where excitation_level
  mainfilename, descr = save_or_start_file(EC, type, excitation_level, false)
  if mainfilename != ""
    filename = mainfilename*"_$excitation_level"
    if file_exists(EC, filename)
      println("Read $descr from file $filename")
      return load(EC, filename, Val(excitation_level*2), skip_error=true)
    end
  end
  return Array{Float64,excitation_level*2}(undef, ntuple(i->0, Val(excitation_level*2)))
end

"""
    try2save_singles!(EC::ECInfo, singles...; type="T")

  Save singles amplitudes (type="T") or Lagrange multipliers (type="LM") 
  to file `EC.options.cc.save[_lm]*"_1"`.
"""
function try2save_singles!(EC::ECInfo, singles...; type="T")
  try2save_amps!(EC, Val(1), singles...; type)
end

"""
    try2save_doubles!(EC::ECInfo, doubles...; type="T")

  Save doubles amplitudes (type="T") or Lagrange multipliers (type="LM") 
  to file `EC.options.cc.save[_lm]*"_2"`.
"""
function try2save_doubles!(EC::ECInfo, doubles...; type="T")
  try2save_amps!(EC, Val(2), doubles...; type)
end

"""
    try2start_singles(EC::ECInfo; type="T")

  Read singles amplitudes (type="T") or Lagrange multipliers (type="LM")
  from file `EC.options.cc.start[_lm]*"_1"`.
"""
function try2start_singles(EC::ECInfo; type="T")
  return try2start_amps(EC, Val(1); type)
end

"""
    try2start_doubles(EC::ECInfo; type="T")

  Read doubles amplitudes (type="T") or Lagrange multipliers (type="LM")
  from file `EC.options.cc.start[_lm]*"_2"`.
"""
function try2start_doubles(EC::ECInfo; type="T")
  return try2start_amps(EC, Val(2); type)
end

"""
    read_starting_guess4amplitudes(EC::ECInfo, ::Val{level}, spins...)

  Read starting guess for excitation `level`.

  The guess will be read from `T_vo`, `T_VO`, `T_vvoo` etc files.
  If the file does not exist, the guess will be a zeroed-vector.
"""
function read_starting_guess4amplitudes(EC::ECInfo, ::Val{level}, spins...) where level
  if length(spins) == 0
    spins = [:α for i in 1:level]
  end
  if length(spins) != level
    error("number of spins does not match level")
  end
  spaces = ""
  for spin in spins
    spaces *= (spin == :α ? "v" : "V")
  end
  for spin in spins
    spaces *= (spin == :α ? "o" : "O")
  end
  filename = "T_"*spaces
  if file_exists(EC, filename)
    return load(EC, filename, Val(level*2))
  else
    return zeros(len_spaces(EC, spaces)...)::Array{Float64,level*2}
  end
end

"""
    save_current_singles(EC::ECInfo, T1; prefix="T")

  Save current singles amplitudes `T1` to file `prefix*"_vo"`
"""
function save_current_singles(EC::ECInfo, T1; prefix="T")
  save!(EC, prefix*"_vo", T1)
end

"""
    save_current_singles(EC::ECInfo, T1a, T1b; prefix="T")

  Save current singles amplitudes `T1a` and `T1b` to files `prefix*"_vo"` and `prefix*"_VO"`
"""
function save_current_singles(EC::ECInfo, T1a, T1b; prefix="T")
  save!(EC, prefix*"_vo", T1a)
  save!(EC, prefix*"_VO", T1b)
end

"""
    save_current_doubles(EC::ECInfo, T2; prefix="T")

  Save current doubles amplitudes `T2` to file `prefix*"_vvoo"`
"""
function save_current_doubles(EC::ECInfo, T2; prefix="T")
  save!(EC, prefix*"_vvoo", T2)
end

"""
    save_current_doubles(EC::ECInfo, T2a, T2b, T2ab; prefix="T")

  Save current doubles amplitudes `T2a`, `T2b`, and `T2ab` to files 
  `prefix*"_vvoo"`, `prefix*"_VVOO"`, and `prefix*"_vVoO"`
"""
function save_current_doubles(EC::ECInfo, T2a, T2b, T2ab; prefix="T")
  save!(EC, prefix*"_vvoo", T2a)
  save!(EC, prefix*"_VVOO", T2b)
  save!(EC, prefix*"_vVoO", T2ab)
end

"""
    transform_amplitudes2lagrange_multipliers!(Amps1, Amps2)

  Transform amplitudes to first guess for Lagrange multipliers.

  The amplitudes are transformed in-place. 
"""
function transform_amplitudes2lagrange_multipliers!(Amps1, Amps2)
  unrestricted = (length(Amps1) == 2)
  @assert (unrestricted && length(Amps2) == 3) || (!unrestricted && length(Amps2) == 1)
  # add singles to doubles
  add_singles2doubles!(Amps2..., Amps1...)
  return
end

"""
    add_singles2doubles!(T2aa, T2bb, T2ab, T1a, T1b)

  Add singles to doubles amplitudes.
"""
function add_singles2doubles!(T2aa, T2bb, T2ab, T1a, T1b)
  if length(T1a) > 0
    @tensoropt T2aa[a,b,i,j] += T1a[a,i] * T1a[b,j] - T1a[b,i] * T1a[a,j]
  end
  if length(T1b) > 0
  @tensoropt T2bb[a,b,i,j] += T1b[a,i] * T1b[b,j] - T1b[b,i] * T1b[a,j]
  end
  if length(T1a) > 0 && length(T1b) > 0
    @tensoropt T2ab[a,b,i,j] += T1a[a,i] * T1b[b,j]
  end 
end

"""
    add_singles2doubles!(T2, T1; make_contravariant=true)

  Add singles to doubles amplitudes.
  
  If `make_contravariant` is true, the amplitudes will be made contravariant.
"""
function add_singles2doubles!(T2, T1; make_contravariant=true)
  if length(T1) > 0
    @tensoropt T2[a,b,i,j] += T1[a,i] * T1[b,j]
  end
  if make_contravariant
    @tensoropt tT2[a,b,i,j] := T2[a,b,i,j] - T2[a,b,j,i]
    @tensoropt T2[a,b,i,j] += tT2[a,b,i,j]
    T1 .+= T1
  end
end

"""
    contra2covariant(T2)

  Transform contravariant doubles amplitudes to covariant.
"""
function contra2covariant(T2)
  @tensoropt U2[a,b,i,j] := (1/3) * (2*T2[a,b,i,j] + T2[b,a,i,j])
  return U2
end

"""
    triples_4ext!(EC::ECInfo, R3, T3)

  Calculate 4-external contraction with triples amplitudes
  and store the result in `R3`.
"""
function triples_4ext!(EC::ECInfo, R3, T3)
  nocc = size(T3, 6)
  nvirt = size(T3, 3)
  d_vvvv = load4idx(EC,"d_vvvv")
  diagindx = [CartesianIndex(i,i) for i in 1:nvirt]
  d_vvvv[:,:,diagindx] *= 0.5
  trivv = [CartesianIndex(i,j) for j in 1:nvirt for i in 1:j]
  d_vvvv = d_vvvv[:,:,trivv]
  X3 = Array{Float64}(undef, nvirt, nvirt, nvirt, nocc, nocc)
  T3k = Array{Float64}(undef, length(trivv), nvirt, nocc, nocc)
  for k = 1:nocc
    T3k .= T3[trivv,:,:,:,k]
    @tensoropt X3[a,b,c,i,j] = d_vvvv[a,b,x] * T3k[x,c,i,j]
    vR3 = selectdim(R3, 6, k)
    @tensoropt vR3[a,b,c,i,j] += X3[a,b,c,i,j] + X3[b,a,c,j,i]
    vR3 = selectdim(R3, 5, k)
    @tensoropt vR3[a,c,b,i,j] += X3[a,b,c,i,j] + X3[b,a,c,j,i]
    vR3 = selectdim(R3, 4, k)
    @tensoropt vR3[c,a,b,i,j] += X3[a,b,c,i,j] + X3[b,a,c,j,i]
  end
  return R3
end

end # module