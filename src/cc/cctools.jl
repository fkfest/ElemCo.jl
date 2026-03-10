"""
    CCTools

A collection of tools for working with coupled cluster theory.
"""
module CCTools
using LinearAlgebra
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.FockFactory
using ..ElemCo.OrbTools
using ..ElemCo.Wavefunctions
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.QMTensors
using ..ElemCo.TrexioInterface
using ..ElemCo.AbstractEC: AbstractDeterminant

export calc_fock_matrix, calc_HF_energy, calc_rotated_HF_energy
export calc_singles_energy_using_dfock
export update_singles, update_doubles, update_singles!, update_doubles!, update_triples!, update_deco_doubles, update_deco_triples
export calc_singles_norm, calc_doubles_norm, calc_triples_norm, calc_contra_singles_norm, calc_contra_doubles_norm, calc_deco_doubles_norm, calc_deco_triples_norm
export read_starting_guess4amplitudes, save_current_singles, save_current_doubles
export transform_amplitudes2lagrange_multipliers!
export save_or_start_file, try2save_amps!, try2start_amps, try2save_singles!, try2save_doubles!, try2start_singles, try2start_doubles
export contra2covariant, rotation_matrix
export spin_project!, spin_project_amplitudes
export clean_cs_triples!
export calc_cs_singles_dot, calc_u_singles_dot
export calc_cs_doubles_dot, calc_samespin_doubles_dot, calc_ab_doubles_dot
export calc_cs_triples_dot, calc_samespin_triples_dot, calc_mixedspin_triples_dot
export calc_contra_cs_singles_dot, calc_contra_cs_doubles_dot
export triples_4ext!
export triples_4ext_aa!, triples_4ext_bb!
export triples_4ext_aab_ab!, triples_4ext_abb_ab!
export project_amplitudes, check_projection_rank
export dump_wavefunction_with_amplitudes!
export dump_wavefunction_with_determinants!, try_fetch_starting_determinants
export try_fetch_restricted_starting_amplitudes, try_fetch_unrestricted_starting_amplitudes
export print_main_singles

""" 
    calc_fock_matrix(EC::ECInfo, closed_shell, print_out=true)

  Calculate fock matrix from FCIDump
"""
function calc_fock_matrix(EC::ECInfo, closed_shell, print_out=true)
  t1 = time_ns()
  if closed_shell
    fock = gen_fock(EC)
    save!(EC, "f_mm", fock)
    save!(EC, "f_MM", fock)
    eps = diag(fock)
    if print_out
      println("Occupied orbital energies: ", eps[EC.space['o']])
    end
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
  if print_out
    t1 = print_time(EC,t1,"fock matrix",1)
  end
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
    calc_rotated_HF_energy(EC::ECInfo)

  Calculate the Hartree-Fock energy in the rotated orbital basis.
"""
function calc_rotated_HF_energy(EC::ECInfo, closed_shell)
  SP = EC.space
  if closed_shell
    ϵo = load1idx(EC,"e_m")[SP['o']]
    int1_r = load2idx(EC,"int1_r")[SP['o'],SP['o']]
    EHF = sum(ϵo) + sum(diag(int1_r)) + EC.fd.int0
  else
    # TODO
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
  @mtensor T2ab[:,dvb,doa,:][a,b,i,j] = (1/6) * (T2a[:,:,doa,doa][a,b,i,j] + T2b[dvb,dvb,:,:][a,b,i,j] + 2*T2abc[a,b,i,j] + T2abc[b,a,i,j] + 2*T2abc[b,a,j,i] + T2abc[a,b,j,i])
  T2abc = nothing
  # calc ``T^{ij}_{at} = \frac{1}{3} ( ^{ββ}T^{ij}_{at} + 2 ^{αβ}T^{ij}_{at} + ^{αβ}T^{ji}_{at})``
  Tvsdd = T2ab[:,svb,doa,:]
  @mtensor T2ab[:,svb,doa,:][a,t,i,j] = (1/3) * (T2b[dvb,svb,:,:][a,t,i,j] + 2*Tvsdd[a,t,i,j] + Tvsdd[a,t,j,i])
  Tvsdd = nothing
  # calc ``T^{tj}_{ab} = \frac{1}{3} ( ^{αα}T^{tj}_{ab} + 2 ^{αβ}T^{tj}_{ab} + ^{αβ}T^{tj}_{ba})``
  Tvvsd = T2ab[:,dvb,soa,:]
  @mtensor T2ab[:,dvb,soa,:][a,b,t,j] = (1/3) * (T2a[:,:,soa,doa][a,b,t,j] + 2*Tvvsd[a,b,t,j] + Tvvsd[b,a,t,j])
  Tvvsd = nothing

  if length(T1b) > 0
    ms2 = length(soa)
    @mtensor T1add[a,i] := T2ab[:,svb,soa,:][a,t,t,i]
    T1c = (1/(2+ms2))*(T1b[dvb,:] - T1a[:,doa] - T1add)
    for i in 1:length(doa)
      T2ab[:,svb,soa,i] .+= T1c[:,i]
    end
    @mtensor T1add[a,i] = 0.5*T2ab[:,svb,soa,:][a,t,t,i]
    T1 = 0.5 * (T1a[:,doa] + T1b[dvb,:])
    T1a[:,doa] .= T1 - T1add
    T1b[dvb,:] .= T1 + T1add
  end
  @mtensor T2a[:,:,:,doa][a,b,i,j] = T2ab[:,dvb,:,:][a,b,i,j] - T2ab[:,dvb,:,:][b,a,i,j]
  @mtensor T2a[:,:,doa,soa][a,b,i,j] = T2a[:,:,soa,doa][b,a,j,i]
  @mtensor T2b[dvb,:,:,:][a,b,i,j] = T2ab[:,:,doa,:][a,b,i,j] - T2ab[:,:,doa,:][a,b,j,i]
  @mtensor T2b[svb,dvb,:,:][a,b,i,j] = T2b[dvb,svb,:,:][b,a,j,i]
end

"""
    spin_project!(EC::ECInfo, T1a, T1b)
  
  Spin-project singles amplitudes/residuals.
  
  Only possible for high-spin states.
"""
function spin_project!(EC::ECInfo, T1a, T1b)
  SP = EC.space
  @assert length(SP['S']) == 0 " Spin-projection only possible for high-spin states!"
  soa = subspace_in_space(SP['s'], SP['o'])
  svb = subspace_in_space(SP['s'], SP['V'])
  @assert length(soa) == length(svb)
  doa = setdiff(1:length(SP['o']), soa)
  @assert length(doa) == length(SP['O'])
  dvb =setdiff(1:length(SP['V']), svb)
  @assert length(dvb) == length(SP['v'])
  if length(T1b) > 0
    T1 = 0.5 * (T1a[:,doa] + T1b[dvb,:])
    T1a[:,doa] .= T1
    T1b[dvb,:] .= T1
  end
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
    fock = load2idx(EC, "f_mm")
    if fock_only
      ET1SS = ET1OS = ET1 = 0.0
    else
      if !file_exists(EC, "dfc_ov") || !file_exists(EC, "dfe_ov")
        error("Files dfc_ov and dfe_ov are required in calc_singles_energy_using_dfock!")
      end
      dfockc_ov = load2idx(EC, "dfc_ov")
      dfocke_ov = load2idx(EC, "dfe_ov")
      @mtensor begin
        ET1d = T1[a,i] * dfockc_ov[i,a] 
        ET1ex = T1[a,i] * dfocke_ov[i,a]
      end
      ET1SS = ET1d - ET1ex
      ET1OS = ET1d
      ET1 = ET1SS + ET1OS
    end
    fov = fock[SP['o'],SP['v']] 
    @mtensor ET1 += 2.0*(fov[i,a] * T1[a,i])
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
  @mtensor NormT1 = 2.0*conj(T1[a,i])*T1[a,i]
  return real(NormT1)
end

"""
    calc_contra_singles_norm(T1)

  Calculate squared norm of closed-shell contravariant singles amplitudes.
"""
function calc_contra_singles_norm(T1)
  @mtensor NormT1 = 0.5*conj(T1[a,i])*T1[a,i]
  return real(NormT1)
end

"""
    calc_singles_norm(T1a, T1b)

  Calculate squared norm of unrestricted singles amplitudes.
"""
function calc_singles_norm(T1a, T1b)
  @mtensor begin
    NormT1 = conj(T1a[a,i])*T1a[a,i]
    NormT1 += conj(T1b[a,i])*T1b[a,i]
  end
  return real(NormT1)
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
    calc_cs_singles_dot(T1, T1_, state=0)

  Calculate dot product of closed-shell singles amplitudes.
"""
function calc_cs_singles_dot(T1::AbstractMatrix{T}, T1_::AbstractMatrix{T}, state=0) where T
  @mtensor DotT1 = 2.0*conj(T1[a,i])*T1_[a,i]
  return DotT1::T
end
calc_cs_singles_dot(T1, T1_, state=0) = error("calc_cs_singles_dot: T1 and T1_ must be matrices!")

"""
    calc_contra_cs_singles_dot(T1, T1_, state=0)

  Calculate dot product of contravariant closed-shell singles amplitudes.
"""
function calc_contra_cs_singles_dot(T1::AbstractMatrix{T}, T1_::AbstractMatrix{T}, state=0) where T
  @mtensor DotT1 = 0.5*conj(T1[a,i])*T1_[a,i]
  return DotT1::T
end
calc_contra_cs_singles_dot(T1, T1_, state=0) = error("calc_contra_cs_singles_dot: T1 and T1_ must be matrices!")

"""
    calc_u_singles_dot(T1, T1_, state=0)

  Calculate dot of unrestricted singles amplitudes.
"""
function calc_u_singles_dot(T1::AbstractMatrix{T}, T1_::AbstractMatrix{T}, state=0) where T
  @mtensor begin
    DotT1 = conj(T1[a,i])*T1_[a,i]
  end
  return DotT1::T
end
calc_u_singles_dot(T1, T1_, state=0) = error("calc_u_singles_dot: T1 and T1_ must be matrices!")

"""
    calc_doubles_norm(T2)

  Calculate squared norm of closed-shell doubles amplitudes.
"""
function calc_doubles_norm(T2)
  @mtensor NormT2 = (2.0*T2[a,b,i,j] - T2[b,a,i,j])*conj(T2[a,b,i,j])
  return real(NormT2)
end

"""
    calc_contra_doubles_norm(T2)

  Calculate squared norm of closed-shell contravariant doubles amplitudes.
"""
function calc_contra_doubles_norm(T2)
  @mtensor NormT2 = (2.0*T2[a,b,i,j] + T2[b,a,i,j])*conj(T2[a,b,i,j])
  return real(NormT2)/3.0
end

"""
    calc_doubles_norm(T2a, T2b, T2ab)

  Calculate squared norm of unrestricted doubles amplitudes.
"""
function calc_doubles_norm(T2a, T2b, T2ab)
  @mtensor begin
    NormT2 = 0.25*(conj(T2a[a,b,i,j])*T2a[a,b,i,j])
    NormT2 += 0.25*(conj(T2b[a,b,i,j])*T2b[a,b,i,j])
    NormT2 += conj(T2ab[a,b,i,j])*T2ab[a,b,i,j]
  end
  return real(NormT2)
end

"""
    calc_cs_doubles_dot(T2, T2_)

  Calculate dot of closed-shell doubles amplitudes.
"""
function calc_cs_doubles_dot(T2::AbstractArray{T,4}, T2_::AbstractArray{T,4}, state=0) where T
  @mtensor DotT2 = conj(T2[a,b,i,j])*(2.0*T2_[a,b,i,j] - T2_[b,a,i,j])
  return DotT2::T
end
calc_cs_doubles_dot(T2, T2_, state=0) = error("calc_cs_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_contra_cs_doubles_dot(T2, T2_)

  Calculate dot of contravariant closed-shell doubles amplitudes.
"""
function calc_contra_cs_doubles_dot(T2::AbstractArray{T,4}, T2_::AbstractArray{T,4}, state=0) where T
  @mtensor DotT2 = conj(T2[a,b,i,j])*(2.0*T2_[a,b,i,j] + T2_[b,a,i,j])
  return DotT2/3.0::T
end
calc_contra_cs_doubles_dot(T2, T2_, state=0) = error("calc_contra_cs_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_samespin_doubles_dot(T2, T2_)

  Calculate dot of unrestricted same-spin doubles amplitudes.
"""
function calc_samespin_doubles_dot(T2::AbstractArray{T,4}, T2_::AbstractArray{T,4}, state=0) where T
  @mtensor begin
    DotT2 = 0.25*(conj(T2[a,b,i,j])*T2_[a,b,i,j])
  end
  return DotT2::T
end
calc_samespin_doubles_dot(T2, T2_, state=0) = error("calc_samespin_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_ab_doubles_dot(T2, T2_)

  Calculate dot of unrestricted αβ doubles amplitudes.
"""
function calc_ab_doubles_dot(T2::AbstractArray{T,4}, T2_::AbstractArray{T,4}, state=0) where T
  @mtensor begin
    DotT2 = conj(T2[a,b,i,j])*T2_[a,b,i,j]
  end
  return DotT2::T
end
calc_ab_doubles_dot(T2, T2_, state=0) = error("calc_ab_doubles_dot: T2 and T2_ must be 4D arrays!")

"""
    calc_triples_norm(T3aaa, T3bbb, T3abb, T3aab)

  Calculate squared norm of unrestricted triples amplitudes.
"""
function calc_triples_norm(T3aaa, T3bbb, T3abb, T3aab)
  @mtensor begin
    NormT3 = (1/36)*(conj(T3aaa[a,b,c,i,j,k])*T3aaa[a,b,c,i,j,k])
    NormT3 += (1/36)*(conj(T3bbb[a,b,c,i,j,k])*T3bbb[a,b,c,i,j,k])
    NormT3 += 0.25*(conj(T3abb[a,b,c,i,j,k])*T3abb[a,b,c,i,j,k])
    NormT3 += 0.25*(conj(T3aab[a,b,c,i,j,k])*T3aab[a,b,c,i,j,k])
  end
  return real(NormT3)
end

"""
    calc_triples_norm(T3)

  Calculate squared norm of triples amplitudes.
"""
function calc_triples_norm(T3)
  NormT3 = zero(real(eltype(T3)))
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
        @mtensor begin
          NormT3_ = 4*(conj(T3_ijk[a,b,c])*T3_ijk[a,b,c])
          NormT3_ -= 2*(conj(T3_ijk[a,b,c])*T3_ijk[a,c,b])
          NormT3_ -= 2*(conj(T3_ijk[a,b,c])*T3_ijk[c,b,a])
          NormT3_ -= 2*(conj(T3_ijk[a,b,c])*T3_ijk[b,a,c])
          NormT3_ += (conj(T3_ijk[a,b,c])*T3_ijk[c,a,b])
          NormT3_ += (conj(T3_ijk[a,b,c])*T3_ijk[b,c,a])
        end
        NormT3 += fac*real(NormT3_)
      end
    end
  end
  return NormT3
end

"""
    calc_cs_triples_dot(T3, T3_)

  Calculate dot of closed-shell triples amplitudes.
"""
function calc_cs_triples_dot(T3::AbstractArray{T,6}, T3_::AbstractArray{T,6}) where T
  DotT3 = zero(T)
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
        @mtensor begin
          DotT3_ = 4*(conj(T3_ijk[a,b,c])*T3_ijk_[a,b,c])
          DotT3_ -= 2*(conj(T3_ijk[a,b,c])*T3_ijk_[a,c,b])
          DotT3_ -= 2*(conj(T3_ijk[a,b,c])*T3_ijk_[c,b,a])
          DotT3_ -= 2*(conj(T3_ijk[a,b,c])*T3_ijk_[b,a,c])
          DotT3_ += (conj(T3_ijk[a,b,c])*T3_ijk_[c,a,b])
          DotT3_ += (conj(T3_ijk[a,b,c])*T3_ijk_[b,c,a])
        end
        DotT3 += fac*DotT3_
      end
    end
  end
  return DotT3::T
end
calc_cs_triples_dot(T3, T3_) = error("calc_cs_triples_dot not implemented for this type of T3")

"""
    calc_samespin_triples_dot(T3, T3_)

  Calculate dot of unrestricted same-spin triples amplitudes.
"""
function calc_samespin_triples_dot(T3::AbstractArray{T,6}, T3_::AbstractArray{T,6}) where T
  @mtensor begin
    DotT3 = (1/36)*(conj(T3[a,b,c,i,j,k])*T3_[a,b,c,i,j,k])
  end
  return DotT3::T
end
calc_samespin_triples_dot(T3, T3_) = error("calc_samespin_triples_dot not implemented for this type of T3")

"""
    calc_mixedspin_triples_dot(T3, T3_)

  Calculate dot of unrestricted mixed-spin triples amplitudes.
"""
function calc_mixedspin_triples_dot(T3::AbstractArray{T,6}, T3_::AbstractArray{T,6}) where T
  @mtensor begin
    DotT3 = 0.25*(conj(T3[a,b,c,i,j,k])*T3_[a,b,c,i,j,k])
  end
  return DotT3::T
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
      @mtensor normT2 = conj(T2[X,Y]) * tT2[X,Y]
    else
      @mtensor normT2 = conj(T2[X,Y]) * T2[X,Y]
    end
  end
  return real(normT2)
end

"""
    calc_deco_triples_norm(T3)

  Calculate a *simple* norm of triples (without contravariant!)
"""
function calc_deco_triples_norm(T3)
  @mtensor NormT3 = conj(T3[X,Y,Z]) * T3[X,Y,Z]
  return real(NormT3)
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
  elseif type == "X"
    descr *= " eigenvectors"
    mainfilename = save ? EC.options.eom.save : EC.options.eom.start
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
function try2start_amps(EC::ECInfo{T}, ::Val{excitation_level}; type="T") where {T, excitation_level}
  mainfilename, descr = save_or_start_file(EC, type, excitation_level, false)
  if mainfilename != ""
    filename = mainfilename*"_$excitation_level"
    if file_exists(EC, filename)
      println("Read $descr from file $filename")
      return load(EC, filename, Val(excitation_level*2), skip_error=true)
    end
  end
  return Array{T,excitation_level*2}(undef, ntuple(i->0, Val(excitation_level*2)))
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
function read_starting_guess4amplitudes(EC::ECInfo{T}, ::Val{level}, spins...) where {T,level}
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
    return zeros(T, len_spaces(EC, spaces)...)::Array{T,level*2}
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
    @mtensor T2aa[a,b,i,j] += T1a[a,i] * T1a[b,j] - T1a[b,i] * T1a[a,j]
  end
  if length(T1b) > 0
  @mtensor T2bb[a,b,i,j] += T1b[a,i] * T1b[b,j] - T1b[b,i] * T1b[a,j]
  end
  if length(T1a) > 0 && length(T1b) > 0
    @mtensor T2ab[a,b,i,j] += T1a[a,i] * T1b[b,j]
  end 
end

"""
    add_singles2doubles!(T2, T1; make_contravariant=true)

  Add singles to doubles amplitudes.
  
  If `make_contravariant` is true, the amplitudes will be made contravariant.
"""
function add_singles2doubles!(T2, T1; make_contravariant=true)
  if length(T1) > 0
    @mtensor T2[a,b,i,j] += T1[a,i] * T1[b,j]
  end
  if make_contravariant
    @mtensor tT2[a,b,i,j] := T2[a,b,i,j] - T2[a,b,j,i]
    @mtensor T2[a,b,i,j] += tT2[a,b,i,j]
    T1 .+= T1
  end
end

"""
    contra2covariant(T2)

  Transform contravariant doubles amplitudes to covariant.
"""
function contra2covariant(T2)
  @mtensor U2[a,b,i,j] := (1/3) * (2*T2[a,b,i,j] + T2[b,a,i,j])
  return U2
end

"""
    rotation_matrix(EC::ECInfo, T1; full=false, beta=false)
  
  Make the integrals rotation matrix with T1.

  If `full` is true, the rotation matrix will be made with core and deleted orbitals included.
"""
function rotation_matrix(EC::ECInfo{T}, T1; full=false, beta=false) where T
  if full
    space_save, _ = restore_full_space!(EC)
  end
  norb = n_orbs(EC)
  SP = EC.space
  occ = beta ? SP['O'] : SP['o']
  virt = beta ? SP['V'] : SP['v']
  Rpq = zeros(T, norb, norb)
  @assert size(T1) == (length(virt), length(occ)) "size of T1 does not match space dimensions"
  Rpq[virt,occ] = T1
  Rpq[occ,virt] = -T1'
  Rpq = exp(Rpq)
  if full
    restore_space!(EC, space_save)
  end
  return Rpq
end

"""
    triples_4ext!(EC::ECInfo, R3, T3)

  Calculate 4-external contraction with triples amplitudes
  and store the result in `R3`.
"""
function triples_4ext!(EC::ECInfo{T}, R3, T3) where T
  nocc = size(T3, 6)
  nvirt = size(T3, 3)
  d_vvvv = load4idx(EC,"d_vvvv")
  diagindx = [CartesianIndex(i,i) for i in 1:nvirt]
  d_vvvv[:,:,diagindx] *= 0.5
  trivv = uppertriangular_cut(nvirt)
  d_vvvv = d_vvvv[:,:,trivv]
  X3 = Array{T}(undef, nvirt, nvirt, nvirt, nocc, nocc)
  T3k = Array{T}(undef, length(trivv), nvirt, nocc, nocc)
  for k = 1:nocc
    T3k .= @view T3[trivv,:,:,:,k]
    @mtensor X3[a,b,c,i,j] = d_vvvv[a,b,x] * T3k[x,c,i,j]
    vR3 = selectdim(R3, 6, k)
    @mtensor vR3[a,b,c,i,j] += X3[a,b,c,i,j] + X3[b,a,c,j,i]
    vR3 = selectdim(R3, 5, k)
    @mtensor vR3[a,c,b,i,j] += X3[a,b,c,i,j] + X3[b,a,c,j,i]
    vR3 = selectdim(R3, 4, k)
    @mtensor vR3[c,a,b,i,j] += X3[a,b,c,i,j] + X3[b,a,c,j,i]
  end
  return R3
end

"""
    antisymmetrize_34!(d, trivv)

  Antisymmetrize dims 3 and 4 of `d` in-place and compress to triangular.
  Returns `d[:,:,trivv]` where `d[p,q,a,b] → d[p,q,a,b] - d[p,q,b,a]` for `a < b`.
"""
function antisymmetrize_34!(d, trivv)
  nv = size(d, 3)
  for b in 1:nv, a in 1:b-1
    @views d[:,:,a,b] .-= d[:,:,b,a]
  end
  return d[:,:,trivv]
end

"""
    triples_4ext_aa!(EC::ECInfo, R3a, R3aab, T3a, T3aab)

  Calculate αα 4-external contractions with triples amplitudes
  for the unrestricted case.
  Uses antisymmetrized integrals ``\\langle cd||ab\\rangle`` compressed to ``a<b``.

  ``R^{ijk}_{cde} += \\sum_{a<b} \\langle cd||ab\\rangle T^{ijk}_{eab}``
  ``R^{ijK}_{cdA} += \\sum_{a<b} \\langle cd||ab\\rangle T^{ijK}_{abA}``
"""
function triples_4ext_aa!(EC::ECInfo{T}, R3a, R3aab, T3a, T3aab) where T
  nva = size(T3a, 1)
  noa = size(T3a, 4)
  nvb = size(T3aab, 3)
  nob = size(T3aab, 6)
  d_vvvv = load4idx(EC, "d_vvvv")
  # d_vvvv[c,d,a,b]: antisymmetrize in (a,b) = dims (3,4) and compress to a<b
  trivv = strict_uppertriangular_cut(nva)
  ntri = length(trivv)
  if ntri == 0
    return
  end
  d_anti = antisymmetrize_34!(d_vvvv, trivv)
  d_vvvv = nothing
  # d_anti[c,d,x] is antisymmetric in (c,d) due to exchange symmetry.
  # Original: X[c,d,e,...] = Σ_{a,b} d[c,d,a,b]*T3[e,a,b,...] = Σ_x d_anti[c,d,x]*T3_comp[e,x,...]
  # R3 accumulation simplifies from 0.5*(X - X[d↔c]) to X (since X[d,c,...] = -X[c,d,...])

  # --- T3a: loop over k ---
  X3 = Array{T}(undef, nva, nva, nva, noa, noa)
  T3k = Array{T}(undef, ntri, nva, noa, noa)
  for k in 1:noa
    T3k .= @view T3a[trivv,:,:,:,k]
    @mtensor X3[c,d,e,i,j] = d_anti[c,d,x] * T3k[x,e,i,j]
    vR3 = selectdim(R3a, 6, k)
    @mtensor begin
      vR3[c,d,e,i,j] += X3[c,d,e,i,j]
      vR3[c,e,d,i,j] -= X3[c,d,e,i,j]
      vR3[e,c,d,i,j] += X3[c,d,e,i,j]
    end
  end

  # --- T3aab: loop over I ---
  X3aab = Array{T}(undef, nva, nva, nvb, noa, noa)
  T3I = Array{T}(undef, ntri, nvb, noa, noa)
  for I in 1:nob
    T3I .= @view T3aab[trivv,:,:,:,I]
    @mtensor X3aab[c,d,A,i,j] = d_anti[c,d,x] * T3I[x,A,i,j]
    vR3 = selectdim(R3aab, 6, I)
    @mtensor vR3[c,d,A,i,j] += X3aab[c,d,A,i,j]
  end
  return
end

"""
    triples_4ext_bb!(EC::ECInfo, R3b, R3abb, T3b, T3abb)

  Calculate ββ 4-external contractions with triples amplitudes
  for the unrestricted case.
  Uses antisymmetrized integrals ``\\langle CD||AB\\rangle`` compressed to ``A<B``.

  ``R^{IJK}_{CDE} += \\sum_{A<B} \\langle CD||AB\\rangle T^{IJK}_{EAB}``
  ``R^{iIJ}_{aCB} += \\sum_{A<B} \\langle CD||AB\\rangle T^{iIJ}_{aAB}``
"""
function triples_4ext_bb!(EC::ECInfo{T}, R3b, R3abb, T3b, T3abb) where T
  nvb = size(T3b, 1)
  nob = size(T3b, 4)
  nva = size(T3abb, 1)
  noa = size(T3abb, 4)
  d_VVVV = load4idx(EC, "d_VVVV")
  # d_VVVV[D,C,B,A]: permute to [C,D,A,B] so contracted pair (A,B) is at dims (3,4)
  # and free pair (C,D) is at dims (1,2), matching the αα convention.
  # Exchange symmetry: d[D,C,B,A] = d[C,D,A,B], so this is just a symmetry.
  d_VVVV = permutedims(d_VVVV, (2,1,4,3))
  # Now d_VVVV[C,D,A,B]: antisymmetrize in (A,B) = dims (3,4) and compress to A<B
  trivv = strict_uppertriangular_cut(nvb)
  ntri = length(trivv)
  if ntri == 0
    return
  end
  d_anti = antisymmetrize_34!(d_VVVV, trivv)
  d_VVVV = nothing
  # d_anti[C,D,x] with x = CI(A,B) for A<B. Antisymmetric in (C,D).
  # Now the structure is identical to the αα case.

  # --- T3b: loop over K ---
  X3 = Array{T}(undef, nvb, nvb, nvb, nob, nob)
  T3k = Array{T}(undef, ntri, nvb, nob, nob)
  for K in 1:nob
    T3k .= @view T3b[trivv,:,:,:,K]
    @mtensor X3[C,D,E,I,J] = d_anti[C,D,x] * T3k[x,E,I,J]
    vR3 = selectdim(R3b, 6, K)
    @mtensor begin
      vR3[C,D,E,I,J] += X3[C,D,E,I,J]
      vR3[C,E,D,I,J] -= X3[C,D,E,I,J]
      vR3[E,C,D,I,J] += X3[C,D,E,I,J]
    end
  end

  # --- T3abb: loop over i ---
  X3abb = Array{T}(undef, nvb, nvb, nva, nob, nob)
  T3i = Array{T}(undef, ntri, nva, nob, nob)
  for i in 1:noa
    for (idx, ci) in enumerate(trivv)
      T3i[idx, :, :, :] .= @view T3abb[:, ci[1], ci[2], i, :, :]
    end
    @mtensor X3abb[C,D,a,I,J] = d_anti[C,D,x] * T3i[x,a,I,J]
    vR3 = selectdim(R3abb, 4, i)
    @mtensor vR3[a,C,D,I,J] += X3abb[C,D,a,I,J]
  end
end

"""
    triples_4ext_aab_ab!(EC::ECInfo, R3aab, T3aab)

  Calculate αβ 4-external contraction of ``d_{vVvV}`` with ``T^{ijK}_{abC}``
  for the unrestricted case.
  No integral compression (different-spin contracted indices), but loops over
  one occupied index to reduce intermediate size.

  ``R^{ijK}_{bcB} -= X_{bcB}^{ijK} - X_{cbB}^{ijK}``
  where ``X_{bcB}^{ijK} = \\sum_{a,A} \\langle bB|aA\\rangle T^{ijK}_{caA}``
"""
function triples_4ext_aab_ab!(EC::ECInfo, R3aab, T3aab)
  nva = size(T3aab, 1)
  nvb = size(T3aab, 3)
  noa = size(T3aab, 4)
  nob = size(T3aab, 6)
  d_vVvV = load4idx(EC, "d_vVvV")
  # d_vVvV[b,B,a,A]: contracted (a,A) at dims (3,4), free (b,B) at dims (1,2)
  # Loop over I (β occ) to reduce intermediate
  X3 = Array{Float64}(undef, nva, nva, nvb, noa, noa)
  for I in 1:nob
    vT3 = @view T3aab[:,:,:,:,:,I]
    @mtensor X3[b,c,B,i,j] = d_vVvV[b,B,a,A] * vT3[c,a,A,i,j]
    vR3 = selectdim(R3aab, 6, I)
    # Original: R3 -= 0.5*X[b,c,...] + 0.5*X[c,b,...] + 0.5*X[b,c,...,j,i] - 0.5*X[c,b,...,j,i]
    # X[b,c,B,j,i] = -X[b,c,B,i,j] by T3aab antisymmetry in (i,j)
    # Combined: R3[b,c,B,...] -= X, R3[c,b,B,...] += X
    @mtensor begin
      vR3[b,c,B,i,j] -= X3[b,c,B,i,j]
      vR3[c,b,B,i,j] += X3[b,c,B,i,j]
    end
  end
  d_vVvV = nothing
end

"""
    triples_4ext_abb_ab!(EC::ECInfo, R3abb, T3abb)

  Calculate αβ 4-external contraction of ``d_{vVvV}`` with ``T^{iIJ}_{aCB}``
  for the unrestricted case.
  No integral compression (different-spin contracted indices), but loops over
  one occupied index to reduce intermediate size.

  ``R^{iIJ}_{bBC} -= X_{bBC}^{iIJ} - X_{bCB}^{iIJ}``
  where ``X_{bBC}^{iIJ} = \\sum_{a,A} \\langle bB|aA\\rangle T^{iIJ}_{aCA}``
"""
function triples_4ext_abb_ab!(EC::ECInfo{T}, R3abb, T3abb) where T
  nva = size(T3abb, 1)
  nvb = size(T3abb, 2)
  noa = size(T3abb, 4)
  nob = size(T3abb, 5)
  d_vVvV = load4idx(EC, "d_vVvV")
  # Loop over i (α occ) to reduce intermediate
  X3 = Array{T}(undef, nva, nvb, nvb, nob, nob)
  for i in 1:noa
    vT3 = @view T3abb[:,:,:,i,:,:]
    @mtensor X3[b,B,C,I,J] = d_vVvV[b,B,a,A] * vT3[a,C,A,I,J]
    vR3 = selectdim(R3abb, 4, i)
    # Original: R3 -= 0.5*X[b,B,C,...] + 0.5*X[b,C,B,...] + 0.5*X[b,B,C,...,J,I] - 0.5*X[b,C,B,...,J,I]
    # X[b,B,C,I,J] → X[b,B,C,J,I] = -X[b,B,C,I,J] by T3abb antisymmetry in (I,J)
    # Combined: R3[b,B,C,...] -= X, R3[b,C,B,...] += X
    @mtensor begin
      vR3[b,B,C,I,J] -= X3[b,B,C,I,J]
      vR3[b,C,B,I,J] += X3[b,B,C,I,J]
    end
  end
  d_vVvV = nothing
end

"""
    check_projection_rank(P, expected_rank::Int, space_name::String)

  Check if the projection matrix `P` has full rank for the expected dimension.

  Returns `(is_full_rank::Bool, actual_rank::Int)`.
  Prints a warning if the rank is less than expected.
"""
function check_projection_rank(P::AbstractMatrix, expected_rank::Int, space_name::String)
  actual_rank = rank(P)
  is_full_rank = (actual_rank >= expected_rank)
  if !is_full_rank
    println("WARNING: $space_name projection is not full rank!")
    println("         Expected rank: $expected_rank, actual rank: $actual_rank")
    println("         Amplitude restart may not be accurate.")
  end
  return is_full_rank, actual_rank
end

"""
    project_amplitudes(EC::ECInfo, T1_old, T2_old, cMO_old::SpinMatrix, cMO_new::SpinMatrix, 
                       basis_old::BasisSet, basis_new::BasisSet; 
                       classes_old::Tuple{Vector{String},Vector{String}}=(String[], String[]),
                       classes_new::Tuple{Vector{String},Vector{String}}=(String[], String[]))

  Project CC amplitudes from an old orbital basis to a new one.

  The projection is performed as:
  - Singles: ``T_{a}^{i,\\text{new}} = P_v^{a'} T_{a'}^{i',\\text{old}} P_o^{i'}``
  - Doubles: ``T_{ab}^{ij,\\text{new}} = P_v^{a'} P_v^{b'} T_{a'b'}^{i'j',\\text{old}} P_o^{i'} P_o^{j'}``

  where ``P`` are projection matrices between old and new orbital spaces.

  If `classes_old` is provided, it is used to determine which orbitals in the old basis
  were occupied ("Inactive"/"Active") vs virtual ("Virtual"). This is essential when
  core orbitals are frozen, as the MO coefficient matrix includes all orbitals but
  amplitudes only involve active orbitals.

  If `classes_new` is provided, it is used similarly for the new basis. Otherwise, 
  `EC.space['o']` and `EC.space['v']` are used (which may have frozen core indices 
  renumbered from FCIDUMP).

  Returns `(T1_new, T2_new)` for closed-shell case.
"""
function project_amplitudes(EC::ECInfo, T1_old::AbstractMatrix, T2_old::AbstractArray{<:Real,4}, 
                            cMO_old::SpinMatrix, cMO_new::SpinMatrix,
                            basis_old::BasisSet, basis_new::BasisSet;
                            classes_old::Tuple{Vector{String},Vector{String}}=(String[], String[]),
                            classes_new::Tuple{Vector{String},Vector{String}}=(String[], String[]))
  SP = EC.space
  nocc_new = length(SP['o'])
  nvirt_new = length(SP['v'])
  
  # Get old orbital dimensions from amplitudes
  if length(T1_old) > 0
    nvirt_old, nocc_old = size(T1_old)
  elseif length(T2_old) > 0
    nvirt_old = size(T2_old, 1)
    nocc_old = size(T2_old, 3)
  else
    # No amplitudes to project
    return zeros(nvirt_new, nocc_new), zeros(nvirt_new, nvirt_new, nocc_new, nocc_new)
  end
  
  # Determine old orbital indices from classes (Core is skipped, Inactive/Active → occ, Virtual → virt)
  if !isempty(classes_old[1])
    occ_old_indices, virt_old_indices = occupied_virtual_from_classes(classes_old[1])
  else
    # No classes available - assume consecutive indices (no frozen core)
    occ_old_indices = collect(1:nocc_old)
    virt_old_indices = collect((nocc_old+1):(nocc_old+nvirt_old))
  end
  
  # Calculate overlap between bases
  if isempty(basis_old) || isempty(basis_new)
    # Assume same basis, just orbital rotation
    SAO = I
    S_old_new = I
  else
    SAO = overlap(basis_new)
    S_old_new = overlap(basis_new, basis_old)
  end
  
  # Build projection matrix in AO basis: P = S_new^{-1} * S_new_old
  if SAO isa UniformScaling
    proj_ao = I
  else
    proj_ao = inv(SAO) * S_old_new
  end
  
  # Project to MO basis
  # P_MO = C_new^T * S_new * proj_ao * C_old = C_new^T * S_old_new * C_old
  if proj_ao isa UniformScaling
    P_full = cMO_new[1]' * cMO_old[1]
  else
    P_full = cMO_new[1]' * S_old_new * cMO_old[1]
  end
  
  # Extract occupied and virtual blocks using orbital class information
  # P_full is (norb_new × norb_old)
  # We need occ_new ← occ_old and virt_new ← virt_old projections
  
  # Determine new orbital indices from classes (Core is skipped, Inactive/Active → occ, Virtual → virt)
  # If no classes provided, fall back to EC.space (which may have renumbered indices from FCIDUMP)
  if !isempty(classes_new[1])
    occ_new_indices, virt_new_indices = occupied_virtual_from_classes(classes_new[1])
  else
    occ_new_indices = collect(SP['o'])
    virt_new_indices = collect(SP['v'])
  end
  
  # Projection matrices for occupied and virtual spaces
  P_o = P_full[occ_new_indices, occ_old_indices]   # (nocc_new × nocc_old)
  P_v = P_full[virt_new_indices, virt_old_indices]  # (nvirt_new × nvirt_old)
  
  # Check projection rank
  check_projection_rank(P_o, min(nocc_new, nocc_old), "Occupied space")
  check_projection_rank(P_v, min(nvirt_new, nvirt_old), "Virtual space")
  
  # Project singles: T1_new[a,i] = P_v[a,a'] * T1_old[a',i'] * P_o[i,i']^T
  if length(T1_old) > 0
    T1_new = P_v * T1_old * P_o'
  else
    T1_new = zeros(nvirt_new, nocc_new)
  end
  
  # Project doubles: T2_new[a,b,i,j] = P_v[a,a'] * P_v[b,b'] * T2_old[a',b',i',j'] * P_o[i,i']^T * P_o[j,j']^T
  if length(T2_old) > 0
    # First, contract virtual indices
    @mtensor T2_tmp1[a,b_old,i_old,j_old] := P_v[a,a_old] * T2_old[a_old,b_old,i_old,j_old]
    @mtensor T2_tmp2[a,b,i_old,j_old] := P_v[b,b_old] * T2_tmp1[a,b_old,i_old,j_old]
    # Then, contract occupied indices  
    @mtensor T2_tmp3[a,b,i,j_old] := T2_tmp2[a,b,i_old,j_old] * P_o[i,i_old]
    @mtensor T2_new[a,b,i,j] := T2_tmp3[a,b,i,j_old] * P_o[j,j_old]
  else
    T2_new = zeros(nvirt_new, nvirt_new, nocc_new, nocc_new)
  end
  
  return T1_new, T2_new
end

"""
    project_amplitudes(EC::ECInfo, T1a_old, T1b_old, T2a_old, T2b_old, T2ab_old,
                       cMO_old::SpinMatrix, cMO_new::SpinMatrix,
                       basis_old::BasisSet, basis_new::BasisSet;
                       classes_old=(String[], String[]), occupations_old=(Float64[], Float64[]))

  Project unrestricted CC amplitudes from an old orbital basis to a new one.

  # Arguments
  - `classes_old`: Tuple of (alpha_classes, beta_classes) orbital class vectors from old calculation.
    Used to identify Core/Deleted orbitals to exclude.
  - `occupations_old`: Tuple of (alpha_occupations, beta_occupations) from old calculation.
    If provided, used to determine occupied/virtual orbitals (more reliable for UHF).
    If empty, falls back to classes_old or assumes contiguous occupied orbitals.

  Returns `(T1a_new, T1b_new, T2a_new, T2b_new, T2ab_new)`.
"""
function project_amplitudes(EC::ECInfo, 
                            T1a_old::AbstractMatrix, T1b_old::AbstractMatrix,
                            T2a_old::AbstractArray{<:Real,4}, T2b_old::AbstractArray{<:Real,4}, 
                            T2ab_old::AbstractArray{<:Real,4},
                            cMO_old::SpinMatrix, cMO_new::SpinMatrix,
                            basis_old::BasisSet, basis_new::BasisSet;
                            classes_old::Tuple{Vector{String},Vector{String}}=(String[], String[]),
                            classes_new::Tuple{Vector{String},Vector{String}}=(String[], String[]),
                            occupations_old::Tuple{Vector{Float64},Vector{Float64}}=(Float64[], Float64[]),
                            occupations_new::Tuple{Vector{Float64},Vector{Float64}}=(Float64[], Float64[]))
  SP = EC.space
  nocc_a_new = length(SP['o'])
  nocc_b_new = length(SP['O'])
  nvirt_a_new = length(SP['v'])
  nvirt_b_new = length(SP['V'])
  
  # Get old orbital dimensions from amplitudes
  if length(T1a_old) > 0
    nvirt_a_old, nocc_a_old = size(T1a_old)
  elseif length(T2a_old) > 0
    nvirt_a_old = size(T2a_old, 1)
    nocc_a_old = size(T2a_old, 3)
  elseif length(T2ab_old) > 0
    nvirt_a_old = size(T2ab_old, 1)
    nocc_a_old = size(T2ab_old, 3)
  else
    nvirt_a_old, nocc_a_old = 0, 0
  end
  
  if length(T1b_old) > 0
    nvirt_b_old, nocc_b_old = size(T1b_old)
  elseif length(T2b_old) > 0
    nvirt_b_old = size(T2b_old, 1)
    nocc_b_old = size(T2b_old, 3)
  elseif length(T2ab_old) > 0
    nvirt_b_old = size(T2ab_old, 2)
    nocc_b_old = size(T2ab_old, 4)
  else
    nvirt_b_old, nocc_b_old = 0, 0
  end
  
  # No amplitudes to project
  if nvirt_a_old == 0 && nvirt_b_old == 0
    return (zeros(nvirt_a_new, nocc_a_new), zeros(nvirt_b_new, nocc_b_new),
            zeros(nvirt_a_new, nvirt_a_new, nocc_a_new, nocc_a_new),
            zeros(nvirt_b_new, nvirt_b_new, nocc_b_new, nocc_b_new),
            zeros(nvirt_a_new, nvirt_b_new, nocc_a_new, nocc_b_new))
  end
  
  # Calculate overlap between bases
  if isempty(basis_old) || isempty(basis_new)
    S_old_new = I
  else
    S_old_new = overlap(basis_new, basis_old)
  end
  
  # Alpha projection matrices
  if is_restricted(cMO_old) 
    cMO_old_a = cMO_old[1]
    cMO_old_b = cMO_old[1]
  else
    cMO_old_a = cMO_old[1]
    cMO_old_b = cMO_old[2]
  end
  if is_restricted(cMO_new)
    cMO_new_a = cMO_new[1]
    cMO_new_b = cMO_new[1]
  else
    cMO_new_a = cMO_new[1]
    cMO_new_b = cMO_new[2]
  end
  
  if S_old_new isa UniformScaling
    P_full_a = cMO_new_a' * cMO_old_a
    P_full_b = cMO_new_b' * cMO_old_b
  else
    P_full_a = cMO_new_a' * S_old_new * cMO_old_a
    P_full_b = cMO_new_b' * S_old_new * cMO_old_b
  end
  
  # Determine old orbital indices
  # Priority: occupations > classes > contiguous from 1
  # For UHF, occupations are more reliable since alpha/beta can have different occ/virt
  classes_a_old, classes_b_old = classes_old
  occ_a_old, occ_b_old = occupations_old
  
  if !isempty(occ_a_old)
    # Use occupations - more reliable for UHF
    occ_a_old_indices, virt_a_old_indices = occupied_virtual_from_occupations(occ_a_old, classes_a_old)
  elseif !isempty(classes_a_old)
    occ_a_old_indices, virt_a_old_indices = occupied_virtual_from_classes(classes_a_old)
  else
    occ_a_old_indices = collect(1:nocc_a_old)
    virt_a_old_indices = collect((nocc_a_old+1):(nocc_a_old+nvirt_a_old))
  end
  
  if !isempty(occ_b_old)
    # Use occupations - more reliable for UHF
    occ_b_old_indices, virt_b_old_indices = occupied_virtual_from_occupations(occ_b_old, classes_b_old)
  elseif !isempty(classes_b_old)
    occ_b_old_indices, virt_b_old_indices = occupied_virtual_from_classes(classes_b_old)
  else
    occ_b_old_indices = collect(1:nocc_b_old)
    virt_b_old_indices = collect((nocc_b_old+1):(nocc_b_old+nvirt_b_old))
  end
  
  # Determine new orbital indices
  # Priority: occupations > classes > EC.space
  occ_a_new, occ_b_new = occupations_new
  if !isempty(occ_a_new)
    occ_a_new_indices, virt_a_new_indices = occupied_virtual_from_occupations(occ_a_new, classes_new[1])
  elseif !isempty(classes_new[1])
    occ_a_new_indices, virt_a_new_indices = occupied_virtual_from_classes(classes_new[1])
  else
    occ_a_new_indices = collect(SP['o'])
    virt_a_new_indices = collect(SP['v'])
  end
  if !isempty(occ_b_new)
    occ_b_new_indices, virt_b_new_indices = occupied_virtual_from_occupations(occ_b_new, classes_new[2])
  elseif !isempty(classes_new[2])
    occ_b_new_indices, virt_b_new_indices = occupied_virtual_from_classes(classes_new[2])
  else
    occ_b_new_indices = collect(SP['O'])
    virt_b_new_indices = collect(SP['V'])
  end
  
  P_oa = P_full_a[occ_a_new_indices, occ_a_old_indices]
  P_va = P_full_a[virt_a_new_indices, virt_a_old_indices]
  P_ob = P_full_b[occ_b_new_indices, occ_b_old_indices]
  P_vb = P_full_b[virt_b_new_indices, virt_b_old_indices]
  
  # Check projection ranks
  check_projection_rank(P_oa, min(nocc_a_new, nocc_a_old), "Alpha occupied space")
  check_projection_rank(P_va, min(nvirt_a_new, nvirt_a_old), "Alpha virtual space")
  check_projection_rank(P_ob, min(nocc_b_new, nocc_b_old), "Beta occupied space")
  check_projection_rank(P_vb, min(nvirt_b_new, nvirt_b_old), "Beta virtual space")
  
  # Project alpha singles
  if length(T1a_old) > 0
    T1a_new = P_va * T1a_old * P_oa'
  else
    T1a_new = zeros(nvirt_a_new, nocc_a_new)
  end
  
  # Project beta singles
  if length(T1b_old) > 0
    T1b_new = P_vb * T1b_old * P_ob'
  else
    T1b_new = zeros(nvirt_b_new, nocc_b_new)
  end
  
  # Project alpha-alpha doubles
  if length(T2a_old) > 0
    @mtensor T2_tmp1[a,b_old,i_old,j_old] := P_va[a,a_old] * T2a_old[a_old,b_old,i_old,j_old]
    @mtensor T2_tmp2[a,b,i_old,j_old] := P_va[b,b_old] * T2_tmp1[a,b_old,i_old,j_old]
    @mtensor T2_tmp3[a,b,i,j_old] := T2_tmp2[a,b,i_old,j_old] * P_oa[i,i_old]
    @mtensor T2a_new[a,b,i,j] := T2_tmp3[a,b,i,j_old] * P_oa[j,j_old]
  else
    T2a_new = zeros(nvirt_a_new, nvirt_a_new, nocc_a_new, nocc_a_new)
  end
  
  # Project beta-beta doubles
  if length(T2b_old) > 0
    @mtensor T2_tmp1[a,b_old,i_old,j_old] := P_vb[a,a_old] * T2b_old[a_old,b_old,i_old,j_old]
    @mtensor T2_tmp2[a,b,i_old,j_old] := P_vb[b,b_old] * T2_tmp1[a,b_old,i_old,j_old]
    @mtensor T2_tmp3[a,b,i,j_old] := T2_tmp2[a,b,i_old,j_old] * P_ob[i,i_old]
    @mtensor T2b_new[a,b,i,j] := T2_tmp3[a,b,i,j_old] * P_ob[j,j_old]
  else
    T2b_new = zeros(nvirt_b_new, nvirt_b_new, nocc_b_new, nocc_b_new)
  end
  
  # Project alpha-beta doubles
  if length(T2ab_old) > 0
    @mtensor T2_tmp1[a,b_old,i_old,j_old] := P_va[a,a_old] * T2ab_old[a_old,b_old,i_old,j_old]
    @mtensor T2_tmp2[a,b,i_old,j_old] := P_vb[b,b_old] * T2_tmp1[a,b_old,i_old,j_old]
    @mtensor T2_tmp3[a,b,i,j_old] := T2_tmp2[a,b,i_old,j_old] * P_oa[i,i_old]
    @mtensor T2ab_new[a,b,i,j] := T2_tmp3[a,b,i,j_old] * P_ob[j,j_old]
  else
    T2ab_new = zeros(nvirt_a_new, nvirt_b_new, nocc_a_new, nocc_b_new)
  end
  
  return T1a_new, T1b_new, T2a_new, T2b_new, T2ab_new
end

"""
    dump_wavefunction_with_amplitudes!(EC::ECInfo, T1, T2; orbopt=false)

  Dump orbitals and CC amplitudes to the TREXIO file specified in `wf.store`.

  Does nothing if `EC.options.wf.store` is empty.
  For closed-shell case with singles `T1` and doubles `T2`.

  If `orbopt=true`, the orbitals are rotated using the T1 amplitudes
  via [`rotation_matrix`](@ref) and stored instead of the original orbitals.
  T1 amplitudes are not stored in that case.
"""
function dump_wavefunction_with_amplitudes!(EC::ECInfo, T1::AbstractMatrix, T2::AbstractArray{<:Number,4};
                                            orbopt::Bool=false)
  if EC.options.wf.store == ""
    return
  end
  println("Storing wavefunction with amplitudes to $(EC.options.wf.store) ...")
  # Pre-fetch orbital data BEFORE opening store file for writing
  # This is crucial when dump and store are the same file
  orbital_data = fetch_orbital_data(EC)
  if orbopt
    # Rotation matrix from T1 amplitudes
    Rpq = rotation_matrix(EC, T1; full=true)
    if !isnothing(orbital_data)
      # Rotate orbitals using T1 and store rotated orbitals instead
      rotate_orbitaldata!(orbital_data, Rpq)
    else
      # store rotation matrix as orbital data if no orbital data is available (e.g. from FCIDUMP)
      orbital_data = OrbitalData(SpinMatrix(Rpq))
    end
    T1 = zeros(0, 0) # Don't store T1 for orbopt methods
  end
  open_dump(EC, "w") do io
    transfer_orbitals_to_store!(io, EC, orbital_data)
    dump_amplitudes(io, EC, T1, T2)
  end
  return
end

"""
    dump_wavefunction_with_amplitudes!(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab; orbopt=false)

  Dump orbitals and unrestricted CC amplitudes to the TREXIO file.

  Does nothing if `EC.options.wf.store` is empty.

  If `orbopt=true`, the orbitals are rotated using the T1 amplitudes
  via [`rotation_matrix`](@ref) and stored instead of the original orbitals.
  T1 amplitudes are not stored in that case.
"""
function dump_wavefunction_with_amplitudes!(EC::ECInfo, 
                                            T1a::AbstractMatrix, T1b::AbstractMatrix,
                                            T2a::AbstractArray{<:Number,4}, T2b::AbstractArray{<:Number,4},
                                            T2ab::AbstractArray{<:Number,4};
                                            orbopt::Bool=false)
  if EC.options.wf.store == ""
    return
  end
  println("Storing wavefunction with amplitudes to $(EC.options.wf.store) ...")
  # Pre-fetch orbital data BEFORE opening store file for writing
  # This is crucial when dump and store are the same file
  orbital_data = fetch_orbital_data(EC)
  if orbopt 
    # Rotation matrices from T1 amplitudes
    Rpq_a = rotation_matrix(EC, T1a; full=true)
    Rpq_b = rotation_matrix(EC, T1b; full=true, beta=true)
    if !isnothing(orbital_data)
    # Rotate orbitals and store rotated orbitals instead
      rotate_orbitaldata!(orbital_data, Rpq_a, Rpq_b)
    else
      # store rotation matrices as orbital data if no orbital data is available (e.g. from FCIDUMP)
      orbital_data = OrbitalData(SpinMatrix(Rpq_a, Rpq_b))
    end
    # Don't store T1 for orbopt methods
    T1a = zeros(0, 0)
    T1b = zeros(0, 0)
  end
  open_dump(EC, "w") do io
    transfer_orbitals_to_store!(io, EC, orbital_data)
    dump_amplitudes(io, EC, T1a, T1b, T2a, T2b, T2ab)
  end
  return
end

# Convenience wrappers for tuple arguments
function dump_wavefunction_with_amplitudes!(EC::ECInfo, T1::Tuple{<:AbstractMatrix}, T2::Tuple{<:AbstractArray{<:Number,4}};
                                            orbopt::Bool=false)
  dump_wavefunction_with_amplitudes!(EC, T1[1], T2[1]; orbopt)
end
function dump_wavefunction_with_amplitudes!(EC::ECInfo, 
                                            T1::Tuple{<:AbstractMatrix,<:AbstractMatrix}, 
                                            T2::Tuple{<:AbstractArray{<:Number,4},<:AbstractArray{<:Number,4},<:AbstractArray{<:Number,4}};
                                            orbopt::Bool=false)
  dump_wavefunction_with_amplitudes!(EC, T1[1], T1[2], T2[1], T2[2], T2[3]; orbopt)
end

# ============================================================================
# CIPHI determinant wavefunction storage
# ============================================================================

"""
    dump_wavefunction_with_determinants!(EC::ECInfo, dets, coeffs; nstates=0)

Dump orbitals and CIPHI determinants with CI coefficients to TREXIO file(s).

For single-state (nstates=0 or nstates=1), writes to `wf.store`.
For multi-state, writes each state to a separate file per TREXIO standard:
- State 1: `wf.store`
- State n>1: `wf.store` with `_stateN` suffix

Does nothing if `EC.options.wf.store` is empty.

# Arguments
- `dets::Vector{<:AbstractDeterminant}`: Determinants 
- `coeffs::AbstractVecOrMat{Float64}`: CI coefficients (vector for 1 state, matrix for multi-state)
- `nstates::Int=0`: Number of states (0 = infer from coeffs)
"""
function dump_wavefunction_with_determinants!(EC::ECInfo, dets::Vector{D}, 
                                              coeffs::AbstractVecOrMat{Float64};
                                              nstates::Int=0) where {D}
  if EC.options.wf.store == ""
    return
  end
  
  # Determine number of states
  if nstates == 0
    nstates = coeffs isa AbstractMatrix ? size(coeffs, 2) : 1
  end
  
  if nstates == 1
    # Single state
    coeffs_vec = coeffs isa AbstractMatrix ? coeffs[:, 1] : coeffs
    dump_determinants(EC, dets, coeffs_vec; state=1)
  else
    # Multi-state: write each state to separate file
    dump_determinants_multistate(EC, dets, coeffs isa AbstractMatrix ? coeffs : reshape(coeffs, :, 1))
  end
  return
end

"""
    try_fetch_starting_determinants(EC::ECInfo; OPattern=UInt64, nstates=1)

Try to read determinants and CI coefficients from a TREXIO file for CIPHI restart.

The logic follows CC amplitude restart:
- If `wf.start` is not empty: read from `wf.start` file(s)
- If `wf.start` is empty: try to read from `wf.dump` file(s)

For multi-state, reads from separate files per TREXIO standard.

Returns `(dets, coeffs, success::Bool)`.
"""
function try_fetch_starting_determinants(EC::ECInfo; OPattern::Type=UInt64, nstates::Int=1)
  use_start = EC.options.wf.start != ""
  
  # Check if ground state determinants exist
  if !has_determinants(EC; start=use_start, state=1)
    return SimpleDeterminant{OPattern}[], zeros(Float64, 0, nstates), false
  end
  
  if nstates == 1
    dets, coeffs = fetch_determinants(EC; start=use_start, OPattern=OPattern, state=1)
    return dets, reshape(coeffs, :, 1), true
  else
    # Multi-state: read from separate files
    dets_gs, coeffs_gs = fetch_determinants(EC; start=use_start, OPattern=OPattern, state=1)
    ndets = length(dets_gs)
    coeffs_matrix = zeros(Float64, ndets, nstates)
    coeffs_matrix[:, 1] = coeffs_gs
    
    for state in 2:nstates
      if has_determinants(EC; start=use_start, state=state)
        dets_s, coeffs_s = fetch_determinants(EC; start=use_start, OPattern=OPattern, state=state)
        if length(dets_s) == ndets
          coeffs_matrix[:, state] = coeffs_s
        else
          @warn "State $state has different number of determinants ($(length(dets_s)) vs $ndets)"
        end
      else
        @warn "State $state determinants not found"
      end
    end
    
    return dets_gs, coeffs_matrix, true
  end
end

"""
    try_fetch_restricted_starting_amplitudes(EC::ECInfo)

  Try to read and project restricted amplitudes from a TREXIO file.

  The logic is:
  - If `wf.start` is not empty: read amplitudes, MOs, and basis from `wf.start`, 
    then project amplitudes to the current MO basis (obtained via `fetch_orbitals` from `wf.dump`).
  - If `wf.start` is empty: try to read amplitudes from `wf.dump` (no projection needed).

  The occupied space projection rank is checked and a warning is printed if not full rank.

  Returns `(T1, T2, success::Bool)` for closed-shell case.
"""
function try_fetch_restricted_starting_amplitudes(EC::ECInfo)
  # Determine whether to use wf.start file
  use_start = EC.options.wf.start != ""
  
  # Check if amplitudes exist
  if !has_amplitudes(EC; unrestricted=false, start=use_start)
    return empty_restricted_amplitudes(EC)
  end
  
  # Read orbitals and orbital classes from source file (start or dump)
  cMO_old, type_old, basis_old = fetch_orbitals(EC; start=use_start)
  classes_old = use_start ? fetch_orbital_classes(EC; start=true) : (String[], String[])
  
  # Get target orbitals and classes from dump file (or same file if not using start)
  use_projection = false
  if use_start && EC.options.wf.dump != EC.options.wf.start
    if has_dumpfile(EC)
      cMO_new, type_new, current_basis = fetch_orbitals(EC)
      classes_new = fetch_orbital_classes(EC)
      use_projection = true
    end
  else
    # Same file - project onto current basis (in case orbitals were modified)
    if !isempty(EC.system)
      current_basis = generate_basis(EC, "ao")
      cMO_new = project_onto_basis(cMO_old, basis_old, current_basis; check=true)
      use_projection = true
    end
    classes_new = (String[], String[])
  end
  
  # Fetch amplitudes
  T1_old, T2_old = fetch_restricted_amplitudes(EC; start=use_start)
  
  if length(T1_old) == 0 && length(T2_old) == 0
    return empty_restricted_amplitudes(EC)
  end
 
  if !use_projection
    # No projection needed, return as-is
    return (T1_old, T2_old, true)
  end 
  # Project amplitudes
  T1, T2 = project_amplitudes(EC, T1_old, T2_old, cMO_old, cMO_new, basis_old, current_basis;
                               classes_old=classes_old, classes_new=classes_new)
  return (T1, T2, true)
end

"""
    try_fetch_unrestricted_starting_amplitudes(EC::ECInfo)

  Try to read and project unrestricted amplitudes from a TREXIO file.

  The logic is:
  - If `wf.start` is not empty: read amplitudes, MOs, and basis from `wf.start`, 
    then project amplitudes to the current MO basis (obtained via `fetch_orbitals` from `wf.dump`).
  - If `wf.start` is empty: try to read amplitudes from `wf.dump` (no projection needed).

  The occupied space projection rank is checked and a warning is printed if not full rank.

  Returns `(T1a, T1b, T2a, T2b, T2ab, success::Bool)` for unrestricted case.
"""
function try_fetch_unrestricted_starting_amplitudes(EC::ECInfo)
  # Determine whether to use wf.start file
  use_start = EC.options.wf.start != ""
  
  # Check if amplitudes exist
  if !has_amplitudes(EC; unrestricted=true, start=use_start)
    return empty_unrestricted_amplitudes(EC)
  end
  
  # Read orbitals, classes, and occupations from source file (start or dump)
  cMO_old, type_old, basis_old = fetch_orbitals(EC; start=use_start)
  classes_old = use_start ? fetch_orbital_classes(EC; start=true) : (String[], String[])
  occupations_old = use_start ? fetch_orbital_occupations(EC, "mo"; start=true) : (Float64[], Float64[])
  
  # Get target orbitals, classes, and occupations from dump file (or same file if not using start)
  use_projection = false
  occupations_new = (Float64[], Float64[])
  if use_start
    if has_dumpfile(EC)
      cMO_new, type_new, current_basis = fetch_orbitals(EC)
      classes_new = fetch_orbital_classes(EC)
      occupations_new = fetch_orbital_occupations(EC, "mo")
      use_projection = true
    end
  else
    # Same file - project onto current basis (in case orbitals were modified)
    if !isempty(EC.system)
      current_basis = generate_basis(EC, "ao")
      cMO_new = project_onto_basis(cMO_old, basis_old, current_basis; check=true)
      use_projection = true
    end
    classes_new = (String[], String[])
  end
  
  # Fetch amplitudes
  T1a_old, T1b_old, T2a_old, T2b_old, T2ab_old = fetch_unrestricted_amplitudes(EC; start=use_start)
  
  if length(T1a_old) == 0 && length(T2a_old) == 0 && length(T2ab_old) == 0
    return empty_unrestricted_amplitudes(EC)
  end
  if !use_projection
    # No projection needed, return as-is
    return (T1a_old, T1b_old, T2a_old, T2b_old, T2ab_old, true)
  end 
  # Project amplitudes using occupations for reliable occ/virt determination
  T1a, T1b, T2a, T2b, T2ab = project_amplitudes(EC, T1a_old, T1b_old, T2a_old, T2b_old, T2ab_old,
                                                 cMO_old, cMO_new, basis_old, current_basis;
                                                 classes_old=classes_old, classes_new=classes_new,
                                                 occupations_old=occupations_old, occupations_new=occupations_new)
  return (T1a, T1b, T2a, T2b, T2ab, true)
end

"""
    empty_restricted_amplitudes(EC::ECInfo)

  Return empty restricted amplitudes of the correct size.

  Returns `(T1, T2, false)` where T1 and T2 are zero arrays.
"""
function empty_restricted_amplitudes(EC::ECInfo)
  SP = EC.space
  nocc = length(SP['o'])
  nvirt = length(SP['v'])
  return (zeros(nvirt, nocc), zeros(nvirt, nvirt, nocc, nocc), false)
end

"""
    empty_unrestricted_amplitudes(EC::ECInfo)

  Return empty unrestricted amplitudes of the correct size.

  Returns `(T1a, T1b, T2a, T2b, T2ab, false)` where all arrays are zero.
"""
function empty_unrestricted_amplitudes(EC::ECInfo)
  SP = EC.space
  nocc_a = length(SP['o'])
  nocc_b = length(SP['O'])
  nvirt_a = length(SP['v'])
  nvirt_b = length(SP['V'])
  return (zeros(nvirt_a, nocc_a), zeros(nvirt_b, nocc_b),
          zeros(nvirt_a, nvirt_a, nocc_a, nocc_a),
          zeros(nvirt_b, nvirt_b, nocc_b, nocc_b),
          zeros(nvirt_a, nvirt_b, nocc_a, nocc_b), false)
end

"""
    print_main_singles(U1, nelem; info="", thr=1e-4)

  Utility function to print the main `nelem` singles.
"""
function print_main_singles(U1::AbstractMatrix, nelem; info="", thr=1e-4)
  println(info * " main singles:")
  nvir, nocc = size(U1)
  n = min(nelem, nvir * nocc)
  Uvec = vec(U1)
  idx = argmaxN(Uvec, n, by=abs)
  for i in idx
    i_occ = (i - 1) ÷ nvir + 1
    i_virt = (i - 1) % nvir + 1
    if abs(Uvec[i]) > thr
      output_single_excitation(Uvec[i], i_occ, i_virt)
    end
  end
  println()
end
end # module
