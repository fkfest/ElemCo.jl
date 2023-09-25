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
using ..ElemCo.OrbTools

export calc_singles_energy_using_dfock
export update_singles, update_doubles, update_singles!, update_doubles!, update_deco_doubles, update_deco_triples
export calc_singles_norm, calc_doubles_norm, calc_deco_doubles_norm, calc_deco_triples_norm
export read_starting_guess4amplitudes, save_current_singles, save_current_doubles, starting_amplitudes
export try2save_singles!, try2save_doubles!, try2start_singles, try2start_doubles

"""
    calc_singles_energy_using_dfock(EC::ECInfo, T1; fock_only=false)

  Calculate coupled-cluster closed-shell singles energy 
  using dressed fock matrix.

  if `fock_only` is true, the energy will be calculated using only non-dressed fock matrix.
"""
function calc_singles_energy_using_dfock(EC::ECInfo, T1; fock_only=false)
  SP = EC.space
  ET1 = 0.0
  if length(T1) > 0
    fock = load(EC, "f_mm")
    if fock_only
      dfock = fock
    else
      dfock = load(EC, "df_mm")
    end
    fov = dfock[SP['o'],SP['v']] + fock[SP['o'],SP['v']] # undressed part should be with factor two
    @tensoropt ET1 = fov[i,a] * T1[a,i]
  end
  return ET1
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
    ϵX = load(EC,"e_X")
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
  ϵX = load(EC,"e_X")
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
    calc_doubles_norm(T2)

  Calculate squared norm of closed-shell doubles amplitudes.
"""
function calc_doubles_norm(T2)
  @tensoropt NormT2 = (2.0*T2[a,b,i,j] - T2[b,a,i,j])*T2[a,b,i,j]
  return NormT2
end

"""
    calc_doubles_norm(T2a, T2b, T2ab)

  Calculate squared norm of unrestricted doubles amplitudes.
"""
function calc_doubles_norm(T2a, T2b, T2ab)
  @tensoropt begin
    NormT2 = 0.25*T2a[a,b,i,j]*T2a[a,b,i,j]
    NormT2 += 0.25*T2b[a,b,i,j]*T2b[a,b,i,j]
    NormT2 += T2ab[a,b,i,j]*T2ab[a,b,i,j]
  end
  return NormT2
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
    try2save_singles!(EC::ECInfo, singles...)

  Save singles amplitudes to file `EC.options.cc.save*"_singles"`.
"""
function try2save_singles!(EC::ECInfo, singles...)
  if EC.options.cc.save != ""
    filename = EC.options.cc.save*"_singles"
    println("Save singles amplitudes to file $filename")
    save!(EC, filename, singles..., description="singles amplitudes")
  end
end

"""
    try2save_doubles!(EC::ECInfo, doubles...)

  Save doubles amplitudes to file `EC.options.cc.save*"_doubles"`.
"""
function try2save_doubles!(EC::ECInfo, doubles...)
  if EC.options.cc.save != ""
    filename = EC.options.cc.save*"_doubles"
    println("Save doubles amplitudes to file $filename")
    save!(EC, filename, doubles..., description="doubles amplitudes")
  end
end

"""
    try2start_singles(EC::ECInfo)

  Read singles amplitudes from file `EC.options.cc.start*"_singles"`.
"""
function try2start_singles(EC::ECInfo)
  if EC.options.cc.start != ""
    filename = EC.options.cc.start*"_singles"
    if file_exists(EC, filename)
      println("Read singles amplitudes from file $filename")
      return load(EC, filename)
    end
  end
  return []
end

"""
    try2start_doubles(EC::ECInfo)

  Read doubles amplitudes from file `EC.options.cc.start*"_doubles"`.
"""
function try2start_doubles(EC::ECInfo)
  if EC.options.cc.start != ""
    filename = EC.options.cc.start*"_doubles"
    if file_exists(EC, filename)
      println("Read doubles amplitudes from file $filename")
      return load(EC, filename)
    end
  end
  return []
end

"""
    read_starting_guess4amplitudes(EC::ECInfo, level::Int, spins...)

  Read starting guess for excitation `level`.

  The guess will be read from `T_vo`, `T_VO`, `T_vvoo` etc files.
  If the file does not exist, the guess will be a zeroed-vector.
"""
function read_starting_guess4amplitudes(EC::ECInfo, level::Int, spins...)
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
    return load(EC, filename)
  else
    return zeros(len_spaces(EC, spaces)...)
  end
end

"""
    save_current_singles(EC::ECInfo, T1)

  Save current singles amplitudes `T1` to file `T_vo`
"""
function save_current_singles(EC::ECInfo, T1)
  save!(EC, "T_vo", T1)
end

"""
    save_current_singles(EC::ECInfo, T1a, T1b)

  Save current singles amplitudes `T1a` and `T1b` to files `T_vo` and `T_VO`
"""
function save_current_singles(EC::ECInfo, T1a, T1b)
  save!(EC, "T_vo", T1a)
  save!(EC, "T_VO", T1b)
end

"""
    save_current_doubles(EC::ECInfo, T2)

  Save current doubles amplitudes `T2` to file `T_vvoo`
"""
function save_current_doubles(EC::ECInfo, T2)
  save!(EC, "T_vvoo", T2)
end

"""
    save_current_doubles(EC::ECInfo, T2a, T2b, T2ab)

  Save current doubles amplitudes `T2a`, `T2b`, and `T2ab` to files `T_vvoo`, `T_VVOO`, and `T_vVoO`
"""
function save_current_doubles(EC::ECInfo, T2a, T2b, T2ab)
  save!(EC, "T_vvoo", T2a)
  save!(EC, "T_VVOO", T2b)
  save!(EC, "T_vVoO", T2ab)
end

"""
    starting_amplitudes(EC::ECInfo, method::ECMethod)

  Prepare starting amplitudes for coupled cluster calculation.
  
  The starting amplitudes are read from files `T_vo`, `T_VO`, `T_vvoo`, etc.
  If the files do not exist, the amplitudes are initialized to zero.
  The order of amplitudes is as follows:
  - singles: `α`, `β`
  - doubles: `αα`, `ββ`, `αβ`
  - triples: `ααα`, `βββ`, `ααβ`, `αββ`
  Return a list of vectors of starting amplitudes 
  and a list of ranges for excitation levels.
"""
function starting_amplitudes(EC::ECInfo, method::ECMethod)
  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 3
    error("starting_amplitudes only implemented upto triples")
  end
  if method.unrestricted
    namps = sum([i + 1 for i in 1:highest_full_exc])
    exc_ranges = [1:2, 3:5, 6:9]
    spins = [(:α,), (:β,), (:α, :α), (:β, :β), (:α, :β), (:α, :α, :α), (:β, :β, :β), (:α, :α, :β), (:α, :β, :β)]
  else
    namps = highest_full_exc
    exc_ranges = [1:1, 2:2, 3:3]
    spins = [(:α,), (:α, :α), (:α, :α, :α)]
  end
  Amps = AbstractArray[Float64[] for i in 1:namps]
  # starting guesses
  for (iex,ex) in enumerate(method.exclevel)
    if ex == :full
      for iamp in exc_ranges[iex]
        Amps[iamp] = read_starting_guess4amplitudes(EC, iex, spins[iamp]...)
      end
    end
  end
  return Amps, exc_ranges
end

end # module