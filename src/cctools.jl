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
export calc_singles_norm, calc_doubles_norm, calc_contra_singles_norm, calc_contra_doubles_norm, calc_deco_doubles_norm, calc_deco_triples_norm
export read_starting_guess4amplitudes, save_current_singles, save_current_doubles, starting_amplitudes
export transform_amplitudes2lagrange_multipliers!
export try2save_amps!, try2start_amps, try2save_singles!, try2save_doubles!, try2start_singles, try2start_doubles
export contra2covariant

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
    NormT2 = 0.25*T2a[a,b,i,j]*T2a[a,b,i,j]
    NormT2 += 0.25*T2b[a,b,i,j]*T2b[a,b,i,j]
    NormT2 += T2ab[a,b,i,j]*T2ab[a,b,i,j]
  end
  return NormT2
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
    save_or_start_file(EC::ECInfo, type, save=true)

  Return filename and description for saving or starting amplitudes/lagrange multipliers.

  `type` is either `"T"` for amplitudes or `"LM"` for Lagrange multipliers.
  If `save` is true, the filename for saving is returned, otherwise the filename for starting.
"""
function save_or_start_file(EC::ECInfo, type, save=true)
  mainfilename = descr = ""
  if type == "T"
    descr = "amplitudes"
    mainfilename = save ? EC.options.cc.save : EC.options.cc.start
  elseif type == "LM"
    descr = "Lagrange multipliers"
    mainfilename = save ? EC.options.cc.save_lm : EC.options.cc.start_lm
  else
    error("unknown type $type")
  end
  return mainfilename, descr
end

"""
    try2save_amps!(EC::ECInfo, excitation_level::AbstractString, amps...; type="T")

  Save amplitudes (type="T") or Lagrange multipliers (type="LM") 
  to file `EC.options.cc.save[_lm]*"_"*excitation_level`.
"""
function try2save_amps!(EC::ECInfo, excitation_level::AbstractString, amps...; type="T")
  mainfilename, descr = save_or_start_file(EC, type)
  if mainfilename != ""
    filename = mainfilename*"_"*excitation_level
    println("Save $excitation_level $descr to file $filename")
    save!(EC, filename, amps..., description=excitation_level*" "*descr)
  end
end

"""
    try2start_amps(EC::ECInfo, excitation_level::AbstractString; type="T")

  Read amplitudes (type="T") or Lagrange multipliers (type="LM") 
  from file `EC.options.cc.start[_lm]*"_"*excitation_level`.
"""
function try2start_amps(EC::ECInfo, excitation_level::AbstractString; type="T")
  mainfilename, descr = save_or_start_file(EC, type)
  if mainfilename != ""
    filename = mainfilename*"_"*excitation_level
    if file_exists(EC, filename)
      println("Read $excitation_level $descr from file $filename")
      return load(EC, filename)
    end
  end
  return []
end

"""
    try2save_singles!(EC::ECInfo, singles...; type="T")

  Save singles amplitudes (type="T") or Lagrange multipliers (type="LM") 
  to file `EC.options.cc.save[_lm]*"_singles"`.
"""
function try2save_singles!(EC::ECInfo, singles...; type="T")
  try2save_amps!(EC, "singles", singles...; type)
end

"""
    try2save_doubles!(EC::ECInfo, doubles...; type="T")

  Save doubles amplitudes (type="T") or Lagrange multipliers (type="LM") 
  to file `EC.options.cc.save[_lm]*"_doubles"`.
"""
function try2save_doubles!(EC::ECInfo, doubles...; type="T")
  try2save_amps!(EC, "doubles", doubles...; type)
end

"""
    try2start_singles(EC::ECInfo; type="T")

  Read singles amplitudes (type="T") or Lagrange multipliers (type="LM")
  from file `EC.options.cc.start[_lm]*"_singles"`.
"""
function try2start_singles(EC::ECInfo; type="T")
  return try2start_amps(EC, "singles"; type)
end

"""
    try2start_doubles(EC::ECInfo; type="T")

  Read doubles amplitudes (type="T") or Lagrange multipliers (type="LM")
  from file `EC.options.cc.start[_lm]*"_doubles"`.
"""
function try2start_doubles(EC::ECInfo; type="T")
  return try2start_amps(EC, "doubles"; type)
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
  if is_unrestricted(method)
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

"""
    transform_amplitudes2lagrange_multipliers!(Amps, exc_ranges)

  Transform amplitudes to first guess for Lagrange multipliers.

  The amplitudes are transformed in-place. 
"""
function transform_amplitudes2lagrange_multipliers!(Amps, exc_ranges)
  singles, doubles, triples = exc_ranges[1:3]
  unrestricted = (length(singles) == 2)
  @assert (unrestricted && length(doubles) == 3) || (!unrestricted && length(doubles) == 1)
  # add singles to doubles
  add_singles2doubles!(Amps[doubles]..., Amps[singles]...)
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

end # module