"""
    OrbTools

A collection of tools for working with orbitals
""" 
module OrbTools
using LinearAlgebra, Printf

using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.MSystems
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions

export guess_orb, guess_pos_orb, load_orbitals, load_rotations, load_left_right_rotations
export orbital_energies, load_positron_orbitals 
export show_orbitals
export rotate_orbs, rotate_orbs!, normalize_phase!
export left_from_right_rotations, project_onto_basis

"""
    guess_hcore(EC::ECInfo)

  Guess MO coefficients from core Hamiltonian.
"""
function guess_hcore(EC::ECInfo)
  hsmall = load(EC, "h_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  ϵ, cMO = eigen(Hermitian(hsmall), Hermitian(sao))
  return SpinMatrix(cMO)
end

"""
    guess_pos_hcore(EC::ECInfo)

  Guess MO coefficients for positron from core Hamiltonian.
"""
function guess_pos_hcore(EC::ECInfo)
  hsmall = load(EC, "h_positron_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  ϵ, cMO = eigen(Hermitian(hsmall), Hermitian(sao))
  return SpinMatrix(cMO)
end
  
"""
    guess_sad(EC::ECInfo)
  
  Guess MO coefficients from atomic densities.
"""
function guess_sad(EC::ECInfo)
  minao = "ano-rcc-mb"
  # minao = "ano-r0"
  # minao = "sto-6g"
  bminao = generate_basis(EC, basisset=minao)
  bao = generate_basis(EC, "ao")
  smin2ao = overlap(bminao, bao)
  smin = overlap(bminao)
  eldist = electron_distribution(EC.system, minao)
  sao = load(EC, "S_AA", Val(2))
  denao = smin2ao' * diagm(eldist./diag(smin)) * smin2ao
  eigs, cMO = eigen(Hermitian(-denao), Hermitian(sao))
  return SpinMatrix(cMO)
end

function guess_gwh(EC::ECInfo)
  error("not implemented yet")
  return SpinMatrix()
end

"""
    guess_positron(EC::ECInfo)

  Initialize positron MO coefficients as zeroes.
"""
function guess_positron(EC::ECInfo)
  hsmall = load(EC, "h_positron_AA", Val(2))
  sao = load(EC, "S_AA", Val(2))
  ϵ, cMO = eigen(Hermitian(hsmall), Hermitian(sao))
  return SpinMatrix(cMO)
end

"""
    guess_orb(EC::ECInfo, guess::Symbol)

  Calculate starting guess for MO coefficients.
  Type of initial guess for MO coefficients is given by `guess`.

  See [`ScfOptions.guess`](@ref ECInfos.ScfOptions) for possible values.
"""
function guess_orb(EC::ECInfo, guess::Symbol)
  if guess == :HCORE || guess == :hcore
    return guess_hcore(EC)
  elseif guess == :SAD || guess == :sad
    return guess_sad(EC)
  elseif guess == :GWH || guess == :gwh
    return guess_gwh(EC)
  elseif guess == :ORB || guess == :orb
    return load_orbitals(EC)
  else
    error("unknown guess type")
    return SpinMatrix()
  end
end

"""
  guess_pos_orb(EC::ECInfo, guess::Symbol)

  Calculate starting guess for MO positron coefficients.
  Type of initial guess for MO coefficients is given by `guess`.

  See [`ScfOptions.guess`](@ref ECInfos.ScfOptions) for possible values.

"""

function guess_pos_orb(EC::ECInfo, guess::Symbol)
  if guess == :HCORE || guess == :hcore
    return guess_pos_hcore(EC)
  else
    error("unknown guess type")
    return SpinMatrix()
  end
end

"""
    load_orbitals(EC::ECInfo)

  Load (last) orbitals from file [`WfOptions.dump`](@ref ECInfos.WfOptions). 

  If the basis has changed, the orbitals will be projected onto the new basis.
  Returns `::SpinMatrix`. 
"""
function load_orbitals(EC::ECInfo)
  cMO, type, basis = fetch_orbitals(EC)
  current_basis = generate_basis(EC, "ao")
  return project_onto_basis(cMO, basis, current_basis; check=true)
end

"""
    left_from_right_rotations(cMOr::SpinMatrix)

  Calculate left biorthogonal rotation coefficients from right BO coefficients.
"""
function left_from_right_rotations(cMOr::SpinMatrix{T}) where {T}
  if is_restricted(cMOr)
    cMOl = SpinMatrix((inv(cMOr[1]))')
    restrict!(cMOl)
  else
    cMOl = SpinMatrix{T}()
    for ispin = 1:2
      cMOl[ispin] = (inv(cMOr[ispin]))'
    end
  end
  return cMOl
end

"""
    load_rotations(EC::ECInfo)

  Load (last) orbital rotations from file [`WfOptions.dump`](@ref ECInfos.WfOptions).

  Returns `::SpinMatrix`. 
"""
function load_rotations(EC::ECInfo)
  cRot, type = fetch_rotations(EC)
  if !is_rotation(type)
    error("Dump file does not contain orbital rotations")
  end
  return cRot
end

"""
    load_left_right_rotations(EC::ECInfo) -> (left::SpinMatrix, right::SpinMatrix)

  Load (last) left and right orbital rotations from file [`WfOptions.dump`](@ref ECInfos.WfOptions).

  If the type of the rotations does not contain the word `biorthogonal`, 
  the same rotation is returned for left and right (can be checked with `===`).
"""
function load_left_right_rotations(EC::ECInfo)
  cRot, type = fetch_rotations(EC)
  if !is_rotation(type)
    error("Dump file does not contain orbital rotations")
  end
  if is_biorthogonal(type)
    cRotL = left_from_right_rotations(cRot)
    return cRotL, cRot
  else
    return cRot, cRot
  end
end

"""
    load_positron_orbitals(EC::ECInfo)

  Load (last) positron orbitals from file [`WfOptions.dump`](@ref ECInfos.WfOptions).
  
  Returns `::SpinMatrix`. 
"""
function load_positron_orbitals(EC::ECInfo)
  cMO, type, basis = fetch_orbitals(EC, "po")
  current_basis = generate_basis(EC, "ao")
  return project_onto_basis(cMO, basis, current_basis; check=true)
end

"""
    orbital_energies(EC::ECInfo, spincase::Symbol=:α)

  Return orbital energies for a given `spincase`∈{`:α`,`:β`}.
"""
function orbital_energies(EC::ECInfo, spincase::Symbol=:α)
  if spincase == :α
    eps = load1idx(EC, "e_m")
    ϵo = eps[EC.space['o']]
    ϵv = eps[EC.space['v']]
  else
    eps = load1idx(EC, "e_M")
    ϵo = eps[EC.space['O']]
    ϵv = eps[EC.space['V']]
  end
  return ϵo, ϵv
end

"""
    rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)

  Rotate orbitals `orb1` and `orb2` from [`WfOptions.dump`](@ref ECInfos.WfOptions) 
  by `angle` degrees. For unrestricted orbitals, `spin` can be `:α` or `:β`.
"""
function rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)
  cMO, descr = fetch_rotations(EC)
  basis = nothing
  if !is_rotation(descr)
    cMO, descr, basis = fetch_orbitals(EC)
  end
  if is_restricted(cMO)
    cMOrot = cMO[1]
  else
    cMOrot = cMO[spin]
  end
  rotate_orbs!(cMOrot, orb1, orb2, angle)
  descr *= " rot$(orb1)&$(orb2)by$(angle)"
  if isnothing(basis)
    dump_rotations(EC, cMO; type=descr)
  else
    dump_orbitals(EC, cMO; basis=basis, type=descr)
  end
end

"""
    rotate_orbs!(cMO::Matrix, orb1, orb2, angle=90)

  Rotate orbitals `orb1` and `orb2` from `cMO` by `angle` degrees.

  `cMO` is a matrix of MO coefficients.
"""
function rotate_orbs!(cMO::Matrix, orb1, orb2, angle=90)
  if orb1 > size(cMO,2) || orb2 > size(cMO,2)
    error("orbital index out of range")
  end
  if orb1 == orb2
    error("orbital indices must be different")
  end
  cMO[:,[orb1,orb2]] = cMO[:,[orb1,orb2]] * [cosd(angle) -sind(angle); sind(angle) cosd(angle)]
end

"""
    show_orbitals(EC::ECInfo, range=nothing)

  Print the MO coefficients in [`WfOptions.dump`](@ref ECInfos.WfOptions) 
  with respect to the atomic orbitals.
  
  `range` is a range of molecular orbitals to be printed.
"""
function show_orbitals(EC::ECInfo, range=nothing)
  cMO, descr, basis = fetch_orbitals(EC)
  if isnothing(range)
    range = 1:size(cMO, 2)
  end
  println(range," orbitals from $descr")
  if is_restricted(cMO)
    show_orbitals(EC, cMO[1], basis, range)
  else
    println("Alpha orbitals:")
    show_orbitals(EC, cMO[1], basis, range)
    println("Beta orbitals:")
    show_orbitals(EC, cMO[2], basis, range)
  end
end

"""
    show_orbitals(EC::ECInfo, cMO::Matrix, basis::BasisSet, range=1:size(cMO,2)

  Print the MO coefficients in `cMO` with respect to the atomic orbitals in `basis`.

  `range` is a range of molecular orbitals to be printed.
"""
function show_orbitals(EC::ECInfo, cMO::Matrix, basis::BasisSet, range=1:size(cMO,2))
  aos = ao_list(basis)
  nao = length(aos)
  nmo = size(cMO,2)
  nlargest = EC.options.wf.print_nlargest
  thr = EC.options.wf.print_thr
  @assert size(cMO,1) == nao "Wrong number of atomic orbitals in cMO: $(size(cMO,1)) vs $(nao)"
  for imo in range
    @assert imo in 1:size(cMO,2) "Wrong range of orbitals: $(range). Number of orbitals: $(nmo)"
    print("$imo: ")
    # get nlargest coefficients (round to 4 digits to avoid numerical noise)
    idx = argmaxN(cMO[:,imo], nlargest, by=x->round(abs(x),digits=4))
    for iao in idx
      if abs(cMO[iao,imo]) < thr
        continue
      end
      @printf("% .3f", cMO[iao,imo])
      print("(")
      print_ao(aos[iao], basis)
      print(") ")
    end
    println()
  end
end

"""
    normalize_phase!(cMO)

  Normalize the phase of the MO coefficients in `cMO`.

  The phase is chosen such that the first largest coefficient is positive.
"""
function normalize_phase!(cMO)
  nmo = size(cMO,2)
  for imo in 1:nmo
    maxao = argmaxN(cMO[:,imo], 1, by=x->round(abs(x),digits=4))[1]
    if cMO[maxao,imo] < 0
      cMO[:,imo] .= -cMO[:,imo]
    end
  end
end

"""
    project_onto_basis(cMO::SpinMatrix, old_basis::BasisSet, new_basis::BasisSet; check=false)

  Project the MO coefficients onto a new basis.

If `check` is true, the function will check whether the projection is needed and return the same
array `cMO` if it is not (i.e., it can be checked with `===`).
"""
function project_onto_basis(cMO::SpinMatrix, old_basis::BasisSet, new_basis::BasisSet; check=false)
  SAO = overlap(new_basis)
  proj = inv(SAO) * overlap(new_basis, old_basis)
  if check && SAO*proj ≈ SAO
    return cMO
  end
  if is_restricted(cMO)
    cMO_new = proj * cMO[1]
    return SpinMatrix(cMO_new)
  else
    cMO_newα = proj * cMO[1]
    cMO_newβ = proj * cMO[2]
    return SpinMatrix(cMO_newα, cMO_newβ)
  end
end

end #module
