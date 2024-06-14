"""
    OrbTools

A collection of tools for working with orbitals
""" 
module OrbTools
using LinearAlgebra, TensorOperations, Printf

using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.MSystem
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions

export guess_orb, load_orbitals, orbital_energies
export show_orbitals
export rotate_orbs, rotate_orbs!, normalize_phase!

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
    guess_sad(EC::ECInfo)
  
  Guess MO coefficients from atomic densities.
"""
function guess_sad(EC::ECInfo)
  # minao = "ano-rcc-mb"
  minao = "ano-r0"
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
    orbs = load_all(EC, EC.options.wf.orb, Val(2))
    return SpinMatrix(orbs...)
  else
    error("unknown guess type")
    return SpinMatrix()
  end
end

"""
    load_orbitals(EC::ECInfo, orbsfile::String="")

  Load (last) orbitals.
  
  - from file `orbsfile` if not empty
  - from file [`WfOptions.orb`](@ref ECInfos.WfOptions) if not empty
  - error if all files are empty

  Returns `::MOs`.
"""
function load_orbitals(EC::ECInfo, orbsfile::String="")
  if !isempty(strip(orbsfile))
    # orbsfile will be used
  elseif !isempty(strip(EC.options.wf.orb))
    orbsfile = EC.options.wf.orb
  else
    error("no orbitals found")
  end
  return SpinMatrix(load_all(EC, orbsfile, Val(2))...)
end

"""
    orbital_energies(EC::ECInfo, spincase::Symbol=:α)

  Return orbital energies for a given `spincase`∈{`:α`,`:β`}.
"""
function orbital_energies(EC::ECInfo, spincase::Symbol=:α)
  if spincase == :α
    eps = load(EC, "e_m")
    ϵo = eps[EC.space['o']]
    ϵv = eps[EC.space['v']]
  else
    eps = load(EC, "e_M")
    ϵo = eps[EC.space['O']]
    ϵv = eps[EC.space['V']]
  end
  return ϵo, ϵv
end

"""
    rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)

  Rotate orbitals `orb1` and `orb2` from [`WfOptions.orb`](@ref ECInfos.WfOptions) 
  by `angle` degrees. For unrestricted orbitals, `spin` can be `:α` or `:β`.
"""
function rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)
  cMO = load_orbitals(EC)
  descr = file_description(EC, EC.options.wf.orb)
  if is_restricted(cMO)
    cMOrot = cMO[1]
  else
    cMOrot = cMO[spin]
  end
  rotate_orbs!(cMOrot, orb1, orb2, angle)
  descr *= " rot$(orb1)&$(orb2)by$(angle)"
  if is_restricted(cMO)
    save!(EC, EC.options.wf.orb, cMO[1], description=descr)
  else
    save!(EC, EC.options.wf.orb, cMO..., description=descr)
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

  Print the MO coefficients in [`WfOptions.orb`](@ref ECInfos.WfOptions) 
  with respect to the atomic orbitals.
  
  `range` is a range of molecular orbitals to be printed.
"""
function show_orbitals(EC::ECInfo, range=nothing)
  basis = generate_basis(EC, "ao")
  cMO = load_orbitals(EC)
  descr = file_description(EC, EC.options.wf.orb)
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

end #module
