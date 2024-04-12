"""
    OrbTools

A collection of tools for working with orbitals
""" 
module OrbTools
using LinearAlgebra, TensorOperations

using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.TensorTools

export guess_orb, load_orbitals, orbital_energies, is_unrestricted_MO
export rotate_orbs, rotate_orbs!

"""
    guess_hcore(EC::ECInfo)

  Guess MO coefficients from core Hamiltonian.
"""
function guess_hcore(EC::ECInfo)
  hsmall = load(EC,"h_AA")
  sao = load(EC,"S_AA")
  ϵ,cMO = eigen(Hermitian(hsmall),Hermitian(sao))
  return cMO
end
  
"""
    guess_sad(EC::ECInfo)
  
  Guess MO coefficients from atomic densities.
"""
function guess_sad(EC::ECInfo)
  # minao = "ano-rcc-mb"
  minao = "ano-r0"
  # minao = "sto-6g"
  bminao = BasisSet(minao,genxyz(EC.system))
  bao = generate_basis(EC.system, "ao")
  smin2ao = overlap(bminao,bao)
  smin = overlap(bminao)
  eldist = electron_distribution(EC.system,minao)
  sao = load(EC,"S_AA")
  denao = smin2ao' * diagm(eldist./diag(smin)) * smin2ao
  eigs,cMO = eigen(Hermitian(-denao),Hermitian(sao))
  return cMO
end

function guess_gwh(EC::ECInfo)
  error("not implemented yet")
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
    return load(EC,EC.options.wf.orb)
  else
    error("unknown guess type")
  end
end

"""
    load_orbitals(EC::ECInfo, orbsfile::String="")

  Load (last) orbitals.
  
  - from file `orbsfile` if not empty
  - from file [`WfOptions.orb`](@ref ECInfos.WfOptions) if not empty
  - error if all files are empty
"""
function load_orbitals(EC::ECInfo, orbsfile::String="")
  if !isempty(strip(orbsfile))
    # orbsfile will be used
  elseif !isempty(strip(EC.options.wf.orb))
    orbsfile = EC.options.wf.orb
  else
    error("no orbitals found")
  end
  return load(EC, orbsfile)
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
    is_unrestricted_MO(cMO)

  Return `true` if `cMO` is unrestricted MO coefficients of the form 
  [CMOα, CMOβ].
"""
function is_unrestricted_MO(cMO)
  if ndims(cMO) == 1
    return true
  elseif ndims(cMO) == 2
    return false
  else
    error("Wrong number of dimensions in cMO: ", ndims(cMO))
  end
end


"""
    rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)

  Rotate orbitals `orb1` and `orb2` from [`WfOptions.orb`](@ref ECInfos.WfOptions) 
  by `angle` degrees. For unrestricted orbitals, `spin` can be `:α` or `:β`.
"""
function rotate_orbs(EC::ECInfo, orb1, orb2, angle=90; spin::Symbol=:α)
  cMO = load_orbitals(EC)
  descr = file_description(EC, EC.options.wf.orb)
  if is_unrestricted_MO(cMO)
    isp = (spin == :α) ? 1 : 2
    cMOrot = cMO[isp]
  else
    cMOrot = cMO
  end
  rotate_orbs!(cMOrot, orb1, orb2, angle)
  descr *= " rot$(orb1)&$(orb2)by$(angle)"
  if is_unrestricted_MO(cMO)
    save!(EC, EC.options.wf.orb, cMO..., description=descr)
  else
    save!(EC, EC.options.wf.orb, cMO, description=descr)
  end
end

"""
    rotate_orbs!(cMO::AbstractArray, orb1, orb2, angle=90)

  Rotate orbitals `orb1` and `orb2` from `cMO` by `angle` degrees.

  `cMO` is a matrix of MO coefficients.
"""
function rotate_orbs!(cMO::AbstractArray, orb1, orb2, angle=90)
  @assert ndims(cMO) == 2 "Wrong number of dimensions in cMO: $(ndims(cMO))"
  if orb1 > size(cMO,2) || orb2 > size(cMO,2)
    error("orbital index out of range")
  end
  if orb1 == orb2
    error("orbital indices must be different")
  end
  cMO[:,[orb1,orb2]] = cMO[:,[orb1,orb2]] * [cosd(angle) -sind(angle); sind(angle) cosd(angle)]
end

end #module
