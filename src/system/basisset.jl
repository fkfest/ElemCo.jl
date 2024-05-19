"""
    BasisSets

  Module for working with basis sets.

  The basis set is stored in the [`BasisSet`](@ref) instance.
  The basis set can be generated using the [`generate_basis`](@ref) function.
"""
module BasisSets
using Unitful, UnitfulAtomic
using AtomsBase
using StaticArrays
using Printf
using DocStringExtensions
using ..ElemCo.Utils
using ..ElemCo.Elements
using ..ElemCo.MSystem
using ..ElemCo.AbstractEC

export BasisCenter, BasisSet
export BasisContraction, AbstractAngularShell, SphericalAngularShell, CartesianAngularShell
export shell_range, center_range, is_cartesian, combine
export n_subshells, n_primitives, n_coefficients, n_angularshells, n_ao
export coefficients_1mat, n_coefficients_1mat
export basis_name, generate_basis, guess_norb

export ILibcint5

include("basiscenter.jl")  
include("parse_basis.jl")
include("intlibs.jl")

"""
    BasisSet

  A basis set with basis centers (atoms) and basis functions.

  $(TYPEDFIELDS)
"""
struct BasisSet
  """ array of basis centers (atoms) with basis functions."""
  centers::Vector{BasisCenter}
  """ indices for angular shells."""
  shell_indices::Vector{CartesianIndex{2}}
  """ center ranges for each basis set in a combined set."""
  center_ranges::Vector{UnitRange{Int}}
  """ angular shell ranges for each basis sets in a combined set."""
  shell_ranges::Vector{UnitRange{Int}}
  """ infos for integral library."""
  lib::AbstractILib
end

function BasisSet(centers::Vector{BasisCenter}, lib::AbstractILib)
  shell_indices = [CartesianIndex(i, j) for (i,c) in enumerate(centers) for j in 1:n_angularshells(c)]
  center_ranges = [1:length(centers)]
  shell_ranges = [1:length(shell_indices)]
  return BasisSet(centers, shell_indices, center_ranges, shell_ranges, lib)
end

function BasisSet(centers::Vector{BasisCenter}, center_ranges, shell_ranges, lib::AbstractILib)
  shell_indices = [CartesianIndex(i, j) for (i,c) in enumerate(centers) for j in 1:n_angularshells(c)]
  return BasisSet(centers, shell_indices, center_ranges, shell_ranges, lib)
end

"""
    Base.length(bs::BasisSet)

  Return the number of angular shells in the basis set.
"""
Base.length(bs::BasisSet) = length(bs.shell_indices)

"""
    Base.getindex(bs::BasisSet, i::Int, j::Int)

  Return the the angular shell.
"""
Base.getindex(bs::BasisSet, i::Int, j::Int) = bs.centers[i].shells[j]

"""
    Base.getindex(bs::BasisSet, i::CartesianIndex{2})

  Return the the angular shell.
"""
Base.getindex(bs::BasisSet, i::CartesianIndex{2}) = bs[i.I[1],i.I[2]]

"""
    Base.getindex(bs::BasisSet, i::Int)

  Return the the angular shell.
"""
Base.getindex(bs::BasisSet, i::Int) = bs[bs.shell_indices[i]]

"""
    Base.iterate(bs::BasisSet, state=1)

  Iterate over the basis angular shells in the basis set.
"""
function Base.iterate(bs::BasisSet, state=1)
  if state > length(bs)
    return nothing
  else
    return (bs[state], state+1)
  end
end

"""
    combine(bs1::BasisSet, bs2::BasisSet)

  Combine two basis sets. 

  The centers are concatenated. 
  The center/shell ranges (`center_ranges`/`shell_ranges`) corresponding to the centers/shells 
  for each basis set   can be used to access the centers/shells in the combined basis set, e.g.
  `bs.centers[i] for i in bs.center_ranges[1]` gives the centers of the first basis set 
  in the combined set.
"""
function combine(bs1::BasisSet, bs2::BasisSet)
  centers = vcat(bs1.centers, bs2.centers)
  set_id!(centers, 1)
  center_ranges = vcat(bs1.center_ranges, [r .+ length(bs1.centers) for r in bs2.center_ranges])
  shell_ranges = vcat(bs1.shell_ranges, [r .+ length(bs1.shell_indices) for r in bs2.shell_ranges])
  return BasisSet(centers, center_ranges, shell_ranges, ILibcint5(centers))
end

function Base.show(io::IO, bs::BasisSet)
  for center in bs.centers
    println(io, center)
  end
end

"""
    shell_range(bs::BasisSet, i::Int)

  Return the range of angular shells for the `i`th basis set.

  The range is used to access the angular shells in the basis set, e.g.,
  `bs[i] for i in shell_range(bs, 1)` gives the angular shells of the first basis set.
"""
shell_range(bs::BasisSet, i::Int) = bs.shell_ranges[i]

"""
    center_range(bs::BasisSet, i::Int)

  Return the range of centers for the `i`th basis set.

  The range is used to access the centers in the basis set, e.g.,
  `bs.centers[i] for i in center_range(bs, 1)` gives the centers of the first basis set.
"""
center_range(bs::BasisSet, i::Int) = bs.center_ranges[i]

"""
    is_cartesian(bs::BasisSet)

  Check if the basis set is Cartesian.
"""
is_cartesian(bs::BasisSet) = bs[1] isa CartesianAngularShell

"""
    basis_name(atoms, type="ao")

  Return the name of the basis set (or `unknown` if not found).
  `atoms` can be a single atom `::Atom` or a system `::AbstractSystem`.
"""
function basis_name(atoms, type="ao")
  if haskey(atoms, :basis) && haskey(atoms[:basis], type)
    return lowercase(atoms[:basis][type])
  else
    return "unknown"
  end
end

"""
    generate_basis(EC::ECInfo, type="ao"; basisset::AbstractString="")

  Generate basis sets for integral calculations.

  The basis set is stored in [`BasisSet`](@ref) object.
  `type` can be `"ao"`, `"mpfit"` or `"jkfit"`.
  If `basisset` is provided, it is used as the basis set.
"""
function generate_basis(EC::AbstractECInfo, type="ao"; basisset::AbstractString="")
  return generate_basis(EC.system, type; EC.options.int.cartesian, basisset)
end

"""
    generate_basis(ms::AbstractSystem, type="ao"; cartesian=false, basisset::AbstractString="")

  Generate basis sets for integral calculations.

  The basis set is stored in [`BasisSet`](@ref) object.
  `type` can be `"ao"`, `"mpfit"` or `"jkfit"`.
  If `basisset` is provided, it is used as the basis set.
"""
function generate_basis(ms::AbstractSystem, type="ao"; cartesian=false, basisset::AbstractString="")
  array_of_centers = BasisCenter[]
  id = 1
  for atom in ms
    if basisset != ""
      basisname = basisset
    else
      basisname = basis_name(atom, type)
      if basisname == "unknown"
        basisname = guess_basis_name(atom, type)
      end
    end
    basisfunctions = parse_basis(basisname, atom; cartesian)
    id = set_id!(basisfunctions, id)
    push!(array_of_centers, BasisCenter(atom, basisname, basisfunctions))
  end
  return BasisSet(array_of_centers, ILibcint5(array_of_centers))
end

"""
    n_angularshells(atoms::BasisSet)

  Return the number of angular shells in the basis set.
"""
n_angularshells(atoms::BasisSet) = sum(n_angularshells, atoms.centers)
n_angularshells(atoms) = sum(n_angularshells, atoms)

"""
    n_subshells(atoms::BasisSet)

  Return the number of subshells in the basis set.
"""
n_subshells(atoms::BasisSet) = sum(n_subshells, atoms.centers)
n_subshells(atoms) = sum(n_subshells, atoms)

"""
    n_primitives(atoms::BasisSet)

  Return the number of primitives in the basis set.
"""
n_primitives(atoms::BasisSet) = sum(n_primitives, atoms.centers)
n_primitives(atoms) = sum(n_primitives, atoms)

"""
    n_ao(atoms::BasisSet)

  Return the number of atomic orbitals in the basis set.
"""
n_ao(atoms::BasisSet) = sum(n_ao, atoms.centers)
n_ao(atoms) = sum(n_ao, atoms)

"""
    guess_basis_name(atom::Atom, type)

  Guess the name of the basis set.
  `type` can be `"ao"`, `"mpfit"` or `"jkfit"`.
"""
function guess_basis_name(atom::Atom, type)
  if type == "ao"
    error("AO basis set for atom $(atomic_symbol(atom)) not defined!")
  end
  aobasis = basis_name(atom, "ao")
  return aobasis * "-" * type
end

"""
    guess_norb(EC::AbstractECInfo)

  Guess the number of orbitals in the system.
"""
function guess_norb(EC::AbstractECInfo)
  bao = generate_basis(EC, "ao")
  return n_ao(bao)
end

end # module