"""
    BasisSets

  Module for working with basis sets.

  The basis set is stored in the [`BasisSet`](@ref) instance.
  The basis set can be generated using the [`generate_basis`](@ref) function.
"""
module BasisSets
using StaticArrays
using Printf
using DocStringExtensions
using ..ElemCo.Utils
using ..ElemCo.Elements
using ..ElemCo.MSystems
using ..ElemCo.AbstractEC

export BasisCentre, BasisSet
export BasisContraction, AngularShell
export shell_range, centre_range, is_cartesian, combine
export n_subshells, n_primitives, n_coefficients, n_angularshells, n_ao
export normalize_contraction
export coefficients_1mat, n_coefficients_1mat
export basis_name, generate_basis, guess_norb
export ao_list, print_ao
export subshell_char, max_l

export ILibcint5

include("basiscentre.jl")  
include("parse_basis.jl")
include("intlibs.jl")
include("aos.jl")

"""
    BasisSet

  A basis set with basis centres (atoms) and basis functions.

  $(TYPEDFIELDS)
"""
struct BasisSet
  """ array of basis centres (atoms) with basis functions."""
  centres::Vector{BasisCentre}
  """ indices for angular shells."""
  shell_indices::Vector{CartesianIndex{2}}
  """ centre ranges for each basis set in a combined set."""
  centre_ranges::Vector{UnitRange{Int}}
  """ angular shell ranges for each basis sets in a combined set."""
  shell_ranges::Vector{UnitRange{Int}}
  """ cartesian basis set """
  cartesian::Bool
  """ infos for integral library (at the moment only libcint5 is possible)."""
  lib::ILibcint5
end

function BasisSet(centres::Vector{BasisCentre}, cartesian::Bool, lib::AbstractILib)
  shell_indices = [CartesianIndex(i, j) for (i,c) in enumerate(centres) for j in 1:n_angularshells(c)]
  centre_ranges = [1:length(centres)]
  shell_ranges = [1:length(shell_indices)]
  return BasisSet(centres, shell_indices, centre_ranges, shell_ranges, cartesian, lib)
end

function BasisSet(centres::Vector{BasisCentre}, centre_ranges, shell_ranges, cartesian::Bool, lib::AbstractILib)
  shell_indices = [CartesianIndex(i, j) for (i,c) in enumerate(centres) for j in 1:n_angularshells(c)]
  return BasisSet(centres, shell_indices, centre_ranges, shell_ranges, cartesian, lib)
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
Base.getindex(bs::BasisSet, i::Int, j::Int) = bs.centres[i].shells[j]

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

  The centres are concatenated. 
  The centre/shell ranges (`centre_ranges`/`shell_ranges`) corresponding to the centres/shells 
  for each basis set   can be used to access the centres/shells in the combined basis set, e.g.
  `bs.centres[i] for i in bs.centre_ranges[1]` gives the centres of the first basis set 
  in the combined set.
"""
function combine(bs1::BasisSet, bs2::BasisSet)
  @assert bs1.cartesian == bs2.cartesian "Basis sets must be both cartesian or both spherical"
  centres = vcat(bs1.centres, bs2.centres)
  set_id!(centres, 1)
  centre_ranges = vcat(bs1.centre_ranges, [r .+ length(bs1.centres) for r in bs2.centre_ranges])
  shell_ranges = vcat(bs1.shell_ranges, [r .+ length(bs1.shell_indices) for r in bs2.shell_ranges])
  cartesian = bs1.cartesian && bs2.cartesian
  return BasisSet(centres, centre_ranges, shell_ranges, cartesian, ILibcint5(centres, cartesian))
end

function Base.show(io::IO, bs::BasisSet)
  for centre in bs.centres
    println(io, centre)
  end
end

"""
    shell_range(bs::BasisSet, i::Int=1)

  Return the range of angular shells for the `i`th basis set.

  The range is used to access the angular shells in the basis set, e.g.,
  `bs[i] for i in shell_range(bs, 1)` gives the angular shells of the first basis set.
"""
shell_range(bs::BasisSet, i::Int=1) = bs.shell_ranges[i]

"""
    centre_range(bs::BasisSet, i::Int=1)

  Return the range of centres for the `i`th basis set.

  The range is used to access the centres in the basis set, e.g.,
  `bs.centres[i] for i in centre_range(bs, 1)` gives the centres of the first basis set.
"""
centre_range(bs::BasisSet, i::Int=1) = bs.centre_ranges[i]

"""
    is_cartesian(bs::BasisSet)

  Check if the basis set is Cartesian.
"""
is_cartesian(bs::BasisSet) = bs.cartesian

"""
    basis_name(atom::ACentre, type="ao")

  Return the name of the basis set (or `unknown` if not found).
"""
function basis_name(atom::ACentre, type="ao")
  if haskey(atom.basis, type)
    return lowercase(atom.basis[type])
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
  return generate_basis(EC.system, type; cartesian=EC.options.int.cartesian, basisset=basisset)
end

"""
    generate_basis(ms::MSystem, type="ao"; cartesian=false, basisset::AbstractString="")

  Generate basis sets for integral calculations.

  The basis set is stored in [`BasisSet`](@ref) object.
  `type` can be `"ao"`, `"mpfit"` or `"jkfit"`.
  If `basisset` is provided, it is used as the basis set.
"""
function generate_basis(ms::MSystem, type="ao"; cartesian::Bool=false, basisset::AbstractString="")
  array_of_centres = BasisCentre[]
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
    basisfunctions = parse_basis(basisname, atom; fallback=(type=="jkfit"))
    id = set_id!(basisfunctions, id)
    push!(array_of_centres, BasisCentre(atom, basisname, basisfunctions))
  end
  return BasisSet(array_of_centres, cartesian, ILibcint5(array_of_centres, cartesian))
end

"""
    max_l(basis::BasisSet)

  Return the maximum angular momentum in the basis set.
"""
function max_l(basis::BasisSet)
  maxl = 0
  for ash in basis
    maxl = max(maxl, ash.l)
  end
  return maxl
end

"""
    n_angularshells(atoms::BasisSet)

  Return the number of angular shells in the basis set.
"""
n_angularshells(atoms::BasisSet) = sum(n_angularshells, atoms.centres)
n_angularshells(atoms) = sum(n_angularshells, atoms)

"""
    n_subshells(atoms::BasisSet)

  Return the number of subshells in the basis set.
"""
n_subshells(atoms::BasisSet) = sum(n_subshells, atoms.centres)
n_subshells(atoms) = sum(n_subshells, atoms)

"""
    n_primitives(atoms::BasisSet)

  Return the number of primitives in the basis set.
"""
n_primitives(atoms::BasisSet) = sum(n_primitives, atoms.centres)
n_primitives(atoms) = sum(n_primitives, atoms)

"""
    n_ao(atoms::BasisSet)

  Return the number of atomic orbitals in the basis set.
"""
n_ao(atoms::BasisSet) = sum(x->n_ao(x, atoms.cartesian), atoms.centres)
n_ao(atoms, cartesian) = sum(x->n_ao(x, cartesian), atoms)


"""
    ao_list(basis::BasisSet, ibas=1)

  Return the list of atomic orbitals in the basis set.

  For a combined basis set, use `ibas` to select the basis set.
"""
function ao_list(basis::BasisSet, ibas=1)
  if is_cartesian(basis)
    AtomicOrbital = CartesianAtomicOrbital
  else
    AtomicOrbital = SphericalAtomicOrbital
  end
  out = AtomicOrbital[]
  nnumber = zeros(Int, length(SUBSHELL2L))
  for ic in centre_range(basis, ibas)
    nnumber .= 0
    for ash in basis.centres[ic].shells
      for (isubshell, con) in enumerate(ash.subshells)
        nnumber[ash.l+1] += 1
        for iao in 1:n_ao4subshell(ash, basis.cartesian)
          ml = iao - 1 - ash.l
          push!(out, AtomicOrbital(ic, ash.id, isubshell, nnumber[ash.l+1], ash.l, ml))
        end
      end
    end
  end
  return out
end

"""
  print_ao(ao::AbstractAtomicOrbital, basis::BasisSet)

  Print the atomic orbital.
"""
function print_ao(ao::AbstractAtomicOrbital, basis::BasisSet)
  @assert ao.icentre <= length(basis.centres) "AO centre index out of range! Use the same basis in print_ao as in ao_list!" 
  icen = 0
  for r in basis.centre_ranges
    if ao.icentre in r
      icen = ao.icentre - r.start + 1
      break
    end
  end
  print(basis.centres[ao.icentre].name, "[", icen, "]", ao)
end

"""
    guess_basis_name(atom::ACentre, type)

  Guess the name of the basis set.
  `type` can be `"ao"`, `"mpfit"` or `"jkfit"`.
"""
function guess_basis_name(atom::ACentre, type)
  if type == "ao"
    error("AO basis set for atom $(atomic_centre_label(atom)) not defined!")
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