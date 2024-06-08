
BVector = Vector{Float64}
# const MAXBSIZE = 40
# BVector = SVector{MAXBSIZE, Float64}


"""
    BasisContraction

  A basis contraction.
  `exprange` is the range of primitives (from exponents in the angular shell).
"""
struct BasisContraction
  exprange::UnitRange{Int}
  coefs::BVector
end

"""
    AbstractAngularShell

  Abstract type for angular shells, i.e, subshells with the same angular momentum.
  For general contracted basis sets, the angular shell is a collection of all subshells 
  with the same l quantum number. 
  For some other basis sets (e.g., the def2-family), the angular shell can be a
  single subshell with a specific l quantum number.
  See [`SphericalAngularShell`](@ref) and [`CartesianAngularShell`](@ref).
  `id` is the index of the angular shell in the basis set.
"""
abstract type AbstractAngularShell end

"""
    SphericalAngularShell

  Type for spherical angular shells, i.e, subshells with the same angular momentum.
  For general contracted basis sets, the angular shell is a collection of all subshells 
  with the same l quantum number. 
  For some other basis sets (e.g., the def2-family), the angular shell can be a
  single subshell with a specific l quantum number.
  `id` is the index of the angular shell in the basis set.

  $(TYPEDFIELDS)
"""
mutable struct SphericalAngularShell <: AbstractAngularShell
  """ element symbol (e.g., "H")"""
  element::String
  """ angular momentum"""
  l::Int
  """ array of exponents"""
  exponents::BVector
  """ array of subshells (contractions)"""
  subshells::Vector{BasisContraction}
  """ index of the angular shell in the basis set"""
  id::Int
end

"""
    CartesianAngularShell

  Type for cartesian angular shells, i.e, subshells with the same angular momentum.
  For general contracted basis sets, the angular shell is a collection of all subshells 
  with the same l quantum number. 
  For some other basis sets (e.g., the def2-family), the angular shell can be a
  single subshell with a specific l quantum number.
  `id` is the index of the angular shell in the basis set.

  $(TYPEDFIELDS)
"""
mutable struct CartesianAngularShell <: AbstractAngularShell
  """ element symbol (e.g., "H")"""
  element::String
  """ angular momentum"""
  l::Int
  """ array of exponents"""
  exponents::BVector
  """ array of subshells (contractions)"""
  subshells::Vector{BasisContraction}
  """ index of the angular shell in the basis set"""
  id::Int
end

"""
    BasisCenter

  A basis center (atom) with basis functions.

  $(TYPEDFIELDS)
"""
struct BasisCenter
  """ basis center name (e.g., "H1")"""
  name::String
  """ atomic position in Bohr (3D vector)"""
  position::SVector{3, Float64}
  """ atomic number"""
  atomic_number::Int
  """ name of the basis set (e.g., "cc-pVDZ")"""
  basis::String
  """ array of angular shells"""
  shells::Vector{AbstractAngularShell}
end

function BasisCenter(atom::Atom, basis="", basisfunctions=[])
  return BasisCenter(string(atomic_symbol(atom)), uconvert.(u"bohr", atom.position)/u"bohr", 
                    atom.atomic_number, basis, basisfunctions)
end

function Base.getindex(bc::BasisContraction, i::Int)
  return bc.coefs[i]
end

function Base.setindex!(bc::BasisContraction, val, i::Int)
  bc.coefs[i] = val
end

function Base.length(bc::BasisContraction)
  return length(bc.coefs)
end

function Base.getindex(ashell::AbstractAngularShell, i::Int)
  return ashell.subshells[i]
end

function Base.setindex!(ashell::AbstractAngularShell, val, i::Int)
  ashell.subshells[i] = val
end

function Base.length(ashell::AbstractAngularShell)
  return length(ashell.subshells)
end

function Base.getindex(bc::BasisCenter, i::Int)
  return bc.shells[i]
end

function Base.setindex!(bc::BasisCenter, val, i::Int)
  bc.shells[i] = val
end

function Base.length(bc::BasisCenter)
  return length(bc.shells)
end

function Base.iterate(bc::BasisCenter, state=1)
  if state > length(bc.shells)
    return nothing
  else
    return bc.shells[state], state+1
  end
end

function Base.show(io::IO, bc::BasisContraction)
  print(io, "c, ", bc.exprange.start, ".", bc.exprange.stop)
  for coef in bc.coefs
    print(io, ", ", coef)
  end
end

function Base.show(io::IO, ashell::AbstractAngularShell)
  print(io, subshell_char(ashell.l), ", ", ashell.element)
  for exp in ashell.exponents
    print(io, ", ", exp)
  end
  # print contractions
  for subshell in ashell.subshells
    print(io, "\n", subshell)
  end
end

function Base.show(io::IO, ashs::Vector{<:AbstractAngularShell})
  for ashell in ashs
    println(io, ashell)
  end
end

function Base.show(io::IO, bc::BasisCenter)
  println(io, "! ", bc.name, " (", bc.atomic_number, ") ", bc.basis, " at ", bc.position)
  for shell in bc.shells
    println(io, shell)
  end
end

angularshells(cen::BasisCenter) = cen.shells

"""
    n_subshells(ashell::AbstractAngularShell)

  Return the number of subshells in the angular shell.
"""
n_subshells(ashell::AbstractAngularShell) = length(ashell.subshells)

"""
    n_primitives(subshell::BasisContraction)

  Return the number of primitives for the subshell.
"""
n_primitives(subshell::BasisContraction) = length(subshell.exprange)

"""
    n_primitives(ashell::AbstractAngularShell)

  Return the total number of primitives in the angular shell.
"""
n_primitives(ashell::AbstractAngularShell) = length(ashell.exponents)

"""
    n_coefficients(subshell::BasisContraction)

  Return the number of coefficients for the subshell.
"""
n_coefficients(subshell::BasisContraction) = length(subshell.coefs)

"""
    n_coefficients(ashell::AbstractAngularShell)

  Return the total number of coefficients in the angular shell.
"""
n_coefficients(ashell::AbstractAngularShell) = sum(n_coefficients, ashell.subshells)

"""
    n_coefficients_1mat(ashell::AbstractAngularShell)

  Return the number of coefficients in the angular shell for a single contraction matrix.

  The missing coefficients will be set to zero.
"""
n_coefficients_1mat(ashell::AbstractAngularShell) = n_primitives(ashell) * n_subshells(ashell) 

"""
    n_ao4subshell(ashell::SphericalAngularShell)

  Return the number of atomic orbitals for the subshell.
"""
n_ao4subshell(ashell::SphericalAngularShell) = 2*ashell.l + 1

"""
    n_ao4subshell(ashell::CartesianAngularShell)

  Return the number of atomic orbitals for the subshell.
"""
n_ao4subshell(ashell::CartesianAngularShell) = (ashell.l + 1)*(ashell.l + 2) ÷ 2

"""
    n_ao(ashell::AbstractAngularShell)

  Return the number of atomic orbitals in the angular shell.
"""
n_ao(ashell::AbstractAngularShell) = n_ao4subshell(ashell) * n_subshells(ashell)

"""
    n_angularshells(atom::BasisCenter)

  Return the number of angular shells in the basis set for `atom`.
"""
function n_angularshells(atom::BasisCenter)
  return length(atom.shells)
end

"""
    n_subshells(atom::BasisCenter)

  Return the number of subshells in the basis set for `atom`.
"""
function n_subshells(atom::BasisCenter)
  return sum(n_subshells, atom.shells)
end

"""
    n_primitives(atom::BasisCenter)

  Return the number of primitives in the basis set for `atom`.
"""
function n_primitives(atom::BasisCenter)
  return sum(n_primitives, atom.shells)
end

"""
    n_ao(atom::BasisCenter)

  Return the number of atomic orbitals in the basis set for `atom`.
"""
function n_ao(atom::BasisCenter)
  return sum(n_ao, atom.shells)
end

"""
    coefficients_1mat(ashell::AbstractAngularShell)

  Return a single contraction matrix of the coefficients in the angular shell 
  (nprimitives × nsubshells). The contractions are normalized.

  The missing coefficients are set to zero in the matrix.
"""
function coefficients_1mat end

function coefficients_1mat(ashell::CartesianAngularShell)
  coefs = zeros(n_primitives(ashell), n_subshells(ashell))
  for (i, subshell) in enumerate(ashell.subshells)
    coefs[subshell.exprange, i] .= normalize_cartesian_contraction(subshell.coefs, ashell.exponents[subshell.exprange], ashell.l)
  end
  return coefs
end

function coefficients_1mat(ashell::SphericalAngularShell)
  coefs = zeros(n_primitives(ashell), n_subshells(ashell))
  for (i, subshell) in enumerate(ashell.subshells)
    coefs[subshell.exprange, i] .= normalize_spherical_contraction(subshell.coefs, ashell.exponents[subshell.exprange], ashell.l)
  end
  return coefs
end

# DOUBLEFACTORIAL[l+1] = (2l+1)!! = 1*3*5*...*(2l+1) for s, p, d, f, g, h, i, k, l
const DOUBLEFACTORIAL = [1, 3, 15, 105, 945, 10395, 135135, 2027025, 34459425]

"""
    normalize_contraction(subshell::BasisContraction, ashell::AbstractAngularShell)

  Normalize the contraction coefficients in `subshell`. 
  
  The subshell has to be part of the angular shell `ashell`.
  Return the normalized contraction.
"""
function normalize_contraction(subshell::BasisContraction, ashell::AbstractAngularShell)
  if isa(ashell, CartesianAngularShell)
    return normalize_cartesian_contraction(subshell.coefs, ashell.exponents[subshell.exprange], ashell.l)
  else
    return normalize_spherical_contraction(subshell.coefs, ashell.exponents[subshell.exprange], ashell.l)
  end
end

"""
    normalize_spherical_contraction(contraction, exponents, l)

  Normalize the spherical subshell.

  Return the normalized contraction.
"""
function normalize_spherical_contraction(contraction, exponents, l)
  contra = map((c,e) -> c * e^((2l+3)/4), contraction, exponents)
  norm = 0.0
  for i in eachindex(contra)
    ci, ei = contra[i], exponents[i]
    for j in eachindex(contra)
      cj, ej = contra[j], exponents[j]
      norm += ci*cj/sqrt((ei+ej)^(2l+3))
    end
  end
  norm *= sqrt(π) * DOUBLEFACTORIAL[l+1]/2^(l+2)
  return contra/sqrt(norm)
end
# function normalize_spherical_contraction(contraction, exponents, l)
#   ff = factorial(l+1) / factorial(2*l+2)
#   return map((c,e) -> c * sqrt(sqrt((8*e)^(2*l+3)/π) * ff), contraction, exponents)
# end

"""
    normalize_cartesian_contraction(contraction, exponents, l)

  Normalize the subshell.

  Return the normalized contraction.
"""
function normalize_cartesian_contraction(contraction, exponents, l)
  contra = map((c,e) -> c * e^((2l+3)/4), contraction, exponents)
  norm = 0.0
  for i in eachindex(contra)
    ci, ei = contra[i], exponents[i]
    for j in eachindex(contra)
      cj, ej = contra[j], exponents[j]
      norm += ci*cj/sqrt((ei+ej)^(2l+3))
    end
  end
  if l < 2
    norm *= sqrt(π) * DOUBLEFACTORIAL[l+1]/2^(l+2) 
  else
    norm *= π^(3/2) * DOUBLEFACTORIAL[l]/2^l 
  end
  return contra/sqrt(norm)
end

"""
    generate_angularshell(elem, l, exponents; cartesian=false)

  Generate an angular shell with angular momentum `l` and exponents.
  The contractions have to be added later.
  Return an angular shell of type [`SphericalAngularShell`](@ref) or [`CartesianAngularShell`](@ref).
"""
function generate_angularshell(elem, l, exponents; cartesian=false)
  if cartesian
    return CartesianAngularShell(elem, l, exponents, BasisContraction[], 0)
  else
    return SphericalAngularShell(elem, l, exponents, BasisContraction[], 0)
  end
end

"""
    add_subshell!(ashell::AbstractAngularShell, exprange, contraction)

  Add a subshell to the angular shell.
"""
function add_subshell!(ashell::AbstractAngularShell, exprange, contraction)
  push!(ashell.subshells, BasisContraction(exprange, BVector(contraction)))
end

function set_id!(ashell::AbstractAngularShell, id)
  ashell.id = id
end

"""
    set_id!(ashells::AbstractArray{AbstractAngularShell}, start_id)

  Set the id for each angular shell in the array.
  Return the next id.
"""
function set_id!(ashells::AbstractArray{AbstractAngularShell}, start_id)
  id = start_id
  for ashell in ashells
    set_id!(ashell, id)
    id += 1
  end
  return id
end

"""
    set_id!(centers::AbstractArray{BasisCenter}, start_id)

  Set the id for each angular shell in the array of centers.
  Return the next id.
"""
function set_id!(centers::AbstractArray{BasisCenter}, start_id)
  id = start_id
  for center in centers
    id = set_id!(center.shells, id)
  end
  return id
end

"""
    subshell_char(l)

  Return the character for the subshell with angular momentum `l`.
"""
subshell_char(l::Int) = SUBSHELLS_NAMES[l+1]