"""
FCI vector implementation with orbital string addressing.
"""

"""
    OrbStringAdrTable{OPattern}

Auxiliary object for addressing orbital occupation patterns for one spin.
Provides addressing of all patterns with a fixed total number of electrons
in a given number of orbitals.
"""
mutable struct OrbStringAdrTable{OPattern}
  n_elec::FCIUInt
  n_orb::FCIUInt
  adr_count::Address
  str_table::Vector{OPattern}

  function OrbStringAdrTable{OPattern}() where OPattern
    new{OPattern}(0, 0, 0, OPattern[])
  end

  function OrbStringAdrTable{OPattern}(n_elec::Integer, n_orb::Integer) where OPattern
    table = new{OPattern}(0, 0, 0, OPattern[])
    init!(table, n_elec, n_orb)
    return table
  end
end

"""
    init!(table::OrbStringAdrTable{OPattern}, n_elec::Integer, n_orb::Integer) where OPattern

Initialize the orbital string addressing table.
"""
function init!(table::OrbStringAdrTable{OPattern}, n_elec::Integer, n_orb::Integer) where OPattern
  table.n_elec = FCIUInt(n_elec)
  table.n_orb = FCIUInt(n_orb)
  table.adr_count = sym_dof(n_orb, n_elec)

  make_string_table!(table)
end

"""
    make_string_table!(table::OrbStringAdrTable{OPattern}) where OPattern

Create the string table with all valid orbital patterns.
"""
function make_string_table!(table::OrbStringAdrTable{OPattern}) where OPattern
  sizehint!(table.str_table, table.adr_count)
  empty!(table.str_table)

  add_strings_to_table_recursive!(table, OPattern(0), Int(table.n_elec), 0)
  sort!(table.str_table)
end

"""
    add_strings_to_table_recursive!(table::OrbStringAdrTable{OPattern}, old_pat::OPattern, 
                                   n_elec_left::Integer, i_first_orb::Integer) where OPattern

Recursively add patterns with n_elec_left electrons in the remaining orbitals.
"""
function add_strings_to_table_recursive!(table::OrbStringAdrTable{OPattern}, old_pat::OPattern,
                                         n_elec_left::Integer, i_first_orb::Integer) where OPattern
  if n_elec_left == 0 && i_first_orb <= table.n_orb
    push!(table.str_table, old_pat)
    return
  end

  for i_orb in i_first_orb:(table.n_orb - 1)
    new_pat = old_pat | (OPattern(1) << i_orb)
    add_strings_to_table_recursive!(table, new_pat, n_elec_left - 1, i_orb + 1)
  end
end

"""
    (table::OrbStringAdrTable{OPattern})(bit_string::OPattern) where OPattern -> Address

Get 1-based address for a given bit string pattern.
"""
function (table::OrbStringAdrTable{OPattern})(bit_string::OPattern)::Address where OPattern
  idx = searchsortedfirst(table.str_table, bit_string)
  @assert idx <= length(table.str_table) && table.str_table[idx] == bit_string "Invalid bit string"
  return Address(idx)  # Return 1-based address
end

"""
    make_pattern(table::OrbStringAdrTable{OPattern}, adr::Address) where OPattern -> OPattern

Create orbital pattern from address.
"""
@inline function make_pattern(table::OrbStringAdrTable{OPattern}, adr)::OPattern where OPattern
  @boundscheck checkbounds(table.str_table, adr)
  return table.str_table[adr]
end

# Accessor functions
n_str(table::OrbStringAdrTable{OPattern}) where OPattern = table.adr_count
n_orb(table::OrbStringAdrTable{OPattern}) where OPattern = table.n_orb
n_elec(table::OrbStringAdrTable{OPattern}) where OPattern = table.n_elec

"""
    FCIVector{OPattern, T}

FCI vector storing coefficients as matrix M[iAdrA, iAdrB] where iAdrA and iAdrB 
are indices of orbital occupation strings for alpha/beta electrons.
"""
mutable struct FCIVector{OPattern, T}
  n_elec_a::FCIUInt
  n_elec_b::FCIUInt
  n_orb::FCIUInt
  n_str_a::Address
  n_str_b::Address
  adr_a::OrbStringAdrTable{OPattern}
  adr_b::OrbStringAdrTable{OPattern}
  is_spin_projected::Bool
  data::Matrix{T}

  function FCIVector{OPattern, T}(n_elec::Integer, n_orb::Integer, n_spin::Integer, is_spin_projected::Bool = false) where {OPattern, T}
    n_elec_a = (n_elec + n_spin) ÷ 2
    n_elec_b = (n_elec - n_spin) ÷ 2

    adr_a = OrbStringAdrTable{OPattern}(n_elec_a, n_orb)
    adr_b = OrbStringAdrTable{OPattern}(n_elec_b, n_orb)

    n_str_a = n_str(adr_a)
    n_str_b = n_str(adr_b)

    # Store data in [alpha, beta] order 
    data = zeros(T, n_str_a, n_str_b)

    new{OPattern, T}(
      FCIUInt(n_elec_a),
      FCIUInt(n_elec_b),
      FCIUInt(n_orb),
      n_str_a,
      n_str_b,
      adr_a,
      adr_b,
      is_spin_projected,
      data,
    )
  end
end

FCIVector{OPattern}(n_elec::Integer, n_orb::Integer, n_spin::Integer, is_spin_projected::Bool = false) where OPattern =
  FCIVector{OPattern, Float64}(n_elec, n_orb, n_spin, is_spin_projected)

"""
    Base.getindex(vec::FCIVector{OPattern}, i_a::Integer, i_b::Integer) where OPattern -> Scalar

Access coefficient vec[i_a, i_b].
"""
@inline function Base.getindex(vec::FCIVector, i_a::Integer, i_b::Integer)
  @boundscheck checkbounds(vec.data, i_a, i_b)
  return vec.data[i_a, i_b]
end

"""
    Base.setindex!(vec::FCIVector{OPattern}, val::Scalar, i_a::Integer, i_b::Integer) where OPattern

Set coefficient vec[i_a, i_b] = val.
"""
@inline function Base.setindex!(vec::FCIVector, val, i_a::Integer, i_b::Integer)
  @boundscheck checkbounds(vec.data, i_a, i_b)
  vec.data[i_a, i_b] = val
end

"""
    n_data(vec::FCIVector{OPattern}) where OPattern -> Int

Total number of data elements.
"""
n_data(vec::FCIVector) = Int(vec.n_str_a * vec.n_str_b)

"""
    n_spin(vec::FCIVector{OPattern}) where OPattern -> FCIUInt

Total spin quantum number.
"""
n_spin(vec::FCIVector) = vec.n_elec_a - vec.n_elec_b

"""
    clear!(vec::FCIVector{OPattern}) where OPattern

Set all coefficients to zero.
"""
function clear!(vec::FCIVector{OPattern, T}) where {OPattern, T}
  fill!(vec.data, zero(T))
end

"""
    Base.copy(vec::FCIVector{OPattern}) where OPattern -> FCIVector{OPattern}

Create a deep copy of an FCIVector, including all data.
"""
function Base.copy(vec::FCIVector{OPattern, T}) where {OPattern, T}
  new_vec = FCIVector{OPattern, T}(
    vec.n_elec_a + vec.n_elec_b,
    vec.n_orb,
    vec.n_elec_a - vec.n_elec_b,
    vec.is_spin_projected,
  )
  copy!(new_vec.data, vec.data)
  return new_vec
end

"""
    Base.zero(vec::FCIVector{OPattern}) where OPattern -> FCIVector{OPattern}

Create a zero FCIVector with the same dimensions.
"""
function Base.zero(vec::FCIVector{OPattern, T}) where {OPattern, T}
  new_vec = FCIVector{OPattern, T}(
    vec.n_elec_a + vec.n_elec_b,
    vec.n_orb,
    vec.n_elec_a - vec.n_elec_b,
    vec.is_spin_projected,
  )
  return new_vec
end

"""
    LinearAlgebra.normalize!(vec::FCIVector{OPattern}) where OPattern

Normalize the FCI vector.
"""
function LinearAlgebra.normalize!(vec::FCIVector)
  norm_val = norm(vec.data)
  if norm_val ≈ 0.0
    throw(ArgumentError("Attempted to normalize a non-normalizable vector"))
  end
  # Use broadcasting for efficiency
  vec.data .*= inv(norm_val)
end

"""
    LinearAlgebra.norm(vec::FCIVector{OPattern}) where OPattern -> Scalar

Compute norm of FCI vector using LinearAlgebra.
"""
function LinearAlgebra.norm(vec::FCIVector)
  return norm(vec.data)
end

"""
    compatible(a::FCIVector{OPattern}, b::FCIVector{OPattern}) where OPattern -> Bool

Check if two FCI vectors are compatible for operations.
"""
@inline function compatible(a::FCIVector, b::FCIVector)::Bool
  return (
    a.n_elec_a == b.n_elec_a &&
    a.n_elec_b == b.n_elec_b &&
    a.n_orb == b.n_orb &&
    a.n_str_a == b.n_str_a &&
    a.n_str_b == b.n_str_b
  )
end

"""
    LinearAlgebra.dot(a::FCIVector{OPattern}, b::FCIVector{OPattern}) where OPattern -> Scalar

Compute dot product using LinearAlgebra.
"""
function LinearAlgebra.dot(a::FCIVector, b::FCIVector)
  @assert compatible(a, b) "Vectors not compatible"
  return LinearAlgebra.dot(a.data, b.data)
end

"""
    add!(r::FCIVector{OPattern}, x::FCIVector{OPattern}, f::Scalar) where OPattern

Compute r += f * x using LinearAlgebra.
"""
function add!(r::FCIVector, x::FCIVector, f)
  @assert compatible(r, x) "Vectors not compatible"
  axpy!(f, x.data, r.data)
end

"""
    orthogonalize_against!(v::FCIVector{OPattern}, u::FCIVector{OPattern}) where OPattern

Remove component of v parallel to u: v = v - (u' * v) * u
Assumes u is normalized. Modifies v in-place.
This is the projection operator: (I - u*u^T) * v
"""
function orthogonalize_against!(v::FCIVector, u::FCIVector)
  projection = dot(u, v)
  add!(v, u, -projection)  # v = v + (-projection) * u
  return nothing
end

"""
    orthogonalize_against!(v::Vector{Scalar}, u::Vector{Scalar})

Vector version for P-space operations.
Remove component of v parallel to u: v = v - (u' * v) * u
Assumes u is normalized. Modifies v in-place.
"""
function orthogonalize_against!(v::AbstractVector, u::AbstractVector)
  projection = LinearAlgebra.dot(u, v)
  LinearAlgebra.axpy!(-projection, u, v)
  return nothing
end