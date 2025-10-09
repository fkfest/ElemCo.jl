"""
FCI vector implementation with orbital string addressing.
"""

"""
    OrbStringAdrTable

Auxiliary object for addressing orbital occupation patterns for one spin.
Provides addressing of all patterns with a fixed total number of electrons
in a given number of orbitals.
"""
mutable struct OrbStringAdrTable
  n_elec::FCIUInt
  n_orb::FCIUInt
  adr_count::Address
  str_table::Vector{OrbPattern}

  function OrbStringAdrTable()
    new(0, 0, 0, OrbPattern[])
  end

  function OrbStringAdrTable(n_elec::Integer, n_orb::Integer)
    table = new(0, 0, 0, OrbPattern[])
    init!(table, n_elec, n_orb)
    return table
  end
end

"""
    init!(table::OrbStringAdrTable, n_elec::Integer, n_orb::Integer)

Initialize the orbital string addressing table.
"""
function init!(table::OrbStringAdrTable, n_elec::Integer, n_orb::Integer)
  table.n_elec = FCIUInt(n_elec)
  table.n_orb = FCIUInt(n_orb)
  table.adr_count = sym_dof(n_orb, n_elec)

  make_string_table!(table)
end

"""
    make_string_table!(table::OrbStringAdrTable)

Create the string table with all valid orbital patterns.
"""
function make_string_table!(table::OrbStringAdrTable)
  sizehint!(table.str_table, table.adr_count)
  empty!(table.str_table)

  add_strings_to_table_recursive!(table, OrbPattern(0), Int(table.n_elec), 0)
  sort!(table.str_table)
end

"""
    add_strings_to_table_recursive!(table::OrbStringAdrTable, old_pat::OrbPattern, 
                                   n_elec_left::Integer, i_first_orb::Integer)

Recursively add patterns with n_elec_left electrons in the remaining orbitals.
"""
function add_strings_to_table_recursive!(table::OrbStringAdrTable, old_pat::OrbPattern,
                                         n_elec_left::Integer, i_first_orb::Integer)
  if n_elec_left == 0 && i_first_orb <= table.n_orb
    push!(table.str_table, old_pat)
    return
  end

  for i_orb in i_first_orb:(table.n_orb - 1)
    new_pat = old_pat | (OrbPattern(1) << i_orb)
    add_strings_to_table_recursive!(table, new_pat, n_elec_left - 1, i_orb + 1)
  end
end

"""
    (table::OrbStringAdrTable)(bit_string::OrbPattern) -> Address

Get 1-based address for a given bit string pattern.
"""
function (table::OrbStringAdrTable)(bit_string::OrbPattern)::Address
  idx = searchsortedfirst(table.str_table, bit_string)
  @assert idx <= length(table.str_table) && table.str_table[idx] == bit_string "Invalid bit string"
  return Address(idx)  # Return 1-based address
end

"""
    make_pattern(table::OrbStringAdrTable, adr::Address) -> OrbPattern

Create orbital pattern from address.
"""
@inline function make_pattern(table::OrbStringAdrTable, adr::Address)::OrbPattern
  @boundscheck checkbounds(table.str_table, adr)
  return table.str_table[adr]
end

# Accessor functions
n_str(table::OrbStringAdrTable) = table.adr_count
n_orb(table::OrbStringAdrTable) = table.n_orb
n_elec(table::OrbStringAdrTable) = table.n_elec

"""
    FCIVector

FCI vector storing coefficients as matrix M[iAdrA, iAdrB] where iAdrA and iAdrB 
are indices of orbital occupation strings for alpha/beta electrons.
"""
mutable struct FCIVector
  n_elec_a::FCIUInt
  n_elec_b::FCIUInt
  n_orb::FCIUInt
  n_str_a::Address
  n_str_b::Address
  adr_a::OrbStringAdrTable
  adr_b::OrbStringAdrTable
  is_spin_projected::Bool
  data::Matrix{Scalar}

  function FCIVector(n_elec::Integer, n_orb::Integer, n_spin::Integer, is_spin_projected::Bool = false)
    n_elec_a = (n_elec + n_spin) ÷ 2
    n_elec_b = (n_elec - n_spin) ÷ 2

    adr_a = OrbStringAdrTable(n_elec_a, n_orb)
    adr_b = OrbStringAdrTable(n_elec_b, n_orb)

    n_str_a = n_str(adr_a)
    n_str_b = n_str(adr_b)

    # Store data in [alpha, beta] order 
    data = zeros(Scalar, n_str_a, n_str_b)

    new(
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

"""
    Base.getindex(vec::FCIVector, i_a::Integer, i_b::Integer) -> Scalar

Access coefficient vec[i_a, i_b].
"""
@inline function Base.getindex(vec::FCIVector, i_a::Integer, i_b::Integer)::Scalar
  @boundscheck checkbounds(vec.data, i_a, i_b)
  return vec.data[i_a, i_b]
end

"""
    Base.setindex!(vec::FCIVector, val::Scalar, i_a::Integer, i_b::Integer)

Set coefficient vec[i_a, i_b] = val.
"""
@inline function Base.setindex!(vec::FCIVector, val::Scalar, i_a::Integer, i_b::Integer)
  @boundscheck checkbounds(vec.data, i_a, i_b)
  vec.data[i_a, i_b] = val
end

"""
    n_data(vec::FCIVector) -> Int

Total number of data elements.
"""
n_data(vec::FCIVector) = Int(vec.n_str_a * vec.n_str_b)

"""
    n_spin(vec::FCIVector) -> FCIUInt

Total spin quantum number.
"""
n_spin(vec::FCIVector) = vec.n_elec_a - vec.n_elec_b

"""
    clear!(vec::FCIVector)

Set all coefficients to zero.
"""
function clear!(vec::FCIVector)
  fill!(vec.data, zero(Scalar))
end

"""
    Base.copy(vec::FCIVector) -> FCIVector

Create a deep copy of an FCIVector, including all data.
"""
function Base.copy(vec::FCIVector)
  new_vec = FCIVector(
    vec.n_elec_a + vec.n_elec_b,
    vec.n_orb,
    vec.n_elec_a - vec.n_elec_b,
    vec.is_spin_projected,
  )
  copy!(new_vec.data, vec.data)
  return new_vec
end

"""
    Base.zero(vec::FCIVector) -> FCIVector

Create a zero FCIVector with the same dimensions.
"""
function Base.zero(vec::FCIVector)
  new_vec = FCIVector(
    vec.n_elec_a + vec.n_elec_b,
    vec.n_orb,
    vec.n_elec_a - vec.n_elec_b,
    vec.is_spin_projected,
  )
  fill!(new_vec.data, zero(Scalar))
  return new_vec
end

"""
    LinearAlgebra.normalize!(vec::FCIVector)

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
    LinearAlgebra.norm(vec::FCIVector) -> Scalar

Compute norm of FCI vector using LinearAlgebra.
"""
function LinearAlgebra.norm(vec::FCIVector)::Scalar
  return norm(vec.data)
end

"""
    compatible(a::FCIVector, b::FCIVector) -> Bool

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
    LinearAlgebra.dot(a::FCIVector, b::FCIVector) -> Scalar

Compute dot product using LinearAlgebra.
"""
function LinearAlgebra.dot(a::FCIVector, b::FCIVector)::Scalar
  @assert compatible(a, b) "Vectors not compatible"
  return LinearAlgebra.dot(a.data, b.data)
end

"""
    add!(r::FCIVector, x::FCIVector, f::Scalar)

Compute r += f * x using LinearAlgebra.
"""
function add!(r::FCIVector, x::FCIVector, f::Scalar)
  @assert compatible(r, x) "Vectors not compatible"
  axpy!(f, x.data, r.data)
end

"""
    orthogonalize_against!(v::FCIVector, u::FCIVector)

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
function orthogonalize_against!(v::Vector{Scalar}, u::Vector{Scalar})
  projection = LinearAlgebra.dot(u, v)
  LinearAlgebra.axpy!(-projection, u, v)
  return nothing
end