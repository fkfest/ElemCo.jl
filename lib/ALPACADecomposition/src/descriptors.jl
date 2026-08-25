"""
    AbstractPrincipalDescriptor

Normalized description of principal matrix elements used as the secondary
pivot signal in ALPACA.
"""
abstract type AbstractPrincipalDescriptor end

"""
    PrincipalPairs{I<:Integer}

Principal descriptor containing only ``(i, j)`` index pairs.
The corresponding matrix elements are fetched via a single
[`elements!`](@ref) call at initialization time.
"""
struct PrincipalPairs{I<:Integer} <: AbstractPrincipalDescriptor
  pairs::Vector{Tuple{I,I}}
end

"""
    PrincipalTriples{I<:Integer,T}

Principal descriptor containing ``(i, j)`` index pairs together
with pre-computed element values of type `T`.  No element access call is
needed — values are copied directly into the cache.
"""
struct PrincipalTriples{I<:Integer,T} <: AbstractPrincipalDescriptor
  pairs::Vector{Tuple{I,I}}
  values::Vector{T}
end

"""
    principal_pairs(pairs) → PrincipalPairs

Construct a [`PrincipalPairs`](@ref) descriptor from index pairs.
"""
principal_pairs(pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}}) =
  PrincipalPairs(collect(pairs))

"""
    principal_triples(entries) → PrincipalPairs | PrincipalTriples

Construct a principal descriptor from a vector of entries.
If entries are `(i, j)` pairs, returns [`PrincipalPairs`](@ref).
If entries are `(i, j, value)` triples, returns [`PrincipalTriples`](@ref).
"""
function principal_triples(entries::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  return PrincipalPairs(collect(entries))
end

function principal_triples(entries::AbstractVector{<:Tuple{<:Integer,<:Integer,<:Number}})
  pairs = Vector{Tuple{Int,Int}}(undef, length(entries))
  values = Vector{promote_type(map(entry -> typeof(entry[3]), entries)...)}(undef, length(entries))
  @inbounds for index in eachindex(entries)
    i, j, value = entries[index]
    pairs[index] = (Int(i), Int(j))
    values[index] = value
  end
  return PrincipalTriples(pairs, values)
end

"""
    normalize_principal_descriptor(symmetry, n, descriptor) → AbstractPrincipalDescriptor

Convert a user-supplied principal descriptor into a canonical form.
Accepted inputs:
- `nothing`: generates diagonal pairs `(1,1), (2,2), ..., (n,n)`.
- `Vector{Tuple{Int,Int}}`: wrapped in [`PrincipalPairs`](@ref).
- `Vector{Tuple{Int,Int,T}}`: wrapped in [`PrincipalTriples`](@ref).
- An existing [`PrincipalPairs`](@ref) or [`PrincipalTriples`](@ref): returned as-is.
"""
function normalize_principal_descriptor(symmetry::Symbol, n::Integer,
                                        descriptor::Nothing)
  if n < 0
    throw(ArgumentError("matrix dimension must be non-negative"))
  end
  return PrincipalPairs([(i, i) for i in 1:n])
end

normalize_principal_descriptor(::Symbol, ::Integer,
                               descriptor::PrincipalPairs) = descriptor

normalize_principal_descriptor(::Symbol, ::Integer,
                               descriptor::PrincipalTriples) = descriptor

function normalize_principal_descriptor(symmetry::Symbol, n::Integer,
                                        descriptor::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  return principal_pairs(descriptor)
end

function normalize_principal_descriptor(symmetry::Symbol, n::Integer,
                                        descriptor::AbstractVector{<:Tuple{<:Integer,<:Integer,<:Number}})
  return principal_triples(descriptor)
end