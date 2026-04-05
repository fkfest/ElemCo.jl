"""
    AbstractALPACAMatrix{T}

Abstract matrix-free interface for on-demand matrix access.

The type parameter `T` specifies the element type of the matrix
(e.g., `Float64`, `ComplexF64`).

Implementors must provide `Base.size(A)` together with:

- **Symmetric / Hermitian**: [`column!`](@ref) and [`elements!`](@ref).
  Rows are inferred from symmetry.
- **General**: [`column!`](@ref), [`row!`](@ref), and [`elements!`](@ref).

[`elements!`](@ref) is called once at initialization to fetch the principal
element values (e.g., the diagonal elements in the default case).  It only
needs to provide these principal elements, not arbitrary matrix entries.
The only exception is when using [`PrincipalTriples`](@ref) (pre-computed
values), which bypasses `elements!` entirely.
"""
abstract type AbstractALPACAMatrix{T} end

"""
    DenseALPACAMatrix(data)

Dense wrapper used for matrices that are already materialized in memory.  
This is the default for `alpaca(matrix)`.
"""
struct DenseALPACAMatrix{T,M<:AbstractMatrix{T}} <: AbstractALPACAMatrix{T}
  data::M
end

Base.size(mat::DenseALPACAMatrix) = size(mat.data)

"""
    column!(buffer, matrix, j)

Fill `buffer` with the `j`-th column of `matrix`.
"""
function column!(buffer::AbstractVector, matrix::AbstractALPACAMatrix, j::Integer)
  throw(MethodError(column!, (buffer, matrix, j)))
end

function column!(buffer::AbstractVector, mat::DenseALPACAMatrix, j::Integer)
  copyto!(buffer, view(mat.data, :, j))
  return buffer
end

"""
    row!(buffer, matrix, i)

Fill `buffer` with the `i`-th row of `matrix`.
"""
function row!(buffer::AbstractVector, matrix::AbstractALPACAMatrix, i::Integer)
  throw(MethodError(row!, (buffer, matrix, i)))
end

function row!(buffer::AbstractVector, mat::DenseALPACAMatrix, i::Integer)
  copyto!(buffer, view(mat.data, i, :))
  return buffer
end

"""
    elements!(buffer, matrix, pairs)

Fill `buffer[k]` with `matrix[i, j]` for `pairs[k] == (i, j)`.
"""
function elements!(buffer::AbstractVector, matrix::AbstractALPACAMatrix,
                   pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  throw(MethodError(elements!, (buffer, matrix, pairs)))
end

function elements!(buffer::AbstractVector, mat::DenseALPACAMatrix,
                   pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  @inbounds for index in eachindex(pairs)
    i, j = pairs[index]
    buffer[index] = mat.data[i, j]
  end
  return buffer
end

"""
    TransposedALPACAMatrix(parent)

Matrix-free wrapper that presents `parent` as its transpose:
columns of `TransposedALPACAMatrix` are rows of `parent` and vice versa.
Used internally by [`llama`](@ref) for column-guided decomposition via the
`d_col` keyword.
"""
struct TransposedALPACAMatrix{T, M<:AbstractALPACAMatrix{T}} <: AbstractALPACAMatrix{T}
  parent::M
end

Base.size(m::TransposedALPACAMatrix) = reverse(size(m.parent))

function column!(buffer::AbstractVector, m::TransposedALPACAMatrix, j::Integer)
  row!(buffer, m.parent, j)
end

function row!(buffer::AbstractVector, m::TransposedALPACAMatrix, i::Integer)
  column!(buffer, m.parent, i)
end

function elements!(buffer::AbstractVector, m::TransposedALPACAMatrix,
                   pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  swapped = [(j, i) for (i, j) in pairs]
  elements!(buffer, m.parent, swapped)
end

"""
    SymmetricALPACAMatrix(parent)

Matrix-free wrapper for symmetric matrices.  Only [`column!`](@ref) must be
implemented on `parent`; [`row!`](@ref) is derived by calling `column!`
(since ``A_{i,:} = A_{:,i}`` for symmetric matrices).
[`elements!`](@ref) delegates directly to `parent`.

See also [`HermitianALPACAMatrix`](@ref).
"""
struct SymmetricALPACAMatrix{T, M<:AbstractALPACAMatrix{T}} <: AbstractALPACAMatrix{T}
  parent::M
end

Base.size(m::SymmetricALPACAMatrix) = size(m.parent)

function column!(buffer::AbstractVector, m::SymmetricALPACAMatrix, j::Integer)
  column!(buffer, m.parent, j)
end

function row!(buffer::AbstractVector, m::SymmetricALPACAMatrix, i::Integer)
  column!(buffer, m.parent, i)
end

function elements!(buffer::AbstractVector, m::SymmetricALPACAMatrix,
                   pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  elements!(buffer, m.parent, pairs)
end

LinearAlgebra.issymmetric(::SymmetricALPACAMatrix) = true
LinearAlgebra.ishermitian(::SymmetricALPACAMatrix) = eltype(m) <: Real

"""
    HermitianALPACAMatrix(parent)

Matrix-free wrapper for Hermitian matrices.  Only [`column!`](@ref) must be
implemented on `parent`; [`row!`](@ref) is derived by calling `column!` and
conjugating the result (since ``A_{i,:} = \\overline{A_{:,i}}`` for Hermitian
matrices).  [`elements!`](@ref) delegates directly to `parent`.

See also [`SymmetricALPACAMatrix`](@ref).
"""
struct HermitianALPACAMatrix{T, M<:AbstractALPACAMatrix{T}} <: AbstractALPACAMatrix{T}
  parent::M
end

Base.size(m::HermitianALPACAMatrix) = size(m.parent)

function column!(buffer::AbstractVector, m::HermitianALPACAMatrix, j::Integer)
  column!(buffer, m.parent, j)
end

function row!(buffer::AbstractVector, m::HermitianALPACAMatrix, i::Integer)
  column!(buffer, m.parent, i)
  conj!(buffer)
  return buffer
end

function elements!(buffer::AbstractVector, m::HermitianALPACAMatrix,
                   pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  elements!(buffer, m.parent, pairs)
end

LinearAlgebra.issymmetric(m::HermitianALPACAMatrix) = eltype(m) <: Real
LinearAlgebra.ishermitian(::HermitianALPACAMatrix) = true

"""
    LinearAlgebra.issymmetric(matrix::AbstractALPACAMatrix) → Bool

Default implementation returning `false`.  Override for custom matrix types
that are known to be symmetric (e.g., integral matrices).
"""
LinearAlgebra.issymmetric(::AbstractALPACAMatrix) = false

"""
    LinearAlgebra.ishermitian(matrix::AbstractALPACAMatrix) → Bool

Default implementation returning `false`.  Override for custom matrix types
that are known to be Hermitian.
"""
LinearAlgebra.ishermitian(::AbstractALPACAMatrix) = false