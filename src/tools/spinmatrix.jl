"""
    SpinMatrix

A type to store a one-electron matrix (spin aware).

The first matrix corresponds to the alpha electron, and the second matrix is beta. 
If the matrix is restricted, the beta matrix refers to the alpha matrix.
"""
mutable struct SpinMatrix{T<:Number}
  α::Matrix{T}
  β::Matrix{T}
  function SpinMatrix(mat1::AbstractMatrix{Ty}, mat2::AbstractMatrix{Ty}) where {Ty}
    return new{Ty}(mat1, mat2)
  end
  function SpinMatrix(mat::AbstractMatrix{Ty}) where {Ty}
    return new{Ty}(mat, mat)
  end
  function SpinMatrix{Ty}() where {Ty}
    return new{Ty}(zeros(Ty,0,0), zeros(Ty,0,0))
  end
  function SpinMatrix(mats::Tuple{Matrix{Ty}, Matrix{Ty}}) where {Ty}
    return new{Ty}(mats[1], mats[2])
  end
end

const FSpinMatrix = SpinMatrix{Float64}
const CSpinMatrix = SpinMatrix{ComplexF64}


function Base.length(mat::SpinMatrix) 
  @assert length(mat.α) == length(mat.β)
  length(mat.α)
end

function Base.size(mat::SpinMatrix) 
  @assert size(mat.α) == size(mat.β)
  size(mat.α)
end

function Base.size(mat::SpinMatrix, i::Int) 
  @assert size(mat.α, i) == size(mat.β, i)
  size(mat.α, i)
end

Base.getindex(mat::SpinMatrix, spincase::Symbol) = getfield(mat, spincase)

Base.getindex(mat::SpinMatrix, i::Int) = getfield(mat, i)

function Base.setindex!(smat::SpinMatrix{T}, mat::AbstractMatrix{T}, spincase::Symbol) where {T}
  if mat isa Matrix{T}
    return setfield!(smat, spincase, mat)
  else
    return setfield!(smat, spincase, copy(mat))
  end
end

function Base.setindex!(smat::SpinMatrix{T}, mat::AbstractMatrix{T}, i::Int) where {T}
  if mat isa Matrix{T}
    setfield!(smat, i, mat)
  else
    setfield!(smat, i, copy(mat))
  end
end

function Base.getindex(mat::SpinMatrix, I, J) 
  if is_restricted(mat) 
    return SpinMatrix(mat.α[I,J])
  else
    return SpinMatrix(mat.α[I,J], mat.β[I,J])
  end
end

function Base.setindex!(mat::SpinMatrix, val::SpinMatrix, I, J)
  setindex!(mat.α, val.α, I, J)
  if !is_restricted(mat) || !is_restricted(val)
    setindex!(mat.β, val.β, I, J)
  end
end

function Base.setindex!(mat::SpinMatrix, val, I, J)
  setindex!(mat.α, val, I, J)
  if !is_restricted(mat)
    setindex!(mat.β, val, I, J)
  end
end

function Base.axes(mat::SpinMatrix) 
  @assert axes(mat.α) == axes(mat.β)
  axes(mat.α)
end

function Base.axes(mat::SpinMatrix, i::Int) 
  @assert axes(mat.α, i) == axes(mat.β, i)
  axes(mat.α, i)
end

Base.iterate(mat::SpinMatrix, state=1) = state > 2 ? nothing : (mat[state], state+1) 

Base.eltype(mat::SpinMatrix) = eltype(mat.α)

Base.copy(mat::SpinMatrix) = SpinMatrix(copy(mat.α), copy(mat.β))

Base.copy!(mat::SpinMatrix, mat2::SpinMatrix) = (copy!(mat.α, mat2.α); copy!(mat.β, mat2.β))

function Base.show(io::IO, mat::SpinMatrix) 
  println(io, "SpinMatrix{$(eltype(mat))}")
  if is_restricted(mat)
    show(io, mat.α)
  else
    println(io, "Alpha:")
    show(io, mat.α)
    println(io, "\nBeta:")
    show(io, mat.β)
  end
end

function Base.show(io::IO, mime::MIME"text/plain", mat::SpinMatrix)
  println(io, "SpinMatrix{$(eltype(mat))}")
  if is_restricted(mat)
    show(io, mime, mat.α)
  else
    println(io, "Alpha:")
    show(io, mime, mat.α)
    println(io, "\nBeta:")
    show(io, mime, mat.β)
  end
end

Base.zero(mat::SpinMatrix) = is_restricted(mat) ? SpinMatrix(zero(mat.α)) : SpinMatrix(zero(mat.α), zero(mat.β))

Base.one(mat::SpinMatrix) = is_restricted(mat) ? SpinMatrix(one(mat.α)) : SpinMatrix(one(mat.α), one(mat.β))

Base.keys(mat::SpinMatrix) = (1, 2)

Base.values(mat::SpinMatrix) = (mat.α, mat.β)

Base.Tuple(mat::SpinMatrix) = (mat.α, mat.β)

"""
    is_restricted(mat::SpinMatrix)

Check if the spin-matrix is restricted (i.e, β === α).
"""
is_restricted(mat::SpinMatrix) = mat.α === mat.β

"""
    unrestrict!(mat::SpinMatrix)

Unrestrict the spin matrix.
"""
function unrestrict!(mat::SpinMatrix)
  if is_restricted(mat)
    mat.β = copy(mat.α)
  end
  return mat
end

"""
    restrict!(mat::SpinMatrix)

Restrict the molecular orbitals (β = α).
"""
function restrict!(mat::SpinMatrix)
  mat.β = mat.α
  return mat
end