"""
This module contains functions for tensor decomposition methods.
"""
module DecompTools
using LinearAlgebra
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.FciDumps
using ..ElemCo.TensorTools
using ..ElemCo.QMTensors
using ..ElemCo.ALPACADecomposition

export calc_integrals_decomposition
export svd_decompose, svd_decompose_dense, svd_decompose_llama

"""
    IntegralMatrix{T} <: AbstractALPACAMatrix{T}

Matrix-free representation of the two-electron integral matrix for ALPACA decomposition.

Uses the raw integral array from `integ2_ss(EC.fd)` directly (mmapped, physicist
notation order, upper triangular storage for the last two indices):
`int2[p, q, tri(r,s)]` = ``v_{pq}^{rs}`` with ``r ≤ s``.

The matrix has compound indices ``I = (r-1) n + p`` and ``J = (s-1) n + q``
(column-major order), with elements ``M_{IJ} = v_{pq}^{rs} = ⟨pq|rs⟩``.

The matrix is symmetric: ``M^T = M`` (complex symmetric for complex MOs).
"""
struct IntegralMatrix{T} <: AbstractALPACAMatrix{T}
  "Two-electron integrals int2[p,q,tri(r,s)] in physicist notation, upper triangular"
  int2::Array{T,3}
  n::Int
end

Base.size(mat::IntegralMatrix) = (mat.n^2, mat.n^2)

function ALPACADecomposition.column!(buffer::AbstractVector, mat::IntegralMatrix, j::Integer)
  n = mat.n
  int2 = mat.int2
  q = ((j - 1) % n) + 1
  s = ((j - 1) ÷ n) + 1
  # buffer[I] = v_{pq}^{rs} where I = (r-1)*n + p
  # For r = 1:s, tri(r,s) = r + s*(s-1)÷2 are contiguous → batch copy
  tri_start = 1 + s * (s - 1) ÷ 2
  @views copyto!(reshape(buffer[1:s*n], n, s), int2[:, q, tri_start:tri_start+s-1])
  # For r > s: v_{pq}^{rs} = v_{qp}^{sr} = int2[q, p, tri(s,r)], use strided view copy
  @inbounds for r in s+1:n
    tri = s + r * (r - 1) ÷ 2
    off = (r - 1) * n
    @views copyto!(buffer[off+1:off+n], int2[q, :, tri])
  end
  return buffer
end

"""
    calc_integrals_decomposition(EC::ECInfo)

  Decompose ``v_{pr}^{qs}`` as ``v_p^{qL} v_r^{sL}`` and store as `mmL`.

  Uses the ALPACA algorithm with a matrix-free interface that accesses
  elements of the ``n^2 \\times n^2`` integral matrix on demand
  directly from the mmapped integral array, avoiding materialization
  of the full dense matrix.

  Diagonal elements ``⟨pq|pq⟩`` are pre-computed and passed as
  `ALPACADecomposition.PrincipalTriples` to avoid scattered I/O via `elements!`.
"""
function calc_integrals_decomposition(EC::ECInfo)
  int2 = integ2_ss(EC.fd)
  n = n_orbs(EC)
  tol = EC.options.cholesky.thr

  # Pre-compute diagonal elements M[I,I] = ⟨pp|rr⟩ where I = (r-1)*n + p
  n2 = n^2
  diag_pairs = Vector{Tuple{Int,Int}}(undef, n2)
  diag_values = Vector{eltype(int2)}(undef, n2)
  @inbounds for r in 1:n
    tri_rr = r * (r + 1) ÷ 2
    off = (r - 1) * n
    for p in 1:n
      I = off + p
      diag_pairs[I] = (I, I)
      diag_values[I] = int2[p, p, tri_rr]
    end
  end
  principal = PrincipalTriples(diag_pairs, diag_values)

  mat = IntegralMatrix(int2, n)
  opts = ALPACAOptions(tol=tol, symmetry=:symmetric)
  result = alpaca(mat; principal=principal, options=opts)

  naux1 = size(result.left, 2)
  if !isempty(result.neg_indices)
    @warn "ALPACA found $(length(result.neg_indices)) negative eigenvalues in integral matrix"
  end
  println("Integral auxiliary space size: ",naux1)
  save!(EC, "mmL", reshape(result.left, (n,n,naux1)))
end

"""
    svd_decompose_llama(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``
  using LLAMA low-rank approximation.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose_llama(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")
  result = llama(Amat; tol=tol)
  naux = length(result.singular_values)
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return reshape(result.Q, (nvirt,nocc,naux))
end

"""
    svd_decompose_llama(Amat, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``
  using LLAMA low-rank approximation.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` and ``Σ_X`` for ``Σ_X`` > `tol`
"""
function svd_decompose_llama(Amat, tol=1e-6; verbose=true, description="")
  result = llama(Amat; tol=tol)
  naux = length(result.singular_values)
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return result.Q, result.singular_values
end

"""
    svd_decompose_dense(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``
  using full dense SVD.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose_dense(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")
  U, S, = svd(Amat; full=false)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose_dense(Amat, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  using full dense SVD.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` and ``Σ_X`` for ``Σ_X`` > `tol`
"""
function svd_decompose_dense(Amat, tol=1e-6; verbose=true, description="")
  U, S, = svd(Amat; full=false)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return U[:,1:naux], S[1:naux]
end

"""
    svd_decompose(Amat, nvirt, nocc, tol=1e-6; method=:llama, verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`.

  `method` selects the algorithm: `:llama` (default) or `:dense`.
"""
function svd_decompose(Amat, nvirt, nocc, tol=1e-6; method=:llama, verbose=true, description="")
  if method == :llama
    return svd_decompose_llama(Amat, nvirt, nocc, tol; verbose, description)
  elseif method == :dense
    return svd_decompose_dense(Amat, nvirt, nocc, tol; verbose, description)
  else
    throw(ArgumentError("Unknown SVD method: $method. Use :llama or :dense."))
  end
end

"""
    svd_decompose(Amat, tol=1e-6; method=:llama, verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` and ``Σ_X`` for ``Σ_X`` > `tol`.

  `method` selects the algorithm: `:llama` (default) or `:dense`.
"""
function svd_decompose(Amat, tol=1e-6; method=:llama, verbose=true, description="")
  if method == :llama
    return svd_decompose_llama(Amat, tol; verbose, description)
  elseif method == :dense
    return svd_decompose_dense(Amat, tol; verbose, description)
  else
    throw(ArgumentError("Unknown SVD method: $method. Use :llama or :dense."))
  end
end

end #module
