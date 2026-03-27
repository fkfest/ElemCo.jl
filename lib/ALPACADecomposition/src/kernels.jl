"""
Matrix-class-specific fetch, deflation, principal update, and Nyström finalization kernels.

Kernels that distinguish Hermitian from symmetric are dispatched via
the compile-time symmetry type parameter `S` on [`ALPACACache`](@ref),
so there is no runtime cost for the distinction.
"""

# ──────────────────────────────────────────────────────────────────
# Helpers that specialize on matrix type (dense vs matrix-free)
# ──────────────────────────────────────────────────────────────────

"""
    _store_orig_col!(cache, matrix)

Store the current column buffer (`cache.cbuf`) into `orig_columns`.
No-op for `DenseALPACAMatrix` (columns can be read from the matrix directly).
"""
@inline _store_orig_col!(cache::ALPACACache, ::DenseALPACAMatrix) = nothing
@inline function _store_orig_col!(cache::ALPACACache, ::AbstractALPACAMatrix)
  @views cache.orig_columns[:, cache.n_cols + 1] .= cache.cbuf
end

"""
    _store_orig_row!(cache, matrix)

Store the current row buffer (`cache.rbuf`) into `orig_rows`.
No-op for `DenseALPACAMatrix`.
"""
@inline _store_orig_row!(cache::ALPACACache, ::DenseALPACAMatrix) = nothing
@inline function _store_orig_row!(cache::ALPACACache, ::AbstractALPACAMatrix)
  @views cache.orig_rows[:, cache.n_rows + 1] .= cache.rbuf
end

"""
    _get_pivot_columns(matrix, cache)

Return the undeflated pivot columns `M[:, pivots]`.
For `DenseALPACAMatrix`, reads directly from the matrix data.
For other matrices, reads from the stored `orig_columns` in the cache.
"""
function _get_pivot_columns(matrix::DenseALPACAMatrix, cache::ALPACACache)
  k = cache.n_cols
  pivots = @view cache.pivot_indices[1:k]
  return matrix.data[:, pivots]
end

function _get_pivot_columns(::AbstractALPACAMatrix, cache::ALPACACache)
  return cache.orig_columns[:, 1:cache.n_cols]
end

"""
    _get_pivot_rows_T(matrix, cache)

Return the transposed pivot rows `RT` where `RT[:, t] = M[row_pivots[t], :]`.
For `DenseALPACAMatrix`, reads directly from the matrix data.
For other matrices, reads from the stored `orig_rows` in the cache.
"""
function _get_pivot_rows_T(matrix::DenseALPACAMatrix, cache::ALPACACache)
  k = cache.n_rows
  row_pivots = @view cache.row_pivot_indices[1:k]
  return permutedims(matrix.data[row_pivots, :])
end

function _get_pivot_rows_T(::AbstractALPACAMatrix, cache::ALPACACache)
  return cache.orig_rows[:, 1:cache.n_rows]
end

# ──────────────────────────────────────────────────────────────────
# Fetch + deflation: symmetric / Hermitian
# ──────────────────────────────────────────────────────────────────

"""
    fetch_and_deflate_symmetric!(cache, matrix, j)

Fetch column `j` from `matrix`, deflate by subtracting all previous pivots
using BLAS-2 gemv, and store the deflated column in the cache.

The symmetry type parameter `S` on the cache controls whether conjugation
is applied (hermitian) or not (symmetric).
Returns the deflated column (a view into `cache.cbuf`).
"""
function fetch_and_deflate_symmetric!(cache::ALPACACache{T,R,:hermitian},
                                      matrix::AbstractALPACAMatrix,
                                      j::Int) where {T,R}
  column!(cache.cbuf, matrix, j)
  _ensure_col_capacity!(cache, cache.n_cols + 1)
  _store_orig_col!(cache, matrix)
  k = cache.n_cols
  if k > 0
    L_view = @view cache.columns[:, 1:k]
    c_view = @view cache.coeffs[1:k]
    @inbounds for t in 1:k
      c_view[t] = cache.pivot_diag[t] * conj(cache.columns[j, t])
    end
    mul!(cache.cbuf, L_view, c_view, -one(T), one(T))
  end
  return cache.cbuf
end

function fetch_and_deflate_symmetric!(cache::ALPACACache{T},
                                      matrix::AbstractALPACAMatrix,
                                      j::Int) where T
  column!(cache.cbuf, matrix, j)
  _ensure_col_capacity!(cache, cache.n_cols + 1)
  _store_orig_col!(cache, matrix)
  k = cache.n_cols
  if k > 0
    L_view = @view cache.columns[:, 1:k]
    c_view = @view cache.coeffs[1:k]
    @inbounds for t in 1:k
      c_view[t] = cache.pivot_diag[t] * cache.columns[j, t]
    end
    mul!(cache.cbuf, L_view, c_view, -one(T), one(T))
  end
  return cache.cbuf
end

# ──────────────────────────────────────────────────────────────────
# Fetch + deflation: general (separate column / row)
# ──────────────────────────────────────────────────────────────────

"""
    fetch_and_deflate_col_general!(cache, matrix, j)

Fetch column `j` from `matrix` and deflate using all previously stored
pivot (column, row) pairs.  Result is in `cache.cbuf`.
"""
function fetch_and_deflate_col_general!(cache::ALPACACache{T},
                                        matrix::AbstractALPACAMatrix,
                                        j::Int) where T
  column!(cache.cbuf, matrix, j)
  _ensure_col_capacity!(cache, cache.n_cols + 1)
  _store_orig_col!(cache, matrix)
  k = cache.n_cols
  if k > 0
    col_view = @view cache.columns[:, 1:k]
    c_view = @view cache.coeffs[1:k]
    @inbounds for t in 1:k
      c_view[t] = cache.pivot_diag[t] * cache.rows[j, t]
    end
    mul!(cache.cbuf, col_view, c_view, -one(T), one(T))
  end
  return cache.cbuf
end

"""
    fetch_and_deflate_row_general!(cache, matrix, i)

Fetch row `i` from `matrix` and deflate using all previously stored
pivot (column, row) pairs.  Result is in `cache.rbuf`.
"""
function fetch_and_deflate_row_general!(cache::ALPACACache{T},
                                        matrix::AbstractALPACAMatrix,
                                        i::Int) where T
  row!(cache.rbuf, matrix, i)
  _ensure_row_capacity!(cache, cache.n_rows + 1)
  _store_orig_row!(cache, matrix)
  k = cache.n_cols
  if k > 0
    row_view = @view cache.rows[:, 1:k]
    c_view = @view cache.coeffs[1:k]
    @inbounds for t in 1:k
      c_view[t] = cache.pivot_diag[t] * cache.columns[i, t]
    end
    mul!(cache.rbuf, row_view, c_view, -one(T), one(T))
  end
  return cache.rbuf
end

# ──────────────────────────────────────────────────────────────────
# Principal residual update
# ──────────────────────────────────────────────────────────────────

"""
    update_principal_residuals!(cache, deflated_col, pivot_val)

Update principal residual values after accepting a pivot with value `pivot_val`.

For each principal pair (a, b):
  val_res[p] -= deflated_col[a] * deflated_col[b] / pivot_val   (symmetric)
  val_res[p] -= deflated_col[a] * conj(deflated_col[b]) / pivot_val  (hermitian)

The hermitian vs symmetric distinction is dispatched via the
cache type parameter `S`.
"""
function update_principal_residuals!(cache::ALPACACache{T,R,:hermitian},
                                     deflated_col::AbstractVector{T},
                                     pivot_val::T) where {T,R}
  @inbounds for p in eachindex(cache.principal_pairs)
    a, b = cache.principal_pairs[p]
    cache.principal_values[p] -= deflated_col[a] * conj(deflated_col[b]) * pivot_val
  end
  return
end

function update_principal_residuals!(cache::ALPACACache{T},
                                     deflated_col::AbstractVector{T},
                                     pivot_val::T) where T
  @inbounds for p in eachindex(cache.principal_pairs)
    a, b = cache.principal_pairs[p]
    cache.principal_values[p] -= deflated_col[a] * deflated_col[b] * pivot_val
  end
  return
end

"""
    update_principal_residuals_general!(cache, deflated_col, deflated_row, pivot_val)

Update principal residuals for general matrices.
  val_res[p] -= deflated_col[a] * deflated_row[b] / pivot_val
"""
function update_principal_residuals_general!(cache::ALPACACache{T},
                                             deflated_col::AbstractVector{T},
                                             deflated_row::AbstractVector{T},
                                             pivot_val::T) where T
  @inbounds for p in eachindex(cache.principal_pairs)
    a, b = cache.principal_pairs[p]
    cache.principal_values[p] -= deflated_col[a] * deflated_row[b] * pivot_val
  end
  return
end

# ──────────────────────────────────────────────────────────────────
# Nyström finalization (symmetric / Hermitian / complex symmetric)
# ──────────────────────────────────────────────────────────────────
# Pivot-loop finalization: construct result directly from cache factors
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────

"""
    nystrom_finalize(cache::ALPACACache, matrix, tol)

Compute Nyström decomposition vectors from stored original (undeflated)
columns in the cache.

For symmetric (real or complex): eigendecomposition / SVD of the pivot block.
For Hermitian: eigendecomposition of the pivot block.

Returns `ALPACAResult{T}`.
"""
function nystrom_finalize(cache::ALPACACache{T,R,S}, matrix::AbstractALPACAMatrix, tol) where {T,R,S}
  k = cache.n_cols
  n = length(cache.cbuf)
  pivots = cache.pivot_indices

  if k == 0
    return ALPACAResult{T}(
      Matrix{T}(undef, n, 0),
      Matrix{T}(undef, n, 0),
      Int[], Int[], Int[], S)
  end

  # C = M[:, pivots]
  C = _get_pivot_columns(matrix, cache)

  # J = M[pivots, pivots]
  J = Matrix{T}(undef, k, k)
  @inbounds for t in 1:k
    for l in 1:k
      J[l, t] = C[pivots[l], t]
    end
  end

  return _nystrom_from_CJ(C, J, pivots, tol, S)
end

"""
    _nystrom_from_CJ(C, J, pivots, tol, sym) → ALPACAResult

Shared Nyström factorization from column matrix `C = M[:, pivots]` and
pivot submatrix `J = M[pivots, pivots]`.  Used by both `nystrom_finalize`
(cache-based) and `_symmetric_refactorize` (matrix-free).

Specialized via dispatch on symmetry type and element type.
"""
function _nystrom_from_CJ(C::AbstractMatrix{T}, J::AbstractMatrix{T},
                          pivots::Vector{Int}, tol, sym::Symbol) where T
  _nystrom_from_CJ(C, J, pivots, tol, Val(sym))
end

# Complex symmetric: SVD/Takagi factorization
function _nystrom_from_CJ(C::AbstractMatrix{T}, J::AbstractMatrix{T},
                          pivots::Vector{Int}, tol, ::Val{:symmetric}) where {T<:Complex}
  F_svd = svd(J)
  nB = max(count(s -> s > tol, F_svd.S), 1)
  A = F_svd.U[:, 1:nB]
  B = (F_svd.Vt[1:nB, :])'
  phases = [conj(sum(A[:,m] .* B[:,m])) for m in 1:nB]
  inv_sqrt_S = 1.0 ./ sqrt.(F_svd.S[1:nB])
  C_inv = conj(A) .* transpose(sqrt.(conj.(phases)) .* inv_sqrt_S)
  L = C * C_inv
  return ALPACAResult{T}(L, L, Int[], pivots, Int[], :symmetric)
end

"""
    _nystrom_eigen_finalize(C, J_wrapped, pivots, tol, sym) → ALPACAResult

Shared Nyström eigen-finalization for real symmetric and complex Hermitian.

Given column matrix ``C = M[:, \\text{pivots}]`` and the symmetry-wrapped
pivot submatrix `J_wrapped` (a `Symmetric` or `Hermitian`), compute the
eigendecomposition, truncate small eigenvalues, and build the low-rank
factor ``L = C \\, V \\, |\\Lambda|^{-1/2}``.

Returns an [`ALPACAResult`](@ref) with `left === right === L` and
`neg_indices` for negative eigenvalues.
"""
function _nystrom_eigen_finalize(C::AbstractMatrix{T}, J_wrapped,
                                pivots::Vector{Int}, tol, sym::Symbol) where T
  E = eigen(J_wrapped)
  nB = max(count(e -> abs(e) > tol, E.values), 1)
  keep = sortperm(abs.(E.values), rev=true)[1:nB]
  vals = E.values[keep]
  vecs = E.vectors[:, keep]
  inv_sqrt_vals = 1.0 ./ sqrt.(abs.(vals))
  C_inv = vecs .* transpose(inv_sqrt_vals)
  neg_indices = findall(v -> v < 0, vals)
  L = C * C_inv
  return ALPACAResult{T}(L, L, neg_indices, pivots, Int[], sym)
end

# Real symmetric: eigendecomposition of Symmetric
function _nystrom_from_CJ(C::AbstractMatrix{T}, J::AbstractMatrix{T},
                          pivots::Vector{Int}, tol, ::Val{:symmetric}) where {T<:Real}
  _nystrom_eigen_finalize(C, Symmetric(J), pivots, tol, :symmetric)
end

# Complex Hermitian: eigendecomposition of Hermitian
function _nystrom_from_CJ(C::AbstractMatrix{T}, J::AbstractMatrix{T},
                          pivots::Vector{Int}, tol, ::Val{:hermitian}) where {T<:Complex}
  _nystrom_eigen_finalize(C, Hermitian(J), pivots, tol, :hermitian)
end

# ──────────────────────────────────────────────────────────────────
# LDU finalization for general matrices: construct from cache factors
# ──────────────────────────────────────────────────────────────────

"""
    svd_finalize_general(cache, matrix, tol)

Compute SVD-based decomposition for general matrices.

M ≈ C J⁻¹ Rᵀ where C = M[:, col_pivots], J = M[row_pivots, col_pivots].
SVD of J = U Σ Vᵀ gives left = C V Σ^{-1/2}, right = R U Σ^{-1/2}.

Returns `ALPACAResult{T}`.
"""
function svd_finalize_general(cache::ALPACACache{T,R,:general},
                              matrix::AbstractALPACAMatrix, tol) where {T,R}
  k = cache.n_cols
  m = length(cache.cbuf)       # number of rows in the matrix
  ncols = length(cache.rbuf)   # number of columns in the matrix
  col_pivots = cache.pivot_indices
  row_pivots = cache.row_pivot_indices

  if k == 0
    return ALPACAResult{T}(
      Matrix{T}(undef, m, 0), Matrix{T}(undef, ncols, 0),
      Int[], col_pivots, row_pivots, :general)
  end

  # C = M[:, col_pivots]
  C = _get_pivot_columns(matrix, cache)
  # RT[:, t] = M[row_pivots[t], :]
  RT = _get_pivot_rows_T(matrix, cache)

  # J = M[row_pivots, col_pivots]
  J = Matrix{T}(undef, k, k)
  @inbounds for t in 1:k
    for l in 1:k
      J[l, t] = C[row_pivots[l], t]
    end
  end

  return _svd_from_CJ_RT(C, J, RT, col_pivots, row_pivots, tol)
end

"""
    _svd_from_CJ_RT(C, J, RT, col_pivots, row_pivots, tol) → ALPACAResult

Shared SVD factorization from `C = M[:, col_pivots]`, `J = M[row_pivots, col_pivots]`,
and `RT[:, t] = M[row_pivots[t], :]`.  Used by both `svd_finalize_general`
(cache-based) and `_general_refactorize` (matrix-free).
"""
function _svd_from_CJ_RT(C::AbstractMatrix{T}, J::AbstractMatrix{T},
                         RT::AbstractMatrix{T},
                         col_pivots::Vector{Int}, row_pivots::Vector{Int},
                         tol) where T
  F = svd(J)
  nk = count(s -> s > tol, F.S)
  nk = max(nk, 1)
  inv_sqrt_S = 1.0 ./ sqrt.(F.S[1:nk])

  left = C * (F.V[:, 1:nk] .* transpose(inv_sqrt_S))
  right = RT * (F.U[:, 1:nk] .* transpose(inv_sqrt_S))

  return ALPACAResult{T}(left, right, Int[], col_pivots, row_pivots, :general)
end

"""
    _matrix_eltype(matrix)

Determine the element type of a matrix.
"""
_matrix_eltype(mat::DenseALPACAMatrix{T}) where T = T

function _matrix_eltype(matrix::AbstractALPACAMatrix)
  # Fallback: try to infer from size + a single element
  n = size(matrix)[1]
  buf = Vector{ComplexF64}(undef, 1)
  try
    elements!(buf, matrix, [(1, 1)])
    return iszero(imag(buf[1])) ? Float64 : ComplexF64
  catch
    return Float64
  end
end
