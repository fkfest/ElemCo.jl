"""
Matrix-class-specific fetch, deflation, and principal update kernels.

Kernels that distinguish Hermitian from symmetric are dispatched via
the compile-time symmetry type parameter `S` on [`ALPACACache`](@ref),
so there is no runtime cost for the distinction.
"""

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
    update_principal_residuals!(cache, stored_col, pivot_val)

Update principal residual values after accepting a pivot with value `pivot_val`.

The `stored_col` is the already-scaled column ``L_{:,k} = c_{:,k} / d_k``
stored in the cache.  For each principal pair ``(a, b)``:

    ``\\Delta_p = L_{a,k} \\cdot L_{b,k} \\cdot d_k`` (symmetric)
    ``\\Delta_p = L_{a,k} \\cdot \\overline{L_{b,k}} \\cdot d_k`` (hermitian)

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
    update_principal_residuals_general!(cache, stored_col, stored_row, pivot_val)

Update principal residuals for general matrices.
The stored column/row are already scaled by ``1/d_k``:

    ``\\Delta_p = L^C_{a,k} \\cdot L^R_{b,k} \\cdot d_k``
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
