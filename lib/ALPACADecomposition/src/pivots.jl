"""
ALPACA pivot selection loops for symmetric/hermitian and general matrices.

Combines Adaptive Cross Approximation (ACA) with principal-element
acceleration.  Symmetric matrices support Bunch-Kaufman 2×2 pivoting
for indefinite or zero-diagonal cases.
"""

# ──────────────────────────────────────────────────────────────────
# Shared pivot-acceptance helpers
# ──────────────────────────────────────────────────────────────────

"""
    _store_symmetric_pivot!(cache, j) → pivot_val

Scale the deflated column in `cache.cbuf` by its diagonal entry at
position `j`, store the result, record the pivot, and update principal
residuals.  Returns the raw pivot value ``d_j`` (before scaling).

Call [`fetch_and_deflate_symmetric!`](@ref) before this function.
"""
function _store_symmetric_pivot!(cache::ALPACACache{T}, j::Int) where T
  n = length(cache.cbuf)
  pivot_val = cache.cbuf[j]
  inv_pv = one(T) / pivot_val
  @inbounds for p in 1:n
    cache.cbuf[p] *= inv_pv
  end
  store_column!(cache, j, cache.cbuf)
  store_pivot!(cache, j, pivot_val)
  k = cache.n_cols
  update_principal_residuals!(cache, @view(cache.columns[:, k]), pivot_val)
  return pivot_val
end

"""
    _store_general_pivot!(cache, col_j, row_i) → pivot_val

Scale both the deflated column (`cache.cbuf`) and row (`cache.rbuf`)
by the intersection value at `(row_i, col_j)`, store both, record the pivot
pair, and update principal residuals.  Returns the raw pivot value.

Call [`fetch_and_deflate_row_general!`](@ref) and
[`fetch_and_deflate_col_general!`](@ref) before this function.
"""
function _store_general_pivot!(cache::ALPACACache{T}, col_j::Int, row_i::Int) where T
  m = length(cache.cbuf)
  ncols = length(cache.rbuf)
  pivot_val = cache.cbuf[row_i]
  inv_pv = one(T) / pivot_val
  @inbounds for p in 1:m
    cache.cbuf[p] *= inv_pv
  end
  @inbounds for p in 1:ncols
    cache.rbuf[p] *= inv_pv
  end
  store_column!(cache, col_j, cache.cbuf)
  store_row!(cache, row_i, cache.rbuf)
  store_pivot!(cache, col_j, pivot_val)
  push!(cache.row_pivot_indices, row_i)
  cache.is_row_pivot[row_i] = true
  k = cache.n_cols
  update_principal_residuals_general!(cache,
    @view(cache.columns[:, k]), @view(cache.rows[:, k]), pivot_val)
  return pivot_val
end

# ──────────────────────────────────────────────────────────────────
# 2×2 Bunch-Kaufman pivoting for symmetric matrices
# ──────────────────────────────────────────────────────────────────

"""
    _eigendecompose_2x2(B, ::Val{S})

Decompose a 2×2 block `B` according to symmetry type `S`:
- `:symmetric` (real): eigendecomposition of `Symmetric(B)`
- `:hermitian`: eigendecomposition of `Hermitian(B)`, eigenvalues converted to `T`
- `:symmetric` (complex): SVD of `B`, singular values converted to `T`

Returns `(values::Vector{T}, vectors::Matrix{T})`.
"""
function _eigendecompose_2x2(B::AbstractMatrix{T}, ::Val{:symmetric}) where {T<:Real}
  F = eigen(Symmetric(B))
  return F.values, F.vectors
end

function _eigendecompose_2x2(B::AbstractMatrix{T}, ::Val{:hermitian}) where T
  F = eigen(Hermitian(B))
  # Convert real eigenvalues to complex type T for consistent typing
  return T.(F.values), F.vectors
end

# Complex symmetric: use SVD to get pseudo-eigendecomposition
function _eigendecompose_2x2(B::AbstractMatrix{T}, ::Val{:symmetric}) where {T<:Complex}
  F = svd(B)
  return T.(F.S), F.U
end

"""
    _attempt_2x2_pivot!(cache, matrix, j, partner, deflated_j, options) → Bool

Attempt a 2×2 Bunch-Kaufman pivot using columns `j` and `partner`.

Column `j` must already be fetched+deflated (passed as `deflated_j`).
Fetches column `partner`, computes the 2×2 intersection block in the
deflated basis, eigendecomposes it, and stores two rotated rank-1 pivots
with the eigenvalues as pivot diagonal values.

The rotated (deflated, scaled) columns are stored in `cache.columns` for
correct subsequent deflation and raw-factor (`lpaca`) reconstruction.

Returns `true` if at least one valid pivot was stored.
"""
function _attempt_2x2_pivot!(cache::ALPACACache{T,R,S},
                              matrix::AbstractALPACAMatrix,
                              j::Int, partner::Int,
                              deflated_j::Vector{T},
                              options::ALPACAOptions) where {T,R,S}
  n = length(cache.cbuf)
  tol = options.pivotol

  # Partner must not already be a pivot
  cache.is_pivot[partner] && return false

  # Save the original (undeflated) column j (no-op for DenseALPACAMatrix)
  saved_orig_j = _save_orig_col(cache, matrix)

  # Fetch and deflate the partner column (dispatches hermitian/symmetric).
  fetch_and_deflate_symmetric!(cache, matrix, partner)
  deflated_partner = copy(cache.cbuf)
  saved_orig_partner = _save_orig_col(cache, matrix)

  # 2×2 intersection block from deflated columns
  B = T[deflated_j[j]       deflated_j[partner];
        deflated_partner[j]  deflated_partner[partner]]

  # Eigendecompose
  λ, V = _eigendecompose_2x2(B, Val(S))

  # Sort by decreasing |eigenvalue|
  order = sortperm(abs.(λ), rev=true)
  λ = λ[order]
  V = V[:, order]

  # Both eigenvalues too small → no usable pivot
  if abs(λ[1]) < tol
    return false
  end

  indices = (j, partner)
  saved_origs = (saved_orig_j, saved_orig_partner)
  stored = 0
  for t in 1:2
    abs(λ[t]) < tol && break

    # Rotated deflated column, scaled by 1/λ[t]
    inv_lambda = one(T) / λ[t]
    v1, v2 = V[1, t], V[2, t]
    @inbounds for p in 1:n
      cache.cbuf[p] = (deflated_j[p] * v1 + deflated_partner[p] * v2) * inv_lambda
    end

    # Store as a standard 1×1 column pivot
    jj = indices[t]
    store_column!(cache, jj, cache.cbuf)
    k = cache.n_cols

    # Restore original column for Nyström finalization (no-op for DenseALPACAMatrix)
    _restore_orig_col!(cache, matrix, k, saved_origs[t])

    # Record pivot + update residuals
    store_pivot!(cache, jj, λ[t])
    update_principal_residuals!(cache, @view(cache.columns[:, k]), λ[t])
    stored += 1
  end

  return stored > 0
end

"""
    alpaca_pivots!(cache, matrix, options, descriptor)

Run the ALPACA pivot selection loop. Modifies `cache` in place.
Returns `cache.pivot_indices` (column pivots).

The symmetry type parameter `S` on the cache provides compile-time dispatch:
- `:general` → `alpaca_pivots_general!` (ACA with separate row/col pivots)
- `:symmetric`, `:hermitian` → shared symmetric path with Bunch-Kaufman

Algorithm (symmetric / Hermitian / complex symmetric):
1. Initialize principal values from descriptor
2. Main loop: pick candidate index `j` from principal (argmax |residual|)
   or ACA (argmax of last stored column); fetch and deflate column `j`
3. Inspect the deflated column: compare diagonal `d` vs largest off-diagonal `g`;
   accept 1×1 pivot if ``d \\ge \\max((1 - 5\\tau)\\,g,\\; \\tau)`` where ``\\tau`` is
   `pivotol`; otherwise attempt 2×2 Bunch-Kaufman pivot with the off-diagonal partner
4. Stop when both ACA and principal proposals fall below tolerance
"""
function alpaca_pivots!(cache::ALPACACache{T,R,:general},
                        matrix::AbstractALPACAMatrix,
                        options::ALPACAOptions,
                        descriptor::AbstractPrincipalDescriptor) where {T,R}
  return alpaca_pivots_general!(cache, matrix, options, descriptor)
end

# ──────────────────────────────────────────────────────────────────
# ACA candidate helpers
# ──────────────────────────────────────────────────────────────────

"""
    _aca_candidate_symmetric(cache, n) → (best_idx, magnitude)

Scan the two most recent stored columns for the largest non-pivot entry,
returning its index and estimated residual magnitude (`|entry| * |d_k|`).

Scanning two columns instead of one gives ACA a wider view of the
residual, which helps when the latest column has small entries
(e.g. after a 2×2 pivot where the second rotated column may be small).

Returns `(0, 0)` when no pivots have been stored yet.
"""
function _aca_candidate_symmetric(cache::ALPACACache{T,R}, n::Int) where {T,R}
  k = cache.n_cols
  k == 0 && return (0, zero(R))

  best_idx = 0
  best_mag = zero(R)

  # Scan last stored column
  last_col = @view cache.columns[:, k]
  abs_dk = abs(cache.pivot_diag[k])
  @inbounds for p in 1:n
    if !cache.is_pivot[p]
      v = abs(last_col[p])
      mag = v * abs_dk
      if mag > best_mag
        best_mag = mag
        best_idx = p
      end
    end
  end

  # Also scan the previous column to widen the search
  if k >= 2
    prev_col = @view cache.columns[:, k - 1]
    abs_dk1 = abs(cache.pivot_diag[k - 1])
    @inbounds for p in 1:n
      if !cache.is_pivot[p]
        v = abs(prev_col[p])
        mag = v * abs_dk1
        if mag > best_mag
          best_idx = p
          best_mag = mag
        end
      end
    end
  end

  return (best_idx, best_mag)
end

"""
    _aca_next_col(cache, ncols) → (best_col, magnitude)

Scan the most recent stored row for the largest non-pivot column entry,
returning its index and estimated residual magnitude (`|entry| * |d_k|`).

Returns `(0, 0)` when no pivots have been stored yet.
"""
function _aca_next_col(cache::ALPACACache{T,R,:general}, ncols::Int) where {T,R}
  k = cache.n_cols
  k == 0 && return (0, zero(R))

  best_col = 0
  best_mag = zero(R)
  last_row = @view cache.rows[:, k]
  abs_dk = abs(cache.pivot_diag[k])
  @inbounds for j in 1:ncols
    if !cache.is_pivot[j]
      v = abs(last_row[j])
      mag = v * abs_dk
      if mag > best_mag
        best_mag = mag
        best_col = j
      end
    end
  end
  return (best_col, best_mag)
end

"""
    _aca_next_row(cache, m) → (best_row, best_val)

Scan `cache.cbuf` for the largest non-row-pivot entry.
Used in the general ACA path after fetching+deflating a column
to find the best row partner.

Returns `(0, 0)` if all rows are already pivoted.
"""
function _aca_next_row(cache::ALPACACache{T}, m::Int) where T
  best_row = 0
  best_val = zero(real(T))
  @inbounds for i in 1:m
    if !cache.is_row_pivot[i]
      v = abs(cache.cbuf[i])
      if v > best_val
        best_val = v
        best_row = i
      end
    end
  end
  return (best_row, best_val)
end

function alpaca_pivots!(cache::ALPACACache{T,R,S},
                        matrix::AbstractALPACAMatrix,
                        options::ALPACAOptions,
                        descriptor::AbstractPrincipalDescriptor) where {T,R,S}
  n = length(cache.cbuf)
  tol = options.pivotol
  tol2 = abs2(tol)
  max_rank = options.max_rank

  # Initialize principal values
  init_principal_values!(cache, matrix, descriptor)

  for iter in 1:min(n, max_rank)
    k = cache.n_cols  # number of pivots so far

    # ── Candidate selection: principal first, ACA as fallback ──
    prin_best_slot = _argmax_principal(cache)
    prin_magnitude = prin_best_slot > 0 ?
      abs(cache.principal_values[prin_best_slot]) : zero(R)

    local next_j::Int
    local from_principal_diagonal::Bool = false
    if prin_magnitude >= (k == 0 ? tol2 : tol) && prin_best_slot > 0
      # Principal: pick either index from the best pair (both are non-pivots,
      # guaranteed by _argmax_principal)
      prin_i, next_j = cache.principal_pairs[prin_best_slot]
      from_principal_diagonal = (prin_i == next_j)
    else
      # Fall back to ACA: argmax |residual| in most recent stored column(s)
      aca_best_idx, aca_magnitude = _aca_candidate_symmetric(cache, n)
      if aca_best_idx > 0 && aca_magnitude >= tol
        next_j = aca_best_idx
      else
        break
      end
    end

    # ── Fetch and deflate column next_j ──
    fetch_and_deflate_symmetric!(cache, matrix, next_j)

    # ── 1×1 vs 2×2 pivot decision ──
    if from_principal_diagonal
      # Principal diagonal element: the diagonal residual was already the
      # largest monitored value, so 1×1 pivot is the right choice.
      # Consistency check: pivot value should match the principal value
      @assert abs(abs(cache.cbuf[next_j]) - prin_magnitude) <
              10 * tol * max(prin_magnitude, one(R)) "pivot value $(cache.cbuf[next_j]) inconsistent with principal value $prin_magnitude"
      _store_symmetric_pivot!(cache, next_j)
    else
      # Find the element with the largest absolute value in the deflated column
      # among non-pivot indices (this is the off-diagonal candidate)
      diag_val = abs(cache.cbuf[next_j])
      offdiag_idx = 0
      offdiag_val = zero(R)
      @inbounds for p in 1:n
        p == next_j && continue
        cache.is_pivot[p] && continue
        v = abs(cache.cbuf[p])
        if v > offdiag_val
          offdiag_val = v
          offdiag_idx = p
        end
      end

      if diag_val < tol && offdiag_val < tol
        # Entire column is negligible → skip (no more rank in this direction)
        continue
      end

      if offdiag_idx == 0 || diag_val >= max((one(R) - 5 * tol) * offdiag_val, tol)
        # Diagonal is large enough for a stable 1×1 pivot
        _store_symmetric_pivot!(cache, next_j)
      else
        # Off-diagonal is significantly larger → 2×2 Bunch-Kaufman
        if !_attempt_2x2_pivot!(cache, matrix, next_j, offdiag_idx,
                                copy(cache.cbuf), options)
          # 2×2 failed (both eigenvalues below tol); try 1×1 if diagonal is usable
          if diag_val >= tol
            # Re-fetch column (2×2 attempt may have overwritten cbuf)
            fetch_and_deflate_symmetric!(cache, matrix, next_j)
            _store_symmetric_pivot!(cache, next_j)
          end
        end
      end
    end
  end

  return cache.pivot_indices
end

# ──────────────────────────────────────────────────────────────────────────────
# General-matrix pivot loop with separate row / column pivots
# ──────────────────────────────────────────────────────────────────────────────

"""
    alpaca_pivots_general!(cache, matrix, options, descriptor)

ACA pivot selection for general (non-symmetric) matrices with principal
element acceleration.

Each iteration selects a **(row, column)** pair using:

1. **Principal proposal**: ``(i, j)`` pair with largest ``|\\text{residual}|``
   — gives row ``i`` and column ``j`` directly.
2. **ACA fallback**: argmax of last deflated **row** → next column index;
   fetch+deflate that column → argmax gives next row index.

Returns `(cache.pivot_indices, cache.row_pivot_indices)`.
"""
function alpaca_pivots_general!(cache::ALPACACache{T},
                                matrix::AbstractALPACAMatrix,
                                options::ALPACAOptions,
                                descriptor::AbstractPrincipalDescriptor) where T
  m = length(cache.cbuf)       # number of rows in the matrix
  ncols = length(cache.rbuf)   # number of columns in the matrix
  tol = options.pivotol
  tol2 = abs2(tol)
  max_rank = options.max_rank

  # Initialize principal values
  init_principal_values!(cache, matrix, descriptor)

  for iter in 1:min(m, ncols, max_rank)
    k = cache.n_cols

    # ── Principal proposal ──
    prin_slot = _argmax_principal(cache)
    prin_magnitude = prin_slot > 0 ?
      abs(cache.principal_values[prin_slot]) : zero(real(T))

    local next_i::Int, next_j::Int

    if prin_magnitude >= (k == 0 ? tol2 : tol) && prin_slot > 0
      # Principal gives both row and column directly
      prin_i, prin_j = cache.principal_pairs[prin_slot]
      if !cache.is_row_pivot[prin_i] && !cache.is_pivot[prin_j]
        next_i = prin_i
        next_j = prin_j

        # Principal path: fetch both column and row
        fetch_and_deflate_col_general!(cache, matrix, next_j)
        fetch_and_deflate_row_general!(cache, matrix, next_i)

        # Consistency check: pivot value should match the principal value
        @assert abs(abs(cache.cbuf[next_i]) - prin_magnitude) <
                10 * tol * max(prin_magnitude, one(real(T))) "pivot value $(cache.cbuf[next_i]) inconsistent with principal value $prin_magnitude"
        _store_general_pivot!(cache, next_j, next_i)
        continue
      end
      # One of them is already used; fall through to ACA
    end

    # ── ACA fallback: argmax of last stored row → next column ──
    aca_col, aca_magnitude = _aca_next_col(cache, ncols)

    if aca_col > 0 && aca_magnitude >= tol && !cache.is_pivot[aca_col]
      # ACA: maxabs of last row gave us a column; fetch+deflate it,
      # then maxabs of this column gives us a row
      next_j = aca_col
      fetch_and_deflate_col_general!(cache, matrix, next_j)

      # Find best unused row from the deflated column
      next_i, best_i_val = _aca_next_row(cache, m)
      if next_i == 0 || best_i_val < tol
        break
      end

      # Fetch and deflate the row
      fetch_and_deflate_row_general!(cache, matrix, next_i)

      # Pivot value at intersection
      if abs(cache.cbuf[next_i]) < tol
        break
      end
      _store_general_pivot!(cache, next_j, next_i)
    else
      break
    end
  end

  return cache.pivot_indices, cache.row_pivot_indices
end

"""
    _argmax_principal(cache) → slot index (0 if empty)

Return the index into cache.principal_pairs/values with the largest |residual| 
among non-pivot entries.

For `:general` caches, checks `is_row_pivot` for the first index
and `is_pivot` for the second.  For symmetric-like caches, checks
`is_pivot` for both.
"""
function _argmax_principal(cache::ALPACACache{T,R,:general}) where {T,R}
  best_slot = 0
  best_val = zero(real(T))
  @inbounds for p in eachindex(cache.principal_values)
    a, b = cache.principal_pairs[p]
    (cache.is_row_pivot[a] || cache.is_pivot[b]) && continue
    v = abs(cache.principal_values[p])
    if v > best_val
      best_val = v
      best_slot = p
    end
  end
  return best_slot
end

function _argmax_principal(cache::ALPACACache{T}) where T
  best_slot = 0
  best_val = zero(real(T))
  @inbounds for p in eachindex(cache.principal_values)
    a, b = cache.principal_pairs[p]
    (cache.is_pivot[a] || cache.is_pivot[b]) && continue
    v = abs(cache.principal_values[p])
    if v > best_val
      best_val = v
      best_slot = p
    end
  end
  return best_slot
end
