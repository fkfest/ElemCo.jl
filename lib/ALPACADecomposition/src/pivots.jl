"""
Main ALPACA pivot selection loop combining ACA residual pivots and
principal-element pivots with dual convergence.
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

"""
    alpaca_pivots!(cache, matrix, options, descriptor)

Run the ALPACA pivot selection loop. Modifies `cache` in place.
Returns `cache.pivot_indices` (column pivots).

The symmetry type parameter `S` on the cache provides compile-time dispatch:
- `:general` → `alpaca_pivots_general!` (alternating ACA with separate row/col pivots)
- `:symmetric`, `:hermitian` → shared symmetric path

Algorithm (symmetric / Hermitian / complex symmetric):
1. Initialize principal values (one-time `elements!` for pairs, copy for triples)
2. Select first pivot from argmax |principal_values|
3. Main loop: at each iteration, compare ACA proposal vs principal proposal,
   accept the larger, fetch+deflate, update residuals
4. Stop when dual convergence (Frobenius + principal) or max_rank
"""
function alpaca_pivots!(cache::ALPACACache{T,R,:general},
                        matrix::AbstractALPACAMatrix,
                        options::ALPACAOptions,
                        descriptor::AbstractPrincipalDescriptor) where {T,R}
  return alpaca_pivots_general!(cache, matrix, options, descriptor)
end

function alpaca_pivots!(cache::ALPACACache{T},
                        matrix::AbstractALPACAMatrix,
                        options::ALPACAOptions,
                        descriptor::AbstractPrincipalDescriptor) where T
  n = length(cache.cbuf)
  tol = options.pivotol
  tol2 = abs2(tol)
  max_rank = options.max_rank

  # Initialize principal values
  init_principal_values!(cache, matrix, descriptor)

  # Select first pivot from principal argmax
  first_pivot = _argmax_principal(cache)
  if first_pivot == 0 || abs(cache.principal_values[first_pivot]) < tol2
    return cache.pivot_indices   # nothing significant
  end

  # The first pivot column index
  first_a, first_b = cache.principal_pairs[first_pivot]
  first_j = first_b  # fetch column b for the first principal pair

  # Fetch and deflate (no prior pivots, so this is just a fetch)
  fetch_and_deflate_symmetric!(cache, matrix, first_j)
  _store_symmetric_pivot!(cache, first_j)

  # ── Main loop ──
  for iter in 2:min(n, max_rank)
    k = cache.n_cols  # number of pivots so far

    # ACA proposal: argmax |residual| in most recent deflated column
    last_col = @view cache.columns[:, k]
    aca_best_idx = 0
    aca_best_val = zero(real(T))
    @inbounds for p in 1:n
      if !cache.is_pivot[p]
        v = abs(last_col[p])
        if v > aca_best_val
          aca_best_val = v
          aca_best_idx = p
        end
      end
    end
    # The ACA magnitude is scaled by pivot diagonal
    aca_magnitude = aca_best_val * abs(cache.pivot_diag[k])

    # Principal proposal: argmax |principal_values|
    prin_best_slot = _argmax_principal(cache)
    prin_magnitude = prin_best_slot > 0 ? abs(cache.principal_values[prin_best_slot]) : zero(real(T))

    # Both below tolerance → converged
    if aca_magnitude < tol && prin_magnitude < tol
      break
    end

    # Prefer principal; use ACA only as fallback when principal is below tol
    if prin_magnitude >= tol && prin_best_slot > 0
      # Accept principal pivot
      _, next_j = cache.principal_pairs[prin_best_slot]
    elseif aca_best_idx > 0
      # ACA fallback
      next_j = aca_best_idx
    else
      break
    end

    # Skip if already a pivot (can happen in edge cases)
    if cache.is_pivot[next_j]
      # Try the other candidate
      if prin_magnitude >= tol && aca_best_idx > 0 && !cache.is_pivot[aca_best_idx]
        next_j = aca_best_idx
      elseif prin_best_slot > 0
        _, next_j = cache.principal_pairs[prin_best_slot]
      else
        break
      end
      cache.is_pivot[next_j] && break
    end

    # Fetch and deflate
    fetch_and_deflate_symmetric!(cache, matrix, next_j)
    if abs(cache.cbuf[next_j]) < options.pivotol
      break
    end
    _store_symmetric_pivot!(cache, next_j)
  end

  return cache.pivot_indices
end

# ──────────────────────────────────────────────────────────────────────────────
# General-matrix pivot loop with separate row / column pivots
# ──────────────────────────────────────────────────────────────────────────────

"""
    alpaca_pivots_general!(cache, matrix, options, descriptor)

Alternating ACA pivot selection for general (non-symmetric) matrices.

Each iteration selects a **(row, column)** pair using:

1. **ACA row proposal**: row with largest magnitude in the most-recently
   deflated column (among non-row-pivots).
2. **Principal proposal**: ``(i, j)`` pair with largest ``|\\text{residual}|``
   from the principal descriptor.

The winning row is fetched and deflated → scan the row for the best column
→ fetch and deflate the column → store both → update.

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

  # Select first pivot from principal argmax
  first_slot = _argmax_principal(cache)
  if first_slot == 0 || abs(cache.principal_values[first_slot]) < tol2
    return cache.pivot_indices, cache.row_pivot_indices
  end

  first_a, _ = cache.principal_pairs[first_slot]

  # ── First pivot: fetch row first_a, scan for best column ──
  fetch_and_deflate_row_general!(cache, matrix, first_a)

  best_col = 0
  best_col_val = zero(real(T))
  @inbounds for j in 1:ncols
    v = abs(cache.rbuf[j])
    if v > best_col_val
      best_col_val = v
      best_col = j
    end
  end
  if best_col == 0
    return cache.pivot_indices, cache.row_pivot_indices
  end

  # Fetch column
  fetch_and_deflate_col_general!(cache, matrix, best_col)

  # Pivot value at intersection (row, col)
  if abs(cache.cbuf[first_a]) < options.pivotol
    return cache.pivot_indices, cache.row_pivot_indices
  end

  _store_general_pivot!(cache, best_col, first_a)

  # ── Main loop ──
  for iter in 2:min(m, ncols, max_rank)
    k = cache.n_cols

    # ── ACA row proposal: argmax |last_col[i]| among non-row-pivots ──
    last_col = @view cache.columns[:, k]
    aca_row = 0
    aca_row_val = zero(real(T))
    @inbounds for i in 1:m
      if !cache.is_row_pivot[i]
        v = abs(last_col[i])
        if v > aca_row_val
          aca_row_val = v
          aca_row = i
        end
      end
    end
    aca_magnitude = aca_row_val * abs(cache.pivot_diag[k])

    # ── Principal proposal ──
    prin_slot = _argmax_principal(cache)
    prin_magnitude = prin_slot > 0 ?
      abs(cache.principal_values[prin_slot]) : zero(real(T))

    # Both below tolerance → converged
    if aca_magnitude < tol && prin_magnitude < tol
      break
    end

    local next_i::Int
    scan_row_for_col = true

    if prin_magnitude > aca_magnitude && prin_slot > 0
      prin_i, prin_j = cache.principal_pairs[prin_slot]
      if !cache.is_row_pivot[prin_i] && !cache.is_pivot[prin_j]
        # Use both row and column from principal
        next_i = prin_i
        fetch_and_deflate_row_general!(cache, matrix, next_i)
        # Prefer the principal column, but verify it is still the row's best
        # among non-col-pivots
        scan_row_for_col = true
      elseif aca_row > 0
        next_i = aca_row
      else
        break
      end
    else
      if aca_row == 0
        break
      end
      next_i = aca_row
    end

    # Fetch and deflate row (if not already done via principal path above)
    if scan_row_for_col && cache.n_cols == k
      # Row was not fetched yet in the principal-with-both-taken path
      fetch_and_deflate_row_general!(cache, matrix, next_i)
    end

    # Find best column from the deflated row
    best_j = 0
    best_j_val = zero(real(T))
    @inbounds for j in 1:ncols
      if !cache.is_pivot[j]
        v = abs(cache.rbuf[j])
        if v > best_j_val
          best_j_val = v
          best_j = j
        end
      end
    end
    if best_j == 0
      break
    end
    next_j = best_j

    # Fetch and deflate column next_j
    fetch_and_deflate_col_general!(cache, matrix, next_j)

    # Pivot value at intersection
    if abs(cache.cbuf[next_i]) < options.pivotol
      break
    end

    _store_general_pivot!(cache, next_j, next_i)
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
