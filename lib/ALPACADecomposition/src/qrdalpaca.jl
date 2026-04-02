# ──────────────────────────────────────────────────────────────────
# qrdalpaca: ALPACA pivot selection + QR refinement
# ──────────────────────────────────────────────────────────────────

"""
    _fetch_columns!(dest, matrix::AbstractALPACAMatrix, indices, buf)

Fetch columns `indices` from `matrix` into `dest[:, 1:length(indices)]`
using the matrix-free `column!` interface.  `buf` is a pre-allocated work vector
of length `size(matrix, 1)`.
"""
function _fetch_columns!(dest::AbstractMatrix{T}, matrix::AbstractALPACAMatrix,
                         indices, buf::AbstractVector{T}) where T
  @inbounds for (k, j) in enumerate(indices)
    column!(buf, matrix, j)
    dest[:, k] .= buf
  end
  return dest
end

"""
    qrdalpaca(matrix; principal=nothing, options=ALPACAOptions(...))

Matrix-free ALPACA decomposition with QR-pivoted refinement.

Runs ALPACA pivot selection, then checks remaining columns via
column-pivoted QR for any significant columns the greedy phase may
have missed.  New pivots found by QR refinement are incorporated into
the cache (fetch + deflate) and the final low-rank factorization is
obtained by decomposition-based finalization.

# Algorithm
1. Run ALPACA pivot loop to build initial factors in the cache.
2. QR-decompose raw pivot columns (reused in finalization).
3. Quick reconstruction pre-check on a random sample; also computes
   projection norms for sampled columns.
4. Compute residual column norms ``\\|M_{:,j} - Q Q^\\dagger M_{:,j}\\|^2``
   for remaining non-pivot columns.
5. Repeatedly select batches of columns with large residuals, run
   column-pivoted QR, and extend ``Q`` with any newly discovered pivots.
6. Extend the cache with new pivots (fetch + deflate), then finalize.
"""
function qrdalpaca(matrix::AbstractALPACAMatrix{T};
                   principal=nothing,
                   options::ALPACAOptions) where T
  m, n = _validate_matrix(matrix, options.symmetry)
  descriptor = normalize_principal_descriptor(options.symmetry, min(m, n), principal)
  return _qrdalpaca_impl(matrix, m, n, descriptor, options, Val(options.symmetry))
end

function _qrdalpaca_impl(matrix::AbstractALPACAMatrix{T},
                         m::Int, n::Int,
                         descriptor::AbstractPrincipalDescriptor,
                         options::ALPACAOptions,
                         ::Val{S}) where {T, S}
  # ── Step 1: ALPACA pivot loop (keeping cache alive) ──
  cache = ALPACACache(T, m, n, Val{S}(), descriptor.pairs)
  alpaca_pivots!(cache, matrix, options, descriptor)
  k_alpaca = cache.n_cols

  RT = real(T)
  sym = options.symmetry
  sig = options.sigma
  tol_val = options.pivotol
  max_pivots = min(m, n, options.max_rank)

  if k_alpaca == 0 || k_alpaca >= max_pivots
    return decomposition_finalize(cache, options.tol)
  end

  # ── Step 2: Quick reconstruction pre-check + QR projection fallback ──
  # Use raw cache factors directly: M ≈ C*D*C' (sym/herm) or C*D*R^T (general)
  cbuf = Vector{T}(undef, m)
  k = k_alpaca
  C = cache.columns[:, 1:k]
  d = cache.pivot_diag[1:k]
  n_basis = k

  is_pivot = falses(n)
  @inbounds for p in cache.pivot_indices
    is_pivot[p] = true
  end

  tol2 = tol_val^2
  col_norms2 = zeros(RT, n)
  non_pivots = [j for j in 1:n if !is_pivot[j]]
  nn = length(non_pivots)

  # Compute reconstruction coefficients from raw cache factors
  _recon_coeffs(indices) = if sym == :general
    Diagonal(d) * transpose(cache.rows[indices, 1:k])  # M ≈ C*D*Rᵀ
  elseif sym == :hermitian
    Diagonal(d) * C[indices, :]'                        # M ≈ C*D*C'
  else
    Diagonal(d) * transpose(C[indices, :])              # M ≈ C*D*Cᵀ
  end

  # QR of raw columns (computed once, reused in decomposition_finalize)
  column_qr = qr(C)
  n_acc = n_basis
  cap = max(2 * n_basis, 64)
  Q_acc = Matrix{T}(undef, m, cap)
  # Build thin Q via lmul! (avoids materializing full m×m Q matrix)
  @views Q_acc[:, 1:n_acc] .= zero(T)
  @inbounds for j in 1:n_acc
    Q_acc[j, j] = one(T)
  end
  lmul!(column_qr.Q, @view Q_acc[:, 1:n_acc])

  # Quick reconstruction pre-check on a random sample of non-pivot columns.
  # Also computes projection norms for the sampled columns to avoid refetching.
  #
  # Sample size for 99.9% confidence of detecting missed columns:
  #   P(detect ≥ 1 from d missed in N) = 1 - (1 - d/N)^k ≥ 0.999
  #   ⟹ k ≥ -ln(0.001) / (d/N) ≈ 6.908 N/d
  # With d = √N:  k ≈ 7√N  (detects ≥ √N missed columns)
  # With d = 0.01N: k ≈ 700 (detects ≥ 1% missed columns)
  n_sample = min(nn, max(ceil(Int, 7*sqrt(nn)), 700))
  sampled = falses(n)  # track which columns already have norms
  if n_sample > 0
    recon_threshold = tol_val

    # Sample non-pivot columns
    sample_idx = if n_sample < nn
      non_pivots[sort(randperm(nn)[1:n_sample])]
    else
      non_pivots
    end
    cols_batch = Matrix{T}(undef, m, length(sample_idx))
    _fetch_columns!(cols_batch, matrix, sample_idx, cbuf)

    # Compute projection norms for sampled columns (before modifying cols_batch)
    col_n2 = vec(sum(abs2, cols_batch, dims=1))
    Qt_cols = @views Q_acc[:, 1:n_acc]' * cols_batch
    proj_n2 = vec(sum(abs2, Qt_cols, dims=1))
    @inbounds for (i, j) in enumerate(sample_idx)
      col_norms2[j] = max(col_n2[i] - proj_n2[i], zero(RT))
      sampled[j] = true
    end

    # Check element-wise reconstruction
    cols_batch .-= C * _recon_coeffs(sample_idx)
    recon_ok = true
    @inbounds for i in eachindex(sample_idx)
      if maximum(abs, @view cols_batch[:, i]) >= recon_threshold
        recon_ok = false
        break
      end
    end
    if recon_ok
      return decomposition_finalize(cache, options.tol; column_qr)
    end
  end

  # Reconstruction check failed → compute projection residuals for remaining columns.
  remaining = [j for j in non_pivots if !sampled[j]]
  n_rem = length(remaining)
  batch_size = max(n_rem, 256)
  for start in 1:batch_size:n_rem
    stop = min(start + batch_size - 1, n_rem)
    batch_idx = remaining[start:stop]
    nb = length(batch_idx)
    cols_batch = Matrix{T}(undef, m, nb)
    _fetch_columns!(cols_batch, matrix, batch_idx, cbuf)
    col_n2 = vec(sum(abs2, cols_batch, dims=1))
    Qt_cols = @views Q_acc[:, 1:n_acc]' * cols_batch
    proj_n2 = vec(sum(abs2, Qt_cols, dims=1))
    @inbounds for (i, j) in enumerate(batch_idx)
      col_norms2[j] = max(col_n2[i] - proj_n2[i], zero(RT))
    end
  end

  D_indices = [j for j in 1:n if col_norms2[j] >= tol2]
  if isempty(D_indices)
    return decomposition_finalize(cache, options.tol; column_qr)
  end

  # ── Step 4: Batched QR refinement ──
  # Q_acc already built from above

  orig_col_norms2 = copy(col_norms2)
  recompute_ratio = eps(RT)^(2/3)
  refinement_pivots = Int[]
  max_batch = max(256, 2 * n_acc)

  while !isempty(D_indices)
    D_max = maximum(col_norms2[j] for j in D_indices)
    D_max < tol2 && break

    threshold = sig * D_max
    batch = sort!(filter(j -> col_norms2[j] >= threshold, D_indices);
                  by=j -> col_norms2[j], rev=true)
    nQ = min(length(batch), max_batch)
    batch = batch[1:nQ]

    # Fetch batch columns via matrix-free interface
    cols = Matrix{T}(undef, m, nQ)
    _fetch_columns!(cols, matrix, batch, cbuf)

    # Project out Q_acc
    if n_acc > 0
      Qv = @view Q_acc[:, 1:n_acc]
      proj = Qv' * cols
      mul!(cols, Qv, proj, -one(T), one(T))
    end

    # Column-pivoted QR
    F_qr = qr!(cols, ColumnNorm())
    R_diag = abs.(diag(F_qr.R))
    n_new = count(rd -> rd > tol_val, R_diag)
    n_new == 0 && break

    @inbounds for k in 1:n_new
      p = batch[F_qr.p[k]]
      push!(refinement_pivots, p)
      is_pivot[p] = true
      col_norms2[p] = zero(RT)
    end

    # Ensure Q_acc capacity
    if n_acc + n_new > size(Q_acc, 2)
      new_cap = max(2 * size(Q_acc, 2), n_acc + n_new)
      Q_new = Matrix{T}(undef, m, new_cap)
      @views Q_new[:, 1:n_acc] .= Q_acc[:, 1:n_acc]
      Q_acc = Q_new
    end

    Q_thin = zeros(T, m, n_new)
    @inbounds for k in 1:n_new
      Q_thin[k, k] = one(T)
    end
    lmul!(F_qr.Q, Q_thin)
    @views Q_acc[:, n_acc+1:n_acc+n_new] .= Q_thin
    n_acc += n_new
    max_batch = max(256, 2 * n_acc)

    # Update remaining D
    D_indices = filter!(j -> !is_pivot[j] && col_norms2[j] >= tol2, D_indices)
    isempty(D_indices) && break

    # Update residual norms via matrix-free interface
    cols_d = Matrix{T}(undef, m, length(D_indices))
    _fetch_columns!(cols_d, matrix, D_indices, cbuf)
    proj_new = Q_thin' * cols_d
    @inbounds for (idx, I) in enumerate(D_indices)
      decrement = sum(abs2, @view proj_new[:, idx])
      col_norms2[I] -= decrement
      col_norms2[I] < 0 && (col_norms2[I] = zero(RT))
    end

    # Recompute catastrophically cancelled norms
    recompute = [I for I in D_indices
                 if col_norms2[I] < recompute_ratio * orig_col_norms2[I]]
    if !isempty(recompute)
      Qv = @view Q_acc[:, 1:n_acc]
      cols_r = Matrix{T}(undef, m, length(recompute))
      _fetch_columns!(cols_r, matrix, recompute, cbuf)
      proj_r = Qv' * cols_r
      mul!(cols_r, Qv, proj_r, -one(T), one(T))
      @inbounds for (k, I) in enumerate(recompute)
        col_norms2[I] = sum(abs2, @view cols_r[:, k])
      end
    end

    filter!(j -> col_norms2[j] >= tol2, D_indices)
  end

  if isempty(refinement_pivots)
    return decomposition_finalize(cache, options.tol; column_qr)
  end

  # ── Step 6: Extend cache with new pivots, then finalize ──
  if S === :general
    _extend_cache_general!(cache, matrix, refinement_pivots, tol_val)
  else
    _extend_cache_symmetric!(cache, matrix, refinement_pivots, tol_val)
  end
  return decomposition_finalize(cache, options.tol)
end

"""
    qrdalpaca(matrix::AbstractMatrix; tol, [symmetry], ...)

Convenience interface for dense matrices.  Auto-detects symmetry unless
overridden.  See [`alpaca`](@ref) for keyword arguments.
"""
function qrdalpaca(matrix::AbstractMatrix;
                   tol::Union{Real,Nothing}=nothing,
                   principal=nothing,
                   symmetry::Union{Symbol,Nothing}=nothing,
                   options::Union{ALPACAOptions,Nothing}=nothing,
                   pivotol::Union{Real,Nothing}=nothing,
                   sigma::Real=0.01,
                   max_rank::Integer=typemax(Int))
  options = _build_options(matrix; tol, symmetry, options, pivotol, sigma,
                           qr=true, max_rank)
  return qrdalpaca(DenseALPACAMatrix(matrix); principal, options)
end

"""
    _extend_cache_symmetric!(cache, matrix, new_pivots, tol)

Incorporate QR-discovered pivots into the ALPACA cache for
symmetric/hermitian matrices.  For each new pivot, fetches and deflates
the column against all existing cache pivots, then stores it as a new
pivot (with the standard Schur-complement scaling).

Pivots whose deflated diagonal entry is below `tol` are skipped.
"""
function _extend_cache_symmetric!(cache::ALPACACache{T}, matrix::AbstractALPACAMatrix,
                                  new_pivots::Vector{Int}, tol) where T
  for j in new_pivots
    fetch_and_deflate_symmetric!(cache, matrix, j)
    pivot_val = cache.cbuf[j]
    abs(pivot_val) < tol && continue
    _store_symmetric_pivot!(cache, j)
  end
end

"""
    _extend_cache_general!(cache, matrix, new_col_pivots, tol)

Incorporate QR-discovered column pivots into the ALPACA cache for
general matrices.  For each new column pivot, fetches and deflates
the column, finds the best row partner, fetches and deflates that row,
then stores both in the cache.

Pivots whose best row entry is below `tol` are skipped.
"""
function _extend_cache_general!(cache::ALPACACache{T,R,:general}, matrix::AbstractALPACAMatrix,
                                new_col_pivots::Vector{Int}, tol) where {T,R}
  m = length(cache.cbuf)
  for j in new_col_pivots
    fetch_and_deflate_col_general!(cache, matrix, j)
    best_row, best_val = _aca_next_row(cache, m)
    if best_row == 0 || best_val < tol
      continue
    end
    fetch_and_deflate_row_general!(cache, matrix, best_row)
    _store_general_pivot!(cache, j, best_row)
  end
end
