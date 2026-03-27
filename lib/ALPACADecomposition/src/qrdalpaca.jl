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
    _fetch_elements!(dest, matrix::AbstractALPACAMatrix, row_indices, col_indices, buf)

Fill `dest[l, t] = matrix[row_indices[l], col_indices[t]]` via `elements!`.
"""
function _fetch_elements!(dest::AbstractMatrix{T}, matrix::AbstractALPACAMatrix,
                          row_indices, col_indices,
                          buf::AbstractVector{T}) where T
  pairs = Tuple{Int,Int}[]
  for t in eachindex(col_indices)
    for l in eachindex(row_indices)
      push!(pairs, (row_indices[l], col_indices[t]))
    end
  end
  resize!(buf, length(pairs))
  elements!(buf, matrix, pairs)
  k = 0
  nr = length(row_indices)
  @inbounds for t in eachindex(col_indices)
    for l in 1:nr
      k += 1
      dest[l, t] = buf[k]
    end
  end
  return dest
end

"""
    qrdalpaca(matrix; principal=nothing, options=ALPACAOptions(...))

Matrix-free ALPACA decomposition with QR-pivoted refinement.

Runs ALPACA, then checks remaining columns via column-pivoted QR for any
significant columns the greedy phase may have missed.  Returns an
`ALPACAResult` with the combined pivot set.

# Algorithm
1. Run `alpaca(matrix; ...)` to obtain initial pivots.
2. Build an orthonormal basis ``Q`` from the selected columns (thin SVD).
3. Compute residual column norms ``\\|M_{:,j} - Q Q^\\dagger M_{:,j}\\|^2``
   for every non-pivot column ``j``.
4. Repeatedly select batches of columns with large residuals, run
   column-pivoted QR, and extend ``Q`` with any newly discovered pivots.
5. Re-finalize: Nyström (symmetric/hermitian) or SVD (general).
"""
function qrdalpaca(matrix::AbstractALPACAMatrix;
                   principal=nothing,
                   options::ALPACAOptions)
  # ── Step 1: ALPACA pilot ──
  result = alpaca(matrix; principal, options)
  alpaca_pivots = result.pivot_indices
  k_alpaca = length(alpaca_pivots)

  T = _matrix_eltype(matrix)
  RT = real(T)
  sym = options.symmetry
  sig = options.sigma
  tol_val = options.pivotol
  m, n = size(matrix)
  max_pivots = min(m, n, options.max_rank)

  if k_alpaca == 0 || k_alpaca >= max_pivots
    return result
  end

  # ── Step 2+3: Quick reconstruction pre-check + QR projection fallback ──
  cbuf = Vector{T}(undef, m)
  L = result.left
  R = result.right
  n_basis = size(L, 2)

  is_pivot = falses(n)
  @inbounds for p in alpaca_pivots
    is_pivot[p] = true
  end

  tol2 = tol_val^2
  col_norms2 = zeros(RT, n)
  non_pivots = [j for j in 1:n if !is_pivot[j]]
  nn = length(non_pivots)

  # Quick reconstruction pre-check: M[:,j] ≈ L S L[j,:] (symmetric) or L R[j,:] (general).
  # Sample non-pivot columns and compare reconstruction errors against tol.
  # If all sampled residuals are small, skip the expensive QR refinement.
  #
  # Sample size for 99.9% confidence of detecting missed columns:
  #   P(detect ≥ 1 from d missed in N) = 1 - (1 - d/N)^k ≥ 0.999
  #   ⟹ k ≥ -ln(0.001) / (d/N) ≈ 6.908 N/d
  # With d = √N:  k ≈ 7√N  (detects ≥ √N missed columns)
  # With d = 0.01N: k ≈ 700 (detects ≥ 1% missed columns)
  n_sample = min(nn, max(ceil(Int, 7*sqrt(nn)), 700))
  if n_sample > 0
    if sym == :general
      # General: M ≈ L R', nothing extra needed
    else
      signs = ones(RT, n_basis)
      signs[result.neg_indices] .= -one(RT)
      L_S = L .* transpose(signs)
    end

    # Compute reconstruction coefficients for a batch of columns
    _recon_coeffs(indices) = if sym == :general
      R[indices, :]'                 # adjoint: M ≈ L R'
    elseif sym == :hermitian
      L_S[indices, :]'              # adjoint: M ≈ L L'
    else
      transpose(L_S[indices, :])    # transpose: M ≈ L Lᵀ
    end

    # Threshold: compare max absolute element deviation directly against tol.
    # After Nyström finalization the factors are stable, consistent with
    # ALPACA's element-wise pivot tolerance.
    recon_threshold = tol_val

    # Sample non-pivot columns
    sample_idx = if n_sample < nn
      non_pivots[sort(randperm(nn)[1:n_sample])]
    else
      non_pivots
    end
    cols_true = Matrix{T}(undef, m, length(sample_idx))
    _fetch_columns!(cols_true, matrix, sample_idx, cbuf)
    cols_true .-= L * _recon_coeffs(sample_idx)
    recon_ok = true
    @inbounds for i in eachindex(sample_idx)
      if maximum(abs, @view cols_true[:, i]) >= recon_threshold
        recon_ok = false
        break
      end
    end
    if recon_ok
      return result
    end
  end

  # Reconstruction check failed → compute QR for precise projection residuals.
  # Projection residual ||col||² - ||Q'col||² measures component outside span(L)
  # and is numerically stable with threshold tol².
  F_qr_basis = qr(L)
  n_acc = n_basis
  cap = max(2 * n_basis, 64)
  Q_acc = Matrix{T}(undef, m, cap)
  @views Q_acc[:, 1:n_acc] .= Matrix(F_qr_basis.Q)[:, 1:n_acc]

  # Compute projection residuals for all non-pivot columns
  batch_size = max(nn, 256)
  for start in 1:batch_size:nn
    stop = min(start + batch_size - 1, nn)
    batch_idx = non_pivots[start:stop]
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
    return result
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
    return result
  end

  # ── Step 5: Re-finalize with combined pivots ──
  all_pivots = vcat(alpaca_pivots, refinement_pivots)

  if sym == :general
    return _general_refactorize(matrix, all_pivots, result.row_pivots, options.tol, cbuf)
  else
    return _symmetric_refactorize(matrix, all_pivots, options.tol, sym, cbuf)
  end
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
    _symmetric_refactorize(matrix, pivots, tol, sym, cbuf) → ALPACAResult

Nyström re-finalization from combined pivot set for symmetric/hermitian matrices.
Fetches data via the matrix-free interface, then delegates to `_nystrom_from_CJ`.
"""
function _symmetric_refactorize(matrix::AbstractALPACAMatrix, pivots::Vector{Int},
                                tol, sym::Symbol, cbuf::AbstractVector{T}) where T
  k = length(pivots)
  m = size(matrix)[1]

  C = Matrix{T}(undef, m, k)
  _fetch_columns!(C, matrix, pivots, cbuf)

  J = Matrix{T}(undef, k, k)
  ebuf = Vector{T}(undef, k * k)
  _fetch_elements!(J, matrix, pivots, pivots, ebuf)

  return _nystrom_from_CJ(C, J, pivots, tol, sym)
end

"""
    _general_refactorize(M, col_pivots, row_pivots, tol, cbuf) → ALPACAResult

SVD factorization from combined pivot set for general matrices.
Uses the matrix-free interface for column/row/element access.
"""
function _general_refactorize(matrix::AbstractALPACAMatrix, col_pivots::Vector{Int},
                              row_pivots::Vector{Int}, tol,
                              cbuf::AbstractVector{T}) where T
  m, n = size(matrix)
  k = length(col_pivots)

  C = Matrix{T}(undef, m, k)
  _fetch_columns!(C, matrix, col_pivots, cbuf)

  # For general: we need row pivots too.  Use the original ALPACA row pivots
  # where available, and use the col_pivots as row indices for any extras.
  # J = M[row_pivots, col_pivots] — but row_pivots may be shorter than col_pivots
  # after QR added new col pivots.  Pad with pivot-column-based row selection.
  kr = length(row_pivots)
  if kr < k
    # Need more row pivots. Use the rows that have max absolute value
    # in the new columns.
    is_row = falses(m)
    for r in row_pivots
      is_row[r] = true
    end
    new_cols = C[:, kr+1:k]
    extra_rows = Int[]
    for t in 1:k-kr
      best_row = 0
      best_val = -one(real(T))
      col_t = @view new_cols[:, t]
      for i in 1:m
        if !is_row[i]
          v = abs(col_t[i])
          if v > best_val
            best_val = v
            best_row = i
          end
        end
      end
      if best_row > 0
        push!(extra_rows, best_row)
        is_row[best_row] = true
      end
    end
    row_pivots = vcat(row_pivots, extra_rows)
  end

  rp = row_pivots[1:k]

  J = Matrix{T}(undef, k, k)
  ebuf = Vector{T}(undef, k * k)
  _fetch_elements!(J, matrix, rp, col_pivots, ebuf)

  # RT[:, t] = M[row_pivots[t], :]  →  fetch rows via row!
  rbuf = Vector{T}(undef, n)
  RT = Matrix{T}(undef, n, k)
  @inbounds for t in 1:k
    row!(rbuf, matrix, rp[t])
    RT[:, t] .= rbuf
  end

  return _svd_from_CJ_RT(C, J, RT, col_pivots, rp, tol)
end
