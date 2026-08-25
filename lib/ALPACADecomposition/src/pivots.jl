"""
ALPACA pivot selection loops for symmetric/hermitian and general matrices.

Combines Adaptive Cross Approximation (ACA) with principal-element
acceleration.  Symmetric matrices support Bunch-Kaufman 2×2 pivoting
for indefinite or zero-diagonal cases.
"""

# ──────────────────────────────────────────────────────────────────
# Smooth pivot scaling helpers
# ──────────────────────────────────────────────────────────────────

"""
    _smoothstep(t) → Float64

Hermite smoothstep: ``3t^2 - 2t^3`` for ``t \\in [0,1]``.
Returns 0 for ``t \\le 0`` and 1 for ``t \\ge 1``.
"""
function _smoothstep(t::Real)
  t <= 0 && return zero(t)
  t >= 1 && return one(t)
  return t * t * (3 - 2 * t)
end

"""
    _smooth_pivot_scale(value, tol, scale_floor) → scale ∈ [0, 1]

Compute a smooth attenuation factor for a pivot value near the threshold.
- `|value| ≥ tol` → 1 (full contribution)
- `|value| ≤ tol * scale_floor` → 0 (no contribution)
- Between: smoothstep interpolation

This ensures that borderline pivots (whose inclusion depends on BLAS
implementation or numerical noise) contribute smoothly rather than
discontinuously, reducing platform sensitivity of the final decomposition.
"""
function _smooth_pivot_scale(value::Real, tol::Real, scale_floor::Real)
  av = abs(value)
  av >= tol && return one(av)
  low = tol * scale_floor
  av <= low && return zero(av)
  return _smoothstep((av - low) / (tol - low))
end

"""
    _apply_smooth_pivot_scaling!(pivot_diag, k, tol, scale_floor)

Scale pivot diagonal values in-place using [`_smooth_pivot_scale`](@ref).
Only pivots `1:k` with `|D| < tol` are affected; pivots well above the
threshold are left unchanged.
"""
function _apply_smooth_pivot_scaling!(pivot_diag::AbstractVector, k::Int, tol::Real, scale_floor::Real)
  @inbounds for i in 1:k
    s = _smooth_pivot_scale(abs(pivot_diag[i]), tol, scale_floor)
    if s < 1
      pivot_diag[i] *= s
    end
  end
end

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

Analytical decomposition of a 2×2 block `B` according to symmetry type `S`:
- `:symmetric` (real): eigendecomposition of symmetric 2×2 matrix
- `:hermitian`: eigendecomposition of Hermitian 2×2 matrix
- `:symmetric` (complex): Takagi factorization of complex symmetric 2×2 matrix

Returns `(values, V)` where `V` is the rotation matrix to be used
by `_attempt_2x2_pivot!`.  For real symmetric and Hermitian, `V` contains
the eigenvectors.  For complex symmetric, `V = conj(U)` where `U` are the
Takagi vectors (``B = U Σ Uᵀ``); the conjugation ensures `B V / Σ = U`,
giving correct ``L D Lᵀ`` factors.
"""
function _eigendecompose_2x2(B::AbstractMatrix{T}, ::Val{:symmetric}) where {T<:Real}
  # F = eigen(Symmetric(B))
  a, d = real(B[1,1]), real(B[2,2])
  b = B[1,2]
  Δ = sqrt((a - d)^2 + 4*b^2)
  λ1 = (a + d - Δ) / 2
  λ2 = (a + d + Δ) / 2
  # Eigenvectors: rotation matrix
  v1 = T[λ1 - d, b]
  v2 = T[λ2 - d, b]
  v1 ./= norm(v1)
  v2 ./= norm(v2)
  if abs(λ1) < abs(λ2)
    V = hcat(v2, v1)
    return T[λ2, λ1], V
  else
    V = hcat(v1, v2)
    return T[λ1, λ2], V
  end
end

function _eigendecompose_2x2(B::AbstractMatrix{T}, ::Val{:hermitian}) where T
  # F = eigen(Hermitian(B))
  a, d = real(B[1,1]), real(B[2,2])
  z = B[1,2]
  Δ = sqrt((a - d)^2 + 4*abs2(z))
  λ1 = (a + d - Δ) / 2
  λ2 = (a + d + Δ) / 2
  v1 = T[T(λ1) - T(d), conj(z)]
  v2 = T[T(λ2) - T(d), conj(z)]
  v1 ./= norm(v1)
  v2 ./= norm(v2)
  if abs(λ1) < abs(λ2)
    V = hcat(v2, v1)
    return T[λ2, λ1], V
  else
    V = hcat(v1, v2)
    return T[λ1, λ2], V
  end
end

# Complex symmetric: Takagi (Autonne–Takagi) factorization B = U Σ Uᵀ.
# For a complex symmetric 2×2 matrix, the Takagi decomposition gives
# real non-negative singular values σ and a unitary U such that B = U Σ Uᵀ.
#
# The caller (_attempt_2x2_pivot!) builds L-columns as [d_j, d_p]·V/λ.
# For the LDLᵀ factorization to be consistent, we need L[pivots,:] = U
# (so LDLᵀ = UΣUᵀ = B).  Since B·conj(U)/Σ = U (Takagi relation), the
# rotation matrix V passed to the caller must be conj(U).
function _eigendecompose_2x2(B::AbstractMatrix{T}, ::Val{:symmetric}) where {T<:Complex}
  σ, U = _takagi_2x2(B)
  return σ, conj.(U)
end

"""
    _takagi_2x2_from_svd(B) → (σ::Vector, U::Matrix)

Reference implementation: Takagi factorization of a 2×2 complex symmetric
matrix via LAPACK SVD.  Given SVD ``B = U_s Σ V_s^†``, the symmetry
``B = B^T`` implies ``\\overline{V_s} = U_s D_φ`` for a diagonal phase matrix
``D_φ``.  The Takagi vectors are ``U_T = U_s \\sqrt{D_φ}``, giving
``B = U_T Σ U_T^T``.

For degenerate singular values (``σ_1 ≈ σ_2``), uses an antilinear
fixed-point iteration ``T(v) = B \\bar v / σ`` whose ``+1`` eigenvectors
are the Takagi vectors.
"""
function _takagi_2x2_from_svd(B::AbstractMatrix{T}) where {T<:Complex}
  RT = real(T)
  F = svd(B)
  σ = F.S
  degen_tol = RT(128) * eps(RT) * max(σ[1], one(RT))
  if abs(σ[1] - σ[2]) < degen_tol
    # Degenerate case: use T-operator approach
    U_takagi = _takagi_degenerate(B, σ[1])
  else
    U_takagi = similar(F.U)
    for k in 1:2
      u = F.U[:, k]
      v = F.V[:, k]
      # Phase: conj(V[:,k]) = U[:,k] * φ_k → φ_k = conj(v[l]) / u[l]
      l = abs(u[1]) >= abs(u[2]) ? 1 : 2
      phase_k = conj(v[l]) / u[l]
      U_takagi[:, k] = u * sqrt(phase_k)
    end
  end
  return T.(σ), U_takagi
end

"""
    _takagi_2x2(B) → (σ::Vector, U::Matrix)

Analytical Takagi factorization of a 2×2 complex symmetric matrix
``B = B^T``.  Returns real non-negative singular values ``σ`` (descending)
and unitary ``U`` such that ``B = U \\operatorname{diag}(σ) U^T``.

Computes eigenvalues of ``\\bar B B`` (Hermitian PSD) analytically to get
``σ_k^2`` and right singular vectors ``v_k``, then derives Takagi vectors
via the SVD phase relation ``Q_k = (B v_k / σ_k) \\sqrt{φ_k}``.

For degenerate singular values (``σ_1 ≈ σ_2``), uses an antilinear
fixed-point iteration ``T(v) = B \\bar v / σ`` whose ``+1`` eigenvectors
are the Takagi vectors.
"""
function _takagi_2x2(B::AbstractMatrix{T}) where {T<:Complex}
  RT = real(T)

  # B̄B is 2×2 Hermitian PSD — eigenvalues give σ², eigenvectors give
  # right singular vectors of B.
  BbarB = B' * B
  h11, h22 = real(BbarB[1,1]), real(BbarB[2,2])
  h12 = BbarB[1,2]
  Δ = sqrt((h11 - h22)^2 + 4 * abs2(h12))
  s2_1 = (h11 + h22 + Δ) / 2                     # larger eigenvalue
  s2_2 = max((h11 + h22 - Δ) / 2, zero(RT))       # smaller eigenvalue
  σ_1 = sqrt(s2_1)
  σ_2 = sqrt(s2_2)

  # Degenerate case: σ₁ ≈ σ₂ → B̄B ≈ σ²I, eigenvectors arbitrary.
  # Use the antilinear T-operator approach instead.
  # Δ suffers catastrophic cancellation when h11 ≈ h22 and h12 ≈ 0,
  # so use a robust threshold (relative to trace of B̄B).
  degen_tol = RT(128) * eps(RT) * max(σ_1, one(RT))
  if abs(σ_1 - σ_2) < degen_tol
    U = _takagi_degenerate(B, σ_1)
    return T[T(σ_1), T(σ_2)], U
  end

  # Eigenvectors of B̄B (Hermitian 2×2)
  if abs(h12) < eps(RT) * max(abs(h11), abs(h22), one(RT))
    # Diagonal B̄B: eigenvectors are standard basis vectors
    if h11 >= h22
      v1 = T[one(T), zero(T)]
      v2 = T[zero(T), one(T)]
    else
      v1 = T[zero(T), one(T)]   # for larger eigenvalue s2_1 = h22
      v2 = T[one(T), zero(T)]
    end
  else
    # Eigenvector for larger eigenvalue s2_1
    v1 = T[s2_1 - h22, conj(h12)]
    v1 ./= norm(v1)
    # Orthogonal eigenvector for smaller eigenvalue s2_2
    v2 = T[-h12, s2_1 - h22]
    v2 ./= norm(v2)
  end

  # Build Takagi vectors from right singular vectors via phase correction.
  # SVD: B v_k = σ_k u_k (left singular vector).
  # For B = Bᵀ: conj(v_k) = u_k φ_k, so Takagi vector Q_k = u_k √φ_k.
  U = Matrix{T}(undef, 2, 2)
  for (col, v, σ_k) in ((1, v1, σ_1), (2, v2, σ_2))
    if σ_k < eps(RT) * max(σ_1, one(RT))
      # Zero singular value: orthogonal complement of other Takagi vector
      if col == 2
        U[:, 2] = T[-conj(U[2,1]), conj(U[1,1])]
      else
        U[:, 1] = v
      end
      continue
    end
    u_svd = B * v / σ_k                    # left singular vector
    l = abs(u_svd[1]) >= abs(u_svd[2]) ? 1 : 2
    phase_k = conj(v[l]) / u_svd[l]        # conj(v) = u * phase
    U[:, col] = u_svd * sqrt(phase_k)
  end

  return T[T(σ_1), T(σ_2)], U
end

"""
    _takagi_degenerate(B, σ) → U::Matrix

Compute Takagi vectors for a 2×2 complex symmetric matrix with degenerate
singular values ``σ_1 ≈ σ_2 ≈ σ``.

Uses the antilinear operator ``T(v) = B \\bar v / σ``, which satisfies
``T^2 = I`` (since ``\\bar B B = σ^2 I``).  The ``+1`` eigenvectors of
``T`` give the Takagi vectors.  For a starting vector ``v``, the projection
``v + T(v)`` lands in the ``+1`` eigenspace.  If the orthogonal complement
falls in the ``-1`` eigenspace, multiplying by ``i`` flips the sign
(``T(iv) = -iT(v) = iv`` when ``T(v)=-v``).
"""
function _takagi_degenerate(B::AbstractMatrix{T}, σ::Real) where {T<:Complex}
  U = Matrix{T}(undef, 2, 2)
  RT = real(T)

  if σ < eps(RT)
    # B ≈ 0: any unitary works
    U[:, 1] = T[one(T), zero(T)]
    U[:, 2] = T[zero(T), one(T)]
    return U
  end

  # First Takagi vector: project e₁ onto +1 eigenspace of T
  v0 = T[one(T), zero(T)]
  Tv = B * conj.(v0) / σ
  u1 = v0 + Tv
  if norm(u1) < RT(0.1)
    # e₁ is (nearly) in the −1 eigenspace; try e₂
    v0 = T[zero(T), one(T)]
    Tv = B * conj.(v0) / σ
    u1 = v0 + Tv
  end
  u1 ./= norm(u1)
  U[:, 1] = u1

  # Second Takagi vector: start from orthogonal complement of u1,
  # project onto +1 eigenspace, then re-orthogonalize.
  v0 = T[-conj(u1[2]), conj(u1[1])]
  Tv = B * conj.(v0) / σ
  u2 = v0 + Tv
  nrm = norm(u2)
  if nrm < RT(0.1)
    # Orthogonal complement is in −1 eigenspace of T.
    # Multiply by i: T(iv) = −iT(v) = −i(−v) = iv, so iv is in +1 eigenspace.
    u2 = T(im) * v0
    Tv = B * conj.(u2) / σ
    u2 = u2 + Tv
  end
  u2 -= (U[:, 1]' * u2) * U[:, 1]     # Gram-Schmidt
  u2 ./= norm(u2)
  U[:, 2] = u2

  return U
end

"""
    _attempt_2x2_pivot!(cache, matrix, j, partner, deflated_j, pivotol) → Bool

Attempt a 2×2 Bunch-Kaufman pivot using columns `j` and `partner`.

Column `j` must already be fetched+deflated (passed as `deflated_j`).
Fetches column `partner`, computes the 2×2 intersection block in the
deflated basis, decomposes it (eigendecomposition for real symmetric /
Hermitian, Takagi factorization for complex symmetric), and stores
two rotated rank-1 pivots with the eigenvalues / singular values as
pivot diagonal values.

The rotated (deflated, scaled) columns are stored in `cache.columns` for
correct subsequent deflation and raw-factor (`lpaca`) reconstruction.

For complex symmetric matrices, uses the Takagi factorization
``B = U Σ Uᵀ``.  The rotation uses ``\bar{U}`` (complex conjugate of
Takagi vectors) so that ``B \bar{U} / Σ = U``, producing correct
``L D Lᵀ`` factors at the pivot positions.

Returns `true` if at least one valid pivot was stored.
"""
function _attempt_2x2_pivot!(cache::ALPACACache{T,R,S},
                              matrix::AbstractALPACAMatrix,
                              j::Int, partner::Int,
                              deflated_j::Vector{T},
                              tol::R) where {T,R,S}
  n = length(cache.cbuf)

  # Partner must not already be a pivot
  cache.is_pivot[partner] && return false

  # Fetch and deflate the partner column (dispatches hermitian/symmetric).
  fetch_and_deflate_symmetric!(cache, matrix, partner)
  deflated_partner = copy(cache.cbuf)

  # 2×2 intersection block M_remaining[{j,partner}, {j,partner}]:
  #   B[r,c] = deflated column c, row r.
  B = T[deflated_j[j]       deflated_partner[j];
        deflated_j[partner]  deflated_partner[partner]]

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
function _aca_candidate_symmetric(cache::ALPACACache{T,R}, n::Int, equiv_tol::Real=zero(R)) where {T,R}
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
      if mag > best_mag + equiv_tol
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
        if mag > best_mag + equiv_tol
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
function _aca_next_col(cache::ALPACACache{T,R,:general}, ncols::Int, equiv_tol::Real=zero(R)) where {T,R}
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
      if mag > best_mag + equiv_tol
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
function _aca_next_row(cache::ALPACACache{T}, m::Int, equiv_tol::Real=zero(real(T))) where T
  best_row = 0
  best_val = zero(real(T))
  @inbounds for i in 1:m
    if !cache.is_row_pivot[i]
      v = abs(cache.cbuf[i])
      if v > best_val + equiv_tol
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
  tol = resolve_pivotol(options, n)
  cache.pivotol = tol
  # Smooth pivot scaling: disable when max_rank is explicitly set — a fixed-rank
  # truncation already pins the set of retained pivots (matches LLAMA's behavior).
  smooth_floor = (options.max_rank < typemax(Int)) ? zero(R) : R(options.smooth_tol)
  use_smooth = smooth_floor > 0
  # Extended threshold: accept pivots down to tol * smooth_floor (they will be attenuated)
  tol_ext = use_smooth ? tol * smooth_floor : tol
  tol2 = abs2(tol_ext)
  max_rank = options.max_rank
  # Equivalence tolerance for stable pivot selection: among candidates within
  # this tolerance of each other, prefer the lower index (original order).
  # Use resolved pivotol scale since all comparisons are against tol or tol2.
  equiv_tol = R(tol_ext / 10)

  # Initialize principal values
  init_principal_values!(cache, matrix, descriptor)

  last_cold_start_k = -1  # guard against repeated cold starts at the same rank

  for iter in 1:min(n, max_rank)
    k = cache.n_cols  # number of pivots so far

    # ── Candidate selection: principal first, ACA as fallback ──
    prin_best_slot = _argmax_principal(cache, equiv_tol)
    prin_magnitude = prin_best_slot > 0 ?
      abs(cache.principal_values[prin_best_slot]) : zero(R)

    local next_j::Int
    local from_principal_diagonal::Bool = false
    if prin_magnitude >= (k == 0 ? tol2 : tol_ext) && prin_best_slot > 0
      # Principal: pick either index from the best pair (both are non-pivots,
      # guaranteed by _argmax_principal)
      prin_i, next_j = cache.principal_pairs[prin_best_slot]
      from_principal_diagonal = (prin_i == next_j)
    else
      # Fall back to ACA: argmax |residual| in most recent stored column(s)
      aca_best_idx, aca_magnitude = _aca_candidate_symmetric(cache, n, equiv_tol)
      if aca_best_idx > 0 && aca_magnitude >= tol_ext
        next_j = aca_best_idx
      else
        # Cold start: both principal and ACA failed (e.g. zero-diagonal matrix).
        # Pick the first non-pivot column so the 1×1/2×2 decision logic can
        # inspect it.  Break if we already tried at this rank (no progress).
        k == last_cold_start_k && break
        last_cold_start_k = k
        cold_j = 0
        @inbounds for p in 1:n
          if !cache.is_pivot[p]
            cold_j = p
            break
          end
        end
        cold_j == 0 && break
        next_j = cold_j
      end
    end

    # ── Fetch and deflate column next_j ──
    fetch_and_deflate_symmetric!(cache, matrix, next_j)

    # ── 1×1 vs 2×2 pivot decision ──
    if from_principal_diagonal
      # Principal diagonal element: the diagonal residual was already the
      # largest monitored value, so 1×1 pivot is the right choice.
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
        if v > offdiag_val + equiv_tol
          offdiag_val = v
          offdiag_idx = p
        end
      end

      if diag_val < tol_ext && offdiag_val < tol_ext
        # Entire column is negligible — no more rank available.
        # ACA already selected the best candidate; bail out.
        break
      end

      if offdiag_idx == 0 || diag_val >= max((one(R) - 5 * tol_ext) * offdiag_val, tol_ext)
        # Diagonal is large enough for a stable 1×1 pivot
        _store_symmetric_pivot!(cache, next_j)
      else
        # Off-diagonal is significantly larger → 2×2 Bunch-Kaufman
        if !_attempt_2x2_pivot!(cache, matrix, next_j, offdiag_idx,
                                copy(cache.cbuf), tol_ext)
          # 2×2 failed (both eigenvalues below tol_ext); try 1×1 if diagonal is usable
          if diag_val >= tol_ext
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
  R = real(T)
  m = length(cache.cbuf)       # number of rows in the matrix
  ncols = length(cache.rbuf)   # number of columns in the matrix
  tol = resolve_pivotol(options, m)
  cache.pivotol = tol
  # Smooth pivot scaling: disable when max_rank is explicitly set — a fixed-rank
  # truncation already pins the set of retained pivots (matches LLAMA's behavior).
  smooth_floor = (options.max_rank < typemax(Int)) ? zero(R) : R(options.smooth_tol)
  use_smooth = smooth_floor > 0
  tol_ext = use_smooth ? tol * smooth_floor : tol
  tol2 = abs2(tol_ext)
  max_rank = options.max_rank
  # Equivalence tolerance for stable pivot selection: use resolved pivotol scale
  equiv_tol = R(tol_ext / 10)

  # Initialize principal values
  init_principal_values!(cache, matrix, descriptor)

  last_cold_start_k = -1  # guard against repeated cold starts at the same rank

  for iter in 1:min(m, ncols, max_rank)
    k = cache.n_cols

    # ── Principal proposal ──
    prin_slot = _argmax_principal(cache, equiv_tol)
    prin_magnitude = prin_slot > 0 ?
      abs(cache.principal_values[prin_slot]) : zero(R)

    local next_i::Int, next_j::Int

    if prin_magnitude >= (k == 0 ? tol2 : tol_ext) && prin_slot > 0
      # Principal gives both row and column directly
      prin_i, prin_j = cache.principal_pairs[prin_slot]
      if !cache.is_row_pivot[prin_i] && !cache.is_pivot[prin_j]
        next_i = prin_i
        next_j = prin_j

        # Principal path: fetch both column and row
        fetch_and_deflate_col_general!(cache, matrix, next_j)
        fetch_and_deflate_row_general!(cache, matrix, next_i)

        _store_general_pivot!(cache, next_j, next_i)
        continue
      end
      # One of them is already used; fall through to ACA
    end

    # ── ACA fallback: argmax of last stored row → next column ──
    aca_col, _ = _aca_next_col(cache, ncols, equiv_tol)

    if aca_col > 0 && !cache.is_pivot[aca_col]
      # ACA: maxabs of last row gave us a column; fetch+deflate it,
      # then maxabs of this column gives us a row.
      # The actual stopping criterion is the max element of the
      # freshly deflated column
      next_j = aca_col
      fetch_and_deflate_col_general!(cache, matrix, next_j)

      # Find best unused row from the deflated column
      next_i, best_i_val = _aca_next_row(cache, m, equiv_tol)
      if next_i == 0 || best_i_val < tol_ext
        break
      end

      # Fetch and deflate the row
      fetch_and_deflate_row_general!(cache, matrix, next_i)

      # Pivot value at intersection
      if abs(cache.cbuf[next_i]) < tol_ext
        break
      end
      _store_general_pivot!(cache, next_j, next_i)
    else
      # Cold start: both principal and ACA failed (e.g. zero-diagonal matrix).
      # Pick the first non-pivot column so we can start the ACA chain.
      k == last_cold_start_k && break
      last_cold_start_k = k
      cold_j = 0
      @inbounds for p in 1:ncols
        if !cache.is_pivot[p]
          cold_j = p
          break
        end
      end
      cold_j == 0 && break

      fetch_and_deflate_col_general!(cache, matrix, cold_j)

      cold_i, cold_val = _aca_next_row(cache, m, equiv_tol)
      if cold_i == 0 || cold_val < tol_ext
        break
      end

      fetch_and_deflate_row_general!(cache, matrix, cold_i)

      if abs(cache.cbuf[cold_i]) < tol_ext
        break
      end
      _store_general_pivot!(cache, cold_j, cold_i)
    end
  end

  return cache.pivot_indices, cache.row_pivot_indices
end

# ──────────────────────────────────────────────────────────────────────────────
# Post-pivot-loop column scaling
# ──────────────────────────────────────────────────────────────────────────────

"""
    _scale_pivot_columns!(cache; from=1)

Transform the raw pivot-loop factors in-place so that downstream code
can use `cache.columns` directly without additional scaling.
Only scales columns `from:k` where `k = cache.n_cols`.

Dispatches on the cache symmetry type `S` and element type `T`.
"""
function _scale_pivot_columns! end

# General: L_C *= d, so M ≈ L_C * L_R^T
function _scale_pivot_columns!(cache::ALPACACache{T,R,:general}; from::Int=1) where {T,R}
  k = cache.n_cols
  @inbounds for t in from:k
    d = cache.pivot_diag[t]
    @views cache.columns[:, t] .*= d
    cache.pivot_diag[t] = one(T)
  end
end

# Complex symmetric: L *= √d, so M ≈ L L^T
function _scale_pivot_columns!(cache::ALPACACache{<:Complex,R,:symmetric}; from::Int=1) where R
  k = cache.n_cols
  @inbounds for t in from:k
    sd = sqrt(Complex(cache.pivot_diag[t]))
    @views cache.columns[:, t] .*= sd
    cache.pivot_diag[t] = one(eltype(cache.pivot_diag))
  end
end

# Real symmetric: L *= √|d|, d ← sign(d)
function _scale_pivot_columns!(cache::ALPACACache{<:Real,R,:symmetric}; from::Int=1) where R
  k = cache.n_cols
  @inbounds for t in from:k
    d = cache.pivot_diag[t]
    @views cache.columns[:, t] .*= sqrt(abs(d))
    cache.pivot_diag[t] = d < zero(R) ? -one(R) : one(R)
  end
end

# Hermitian: L *= √|Re(d)|, d ← sign(Re(d))
function _scale_pivot_columns!(cache::ALPACACache{T,R,:hermitian}; from::Int=1) where {T,R}
  k = cache.n_cols
  @inbounds for t in from:k
    rd = real(cache.pivot_diag[t])
    @views cache.columns[:, t] .*= sqrt(abs(rd))
    cache.pivot_diag[t] = rd < zero(R) ? -one(T) : one(T)
  end
end

"""
    _argmax_principal(cache) → slot index (0 if empty)

Return the index into cache.principal_pairs/values with the largest |residual| 
among non-pivot entries.

For `:general` caches, checks `is_row_pivot` for the first index
and `is_pivot` for the second.  For symmetric-like caches, checks
`is_pivot` for both.
"""
function _argmax_principal(cache::ALPACACache{T,R,:general}, equiv_tol::Real=zero(R)) where {T,R}
  best_slot = 0
  best_val = zero(real(T))
  @inbounds for p in eachindex(cache.principal_values)
    a, b = cache.principal_pairs[p]
    (cache.is_row_pivot[a] || cache.is_pivot[b]) && continue
    v = abs(cache.principal_values[p])
    if v > best_val + equiv_tol
      best_val = v
      best_slot = p
    end
  end
  return best_slot
end

function _argmax_principal(cache::ALPACACache{T}, equiv_tol::Real=zero(real(T))) where T
  best_slot = 0
  best_val = zero(real(T))
  @inbounds for p in eachindex(cache.principal_values)
    a, b = cache.principal_pairs[p]
    (cache.is_pivot[a] || cache.is_pivot[b]) && continue
    v = abs(cache.principal_values[p])
    if v > best_val + equiv_tol
      best_val = v
      best_slot = p
    end
  end
  return best_slot
end
