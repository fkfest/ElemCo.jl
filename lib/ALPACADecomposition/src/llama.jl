# ──────────────────────────────────────────────────────────────────
# LLAMA: Left Lowrank Amended Matrix Approximation
# ──────────────────────────────────────────────────────────────────
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║                        OVERVIEW                              ║
# ╚═══════════════════════════════════════════════════════════════╝
#
# Given a general matrix A (m×n) with numerical rank r ≪ min(m,n),
# compute an orthonormal basis Q (m×r) for the column space of A
# using only O(r) row and column accesses.
#
# The user must supply d_row = diag(AA') as a precomputed guidance
# vector to steer row pivot selection toward high-norm rows.
#
# LLAMA extends ALPACA's cross-coupled Schur complement deflation
# for general (non-symmetric) matrices.  The key insight is to
# track a residual norm indicator for each row, derived from d_row,
# that is deflated each iteration.  This guides the algorithm toward
# rows carrying the most uncaptured information.
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║                     ALGORITHM SKETCH                         ║
# ╚═══════════════════════════════════════════════════════════════╝
#
# The algorithm has two nested loops: an OUTER loop for SVD-based
# correction and an INNER loop for Gram-guided row/column pivoting.
#
# ── Inner loop (ACA-style cross-coupled deflation) ──
#
#   1. Initialize residual[i] = d_row[i] for all rows.
#
#   2. Repeat:
#      a. Pick the row i* with the largest residual.
#         If max residual < tol², done — break to finalization.
#      b. Fetch row A[i*,:] and deflate using stored columns:
#            r̃ = A[i*,:] − R · (D · C[i*,:])
#      c. Pick the column j* where |r̃[j]| is largest.
#         If |r̃|∞ < tol, mark i* as "exhausted" (already well-
#         captured) and try the next row.
#      d. Fetch column A[:,j*] and deflate using stored rows:
#            c̃ = A[:,j*] − C · (D · R[j*,:])
#      e. Record the pivot value p = c̃[i*] and store:
#            C[:,k] = c̃/p,  R[:,k] = r̃/p,  D[k] = p.
#      f. Update all residuals via the Gram-corrected formula
#         (see "Gram-corrected residual" below).
#
# ── SVD finalization ──
#
#   3. Factor the stored columns via Cholesky of the column
#      Gram matrix (rank×rank, much cheaper than full QR on
#      m×rank):
#        R_L = chol(C^H C)
#      Then SVD of the rank×n matrix B = R_L D R^T:
#        B = U_B Σ V_B^H
#      Truncate at tol:  Q = C · R_L⁻¹ · U_B[:,1:nk].
#
# ── SVD-corrected residual (outer loop) ──
#
#   4. After the SVD, LLAMA checks whether any rank was missed
#      by computing a corrected residual.  The key formula is:
#
#        P = Q[pivots,:]^H Q[pivots,:]       (nk × nk)
#        G̃_A = P Σ² P                        (estimated Gram)
#        corrected[i] = d_row[i] − qᵢᴴ G̃_A qᵢ
#
#      If any corrected[i] > tol², re-enter the inner loop with
#      the corrected residuals to find more pivots.  Stop after
#      at most 10 outer iterations or when no new pivots are found.
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║               GRAM-CORRECTED RESIDUAL                        ║
# ╚═══════════════════════════════════════════════════════════════╝
#
# The tracked residual quantity is:
#
#   residual[i] = ||A[i,:]||² − ||approx[i,:]||²
#                = d_row[i] − ||C[i,:] D R^T[i,:]||²
#
# This is NOT the same as the true error ||A[i,:] − approx[i,:]||².
# They differ by a cross-term 2 Re⟨E, approx⟩.  For orthogonal
# projections (like the SVD), the cross-term is zero and the two
# coincide.  But for ACA (which is more like an LU-based scheme),
# the cross-term can make the residual dip below zero in individual
# rows even when the true error is large.
#
# Mathematically, for row i:
#
#   ||A[i,:]||² = ||E[i,:]||² + ||approx[i,:]||² + 2 Re⟨E[i,:], approx[i,:]⟩
#
# Rearranging:
#
#   ||A[i,:]||² − ||approx[i,:]||² = ||E[i,:]||² + 2 Re⟨E[i,:], approx[i,:]⟩
#
# The left side (which we track) can be negative when the cross-term
# is large and negative.  We clamp to zero, but this means the
# residual might indicate "converged" when there's still real error.
#
# This overshoot phenomenon affects ANY non-SVD decomposition
# (LU, ACA, QR without full re-orthogonalization) — it is not
# specific to LLAMA.
#
# The Gram update is computed incrementally.  When storing column k,
# we compute the row Gram entries G_{tk} = ⟨R[:,t], R[:,k]⟩ and
# update each residual:
#
#   Δ[i] = |d_k C[i,k]|² G_{kk} + 2 Re(d̄_k C̄[i,k] Σ_{t<k} d_t C[i,t] G_{tk})
#   residual[i] ← max(residual[i] − Δ[i], 0)
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║             WHY THE NAIVE SVD CORRECTION FAILS               ║
# ╚═══════════════════════════════════════════════════════════════╝
#
# A natural idea after SVD finalization: compute
#
#   d_row[i] − Σ_k σ_k² |Q[i,k]|²
#
# as a corrected residual.  This LOOKS like it subtracts the energy
# captured by Q — but it doesn't work.
#
# The reason: Q comes from the SVD of CDR^H (the ACA approximation),
# not from A itself.  Since CDR^H = Q Σ V^H Q_R^H, we have
# ||CDR^H[i,:]||² = Σ_k σ_k² |Q[i,k]|² (by orthonormality of V Q_R).
# So the formula reduces to:
#
#   d_row[i] − ||CDR^H[i,:]||²
#
# This is EXACTLY the Gram residual we already tracked!  The SVD
# didn't add any information.  To compute the TRUE projection error
# ||A − QQ^H A||² per row, we'd need Q^H A (an r×n matrix), which
# requires accessing all m rows of A — defeating the purpose.
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║    THE ACCESSED-ROW GRAM ESTIMATE (P Σ² P CORRECTION)       ║
# ╚═══════════════════════════════════════════════════════════════╝
#
# LLAMA's solution: estimate the true Gram matrix G_A = Q^H A A^H Q
# using ONLY the rows already accessed during pivoting.
#
# Key observation (ACA interpolation property):
#   For every pivot row i_s:  A[i_s,:] = CDR^H[i_s,:]  (exactly)
#
# This lets us compute a partial estimate of Q^H A:
#
#   (Q^H A)_partial = Q[pivots,:]^H · A[pivots,:]
#                   = Q[pivots,:]^H · CDR^H[pivots,:]    (by interpolation)
#                   = Q[pivots,:]^H · Q[pivots,:] · Σ V^H Q_R^H
#                   = P · Σ V^H Q_R^H
#
# where P = Q[pivots,:]^H Q[pivots,:] is the pivot-row "energy
# capture" matrix (nk × nk).  Since Q has orthonormal columns with
# energy spread across all m rows, and we only use the r pivot rows:
#
#   P ⪯ I    (positive semidefinite, dominated by identity)
#
# The estimated Gram matrix is:
#
#   G̃_A = (Q^H A)_partial · (Q^H A)_partial^H = P Σ² P
#
# And the corrected residual:
#
#   corrected[i] = d_row[i] − q_i^H (P Σ² P) q_i
#
# Since P ⪯ I, we have P Σ² P ⪯ Σ², so the subtraction is LESS
# aggressive than the Gram formula.  This means:
#
# - Rows that are genuinely converged (small true error) have
#   corrected residual ≈ 0 (P captures most of their energy).
# - Rows where the Gram overshot have corrected residual > 0
#   (P doesn't fully capture the energy from non-pivot rows,
#   leaving a positive residual that triggers re-exploration).
#
# Example: for a block-diagonal matrix with 3 blocks, each rank 3:
# - Block 2 has 10 rows but only 2 pivot rows
# - P's block-2 diagonal ≈ 2/10 = 0.2
# - The correction subtracts only ~4% (= 0.2²) of the Gram value
# - So block-2 rows' corrected residual stays positive → they
#   are re-explored in the next outer pass → missing component found
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║               EXHAUSTED-ROW SKIP LOGIC                       ║
# ╚═══════════════════════════════════════════════════════════════╝
#
# When a row is selected by residual but its deflated content is
# near-zero (|r̃|∞ < tol or |p| < tol), the row is "exhausted" —
# its information is already captured.  LLAMA marks the row and
# uses a `needs_recompute` flag to decide the next action:
#
# - If `needs_recompute` is true (i.e., new Gram updates have
#   occurred since the last recomputation), LLAMA recomputes ALL
#   non-pivot residuals from scratch using the stored ACA factors
#   and the row Gram matrix, then continues the inner loop.
# - If `needs_recompute` is false (residuals were already
#   recomputed after the most recent Gram update), the exhaustion
#   is genuine and the inner loop breaks to finalization.
#
# The flag is set to true after each successful pivot (which adds
# a Gram update that may overshoot residuals) and reset to false
# after each recomputation.  This limits the expensive O(m·r²)
# recomputation to at most r times total (once per successful
# pivot), while still ensuring that Gram-induced false convergence
# is detected.
#
# This is essential for non-square matrices where the Gram
# cross-terms accumulate quickly: after a few pivots, many rows
# appear converged (residual clamped to zero) even though their
# true error is significant.  Recomputation restores the correct
# values, allowing the algorithm to discover all components.
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║                    NOTATION                                  ║
# ╚═══════════════════════════════════════════════════════════════╝
#
#   C = cols_store (m × rank): stored scaled deflated columns
#   R = rows_store (n × rank): stored scaled deflated rows
#   D = diag(pivot_diag[1:rank]): pivot values
#   Approximation: A ≈ C D R^H
#
# Column deflation: c̃ = A[:,j] − C · (D · R[j,:])
# Row deflation:    r̃ = A[i,:] − R · (D · C[i,:])
#
# Complexity:
#   Accesses: r rows + r columns (plus exhausted-row overhead).
#   Memory:   O(m·r + n·r) for stored columns and rows,
#             O(r²) for the incrementally accumulated row Gram.
#   Per-step: O((m+n)·k) for two BLAS-2 gemv deflations
#             + O(n·k) for row Gram entries (via BLAS gemv).
#   Total:    O((m+2n)·r²) ≈ O((m+n)·r²) when m ≈ n.
#
# Finalization (Cholesky + SVD):
#   O(m·r²) for column Gram C^H C.  Row Gram R^H R is accumulated
#   incrementally during the inner loop at no extra finalization cost.
#   O(r³) for Cholesky + SVD of the rank×rank core.

using LinearAlgebra: mul!, svd, Diagonal, Hermitian, cholesky!

"""
    LLAMAResult{T, R}

Result of a LLAMA column-space decomposition.

# Fields
- `Q::Matrix{T}`: orthonormal column-space basis (``m \\times r``).
- `singular_values::Vector{R}`: approximate singular values from the
  SVD finalization of the ``r \\times n`` matrix ``B = R_L D R^T``.
- `col_pivots::Vector{Int}`: column indices fetched from `A` (length ``r``).
- `row_pivots::Vector{Int}`: row indices used as successful pivots (length ``r``).
- `V::Union{Nothing, Matrix{T}}`: right singular vectors (``n \\times r``).
  Populated only when `fullsvd=true`; otherwise `nothing`.
  When present, ``A \\approx Q \\, \\mathrm{diag}(\\text{singular\\_values}) \\, V^H``.
"""
struct LLAMAResult{T, R<:Real}
  Q::Matrix{T}
  singular_values::Vector{R}
  col_pivots::Vector{Int}
  row_pivots::Vector{Int}
  V::Union{Nothing, Matrix{T}}
end

"""Grow buffer matrix by doubling along dimension `dim` if `needed` exceeds capacity."""
function _ensure_capacity!(buf::Matrix{T}, needed::Int, dim::Int) where T
  current = size(buf, dim)
  needed <= current && return buf
  new_cap = max(2 * current, needed)
  if dim == 2
    new_buf = Matrix{T}(undef, size(buf, 1), new_cap)
    @views new_buf[:, 1:current] .= buf
  else
    new_buf = Matrix{T}(undef, new_cap, size(buf, 2))
    @views new_buf[1:current, :] .= buf
  end
  return new_buf
end

"""
    _llama_finalize(cols_store, rows_store, pivot_diag, row_gram, rank, tol_rt, fullsvd)

SVD finalization of the ACA factors.

**Fast path** (`fullsvd=false` and real `T`): Cholesky of both Gram
matrices, SVD of the ``r \\times r`` core matrix.  Cost: ``O(mr^2 + r^3)``.

**Full path** (`fullsvd=true` or complex `T`): Cholesky of the column
Gram, SVD of the ``r \\times n`` matrix ``B = R_L D R^T``.
Cost: ``O(mr^2 + r^2 n)``.

Returns `(Q, sv, V_or_nothing, nk)` where `nk` is the number of
singular values above `tol_rt`.
"""
function _llama_finalize(cols_store::AbstractMatrix{T}, rows_store::AbstractMatrix{T},
                         pivot_diag::AbstractVector{T}, row_gram::AbstractMatrix{T},
                         rank::Int, tol_rt::Real, fullsvd::Bool) where T
  RT = real(T)
  m = size(cols_store, 1)
  d = @view pivot_diag[1:rank]
  Cv = @view cols_store[:, 1:rank]
  Rv = @view rows_store[:, 1:rank]

  # Column Gram → Cholesky: C = Q_L R_L
  CtC = Cv' * Cv                        # O(m × r²)
  CL = cholesky!(Hermitian(CtC))

  if !fullsvd && T <: Real
    # ── Fast path: rank×rank Core SVD ──
    # Core = R_L D R_R^T  (rank×rank, since R^T = R^H for real T)
    RtR = @view row_gram[1:rank, 1:rank]
    CR = cholesky(Hermitian(RtR))        # non-mutating: preserves row_gram
    Core = Matrix(CL.U)
    Core .*= transpose(d)               # CL.U * diag(D)
    Core = Core * Matrix(CR.U)'         # rank × rank
    F = svd(Core)

    nk = count(s -> s > tol_rt, F.S)
    nk == 0 && return (Matrix{T}(undef, m, 0), RT[], nothing, 0)

    sv = F.S[1:nk]
    W = CL.U \ @view(F.U[:, 1:nk])     # rank × nk
    Q = Cv * W                          # m × nk
    return (Q, sv, nothing, nk)
  else
    # ── Full path: rank×n matrix B SVD ──
    # B = R_L D R^T  (rank × n, uses transpose for complex correctness)
    UL = Matrix(CL.U)
    UL .*= transpose(d)                 # CL.U * diag(D)
    B = UL * transpose(Rv)              # rank × n
    F = svd(B)

    nk = count(s -> s > tol_rt, F.S)
    nk == 0 && return (Matrix{T}(undef, m, 0), RT[], nothing, 0)

    sv = F.S[1:nk]
    W = CL.U \ @view(F.U[:, 1:nk])     # rank × nk
    Q = Cv * W                          # m × nk
    V_final = fullsvd ? F.V[:, 1:nk] : nothing
    return (Q, sv, V_final, nk)
  end
end

"""
    _llama_correction!(residual, Q, sv, d_row, row_pivots, is_row_pivot)

Compute SVD-corrected residuals using the accessed-row Gram estimate.
Updates `residual` in-place and returns the maximum corrected residual
over non-pivot rows.
"""
function _llama_correction!(residual::AbstractVector{RT}, Q::AbstractMatrix{T},
                            sv::AbstractVector{RT}, d_row::AbstractVector{<:Real},
                            row_pivots::AbstractVector{Int},
                            is_row_pivot::AbstractVector{Bool}) where {T, RT<:Real}
  m = length(residual)
  nk = length(sv)

  # Estimate G_A ≈ P Σ² P  where P = Q[pivots,:]^H Q[pivots,:]
  Qp = Q[row_pivots, :]                 # copy: fancy indexing → contiguous for BLAS
  P = Qp' * Qp                          # nk × nk
  Y = Diagonal(sv) * P                  # nk × nk
  Z = Q * Y'                             # m × nk

  max_corr_res = zero(RT)
  @inbounds for i in 1:m
    if !is_row_pivot[i]
      s = RT(d_row[i])
      for k in 1:nk
        s -= RT(abs2(Z[i, k]))
      end
      residual[i] = max(s, zero(RT))
      max_corr_res = max(max_corr_res, residual[i])
    end
  end
  return max_corr_res
end

"""
    _ensure_recompute_buffers!(W_buf, DG_vec, rank) -> W_buf

Ensure `W_buf` (``m \\times r``) and `DG_vec` (length ``\\geq r^2``) have
sufficient capacity for rank `r`.  Returns the (possibly reallocated) `W_buf`.
"""
function _ensure_recompute_buffers!(W_buf::Matrix{T}, DG_vec::Vector{T}, rank::Int) where T
  W_buf = _ensure_capacity!(W_buf, rank, 2)
  if rank * rank > length(DG_vec)
    resize!(DG_vec, rank * rank)
  end
  return W_buf
end

"""
    _llama_recompute_residuals!(residual, d_row, cols_store, pivot_diag, row_gram,
                                rank, is_row_pivot, phi_buf, W_buf)

Recompute all non-pivot residuals from scratch using the stored ACA
factors and the incrementally accumulated row Gram matrix.

For each non-pivot row ``i``:

``\\text{residual}[i] = \\max\\bigl(d_{\\text{row}}[i]
- \\Re\\bigl(\\phi_i^H\\, G_R\\, \\phi_i\\bigr),\\; 0\\bigr)``

where ``\\phi_i[t] = d_t \\cdot C[i,t]`` and ``G_R = R^H R``.

This avoids the accumulated numerical drift of the incremental
Gram update, which can make residuals overshoot to zero due to
the cross-term in the non-orthogonal ACA factorization.

Uses one ``m \\times r`` buffer (`W_buf`) and one length-``r^2`` vector
(`DG_vec`, reshaped to ``r \\times r``): computes ``DG = \\mathrm{diag}(d) \\cdot G_R``
(``O(r^2)``), then BLAS GEMM ``W = C \\cdot DG`` (``O(m r^2)``), and a scalar
inner loop for the per-row dot products (``O(m r)``).

Cost: ``O(m \\cdot r^2)`` where ``r`` is the current rank.
"""
function _llama_recompute_residuals!(residual::AbstractVector{RT},
                                     d_row::AbstractVector{<:Real},
                                     cols_store::AbstractMatrix{T},
                                     pivot_diag::AbstractVector{T},
                                     row_gram::AbstractMatrix{T},
                                     rank::Int,
                                     is_row_pivot::AbstractVector{Bool},
                                     W_buf::AbstractMatrix{T},
                                     DG_vec::AbstractVector{T}) where {T, RT<:Real}
  m = length(residual)
  rank == 0 && return zero(RT)

  Cv = @view cols_store[:, 1:rank]
  G_R = @view row_gram[1:rank, 1:rank]
  dv = @view pivot_diag[1:rank]

  # DG[t,u] = dv[t] * G_R[t,u]  — O(r²), row-scale
  DG = reshape(@view(DG_vec[1:rank*rank]), rank, rank)
  @inbounds for u in 1:rank
    for t in 1:rank
      DG[t, u] = dv[t] * G_R[t, u]
    end
  end

  # BLAS GEMM: W = Cv * DG  — O(m·r²)
  W = @view W_buf[:, 1:rank]
  mul!(W, Cv, DG)

  # Per-row: s_i = Re(Σ_t conj(dv[t]*Cv[i,t]) * W[i,t])  — O(m·r)
  max_res = zero(RT)
  @inbounds for i in 1:m
    if !is_row_pivot[i]
      s = zero(RT)
      for t in 1:rank
        s += RT(real(conj(dv[t] * Cv[i, t]) * W[i, t]))
      end
      residual[i] = max(RT(d_row[i]) - s, zero(RT))
      max_res = max(max_res, residual[i])
    end
  end
  return max_res
end

"""
    llama(matrix::AbstractALPACAMatrix{T};
          d_row, tol, max_rank=typemax(Int),
          oversample=0, fullsvd=false) → LLAMAResult

Compute an orthonormal basis for the column space of a general matrix
using LLAMA (Left Lowrank Amended Matrix Approximation).

The guidance vector ``d_{\\text{row}} = \\text{diag}(AA^H)`` provides the
squared ``\\ell_2`` norms of the rows (mnemonic: **LL**AMA ↔ ``\\ell_2``).

Alternatively, pass `d_col` (= ``\\text{diag}(A^H A)``, the squared column
norms) instead of `d_row` for **column-guided** decomposition.  When `d_col`
is given, LLAMA internally works on the transposed matrix, which is
faster when ``n > m`` (cost ``O((2m+n)r^2)`` instead of ``O((m+2n)r^2)``).
At most one of `d_row` or `d_col` may be provided; the dense
convenience wrapper auto-selects the faster mode when neither is given.

# How it works

LLAMA computes ``Q`` (an ``m \\times r`` orthonormal matrix) such that
``Q Q^H A \\approx A``, using only ``O(r)`` row and column accesses.
It proceeds in three stages:

**1. Inner loop — Gram-guided ACA with cross-coupled deflation:**

Each iteration picks the row ``i^*`` with the largest residual
``\\text{residual}[i]`` (initialized from ``d_{\\text{row}}``),
fetches and deflates it, then picks the best column from the deflated
row.  Both the column and row are stored as factors of a rank-1 update
``A \\approx C D R^H``.  Residuals are updated incrementally using the
Gram formula (see header comments for the mathematical derivation).

**2. SVD finalization — Cholesky + truncated SVD:**

Once the inner loop converges (all residuals below ``\\text{tol}^2``),
LLAMA extracts orthonormal ``Q`` from the stored factors ``C``, ``D``,
``R`` via Cholesky factorization of the column Gram (``r \\times r``),
followed by SVD of the ``r \\times n`` matrix ``B = R_L D R^T``.
When `fullsvd=true`, the right singular vectors ``V`` from this SVD
are also returned, yielding ``A \\approx Q \\Sigma V^H``.

**3. Iterative correction — accessed-row Gram estimate:**

After the SVD, LLAMA checks for missed rank by estimating the true
Gram matrix ``G_A = Q^H A A^H Q`` from the pivot rows.  Using the ACA
interpolation property (``A[\\text{pivot},:] = (C D R^H)[\\text{pivot},:]``
exactly), the estimate is ``\\tilde G_A = P \\Sigma^2 P`` where
``P = Q[\\text{pivots},:]^H Q[\\text{pivots},:]``.  Since ``P \\preceq I``,
the corrected residual ``d_{\\text{row}}[i] - q_i^H \\tilde G_A q_i``
is less aggressive than the Gram formula, correctly revealing rows
where the ACA approximation overshot.  If any corrected residual
exceeds ``\\text{tol}^2``, the inner loop re-enters with updated values.

# Arguments
- `matrix`: matrix-free wrapper implementing [`column!`](@ref) and [`row!`](@ref).
- `d_row`: precomputed ``\\text{diag}(A A^H)``, length `m`.  Used as the
  initial residual norm indicator for row pivot selection.
  Mutually exclusive with `d_col`.
- `d_col`: precomputed ``\\text{diag}(A^H A)``, length `n`.  When given
  instead of `d_row`, LLAMA works on the transposed matrix internally,
  which is faster when ``n > m``.
- `tol`: approximation tolerance.  Controls both convergence
  (``\\max_i \\text{residual}[i] < \\text{tol}^2``) and singular value
  truncation in the finalization SVD.
- `max_rank`: upper bound on rank (default: `typemax(Int)`).
- `oversample`: reserved for future robustness extensions (unused).
- `fullsvd`: if `true`, also compute the right singular vectors `V`.
  The full approximation ``A \\approx Q \\, \\Sigma \\, V^H`` is then
  available at cost ``O(n r^2)`` extra (one triangular solve + one gemm).

# Returns
A [`LLAMAResult`](@ref) containing:
- `Q`: orthonormal column-space basis (``m \\times r``).
- `singular_values`: approximate singular values from the core SVD.
- `col_pivots`, `row_pivots`: indices of accessed columns and rows.
- `V`: right singular vectors (``n \\times r``) when `fullsvd=true`;
  `nothing` otherwise.

# Complexity
For an ``m \\times n`` matrix of numerical rank ``r``:
- **Accesses**: ``r`` columns + up to ``m`` rows (at most ``r`` successful
  pivots; exhausted rows cost one row access each).
- **Memory**: ``O(m r + n r)`` for stored factors, ``O(r^2)`` for
  Cholesky/SVD workspace.
- **Arithmetic**: ``O((m+n) r^2)`` for the inner loop (BLAS-2 deflation
  + row Gram entries), ``O(m r^2 + r^3)`` for finalization (row Gram
  is accumulated incrementally during the inner loop).

# Numerical stability

Residuals are tracked via the Gram-corrected formula
``||A[i,:]||^2 - ||\\text{approx}[i,:]||^2``, which can undershoot
zero due to the non-orthogonal nature of ACA (shared by LU and
any non-SVD decomposition).  When a row appears exhausted (deflated
content near zero), LLAMA uses a `needs_recompute` flag to decide
whether to recompute all residuals from scratch or break: after each
successful pivot the flag is set, and at most one recomputation is
performed per pivot.  This limits the ``O(m r^2)`` recomputation
cost to at most ``r`` times total.  The iterative ``P \\Sigma^2 P``
correction catches any remaining overshoots across outer iterations.
Singular values near the tolerance boundary may be captured or not,
depending on the accumulated deflation residual.

# Example
```julia
using ALPACADecomposition

A = randn(100, 200)  # or a low-rank matrix
d_row = vec(sum(abs2, A, dims=2))
result = llama(DenseALPACAMatrix(A); d_row, tol=1e-10)

Q = result.Q  # orthonormal basis for column space
```
"""
function llama(matrix::AbstractALPACAMatrix{T};
               d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
               d_col::Union{AbstractVector{<:Real}, Nothing}=nothing,
               tol::Real,
               pivotol::Real=NaN,
               max_rank::Integer=typemax(Int),
               oversample::Integer=0,
               fullsvd::Bool=false) where T
  # ── Column-guided dispatch: transpose the problem ──
  if d_col !== nothing
    d_row === nothing || throw(ArgumentError(
      "cannot specify both d_row and d_col"))
    m, n = size(matrix)
    length(d_col) == n || throw(ArgumentError(
      "d_col length $(length(d_col)) must match column count $n"))
    result_t = llama(TransposedALPACAMatrix(matrix);
                     d_row=d_col, tol=tol, pivotol=pivotol, max_rank=max_rank,
                     oversample=oversample, fullsvd=true)
    result_t.V === nothing && return LLAMAResult{T, real(T)}(
      Matrix{T}(undef, m, 0), real(T)[], result_t.row_pivots, result_t.col_pivots, nothing)
    # V of A^T spans col(conj(A)); conjugate to get col(A)
    Q_a = T <: Real ? result_t.V : conj(result_t.V)
    V_a = fullsvd ? (T <: Real ? result_t.Q : conj(result_t.Q)) : nothing
    return LLAMAResult{T, real(T)}(Q_a, result_t.singular_values,
                                   result_t.row_pivots, result_t.col_pivots, V_a)
  end

  d_row === nothing && throw(ArgumentError(
    "must specify either d_row or d_col"))

  m, n = size(matrix)
  RT = real(T)
  tol_rt = RT(tol)
  # Pivot tolerance: adaptive via effective dimensionality from d_row
  if isnan(pivotol)
    d_max = maximum(d_row)
    if d_max > 0
      m_eff = max(one(RT), RT(sum(sqrt, d_row) / sqrt(d_max)))
    else
      m_eff = RT(m)
    end
    pivotol_rt = RT(tol / sqrt(m_eff))
  else
    pivotol_rt = RT(pivotol)
  end
  pivotol2 = pivotol_rt^2

  length(d_row) == m || throw(ArgumentError(
    "d_row length $(length(d_row)) must match row count $m"))

  max_r = min(m, n, max_rank)

  # Work buffers
  cbuf = Vector{T}(undef, m)
  rbuf = Vector{T}(undef, n)
  coeffs = Vector{T}(undef, max_r)

  row_pivots = Int[]
  col_pivots = Int[]
  is_col_pivot = falses(n)
  is_row_pivot = falses(m)

  residual = RT.(d_row)

  # Cross-coupled storage (amortized)
  init_cap = min(max_r, 64)
  cols_store = Matrix{T}(undef, m, init_cap)   # scaled deflated columns
  rows_store = Matrix{T}(undef, n, init_cap)   # scaled deflated rows
  recomp_DG = Vector{T}(undef, 0)              # DG vector (reshaped to r×r) for residual recomputation
  recomp_W = Matrix{T}(undef, m, 0)            # W buffer for residual recomputation
  pivot_diag = Vector{T}(undef, init_cap)      # pivot values
  row_gram = Matrix{T}(undef, init_cap, init_cap) # R^H R accumulated incrementally

  rank = 0

  # Outer-loop state (must be visible after loop for fallthrough return)
  Q_final = Matrix{T}(undef, m, 0)
  V_final::Union{Nothing, Matrix{T}} = nothing
  sv = RT[]
  nk = 0

  # ── Outer loop: SVD-corrected residual from accessed rows ──
  max_outer = 10
  rank_at_start = 0
  for outer in 1:max_outer

  rank_at_start = rank
  needs_recompute = false  # set true after each Gram update, false after recomputation
  for iter in 1:m
    # ── Select row: argmax residual ──
    max_res = zero(RT)
    best_row = 0
    @inbounds for i in 1:m
      if !is_row_pivot[i] && residual[i] > max_res
        max_res = residual[i]
        best_row = i
      end
    end
    (best_row == 0 || max_res < pivotol2) && break

    # ── Fetch and deflate row (cross-coupled: uses stored columns) ──
    row!(rbuf, matrix, best_row)
    if rank > 0
      cv = @view coeffs[1:rank]
      @inbounds for t in 1:rank
        cv[t] = pivot_diag[t] * cols_store[best_row, t]
      end
      Rv = @view rows_store[:, 1:rank]
      mul!(rbuf, Rv, cv, -one(T), one(T))
    end

    # ── Select column: argmax |deflated_row| ──
    best_col = 0
    best_val = zero(RT)
    @inbounds for j in 1:n
      if !is_col_pivot[j]
        v = abs(rbuf[j])
        if v > best_val
          best_val = v
          best_col = j
        end
      end
    end

    # If deflated row is near-zero, this row is already well-captured
    if best_col == 0 || best_val < pivotol_rt
      is_row_pivot[best_row] = true
      residual[best_row] = zero(RT)
      if needs_recompute
        # Gram overshoot may have zeroed residuals prematurely — recompute once
        recomp_W = _ensure_recompute_buffers!(recomp_W, recomp_DG, rank)
        max_res = _llama_recompute_residuals!(residual, d_row, cols_store, pivot_diag,
                                               row_gram, rank, is_row_pivot, recomp_W, recomp_DG)
        needs_recompute = false
        max_res < pivotol2 && break
        continue
      else
        break
      end
    end

    # ── Fetch and deflate column (cross-coupled: uses stored rows) ──
    column!(cbuf, matrix, best_col)
    if rank > 0
      cv = @view coeffs[1:rank]
      @inbounds for t in 1:rank
        cv[t] = pivot_diag[t] * rows_store[best_col, t]
      end
      Cv = @view cols_store[:, 1:rank]
      mul!(cbuf, Cv, cv, -one(T), one(T))
    end

    # ── Pivot value at intersection ──
    pivot_val = cbuf[best_row]
    if abs(pivot_val) < pivotol_rt
      is_row_pivot[best_row] = true
      residual[best_row] = zero(RT)
      if needs_recompute
        recomp_W = _ensure_recompute_buffers!(recomp_W, recomp_DG, rank)
        max_res = _llama_recompute_residuals!(residual, d_row, cols_store, pivot_diag,
                                               row_gram, rank, is_row_pivot, recomp_W, recomp_DG)
        needs_recompute = false
        max_res < pivotol2 && break
        continue
      else
        break
      end
    end

    # ── Scale and store ──
    inv_pv = one(T) / pivot_val
    cbuf .*= inv_pv
    rbuf .*= inv_pv

    rank += 1
    cols_store = _ensure_capacity!(cols_store, rank, 2)
    rows_store = _ensure_capacity!(rows_store, rank, 2)
    if rank > length(pivot_diag)
      new_cap = max(2 * length(pivot_diag), rank)
      resize!(pivot_diag, new_cap)
    end
    if rank > size(row_gram, 1)
      new_cap = max(2 * size(row_gram, 1), rank)
      new_rg = Matrix{T}(undef, new_cap, new_cap)
      old_sz = size(row_gram, 1)
      @views new_rg[1:old_sz, 1:old_sz] .= row_gram
      row_gram = new_rg
    end

    @views cols_store[:, rank] .= cbuf
    @views rows_store[:, rank] .= rbuf
    pivot_diag[rank] = pivot_val

    # ── Update residual norm indicators (Gram-corrected) ──
    # Incremental update of ||A[i,:]||² − ||approx[i,:]||²:
    #   Δ[i] = |d_k C[i,k]|² G_kk + 2 Re(d̄_k C̄[i,k] · Σ_{t<k} d_t C[i,t] G_tk)
    # where G_tk = ⟨R[:,t], R[:,k]⟩ (row Gram matrix entries).
    # We also accumulate G into row_gram for use in finalization.
    rv_k = @view rows_store[:, rank]
    G_kk = zero(RT)
    @inbounds for j in 1:n
      G_kk += RT(abs2(rv_k[j]))
    end
    row_gram[rank, rank] = T(G_kk)
    d_k = pivot_diag[rank]

    if rank > 1
      # Compute all Gram entries at once via BLAS gemv instead of individual dots
      Rv_prev = @view rows_store[:, 1:rank-1]
      wv = @view coeffs[1:rank-1]
      mul!(wv, Rv_prev', rv_k)           # wv[t] = ⟨R[:,t], R[:,k]⟩ = G_tk
      @inbounds for t in 1:rank-1
        row_gram[t, rank] = wv[t]
        row_gram[rank, t] = conj(wv[t])
        wv[t] *= pivot_diag[t]           # coeffs[t] = d_t G_tk
      end
      Cv = @view cols_store[:, 1:rank-1]
      mul!(cbuf, Cv, wv)

      @inbounds for i in 1:m
        if !is_row_pivot[i]
          phi = d_k * cols_store[i, rank]
          residual[i] -= RT(abs2(phi)) * G_kk + 2 * RT(real(conj(phi) * cbuf[i]))
          residual[i] = max(residual[i], zero(RT))
        end
      end
    else
      pv2_Gkk = abs2(d_k) * G_kk
      @inbounds for i in 1:m
        if !is_row_pivot[i]
          residual[i] -= pv2_Gkk * RT(abs2(cols_store[i, rank]))
          residual[i] = max(residual[i], zero(RT))
        end
      end
    end

    push!(col_pivots, best_col)
    push!(row_pivots, best_row)
    is_col_pivot[best_col] = true
    is_row_pivot[best_row] = true
    residual[best_row] = zero(RT)
    needs_recompute = true  # Gram update may overshoot for next rows

    rank >= max_r && break
  end  # inner loop

  # ── SVD finalization ──
  rank == 0 && return LLAMAResult{T, RT}(Matrix{T}(undef, m, 0), RT[], col_pivots, row_pivots, nothing)
  new_pivots = rank - rank_at_start

  Q_final, sv, V_final, nk = _llama_finalize(cols_store, rows_store, pivot_diag, row_gram, rank, tol_rt, fullsvd)
  nk == 0 && return LLAMAResult{T, RT}(Matrix{T}(undef, m, 0), RT[], col_pivots, row_pivots, nothing)

  rank >= max_r && return LLAMAResult{T, RT}(Q_final, sv, col_pivots, row_pivots, V_final)
  new_pivots == 0 && return LLAMAResult{T, RT}(Q_final, sv, col_pivots, row_pivots, V_final)

  # ── SVD-corrected residual ──
  max_corr_res = _llama_correction!(residual, Q_final, sv, d_row, row_pivots, is_row_pivot)
  max_corr_res < pivotol2 && return LLAMAResult{T, RT}(Q_final, sv, col_pivots, row_pivots, V_final)

  # Not converged — reset exhausted (non-pivot) rows for retry
  fill!(is_row_pivot, false)
  for rp in row_pivots
    is_row_pivot[rp] = true
  end

  end  # outer loop

  # Outer loop exhausted without convergence — return last SVD result
  return LLAMAResult{T, RT}(Q_final, sv, col_pivots, row_pivots, V_final)
end

# ──────────────────────────────────────────────────────────────────
# Dense-matrix convenience wrapper
# ──────────────────────────────────────────────────────────────────

"""
    llama(matrix::AbstractMatrix; tol, d_row=nothing, d_col=nothing, kwargs...)

Convenience interface for dense matrices.  If neither `d_row` nor `d_col`
is provided, automatically selects the faster mode: column-guided
(`d_col`, squared ``\ell_2`` column norms) when ``n > m``, row-guided
(`d_row`, squared ``\ell_2`` row norms) otherwise.
"""
function llama(matrix::AbstractMatrix{T};
               tol::Real,
               pivotol::Real=NaN,
               d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
               d_col::Union{AbstractVector{<:Real}, Nothing}=nothing,
               max_rank::Integer=typemax(Int),
               oversample::Integer=0,
               fullsvd::Bool=false) where T
  if d_row === nothing && d_col === nothing
    m, n = size(matrix)
    if n > m
      d_col = vec(sum(abs2, matrix, dims=1))
    else
      d_row = vec(sum(abs2, matrix, dims=2))
    end
  end
  return llama(DenseALPACAMatrix(matrix); d_row, d_col, tol, pivotol, max_rank, oversample, fullsvd)
end

"""
    llama(matrix::Symmetric; tol, [d_row], kwargs...)

Convenience interface for `Symmetric` matrices.  Only needs column access
since rows equal columns.  If `d_row` is not provided, it is computed as
the squared ``\\ell_2`` norms of the columns (equal to row norms for
symmetric matrices).
"""
function llama(matrix::Symmetric{T};
               tol::Real,
               pivotol::Real=NaN,
               d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
               max_rank::Integer=typemax(Int),
               oversample::Integer=0,
               fullsvd::Bool=false) where T
  if d_row === nothing
    d_row = vec(sum(abs2, parent(matrix), dims=1))
  end
  wrapped = SymmetricALPACAMatrix(DenseALPACAMatrix(parent(matrix)))
  return llama(wrapped; d_row, tol, pivotol, max_rank, oversample, fullsvd)
end

"""
    llama(matrix::Hermitian; tol, [d_row], kwargs...)

Convenience interface for `Hermitian` matrices.  Only needs column access
since rows are the complex conjugates of columns.  If `d_row` is not
provided, it is computed as the squared ``\\ell_2`` norms of the columns
(equal to row norms for Hermitian matrices).
"""
function llama(matrix::Hermitian{T};
               tol::Real,
               pivotol::Real=NaN,
               d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
               max_rank::Integer=typemax(Int),
               oversample::Integer=0,
               fullsvd::Bool=false) where T
  if d_row === nothing
    d_row = vec(sum(abs2, parent(matrix), dims=1))
  end
  wrapped = HermitianALPACAMatrix(DenseALPACAMatrix(parent(matrix)))
  return llama(wrapped; d_row, tol, pivotol, max_rank, oversample, fullsvd)
end

# ──────────────────────────────────────────────────────────────────
# SVD extraction
# ──────────────────────────────────────────────────────────────────

"""
    llama_svd(matrix; kwargs...) → (U, S, Vt)

Compute a low-rank SVD via LLAMA.

Returns `(U, S, Vt)` where `U` contains the left singular vectors,
`S` the approximate singular values, and `Vt` the conjugate-transposed
right singular vectors.

All keyword arguments are forwarded to [`llama`](@ref).
"""
function llama_svd(matrix::AbstractALPACAMatrix{T};
                   d_row::AbstractVector{<:Real},
                   tol::Real,
                   pivotol::Real=NaN,
                   max_rank::Integer=typemax(Int),
                   oversample::Integer=0) where T
  m, n = size(matrix)
  result = llama(matrix; d_row, tol, pivotol, max_rank, oversample, fullsvd=true)
  r = size(result.Q, 2)
  RT = real(T)
  if r == 0
    return (U = Matrix{T}(undef, m, 0),
            S = Vector{RT}(undef, 0),
            Vt = Matrix{T}(undef, 0, n))
  end
  return (U = result.Q, S = result.singular_values,
          Vt = result.V')
end

"""
    llama_svd(matrix::AbstractMatrix; kwargs...) → (U, S, Vt)

Dense convenience interface.  Automatically uses column-guided
decomposition (faster) when `n > m` and neither `d_row` nor `d_col`
is provided.  See [`llama_svd`](@ref) for details.
"""
function llama_svd(matrix::AbstractMatrix{T};
                   tol::Real,
                   pivotol::Real=NaN,
                   d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
                   d_col::Union{AbstractVector{<:Real}, Nothing}=nothing,
                   max_rank::Integer=typemax(Int),
                   oversample::Integer=0) where T
  m, n = size(matrix)
  if d_row === nothing && d_col === nothing
    if n > m
      d_col = vec(sum(abs2, matrix, dims=1))
    else
      d_row = vec(sum(abs2, matrix, dims=2))
    end
  end
  result = llama(DenseALPACAMatrix(matrix); d_row, d_col, tol, pivotol, max_rank, oversample, fullsvd=true)
  r = size(result.Q, 2)
  RT = real(T)
  if r == 0
    return (U = Matrix{T}(undef, m, 0),
            S = Vector{RT}(undef, 0),
            Vt = Matrix{T}(undef, 0, n))
  end
  return (U = result.Q, S = result.singular_values,
          Vt = result.V')
end

"""
    llama_svd(matrix::Symmetric; tol, [d_row], kwargs...) → (U, S, Vt)

Convenience interface for `Symmetric` matrices.  See [`llama`](@ref) for details.
"""
function llama_svd(matrix::Symmetric{T};
                   tol::Real,
                   pivotol::Real=NaN,
                   d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
                   max_rank::Integer=typemax(Int),
                   oversample::Integer=0) where T
  result = llama(matrix; d_row, tol, pivotol, max_rank, oversample, fullsvd=true)
  m, n = size(matrix)
  r = size(result.Q, 2)
  RT = real(T)
  if r == 0
    return (U = Matrix{T}(undef, m, 0),
            S = Vector{RT}(undef, 0),
            Vt = Matrix{T}(undef, 0, n))
  end
  return (U = result.Q, S = result.singular_values,
          Vt = result.V')
end

"""
    llama_svd(matrix::Hermitian; tol, [d_row], kwargs...) → (U, S, Vt)

Convenience interface for `Hermitian` matrices.  See [`llama`](@ref) for details.
"""
function llama_svd(matrix::Hermitian{T};
                   tol::Real,
                   pivotol::Real=NaN,
                   d_row::Union{AbstractVector{<:Real}, Nothing}=nothing,
                   max_rank::Integer=typemax(Int),
                   oversample::Integer=0) where T
  result = llama(matrix; d_row, tol, pivotol, max_rank, oversample, fullsvd=true)
  m, n = size(matrix)
  r = size(result.Q, 2)
  RT = real(T)
  if r == 0
    return (U = Matrix{T}(undef, m, 0),
            S = Vector{RT}(undef, 0),
            Vt = Matrix{T}(undef, 0, n))
  end
  return (U = result.Q, S = result.singular_values,
          Vt = result.V')
end
