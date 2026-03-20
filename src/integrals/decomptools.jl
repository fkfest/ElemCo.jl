"""
This module contains functions for tensor decomposition methods.
"""
module DecompTools
using LinearAlgebra
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.QMTensors

export calc_integrals_decomposition, calc_df_integrals
export eigen_decompose, svd_decompose
export rotate_U2pseudocanonical
export qr_pivoted_symmetric_decompose
export ldlt_pivoted_symmetric_decompose
export orthogonalize

"""
    symmetric_pivoted_cholesky(M, tol)

  Pivoted Cholesky factorization for a complex symmetric positive semi-definite
  matrix: ``M = P^T L L^T P``, where ``L`` is lower triangular.
  This differs from the standard Hermitian Cholesky (``M = L L^†``) by using
  transpose instead of adjoint, which is required for complex symmetric matrices
  arising from two-electron integrals with complex MO coefficients.

  Returns `(pqP, rank)` where `pqP[i,L]` are the unpivoted Cholesky vectors
  such that ``M ≈ pqP \\cdot pqP^T``.
"""
function symmetric_pivoted_cholesky(M_in::AbstractMatrix{T}, tol) where T<:Number
  #TODO implement a more efficient version or check if using SVD is faster
  n = size(M_in, 1)
  M = copy(M_in)
  L = zeros(T, n, n)
  perm = collect(1:n)
  rank = 0
  for k in 1:n
    max_diag = 0.0
    max_idx = k
    for i in k:n
      d = abs(M[i,i])
      if d > max_diag
        max_diag = d
        max_idx = i
      end
    end
    max_diag < tol && break
    if max_idx != k
      for j in 1:n
        M[j,k], M[j,max_idx] = M[j,max_idx], M[j,k]
      end
      for j in 1:n
        M[k,j], M[max_idx,j] = M[max_idx,j], M[k,j]
      end
      for j in 1:k-1
        L[k,j], L[max_idx,j] = L[max_idx,j], L[k,j]
      end
      perm[k], perm[max_idx] = perm[max_idx], perm[k]
    end
    L[k,k] = sqrt(M[k,k])
    for i in (k+1):n
      L[i,k] = M[i,k] / L[k,k]
    end
    # Update remaining submatrix: M -= L[:,k] * L[:,k]^T (symmetric, not Hermitian)
    for j in (k+1):n
      for i in j:n
        M[i,j] -= L[i,k] * L[j,k]
        if i != j
          M[j,i] = M[i,j]
        end
      end
    end
    rank += 1
  end
  pqP = L[invperm(perm), 1:rank]
  return pqP, rank
end

"""
    orthogonalize(L, neg_indices=Int[]) → (ortho, diag)

Convert decomposition vectors `L` and sign information `neg_indices` into
an approximate eigendecomposition (real) or Takagi factorization (complex).

Given `L` such that ``M ≈ L \\cdot S \\cdot L^T`` (where ``S`` is the sign diagonal
with ``S_{kk} = -1`` for ``k ∈`` `neg_indices`), compute orthonormal matrix `ortho`
and values `diag` such that ``M ≈ \\text{ortho} \\cdot \\text{diag}(d) \\cdot \\text{ortho}^T``.

Uses QR decomposition of `L` followed by eigendecomposition (real) or
Takagi/SVD decomposition (complex) of the small ``r × r`` factor.

# Arguments
- `L`: decomposition vectors, shape `(n, r)`
- `neg_indices`: column indices with negative sign (default: empty)

# Returns
- `ortho`: orthonormal matrix `(n × r)`
- `diag::Vector{Float64}`: eigenvalues (real, can be negative) or singular values (complex, ≥ 0), length `r`
"""
function orthogonalize(L::AbstractMatrix{T}, neg_indices::Vector{Int}=Int[]) where T
  n, r = size(L)
  if r == 0
    return L, Float64[]
  end

  F_L = qr(L)
  Q = Matrix(F_L.Q)
  R = F_L.R

  if T <: Complex
    # Takagi decomposition of R * R^T
    B = R * transpose(R)
    F_B = svd(B)
    A_B = F_B.U
    V_B = (F_B.Vt)'
    phases_B = [conj(sum(A_B[:,k] .* V_B[:,k])) for k in 1:r]
    U_T = A_B .* transpose(sqrt.(phases_B))
    return Q * U_T, F_B.S
  else
    # Eigendecomposition of R * S * R^T
    signs = ones(r)
    signs[neg_indices] .= -1.0
    B = R * Diagonal(signs) * transpose(R)
    E = eigen(Symmetric(B))
    return Q * E.vectors, E.values
  end
end

"""
    _nystrom_vectors(M, pivots, tol) → (L, rank, neg_indices)

Compute Nyström decomposition vectors from a symmetric matrix and a set of pivot indices.

Given ``M`` and pivot set ``B``, computes ``L`` such that
``M ≈ L \\cdot S \\cdot L^T`` using the RI/Nyström formula
``M ≈ M_{:,B} J^{-1} M_{B,:}^T`` where ``J = M_{B,B}``.

- Complex: SVD/Takagi factorization ``J = U_T \\Sigma U_T^T`` →
  ``C = \\overline{U_T} \\Sigma^{-1/2}``
- Real: eigendecomposition ``J = Q \\Lambda Q^T`` →
  ``C_k = q_k / \\sqrt{|\\lambda_k|}``; negative ``\\lambda_k`` tracked in `neg_indices`

# Arguments
- `M`: symmetric matrix (``M = M^T``), real or complex
- `pivots`: vector of pivot column indices
- `tol`: threshold for rank determination

# Returns NamedTuple
- `L`: decomposition vectors, shape `(n, rank)`, type matches input
- `rank`: number of decomposition vectors
- `neg_indices`: column indices in `L` corresponding to negative eigenvalues
  (empty for complex matrices)
"""
function _nystrom_vectors(M::AbstractMatrix{T}, pivots::Vector{Int}, tol) where T
  nB = length(pivots)
  if nB == 0
    return (L = zeros(T, n, 0), rank = 0, neg_indices = Int[])
  end

  J = M[pivots, pivots]

  if T <: Complex
    # SVD-based Takagi: J = U_T Σ U_T^T with U_T unitary, Σ ≥ 0
    # J^{-1} = conj(U_T) Σ^{-1} conj(U_T)^T, so C = conj(U_T) Σ^{-1/2}
    F_svd = svd(J)
    nB_final = count(s -> s > tol, F_svd.S)
    nB_final = max(nB_final, 1)
    A = F_svd.U[:, 1:nB_final]
    B = F_svd.Vt[1:nB_final, :]'
    phases = [conj(sum(A[:,k] .* B[:,k])) for k in 1:nB_final]
    inv_sqrt_S = 1.0 ./ sqrt.(F_svd.S[1:nB_final])
    C = conj(A) .* transpose(sqrt.(conj.(phases)) .* inv_sqrt_S)
    neg_indices = Int[]
  else
    # Eigendecomposition: J = Q Λ Q^T → J^{-1} = Q |Λ|^{-1} Q^T
    # C_k = q_k / √|λ_k|; negative λ_k tracked in neg_indices
    E = eigen(Symmetric(J))
    nB_final = count(e -> abs(e) > tol, E.values)
    nB_final = max(nB_final, 1)
    keep = sortperm(abs.(E.values), rev=true)[1:nB_final]
    vals = E.values[keep]
    vecs = E.vectors[:, keep]
    inv_sqrt_vals = 1.0 ./ sqrt.(abs.(vals))
    C = vecs .* transpose(inv_sqrt_vals)
    neg_indices = findall(v -> v < 0, vals)
  end

  L = M[:, pivots] * C
  return (L = L, rank = nB_final, neg_indices = neg_indices)
end

"""
    qr_pivoted_symmetric_decompose(M, tol; sigma=0.01, pivotol=tol) → (L, rank, neg_indices)

Two-step decomposition of a symmetric matrix using QR-based pivot
selection and the Nyström formula.

Given a symmetric matrix ``M = M^T`` (real or complex), produces `L` such that ``M ≈ L \\cdot S \\cdot L^T`` (transpose, not adjoint),
where ``S`` is a diagonal sign matrix with ``S_{kk} = -1`` for
``k ∈`` `neg_indices` and ``+1`` otherwise.

**Step I** uses a span-factor batched QR algorithm to determine the pivot set:
- Column norms ``r_I = \\|M_{:,I}\\|^2`` serve as importance metric (always non-negative)
- Batch selection: ``Q = \\{I ∈ D : r_I ≥ σ \\cdot \\max_D r\\}``
- Within each batch, column-pivoted QR selects the important columns
- Residual norms are updated by subtracting projections onto selected columns
- Diagonal screening: indices below threshold are removed

**Step II** uses the RI/Nyström formula ``M ≈ M_{:,B} J^{-1} M_{B,:}^T``
where ``J = M_{B,B}``:
- Complex: SVD/Takagi factorization ``J = U_T \\Sigma U_T^T`` →
  ``C = \\overline{U_T} \\Sigma^{-1/2}`` with ``C C^T = J^{-1}``
- Real: eigendecomposition ``J = Q \\Lambda Q^T`` →
  ``C_k = q_k / \\sqrt{|\\lambda_k|}``; negative ``\\lambda_k`` tracked in `neg_indices`

# Arguments
- `M`: symmetric matrix (``M = M^T``), real or complex
- `tol`: threshold for rank determination
- `sigma`: span factor for batch pivot selection (default: 0.01)
- `pivotol`: tolerance for column pivoting in QR (default: `tol`)

# Returns NamedTuple
- `L`: decomposition vectors, shape `(n, rank)`, type matches input
- `rank`: number of decomposition vectors
- `neg_indices`: column indices in `L` corresponding to negative eigenvalues
  (empty for PSD or complex matrices)
"""
function qr_pivoted_symmetric_decompose(M_in::AbstractMatrix{T}, tol; sigma::Float64=0.01, pivotol::Float64=tol) where T
  n = size(M_in, 1)

  # ═══════════════════════════════════════════════════════════════
  # STEP I: Span-factor batched QR pivot selection
  # ═══════════════════════════════════════════════════════════════
  # Initial squared column norms as importance metric (always non-negative)
  col_norms2 = vec(sum(abs2, M_in, dims=1))
  tol2 = pivotol^2

  D_indices = findall(r -> r >= tol2, col_norms2)
  nD = length(D_indices)

  est_cap = min(n, max(64, isqrt(nD)))
  pivots = Int[]
  sizehint!(pivots, est_cap)
  is_pivot = falses(n)  # BitVector for O(1) pivot lookup

  # Pre-allocate orthonormal basis with amortized doubling (avoids hcat)
  Q_acc = Matrix{T}(undef, n, est_cap)
  n_acc = 0

  # Pre-allocate work buffers
  Q_batch_buf = Vector{Int}(undef, n)
  new_D_buf = Vector{Int}(undef, n)

  while nD > 0
    # Find max squared column norm in D
    D_max = zero(real(T))
    @inbounds for I in D_indices
      r = col_norms2[I]
      r > D_max && (D_max = r)
    end
    D_max < tol2 && break

    # Batch: select all I ∈ D with col_norms2[I] ≥ σ * D_max
    threshold = sigma * D_max
    nQ = 0
    @inbounds for I in D_indices
      if col_norms2[I] >= threshold
        nQ += 1
        Q_batch_buf[nQ] = I
      end
    end
    Q_view = @view Q_batch_buf[1:nQ]
    sort!(Q_view, by=I -> col_norms2[I], rev=true)

    # Extract batch columns (allocation needed since columns are non-contiguous)
    cols = M_in[:, Q_view]

    # Project out accumulated basis: cols -= Q_acc * (Q_acc' * cols)
    if n_acc > 0
      Q_view_acc = @view Q_acc[:, 1:n_acc]
      proj = Q_view_acc' * cols
      mul!(cols, Q_view_acc, proj, -one(T), one(T))
    end

    # Column-pivoted QR of residual columns to select important ones
    F_qr = qr(cols, ColumnNorm())
    R_diag = abs.(diag(F_qr.R))
    n_new = count(rd -> rd > pivotol, R_diag)
    n_new == 0 && break

    # Record new pivots
    @inbounds for k in 1:n_new
      p = Q_batch_buf[F_qr.p[k]]
      push!(pivots, p)
      is_pivot[p] = true
      col_norms2[p] = 0.0
    end

    # Ensure Q_acc capacity (amortized doubling)
    if n_acc + n_new > size(Q_acc, 2)
      new_cap = max(2 * size(Q_acc, 2), n_acc + n_new)
      Q_new_buf = Matrix{T}(undef, n, new_cap)
      @views Q_new_buf[:, 1:n_acc] .= Q_acc[:, 1:n_acc]
      Q_acc = Q_new_buf
    end

    # Update orthonormal basis — copy QR columns directly into Q_acc (no hcat)
    Q_new = Matrix(F_qr.Q)[:, 1:n_new]
    @views Q_acc[:, n_acc+1:n_acc+n_new] .= Q_new
    n_acc += n_new

    # Update remaining D indices into pre-allocated buffer
    nD_new = 0
    @inbounds for I in D_indices
      if !is_pivot[I] && col_norms2[I] >= tol2
        nD_new += 1
        new_D_buf[nD_new] = I
      end
    end

    if nD_new > 0
      # Update residual column norms: r_new[I] -= ||Q_new' * M[:,I]||²
      remaining = @view new_D_buf[1:nD_new]
      proj_new = Q_new' * M_in[:, remaining]
      @inbounds for idx in 1:nD_new
        decrement = zero(real(T))
        for k in 1:n_new
          decrement += abs2(proj_new[k, idx])
        end
        I = new_D_buf[idx]
        col_norms2[I] -= decrement
        col_norms2[I] < 0 && (col_norms2[I] = 0.0)
      end

      # Re-screen after norm update
      nD_final = 0
      @inbounds for idx in 1:nD_new
        I = new_D_buf[idx]
        if col_norms2[I] >= tol2
          nD_final += 1
          new_D_buf[nD_final] = I
        end
      end
      D_indices = @view(new_D_buf[1:nD_final]) |> collect
      nD = nD_final
    else
      nD = 0
    end
  end

  # STEP II: Nyström approximation  M ≈ L S Lᵀ
  return _nystrom_vectors(M_in, pivots, tol)
end

"""
    ldlt_pivoted_symmetric_decompose(M, tol; sigma=0.01, pivotol=tol) → (L, rank, neg_indices)

Two-step decomposition of a symmetric matrix using LDLT-based pivot
selection and the Nyström formula.

Produces `L` such that ``M ≈ L \\cdot S \\cdot L^T`` (transpose, not adjoint),
where ``S`` is a diagonal sign matrix with ``S_{kk} = -1`` for
``k ∈`` `neg_indices` and ``+1`` otherwise.

Works for non-positive definite matrices, including real symmetric indefinite
and complex symmetric matrices.

**Step I** uses a span-factor batched LDLT algorithm to determine the pivot set,
following the spirit of the Cholesky-based approach of Folkestad et al. [JCP 150, 194112 (2019)]
but adapted for indefinite matrices:
- Diagonal elements ``d_I = M_{II}`` of the Schur complement serve as importance
  metric, using ``|d_I|`` for screening to handle negative/complex values
- Batch selection: ``Q = \\{I ∈ D : |d_I| ≥ σ \\cdot \\max_D |d|\\}``
- Within each batch, pivoted LDLT (``M = L D L^T`` with unit lower triangular ``L``)
  selects the important columns
- Residual diagonals are updated via Schur complement:
  ``d_I \\leftarrow d_I - \\ell_{Ik}^2 \\cdot d_k``
- Diagonal screening: indices with ``|d_I|`` below threshold are removed

**Step II** uses the RI/Nyström formula ``M ≈ M_{:,B} J^{-1} M_{B,:}^T``
where ``J = M_{B,B}``:
- Complex: SVD/Takagi factorization ``J = U_T \\Sigma U_T^T`` →
  ``C = \\overline{U_T} \\Sigma^{-1/2}`` with ``C C^T = J^{-1}``
- Real: eigendecomposition ``J = Q \\Lambda Q^T`` →
  ``C_k = q_k / \\sqrt{|\\lambda_k|}``; negative ``\\lambda_k`` tracked in `neg_indices`

# Arguments
- `M`: symmetric matrix (``M = M^T``), real or complex
- `tol`: threshold for rank determination
- `sigma`: span factor for batch pivot selection (default: 0.01)
- `pivotol`: tolerance for pivoting in LDLT (default: `tol`)

# Returns NamedTuple
- `L`: decomposition vectors, shape `(n, rank)`, type matches input
- `rank`: number of decomposition vectors
- `neg_indices`: column indices in `L` corresponding to negative eigenvalues
  (empty for PSD or complex matrices)
"""
function ldlt_pivoted_symmetric_decompose(M_in::AbstractMatrix{T}, tol; sigma::Float64=0.01, pivotol::Float64=tol) where T
  n = size(M_in, 1)

  # ═══════════════════════════════════════════════════════════════
  # STEP I: Span-factor batched LDLT pivot selection
  # ═══════════════════════════════════════════════════════════════
  # Diagonal elements as importance metric (can be negative/complex for indefinite matrices)
  d = Vector{T}(diag(M_in))

  # D: set of significant diagonal indices (using absolute value for screening)
  D_indices = findall(i -> abs(d[i]) >= pivotol, 1:n)
  nD = length(D_indices)

  # Map from full index I to compressed index in D
  D_map = zeros(Int, n)
  @inbounds for (k, I) in enumerate(D_indices)
    D_map[I] = k
  end

  # Pre-allocate LDLT vector storage with amortized doubling
  est_cap = min(n, max(64, isqrt(nD)))
  L_storage = Matrix{T}(undef, nD, est_cap)
  D_storage = Vector{T}(undef, est_cap)
  n_stored = 0

  pivots = Int[]
  sizehint!(pivots, est_cap)
  is_pivot = falses(n)  # BitVector for O(1) pivot lookup

  # Pre-allocate work buffers
  V_col = Vector{T}(undef, nD)
  coeffs = Vector{T}(undef, est_cap)
  Q_batch = Vector{Int}(undef, n)
  new_D_buf = Vector{Int}(undef, n)

  while nD > 0
    # Find D_max = max |d[I]| for I ∈ D
    D_max = zero(real(T))
    @inbounds for I in D_indices
      aI = abs(d[I])
      aI > D_max && (D_max = aI)
    end
    D_max < pivotol && break

    # Qualify batch into pre-allocated buffer
    threshold = sigma * D_max
    nQ = 0
    @inbounds for I in D_indices
      if abs(d[I]) >= threshold
        nQ += 1
        Q_batch[nQ] = I
      end
    end
    Q_view = @view Q_batch[1:nQ]
    sort!(Q_view, by=I -> abs(d[I]), rev=true)

    n_batch = 0

    for _ in 1:nQ
      # Find best pivot: max |d[I]| in remaining Q
      best_j = 0
      best_val = zero(real(T))
      @inbounds for j in 1:nQ
        av = abs(d[Q_batch[j]])
        if av > best_val
          best_val = av
          best_j = j
        end
      end
      best_val < pivotol && break

      q = Q_batch[best_j]  # full index of the new pivot
      q_local = D_map[q]
      n_batch += 1

      # Compute column from M_in into pre-allocated buffer (no allocation)
      @inbounds for (k, I) in enumerate(D_indices)
        V_col[k] = M_in[I, q]
      end

      # Subtract ALL previous contributions using BLAS-2 gemv
      if n_stored > 0
        # coeffs[j] = D_storage[j] * L_storage[q_local, j]
        @inbounds for j in 1:n_stored
          coeffs[j] = D_storage[j] * L_storage[q_local, j]
        end
        # BLAS-2 gemv: V_col = -1 * L_storage[:,1:n_stored] * coeffs[1:n_stored] + V_col
        mul!(V_col, @view(L_storage[:, 1:n_stored]), @view(coeffs[1:n_stored]), -one(T), one(T))
      end

      dq = d[q]
      @inbounds for k in 1:nD
        V_col[k] /= dq
      end
      # V_col is now the unit lower triangular LDLT column: ℓ[q,q] = 1

      # Update diagonals via Schur complement: d[I] -= ℓ[I,k]² * D[k,k]
      @inbounds for (k, I) in enumerate(D_indices)
        d[I] -= V_col[k] * V_col[k] * dq
      end
      d[q] = zero(T)  # pivot's diagonal is fully consumed

      # Ensure L_storage capacity (amortized doubling)
      if n_stored + 1 > size(L_storage, 2)
        new_cap = 2 * size(L_storage, 2)
        L_new = Matrix{T}(undef, nD, new_cap)
        @views L_new[:, 1:n_stored] .= L_storage[:, 1:n_stored]
        L_storage = L_new
        D_new = Vector{T}(undef, new_cap)
        @views D_new[1:n_stored] .= D_storage[1:n_stored]
        D_storage = D_new
        coeffs = Vector{T}(undef, new_cap)
      end

      # Store LDLT vector directly into L_storage (no intermediate allocation)
      @inbounds for k in 1:nD
        L_storage[k, n_stored + 1] = V_col[k]
      end
      D_storage[n_stored + 1] = dq
      n_stored += 1

      push!(pivots, q)
      is_pivot[q] = true
    end

    # No progress in this batch — stop
    n_batch == 0 && break

    # Screen D into pre-allocated buffer
    nD_new = 0
    @inbounds for I in D_indices
      if !is_pivot[I] && abs(d[I]) >= pivotol
        nD_new += 1
        new_D_buf[nD_new] = I
      end
    end

    if nD_new < nD
      old_D_indices = D_indices
      D_indices = @view(new_D_buf[1:nD_new]) |> collect
      nD = nD_new
      fill!(D_map, 0)
      @inbounds for (k, I) in enumerate(D_indices)
        D_map[I] = k
      end
      # Compress L_storage rows to match new D_indices
      if nD > 0
        L_compressed = Matrix{T}(undef, nD, size(L_storage, 2))
        @inbounds for j in 1:n_stored
          for (old_k, I) in enumerate(old_D_indices)
            new_k = D_map[I]
            if new_k > 0
              L_compressed[new_k, j] = L_storage[old_k, j]
            end
          end
        end
        L_storage = L_compressed
      end
      V_col = Vector{T}(undef, nD)
    else
      D_indices = @view(new_D_buf[1:nD_new]) |> collect
      nD = nD_new
    end
  end

  # STEP II: Nyström approximation  M ≈ L S Lᵀ
  return _nystrom_vectors(M_in, pivots, tol)
end

"""
    calc_integrals_decomposition(EC::ECInfo)

  Decompose ``v_{pr}^{qs}`` as ``v_p^{qL} v_r^{sL}`` and store as `mmL`.
"""
function calc_integrals_decomposition(EC::ECInfo)
  pqrs = permutedims(ints2(EC,"::::",:α),(1,3,2,4))
  n = size(pqrs,1)
  if EC.options.cc.usecholesky
    Mmat = reshape(pqrs, (n^2,n^2))
    pqrs = nothing
    if ec_eltype(EC) <: Complex
      # Complex symmetric PSD matrix: need M = L*L^T (not L*L†)
      pqP, naux1 = symmetric_pivoted_cholesky(Mmat, EC.options.cholesky.thr)
    else
      CA = cholesky(Hermitian(Mmat), RowMaximum(), check = false, tol = EC.options.cholesky.thr)
      naux1 = CA.rank
      pqP = CA.U[1:naux1,invperm(CA.p)]'
    end
  else
    F = svd(reshape(pqrs, (n^2,n^2)))
    S = F.S
    pqrs = nothing

    naux1 = 0
    for s in S
      if s > EC.options.cholesky.thr
        naux1 += 1
      else
        break
      end
    end

    if ec_eltype(EC) <: Complex
      # Takagi factorization for complex symmetric M = U Σ U^T
      # From SVD M = A Σ B†, phases: e^{iφ_k} = conj(Aₖ^T Bₖ)
      A = F.U[:, 1:naux1]
      B = F.V[:, 1:naux1]
      phases = [conj(sum(A[:,k] .* B[:,k])) for k in 1:naux1]
      pqP = A .* transpose(sqrt.(phases) .* sqrt.(S[1:naux1]))
    else
      pqP = F.U[:,1:naux1].*sqrt.(S[1:naux1]')
    end
  end
  println("Integral auxiliary space size: ",naux1)
  save!(EC, "mmL", reshape(pqP, (n,n,naux1)))
  #B_comparison = pqP * pqP'
  #println( B_comparison ≈ reshape(pqrs, (n^2,n^2)) )
end

"""
    calc_df_integrals(EC::ECInfo)

  Calculate 3-index integrals and store them in `mmL` file.
  The routine is intended to be used in a combination with FDump integrals.
"""
function calc_df_integrals(EC::ECInfo)
  space_save, _ = restore_system_space!(EC)
  cMO = load_orbitals(EC)
  # correlated MOs
  SP = EC.space
  if is_restricted(cMO) && SP['o'] == SP['O']
    coMO = SpinMatrix(cMO[1][:,vcat(SP['o'],SP['v'])])
  else
    coMO = SpinMatrix(cMO[1][:,vcat(SP['o'],SP['v'])], cMO[2][:,vcat(SP['O'],SP['V'])])
  end
  generate_3idx_integrals(EC, coMO, "mpfit")
  restore_space!(EC, space_save)
end

"""
    eigen_decompose(T2mat, nvirt, nocc, tol=1e-6)

  Eigenvector-decompose symmetric doubles `T2[ai,bj]` matrix: 
  ``T^{ij}_{ab} = U^{iX}_a T_{XY} U^{jY}_b δ_{XY}``.
  Return ``U^iX_a`` as `U[a,i,X]` for ``T_{XX}`` > `tol`
"""
function eigen_decompose(T2mat, nvirt, nocc, tol=1e-6)
  Tval, U = eigen(Hermitian(-T2mat))
  naux = 0
  for s in Tval
    if -s < tol
      break
    end
    naux += 1
  end
  # display(Tval[1:naux])
  # println(naux)
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")
  U, S, = svd(Amat)
  # display(S)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  # display(S[1:naux])
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose(Amat, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose(Amat, tol=1e-6; verbose=true, description="")
  U, S, = svd(Amat)
  # display(S)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  # display(S[1:naux])
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return U[:,1:naux], S[1:naux]
end

""" 
    rotate_U2pseudocanonical(EC::ECInfo, UaiX)

  Diagonalize ϵv - ϵo transformed with UaiX (for update).
  Return eigenvalues and rotated UaiX
"""
function rotate_U2pseudocanonical(EC::ECInfo, UaiX)
  SP = EC.space
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  UaiX2 = deepcopy(UaiX)
  ϵo, ϵv = orbital_energies(EC)
  for a in 1:nvirt
    for i in 1:nocc
      UaiX2[a,i,:] *= ϵv[a] - ϵo[i]
    end
  end

  @mtensor Fdiff[X,Y] := conj(UaiX[a,i,X]) * UaiX2[a,i,Y]
  diagFdiff = eigen(Hermitian(Fdiff))

  @mtensor UaiX2[a,i,Y] = diagFdiff.vectors[X,Y] * UaiX[a,i,X]
  return eltype(Fdiff).(diagFdiff.values), UaiX2
end


end #module
