# ──────────────────────────────────────────────────────────────────
# Decomposition extraction from ALPACAResult
# ──────────────────────────────────────────────────────────────────
#
# Each _extract_* function works on a pre-computed ALPACAResult.
# Convenience wrappers (alpaca_svd, lpaca_eigen, qrdalpaca_qr, ...)
# are generated at the bottom for all three ALPACA variants.
#
# Shared helpers:
#   _sign_diagonal     — build ±1 diagonal from neg_indices
#   _reduced_eigen     — QR-compress L, eigendecompose the reduced matrix
#   _svd_general       — dual-QR + SVD for general (or complex symmetric)
#
# Symmetry dispatch:
#   real symmetric + complex hermitian → _reduced_eigen path
#   complex symmetric → _svd_general with R = conj(L)
#   general → _svd_general with separate L, R

# ── Shared helpers ───────────────────────────────────────────────

"""
    _sign_diagonal(T, r, neg_indices) → Diagonal

Construct an `r×r` `Diagonal` matrix with entries ``±1`` of type `T`.
Entries at positions listed in `neg_indices` are ``-1``; all others are ``+1``.
"""
function _sign_diagonal(::Type{T}, r::Int, neg_indices::Vector{Int}) where T
  d = ones(T, r)
  for i in neg_indices
    d[i] = -one(T)
  end
  return Diagonal(d)
end

"""
    _reduced_eigen(L, neg_indices) → (Q, values, vectors)

QR-compress the factor matrix `L` and eigendecompose the reduced signed matrix
``R \\, D \\, R^\\dagger`` where ``L = Q R`` (thin QR) and
``D = \\text{diag}(\\pm 1)`` is determined by `neg_indices`.

Returns:
- `Q`: ``n × r`` matrix with orthonormal columns from the QR factorization of `L`.
- `values`: length-``r`` vector of real eigenvalues (ascending order).
- `vectors`: ``r × r`` eigenvector matrix in the compressed basis.

For real `T`, the reduced matrix is wrapped in `Symmetric`;
for complex `T`, in `Hermitian`.
"""
function _reduced_eigen(L::Matrix{T}, neg_indices::Vector{Int}) where T
  r = size(L, 2)
  RT = real(T)
  QL = qr(L)
  D = _sign_diagonal(RT, r, neg_indices)
  M = QL.R * D * QL.R'
  E = T <: Real ? eigen(Symmetric(M)) : eigen(Hermitian(M))
  return QL.Q, E.values, E.vectors
end

# ── SVD ──────────────────────────────────────────────────────────

"""
    _extract_svd(result::ALPACAResult) → (U, S, Vt)

Extract a thin SVD from an ALPACA low-rank factorization.

Returns a named tuple `(U, S, Vt)` such that ``A ≈ U \\, \\text{diag}(S) \\, V^\\dagger``.
Singular values `S` are sorted in descending order of magnitude.

Dispatches internally based on the symmetry type stored in `result`:
- **Real symmetric / complex Hermitian**: eigendecomposition of the reduced matrix
  via [`_reduced_eigen`](@ref), singular values are ``|\\lambda_k|``.
- **Complex symmetric**: delegates to [`_svd_general`](@ref) with ``R = \\bar{L}``,
  exploiting ``A ≈ L L^T = L \\overline{L}^\\dagger``.
- **General**: dual QR + SVD of the small core matrix via [`_svd_general`](@ref).
"""
function _extract_svd(result::ALPACAResult{T}) where T
  L = result.left
  R = result.right
  r = size(L, 2)
  if r == 0
    n = size(L, 1); m = size(R, 1)
    return (U = Matrix{T}(undef, n, 0),
            S = Vector{real(T)}(undef, 0),
            Vt = Matrix{T}(undef, 0, m))
  end

  sym = result.symmetry
  if sym == :general
    return _svd_general(L, R)
  elseif sym == :symmetric && T <: Complex
    # A ≈ L * Lᵀ = L * conj(L)^*
    return _svd_general(L, conj(L))
  else  # real symmetric or hermitian
    return _svd_symmetric_hermitian(L, result.neg_indices)
  end
end

"""
    _svd_general(L, R) → (U, S, Vt)

SVD extraction for a general low-rank factorization ``A ≈ L R^\\dagger``.

Computes thin QR of both `L` and `R`, then SVD of the small ``r × r``
core matrix ``R_L \\, R_R^\\dagger`` to obtain the full-size factors.

Also used for complex symmetric matrices with ``R = \\bar{L}``.
"""
function _svd_general(L::Matrix{T}, R::Matrix{T}) where T
  QL = qr(L)
  QR_f = qr(R)
  M = QL.R * QR_f.R'
  F = svd(M)
  U = QL.Q * F.U
  V = QR_f.Q * F.V
  return (U = U, S = F.S, Vt = Matrix(V'))
end

"""
    _svd_symmetric_hermitian(L, neg_indices) → (U, S, Vt)

SVD extraction for real symmetric or complex Hermitian factorization
``A ≈ L \\, D \\, L^\\dagger`` where ``D = \\text{diag}(\\pm 1)``.

Uses [`_reduced_eigen`](@ref) to obtain eigenvalues ``\\lambda_k`` and
eigenvectors of the reduced matrix, then converts to SVD form:
- ``S_k = |\\lambda_k|``
- ``U = Q \\, (V_k \\cdot \\text{sgn}(\\lambda_k))``
- ``V^\\dagger = (Q \\, V_k)^\\dagger``

Sorted by descending singular value.
"""
function _svd_symmetric_hermitian(L::Matrix{T}, neg_indices::Vector{Int}) where T
  Q, vals, vecs = _reduced_eigen(L, neg_indices)
  perm = sortperm(abs.(vals), rev=true)
  sorted_vals = vals[perm]
  sorted_vecs = vecs[:, perm]
  S = abs.(sorted_vals)
  signs = T <: Real ? T.(sign.(sorted_vals)) : complex.(sign.(sorted_vals))
  U = Q * (sorted_vecs .* transpose(signs))
  V = Q * sorted_vecs
  return (U = U, S = S, Vt = Matrix(V'))
end

# ── Eigendecomposition ───────────────────────────────────────────

"""
    _extract_eigen(result::ALPACAResult) → (values, vectors)

Extract eigenvalues and eigenvectors from an ALPACA low-rank factorization.

Returns a named tuple `(values, vectors)`:
- **Symmetric / Hermitian**: real eigenvalues in ascending order, with
  eigenvectors as columns of `vectors`.  Satisfies ``A \\, v_k ≈ \\lambda_k \\, v_k``.
- **General / complex symmetric**: eigenvalues of the small ``r × r`` matrix
  ``R^\\dagger L``, with approximate right eigenvectors of ``A``.  Sorted by
  descending ``|\\lambda|``; eigenvectors are normalized.

!!! note
    For general matrices, eigendecomposition requires a square matrix.
"""
function _extract_eigen(result::ALPACAResult{T}) where T
  L = result.left
  R = result.right
  r = size(L, 2)
  if r == 0
    n = size(L, 1)
    return (values = Vector{complex(T)}(undef, 0),
            vectors = Matrix{T}(undef, n, 0))
  end

  sym = result.symmetry
  if (sym == :symmetric && T <: Real) || sym == :hermitian
    return _eigen_symmetric_hermitian(L, result.neg_indices)
  else  # general or complex symmetric
    return _eigen_general(L, R)
  end
end

"""
    _eigen_symmetric_hermitian(L, neg_indices) → (values, vectors)

Eigendecomposition for real symmetric or complex Hermitian factorization
``A ≈ L \\, D \\, L^\\dagger``.

Uses [`_reduced_eigen`](@ref) and maps eigenvectors back to the full space
via ``Q \\cdot V_k``.  Eigenvalues are returned in ascending order.
"""
function _eigen_symmetric_hermitian(L::Matrix{T}, neg_indices::Vector{Int}) where T
  Q, vals, vecs = _reduced_eigen(L, neg_indices)
  return (values = vals, vectors = Q * vecs)
end

"""
    _eigen_general(L, R) → (values, vectors)

Eigendecomposition for a general low-rank factorization ``A ≈ L R^\\dagger``.

Computes eigenvalues of the small matrix ``R^\\dagger L``, which share the
non-zero eigenvalues of ``A``.  Right eigenvectors are obtained via
``v_k = L w_k / \\|L w_k\\|`` where ``w_k`` is an eigenvector of ``R^\\dagger L``.

Eigenvalues are sorted by descending ``|\\lambda|``.

!!! warning
    Requires a square matrix (``m = n``).
"""
function _eigen_general(L::Matrix{T}, R::Matrix{T}) where T
  n = size(L, 1)
  m = size(R, 1)
  if n != m
    throw(ArgumentError(
      "Eigendecomposition requires a square matrix (got $(n)×$(m))"))
  end
  # Eigenvalues of A ≈ L * R' are eigenvalues of the small matrix R' * L
  M = R' * L
  E = eigen(M)
  perm = sortperm(abs.(E.values), rev=true)
  vals = E.values[perm]
  # Right eigenvectors: if M*v = λ*v, then (L*R')*(L*v) = L*(M*v) = λ*(L*v)
  vecs = L * E.vectors[:, perm]
  # Normalize eigenvectors
  for j in axes(vecs, 2)
    nrm = norm(@view vecs[:, j])
    if nrm > 0
      vecs[:, j] ./= nrm
    end
  end
  return (values = vals, vectors = vecs)
end

# ── Takagi decomposition ─────────────────────────────────────────

"""
    _extract_takagi(result::ALPACAResult) → (U, D)

Extract the Autonne-Takagi decomposition ``A = U \\, \\text{diag}(D) \\, U^T``
from an ALPACA factorization of a complex symmetric matrix.

Returns a named tuple `(U, D)`:
- `D`: non-negative real Takagi values, sorted in descending order.
- `U`: unitary matrix of Takagi vectors satisfying ``U^\\dagger U = I``.

Only supported for complex symmetric input (`symmetry = :symmetric`
with `T <: Complex`).  For real symmetric matrices, the Takagi decomposition
reduces to the eigendecomposition with absolute values;
use [`_extract_eigen`](@ref) instead.

# Algorithm
1. Thin QR: ``L = Q R``.
2. Form the small complex symmetric matrix ``M = R R^T`` (transpose, not adjoint).
3. SVD: ``M = F_U \\, \\Sigma \\, F_V^\\dagger``.
4. Autonne-Takagi phase correction: ``p_k = \\sum_i (F_U)_{ik} (F_V)_{ik}``
   (element-wise product, no conjugation).
5. Takagi vectors: ``W = Q \\, F_U \\, \\text{diag}(\\sqrt{\\bar{p}})``
   so that ``M = W \\, \\Sigma \\, W^T``.
"""
function _extract_takagi(result::ALPACAResult{T}) where T
  if !(T <: Complex && result.symmetry == :symmetric)
    throw(ArgumentError(
      "Takagi decomposition requires complex symmetric input " *
      "(symmetry=:symmetric with complex element type)"))
  end
  L = result.left
  r = size(L, 2)
  if r == 0
    n = size(L, 1)
    return (U = Matrix{T}(undef, n, 0),
            D = Vector{real(T)}(undef, 0))
  end

  QL = qr(L)
  M = QL.R * transpose(QL.R)  # r×r complex symmetric (no conjugate)
  F = svd(M)
  nk = size(F.U, 2)

  # Autonne-Takagi phase correction: p_k = Σ_i U[i,k]*V[i,k] (no conjugation)
  phases = [sum(F.U[:, m] .* F.V[:, m]) for m in 1:nk]
  U_takagi = QL.Q * (F.U .* transpose(sqrt.(conj.(phases))))

  perm = sortperm(F.S, rev=true)
  return (U = U_takagi[:, perm], D = F.S[perm])
end

# ── QR decomposition ────────────────────────────────────────────

"""
    _extract_qr(result::ALPACAResult) → (Q, R)

Extract a thin QR factorization from an ALPACA low-rank approximation.

Returns a named tuple `(Q, R)` where `Q` is ``n × r`` with orthonormal columns
and ``A ≈ Q R``.

The `R` factor depends on the symmetry type:
- **Real symmetric / Hermitian**: ``R = R_\\text{upper} \\, D \\, L^\\dagger``
- **Complex symmetric**: ``R = R_\\text{upper} \\, L^T``
- **General**: ``R = R_\\text{upper} \\, R_\\text{right}^\\dagger``

where ``L = Q \\, R_\\text{upper}`` is the thin QR of the left factor.
"""
function _extract_qr(result::ALPACAResult{T}) where T
  L = result.left
  R = result.right
  r = size(L, 2)
  if r == 0
    n = size(L, 1); m = size(R, 1)
    return (Q = Matrix{T}(undef, n, 0),
            R = Matrix{T}(undef, 0, m))
  end

  sym = result.symmetry
  QL = qr(L)
  Q = QL.Q * Matrix{T}(I, r, r)

  R_part = if sym == :general
    QL.R * R'
  elseif sym == :symmetric && T <: Complex
    QL.R * transpose(L)
  else  # real symmetric or hermitian
    D = _sign_diagonal(real(T), r, result.neg_indices)
    QL.R * D * L'
  end

  return (Q = Q, R = R_part)
end

# ── Convenience wrappers ─────────────────────────────────────────
# Generate alpaca_svd, lpaca_svd, qrdalpaca_svd, etc.

const _DECOMP_VARIANTS = (
  (:alpaca, "ALPACA (Amended Low-rank Principal-element Adaptive Cross Approximation)"),
  (:lpaca, "LPACA (un-amended variant, raw pivot columns)"),
  (:qrdalpaca, "QRdALPACA (ALPACA with QR-pivoted refinement)"),
)

const _DECOMP_TYPES = (
  (:_extract_svd, :svd, "(U, S, Vt)", "``A ≈ U \\, \\text{diag}(S) \\, V^\\dagger``"),
  (:_extract_eigen, :eigen, "(values, vectors)", "``A \\, v_k ≈ \\lambda_k \\, v_k``"),
  (:_extract_takagi, :takagi, "(U, D)", "``A = U \\, \\text{diag}(D) \\, U^T`` (complex symmetric only)"),
  (:_extract_qr, :qr, "(Q, R)", "``A ≈ Q R``"),
)

for (run_func, variant_desc) in _DECOMP_VARIANTS
  for (extract_func, suffix, returns, equation) in _DECOMP_TYPES
    fname = Symbol(run_func, :_, suffix)
    docstring = """
        $fname(result::ALPACAResult)
        $fname(matrix::AbstractMatrix; kwargs...)
        $fname(matrix::AbstractALPACAMatrix; kwargs...)

    Compute a low-rank $suffix decomposition via $variant_desc.

    Returns a named tuple `$returns` satisfying $equation.

    When called with a matrix, all keyword arguments are forwarded to
    [`$run_func`](@ref).  When called with a pre-computed [`ALPACAResult`](@ref),
    only the extraction step is performed.
    """
    @eval begin
      @doc $docstring
      $fname(result::ALPACAResult) = $extract_func(result)
      function $fname(matrix::AbstractMatrix; kwargs...)
        result = $run_func(matrix; kwargs...)
        return $extract_func(result)
      end
      function $fname(matrix::AbstractALPACAMatrix; kwargs...)
        result = $run_func(matrix; kwargs...)
        return $extract_func(result)
      end
    end
  end
end
