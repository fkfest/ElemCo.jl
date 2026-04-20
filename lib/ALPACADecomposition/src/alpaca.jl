function _validate_matrix(matrix, symmetry::Symbol)
  m, n = size(matrix)
  if symmetry != :general && m != n
    throw(ArgumentError("symmetry=:$(symmetry) requires a square matrix; got size $(m)×$(n)"))
  end
  return m, n
end

"""
    alpaca(matrix; principal=nothing, options=ALPACAOptions(...))

Compute a low-rank decomposition of a square matrix using the ALPACA algorithm
(Amended Low-rank Principal-element Adaptive Cross Approximation).

The amendment step uses QR-compressed eigendecomposition (symmetric/Hermitian),
SVD (general), or Takagi factorization (complex symmetric) of the pivot-loop
factors to remove redundancies and truncate small eigenvalues/singular values.

By default (`principal=nothing`), the diagonal entries are used as principal
elements.  Passing `principal=Tuple{Int,Int}[]` (empty) disables all principal
element monitoring, reducing the pivot selection to pure Adaptive Cross
Approximation (ACA).  The finalization step is still applied.

Returns an `ALPACAResult` with left/right factors, negative-sign indices,
and accepted pivot indices.
"""
function alpaca(matrix::AbstractALPACAMatrix;
                principal=nothing,
                options::ALPACAOptions)
  m, n = _validate_matrix(matrix, options.symmetry)
  descriptor = normalize_principal_descriptor(options.symmetry, min(m, n), principal)
  # Function barrier: Val(symmetry) makes S a compile-time type parameter
  # for all downstream dispatch (cache construction, pivot loop, finalization).
  return _alpaca_impl(matrix, m, n, descriptor, options, Val(options.symmetry))
end

function _alpaca_impl(matrix::AbstractALPACAMatrix{T},
                      m::Int, n::Int,
                      descriptor::AbstractPrincipalDescriptor,
                      options::ALPACAOptions,
                      ::Val{S}) where {T, S}

  cache = ALPACACache(T, m, n, Val{S}(), descriptor.pairs)
  alpaca_pivots!(cache, matrix, options, descriptor)
  return decomposition_finalize(cache, options.tol; smooth_tol=options.smooth_tol)
end

# ──────────────────────────────────────────────────────────────────
# Decomposition-based finalization
# ──────────────────────────────────────────────────────────────────

"""
    decomposition_finalize(cache::ALPACACache, tol; smooth_tol=0.0) → ALPACAResult

Build the low-rank factorization from the pivot-loop factors stored in
the cache.  When `smooth_tol > 0`, borderline pivot values (between
`pivotol * smooth_tol` and `pivotol`) are attenuated via smoothstep
scaling before the decomposition step.

Dispatches to:
- [`_decomposition_finalize_general`](@ref) for general matrices (dual-QR + SVD),
- [`_decomp_finalize_eigen`](@ref) for real symmetric / Hermitian (QR + eigendecomposition),
- [`_decomp_finalize_takagi`](@ref) for complex symmetric (QR + Autonne-Takagi).
"""
function decomposition_finalize(cache::ALPACACache{T,R,S}, tol;
                                column_qr=nothing,
                                smooth_tol::Real=0.0) where {T,R,S}
  if cache.n_cols == 0
    m = length(cache.cbuf)
    ncols = S === :general ? length(cache.rbuf) : m
    return ALPACAResult{T}(
      Matrix{T}(undef, m, 0), Matrix{T}(undef, ncols, 0),
      Int[], cache.pivot_indices, Int[], S, R[])
  end
  # Apply smooth scaling to borderline pivots before decomposition
  if smooth_tol > 0
    _apply_smooth_pivot_scaling!(cache.pivot_diag, cache.n_cols, cache.pivotol, R(smooth_tol))
  end
  if S === :general
    return _decomposition_finalize_general(cache, tol; column_qr)
  elseif S === :symmetric && !(T <: Real)
    return _decomp_finalize_takagi(cache, tol; column_qr)
  else
    return _decomp_finalize_eigen(cache, tol, S; column_qr)
  end
end

"""
    _decomp_finalize_eigen(cache, tol, sym) → ALPACAResult

QR-compressed eigendecomposition finalization for real symmetric or
complex Hermitian factorizations ``M ≈ C \\, D \\, C^\\dagger``.

Operates directly on the raw factors from `alpaca_pivots!`:
  1. QR decompose ``C = Q R``
  2. Eigendecompose ``R \\, D \\, R^\\dagger``
  3. Truncate eigenvalues with ``|\\lambda| < \\text{tol}``
  4. Build ``L = Q \\, V \\, \\sqrt{|\\lambda|}`` (scaled eigenvectors)

The result stores `left` columns scaled by ``\\sqrt{|\\lambda|}`` and
the `eigenvalues` vector, such that
``A ≈ L \\, D_{\\pm} \\, L^\\dagger`` where ``D_{\\pm} = \\text{diag}(\\pm 1)``.
The eigenvalues enable fast SVD/eigen extraction without re-doing QR+eigen.
"""
function _decomp_finalize_eigen(cache::ALPACACache{T,R}, tol, sym::Symbol;
                                column_qr=nothing) where {T,R}
  k = cache.n_cols
  d = real.(@view cache.pivot_diag[1:k])
  n = size(cache.columns, 1)

  if column_qr !== nothing
    QC = column_qr
  else
    QC = qr!(cache.columns[:, 1:k])
  end
  M_small = QC.R * Diagonal(d) * QC.R'
  E = T <: Real ? eigen(Symmetric(M_small)) : eigen(Hermitian(M_small))

  # Truncate small eigenvalues, sort by descending magnitude
  keep = sortperm(abs.(E.values), rev=true)
  nk = count(s -> abs(s) > tol, E.values)
  if nk == 0
    return ALPACAResult{T}(
      Matrix{T}(undef, n, 0), Matrix{T}(undef, n, 0),
      Int[], cache.pivot_indices, Int[], sym, R[])
  end
  keep = keep[1:nk]
  kept_vals = R.(E.values[keep])
  kept_vecs = E.vectors[:, keep]

  # Scaled eigenvectors: A ≈ L * D_± * L' where L = Q * V * sqrt(|λ|)
  sqrt_vals = sqrt.(abs.(kept_vals))
  L_final = QC.Q * (kept_vecs .* transpose(sqrt_vals))
  neg_final = findall(v -> v < 0, kept_vals)

  return ALPACAResult{T}(L_final, L_final, neg_final, cache.pivot_indices, Int[], sym, kept_vals)
end

"""
    _decomp_finalize_takagi(cache, tol) → ALPACAResult

SVD/Takagi finalization for complex symmetric factorizations
``M ≈ C \\, D \\, C^T``.

Operates directly on the raw factors from `alpaca_pivots!`:
  1. QR decompose ``C = Q R``
  2. SVD of ``R \\, D \\, R^T`` with Autonne-Takagi phase correction
  3. Truncate singular values below `tol`
  4. Build ``L_\\text{final}`` from Takagi vectors and singular values
"""
function _decomp_finalize_takagi(cache::ALPACACache{T}, tol;
                                 column_qr=nothing) where T
  k = cache.n_cols
  d = @view cache.pivot_diag[1:k]
  n = size(cache.columns, 1)

  if column_qr !== nothing
    QC = column_qr
  else
    QC = qr!(cache.columns[:, 1:k])
  end
  M_small = QC.R * Diagonal(d) * transpose(QC.R)  # complex symmetric
  F = svd(M_small)

  # Truncate small singular values
  nk = count(s -> s > tol, F.S)
  if nk == 0
    return ALPACAResult{T}(
      Matrix{T}(undef, n, 0), Matrix{T}(undef, n, 0),
      Int[], cache.pivot_indices, Int[], :symmetric, real(T)[])
  end

  # Autonne-Takagi phase correction
  phases = [sum(F.U[:, m] .* F.V[:, m]) for m in 1:nk]
  sqrt_phases = sqrt.(conj.(phases))
  takagi_values = F.S[1:nk]
  sqrt_S = sqrt.(takagi_values)
  L_final = QC.Q * (F.U[:, 1:nk] .* transpose(sqrt_phases) .* transpose(sqrt_S))

  return ALPACAResult{T}(L_final, L_final, Int[], cache.pivot_indices, Int[], :symmetric, takagi_values)
end

"""
    _decomposition_finalize_general(cache, tol) → ALPACAResult

Dual-QR + SVD finalization for general matrix factorizations
``M ≈ C_L \\, D \\, C_R^T``.

Operates directly on the raw factors from `alpaca_pivots!`.
"""
function _decomposition_finalize_general(cache::ALPACACache{T,R,:general}, tol;
                                         column_qr=nothing) where {T,R}
  k = cache.n_cols
  m = length(cache.cbuf)
  ncols = length(cache.rbuf)
  col_pivots = copy(cache.pivot_indices)
  row_pivots = copy(cache.row_pivot_indices[1:k])

  d = @view cache.pivot_diag[1:k]

  # Dual QR + SVD of core: R_L * diag(d) * R_R^T
  if column_qr !== nothing
    QL = column_qr
  else
    QL = qr!(cache.columns[:, 1:k])
  end
  QR_f = qr!(cache.rows[:, 1:k])
  Core = QL.R * Diagonal(d) * transpose(QR_f.R)
  F = svd(Core)

  nk = count(s -> s > tol, F.S)
  if nk == 0
    return ALPACAResult{T}(
      Matrix{T}(undef, m, 0), Matrix{T}(undef, ncols, 0),
      Int[], col_pivots, row_pivots, :general, R[])
  end

  singular_values = F.S[1:nk]
  sqrt_S = sqrt.(singular_values)
  left_final = QL.Q * (F.U[:, 1:nk] .* transpose(sqrt_S))
  # Factorization is A ≈ C_L * D * C_R^T (transpose), but reconstruction uses
  # adjoint: A = left * right'.  For complex T, conjugate to compensate.
  right_final = conj.(QR_f.Q * (conj.(F.V[:, 1:nk]) .* transpose(sqrt_S)))

  return ALPACAResult{T}(left_final, right_final, Int[], col_pivots, row_pivots, :general, singular_values)
end

"""
    lpaca(matrix; principal=nothing, options=ALPACAOptions(...))

Low-rank Principal-element Adaptive Cross Approximation — the raw-factor un-amended
variant of [`alpaca`](@ref).

Returns the raw factors directly from the pivot selection step,
without the QR-compressed eigendecomposition / SVD finalization step.
For symmetric/Hermitian this gives `left * left'` factors
(with `neg_indices` tracking sign flips for indefinite matrices);
for general matrices `left * right'`.

With `principal=Tuple{Int,Int}[]` (empty), no principal elements are monitored
and the algorithm reduces to pure Adaptive Cross Approximation (ACA).
"""
function lpaca(matrix::AbstractALPACAMatrix;
               principal=nothing,
               options::ALPACAOptions)
  m, n = _validate_matrix(matrix, options.symmetry)
  descriptor = normalize_principal_descriptor(options.symmetry, min(m, n), principal)
  return _lpaca_impl(matrix, m, n, descriptor, options, Val(options.symmetry))
end

function _lpaca_impl(matrix::AbstractALPACAMatrix{T},
                     m::Int, n::Int,
                     descriptor::AbstractPrincipalDescriptor,
                     options::ALPACAOptions,
                     ::Val{S}) where {T, S}
  RT = real(T)

  # Disable smooth scaling for lpaca: no finalization step to truncate
  # borderline pivots, so the pivot loop must not extend below tol.
  opts = options.smooth_tol > 0 ? ALPACAOptions(options; smooth_tol=0.0) : options
  cache = ALPACACache(T, m, n, Val{S}(), descriptor.pairs)
  alpaca_pivots!(cache, matrix, opts, descriptor)
  _scale_pivot_columns!(cache)

  # After _scale_pivot_columns!:
  #   symmetric/hermitian: columns are scaled by √|d|, pivot_diag holds ±1 signs
  #   complex symmetric:   columns are scaled by √d,   pivot_diag holds ones
  #   general:             columns have D absorbed,     pivot_diag holds ones
  k = cache.n_cols
  pivots = cache.pivot_indices[1:k]

  if S === :general
    left = copy(cache.columns[:, 1:k])
    right = conj.(cache.rows[:, 1:k])
    row_pivots = cache.row_pivot_indices[1:k]
    return ALPACAResult{T}(left, right, Int[], pivots, row_pivots, :general, RT[])
  elseif S === :symmetric && (T <: Complex)
    L = copy(cache.columns[:, 1:k])
    return ALPACAResult{T}(L, L, Int[], pivots, Int[], :symmetric, RT[])
  else
    L = copy(cache.columns[:, 1:k])
    d_sign = real.(cache.pivot_diag[1:k])
    neg_indices = findall(<(zero(RT)), d_sign)
    return ALPACAResult{T}(L, L, neg_indices, pivots, Int[], S, RT[])
  end
end

"""
    _build_options(matrix; tol, symmetry, pivotol, sigma, qr, max_rank, options) → (ALPACAOptions, Symbol)

Shared option construction for the [`alpaca`](@ref), [`lpaca`](@ref),
and [`qrdalpaca`](@ref) convenience wrappers.  Auto-detects symmetry
from the matrix when not explicitly provided.

Returns the constructed `ALPACAOptions`.
"""
function _build_options(matrix::AbstractMatrix;
                        tol::Union{Real,Nothing},
                        symmetry::Union{Symbol,Nothing},
                        options::Union{ALPACAOptions,Nothing},
                        pivotol::Union{Real,Nothing},
                        sigma::Real,
                        qr::Bool,
                        max_rank::Integer,
                        smooth_tol::Real=0.5)
  options !== nothing && return options
  tol === nothing && throw(ArgumentError("provide either `tol` or `options`"))
  if symmetry === nothing
    symmetry = _detect_symmetry(matrix)
  end
  if pivotol !== nothing
    return ALPACAOptions(; tol, pivotol, sigma, qr, symmetry, max_rank, smooth_tol)
  end
  return ALPACAOptions(; tol, sigma, qr, symmetry, max_rank, smooth_tol)
end

"""
    _detect_symmetry(matrix) → Symbol

Detect the symmetry type of `matrix` from its wrapper type.
Returns `:hermitian` for `Hermitian`, `:symmetric` for `Symmetric`
or real `Hermitian`, or `:general` for anything else.
No element-by-element check is performed.
"""
_detect_symmetry(::AbstractMatrix) = :general
_detect_symmetry(::Symmetric) = :symmetric
function _detect_symmetry(matrix::Hermitian)
  return eltype(matrix) <: Complex ? :hermitian : :symmetric
end

"""
    alpaca(matrix; tol, [symmetry], [principal], [pivotol], [sigma], [qr], [max_rank])
    alpaca(matrix; options, [principal])

Convenience interface for dense matrices.  When `tol` is given, automatically
detects `:symmetric`, `:hermitian`, or `:general` from the matrix (override
with `symmetry`).  Alternatively, pass a pre-built `ALPACAOptions` via `options`.

# Arguments
- `tol::Real`: convergence tolerance.
- `symmetry::Symbol`: override auto-detection (`:symmetric`, `:hermitian`, `:general`).
- `principal`: principal element descriptor (default: diagonal pairs).
    Pass `Tuple{Int,Int}[]` for pure ACA (no principal monitoring).
- `pivotol`, `sigma`, `qr`, `max_rank`: forwarded to `ALPACAOptions`.
- `options::ALPACAOptions`: use instead of individual keyword arguments.
"""
function alpaca(matrix::AbstractMatrix;
                tol::Union{Real,Nothing}=nothing,
                principal=nothing,
                symmetry::Union{Symbol,Nothing}=nothing,
                options::Union{ALPACAOptions,Nothing}=nothing,
                pivotol::Union{Real,Nothing}=nothing,
                sigma::Real=0.01,
                qr::Bool=false,
                max_rank::Integer=typemax(Int),
                smooth_tol::Real=0.5)
  options = _build_options(matrix; tol, symmetry, options, pivotol, sigma, qr, max_rank, smooth_tol)
  return alpaca(DenseALPACAMatrix(matrix); principal, options)
end

"""
    lpaca(matrix; tol, [symmetry], [principal], [pivotol], [sigma], [max_rank])
    lpaca(matrix; options, [principal])

Convenience interface for dense matrices.  Same keyword interface as
[`alpaca`](@ref) but returns the un-amended result (raw pivot columns).
With `principal=Tuple{Int,Int}[]`, reduces to pure ACA.
"""
function lpaca(matrix::AbstractMatrix;
               tol::Union{Real,Nothing}=nothing,
               principal=nothing,
               symmetry::Union{Symbol,Nothing}=nothing,
               options::Union{ALPACAOptions,Nothing}=nothing,
               pivotol::Union{Real,Nothing}=nothing,
               sigma::Real=0.01,
               max_rank::Integer=typemax(Int),
               smooth_tol::Real=0.5)
  options = _build_options(matrix; tol, symmetry, options, pivotol, sigma,
                           qr=false, max_rank, smooth_tol)
  return lpaca(DenseALPACAMatrix(matrix); principal, options)
end