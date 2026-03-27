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

  # Cache type ALPACACache{T,R,S} is fully specified → all dispatch is static
  cache = ALPACACache(T, m, n, Val{S}(), descriptor.pairs)
  alpaca_pivots!(cache, matrix, options, descriptor)

  if S === :general
    return svd_finalize_general(cache, matrix, options.tol)
  else
    return nystrom_finalize(cache, matrix, options.tol)
  end
end

"""
    lpaca(matrix; principal=nothing, options=ALPACAOptions(...))

Low-rank Principal-element Adaptive Cross Approximation — the un-amended
variant of [`alpaca`](@ref).

Returns the raw factors from the pivot selection step,
without the Nyström eigendecomposition / SVD amendment step.
For symmetric/Hermitian this gives `left * left'` factors
(with `neg_indices` tracking sign flips for indefinite matrices);
for general matrices `left * right'`.
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

  cache = ALPACACache(T, m, n, Val{S}(), descriptor.pairs)
  alpaca_pivots!(cache, matrix, options, descriptor)

  # Use the factorization already computed during the pivot loop:
  #   cache.columns[:, 1:k] = L (deflated factor, scaled by pivot values)
  #   cache.pivot_diag[1:k] = d_k (pivot values)
  #   M ≈ L Lᵀ (symmetric) or L_C L_Rᵀ (general)
  k = cache.n_cols
  pivots = cache.pivot_indices[1:k]
  d = cache.pivot_diag[1:k]

  if S === :general
    # M ≈ L_C * D * L_Rᵀ.  Encode as left * right' by absorbing D into right:
    #   right[:, t] = conj(d_t) * conj(L_R[:, t])  so that  right'[t,:] = d_t * L_R[:,t]ᵀ
    left = cache.columns[:, 1:k] |> copy
    right = conj.(cache.rows[:, 1:k] .* transpose(d))
    row_pivots = cache.row_pivot_indices[1:k]
    return ALPACAResult{T}(left, right, Int[], pivots, row_pivots, :general)
  elseif S === :hermitian
    # d_k is real for Hermitian.  Scale: left[:,k] = L[:,k] * √|d_k|
    rd = real.(d)
    sqrt_abs_d = sqrt.(abs.(rd))
    L = cache.columns[:, 1:k] .* transpose(sqrt_abs_d)
    neg_indices = findall(<(zero(RT)), rd)
    return ALPACAResult{T}(copy(L), copy(L), neg_indices, pivots, Int[], :hermitian)
  else  # :symmetric
    if T <: Real
      sqrt_abs_d = sqrt.(abs.(d))
      L = cache.columns[:, 1:k] .* transpose(sqrt_abs_d)
      neg_indices = findall(<(zero(RT)), d)
    else
      # Complex symmetric: d_k is complex; absorb phase via √d_k
      sqrt_d = sqrt.(Complex.(d))
      L = cache.columns[:, 1:k] .* transpose(sqrt_d)
      neg_indices = Int[]
    end
    return ALPACAResult{T}(copy(L), copy(L), neg_indices, pivots, Int[], :symmetric)
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
                        max_rank::Integer)
  options !== nothing && return options
  tol === nothing && throw(ArgumentError("provide either `tol` or `options`"))
  if symmetry === nothing
    symmetry = _detect_symmetry(matrix)
  end
  pivotol = pivotol === nothing ? tol : pivotol
  return ALPACAOptions(; tol, pivotol, sigma, qr, symmetry, max_rank)
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
                max_rank::Integer=typemax(Int))
  options = _build_options(matrix; tol, symmetry, options, pivotol, sigma, qr, max_rank)
  return alpaca(DenseALPACAMatrix(matrix); principal, options)
end

"""
    lpaca(matrix; tol, [symmetry], [principal], [pivotol], [sigma], [max_rank])
    lpaca(matrix; options, [principal])

Convenience interface for dense matrices.  Same keyword interface as
[`alpaca`](@ref) but returns the un-amended result (raw pivot columns).
"""
function lpaca(matrix::AbstractMatrix;
               tol::Union{Real,Nothing}=nothing,
               principal=nothing,
               symmetry::Union{Symbol,Nothing}=nothing,
               options::Union{ALPACAOptions,Nothing}=nothing,
               pivotol::Union{Real,Nothing}=nothing,
               sigma::Real=0.01,
               max_rank::Integer=typemax(Int))
  options = _build_options(matrix; tol, symmetry, options, pivotol, sigma,
                           qr=false, max_rank)
  return lpaca(DenseALPACAMatrix(matrix); principal, options)
end