"""
    ALPACAOptions(; tol, pivotol=NaN, sigma=0.01, qr=false, symmetry=:symmetric, max_rank=typemax(Int), smooth_tol=0.5)

Configuration for ALPACA decompositions.

# Fields
- `tol::Float64`: convergence tolerance for the low-rank approximation.
- `pivotol::Float64`: pivot acceptance threshold. `NaN` (default) means
  auto-scale to `tol / √m` at the call site, where `m` is the row count.
  This ensures that singular values above `tol` are reliably detected even
  in large matrices where per-element magnitudes are much smaller than
  the singular values.
- `sigma::Float64`: batch-screening ratio for QR refinement (default `0.01`).
- `qr::Bool`: whether QR refinement is enabled.
- `symmetry::Symbol`: matrix class — `:symmetric`, `:hermitian`, or `:general`.
- `max_rank::Int`: upper bound on the rank of the approximation.
- `smooth_tol::Float64`: smooth truncation floor as a fraction of `pivotol`.
  Pivots with `|D| ∈ [pivotol * smooth_tol, pivotol]` are smoothly attenuated
  via Hermite smoothstep.  Set to `0.0` to disable (hard cutoff).
  Default `0.5`.
"""
struct ALPACAOptions
  tol::Float64
  pivotol::Float64
  sigma::Float64
  qr::Bool
  symmetry::Symbol
  max_rank::Int
  smooth_tol::Float64

  function ALPACAOptions(; tol::Real,
                         pivotol::Real=NaN,
                         sigma::Real=0.01,
                         qr::Bool=false,
                         symmetry::Symbol=:symmetric,
                         max_rank::Integer=typemax(Int),
                         smooth_tol::Real=0.5)
    if tol <= 0
      throw(ArgumentError("tol must be positive"))
    end
    if !isnan(pivotol) && pivotol <= 0
      throw(ArgumentError("pivotol must be positive or NaN (auto)"))
    end
    if sigma <= 0
      throw(ArgumentError("sigma must be positive"))
    end
    if max_rank <= 0
      throw(ArgumentError("max_rank must be positive"))
    end
    valid_symmetry = (:symmetric, :hermitian, :general)
    if symmetry ∉ valid_symmetry
      throw(ArgumentError("symmetry must be one of $(valid_symmetry)"))
    end
    if smooth_tol < 0 || smooth_tol >= 1
      throw(ArgumentError("smooth_tol must be in [0, 1)"))
    end
    return new(Float64(tol), Float64(pivotol), Float64(sigma), qr,
               symmetry, Int(max_rank), Float64(smooth_tol))
  end
end

"""
    resolve_pivotol(options::ALPACAOptions, m::Integer) → Float64

Resolve the effective pivot tolerance. If `options.pivotol` is `NaN`
(auto-scale), returns `options.tol / √m`. Otherwise returns `options.pivotol`.
"""
function resolve_pivotol(options::ALPACAOptions, m::Integer)
  isnan(options.pivotol) ? options.tol / sqrt(m) : options.pivotol
end

"""
    resolve_pivotol(options::ALPACAOptions, d_row::AbstractVector{<:Real}) → Float64

Resolve the effective pivot tolerance using row norms for adaptive scaling.
If `options.pivotol` is `NaN` (auto-scale), computes an effective dimensionality
``m_{\\text{eff}} = \\|d_{\\text{row}}\\|_1 / \\|d_{\\text{row}}\\|_\\infty``
and returns `options.tol / √m_eff`.  For uniformly distributed singular vectors
this equals `tol / √m`; for localized singular vectors ``m_{\\text{eff}} < m``,
yielding a less aggressive (larger) pivot tolerance.
"""
function resolve_pivotol(options::ALPACAOptions, d_row::AbstractVector{<:Real})
  isnan(options.pivotol) || return options.pivotol
  d_max = maximum(d_row)
  d_max <= 0 && return options.tol  # degenerate: fall back to tol
  m_eff = max(1.0, sum(d_row) / d_max)
  return options.tol / sqrt(m_eff)
end

"""
    ALPACAOptions(options::ALPACAOptions; kwargs...)

Copy-with-modification constructor.  Any keyword argument overrides
the corresponding field of `options`.
"""
function ALPACAOptions(options::ALPACAOptions; kwargs...)
  return ALPACAOptions(; tol=options.tol, pivotol=options.pivotol,
                       sigma=options.sigma, qr=options.qr,
                       symmetry=options.symmetry, max_rank=options.max_rank,
                       smooth_tol=options.smooth_tol,
                       kwargs...)
end

"""
    ALPACAResult{T}

Result of an ALPACA low-rank decomposition with element type `T`.

# Fields
- `left::Matrix{T}`: left factor (`n × r`).  For symmetric / Hermitian
  via `alpaca()`, columns are eigenvectors scaled by ``\\sqrt{|\\lambda_k|}``,
  so that ``A ≈ L \\, \\text{diag}(\\pm 1) \\, L^\\dagger``.
  The eigenvalues ``\\lambda_k`` are stored in the `eigenvalues` field.
  For `lpaca()`, columns are raw deflated pivot columns (unscaled).
  For general matrices, `A ≈ L R'`.
- `right::Matrix{T}`: right factor.  Equals `left` for symmetric / Hermitian;
  independent for general matrices where `A ≈ L R'`.
- `neg_indices::Vector{Int}`: column indices with negative eigenvalue.
  For backward compatibility with `lpaca()` results where eigenvalues may be empty.
- `pivot_indices::Vector{Int}`: accepted column pivot indices.
- `row_pivots::Vector{Int}`: accepted row pivot indices (general only; empty otherwise).
- `symmetry::Symbol`: matrix class (`:symmetric`, `:hermitian`, or `:general`).
- `eigenvalues::Vector{R}`: spectral values of the approximation.
  The semantics depend on the symmetry type:
  - **Real symmetric / Hermitian** via `alpaca()`: eigenvalues ``\\lambda_k``
    (can be negative) such that ``A ≈ L \\, \\text{diag}(\\lambda) \\, L^\\dagger``.
    Column ``k`` of ``L`` has norm ``\\sqrt{|\\lambda_k|}``
    and carries the corresponding eigenvector direction.
  - **Complex symmetric** via `alpaca()`: Takagi (singular) values ``\\sigma_k \\geq 0``
    such that ``A ≈ L \\, \\text{diag}(\\sigma) \\, L^T`` with Takagi vectors
    recoverable as ``L \\, \\text{diag}(1/\\sqrt{\\sigma})``.
  - **General** via `alpaca()`: singular values ``\\sigma_k \\geq 0`` such that
    ``A ≈ L \\, \\text{diag}(\\sigma) \\, R^\\dagger`` with orthonormal factors
    recoverable as ``L \\, \\text{diag}(1/\\sqrt{\\sigma})`` and
    ``R \\, \\text{diag}(1/\\sqrt{\\sigma})``.
  - Empty for `lpaca()` results.
"""
struct ALPACAResult{T, R<:Real}
  left::Matrix{T}
  right::Matrix{T}
  neg_indices::Vector{Int}
  pivot_indices::Vector{Int}
  row_pivots::Vector{Int}
  symmetry::Symbol
  eigenvalues::Vector{R}
end

function ALPACAResult{T}(left, right, neg, pivots, row_pivots, sym, eigenvalues::Vector{R}) where {T, R<:Real}
  return ALPACAResult{T, R}(left, right, neg, pivots, row_pivots, sym, eigenvalues)
end

function ALPACAResult{T}(left, right, neg, pivots, row_pivots, sym) where T
  return ALPACAResult{T, real(T)}(left, right, neg, pivots, row_pivots, sym, real(T)[])
end