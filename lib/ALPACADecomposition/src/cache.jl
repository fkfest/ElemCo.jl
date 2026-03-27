"""
    ALPACACache{T,R,S}

Pre-allocated cache for zero-allocation ALPACA inner loop.

`T` is the element type (e.g. `Float64`, `ComplexF64`),
`R` is the corresponding real type (e.g. `Float64`),
`S` is the symmetry symbol (`:symmetric`, `:hermitian`,
or `:general`).

The symmetry is baked into the type so that dispatch-based specialization
eliminates runtime `if` branches in the hot path.

Stores fetched columns, rows (general case), principal element residuals,
and work buffers. All storage grows by amortized doubling.
"""
mutable struct ALPACACache{T,R<:Real,S}
  # Column cache: columns[:,1:n_cols] stores deflated pivot columns
  columns::Matrix{T}
  # Original (undeflated) pivot columns for Nyström/SVD finalization
  orig_columns::Matrix{T}
  n_cols::Int
  col_index::Vector{Int}    # col_index[k] = original column index of columns[:,k]
  col_map::Vector{Int}      # col_map[j] = cache slot for column j (0 = not fetched)

  # Row cache (general case only): rows[:,1:n_rows] stores deflated pivot rows
  rows::Matrix{T}
  # Original (undeflated) pivot rows (general case only)
  orig_rows::Matrix{T}
  n_rows::Int
  row_index::Vector{Int}
  row_map::Vector{Int}

  # Principal element cache
  principal_pairs::Vector{Tuple{Int,Int}}
  principal_values::Vector{T}     # residual values (updated in-place)
  principal_fetched::Bool         # true after elements!() call

  # Pivot bookkeeping
  pivot_indices::Vector{Int}      # accepted pivot column indices
  pivot_diag::Vector{T}           # pivot diagonal values d[k] for Schur complement
  is_pivot::BitVector             # O(1) pivot lookup

  # Work buffers (allocated once, never reallocated unless capacity grows)
  cbuf::Vector{T}          # column work buffer (length n)
  rbuf::Vector{T}          # row work buffer (length n)
  coeffs::Vector{T}        # deflation coefficients (length ≥ capacity)
  ebuf::Vector{T}          # elements! buffer (length = number of principal pairs)

  # Row pivot tracking (general case only)
  row_pivot_indices::Vector{Int}
  is_row_pivot::BitVector
end

const INITIAL_CAPACITY = 256

"""
    ALPACACache(T, m, n, symmetry, principal_pairs)

Create a cache for an m×n matrix with element type `T`.
For symmetric/hermitian, m must equal n.
The `symmetry` argument (a `Symbol`) becomes a compile-time type parameter
so that downstream functions can dispatch on it.
"""
ALPACACache(::Type{T}, m::Int, n::Int, symmetry::Symbol,
            pairs::Vector{Tuple{Int,Int}}) where T =
  ALPACACache(T, m, n, Val(symmetry), pairs)

# Convenience: square matrix (backward compatible)
ALPACACache(::Type{T}, n::Int, symmetry::Symbol,
            pairs::Vector{Tuple{Int,Int}}) where T =
  ALPACACache(T, n, n, Val(symmetry), pairs)

ALPACACache(::Type{T}, n::Int, v::Val,
            pairs::Vector{Tuple{Int,Int}}) where T =
  ALPACACache(T, n, n, v, pairs)

function ALPACACache(::Type{T}, m::Int, n::Int, ::Val{S},
                     pairs::Vector{Tuple{Int,Int}}) where {T, S}
  R = real(T)
  cap = min(min(m, n), INITIAL_CAPACITY)
  np = length(pairs)

  use_rows = S == :general

  # columns have m rows (one entry per matrix row)
  columns = Matrix{T}(undef, m, cap)
  orig_columns = Matrix{T}(undef, m, cap)
  # rows have n entries (one entry per matrix column), general only
  rows = use_rows ? Matrix{T}(undef, n, cap) : Matrix{T}(undef, 0, 0)
  orig_rows = use_rows ? Matrix{T}(undef, n, cap) : Matrix{T}(undef, 0, 0)

  ALPACACache{T,R,S}(
    columns, orig_columns, 0,
    Vector{Int}(undef, cap),
    zeros(Int, n),              # col_map: maps column index j (1..n)
    rows, orig_rows, 0,
    use_rows ? Vector{Int}(undef, cap) : Int[],
    use_rows ? zeros(Int, m) : Int[],  # row_map: maps row index i (1..m)
    copy(pairs),
    Vector{T}(undef, np),
    false,
    Int[],
    Vector{T}(undef, cap),
    falses(n),              # is_pivot: tracks column indices (1..n)
    Vector{T}(undef, m),    # cbuf: column work buffer (length m)
    Vector{T}(undef, n),    # rbuf: row work buffer (length n)
    Vector{T}(undef, cap),
    Vector{T}(undef, np),
    Int[],
    use_rows ? falses(m) : falses(0)  # is_row_pivot: tracks row indices (1..m)
  )
end

"""
    _ensure_col_capacity!(cache, needed)

Grow column storage by amortized doubling if needed.
"""
function _ensure_col_capacity!(cache::ALPACACache{T}, needed::Int) where T
  current_cap = size(cache.columns, 2)
  needed <= current_cap && return
  new_cap = max(2 * current_cap, needed)
  n = size(cache.columns, 1)

  new_cols = Matrix{T}(undef, n, new_cap)
  @views new_cols[:, 1:cache.n_cols] .= cache.columns[:, 1:cache.n_cols]
  cache.columns = new_cols

  new_orig = Matrix{T}(undef, n, new_cap)
  @views new_orig[:, 1:cache.n_cols] .= cache.orig_columns[:, 1:cache.n_cols]
  cache.orig_columns = new_orig

  new_idx = Vector{Int}(undef, new_cap)
  @views new_idx[1:cache.n_cols] .= cache.col_index[1:cache.n_cols]
  cache.col_index = new_idx

  new_diag = Vector{T}(undef, new_cap)
  @views new_diag[1:cache.n_cols] .= cache.pivot_diag[1:cache.n_cols]
  cache.pivot_diag = new_diag

  cache.coeffs = Vector{T}(undef, new_cap)
  return
end

"""
    _ensure_row_capacity!(cache, needed)

Grow row storage by amortized doubling if needed (general case only).
"""
function _ensure_row_capacity!(cache::ALPACACache{T}, needed::Int) where T
  current_cap = size(cache.rows, 2)
  needed <= current_cap && return
  new_cap = max(2 * current_cap, needed)
  n = size(cache.rows, 1)

  new_rows = Matrix{T}(undef, n, new_cap)
  @views new_rows[:, 1:cache.n_rows] .= cache.rows[:, 1:cache.n_rows]
  cache.rows = new_rows

  new_orig = Matrix{T}(undef, n, new_cap)
  @views new_orig[:, 1:cache.n_rows] .= cache.orig_rows[:, 1:cache.n_rows]
  cache.orig_rows = new_orig

  new_idx = Vector{Int}(undef, new_cap)
  @views new_idx[1:cache.n_rows] .= cache.row_index[1:cache.n_rows]
  cache.row_index = new_idx
  return
end

"""
    store_column!(cache, j, col)

Store a deflated column in the cache. Returns the cache slot index.
"""
function store_column!(cache::ALPACACache{T}, j::Int, col::AbstractVector{T}) where T
  _ensure_col_capacity!(cache, cache.n_cols + 1)
  cache.n_cols += 1
  k = cache.n_cols
  @views cache.columns[:, k] .= col
  cache.col_index[k] = j
  cache.col_map[j] = k
  return k
end

"""
    store_row!(cache, i, row)

Store a deflated row in the cache (general case). Returns the cache slot index.
"""
function store_row!(cache::ALPACACache{T}, i::Int, row::AbstractVector{T}) where T
  _ensure_row_capacity!(cache, cache.n_rows + 1)
  cache.n_rows += 1
  k = cache.n_rows
  @views cache.rows[:, k] .= row
  cache.row_index[k] = i
  cache.row_map[i] = k
  return k
end

"""
    store_pivot!(cache, j, dval)

Record that column j has been accepted as a pivot with diagonal value dval.
"""
function store_pivot!(cache::ALPACACache, j::Int, dval)
  push!(cache.pivot_indices, j)
  cache.is_pivot[j] = true
  k = length(cache.pivot_indices)
  cache.pivot_diag[k] = dval
  return
end

"""
    init_principal_values!(cache, matrix, descriptor)

Initialize principal values in the cache. For `PrincipalPairs`, calls
`elements!` exactly once. For `PrincipalTriples`, copies values directly.
"""
function init_principal_values!(cache::ALPACACache{T},
                                matrix::AbstractALPACAMatrix,
                                descriptor::PrincipalPairs) where T
  cache.principal_fetched && return
  elements!(cache.ebuf, matrix, cache.principal_pairs)
  copyto!(cache.principal_values, cache.ebuf)
  cache.principal_fetched = true
  return
end

function init_principal_values!(cache::ALPACACache{T},
                                ::AbstractALPACAMatrix,
                                descriptor::PrincipalTriples) where T
  cache.principal_fetched && return
  copyto!(cache.principal_values, descriptor.values)
  cache.principal_fetched = true
  return
end
