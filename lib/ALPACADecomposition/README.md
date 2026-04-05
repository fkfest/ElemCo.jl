# ALPACADecomposition.jl

ALPACA (**A**mended **L**ow-rank **P**rincipal-element **A**daptive **C**ross **A**pproximation) is a standalone Julia package for low-rank matrix decomposition with minimal element access.

## Overview

The algorithm combines two pivoting signals:

1. **Principal-element pivots** (primary) — residuals of user-provided matrix elements
   (defaulting to the diagonal for square matrices), tracked throughout the
   decomposition.
2. **ACA-style residual pivots** (fallback) — extracted from already fetched
   columns/rows, used only when all principal residuals fall below the tolerance.

Three variants are provided:

| Variant       | Description |
|:--------------|:------------|
| `alpaca`      | Decomposition-finalized factors (eigenvalues below `tol` are truncated) |
| `lpaca`       | Raw factors from the pivot loop (no eigen/SVD amendment) |
| `qrdalpaca`   | `alpaca` followed by column-pivoted QR refinement to recover missed columns |

## Supported Matrix Classes

- **Real symmetric** (`symmetry=:symmetric`, `T<:Real`): `A ≈ L Lᵀ` (with `neg_indices` tracking sign flips for indefinite matrices)
- **Complex Hermitian** (`symmetry=:hermitian`): `A ≈ L L†` (with `neg_indices` tracking sign flips)
- **Complex symmetric** (`symmetry=:symmetric`, `T<:Complex`): `A ≈ L Lᵀ`
- **General** (`symmetry=:general`): `A ≈ L R†` (rectangular matrices supported)

For indefinite symmetric/Hermitian matrices, ALPACA uses **2×2 Bunch-Kaufman pivoting**
to handle cases where the diagonal is small but off-diagonal elements are significant.

## Quick Start

```julia
using ALPACADecomposition
using LinearAlgebra

# Dense matrix — symmetry is auto-detected
A = Symmetric(randn(100, 100))
result = alpaca(A; tol=1e-10)

# Decomposition extraction
U, S, Vt = alpaca_svd(A; tol=1e-10)       # SVD
vals, vecs = alpaca_eigen(A; tol=1e-10)    # Eigendecomposition
Q, R = alpaca_qr(A; tol=1e-10)            # QR

# Takagi decomposition (complex symmetric only)
A_cs = let M = randn(ComplexF64, 50, 50); M + transpose(M); end
U_t, D_t = alpaca_takagi(A_cs; tol=1e-10)

# QR-refined variant
result = qrdalpaca(A; tol=1e-10)

# Un-amended (raw factors) variant
result = lpaca(A; tol=1e-10)
```

## Matrix-free Interface

For large or implicit matrices, implement the matrix-free interface to avoid
materializing the full matrix:

```julia
struct MyMatrix <: AbstractALPACAMatrix{Float64}
  # ...
end

Base.size(o::MyMatrix) = (m, n)
ALPACADecomposition.column!(buf, o::MyMatrix, j) = # fill buf with column j
ALPACADecomposition.row!(buf, o::MyMatrix, i)    = # fill buf with row i
ALPACADecomposition.elements!(buf, o::MyMatrix, pairs) = # fill buf[k] with M[pairs[k]...]

result = alpaca(MyMatrix(...); options=ALPACAOptions(tol=1e-8, symmetry=:symmetric))
```

The interface guarantees:
- `elements!` is called **exactly once** at initialization (for `PrincipalPairs`).
- Every column and row is fetched **at most once** during the pivot selection phase.
  The QR refinement variant (`qrdalpaca`) may re-fetch columns for residual norm computation.

## Principal Descriptors

Control the secondary pivot signal:

```julia
# Default: diagonal elements (auto-generated)
result = alpaca(A; tol=1e-8)

# Custom index pairs (values fetched via elements!)
result = alpaca(A; tol=1e-8, principal=[(1,2), (3,4)])

# Pre-computed triples (no elements! call needed)
result = alpaca(A; tol=1e-8, principal=[(1,1,5.0), (2,2,3.0)])
```

## Options

```julia
ALPACAOptions(;
  tol,                    # convergence tolerance (required)
  pivotol = NaN,          # pivot acceptance threshold (NaN = auto-scale to tol/√m)
  sigma = 0.01,           # QR refinement batch-screening ratio
  qr = false,             # enable QR refinement
  symmetry = :symmetric,  # :symmetric, :hermitian, or :general
  max_rank = typemax(Int)  # upper bound on approximation rank
)
```

## Result Type

```julia
struct ALPACAResult{T}
  left::Matrix{T}           # left factor (n × r)
  right::Matrix{T}          # right factor (equals left for symmetric/hermitian)
  neg_indices::Vector{Int}   # column indices where the sign diagonal entry is −1
                             # (only for symmetric/hermitian; empty otherwise)
  pivot_indices::Vector{Int} # accepted column pivot indices
  row_pivots::Vector{Int}    # accepted row pivot indices (general only)
  symmetry::Symbol           # matrix class tag
end
```

For symmetric/Hermitian results with `neg_indices`, the reconstruction is:
```julia
L = result.left
D = ones(size(L, 2))
D[result.neg_indices] .= -1
A_approx = L * Diagonal(D) * L'
```

## Source Layout

```
src/
├── ALPACADecomposition.jl  # Module definition and exports
├── access.jl               # Matrix-free interface (AbstractALPACAMatrix, DenseALPACAMatrix)
├── descriptors.jl          # Principal descriptor types and normalization
├── results.jl              # ALPACAOptions, ALPACAResult
├── cache.jl                # ALPACACache with zero-allocation inner loop
├── kernels.jl              # Fetch and deflation kernels
├── pivots.jl               # Main pivot selection loop (ACA + principal)
├── alpaca.jl               # Public API (alpaca, lpaca, convenience wrappers)
├── qrdalpaca.jl            # QR-refined variant (qrdalpaca)
├── decompositions.jl       # Post-hoc decomposition extraction (SVD, Eigen, Takagi, QR)
└── llama.jl                # LLAMA: column-space basis for general matrices via d_row guidance
```

## LLAMA: Column-Space Basis for General Matrices

**LLAMA** (**L**eft **L**owrank **A**mended **M**atrix **A**pproximation) computes an
orthonormal basis `Q` for the column space of a general `m×n` matrix using
`diag(AA')` — the squared ℓ₂ norms of the rows (mnemonic: **LL**AMA ↔ ℓ₂) —
as a precomputed guidance vector for row pivot selection.

### Motivation

When working with general (non-symmetric) matrices, one often needs to compute
the column-space basis `Q` using minimal row and column accesses.  ALPACA handles
general matrices via ACA-style row/column cycling, but does not use external
guidance for row selection.  LLAMA extends ALPACA's cross-coupled Schur complement
deflation with `diag(AA')`-guided row pivoting, which is particularly useful when
`diag(AA')` is available cheaply (e.g., from Cholesky-decomposed integrals in
quantum chemistry applications).

### Algorithm

LLAMA uses ALPACA's cross-coupled Schur complement deflation with a
modified pivot selection strategy and an iterative SVD correction:

**Inner loop — Gram-guided ACA:**

1. **Row selection**: Pick the row with the largest residual norm indicator
   (initialized from `diag(AA')`, deflated each iteration via the Gram formula).
2. **Row deflation**: Fetch the row and deflate using stored columns:
   `r̃ = A[i*,:] − R · (D · C[i*,:])`
3. **Column selection**: Pick the column with the largest entry in `r̃`.
4. **Column deflation**: Fetch and deflate:
   `c̃ = A[:,j*] − C · (D · R[j*,:])`
5. **Store**: Scale by pivot value, update Gram-corrected residuals.

**Finalization — Cholesky + SVD:**

6. Cholesky of rank×rank Gram matrices (column Gram computed at finalization;
   row Gram accumulated incrementally during the inner loop — avoids the
   O(n·r²) recomputation), then truncated SVD of the core matrix →
   orthonormal `Q` and approximate singular values `σ`.

**Iterative correction — accessed-row Gram estimate:**

7. Compute `P = Q[pivots,:]ᴴ Q[pivots,:]` and the corrected residual
   `d_row[i] − qᵢᴴ (PΣ²P) qᵢ`.  If any corrected residual exceeds `tol²`,
   re-enter the inner loop.  Since `P ⪯ I`, the correction is less
   aggressive than the Gram formula, correctly revealing rows where the
   ACA approximation overshot.

For block-structured matrices, the algorithm skips exhausted rows (whose deflated
content is below tolerance) rather than terminating, allowing discovery of all
independent blocks.  The `PΣ²P` correction catches components missed by the
Gram overshoot.

### Column-Guided Mode (`d_col`)

When `n > m`, the inner loop is dominated by the n-dependent row Gram accumulation
(O(nr²)).  By passing `d_col = diag(A'A)` instead of `d_row`, LLAMA internally
works on the transposed matrix, effectively swapping the roles of m and n.
This reduces the dominant cost from O((m+2n)r²) to O((2m+n)r²), approaching
a 2× speedup as n/m → ∞.

The returned `Q` still spans the column space of the original matrix A —
the transposition is handled transparently.

For dense matrices, if neither `d_row` nor `d_col` is passed, the algorithm
automatically uses column-guided mode when `n > m` and row-guided mode
otherwise.

### Quick Start

```julia
using ALPACADecomposition

A = randn(100, 200)  # or any m×n matrix
result = llama(A; tol=1e-10)
Q = result.Q           # orthonormal column-space basis
S = result.singular_values  # approximate singular values

# Full SVD: A ≈ Q * diag(S) * V'
result = llama(A; tol=1e-10, fullsvd=true)
V = result.V           # right singular vectors (n × r)

# Column-guided: faster when n >> m
# Uses d_col = diag(A'A) = squared column norms instead of d_row
d_col = vec(sum(abs2, A, dims=1))
result = llama(A; d_col, tol=1e-10)
Q = result.Q           # same quality column-space basis, ~1.5× faster for n >> m

# Matrix-free interface
mat = DenseALPACAMatrix(A)
d_row = vec(sum(abs2, A, dims=2))
result = llama(mat; d_row, tol=1e-10)

# SVD extraction (fullsvd=true by default)
U, S, Vt = llama_svd(A; tol=1e-10)
```

### Result Type

```julia
struct LLAMAResult{T, R}
  Q::Matrix{T}                      # orthonormal column-space basis (m × r)
  singular_values::Vector{R}        # approximate singular values
  col_pivots::Vector{Int}           # column indices accessed
  row_pivots::Vector{Int}           # row indices used as successful pivots
  V::Union{Nothing, Matrix{T}}      # right singular vectors (n × r), or nothing
end
```

### Complexity

For an `m×n` matrix of rank `r`:
- **Accesses**: `r` columns + up to `m` rows (successful pivots + exhausted-row overhead)
- **Memory**: O(mr + nr) for stored factors, O(r²) for row Gram and Cholesky/SVD workspace
- **Arithmetic (row-guided, `d_row`)**: O((m+2n)r²) for ACA loop (column/row
  deflation + row Gram entries via BLAS gemv), O(mr² + r³) for finalization
- **Arithmetic (column-guided, `d_col`)**: O((2m+n)r²) for ACA loop (internally
  works on the transposed matrix), O(nr² + r²m) for finalization.
  Faster when `n > m`; approaches 2× speedup as `n/m → ∞`

## Documentation

Full documentation including theory, tutorial, and API reference is available
in the `docs/` directory.  Build it locally with:

```bash
cd docs && julia --project=. make.jl
```

The documentation covers:
- **Theory**: mathematical foundations, pivot selection, decomposition finalization, QR refinement
- **Tutorial**: step-by-step examples from basic usage to custom matrix-free interfaces
- **API Reference**: complete docstrings for all exported functions and types
