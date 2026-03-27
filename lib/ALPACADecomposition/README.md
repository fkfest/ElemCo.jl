# ALPACADecomposition.jl

ALPACA (**A**mended **L**ow-rank **P**rincipal-element **A**daptive **C**ross **A**pproximation) is a standalone Julia package for low-rank matrix decomposition with minimal element access.

## Overview

The algorithm combines two pivoting signals:

1. **ACA-style residual pivots** extracted from already requested rows and columns.
2. **Principal-element pivots** extracted from a user-provided descriptor, defaulting to the diagonal for square matrices.

Three variants are provided:

| Variant       | Description |
|:--------------|:------------|
| `alpaca`      | Nyström / SVD-finalized factors (eigenvalues below `tol` are truncated) |
| `lpaca`       | Raw factors from the pivot loop (no eigen/SVD amendment) |
| `qrdalpaca`   | `alpaca` followed by column-pivoted QR refinement to recover missed columns |

## Supported Matrix Classes

- **Real symmetric** (`symmetry=:symmetric`, `T<:Real`): `A ≈ L Lᵀ` (with a vector of negative indices)
- **Complex Hermitian** (`symmetry=:hermitian`): `A ≈ L L†` (with a vector of negative indices)
- **Complex symmetric** (`symmetry=:symmetric`, `T<:Complex`): `A ≈ L Lᵀ`
- **General** (`symmetry=:general`): `A ≈ L R†` (rectangular matrices supported)

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
struct MyMatrix <: AbstractALPACAMatrix
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
- Every column and row is fetched **at most once**.

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
  pivotol = tol,          # pivot acceptance threshold
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
  neg_indices::Vector{Int}   # indices with −1 in sign diagonal D
  pivot_indices::Vector{Int} # accepted column pivot indices
  row_pivots::Vector{Int}    # accepted row pivot indices (general only)
  symmetry::Symbol           # matrix class tag
end
```

## Source Layout

```
src/
├── ALPACADecomposition.jl  # Module definition and exports
├── access.jl               # Matrix-free interface (AbstractALPACAMatrix, DenseALPACAMatrix)
├── descriptors.jl          # Principal descriptor types and normalization
├── results.jl              # ALPACAOptions, ALPACAResult
├── cache.jl                # ALPACACache with zero-allocation inner loop
├── kernels.jl              # Fetch, deflation, Nyström/SVD finalization
├── pivots.jl               # Main pivot selection loop (ACA + principal)
├── alpaca.jl               # Public API (alpaca, lpaca, convenience wrappers)
├── qrdalpaca.jl            # QR-refined variant (qrdalpaca)
└── decompositions.jl       # Post-hoc decomposition extraction (SVD, Eigen, Takagi, QR)
```
