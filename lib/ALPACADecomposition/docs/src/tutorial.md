# Tutorial

This tutorial walks through progressively more advanced uses of
ALPACADecomposition, from basic symmetric decomposition to
custom matrix-free interfaces and performance tuning.

## Getting Started

```julia
using ALPACADecomposition
using LinearAlgebra
```

## 1. Basic Symmetric Decomposition

Create a low-rank symmetric positive-definite matrix and decompose it:

```julia
# Build a rank-5 matrix of size 100×100
n = 100
V = randn(n, 5)
A = V * V'   # symmetric PSD, rank 5

result = alpaca(A; tol=1e-8)
```

The returned [`ALPACAResult`](@ref) contains the decomposition factors.
Check that the reconstruction is close to the original:

```julia
L = result.left
# For symmetric: A ≈ L * L'  (with sign flips for indefinite matrices)
if !isempty(result.neg_indices)
  L_signed = copy(L)
  L_signed[:, result.neg_indices] .*= -1
  reconstruction = L * L_signed'
else
  reconstruction = L * L'
end

@assert norm(A - reconstruction) / norm(A) < 1e-7
println("Rank found: ", length(result.pivot_indices))  # → 5
```

## 2. Extracting Standard Decompositions

Rather than working with the raw factors, you can extract familiar
decompositions directly.

### SVD

```julia
U, S, Vt = alpaca_svd(A; tol=1e-8)
# A ≈ U * Diagonal(S) * Vt
@assert norm(A - U * Diagonal(S) * Vt) / norm(A) < 1e-7
```

### Eigendecomposition

```julia
vals, vecs = alpaca_eigen(A; tol=1e-8)
# A * vecs ≈ vecs * Diagonal(vals)
@assert norm(A * vecs - vecs * Diagonal(vals)) / norm(A) < 1e-7
```

### QR Decomposition

```julia
Q, R = alpaca_qr(A; tol=1e-8)
# A ≈ Q * R
@assert norm(A - Q * R) / norm(A) < 1e-7
```

### Takagi Decomposition (complex symmetric)

For complex symmetric matrices (``A = A^\top``, ``A \neq A^\dagger``):

```julia
B = randn(ComplexF64, 50, 5)
A_csym = B * transpose(B)   # complex symmetric, not Hermitian

U_takagi, D_takagi = alpaca_takagi(A_csym; tol=1e-8)
# A ≈ U * diag(D) * Uᵀ
@assert norm(A_csym - U_takagi * Diagonal(D_takagi) * transpose(U_takagi)) / norm(A_csym) < 1e-7
```

## 3. Algorithm Variants

### `alpaca` — Standard (Nyström-Amended)

The default. Produces the cleanest factors by eigendecomposing (or
Takagi-decomposing for complex symmetric) the pivot submatrix:

```julia
result = alpaca(A; tol=1e-8)
```

### `lpaca` — Raw Factors

Skips the Nyström amendment step. Returns the raw decomposition
from the pivot loop. Faster, but the factors may contain small
spurious components:

```julia
result = lpaca(A; tol=1e-8)
# A ≈ L * L'  (symmetric) or L * R† (general)
```

### `qrdalpaca` — QRdALPACA

Adds a post-processing QR refinement step that discovers pivots
ALPACA's greedy search may have missed:

```julia
result = qrdalpaca(A; tol=1e-8)
```

This is recommended when the matrix has well-separated
groups of important columns.  All extraction wrappers exist for
every variant:

```julia
# qrdalpaca_svd, qrdalpaca_eigen, qrdalpaca_qr, qrdalpaca_takagi
# lpaca_svd, lpaca_eigen, lpaca_qr, lpaca_takagi
U, S, Vt = qrdalpaca_svd(A; tol=1e-8)
```

## 4. Matrix Symmetry Classes

ALPACA auto-detects the symmetry class from the matrix via `ishermitian`
and `issymmetric`, but you can override it:

```julia
# Force :general treatment (separate left/right factors)
result = alpaca(A; tol=1e-8, symmetry=:general)
# A ≈ result.left * result.right'
```

The four supported classes are:

| Class | Julia condition | Factorization |
|---|---|---|
| `:symmetric` (real) | `issymmetric(A)` | ``A \approx L L^\top`` |
| `:symmetric` (complex) | `issymmetric(A)` and `!ishermitian(A)` | ``A \approx L L^\top`` |
| `:hermitian` | `ishermitian(A)` (complex) | ``A \approx L L^\dagger`` |
| `:general` | neither | ``A \approx L R^\dagger`` |

!!! tip "Symmetric is much faster than general"
    The symmetric/Hermitian code path is significantly faster than the general
    path because it fetches only *columns* from the matrix (one per pivot),
    whereas the general path fetches both a column and a row per pivot — doubling
    the number of element access calls and the cache memory.

    If your matrix is symmetric or Hermitian, **make sure ALPACA knows about it**:

    - **Dense matrices**: wrap with `Symmetric(A)` or `Hermitian(A)` so that
      `issymmetric` / `ishermitian` returns `true`.
    - **Custom matrix types**: implement `Base.issymmetric` or `LinearAlgebra.ishermitian`
      for your `AbstractALPACAMatrix` subtype, or pass `symmetry=:symmetric`
      (or `:hermitian`) explicitly.
    - **Keyword override**: use `symmetry=:symmetric` or `symmetry=:hermitian`
      in the function call to bypass auto-detection.

    ```julia
    # Dense: wrap to enable fast path
    result = alpaca(Symmetric(A); tol=1e-8)

    # Matrix-free: set symmetry explicitly
    opts = ALPACAOptions(tol=1e-8, symmetry=:symmetric)
    result = alpaca(my_matrix; options=opts)

    # Keyword override
    result = alpaca(A; tol=1e-8, symmetry=:symmetric)
    ```

## 5. Controlling Options

### Tolerance (`tol`)

The convergence threshold for the approximation.  Smaller values
give more accurate results but discover more pivots:

```julia
# Coarse approximation (fast)
result = alpaca(A; tol=1e-4)

# Tight approximation (more pivots found)
result = alpaca(A; tol=1e-12)
```

### Pivot Tolerance (`pivotol`)

Controls when pivots are accepted.  By default equals `tol`.
Setting it smaller than `tol` finds more pivots before the
Nyström finalization filters them:

```julia
result = alpaca(A; tol=1e-6, pivotol=1e-8)
```

### Screening Ratio (`sigma`)

Used by `qrdalpaca` to select batches of candidate columns
for QR refinement.  Smaller values select fewer candidates per
batch (more iterations, but cheaper per batch):

```julia
result = qrdalpaca(A; tol=1e-8, sigma=0.1)   # more aggressive
result = qrdalpaca(A; tol=1e-8, sigma=0.001)  # finer batches
```

### Max Rank (`max_rank`)

Caps the number of pivots:

```julia
result = alpaca(A; tol=1e-12, max_rank=3)
println(length(result.pivot_indices))  # ≤ 3
```

### Using `ALPACAOptions`

For repeated calls with the same settings, pre-build an options object:

```julia
opts = ALPACAOptions(tol=1e-8, symmetry=:symmetric)
result1 = alpaca(A; options=opts)
result2 = alpaca(B; options=opts)
```

## 6. Principal Element Descriptors

By default, ALPACA tracks the diagonal elements as the primary
pivot signal.  You can customize which elements are monitored.

### Custom Pairs

Specify a set of ``(i, j)`` indices to monitor:

```julia
# Only monitor a subset of diagonal elements
my_pairs = [(i, i) for i in 1:10]
result = alpaca(A; tol=1e-8, principal=my_pairs)
```

### Pre-computed Values (Triples)

If you already know the values of the principal elements (e.g. from
a previous calculation), supply triples ``(i, j, \text{value})``:

```julia
# Provide element values to avoid element access calls
my_triples = [(i, i, A[i,i]) for i in 1:10]
result = alpaca(A; tol=1e-8, principal=my_triples)
```

### Why Customize?

The principal elements provide a *global* convergence signal.  If
only some elements are informative (e.g. you know the important
subspace lies in the first 10 rows/columns), restricting the
principal set can:
1. Reduce initialization cost (fewer `elements!` calls).
2. Focus the search on the relevant part of the matrix.

However, if the principal set misses important matrix regions,
consider using `qrdalpaca` which can recover via QR refinement.

## 7. Custom Matrix-free Interface

For matrices where element access is expensive (e.g. computed
on-the-fly), implement the matrix-free interface instead of materializing:

```julia
using ALPACADecomposition

# Example: a kernel matrix K(i,j) = exp(-|xᵢ - xⱼ|²/σ²)
struct KernelMatrix <: AbstractALPACAMatrix{Float64}
  points::Matrix{Float64}   # d × n
  sigma2::Float64
end

Base.size(K::KernelMatrix) = (n = size(K.points, 2); (n, n))

function ALPACADecomposition.column!(buf, K::KernelMatrix, j)
  xj = @view K.points[:, j]
  @inbounds for i in eachindex(buf)
    xi = @view K.points[:, i]
    buf[i] = exp(-sum((xi .- xj).^2) / K.sigma2)
  end
  return buf
end

function ALPACADecomposition.row!(buf, K::KernelMatrix, i)
  xi = @view K.points[:, i]
  @inbounds for j in eachindex(buf)
    xj = @view K.points[:, j]
    buf[j] = exp(-sum((xi .- xj).^2) / K.sigma2)
  end
  return buf
end

function ALPACADecomposition.elements!(buf, K::KernelMatrix,
                                       pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}})
  @inbounds for k in eachindex(pairs)
    i, j = pairs[k]
    xi = @view K.points[:, i]
    xj = @view K.points[:, j]
    buf[k] = exp(-sum((xi .- xj).^2) / K.sigma2)
  end
  return buf
end

# Use it
points = randn(3, 200)
K = KernelMatrix(points, 1.0)
opts = ALPACAOptions(tol=1e-6, symmetry=:symmetric)
result = alpaca(K; options=opts)
println("Kernel approximation rank: ", length(result.pivot_indices))
```

The interface is called at most:
- ``k`` times for `column!` (one per pivot)
- ``k`` times for `row!` (general matrices only)
- Once for `elements!` (to initialize principal values)

## 8. General (Non-Symmetric) Matrices

For non-symmetric matrices, ALPACA finds separate left and right factors:

```julia
m, n, r = 80, 60, 4
A_gen = randn(m, r) * randn(r, n)   # rank-4 rectangular-ish

result = alpaca(A_gen; tol=1e-8, symmetry=:general)
# A ≈ result.left * result.right'
@assert norm(A_gen - result.left * result.right') / norm(A_gen) < 1e-7
```

Note: general matrices cannot use `alpaca_eigen` or `alpaca_takagi`
for eigenvalue / Takagi extraction in the usual sense, but SVD and
QR extraction work for any matrix class.

## 9. Block Diagonal Matrices — When QRdALPACA Helps

A common scenario where `qrdalpaca` shines: the matrix has multiple
blocks, and the principal descriptor only covers one of them.

```julia
n1, n2 = 10, 10
B1 = randn(n1, 3); B2 = 0.1 * randn(n2, 3)
A_block = zeros(n1 + n2, n1 + n2)
A_block[1:n1, 1:n1] = B1 * B1'          # strong block
A_block[n1+1:end, n1+1:end] = B2 * B2'  # weak block

# Principal pairs only cover block 1 → ALPACA misses block 2
bad_principal = [(i, i) for i in 1:n1]

result_plain = alpaca(A_block; tol=1e-8, principal=bad_principal)
println("alpaca found: ", length(result_plain.pivot_indices), " pivots")

result_qrd = qrdalpaca(A_block; tol=1e-8, principal=bad_principal)
println("qrdalpaca found: ", length(result_qrd.pivot_indices), " pivots")

# qrdalpaca recovers the pivots from block 2 that alpaca missed
```

## 10. Hermitian Complex Matrices

Complex Hermitian matrices (``A = A^\dagger``) are handled seamlessly:

```julia
n = 50
V = randn(ComplexF64, n, 4)
A_herm = V * V'   # Hermitian PSD
@assert ishermitian(A_herm)

U, S, Vt = alpaca_svd(A_herm; tol=1e-8)
@assert norm(A_herm - U * Diagonal(S) * Vt) / norm(A_herm) < 1e-7
```

## 11. Tips and Best Practices

1. **Start with `alpaca`**.  Only switch to `qrdalpaca` if you suspect
   missing pivots (e.g. reconstruction error is unexpectedly large).

2. **Always declare symmetry**.  The symmetric/Hermitian path is much
   faster than the general path.  Use `Symmetric(A)`, `Hermitian(A)`,
   or `symmetry=:symmetric` / `symmetry=:hermitian` whenever applicable.

3. **Set `tol` to the desired reconstruction accuracy**.  The pivot
   tolerance `pivotol` can be left at its default (equal to `tol`).

4. **For large matrices**, implement `elements!` efficiently.
   Bulk element access (one call with many pairs) is cheaper than many
   individual `column!` calls.

5. **Use `lpaca` for intermediate results** when you only need the raw
   factors and will post-process yourself.

6. **Monitor `result.neg_indices`** for indefinite matrices.
   A non-empty `neg_indices` means the matrix has negative
   eigenvalues — the reconstruction formula must include the sign
   diagonal.
