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

### `alpaca` — Standard (Decomposition-Finalized)

The default. Produces the cleanest factors by QR-compressed
eigendecomposition (or Takagi factorization for complex symmetric)
of the pivot-loop factors:

```julia
result = alpaca(A; tol=1e-8)
```

### `lpaca` — Raw Factors

Skips the decomposition finalization step. Returns the raw decomposition
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
decomposition finalization filters them:

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

### Smooth Pivot Scaling (`smooth_tol`)

Pivots whose magnitude sits right at the acceptance threshold are
*borderline*: whether they are kept can flip with the BLAS
implementation or last-bit rounding, making the rank (and the whole
decomposition) platform-dependent.  By default (`smooth_tol = 0.5`)
ALPACA smoothly attenuates these borderline pivots — those with
magnitude between `pivotol * smooth_tol` and `pivotol` — via a Hermite
smoothstep before finalization, so the retained energy varies
continuously and the result is reproducible across platforms.  See
[Smooth Pivot Scaling](@ref) in the theory guide for the formula.

```julia
result = alpaca(A; tol=1e-8, smooth_tol=0.5)  # default
result = alpaca(A; tol=1e-8, smooth_tol=0.0)  # hard cut-off (no smoothing)
```

Smoothing is automatically disabled when an explicit `max_rank` is
requested, since a fixed-rank truncation already pins the pivot set.

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

## 9. LLAMA: Column-Space Basis for General Matrices

**LLAMA** (**L**eft **L**owrank **A**mended **M**atrix **A**pproximation) is a
dedicated algorithm for computing an orthonormal column-space basis
``\mathbf{Q}`` for general (non-symmetric) matrices.  It uses the
squared ``\ell_2`` row norms ``\mathbf{d}_\text{row} = \text{diag}(\mathbf{AA}^H)``
(mnemonic: **LL**AMA ↔ ``\ell_2``) as an external guidance signal for
row selection, replacing ALPACA's diagonal-based principal element signal.

### When to use LLAMA instead of ALPACA

Use LLAMA when:

1. **You have ``\text{diag}(\mathbf{AA}^H)`` available cheaply** —
   for example, from Cholesky-decomposed integrals in quantum chemistry.
2. **The matrix is block-diagonal or has localized singular vectors** —
   LLAMA's row-norm guidance discovers all blocks, while ALPACA/ACA
   can get stuck cycling within the first block found.
3. **You need an orthonormal ``\mathbf{Q}`` directly** — LLAMA's SVD
   finalization produces ``\mathbf{Q}`` automatically, whereas ALPACA
   returns raw factors that require post-processing.
4. **The matrix is rectangular with ``n \gg m``** — column-guided mode
   (`d_col`) provides up to 2× speedup.

Use ALPACA instead when:

- The matrix is **symmetric or Hermitian** — ALPACA's symmetric path
  fetches only columns (2× fewer accesses than LLAMA).
- You need **eigendecomposition** or **Takagi decomposition** —
  these are only available from ALPACA results.
- You **don't have ``\mathbf{d}_\text{row}``** and computing it would
  be expensive for your matrix-free interface.

### Basic Usage

```julia
using ALPACADecomposition
using LinearAlgebra

# Build a rank-4 rectangular matrix
m, n, r = 100, 200, 4
U0 = Matrix(qr(randn(m, r)).Q)
V0 = Matrix(qr(randn(n, r)).Q)
A = U0 * Diagonal([10.0, 5.0, 2.0, 1.0]) * V0'

# Compute d_row = diag(AA')
d_row = vec(sum(abs2, A, dims=2))

# Run LLAMA
result = llama(DenseALPACAMatrix(A); d_row, tol=1e-10)

Q = result.Q                  # orthonormal column-space basis (m × r)
S = result.singular_values    # approximate singular values
println("Rank found: ", size(Q, 2))   # → 4

# Check: projector onto column space matches
P_approx = Q * Q'
P_exact = U0 * U0'
@assert norm(P_approx - P_exact) < 1e-6
```

### Dense Convenience (auto-computed norms)

For dense matrices, LLAMA can auto-compute the row/column norms:

```julia
# No need to pass d_row — computed automatically
result = llama(A; tol=1e-10)
Q = result.Q
```

When ``n > m``, the convenience wrapper automatically switches to
column-guided mode (using ``\mathbf{d}_\text{col}`` internally) for
better performance.

### Full SVD Extraction

Request right singular vectors with `fullsvd=true`:

```julia
result = llama(A; tol=1e-10, fullsvd=true)
Q = result.Q          # left singular vectors  (m × r)
S = result.singular_values
V = result.V          # right singular vectors (n × r)

# Verify: A ≈ Q * diag(S) * V'
@assert norm(A - Q * Diagonal(S) * V') / norm(A) < 1e-7
```

Or use the `llama_svd` convenience function that returns
`(U, S, Vt)` directly:

```julia
U, S, Vt = llama_svd(A; tol=1e-10)
# A ≈ U * Diagonal(S) * Vt
@assert norm(A - U * Diagonal(S) * Vt) / norm(A) < 1e-7
```

### Column-Guided Mode

When the matrix has many more columns than rows (``n \gg m``),
pass ``\mathbf{d}_\text{col} = \text{diag}(\mathbf{A}^H\mathbf{A})``
instead for faster execution:

```julia
d_col = vec(sum(abs2, A, dims=1))
result = llama(A; d_col, tol=1e-10)
Q = result.Q  # same quality, ~1.5–2× faster when n >> m
```

Under the hood, LLAMA transposes the problem and works on
``\mathbf{A}^T``, reducing the dominant inner-loop cost from
``O((m+2n)r^2)`` to ``O((2m+n)r^2)``.  The returned ``\mathbf{Q}``
still spans the column space of the *original* ``\mathbf{A}``.

!!! warning "`d_row` and `d_col` are mutually exclusive"
    You cannot pass both.  Choose based on which is cheaper to compute
    or which dimension is smaller.

### Symmetric and Hermitian Convenience

Although LLAMA is designed for general matrices, convenience
methods for `Symmetric` and `Hermitian` wrappers are provided.
These auto-compute `d_row` from column norms (which equal row
norms by symmetry):

```julia
A_sym = Symmetric(V * V')
result = llama(A_sym; tol=1e-10)

A_herm = Hermitian(randn(ComplexF64, 50, 5) * randn(ComplexF64, 5, 50))
result = llama(A_herm; tol=1e-10)
```

!!! tip "For symmetric matrices, prefer ALPACA"
    ALPACA's symmetric path is 2× faster because it fetches only
    columns.  Use LLAMA for symmetric matrices only when you specifically
    need its row-norm-guided behavior or the iterative
    ``P\Sigma^2 P`` correction.

### Matrix-Free Interface

LLAMA uses the same `AbstractALPACAMatrix` interface as ALPACA.
Implement `column!` and `row!` for your custom matrix type:

```julia
struct MyFactoredMatrix{T} <: AbstractALPACAMatrix{T}
  U::Matrix{T}
  s::Vector{T}
  V::Matrix{T}
end

Base.size(A::MyFactoredMatrix) = (size(A.U, 1), size(A.V, 1))

function ALPACADecomposition.column!(buf, A::MyFactoredMatrix, j)
  mul!(buf, A.U, A.s .* A.V[j, :])
  return buf
end

function ALPACADecomposition.row!(buf, A::MyFactoredMatrix, i)
  mul!(buf, A.V, A.s .* A.U[i, :])
  return buf
end

# Precompute d_row externally (e.g. from Cholesky factors)
d_row = vec(sum(abs2, U0 * Diagonal(S0) * V0', dims=2))
mat = MyFactoredMatrix(U0, diag(S0), V0)
result = llama(mat; d_row, tol=1e-10)
```

!!! note "No `elements!` needed"
    Unlike ALPACA, LLAMA does **not** call `elements!` — it has no
    principal element descriptor.  You only need `column!`, `row!`,
    and `Base.size`.

### Controlling Options

| Parameter | Default | Effect |
|---|---|---|
| `tol` | *(required)* | Convergence threshold: inner loop stops when all residuals < `tol²`; SVD truncates at `tol` |
| `pivotol` | `NaN` (auto) | Pivot acceptance threshold. `NaN` → adaptive: `tol / √m_eff` (see below) |
| `max_rank` | `typemax(Int)` | Upper bound on discovered rank |
| `smooth_tol` | `0.5` | Smooth-attenuation floor for borderline pivots (reproducibility across platforms); disabled when `max_rank` is set |
| `fullsvd` | `false` | If `true`, compute and return right singular vectors `V` |
| `d_row` | `nothing` | Squared ℓ₂ row norms; drives row pivot selection |
| `d_col` | `nothing` | Squared ℓ₂ column norms; triggers column-guided mode |

**Adaptive pivot tolerance.** When `pivotol = NaN` (the default),
LLAMA computes the **effective dimensionality** from the row norms:

```math
m_{\text{eff}} = \frac{\sum_i \sqrt{d_i}}{\sqrt{\max_i d_i}}
  = \frac{\sum_i \|\mathbf{A}_{i,:}\|}{\max_i \|\mathbf{A}_{i,:}\|}
```

and sets `pivotol = tol / √m_eff`.  For matrices with
block-localized singular vectors, ``m_\text{eff} \ll m``, yielding
a larger (less aggressive) pivot tolerance that avoids unnecessary
work on negligible rows while still detecting all significant
singular values.

### Result Type

```julia
struct LLAMAResult{T, R}
  Q::Matrix{T}                      # orthonormal column-space basis (m × r)
  singular_values::Vector{R}        # approximate singular values
  col_pivots::Vector{Int}           # column indices accessed from A
  row_pivots::Vector{Int}           # row indices used as successful pivots
  V::Union{Nothing, Matrix{T}}      # right singular vectors (n × r), or nothing
end
```

### Example: Discovering Block Structure

This example shows LLAMA's key advantage on a block-diagonal matrix
where the diagonal is zero — making ALPACA's principal elements
uninformative:

```julia
using ALPACADecomposition
using LinearAlgebra

# Two disjoint blocks placed off-diagonal
n = 100
r1, r2 = 3, 3
B1 = randn(n÷2, r1)
B2 = 0.1 * randn(n÷2, r2)

A = zeros(n, n)
A[1:n÷2, n÷2+1:n] = B1 * B2'      # upper-right block
A[n÷2+1:n, 1:n÷2] = B2 * B1'      # lower-left block
# Note: diag(A) is all zeros!

# ALPACA with default diagonal principal elements → may miss a block
result_alpaca = alpaca(A; tol=1e-8, symmetry=:general)
println("ALPACA pivots: ", length(result_alpaca.pivot_indices))

# LLAMA with d_row guidance → finds all blocks
d_row = vec(sum(abs2, A, dims=2))
result_llama = llama(A; d_row, tol=1e-8)
println("LLAMA rank: ", size(result_llama.Q, 2))   # finds the full rank
```

## 10. Block Diagonal Matrices — When QRdALPACA Helps

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

## 11. Hermitian Complex Matrices

Complex Hermitian matrices (``A = A^\dagger``) are handled seamlessly:

```julia
n = 50
V = randn(ComplexF64, n, 4)
A_herm = V * V'   # Hermitian PSD
@assert ishermitian(A_herm)

U, S, Vt = alpaca_svd(A_herm; tol=1e-8)
@assert norm(A_herm - U * Diagonal(S) * Vt) / norm(A_herm) < 1e-7
```

## 12. Tips and Best Practices

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

7. **Use `llama` for general matrices with available row norms.**
   When ``\text{diag}(\mathbf{AA}^H)`` is cheap (e.g. quantum chemistry
   integrals), LLAMA's row-norm guidance gives better block-structure
   discovery than ALPACA's general mode.

8. **Use `llama` with `d_col` when ``n \gg m``.**
   Column-guided mode provides up to 2× speedup for wide matrices.
