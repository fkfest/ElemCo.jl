# Theory

This section explains the mathematical foundation behind ALPACA,
building up from basic concepts to the full algorithm.

## Background: Low-Rank Matrices

Many matrices that arise in scientific computing are (approximately) low-rank:
their information content is much smaller than the total number of entries.
Formally, a matrix ``\mathbf{A} \in \mathbb{R}^{m \times n}`` has rank
``r \ll \min(m, n)`` if it can be written as

```math
\mathbf{A} = \sum_{k=1}^{r} \sigma_k \, \mathbf{u}_k \mathbf{v}_k^\top
```

where ``\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0`` are the
singular values.  In practice we want a *numerical* low-rank approximation:
we truncate to rank ``k`` keeping only components with ``\sigma_i > \tau``
for some tolerance ``\tau``.

Computing the full SVD costs ``O(mn \min(m,n))`` — prohibitive for large
matrices.  ALPACA constructs an equally good approximation by accessing
only ``O((m+n)k)`` matrix entries.

## Adaptive Cross Approximation (ACA)

The classical ACA algorithm builds a rank-``k`` approximation by
selecting one row and one column per iteration.

**Idea:** Start from an initial row ``i_1``.
Find the column ``j_1`` where the residual is largest:
```math
j_1 = \arg\max_j \lvert \mathbf{A}_{i_1, j} \rvert.
```
Now column ``j_1`` tells us which row has the largest residual, giving
``i_2``, and so on.  Each step adds a rank-1 term:

```math
\mathbf{A} \approx \sum_{k=1}^{r} \frac{1}{d_k} \, \mathbf{c}_k \, \mathbf{r}_k^\top
```

where ``\mathbf{c}_k = \mathbf{A}_{:, j_k}`` (deflated column),
``\mathbf{r}_k = \mathbf{A}_{i_k, :}`` (deflated row), and
``d_k = \mathbf{A}_{i_k, j_k}`` is the pivot element.

**Limitation:** ACA's greedy residual-based pivoting can miss important
columns if the largest residual happens to lie in a subspace that is
already well-represented.  It has no global view of the matrix.

## The Principal Element Signal

ALPACA's key innovation is using **principal elements** — user-specified
matrix entries whose residuals are tracked throughout the decomposition —
as the **primary** pivot selection signal.

### Default: Diagonal Elements

For a symmetric matrix, the diagonal ``\{A_{ii}\}`` is the default
and most natural choice of principal elements.  As pivots are accepted,
the deflated diagonal residual of index ``i`` is

```math
p_i^{(k)} = A_{ii} - \sum_{\ell=1}^{k} \frac{c_\ell(i)^2}{d_\ell}
```

where ``c_\ell(i)`` is the ``i``-th entry of the ``\ell``-th deflated
column.  The magnitude ``\lvert p_i^{(k)} \rvert`` estimates
the importance of adding index ``i`` as a pivot.

The principal element with the largest residual is chosen as the next
pivot.  **ACA serves only as a fallback**: ALPACA switches to the ACA
candidate only when all principal residuals have fallen below the
tolerance.

### General Update Formula

The principal elements are not restricted to diagonal entries — any set
of ``(a, b)`` index pairs can be tracked.  The residual update for
a principal pair ``(a, b)`` after pivot ``k`` is:

- **Symmetric** (real or complex):
  ```math
  p_{ab}^{(k)} = p_{ab}^{(k-1)} - \frac{c_k(a) \cdot c_k(b)}{d_k}
  ```
- **Hermitian**: uses complex conjugation on the second index:
  ```math
  p_{ab}^{(k)} = p_{ab}^{(k-1)} - \frac{c_k(a) \cdot \overline{c_k(b)}}{d_k}
  ```
  For diagonal elements (``a = b``), this reduces to
  ``p_{aa}^{(k)} = p_{aa}^{(k-1)} - |c_k(a)|^2 / d_k``,
  which is always real (since ``A_{aa}`` is real for Hermitian matrices).
- **General**: uses the row factor instead of the column factor:
  ```math
  p_{ab}^{(k)} = p_{ab}^{(k-1)} - \frac{c_k(a) \cdot r_k(b)}{d_k}
  ```
  where ``r_k(b)`` is the ``b``-th entry of the ``k``-th deflated row.

For all symmetry classes, diagonal elements remain the default choice
because they provide the most informative convergence signal: once all
diagonal residuals are small, the matrix is well-approximated.
Custom off-diagonal pairs can be specified for matrices where other
entries are more informative (see the [Tutorial](@ref) for examples).

### Convergence

The principal signal is checked **first** at each iteration.  The ACA 
fallback is computed **only** when the principal signal is insufficient:

1. If ``\max_i \lvert p_i^{(k)} \rvert \ge \texttt{pivotol}``,
   the best principal element is selected as the next pivot.  No ACA
   computation is performed.
2. Otherwise, the ACA candidate is evaluated:
   ``\max_{j \notin \text{pivots}} \lvert L_{j,k} \rvert \cdot \lvert d_k \rvert``.
   If this exceeds `pivotol`, the ACA candidate becomes the pivot.
3. If *both* signals are below `pivotol`, the loop terminates.

This principal-first, lazy-ACA strategy avoids unnecessary work when
the principal signal is informative (the common case), while still
using ACA as a safety net to catch pivots that the principal
descriptor may have missed.

## The Pivot Loop

At its core, ALPACA performs an incremental rank-revealing factorization
by selecting one pivot per iteration.

### Symmetric / Hermitian Matrices

For a symmetric matrix ``\mathbf{A}``, ALPACA constructs a low-rank
factorization:

```math
\mathbf{A} \approx \mathbf{L} \mathbf{L}^\top
```

where ``\mathbf{L} \in \mathbb{R}^{n \times k}`` contains the deflated
columns (normalized by the pivot values).  For indefinite matrices,
some columns of ``\mathbf{L}`` absorb a sign flip, tracked via
`neg_indices`.  Each iteration proceeds as follows:

1. **Select pivot** ``j``: Use the principal element with the largest
   residual.  If all principal residuals are below the tolerance, fall
   back to the ACA candidate (largest entry in the most recent deflated
   column).
2. **Fetch column** ``\mathbf{A}_{:,j}``: request the column from the matrix.
3. **Deflate**: remove the contribution of all previous pivots:
   ```math
   \tilde{\mathbf{c}} = \mathbf{A}_{:,j} - \mathbf{L}_{1:k-1} \, \mathbf{L}_{j, 1:k-1}^\top
   ```
   This is a BLAS-2 operation (GEMV) on the cached factors.
4. **1×1 vs 2×2 pivot decision**: See [below](@ref bk_pivot) for details.
5. **Record pivot**: store ``d_k = \tilde{c}_j`` (the pivot element) and
   ``\mathbf{L}_{:,k} = \tilde{\mathbf{c}} / d_k`` (the scaled column).
   If ``d_k < 0``, record ``k`` as a negative index.
6. **Update principal residuals**: subtract the new pivot's contribution
   from all tracked principal elements.

The raw factorization maintained during the loop is
```math
\mathbf{A} \approx \sum_{k=1}^{r} d_k \, \mathbf{L}_{:,k} \, \mathbf{L}_{:,k}^\top
```
where the ``d_k`` are stored separately.  The Nyström finalization
(see [below](@ref nystrom)) converts these raw factors into the
cleaner ``\mathbf{L}\mathbf{L}^\top`` form.

Because ``\mathbf{A} = \mathbf{A}^\top``, only columns are fetched —
the rows are implicitly available from the symmetry.

### General Matrices

For a general (non-symmetric) matrix ``\mathbf{A} \in \mathbb{R}^{m \times n}``,
the factorization takes the form:

```math
\mathbf{A} \approx \mathbf{L}_C \, \mathbf{L}_R^\top
```

where ``\mathbf{L}_C \in \mathbb{R}^{m \times k}`` stores deflated
columns and ``\mathbf{L}_R \in \mathbb{R}^{n \times k}`` stores deflated
rows.  At each step, ALPACA selects a column pivot ``j_k`` *and* a row
pivot ``i_k``, fetching both ``\mathbf{A}_{:,j_k}`` and
``\mathbf{A}_{i_k,:}`` from the matrix.  The pivot element is
``d_k = \tilde{c}_{i_k}``, and both column and row are deflated against
all previous pivots before being stored.

### [1×1 vs 2×2 Pivot Selection (Bunch-Kaufman)](@id bk_pivot)

After fetching and deflating column ``j``, ALPACA inspects the deflated
column ``\tilde{\mathbf{c}}`` to decide between a standard 1×1 pivot
and a 2×2 Bunch-Kaufman pivot.  This is important for **indefinite**
matrices where the diagonal element can be small or zero even though
the column contains significant off-diagonal entries.

Let ``d = |\tilde{c}_j|`` (diagonal) and
``g = \max_{p \neq j,\, p \notin \text{pivots}} |\tilde{c}_p|``
(largest off-diagonal).

- **If the pivot was selected by a principal diagonal element**
  (i.e., the principal pair has ``i = j``), the diagonal was already
  the dominant monitored value, so a **1×1 pivot** is accepted directly
  without scanning the column.

- **1×1 pivot** is accepted when
  ```math
  d \ge \max\!\big((1 - 5\tau)\,g,\;\tau\big)
  ```
  where ``\tau`` is the pivot tolerance (`pivotol`).  This ensures
  the diagonal is large enough relative to the off-diagonal for
  numerical stability.

- **2×2 Bunch-Kaufman pivot**: when the off-diagonal element ``g``
  at index ``p^*`` dominates, ALPACA fetches and deflates column
  ``p^*`` as well.  The 2×2 intersection block
  ```math
  \mathbf{B} = \begin{pmatrix}
    \tilde{c}_j(j) & \tilde{c}_j(p^*) \\
    \tilde{c}_{p^*}(j) & \tilde{c}_{p^*}(p^*)
  \end{pmatrix}
  ```
  is eigendecomposed (for real symmetric and complex Hermitian) or
  Takagi-decomposed (for complex symmetric).  Each eigenvalue /
  singular value ``\lambda_t`` above the tolerance produces a rotated
  rank-1 pivot:
  ```math
  \mathbf{L}_{:,k} = \frac{v_{1t}\,\tilde{\mathbf{c}}_j +
                           v_{2t}\,\tilde{\mathbf{c}}_{p^*}}{\lambda_t}
  ```
  This allows ALPACA to handle indefinite matrices where the diagonal
  is small but the 2×2 block has a large eigenvalue.

!!! tip "Performance: symmetric vs general"
    The symmetric/Hermitian code path is significantly faster than the
    general path.  Symmetric mode fetches only one column per pivot,
    whereas general mode fetches both a column and a row — doubling the
    number of element access calls and the cache memory.  If your matrix is
    symmetric or Hermitian, ensure ALPACA uses the fast path by wrapping
    dense matrices in `Symmetric()` or `Hermitian()`, implementing
    `issymmetric` / `ishermitian` for custom matrix types, or passing
    `symmetry=:symmetric` (or `:hermitian`) explicitly.

## [Nyström Finalization](@id nystrom)

The raw factors from the pivot loop (returned by [`lpaca`](@ref))
may contain small or spurious components due to finite-precision
arithmetic.  The **Nyström finalization** step in [`alpaca`](@ref) cleans
up the result.

Given the column pivots ``\{j_1, \ldots, j_k\}``, define:
- ``\mathbf{C} = \mathbf{A}_{:, \text{pivots}} \in \mathbb{R}^{m \times k}``
  — the full (undeflated) pivot columns
- ``\mathbf{J} = \mathbf{A}_{\text{pivots}, \text{pivots}} \in \mathbb{R}^{k \times k}``
  — the pivot submatrix

For **symmetric/Hermitian matrices**, we diagonalize ``\mathbf{J}``:
```math
\mathbf{J} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\dagger
```
and retain only components with ``\lvert \lambda_i \rvert > \tau``.  The
amended left factor is:
```math
\mathbf{L}_{\text{amended}} = \mathbf{C} \, \mathbf{V}_r \, \lvert \boldsymbol{\Lambda}_r \rvert^{-1/2}
```
where the subscript ``r`` denotes the retained components. The signs of
the eigenvalues are stored in `neg_indices` so that
```math
\mathbf{A} \approx \mathbf{L}_{\text{amended}} \, \tilde{\mathbf{D}} \, \mathbf{L}_{\text{amended}}^\dagger,
\qquad \tilde{\mathbf{D}} = \text{diag}(\pm 1).
```

For **general matrices**, the finalization uses the SVD of
``\mathbf{J}`` to produce left and right factors ``\mathbf{L}, \mathbf{R}``
such that ``\mathbf{A} \approx \mathbf{L}\mathbf{R}^\dagger``.

## QR Refinement (QRdALPACA)

The [`qrdalpaca`](@ref) variant adds a post-processing step that can
discover pivots that ALPACA's greedy search missed.  This is particularly
useful when:

- The "principal" descriptor is incomplete or misleading.
- The matrix has well-separated clusters of significant columns that
  ACA's local search cannot bridge.

### Reconstruction Pre-check

Before invoking the expensive QR machinery, `qrdalpaca` performs a
cheap reconstruction test on a random sample of non-pivot columns.

#### Sample size

If ALPACA missed ``d`` out of ``N`` non-pivot columns, the probability
that a random sample of ``k`` columns detects at least one is

```math
P(\text{detect} \geq 1 \mid d) = 1 - \left(1 - \frac{d}{N}\right)^k.
```

Requiring ``P \geq 1 - \alpha`` and using ``\ln(1 - x) \approx -x``
for small ``x = d/N`` gives

```math
k \geq \frac{-\ln\alpha}{d/N} = \frac{N \ln(1/\alpha)}{d}.
```

For 99.9% confidence (``\alpha = 0.001``, ``\ln(1/\alpha) \approx 6.908``):

| Missed columns ``d`` | Required ``k`` |
|---|---|
| ``\sqrt{N}`` | ``\lceil 7\sqrt{N}\rceil`` |
| ``0.01\, N`` (1%) | ``\approx 700`` |

The implementation uses ``k = \min\!\big(N,\; \max(\lceil 7\sqrt{N}\rceil, 700)\big)``
to cover both regimes with 99.9% detection probability.

#### Threshold

Each sampled column ``\mathbf{a}_j`` is reconstructed from the ALPACA
factors and the maximum absolute element deviation is compared against
the tolerance ``\tau``:

```math
\max_i \lvert \mathbf{a}_j - \mathbf{L}\,\mathbf{c}_j \rvert_i < \tau
```

where ``\mathbf{c}_j`` are the reconstruction coefficients derived from
the right factor (or from ``\mathbf{L}`` itself via symmetry).  This
element-wise check is consistent with ALPACA's pivot selection criterion.

If all sampled columns satisfy the bound, the ALPACA result is already
good — return early without QR refinement.

### Projection Residuals

If the pre-check fails, compute exact projection residuals for all
non-pivot columns:

1. Form a QR basis ``\mathbf{Q}`` from the ALPACA left factor
   ``\mathbf{L}``.
2. For each non-pivot column ``\mathbf{a}_j``:
   ```math
   r_j^2 = \lVert \mathbf{a}_j \rVert^2 - \lVert \mathbf{Q}^\dagger \mathbf{a}_j \rVert^2
   ```
   This is the squared norm of the component outside ``\text{span}(\mathbf{L})``.
3. Columns with ``r_j^2 \geq \tau^2`` have significant content missed
   by ALPACA.

### Batched QR Iteration

The refinement then proceeds in batches:

1. **Select batch**: columns with the largest projection residuals,
   above ``\sigma \cdot r_{\max}^2`` (screening ratio ``\sigma``).
2. **Project out** the current basis ``\mathbf{Q}``.
3. **Column-pivoted QR** on the projected batch to find new
   significant directions.
4. **Extend** ``\mathbf{Q}`` with the new orthogonal vectors.
5. **Update** projection residuals for remaining candidates.
6. Repeat until no residuals exceed ``\tau^2``.

### Catastrophic Cancellation Recovery

During residual updates, subtracting ``\lVert \mathbf{Q}_{\text{new}}^\dagger \mathbf{a}_j \rVert^2`` from ``r_j^2``
can cause catastrophic cancellation when the decrement is close to
``r_j^2``.  When ``r_j^2`` drops below
``\epsilon^{2/3} \cdot r_{j,\text{orig}}^2``, the residual is
recomputed from scratch to maintain numerical stability.

### Re-finalization

Once all significant columns have been found, the combined pivot set
(ALPACA pivots + QR-discovered pivots) is used to re-run the Nyström
(symmetric/Hermitian), Takagi (complex symmetric), or SVD (general)
finalization to produce the final amended factors.

## Decomposition Extraction

The [`ALPACAResult`](@ref) stores the low-rank factors in a compact form.
Several standard decompositions can be extracted from it:

### SVD

The thin SVD ``\mathbf{A} \approx \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\dagger``
is obtained by:

- **Symmetric/Hermitian**: the eigendecomposition of the sign-corrected
  Gram matrix ``\mathbf{L}^\dagger \mathbf{L}`` gives the singular
  values (as ``\lvert \lambda_i \rvert``) and right singular vectors.
- **General**: QR factorizations of ``\mathbf{L}`` and ``\mathbf{R}``
  followed by SVD of the ``k \times k`` core ``\mathbf{R}_L^\top \mathbf{R}_R``.

### Eigendecomposition

For symmetric/Hermitian matrices, eigenvalues are extracted from the
sign-corrected factorization:
```math
\lambda_i = \pm \sigma_i, \qquad \mathbf{v}_i = \mathbf{L} \mathbf{w}_i / \sigma_i
```
where ``\sigma_i, \mathbf{w}_i`` come from the reduced eigendecomposition.

For general matrices, eigenvalues are computed from
``\mathbf{R}^\dagger \mathbf{L}``.

### Takagi Decomposition

For complex symmetric matrices (``\mathbf{A} = \mathbf{A}^\top``,
``\mathbf{A} \neq \mathbf{A}^\dagger``), the Autonne–Takagi factorization:
```math
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{U}^\top
```
is computed from the SVD with a phase correction:
``\mathbf{U} = \mathbf{U}_{\text{SVD}} \cdot \text{diag}(e^{i\phi_j / 2})``
where ``\phi_j`` matches the phases of ``\mathbf{U}_{\text{SVD}}^\top \mathbf{U}_{\text{SVD}}``.

### QR Decomposition

A thin QR factorization ``\mathbf{A} \approx \mathbf{Q}\mathbf{R}`` is
extracted by QR-factorizing the left factor and propagating the
triangular part to the right factor.

## Matrix-free Interface

ALPACA communicates with the matrix through three operations:

1. **`column!(buf, M, j)`**: Fill `buf` with column ``j`` of ``\mathbf{M}``.
2. **`row!(buf, M, i)`**: Fill `buf` with row ``i`` of ``\mathbf{M}``.
3. **`elements!(buf, M, pairs)`**: Fill `buf[k]` with ``\mathbf{M}_{i_k, j_k}``
   for each pair ``(i_k, j_k)``.

### Required methods by symmetry class

Not every method is needed in every case.  The table below summarizes
which operations must be implemented for each matrix class:

| Method | `:symmetric` / `:hermitian` | `:general` |
|---|---|---|
| `Base.size` | **required** | **required** |
| `column!` | **required** | **required** |
| `row!` | not called | **required** |
| `elements!` | **required**\* | **required**\* |

\* `elements!` is called exactly once at initialization to fetch the
principal element values (e.g., the diagonal elements in the default case).
It only needs to be able to provide these principal elements, not arbitrary
matrix entries.  The only exception is when using [`PrincipalTriples`](@ref),
which carry pre-computed values and bypass `elements!` entirely.

- **Symmetric / Hermitian matrices** need `column!`, `elements!`, and `size`.
  Because ``\mathbf{A} = \mathbf{A}^\top`` (or ``\mathbf{A}^\dagger``),
  the rows are implicitly available from the columns.
- **General matrices** additionally require `row!`, since the algorithm
  fetches both a column and a row at each pivot step.

### Access guarantees

The algorithm issues at most:
- ``k`` `column!` calls (one per pivot)
- ``k`` `row!` calls (general matrices only, one per pivot)
- 1 `elements!` call at initialization (for `PrincipalPairs` descriptors)

This makes ALPACA ideal for matrices where element access is expensive —
for example integral matrices in quantum chemistry, kernel matrices in
machine learning, or matrices stored on disk.
