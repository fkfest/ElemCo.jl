```@meta
CurrentModule = ElemCo.ALPACADecomposition
```

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

## Pivot Tolerance Scaling

The pivot acceptance threshold (`pivotol`) determines when a candidate
pivot is large enough to be included in the decomposition.  For a matrix
``\mathbf{A} \in \mathbb{R}^{m \times n}`` with singular value ``\sigma``,
the corresponding per-element signal magnitude scales as

```math
|A_{ij}| \sim \frac{\sigma}{\sqrt{m}}
```

because the singular vectors have unit norm distributed across ``m``
entries.  Using `pivotol = tol` in this regime causes the algorithm to
miss singular values near `tol` in large matrices, since their
per-element contributions fall well below `tol`.

### Default auto-scaling: ``\text{tol}/\sqrt{m}``

By default (`pivotol = NaN`), the effective pivot tolerance is
automatically scaled:

```math
\text{pivotol} = \frac{\text{tol}}{\sqrt{m}}
```

This compensates for the ``1/\sqrt{m}`` dilution of singular-value
signal across matrix rows, ensuring that singular values above `tol`
produce element magnitudes safely above `pivotol`.

### Adaptive scaling via effective dimensionality (LLAMA)

For matrices with block-localized singular vectors, the signal is
concentrated in a subset of the ``m`` rows rather than spread uniformly.
Using ``\sqrt{m}`` in this case is overly conservative — the effective
number of rows carrying signal is smaller than ``m``.

LLAMA exploits the row-norm vector
``\mathbf{d} = \text{diag}(\mathbf{A}\mathbf{A}^\top)``
(which is always available as input) to estimate the **effective
dimensionality**:

```math
m_{\text{eff}} = \frac{\sum_i \sqrt{d_i}}{\sqrt{\max_i d_i}}
  = \frac{\sum_i \|\mathbf{A}_{i,:}\|}{\max_i \|\mathbf{A}_{i,:}\|}
```

This ratio measures how uniformly the signal amplitude (row norms)
is distributed:
- For globally distributed singular vectors: ``m_{\text{eff}} \approx m``
- For block-localized singular vectors: ``m_{\text{eff}} \ll m``

The adaptive pivot tolerance is then:

```math
\text{pivotol}_{\text{LLAMA}} = \frac{\text{tol}}{\sqrt{m_{\text{eff}}}}
```

Since ``m_{\text{eff}} \leq m``, this yields a less aggressive (larger)
pivot tolerance that better matches the actual signal concentration,
avoiding unnecessary work on rows with negligible content while still
reliably detecting all significant singular values.

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
where the ``d_k`` are stored separately.  The decomposition finalization
(see [below](@ref decomposition_finalization)) converts these raw factors into the
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

## [Decomposition Finalization](@id decomposition_finalization)

The raw factors from the pivot loop (returned by [`lpaca`](@ref))
may contain small or spurious components due to finite-precision
arithmetic.  The **decomposition finalization** step in [`alpaca`](@ref) cleans
up the result.

Given the pivot-loop factors
``\mathbf{L} = [\mathbf{L}_1, \ldots, \mathbf{L}_k]`` and pivot
diagonal values ``d_1, \ldots, d_k``, we form the full factor matrix:
```math
\hat{\mathbf{L}} = \mathbf{L} \cdot \text{diag}(\sqrt{|d_k|})
```

For **real symmetric** and **complex Hermitian matrices**, we apply
QR-compressed eigendecomposition:
1. QR decompose ``\hat{\mathbf{L}} = \mathbf{Q} \mathbf{R}``
2. Form ``\mathbf{M} = \mathbf{R} \, \mathbf{D} \, \mathbf{R}^\dagger``
   where ``\mathbf{D} = \text{diag}(\pm 1)`` from the pivot signs
3. Eigendecompose ``\mathbf{M} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\dagger``
4. Retain only components with ``|\lambda_i| > \tau``
5. Build ``\mathbf{L}_{\text{final}} = \mathbf{Q} \mathbf{V}_r \sqrt{|\boldsymbol{\Lambda}_r|}``

The signs of the retained eigenvalues are stored in `neg_indices` so that
```math
\mathbf{A} \approx \mathbf{L}_{\text{final}} \, \tilde{\mathbf{D}} \, \mathbf{L}_{\text{final}}^\dagger,
\qquad \tilde{\mathbf{D}} = \text{diag}(\pm 1).
```

For **complex symmetric matrices**, the finalization uses the SVD of
``\mathbf{R} \mathbf{R}^T`` with Autonne-Takagi phase correction.

For **general matrices**, dual QR factorizations of the left and right
factors are combined with SVD of the core matrix to produce left and
right factors ``\mathbf{L}, \mathbf{R}`` such that
``\mathbf{A} \approx \mathbf{L}\mathbf{R}^\dagger``.

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

Once all significant columns have been found, the new pivots are
incorporated into the cache (fetch + deflate against all existing
pivots) and the decomposition finalization is re-run on the extended
factor set to produce the final amended factors.

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

## LLAMA: Left Lowrank Amended Matrix Approximation

LLAMA (**L**eft **L**owrank **A**mended **M**atrix **A**pproximation) is a
specialized algorithm for computing an orthonormal basis
``\mathbf{Q} \in \mathbb{R}^{m \times r}`` (or ``\mathbb{C}^{m \times r}``)
for the column space of a general ``m \times n`` matrix ``\mathbf{A}`` of
numerical rank ``r \ll \min(m, n)`` using only ``O(r)`` row and column
accesses.  The mnemonic **LL**AMA refers to the ``\ell_2`` norms
(``\mathbf{d}_{\text{row}} = \text{diag}(\mathbf{A}\mathbf{A}^H)``)
that guide its pivot selection.

### Motivation: Why Not Just Use ALPACA for General Matrices?

ALPACA handles general (non-symmetric) matrices via ACA-style
row/column cycling with principal element guidance.  However,
there are two scenarios where this approach can struggle:

1. **Block-diagonal matrices with zero (or small) diagonal.**
   When the diagonal of ``\mathbf{A}`` is zero — for instance because
   the nonzero blocks occupy off-diagonal positions — the principal
   element signal (which defaults to diagonal entries) provides no
   guidance.  ACA's greedy column/row cycling can get trapped
   exploring a single block forever, missing the others entirely.

2. **Precomputed row-norm guidance is not too expensive to obtain.**
   The squared row norms ``d_i = \|\mathbf{A}_{i,:}\|^2`` can often
   be computed or estimated at moderate cost — for instance from
   already-available factored representations of the matrix, from
   diagonal blocks, or from a cheap preliminary pass.  When such
   a guidance vector is available, relying on diagonal elements
   alone discards useful information about the global row-norm
   distribution.

LLAMA addresses both scenarios by replacing the principal element
signal with a **row-norm residual indicator** derived from
``\mathbf{d}_{\text{row}} = \text{diag}(\mathbf{A}\mathbf{A}^H)``.
This steers the algorithm toward rows carrying the most uncaptured
information, regardless of the diagonal structure of ``\mathbf{A}``.

### When to Use LLAMA

| Scenario | Recommended | Alternative |
|---|---|---|
| General matrix with ``\mathbf{d}_{\text{row}}`` available cheaply | **LLAMA** | ALPACA general mode |
| Block-diagonal with zero / small diagonal | **LLAMA** | QRdALPACA |
| Rectangular ``m \times n`` with ``n \gg m`` (column norms available) | **LLAMA** (column-guided via ``\mathbf{d}_{\text{col}}``) | ALPACA general mode |
| Symmetric / Hermitian matrix | ALPACA symmetric path (2× faster) | LLAMA (works, but slower) |
| Need eigendecomposition or Takagi extraction | ALPACA | Not directly supported by LLAMA |

The key decision criterion: **do you have
``\text{diag}(\mathbf{A}\mathbf{A}^H)`` or
``\text{diag}(\mathbf{A}^H\mathbf{A})`` precomputed or cheaply
available?**  If yes, LLAMA will typically outperform ALPACA's general
mode on non-symmetric matrices, especially those with block structure
or localized singular vectors.

### Algorithm Overview

LLAMA consists of three stages: an **inner loop** for Gram-guided
ACA pivoting, an **SVD finalization** step, and an **outer loop**
for iterative correction.

#### Stage 1: Inner Loop — Gram-Guided ACA

The inner loop performs cross-coupled Schur complement deflation
(the same ACA kernel used by ALPACA for general matrices), but
with a different pivot selection strategy: instead of tracking
principal element residuals, LLAMA tracks a **row-norm residual
indicator** initialized from ``\mathbf{d}_{\text{row}}``.

**State maintained during the loop:**

- ``\mathbf{C} \in \mathbb{R}^{m \times k}``: scaled deflated columns.
- ``\mathbf{R} \in \mathbb{R}^{n \times k}``: scaled deflated rows.
- ``\mathbf{D} = \text{diag}(d_1, \ldots, d_k)``: pivot values.
- ``\text{residual}[i]``: Gram-corrected row-norm indicator for each row.

The approximation is ``\mathbf{A} \approx \mathbf{C} \mathbf{D} \mathbf{R}^H``.

**Iteration ``k``:**

1. **Row selection**: pick the unpivoted row ``i^*`` with the largest
   residual:
   ```math
   i^* = \arg\max_{i \notin \text{pivots}} \text{residual}[i]
   ```
   If ``\max \text{residual}[i] < \text{pivotol}^2``, the inner loop
   terminates.

2. **Row deflation**: fetch row ``\mathbf{A}_{i^*,:}`` and deflate:
   ```math
   \tilde{\mathbf{r}} = \mathbf{A}_{i^*,:} - \mathbf{R} \cdot (\mathbf{D} \cdot \mathbf{C}_{i^*,:}^H)
   ```

3. **Column selection**: pick the column with the largest entry:
   ```math
   j^* = \arg\max_{j \notin \text{col\_pivots}} \lvert \tilde{r}_j \rvert
   ```
   If ``\|\tilde{\mathbf{r}}\|_\infty < \text{pivotol}``, the row
   is **exhausted** — its information is already captured.  Mark it
   and move to the next row candidate.

4. **Column deflation**: fetch column ``\mathbf{A}_{:,j^*}`` and deflate:
   ```math
   \tilde{\mathbf{c}} = \mathbf{A}_{:,j^*} - \mathbf{C} \cdot (\mathbf{D} \cdot \mathbf{R}_{j^*,:}^H)
   ```

5. **Pivot acceptance**: the pivot value is ``p = \tilde{c}_{i^*}``.
   If ``|p| < \text{pivotol}``, the row is exhausted; skip it.
   Otherwise, store the scaled column and row:
   ```math
   \mathbf{C}_{:,k} = \tilde{\mathbf{c}} / p, \quad
   \mathbf{R}_{:,k} = \tilde{\mathbf{r}} / p, \quad
   d_k = p.
   ```

6. **Update residuals** (Gram-corrected formula — see below).

##### Gram-Corrected Residual Update

The tracked quantity at each row ``i`` is an approximation to the
difference between the true squared row norm and the squared norm of
its approximation:

```math
\text{residual}[i] \approx \|\mathbf{A}_{i,:}\|^2 - \|\text{approx}_{i,:}\|^2
```

When pivot ``k`` is stored, the incremental update uses the row Gram
matrix entries ``G_{tk} = \langle \mathbf{R}_{:,t}, \mathbf{R}_{:,k} \rangle``:

```math
\Delta[i] = |d_k \, C_{i,k}|^2 \, G_{kk}
          + 2 \operatorname{Re}\!\left(
              \overline{d_k \, C_{i,k}} \sum_{t < k} d_t \, C_{i,t} \, G_{tk}
            \right)
```
```math
\text{residual}[i] \leftarrow \max\!\bigl(\text{residual}[i] - \Delta[i],\; 0\bigr)
```

The ``G_{tk}`` entries are accumulated incrementally via BLAS-2
operations (one `gemv` per pivot) rather than recomputed from scratch
at finalization.  The ``\max(\cdot, 0)`` clamp is necessary because
the residual can undershoot zero — a phenomenon inherent to any
non-orthogonal decomposition (ACA, LU, etc.) due to a nonzero
cross-term ``2 \operatorname{Re}\langle \mathbf{E}, \text{approx} \rangle``
between the error and the approximation.

##### Exhausted-Row Skip Logic

When a row is selected by residual but its deflated content is
near-zero (``\|\tilde{\mathbf{r}}\|_\infty < \text{pivotol}`` or
``|p| < \text{pivotol}``), the row is **exhausted**: its information
is already captured by the existing factors.  LLAMA marks it and
uses a `needs_recompute` flag to decide the next action:

- If the flag is **true** (new Gram updates have occurred since the
  last recomputation), LLAMA **recomputes all non-pivot residuals
  from scratch** using the stored ACA factors and the row Gram matrix,
  resets the flag, and continues the inner loop with accurate residuals.
- If the flag is **false** (residuals were already recomputed after the
  most recent Gram update), the exhaustion is genuine and the inner
  loop **breaks** to finalization.

The flag is set after each successful pivot (whose Gram update may
cause residuals to overshoot to zero) and cleared after each
recomputation.  This limits the expensive ``O(m r^2)``
recomputation to at most ``r`` times total (once per successful
pivot).

This is particularly important for block-structured and non-square
matrices: rows in already-captured blocks have near-zero deflated
content but may still have large Gram residuals (false positives
from the residual overshoot).  The ``\mathbf{P}\boldsymbol{\Sigma}^2
\mathbf{P}`` correction (Stage 3) provides additional safeguards
across outer iterations.

#### Stage 2: SVD Finalization

After the inner loop converges (all residuals below the tolerance
squared), the raw cross-coupled factors are converted into an
orthonormal column-space basis via Cholesky and SVD of small
rank-sized matrices:

1. **Column Gram**: compute ``\mathbf{C}^H \mathbf{C}`` and
   Cholesky-factor it: ``\mathbf{L}_C = \text{chol}(\mathbf{C}^H \mathbf{C})``.
2. **Row Gram**: already accumulated as ``\mathbf{R}^H \mathbf{R}``
   during the inner loop.  Cholesky-factor:
   ``\mathbf{L}_R = \text{chol}(\mathbf{R}^H \mathbf{R})``.
3. **Core matrix**: form the ``r \times r`` core
   ``\mathbf{B} = \mathbf{L}_C^{-H} \, \mathbf{D} \, \mathbf{L}_R^{-H}``.
4. **SVD**: decompose ``\mathbf{B} = \mathbf{U}_B \boldsymbol{\Sigma} \mathbf{V}_B^H``.
5. **Truncate**: retain only components with ``\sigma_i > \tau``,
   giving effective rank ``n_k``.
6. **Orthonormal basis**:
   ``\mathbf{Q} = \mathbf{C} \, \mathbf{L}_C^{-H} \, \mathbf{U}_{B}[:,1:n_k]``
   ``\in \mathbb{R}^{m \times n_k}``.
7. **Right singular vectors** (when `fullsvd=true`):
   ``\mathbf{V} = \mathbf{R} \, \mathbf{L}_R^{-H} \, \mathbf{V}_{B}[:,1:n_k]``
   ``\in \mathbb{R}^{n \times n_k}``.

The finalization cost is ``O(mr^2 + r^3)`` (or ``O(mr^2 + r^2 n)``
with `fullsvd`), dominated by the column Gram computation and the
Q formation.

#### Stage 3: Iterative Correction — the ``\mathbf{P}\boldsymbol{\Sigma}^2\mathbf{P}`` Estimate

After finalization, the Gram residuals may have falsely converged
to zero in rows where the ACA cross-term caused undershoot.  To
detect this, LLAMA computes a **corrected residual** using an
estimate of the true Gram matrix
``\mathbf{G}_A = \mathbf{Q}^H \mathbf{A} \mathbf{A}^H \mathbf{Q}``
from only the accessed pivot rows.

**Key observation (ACA interpolation property):**
For every pivot row ``i_s``, the ACA interpolation condition
guarantees ``\mathbf{A}_{i_s,:} = (\mathbf{C}\mathbf{D}\mathbf{R}^H)_{i_s,:}``
exactly.

This lets us build a partial estimate of ``\mathbf{Q}^H \mathbf{A}``:

```math
(\mathbf{Q}^H \mathbf{A})_{\text{partial}}
  = \mathbf{Q}[\text{pivots},:]^H \cdot \mathbf{A}[\text{pivots},:]
  = \mathbf{Q}[\text{pivots},:]^H \cdot (\mathbf{C}\mathbf{D}\mathbf{R}^H)[\text{pivots},:]
```

Define the **pivot energy-capture matrix**:

```math
\mathbf{P} = \mathbf{Q}[\text{pivots},:]^H \, \mathbf{Q}[\text{pivots},:] \quad (n_k \times n_k)
```

Since ``\mathbf{Q}`` has orthonormal columns with energy spread
across all ``m`` rows, and we only use the ``r`` pivot rows, we have
``\mathbf{P} \preceq \mathbf{I}`` (positive semidefinite, dominated
by identity).

The estimated Gram matrix is:

```math
\tilde{\mathbf{G}}_A = \mathbf{P} \boldsymbol{\Sigma}^2 \mathbf{P}
```

And the corrected residual for each row:

```math
\text{corrected}[i] = d_{\text{row}}[i]
  - \mathbf{q}_i^H \, (\mathbf{P} \boldsymbol{\Sigma}^2 \mathbf{P}) \, \mathbf{q}_i
```

where ``\mathbf{q}_i = \mathbf{Q}[i,:]``.

**Why this works:**
Since ``\mathbf{P} \preceq \mathbf{I}``, the correction
``\mathbf{P}\boldsymbol{\Sigma}^2\mathbf{P}`` subtracts *less* energy
than the Gram formula.  This positive bias correctly reveals rows
where the ACA approximation overshot:

- **Genuinely converged rows** have ``\text{corrected}[i] \approx 0``
  because ``\mathbf{P}`` captures most of their energy.
- **Rows where the Gram overshot** have ``\text{corrected}[i] > 0``
  because ``\mathbf{P}`` doesn't capture energy from non-pivot rows,
  leaving a positive residual that triggers re-exploration.

If ``\max_i \text{corrected}[i] \geq \text{pivotol}^2``, the algorithm
re-enters the inner loop with the corrected residuals to discover
more pivots.  This outer loop runs at most 10 iterations.

**Example: block-diagonal matrix with 4 blocks.**
After the first inner pass, LLAMA may discover only one block via
ACA cycling.  The ``\mathbf{P}\boldsymbol{\Sigma}^2\mathbf{P}``
correction reveals that rows in the undiscovered blocks have large
corrected residuals.  The next inner pass discovers those blocks.
In practice, LLAMA discovers all blocks within 2–3 outer iterations.

### Column-Guided Mode

When the number of columns ``n`` exceeds the number of rows ``m``,
the inner loop is dominated by the row Gram accumulation cost
``O(n r^2)`` (one `gemv` of length ``n`` per pivot iteration).
By passing ``\mathbf{d}_{\text{col}} = \text{diag}(\mathbf{A}^H \mathbf{A})``
instead of ``\mathbf{d}_{\text{row}}``, LLAMA internally transposes
the problem: it works on ``\mathbf{A}^T`` (an ``n \times m`` matrix),
swapping the roles of rows and columns.

This reduces the dominant loop cost from ``O((m + 2n) r^2)`` to
``O((2m + n) r^2)``, approaching a 2× speedup as ``n/m \to \infty``.
The returned ``\mathbf{Q}`` still spans the column space of the
original matrix ``\mathbf{A}`` — the transposition is handled
transparently.

For dense matrices, if neither ``\mathbf{d}_{\text{row}}`` nor
``\mathbf{d}_{\text{col}}`` is provided, LLAMA automatically selects
column-guided mode when ``n > m`` and row-guided mode otherwise.

### Complexity

For an ``m \times n`` matrix of numerical rank ``r``:

| | Row-guided (``\mathbf{d}_{\text{row}}``) | Column-guided (``\mathbf{d}_{\text{col}}``) |
|---|---|---|
| **Element accesses** | ``r`` columns + up to ``m`` rows | ``r`` rows + up to ``n`` columns |
| **Memory** | ``O(mr + nr + r^2)`` | ``O(mr + nr + r^2)`` |
| **Inner loop arithmetic** | ``O((m + 2n) r^2)`` | ``O((2m + n) r^2)`` |
| **Finalization** | ``O(mr^2 + r^3)`` | ``O(nr^2 + r^3)`` |

The row Gram entries are accumulated incrementally via BLAS-2
operations during the inner loop, avoiding an ``O(nr^2)``
recomputation at finalization.

### Output

Unlike ALPACA (which produces raw left/right factors), LLAMA
directly outputs an **orthonormal column-space basis** ``\mathbf{Q}``
and **approximate singular values** ``\boldsymbol{\sigma}`` from the
SVD finalization.  When `fullsvd=true`, right singular vectors
``\mathbf{V}`` are also returned, giving the full low-rank SVD
``\mathbf{A} \approx \mathbf{Q} \, \text{diag}(\boldsymbol{\sigma}) \, \mathbf{V}^H``.

### Relationship to ALPACA

LLAMA reuses ALPACA's cross-coupled Schur complement deflation
engine (the ACA inner loop) and shares the same matrix-free
interface (`column!`, `row!`).  The key differences are:

| Aspect | ALPACA (general mode) | LLAMA |
|---|---|---|
| **Row pivot selection** | ACA greedy + principal elements | ``\mathbf{d}_{\text{row}}``-guided residual |
| **Column pivot selection** | ACA greedy | ACA greedy (same) |
| **Convergence signal** | Principal element residuals | Row-norm residuals |
| **Iterative correction** | None (or QRdALPACA post-hoc) | ``\mathbf{P}\boldsymbol{\Sigma}^2\mathbf{P}`` outer loop |
| **Output** | Raw factors ``\mathbf{L}_C, \mathbf{L}_R`` | Orthonormal ``\mathbf{Q}`` + singular values |
| **Block-structure robustness** | May miss blocks | Discovers all blocks |
| **External information** | Optional principal pairs | Requires ``\mathbf{d}_{\text{row}}`` or ``\mathbf{d}_{\text{col}}`` |

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
