# Pivoted Symmetric Decomposition

This document explains the pivoted symmetric decomposition functions:
`ldlt_pivoted_symmetric_decompose` and `qr_pivoted_symmetric_decompose`.
Both share the same two-step structure — pivot selection followed by
Nyström approximation — but use different strategies for Step I.
The LDLT variant is covered in full detail first, followed by the QR
variant with a comparison of their strengths and weaknesses.

## Motivation

Suppose we have a large symmetric matrix $M$ of size $n \times n$.
In quantum chemistry, such matrices routinely appear as two-electron
integral matrices, where $n$ can be tens of thousands or more.

Three important properties of these matrices make decomposition attractive:

1. **They are symmetric**: $M = M^T$ (transpose, not conjugate transpose).
2. **They are often low-rank**: only $r \ll n$ eigenvalues are significantly
   different from zero.
3. **We don't need the matrix itself** — we need a compact factored form
   $M \approx L \, S \, L^T$, where $L$ is $n \times r$ and $S$ is a
   diagonal sign matrix ($\pm 1$ on the diagonal).

Computing $L$ directly (e.g., by eigendecomposition of $M$) costs $O(n^3)$.
The LDLT pivoted decomposition achieves this in roughly $O(n \, r^2)$ by
being smart about *which* columns of $M$ to look at.

## Overview of the Two-Step Algorithm

The algorithm has two phases:

| Step | What it does | Cost |
|---|---|---|
| **Step I** — Pivot selection | Identify $r$ "important" column indices of $M$ | $O(n \, r^2)$ |
| **Step II** — Nyström approximation | Build $L$ from those $r$ columns | $O(n \, r^2)$ |

After both steps, we have $M \approx L \, S \, L^T$ with $L$ of shape
$(n, r)$.

---

## Step I: Pivot Selection via Batched LDLT

### The Key Idea

The LDLT factorization of a symmetric matrix is:

$$M = L \, D \, L^T$$

where $L$ is unit lower triangular (ones on the diagonal) and $D$ is
diagonal.  Unlike Cholesky ($M = G G^T$, which requires $M$ to be positive
semi-definite), LDLT works for **indefinite** matrices because $D$ can
contain negative entries.

We don't compute the full LDLT factorization.  Instead, we use it as a
**pivot selection strategy**: we iteratively pick the columns that carry
the most "weight" in the matrix, until the remaining matrix is negligible.

### Diagonal Elements as Importance Metric

In the LDLT factorization, the diagonal of $D$ equals the diagonal of
the *Schur complement* at each step.  After selecting pivots
$\{p_1, \ldots, p_k\}$, the remaining diagonal is:

$$d_I^{(\text{new})} = d_I^{(\text{old})} - \ell_{I,k}^2 \cdot d_{p_k}$$

where $\ell_{I,k}$ is the LDLT coefficient (the $I$-th entry of the
$k$-th LDLT column divided by $d_{p_k}$).

The absolute values $|d_I|$ tell us how much "energy" column $I$ still
carries.  When $|d_I| < \text{tol}$, column $I$ is negligible and we
discard it.

!!! note "Why absolute values?"
    For positive definite matrices, all $d_I > 0$ and we could compare
    directly.  For indefinite or complex symmetric matrices, $d_I$ can
    be negative or complex.  Using $|d_I|$ for screening handles all
    cases uniformly.

### Batched Processing with the Span Factor

Processing one pivot at a time is correct but can be slow (many iterations
with small updates).  Instead, the algorithm works in **batches**.

In each iteration:

1. **Find the maximum**: $d_{\max} = \max_{I \in D} |d_I|$
2. **Select a batch**: collect all indices $I$ where
   $|d_I| \geq \sigma \cdot d_{\max}$

The parameter $\sigma$ (default: 0.01) is called the **span factor**.
It controls how "greedy" each batch is:

| $\sigma$ | Batch size | Behavior |
|---|---|---|
| Close to 1 | Small (only near-maximum elements) | Many iterations, very selective |
| Close to 0 | Large (almost everything) | Few iterations, less selective |
| 0.01 (default) | Moderate | Good trade-off for typical matrices |

!!! tip "Analogy"
    Think of the span factor like a fishing net.  A tight net
    ($\sigma \approx 0$) catches everything — all fish end up in the
    same batch.  A wide net ($\sigma \approx 1$) lets the small fish
    slip through and only catches the biggest ones each round.
    The default $\sigma = 0.01$ catches all fish within two orders of
    magnitude of the biggest one.

### The Inner Loop: Pivoted LDLT Within Each Batch

Within a batch of $b$ candidate indices, we perform pivoted LDLT
decomposition.  This means:

```
for each candidate in the batch (sorted by |d_I| descending):
    1. Pick the candidate with the largest |d_I| as the next pivot
    2. Compute its LDLT column vector from M
    3. Subtract contributions from ALL previously selected pivots (BLAS-2 operation)
    4. Divide by d[pivot] to get unit triangular coefficients
    5. Update all remaining diagonals via the Schur complement formula
    6. Mark the pivot as "done"
```

Here is the Schur complement update in detail.  After selecting pivot $q$
with diagonal value $d_q$, each remaining diagonal element updates as:

$$d_I \leftarrow d_I - \ell_{I,q}^2 \cdot d_q$$

where $\ell_{I,q} = M_{I,q}^{\text{(residual)}} / d_q$ is the LDLT coefficient.

!!! note "What is BLAS-2?"
    BLAS (Basic Linear Algebra Subprograms) is a highly optimized library
    for linear algebra operations.  "BLAS-2" refers to matrix-vector
    operations like $y \leftarrow y - A \cdot x$, which run at near
    peak memory bandwidth.  The implementation uses Julia's `mul!` function
    which calls BLAS internally.

### Screening and Compression

After each batch, the algorithm removes indices whose $|d_I|$ has dropped
below the tolerance.  This progressively shrinks the working set $D$,
so later iterations work on fewer indices and run faster.

When the working set shrinks significantly, the internal storage is
**compressed**: the LDLT vectors (stored in a pre-allocated matrix) are
remapped to use only the surviving indices.  This keeps the memory
footprint proportional to the current working set size, not the original
matrix size.

### Step I: Complete Algorithm

```
Algorithm: LDLT Pivot Selection
─────────────────────────────────────────────────────
Input:  M (n×n symmetric), tol, σ
Output: pivots (list of r selected column indices)

1.  d[I] ← M[I,I]  for all I = 1, …, n
2.  D ← { I : |d[I]| ≥ tol }
3.  pivots ← []

4.  while D is non-empty:
5.    d_max ← max{ |d[I]| : I ∈ D }
6.    if d_max < tol: break
7.    batch ← { I ∈ D : |d[I]| ≥ σ · d_max }, sorted by |d[I]| desc
8.    
9.    for each candidate in batch:
10.     q ← candidate with largest |d[I]|
11.     if |d[q]| < tol: break
12.     
13.     Compute LDLT column v[I] = M[I, q]  for I ∈ D
14.     Subtract previous pivots: v ← v − L_stored · (D_stored ∘ L_stored[q,:])
15.     Divide by diagonal:  ℓ[I] = v[I] / d[q]
16.     Update diagonals:    d[I] ← d[I] − ℓ[I]² · d[q]  for I ∈ D
17.     Store ℓ and d[q];  append q to pivots
18.     
19.   Screen D: remove indices with |d[I]| < tol
20.   Compress storage to match surviving indices

21. return pivots
```

---

## Step II: The Nyström Approximation

Once we have the $r$ pivot indices $B = \{p_1, \ldots, p_r\}$, we build
the final decomposition using the **Nyström formula** (also known as the
Resolution of the Identity, or RI, approximation):

$$M \approx M_{:,B} \; J^{-1} \; M_{B,:}^T$$

where $J = M_{B,B}$ is the $r \times r$ submatrix at the pivot rows and
columns, and $M_{:,B}$ denotes all $n$ rows but only the $r$ pivot
columns.

### Why Does This Work?

If the pivot columns span the column space of $M$, then every column of
$M$ can be written as a linear combination of the pivot columns.  The
Nyström formula finds exactly these coefficients.

More precisely, if $M$ has rank $r$ and the pivot columns form a basis
for the column space, then the approximation is **exact**.  If $r$ is an
approximation to the true rank, the error is controlled by the
eigenvalues we dropped.

### Computing $L$ from the Nyström Formula

We want $L$ such that $M \approx L \, S \, L^T$.  Starting from
$M \approx M_{:,B} \, J^{-1} \, M_{B,:}^T$, we factorize $J^{-1}$:

**For real symmetric matrices:**

We compute the eigendecomposition of the small $r \times r$ matrix $J$:

$$J = Q \, \Lambda \, Q^T$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_r)$.  Then:

$$J^{-1} = Q \, |\Lambda|^{-1} \, Q^T$$

and we define:

$$C_k = \frac{q_k}{\sqrt{|\lambda_k|}}$$

so that $C \, C^T = J^{-1}$ (up to signs).  The final decomposition
vectors are:

$$L = M_{:,B} \cdot C$$

Negative eigenvalues $\lambda_k < 0$ are tracked in `neg_indices`, and
the sign matrix $S$ has $S_{kk} = -1$ for those indices.

**For complex symmetric matrices:**

We use a Takagi-like factorization via the SVD of $J$:

$$J = U \, \Sigma \, V^T$$

From the SVD, we extract Takagi vectors $U_T$ such that $J = U_T \, \Sigma \, U_T^T$
(with $U_T$ unitary).  Then:

$$C = \overline{U_T} \, \Sigma^{-1/2}$$

and again $L = M_{:,B} \cdot C$.  For complex symmetric matrices, all
singular values are non-negative, so `neg_indices` is always empty.

### Step II: Complete Algorithm

```
Algorithm: Nyström Vectors
─────────────────────────────────────────────────────
Input:  M (n×n symmetric), pivots B = {p₁, …, pᵣ}, tol
Output: L (n×r), rank, neg_indices

1.  J ← M[B, B]                          # r×r submatrix
2.  if real:
3.    Eigendecompose J = Q Λ Qᵀ
4.    Keep eigenvalues with |λₖ| > tol
5.    Cₖ = qₖ / √|λₖ|
6.    neg_indices ← { k : λₖ < 0 }
7.  if complex:
8.    SVD/Takagi: J = U_T Σ U_Tᵀ
9.    Keep singular values σₖ > tol
10.   C = conj(U_T) · Σ^{-1/2}
11.   neg_indices ← ∅
12. L ← M[:, B] · C                      # n×r matrix
13. return (L, rank, neg_indices)
```

---

## Worked Example

Consider a simple $4 \times 4$ rank-2 matrix ($\text{tol} = 0.01$,
$\sigma = 0.1$):

$$M = \begin{pmatrix} 4 & 2 & 0 & 2 \\ 2 & 1 & 0 & 1 \\ 0 & 0 & 9 & 3 \\ 2 & 1 & 3 & 2 \end{pmatrix}$$

**Step I: Pivot Selection**

1. Initial diagonals: $d = [4, 1, 9, 2]$.
2. $d_{\max} = 9$.  Threshold: $\sigma \cdot d_{\max} = 0.9$.
3. Batch: all indices with $|d_I| \geq 0.9$ → $\{1, 2, 3, 4\}$ (all qualify).
4. Sort by $|d_I|$ descending: $[3, 1, 4, 2]$.

   **Pivot 1**: $q = 3$, $d_q = 9$.
   - LDLT column (before dividing): $v = M[:,3] = [0, 0, 9, 3]$
   - Dividing by $d_q = 9$: $\ell = [0, 0, 1, 1/3]$
   - Update diagonals: $d_I \leftarrow d_I - \ell_I^2 \cdot 9$
     - $d_1 = 4 - 0 = 4$, $d_2 = 1 - 0 = 1$, $d_3 = 0$ (consumed), $d_4 = 2 - 1 = 1$

   **Pivot 2**: $q = 1$ (now has $|d_1| = 4$, largest remaining).
   - LDLT column: $v = M[:,1] = [4, 2, 0, 2]$
   - Subtract previous pivot's contribution:
     $v \leftarrow v - \ell_{:,3} \cdot (d_3 \cdot \ell_{1,3})$.
     Since $\ell_{1,3} = 0$ (row 1 in pivot 3's column), no change.
   - Dividing by $d_q = 4$: $\ell = [1, 1/2, 0, 1/2]$
   - Update diagonals: $d_I \leftarrow d_I - \ell_I^2 \cdot 4$
     - $d_2 = 1 - 1 = 0$, $d_4 = 1 - 1 = 0$

5. All remaining diagonals are 0 → **done**.
   Pivots: $B = [3, 1]$, rank $= 2$.

**Step II: Nyström Approximation**

1. $J = M_{B,B} = M[\{3,1\}, \{3,1\}] = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$
2. Eigendecompose: $\Lambda = \text{diag}(4, 9)$, $Q = I$ (already diagonal).
3. $C = \text{diag}(1/\sqrt{4},\; 1/\sqrt{9}) = \text{diag}(1/2,\; 1/3)$
4. $L = M_{:,B} \cdot C = \begin{pmatrix} 0 & 4 \\ 0 & 2 \\ 9 & 0 \\ 3 & 2 \end{pmatrix} \cdot \begin{pmatrix} 1/2 & 0 \\ 0 & 1/3 \end{pmatrix} = \begin{pmatrix} 0 & 4/3 \\ 0 & 2/3 \\ 9/2 & 0 \\ 3/2 & 2/3 \end{pmatrix}$

Reconstruction: $L L^T = M$ exactly (because $M$ is rank 2 and we found both pivots).

---

## Why Is It So Efficient?

### 1. Only Diagonal Updates in Step I

The LDLT pivot selection never forms the full Schur complement matrix.
It only tracks the **diagonal** elements $d_I$, which update via:

$$d_I \leftarrow d_I - \ell_{I,k}^2 \cdot d_k$$

This is a scalar operation per index per pivot — much cheaper than
updating an $n \times n$ matrix.  The full column from $M$ is read once
per pivot (a single column, $O(n)$), and the update of all diagonals
is also $O(n)$ per pivot.  Over $r$ pivots, Step I costs $O(n \cdot r)$
for the diagonal updates, plus $O(n \cdot r^2)$ for subtracting previous
pivots via BLAS-2.

### 2. BLAS-2 for Pivot Subtraction

The most expensive part of the inner loop is subtracting the contribution
of all previous pivots from the new column:

$$v \leftarrow v - L_{\text{stored}} \cdot (\text{coeffs})$$

This is a matrix-vector multiply (`mul!` → BLAS `gemv`), which runs at
near memory bandwidth on modern hardware.  The implementation uses
in-place operations (no allocation) and pre-allocated buffers.

### 3. Pre-allocated Buffers with Amortized Doubling

Instead of allocating new arrays each iteration, the implementation
pre-allocates storage and grows it by doubling when needed.  This is
the same strategy used by `push!` on Julia `Vector`s.  Specifically:

- `L_storage`: pre-allocated matrix for LDLT column vectors
- `D_storage`: pre-allocated vector for diagonal values
- `coeffs`, `V_col`, `Q_batch`, `new_D_buf`: work buffers, allocated
  once and reused

### 4. BitVector for O(1) Pivot Lookup

To check whether an index has already been selected as a pivot, the
implementation uses a `BitVector` (`is_pivot = falses(n)`).  This gives
$O(1)$ lookup and uses only $n$ bits of memory, compared to a `Set{Int}`
which would use $O(r)$ memory and have higher constant-factor overhead.

### 5. Progressive Screening Shrinks the Working Set

After each batch, indices with $|d_I| < \text{tol}$ are removed from the
active set $D$.  For a rank-$r$ matrix, most indices become negligible
after $O(r)$ pivots.  This means later iterations work on progressively
smaller sets — the algorithm naturally speeds up as it goes.

### 6. The Nyström Step Is a Single Matrix Multiply

Step II requires:
- One eigendecomposition (or SVD) of the $r \times r$ matrix $J$: $O(r^3)$
- One matrix multiply $M_{:,B} \cdot C$: $O(n \cdot r^2)$

Since $r \ll n$, the $O(r^3)$ part is negligible, and the matrix multiply
dominates.  This is a BLAS-3 operation (matrix-matrix multiply), which
is the most efficient operation on modern CPUs.

### Cost Summary

| Operation | Cost | Dominates when |
|---|---|---|
| Step I: diagonal updates | $O(n \cdot r)$ | Always fast |
| Step I: BLAS-2 column subtraction | $O(n \cdot r^2)$ | $r$ is moderate |
| Step II: eigendecompose $J$ | $O(r^3)$ | Never (small $r$) |
| Step II: form $L = M_{:,B} \cdot C$ | $O(n \cdot r^2)$ | Large $n$ |
| **Total** | **$O(n \cdot r^2)$** | |

Compare this to a full eigendecomposition at $O(n^3)$.  For a typical
quantum chemistry problem with $n = 10{,}000$ and $r = 500$, the
speedup is roughly $(10^4)^3 / (10^4 \cdot 500^2) = 4 \times 10^5$.

---

## Assumptions and Approximations

### 1. The Matrix Must Be Symmetric

The algorithm requires $M = M^T$ (for complex matrices, transpose, **not**
conjugate transpose $M^\dagger$).  This is always satisfied for
two-electron integrals in quantum chemistry, but should be verified
for other applications.

### 2. Low-Rank Assumption (Nyström Approximation)

The Nyström formula $M \approx M_{:,B} \, J^{-1} \, M_{B,:}^T$ is
exact when the pivot columns span the column space of $M$.  It is an
approximation when $r < \text{rank}(M)$.  The quality of the
approximation depends on how well the selected pivots capture the
important directions.

### 3. Diagonal Elements Reflect Column Importance

The LDLT pivot selection strategy assumes that $|d_I|$ (diagonal of the
Schur complement) is a good proxy for the importance of column $I$.
This is justified by the following:

- For positive definite $M$: the diagonal elements of the Schur complement
  are the variances of the residual — large values mean important columns.
- For indefinite $M$: the absolute value $|d_I|$ still measures the
  magnitude of the residual contribution.

This is the same principle used in pivoted Cholesky decomposition, but
generalized to indefinite matrices.

### 4. The Span Factor Groups Similar-Magnitude Columns

The span factor $\sigma$ assumes that columns within a factor of
$1/\sigma$ in importance can be processed together without loss of
quality.  This is a heuristic that works well in practice but has no
strict theoretical guarantee for the batch ordering.

### 5. Complex Symmetric ≠ Hermitian

For complex matrices, the algorithm decomposes $M = M^T$, **not**
$M = M^\dagger$.  This distinction matters: a complex symmetric matrix
$M = M^T$ is factorized via the Takagi decomposition ($M = U \Sigma U^T$
with $U$ unitary), while a Hermitian matrix $M = M^\dagger$ is factorized
via eigendecomposition ($M = U \Lambda U^\dagger$).  The LDLT
decomposition and Nyström formula both use transpose consistently.

---

## Where It Works Well

1. **Low-rank positive semi-definite matrices**: The "textbook" case.
   Pivoted LDLT (like pivoted Cholesky) finds the important columns
   quickly, and the Nyström approximation is near-exact.

2. **Low-rank indefinite matrices**: Unlike Cholesky (which fails for
   indefinite matrices), LDLT handles negative eigenvalues naturally
   through the $D$ diagonal.  The `neg_indices` output tracks which
   components have negative sign.

3. **Complex symmetric matrices**: These appear in quantum chemistry
   with complex MO coefficients (e.g., periodic boundary conditions or
   magnetic fields).  The algorithm handles them seamlessly via the
   Takagi-based Nyström step.

4. **Matrices with rapidly decaying spectrum**: When eigenvalues decay
   quickly (as is typical for two-electron integrals), few pivots are
   needed and the algorithm is extremely fast.

5. **Large matrices where only a few columns matter**: The algorithm's
   cost scales with the *compressed rank* $r$, not the full size $n$.
   For $n = 50{,}000$ and $r = 200$, only 200 columns of $M$ are ever
   read.

---

## Where It Can Fail or Perform Poorly

### 1. Full-Rank Matrices

If the matrix has no low-rank structure ($r \approx n$), the algorithm
selects nearly all columns as pivots.  The cost becomes $O(n^3)$ — no
better than a full eigendecomposition, and likely slower due to the
overhead of the pivot selection machinery.

### 2. Diagonal Elements Are Poor Importance Indicators

The LDLT strategy relies on $|d_I|$ as a proxy for column importance.
In rare cases, a column can be important despite having a small diagonal
element (or vice versa).  For example, consider:

$$M = \begin{pmatrix} \epsilon & 1 \\ 1 & \epsilon \end{pmatrix}$$

with $\epsilon \ll 1$.  The diagonal elements are tiny, but the matrix
has rank 2 and both columns are important.  With a tolerance larger than
$\epsilon$, the algorithm would incorrectly discard both columns.

In practice, this pathology is rare for matrices arising from physical
problems — they tend to have large diagonal elements for important columns.

### 3. Highly Ill-Conditioned Pivot Submatrices

If the selected pivots produce a nearly singular $J = M_{B,B}$, the
Nyström step amplifies numerical errors through $J^{-1}$.  The algorithm
mitigates this by screening eigenvalues of $J$ below the tolerance, but
extreme ill-conditioning can still degrade accuracy.

### 4. The Span Factor Is a Heuristic

The choice of $\sigma$ affects both speed and accuracy.  Too large
($\sigma \to 1$): many small batches, slower but more selective.  Too
small ($\sigma \to 0$): one giant batch, fast but the within-batch
LDLT may miss important columns because it processes the entire batch
in one shot.  The default $\sigma = 0.01$ is empirically good but
not optimal for all matrices.

### 5. Schur Complement Diagonal Updates Can Accumulate Errors

The diagonal update $d_I \leftarrow d_I - \ell_{I,k}^2 \cdot d_k$ is
applied sequentially.  For many pivots, floating-point rounding errors
in $d_I$ can accumulate.  In the worst case, $d_I$ could become slightly
negative for a PSD matrix (the implementation clamps only the residual
norms in the QR variant, not in the LDLT variant, because negative $d_I$
values are valid for indefinite matrices).

---

## The Orthogonalization Function

After the decomposition, we have $L$ such that:

$$M \approx L \, S \, L^T$$

where $S = \text{diag}(\pm 1)$.  The columns of $L$ are **not**
orthogonal.  Sometimes we need an orthonormal basis — for example, to
obtain eigenvalues or to use the decomposition as a spectral
approximation.

The `orthogonalize` function converts $L$ and `neg_indices` into:

$$M \approx U \, \text{diag}(\lambda_1, \ldots, \lambda_r) \, U^T$$

where $U$ has orthonormal columns and $\lambda_k$ are (approximate)
eigenvalues of $M$.

### How It Works

The key insight is that we can reduce the $n \times r$ orthogonalization
problem to a small $r \times r$ problem.

**Step 1: QR factorization of $L$**

$$L = Q \, R$$

where $Q$ is $n \times r$ with orthonormal columns and $R$ is $r \times r$
upper triangular.  This costs $O(n \, r^2)$.

**Step 2: Transform the small matrix**

Since $M \approx L \, S \, L^T = Q R \, S \, R^T Q^T$, the matrix
$B = R \, S \, R^T$ is $r \times r$ and captures all the spectral
information.

**Step 3: Eigendecompose (or Takagi-decompose) $B$**

For real matrices:

$$B = R \, S \, R^T = V \, \Lambda \, V^T$$

This is a standard eigendecomposition of a small $r \times r$ symmetric
matrix.  Then $U = Q \cdot V$ and $\lambda_k$ are the eigenvalues.

For complex symmetric matrices:

$$B = R \, R^T = U_T \, \Sigma \, U_T^T$$

This is a Takagi factorization (computed via SVD), giving $U = Q \cdot U_T$
and the singular values $\sigma_k$ (all non-negative).

### Why This Is Efficient

The full eigendecomposition of $M$ (size $n \times n$) costs $O(n^3)$.
By first compressing via LDLT ($O(n \, r^2)$), then orthogonalizing
the compressed representation ($O(n \, r^2)$ for QR plus $O(r^3)$ for
the small eigendecomposition), we get the same result for $O(n \, r^2)$
total.

### Complete Pipeline

```
M (n×n symmetric)
  │
  │  ldlt_pivoted_symmetric_decompose(M, tol)     O(n·r²)
  ▼
L (n×r), neg_indices
  │
  │  orthogonalize(L, neg_indices)                 O(n·r²)
  ▼
U (n×r orthonormal), λ (r eigenvalues)
  │
  result: M ≈ U · diag(λ) · Uᵀ                   approximate eigendecomposition
```

This pipeline gives an approximate eigendecomposition (or Takagi
factorization) of a large matrix $M$ in $O(n \, r^2)$ time — vastly
faster than the $O(n^3)$ full decomposition when $r \ll n$.

---

## QR Pivoted Symmetric Decomposition

The QR variant (`qr_pivoted_symmetric_decompose`) shares the same
two-step structure as the LDLT method: **Step I** selects pivot columns,
**Step II** applies the Nyström formula (identical to the LDLT case).
The difference lies entirely in how Step I chooses and validates pivots.

### Importance Metric: Squared Column Norms

Instead of diagonal elements $d_I$, the QR variant uses the squared
column norm as its importance metric:

$$r_I = \|M_{:,I}\|^2 = \sum_j |M_{j,I}|^2$$

This is always non-negative (even for indefinite or complex matrices),
which simplifies the screening logic — no absolute values are needed.

!!! note "Column norms vs. diagonal elements"
    The squared column norm $r_I$ uses information from the *entire*
    column $M_{:,I}$, while the LDLT diagonal $d_I = M_{II}$ uses only
    one element.  This makes norms a more robust importance indicator
    in some cases, but computing them initially costs $O(n^2)$ (one
    pass over the full matrix) versus $O(n)$ for the diagonal.

### Step I: Batched Column-Pivoted QR

The algorithm maintains a set of candidate indices $D$ and an
orthonormal basis $Q_{\text{acc}}$ accumulated from previous batches.
Each iteration proceeds as follows:

```
Algorithm: QR Pivot Selection
─────────────────────────────────────────────────────
Input:  M (n×n symmetric), tol, σ
Output: pivots (list of r selected column indices)

1.  r[I] ← ‖M[:,I]‖²  for all I = 1, …, n
2.  D ← { I : r[I] ≥ tol² }
3.  pivots ← [],  Q_acc ← ∅  (orthonormal basis)

4.  while D is non-empty:
5.    r_max ← max{ r[I] : I ∈ D }
6.    if r_max < tol²: break
7.    batch ← { I ∈ D : r[I] ≥ σ · r_max }, sorted by r[I] desc
8.    Cap batch to at most max_batch columns
9.
10.   Extract cols ← M[:, batch]
11.   Project out Q_acc:  cols ← cols − Q_acc · (Q_acc' · cols)
12.   Column-pivoted QR:  cols = Q · R · P
13.   n_new ← number of |R[k,k]| > tol
14.   if n_new = 0: break
15.
16.   Append top n_new pivots (by QR ordering) to pivots
17.   Append corresponding Q columns to Q_acc
18.
19.   Update residual norms:
20.     r[I] ← r[I] − ‖Q_new' · M[:,I]‖²   for I ∈ D
21.   Screen D: remove indices with r[I] < tol²

22. return pivots
```

### Key Differences from LDLT

**Within-batch pivot selection:**  
The LDLT variant processes candidates *one by one* inside each batch,
using the Schur complement formula to update diagonals after each pivot.
The QR variant hands the entire batch (up to `max_batch` columns) to a
single column-pivoted QR factorization, which selects and orders the
important columns in one shot.  This is a LAPACK-level operation
(`geqp3`) that is heavily optimized.

**Residual updates:**  
LDLT updates the diagonal via the cheap scalar formula
$d_I \leftarrow d_I - \ell_{I,k}^2 \cdot d_k$, which is $O(n)$ per
pivot.  QR must update the residual column norms by projecting
*all remaining columns* of $M$ against the new basis vectors:
$r_I \leftarrow r_I - \|Q_{\text{new}}^T \, M_{:,I}\|^2$.  This
requires reading the corresponding columns of $M$ and costs
$O(n \cdot n_{\text{new}} \cdot n_D)$ per batch.  It is the main
performance bottleneck of the QR variant.

**Batch size control:**  
The QR variant caps the batch size (default: 256, growing adaptively
with discovered rank) to prevent the QR factorization from becoming
$O(n^3)$ when the span factor selects too many candidates at once.
The LDLT variant does not need this cap because its within-batch
processing is $O(n)$ per pivot regardless of batch size.

### Performance Characteristics

Both methods share the same asymptotic complexity $O(n \, r^2)$, but
with different constant factors:

| Operation | LDLT | QR |
|---|---|---|
| Importance metric | $d_I = M_{II}$, $O(n)$ to initialize | $r_I = \|M_{:,I}\|^2$, $O(n^2)$ to initialize |
| Within-batch selection | Sequential LDLT, $O(n \cdot b)$ | Column-pivoted QR, $O(n \cdot b^2)$ |
| Residual update | Scalar: $d_I \leftarrow d_I - \ell^2 d$, $O(n)$ per pivot | Projection: $Q^T M$, $O(n \cdot n_{\text{new}} \cdot n_D)$ per batch |
| Memory for basis | Not needed (diagonals suffice) | Accumulates $Q_{\text{acc}}$ ($n \times r$) |

In practice, for a $10{,}000 \times 10{,}000$ matrix of rank 500, LDLT
is roughly 4–5× faster than QR due to the cheaper residual updates.

### When to Prefer QR over LDLT

1. **Matrices where diagonals are misleading:**  QR uses full column
   norms, which capture off-diagonal structure that the LDLT diagonal
   $d_I = M_{II}$ might miss.  For matrices where important columns
   have small diagonal elements but large off-diagonal entries, QR
   can find pivots that LDLT would overlook.

2. **Cross-validation:**  Since QR and LDLT use independent importance
   metrics, running both and comparing the selected pivots can reveal
   whether the decomposition is robust or sensitive to the pivot
   selection strategy.

3. **Matrices with uniform diagonal:**  If all diagonal elements are
   similar (e.g., a correlation matrix with ones on the diagonal), the
   LDLT strategy has little signal to distinguish important from
   unimportant columns in the first batch.  Column norms may provide
   better discrimination.

### When LDLT Is Better

1. **Speed:**  For typical quantum chemistry matrices, LDLT is 4–5×
   faster than QR due to $O(n)$ per-pivot diagonal updates vs.
   $O(n \cdot n_{\text{new}} \cdot n_D)$ per-batch matrix projections.

2. **Memory:**  LDLT needs only the diagonal vector ($n$ values) plus
   the LDLT column storage ($n \times r$).  QR additionally maintains
   the orthonormal basis $Q_{\text{acc}}$ ($n \times r$), roughly
   doubling the memory footprint.

3. **Indefinite matrices:**  The LDLT diagonal $d_I$ naturally tracks
   the sign of each pivot (positive or negative), giving it direct
   insight into the spectral structure.  QR column norms are always
   non-negative and cannot distinguish positive from negative
   eigenvalue contributions — this information is recovered only in
   Step II.

---

## Comparison with Other Methods

| Method | PSD required? | Cost | Handles complex symmetric? |
|---|---|---|---|
| Pivoted Cholesky | Yes | $O(n \, r^2)$ | Yes (but only PSD) |
| Full eigendecomposition | No | $O(n^3)$ | No (for Hermitian) |
| Full SVD / Takagi | No | $O(n^3)$ | Yes |
| QR-pivoted decompose | No | $O(n \, r^2)$ | Yes |
| **LDLT-pivoted decompose** | **No** | **$O(n \, r^2)$** | **Yes** |

Both the LDLT-pivoted and QR-pivoted methods combine the efficiency of
pivoted Cholesky with the generality of eigendecomposition, handling
indefinite and complex symmetric matrices at $O(n \, r^2)$ cost.
LDLT is the faster of the two due to its cheaper per-pivot updates,
while QR offers a more robust importance metric based on full column
norms.

## Extension to Hermitian Matrices

The entire algorithm can be applied to **Hermitian** matrices
($M = M^\dagger$) by replacing every transpose $(\cdot)^T$ with the
conjugate transpose $(\cdot)^\dagger$ throughout:

- The decomposition becomes $M \approx L \, S \, L^\dagger$
- The Nyström formula becomes $M \approx M_{:,B} \, J^{-1} \, M_{B,:}^\dagger$
- The Takagi factorization is replaced by the standard eigendecomposition
  $J = U \Lambda U^\dagger$
- The orthogonalization step solves $R \, S \, R^\dagger = V \Lambda V^\dagger$

Since Hermitian matrices have real eigenvalues by construction, the
`neg_indices` mechanism works the same way: negative eigenvalues of $J$
are tracked and carried through to the final result.

This makes the LDLT pivoted decomposition a general-purpose tool for
approximate eigendecomposition of any large matrix that is either
symmetric ($M = M^T$) or Hermitian ($M = M^\dagger$), with or without
positive definiteness.

## References

- Folkestad, Koch, et al., "An efficient algorithm for Cholesky
  decomposition of electron repulsion integrals", *J. Chem. Phys.* **150**,
  194112 (2019). — The span-factor batched approach for pivot selection
  originates from this work (for positive definite Cholesky).
- Nyström, "Über die praktische Auflösung von Integralgleichungen mit
  Anwendungen auf Randwertaufgaben", *Acta Math.* **54**, 185–204 (1930).
  — The Nyström approximation formula.
- Takagi, "On an algebraic problem related to an analytic theorem of
  Carathéodory and Fejér and on an allied theorem of Landau",
  *Japanese J. Math.* **1**, 83–93 (1924). — Takagi factorization for
  complex symmetric matrices.
