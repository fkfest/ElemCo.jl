# ALPACADecomposition.jl

```@raw html
<div style="text-align: center; margin-bottom: 1.5em;">
  <img src="assets/logo.png" alt="ALPACA logo" width="160" />
</div>
```

ALPACA (**A**mended **L**ow-rank **P**rincipal-element **A**daptive **C**ross **A**pproximation) is a Julia package for low-rank matrix decomposition with minimal element access.

## What Does ALPACA Do?

Given a large matrix ``\mathbf{A}`` (possibly only accessible through a matrix-free interface), ALPACA finds a low-rank approximation

```math
\mathbf{A} \approx \mathbf{L} \mathbf{R}^\dagger
```

where ``\mathbf{L}`` and ``\mathbf{R}`` are tall-skinny matrices.  For symmetric and Hermitian matrices ``\mathbf{R} = \mathbf{L}`` and a vector of negative indices marks sign changes (``\pm 1`` on the implicit sign diagonal).  The key property is that ALPACA accesses only a small fraction of the entries of ``\mathbf{A}``:  each column and row is fetched **at most once**, and element queries are minimized through the principal descriptor mechanism.

## Algorithm Variants

Three variants with increasing robustness are provided:

| Variant       | Description |
|:--------------|:------------|
| [`lpaca`](@ref)     | Raw factors from the pivot loop (no eigen/SVD amendment) |
| [`alpaca`](@ref)    | Nyström / SVD-finalized factors (eigenvalues below `tol` are truncated) |
| [`qrdalpaca`](@ref) | `alpaca` followed by column-pivoted QR refinement to recover missed columns |

## Supported Matrix Classes

| Matrix class | Symmetry keyword | Factorization | Element type |
|:-------------|:-----------------|:--------------|:-------------|
| Real symmetric | `:symmetric` | ``\mathbf{A} \approx \mathbf{L}\mathbf{L}^\top`` | `Float64` |
| Complex Hermitian | `:hermitian` | ``\mathbf{A} \approx \mathbf{L}\mathbf{L}^\dagger`` | `ComplexF64` |
| Complex symmetric | `:symmetric` | ``\mathbf{A} \approx \mathbf{L}\mathbf{L}^\top`` | `ComplexF64` |
| General | `:general` | ``\mathbf{A} \approx \mathbf{L}\mathbf{R}^\dagger`` | any |

For real symmetric and complex Hermitian cases, the result includes a vector of negative indices to track sign changes (``\pm 1``).

## Quick Start

```julia
using ALPACADecomposition
using LinearAlgebra

# Dense matrix — symmetry is auto-detected
A = Symmetric(randn(100, 100))
result = alpaca(A; tol=1e-10)

# Decomposition extraction
U, S, Vt = alpaca_svd(A; tol=1e-10)
vals, vecs = alpaca_eigen(A; tol=1e-10)
Q, R = alpaca_qr(A; tol=1e-10)
```

For a step-by-step introduction, see the [Tutorial](@ref).

## Installation

ALPACADecomposition is part of the [ElemCo.jl](https://github.com/fkfest/ElemCo.jl) package.  To use it standalone:

```julia
using Pkg
Pkg.develop(path="lib/ALPACADecomposition")
using ALPACADecomposition
```
