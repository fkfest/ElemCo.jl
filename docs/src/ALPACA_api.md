```@meta
CurrentModule = ElemCo.ALPACADecomposition
```

# API Reference

## Matrix-free Interface

```@docs
AbstractALPACAMatrix
DenseALPACAMatrix
SymmetricALPACAMatrix
HermitianALPACAMatrix
column!
row!
elements!
```

## Principal Descriptors

```@docs
AbstractPrincipalDescriptor
PrincipalPairs
PrincipalTriples
principal_pairs
principal_triples
normalize_principal_descriptor
```

## Options and Results

```@docs
ALPACAOptions
ALPACAResult
LLAMAResult
```

## Cache

```@docs
ALPACACache
```

## Core Algorithms

```@docs
alpaca
lpaca
qrdalpaca
llama
```

## Decomposition Extraction

### SVD

```@docs
alpaca_svd
lpaca_svd
qrdalpaca_svd
llama_svd
```

### Eigendecomposition

```@docs
alpaca_eigen
lpaca_eigen
qrdalpaca_eigen
```

### Takagi Decomposition

```@docs
alpaca_takagi
lpaca_takagi
qrdalpaca_takagi
```

### QR Decomposition

```@docs
alpaca_qr
lpaca_qr
qrdalpaca_qr
```

## Internals

These are internal functions documented for developers.  They are not
exported and may change without notice.

```@autodocs
Modules = [ALPACADecomposition]
Public = false
```
