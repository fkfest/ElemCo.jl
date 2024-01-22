# Tensor tools

```@meta
CurrentModule = ElemCo.TensorTools
```

```@docs
TensorTools
```

## I/O functions

```@docs
save!
load
mmap
newmmap
closemmap
```

## Integral extraction

```@docs
ints1
ints2
```

## Tensor manipulation

```@docs
sqrtinvchol
invchol
rotate_eigenvectors_to_real!
```

## Other exported functions

```@autodocs
Modules = [TensorTools]
Private = false
Order = [:function]
Filter = t -> t ∉ [ElemCo.save!, ElemCo.load, ElemCo.mmap, ElemCo.newmmap, ElemCo.closemmap, ElemCo.ints1, ElemCo.ints2, ElemCo.sqrtinvchol, ElemCo.invchol, ElemCo.rotate_eigenvectors_to_real! ]
```

## Internal functions
```@autodocs
Modules = [TensorTools]
Public = false
Order = [:function]
```
