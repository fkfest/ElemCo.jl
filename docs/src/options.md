# Options

```@meta
CurrentModule = ElemCo.ECInfos
```

Options control the behavior of various calculations in ElemCo.jl. They are organized into categories such as `wf` (wavefunction), `scf` (self-consistent field), `cc` (coupled cluster), etc.

## Setting Options

The **recommended way** to set options is using local options within calculation macros:

```julia
@cc ccsd begin
  @set wf charge=-1 ms2=1
  @set cc maxit=100
end
```

This ensures options are automatically restored after the calculation. See the [Calculations](@ref) page for more details.

Options can also be set globally using the [`@set`](@ref ElemCo.@set) macro:

```julia
@set cc maxit=100 thr=1.e-12
```

## Options Structure

```@docs
Options
```

```@autodocs
Modules = [ECInfos]
Pages = ["options.jl"]
Filter = t -> typeof(t) !== DataType || !(t <: ElemCo.ECInfos.Options)
```

