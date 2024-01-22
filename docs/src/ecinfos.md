# ElemCo.jl global information

```@meta
CurrentModule = ElemCo.ECInfos
```

```@docs
ECInfos
```

## Main structure
```@docs
ECInfo
```

## Exported functions
```@autodocs
Modules = [ECInfos]
Private = false
Order = [:function]
Filter = t -> t ∉ [ElemCo.file_exists, ElemCo.add_file!, ElemCo.copy_file!, ElemCo.delete_file!, ElemCo.delete_files!, ElemCo.delete_temporary_files!]
```

## File management
```@docs
file_exists
add_file!
copy_file!
delete_file!
delete_files!
delete_temporary_files!
```

## Internal functions
```@autodocs
Modules = [ECInfos]
Public = false
Order = [:function]
```

## Abstract types
```@meta
CurrentModule = ElemCo.AbstractEC
```
```@docs
AbstractEC
```

