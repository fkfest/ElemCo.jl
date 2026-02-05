# TREXIO Format Support in ElemCo.jl

ElemCo.jl includes support for the TREXIO format, a standardized HDF5-based format for quantum chemistry data exchange. This enables efficient storage and retrieval of orbitals, amplitudes, and other data structures, facilitating interoperability with other quantum chemistry software.

## API Reference

### Standalone TREXIO Module

```@docs
ElemCo.TREXIO
```

#### TREXIO Core Types and Functions

```@autodocs
Modules = [ElemCo.TREXIO]
Private = false
Order = [:type, :function, :macro, :constant]
```

### TrexioInterface Module

```@docs
ElemCo.TrexioInterface
```

#### ElemCo Integration Functions

```@autodocs
Modules = [ElemCo.TrexioInterface]
Private = false
Order = [:function, :type, :macro, :constant]
```

### Internal Functions

```@autodocs
Modules = [ElemCo.TrexioInterface]
Public = false
Order = [:function, :type, :macro, :constant]
```
