# TREXIO Functionality Separation

This document describes the separation of TREXIO functionality from ElemCo.jl to enable independent usage.

## Overview

The TREXIO (Table of Results Exchange) functionality has been separated into a standalone module that can be used independently of ElemCo.jl while maintaining full backward compatibility with existing ElemCo workflows.

## Structure

### Standalone TREXIO Module

**Location**: `lib/TREXIO/`

**Purpose**: Provides a complete, independent implementation of the TREXIO format specification.

**Features**:
- Follows TREXIO specification 2.4.0 exactly
- Uses only HDF5 and standard Julia types
- No dependencies on ElemCo.jl internals
- Column-major representation as required by TREXIO
- Supports nucleus, basis, MO, and metadata operations
- Provides both low-level and high-level APIs

**Key Files**:
- `lib/TREXIO/src/TREXIO.jl` - Main module implementation
- `lib/TREXIO/Project.toml` - Package configuration
- `lib/TREXIO/test/runtests.jl` - Comprehensive test suite

### ElemCo Integration Layer

**Location**: `src/interfaces/trexio.jl`

**Purpose**: Provides ElemCo-specific convenience functions and type conversions while using the standalone TREXIO module for core operations.

**Features**:
- Converts between ElemCo types and TREXIO standard formats
- Maintains existing ElemCo TREXIO API for backward compatibility
- Uses standalone TREXIO module for all core I/O operations
- Provides convenience functions for ElemCo workflows

## Usage

### Independent Usage

The TREXIO module can be used completely independently:

```julia
# Add to load path
push!(LOAD_PATH, "path/to/lib/TREXIO/src")
using TREXIO

# Create TREXIO file
trexio = TREXIO.TrexioFile("data.h5", "w")
TREXIO.write_metadata(trexio, created_by="My Code")

# Write nuclear data
charges = [6.0, 1.0, 1.0, 1.0, 1.0]  # CH4
coords = [0.0 1.0 -1.0  0.0  0.0;     # x coordinates
          0.0 0.0  0.0  1.0 -1.0;     # y coordinates
          0.0 0.0  0.0  0.0  0.0]     # z coordinates  
labels = ["C1", "H1", "H2", "H3", "H4"]

TREXIO.write_nucleus(trexio, charges, coords, labels)
TREXIO.close_trexio(trexio)

# Read data back
data = TREXIO.read_trexio_file("data.h5")
```

### ElemCo Integration

The existing ElemCo TREXIO API continues to work unchanged:

```julia
using ElemCo

geometry = "C 0.0 0.0 0.0\nH 1.0 0.0 0.0"
basis = "sto-3g"
@ECinit
@dfhf

# Export using existing API
@write_trexio "molecule.h5"

# Read using existing API  
data = @read_trexio "molecule.h5"
```

## API Reference

### Standalone TREXIO Module

#### Core Types

- `TrexioFile`: Represents a TREXIO file handle
- `open_trexio(trexio)`: Open a TREXIO file
- `close_trexio(trexio)`: Close a TREXIO file

#### Data Writing Functions

- `write_metadata(trexio; format_version, created_by)`: Write file metadata
- `write_nucleus(trexio, charges, coords, labels)`: Write nuclear data
- `write_basis(trexio, shell_data...)`: Write basis set data (TREXIO standard)
- `write_mo(trexio, coefficients; orbital_type, spin)`: Write molecular orbitals

#### Data Reading Functions

- `read_metadata(trexio)`: Read file metadata
- `read_nucleus(trexio)`: Read nuclear data
- `read_basis(trexio)`: Read basis set data
- `read_mo(trexio)`: Read molecular orbitals

#### High-Level Functions

- `create_trexio_file(filename, nucleus_data, basis_data, mo_data)`: Create complete file
- `read_trexio_file(filename)`: Read all available data

### ElemCo Integration Layer

#### Conversion Functions

- `write_trexio_molecule(trexio, system)`: Convert MSystem to TREXIO nucleus
- `read_trexio_molecule(trexio)`: Convert TREXIO nucleus to MSystem
- `write_trexio_orbitals(trexio, orbitals)`: Convert SpinMatrix to TREXIO MO
- `read_trexio_orbitals(trexio)`: Convert TREXIO MO to SpinMatrix

#### High-Level ElemCo Functions

- `write_trexio(filename, EC; options...)`: Export ElemCo calculation
- `read_trexio(filename)`: Import TREXIO data for ElemCo

## TREXIO Specification Compliance

The standalone module strictly follows the TREXIO specification:

- **HDF5 backend**: Uses HDF5 for efficient, cross-platform storage
- **Column-major arrays**: All coordinate and orbital data in column-major format
- **Standard groups**: nucleus, basis, mo groups follow TREXIO schema
- **Metadata**: Standard format_version, created_by, created_date attributes
- **Units**: Proper unit specification (default: bohr for coordinates)

## Testing

### Standalone Module Tests

```bash
cd lib/TREXIO
julia --project=. test/runtests.jl
```

Tests cover:
- File operations (create, open, close)
- Metadata read/write
- Nuclear data I/O
- Basis set data I/O  
- Molecular orbital I/O
- High-level API functions
- Error handling

### ElemCo Integration Tests

The existing ElemCo TREXIO tests continue to work and verify backward compatibility.

## Benefits

1. **Independence**: TREXIO functionality can be used without ElemCo.jl
2. **Standardization**: Strict adherence to TREXIO specification
3. **Interoperability**: Compatible with other TREXIO-supporting codes
4. **Maintainability**: Clear separation of concerns
5. **Backward Compatibility**: Existing ElemCo code continues to work
6. **Extensibility**: Easy to add new TREXIO features

## Migration Guide

### For Independent TREXIO Usage

No migration needed - the module is designed for independent use from the start.

### For Existing ElemCo Users

No migration needed - all existing APIs continue to work unchanged. The separation is transparent to end users.

### For Developers

- Core TREXIO operations should use the standalone module
- ElemCo-specific functionality should use the integration layer
- New TREXIO features should be implemented in the standalone module first
- ElemCo integration should be added as needed

## Future Enhancements

The standalone TREXIO module provides a foundation for:

1. **Full basis set support**: Complete TREXIO-compliant basis set I/O
2. **Additional data types**: Integrals, densities, properties
3. **Performance optimizations**: Chunked I/O for large datasets
4. **Validation tools**: TREXIO file validation and schema checking
5. **Conversion utilities**: Tools for converting between formats

## Example Files

- `examples/standalone_trexio_example.jl`: Demonstrates independent usage
- `examples/trexio_usage.jl`: Shows ElemCo integration
- `test/trexio_test.jl`: ElemCo integration tests
- `lib/TREXIO/test/runtests.jl`: Standalone module tests