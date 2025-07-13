# TREX Format Support in ElemCo.jl

ElemCo.jl includes support for the TREX (Table of Results Exchange) format, a standardized HDF5-based format for quantum chemistry data exchange. This enables efficient storage and retrieval of orbitals, amplitudes, and other data structures, facilitating interoperability with other quantum chemistry software.

## Overview

The TREX format provides:
- Standardized structure for quantum chemistry data
- HDF5-based efficient storage and retrieval
- Cross-platform compatibility
- Support for molecular geometries, orbitals, amplitudes, and other QC data

## Key Features

- **Standardized Format**: Follows the TREX specification for maximum compatibility
- **Efficient Storage**: Built on HDF5 for high-performance I/O
- **Comprehensive Data Support**: Handles molecules, orbitals, CC amplitudes, and more
- **Easy Integration**: Simple macros for common use cases
- **Extensible**: Low-level interface for custom data handling

## Quick Start

### Basic Usage with Macros

```julia
using ElemCo

# Define your molecular system
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = "cc-pVDZ"

# Initialize and run calculation
@ECinit
@dfhf

# Export to TREX format
@write_trex "water_results.h5"

# Read TREX data
data = @read_trex "water_results.h5"
println("Available data: ", keys(data))
```

### Reading TREX Data

```julia
# Read a TREX file
data = @read_trex "calculation_results.h5"

# Access different data sections
if haskey(data, "molecule")
    molecule = data["molecule"]
    println("Number of atoms: ", length(molecule))
end

if haskey(data, "orbitals")
    orbitals = data["orbitals"]
    println("Orbital matrix dimensions: ", size(orbitals))
end

if haskey(data, "amplitudes")
    amplitudes = data["amplitudes"]
    println("Available amplitudes: ", keys(amplitudes))
end
```

## Macro Reference

### `@write_trex`

Export ElemCo data to TREX format.

```julia
@write_trex filename [options...]
```

**Options:**
- `include_orbitals::Bool=true`: Include molecular orbitals and basis sets
- `include_amplitudes::Bool=false`: Include CC amplitudes  
- `include_molecule::Bool=true`: Include molecular geometry

**Note:** When `include_orbitals=true`, basis set information is automatically included.

**Examples:**
```julia
# Export everything (default)
@write_trex "results.h5"

# Export only geometry
@write_trex "geometry.h5" include_orbitals=false

# Export with amplitudes (when available)
@write_trex "full_results.h5" include_amplitudes=true
```

### `@read_trex`

Read data from TREX format file.

```julia
data = @read_trex filename
```

Returns a dictionary with available data sections.

## Low-Level Interface

For more control, use the low-level `TrexInterface` module:

```julia
using ElemCo.TrexInterface

# Create a TREX file
trex = TrexFile("custom.h5", "w")

# Write molecular data
write_trex_molecule(trex, EC.system)

# Write orbital data
orbitals = load(EC, EC.options.wf.orb)
write_trex_orbitals(trex, orbitals)

# Write amplitude data
amplitudes = Dict("t1" => t1_amplitudes, "t2" => t2_amplitudes)
write_trex_amplitudes(trex, amplitudes)

# Close file
close_trex(trex)

# Read data back
trex_read = TrexFile("custom.h5", "r")
molecule = read_trex_molecule(trex_read)
orbitals = read_trex_orbitals(trex_read)
amplitudes = read_trex_amplitudes(trex_read)
close_trex(trex_read)
```

## TREX File Structure

The TREX format organizes data in a standardized HDF5 hierarchy:

```
/trex/
  ├── nucleus/           # Molecular geometry
  │   ├── num            # Number of atoms
  │   ├── charge         # Nuclear charges
  │   ├── coord          # Atomic coordinates
  │   └── label          # Atom labels
  ├── basis/             # Basis set information (automatically included with orbitals)
  │   ├── shell_num      # Number of shells (TREXIO format)
  │   ├── prim_num       # Number of primitives (TREXIO format)
  │   ├── shell_nucleus_index # Nucleus index for each shell (TREXIO format)
  │   ├── shell_ang_mom  # Angular momentum for each shell (TREXIO format)
  │   ├── shell_factor   # Normalization factors (TREXIO format)
  │   ├── shell_range    # Range of primitives for each shell (TREXIO format)
  │   ├── exponent       # Primitive exponents (TREXIO format)
  │   ├── coefficient    # Contraction coefficients (TREXIO format)
  │   └── type           # Basis set name (stored as attribute)
  │   # Legacy format (for backward compatibility):
  │   ├── num            # Number of basis sets (legacy)
  │   ├── nucleus_index  # Atom index for each basis set (legacy)
  │   └── type           # Basis set names (legacy)
  ├── mo/                # Molecular orbitals
  │   ├── num            # Number of MOs
  │   └── coefficient    # MO coefficients
  └── amplitudes/        # CC amplitudes
      ├── t1             # Singles amplitudes
      ├── t2             # Doubles amplitudes
      └── ...            # Other amplitude tensors
```

## Integration with Workflows

### Export After Calculations

```julia
# Standard workflow
@dfhf
@cc ccsd
@write_trex "final_results.h5" include_amplitudes=true
```

### Data Sharing and Collaboration

```julia
# Export for sharing
@write_trex "shared_data.h5"

# Import shared data (in another session/code)
shared = @read_trex "shared_data.h5"
# Use shared["molecule"], shared["orbitals"] as starting point
```

### Restart Calculations

```julia
# Read previous results
previous = @read_trex "checkpoint.h5"

# Use previous orbitals as initial guess
if haskey(previous, "orbitals")
    initial_orbs = previous["orbitals"]
    # Set as initial guess for new calculation
end
```

## Error Handling

The TREX interface includes comprehensive error handling:

```julia
try
    data = @read_trex "nonexistent.h5"
catch e
    println("Error reading TREX file: ", e)
end

# Check file existence
if isfile("results.h5")
    data = @read_trex "results.h5"
else
    println("TREX file not found")
end
```

## TREXIO Standard Compliance

ElemCo.jl's TREX implementation follows the TREXIO standard for basis set storage:

- **TREXIO Format**: When detailed basis set information is available, data is stored using TREXIO-compliant field names (`shell_num`, `prim_num`, `shell_nucleus_index`, `shell_ang_mom`, `shell_factor`, `shell_range`, `exponent`, `coefficient`)
- **Legacy Support**: Maintains backward compatibility with simplified basis set storage (`num`, `nucleus_index`, `type`)
- **Automatic Detection**: The reader automatically detects and handles both formats

The implementation automatically uses the TREXIO-compliant format when basis set details are available from ElemCo calculations, ensuring maximum interoperability with other quantum chemistry codes.

## Performance Considerations

- **Large Data**: HDF5 provides efficient storage for large orbital and amplitude tensors
- **Compression**: Automatic compression for better storage efficiency
- **Memory Usage**: Data is loaded on-demand to minimize memory footprint
- **Parallel I/O**: HDF5 supports parallel access for large-scale calculations

## Compatibility

This implementation follows the TREX format specification version 2.4.0 and is compatible with:
- Other TREX-supporting quantum chemistry codes
- Standard HDF5 tools and libraries
- Cross-platform data exchange (Linux, Windows, macOS)

## Troubleshooting

### Common Issues

**File not found errors:**
```julia
# Always check file existence
if !isfile("results.h5")
    error("TREX file not found")
end
```

**Missing data sections:**
```julia
data = @read_trex "file.h5"
if !haskey(data, "orbitals")
    @warn "No orbital data found in TREX file"
end
```

**Version compatibility:**
Check TREX format version in file attributes if compatibility issues arise.

## References

- [TREX Format Specification](https://trex-coe.github.io/trexio/lib.html)
- [TREX Paper](https://arxiv.org/abs/2302.14793)
- [HDF5 Documentation](https://www.hdfgroup.org/solutions/hdf5/)

## Examples

See `examples/trex_usage.jl` for comprehensive usage examples.