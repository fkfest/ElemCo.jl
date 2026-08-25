# ElemCo.jl Copilot Instructions

This repository contains **ElemCo.jl** (*elemcoil*), a Julia package for electronic structure calculations and quantum chemistry computations, with a focus on coupled cluster methods and electron correlation techniques.

## Project Overview

ElemCo.jl is a scientific computing package that provides:
- Coupled cluster methods (CCSD, DCSD, CCSDT, etc.)
- Density-fitted Hartree-Fock (DF-HF) calculations
- Post-Hartree-Fock methods including MP2
- Quantum chemistry interfaces (Molpro, TREXIO, FCIDUMP)
- Advanced tensor operations and orbital tools
- DMRG (Density Matrix Renormalization Group) integration
- Full Configuration Interaction (FCI) with Selected CI and Heat-Bath CI

## Architecture Overview

### Core Data Flow
1. **System Definition** → 2. **Integrals** → 3. **SCF** → 4. **CC/Post-HF** → 5. **Properties**
   - `MSystems` (molecular geometry, basis sets) → `Integrals` (FCIDUMP or DF) → `DFHF`/`BOHF` (orbitals) → `CoupledCluster` (amplitudes) → `CCTools` (properties)

### Central State Object: `ECInfo`
- **Location**: `src/infos/ecinfos.jl`
- **Purpose**: Global state container for all calculations
- **Key fields**:
  - `EC.system`: Molecular system (`MSystems.MolecularSystem`)
  - `EC.fd`: FCIDUMP integrals (`FciDumps.FciDump`)
  - `EC.options`: All calculation options (nested structure: `scf`, `cc`, `cholesky`, `wf`, etc.)
  - `EC.space`: Orbital space dictionary (`'o'` = occupied, `'v'` = virtual, etc.)
  - **Usage**: Always passed as first argument: `function_name(EC::ECInfo, ...)`

### Module Organization
```
src/
├── ElemCo.jl              # Main module, includes all submodules, defines macros
├── infos/                 # ECInfo, Options, ECMethod
├── system/                # MolecularSystem, BasisSet, Elements
├── integrals/             # FciDump, DumpTools, DFTools
├── scf/                   # DFHF, BOHF, DFMCSCF, OrbTools, FockFactory
├── cc/                    # CoupledCluster, CCTools, Drivers, DMRG
├── fci/                   # FCI, Davidson, Selected CI, Heat-Bath CI
├── solvers/               # DIIS, Davidson
├── tools/                 # TensorTools, QMTensors, MIO, Utils
└── interfaces/            # Molpro, TREXIO, Molden
```

## Code Style and Format

### Julia Conventions

**General Style:**
- Use 2-space indentation consistently
- Follow Julia standard naming conventions:
  - `snake_case` for functions and variables
  - `PascalCase` for types and modules
  - `UPPER_CASE` for constants
- Line length: aim for 80-100 characters, but scientific formulas may exceed this
- Use descriptive variable names, especially for physical quantities

**Function Documentation:**
- Use Julia docstrings with triple quotes `"""`
- Include mathematical formulas using LaTeX notation when relevant
- Document parameters, return values, and provide examples for complex functions
- Use proper LaTeX formatting for equations, e.g., `[``units``]` for units

**Example:**
```julia
"""
    calc_ccsd_energy(EC::ECInfo, T1, T2)

Calculate the CCSD correlation energy using cluster amplitudes.

# Arguments
- `EC::ECInfo`: Electronic structure information object
- `T1`: Single excitation amplitudes [``T_i^a``]
- `T2`: Double excitation amplitudes [``T_{ij}^{ab}``]

# Returns
- `Float64`: CCSD correlation energy in atomic units

# Example
```julia
T1 = load2idx(EC, "T_vo")
T2 = load4idx(EC, "T_vvoo") 
E_corr = calc_ccsd_energy(EC, T1, T2)
```
"""
```

**Macros:**
- The package uses domain-specific macros extensively (e.g., `@dfhf`, `@cc`, `@set`)
- Macro names use lowercase with underscores
- Reserved variable names: `fcidump`, `geometry`, `basis`
- Always include `@print_input` at the beginning of input scripts

**Tensor Operations:**
- Use `@mtensor` macro for tensor contractions (wraps `TensorOperations.@tensor`)
- Follow Einstein summation notation in comments
- Include LaTeX expressions for tensor equations in comments
- Example: `# R_e^m += D_{id}^{el} (\\hat v_{ml}^{di}-\\hat v_{lm}^{di})`
- Example code: `@mtensor A[p,q,L] = B[p,r,L] * C[r,q]`
- Use `@mview` for memory-efficient array views (based on `StridedViews`)

### Domain-Specific Language (DSL)
ElemCo uses **macro-based DSL** for user-facing API (see `src/ElemCo.jl` lines 250-600):

**Key Macros:**
- `@ECinit` / `@tryECinit` - Initialize `EC::ECInfo` global state
- `@print_input` - prints input for reproducibility
- `@dfhf` / `@dfuhf` / `@dfmcscf` - Run SCF calculations, store orbitals in `EC.options.wf.orb`
- `@cc <method>` - Run CC calculations (automatically calls `@dfints` if needed)
- `@dfcc <method>` - Run CC with on-the-fly density fitting
- `@set <opt> <key>=<val>` - Set options (e.g., `@set scf thr=1.e-14 maxit=100`)
- `@fci` - Run FCI calculation

**Reserved Variables:**
- `fcidump::String` - Path to FCIDUMP file
- `geometry::String` - Molecular geometry (Cartesian or Z-matrix)
- `basis::String` or `Dict` - Basis set specification
- `EC::ECInfo` - Global state object (auto-created by macros)

**Example Input Pattern:**
```julia
using ElemCo
@print_input              # Always first!

geometry = "O 0 0 0; H 0 0 1.8; H 0 1.8 0"
basis = "cc-pVDZ"
@dfhf                     # Run HF, stores orbitals
@cc dcsd                  # Run DCSD using stored orbitals
```

### Module Structure

**File Organization:**
- Main module: `src/ElemCo.jl`
- Submodules organized by functionality:
  - `cc/` - Coupled cluster methods (see `drivers.jl` for entry points)
  - `scf/` - Self-consistent field methods
  - `integrals/` - Integral handling and transformations
  - `system/` - Molecular systems and basis sets
  - `tools/` - Utilities and tensor operations
  - `interfaces/` - External program interfaces (Molpro, TREXIO)
  - `fci/` - Full CI implementation

**Constants and Physical Units:**
- Define physical constants in `Constants` module
- Include proper units in docstrings: `[``m~s^{-1}``]`
- Use atomic units as the default unit system

## Development Guidelines

### Testing
- Tests are located in `test/` directory
- Use descriptive test names that indicate the method being tested
- Test files follow pattern: `method_system.jl` (e.g., `h2o_dcsd.jl`)
- Tests use `@testset` with energy comparisons and numerical thresholds
- Standard test pattern:
```julia
@testset "System Method Test" begin
    epsilon = 1.e-6
    E_ref = -75.6457645933  # Reference energy
    
    @print_input
    fcidump = joinpath(@__DIR__, "files", "system.FCIDUMP")
    energies = @cc method
    
    @test abs(energies["METHOD"] - E_ref) < epsilon
end
```
- Run quick tests with: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Run all tests with: `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["all"])'`
- Select a subset by tag or item name, e.g.: `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["df"])'` (or `["h2o","complex"]`)

### Dependencies
- Minimize external dependencies
- Use `LinearAlgebra`, `TensorOperations` for mathematical operations
- HDF5 for data storage, XML for configuration files
- `libcint_jll` for integral calculations

### Performance Considerations
- **Type stability is essential**: All performance-critical functions must be type-stable
  - Ensure return types are inferrable from input types at compile time
  - Use `@code_warntype` to check for type instabilities
  - Avoid abstract types in struct fields (use parametric types or concrete types)
  - Use `Val{N}` for dimension-dependent code (see `mioload` in `src/tools/myio.jl`)
  - Example: Return `Array{Float64,N}` not `Array` from functions
- Use in-place operations where possible (functions ending with `!`)
- Leverage BLAS operations via `LinearAlgebra`
- Memory management is crucial for large tensor operations
- Use `load4idx()` (`load3idx`, `load2idx`, etc) and `save4idx()` (`save3idx()`, `save2idx()`, etc) for tensor disk I/O

### Type Stability Checking with JET

Use **JET.jl** for comprehensive type stability analysis. The analysis script is in `profile/jet.jl`.

**Running JET Analysis:**
```bash
julia --project=. profile/jet.jl
```

**How it works:**
- Uses `@report_opt` to analyze optimization issues and runtime dispatches
- Targets all ElemCo modules to catch type instabilities across the codebase
- Reports "possible errors" which are typically runtime dispatches due to type instability

**Fixing Type Instabilities - Key Principles:**

1. **Minimize type annotations**: Do NOT add return type annotations as a first solution
2. **Find and fix the root cause**: Trace the instability back to its origin
3. **Common root causes:**
   - Functions returning abstract types (e.g., `Matrix{T} where T` instead of `Matrix{Float64}`)
   - Closures with `f::Function` abstract type preventing inference
   - Type-unstable data flowing through multiple function calls
   - Reading data from files/interfaces without concrete type conversion

4. **Fixing strategies (in order of preference):**
   - Fix the source function to return concrete types
   - Add explicit type conversion at data boundaries (e.g., `Matrix{Float64}(data)`)
   - Use concrete types in struct fields
   - Only as last resort: add return type annotations

5. **Known acceptable instabilities:**
   - `kwcall` runtime dispatch (inherent Julia limitation with keyword arguments)
   - Dynamic dispatch in initialization code (not performance-critical)

**Example - Fixing at the source:**
```julia
# BAD: Adding annotation to hide the problem
function process_data(data)::Matrix{Float64}
    return compute(data)  # compute() returns abstract type
end

# GOOD: Fix compute() to return concrete type
function compute(data)
    result = some_operation(data)
    return Matrix{Float64}(result)  # Convert at the source
end

function process_data(data)
    return compute(data)  # Now type-stable without annotation
end
```

**After making changes:**
- Re-run `profile/jet.jl` to verify improvements
- Run test suite to ensure correctness: `julia --project=. -e 'using Pkg; Pkg.test()'`

## Quantum Chemistry Specifics

### Mathematical Notation
- Use standard quantum chemistry notation
- Greek letters for spin indices (α, β)
- Latin letters for spatial orbitals (i,j,k... occupied, a,b,c... virtual)
- Tensor indices follow physicist's notation

### Method Implementations
- Coupled cluster amplitudes: T1 (singles), T2 (doubles), T3 (triples)
- Density matrices: 1RDM, 2RDM with proper symmetry
- Fock matrices with density fitting approximations
- Molecular orbital coefficients and transformations

### Input File Format
Standard input files should start with:
```julia
using ElemCo
@print_input

# Option 1: Using FCIDUMP file
fcidump = "path/to/file.FCIDUMP"
@cc dcsd

# Option 2: Define molecular system
geometry = "H 0.0 0.0 0.0
           H 0.0 0.0 1.0"
basis = "cc-pVDZ"
@dfhf
@cc dcsd

# Option 3: Using ccdriver function
EC = ECInfo()
energies = ElemCo.ccdriver(EC, "ccsd(t)"; fcidump="file.FCIDUMP")
```

**Key Input Patterns:**
- Always include `@print_input` for reproducibility
- Use `fcidump`, `geometry`, `basis` as reserved variable names
- Methods: `dcsd`, `ccsd`, `ccsd(t)`, `λccsd(t)`, `mp2`, etc.
- Options set via `@set` macro: `@set scf maxit=50`
- Occupation can be specified: `@cc dcsd occa="1-5" occb="1-4"`

## Common Patterns

### Error Handling
- Use Julia's exception system
- Provide meaningful error messages with context
- Include suggestions for fixing common user errors

### Logging and Output
- Use `println()` for user-facing output
- Include timing information for expensive operations
- Progress reporting for iterative methods
- ASCII art headers for major calculation sections

### Memory Management
- Use `NOTHING4idx` (`NOTHING3idx`, `NOTHING2idx`, etc) constant for clearing large tensors
- Implement scratch directory management (default: system temp dir + "elemcojlscr")
- Handle temporary files appropriately
- Memory-mapped I/O for large tensors via `MIO` module (`miosave`, `mioload`, `miommap`)

## Contributing Guidelines

### Code Reviews
- Ensure mathematical correctness of implementations
- Verify numerical stability and convergence
- Check performance implications of changes
- Validate against reference implementations when available

### Documentation
- Update docstrings for any API changes
- Include examples in documentation
- Mathematical derivations should be clear and complete
- Reference papers and methods appropriately

### Backward Compatibility
- Maintain compatibility with existing input files
- Deprecate features gracefully with warnings
- Preserve numerical results for regression testing

Remember: This is scientific software where correctness and numerical stability are paramount. Always validate implementations against established quantum chemistry references and test with multiple molecular systems.
