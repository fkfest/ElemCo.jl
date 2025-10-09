---
applyTo: 'src/fci/*.jl'
---
# FCI Implementation Instructions

Julia implementation of Full Configuration Interaction (FCI) with Selected CI and Heat-Bath CI extensions.

## Core Architecture

### Key Modules
- `fci_main.jl` - Main FCI driver and context
- `fci_davidson.jl` - Davidson iterative diagonalization
- `fci_selected_ci.jl` - Selected CI and Heat-Bath CI
- `fci_pspace.jl` - P-space initial guess generation
- `fci_ops.jl` - Hamiltonian operations and RDMs
- `fci_vec.jl` - CI vector operations
- `fci_types.jl` - Data structures
- `fci_dump.jl` - FCIDUMP I/O

### Central Data Structure: `FCIContext`
```julia
struct FCIContext
  fcidump::FCIDump           # Integrals (h1, h2, or h1a/h1b, h2aa/h2bb/h2ab)
  options::FCIOptions        # All calculation options
  n_elec, n_alpha, n_beta    # Electron counts
  n_orb, n_spin              # Orbital/spin info
  # String addressing tables, P-space data, etc.
end
```

## Critical Performance Requirements

### 1. Selected CI Efficiency: Direct Matrix Elements ONLY

**NEVER map selected determinants to full CI space!**

```julia
# ✅ CORRECT - Direct computation O(N_selected²)
for i in 1:n_selected, j in 1:n_selected
    h_ij = compute_matrix_element_direct(det_i, det_j, ctx)
    result[i] += h_ij * input[j]
end

# ❌ WRONG - Full-space mapping O(N_selected × N_full)
v_full[selected_addresses] = input
contract_hamiltonian!(v_out, v_full, ctx)  # Wasteful!
result = v_out[selected_addresses]
```

**Why**: For 100 selected determinants out of 25,200 total, direct computation is **250x faster**.



### 2. Type Stability

All performance-critical functions must be type-stable:
- Ensure return types are inferrable at compile time
- Use `@code_warntype` to check for type instabilities
- Avoid abstract types in struct fields
- Use `Val{N}` for dimension-dependent code

### 3. Zero Allocations in Hot Paths

Tight loops must not allocate memory:
```julia
# ✅ GOOD - Pre-allocated buffer
function op!(result::Vector{T}, input::Vector{T}, ...) where T
    # Reuse result vector
    return n_filled
end

# ❌ BAD - Allocates in loop
function op(input)
    result = similar(input)  # Allocation!
    return result
end
```

Monitor with: `@allocated contract_hamiltonian!(...)` should be ~0 after warmup.

### 4. Function Conventions

- Mutating functions end with `!` and modify first argument(s)
- Return counts/status, not newly allocated arrays
- Document which arguments are modified
- Use in-place operations where possible

## Heat-Bath CI Algorithm

**Purpose**: Efficiently select important determinants using pre-computed matrix element lists.

### Setup phase (One-time O(M⁴ log M))

Pre-compute and store sorted excitation lists:
```julia
struct HBCISetupData
  double_excitations::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}
  h_doub_max::Float64  # Maximum |H(rs ← pq)|
  is_uhf::Bool
end

function setup_hbci!(ctx::FCIContext)::HBCISetupData
  # For each orbital pair (p,q), store list of (r,s,|H|) sorted descending
  # Enables early termination when |H| < ε
end
```

### Selection phase (Per-iteration O(N_εcon M²))

```julia
function generate_excitations_with_threshold!(
  excitations, det, ctx, setup_data, epsilon
)
  # Use pre-sorted lists: stop when |H| < epsilon
  # Much faster than computing all O(M⁴) excitations
end
```

**Key insight**: Skip small matrix elements WITHOUT computing them.

**Performance**: Heat-Bath CI achieves 574x speedup (RHF) and 20-26x speedup (UHF) vs naive implementation.

## Multi-State Calculations

### Davidson for Multiple Roots

```julia
ctx.options.n_roots = 3  # Request 3 states
energies, states = davidson_fci!(ctx)
```

**Implementation notes**:
- Maintains orthogonality via Gram-Schmidt
- Convergence checked for ALL states
- Subspace expansion adds vectors for all unconverged states
- Initial guess: P-space provides good starting vectors

### Multi-State HBCI

Uses state-maximum selection strategy:
```julia
# Select determinant if important for ANY state
for state in 1:n_roots
    probability = compute_heatbath_probability(det, state)
    max_prob = max(max_prob, probability)
end
```

## Testing

### Running Tests
```bash
julia --project=. test/runtests.jl          # All tests
julia --project=. test/runtests.jl quick    # Quick tests only
```

### Test Pattern
```julia
@testset "FCI Energy Test" begin
    ctx = FCIContext(read_fcidump("test.FCIDUMP"))
    E = run_fci!(ctx)
    @test abs(E - E_reference) < 1e-8
end
```

### Validation Checklist
- [ ] Energy matches reference within 1e-8 Hartree
- [ ] Multi-state: all states converge and are orthogonal
- [ ] Selected CI: energies independent of P-space size
- [ ] Heat-Bath CI: setup phase provides speedup
- [ ] No performance regressions

## Code Style

### Formatting
- 2-space indentation (NOT tabs, NOT 4 spaces)
- Maximum 100 characters per line (prefer 80-90)
- Spaces around operators: `a + b` not `a+b`
- One blank line between functions

### Conventions
- Use Julia column-major indexing: `matrix[row, col]`
- Leverage BLAS via `LinearAlgebra` for matrix operations
- Pre-allocate buffers for repeated operations
- Document algorithm sources in comments (e.g., "Holmes et al. 2016")

## Common Patterns

### RDM Calculations
```julia
# 1-RDM: transition density between states
rdm1a, rdm1b = make_1rdms!(ctx, state_i, state_j)

# 2-RDM: for energy verification
rdm2 = make_2rdm!(ctx, state)
E_from_rdm = contract_rdm_with_integrals(rdm2, h1, h2)
```

### Memory Management
```julia
# Memory-mapped I/O for large arrays
save4idx(EC, tensor, "filename")
tensor = load4idx(EC, "filename")

# Clear large tensors
tensor = NOTHING4idx  # Signals GC to reclaim
```

### Integral Access
```julia
# RHF: spatial orbitals
h1 = ctx.fcidump.h1[i, a]
h2 = ctx.fcidump.h2[i, j, a, b]

# UHF: spin-separated
h1a = ctx.fcidump.h1a[i, a]
h2aa = ctx.fcidump.h2aa[i, j, a, b]
h2ab = ctx.fcidump.h2ab[i, j, a, b]
```

## Key References

- Knowles & Handy (1984): String-based FCI algorithm
- Holmes et al. (2016): Heat-Bath CI with perturbative selection
- Davidson (1975): Iterative diagonalization method
- Sleijpen & Van der Vorst (2000): Jacobi-Davidson preconditioner

## Status Summary

**Implemented & Working**:
- ✅ Full CI (RHF and UHF)
- ✅ Multi-state Davidson solver
- ✅ Selected CI with direct matrix elements
- ✅ Heat-Bath CI (RHF and UHF)
- ✅ PT2 perturbative correction
- ✅ P-space initial guess (including HBCI-based)
- ✅ 1-RDM and 2-RDM calculations
- ✅ Jacobi-Davidson preconditioner

**Performance**:
- Heat-Bath CI: 574x speedup (RHF), 20-26x speedup (UHF) vs naive
- Selected CI: O(N_selected²) scaling, not O(N_full²)
- Davidson: Efficient for N > 100 determinants
- Zero allocations in hot paths achieved
