---
applyTo: 'src/fci/*.jl'
---
# FCI Implementation Instructions

Julia implementation of Full Configuration Interaction (FCI) with Selected CI and Heat-Bath CI extensions.

## Type Stability Status

**Last checked:** 2025-10-13  
**JET Issues:** ✅ **0** (All type instabilities resolved!)

### All Issues Fixed (8/8)
1. ✅ `davidson_fci!` return type annotation added
2. ✅ `hf_energy` captured variable type annotations (3 locations in fci_pspace.jl)
3. ✅ `initial_guesses` type stability in `davidson_selected_ci!` fixed
4. ✅ `diagonalize_selected_space` return type annotation added
5. ✅ `run_heatbath_ci!` return type annotation added
6. ✅ `n_selected` and `n_guess` type annotations in `davidson_selected_ci!`
7. ✅ Variable name fix in `apply_1e_op!` (coeffs_beta, removed invalid free! call)
8. ✅ Replaced `@assert` with `||error()` to avoid AssertionError type instability

**Testing:** Run `julia --project=.. jet_fci.jl` from `profile/` directory

**Result:** The FCI module is now fully type-stable according to JET analysis! 🎉

---

## Running FCI/HCI Calculations

### Using ElemCo Macros

**Full Configuration Interaction (FCI):**
```julia
using ElemCo
@print_input

geometry = "
    O      0.000000000    0.000000000   -0.130186067
    H1     0.000000000    1.489124508    1.033245507
    H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")

@dfhf
@fci
```

**Heat-Bath Configuration Interaction (HBCI/Selected CI):**
```julia
using ElemCo
@print_input

geometry = "
    O      0.000000000    0.000000000   -0.130186067
    H1     0.000000000    1.489124508    1.033245507
    H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")

@dfhf
@hci
```

**Using FCIDUMP file:**
```julia
using ElemCo
@print_input

fcidump = "path/to/file.FCIDUMP"
@fci
# or
@hci
```

**Setting FCI/HCI options: (to be implemented!)**
```julia
@set fci n_roots=3        # Number of states to compute
@set fci max_iter=100     # Maximum Davidson iterations
@set fci threshold=1.e-6  # Energy convergence threshold
@set fci hbci_eps=1.e-4   # HBCI selection threshold
@dfhf
@hci
```

## Development Workflow: Type Stability with JET

**IMPORTANT**: Before implementing any new features or optimizations, ensure type stability!

### Step 1: Run JET Analysis

```bash
cd profile
julia --project=.. jet.jl
```

This runs FCI code through JET's type inference analyzer and reports any type instabilities.

### Step 2: Identify Issues

JET output will show:
```julia
┌ @ ElemCo.FCI src/fci/fci_selected_ci.jl:1234 my_function(arg1, arg2)
│┌ @ ElemCo.FCI src/fci/fci_selected_ci.jl:1240 internal_call(x)
││ runtime dispatch detected: internal_call(::Any)
```

This means `internal_call` is being dispatched dynamically because `x` has type `Any`.

### Step 3: Diagnose with @code_warntype

```julia
julia> using ElemCo, ElemCo.FCI
julia> @code_warntype my_function(arg1, arg2)
```

Look for:
- **Red variables** = type-unstable (bad!)
- **Yellow variables** = Union types (acceptable for initialization, bad in loops)
- **Blue/green variables** = concrete types (good!)

### Step 4: Fix Type Instabilities

Common fixes:
1. **Add type annotations to function signatures**
2. **Use parametric types in structs**
3. **Avoid returning different types from branches**
4. **Use function barriers for unavoidable type instabilities**
5. **Replace abstract types with concrete ones**

### Step 5: Verify Fix

Re-run JET and `@code_warntype` to confirm the issue is resolved.

### Step 6: Run Correctness Tests

```bash
julia --project=. test/runtests.jl
```

Ensure numerical results remain correct after refactoring.

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

### 1. Type Stability (PRIMARY GOAL)

**All FCI code must be fully type-stable.** This is the current priority for development.

#### Testing Type Stability with JET

Use JET.jl to automatically detect type instabilities:

```bash
# Run JET analysis on FCI code
cd profile
julia --project=.. jet.jl
```

The `jet.jl` file is configured to analyze the FCI module:
```julia
@report_opt target_modules=(
    ElemCo.FCI,  # FCI module is included
    ...
) ElemCo.Drivers.fcidriver(EC)  # Tests FCI execution
```

#### Type Stability Requirements

**All performance-critical functions must be type-stable:**
- Return types must be inferrable at compile time
- No `Union` types in hot paths
- Avoid abstract types in struct fields
- Use `Val{N}` for dimension-dependent code
- All loop variables must have concrete types

**Check functions with:**
```julia
using JET
@report_opt my_function(args...)
```

Or manually:
```julia
@code_warntype my_function(args...)
```

#### Common Type Instabilities to Fix

**❌ Bad: Abstract field types**
```julia
struct MyStruct
    data::AbstractArray  # Type-unstable!
end
```

**✅ Good: Parametric types**
```julia
struct MyStruct{T<:AbstractFloat, N}
    data::Array{T, N}
end
```

**❌ Bad: Type-unstable returns**
```julia
function maybe_compute(flag)
    if flag
        return 1.0
    else
        return nothing  # Union{Float64, Nothing}
    end
end
```

**✅ Good: Consistent return type**
```julia
function maybe_compute(flag)
    if flag
        return 1.0
    else
        return 0.0  # Always Float64
    end
end
```

**❌ Bad: Dynamic dispatch in loops**
```julia
function process(items::Vector)  # Abstract!
    for item in items
        compute(item)  # Type-unstable dispatch
    end
end
```

**✅ Good: Concrete types**
```julia
function process(items::Vector{MyType})
    for item in items
        compute(item)  # Type-stable dispatch
    end
end
```

### 2. Selected CI Efficiency: Direct Matrix Elements ONLY

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

### Type Stability Testing with JET (PRIMARY)

**Goal**: Ensure all FCI code is type-stable for optimal performance.

```bash
# Run JET analysis
cd profile
julia --project=.. jet.jl
```

JET will report:
- Type instabilities in function calls
- Runtime dispatch issues
- Potential performance problems
- Optimization opportunities

**Expected output**: No type instability warnings from FCI module functions.

**Interpreting JET output:**
- `✓ No errors found` - Code is type-stable
- `┌ @ ElemCo.FCI file.jl:123 function(...)` - Type instability detected at this location
- Look for red/yellow warnings about dynamic dispatch or type uncertainty

**Fixing issues:**
1. Identify the problematic function from JET output
2. Run `@code_warntype` on that specific function
3. Look for variables marked in red (type-unstable)
4. Add type annotations or refactor to make types concrete
5. Re-run JET to verify fix

### Running Correctness Tests
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
- [ ] **JET reports no type instabilities** (PRIMARY)
- [ ] Energy matches reference within 1e-8 Hartree
- [ ] Multi-state: all states converge and are orthogonal
- [ ] Selected CI: energies independent of P-space size
- [ ] Heat-Bath CI: setup phase provides speedup
- [ ] No performance regressions
- [ ] `@code_warntype` shows no red variables in hot paths

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

## Planned Improvements

### 1. Optimize Fock Element Calculation with Precomputed h1e2 (High Priority)
**Location**: `fci_selected_ci.jl:1095` (single excitations in Heat-Bath CI)

Currently, single excitations use only 1-electron integrals for threshold filtering:
```julia
h_val = is_uhf ? abs(ctx.fcidump.h1a[i+1, a+1]) : abs(ctx.fcidump.h1[i+1, a+1])
```

**Goal**: Use proper Fock matrix elements for more accurate threshold filtering using direct computation with precomputed h1e2 terms.

#### Implementation: Precompute h1e2 and Direct Computation

Pre-compute the two-electron contribution to Fock elements once during setup:
```julia
# Add to HBCISetupData or FCIContext
struct HBCISetupData
  double_excitations::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}
  h_doub_max::Float64
  
  # Precomputed h1e2 terms for efficient Fock calculation
  h1e2::Array{Float64, 3}      # RHF: h1e2[i, p, q] = v_{pi}^{qi} - v_{pi}^{iq}
  h1e2_aa::Array{Float64, 3}   # UHF: alpha-alpha
  h1e2_bb::Array{Float64, 3}   # UHF: beta-beta
  h1e2_ab::Array{Float64, 3}   # UHF: alpha-beta (no exchange)
  
  is_uhf::Bool
end

function setup_hbci!(ctx::FCIContext)::HBCISetupData
  # ... existing double excitation setup ...
  
  # Precompute h1e2 terms
  n_orb = ctx.n_orb
  if ctx.fcidump.is_uhf
    h1e2_aa = zeros(n_orb, n_orb, n_orb)
    h1e2_bb = zeros(n_orb, n_orb, n_orb)
    h1e2_ab = zeros(n_orb, n_orb, n_orb)
    
    for i in 1:n_orb, p in 1:n_orb, q in 1:n_orb
      h1e2_aa[i, p, q] = ctx.fcidump.h2aa[p, i, q, i] - ctx.fcidump.h2aa[p, i, i, q]
      h1e2_bb[i, p, q] = ctx.fcidump.h2bb[p, i, q, i] - ctx.fcidump.h2bb[p, i, i, q]
      h1e2_ab[i, p, q] = ctx.fcidump.h2ab[p, i, q, i]  # No exchange for mixed spin
    end
    
    return HBCISetupData(..., h1e2_aa, h1e2_bb, h1e2_ab, true)
  else
    h1e2 = zeros(n_orb, n_orb, n_orb)
    for i in 1:n_orb, p in 1:n_orb, q in 1:n_orb
      h1e2[i, p, q] = ctx.fcidump.h2[p, i, q, i] - ctx.fcidump.h2[p, i, i, q]
    end
    return HBCISetupData(..., h1e2, ..., false)
  end
end
```

**Direct Fock Element Computation** (no caching needed):
```julia
"""
    compute_fock_ai_direct(ctx, setup_data, occ_alpha, occ_beta, a, i, is_alpha)

Compute Fock matrix element f_ai directly using precomputed h1e2 terms.

f_ai = h1_ai + Σ_j (v_aijj - v_ajji)

# Arguments
- `ctx`: FCIContext with integrals
- `setup_data`: HBCISetupData with precomputed h1e2 arrays
- `occ_alpha`, `occ_beta`: Occupied orbital lists (0-based)
- `a, i`: Virtual and occupied orbital indices (0-based)
- `is_alpha`: true for alpha spin, false for beta
"""
function compute_fock_ai_direct(ctx::FCIContext, 
                                setup_data::HBCISetupData,
                                occ_alpha::Vector{Int}, 
                                occ_beta::Vector{Int},
                                a::Int, i::Int, 
                                is_alpha::Bool)::Float64
    # Convert to 1-based indexing
    a1, i1 = a + 1, i + 1
    
    if setup_data.is_uhf
        h1 = is_alpha ? ctx.fcidump.h1a : ctx.fcidump.h1b
        h1e2_same = is_alpha ? setup_data.h1e2_aa : setup_data.h1e2_bb
        occ_same = is_alpha ? occ_alpha : occ_beta
        occ_opp = is_alpha ? occ_beta : occ_alpha
        
        # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
        fock_val = h1[i1, a1]
        @inbounds @simd for j in occ_same
            fock_val += h1e2_same[j+1, a1, i1]
        end
        @inbounds @simd for j in occ_opp
            fock_val += setup_data.h1e2_ab[j+1, a1, i1]
        end
    else
        # RHF: sum over all occupied orbitals
        fock_val = ctx.fcidump.h1[i1, a1]
        @inbounds @simd for j in occ_alpha
            fock_val += setup_data.h1e2[j+1, a1, i1]
        end
    end
    
    return fock_val
end
```

**Integration into Heat-Bath Single Excitation Selection:**
```julia
# In generate_excitations_with_threshold! (around line 1095)

# Alpha single excitations with proper Fock elements
for i in alpha_occ
    for a in alpha_virt
        h_val = abs(compute_fock_ai_direct(ctx, setup_data, 
                                          alpha_occ, beta_occ,
                                          i, a, true))
        
        if h_val >= epsilon
            new_det = single_excitation_alpha(det, i, a)
            push!(excitations, new_det)
        end
    end
end

# Beta single excitations
for i in beta_occ
    for a in beta_virt
        h_val = abs(compute_fock_ai_direct(ctx, setup_data,
                                          alpha_occ, beta_occ,
                                          i, a, false))
        
        if h_val >= epsilon
            new_det = single_excitation_beta(det, i, a)
            push!(excitations, new_det)
        end
    end
end
```

**Performance Characteristics**:
- **Memory overhead**: `8 × n_orb³` bytes per h1e2 array (e.g., 2 MB for 64 orbitals)
- **Computation**: Direct sum over occupied orbitals O(n_elec) per (i,a) pair
- **Simple and robust**: No caching complexity, no search overhead
- **SIMD-friendly**: Inner loops vectorize well

**Key Advantages**:
- Clean, straightforward implementation
- No caching infrastructure needed
- Memory overhead is modest and predictable
- Easy to debug and maintain
- More accurate threshold filtering than using only h1

### 2. Implement Slater-Condon Rule Screening (Medium Priority)
**Locations**: `fci_selected_ci.jl:1289, 1383`

In Heat-Bath CI probability calculation, currently all determinants are considered:
```julia
for (i, det_I) in enumerate(variational_dets)
  # TODO: skip determinants according to Slater-Condon rules
  c_I = variational_coeffs[i]
  H_IJ = compute_matrix_element_direct(det_I, det_J, ctx)
  sum_term += c_I * H_IJ
end
```

**TODO**: Skip determinants that have zero matrix elements by Slater-Condon rules:
- If det_I and det_J differ by more than 2 orbitals, H_IJ = 0
- Pre-compute excitation degree or use bit operations to quickly determine
- Expected benefit: Reduce O(N_var) loop cost in probability computation

**Implementation approach**:
```julia
# Quick check: excitation degree > 2 → skip
if count_excitation_degree(det_I, det_J) > 2
  continue
end
H_IJ = compute_matrix_element_direct(det_I, det_J, ctx)
```

### 3. Store and Use PT2 Contributions (Low Priority)
**Location**: `fci_selected_ci.jl:1299`

Currently, perturbative energy contributions are computed but not used:
```julia
contrib_J = prob_J * ΔE_J  # Perturbative energy contribution
#TODO: use contrib_J to calculate the PT2 correction later
```

**TODO**: Accumulate and report PT2 corrections:
- Sum contributions from all external determinants
- Report variational + PT2 energy
- Useful for convergence assessment and comparison with CIPSI

**Implementation approach**:
```julia
# In run_heatbath_ci!:
E_PT2 = 0.0
for candidate in candidates
  if !selected
    E_PT2 += candidate.contrib
  end
end
return E_variational, E_PT2
```

### 4. Migrate to QFDump (Low Priority)
**Location**: `fci_dump.jl:9`

Current `FCIDump` struct is FCI-specific:
```julia
mutable struct FCIDump
  # TODO: Use QFDump instead!
```

**TODO**: Migrate to unified `QFDump` from ElemCo main codebase:
- Use standardized integral storage format
- Better integration with rest of ElemCo.jl
- Consistent FCIDUMP reading across modules

## Key References

- Knowles & Handy (1984): String-based FCI algorithm
- Holmes et al. (2016): Heat-Bath CI with perturbative selection
- Davidson (1975): Iterative diagonalization method
- Sleijpen & Van der Vorst (2000): Jacobi-Davidson preconditioner

## Status Summary

**Current Priority**:
- 🎯 **Type stability testing with JET** - Ensure all FCI code is fully type-stable
- Test with `profile/jet.jl` and fix any reported type instabilities
- Goal: Zero type instability warnings from JET for FCI module

**Implemented & Working**:
- ✅ Full CI (RHF and UHF)
- ✅ Multi-state Davidson solver
- ✅ Selected CI with direct matrix elements
- ✅ Heat-Bath CI (RHF and UHF)
- ✅ PT2 perturbative correction
- ✅ P-space initial guess (including HBCI-based)
- ✅ 1-RDM and 2-RDM calculations
- ✅ Jacobi-Davidson preconditioner
- ✅ Previous eigenvector warm start for Davidson

**Performance**:
- Heat-Bath CI: 574x speedup (RHF), 20-26x speedup (UHF) vs naive
- Selected CI: O(N_selected²) scaling, not O(N_full²)
- Davidson: Efficient for N > 100 determinants
- Zero allocations in hot paths achieved
- Warm start improves multi-state convergence

**Type Stability Status**:
- ⏳ Testing in progress with JET
- Run `profile/jet.jl` to check current status
- Fix any reported type instabilities before adding new features
