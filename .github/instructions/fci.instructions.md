---
applyTo: 'src/fci/*.jl'
---
# FCI Implementation Instructions

Julia implementation of Full Configuration Interaction (FCI) with Selected CI and CIPHI (CIΦ - CI via Perturbative and Heat-Bath Iterative selection) extensions.

## Type Stability

**Status:** ✅ All type instabilities resolved (Last checked: 2025-10-13)

**Testing:** Run `julia --project=.. jet_fci.jl` from `profile/` directory to verify type stability.


## CIPHIContext Implementation

**Status:** ✅ Completed and working correctly

**Key Points:**
- CIPHIContext is a lightweight alternative to FCIContext for CIPHI
- Computes diagonal elements ONLY for selected determinants (not full space)
- Uses `compute_diagonal_element()` that replicates DiagonalHEvalData formula
- Handles absorbed integrals correctly (loops over ALL orbitals with occupation factors)

**Files:**
- `src/fci/fci_ciphi_context.jl` - CIPHIContext struct definition
- `src/fci/fci_selected_ci.jl` - compute_diagonal_element implementation
- `src/infos/options.jl` - FCI options (moved from fci_options.jl)


## Configuration

**FCI Options:**
FCI options are stored in the main `Options` structure (`src/infos/options.jl`) and can be set using the `@set` macro:

```julia
@set fci nstates=3           # Number of states to compute
@set fci max_iter=100        # Maximum Davidson iterations  
@set fci threshold=1.e-6     # Energy convergence threshold
@set ciphi epsilon_h=1.e-4   # CIPHI selection threshold
@set fci pspace_selection_method=:ciphi      # Use lightweight CIPHIContext
```

**Integral Storage:**
FCI now uses the unified `QFDump` structure instead of the old FCI-specific `FCIDump` type:
- Integrals still accessed via `ctx.fcidump` (field name unchanged)
- `ctx.fcidump` is now of type `QFDump` (not the old `FCIDump`)
- Better integration with rest of ElemCo.jl
- Consistent with other quantum chemistry modules

## Usage Examples

**Full CI:**
```julia
using ElemCo
@print_input
geometry = "O 0 0 0; H 0 0 1.8; H 0 1.8 0"
basis = "6-31g"
@dfhf
@fci
```

**CIPHI with options:**
```julia
using ElemCo
@print_input
@set ciphi epsilon=1.e-4
@set ciphi nstates=2
fcidump = "path/to/file.FCIDUMP"
@ciphi
```


## Key Performance Rules

1. **Type Stability**: All functions must be type-stable (verify with `julia --project=.. jet.jl` from `profile/`)
2. **Direct Matrix Elements**: Selected CI computes H·v directly, never via full-space mapping
3. **Zero Allocations**: Hot paths use pre-allocated buffers (functions end with `!`)
4. **Concrete Types**: Avoid abstract types in struct fields and hot loops


## Algorithm Notes

**CIPHI:**
- Setup phase: Pre-computes sorted excitation lists for fast threshold-based selection
- Selection: Skips small matrix elements without computing them
- Performance: 574x speedup (RHF), 20-26x speedup (UHF) vs naive

**Multi-State:**
- Davidson solver maintains orthogonality via Gram-Schmidt
- State-maximum selection: Include determinant if important for ANY state

## Testing

```bash
julia --project=. test/runtests.jl          # All tests
julia --project=. test/runtests.jl quick    # Quick tests
cd profile && julia --project=.. jet.jl     # Type stability check
```


## Recent Updates

**October 2025:**
- ✅ FCI options moved to main `Options` structure in `src/infos/options.jl`
- ✅ Migrated to unified `QFDump` type - `ctx.fcidump` now uses `QFDump` instead of old `FCIDump` type
- ✅ Options configurable via standard `@set fci <option>=<value>` macro

## Future Improvements

1. **PT2 Energy Reporting** - Accumulate and report perturbative corrections from external determinants

## References

- Knowles & Handy (1984): String-based FCI
- Holmes et al. (2016): Heat-Bath CI
- Davidson (1975): Iterative diagonalization
- Sleijpen & Van der Vorst (2000): Jacobi-Davidson preconditioner
