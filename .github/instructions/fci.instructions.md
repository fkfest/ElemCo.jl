---
applyTo: 'src/fci/*.jl'
---
# FCI Implementation Instructions

Julia implementation of Full Configuration Interaction (FCI) with Selected CI and Heat-Bath CI extensions.

## Type Stability

**Status:** ✅ All type instabilities resolved (Last checked: 2025-10-13)

**Testing:** Run `julia --project=.. jet_fci.jl` from `profile/` directory to verify type stability.


## HCIContext Implementation

**Status:** ✅ Completed and working correctly

**Key Points:**
- HCIContext is a lightweight alternative to FCIContext for Heat-Bath CI
- Computes diagonal elements ONLY for selected determinants (not full space)
- Uses `compute_diagonal_element()` that replicates DiagonalHEvalData formula
- Handles absorbed integrals correctly (loops over ALL orbitals with occupation factors)

**Files:**
- `src/fci/fci_hci_context.jl` - HCIContext struct definition
- `src/fci/fci_selected_ci.jl` - compute_diagonal_element implementation


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

**Heat-Bath CI:**
```julia
using ElemCo
@print_input
fcidump = "path/to/file.FCIDUMP"
@hci
```


## Key Performance Rules

1. **Type Stability**: All functions must be type-stable (verify with `julia --project=.. jet.jl` from `profile/`)
2. **Direct Matrix Elements**: Selected CI computes H·v directly, never via full-space mapping
3. **Zero Allocations**: Hot paths use pre-allocated buffers (functions end with `!`)
4. **Concrete Types**: Avoid abstract types in struct fields and hot loops


## Algorithm Notes

**Heat-Bath CI:**
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


## Future Improvements

1. **Fock Elements in HBCI Selection** - Use proper Fock matrix elements (h1 + 2e contributions) instead of just h1 for single excitation thresholding
2. **Slater-Condon Screening** - Skip zero matrix elements by checking excitation degree before computation
3. **PT2 Energy Reporting** - Accumulate and report perturbative corrections from external determinants
4. **QFDump Migration** - Use unified QFDump format for better integration with ElemCo.jl

## References

- Knowles & Handy (1984): String-based FCI
- Holmes et al. (2016): Heat-Bath CI
- Davidson (1975): Iterative diagonalization
- Sleijpen & Van der Vorst (2000): Jacobi-Davidson preconditioner
