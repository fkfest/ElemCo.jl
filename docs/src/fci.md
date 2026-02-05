# Full/CIPHI CI

```@meta
CurrentModule = ElemCo.FCI
```

```@docs
FCI
```

## Store and Restart

CIPHI calculations can be stored and restarted using the `wf.store` and `wf.start` options. This is useful for:
- Continuing a calculation with tighter thresholds
- Computing PT2 corrections on a pre-converged determinant space
- Multi-state calculations where states are stored separately

### Storing CIPHI Calculations

```julia
@ciphi begin
  @set ciphi epsilon=1e-3 nstates=2
  @set wf store="my_ciphi.h5"
end
```

This stores the final determinants and CI coefficients to TREXIO files. For multi-state calculations, each state is stored in a separate file (e.g., `my_ciphi.h5`, `my_ciphi_state2.h5`).

### Restarting from Stored Determinants

```julia
@ciphi begin
  @set ciphi epsilon=5e-4
  @set wf start="my_ciphi.h5"
end
```

The restart loads the stored determinants as the initial space and uses the stored CI coefficients as a warm start for the Davidson solver, significantly accelerating convergence.

### PT2-Only Mode

To compute only the PT2 correction without additional variational iterations:

```julia
@ciphi begin
  @set ciphi pt2_only=true
  @set wf start="my_ciphi.h5"
end
```

This is useful when you have a converged determinant space and want to compute or recompute the PT2 correction with different parameters.

## Exported functions

```@autodocs
Modules = [FCI]
Private = false
Order = [:type, :function, :macro]
```

## Internal functions

```@autodocs
Modules = [FCI]
Public = false
Order = [:type, :function, :macro]
```
