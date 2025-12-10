"""
    DMRG

Density Matrix Renormalization Group (DMRG) calculations
using `ITensors.jl` package.

The functionality is moved to an extension, i.e., one has to load `ITensors.jl` 
and `ITensorMPS.jl` packages to run DMRG calculations.

# Example
```julia
using ITensors, ITensorMPS
using ElemCo 
fcidump = "h2o.fcidump"
@cc dmrg
```
"""
module DMRG
using ..ElemCo.Utils
export calc_dmrg

"""
    calc_dmrg

  Perform DMRG calculation
"""
function calc_dmrg()
  warnerror("For DMRG calculations, please load ITensors.jl and ITensorMPS.jl packages.", true)
end

end # module DMRG
