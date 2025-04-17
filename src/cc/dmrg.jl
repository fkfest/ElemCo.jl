"""
    DMRG

Density Matrix Renormalization Group (DMRG) calculations
using `ITensors.jl` package.

The functionality is moved to an extension, i.e., one has to load `ITensors.jl` package 
to use DMRG calculations.

# Example
```julia
using ElemCo, ITensors
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
  warn("For DMRG calculations, please load ITensors.jl package.", true)
end

end # module DMRG
