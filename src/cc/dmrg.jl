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
using ..ElemCo.ECInfos
export calc_dmrg, calc_dmrg_dispatch

function calc_dmrg_dispatch(EC)
  ext = Base.get_extension(@__MODULE__, :DmrgExt)
  if isnothing(ext)
    return calc_dmrg()
  else
    return ext.calc_dmrg(EC)::OutDict
  end
end

"""
    calc_dmrg() -> OutDict

  Performm DMRG calculation.
  Requires ITensors.jl and ITensorMPS.jl packages to be loaded.
"""
function calc_dmrg()
  warnerror("For DMRG calculations, please load ITensors.jl and ITensorMPS.jl packages.", true)
  return OutDict()
end

end # module DMRG
