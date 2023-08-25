using Test

using LinearAlgebra

try
  EC = ECInfo()
catch
  import ElemCo: ECdriver
  using ElemCo.Utils
  using ElemCo.ECInfos
  using ElemCo.ECMethods
  using ElemCo.TensorTools
  using ElemCo.FockFactory
  using ElemCo.CoupledCluster
  using ElemCo.FciDump
end

@testset verbose = true "FCIDUMP Calculations" begin

include("h2o.jl")
include("h2o_st1.jl")
include("h2o_cation.jl")
include("df_hf.jl")
include("df_mcscf.jl")

end
