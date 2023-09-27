using Test

using LinearAlgebra

try
  EC = ECInfo()
catch
  import ElemCo: ECdriver,@ECinit, @tryECinit, @opt
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
include("2d_cc.jl")

end

@testset verbose = true "DF Calculations" begin

include("df_hf.jl")
include("df_mcscf.jl")

end

@testset verbose = true "Unit tests" begin

include("unit_tests.jl")

end
