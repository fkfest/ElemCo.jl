using ElemCo

@testset "DF-MCSCF HIGH-SPIN OPEN SHELL Test" begin
epsilon    =   1.e-6
EMCSCF_test   = -75.39523234954376

geometry=joinpath(@__DIR__,"files","h2o.xyz")

basis = Dict("ao"=>"cc-pVDZ",
            "jkfit"=>"cc-pvtz-jkfit",
            "mpfit"=>"cc-pvdz-rifit")

@opt wf ms2=2 charge=-2

E = @dfmcscf
@test abs(E-EMCSCF_test) < epsilon

end
