using ElemCo

@testset "DF-MCSCF HIGH-SPIN OPEN SHELL Test" begin
epsilon    =   1.e-8
EMCSCF_test   = -75.39523234954376

geometry="bohr
    O      0.000000000    0.000000000   -0.130186067
    H1     0.000000000    1.489124508    1.033245507
    H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
            "jkfit"=>"cc-pvtz-jkfit",
            "mp2fit"=>"cc-pvdz-rifit")

@opt wf ms2=2 charge=-2

E,cMO =  ElemCo.dfmcscf(EC,direct=false)

@test abs(E-EMCSCF_test) < epsilon

end
