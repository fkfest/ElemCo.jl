@testset "DF-MCSCF HIGH-SPIN OPEN SHELL Test" begin
epsilon    =   1.e-8
EMCSCF_test   = -75.39523234954376

try
using ElemCo.MSystem
using ElemCo.DFMCSCF
using ElemCo.DfDump
catch
#using .MSystem
end

xyz="bohr
    O      0.000000000    0.000000000   -0.130186067
    H1     0.000000000    1.489124508    1.033245507
    H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
            "jkfit"=>"cc-pvtz-jkfit",
            "mp2fit"=>"cc-pvdz-rifit")

EC = ECInfo(ms=MSys(xyz,basis))
setup!(EC,ms2=2,charge=-2)

E,cMO =  dfmcscf(EC,direct=false)

@test abs(E-EMCSCF_test) < epsilon

end