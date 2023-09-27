using ElemCo
using ElemCo.ECInfos

@testset "H2O Closed-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -75.6457645933
EMP2_test  =  -0.287815830908
ECCSD_T_test =  -0.329259440500
EDCSD_test =  -0.328754956597
EDC_CCSDT_useT3_test = -0.330054209137
EDC_CCSDT_test = -0.33024914396392

fcidump = joinpath(@__DIR__,"H2O.FCIDUMP")

EC = ECInfo()
EHF, EMP2, ECCSD, ET3 = ECdriver(EC, "ccsd(t)"; fcidump)
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(ECCSD+ET3-ECCSD_T_test) < epsilon

EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump)
@test abs(EDCSD-EDCSD_test) < epsilon

@opt cholesky thr = 1.e-4
@opt cc ampsvdtol = 1.e-2
EHF, EMP2, EDC_CCSDT = ECdriver(EC, "dc-ccsdt"; fcidump="")
@test abs(EDC_CCSDT-EDC_CCSDT_test) < epsilon

@opt cc calc_t3_for_decomposition = true
EHF, EMP2, EDC_CCSDT = ECdriver(EC, "dc-ccsdt"; fcidump="")
@test abs(EDC_CCSDT-EDC_CCSDT_useT3_test) < epsilon

end
