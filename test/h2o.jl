using ElemCo
using ElemCo.ECInfos

@testset "H2O Closed-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -75.6457645933
EMP2_test  =  -0.287815830908
ECCSD_T_test =  -0.329259440500 + EHF_test
EΛCCSD_T_test = -0.326915143863 + EHF_test
EDCSD_test =  -0.328754956597 + EHF_test
EDC_CCSDT_useT3_test = -0.330053201279 + EHF_test
EDC_CCSDT_test = -0.330450039481 + EHF_test

@print_input

fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

EC = ECInfo()
energies = ElemCo.ccdriver(EC, "ccsd(t)"; fcidump)
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(energies["MP2c"]-EMP2_test) < epsilon
@test abs(energies["CCSD(T)"]-ECCSD_T_test) < epsilon

energies = @cc λccsd(t)
@test abs(energies["ΛCCSD(T)"]-EΛCCSD_T_test) < epsilon

energies = ElemCo.ccdriver(EC, "dcsd"; fcidump)
@test abs(last_energy(energies)-EDCSD_test) < epsilon

@set cholesky thr = 1.e-4
@set cc ampsvdtol = 1.e-4
energies = ElemCo.ccdriver(EC, "svd-dc-ccsdt"; fcidump="")
@test abs(last_energy(energies)-EDC_CCSDT_test) < epsilon

@set cc calc_t3_for_decomposition = true
energies = ElemCo.ccdriver(EC, "svd-dc-ccsdt"; fcidump="")
@test abs(last_energy(energies)-EDC_CCSDT_useT3_test) < epsilon

end
