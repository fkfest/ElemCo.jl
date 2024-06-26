using ElemCo

@testset "H2O Open-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -76.150793582159
EMP2_test  =  -0.073645765995 + EHF_test
ECCSD_test =  -0.086578672000 + EHF_test
EDCSD_test =  -0.087143018852 + EHF_test

fcidump = joinpath(@__DIR__,"files","H2O_ST1.FCIDUMP")

@opt wf charge=-1
@bohf
@transform_ints biorthogonal
energies = @cc ccsd
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test abs(last_energy(energies)-ECCSD_test) < epsilon

energies = @cc dcsd
@test abs(last_energy(energies)-EDCSD_test) < epsilon

end
