using ElemCo
@testset "H2O Open-Shell Test" begin
epsilon     =      1.e-6
EHF_test    =    -75.337282954481
EMP2_test   =     -0.207727619864 + EHF_test
ECCD_test   =     -0.213495204514 + EHF_test
ECCSD_test  =     -0.232320803220 + EHF_test
EDCSD_test  =     -0.243223819179 + EHF_test
ERDCSD_test =     -0.241910345272 + EHF_test
EUHF_test   =    -75.631764795601
ECCSD_UHF_test = -0.168407943239 + EUHF_test

fcidump = joinpath(@__DIR__,"files","H2O_CATION.FCIDUMP")

energies = @cc uccd
@test abs(last_energy(energies)-ECCD_test) < epsilon
energies = @cc uccsd
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test abs(last_energy(energies)-ECCSD_test) < epsilon

energies = @cc dcsd
@test abs(last_energy(energies)-EDCSD_test) < epsilon

energies = @cc rdcsd
@test abs(last_energy(energies)-ERDCSD_test) < epsilon

fcidump = joinpath(@__DIR__,"files","H2OP_UHF.FCIDUMP")
@ECinit
energies = @cc uccsd
@test abs(energies["HF"]-EUHF_test) < epsilon
@test abs(last_energy(energies)-ECCSD_UHF_test) < epsilon

@set cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true triangular_kext = false 
energies = @cc uccsd
@test abs(last_energy(energies)-ECCSD_UHF_test) < epsilon

end
