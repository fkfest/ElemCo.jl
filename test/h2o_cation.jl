using ElemCo
@testset "H2O Open-Shell Test" begin
epsilon    =      1.e-6
EHF_test   =    -75.337282954481
EMP2_test  =     -0.207727619864
ECCD_test =      -0.213495204514
ECCSD_test =     -0.232320803220
EDCSD_test =     -0.243223819179
ERDCSD_test =    -0.241910345272
ECCSD_UHF_test = -0.168407943239

fcidump = joinpath(@__DIR__,"H2O_CATION.FCIDUMP")

EC = ElemCo.ECInfo()
EHF, EMP2, ECCD = ECdriver(EC, "uccd"; fcidump)
@test abs(ECCD-ECCD_test) < epsilon
EHF, EMP2, ECCSD = ECdriver(EC, "uccsd"; fcidump)
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(ECCSD-ECCSD_test) < epsilon

EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump)
@test abs(EDCSD-EDCSD_test) < epsilon

EHF, EMP2, ERDCSD = @cc rdcsd
@test abs(ERDCSD-ERDCSD_test) < epsilon

fcidump = joinpath(@__DIR__,"H2OP_UHF.FCIDUMP")
EHF, EMP2, ECCSD = ECdriver(EC, "uccsd"; fcidump)
@test abs(ECCSD-ECCSD_UHF_test) < epsilon

@opt cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true triangular_kext = false 
EHF, EMP2, ECCSD_nokext = ECdriver(EC, "uccsd"; fcidump)
@test abs(ECCSD_nokext-ECCSD_UHF_test) < epsilon

end
