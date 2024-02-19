using ElemCo
@testset "H2O Open-Shell UCCSDT and UDC-CCSDT" begin
epsilon    =      1.e-6
ECCSDT_test    = -0.170787150063
EDCCCSDT_test  = -0.170829455099

EC = ElemCo.ECInfo()

fcidump = joinpath(@__DIR__,"H2OP_UHF.FCIDUMP")
@opt cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true triangular_kext = false 

EHF, EMP2, ECCSDT = ECdriver(EC, "uccsdt"; fcidump)
@test abs(ECCSDT-ECCSDT_test) < epsilon

EHF, EMP2, EDCCCSDT = ECdriver(EC, "udc-ccsdt"; fcidump)
@test abs(EDCCCSDT-EDCCCSDT_test) < epsilon
end

