using ElemCo
@testset "H2O Open-Shell UCCSDT and UDC-CCSDT" begin
epsilon    =      1.e-6
ECCSDT_test    = -0.170787150063
EDCCCSDT_test  = -0.170829455099

fcidump = joinpath(@__DIR__,"H2OP_UHF.FCIDUMP")
@set cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true triangular_kext = false 

energies = @cc uccsdt
@test abs(last(energies)-energies.HF-ECCSDT_test) < epsilon

energies = @cc udc-ccsdt
@test abs(last(energies)-energies.HF-EDCCCSDT_test) < epsilon
end

