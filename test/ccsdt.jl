using ElemCo
@testset "H2O Closed-Shell CCSDT and DC-CCSDT" begin
epsilon    =      1.e-6
ECCSDT_test    = -0.328471431306
EDCCCSDT_test  = -0.329165986996

fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")
@set cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true

energies = @cc ccsdt
@test abs(last_energy(energies)-energies["HF"]-ECCSDT_test) < epsilon

energies = @cc dc-ccsdt
@test abs(last_energy(energies)-energies["HF"]-EDCCCSDT_test) < epsilon
end

