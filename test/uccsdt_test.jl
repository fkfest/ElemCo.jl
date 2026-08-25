@testitem "uccsdt" tags=[:highorder, :long] begin
using ElemCo
@testset "H2O Open-Shell UCCSDT and UDC-CCSDT" begin
epsilon    =      1.e-6
ECCSDT_test    = -0.170787150063
EDCCCSDT_test  = -0.170829455099

fcidump = joinpath(@__DIR__,"files","H2OP_UHF.FCIDUMP")
@set cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true

energies = @cc uccsdt
@test abs(last_energy(energies)-energies["HF"]-ECCSDT_test) < epsilon

energies = @cc udc-ccsdt
@test abs(last_energy(energies)-energies["HF"]-EDCCCSDT_test) < epsilon
end

@testset "CH2 Frozen-Reference UDC-CCSDT" begin
epsilon          =      1.e-6
EFRSDCCCSDT_test = -38.985063888296
EFRTDCCCSDT_test = -39.046441976871

fcidump = joinpath(@__DIR__,"files","CH2.3B1.DZP.ROHF.FCIDUMP")
@set cc use_kext = false calc_d_vvvv = true calc_d_vvvo = true calc_d_vovv = true calc_d_vvoo = true

energies = @cc frs-dc-ccsdt occa="-2.1+1.3" occb="1.1+1.2+1.3"
@test abs(energies["FRS-UDC-CCSDT"]-EFRSDCCCSDT_test) < epsilon

energies = @cc frt-dc-ccsdt occa="-2.1+1.3" occb="1.1+1.2+1.3"
@test abs(energies["FRT-UDC-CCSDT"]-EFRTDCCCSDT_test) < epsilon
end

end
