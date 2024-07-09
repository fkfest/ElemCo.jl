using ElemCo

@testset "N Open-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -54.510599961049
EMP2_test  =  -0.040618302979 + EHF_test
ccmethods = ["uccsd", "udcsd"]
ECC_test =  [-0.051205093083, -0.051642622503]
ECCSD_T_test =  -0.051388091562
EBODCSDfc_test =  -0.051994090819 + EHF_test

fcidump = joinpath(@__DIR__,"files","N_ST1.FCIDUMP")

@ECinit
for (ime,method) in enumerate(ccmethods)
  energies = @cc method 
  @test abs(energies["HF"]-EHF_test) < epsilon
  @test abs(energies["UMP2"]-EMP2_test) < epsilon
  @test abs(last_energy(energies)-EHF_test-ECC_test[ime]) < epsilon
end

#EC.fd = read_fcidump(fcidump)
@set scf pseudo=true
EBOHF = @bouhf
@transform_ints biorthogonal
@test abs(EBOHF-EHF_test) < epsilon
energies = @cc udcsd
@test abs(last_energy(energies)-EHF_test-ECC_test[2]) < epsilon
energies = @cc λuccsd(t)
@test abs(last_energy(energies)-EHF_test-ECCSD_T_test) < epsilon

@freeze_orbs [1]
energies = @cc udcsd
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(last_energy(energies)-EBODCSDfc_test) < epsilon
end
