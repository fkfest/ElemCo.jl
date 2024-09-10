using ElemCo
using ElemCo.BOHF

@testset "H2O Closed-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -76.298014304953
EMP2_test  =  -0.069545740864 + EHF_test
ccmethods = ["ccsd", "dcsd"]
ECC_test =  [-0.082041632192, -0.082498102641]
EBOHF_test = -76.29524839981325
EBODCSD_test =  -0.0852347071335213 + EBOHF_test
EBODCSDfc_test =  -0.08583428759404194 + EBOHF_test

fcidump = joinpath(@__DIR__,"files","H2O_ST1.FCIDUMP")

@ECinit
for (ime,method) in enumerate(ccmethods)
  energies = @cc method 
  @test abs(energies["HF"]-EHF_test) < epsilon
  @test abs(energies["MP2"]-EMP2_test) < epsilon
  @test abs(last_energy(energies)-EHF_test-ECC_test[ime]) < epsilon
end

#EC.fd = read_fcidump(fcidump)
EBOHF = bohf(EC)
CMOr = @loadfile EC.options.wf.orb
CMOl = @loadfile EC.options.wf.orb*EC.options.wf.left
ElemCo.transform_fcidump(EC.fd, CMOl, CMOr)
energies = @cc dcsd
@test abs(EBOHF-EBOHF_test) < epsilon
@test abs(last_energy(energies)-EBODCSD_test) < epsilon

@freeze_orbs [1]
energies = @cc dcsd
@test abs(energies["HF"]-EBOHF_test) < epsilon
@test abs(last_energy(energies)-EBODCSDfc_test) < epsilon
end
