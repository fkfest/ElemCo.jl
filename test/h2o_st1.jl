using ElemCo
using ElemCo.BOHF

@testset "H2O Closed-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -76.298014304953
EMP2_test  =  -0.069545740864
ccmethods = ["ccsd", "dcsd"]
ECC_test =  [-0.082041632192, -0.082498102641]
EBODCSD_test =  -0.0852347071335213
EBODCSDfc_test =  -0.08583428759404194

fcidump = joinpath(@__DIR__,"H2O_ST1.FCIDUMP")

@ECinit
for (ime,method) in enumerate(ccmethods)
  EHF, EMP2, ECC = @cc method 
  @test abs(EHF-EHF_test) < epsilon
  @test abs(EMP2-EMP2_test) < epsilon
  @test abs(ECC-ECC_test[ime]) < epsilon
end

#EC.fd = read_fcidump(fcidump)
EBOHF = bohf(EC)
CMOr = @loadfile EC.options.wf.orb
CMOl = @loadfile EC.options.wf.orb*EC.options.wf.left
ElemCo.transform_fcidump(EC.fd, CMOl, CMOr)
EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump="")
@test abs(EBOHF-EHF) < epsilon
@test abs(EDCSD-EBODCSD_test) < epsilon

@freeze_orbs [1]
EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump="")
@test abs(EBOHF-EHF) < epsilon
@test abs(EDCSD-EBODCSDfc_test) < epsilon
end
