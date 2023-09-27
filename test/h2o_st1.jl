using ElemCo
using ElemCo.BOHF

@testset "H2O Closed-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -76.298014304953
EMP2_test  =  -0.069545740864
ECCSD_test =  -0.082041632192
EDCSD_test =  -0.082498102641
EBODCSD_test =  -0.0852347071335213

fcidump = joinpath(@__DIR__,"H2O_ST1.FCIDUMP")

EC = ElemCo.ECInfo()
EHF, EMP2, ECCSD = ECdriver(EC, "ccsd"; fcidump)
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(ECCSD-ECCSD_test) < epsilon

EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump)
@test abs(EDCSD-EDCSD_test) < epsilon

#EC.fd = read_fcidump(fcidump)
EBOHF = bohf(EC)
CMOr = @loadfile EC.options.wf.orb
CMOl = @loadfile EC.options.wf.orb*EC.options.wf.left
ElemCo.transform_fcidump(EC.fd, CMOl, CMOr)
EHF, EMP2, EDCSD = ECdriver(EC, "dcsd"; fcidump="")
@test abs(EBOHF-EHF) < epsilon
@test abs(EDCSD-EBODCSD_test) < epsilon


end
