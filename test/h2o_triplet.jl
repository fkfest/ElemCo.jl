using ElemCo

@testset "H2O Triplet Open-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -75.62407982361415
EMP2_test  =  -0.22401008330020
EUCCSD_test = -0.27656151568706
ERCCSD_test = -0.27614920708496
ERDCSD_test = -0.28995913689122

fcidump = joinpath(@__DIR__,"H2O.FCIDUMP")

@opt wf ms2=2
EHF, EMP2, EUCCSD = @cc uccsd
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(EUCCSD-EUCCSD_test) < epsilon

EHF, EMP2, ERCCSD = @cc rccsd
@test abs(ERCCSD-ERCCSD_test) < epsilon

EHF, EMP2, ERDCSD = @cc rdcsd
@test abs(ERDCSD-ERDCSD_test) < epsilon

end
