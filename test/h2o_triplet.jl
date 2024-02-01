using ElemCo

@testset "H2O Triplet Open-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -75.62407982361415
EMP2_test  =  -0.22401008330020
EUCCSD_test = -0.27656151568706
EUCCSD_T_test = -0.2953883330999
ERCCSD_test = -0.27614920708496
ERDCSD_test = -0.28995913689122
EΛUCCSD_T_test = -0.2903721324779

fcidump = joinpath(@__DIR__,"H2O.FCIDUMP")

@opt wf ms2=2
EHF, EMP2, EUCCSD, ET3 = @cc uccsd(t)
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(EUCCSD-EUCCSD_test) < epsilon
@test abs(EUCCSD+ET3-EUCCSD_T_test) < epsilon

EHF, EMP2, ERCCSD = @cc rccsd
@test abs(ERCCSD-ERCCSD_test) < epsilon

EHF, EMP2, ERDCSD = @cc rdcsd
@test abs(ERDCSD-ERDCSD_test) < epsilon

EHF, EMP2, EUCCSD, ET3 = @cc λuccsd(t)
@test abs(EUCCSD+ET3-EΛUCCSD_T_test) < epsilon

end
