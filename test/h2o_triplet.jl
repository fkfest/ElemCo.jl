using ElemCo

@testset "H2O Triplet Open-Shell Test" begin
epsilon    =   1.e-6
EHF_test   = -75.62407982361415
EMP2_test  =  -0.22401008330020 + EHF_test
EUCCSD_test = -0.27656151568706 + EHF_test
EUCCSD_T_test = -0.2953883330999 + EHF_test
ERCCSD_test = -0.27614920708496 + EHF_test
ERDCSD_test = -0.28995913689122 + EHF_test
EΛUCCSD_T_test = -0.2903721324779 + EHF_test

fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

@opt wf ms2=2
energies = @cc uccsd(t)
@test abs(energies.HF-EHF_test) < epsilon
@test abs(energies.MP2-EMP2_test) < epsilon
@test abs(energies.UCCSD-EUCCSD_test) < epsilon
@test abs(energies.UCCSD_T-EUCCSD_T_test) < epsilon

energies = @cc rccsd
@test abs(energies.RCCSD-ERCCSD_test) < epsilon

energies = @cc rdcsd
@test abs(energies.RDCSD-ERDCSD_test) < epsilon

energies = @cc λuccsd(t)
@test abs(last(energies)-EΛUCCSD_T_test) < epsilon

end
