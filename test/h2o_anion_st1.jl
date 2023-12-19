using ElemCo

@testset "H2O Open-Shell ST Test" begin
epsilon    =   1.e-6
EHF_test   = -76.150793582159
EMP2_test  =  -0.073645765995
ECCSD_test =  -0.086578672000
EDCSD_test =  -0.087143018852

fcidump = joinpath(@__DIR__,"H2O_ST1.FCIDUMP")

@opt wf charge=-1
@bohf
@transform_ints biorthogonal
EHF, EMP2, ECCSD = @cc ccsd
@test abs(EHF-EHF_test) < epsilon
@test abs(EMP2-EMP2_test) < epsilon
@test abs(ECCSD-ECCSD_test) < epsilon

EHF, EMP2, EDCSD = @cc dcsd
@test abs(EDCSD-EDCSD_test) < epsilon

end
