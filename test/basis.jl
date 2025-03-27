using ElemCo

@testset "DF-HF Closed-Shell Test 2" begin
epsilon    =  1.e-6
EHF_test   =      -76.03518063983151
EMP2_test  =      -76.25688653055902
EDCSD_test =      -76.27021693741234


geometry="
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVTZ; o=avdz; 
             h={! hydrogen             (4s,1p) -> [2s,1p]
                s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000
                c, 1.4, 0.0196850, 0.1379770, 0.4781480, 0.5012400
                c, 4.4, 1.0000000
                p, H , 0.7270000
                c, 1.1, 1.0000000}",
             "jkfit"=>"vtz-jkfit",
             "mpfit"=>"avtz-mpfit")

@dfhf
energies = @cc dcsd
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test abs(energies["DCSD"]-EDCSD_test) < epsilon

end
