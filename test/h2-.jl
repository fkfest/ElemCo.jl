using ElemCo

@testset "H2- Empty Subspace Test" begin
epsilon    =  1.e-6
EHF_test   =      -0.424464539648
EMP2_test  =      -0.002487553782 + EHF_test
EDCSD_test =      -0.003327831586 + EHF_test
EΛUCCSD_T_test = -0.003324996250 + EHF_test

geometry="bohr
     H1 0.0 0.0 0.0 
     H2 1.4 0.0 0.0" 

basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-rifit")

@set wf ms2=3 charge=-1
@dfuhf

energies = @cc dcsd
@test abs(energies.HF-EHF_test) < epsilon
@test abs(energies.MP2-EMP2_test) < epsilon
@test abs(energies.UDCSD-EDCSD_test) < epsilon

energies = @cc λuccsd(t)
@test abs(energies.ΛUCCSD_T-EΛUCCSD_T_test) < epsilon
@test abs(energies.T3+energies.ΛUCCSD-EΛUCCSD_T_test) < epsilon

end
