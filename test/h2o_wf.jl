using ElemCo
using LinearAlgebra

@testset "H2O WF Test" begin
epsilon    =   1.e-6
EHF_test   = -75.6457645933
EMP2_test  =  -0.287815830908
ECCSD_T_test =  -0.329259440500 + EHF_test
EΛCCSD_T_test = -0.326915143863 + EHF_test
EDCSD_test =  -0.328754956597 + EHF_test
EDC_CCSDT_useT3_test = -0.330053201279 + EHF_test
EDC_CCSDT_test = -0.330450039481 + EHF_test

geometry = "
     O      0.000000000    0.000000000   -0.130186067
     H      0.000000000    1.489124508    1.033245507
     H      0.000000000   -1.489124508    1.033245507
     O1     4.000000000    0.000000000   -0.130186067
     H1     4.000000000    1.489124508    1.033245507
     H1     4.000000000   -1.489124508    1.033245507"

basis = "avdz"

@print_input

@dfhf

orbs = @loadwf

SAO = ElemCo.Integrals.overlap(orbs["basis"])
CMO = orbs["orbitals"][1]
@test norm(CMO'*SAO*CMO - I) < epsilon

@dummy ["O1", "H1", "H1"]
ehf = @dfhf
@copywf "mywf.h5"
@dfints begin
  @set wf freeze_nvirt=10
end
ehf2 = @bohf
@test abs(last_energy(ehf)-last_energy(ehf2)) < epsilon
en1 = @cc dcsd
@usewf "mywf.h5"
@dfints begin
  @set wf freeze_nvirt=10
end
en2 = @cc dcsd
@test abs(last_energy(en1)-last_energy(en2)) < epsilon

rm("mywf.h5")

end
