@testitem "dummy" tags=[:system, :quick] begin
using ElemCo

@testset "Dummy Closed-Shell Test" begin
epsilon    =  1.e-6
EHFdim_test   =     -151.7448996058252
EMP2dim_test  =     -0.431859473443 + EHFdim_test
EDCSDdim_test =     -0.460041528113 + EHFdim_test
EHF_test      =     -76.02626385089732
EMP2_test     =     -0.210310338556 + EHF_test
EDCSD_test    =     -0.224416212543 + EHF_test

geometry="bohr
     O1     0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H1     0.000000000   -1.489124508    1.033245507
     O2     3.000000000    0.000000000   -0.130186067
     H2     3.000000000    1.489124508    1.033245507
     H2     3.000000000   -1.489124508    1.033245507"


basis = "vdz"
energy = @dfhf
@test abs(energy["HF"]-EHFdim_test) < epsilon
energy = @dfmp2
@test abs(energy["MP2"]-EMP2dim_test) < epsilon
energy = @cc dcsd
@test abs(energy["DCSD"]-EDCSDdim_test) < epsilon
@dummy [4, "H2"]
energy = @dfhf
@test abs(energy["HF"]-EHF_test) < epsilon
energy = @dfmp2
@test abs(energy["MP2"]-EMP2_test) < epsilon
energy = @cc dcsd
@test abs(energy["DCSD"]-EDCSD_test) < epsilon

@dummy []
energy = @dfhf
@test abs(energy["HF"]-EHFdim_test) < epsilon

end
end
