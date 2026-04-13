using ElemCo

@testset "SVD-DCSD Closed-Shell Test" begin
epsilon    =  1.e-6
EHF_test   =      -76.02145513971418
ESVDDCSD_test =  -0.220422025511 + EHF_test
ESVDDCD_test =   -76.240864671122
ESVDDCSD_px_test =-0.220836904556 + EHF_test
ESVDDCSD_ft_test =-0.220523567255 + EHF_test
ESVDDCSD_ft0_test =-0.220504680811 + EHF_test
ESVDDCSD_ft1_test =-0.220587552033 + EHF_test
ESVDDCSD_ft2_test =-0.220389510164 + EHF_test
ESVDDCSD_ft3_test =-0.220515977661 + EHF_test
ESVDDCSD_fd_test =-0.220051799461 + EHF_test

geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-rifit")

@dfhf

energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_test) < epsilon

energies = @dfcc svd-dcd
@test abs(energies["SVD-DCD"]-ESVDDCD_test) < epsilon

energies = @dfcc svd-dcsd begin
  @set cc use_projx=true decompose_using_mp2=true
end
@test abs(energies["SVD-DCSD"]-ESVDDCSD_px_test) < epsilon

@set cc use_full_t2=true
energies = @dfcc svd-dcsd begin
end
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft_test) < epsilon

@set cc project_vovo_t2=0
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft0_test) < epsilon

@set cc project_vovo_t2=1
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft1_test) < epsilon

@set cc project_vovo_t2=2
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft2_test) < epsilon

@set cc project_vovo_t2=3
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft3_test) < epsilon

@set cc ampsvdtol=1.e-6
@set cc decompose_full_doubles=true
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_fd_test) < epsilon

end
