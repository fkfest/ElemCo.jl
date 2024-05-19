using ElemCo

@testset "SVD-DCSD Closed-Shell Test" begin
epsilon    =  1.e-6
EHF_test   =      -76.02145513971418
ESVDDCSD_test =   -0.220331906783324 + EHF_test
ESVDDCD_test =   -76.240776272982
ESVDDCSD_px_test = -0.220423917054 + EHF_test
ESVDDCSD_ft_test =-0.219961375476643 + EHF_test
ESVDDCSD_ft0_test =-0.220062044710 + EHF_test
ESVDDCSD_ft1_test =-0.220069358696 + EHF_test
ESVDDCSD_ft2_test =-0.219961375476643 + EHF_test
ESVDDCSD_ft3_test =-0.220069165297 + EHF_test
ESVDDCSD_fd_test =-0.220238449366 + EHF_test

geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-rifit")

@dfhf

energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_test) < epsilon

energies = @dfcc svd-dcd
@test abs(energies.SVD_DCD-ESVDDCD_test) < epsilon

@opt cc use_projx=true
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_px_test) < epsilon

@opt cc use_projx=false
@set cc use_full_t2=true
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_ft_test) < epsilon

@opt cc project_vovo_t2=0
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_ft0_test) < epsilon

@opt cc project_vovo_t2=1
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_ft1_test) < epsilon

@opt cc project_vovo_t2=2
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_ft2_test) < epsilon

@opt cc project_vovo_t2=3
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_ft3_test) < epsilon

@opt cc ampsvdtol=1.e-3
@opt cc decompose_full_doubles=true
energies = @dfcc svd-dcsd
@test abs(energies.SVD_DCSD-ESVDDCSD_fd_test) < epsilon

end
