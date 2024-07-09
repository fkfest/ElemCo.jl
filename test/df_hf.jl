using ElemCo

@testset "DF-HF Closed-Shell Test" begin
epsilon    =  1.e-6
EHF_test   =      -76.02145513971418
EMP2_test  =      -0.204723138509385 + EHF_test
EDCSD_test =      -0.219150244853825 + EHF_test
ESVDDCSD_test =   -0.220331906783324 + EHF_test
ESVDDCSD_ft_test =-0.219961375476643 + EHF_test
EUHF_test  =      -75.79199546193901

orbital_printout_test = "4:5 orbitals from DFHF orbitals\n4:  0.788(O[1]1p{z})  0.353(H1[2]1s)  0.353(H2[3]1s) -0.290(O[1]2s) -0.170(O[1]3s) \n5:  0.922(O[1]1p{x}) \n"

xyz="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvtz-jkfit",
             "mpfit"=>"cc-pvdz-mpfit")

EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz,basis))

@set scf direct=true
@dfhf
# store orbital printout in a string
original_stdout = stdout
(rd, wr) = redirect_stdout();
@show_orbs 4:5
redirect_stdout(original_stdout)
close(wr)
orbital_printout = read(rd, String)
close(rd)
println(orbital_printout)
@test orbital_printout == orbital_printout_test 
fcidump = "DF_HF_TEST.FCIDUMP"
@set int fcidump=fcidump
@dfints

energies = ElemCo.ccdriver(EC, "dcsd"; fcidump)
@test abs(energies["HF"]-EHF_test) < epsilon
@test abs(energies["MP2"]-EMP2_test) < epsilon
@test abs(energies["DCSD"]-EDCSD_test) < epsilon

rm(fcidump)

energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_test) < epsilon
@set cc use_full_t2=true
energies = @dfcc svd-dcsd
@test abs(energies["SVD-DCSD"]-ESVDDCSD_ft_test) < epsilon

@set scf direct=false
@set wf ms2=2
EUHF = @dfuhf
@test abs(EUHF["HF"]-EUHF_test) < epsilon
end
