using ElemCo

@testset "Positron DF-HF Closed-Shell Test" begin
epsilon    =  1.e-6
EHF_H_test     =     -0.660770127162853
Ecorr_MP2_H_test     =     -0.03503188483173242
EHF_LiH_test   =      -7.991998796257979

xyz_H="bohr
H 0.000000 0.000000 0.000000"

xyz_LiH="bohr
            Li 0.000000 0.000000 0.000000
            H  0.000000 0.000000 3.0196"


basis_H = Dict("ao"=>"aug-cc-pVDZ",
     "jkfit"=>"def2-universal-jkfit",
     "mpfit"=>"cc-pvtz-rifit")
basis_LiH = Dict("ao"=>"aug-cc-pVqZ",
     "jkfit"=>"def2-universal-jkfit",
     "mpfit"=>"cc-pvqz-rifit")

EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_H,basis_H))
@set wf charge=-1
@set wf npositron=1
E_H=@dfhf
@dfints
E_H = @cc MP2
@show E_H
@test abs(E_H["HF"]-EHF_H_test) < epsilon
@test abs(E_H["MP2c"]-Ecorr_MP2_H_test) < epsilon
EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_LiH,basis_LiH))
@set wf charge=0
@set wf npositron=1
E_LiH=@dfhf
@test abs(E_LiH["HF"]-EHF_LiH_test) < epsilon
EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_LiH,basis_LiH))
@set wf charge=0
@set wf npositron=1
@set scf direct=1
E_LiH_direct=@dfhf
@test abs(E_LiH_direct["HF"]-EHF_LiH_test) < epsilon

end
