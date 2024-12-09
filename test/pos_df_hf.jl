using ElemCo

@testset "Positron DF-HF Closed-Shell Test" begin
epsilon    =  1.e-6
EHF_H_test     =     -0.6122401829972024
EHF_LiH_test   =      -7.959702291033732

xyz_H="bohr
H 0.000000 0.000000 0.000000"

xyz_LiH="bohr
            Li 0.000000 0.000000 0.000000
            H  0.000000 0.000000 3.0196"


basis = Dict("ao"=>"cc-pVDZ",
     "jkfit"=>"def2-universal-jkfit",
     "mp2fit"=>"cc-pvdz-rifit")

EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_H,basis))
@set wf charge=-1
E_H=@dfhf_positron
@test abs(E_H["HF"]-EHF_H_test) < epsilon
EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(xyz_LiH,basis))
@set wf charge=0
E_LiH=@dfhf_positron
@test abs(E_LiH["HF"]-EHF_LiH_test) < epsilon

end
