using ElemCo

@testset "DMRG Closed-Shell Test" begin
epsilon    =  1.e-6
EDMRG_test =     -75.020033815465 

geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"sto-3g",
             "jkfit"=>"cc-pvdz-jkfit",
             "mpfit"=>"cc-pvdz-mpfit")

@dfhf
energies = @cc dmrg

@test abs(energies["DMRG"]-EDMRG_test) < epsilon

end
