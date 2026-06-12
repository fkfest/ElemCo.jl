@testitem "h2o_bohf" tags=[:fcidump, :quick] begin
using ElemCo

@testset "H2O Closed-Shell BOHF Test" begin
epsilon    =   1.e-6
EHF_test   = -75.645764593292

fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

@opt scf guess=:hcore temperature_guess=1e9
energies=@bohf

@test abs(energies["HF"]-EHF_test) < epsilon
end
end
