using ElemCo

@testset "H2O Molpro XML Import Test" begin
epsilon    =   1.e-8

@print_input

@molpro_input

energy = @cc dcsd

@test isapprox(last_energy(energy), -76.240578778725, atol=epsilon)

basis = Dict("mpfit" => "vqz-mpfit")

@molpro_input

energy = @cc dcsd

@molpro_output energy

@test isapprox(last_energy(energy), -76.240437440160, atol=epsilon)

rm(MI["ECVARIABLES"])

end
