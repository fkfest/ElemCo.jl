@testitem "h2o_bohf" tags=[:fcidump, :quick] begin
using ElemCo
using LinearAlgebra

@testset "H2O Closed-Shell BOHF Test" begin
epsilon    =   1.e-6
EHF_test   = -75.645764593292

fcidump = joinpath(@__DIR__,"files","H2O.FCIDUMP")

@opt scf guess=:hcore temperature_guess=1e9
energies=@bohf

@test abs(energies["HF"]-EHF_test) < epsilon

# the rotation dump stores the Fock in the original MO basis; its eigenvalues are the BO-HF
# orbital energies (the rotation diagonalizes it)
F = ElemCo.Wavefunctions.fetch_ao_fock(EC)
@test !isnothing(F)
eps = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
@test maximum(abs.(sort(real.(eigvals(F[1]))) - sort(eps[1]))) < 1.e-4
end
end
