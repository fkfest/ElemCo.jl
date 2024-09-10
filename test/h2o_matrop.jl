using ElemCo
using LinearAlgebra

@testset "H2O Molpro Import Test" begin
epsilon    =   1.e-8

@print_input

geometry = "bohr
O      0.000000000   0.000000000  -0.130186067
H1     0.000000000   1.489124508   1.033245507
H2     0.000000000  -1.489124508   1.033245507"

basis = "v5z"

matropfile = joinpath(@__DIR__,"files","orbs.matrop")
orbs = @import_matrix matropfile
basisset = ElemCo.generate_basis(EC)
overlap = ElemCo.Integrals.overlap(basisset)
unity = orbs'*overlap*orbs
nao = size(unity, 1) 
@test isapprox(unity, Matrix{Float64}(I, nao, nao), atol=epsilon)

end
