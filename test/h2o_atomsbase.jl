using ElemCo
using AtomsBase, Unitful, UnitfulAtomic

@testset "H2O AtomsBase interface Test" begin

geometry ="bohr
     O      0.000000000    0.000000000   -0.130186067
     H      0.000000000    1.489124508    1.033245507
     H      0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "mpfit"=>"cc-pvdz-mpfit")

@ECinit

fs = FlexibleSystem(EC.system)
ms = ElemCo.MSystems.MSystem(fs, basis)
@test ms ≈ EC.system

end

