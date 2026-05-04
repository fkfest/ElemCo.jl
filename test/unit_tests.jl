using ElemCo
@testset "parse_orbstring" begin

orbs = "-3.1+-2.2+1.3"
orbsym = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
 2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
 3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
orbs_ref = [1,2,3,53,54,82]
@test ElemCo.parse_orbstring(orbs;orbsym) == orbs_ref

orbs = "-5+7-9"
orbsym = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
 1,1,1,1,1,1,1]
orbs_ref = [1,2,3,4,5,7,8,9]
@test ElemCo.parse_orbstring(orbs;orbsym) == orbs_ref
end

@testset "SpinMatrix copy" begin
restricted = ElemCo.SpinMatrix([1.0 0.0; 0.0 1.0])
restricted_copy = copy(restricted)
@test ElemCo.is_restricted(restricted_copy)
@test restricted_copy.α !== restricted.α

unrestricted = ElemCo.SpinMatrix([1.0 0.0; 0.0 1.0], [0.0 1.0; 1.0 0.0])
unrestricted_copy = copy(unrestricted)
@test !ElemCo.is_restricted(unrestricted_copy)
@test unrestricted_copy.α !== unrestricted.α
@test unrestricted_copy.β !== unrestricted.β
end
