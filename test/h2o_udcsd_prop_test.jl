@testitem "h2o_udcsd_prop" tags=[:properties, :long] begin
using ElemCo
using LinearAlgebra

@testset "ΛUDCSD Test" begin
epsilon    =  1.e-6
EUDCSD_test =     -0.1866586054908987
EUHF1_test  =     -75.63312357707606
EUCCSD1_test =    -0.1706009099216159

geometry="bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"


basis = Dict("ao"=>"cc-pVDZ",
             "jkfit"=>"cc-pvdz-rifit",
             "mpfit"=>"cc-pvdz-rifit")
let
  @opt wf ms2=2
  @dfuhf
  @rotate_orbs 6 7 90
  # occa/occb refer to the full MO space (issue #191): full α=[1,2,3,4,5,7], β=[1,2,3,4];
  # the frozen core (orbital 1) is removed and the rest renumbered to the active dump
  @cc λudcsd occa="1-5+7" occb="1-4"
  U1a, U1b = @loadfile("cc_multipliers_1") 
  U2a, U2b, U2ab = @loadfile("cc_multipliers_2") 
  T1a, T1b = @loadfile("cc_amplitudes_1") 
  T2a, T2b, T2ab = @loadfile("cc_amplitudes_2") 
  D1a, dD1a = ElemCo.calc_1RDM(EC, U1a, U1b, U2a, U2ab, T1a, T2a, T2ab, :α)
  D1b, dD1b = ElemCo.calc_1RDM(EC, U1b, U1a, U2b, U2ab, T1b, T2b, T2ab, :β)
  @test abs(tr(dD1a)) < epsilon
  @test abs(tr(dD1b)) < epsilon
end

end
end
