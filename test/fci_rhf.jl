using Test
using ElemCo

@testset "Full CI - RHF Systems" begin
  println("\n=== Testing Full CI with RHF ===")
    
  # Reference energies (approximate, adjust based on actual calculations)
  epsilon = 1.e-6
  E_FCI_test = -76.120273657749
  omega1_test = 0.26700652210
  omega2_test = 0.29442080996
  
  @print_input
  
  geometry = "
             O      0.000000000    0.000000000   -0.130186067
             H1     0.000000000    1.489124508    1.033245507
             H2     0.000000000   -1.489124508    1.033245507"
  basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
  
  @dfhf
  energies = @fci
  
  @test haskey(energies, "FCI")
  E_fci = energies["FCI"]
  
  println("FCI Energy (H2O, RHF): $E_fci")
  @test abs(E_fci - E_FCI_test) < epsilon

  @set fci nstates=3
  energies = @fci

  @test abs(energies["FCI"] - E_FCI_test) < epsilon
  @test abs(energies["ω1"] - omega1_test) < epsilon
  @test abs(energies["ω2"] - omega2_test) < epsilon
      
  @set fci nstates=1  # Reset to single root 
  @set fci conv_tol=1.e-8
  @set fci max_iter=50
  @set fci pspace_selection_method=:hci
  energies = @fci
      
  @test abs(energies["FCI"] - E_FCI_test) < epsilon
      
  
  println("\n=== Full CI RHF Tests Passed ===\n")
end
