@testitem "ciphi_rhf" tags=[:ciphi, :quick] begin
using Test
using ElemCo

@testset "CIPHI - RHF Systems" begin
  println("\n=== Testing CIPHI with RHF ===")
  
  epsilon = 1.e-6
  E_CIPHI_test = -76.119498255053
  E_CIPHI_PT2_test = -76.120246986514
  E_CIPHI_tight_test = -76.120270157954
  E_CIPHI_loose_test = -76.116551675682
  E_CIPHI_ms_test = -76.11876589520229
  omega1_test = 0.26760856065699556
  omega2_test = 0.2949840583669783
  
  @print_input
  
  geometry = "
             O      0.000000000    0.000000000   -0.130186067
             H1     0.000000000    1.489124508    1.033245507
             H2     0.000000000   -1.489124508    1.033245507"
  basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
  
  @dfhf
  energies = @ciphi
  
  @test haskey(energies, "CIPHI")
  E_ciphi = energies["CIPHI"]
  
  println("CIPHI Energy (H2O, RHF): $E_ciphi")
  @test abs(E_ciphi - E_CIPHI_test) < epsilon
  @test abs(E_ciphi + energies["CIPHI-correction"] - E_CIPHI_PT2_test) < epsilon

  
  energies = @ciphi begin
    ciphi(epsilon=1.e-5)
  end
  E_ciphi_tight = energies["CIPHI"]

  println("CIPHI Energy (tight selection, ε=1e-5): $E_ciphi_tight")
  @test abs(E_ciphi_tight - E_CIPHI_tight_test) < epsilon

  energies = @ciphi begin
    ciphi(epsilon=1.e-3)
  end
  E_ciphi_loose = energies["CIPHI"]
      
  println("CIPHI Energy (loose selection, ε=1e-3): $E_ciphi_loose")
  @test abs(E_ciphi_loose - E_CIPHI_loose_test) < epsilon
  
  energies = @ciphi begin
    @set ciphi epsilon=5.e-4 nstates=3
  end

  println("CIPHI Multi-root energies: ", energies["CIPHI"], ", ", energies["ω1"], ", ", energies["ω2"])
  @test abs(energies["CIPHI"] - E_CIPHI_ms_test) < epsilon
  @test abs(energies["ω1"] - omega1_test) < epsilon
  @test abs(energies["ω2"] - omega2_test) < epsilon

  println("\n=== CIPHI RHF Tests Passed ===\n")
end
end
