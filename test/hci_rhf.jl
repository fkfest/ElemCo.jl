using Test
using ElemCo

@testset "Heat-Bath CI - RHF Systems" begin
  println("\n=== Testing Heat-Bath CI with RHF ===")
  
  epsilon = 1.e-6
  E_HCI_test = -76.119399679024
  E_HCI_PT2_test = -76.119932129386
  E_HCI_tight_test = -76.120269539723
  E_HCI_loose_test = -76.115687182148
  E_HCI_ms_test = -76.11860647185388
  omega1_test = 0.2674567782209607
  omega2_test = 0.2948479230453245
  
  @print_input
  
  geometry = "
             O      0.000000000    0.000000000   -0.130186067
             H1     0.000000000    1.489124508    1.033245507
             H2     0.000000000   -1.489124508    1.033245507"
  basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
  
  @dfhf
  energies = @hci
  
  @test haskey(energies, "HCI")
  E_hci = energies["HCI"]
  
  println("HCI Energy (H2O, RHF): $E_hci")
  @test abs(E_hci - E_HCI_test) < epsilon
  @test abs(E_hci + energies["HCI-correction"] - E_HCI_PT2_test) < epsilon

  
  @set hci epsilon=1.e-5
      
  energies = @hci

  E_hci_tight = energies["HCI"]

  println("HCI Energy (tight selection, ε=1e-5): $E_hci_tight")
  @test abs(E_hci_tight - E_HCI_tight_test) < epsilon

  @set hci epsilon=1.e-3

  energies = @hci
      
  E_hci_loose = energies["HCI"]
      
  println("HCI Energy (loose selection, ε=1e-3): $E_hci_loose")
  @test abs(E_hci_loose - E_HCI_loose_test) < epsilon
  
  @set hci epsilon=5.e-4 nstates=3
  energies = @hci

  println("HCI Multi-root energies: ", energies["HCI"], ", ", energies["ω1"], ", ", energies["ω2"])
  @test abs(energies["HCI"] - E_HCI_ms_test) < epsilon
  @test abs(energies["ω1"] - omega1_test) < epsilon
  @test abs(energies["ω2"] - omega2_test) < epsilon

  println("\n=== Heat-Bath CI RHF Tests Passed ===\n")
end
