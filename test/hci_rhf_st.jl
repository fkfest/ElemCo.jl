using Test
using ElemCo

@testset "Heat-Bath CI - RHF Similarity-Transformed Systems" begin
  println("\n=== Testing Heat-Bath CI with RHF xTC ===")
  
  epsilon = 1.e-6
  E_HCI_test = -76.379227285415
  E_HCI_PT2_test = -76.379742592201
  
  @print_input
  
  fcidump = joinpath(@__DIR__,"files","H2O_ST1_SWAP.FCIDUMP") 

  @set hci use_mp2=true epsilon=1.e-4
  energies = @hci
  
  @test haskey(energies, "HCI")
  E_hci = energies["HCI"]
  
  println("HCI Energy (H2O, RHF, ST): $E_hci")
  @test abs(E_hci - E_HCI_test) < epsilon
  @test abs(E_hci + energies["HCI-correction"] - E_HCI_PT2_test) < epsilon

  println("\n=== Heat-Bath CI RHF Similarity-Transformed Tests Passed ===\n")
end
