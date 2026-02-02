using Test
using ElemCo

@testset "CIPHI - RHF Similarity-Transformed Systems" begin
  println("\n=== Testing CIPHI with RHF xTC ===")
  
  epsilon = 1.e-6
  E_CIPHI_test = -76.379229376455
  E_CIPHI_PT2_test = -76.379775159873
  
  @print_input
  
  fcidump = joinpath(@__DIR__,"files","H2O_ST1_SWAP.FCIDUMP") 

  @set ciphi use_mp2=true epsilon=1.e-4
  energies = @ciphi
  
  @test haskey(energies, "CIPHI")
  E_ciphi = energies["CIPHI"]
  
  println("CIPHI Energy (H2O, RHF, ST): $E_ciphi")
  @test abs(E_ciphi - E_CIPHI_test) < epsilon
  @test abs(E_ciphi + energies["CIPHI-correction"] - E_CIPHI_PT2_test) < epsilon

  println("\n=== CIPHI RHF Similarity-Transformed Tests Passed ===\n")
end
