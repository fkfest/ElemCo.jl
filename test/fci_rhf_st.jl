using Test
using ElemCo

@testset "FCI - RHF Similarity-Transformed Systems" begin
  println("\n=== Testing FCI with RHF xTC (N atom, ST) ===")

  epsilon = 1.e-6
  E_FCI_test = -54.562038540256

  @print_input
  geometry = nothing
  fcidump = joinpath(@__DIR__, "files", "N_ST1.FCIDUMP")

  # Run full FCI
  energies_fci = @fci

  @test haskey(energies_fci, "FCI")
  E_fci = energies_fci["FCI"]
  println("FCI Energy (N, ST): $E_fci")
  @test abs(E_fci - E_FCI_test) < epsilon

  println("\n=== FCI RHF Similarity-Transformed Tests Passed ===\n")
end
