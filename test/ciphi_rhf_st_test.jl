@testitem "ciphi_rhf_st" tags=[:ciphi, :quick] begin
using Test
using ElemCo

@testset "CIPHI - RHF Similarity-Transformed Systems" begin
  println("\n=== Testing CIPHI with RHF xTC ===")
  
  epsilon = 1.e-6
  E_CIPHI_test = -76.379229376455
  E_CIPHI_PT2_test = -76.379775159873
  
  @print_input
  geometry = nothing 
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

@testset "CIPHI - RHF ST with frozen orbitals" begin
  println("\n=== Testing CIPHI with RHF xTC + frozen orbitals ===")

  epsilon = 1.e-4
  E_CIPHI_test = -76.35024356548195
  E_CIPHI_PT2_test = -76.350524245628

  geometry = nothing
  fcidump = joinpath(@__DIR__, "files", "H2O_ST1.FCIDUMP")

  @freeze_orbs [1,[20:24;]...]
  @set ciphi epsilon=1.e-4
  energies = @ciphi

  @test haskey(energies, "CIPHI")
  E_ciphi = energies["CIPHI"]

  println("CIPHI Energy (H2O, RHF, ST, frozen): $E_ciphi")
  @test abs(E_ciphi - E_CIPHI_test) < epsilon
  @test abs(E_ciphi + energies["CIPHI-correction"] - E_CIPHI_PT2_test) < epsilon

  println("\n=== CIPHI RHF ST Frozen Orbitals Tests Passed ===\n")
end
end
