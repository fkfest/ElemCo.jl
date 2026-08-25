@testitem "fci_uhf" tags=[:fci, :quick] begin
using Test
using ElemCo

@testset "Full CI - UHF Systems" begin
  println("\n=== Testing Full CI with UHF ===")
    
  epsilon = 1.e-6
  E_FCIa_test = -75.941006867518
  E_FCIc_test = -75.688631961833
  E_FCIt_test = -75.853283647874

  omega1_test = 0.08961778672348

  @testset "FCI Basic - H2O Anion" begin
    
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @dfuhf
    energies = @fci
    
    @test haskey(energies, "FCI")
    E_fci = energies["FCI"]
    
    println("FCI Energy (H2O anion, UHF): $E_fci")
    @test abs(E_fci - E_FCIa_test) < epsilon
  end
  
  @testset "FCI - H2O Cation" begin
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=1
    @dfuhf
    energies = @fci
    
    E_fci = energies["FCI"]
    
    println("FCI Energy (H2O cation, UHF): $E_fci")
    @test abs(E_fci - E_FCIc_test) < epsilon
  end
  
  @testset "FCI - H2O Triplet" begin
    
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf ms2=2
    @dfuhf
    energies = @fci
    
    E_fci = energies["FCI"]
    
    println("FCI Energy (H2O triplet, UHF): $E_fci")
    @test abs(E_fci - E_FCIt_test) < epsilon
  end
  
  @testset "FCI UHF Multi-root" begin

    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set fci nstates=2
    @dfuhf
    energies = @fci
    
    println("FCI Multi-root energies (UHF): ", energies["FCI"])
    
    @test abs(energies["FCI"] - E_FCIa_test) < epsilon
    @test abs(energies["ω1"] - omega1_test) < epsilon
  end
  
  println("\n=== Full CI UHF Tests Passed ===\n")
end
end
