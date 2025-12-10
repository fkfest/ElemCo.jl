using Test
using ElemCo

@testset "Heat-Bath CI - UHF Systems" begin
  println("\n=== Testing Heat-Bath CI with UHF ===")
    
  epsilon = 1.e-6
  E_HCIa_test = -75.940790606682
  E_HCIa_PT2_test = -75.941009904381
  E_HCIc_test = -75.688047789381
  E_HCIt_test = -75.852590484068

  E_HCIa_tight_test = -75.941001448353

  E_HCIa_ms_test = -75.939226562784
  omega1_test = 0.08963377462914934
    
  @testset "HCI Basic - H2O Anion" begin
        
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set hci epsilon=1.e-4
    
    @dfuhf
    energies = @hci
    
    @test haskey(energies, "HCI")
    E_hci = energies["HCI"]
    
    println("HCI Energy (H2O anion, UHF): $E_hci")
    @test abs(E_hci - E_HCIa_test) < epsilon
  end
    
  @testset "HCI - H2O Cation" begin
    
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=1

    @dfuhf
    energies = @hci
    
    E_hci = energies["HCI"]
    
    println("HCI Energy (H2O cation, UHF): $E_hci")
    @test abs(E_hci - E_HCIc_test) < epsilon
  end
    
  @testset "HCI - H2O Triplet" begin
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")

    @set wf ms2=2
    
    @dfuhf
    energies = @hci
    
    E_hci = energies["HCI"]
    
    println("HCI Energy (H2O triplet, UHF): $E_hci")
    @test abs(E_hci - E_HCIt_test) < epsilon
  end
    
  @testset "HCI UHF Selection Thresholds" begin
        
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set hci epsilon=1.e-5
    
    @dfuhf
    energies = @hci
    
    E_hci = energies["HCI"]
    
    println("HCI Energy (UHF, tight selection): $E_hci")
    @test abs(E_hci - E_HCIa_tight_test) < epsilon
  end
    
  @testset "HCI UHF Multi-root" begin
    geometry = "
               O      0.000000000    0.000000000   -0.130186067
               H1     0.000000000    1.489124508    1.033245507
               H2     0.000000000   -1.489124508    1.033245507"
    basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
    
    @set wf charge=-1
    @set hci nstates=2 epsilon=5.e-4
    
    @dfuhf
    energies = @hci
    
    println("HCI Multi-root energies (UHF): ", energies["HCI"])
    
    @test abs(energies["HCI"] - E_HCIa_ms_test) < epsilon
    @test abs(energies["ω1"] - omega1_test) < epsilon
  end

  println("\n=== Heat-Bath CI UHF Tests Passed ===\n")
end
