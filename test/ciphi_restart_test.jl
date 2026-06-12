@testitem "ciphi_restart" tags=[:ciphi, :quick] begin
using Test
using ElemCo

@testset "CIPHI - Store/Restart" begin
  println("\n=== Testing CIPHI Store/Restart ===")
  
  epsilon = 1.e-8  # Tolerance for comparing essentially identical calculations
  epsilon_energy = 1.e-5  # Standard tolerance for energy comparisons (allows for small selection differences)
  
  @print_input
  
  geometry = "
             O      0.000000000    0.000000000   -0.130186067
             H1     0.000000000    1.489124508    1.033245507
             H2     0.000000000   -1.489124508    1.033245507"
  basis = Dict("ao"=>"6-31g", "jkfit"=>"vtz-jkfit", "mpfit"=>"vtz-mpfit")
  
  @dfhf
  
  #==========================================================================
  Test 1: Basic single-state store and restart with pt2_only (exact match)
  ===========================================================================#
  @testset "Single-state store/restart (pt2_only)" begin
    println("\n--- Test 1: Single-state store/restart (pt2_only) ---")
    
    # First calculation: store determinants
    energies1 = @ciphi begin
      @set wf store="ciphi_test1.h5"
      @set ciphi epsilon=1.e-3 nstates=1
    end
    E1_var = energies1["CIPHI"]
    E1_corr = energies1["CIPHI-correction"]
    println("First run - Variational: $E1_var, PT2 correction: $E1_corr")
    
    # Second calculation: restart with pt2_only (should give identical result)
    energies2 = @ciphi begin
      @set wf start="ciphi_test1.h5"
      @set ciphi epsilon=1.e-3 nstates=1 pt2_only=true
    end
    E2_var = energies2["CIPHI"]
    E2_corr = energies2["CIPHI-correction"]
    println("Restart (pt2_only) - Variational: $E2_var, PT2 correction: $E2_corr")
    
    # Energies should be identical (same space, pt2_only)
    @test abs(E1_var - E2_var) < epsilon
    @test abs(E1_corr - E2_corr) < epsilon
    println("✓ Single-state store/restart (pt2_only): energies match")
    
    # Cleanup
    @deletefile("ciphi_test1.h5")
  end
  
  #==========================================================================
  Test 2: Multi-state store and restart with pt2_only (exact match)
  ===========================================================================#
  @testset "Multi-state store/restart (pt2_only)" begin
    println("\n--- Test 2: Multi-state store/restart (pt2_only) ---")
    
    # First calculation: store determinants for 3 states
    energies1 = @ciphi begin
      @set wf store="ciphi_test2.h5" 
      @set ciphi epsilon=5.e-4 nstates=3
    end
    E1_gs = energies1["CIPHI"]
    E1_omega1 = energies1["ω1"]
    E1_omega2 = energies1["ω2"]
    println("First run - GS: $E1_gs, ω1: $E1_omega1, ω2: $E1_omega2")
    
    # Second calculation: restart with pt2_only (exact match)
    energies2 = @ciphi begin
      @set wf start="ciphi_test2.h5"
      @set ciphi epsilon=5.e-4 nstates=3 pt2_only=true
    end
    E2_gs = energies2["CIPHI"]
    E2_omega1 = energies2["ω1"]
    E2_omega2 = energies2["ω2"]
    println("Restart (pt2_only) - GS: $E2_gs, ω1: $E2_omega1, ω2: $E2_omega2")
    
    # All state energies should match exactly
    @test abs(E1_gs - E2_gs) < epsilon
    @test abs(E1_omega1 - E2_omega1) < epsilon
    @test abs(E1_omega2 - E2_omega2) < epsilon
    println("✓ Multi-state store/restart (pt2_only): all state energies match")
    
    # Cleanup
    @deletefile("ciphi_test2.h5")
    @deletefile("ciphi_test2_state2.h5")
    @deletefile("ciphi_test2_state3.h5")
  end
  
  #==========================================================================
  Test 3: Restart with tighter threshold (warm start)
  ===========================================================================#
  @testset "Warm start with tighter threshold" begin
    println("\n--- Test 3: Warm start with tighter threshold ---")
    
    # First calculation: loose threshold
    energies1 = @ciphi begin
      @set wf store="ciphi_test3.h5"
      @set ciphi epsilon=1.e-3 nstates=2
    end
    E1_var = energies1["CIPHI"]
    println("First run (ε=1e-3) - Energy: $E1_var")
    
    # Second calculation: restart with tighter threshold
    energies2 = @ciphi begin
      @set wf start="ciphi_test3.h5"
      @set ciphi epsilon=3.e-4 nstates=2
    end
    E2_var = energies2["CIPHI"]
    println("Restart run (ε=3e-4) - Energy: $E2_var")
    
    # Energy should improve with tighter threshold
    @test E2_var < E1_var
    @test abs(E2_var - E1_var) > 1e-4  # Significant improvement expected
    println("✓ Warm start: energy improved from $E1_var to $E2_var")
    
    # Cleanup
    @deletefile("ciphi_test3.h5")
    @deletefile("ciphi_test3_state2.h5")
  end
  
  #==========================================================================
  Test 4: pt2_only mode
  ===========================================================================#
  @testset "PT2-only mode" begin
    println("\n--- Test 4: PT2-only mode ---")
    
    # First calculation: full CIPHI with store
    energies1 = @ciphi begin
      @set wf store="ciphi_test4.h5"
      @set ciphi epsilon=1.e-3 nstates=2
    end
    E1_var = energies1["CIPHI"]
    E1_corr = energies1["CIPHI-correction"]
    E1_total = E1_var + E1_corr
    println("Full CIPHI - Var: $E1_var, PT2: $E1_corr, Total: $E1_total")
    
    # Second calculation: pt2_only mode (skip variational iterations)
    energies2 = @ciphi begin
      @set wf start="ciphi_test4.h5"
      @set ciphi pt2_only=true nstates=2
    end
    E2_var = energies2["CIPHI"]
    E2_corr = energies2["CIPHI-correction"]
    E2_total = E2_var + E2_corr
    println("PT2-only - Var: $E2_var, PT2: $E2_corr, Total: $E2_total")
    
    # Energies should be identical
    @test abs(E1_var - E2_var) < epsilon
    @test abs(E1_corr - E2_corr) < epsilon
    @test abs(E1_total - E2_total) < epsilon
    println("✓ PT2-only mode: energies match full calculation")
    
    # Cleanup
    @deletefile("ciphi_test4.h5")
    @deletefile("ciphi_test4_state2.h5")
  end
  
  #==========================================================================
  Test 5: pt2_only with different epsilon_pt2
  ===========================================================================#
  @testset "PT2-only with different thresholds" begin
    println("\n--- Test 5: PT2-only with different epsilon_pt2 ---")
    
    # First calculation: store determinants
    energies1 = @ciphi begin
      @set wf store="ciphi_test5.h5"
      @set ciphi epsilon=5.e-4 epsilon_pt2=1.e-6
    end
    E1_corr = energies1["CIPHI-correction"]
    println("Original PT2 (ε_pt2=1e-6): $E1_corr")
    
    # Second calculation: pt2_only with tighter epsilon_pt2
    energies2 = @ciphi begin
      @set wf start="ciphi_test5.h5"
      @set ciphi pt2_only=true epsilon_pt2=1.e-8
    end
    E2_corr = energies2["CIPHI-correction"]
    println("PT2-only (ε_pt2=1e-8): $E2_corr")
    
    # PT2 corrections should be similar but may differ slightly due to threshold
    @test abs(E1_corr - E2_corr) < 1e-5  # Should be close but not identical
    println("✓ PT2-only with different threshold: corrections are consistent")
    
    # Cleanup
    @deletefile("ciphi_test5.h5")
  end
  
  #==========================================================================
  Test 6: pt2_only requires restart (error handling)
  ===========================================================================#
  @testset "PT2-only error handling" begin
    println("\n--- Test 6: PT2-only error handling ---")
    
    # pt2_only without restart should throw an error
    
    @test_throws ErrorException @ciphi begin
      @set ciphi pt2_only=true
    end
    println("✓ PT2-only without restart correctly throws error")
  end
  
  #==========================================================================
  Test 7: Store and restart across different nstates (pt2_only for exact match)
  ===========================================================================#
  @testset "Restart with different nstates" begin
    println("\n--- Test 7: Restart with different nstates ---")
    
    # First calculation: 3 states
    energies1 = @ciphi begin
      @set wf store="ciphi_test7.h5"
      @set ciphi epsilon=5.e-4 nstates=3
    end
    E1_gs = energies1["CIPHI"]
    println("First run (3 states) - GS: $E1_gs")
    
    # Second calculation: restart with pt2_only, request only 2 states
    # Using pt2_only ensures we get exact match on the same determinant space
    energies2 = @ciphi begin
      @set wf start="ciphi_test7.h5"
      @set ciphi epsilon=5.e-4 nstates=2 pt2_only=true
    end
    E2_gs = energies2["CIPHI"]
    println("Restart (pt2_only, 2 states) - GS: $E2_gs")
    
    # Ground state energy should be identical (same determinant space, pt2_only)
    @test abs(E1_gs - E2_gs) < epsilon
    println("✓ Restart with different nstates: ground state energies match")
    
    # Cleanup
    @deletefile("ciphi_test7.h5")
    @deletefile("ciphi_test7_state2.h5")
    @deletefile("ciphi_test7_state3.h5")
  end
  
  #==========================================================================
  Test 8: Chain of restarts (A -> B -> C)
  ===========================================================================#
  @testset "Chain of restarts" begin
    println("\n--- Test 8: Chain of restarts ---")
    
    # First calculation: very loose threshold
    energies_a = @ciphi begin
      @set wf store="ciphi_chain_a.h5"
      @set ciphi epsilon=2.e-3
    end
    E_a = energies_a["CIPHI"]
    println("Chain A (ε=2e-3): $E_a")
    
    # Second calculation: restart from A, use medium threshold, store as B
    energies_b = @ciphi begin
      @set wf start="ciphi_chain_a.h5" store="ciphi_chain_b.h5"
      @set ciphi epsilon=1.e-3
    end
    E_b = energies_b["CIPHI"]
    println("Chain B (ε=1e-3): $E_b")
    
    # Third calculation: restart from B, use tight threshold
    energies_c = @ciphi begin
      @set wf start="ciphi_chain_b.h5"
      @set ciphi epsilon=5.e-4
    end
    E_c = energies_c["CIPHI"]
    println("Chain C (ε=5e-4): $E_c")
    
    # Energy should monotonically decrease
    @test E_b < E_a
    @test E_c < E_b
    println("✓ Chain of restarts: energies improve monotonically")
    println("  $E_a -> $E_b -> $E_c")
    
    # Cleanup
    @deletefile("ciphi_chain_a.h5")
    @deletefile("ciphi_chain_b.h5")
  end
  
  #==========================================================================
  Test 9: Verify warm restart can only improve or maintain energy
  ===========================================================================#
  @testset "Warm restart energy improvement" begin
    println("\n--- Test 9: Warm restart energy improvement ---")
    
    # Store determinants with 2 states
    energies1 = @ciphi begin
      @set wf store="ciphi_test9.h5"
      @set ciphi epsilon=5.e-4 nstates=2
    end
    E1_gs = energies1["CIPHI"]
    E1_omega = energies1["ω1"]
    println("Original - GS: $E1_gs, ω1: $E1_omega")
    
    # Restart with same parameters - may add more determinants, energy can only improve
    energies2 = @ciphi begin
      @set wf start="ciphi_test9.h5"
      @set ciphi epsilon=5.e-4 nstates=2
    end
    E2_gs = energies2["CIPHI"]
    E2_omega = energies2["ω1"]
    println("Warm restart - GS: $E2_gs, ω1: $E2_omega")
    
    # Energy should improve or stay the same (variational principle)
    @test E2_gs <= E1_gs + epsilon
    # Excitation energies should be consistent within tolerance
    @test abs(E1_omega - E2_omega) < epsilon_energy
    println("✓ Warm restart: energy improved or maintained")
    
    # Cleanup
    @deletefile("ciphi_test9.h5")
    @deletefile("ciphi_test9_state2.h5")
  end

  println("\n=== CIPHI Store/Restart Tests Passed ===\n")
end
end
