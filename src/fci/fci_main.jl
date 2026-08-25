
"""
    run_fci!(context::FCIContext) -> Vector{Float64}

Run full FCI calculation.
"""
function run_fci!(context::FCIContext{O,T}) where {O, T}
  println("="^80)
  println("FCI CALCULATION")
  println("="^80)

  # Run Davidson diagonalization
  energies, eigenvectors = davidson_fci!(context)
  energy = energies[1]  # Ground state energy

  println("\n" * "="^80)
  println("FCI RESULTS")
  println("="^80)
  println("Ground state energy: $(energy) Hartree")
  println("Electronic energy:   $(energy - context.fcidump.int0) Hartree")
  
  # Compute RDMs if requested
  if context.options.compute_rdms
    println("\nComputing 1-RDMs...")
    make_1rdms!(context.rdm1_a, context.rdm1_b, context.coeff)
    
    # Verify trace
    n_elec = context.n_elec[1] + context.n_elec[2]
    trace_1rdm = tr(context.rdm1_a) + tr(context.rdm1_b)
    println("  1-RDM trace: $(trace_1rdm) (expected: $(n_elec))")
    
    if abs(trace_1rdm - n_elec) > 1e-8
      @warn "1-RDM trace error: $(abs(trace_1rdm - n_elec))"
    else
      println("  ✓ 1-RDM trace verified")
    end
    
    # Compute 2-RDM if requested
    if context.options.compute_2rdm
      n_orb = context.n_orb
      println("\nComputing 2-RDM...")
      
      # Allocate 2-RDM array if not already allocated
      if context.rdm2 === nothing
        context.rdm2 = zeros(T, n_orb, n_orb, n_orb, n_orb)
      end
      
      # Compute 2-RDM
      rdm1 = context.rdm1_a + context.rdm1_b
      make_2rdm!(context.rdm2, context.coeff, rdm1, context.options.thr_negligible)
      
      # Verify energy from 2-RDM
      if context.fcidump.uhf
        error("2-RDM energy verification not implemented for UHF FCI yet.")
      end
      int2 = context.fcidump.int2
      e_2rdm = 0.0
      for l in 1:n_orb, k in 1:n_orb, j in 1:n_orb, i in 1:n_orb
        e_2rdm += 0.5 * context.rdm2[i,j,k,l] * int2[i,j,k,l]
      end
      
      # Add 1-electron contribution if absorb_1e is true
      if context.absorb_1e
        e_1rdm = tr(rdm1 * context.fcidump.int1)
        e_total_rdm = e_1rdm + e_2rdm + context.fcidump.int0
      else
        e_total_rdm = e_2rdm + context.fcidump.int0
      end
      
      println("  Energy from 2-RDM: $(e_total_rdm) Hartree")
      println("  FCI energy:        $(energy) Hartree")
      println("  Difference:        $(abs(e_total_rdm - energy)) Hartree")
      
      if abs(e_total_rdm - energy) > 1e-6
        error("2-RDM energy check failed! Difference: $(abs(e_total_rdm - energy))")
      else
        println("  ✓ 2-RDM energy verified")
      end
    end
  end
  
  println("="^80)

  return energies
end