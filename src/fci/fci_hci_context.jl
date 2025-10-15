
"""
    HCIContext

Lightweight context for Heat-Bath Configuration Interaction (HCI) calculations.

Unlike FCIContext, HCIContext does NOT pre-compute full-space address tables
or diagonal Hamiltonian elements. Instead, it only stores the minimal data needed:
- Integral data (FCIDump)
- Calculation options (HeatBathCIOptions)
- System size parameters
- Modified core Hamiltonian (for correct diagonal element calculation)

Address tables and diagonal elements are computed on-demand only for the selected
determinants during HCI iterations. This provides:
- Faster initialization (no full-space addressing)
- Lower memory usage (proportional to N_selected, not N_determinants)
- Better scaling for large orbital spaces

For a system with 23 orbitals:
- Full FCI space: ~4 million determinants
- HCI selected space: ~1500-4000 determinants
- Memory savings: ~99.9% (only store what's needed)

# Fields
- `fcidump::FCIDump` - Integral data (h1, h2 tensors)
- `options::HeatBathCIOptions` - HCI-specific calculation options
- `n_orb::Int` - Number of spatial orbitals
- `n_elec::Tuple{Int,Int}` - (n_alpha, n_beta) electron counts
- `reference_det::Determinant` - HF reference determinant
- `is_uhf::Bool` - Whether using UHF integrals
- `mod_core_h_a::Matrix{Scalar}` - Modified core Hamiltonian for alpha spin
- `mod_core_h_b::Matrix{Scalar}` - Modified core Hamiltonian for beta spin
"""
struct HCIContext
  fcidump::FCIDump
  options::HeatBathCIOptions
  n_orb::Int
  n_elec::Tuple{Int,Int}
  reference_det::Determinant
  mod_core_h_a::Matrix{Scalar}
  mod_core_h_b::Matrix{Scalar}

  function HCIContext(fcidump::FCIDump, options::HeatBathCIOptions = HeatBathCIOptions())
    n_orb = fcidump.n_orb
    n_elec = fcidump.n_elec
    n_spin = fcidump.n_spin
    is_uhf = fcidump.is_uhf

    # Validate integrals
    if is_uhf
      if fcidump.h1a === nothing || fcidump.h1b === nothing
        error("UHF calculation requires h1a, h1b integrals")
      end
      if fcidump.h2aa === nothing || fcidump.h2bb === nothing || fcidump.h2ab === nothing
        error("UHF calculation requires h2aa, h2bb, h2ab integrals")
      end
    else
      if fcidump.h1 === nothing || fcidump.h2 === nothing
        error("RHF calculation requires h1, h2 integrals")
      end
    end

    # Compute modified core Hamiltonian (needed for correct diagonal elements)
    # This matches the calculation in FCIContext.init_hamiltonian_terms!
    if is_uhf
      # UHF: use spin-separated integrals
      mod_core_h_a = copy(fcidump.h1a)
      mod_core_h_b = copy(fcidump.h1b)
      
      # Add 2e contributions: mod_core_h[m,n] = h1[m,n] - 0.5 * sum_i(h2[m,i,n,i])
      for m in 1:n_orb
        for n in 1:n_orb
          for i in 1:n_orb
            mod_core_h_a[m,n] -= 0.5 * fcidump.h2aa[m,i,n,i]
            mod_core_h_b[m,n] -= 0.5 * fcidump.h2bb[m,i,n,i]
          end
        end
      end
    else
      # RHF: use spatial integrals
      mod_core_h_a = copy(fcidump.h1)
      mod_core_h_b = copy(fcidump.h1)
      
      # Add 2e contributions
      for m in 1:n_orb
        for n in 1:n_orb
          for i in 1:n_orb
            mod_core_h_a[m,n] -= 0.5 * fcidump.h2[m,i,n,i]
            mod_core_h_b[m,n] -= 0.5 * fcidump.h2[m,i,n,i]
          end
        end
      end
    end

    # Create HF reference determinant
    n_alpha = (n_elec + n_spin) ÷ 2
    n_beta = (n_elec - n_spin) ÷ 2
    
    # HF determinant: occupy first n_alpha/n_beta orbitals
    alpha_pattern = (OrbPattern(1) << n_alpha) - OrbPattern(1)
    beta_pattern = (OrbPattern(1) << n_beta) - OrbPattern(1)
    reference_det = Determinant(alpha_pattern, beta_pattern)

    new(fcidump, options, n_orb, (n_alpha, n_beta), reference_det, mod_core_h_a, mod_core_h_b)
  end
end
