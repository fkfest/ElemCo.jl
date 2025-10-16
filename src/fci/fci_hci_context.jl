"""
    HamiltonianTerm

Abstract base type for Hamiltonian terms.
"""
abstract type HamiltonianTerm end

"""
    FCIContext

Main FCI calculation context.
"""
mutable struct FCIContext
  fcidump::QFDump
  options::FCIOptions
  n_orb::Int
  n_elec::Tuple{Int,Int}
  adr_a::OrbStringAdrTable
  adr_b::OrbStringAdrTable
  coeff::FCIVector
  resid::FCIVector
  diag_h::FCIVector
  absorb_1e::Bool
  hamiltonian_terms::Vector{HamiltonianTerm}
  reference_det::Determinant
  mod_core_h_a::Matrix{Scalar}
  mod_core_h_b::Matrix{Scalar}
  basis_a::Matrix{Scalar}
  basis_b::Matrix{Scalar}
  rdm1_a::Matrix{Scalar}
  rdm1_b::Matrix{Scalar}
  rdm2::Union{Nothing, Array{Scalar, 4}}  # 2-RDM (optional)
  energy_fci::Scalar
  energy_ptrace::Scalar
  method_name::String
  pspace_data::PSpaceData             # P-space calculation data

  function FCIContext(fcidump::QFDump, options::FCIOptions = FCIOptions(); occa=nothing, occb=nothing)
    n_elec = headvar(fcidump, "NELEC", Int) 
    n_orb = headvar(fcidump, "NORB", Int)
    ms2 = headvar(fcidump, "MS2", Int)
    n_alpha = (n_elec + ms2) ÷ 2
    n_beta = (n_elec - ms2) ÷ 2
    # Initialize FCI vectors
    coeff = FCIVector(n_elec, n_orb, ms2, false)
    resid = FCIVector(n_elec, n_orb, ms2, false)
    diag_h = FCIVector(n_elec, n_orb, ms2, false)

    # Select integrals based on RHF vs UHF
    if fcidump.uhf
      # UHF: Use spin-separated integrals
      if length(fcidump.int1a) == 0 || length(fcidump.int1b) == 0
        error("UHF calculation requires int1a, int1b integrals")
      end
      if length(fcidump.int2aa) == 0 || length(fcidump.int2bb) == 0 || length(fcidump.int2ab) == 0
        error("UHF calculation requires int2aa, int2bb, int2ab integrals")
      end

      mod_core_h_a = copy(fcidump.int1a)
      mod_core_h_b = copy(fcidump.int1b)
    else
      # RHF: Use spatial integrals
      mod_core_h_a = copy(fcidump.int1)
      mod_core_h_b = copy(fcidump.int1)
    end

    if occa !== nothing && occb !== nothing
      # User provided occupation patterns for reference determinant
      if length(occa) != n_alpha || length(occb) != n_beta
        error("Provided occupation patterns do not match n_alpha/n_beta")
      end
      reference_det = Determinant(occa, occb)
    else
      # HF determinant: occupy first n_alpha/n_beta orbitals
      alpha_pattern = (OrbPattern(1) << n_alpha) - OrbPattern(1)
      beta_pattern = (OrbPattern(1) << n_beta) - OrbPattern(1)
      reference_det = Determinant(alpha_pattern, beta_pattern)
    end

    context = new(
      fcidump,
      options,
      n_orb,
      (n_alpha, n_beta),
      coeff.adr_a,
      coeff.adr_b,
      coeff,
      resid,
      diag_h,
      true,
      HamiltonianTerm[],  # Set absorb_1e = true by default
      reference_det,
      mod_core_h_a,
      mod_core_h_b,
      Matrix{Scalar}(I, n_orb, n_orb),
      Matrix{Scalar}(I, n_orb, n_orb),
      zeros(Scalar, n_orb, n_orb),
      zeros(Scalar, n_orb, n_orb),
      nothing,  # rdm2 - not computed by default
      0.0,
      0.0,
      fcidump.uhf ? "UHF-FCI" : "FCI",
      PSpaceData(),
    )  # Initialize empty P-space data

    # Initialize Hamiltonian terms
    init_hamiltonian_terms!(context)

    return context
  end
end


"""
    HCIContext

Lightweight context for Heat-Bath Configuration Interaction (HCI) calculations.

Unlike FCIContext, HCIContext does NOT pre-compute full-space address tables
or diagonal Hamiltonian elements. Instead, it only stores the minimal data needed:
- Integral data (FCIDump)
- Calculation options (HCIOptions)
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
- `fcidump::FCIDump` - Integral data
- `options::HCIOptions` - HCI-specific calculation options
- `n_orb::Int` - Number of spatial orbitals
- `n_elec::Tuple{Int,Int}` - (n_alpha, n_beta) electron counts
- `reference_det::Determinant` - Reference determinant
- `is_uhf::Bool` - Whether using UHF integrals
- `mod_core_h_a::Matrix{Scalar}` - Modified core Hamiltonian for alpha spin
- `mod_core_h_b::Matrix{Scalar}` - Modified core Hamiltonian for beta spin
"""
struct HCIContext
  fcidump::QFDump
  options::HCIOptions
  n_orb::Int
  n_elec::Tuple{Int,Int}
  reference_det::Determinant
  mod_core_h_a::Matrix{Scalar}
  mod_core_h_b::Matrix{Scalar}

  function HCIContext(fcidump::QFDump, options::HCIOptions = HCIOptions(); occa=nothing, occb=nothing)
    n_orb = headvar(fcidump, "NORB", Int)
    n_elec = headvar(fcidump, "NELEC", Int)
    ms2 = headvar(fcidump, "MS2", Int)
    is_uhf = fcidump.uhf

    # Validate integrals
    if is_uhf
      if length(fcidump.int1a) == 0 || length(fcidump.int1b) == 0
        error("UHF-based calculation requires int1a, int1b integrals")
      end
      if length(fcidump.int2aa) == 0 || length(fcidump.int2bb) == 0 || length(fcidump.int2ab) == 0
        error("UHF-based calculation requires int2aa, int2bb, int2ab integrals")
      end
    else
      if length(fcidump.int1) == 0 || length(fcidump.int2) == 0
        error("RHF calculation requires int1, int2 integrals")
      end
    end

    # Compute modified core Hamiltonian (needed for correct diagonal elements)
    # This matches the calculation in FCIContext.init_hamiltonian_terms!
    if is_uhf
      # UHF: use spin-separated integrals
      mod_core_h_a = copy(fcidump.int1a)
      mod_core_h_b = copy(fcidump.int1b)
      int2aa = fcidump.int2aa
      int2bb = fcidump.int2bb
    else
      # RHF: use spatial integrals
      mod_core_h_a = copy(fcidump.int1)
      mod_core_h_b = copy(fcidump.int1)
      int2aa = fcidump.int2
      int2bb = fcidump.int2
    end
    @tensor mod_core_h_a[m,n] -= 0.5 * int2aa[m,i,n,i]
    @tensor mod_core_h_b[m,n] -= 0.5 * int2bb[m,i,n,i]

    n_alpha = (n_elec + ms2) ÷ 2
    n_beta = (n_elec - ms2) ÷ 2
    if occa !== nothing && occb !== nothing
      # User provided occupation patterns for reference determinant
      if length(occa) != n_alpha || length(occb) != n_beta
        error("Provided occupation patterns do not match n_alpha/n_beta")
      end
      reference_det = Determinant(occa, occb)
    else
      # HF determinant: occupy first n_alpha/n_beta orbitals
      alpha_pattern = (OrbPattern(1) << n_alpha) - OrbPattern(1)
      beta_pattern = (OrbPattern(1) << n_beta) - OrbPattern(1)
      reference_det = Determinant(alpha_pattern, beta_pattern)
    end

    new(fcidump, options, n_orb, (n_alpha, n_beta), reference_det, mod_core_h_a, mod_core_h_b)
  end
end
