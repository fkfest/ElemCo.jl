"""
    HamiltonianTerm

Abstract base type for Hamiltonian terms.
"""
abstract type HamiltonianTerm end

"""
    FCIContext

Main FCI calculation context.
"""
mutable struct FCIContext{OPattern}
  fcidump::QFDump
  options::FCIOptions
  n_orb::Int
  n_elec::Tuple{Int,Int}
  adr_a::OrbStringAdrTable{OPattern}
  adr_b::OrbStringAdrTable{OPattern}
  coeff::FCIVector{OPattern}
  resid::FCIVector{OPattern}
  diag_h::FCIVector{OPattern}
  absorb_1e::Bool
  hamiltonian_terms::Vector{HamiltonianTerm}
  reference_det::Determinant{OPattern}
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
  pspace_data::PSpaceData{OPattern}  # P-space calculation data
  heval_data::HEvalData              # Precomputed arrays for diagonal and Fock elements
  int1a::Matrix{Scalar}              # Reference to one-electron integrals for alpha spin
  int1b::Matrix{Scalar}              # Reference to one-electron integrals for beta spin
  int2aa::Array{Scalar,4}            # Reference to two-electron integrals for alpha-alpha spin
  int2bb::Array{Scalar,4}            # Reference to two-electron integrals for beta-beta spin
  int2ab::Array{Scalar,4}            # Reference to two-electron integrals for alpha-beta spin

  function FCIContext{OPattern}(fcidump::QFDump, options::FCIOptions = FCIOptions(); occa=nothing, occb=nothing) where OPattern
    n_elec = headvar(fcidump, "NELEC", Int) 
    n_orb = headvar(fcidump, "NORB", Int)
    ms2 = headvar(fcidump, "MS2", Int)
    n_alpha = (n_elec + ms2) ÷ 2
    n_beta = (n_elec - ms2) ÷ 2
    # Initialize FCI vectors
    coeff = FCIVector{OPattern}(n_elec, n_orb, ms2, false)
    resid = FCIVector{OPattern}(n_elec, n_orb, ms2, false)
    diag_h = FCIVector{OPattern}(n_elec, n_orb, ms2, false)

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
      reference_det = Determinant{OPattern}(occa, occb)
    else
      # HF determinant: occupy first n_alpha/n_beta orbitals
      alpha_pattern = (OPattern(1) << n_alpha) - OPattern(1)
      beta_pattern = (OPattern(1) << n_beta) - OPattern(1)
      reference_det = Determinant(alpha_pattern, beta_pattern)
    end

    if fcidump.uhf
      int1a = fcidump.int1a
      int1b = fcidump.int1b
      int2aa = fcidump.int2aa
      int2bb = fcidump.int2bb
      int2ab = fcidump.int2ab
    else
      int1a = fcidump.int1
      int1b = fcidump.int1
      int2aa = fcidump.int2
      int2bb = fcidump.int2
      int2ab = fcidump.int2
    end

    context = new{OPattern}(
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
      PSpaceData{OPattern}(),
      HEvalData(),  # computed later if needed
      int1a, int1b, int2aa, int2bb, int2ab
    )

    # Initialize Hamiltonian terms
    init_hamiltonian_terms!(context)

    return context
  end
end

# Convenience constructor that defaults to UInt64
FCIContext(fcidump::QFDump, options::FCIOptions = FCIOptions(); kwargs...) = FCIContext{UInt64}(fcidump, options; kwargs...)


"""
    CIPHIContext

Lightweight context for CIPHI (CIΦ - CI via Perturbative and Heat-Bath Iterative selection) calculations.

Unlike FCIContext, CIPHIContext does NOT pre-compute full-space address tables
or diagonal Hamiltonian elements. Instead, it only stores the minimal data needed:
- Integral data (FCIDump)
- Calculation options (CIPHIOptions)
- System size parameters
- Modified core Hamiltonian (for correct diagonal element calculation)

Address tables and diagonal elements are computed on-demand only for the selected
determinants during CIPHI iterations. This provides:
- Faster initialization (no full-space addressing)
- Lower memory usage (proportional to N_selected, not N_determinants)
- Better scaling for large orbital spaces

# Fields
- `fcidump::FCIDump` - Integral data
- `options::CIPHIOptions` - CIPHI-specific calculation options
- `n_orb::Int` - Number of spatial orbitals
- `n_elec::Tuple{Int,Int}` - (n_alpha, n_beta) electron counts
- `reference_det::Determinant` - Reference determinant
- `mod_core_h_a::Matrix{Scalar}` - Modified core Hamiltonian for alpha spin
- `mod_core_h_b::Matrix{Scalar}` - Modified core Hamiltonian for beta spin
- `heval_data::HEvalData` - Precomputed arrays for diagonal and Fock elements
"""
mutable struct CIPHIContext{OPattern}
  fcidump::QFDump
  options::CIPHIOptions
  n_orb::Int
  n_elec::Tuple{Int,Int}
  reference_det::Determinant{OPattern}
  mod_core_h_a::Matrix{Scalar}
  mod_core_h_b::Matrix{Scalar}
  heval_data::HEvalData              # Precomputed heval_data arrays for diagonal and Fock elements
  int1a::Matrix{Scalar}              # Reference to one-electron integrals for alpha spin
  int1b::Matrix{Scalar}              # Reference to one-electron integrals for beta spin
  int2aa::Array{Scalar,4}            # Reference to two-electron integrals for alpha-alpha spin
  int2bb::Array{Scalar,4}            # Reference to two-electron integrals for beta-beta spin
  int2ab::Array{Scalar,4}            # Reference to two-electron integrals for alpha-beta spin

  function CIPHIContext{OPattern}(fcidump::QFDump, options::CIPHIOptions = CIPHIOptions(); occa=nothing, occb=nothing) where OPattern
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
    @tensor mod_core_h_a[m,n] -= 0.5 * int2aa[m,i,i,n]
    @tensor mod_core_h_b[m,n] -= 0.5 * int2bb[m,i,i,n]

    n_alpha = (n_elec + ms2) ÷ 2
    n_beta = (n_elec - ms2) ÷ 2
    if occa !== nothing && occb !== nothing
      # User provided occupation patterns for reference determinant
      if length(occa) != n_alpha || length(occb) != n_beta
        error("Provided occupation patterns do not match n_alpha/n_beta")
      end
      reference_det = Determinant{OPattern}(occa, occb)
    else
      # HF determinant: occupy first n_alpha/n_beta orbitals
      alpha_pattern = (OPattern(1) << n_alpha) - OPattern(1)
      beta_pattern = (OPattern(1) << n_beta) - OPattern(1)
      reference_det = Determinant(alpha_pattern, beta_pattern)
    end

    if fcidump.uhf
      int1a = fcidump.int1a
      int1b = fcidump.int1b
      int2aa = fcidump.int2aa
      int2bb = fcidump.int2bb
      int2ab = fcidump.int2ab
    else
      int1a = fcidump.int1
      int1b = fcidump.int1
      int2aa = fcidump.int2
      int2bb = fcidump.int2
      int2ab = fcidump.int2
    end

    context = new{OPattern}(fcidump, options, n_orb, (n_alpha, n_beta), reference_det,
        mod_core_h_a, mod_core_h_b, HEvalData(), int1a, int1b, int2aa, int2bb, int2ab)

    # Initialize Hamiltonian terms
    init_hamiltonian_terms!(context)
    return context
  end
end

# Convenience constructor that defaults to UInt128
CIPHIContext(fcidump::QFDump, options::CIPHIOptions = CIPHIOptions(); kwargs...) = CIPHIContext{UInt128}(fcidump, options; kwargs...)

is_hermitian(ctx::Union{CIPHIContext, FCIContext}) = !is_similarity_transformed(ctx.fcidump)