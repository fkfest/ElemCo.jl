
"""
Selected CI implementation that computes H*c matrix-vector products directly
without constructing or storing full Hamiltonian matrices.

This provides:
1. Efficient P-space calculations with O(N²) scaling and O(N) memory
2. Standalone Selected CI solver capability  
3. Direct integration with Davidson solvers
4. On-the-fly matrix element evaluation using Slater-Condon rules
"""

using LinearAlgebra

# ===========================================
# Core Selected CI Data Structures  
# ===========================================
"""
    SelectedCIVector

Vector storing coefficients for a selected set of determinants only.
Much more memory efficient than full FCIVector for small selected spaces.
"""
struct SelectedCIVector
  coefficients::Vector{Scalar}          # Coefficients for selected determinants only
  n_selected::Int                       # Number of selected determinants

  function SelectedCIVector(n_selected::Int)
    new(zeros(Scalar, n_selected), n_selected)
  end
end

"""
    ExcitationInfo

Information about orbital excitation between two determinants.
Used for efficient matrix element evaluation.
"""
struct ExcitationInfo
  excitation_type::Symbol              # :same, :single_alpha, :single_beta, :double, :invalid
  orb_indices::NTuple{4, Int}         # Orbital indices involved in excitation (i,j,a,b)
  phase::Int8                         # Sign factor from orbital reordering (+1 or -1)

  function ExcitationInfo(
    excitation_type::Symbol,
    orb_indices::NTuple{4, Int} = (0, 0, 0, 0),
    phase::Int8 = Int8(1),
  )
    new(excitation_type, orb_indices, phase)
  end
end

"""
    SelectedCIDeterminants

Container for selected determinants with efficient storage.
"""
struct SelectedCIDeterminants
  determinants::Vector{Determinant}     # Selected determinants
  addresses::Vector{Address}           # Corresponding addresses in full CI space
  n_selected::Int                      # Number of selected determinants

  function SelectedCIDeterminants(determinants::Vector{Determinant}, addresses::Vector{Address})
    @assert length(determinants) == length(addresses) "Determinants and addresses must have same length"
    new(determinants, addresses, length(determinants))
  end
end

"""
    SelectedCIContext

Context for Selected CI calculations using direct H*c operations.
"""
mutable struct SelectedCIContext
  base_context::FCIContext             # Full FCI context for integrals and addressing
  selected_dets::SelectedCIDeterminants # Selected determinants and addresses

  function SelectedCIContext(base_context::FCIContext, determinants::Vector{Determinant})
    # Convert determinants to addresses
    addresses = [address_from_determinant(base_context, det) for det in determinants]
    selected_dets = SelectedCIDeterminants(determinants, addresses)
    new(base_context, selected_dets)
  end
end

# ===========================================
# Orbital Excitation Analysis
# ===========================================

"""
    analyze_determinant_excitation(det_i::Determinant, det_j::Determinant) -> ExcitationInfo

Analyze orbital excitation between two determinants.
Returns excitation type and orbital indices for matrix element evaluation.
"""
function analyze_determinant_excitation(det_i::Determinant, det_j::Determinant)::ExcitationInfo
  # Find differences in alpha and beta strings
  alpha_diff = det_i.alpha ⊻ det_j.alpha  # XOR to find differing bits
  beta_diff = det_i.beta ⊻ det_j.beta

  # Count number of differing orbitals
  n_alpha_diff = count_ones(alpha_diff)
  n_beta_diff = count_ones(beta_diff)

  # Same determinant
  if n_alpha_diff == 0 && n_beta_diff == 0
    return ExcitationInfo(:same)
  end

  # Single excitation in alpha string
  if n_alpha_diff == 2 && n_beta_diff == 0
    orb_i, orb_a = find_excitation_orbitals(det_i.alpha, det_j.alpha)
    phase = calculate_excitation_phase(det_i.alpha, orb_i, orb_a)
    return ExcitationInfo(:single_alpha, (orb_i, orb_a, 0, 0), phase)
  end

  # Single excitation in beta string  
  if n_alpha_diff == 0 && n_beta_diff == 2
    orb_i, orb_a = find_excitation_orbitals(det_i.beta, det_j.beta)
    phase = calculate_excitation_phase(det_i.beta, orb_i, orb_a)
    return ExcitationInfo(:single_beta, (orb_i, orb_a, 0, 0), phase)
  end

  # Double excitation
  if n_alpha_diff == 4 && n_beta_diff == 0
    # Double excitation in alpha
    orb_i, orb_j, orb_a, orb_b = find_double_excitation_orbitals(det_i.alpha, det_j.alpha)
    phase = calculate_double_excitation_phase(det_i.alpha, orb_i, orb_j, orb_a, orb_b)
    return ExcitationInfo(:double, (orb_i, orb_j, orb_a, orb_b), phase)
  elseif n_alpha_diff == 0 && n_beta_diff == 4
    # Double excitation in beta
    orb_i, orb_j, orb_a, orb_b = find_double_excitation_orbitals(det_i.beta, det_j.beta)
    phase = calculate_double_excitation_phase(det_i.beta, orb_i, orb_j, orb_a, orb_b)
    return ExcitationInfo(:double, (orb_i, orb_j, orb_a, orb_b), phase)
  elseif n_alpha_diff == 2 && n_beta_diff == 2
    # Mixed double excitation (alpha and beta)
    orb_i_alpha, orb_a_alpha = find_excitation_orbitals(det_i.alpha, det_j.alpha)
    orb_i_beta, orb_a_beta = find_excitation_orbitals(det_i.beta, det_j.beta)
    phase_alpha = calculate_excitation_phase(det_i.alpha, orb_i_alpha, orb_a_alpha)
    phase_beta = calculate_excitation_phase(det_i.beta, orb_i_beta, orb_a_beta)
    total_phase = phase_alpha * phase_beta
    return ExcitationInfo(:double, (orb_i_alpha, orb_i_beta, orb_a_alpha, orb_a_beta), total_phase)
  end

  # Invalid excitation (more than double)
  return ExcitationInfo(:invalid)
end

"""
    find_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern) -> (Int, Int)

Find the two orbitals involved in a single excitation i -> a.
"""
function find_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern)::Tuple{Int, Int}
  diff = pattern_i ⊻ pattern_j

  orb_i = -1  # Orbital being destroyed
  orb_a = -1  # Orbital being created

  bit_pos = 0
  while diff != 0
    if (diff & 1) != 0
      if (pattern_i & (OrbPattern(1) << bit_pos)) != 0
        orb_i = bit_pos  # This orbital is in pattern_i but not pattern_j
      else
        orb_a = bit_pos  # This orbital is in pattern_j but not pattern_i
      end
    end
    diff >>= 1
    bit_pos += 1
  end

  return (orb_i, orb_a)
end

"""
    find_double_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern) -> (Int, Int, Int, Int)

Find the four orbitals involved in a double excitation ij -> ab.
"""
function find_double_excitation_orbitals(
  pattern_i::OrbPattern,
  pattern_j::OrbPattern,
)::NTuple{4, Int}
  diff = pattern_i ⊻ pattern_j

  destroyed = Int[]  # Orbitals in pattern_i but not pattern_j
  created = Int[]    # Orbitals in pattern_j but not pattern_i

  bit_pos = 0
  while diff != 0
    if (diff & 1) != 0
      if (pattern_i & (OrbPattern(1) << bit_pos)) != 0
        push!(destroyed, bit_pos)
      else
        push!(created, bit_pos)
      end
    end
    diff >>= 1
    bit_pos += 1
  end

  @assert length(destroyed) == 2 && length(created) == 2 "Invalid double excitation"

  # Order: (i, j, a, b) where i < j and a < b
  orb_i, orb_j =
    destroyed[1] < destroyed[2] ? (destroyed[1], destroyed[2]) : (destroyed[2], destroyed[1])
  orb_a, orb_b = created[1] < created[2] ? (created[1], created[2]) : (created[2], created[1])

  return (orb_i, orb_j, orb_a, orb_b)
end

"""
    calculate_excitation_phase(pattern::OrbPattern, orb_i::Int, orb_a::Int) -> Int8

Calculate phase factor for single excitation i -> a.
"""
function calculate_excitation_phase(pattern::OrbPattern, orb_i::Int, orb_a::Int)::Int8
  # Count electrons between orb_i and orb_a
  min_orb = min(orb_i, orb_a)
  max_orb = max(orb_i, orb_a)

  n_electrons = 0
  for orb in (min_orb + 1):(max_orb - 1)
    if (pattern & (OrbPattern(1) << orb)) != 0
      n_electrons += 1
    end
  end

  # Phase is (-1)^n_electrons
  return (n_electrons % 2 == 0) ? Int8(1) : Int8(-1)
end

"""
    calculate_double_excitation_phase(pattern::OrbPattern, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Int8

Calculate phase factor for double excitation ij -> ab.

The phase is calculated by decomposing into two successive single excitations:
  1. First excitation:  i -> a, with phase φ₁
  2. Second excitation: j -> b in modified determinant, with phase φ₂
  Total phase = φ₁ × φ₂
  
This approach correctly handles the fermionic anticommutation relations.
"""
function calculate_double_excitation_phase(pattern::OrbPattern, orb_i::Int, orb_j::Int,
                                           orb_a::Int, orb_b::Int)::Int8
  # First excitation: i -> a
  phase1 = calculate_excitation_phase(pattern, orb_i, orb_a)
  
  # Create intermediate determinant: remove electron from orb_i, add to orb_a
  intermediate = pattern
  intermediate &= ~(OrbPattern(1) << orb_i)  # Remove electron from i
  intermediate |= (OrbPattern(1) << orb_a)   # Add electron to a
  
  # Second excitation: j -> b in the intermediate determinant
  phase2 = calculate_excitation_phase(intermediate, orb_j, orb_b)
  
  # Total phase is the product
  return phase1 * phase2
end

# ===========================================
# Direct Matrix Element Evaluation
# ===========================================

"""
    compute_matrix_element_direct(det_i::Determinant, det_j::Determinant, 
                                 context::FCIContext) -> Scalar

Compute ⟨det_i|Ĥ|det_j⟩ directly using orbital excitation analysis.
"""
function compute_matrix_element_direct(det_i::Determinant, det_j::Determinant, context::FCIContext)::Scalar
  excitation = analyze_determinant_excitation(det_i, det_j)
  return evaluate_matrix_element(det_i, det_j, context, excitation)
end

"""
    evaluate_matrix_element(det_i::Determinant, det_j::Determinant, 
                           context::FCIContext, excitation::ExcitationInfo) -> Scalar

Evaluate matrix element based on excitation type.
"""
function evaluate_matrix_element(det_i::Determinant, det_j::Determinant, context::FCIContext,
                                 excitation::ExcitationInfo)::Scalar
  if excitation.excitation_type == :same
    return diagonal_matrix_element(det_i, context)
  elseif excitation.excitation_type == :single_alpha || excitation.excitation_type == :single_beta
    return single_excitation_matrix_element(det_i, det_j, context, excitation)
  elseif excitation.excitation_type == :double
    return double_excitation_matrix_element(det_i, det_j, context, excitation)
  else
    return 0.0  # Invalid excitation
  end
end

"""
    diagonal_matrix_element(det::Determinant, context::FCIContext) -> Scalar

Compute diagonal matrix element ⟨det|Ĥ|det⟩.
"""
function diagonal_matrix_element(det::Determinant, context::FCIContext)::Scalar
  # Get the address and use existing diagonal computation
  addr = address_from_determinant(context, det)
  return context.diag_h.data[addr]
end

"""
    single_excitation_matrix_element(det_i::Determinant, det_j::Determinant, 
                                    context::FCIContext, excitation::ExcitationInfo) -> Scalar

Compute matrix element for single excitation.
"""
function single_excitation_matrix_element(det_i::Determinant, det_j::Determinant,
                                          context::FCIContext, excitation::ExcitationInfo)::Scalar
  orb_i = excitation.orb_indices[1] + 1  # Convert to 1-based indexing
  orb_a = excitation.orb_indices[2] + 1
  phase = excitation.phase

  # Select correct integrals based on RHF vs UHF
  is_uhf = context.fcidump.is_uhf

  # One-electron contribution: h_ia
  if excitation.excitation_type == :single_alpha
    h_element = is_uhf ? context.fcidump.h1a[orb_i, orb_a] : context.fcidump.h1[orb_i, orb_a]
  else  # single_beta
    h_element = is_uhf ? context.fcidump.h1b[orb_i, orb_a] : context.fcidump.h1[orb_i, orb_a]
  end

  # Two-electron contributions: sum over occupied orbitals in det_i
  # For single excitation i->a: ⟨det_i|Ĥ|det_j⟩ = phase * (h_ia + sum_k (ia|kk) - (ik|ka))

  if excitation.excitation_type == :single_alpha
    # Get correct alpha-alpha integrals
    h2aa = is_uhf ? context.fcidump.h2aa : context.fcidump.h2
    h2ab = is_uhf ? context.fcidump.h2ab : context.fcidump.h2
    
    # Sum over all occupied alpha orbitals in det_i (excluding orbital i)
    pattern_alpha = det_i.alpha
    bit_pos = 0
    while pattern_alpha != 0
      if (pattern_alpha & 1) != 0  # Orbital k is occupied
        k = bit_pos + 1  # Convert to 1-based
        if k != orb_i  # Don't include the orbital being removed
          h_element += h2aa[orb_i, orb_a, k, k] - h2aa[orb_i, k, k, orb_a]
        end
      end
      pattern_alpha >>= 1
      bit_pos += 1
    end

    # Sum over all occupied beta orbitals in det_i
    pattern_beta = det_i.beta
    bit_pos = 0
    while pattern_beta != 0
      if (pattern_beta & 1) != 0  # Orbital k is occupied
        k = bit_pos + 1  # Convert to 1-based
        h_element += h2ab[orb_i, orb_a, k, k]
      end
      pattern_beta >>= 1
      bit_pos += 1
    end

  else  # single_beta
    # Get correct beta-beta and alpha-beta integrals
    h2bb = is_uhf ? context.fcidump.h2bb : context.fcidump.h2
    h2ab = is_uhf ? context.fcidump.h2ab : context.fcidump.h2
    
    # Sum over all occupied alpha orbitals in det_i
    pattern_alpha = det_i.alpha
    bit_pos = 0
    while pattern_alpha != 0
      if (pattern_alpha & 1) != 0  # Orbital k is occupied
        k = bit_pos + 1  # Convert to 1-based
        # Note: For UHF, h2ab[k, k, orb_i, orb_a] = (k_α k_α | i_β a_β) = (i_β a_β | k_α k_α)
        h_element += h2ab[k, k, orb_i, orb_a]
      end
      pattern_alpha >>= 1
      bit_pos += 1
    end

    # Sum over all occupied beta orbitals in det_i (excluding orbital i)
    pattern_beta = det_i.beta
    bit_pos = 0
    while pattern_beta != 0
      if (pattern_beta & 1) != 0  # Orbital k is occupied
        k = bit_pos + 1  # Convert to 1-based
        if k != orb_i  # Don't include the orbital being removed
          h_element += h2bb[orb_i, orb_a, k, k] - h2bb[orb_i, k, k, orb_a]
        end
      end
      pattern_beta >>= 1
      bit_pos += 1
    end
  end

  return Scalar(phase) * h_element
end

"""
    double_excitation_matrix_element(det_i::Determinant, det_j::Determinant, 
                                    context::FCIContext, excitation::ExcitationInfo) -> Scalar

Compute matrix element for double excitation.
Handles both same-spin and alpha-beta mixed double excitations correctly.
"""
function double_excitation_matrix_element(det_i::Determinant, det_j::Determinant,
                                          context::FCIContext, excitation::ExcitationInfo)::Scalar
  # Determine if this is same-spin or alpha-beta mixed
  alpha_diff = det_i.alpha ⊻ det_j.alpha
  beta_diff = det_i.beta ⊻ det_j.beta
  n_alpha_diff = count_ones(alpha_diff)
  n_beta_diff = count_ones(beta_diff)

  phase = excitation.phase
  is_uhf = context.fcidump.is_uhf

  if n_alpha_diff == 4 && n_beta_diff == 0
    # Double alpha excitation: both electrons in alpha spin
    # Orbitals: (i, j, a, b) where i,j → a,b
    orb_i = excitation.orb_indices[1] + 1  # Convert to 1-based
    orb_j = excitation.orb_indices[2] + 1
    orb_a = excitation.orb_indices[3] + 1
    orb_b = excitation.orb_indices[4] + 1

    # Get correct alpha-alpha integrals
    h2aa = is_uhf ? context.fcidump.h2aa : context.fcidump.h2
    
    # Matrix element: phase * [(ia|jb) - (ib|ja)]
    h_element = h2aa[orb_i, orb_a, orb_j, orb_b] - h2aa[orb_i, orb_b, orb_j, orb_a]
    return Scalar(phase) * h_element

  elseif n_alpha_diff == 0 && n_beta_diff == 4
    # Double beta excitation: both electrons in beta spin
    # Orbitals: (i, j, a, b) where i,j → a,b
    orb_i = excitation.orb_indices[1] + 1
    orb_j = excitation.orb_indices[2] + 1
    orb_a = excitation.orb_indices[3] + 1
    orb_b = excitation.orb_indices[4] + 1

    # Get correct beta-beta integrals
    h2bb = is_uhf ? context.fcidump.h2bb : context.fcidump.h2
    
    # Matrix element: phase * [(ia|jb) - (ib|ja)]
    h_element = h2bb[orb_i, orb_a, orb_j, orb_b] - h2bb[orb_i, orb_b, orb_j, orb_a]
    return Scalar(phase) * h_element

  elseif n_alpha_diff == 2 && n_beta_diff == 2
    # Alpha-beta mixed excitation: one electron in each spin
    # Orbitals: (i_α, i_β, a_α, a_β) where i_α → a_α (alpha) and i_β → a_β (beta)
    orb_i_alpha = excitation.orb_indices[1] + 1
    orb_i_beta = excitation.orb_indices[2] + 1
    orb_a_alpha = excitation.orb_indices[3] + 1
    orb_a_beta = excitation.orb_indices[4] + 1

    # Get correct alpha-beta integrals
    h2ab = is_uhf ? context.fcidump.h2ab : context.fcidump.h2
    
    # Matrix element: phase * (i_α a_α | i_β a_β) - NO exchange term!
    h_element = h2ab[orb_i_alpha, orb_a_alpha, orb_i_beta, orb_a_beta]
    return Scalar(phase) * h_element

  else
    # Should never reach here if analyze_determinant_excitation is correct
    return 0.0
  end
end

# ===========================================
# Direct H*c Matrix-Vector Products
# ===========================================

"""
    contract_hamiltonian_selected!(result::Vector{Scalar}, input::Vector{Scalar}, 
                                  selected_ctx::SelectedCIContext, prefactor::Scalar)

Compute H*c matrix-vector product directly without storing Hamiltonian matrix.
This is the core function for Selected CI calculations.

Applies Slater-Condon rule screening: only computes matrix elements between
determinants differing by at most 2 orbital occupations (same, single, or double excitations).
"""
function contract_hamiltonian_selected!(result::Vector{Scalar}, input::Vector{Scalar},
                                        selected_ctx::SelectedCIContext, prefactor::Scalar)
  n_selected = selected_ctx.selected_dets.n_selected
  @assert length(result) == n_selected "Result vector size mismatch"
  @assert length(input) == n_selected "Input vector size mismatch"

  fill!(result, 0.0)

  for i in 1:n_selected
    det_i = selected_ctx.selected_dets.determinants[i]

    for j in 1:n_selected
      if abs(input[j]) < ThrNeglect
        continue  # Skip negligible coefficients for efficiency
      end

      det_j = selected_ctx.selected_dets.determinants[j]
      
      # Slater-Condon screening: skip if determinants differ by > 2 orbitals
      # Count differing orbitals in alpha and beta spins
      alpha_diff = det_i.alpha ⊻ det_j.alpha
      beta_diff = det_i.beta ⊻ det_j.beta
      n_alpha_diff = count_ones(alpha_diff)
      n_beta_diff = count_ones(beta_diff)
      
      # Matrix element is zero if total differences > 4
      # (4 because each single excitation changes 2 bits: remove one, add one)
      if n_alpha_diff + n_beta_diff > 4
        continue  # Skip: matrix element is exactly zero
      end
      
      h_ij = compute_matrix_element_direct(det_i, det_j, selected_ctx.base_context)
      result[i] += prefactor * h_ij * input[j]
    end
  end
end

"""
    contract_hamiltonian!(selected_ctx::SelectedCIContext, result::Vector{Scalar}, 
                         input::Vector{Scalar}, prefactor::Scalar)

Interface to existing Davidson solver infrastructure for Selected CI.
"""
function contract_hamiltonian!(selected_ctx::SelectedCIContext, result::Vector{Scalar},
                               input::Vector{Scalar}, prefactor::Scalar)
  contract_hamiltonian_selected!(result, input, selected_ctx, prefactor)
end

# ===========================================
# Selected CI Solver Integration
# ===========================================

"""
    setup_selected_ci_from_determinants!(context::FCIContext, determinants::Vector{Determinant}) -> SelectedCIContext

Create SelectedCIContext from list of determinants.
"""
function setup_selected_ci_from_determinants!(context::FCIContext, determinants::Vector{Determinant})
  return SelectedCIContext(context, determinants)
end

"""
    setup_selected_ci_from_addresses!(context::FCIContext, addresses::Vector{Address}) -> SelectedCIContext

Create SelectedCIContext from list of addresses.
"""
function setup_selected_ci_from_addresses!(context::FCIContext, addresses::Vector{Address})
  determinants = [determinant_from_address(context, addr) for addr in addresses]
  return SelectedCIContext(context, determinants)
end

"""
    project_selected_to_full!(v_full::FCIVector, v_selected::Vector{Scalar}, 
                             selected_ctx::SelectedCIContext)

Project selected CI vector onto full CI space.
"""
function project_selected_to_full!(v_full::FCIVector, v_selected::Vector{Scalar},
                                   selected_ctx::SelectedCIContext)
  fill!(v_full.data, 0.0)

  for i in 1:(selected_ctx.selected_dets.n_selected)
    addr = selected_ctx.selected_dets.addresses[i]
    v_full.data[addr] = v_selected[i]
  end

  normalize!(v_full)
end

"""
    extract_full_to_selected!(v_selected::Vector{Scalar}, v_full::FCIVector, 
                             selected_ctx::SelectedCIContext)

Extract selected CI coefficients from full CI vector.
"""
function extract_full_to_selected!(v_selected::Vector{Scalar}, v_full::FCIVector, selected_ctx::SelectedCIContext)
  for i in 1:(selected_ctx.selected_dets.n_selected)
    addr = selected_ctx.selected_dets.addresses[i]
    v_selected[i] = v_full.data[addr]
  end
end

# ===========================================
# Heat-Bath Configuration Interaction (HBCI)
# ===========================================

"""
    HeatBathCIOptions

Configuration for Heat-Bath CI selection algorithm.

Based on Holmes et al. (2016): "Heat-Bath Configuration Interaction: An Efficient 
Selected Configuration Interaction Algorithm Inspired by Heat-Bath Sampling"
"""
struct HeatBathCIOptions
  target_selection::Int              # Target number of determinants to select
  epsilon_h::Float64                # HCI selection threshold (default: 1e-4)
  epsilon_p::Float64                # CIPSI selection threshold (default: 1e-4)
  tol::Float64                      # convergence threshold for Davidson (default: 1e-8)
  max_iterations::Int               # Maximum HBCI iterations (default: 10)
  verbose::Bool                     # Print iteration details
  use_setup_phase::Bool             # Perform setup (all singles+doubles from HF)
  compute_pt2::Bool                 # Compute PT2 perturbative correction
  epsilon_pt2::Float64              # Threshold for PT2 contributions (default: 1e-6)
  n_roots::Int                      # Number of states to compute (default: 1 = ground state only)
  use_small_space_guess::Bool       # Use small-space Hamiltonian for initial guess
  small_space_size::Int             # Size of small space (0 = auto: max(100, target÷10, 5*n_roots))
  small_space_method::Symbol        # Selection method: :hybrid (energy + excitation)
  
  function HeatBathCIOptions(;
    target_selection::Int = 10000,
    epsilon_h::Float64 = 5e-4,
    epsilon_p::Float64 = epsilon_h,
    tol::Float64 = 1e-6,
    max_iterations::Int = 50,
    verbose::Bool = true,
    use_setup_phase::Bool = true,
    compute_pt2::Bool = true,
    epsilon_pt2::Float64 = 1e-6,
    n_roots::Int = 1,
    use_small_space_guess::Bool = true,
    small_space_size::Int = 0,
    small_space_method::Symbol = :hybrid
  )
    new(target_selection, epsilon_h, epsilon_p, tol, max_iterations, 
        verbose, use_setup_phase,
        compute_pt2, epsilon_pt2, n_roots,
        use_small_space_guess, small_space_size, small_space_method)
  end
end

"""
    HBCandidate

Heat-Bath selection candidate with probability and energy contribution.
"""
struct HBCandidate
  determinant::Determinant
  probability::Float64              # Selection probability: |c_I × H_IJ|² / ΔE²
  contribution::Float64             # Perturbative energy contribution
  
  function HBCandidate(det::Determinant, prob::Float64, contrib::Float64)
    new(det, prob, contrib)
  end
end

"""
    HBCISetupData

Setup data: Pre-computed and sorted double excitation matrix elements.

For each pair of orbitals {p,q}, stores a list of triplets {r,s,|H(rs←pq)|},
sorted by |H| in decreasing order. This enables efficient generation of only
important excitations during iterative selection.

Following Holmes et al. (2016), Algorithm Step IIa.
"""
struct HBCISetupData
  # For RHF: only double_excitations is used
  # For UHF: all three dictionaries are used (aa, bb, ab)
  double_excitations::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}  # RHF or UHF alpha-alpha
  double_excitations_bb::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}  # UHF beta-beta
  double_excitations_ab::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}  # RHF or UHF alpha-beta mixed
  h_doub_max::Float64              # Maximum |H(rs ← pq)| over all excitations
  
  # Precomputed h1e2 terms for efficient Fock element calculation
  # h1e2[i, p, q] = v_{pi}^{qi} - v_{pi}^{iq}
  h1e2::Array{Float64, 3}          # RHF: spatial orbitals
  h1e2_aa::Array{Float64, 3}       # UHF: alpha-alpha
  h1e2_bb::Array{Float64, 3}       # UHF: beta-beta
  h1e2_ab::Array{Float64, 3}       # UHF and RHF: alpha-beta (no exchange)
  h1e2_ba::Array{Float64, 3}       # UHF: beta-alpha (no exchange)

  is_uhf::Bool                     # Whether this is UHF data
  
  function HBCISetupData()
    new(Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}(), 
        Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}(),
        Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}(),
        0.0, 
        zeros(0,0,0), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0), zeros(0,0,0),
        false)
  end
  
  # RHF constructor
  function HBCISetupData(double_exc::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}, 
                        double_exc_ab::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}},
                        h_max::Float64, 
                        h1e2::Array{Float64, 3}, h1e2_ab::Array{Float64, 3})
    new(double_exc, 
        Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}(),
        double_exc_ab,
        h_max, 
        h1e2, zeros(0,0,0), zeros(0,0,0), h1e2_ab, zeros(0,0,0),
        false)
  end
  
  # UHF constructor
  function HBCISetupData(double_exc_aa::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}},
                        double_exc_bb::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}},
                        double_exc_ab::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}},
                        h_max::Float64,
                        h1e2_aa::Array{Float64, 3}, h1e2_bb::Array{Float64, 3},
                        h1e2_ab::Array{Float64, 3}, h1e2_ba::Array{Float64, 3})
    new(double_exc_aa, double_exc_bb, double_exc_ab, h_max, 
        zeros(0,0,0), h1e2_aa, h1e2_bb, h1e2_ab, h1e2_ba,
        true)
  end
end

"""
    PT2Options

Options for second-order perturbative correction (PT2) computation.

Following Holmes et al. (2016), Section III.B.
"""
struct PT2Options
  epsilon_pt2::Float64               # Threshold for PT2 contributions
  compute_pt2::Bool               # Whether to compute PT2 correction
  verbose::Bool                   # Print PT2 details
  
  function PT2Options(;
    epsilon_pt2::Float64 = 1e-6,
    compute_pt2::Bool = false,
    verbose::Bool = true
  )
    new(epsilon_pt2, compute_pt2, verbose)
  end
end

"""
    PT2Result

Results from PT2 perturbative correction computation.
"""
struct PT2Result
  energy_correction::Float64      # ΔE = PT2 correction
  n_external_dets::Int           # Number of external determinants contributing
  largest_contribution::Float64  # Largest single |numerator²/denominator|
  smallest_contribution::Float64 # Smallest (above threshold)
  
  function PT2Result(
    energy::Float64 = 0.0,
    n_ext::Int = 0,
    largest::Float64 = 0.0,
    smallest::Float64 = 0.0
  )
    new(energy, n_ext, largest, smallest)
  end
end

# ===========================================
# Helper Functions for Determinant Manipulation
# ===========================================

"""
    hartree_fock_determinant(ctx::FCIContext) -> Determinant

Create Hartree-Fock determinant (lowest n_elec orbitals occupied).
"""
function hartree_fock_determinant(ctx::FCIContext)::Determinant
  # Calculate n_alpha and n_beta from n_elec and n_spin
  # This is more robust than relying on fcidump.n_alpha/n_beta being set
  n_elec = ctx.fcidump.n_elec
  n_spin = ctx.fcidump.n_spin
  n_alpha = (n_elec + n_spin) ÷ 2
  n_beta = (n_elec - n_spin) ÷ 2
  
  # Occupy lowest n_alpha alpha orbitals (0-based bit positions)
  alpha_pattern = OrbPattern(0)
  for i in 0:(n_alpha-1)
    alpha_pattern |= (OrbPattern(1) << i)
  end
  
  # Occupy lowest n_beta beta orbitals (0-based bit positions)
  beta_pattern = OrbPattern(0)
  for i in 0:(n_beta-1)
    beta_pattern |= (OrbPattern(1) << i)
  end
  
  return Determinant(alpha_pattern, beta_pattern)
end

"""
    occupied_orbitals(pattern::OrbPattern, n_orb::Int) -> Vector{Int}

Get list of occupied orbital indices (0-based to match bit positions).
"""
function occupied_orbitals(pattern::OrbPattern, n_orb::Int)::Vector{Int}
  occ = Int[]
  for i in 0:(n_orb-1)
    if (pattern & (OrbPattern(1) << i)) != 0
      push!(occ, i)
    end
  end
  return occ
end

"""
    virtual_orbitals(pattern::OrbPattern, n_orb::Int) -> Vector{Int}

Get list of virtual (unoccupied) orbital indices (0-based).
"""
function virtual_orbitals(pattern::OrbPattern, n_orb::Int)::Vector{Int}
  virt = Int[]
  for i in 0:(n_orb-1)
    if (pattern & (OrbPattern(1) << i)) == 0
      push!(virt, i)
    end
  end
  return virt
end

"""
    single_excitation_alpha(det::Determinant, i::Int, a::Int) -> Determinant

Create determinant with alpha electron moved from orbital i to orbital a (0-based indices).
"""
function single_excitation_alpha(det::Determinant, i::Int, a::Int)::Determinant
  # Remove electron from orbital i, add to orbital a (0-based indexing)
  new_alpha = det.alpha & ~(OrbPattern(1) << i)  # Remove i
  new_alpha |= (OrbPattern(1) << a)              # Add a
  return Determinant(new_alpha, det.beta)
end

"""
    single_excitation_beta(det::Determinant, i::Int, a::Int) -> Determinant

Create determinant with beta electron moved from orbital i to orbital a (0-based indices).
"""
function single_excitation_beta(det::Determinant, i::Int, a::Int)::Determinant
  new_beta = det.beta & ~(OrbPattern(1) << i)
  new_beta |= (OrbPattern(1) << a)
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_alpha(det::Determinant, i::Int, j::Int, a::Int, b::Int) -> Determinant

Create determinant with alpha electrons moved from orbitals i,j to orbitals a,b (0-based).
"""
function double_excitation_alpha(det::Determinant, i::Int, j::Int, a::Int, b::Int)::Determinant
  new_alpha = det.alpha & ~(OrbPattern(1) << i)  # Remove i
  new_alpha &= ~(OrbPattern(1) << j)             # Remove j
  new_alpha |= (OrbPattern(1) << a)              # Add a
  new_alpha |= (OrbPattern(1) << b)              # Add b
  return Determinant(new_alpha, det.beta)
end

"""
    double_excitation_beta(det::Determinant, i::Int, j::Int, a::Int, b::Int) -> Determinant

Create determinant with beta electrons moved from orbitals i,j to orbitals a,b (0-based).
"""
function double_excitation_beta(det::Determinant, i::Int, j::Int, a::Int, b::Int)::Determinant
  new_beta = det.beta & ~(OrbPattern(1) << i)
  new_beta &= ~(OrbPattern(1) << j)
  new_beta |= (OrbPattern(1) << a)
  new_beta |= (OrbPattern(1) << b)
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_mixed(det::Determinant, i_alpha::Int, i_beta::Int, a_alpha::Int, a_beta::Int) -> Determinant

Create determinant with one alpha excitation i_alpha->a_alpha and one beta excitation i_beta->a_beta (0-based).
"""
function double_excitation_mixed(det::Determinant, i_alpha::Int, i_beta::Int, 
                                 a_alpha::Int, a_beta::Int)::Determinant
  new_alpha = det.alpha & ~(OrbPattern(1) << i_alpha)
  new_alpha |= (OrbPattern(1) << a_alpha)
  new_beta = det.beta & ~(OrbPattern(1) << i_beta)
  new_beta |= (OrbPattern(1) << a_beta)
  return Determinant(new_alpha, new_beta)
end

# ===========================================
# Connected Determinant Generation
# ===========================================

"""
    generate_connected_determinants!(connected::Vector{Determinant}, 
                                     det::Determinant, 
                                     ctx::FCIContext) -> Int

Generate all determinants connected to `det` by single and double excitations.
Returns number of connected determinants generated.
"""
function generate_connected_determinants!(connected::Vector{Determinant}, 
                                         det::Determinant,
                                         ctx::FCIContext)::Int
  empty!(connected)
  n_orb = ctx.fcidump.n_orb
  
  # Get occupied and virtual orbitals for alpha and beta
  alpha_occ = occupied_orbitals(det.alpha, n_orb)
  alpha_virt = virtual_orbitals(det.alpha, n_orb)
  beta_occ = occupied_orbitals(det.beta, n_orb)
  beta_virt = virtual_orbitals(det.beta, n_orb)
  
  # Alpha single excitations: i -> a
  for i in alpha_occ
    for a in alpha_virt
      new_det = single_excitation_alpha(det, i, a)
      push!(connected, new_det)
    end
  end
  
  # Beta single excitations: i -> a
  for i in beta_occ
    for a in beta_virt
      new_det = single_excitation_beta(det, i, a)
      push!(connected, new_det)
    end
  end
  
  # Alpha double excitations: ij -> ab
  for (idx_i, i) in enumerate(alpha_occ)
    for j in alpha_occ[(idx_i+1):end]  # j > i to avoid duplicates
      for (idx_a, a) in enumerate(alpha_virt)
        for b in alpha_virt[(idx_a+1):end]  # b > a to avoid duplicates
          new_det = double_excitation_alpha(det, i, j, a, b)
          push!(connected, new_det)
        end
      end
    end
  end
  
  # Beta double excitations: ij -> ab
  for (idx_i, i) in enumerate(beta_occ)
    for j in beta_occ[(idx_i+1):end]
      for (idx_a, a) in enumerate(beta_virt)
        for b in beta_virt[(idx_a+1):end]
          new_det = double_excitation_beta(det, i, j, a, b)
          push!(connected, new_det)
        end
      end
    end
  end
  
  # Mixed double excitations: i_alpha -> a_alpha, i_beta -> a_beta
  for i_alpha in alpha_occ
    for a_alpha in alpha_virt
      for i_beta in beta_occ
        for a_beta in beta_virt
          new_det = double_excitation_mixed(det, i_alpha, i_beta, a_alpha, a_beta)
          push!(connected, new_det)
        end
      end
    end
  end
  
  return length(connected)
end

"""
    generate_excitations_with_threshold!(excitations::Vector{Determinant},
                                         det::Determinant,
                                         ctx::FCIContext,
                                         setup_data::HBCISetupData,
                                         epsilon::Float64) -> Int

Generate only excitations with |H| > epsilon using pre-computed data.

This is the efficient excitation generation from Holmes et al. (2016):
1. For doubles: Use pre-sorted lists to stop early when |H| < epsilon
2. For singles: Compute on-the-fly and discard if |H| < epsilon

Time complexity: O(N_εcon × M^2) where N_εcon is number of excitations above threshold,
much faster than O(M^4) for generating all possible excitations.
"""
function generate_excitations_with_threshold!(excitations::Vector{Determinant},
                                             det::Determinant,
                                             ctx::FCIContext,
                                             setup_data::HBCISetupData,
                                             epsilon::Float64)::Int
  empty!(excitations)
  n_orb = ctx.fcidump.n_orb
  
  # Get occupied and virtual orbitals
  alpha_occ = occupied_orbitals(det.alpha, n_orb)
  alpha_virt = virtual_orbitals(det.alpha, n_orb)
  beta_occ = occupied_orbitals(det.beta, n_orb)
  beta_virt = virtual_orbitals(det.beta, n_orb)
  
  # ===========================================
  # 1. Generate double excitations using pre-computed lists
  # ===========================================
  
  # Check if we should skip all double excitations
  if epsilon <= setup_data.h_doub_max
    if !setup_data.is_uhf
      # RHF: Use single dictionary for all double excitations
      # Alpha-alpha double excitations
      for (idx_i, i) in enumerate(alpha_occ)
        for j in @view(alpha_occ[(idx_i+1):end])
          # Look up pre-sorted list for (i,j) pair
          pq_key = i < j ? (i, j) : (j, i)
          if haskey(setup_data.double_excitations, pq_key)
            for (r, s, h_val) in setup_data.double_excitations[pq_key]
              # Stop when matrix element falls below threshold
              if h_val < epsilon
                break
              end
              
              # Check if r and s are virtual (not occupied)
              if !(r in alpha_occ) && !(s in alpha_occ)
                new_det = double_excitation_alpha(det, i, j, r, s)
                push!(excitations, new_det)
              end
            end
          end
        end
      end
      
      # Beta-beta double excitations (RHF uses same integrals)
      for (idx_i, i) in enumerate(beta_occ)
        for j in @view(beta_occ[(idx_i+1):end])
          pq_key = i < j ? (i, j) : (j, i)
          if haskey(setup_data.double_excitations, pq_key)
            for (r, s, h_val) in setup_data.double_excitations[pq_key]
              if h_val < epsilon
                break
              end
              if !(r in beta_occ) && !(s in beta_occ)
                new_det = double_excitation_beta(det, i, j, r, s)
                push!(excitations, new_det)
              end
            end
          end
        end
      end

      # Mixed double excitations (alpha-beta) (use h2ab pre-computed lists, i.e., no exchange)
      for i_alpha in alpha_occ
        for i_beta in beta_occ
          pq_key = (i_alpha, i_beta)
          if haskey(setup_data.double_excitations_ab, pq_key)
            for (r, s, h_val) in setup_data.double_excitations_ab[pq_key]
              if h_val < epsilon
                break
              end
              # r is alpha virtual, s is beta virtual
              if !(r in alpha_occ) && !(s in beta_occ)
                new_det = double_excitation_mixed(det, i_alpha, i_beta, r, s)
                push!(excitations, new_det)
              end
            end
          end
        end
      end
      
    else
      # UHF: Use spin-separated dictionaries
      # Alpha-alpha double excitations (use h2aa pre-computed lists)
      for (idx_i, i) in enumerate(alpha_occ)
        for j in alpha_occ[(idx_i+1):end]
          pq_key = i < j ? (i, j) : (j, i)
          if haskey(setup_data.double_excitations, pq_key)
            for (r, s, h_val) in setup_data.double_excitations[pq_key]
              if h_val < epsilon
                break
              end
              if !(r in alpha_occ) && !(s in alpha_occ)
                new_det = double_excitation_alpha(det, i, j, r, s)
                push!(excitations, new_det)
              end
            end
          end
        end
      end
      
      # Beta-beta double excitations (use h2bb pre-computed lists)
      for (idx_i, i) in enumerate(beta_occ)
        for j in beta_occ[(idx_i+1):end]
          pq_key = i < j ? (i, j) : (j, i)
          if haskey(setup_data.double_excitations_bb, pq_key)
            for (r, s, h_val) in setup_data.double_excitations_bb[pq_key]
              if h_val < epsilon
                break
              end
              if !(r in beta_occ) && !(s in beta_occ)
                new_det = double_excitation_beta(det, i, j, r, s)
                push!(excitations, new_det)
              end
            end
          end
        end
      end
      
      # Mixed alpha-beta double excitations (use h2ab pre-computed lists)
      for i_alpha in alpha_occ
        for i_beta in beta_occ
          pq_key = (i_alpha, i_beta)
          if haskey(setup_data.double_excitations_ab, pq_key)
            for (r, s, h_val) in setup_data.double_excitations_ab[pq_key]
              if h_val < epsilon
                break
              end
              # r is alpha virtual, s is beta virtual
              if !(r in alpha_occ) && !(s in beta_occ)
                new_det = double_excitation_mixed(det, i_alpha, i_beta, r, s)
                push!(excitations, new_det)
              end
            end
          end
        end
      end
    end
  end
  
  # ===========================================
  # 2. Generate single excitations with on-the-fly filtering using Fock elements
  # ===========================================
  is_uhf = ctx.fcidump.is_uhf

  @inline function sum_h1e2(h1e2, occ, a1, i1)
    total = 0.0
    @inbounds @simd for j in occ
      total += h1e2[j+1, a1, i1]
    end
    return total
  end
  # Helper function to compute Fock matrix element f_ai
  # f_ai = h1_ai + Σ_j (v_aijj - v_ajji)
  function compute_fock_element(setup_data::HBCISetupData, 
                                occ_same::Vector{Int}, occ_opp::Vector{Int},
                                a::Int, i::Int, is_alpha::Bool)::Float64
    # Convert to 1-based indexing
    a1, i1 = a + 1, i + 1
    if is_uhf
      h1 = is_alpha ? ctx.fcidump.h1a : ctx.fcidump.h1b
      h1e2_same = is_alpha ? setup_data.h1e2_aa : setup_data.h1e2_bb
      h1e2_opp = is_alpha ? setup_data.h1e2_ab : setup_data.h1e2_ba
    else
      h1 = ctx.fcidump.h1
      h1e2_same = setup_data.h1e2
      h1e2_opp = setup_data.h1e2_ab
    end
    # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
    return h1[a1, i1] + sum_h1e2(h1e2_same, occ_same, a1, i1)
                      + sum_h1e2(h1e2_opp, occ_opp, a1, i1)
  end
  
  # Alpha single excitations
  for i in alpha_occ
    for a in alpha_virt
      # Compute Fock matrix element f_ai
      h_val = abs(compute_fock_element(setup_data, alpha_occ, beta_occ, a, i, true))
      if h_val >= epsilon
        new_det = single_excitation_alpha(det, i, a)
        push!(excitations, new_det)
      end
    end
  end
  
  # Beta single excitations
  for i in beta_occ
    for a in beta_virt
      # Compute Fock matrix element f_ai
      h_val = abs(compute_fock_element(setup_data, beta_occ, alpha_occ, i, a, false))
      if h_val >= epsilon
        new_det = single_excitation_beta(det, i, a)
        push!(excitations, new_det)
      end
    end
  end
  
  return length(excitations)
end

# ===========================================
# Diagonal Element Computation
# ===========================================

"""
    compute_diagonal_element(det::Determinant, ctx::FCIContext) -> Scalar

Compute diagonal matrix element ⟨det|H|det⟩ for a single determinant.
"""
function compute_diagonal_element(det::Determinant, ctx::FCIContext)::Scalar
  # For diagonal elements, we need:
  # 1. One-electron contributions: Σᵢ hᵢᵢ
  # 2. Two-electron contributions: Σᵢⱼ (ii|jj) - (ij|ij) for same spin, (ii|jj) for opposite spin
  
  n_orb = ctx.fcidump.n_orb
  fc = ctx.fcidump
  
  # Get occupied orbitals (0-based indexing from bit patterns)
  alpha_occ = occupied_orbitals(det.alpha, n_orb)
  beta_occ = occupied_orbitals(det.beta, n_orb)
  
  # One-electron contribution
  H_diag = 0.0
  if fc.is_uhf
    # UHF: use spin-separated integrals
    for i in alpha_occ
      H_diag += fc.h1a[i+1, i+1]  # Convert to 1-based indexing for array access
    end
    for i in beta_occ
      H_diag += fc.h1b[i+1, i+1]
    end
  else
    # RHF: use spatial integrals
    for i in alpha_occ
      H_diag += fc.h1[i+1, i+1]
    end
    for i in beta_occ
      H_diag += fc.h1[i+1, i+1]
    end
  end
  
  # Two-electron contribution
  if fc.is_uhf
    # UHF: use spin-separated integrals
    # Alpha-alpha
    for i in alpha_occ
      for j in alpha_occ
        H_diag += 0.5 * fc.h2aa[i+1, i+1, j+1, j+1]  # Coulomb
        if i != j
          H_diag -= 0.5 * fc.h2aa[i+1, j+1, j+1, i+1]  # Exchange
        end
      end
    end
    
    # Beta-beta
    for i in beta_occ
      for j in beta_occ
        H_diag += 0.5 * fc.h2bb[i+1, i+1, j+1, j+1]  # Coulomb
        if i != j
          H_diag -= 0.5 * fc.h2bb[i+1, j+1, j+1, i+1]  # Exchange
        end
      end
    end
    
    # Alpha-beta (no exchange)
    for i in alpha_occ
      for j in beta_occ
        H_diag += fc.h2ab[i+1, i+1, j+1, j+1]  # Coulomb only
      end
    end
  else
    # RHF: use spatial integrals
    # Alpha-alpha
    for i in alpha_occ
      for j in alpha_occ
        H_diag += 0.5 * fc.h2[i+1, i+1, j+1, j+1]  # Coulomb
        if i != j
          H_diag -= 0.5 * fc.h2[i+1, j+1, j+1, i+1]  # Exchange
        end
      end
    end
    
    # Beta-beta
    for i in beta_occ
      for j in beta_occ
        H_diag += 0.5 * fc.h2[i+1, i+1, j+1, j+1]  # Coulomb
        if i != j
          H_diag -= 0.5 * fc.h2[i+1, j+1, j+1, i+1]  # Exchange
        end
      end
    end
    
    # Alpha-beta (no exchange)
    for i in alpha_occ
      for j in beta_occ
        H_diag += fc.h2[i+1, i+1, j+1, j+1]  # Coulomb only
      end
    end
  end
  
  return Scalar(H_diag)
end

# ===========================================
# Heat-Bath Probability Computation
# ===========================================

"""
    compute_heatbath_probabilities!(candidates::Vector{HBCandidate},
                                    variational_dets::Vector{Determinant},
                                    variational_coeffs::Vector{Float64},
                                    ctx::FCIContext,
                                    E_current::Float64,
                                    setup_data::Union{HBCISetupData,Nothing}=nothing,
                                    epsilon::Float64) -> Float64

Compute Heat-Bath selection probabilities for all candidates.

If setup_data is provided, uses efficient excitation generation
with threshold-based filtering. Otherwise, generates all connected determinants.

Returns total selection probability sum.
"""
function compute_heatbath_probabilities!(candidates::Vector{HBCandidate},
                                        variational_dets::Vector{Determinant},
                                        variational_coeffs::Vector{Float64},
                                        ctx::FCIContext,
                                        E_current::Float64,
                                        setup_data::Union{HBCISetupData,Nothing}=nothing,
                                        epsilon::Float64=1e-10)::Float64
  empty!(candidates)
  
  # Get all connected determinants from variational space
  connected = Set{Determinant}()
  temp_buffer = Determinant[]
  
  if setup_data !== nothing
    # Setup enabled: Use efficient threshold-based excitation generation
    for (i, det) in enumerate(variational_dets)
      c_I = variational_coeffs[i]
      if abs(c_I) < 1e-10
        continue  # Skip negligible coefficients
      end
      eps = epsilon / abs(c_I)
      generate_excitations_with_threshold!(temp_buffer, det, ctx, setup_data, eps)
      union!(connected, temp_buffer)
    end
  else
    # Setup disabled: Generate all connected determinants
    for det in variational_dets
      generate_connected_determinants!(temp_buffer, det, ctx)
      union!(connected, temp_buffer)
    end
  end
  # Remove determinants already in variational space
  variational_set = Set(variational_dets)
  setdiff!(connected, variational_set)
  
  println("Generated $(length(connected)) connected determinants from variational space of size $(length(variational_dets))") 
  total_prob = 0.0
  
  for det_J in connected
    # Compute H_JJ (diagonal element)
    H_JJ = compute_diagonal_element(det_J, ctx)
    ΔE_J = E_current - H_JJ
    
    # Compute perturbative contribution: sum over I in variational space
    sum_term = 0.0
    for (i, det_I) in enumerate(variational_dets)
      # Slater-Condon screening: skip if determinants differ by > 2 orbitals
      # Count differing orbitals in alpha and beta spins
      alpha_diff = det_I.alpha ⊻ det_J.alpha
      beta_diff = det_I.beta ⊻ det_J.beta
      n_alpha_diff = count_ones(alpha_diff)
      n_beta_diff = count_ones(beta_diff)
      
      # Matrix element is zero if total differences > 4
      # (4 because each single excitation changes 2 bits: remove one, add one)
      if n_alpha_diff + n_beta_diff > 4
        continue  # Skip: matrix element is exactly zero
      end
      c_I = variational_coeffs[i]
      H_IJ = compute_matrix_element_direct(det_I, det_J, ctx)
      sum_term += c_I * H_IJ
    end
    
    # Selection probability: |Σ c_I H_IJ|² / ΔE²
    # Add small epsilon for numerical stability
    prob_J = abs2(sum_term) / (ΔE_J^2 + 1e-10)
    contrib_J = prob_J * ΔE_J  # Perturbative energy contribution
    #TODO: use contrib_J to calculate the PT2 correction later
    push!(candidates, HBCandidate(det_J, prob_J, contrib_J))
    total_prob += prob_J
  end
  
  return total_prob
end

"""
    compute_heatbath_probabilities_multistate!(candidates::Vector{HBCandidate},
                                              variational_dets::Vector{Determinant},
                                              variational_coeffs::Matrix{Float64},
                                              ctx::FCIContext,
                                              E_states::Vector{Float64},
                                              setup_data::Union{HBCISetupData,Nothing}=nothing,
                                              epsilon::Float64=1e-10)::Float64

Compute Heat-Bath selection probabilities for multiple states simultaneously.

For each candidate determinant, computes contributions from ALL states and uses
the maximum probability across states (state-max selection strategy).

# Arguments
- `candidates`: Output vector to store candidates
- `variational_dets`: Current variational space determinants
- `variational_coeffs`: Matrix (n_dets × n_states) of coefficients for all states
- `ctx`: FCI context
- `E_states`: Vector of energies for all states
- `setup_data`: Optional setup data
- `epsilon`: Threshold for excitation generation

# Returns
- Total selection probability sum across all candidates
"""
function compute_heatbath_probabilities_multistate!(candidates::Vector{HBCandidate},
                                                   variational_dets::Vector{Determinant},
                                                   variational_coeffs::Matrix{Float64},
                                                   ctx::FCIContext,
                                                   E_states::Vector{Float64},
                                                   setup_data::Union{HBCISetupData,Nothing}=nothing,
                                                   epsilon::Float64=1e-10)::Float64
  empty!(candidates)
  
  n_states = length(E_states)
  @assert size(variational_coeffs, 2) == n_states "Coefficient matrix must have n_states columns"
  
  # Get all connected determinants from variational space
  connected = Set{Determinant}()
  temp_buffer = Determinant[]
 
  # Helper to get maximum absolute coefficient for a determinant allocfree
  function absmax(a, i)
    am = 0.0
    for j in axes(a,2)
      am = max(am, abs(a[i,j]))
    end
    return am
  end
  if setup_data !== nothing
    # Setup enabled: Use efficient threshold-based excitation generation
    for (i, det) in enumerate(variational_dets)
      abs_c_I = absmax(variational_coeffs, i)
      if abs_c_I < 1e-10
        continue  # Skip negligible coefficients
      end
      eps = epsilon / abs_c_I
      generate_excitations_with_threshold!(temp_buffer, det, ctx, setup_data, eps)
      union!(connected, temp_buffer)
    end
  else
    # Setup disabled: Generate all connected determinants
    for det in variational_dets
      generate_connected_determinants!(temp_buffer, det, ctx)
      union!(connected, temp_buffer)
    end
  end
  
  # Remove determinants already in variational space
  variational_set = Set(variational_dets)
  setdiff!(connected, variational_set)
  
  total_prob = 0.0
  
  for det_J in connected
    # Compute H_JJ (diagonal element) once
    H_JJ = compute_diagonal_element(det_J, ctx)
    
    # Compute probabilities for each state
    max_prob = 0.0
    max_contrib = 0.0
    
    sum_terms = zeros(Float64, n_states)
    # Compute perturbative contribution for each state: sum over I in variational space
    for (i, det_I) in enumerate(variational_dets)
      # Slater-Condon screening: skip if determinants differ by > 2 orbitals
      # Count differing orbitals in alpha and beta spins
      alpha_diff = det_I.alpha ⊻ det_J.alpha
      beta_diff = det_I.beta ⊻ det_J.beta
      n_alpha_diff = count_ones(alpha_diff)
      n_beta_diff = count_ones(beta_diff)
    
      # Matrix element is zero if total differences > 4
      # (4 because each single excitation changes 2 bits: remove one, add one)
      if n_alpha_diff + n_beta_diff > 4
        continue  # Skip: matrix element is exactly zero
      end
      H_IJ = compute_matrix_element_direct(det_I, det_J, ctx)
      for state in 1:n_states
        c_I = variational_coeffs[i, state]
        sum_terms[state] += c_I * H_IJ
      end
    end
      
    for state in 1:n_states
      ΔE_J = E_states[state] - H_JJ
      # Selection probability for this state: |Σ c_I H_IJ|² / ΔE²
      prob_state = abs2(sum_terms[state]) / (ΔE_J^2 + 1e-10)
      contrib_state = prob_state * ΔE_J  # Perturbative energy contribution
      
      # Keep maximum across states (state-max selection)
      if prob_state > max_prob
        max_prob = prob_state
        max_contrib = contrib_state
      end
    end
    
    push!(candidates, HBCandidate(det_J, max_prob, max_contrib))
    total_prob += max_prob
  end
  
  return total_prob
end

# ===========================================
# Determinant Selection
# ===========================================

"""
    select_determinants_heatbath!(selected::Vector{Determinant},
                                 candidates::Vector{HBCandidate},
                                 options::HeatBathCIOptions) -> Int

Select determinants from candidates using Heat-Bath sampling.
Returns number of determinants selected.
"""
function select_determinants_heatbath!(selected::Vector{Determinant},
                                      candidates::Vector{HBCandidate},
                                      options::HeatBathCIOptions)::Int
  empty!(selected)
  return select_deterministic!(selected, candidates, options)
end

"""
    select_deterministic!(selected::Vector{Determinant},
                         candidates::Vector{HBCandidate},
                         options::HeatBathCIOptions) -> Int

Deterministic selection: select top-N by probability.
"""
function select_deterministic!(selected::Vector{Determinant},
                              candidates::Vector{HBCandidate},
                              options::HeatBathCIOptions)::Int
  # Sort by probability (descending)
  sort!(candidates, by=c->c.probability, rev=true)
  
  # Select determinants above threshold or until target reached
  # use square of epsilon_p to match probability definition (T_2^2)
  epsilon = options.epsilon_p^2
  n_selected = 0
  for candidate in candidates
    if n_selected >= options.target_selection
      break
    end
    if candidate.probability > epsilon
      push!(selected, candidate.determinant)
      n_selected += 1
    end
  end
  
  return n_selected
end

# ===========================================
# Main HBCI Iteration Loop
# ===========================================

"""
    setup_hbci!(ctx::FCIContext) -> HBCISetupData

Setup: Pre-compute and store sorted double excitation matrix elements.

For each pair of orbitals {p,q}, computes H(rs ← pq) for all distinct {r,s} pairs
that don't include {p,q}, and stores them sorted by |H| in decreasing order.

This enables efficient generation of only important excitations,
avoiding computation of matrix elements that would be below threshold.

Algorithm from Holmes et al. (2016), IIa:
- Time complexity: O(M^4 log M)
- Space complexity: O(M^4)
where M is the number of orbitals.
"""
function setup_hbci!(ctx::FCIContext)::HBCISetupData
  n_orb = ctx.fcidump.n_orb
  is_uhf = ctx.fcidump.is_uhf
  
  if !is_uhf
    # RHF case: use standard h2 integrals
    return setup_hbci_rhf!(ctx)
  else
    # UHF case: use spin-separated integrals
    return setup_hbci_uhf!(ctx)
  end
end

function gen_triplets_list(n_orb::Int, h2::Array{Float64,4})
  double_exc_lists = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}()
  h_doub_max = 0.0
  
  # Loop over all pairs of orbitals {p, q} (0-based)
  for p in 0:(n_orb-1)
    for q in (p+1):(n_orb-1)  # Only consider p < q to avoid duplicates
      # List of triplets {r, s, |H(rs ← pq)|} for this (p,q) pair
      triplets = Tuple{Int,Int,Float64}[]
      
      # Loop over all distinct pairs of orbitals {r, s} that don't include {p, q}
      for r in 0:(n_orb-1)
        if r == p || r == q
          continue
        end
        for s in (r+1):(n_orb-1)  # Only consider r < s to avoid duplicates
          if s == p || s == q
            continue
          end
          
          # Compute antisymmetrized two-electron integral <pq||rs>
          # Using same convention as double_excitation_matrix_element:
          # Matrix element for double excitation p,q → r,s is (pr|qs) - (ps|qr)
          # Note: h2[i,a,j,b] represents integral (ia|jb)
          # Convert to 1-based for array indexing
          h_val = abs(h2[p+1, r+1, q+1, s+1] - h2[p+1, s+1, q+1, r+1])
          
          if h_val > 1e-14  # Skip negligible matrix elements
            push!(triplets, (r, s, h_val))
            h_doub_max = max(h_doub_max, h_val)
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(triplets, by=x->x[3], rev=true)
      
      # Store sorted list for this (p,q) pair (both keys and values use 0-based indices)
      double_exc_lists[(p, q)] = triplets
    end
  end
  return double_exc_lists, h_doub_max
end

function gen_triplets_list_ab(n_orb::Int, h2ab::Array{Float64,4})
  double_exc_ab_lists = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64}}}()
  h_doub_max = 0.0
  
  # Loop over all pairs of orbitals {p, q} (0-based)
  # For mixed excitations, we don't need antisymmetrization (different spins)
  for p in 0:(n_orb-1)
    for q in 0:(n_orb-1)  # Note: can have p >= q for mixed
      if p == q; continue; end  # Skip same orbital pairs
      
      triplets = Tuple{Int,Int,Float64}[]
      
      for r in 0:(n_orb-1)
        if r == p; continue; end  # Alpha r cannot equal alpha p
        for s in 0:(n_orb-1)
          if s == q; continue; end  # Beta s cannot equal beta q
          
          # Mixed integral (pr|qs)_αβ (no antisymmetrization for different spins)
          h_val = abs(h2ab[p+1, r+1, q+1, s+1])
          
          if h_val > 1e-14
            push!(triplets, (r, s, h_val))
            h_doub_max = max(h_doub_max, h_val)
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(triplets, by=x->x[3], rev=true)
      double_exc_ab_lists[(p, q)] = triplets
    end
  end
  return double_exc_ab_lists, h_doub_max
end
"""
    setup_hbci_rhf!(ctx::FCIContext) -> HBCISetupData

Setup for RHF systems using spatial orbital integrals.
"""
function setup_hbci_rhf!(ctx::FCIContext)::HBCISetupData
  n_orb = ctx.fcidump.n_orb
  
  # Dictionary to store sorted lists for each (p,q) pair
  # Note: Using 0-based indices to match occupied_orbitals/virtual_orbitals convention
  double_exc_lists, h_doub_max = gen_triplets_list(n_orb, ctx.fcidump.h2)
  double_exc_ab_lists, h_doub_max_ab = gen_triplets_list_ab(n_orb, ctx.fcidump.h2)
  h_doub_max = max(h_doub_max, h_doub_max_ab)
  
  # Precompute h1e2 terms for efficient Fock element calculation
  # h1e2[i, p, q] = v_{pi}^{qi} - v_{pi}^{iq} = (pq|ii) - (pi|iq)
  h1e2 = zeros(Float64, n_orb, n_orb, n_orb)
  h1e2_ab = zeros(Float64, n_orb, n_orb, n_orb)
  for i in 1:n_orb, p in 1:n_orb, q in 1:n_orb
    h1e2[i, p, q] = ctx.fcidump.h2[p, q, i, i] - ctx.fcidump.h2[p, i, i, q]
    h1e2_ab[i, p, q] = ctx.fcidump.h2[p, q, i, i]  # (without exchange for opposite spins)
  end
  return HBCISetupData(double_exc_lists, double_exc_ab_lists, h_doub_max, h1e2, h1e2_ab)
end

"""
    setup_hbci_uhf!(ctx::FCIContext) -> HBCISetupData

Setup for UHF systems using spin-separated integrals.
Handles three types of double excitations:
- Alpha-alpha (using h2aa)
- Beta-beta (using h2bb)  
- Mixed alpha-beta (using h2ab)
"""
function setup_hbci_uhf!(ctx::FCIContext)::HBCISetupData
  n_orb = ctx.fcidump.n_orb
  
  # Three dictionaries for the three types of double excitations
  double_exc_aa, h_doub_max_aa = gen_triplets_list(n_orb, ctx.fcidump.h2aa)
  double_exc_bb, h_doub_max_bb = gen_triplets_list(n_orb, ctx.fcidump.h2bb)
  double_exc_ab, h_doub_max_ab = gen_triplets_list_ab(n_orb, ctx.fcidump.h2ab)
  h_doub_max = max(h_doub_max_aa, h_doub_max_bb, h_doub_max_ab)

  # Precompute h1e2 terms for efficient Fock element calculation (UHF)
  # For alpha: h1e2_aa[i, p, q] = v_{pi}^{qi}_αα - v_{pi}^{iq}_αα 
  # For beta:  h1e2_bb[i, p, q] = v_{pi}^{qi}_ββ - v_{pi}^{iq}_ββ 
  # For mixed: h1e2_ab[i, p, q] = v_{pi}^{qi}_αβ (no exchange for different spins)
  # For mixed: h1e2_ba[i, p, q] = v_{ip}^{iq}_αβ (no exchange for different spins)
  h1e2_aa = zeros(Float64, n_orb, n_orb, n_orb)
  h1e2_bb = zeros(Float64, n_orb, n_orb, n_orb)
  h1e2_ab = zeros(Float64, n_orb, n_orb, n_orb)
  h1e2_ba = zeros(Float64, n_orb, n_orb, n_orb)
  
  for i in 1:n_orb, p in 1:n_orb, q in 1:n_orb
    h1e2_aa[i, p, q] = ctx.fcidump.h2aa[p, q, i, i] - ctx.fcidump.h2aa[p, i, i, q]
    h1e2_bb[i, p, q] = ctx.fcidump.h2bb[p, q, i, i] - ctx.fcidump.h2bb[p, i, i, q]
    h1e2_ab[i, p, q] = ctx.fcidump.h2ab[p, q, i, i]  # No exchange for mixed spin
    h1e2_ba[i, p, q] = ctx.fcidump.h2ab[i, i, p, q]  # No exchange for mixed spin
  end
  
  return HBCISetupData(double_exc_aa, double_exc_bb, double_exc_ab, h_doub_max,
                      h1e2_aa, h1e2_bb, h1e2_ab, h1e2_ba)
end

"""
    compute_pt2_correction!(ctx, variational_dets, coefficients, E_var, setup_data, options)

Compute second-order perturbative correction to variational energy.

Following Holmes et al. (2016), Section III.B:
    ΔE = ∑_k [ (∑_i H_ki c_i)² / (E⁽⁰⁾ - H_kk) ]

where k runs over external determinants (not in variational space) and
i runs over internal determinants (in variational space).

Uses adaptive thresholding: for each internal det i with coefficient c_i,
generate external dets k with |H_ki| > ε_PT2/|c_i|. This focuses effort
on important contributions.

Returns PT2Result with energy correction and diagnostics.
"""
function compute_pt2_correction!(
  ctx::FCIContext,
  variational_dets::Vector{Determinant},
  coefficients::Vector{Float64},
  E_variational::Float64,
  setup_data::Union{HBCISetupData, Nothing},
  options::PT2Options
)::PT2Result
  
  if !options.compute_pt2
    return PT2Result()
  end
  
  if options.verbose
    println("\n" * "="^70)
    println("Computing PT2 Perturbative Correction")
    println("="^70)
    println("  Variational energy: $E_variational Ha")
    println("  Threshold ε_PT2: $(options.epsilon_pt2)")
    println("  Variational space size: $(length(variational_dets))")
  end
  
  # Dictionary to accumulate contributions: det_k => ∑_i H_ki c_i
  external_contributions = Dict{Determinant, Float64}()
  
  # Create set of variational determinants for fast lookup
  variational_set = Set(variational_dets)
  
  # Buffer for generated excitations
  connected = Vector{Determinant}()
  
  # Step 1: Generate external determinants and accumulate contributions
  n_internal_processed = 0
  total_excitations_generated = 0
  
  for (i, det_i) in enumerate(variational_dets)
    c_i = coefficients[i]
    
    # Skip negligible coefficients
    if abs(c_i) < 1e-12
      continue
    end
    
    n_internal_processed += 1
    
    # Adaptive threshold: ε = ε_PT2 / |c_i|
    # Larger |c_i| → smaller ε → generate more excitations
    epsilon_adaptive = options.epsilon_pt2 / abs(c_i)
    
    # Generate connected determinants with |H_ki| > epsilon_adaptive
    empty!(connected)
    if setup_data !== nothing
      # Use efficient setup if available
      generate_excitations_with_threshold!(connected, det_i, ctx, setup_data, epsilon_adaptive)
    else
      # Fallback: generate all connected determinants (slower)
      generate_connected_determinants!(connected, det_i, ctx)
    end
    
    total_excitations_generated += length(connected)
    
    for det_k in connected
      # Skip if det_k is in variational space
      if det_k in variational_set
        continue
      end
      
      # Compute H_ki
      H_ki = compute_matrix_element_direct(det_i, det_k, ctx)
      contribution = H_ki * c_i
      
      # Only accumulate if |contribution| >= ε_PT2
      if abs(contribution) >= options.epsilon_pt2
        external_contributions[det_k] = get(external_contributions, det_k, 0.0) + contribution
      end
    end
  end
  
  if options.verbose
    println("  Internal dets processed: $n_internal_processed / $(length(variational_dets))")
    println("  Total excitations generated: $total_excitations_generated")
    println("  Unique external dets: $(length(external_contributions))")
  end
  
  # Step 2 & 3: Compute PT2 energy
  ΔE = 0.0
  contributions = Float64[]
  
  for (det_k, numerator) in external_contributions
    H_kk = compute_diagonal_element(det_k, ctx)
    denominator = E_variational - H_kk
    
    # Guard against near-zero denominators
    if abs(denominator) < 1e-12
      @warn "Near-zero denominator in PT2: |E_var - H_kk| = $(abs(denominator)). Skipping."
      continue
    end
    
    # Compute contribution to PT2 energy
    pt2_term = numerator^2 / denominator
    ΔE += pt2_term
    push!(contributions, abs(pt2_term))
  end
  
  # Diagnostics
  largest = isempty(contributions) ? 0.0 : maximum(contributions)
  smallest = isempty(contributions) ? 0.0 : minimum(contributions)
  
  if options.verbose
    println("  PT2 correction: $ΔE Ha")
    println("  Total energy (VAR+PT2): $(E_variational + ΔE) Ha")
    println("  Largest contribution: $largest")
    println("  Smallest contribution: $smallest")
    if !isempty(contributions)
      println("  Mean contribution: $(sum(contributions) / length(contributions))")
    end
    println("="^70)
  end
  
  return PT2Result(ΔE, length(external_contributions), largest, smallest)
end

"""
    run_heatbath_ci!(ctx::FCIContext, options::HeatBathCIOptions) 
      -> (Vector{Float64}, Matrix{Float64}, Vector{Determinant}, PT2Result)

Run Heat-Bath CI calculation with support for multiple states.

# Arguments
- `ctx`: FCI context
- `options`: Heat-Bath CI options (including n_roots for multi-state)

# Returns
- `energies`: Vector of length n_roots with total energies (electronic + nuclear)
- `coefficients`: Matrix (n_dets × n_roots) with CI coefficients for all states
- `variational_dets`: Vector of determinants in final variational space
- `pt2_result`: PT2 correction result (currently only for ground state)

# Notes
- For n_roots=1 (default), uses single-state selection strategy
- For n_roots>1, uses multi-state selection with state-maximum probability
- PT2 correction currently only computed for ground state
"""
function run_heatbath_ci!(ctx::FCIContext, options::HeatBathCIOptions)::Tuple{Vector{Scalar}, Matrix{Scalar}, Vector{Determinant}, PT2Result}
  if options.verbose
    println("\n" * "="^70)
    println("Heat-Bath Configuration Interaction (HBCI)")
    println("="^70)
    println("Target selection: $(options.target_selection)")
    println("Selection threshold (εₕ): $(options.epsilon_h)")
    println("Selection threshold (εₚ): $(options.epsilon_p)")
    println("PT2 threshold (εₚₜ₂): $(options.epsilon_pt2)")
    println("Number of states (n_roots): $(options.n_roots)")
    if options.n_roots > 1
      println("Multi-state selection: State-maximum probability")
    end
    println("="^70)
  end
  
  # Initialization
  hf_det = hartree_fock_determinant(ctx)
  
  # Setup (if enabled)
  # Pre-compute and store sorted double excitation matrix elements
  setup_data = nothing
  if options.use_setup_phase
    if options.verbose
      println("\nSetup - Pre-computing double excitation matrix elements")
      println("  Computing and sorting |H(rs ← pq)| for all orbital pairs...")
    end
    
    setup_data = setup_hbci!(ctx)
    
    if options.verbose
      n_pairs = length(setup_data.double_excitations)
      total_triplets = sum(length(v) for v in values(setup_data.double_excitations))
      println("  Stored $(n_pairs) (p,q) pairs with $(total_triplets) total (r,s) triplets")
      println("  Maximum |H_doub|: $(setup_data.h_doub_max)")
    end
  end
  
  # Enhanced initial guess using small-space Hamiltonian (if enabled)
  variational_dets = Determinant[]
  E_init_vec = Float64[]
  
  if options.use_small_space_guess && options.n_roots > 1
    # Use small-space Hamiltonian diagonalization for better initial guess
    if options.verbose
      println("\nInitialization (Small-Space Method)")
      println("  Using small-space Hamiltonian for multi-state initial guess")
    end
    
    # Determine small-space size (adaptive or user-specified)
    small_space_size = options.small_space_size > 0 ? 
                       options.small_space_size : 
                       max(100, options.target_selection ÷ 10, 5 * options.n_roots)
    
    # Generate initial guess from small-space diagonalization
    small_space_result = initialize_multistate_from_small_space(
      ctx, options.target_selection, options.n_roots
    )
    
    # Start with all small-space determinants
    variational_dets = copy(small_space_result.determinants)
    
    # Initial energies from small-space diagonalization
    E_init_vec = small_space_result.eigenvalues .+ ctx.fcidump.e_nuc
    
    if options.verbose
      println("\n  Small-space initial guess:")
      println("    Space size: $(small_space_result.n_small) determinants")
      println("    Initial energies ($(options.n_roots) states):")
      for (i, E) in enumerate(E_init_vec)
        println("      State $i: $E Hartree")
      end
    end
  else
    # Traditional initialization: Start with HF determinant only
    if options.verbose
      println("\nInitialization (HF Reference)")
      println("  Starting from HF reference determinant")
    end
    
    push!(variational_dets, hf_det)
    
    # Get initial HF energy (all states)
    selected_ctx_hf = SelectedCIContext(ctx, variational_dets)
    E_electronic_hf_vec, _ = diagonalize_selected_space(selected_ctx_hf, n_roots=options.n_roots)
    E_init_vec = E_electronic_hf_vec .+ ctx.fcidump.e_nuc
    
    if options.verbose
      if options.n_roots == 1
        println("  HF reference energy: $(E_init_vec[1]) Hartree")
      else
        println("  HF reference energies ($(options.n_roots) states):")
        for (i, E) in enumerate(E_init_vec)
          println("    State $i: $E Hartree")
        end
      end
    end
  end
  
  if options.verbose
    println("\nIterative perturbative selection")
  end
  
  E_prev_vec = copy(E_init_vec)
  previous_eigenvectors = nothing  # Track previous eigenvectors for warm start
  converged = false
  
  for iter in 1:options.max_iterations
    if options.verbose
      println("\nHBCI Iteration $iter:")
      println("  Current space size: $(length(variational_dets)) determinants")
    end
    
    # 1. Diagonalize Hamiltonian in current space (all requested states)
    selected_ctx = SelectedCIContext(ctx, variational_dets)
    
    E_electronic_vec, coeffs_matrix = diagonalize_selected_space(selected_ctx, 
                                                                 n_roots=options.n_roots,
                                                                 previous_vectors=previous_eigenvectors)
    E_current_vec = E_electronic_vec .+ ctx.fcidump.e_nuc  # Add nuclear repulsion
    
    if options.verbose
      if options.n_roots == 1
        println("  Energy: $(E_current_vec[1]) Hartree")
      else
        println("  Energies ($(options.n_roots) states):")
        for (i, E) in enumerate(E_current_vec)
          println("    State $i: $E Hartree")
        end
      end
    end
    
    # 2. Check convergence (all states must converge)
    ΔE_vec = abs.(E_current_vec .- E_prev_vec)
    ΔE_max = maximum(ΔE_vec)
    
    if options.verbose
      if options.n_roots == 1
        println("  ΔE: $(ΔE_vec[1])")
      else
        println("  ΔE (max): $ΔE_max")
        for (i, ΔE) in enumerate(ΔE_vec)
          println("    State $i: $ΔE")
        end
      end
    end
    
    if ΔE_max < options.tol && length(variational_dets) >= options.target_selection
      if options.verbose
        println("  ✓ Converged! max(ΔE) = $ΔE_max < $(options.tol)")
      end
      converged = true
      break
    end
    
    # 3. Generate candidates and compute probabilities
    candidates = HBCandidate[]
    if options.n_roots == 1
      # Single-state: use original single-state function
      total_prob = compute_heatbath_probabilities!(candidates, variational_dets, coeffs_matrix[:,1], ctx, 
                                                  E_electronic_vec[1], setup_data, options.epsilon_h)
    else
      # Multi-state: use new multi-state function
      total_prob = compute_heatbath_probabilities_multistate!(candidates, variational_dets, coeffs_matrix, ctx, 
                                                              E_electronic_vec, setup_data, options.epsilon_h)
    end
    
    if options.verbose
      println("  Generated $(length(candidates)) candidate determinants")
      println("  Total selection probability: $total_prob")
    end
    
    if isempty(candidates)
      if options.verbose
        println("  No new candidates found. Converged.")
      end
      converged = true
      break
    end
    
    # 4. Select new determinants
    new_dets = Determinant[]
    n_new = select_determinants_heatbath!(new_dets, candidates, options)
    
    if options.verbose
      println("  Selected $n_new new determinants")
    end
    
    if isempty(new_dets)
      if options.verbose
        println("  No new determinants selected. Converged.")
      end
      converged = true
      break
    end
    
    # 5. Update variational space
    append!(variational_dets, new_dets)
    unique!(variational_dets)
    
    E_prev_vec = copy(E_current_vec)
    # Save eigenvectors for next iteration as warm start
    previous_eigenvectors = coeffs_matrix
  end
  
  if !converged
    @warn "HBCI did not converge in $(options.max_iterations) iterations"
  end
  
  # Final diagonalization
  if options.verbose
    println("\nFinal diagonalization with $(length(variational_dets)) determinants...")
  end
  
  selected_ctx = SelectedCIContext(ctx, variational_dets)
  E_electronic_vec, coeffs_final_matrix = diagonalize_selected_space(selected_ctx, 
                                                                     n_roots=options.n_roots,
                                                                     previous_vectors=previous_eigenvectors)
  
  # Add nuclear repulsion energy for total energy
  E_final_vec = E_electronic_vec .+ ctx.fcidump.e_nuc
  
  if options.verbose
    println("="^70)
    println("HBCI Complete!")
    if options.n_roots == 1
      println("Electronic energy: $(E_electronic_vec[1]) Hartree")
      println("Nuclear repulsion: $(ctx.fcidump.e_nuc) Hartree")
      println("Total energy: $(E_final_vec[1]) Hartree")
    else
      println("Electronic energies ($(options.n_roots) states):")
      for (i, E) in enumerate(E_electronic_vec)
        println("  State $i: $E Hartree")
      end
      println("Nuclear repulsion: $(ctx.fcidump.e_nuc) Hartree")
      println("Total energies:")
      for (i, E) in enumerate(E_final_vec)
        println("  State $i: $E Hartree")
      end
    end
    println("Final space size: $(length(variational_dets)) determinants")
    println("="^70)
  end
  
  # Compute PT2 correction if requested (only for ground state currently)
  pt2_result = PT2Result()
  if options.compute_pt2
    pt2_options = PT2Options(
      epsilon_pt2 = options.epsilon_pt2,
      compute_pt2 = true,
      verbose = options.verbose
    )
    
    # Note: PT2 correction currently only computed for ground state (state 1)
    pt2_result = compute_pt2_correction!(
      ctx, variational_dets, coeffs_final_matrix[:,1], E_electronic_vec[1], setup_data, pt2_options
    )
    
    E_total_with_pt2 = E_final_vec[1] + pt2_result.energy_correction
    
    if options.verbose
      println("\nFinal Energies (Ground State with PT2):")
      println("  Variational:     $(E_final_vec[1]) Ha")
      println("  PT2 correction:  $(pt2_result.energy_correction) Ha")
      println("  Total (VAR+PT2): $E_total_with_pt2 Ha")
      if options.n_roots > 1
        println("  Note: PT2 currently only computed for ground state")
      end
    end
  end
  
  # Return format: energies (vector), coefficients (matrix), determinants, pt2_result
  return E_final_vec, coeffs_final_matrix, variational_dets, pt2_result
end

"""
    diagonalize_selected_space(selected_ctx::SelectedCIContext; 
                               n_roots::Int=1,
                               previous_vectors::Union{Nothing,Matrix{Float64}}=nothing) 
      -> (Vector{Float64}, Matrix{Float64})

Diagonalize the Hamiltonian in the selected CI space.
Returns eigenvalues and eigenvectors for n_roots lowest states.

For small spaces (< 1000 determinants), uses direct diagonalization via eigen().
For large spaces (≥ 1000 determinants), uses Davidson iterative diagonalization.

# Arguments
- `selected_ctx`: Selected CI context with determinants
- `n_roots`: Number of lowest eigenstates to compute (default: 1)
- `previous_vectors`: Optional previous eigenvectors to use as initial guess for Davidson.
                     Should be a matrix of size (n_prev, n_roots) where n_prev is the
                     number of determinants in the previous iteration. Will be projected
                     onto the current determinant space.

# Returns
- `eigenvalues`: Vector of length n_roots with lowest eigenvalues
- `eigenvectors`: Matrix of size (n_selected, n_roots) with eigenvectors
"""
function diagonalize_selected_space(selected_ctx::SelectedCIContext; 
                                   n_roots::Int=1,
                                   previous_vectors::Union{Nothing,Matrix{Float64}}=nothing)::Tuple{Vector{Scalar}, Matrix{Scalar}}
  n_selected = selected_ctx.selected_dets.n_selected
  n_roots = min(n_roots, n_selected)  # Can't compute more roots than determinants
  
  # For small spaces, use direct diagonalization (faster startup)
  if n_selected < 1000
    H_matrix = zeros(Scalar, n_selected, n_selected)
    
    for i in 1:n_selected
      det_i = selected_ctx.selected_dets.determinants[i]
      for j in i:n_selected
        det_j = selected_ctx.selected_dets.determinants[j]
        H_matrix[i,j] = compute_matrix_element_direct(det_i, det_j, selected_ctx.base_context)
        if i != j
          H_matrix[j,i] = H_matrix[i,j]
        end
      end
    end
    
    nval = min(n_roots+5, n_selected)
    eigenvalues, eigenvectors = eigen(Hermitian(H_matrix), 1:nval)
    return real.(eigenvalues[1:n_roots]), real.(eigenvectors[:, 1:n_roots])
  end

  # For large spaces, use Davidson iterative diagonalization
  # This is much faster: O(N²) per iteration vs O(N³) for direct
  
  # Create initial guess(es)
  if previous_vectors !== nothing && size(previous_vectors, 1) <= n_selected
    # Use previous eigenvectors as initial guesses
    # The previous vectors correspond to a subset of current determinants
    # (the current space is a superset of the previous space)
    n_prev = size(previous_vectors, 1)
    n_prev_roots = size(previous_vectors, 2)
    
    # Project previous eigenvectors onto current space
    # Assume first n_prev determinants are the same (newly added dets are at the end)
    # Pass all available previous eigenvectors for their respective roots
    n_use_prev = min(n_roots, n_prev_roots)
    initial_guesses = zeros(Scalar, n_selected, n_use_prev)
    
    for i in 1:n_use_prev
      # Copy previous eigenvector (zero-padded for new determinants)
      initial_guesses[1:n_prev, i] .= previous_vectors[:, i]
    end
    
    # Call Davidson with all previous eigenvectors
    eigenvalues, eigenvectors = davidson_selected_ci!(
      selected_ctx, 
      initial_guesses,
      n_roots = n_roots,
      max_iterations = 50,
      convergence_threshold = 1e-8,
      verbose = false
    )
    return real.(eigenvalues), real.(eigenvectors)
  end
  # No previous vectors: use determinant with lowest diagonal element
  diagonal = [compute_diagonal_element(det, selected_ctx.base_context) 
              for det in selected_ctx.selected_dets.determinants]
  min_idx = argmin(diagonal)
  initial_guess = zeros(Scalar, n_selected, 1)
  initial_guess[min_idx, 1] = 1.0
  
  # Call Davidson solver with single initial guess
  eigenvalues, eigenvectors = davidson_selected_ci!(
    selected_ctx, 
    initial_guess,
    n_roots = n_roots,
    max_iterations = 50,
    convergence_threshold = 1e-8,
    verbose = false
  )
  return real.(eigenvalues), real.(eigenvectors)
  
end