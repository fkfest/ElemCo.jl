
"""
Selected CI implementation

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
    SelectedCIDeterminants

Container for selected determinants with efficient storage.
"""
struct SelectedCIDeterminants
  determinants::Vector{Determinant}     # Selected determinants
  addresses::Vector{Address}           # Corresponding addresses in full CI space

  function SelectedCIDeterminants(determinants::Vector{Determinant}, addresses::Vector{Address})
    @assert length(determinants) == length(addresses) "Determinants and addresses must have same length"
    new(determinants, addresses)
  end
end

"""
    n_selected(selci::SelectedCIDeterminants) -> Int

Get number of selected determinants.
"""
n_selected(selci::SelectedCIDeterminants) = length(selci.determinants)

"""
    extend!(dets::SelectedCIDeterminants, new_dets::Vector{Determinant}, new_addresses::Vector{Address})

Extend SelectedCIDeterminants with new determinants and addresses.
"""
function extend!(dets::SelectedCIDeterminants, new_dets::Vector{Determinant}, new_addresses::Vector{Address})
  @assert length(new_dets) == length(new_addresses) "New determinants and addresses must have same length"
  append!(dets.determinants, new_dets)
  append!(dets.addresses, new_addresses)
end

"""
    SelectedHamiltonianRow

Container for one row of the Selected CI Hamiltonian matrix.
"""
struct SelectedHamiltonianRow
  data::Vector{Scalar}                # Hamiltonian row for one determinant
  idet::Vector{Int}                   # Determinants connected to this determinant

  function SelectedHamiltonianRow()
    new([], [])
  end
end

function Base.push!(row::SelectedHamiltonianRow, value::Scalar, idet::Int)
  push!(row.data, value)
  push!(row.idet, idet)
end

struct SelectedHamiltonianMatrix
  rows::Vector{SelectedHamiltonianRow}  # Rows of the Hamiltonian matrix

  function SelectedHamiltonianMatrix()
    new([])
  end
end

function Base.push!(sel_ham::SelectedHamiltonianMatrix, row::SelectedHamiltonianRow)
  push!(sel_ham.rows, row)
end

"""
    extend!(sel_ham::SelectedHamiltonianMatrix, dets::SelectedCIDeterminants, context)

Extend the SelectedHamiltonianMatrix to include new determinants.
"""
function extend!(sel_ham::SelectedHamiltonianMatrix, dets::SelectedCIDeterminants, context::Union{FCIContext, HCIContext})
  ndet_old = length(sel_ham.rows)
  ndet = n_selected(dets)
  # add new connected determinants to existing rows of Hamiltonian
  for i in 1:ndet_old
    det_i = dets.determinants[i]
    for j in (ndet_old+1):ndet
      det_j = dets.determinants[j]
      if slater_condon_allowed(det_i, det_j)
        h_ij = compute_matrix_element_direct(det_i, det_j, context)
        push!(sel_ham.rows[i], h_ij, j)
      end
    end
  end
  # add new rows for new determinants
  for i in (ndet_old+1):ndet
    det_i = dets.determinants[i]
    new_row = SelectedHamiltonianRow()
    for j in 1:ndet
      det_j = dets.determinants[j]
      if slater_condon_allowed(det_i, det_j)
        h_ij = compute_matrix_element_direct(det_i, det_j, context)
        push!(new_row, h_ij, j)
      end
    end
    push!(sel_ham, new_row)
  end
end

"""
    SelectedCIContext

Context for Selected CI calculations using direct H*c operations.
"""
struct SelectedCIContext{ContextType <:Union{FCIContext, HCIContext}}
  base_context::ContextType                   # Context for integrals and (optionally) addressing
  selected_dets::SelectedCIDeterminants       # Selected determinants and addresses
  hamiltonian::SelectedHamiltonianMatrix      # Hamiltonian matrix for selected determinants

  # Constructor for FCIContext (uses full addressing)
  function SelectedCIContext(base_context::FCIContext, determinants::Vector{Determinant}, hamiltonian::SelectedHamiltonianMatrix)
    # Convert determinants to addresses using full addressing
    addresses = [address_from_determinant(base_context, det) for det in determinants]
    selected_dets = SelectedCIDeterminants(determinants, addresses)
    extend!(hamiltonian, selected_dets, base_context)
    new{FCIContext}(base_context, selected_dets, hamiltonian)
  end

  # Constructor for HCIContext (on-demand addressing)
  function SelectedCIContext(base_context::HCIContext, determinants::Vector{Determinant}, hamiltonian::SelectedHamiltonianMatrix)
    # For HCI, we use on-demand addressing: addresses are just indices (1, 2, 3, ...)
    addresses = UInt64.(1:length(determinants))
    selected_dets = SelectedCIDeterminants(determinants, addresses)
    extend!(hamiltonian, selected_dets, base_context)
    new{HCIContext}(base_context, selected_dets, hamiltonian)
  end
end

"""
    n_selected(selci::SelectedCIContext) -> Int

Get number of selected determinants.
"""
n_selected(selci::SelectedCIContext) = length(selci.selected_dets.determinants)

"""
    determinants(selected_ctx::SelectedCIContext) -> Vector{Determinant}

Get selected determinants.
"""
determinants(selected_ctx::SelectedCIContext) = selected_ctx.selected_dets.determinants

"""
    extend!(selected_ctx::SelectedCIContext, new_dets::Vector{Determinant})

Extend the SelectedCIContext with new determinants.
"""
function extend!(selected_ctx::SelectedCIContext, new_dets::Vector{Determinant})
  if selected_ctx.base_context isa FCIContext
    new_addresses = [address_from_determinant(selected_ctx.base_context, det) for det in new_dets]
  else
    # For HCIContext, addresses are just indices
    start_index = Address(n_selected(selected_ctx)) + 1
    new_addresses = collect(start_index:(start_index + length(new_dets) - 1))
  end
  extend!(selected_ctx.selected_dets, new_dets, new_addresses)
  extend!(selected_ctx.hamiltonian, selected_ctx.selected_dets, selected_ctx.base_context)
end

# ===========================================
# Orbital Excitation Analysis
# ===========================================

"""
    find_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern) -> (Int, Int)

Find the two orbitals involved in a single excitation i -> a.
"""
function find_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern)::Tuple{Int, Int}
  diff = pattern_i ⊻ pattern_j
  occ = pattern_i & diff
  virt = pattern_j & diff
 
  orb_a = trailing_zeros(virt) + 1  # Orbital being created
  orb_i = trailing_zeros(occ) + 1  # Orbital being destroyed

  return (orb_i, orb_a)
end

"""
    find_double_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern) -> (Int, Int, Int, Int)

Find the four orbitals involved in a double excitation ij -> ab.
"""
function find_double_excitation_orbitals(pattern_i::OrbPattern, pattern_j::OrbPattern)
  diff = pattern_i ⊻ pattern_j
  occ = pattern_i & diff
  virt = pattern_j & diff
 
  orb_a = trailing_zeros(virt)  # Orbital being created
  virt ⊻= (OrbPattern(1) << orb_a)
  orb_b = trailing_zeros(virt)  # Second orbital being created
  orb_i = trailing_zeros(occ)  # Orbital being destroyed
  occ ⊻= (OrbPattern(1) << orb_i)
  orb_j = trailing_zeros(occ)  # Second orbital being destroyed
  return (orb_i+1, orb_j+1, orb_a+1, orb_b+1)
end

"""
    calculate_excitation_phase(pattern::OrbPattern, orb_i::Int, orb_a::Int) -> Int8

Calculate phase factor for single excitation i -> a.
"""
function calculate_excitation_phase(pattern::OrbPattern, orb_i::Int, orb_a::Int)::Int8
  # Count electrons between orb_i and orb_a
  min_orb = min(orb_i, orb_a)
  max_orb = max(orb_i, orb_a) - 1
  # pattern between min_orb and max_orb (exclusive)
  sub_pattern = pattern & ((OrbPattern(1) << max_orb) - 1) & ~( (OrbPattern(1) << min_orb) - 1)
  # Count number of electrons in sub_pattern
  n_electrons = count_ones(sub_pattern)
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
  intermediate &= ~(OrbPattern(1) << (orb_i - 1))  # Remove electron from i
  intermediate |= (OrbPattern(1) << (orb_a - 1))   # Add electron to a

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
                                 context) -> Scalar

Compute ⟨det_i|Ĥ|det_j⟩ directly using orbital excitation analysis.
Works with both FCIContext and HCIContext.
"""
function compute_matrix_element_direct(det_i::Determinant, det_j::Determinant, context::Union{FCIContext, HCIContext})::Scalar
  # Find differences in alpha and beta strings
  alpha_diff = det_i.alpha ⊻ det_j.alpha  # XOR to find differing bits
  beta_diff = det_i.beta ⊻ det_j.beta

  # Count number of differing orbitals
  n_alpha_diff = count_ones(alpha_diff)
  n_beta_diff = count_ones(beta_diff)

  # Same determinant
  if n_alpha_diff == 0 && n_beta_diff == 0
    return diagonal_matrix_element(det_i, context)
  end

  # Single excitation in alpha string
  if n_alpha_diff == 2 && n_beta_diff == 0
    orb_i, orb_a = find_excitation_orbitals(det_i.alpha, det_j.alpha)
    phase = calculate_excitation_phase(det_i.alpha, orb_i, orb_a)
    return single_alpha_excitation_matrix_element(det_i, orb_i, orb_a, context) * phase
  end

  # Single excitation in beta string  
  if n_alpha_diff == 0 && n_beta_diff == 2
    orb_i, orb_a = find_excitation_orbitals(det_i.beta, det_j.beta)
    phase = calculate_excitation_phase(det_i.beta, orb_i, orb_a)
    return single_beta_excitation_matrix_element(det_i, orb_i, orb_a, context) * phase
  end

  # Double excitation
  if n_alpha_diff == 4 && n_beta_diff == 0
    # Double excitation in alpha
    orb_i, orb_j, orb_a, orb_b = find_double_excitation_orbitals(det_i.alpha, det_j.alpha)
    phase = calculate_double_excitation_phase(det_i.alpha, orb_i, orb_j, orb_a, orb_b)
    return double_alpha_excitation_matrix_element(context, orb_i, orb_j, orb_a, orb_b) * phase
  elseif n_alpha_diff == 0 && n_beta_diff == 4
    # Double excitation in beta
    orb_i, orb_j, orb_a, orb_b = find_double_excitation_orbitals(det_i.beta, det_j.beta)
    phase = calculate_double_excitation_phase(det_i.beta, orb_i, orb_j, orb_a, orb_b)
    return double_beta_excitation_matrix_element(context, orb_i, orb_j, orb_a, orb_b) * phase
  elseif n_alpha_diff == 2 && n_beta_diff == 2
    # Mixed double excitation (alpha and beta)
    orb_i_alpha, orb_a_alpha = find_excitation_orbitals(det_i.alpha, det_j.alpha)
    orb_i_beta, orb_a_beta = find_excitation_orbitals(det_i.beta, det_j.beta)
    phase_alpha = calculate_excitation_phase(det_i.alpha, orb_i_alpha, orb_a_alpha)
    phase_beta = calculate_excitation_phase(det_i.beta, orb_i_beta, orb_a_beta)
    total_phase = phase_alpha * phase_beta
    return double_alpha_beta_excitation_matrix_element(context, orb_i_alpha, orb_i_beta, orb_a_alpha, orb_a_beta) * total_phase
  end
  return 0.0  # Invalid excitation
end

"""
    diagonal_matrix_element(det::Determinant, context) -> Scalar

Compute diagonal matrix element ⟨det|Ĥ|det⟩.
For FCIContext uses precomputed diagonal, for HCIContext computes on-the-fly.
"""
function diagonal_matrix_element(det::Determinant, context::FCIContext)::Scalar
  # Get the address and use existing diagonal computation
  addr = address_from_determinant(context, det)
  return context.diag_h.data[addr]
end

function diagonal_matrix_element(det::Determinant, context::HCIContext)::Scalar
  # For HCI, compute diagonal element on-the-fly
  return compute_diagonal_element(det, context)
end

"""
    single_alpha_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int, context) -> Scalar 

Compute matrix element for single alpha excitation.
"""
function single_alpha_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int, 
                                                context::Union{FCIContext, HCIContext})
  if context.fcidump.uhf
    int1 = context.fcidump.int1a
  else
    int1 = context.fcidump.int1
  end
  h1e2_same = context.heval_data.h1e2_aa
  h1e2_opp = context.heval_data.h1e2_ab
  return compute_fock_element(int1, h1e2_same, h1e2_opp, det_i.alpha, det_i.beta, orb_a, orb_i)
end

"""
    single_beta_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int, context) -> Scalar 

Compute matrix element for single beta excitation.
"""
function single_beta_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int, 
                                                context::Union{FCIContext, HCIContext})
  if context.fcidump.uhf
    int1 = context.fcidump.int1b
    h1e2_same = context.heval_data.h1e2_bb
    h1e2_opp = context.heval_data.h1e2_ba
  else
    int1 = context.fcidump.int1
    h1e2_same = context.heval_data.h1e2_aa
    h1e2_opp = context.heval_data.h1e2_ab
  end
  return compute_fock_element(int1, h1e2_same, h1e2_opp, det_i.beta, det_i.alpha, orb_a, orb_i)
end

"""
    double_alpha_excitation_matrix_element(context, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Scalar

Compute matrix element for double alpha excitation.
"""
function double_alpha_excitation_matrix_element(context::Union{FCIContext, HCIContext}, 
                                                orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int)
  fcidump = context.fcidump
  int2aa = fcidump.uhf ? fcidump.int2aa : fcidump.int2
  return int2aa[orb_a, orb_b, orb_i, orb_j] - int2aa[orb_a, orb_b, orb_j, orb_i]
end

"""
    double_beta_excitation_matrix_element(context, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Scalar

Compute matrix element for double beta excitation.
"""
function double_beta_excitation_matrix_element(context::Union{FCIContext, HCIContext}, 
                                                orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int)
  fcidump = context.fcidump
  int2bb = fcidump.uhf ? fcidump.int2bb : fcidump.int2
  return int2bb[orb_a, orb_b, orb_i, orb_j] - int2bb[orb_a, orb_b, orb_j, orb_i]
end

"""
    double_alpha_beta_excitation_matrix_element(context, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Scalar

Compute matrix element for double alpha beta excitation.
"""
function double_alpha_beta_excitation_matrix_element(context::Union{FCIContext, HCIContext}, 
                                                orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int)
  fcidump = context.fcidump
  int2ab = fcidump.uhf ? fcidump.int2ab : fcidump.int2
  return int2ab[orb_a, orb_b, orb_i, orb_j]
end

"""
    slater_condon_allowed(det_i::Determinant, det_j::Determinant) -> Bool

Check if two determinants differ by at most two orbital occupations (same, single, or double excitations).
Returns true if they differ by ≤ 2 orbitals, false otherwise.
"""
@inline function slater_condon_allowed(det_i::Determinant, det_j::Determinant)::Bool
  alpha_diff = det_i.alpha ⊻ det_j.alpha
  beta_diff = det_i.beta ⊻ det_j.beta
  n_alpha_diff = count_ones(alpha_diff)
  n_beta_diff = count_ones(beta_diff)
  
  return (n_alpha_diff + n_beta_diff) <= 4
end

# ===========================================
# Direct H*c Matrix-Vector Products
# ===========================================

"""
    contract_hamiltonian_selected!(result::Vector{Scalar}, input::Vector{Scalar}, 
                                  selected_ctx::SelectedCIContext, prefactor::Scalar)

Compute H*c matrix-vector product using precomputed Hamiltonian elements.
"""
function contract_hamiltonian_selected!(result::Vector{Scalar}, input::Vector{Scalar},
                                        selected_ctx::SelectedCIContext, prefactor::Scalar)
  n_det = n_selected(selected_ctx)
  @assert length(result) == n_det "Result vector size mismatch"
  @assert length(input) == n_det "Input vector size mismatch"

  fill!(result, 0.0)
  if !isempty(selected_ctx.hamiltonian.rows)
    for i in 1:n_det
      row = selected_ctx.hamiltonian.rows[i]
      @inbounds @simd for k in 1:length(row.data)
        j = row.idet[k]
        h_ij = row.data[k]
        result[i] += h_ij * input[j]
      end
    end
  else
    # No precomputed Hamiltonian: compute on-the-fly
    # This part is currently not used 
    dets = determinants(selected_ctx)
    for i in 1:n_det
      det_i = dets[i]
      for j in 1:n_det
        if abs(input[j]) < ThrNeglect
          continue  # Skip negligible coefficients for efficiency
        end
        det_j = dets[j]
        if slater_condon_allowed(det_i, det_j)
          h_ij = compute_matrix_element_direct(det_i, det_j, selected_ctx.base_context)
          result[i] += h_ij * input[j]
        end
      end
    end
  end
  result .*= prefactor
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

"""
    hamiltonian_matrix(selected_ctx::SelectedCIContext) -> Matrix{Scalar}

Construct full Hamiltonian matrix for selected determinants.
"""
function hamiltonian_matrix(selected_ctx::SelectedCIContext)
  n_det = n_selected(selected_ctx)
  H_matrix = zeros(Scalar, n_det, n_det)
  
  for i in 1:n_det
    row = selected_ctx.hamiltonian.rows[i]
    @inbounds @simd for k in 1:length(row.data)
      j = row.idet[k]
      h_ij = row.data[k]
      H_matrix[i,j] = h_ij
    end
  end
  return H_matrix
end

# ===========================================
# Selected CI Solver Integration
# ===========================================

"""
    setup_selected_ci_from_determinants!(context::Union{FCIContext, HCIContext}, determinants::Vector{Determinant}, hamiltonian=SelectedHamiltonianMatrix()) -> SelectedCIContext

Create SelectedCIContext from list of determinants.
"""
function setup_selected_ci_from_determinants!(context::Union{FCIContext, HCIContext}, determinants::Vector{Determinant}, 
                                              hamiltonian::SelectedHamiltonianMatrix=SelectedHamiltonianMatrix())
  return SelectedCIContext(context, determinants, hamiltonian)
end

"""
    setup_selected_ci_from_addresses!(context::FCIContext, addresses::Vector{Address}) -> SelectedCIContext

Create SelectedCIContext from list of addresses.
"""
function setup_selected_ci_from_addresses!(context::FCIContext, addresses::Vector{Address})
  determinants = [determinant_from_address(context, addr) for addr in addresses]
  return SelectedCIContext(context, determinants, SelectedHamiltonianMatrix())
end

"""
    project_selected_to_full!(v_full::FCIVector, v_selected::Vector{Scalar}, 
                             selected_ctx::SelectedCIContext)

Project selected CI vector onto full CI space.
"""
function project_selected_to_full!(v_full::FCIVector, v_selected::Vector{Scalar},
                                   selected_ctx::SelectedCIContext)
  fill!(v_full.data, 0.0)

  for i in 1:n_selected(selected_ctx)
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
  for i in 1:n_selected(selected_ctx)
    addr = selected_ctx.selected_dets.addresses[i]
    v_selected[i] = v_full.data[addr]
  end
end

# ===========================================
# Heat-Bath Configuration Interaction (HBCI)
# ===========================================

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
  double_excitations::Vector{Vector{Tuple{Int,Int,Float64}}}  # RHF or UHF alpha-alpha
  double_excitations_bb::Vector{Vector{Tuple{Int,Int,Float64}}}  # UHF beta-beta
  double_excitations_ab::Vector{Vector{Tuple{Int,Int,Float64}}}  # RHF or UHF alpha-beta mixed
  h_doub_max::Float64              # Maximum |H(rs ← pq)| over all excitations

  function HBCISetupData()
    new(Vector{Tuple{Int,Int,Float64}}[], 
        Vector{Tuple{Int,Int,Float64}}[],
        Vector{Tuple{Int,Int,Float64}}[],
        0.0) 
  end
  
  # RHF constructor
  function HBCISetupData(double_exc::Vector{Vector{Tuple{Int,Int,Float64}}}, 
                        double_exc_ab::Vector{Vector{Tuple{Int,Int,Float64}}},
                        h_max::Float64)
    new(double_exc, 
        Vector{Tuple{Int,Int,Float64}}[],  # Empty for beta-beta
        double_exc_ab,
        h_max)
  end
  
  # UHF constructor
  function HBCISetupData(double_exc_aa::Vector{Vector{Tuple{Int,Int,Float64}}},
                        double_exc_bb::Vector{Vector{Tuple{Int,Int,Float64}}},
                        double_exc_ab::Vector{Vector{Tuple{Int,Int,Float64}}},
                        h_max::Float64)
    new(double_exc_aa, double_exc_bb, double_exc_ab, h_max) 
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
    single_excitation_alpha(det::Determinant, i::Int, a::Int) -> Determinant

Create determinant with alpha electron moved from orbital i to orbital a.
"""
function single_excitation_alpha(det::Determinant, i::Int, a::Int)::Determinant
  # Remove electron from orbital i, add to orbital a
  new_alpha = det.alpha & ~(OrbPattern(1) << (i-1))  # Remove i
  new_alpha |= (OrbPattern(1) << (a-1))              # Add a
  return Determinant(new_alpha, det.beta)
end

"""
    single_excitation_beta(det::Determinant, i::Int, a::Int) -> Determinant

Create determinant with beta electron moved from orbital i to orbital a.
"""
function single_excitation_beta(det::Determinant, i::Int, a::Int)::Determinant
  new_beta = det.beta & ~(OrbPattern(1) << (i-1))
  new_beta |= (OrbPattern(1) << (a-1))
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_alpha(det::Determinant, i::Int, j::Int, a::Int, b::Int) -> Determinant

Create determinant with alpha electrons moved from orbitals i,j to orbitals a,b.
"""
function double_excitation_alpha(det::Determinant, i::Int, j::Int, a::Int, b::Int)::Determinant
  new_alpha = det.alpha & ~(OrbPattern(1) << (i-1))  # Remove i
  new_alpha &= ~(OrbPattern(1) << (j-1))              # Remove j
  new_alpha |= (OrbPattern(1) << (a-1))               # Add a
  new_alpha |= (OrbPattern(1) << (b-1))               # Add b
  return Determinant(new_alpha, det.beta)
end

"""
    double_excitation_beta(det::Determinant, i::Int, j::Int, a::Int, b::Int) -> Determinant

Create determinant with beta electrons moved from orbitals i,j to orbitals a,b.
"""
function double_excitation_beta(det::Determinant, i::Int, j::Int, a::Int, b::Int)::Determinant
  new_beta = det.beta & ~(OrbPattern(1) << (i-1))
  new_beta &= ~(OrbPattern(1) << (j-1))
  new_beta |= (OrbPattern(1) << (a-1))
  new_beta |= (OrbPattern(1) << (b-1))
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_mixed(det::Determinant, i_alpha::Int, i_beta::Int, a_alpha::Int, a_beta::Int) -> Determinant

Create determinant with one alpha excitation i_alpha->a_alpha and one beta excitation i_beta->a_beta.
"""
function double_excitation_mixed(det::Determinant, i_alpha::Int, i_beta::Int, 
                                 a_alpha::Int, a_beta::Int)::Determinant
  new_alpha = det.alpha & ~(OrbPattern(1) << (i_alpha-1))
  new_alpha |= (OrbPattern(1) << (a_alpha-1))
  new_beta = det.beta & ~(OrbPattern(1) << (i_beta-1))
  new_beta |= (OrbPattern(1) << (a_beta-1))
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
  n_orb = ctx.n_orb
  
  # Get occupied and virtual orbitals for alpha and beta
  alpha_occ = BufVec(@view(ctx.ibuf[1:n_orb]))
  alpha_virt = BufVec(@view(ctx.ibuf[n_orb+1:n_orbs*2]))
  occupied_and_virtual_orbitals!(alpha_occ, alpha_virt, det.alpha, n_orb)
  beta_occ = BufVec(@view(ctx.ibuf[n_orb*2+1:n_orb*3]))
  beta_virt = BufVec(@view(ctx.ibuf[n_orb*3+1:end]))
  occupied_and_virtual_orbitals!(beta_occ, beta_virt, det.beta, n_orb)
  
  # Alpha single excitations: i -> a
  @inbounds for i in alpha_occ
    for a in alpha_virt
      new_det = single_excitation_alpha(det, i, a)
      push!(connected, new_det)
    end
  end
  
  # Beta single excitations: i -> a
  @inbounds for i in beta_occ
    for a in beta_virt
      new_det = single_excitation_beta(det, i, a)
      push!(connected, new_det)
    end
  end
  
  # Alpha double excitations: ij -> ab
  @inbounds for (idx_i, i) in enumerate(alpha_occ)
    for j in @view(alpha_occ[(idx_i+1):end])  # j > i to avoid duplicates
      for (idx_a, a) in enumerate(alpha_virt)
        for b in @view(alpha_virt[(idx_a+1):end])  # b > a to avoid duplicates
          new_det = double_excitation_alpha(det, i, j, a, b)
          push!(connected, new_det)
        end
      end
    end
  end
  
  # Beta double excitations: ij -> ab
  for (idx_i, i) in enumerate(beta_occ)
    for j in @view(beta_occ[(idx_i+1):end])
      for (idx_a, a) in enumerate(beta_virt)
        for b in @view(beta_virt[(idx_a+1):end])
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
          @assert_devel count_ones(new_det.alpha) == count_ones(det.alpha) "Alpha electron count mismatch"
          @assert_devel count_ones(new_det.beta) == count_ones(det.beta) "Beta electron count mismatch"
          push!(connected, new_det)
        end
      end
    end
  end
  
  return length(connected)
end

"""
    add_excitations!(excitations::Vector{Determinant}, det::Determinant,
                    double_excitation::Function,
                    occ, pchb_excitations, epsilon::Float64)

Generate same-spin double excitations from occupied orbitals using pre-computed
Heat-Bath lists, adding only those with |H| > epsilon.
"""
function add_excitations!(excitations::Vector{Determinant}, det::Determinant,
                          double_excitation::Function,
                          occ, pchb_excitations, epsilon::Float64)
  for (idx_i, i) in enumerate(occ)
    for j in @view(occ[(idx_i+1):end])
      # Look up pre-sorted list for (i,j) pair
      pq_key = trip_index(i, j)
      for (r, s, h_val) in pchb_excitations[pq_key]
        # Stop when matrix element falls below threshold
        if h_val < epsilon
          break
        end
        
        # Check if r and s are virtual (not occupied)
        if !(r in occ) && !(s in occ)
          new_det = double_excitation(det, i, j, r, s)
          @assert_devel count_ones(new_det.alpha) == count_ones(det.alpha) "Alpha electron count mismatch $i $j -> $r $s from $(bitstring(det.alpha))"
          @assert_devel count_ones(new_det.beta) == count_ones(det.beta) "Beta electron count mismatch $i $j -> $r $s from $(bitstring(det.beta))"
          push!(excitations, new_det)
        end
      end
    end
  end
  return
end

"""
    add_excitations!(excitations::Vector{Determinant}, det::Determinant,
                    alpha_occ, beta_occ, n_orb, pchb_excitations, epsilon::Float64)

Generate mixed-spin double excitations from occupied alpha and beta orbitals
using pre-computed Heat-Bath lists, adding only those with |H| > epsilon.
"""
function add_excitations!(excitations::Vector{Determinant}, det::Determinant,
                          alpha_occ, beta_occ, n_orb, pchb_excitations,
                          epsilon::Float64)
  for i_alpha in alpha_occ
    for i_beta in beta_occ
      pq_key = trip_index(i_alpha, i_beta, n_orb)
      for (r, s, h_val) in pchb_excitations[pq_key]
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
  return
end

"""
    sum_h1e2(h1e2, occ, a, i) -> Float64

Compute Σ_j h1e2[j, a, i] over occupied orbitals j.
"""
@inline function sum_h1e2(h1e2, occ, a, i)
  total = 0.0
  @inbounds @simd for j in occ
    total += h1e2[j, a, i]
  end
  return total
end

"""
    compute_fock_element(int1, h1e2_same, h1e2_opp, occ_same, occ_opp,
                        a::Int, i::Int) -> Float64

Compute Fock matrix element f_ai
f_ai = h_ai + Σ_j (v_aijj - v_ajji)_SS + Σ_j (v_aijj)_OS
where SS = same spin, OS = opposite spin. 
"""
function compute_fock_element(int1, h1e2_same, h1e2_opp, occ_same, occ_opp,
                              a::Int, i::Int)::Float64
  # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
  return int1[a, i] + sum_h1e2(h1e2_same, occ_same, a, i) + sum_h1e2(h1e2_opp, occ_opp, a, i)
end

"""
    sum_h1e2(h1e2, str::OrbPattern, a, i) -> Float64

Compute Σ_j h1e2[j, a, i] over occupied orbitals j.
"""
@inline function sum_h1e2(h1e2, str::OrbPattern, a, i)
  total = 0.0
  @inbounds @simd for k in 1:size(h1e2, 1)
    if (str >>> (k-1)) & one(str) != zero(str)
      total += h1e2[k, a, i]
    end
  end
  return total
end
"""
    compute_fock_element(int1, h1e2_same, h1e2_opp, str_same::OrbPattern, str_opp::OrbPattern,
                        a::Int, i::Int) -> Float64

Compute Fock matrix element f_ai
f_ai = h_ai + Σ_j (v_aijj - v_ajji)_SS + Σ_j (v_aijj)_OS
where SS = same spin, OS = opposite spin. 
"""
function compute_fock_element(int1, h1e2_same, h1e2_opp, str_same::OrbPattern, str_opp::OrbPattern,
                              a::Int, i::Int)::Float64
  # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
  return int1[a, i] + sum_h1e2(h1e2_same, str_same, a, i) + sum_h1e2(h1e2_opp, str_opp, a, i)
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
Works with both FCIContext and HCIContext.
"""
function generate_excitations_with_threshold!(excitations::Vector{Determinant},
                                             det::Determinant,
                                             ctx::Union{FCIContext, HCIContext},
                                             setup_data::HBCISetupData,
                                             epsilon::Float64)::Int
  empty!(excitations)
  n_orb = ctx.n_orb
  is_uhf = ctx.fcidump.uhf
  
  # Get occupied and virtual orbitals
  alpha_occ = BufVec(reshape_buf(ctx.ibuf, n_orb))
  alpha_virt = BufVec(reshape_buf(ctx.ibuf, n_orb; offset=n_orb))
  occupied_and_virtual_orbitals!(alpha_occ, alpha_virt, det.alpha, n_orb)
  beta_occ = BufVec(reshape_buf(ctx.ibuf, n_orb; offset=n_orb*2))
  beta_virt = BufVec(reshape_buf(ctx.ibuf, n_orb; offset=n_orb*3))
  occupied_and_virtual_orbitals!(beta_occ, beta_virt, det.beta, n_orb)
  
  # ===========================================
  # 1. Generate double excitations using pre-computed lists
  # ===========================================
  
  # Check if we should skip all double excitations
  if epsilon <= setup_data.h_doub_max
    double_excitations_bb = is_uhf ? setup_data.double_excitations_bb : setup_data.double_excitations
    # Alpha-alpha double excitations
    add_excitations!(excitations, det, double_excitation_alpha, alpha_occ, setup_data.double_excitations, epsilon)
    # Beta-beta double excitations
    add_excitations!(excitations, det, double_excitation_beta, beta_occ, double_excitations_bb, epsilon)
    # Mixed double excitations (alpha-beta) (use int2ab pre-computed lists, i.e., no exchange)
    add_excitations!(excitations, det, alpha_occ, beta_occ, n_orb, setup_data.double_excitations_ab, epsilon)
  end
  # ===========================================
  # 2. Generate single excitations with on-the-fly filtering using Fock elements
  # ===========================================
  
  if is_uhf
    int1 = ctx.fcidump.int1a
  else
    int1 = ctx.fcidump.int1
  end
  h1e2_same = ctx.heval_data.h1e2_aa
  h1e2_opp = ctx.heval_data.h1e2_ab
  # Alpha single excitations
  for i in alpha_occ
    for a in alpha_virt
      # Compute Fock matrix element f_ai
      h_val = abs(compute_fock_element(int1, h1e2_same, h1e2_opp, alpha_occ, beta_occ, a, i))
      if h_val >= epsilon
        new_det = single_excitation_alpha(det, i, a)
        push!(excitations, new_det)
      end
    end
  end
  
  if is_uhf
    int1 = ctx.fcidump.int1b
    h1e2_same = ctx.heval_data.h1e2_bb
    h1e2_opp = ctx.heval_data.h1e2_ba
  end
  # Beta single excitations
  for i in beta_occ
    for a in beta_virt
      # Compute Fock matrix element f_ai
      h_val = abs(compute_fock_element(int1, h1e2_same, h1e2_opp, beta_occ, alpha_occ, a, i))
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
    compute_diagonal_element(det::Determinant, ctx) -> Scalar

Compute diagonal matrix element ⟨det|H|det⟩ for a single determinant using HEvalData.
Works with both FCIContext and HCIContext.
"""
function compute_diagonal_element(det::Determinant, ctx::Union{FCIContext, HCIContext})::Scalar
  return calc_diagonalH(ctx.heval_data, det.alpha, det.beta)
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
Works with both FCIContext and HCIContext.

Returns total selection probability sum.
"""
function compute_heatbath_probabilities!(candidates::Vector{HBCandidate},
                                        variational_dets::Vector{Determinant},
                                        variational_coeffs::Vector{Float64},
                                        ctx::Union{FCIContext, HCIContext},
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
  
  total_prob = 0.0
  for det_J in connected
    # Compute H_JJ (diagonal element)
    H_JJ = compute_diagonal_element(det_J, ctx)
    ΔE_J = E_current - H_JJ
    
    # Compute perturbative contribution: sum over I in variational space
    sum_term = 0.0
    for (i, det_I) in enumerate(variational_dets)
      if slater_condon_allowed(det_I, det_J)
        c_I = variational_coeffs[i]
        H_IJ = compute_matrix_element_direct(det_I, det_J, ctx)
        sum_term += c_I * H_IJ
      end
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
                                              ctx::Union{FCIContext, HCIContext},
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
- `ctx`: FCI or HCI context
- `E_states`: Vector of energies for all states
- `setup_data`: Optional setup data
- `epsilon`: Threshold for excitation generation

# Returns
- Total selection probability sum across all candidates
"""
function compute_heatbath_probabilities_multistate!(candidates::Vector{HBCandidate},
                                                   variational_dets::Vector{Determinant},
                                                   variational_coeffs::Matrix{Float64},
                                                   ctx::Union{FCIContext, HCIContext},
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
      if slater_condon_allowed(det_I, det_J)
        H_IJ = compute_matrix_element_direct(det_I, det_J, ctx)
        for state in 1:n_states
          c_I = variational_coeffs[i, state]
          sum_terms[state] += c_I * H_IJ
        end
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
    select_determinants_perturbatively!(selected::Vector{Determinant},
                                 candidates::Vector{HBCandidate},
                                 options::HCIOptions,
                                 target_size::Int) -> Int

Select determinants from candidates using precalculated Epstein-Nesbet energy.
Returns number of determinants selected.
"""
function select_determinants_perturbatively!(selected::Vector{Determinant},
                                      candidates::Vector{HBCandidate},
                                      options::HCIOptions,
                                      target_size::Int)
  empty!(selected)
  # Sort by probability (descending)
  sort!(candidates, by=c->c.probability, rev=true)
  
  # Select determinants above threshold or until target reached
  # use square of epsilon_p to match probability definition (T_2^2)
  eps_p = options.epsilon_p > -0.1 ? options.epsilon_p : options.epsilon
  epsilon = eps_p^2
  n_selected = 0
  for candidate in candidates
    if n_selected >= target_size
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
function setup_hbci!(ctx::Union{FCIContext, HCIContext})::HBCISetupData
  n_orb = ctx.n_orb
  is_uhf = ctx.fcidump.uhf
  
  if !is_uhf
    # RHF case: use standard int2 integrals
    return setup_hbci_rhf!(ctx)
  else
    # UHF case: use spin-separated integrals
    return setup_hbci_uhf!(ctx)
  end
end

"""
    trip_index(p, q) -> Int

Compute unique index for orbital pair (p, q) with p < q.
"""
function trip_index(p, q)
  @assert_devel p < q "trip_index requires p < q"
  return p + (q - 1) * (q - 2) ÷ 2
end

"""
    trip_index(p, q, n) -> Int

Compute unique index for orbital pair (p, q),
with n orbitals per spin.
"""
function trip_index(p, q, n)
  return p + (q - 1) * n
end

function gen_triplets_list(n_orb::Int, int2::Array{Float64,4})
  double_exc_lists = Vector{Tuple{Int,Int,Float64}}[]
  h_doub_max = 0.0
  
  # Loop over all pairs of orbitals {p, q}
  for q in 2:n_orb
    for p in 1:(q-1)  # Only consider p < q to avoid duplicates
      # List of triplets {r, s, |H(rs ← pq)|} for this (p,q) pair
      triplets = Tuple{Int,Int,Float64}[]
      
      # Loop over all distinct pairs of orbitals {r, s} that don't include {p, q}
      for s in 2:n_orb
        if s == p || s == q
          continue
        end
        for r in 1:(s-1)  # Only consider r < s to avoid duplicates
          if r == p || r == q
            continue
          end
          
          # Compute antisymmetrized two-electron integral <pq||rs>
          # Matrix element for double excitation p,q → r,s is v_pq^rs - v_pq^sr
          h_val = abs(int2[p, q, r, s] - int2[p, q, s, r])
          
          if h_val > 1e-10  # Skip negligible matrix elements
            push!(triplets, (r, s, h_val))
            h_doub_max = max(h_doub_max, h_val)
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(triplets, by=x->x[3], rev=true)
      
      # Store sorted list for this (p,q) pair
      push!(double_exc_lists, triplets)
    end
  end
  return double_exc_lists, h_doub_max
end

function gen_triplets_list_ab(n_orb::Int, int2ab::Array{Float64,4})
  double_exc_ab_lists = Vector{Tuple{Int,Int,Float64}}[]
  h_doub_max = 0.0
  
  # Loop over all pairs of orbitals {p, q}
  # For mixed excitations, we don't need antisymmetrization (different spins)
  for q in 1:n_orb
    for p in 1:n_orb
      triplets = Tuple{Int,Int,Float64}[]
      for r in 1:n_orb
        if r == p; continue; end  # Alpha r cannot equal alpha p
        for s in 1:n_orb
          if s == q; continue; end  # Beta s cannot equal beta q
          
          # Mixed integral v_pq^rs (αβ) (no antisymmetrization for different spins)
          h_val = abs(int2ab[p, q, r, s])

          if h_val > 1e-10
            push!(triplets, (r, s, h_val))
            h_doub_max = max(h_doub_max, h_val)
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(triplets, by=x->x[3], rev=true)
      push!(double_exc_ab_lists, triplets)
    end
  end
  return double_exc_ab_lists, h_doub_max
end
"""
    setup_hbci_rhf!(ctx::Union{FCIContext, HCIContext}) -> HBCISetupData

Setup for RHF systems using spatial orbital integrals.
"""
function setup_hbci_rhf!(ctx::Union{FCIContext, HCIContext})::HBCISetupData
  n_orb = ctx.n_orb
  
  # Dictionary to store sorted lists for each (p,q) pair
  double_exc_lists, h_doub_max = gen_triplets_list(n_orb, ctx.fcidump.int2)
  double_exc_ab_lists, h_doub_max_ab = gen_triplets_list_ab(n_orb, ctx.fcidump.int2)
  h_doub_max = max(h_doub_max, h_doub_max_ab)
  return HBCISetupData(double_exc_lists, double_exc_ab_lists, h_doub_max)
end

"""
    setup_hbci_uhf!(ctx::Union{FCIContext, HCIContext}) -> HBCISetupData

Setup for UHF systems using spin-separated integrals.
Handles three types of double excitations:
- Alpha-alpha (using int2aa)
- Beta-beta (using int2bb)
- Mixed alpha-beta (using int2ab)
"""
function setup_hbci_uhf!(ctx::Union{FCIContext, HCIContext})::HBCISetupData
  n_orb = ctx.n_orb
  
  # Three dictionaries for the three types of double excitations
  double_exc_aa, h_doub_max_aa = gen_triplets_list(n_orb, ctx.fcidump.int2aa)
  double_exc_bb, h_doub_max_bb = gen_triplets_list(n_orb, ctx.fcidump.int2bb)
  double_exc_ab, h_doub_max_ab = gen_triplets_list_ab(n_orb, ctx.fcidump.int2ab)
  h_doub_max = max(h_doub_max_aa, h_doub_max_bb, h_doub_max_ab)

  return HBCISetupData(double_exc_aa, double_exc_bb, double_exc_ab, h_doub_max)
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
  ctx::Union{FCIContext, HCIContext},
  variational_dets::Vector{Determinant},
  coefficients::Vector{Float64},
  E_variational::Float64,
  setup_data::Union{HBCISetupData, Nothing},
  options::HCIOptions
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
    run_heatbath_ci!(ctx::Union{FCIContext, HCIContext}, options::HCIOptions) 
      -> (Vector{Float64}, Matrix{Float64}, Vector{Determinant}, PT2Result)

Run Heat-Bath CI calculation with support for multiple states.

# Arguments
- `ctx`: FCI context
- `options`: Heat-Bath CI options (including nstates for multi-state)

# Returns
- `energies`: Vector of length nstates with total energies (electronic + nuclear)
- `coefficients`: Matrix (n_dets × nstates) with CI coefficients for all states
- `variational_dets`: Vector of determinants in final variational space
- `pt2_result`: PT2 correction result (currently only for ground state)

# Notes
- For nstates=1 (default), uses single-state selection strategy
- For nstates>1, uses multi-state selection with state-maximum probability
- PT2 correction currently only computed for ground state
"""
function run_heatbath_ci!(ctx::Union{FCIContext, HCIContext}, options::HCIOptions)::Tuple{Vector{Scalar}, Matrix{Scalar}, Vector{Determinant}, PT2Result}
  if options.verbose
    println("\n" * "="^70)
    println("Heat-Bath Configuration Interaction (HBCI)")
    println("="^70)
    println("Target selection: $(options.target_selection)")
    println("Selection threshold (ε): $(options.epsilon)")
    if options.epsilon_h > -0.1
      println("Selection threshold (εₕ): $(options.epsilon_h)")
    end
    if options.epsilon_p > -0.1
      println("Selection threshold (εₚ): $(options.epsilon_p)")
    end
    println("PT2 threshold (εₚₜ₂): $(options.epsilon_pt2)")
    println("Number of states (nstates): $(options.nstates)")
    if options.nstates > 1
      println("Multi-state selection: State-maximum probability")
    end
    println("="^70)
    flush(stdout)
  end
  
  # Initialization
  hf_det = get_reference_determinant(ctx)
  
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

  if options.use_small_space_guess && options.nstates > 1
    # Use small-space Hamiltonian diagonalization for better initial guess
    if options.verbose
      println("\nInitialization (Small-Space Method)")
      println("  Using small-space Hamiltonian for multi-state initial guess")
    end
    
    # Determine small-space size (adaptive or user-specified)
    small_space_size = options.small_space_size > 0 ? 
                       options.small_space_size : 
                       max(100, trunc(Int,sqrt(options.target_selection)), 5 * options.nstates)
    
    # Generate initial guess from small-space diagonalization
    small_space_result = initialize_multistate_from_small_space(
      ctx, options.target_selection, options.nstates
    )
    
    # Start with all small-space determinants
    variational_dets = copy(small_space_result.determinants)
    selected_ctx = SelectedCIContext(ctx, variational_dets, SelectedHamiltonianMatrix())
    
    # Initial energies from small-space diagonalization
    E_init_vec = small_space_result.eigenvalues .+ ctx.fcidump.int0
    
    if options.verbose
      println("\n  Small-space initial guess:")
      println("    Space size: $(small_space_result.n_small) determinants")
      println("    Initial energies ($(options.nstates) states):")
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
    selected_ctx = SelectedCIContext(ctx, variational_dets, SelectedHamiltonianMatrix())
    E_electronic_hf_vec, _ = diagonalize_selected_space(selected_ctx, nstates=options.nstates)
    E_init_vec = E_electronic_hf_vec .+ ctx.fcidump.int0
    
    if options.verbose
      if options.nstates == 1
        println("  HF reference energy: $(E_init_vec[1]) Hartree")
      else
        println("  HF reference energies ($(options.nstates) states):")
        for (i, E) in enumerate(E_init_vec)
          println("    State $i: $E Hartree")
        end
      end
    end
  end
  
  if options.verbose
    println("\nIterative perturbative selection")
  end
  
  E_prev_vec = zero(E_init_vec)
  previous_eigenvectors = nothing  # Track previous eigenvectors for warm start
  converged = false
  res_tol = options.res_tol * 100  # Looser tolerance for main loop diagonalization
  for iter in 1:options.max_iterations
    if options.verbose
      println("\nHBCI Iteration $iter:")
      println("  Current space size: $(n_selected(selected_ctx)) determinants")
    end
    
    # 1. Diagonalize Hamiltonian in current space (all requested states)
    E_electronic_vec, coeffs_matrix = diagonalize_selected_space(selected_ctx,
                                                                 nstates=options.nstates,
                                                                 previous_vectors=previous_eigenvectors,
                                                                 conv_tol=res_tol)
    E_current_vec = E_electronic_vec .+ ctx.fcidump.int0  # Add nuclear repulsion
    
    if options.verbose
      if options.nstates == 1
        println("  Energy: $(E_current_vec[1]) Hartree")
      else
        println("  Energies ($(options.nstates) states):")
        for (i, E) in enumerate(E_current_vec)
          println("    State $i: $E Hartree")
        end
      end
    end
    
    # 2. Check convergence (all states must converge)
    ΔE_vec = abs.(E_current_vec .- E_prev_vec)
    ΔE_max = maximum(ΔE_vec)

    res_tol = max(min(res_tol, ΔE_max/10), options.res_tol)  # Tighten tolerance for next iteration
    
    if options.verbose
      if options.nstates == 1
        println("  ΔE: $(ΔE_vec[1])")
      else
        println("  ΔE (max): $ΔE_max")
        for (i, ΔE) in enumerate(ΔE_vec)
          println("    State $i: $ΔE")
        end
      end
    end

    if ΔE_max < options.tol 
      if options.verbose
        println("  ✓ Converged! max(ΔE) = $ΔE_max < $(options.tol)")
      end
      converged = true
      break
    end

    if n_selected(selected_ctx) >= options.target_selection
      if options.verbose
        println("  ✓ Reached target selection size: $(options.target_selection) determinants")
      end
      converged = true
      break
    end
    
    # 3. Generate candidates and compute probabilities
    eps_h = options.epsilon_h > -0.1 ? options.epsilon_h : options.epsilon
    candidates = HBCandidate[]
    if options.nstates == 1
      # Single-state: use original single-state function
      total_prob = compute_heatbath_probabilities!(candidates, determinants(selected_ctx), coeffs_matrix[:,1], ctx, 
                                                  E_electronic_vec[1], setup_data, eps_h)
    else
      # Multi-state: use new multi-state function
      total_prob = compute_heatbath_probabilities_multistate!(candidates, determinants(selected_ctx), coeffs_matrix, ctx, 
                                                              E_electronic_vec, setup_data, eps_h)
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
    target_size = options.target_selection - n_selected(selected_ctx)
    n_new = select_determinants_perturbatively!(new_dets, candidates, options, target_size)
    
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
    extend!(selected_ctx, new_dets)
    
    E_prev_vec = copy(E_current_vec)
    # Save eigenvectors for next iteration as warm start
    previous_eigenvectors = coeffs_matrix
  end
  
  if !converged
    @warn "HBCI did not converge in $(options.max_iterations) iterations"
  end
  
  # Final diagonalization
  if options.verbose
    println("\nFinal diagonalization with $(n_selected(selected_ctx)) determinants...")
  end
  
  E_electronic_vec, coeffs_final_matrix = diagonalize_selected_space(selected_ctx,
                                                                     nstates=options.nstates,
                                                                     previous_vectors=previous_eigenvectors,
                                                                     conv_tol=options.res_tol)
  
  # Add nuclear repulsion energy for total energy
  E_final_vec = E_electronic_vec .+ ctx.fcidump.int0
  
  if options.verbose
    println("="^70)
    println("HBCI Complete!")
    if options.nstates == 1
      println("Electronic energy: $(E_electronic_vec[1]) Hartree")
      println("Nuclear repulsion: $(ctx.fcidump.int0) Hartree")
      println("Total energy: $(E_final_vec[1]) Hartree")
    else
      println("Electronic energies ($(options.nstates) states):")
      for (i, E) in enumerate(E_electronic_vec)
        println("  State $i: $E Hartree")
      end
      println("Nuclear repulsion: $(ctx.fcidump.int0) Hartree")
      println("Total energies:")
      for (i, E) in enumerate(E_final_vec)
        println("  State $i: $E Hartree")
      end
    end
    println("Final space size: $(n_selected(selected_ctx)) determinants")
    println("="^70)
  end
  
  # Compute PT2 correction if requested (only for ground state currently)
  pt2_result = PT2Result()
  if options.compute_pt2
    # Note: PT2 correction currently only computed for ground state (state 1)
    pt2_result = compute_pt2_correction!(ctx, determinants(selected_ctx), coeffs_final_matrix[:,1], 
                                         E_electronic_vec[1], setup_data, options)
    
    E_total_with_pt2 = E_final_vec[1] + pt2_result.energy_correction
    
    if options.verbose
      println("\nFinal Energies (Ground State with PT2):")
      println("  Variational:     $(E_final_vec[1]) Ha")
      println("  PT2 correction:  $(pt2_result.energy_correction) Ha")
      println("  Total (VAR+PT2): $E_total_with_pt2 Ha")
      if options.nstates > 1
        println("  Note: PT2 currently only computed for ground state")
      end
    end
  end
  
  # Return format: energies (vector), coefficients (matrix), determinants, pt2_result
  return E_final_vec, coeffs_final_matrix, determinants(selected_ctx), pt2_result
end

"""
    run_heatbath_ci!(ctx::HCIContext) -> (energies, coeffs, determinants, pt2_result)

Convenience method for HCIContext that uses bundled options.

This lightweight interface avoids FCIContext initialization overhead:
- No full-space address tables computed upfront
- No full-space diagonal H computed upfront  
- Initialization time proportional to N_orbitals, not N_determinants

# Returns
Same as run_heatbath_ci!(ctx, options): (energies, coeffs, determinants, pt2_result)
"""
function run_heatbath_ci!(ctx::HCIContext)::Tuple{Vector{Scalar}, Matrix{Scalar}, Vector{Determinant}, PT2Result}
  return run_heatbath_ci!(ctx, ctx.options)
end

"""
    diagonalize_selected_space(selected_ctx::SelectedCIContext; 
                               nstates::Int=1,
                               previous_vectors::Union{Nothing,Matrix{Float64}}=nothing,
                               conv_tol::Float64=1e-6) 
      -> (Vector{Float64}, Matrix{Float64})

Diagonalize the Hamiltonian in the selected CI space.
Returns eigenvalues and eigenvectors for nstates lowest states.

For small spaces (< 1000 determinants), uses direct diagonalization via eigen().
For large spaces (≥ 1000 determinants), uses Davidson iterative diagonalization.

# Arguments
- `selected_ctx`: Selected CI context with determinants
- `nstates`: Number of lowest eigenstates to compute (default: 1)
- `previous_vectors`: Optional previous eigenvectors to use as initial guess for Davidson.
                     Should be a matrix of size (n_prev, nstates) where n_prev is the
                     number of determinants in the previous iteration. Will be projected
                     onto the current determinant space.

# Returns
- `eigenvalues`: Vector of length nstates with lowest eigenvalues
- `eigenvectors`: Matrix of size (n_selected, nstates) with eigenvectors
"""
function diagonalize_selected_space(selected_ctx::SelectedCIContext; 
                                   nstates::Int=1,
                                   previous_vectors::Union{Nothing,Matrix{Float64}}=nothing,
                                   conv_tol::Float64=1e-6)::Tuple{Vector{Scalar}, Matrix{Scalar}}
  n_dets = n_selected(selected_ctx)
  dets = determinants(selected_ctx)
  nstates = min(nstates, n_dets)  # Can't compute more roots than determinants
  
  # For small spaces, use direct diagonalization (faster startup)
  if n_dets < 1000
    H_matrix = hamiltonian_matrix(selected_ctx)
    
    nval = min(nstates+5, n_dets)
    eigenvalues, eigenvectors = eigen(Hermitian(H_matrix), 1:nval)
    return real.(eigenvalues[1:nstates]), real.(eigenvectors[:, 1:nstates])
  end

  # For large spaces, use Davidson iterative diagonalization
  # This is much faster: O(N²) per iteration vs O(N³) for direct
  
  # Create initial guess(es)
  if previous_vectors !== nothing && size(previous_vectors, 1) <= n_dets
    # Use previous eigenvectors as initial guesses
    # The previous vectors correspond to a subset of current determinants
    # (the current space is a superset of the previous space)
    n_prev = size(previous_vectors, 1)
    n_prev_roots = size(previous_vectors, 2)
    
    # Project previous eigenvectors onto current space
    # Assume first n_prev determinants are the same (newly added dets are at the end)
    # Pass all available previous eigenvectors for their respective roots
    n_use_prev = min(nstates, n_prev_roots)
    initial_guesses = zeros(Scalar, n_dets, n_use_prev)
    
    for i in 1:n_use_prev
      # Copy previous eigenvector (zero-padded for new determinants)
      initial_guesses[1:n_prev, i] .= previous_vectors[:, i]
    end
    
    # Call Davidson with all previous eigenvectors
    eigenvalues, eigenvectors = davidson_selected_ci!(
      selected_ctx, 
      initial_guesses,
      nstates = nstates,
      max_iterations = 50,
      convergence_threshold = conv_tol,
      verbose = false
    )
    return real.(eigenvalues), real.(eigenvectors)
  end
  # No previous vectors: use determinant with lowest diagonal element
  diagonal = [compute_diagonal_element(det, selected_ctx.base_context) 
              for det in dets]
  min_idx = argmin(diagonal)
  initial_guess = zeros(Scalar, n_dets, 1)
  initial_guess[min_idx, 1] = 1.0
  
  # Call Davidson solver with single initial guess
  eigenvalues, eigenvectors = davidson_selected_ci!(
    selected_ctx, 
    initial_guess,
    nstates = nstates,
    max_iterations = 50,
    convergence_threshold = conv_tol,
    verbose = false
  )
  return real.(eigenvalues), real.(eigenvectors)
  
end