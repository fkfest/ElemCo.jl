
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
    DeterminantLookupIndex{OPattern}

Fast lookup index for finding connected determinants in O(1) time.

Stores three dictionaries:
- `same_alpha`: Maps alpha OPattern → Vector{Int} of determinant indices sharing that alpha
- `same_beta`: Maps beta OPattern → Vector{Int} of determinant indices sharing that beta  
- `singlexc_alpha`: Maps alpha OPattern → Vector{OPattern} of alpha patterns connected by single excitation

This enables efficient lookup of candidates for:
- Double-beta excitations (via same_alpha)
- Double-alpha excitations (via same_beta)
- Mixed alpha-beta excitations (via singlexc_alpha then same_alpha)

Reduces extend_SelectedHamiltonianMatrix from O(N²) to O(N×N_connections) where N_connections << N.
"""
struct DeterminantLookupIndex{OPattern}
  same_alpha::Dict{OPattern, Vector{Int}}
  same_beta::Dict{OPattern, Vector{Int}}
  singlexc_alpha::Dict{OPattern, Vector{OPattern}}

  function DeterminantLookupIndex{OPattern}() where OPattern
    new{OPattern}(Dict{OPattern, Vector{Int}}(),
        Dict{OPattern, Vector{Int}}(),
        Dict{OPattern, Vector{OPattern}}())
  end
end

function add_det2same_spin(dict::Dict{OPattern, Vector{Int}}, key::OPattern, idet::Int) where OPattern
  if haskey(dict, key)
    push!(dict[key], idet)
  else
    dict[key] = [idet]
  end
end

"""
    build_lookup_index!(index::DeterminantLookupIndex{OPattern}, determinants::Vector{Determinant{OPattern}}, 
                       start_idx::Int, end_idx::Int, verbosity::Int) where OPattern

Build same_alpha, same_beta and singleexc_alpha index dictionaries for determinants in range [start_idx:end_idx].
same_alpha/beta: Appends determinant indices to vectors for each unique alpha/beta pattern.
singleexc_alpha: Maps alpha patterns to vectors of connected alpha patterns via single excitations.
"""
function build_lookup_index!(index::DeterminantLookupIndex{OPattern}, determinants::Vector{Determinant{OPattern}},
                             start_idx::Int, end_idx::Int, verbosity::Int) where OPattern
  t1 = time_ns()
  old_alpha_patterns = collect(keys(index.same_alpha))
  for i in start_idx:end_idx
    det = determinants[i]
    add_det2same_spin(index.same_alpha, det.alpha, i)
    add_det2same_spin(index.same_beta, det.beta, i)
  end
  
  # Build singlexc_alpha for new alpha patterns only
  all_alpha_patterns = Set(keys(index.same_alpha))
  setdiff!(all_alpha_patterns, old_alpha_patterns)
  new_alpha_patterns = collect(all_alpha_patterns)

  new_connections = VecDict{OPattern, Vector{OPattern}}()
  # Check each new pattern against old patterns 
  connected = OPattern[]
  @inbounds for i in eachindex(new_alpha_patterns)
    α_new = new_alpha_patterns[i]
    @simd for j in eachindex(old_alpha_patterns)
      α_old = old_alpha_patterns[j]
      # Check if patterns differ by exactly 2 bits (single excitation)
      if count_ones(α_new ⊻ α_old) == 2
        push!(connected, α_old)
      end
    end
    new_connections[α_new] = copy(connected)
    empty!(connected)
  end
  
  # Also add reverse connections (α_old → α_new)
  for (α_new, connected_list) in new_connections
    for α_old in connected_list
      push!(index.singlexc_alpha[α_old], α_new)
    end
  end
  # Check each new pattern against new patterns
  @inbounds for i in eachindex(new_alpha_patterns)
    α_i = new_alpha_patterns[i]
    connected2i = new_connections.values[i]
    @simd for j in eachindex(new_alpha_patterns)
      α_j = new_alpha_patterns[j]
      # Check if patterns differ by exactly 2 bits (single excitation)
      if count_ones(α_i ⊻ α_j) == 2
        push!(connected2i, α_j)
      end
    end
  end
  # Merge new connections into main dict using append! to combine vectors
  mergewith!(append!, index.singlexc_alpha, new_connections)
  print_time(verbosity, t1, "build DeterminantLookupIndex", 1)
end

function Base.issorted(index::DeterminantLookupIndex)::Bool
  for vec in values(index.same_alpha)
    if !issorted(vec)
      return false
    end
  end
  for vec in values(index.same_beta)
    if !issorted(vec)
      return false
    end
  end
  return true
end

"""
    SelectedCIDeterminants{OPattern}

Container for selected determinants with efficient storage and fast lookup index.
"""
struct SelectedCIDeterminants{OPattern}
  determinants::Vector{Determinant{OPattern}}     # Selected determinants
  addresses::Vector{Address}           # Corresponding addresses in full CI space
  lookup_index::DeterminantLookupIndex{OPattern}  # Fast lookup for connected determinants
  verbosity::Int                        # Verbosity level for logging

  function SelectedCIDeterminants{OPattern}(determinants::Vector{Determinant{OPattern}}, addresses::Vector{Address}, verbosity::Int) where OPattern
    @assert length(determinants) == length(addresses) "Determinants and addresses must have same length"
    lookup_index = DeterminantLookupIndex{OPattern}()
    build_lookup_index!(lookup_index, determinants, 1, length(determinants), verbosity)
    new{OPattern}(determinants, addresses, lookup_index, verbosity)
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
Incrementally updates the lookup index for efficient connected determinant queries.
"""
function extend!(dets::SelectedCIDeterminants, new_dets, new_addresses::Vector{Address})
  @assert length(new_dets) == length(new_addresses) "New determinants and addresses must have same length"
  
  ndet_old = length(dets.determinants)
  append!(dets.determinants, new_dets)
  append!(dets.addresses, new_addresses)
  
  # Incrementally update lookup index for new determinants only
  ndet_new = length(dets.determinants)
  build_lookup_index!(dets.lookup_index, dets.determinants, ndet_old+1, ndet_new, dets.verbosity)
end

const SelectedHamiltonianRow = VecDict{Int, Scalar}

struct SelectedHamiltonianMatrix
  rows::Vector{SelectedHamiltonianRow}  # Rows of the Hamiltonian matrix
  hermitian::Bool                       # Whether the matrix is Hermitian (we still store full matrix)
  function SelectedHamiltonianMatrix()
    new([], true)
  end
end

function Base.push!(sel_ham::SelectedHamiltonianMatrix, row::SelectedHamiltonianRow)
  push!(sel_ham.rows, row)
end

"""
    get_connections(index::DeterminantLookupIndex, determinants::Vector{Determinant}, 
                    start_idx::Int, end_idx::Int) -> Vector{Vector{Int}}

Get connected determinants for each determinant in range [start_idx:end_idx]
"""
function get_connections(index::DeterminantLookupIndex, determinants, 
                         start_idx::Int, end_idx::Int)
  connections = Vector{Int}[]
  connected_dets = Vector{Int}(undef, length(determinants))
  @inbounds for i in start_idx:end_idx
    det_i = determinants[i]
    α_i = det_i.alpha
    β_i = det_i.beta
    nexc = 0
    # alpha excitations (same beta)
    candidates = index.same_beta[β_i]
    last_index = searchsortedfirst(candidates, i) - 1 # Only consider j < i to avoid double counting
    @simd for j_indx in 1:last_index
      j = candidates[j_indx]
      det_j = determinants[j]
      if count_ones(α_i ⊻ det_j.alpha) <= 4
        nexc += 1
        connected_dets[nexc] = j
      end
    end
    # beta excitations (same alpha)
    candidates = index.same_alpha[α_i]
    last_index = searchsortedfirst(candidates, i) - 1 # Only consider j < i to avoid double counting
    @simd for j_indx in 1:last_index
      j = candidates[j_indx]
      det_j = determinants[j]
      if count_ones(β_i ⊻ det_j.beta) <= 4  # single or double beta excitation
        nexc += 1
        connected_dets[nexc] = j
      end
    end
    # alpha-beta mixed excitations
    for α_excited in index.singlexc_alpha[α_i]
      candidates = index.same_alpha[α_excited]
      last_index = searchsortedfirst(candidates, i) - 1 # Only consider j < i to avoid double counting
      @simd for j_indx in 1:last_index
        j = candidates[j_indx]
        det_j = determinants[j]
        if count_ones(β_i ⊻ det_j.beta) == 2
          nexc += 1
          connected_dets[nexc] = j
        end
      end
    end
    push!(connections, connected_dets[1:nexc])
  end
  return connections
end


"""
    resize!(sel_ham::SelectedHamiltonianMatrix, dets::SelectedCIDeterminants, connections::Vector{Vector{Int}}, cursize)

Resize the SelectedHamiltonianMatrix to preallocate space for new elements.
"""
function Base.resize!(sel_ham::SelectedHamiltonianMatrix, dets::SelectedCIDeterminants, 
                      connections::Vector{Vector{Int}}, cursize)
  ndet_old = length(sel_ham.rows)
  ndet = n_selected(dets)
  index = dets.lookup_index
  # Calculate new sizes
  if isnothing(cursize)
    newsize = ones(Int, ndet) # At least diagonal element
    for (i, row) in enumerate(sel_ham.rows)
      newsize[i] = length(row)
    end
  else
    newsize = copy(cursize)
  end
  # connected determinants
  @inbounds for i in (ndet_old+1):ndet
    inew = i - ndet_old
    connected_dets = connections[inew]
    @simd for j in connected_dets
      newsize[j] += 1
    end
    newsize[i] += length(connected_dets)
  end

  # New rows for new determinants
  for i in (ndet_old+1):ndet
    push!(sel_ham.rows, SelectedHamiltonianRow())
  end
  # Resize rows
  for (i, row) in enumerate(sel_ham.rows)
    resize!(row, newsize[i])
  end
  return newsize
end

"""
    extend!(sel_ham::SelectedHamiltonianMatrix, dets::SelectedCIDeterminants, context)

Extend the SelectedHamiltonianMatrix to include new determinants.
Uses fast lookup index to find connected determinants in O(N×N_connections) instead of O(N²).
"""
function extend!(sel_ham::SelectedHamiltonianMatrix, dets::SelectedCIDeterminants, 
                 context::Union{FCIContext, HCIContext})
  t0 = time_ns()
  ndet_old = length(sel_ham.rows)
  ndet = n_selected(dets)
  ThrNeglect = context.options.thr_negligible
  index = dets.lookup_index
  @assert issorted(index) "DeterminantLookupIndex is not sorted. Skipping lower-triangle will not work correctly."
  # Get connected determinants for new determinants only
  connections = get_connections(index, dets.determinants, ndet_old + 1, ndet)
  t0 = print_time(context.options.print_level, t0, "generate connections", 1)
  # Current size of sel_ham, including the diagonal elements for new determinants
  cursize = ones(Int, ndet) 
  for (i, row) in enumerate(sel_ham.rows)
    cursize[i] = length(row)
  end
  resize!(sel_ham, dets, connections, cursize)
  t0 = print_time(context.options.print_level, t0, "resize Hamiltonian matrix", 1)

  n_orb = context.n_orb
  occa = zeros(Int, n_orb)
  occb = zeros(Int, n_orb)

  # Add new entries to existing rows
  # Find connected determinants using connections list
  @inbounds for i in (ndet_old+1):ndet
    det_i = dets.determinants[i]
    inew = i - ndet_old
    occupied_orbitals!(occa, det_i.alpha, n_orb)
    occupied_orbitals!(occb, det_i.beta, n_orb)
    new_row = sel_ham.rows[i]
    cursize_i = 1
    # Diagonal element
    setat!(new_row, cursize_i, i, diagonal_matrix_element(occa, occb, context))

    @simd for j in connections[inew]
      det_j = dets.determinants[j]
      h_ij = compute_matrix_element_direct(det_i, det_j, context, occa, occb)
      if sel_ham.hermitian
        h_ji = h_ij
      else
        h_ji = compute_matrix_element_direct(det_j, det_i, context)
      end
      if abs(h_ij) > ThrNeglect
        cursize_i += 1
        setat!(new_row, cursize_i, j, h_ij)
      end
      if abs(h_ji) > ThrNeglect
        cursize[j] += 1
        setat!(sel_ham.rows[j], cursize[j], i, h_ji)
      end
    end
    cursize[i] = cursize_i
  end
  for (i, row) in enumerate(sel_ham.rows)
    resize!(row, cursize[i])
  end
  print_time(context.options.print_level, t0, "extend Hamiltonian matrix", 1)
end

"""
    SelectedCIContext

Context for Selected CI calculations using direct H*c operations.
"""
struct SelectedCIContext{OPattern, ContextType <:Union{FCIContext{OPattern}, HCIContext{OPattern}}}
  base_context::ContextType                   # Context for integrals and (optionally) addressing
  selected_dets::SelectedCIDeterminants{OPattern}       # Selected determinants and addresses
  hamiltonian::SelectedHamiltonianMatrix      # Hamiltonian matrix for selected determinants
  old_ndet::Base.RefValue{Int}                # Number of determinants in previous iteration

  # Constructor for FCIContext (uses full addressing)
  function SelectedCIContext(base_context::FCIContext{OPattern}, determinants::Vector{Determinant{OPattern}}, 
                             hamiltonian::SelectedHamiltonianMatrix) where OPattern
    # Convert determinants to addresses using full addressing
    addresses = [address_from_determinant(base_context, det) for det in determinants]
    selected_dets = SelectedCIDeterminants{OPattern}(determinants, addresses, base_context.options.print_level)
    extend!(hamiltonian, selected_dets, base_context)
    new{OPattern, FCIContext{OPattern}}(base_context, selected_dets, hamiltonian, Ref(0))
  end

  # Constructor for HCIContext (on-demand addressing)
  function SelectedCIContext(base_context::HCIContext{OPattern}, determinants::Vector{Determinant{OPattern}}, 
                             hamiltonian::SelectedHamiltonianMatrix) where OPattern
    # For HCI, we use on-demand addressing: addresses are just indices (1, 2, 3, ...)
    addresses = Address.(1:length(determinants))
    selected_dets = SelectedCIDeterminants{OPattern}(determinants, addresses, base_context.options.print_level)
    extend!(hamiltonian, selected_dets, base_context)
    new{OPattern, HCIContext{OPattern}}(base_context, selected_dets, hamiltonian, Ref(0))
  end
end

"""
    n_selected(selci::SelectedCIContext) -> Int

Get number of selected determinants.
"""
n_selected(selci::SelectedCIContext) = length(selci.selected_dets.determinants)

"""
    n_old_dets(selci::SelectedCIContext) -> Int

Get the number of determinants in the previous iteration.
"""
n_old_dets(selci::SelectedCIContext) = selci.old_ndet[]

"""
    set_n_old_dets!(selci::SelectedCIContext, n::Int)

Set the number of determinants in the previous iteration.
"""
function set_n_old_dets!(selci::SelectedCIContext, n::Int)
  selci.old_ndet[] = n
end

"""
    determinants(selected_ctx::SelectedCIContext) -> Vector{Determinant}

Get selected determinants.
"""
determinants(selected_ctx::SelectedCIContext) = selected_ctx.selected_dets.determinants

"""
    extend!(selected_ctx::SelectedCIContext, new_dets::Vector{Determinant})

Extend the SelectedCIContext with new determinants.
"""
function extend!(selected_ctx::SelectedCIContext, new_dets)
  if selected_ctx.base_context isa FCIContext
    new_addresses = [address_from_determinant(selected_ctx.base_context, det) for det in new_dets]
  else
    # For HCIContext, addresses are just indices
    start_index = Address(n_selected(selected_ctx)) + 1
    new_addresses = collect(start_index:(start_index + length(new_dets) - 1))
  end
  old_ndets = n_selected(selected_ctx)
  extend!(selected_ctx.selected_dets, new_dets, new_addresses)
  extend!(selected_ctx.hamiltonian, selected_ctx.selected_dets, selected_ctx.base_context)
  selected_ctx.old_ndet[] = old_ndets
end

# ===========================================
# Orbital Excitation Analysis
# ===========================================

"""
    find_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern) where OPattern -> (Int, Int)

Find the two orbitals involved in a single excitation i -> a.
"""
function find_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern)::Tuple{Int, Int} where OPattern
  diff = pattern_i ⊻ pattern_j
  occ = pattern_i & diff
  virt = pattern_j & diff
 
  orb_a = trailing_zeros(virt) + 1  # Orbital being created
  orb_i = trailing_zeros(occ) + 1  # Orbital being destroyed

  return (orb_i, orb_a)
end

"""
    find_double_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern) where OPattern -> (Int, Int, Int, Int)

Find the four orbitals involved in a double excitation ij -> ab.
"""
function find_double_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern) where OPattern
  diff = pattern_i ⊻ pattern_j
  occ = pattern_i & diff
  virt = pattern_j & diff
 
  orb_a = trailing_zeros(virt)  # Orbital being created
  virt ⊻= (OPattern(1) << orb_a)
  orb_b = trailing_zeros(virt)  # Second orbital being created
  orb_i = trailing_zeros(occ)  # Orbital being destroyed
  occ ⊻= (OPattern(1) << orb_i)
  orb_j = trailing_zeros(occ)  # Second orbital being destroyed
  return (orb_i+1, orb_j+1, orb_a+1, orb_b+1)
end

"""
    calculate_excitation_phase(pattern::OPattern, orb_i::Int, orb_a::Int) where OPattern -> Int

Calculate phase factor for single excitation i -> a.
"""
function calculate_excitation_phase(pattern::OPattern, orb_i::Int, orb_a::Int)::Int where OPattern
  # Count electrons between orb_i and orb_a
  min_orb = min(orb_i, orb_a)
  max_orb = max(orb_i, orb_a) - 1
  # pattern between min_orb and max_orb (exclusive)
  sub_pattern = pattern & ((OPattern(1) << max_orb) - 1) & ~( (OPattern(1) << min_orb) - 1)
  # Count number of electrons in sub_pattern
  n_electrons = count_ones(sub_pattern)
  # Phase is (-1)^n_electrons
  return (n_electrons % 2 == 0) ? 1 : -1
end

"""
    calculate_double_excitation_phase(pattern::OPattern, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) where OPattern -> Int

Calculate phase factor for double excitation ij -> ab.

The phase is calculated by decomposing into two successive single excitations:
  1. First excitation:  i -> a, with phase φ₁
  2. Second excitation: j -> b in modified determinant, with phase φ₂
  Total phase = φ₁ × φ₂
  
This approach correctly handles the fermionic anticommutation relations.
"""
function calculate_double_excitation_phase(pattern::OPattern, orb_i::Int, orb_j::Int,
                                           orb_a::Int, orb_b::Int)::Int where OPattern
  # First excitation: i -> a
  phase1 = calculate_excitation_phase(pattern, orb_i, orb_a)
  
  # Create intermediate determinant: remove electron from orb_i, add to orb_a
  intermediate = pattern
  intermediate &= ~(OPattern(1) << (orb_i - 1))  # Remove electron from i
  intermediate |= (OPattern(1) << (orb_a - 1))   # Add electron to a

  # Second excitation: j -> b in the intermediate determinant
  phase2 = calculate_excitation_phase(intermediate, orb_j, orb_b)
  
  # Total phase is the product
  return phase1 * phase2
end

"""
    single_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int) -> (Determinant{OPattern}, Int)

Create determinant and calculate phase factor for single alpha excitation i -> a.
"""
function single_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int)
  new_det = single_excitation_alpha(det, orb_i, orb_a)
  return new_det, calculate_excitation_phase(det.alpha, orb_i, orb_a)
end
"""
    single_beta_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int) -> (Determinant{OPattern}, Int)

Create determinant and calculate phase factor for single beta excitation i -> a.
"""
function single_beta_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int)
  new_det = single_excitation_beta(det, orb_i, orb_a)
  return new_det, calculate_excitation_phase(det.beta, orb_i, orb_a)
end

"""
    double_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> (Determinant, Int)

Create determinant and calculate phase factor for double alpha excitation ij -> ab.
"""
function double_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int,
                                       orb_a::Int, orb_b::Int)
  new_det = double_excitation_alpha(det, orb_i, orb_j, orb_a, orb_b)
  return new_det, calculate_double_excitation_phase(det.alpha, orb_i, orb_j, orb_a, orb_b) 
end
"""
    double_beta_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> (Determinant, Int)

Create determinant and calculate phase factor for double beta excitation ij -> ab.
"""
function double_beta_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int,
                                      orb_a::Int, orb_b::Int)
  new_det = double_excitation_beta(det, orb_i, orb_j, orb_a, orb_b)
  return new_det, calculate_double_excitation_phase(det.beta, orb_i, orb_j, orb_a, orb_b)
end

"""
    double_alpha_beta_excitation_phase(det::Determinant, orb_i_alpha::Int, orb_i_beta::Int,
                                       orb_a_alpha::Int, orb_a_beta::Int) -> (Determinant, Int)

Calculate phase factor for double alpha-beta excitation iα jβ -> aα bβ.
"""
function double_alpha_beta_excitation_phase(det::Determinant, orb_i_alpha::Int, orb_i_beta::Int,
                                               orb_a_alpha::Int, orb_a_beta::Int)
  new_det = double_excitation_mixed(det, orb_i_alpha, orb_i_beta, orb_a_alpha, orb_a_beta)
  phase_alpha = calculate_excitation_phase(det.alpha, orb_i_alpha, orb_a_alpha)
  phase_beta = calculate_excitation_phase(det.beta, orb_i_beta, orb_a_beta)
  return new_det, phase_alpha * phase_beta
end

# ===========================================
# Direct Matrix Element Evaluation
# ===========================================

@pib function compute_matrix_element_beta_excitations(det_i::Determinant, det_j::Determinant, 
                      n_beta_diff, context::Union{FCIContext, HCIContext},
                      occa=nothing, occb=nothing)::Scalar
  if n_beta_diff == 2
    # Single beta excitation
    orb_i, orb_a = find_excitation_orbitals(det_i.beta, det_j.beta)
    phase = calculate_excitation_phase(det_i.beta, orb_i, orb_a)
    return single_beta_excitation_matrix_element(det_i, orb_i, orb_a, context, occa, occb) * phase
  elseif n_beta_diff == 4
    # Double beta excitation
    orb_i, orb_j, orb_a, orb_b = find_double_excitation_orbitals(det_i.beta, det_j.beta)
    phase = calculate_double_excitation_phase(det_i.beta, orb_i, orb_j, orb_a, orb_b)
    return double_beta_excitation_matrix_element(context, orb_i, orb_j, orb_a, orb_b) * phase
  end
  return 0.0  # Invalid excitation
end

@pib function compute_matrix_element_alpha_excitations(det_i::Determinant, det_j::Determinant,
                      n_alpha_diff, context::Union{FCIContext, HCIContext},
                      occa=nothing, occb=nothing)::Scalar
  if n_alpha_diff == 2
    # Single alpha excitation
    orb_i, orb_a = find_excitation_orbitals(det_i.alpha, det_j.alpha)
    phase = calculate_excitation_phase(det_i.alpha, orb_i, orb_a)
    return single_alpha_excitation_matrix_element(det_i, orb_i, orb_a, context, occa, occb) * phase
  elseif n_alpha_diff == 4
    # Double alpha excitation
    orb_i, orb_j, orb_a, orb_b = find_double_excitation_orbitals(det_i.alpha, det_j.alpha)
    phase = calculate_double_excitation_phase(det_i.alpha, orb_i, orb_j, orb_a, orb_b)
    return double_alpha_excitation_matrix_element(context, orb_i, orb_j, orb_a, orb_b) * phase
  end
  return 0.0  # Invalid excitation
end

@pib function compute_matrix_element_mixed_excitations(det_i::Determinant, det_j::Determinant,
                      context::Union{FCIContext, HCIContext})::Scalar
  # Mixed single excitations in alpha and beta
  orb_i_alpha, orb_a_alpha = find_excitation_orbitals(det_i.alpha, det_j.alpha)
  orb_i_beta, orb_a_beta = find_excitation_orbitals(det_i.beta, det_j.beta)
  phase_alpha = calculate_excitation_phase(det_i.alpha, orb_i_alpha, orb_a_alpha)
  phase_beta = calculate_excitation_phase(det_i.beta, orb_i_beta, orb_a_beta)
  total_phase = phase_alpha * phase_beta
  return double_alpha_beta_excitation_matrix_element(context, orb_i_alpha, orb_i_beta, orb_a_alpha, orb_a_beta) * total_phase
end

"""
    compute_matrix_element_direct(det_i::Determinant, det_j::Determinant, 
                                 context, occa=nothing, occb=nothing) -> Scalar

Compute ⟨det_i|Ĥ|det_j⟩ directly using orbital excitation analysis.
Works with both FCIContext and HCIContext.

occa/occb are either Nothing or lists of occupied orbitals (makes the calculation more efficient).
"""
@pib function compute_matrix_element_direct(det_i::Determinant, det_j::Determinant, 
                                            context::Union{FCIContext, HCIContext}, 
                                            occa=nothing, occb=nothing)::Scalar
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

  if n_alpha_diff == 0
    return compute_matrix_element_beta_excitations(det_i, det_j, n_beta_diff, context, occa, occb)
  end

  if n_beta_diff == 0
    return compute_matrix_element_alpha_excitations(det_i, det_j, n_alpha_diff, context, occa, occb)
  end
  # Mixed excitations
  if n_alpha_diff == 2 && n_beta_diff == 2
    # Mixed single excitations in alpha and beta
    return compute_matrix_element_mixed_excitations(det_i, det_j, context)
  end
  return 0.0  # Invalid excitation
end

"""
    diagonal_matrix_element(det::Determinant, context) -> Scalar

Compute diagonal matrix element ⟨det|Ĥ|det⟩.
For FCIContext uses precomputed diagonal, for HCIContext computes on-the-fly.
"""
@pib function diagonal_matrix_element(det::Determinant, context::FCIContext)::Scalar
  # Get the address and use existing diagonal computation
  addr = address_from_determinant(context, det)
  return context.diag_h.data[addr]
end

@pib function diagonal_matrix_element(det::Determinant, context::HCIContext)::Scalar
  # For HCI, compute diagonal element on-the-fly
  return compute_diagonal_element(det, context)
end

@pib function diagonal_matrix_element(occa::AbstractVector{Int}, occb::AbstractVector{Int}, 
                                      context::Union{HCIContext,FCIContext})::Scalar
  # For HCI, compute diagonal element on-the-fly
  return compute_diagonal_element(occa, occb, context)
end

"""
    single_alpha_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int, context,
                                           occa=nothing, occb=nothing) -> Scalar

Compute matrix element for single alpha excitation.
"""
@pib function single_alpha_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int,
                                                     context::Union{FCIContext, HCIContext}, 
                                                     occa=nothing, occb=nothing)
  int1 = context.int1a
  h1e2_same = context.heval_data.h1e2_aa
  h1e2_opp = context.heval_data.h1e2_ab
  if isnothing(occb)
    return compute_fock_element(int1, h1e2_same, h1e2_opp, det_i.alpha, det_i.beta, orb_a, orb_i)
  else
    return compute_fock_element(int1, h1e2_same, h1e2_opp, occa, occb, orb_a, orb_i)
  end
end

"""
    single_beta_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int, context,
                                          occa=nothing, occb=nothing) -> Scalar 

Compute matrix element for single beta excitation.
"""
@pib function single_beta_excitation_matrix_element(det_i::Determinant, orb_i::Int, orb_a::Int,
                                                    context::Union{FCIContext, HCIContext}, 
                                                    occa=nothing, occb=nothing)
  int1 = context.int1b
  h1e2_same = context.heval_data.h1e2_bb
  h1e2_opp = context.heval_data.h1e2_ba
  if isnothing(occb)
    return compute_fock_element(int1, h1e2_same, h1e2_opp, det_i.beta, det_i.alpha, orb_a, orb_i)
  else
    return compute_fock_element(int1, h1e2_same, h1e2_opp, occb, occa, orb_a, orb_i)
  end
end

"""
    double_alpha_excitation_matrix_element(context, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Scalar

Compute matrix element for double alpha excitation.
"""
@pib function double_alpha_excitation_matrix_element(context::Union{FCIContext, HCIContext}, 
                                                orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int)
  int2aa = context.int2aa
  return int2aa[orb_a, orb_b, orb_i, orb_j] - int2aa[orb_a, orb_b, orb_j, orb_i]
end

"""
    double_beta_excitation_matrix_element(context, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Scalar

Compute matrix element for double beta excitation.
"""
@pib function double_beta_excitation_matrix_element(context::Union{FCIContext, HCIContext}, 
                                                orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int)
  int2bb = context.int2bb
  return int2bb[orb_a, orb_b, orb_i, orb_j] - int2bb[orb_a, orb_b, orb_j, orb_i]
end

"""
    double_alpha_beta_excitation_matrix_element(context, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> Scalar

Compute matrix element for double alpha beta excitation.
"""
@pib function double_alpha_beta_excitation_matrix_element(context::Union{FCIContext, HCIContext}, 
                                                orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int)
  int2ab = context.int2ab
  return int2ab[orb_a, orb_b, orb_i, orb_j]
end

"""
    slater_condon_allowed(det_i::Determinant, det_j::Determinant) -> Bool

Check if two determinants differ by at most two orbital occupations (same, single, or double excitations).
Returns true if they differ by ≤ 2 orbitals, false otherwise.
"""
@inline function slater_condon_allowed(det_i::Determinant, det_j::Determinant)::Bool
  alpha_diff = det_i.alpha ⊻ det_j.alpha
  n_alpha_diff = UInt(count_ones(alpha_diff))
  beta_diff = det_i.beta ⊻ det_j.beta
  n_beta_diff = UInt(count_ones(beta_diff))

  return UInt(n_alpha_diff + n_beta_diff) <= 4
end

function is_singles_only(det_i::Determinant, det_j::Determinant)::Bool
  alpha_diff = det_i.alpha ⊻ det_j.alpha
  beta_diff = det_i.beta ⊻ det_j.beta
  n_alpha_diff = count_ones(alpha_diff)
  n_beta_diff = count_ones(beta_diff)
  
  return (n_alpha_diff + n_beta_diff) == 2
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
      @inbounds @simd for k in 1:length(row)
        j = row.keys[k]
        h_ij = row.values[k]
        result[i] += h_ij * input[j]
      end
    end
  else
    # No precomputed Hamiltonian: compute on-the-fly
    # This part is currently not used 
    dets = determinants(selected_ctx)
    ThrNeglect = selected_ctx.base_context.options.thr_negligible
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
    @inbounds @simd for k in 1:length(row)
      j = row.keys[k]
      h_ij = row.values[k]
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
function setup_selected_ci_from_determinants!(context::Union{FCIContext, HCIContext}, determinants, 
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
# Heat-Bath Configuration Interaction (HCI)
# ===========================================

"""
    HCISetupData

Setup data: Pre-computed and sorted double excitation matrix elements.

For each pair of orbitals {p,q}, stores a list of triplets {r,s,|H(rs←pq)|},
sorted by |H| in decreasing order. This enables efficient generation of only
important excitations during iterative selection.

Following Holmes et al. (2016), Algorithm Step IIa.
"""
struct HCISetupData
  # For RHF: only double_excitations is used
  # For UHF: all three dictionaries are used (aa, bb, ab)
  double_excitations::Vector{Vector{Tuple{Int,Int,Float64}}}  # RHF or UHF alpha-alpha
  double_excitations_bb::Vector{Vector{Tuple{Int,Int,Float64}}}  # UHF beta-beta
  double_excitations_ab::Vector{Vector{Tuple{Int,Int,Float64}}}  # RHF or UHF alpha-beta mixed
  h_doub_max::Float64              # Maximum |H(rs ← pq)| over all excitations

  function HCISetupData()
    new(Vector{Tuple{Int,Int,Float64}}[], 
        Vector{Tuple{Int,Int,Float64}}[],
        Vector{Tuple{Int,Int,Float64}}[],
        0.0) 
  end
  
  # RHF constructor
  function HCISetupData(double_exc::Vector{Vector{Tuple{Int,Int,Float64}}}, 
                        double_exc_ab::Vector{Vector{Tuple{Int,Int,Float64}}},
                        h_max::Float64)
    new(double_exc, 
        Vector{Tuple{Int,Int,Float64}}[],  # Empty for beta-beta
        double_exc_ab,
        h_max)
  end
  
  # UHF constructor
  function HCISetupData(double_exc_aa::Vector{Vector{Tuple{Int,Int,Float64}}},
                        double_exc_bb::Vector{Vector{Tuple{Int,Int,Float64}}},
                        double_exc_ab::Vector{Vector{Tuple{Int,Int,Float64}}},
                        h_max::Float64)
    new(double_exc_aa, double_exc_bb, double_exc_ab, h_max) 
  end
end

# ===========================================
# Helper Functions for Determinant Manipulation
# ===========================================

"""
    single_excitation_alpha(det::Determinant{OPattern}, i::Int, a::Int) where OPattern -> Determinant{OPattern}

Create determinant with alpha electron moved from orbital i to orbital a.
"""
function single_excitation_alpha(det::Determinant{OPattern}, i::Int, a::Int)::Determinant{OPattern} where OPattern
  # Remove electron from orbital i, add to orbital a
  new_alpha = det.alpha & ~(OPattern(1) << (i-1))  # Remove i
  new_alpha |= (OPattern(1) << (a-1))              # Add a
  return Determinant(new_alpha, det.beta)
end

"""
    single_excitation_beta(det::Determinant{OPattern}, i::Int, a::Int) where OPattern -> Determinant{OPattern}

Create determinant with beta electron moved from orbital i to orbital a.
"""
function single_excitation_beta(det::Determinant{OPattern}, i::Int, a::Int)::Determinant{OPattern} where OPattern
  new_beta = det.beta & ~(OPattern(1) << (i-1))
  new_beta |= (OPattern(1) << (a-1))
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_alpha(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int) where OPattern -> Determinant{OPattern}

Create determinant with alpha electrons moved from orbitals i,j to orbitals a,b.
"""
function double_excitation_alpha(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int)::Determinant{OPattern} where OPattern
  new_alpha = det.alpha & ~(OPattern(1) << (i-1))  # Remove i
  new_alpha &= ~(OPattern(1) << (j-1))              # Remove j
  new_alpha |= (OPattern(1) << (a-1))               # Add a
  new_alpha |= (OPattern(1) << (b-1))               # Add b
  return Determinant(new_alpha, det.beta)
end

"""
    double_excitation_beta(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int) where OPattern -> Determinant{OPattern}

Create determinant with beta electrons moved from orbitals i,j to orbitals a,b.
"""
function double_excitation_beta(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int)::Determinant{OPattern} where OPattern
  new_beta = det.beta & ~(OPattern(1) << (i-1))
  new_beta &= ~(OPattern(1) << (j-1))
  new_beta |= (OPattern(1) << (a-1))
  new_beta |= (OPattern(1) << (b-1))
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_mixed(det::Determinant, i_alpha::Int, i_beta::Int, a_alpha::Int, a_beta::Int) -> Determinant

Create determinant with one alpha excitation i_alpha->a_alpha and one beta excitation i_beta->a_beta.
"""
function double_excitation_mixed(det::Determinant{OPattern}, i_alpha::Int, i_beta::Int, 
                                 a_alpha::Int, a_beta::Int)::Determinant{OPattern} where OPattern
  new_alpha = det.alpha & ~(OPattern(1) << (i_alpha-1))
  new_alpha |= (OPattern(1) << (a_alpha-1))
  new_beta = det.beta & ~(OPattern(1) << (i_beta-1))
  new_beta |= (OPattern(1) << (a_beta-1))
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
function generate_connected_determinants!(connected::Vector{Determinant{OPattern}}, 
                                         det::Determinant{OPattern},
                                         ctx::FCIContext{OPattern})::Int where OPattern
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
    add_excitations!(excitations::VecDict{Determinant, Scalar}, det::Determinant,
                    coef::Scalar, double_excitation_phase::Function,
                    occ, pchb_excitations, epsilon::Float64)

Generate same-spin double excitations from occupied orbitals using pre-computed
Heat-Bath lists, adding only those with |H| > epsilon and storing in excitations dictionary
together with their weighted matrix elements.
"""
function add_excitations!(excitations::VecDict{Determinant{OPattern}, Scalar}, det::Determinant{OPattern},
                          coef::Scalar, double_excitation_phase::Function,
                          occ, pchb_excitations, epsilon::Float64) where OPattern
  for (idx_i, i) in enumerate(occ)
    for j in @view(occ[(idx_i+1):end])
      # Look up pre-sorted list for (i,j) pair
      pq_key = trip_index(i, j)
      for (r, s, h_val) in pchb_excitations[pq_key]
        # Stop when matrix element falls below threshold
        if abs(h_val) < epsilon
          break
        end
        
        # Check if r and s are virtual (not occupied)
        if !(r in occ) && !(s in occ)
          new_det, phase = double_excitation_phase(det, i, j, r, s)
          @assert_devel count_ones(new_det.alpha) == count_ones(det.alpha) "Alpha electron count mismatch $i $j -> $r $s from $(bitstring(det.alpha))"
          @assert_devel count_ones(new_det.beta) == count_ones(det.beta) "Beta electron count mismatch $i $j -> $r $s from $(bitstring(det.beta))"
          excitations[new_det] = coef * h_val * phase
        end
      end
    end
  end
  return
end

"""
    add_excitations!(excitations::VecDict{Determinant, Scalar}, det::Determinant, coef::Scalar,
                    alpha_occ, beta_occ, n_orb, pchb_excitations, epsilon::Float64)

Generate mixed-spin double excitations from occupied alpha and beta orbitals
using pre-computed Heat-Bath lists, adding only those with |H| > epsilon.
"""
function add_excitations!(excitations::VecDict{Determinant{OPattern}, Scalar}, det::Determinant{OPattern},
                          coef::Scalar, alpha_occ, beta_occ, n_orb, pchb_excitations,
                          epsilon::Float64) where OPattern
  for i_alpha in alpha_occ
    for i_beta in beta_occ
      pq_key = trip_index(i_alpha, i_beta, n_orb)
      for (r, s, h_val) in pchb_excitations[pq_key]
        if abs(h_val) < epsilon
          break
        end
        # r is alpha virtual, s is beta virtual
        if !(r in alpha_occ) && !(s in beta_occ)
          new_det, phase = double_alpha_beta_excitation_phase(det, i_alpha, i_beta, r, s)
          excitations[new_det] = coef * h_val * phase
        end
      end
    end
  end
  return
end

"""
    sum_h1e2(h1e2, occ::AbstractVector, a, i) -> Float64

Compute Σ_j h1e2[j, a, i] over occupied orbitals j.
"""
@pib function sum_h1e2(h1e2, occ::AbstractVector, a, i)
  total = 0.0
  @inbounds @simd for j in occ
    total += h1e2[j, a, i]
  end
  return total
end

"""
    compute_fock_element(int1, h1e2_same, h1e2_opp, occ_same::AbstractVector, occ_opp::AbstractVector,
                        a::Int, i::Int) -> Float64

Compute Fock matrix element f_ai
f_ai = h_ai + Σ_j (v_aijj - v_ajji)_SS + Σ_j (v_aijj)_OS
where SS = same spin, OS = opposite spin. 
"""
@pib function compute_fock_element(int1, h1e2_same, h1e2_opp, occ_same::AbstractVector, occ_opp::AbstractVector,
                              a::Int, i::Int)::Float64
  # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
  return int1[a, i] + sum_h1e2(h1e2_same, occ_same, a, i) + sum_h1e2(h1e2_opp, occ_opp, a, i)
end

"""
    sum_h1e2(h1e2, str::OPattern, a, i) where OPattern -> Float64

Compute Σ_j h1e2[j, a, i] over occupied orbitals j.
"""
@pib function sum_h1e2(h1e2, str::OPattern, a, i) where OPattern
  total = 0.0
  @inbounds @simd for k in axes(h1e2, 1)
    if (str >>> (k-1)) & one(str) != zero(str)
      total += h1e2[k, a, i]
    end
  end
  return total
end
"""
    compute_fock_element(int1, h1e2_same, h1e2_opp, str_same::OPattern, str_opp::OPattern,
                        a::Int, i::Int) where OPattern -> Float64

Compute Fock matrix element f_ai
f_ai = h_ai + Σ_j (v_aijj - v_ajji)_SS + Σ_j (v_aijj)_OS
where SS = same spin, OS = opposite spin. 
"""
@pib function compute_fock_element(int1, h1e2_same, h1e2_opp, str_same::OPattern, str_opp::OPattern,
                              a::Int, i::Int)::Float64 where OPattern
  # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
  return int1[a, i] + sum_h1e2(h1e2_same, str_same, a, i) + sum_h1e2(h1e2_opp, str_opp, a, i)
end

"""
    generate_excitations!(excitations::VecDict{Determinant, Scalar},
                                         det::Determinant,
                                         coef::Scalar,
                                         ctx::FCIContext,
                                         setup_data::HCISetupData,
                                         epsilon::Float64) -> Int

Generate only excitations with |H| > epsilon using pre-computed data.

This is the efficient excitation generation from Holmes et al. (2016):
1. For doubles: Use pre-sorted lists to stop early when |H| < epsilon
2. For singles: Compute on-the-fly and discard if |H| < epsilon

Additionally, `H*coef` is computed and stored during generation for efficiency.

Works with both FCIContext and HCIContext.
"""
function generate_excitations!(excitations::VecDict{Determinant{OPattern}, Scalar},
                                             det::Determinant{OPattern},
                                             coef::Scalar,  
                                             ctx::Union{FCIContext{OPattern}, HCIContext{OPattern}},
                                             setup_data::HCISetupData,
                                             epsilon::Float64)::Int where OPattern
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
    add_excitations!(excitations, det, coef, double_alpha_excitation_phase, alpha_occ, setup_data.double_excitations, epsilon)
    # Beta-beta double excitations
    add_excitations!(excitations, det, coef, double_beta_excitation_phase, beta_occ, double_excitations_bb, epsilon)
    # Mixed double excitations (alpha-beta) (use int2ab pre-computed lists, i.e., no exchange)
    add_excitations!(excitations, det, coef, alpha_occ, beta_occ, n_orb, setup_data.double_excitations_ab, epsilon)
  end
  # ===========================================
  # 2. Generate single excitations with on-the-fly filtering using Fock elements
  # ===========================================
  
  int1 = ctx.int1a
  h1e2_same = ctx.heval_data.h1e2_aa
  h1e2_opp = ctx.heval_data.h1e2_ab
  # Alpha single excitations
  for i in alpha_occ
    for a in alpha_virt
      # Compute Fock matrix element f_ai
      h_val = compute_fock_element(int1, h1e2_same, h1e2_opp, alpha_occ, beta_occ, a, i)
      if abs(h_val) >= epsilon
        new_det, phase = single_alpha_excitation_phase(det, i, a)
        excitations[new_det] = coef * h_val * phase
      end
    end
  end
  
  int1 = ctx.int1b
  h1e2_same = ctx.heval_data.h1e2_bb
  h1e2_opp = ctx.heval_data.h1e2_ba
  # Beta single excitations
  for i in beta_occ
    for a in beta_virt
      # Compute Fock matrix element f_ai
      h_val = compute_fock_element(int1, h1e2_same, h1e2_opp, beta_occ, alpha_occ, a, i)
      if abs(h_val) >= epsilon
        new_det, phase = single_beta_excitation_phase(det, i, a)
        excitations[new_det] = coef * h_val * phase
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
@pib function compute_diagonal_element(det::Determinant, ctx::Union{FCIContext, HCIContext})::Scalar
  return calc_diagonalH(ctx.heval_data, det.alpha, det.beta)
end

"""
    compute_diagonal_element(occa::AbstractVector{Int}, occb::AbstractVector{Int}, ctx) -> Scalar

Compute diagonal matrix element ⟨det|H|det⟩ for a single determinant using HEvalData.
Works with both FCIContext and HCIContext.
"""
@pib function compute_diagonal_element(occa::AbstractVector{Int}, occb::AbstractVector{Int}, 
                                       ctx::Union{FCIContext, HCIContext})::Scalar
  return calc_diagonalH(ctx.heval_data, occa, occb)
end


# ===========================================
# Heat-Bath Selection 
# ===========================================

"""
    heatbath_selection(selected_ctx::SelectedCIContext,
                       variational_coeffs::AbstractVector{Scalar},
                       options::HCIOptions,
                       E_current::Float64,
                       setup_data::HCISetupData,
                       store_dets::Bool=true) -> (VecDict{Determinant, Scalar}, Float64)

Select determinants using Heat-Bath preselection + perturbative selection.
Uses efficient excitation generation with threshold-based filtering.
Works with both FCIContext and HCIContext.

Returns selected determinants together with the weights and PT2 energy correction.
If `store_dets` is false, only the PT2 energy is returned (with empty determinant list).
"""
function heatbath_selection(selected_ctx::SelectedCIContext,
                            variational_coeffs::AbstractVector{Scalar},
                            options::HCIOptions,
                            E_current::Float64,
                            setup_data::HCISetupData, store_dets::Bool=true)
  t0 = time_ns()
  variational_dets = determinants(selected_ctx)
  ctx = selected_ctx.base_context
  ThrNeglect = options.thr_negligible
  # Get HB candidates determinants from variational space together with the H*c values
  DetType = eltype(variational_dets)
  candidates = Dict{DetType, Scalar}()
  temp_buffer = VecDict{DetType, Scalar}()

  # Use efficient threshold-based excitation generation
  eps_h = options.epsilon_h > -0.1 ? options.epsilon_h : options.epsilon/10.0
  # first we run through new determinants
  n_olddet = n_old_dets(selected_ctx)
  for i in (n_olddet+1):length(variational_dets)
    det = variational_dets[i]
    c_I = variational_coeffs[i]
    if abs(c_I) < ThrNeglect
      continue  # Skip negligible coefficients
    end
    eps = eps_h / abs(c_I)
    generate_excitations!(temp_buffer, det, c_I, ctx, setup_data, eps)
    mergewith!(+, candidates, temp_buffer)
  end
  # Remove determinants already in variational space
  setdiff4dict!(candidates, variational_dets)
  # for the old determinants we only update those that were already added by the new ones.
  # this avoids adding the same determinants over and over again that will be neglected anyway by PT2
  # this has a very small effect on the final energy (coming only from the change of the correlation energy
  # in the denominator in PT2 step), but speeds up the selection significantly in later iterations.
  for i in 1:n_olddet
    det = variational_dets[i]
    c_I = variational_coeffs[i]
    if abs(c_I) < ThrNeglect
      continue  # Skip negligible coefficients
    end
    eps = eps_h / abs(c_I)
    generate_excitations!(temp_buffer, det, c_I, ctx, setup_data, eps)
    modifyvalueswith!(+, candidates, temp_buffer)
  end
  if options.verbose
    println("  Generated $(length(candidates)) candidate determinants")
  end
  t0 = print_time(options.print_level, t0, "candidate determinants", 1)

  # Select determinants above threshold or until target reached
  new_dets = VecDict{DetType, Scalar}()
  pt2_correction = 0.0
  # use square of epsilon_p to match probability definition (T_2^2)
  eps_p = options.epsilon_p > -0.1 ? options.epsilon_p : options.epsilon
  epsilon = eps_p^2
  for (det_J, Hc) in candidates
    # Compute H_JJ (diagonal element)
    H_JJ = compute_diagonal_element(det_J, ctx)
    ΔE_J = E_current - H_JJ
    
    # Selection probability: |Σ c_I H_IJ|² / ΔE²
    # Add small shift for numerical stability
    c_J² = abs2(Hc) / (ΔE_J^2 + 1e-10)
    pt2_correction += c_J² * ΔE_J  # Perturbative energy contribution
    if store_dets && c_J² >= epsilon
      new_dets[det_J] = c_J²
    end
  end
  t0 = print_time(options.print_level, t0, "perturbative selection", 1)
  return new_dets, pt2_correction
end

# ===========================================
# Main HCI Iteration Loop
# ===========================================

"""
    setup_hci!(ctx::FCIContext) -> HCISetupData

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
function setup_hci!(ctx::Union{FCIContext, HCIContext})::HCISetupData
  is_uhf = ctx.fcidump.uhf
  
  if !is_uhf
    # RHF case: use standard int2 integrals
    return setup_hci_rhf!(ctx)
  else
    # UHF case: use spin-separated integrals
    return setup_hci_uhf!(ctx)
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

function gen_triplets_list(n_orb::Int, int2::Array{Float64,4}, ThrNeglect::Float64=1e-10)
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
          h_val = int2[p, q, r, s] - int2[p, q, s, r]

          if abs(h_val) > ThrNeglect # Skip negligible matrix elements
            push!(triplets, (r, s, h_val))
            h_doub_max = max(h_doub_max, abs(h_val))
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(triplets, by=x->abs(x[3]), rev=true)
      
      # Store sorted list for this (p,q) pair
      push!(double_exc_lists, triplets)
    end
  end
  return double_exc_lists, h_doub_max
end

function gen_triplets_list_ab(n_orb::Int, int2ab::Array{Float64,4}, ThrNeglect::Float64=1e-10)
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
          h_val = int2ab[p, q, r, s]
          if abs(h_val) > ThrNeglect
            push!(triplets, (r, s, h_val))
            h_doub_max = max(h_doub_max, abs(h_val))
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(triplets, by=x->abs(x[3]), rev=true)
      push!(double_exc_ab_lists, triplets)
    end
  end
  return double_exc_ab_lists, h_doub_max
end
"""
    setup_hci_rhf!(ctx::Union{FCIContext, HCIContext}) -> HCISetupData

Setup for RHF systems using spatial orbital integrals.
"""
function setup_hci_rhf!(ctx::Union{FCIContext, HCIContext})::HCISetupData
  n_orb = ctx.n_orb
  
  # Dictionary to store sorted lists for each (p,q) pair
  double_exc_lists, h_doub_max = gen_triplets_list(n_orb, ctx.fcidump.int2, ctx.options.thr_negligible)
  double_exc_ab_lists, h_doub_max_ab = gen_triplets_list_ab(n_orb, ctx.fcidump.int2, ctx.options.thr_negligible)
  h_doub_max = max(h_doub_max, h_doub_max_ab)
  return HCISetupData(double_exc_lists, double_exc_ab_lists, h_doub_max)
end

"""
    setup_hci_uhf!(ctx::Union{FCIContext, HCIContext}) -> HCISetupData

Setup for UHF systems using spin-separated integrals.
Handles three types of double excitations:
- Alpha-alpha (using int2aa)
- Beta-beta (using int2bb)
- Mixed alpha-beta (using int2ab)
"""
function setup_hci_uhf!(ctx::Union{FCIContext, HCIContext})::HCISetupData
  n_orb = ctx.n_orb
  
  # Three dictionaries for the three types of double excitations
  double_exc_aa, h_doub_max_aa = gen_triplets_list(n_orb, ctx.fcidump.int2aa, ctx.options.thr_negligible)
  double_exc_bb, h_doub_max_bb = gen_triplets_list(n_orb, ctx.fcidump.int2bb, ctx.options.thr_negligible)
  double_exc_ab, h_doub_max_ab = gen_triplets_list_ab(n_orb, ctx.fcidump.int2ab, ctx.options.thr_negligible)
  h_doub_max = max(h_doub_max_aa, h_doub_max_bb, h_doub_max_ab)

  return HCISetupData(double_exc_aa, double_exc_bb, double_exc_ab, h_doub_max)
end

"""
    compute_pt2_correction!(selected_ctx, coefficients, E_var, setup_data, options)

Compute second-order perturbative correction to variational energy.

Following Holmes et al. (2016), Section III.B:
    ΔE = ∑_k [ (∑_i H_ki c_i)² / (E⁽⁰⁾ - H_kk) ]

where k runs over external determinants (not in variational space) and
i runs over internal determinants (in variational space).

Returns PT2 energies as a vector.
"""
function compute_pt2_correction!(selected_ctx::SelectedCIContext,
                                 coefficients::Matrix{Float64}, E_variational::Vector{Float64},
                                 setup_data::HCISetupData, options::HCIOptions)
  
  if !options.compute_pt2
    return Float64[]
  end
  
  if options.verbose
    println("\n" * "="^70)
    println("Computing PT2 Perturbative Correction")
    println("="^70)
    println("  Variational energy: $E_variational Ha")
    println("  Threshold ε_PT2: $(options.epsilon_pt2)")
    println("  Variational space size: $(n_selected(selected_ctx)) determinants")
  end
  # set old ndets to zero to ensure all determinants are used for PT2
  set_n_old_dets!(selected_ctx, 0)
  nstates = size(coefficients, 2)
  ΔE = zeros(Float64, nstates)
  save_epsilon_h = options.epsilon_h
  options.epsilon_h = options.epsilon_pt2
  for state_idx in 1:nstates
    if options.verbose
      println("\nState $state_idx:")
    end
    _, ΔE[state_idx] = heatbath_selection(selected_ctx, @view(coefficients[:, state_idx]), options,
                               E_variational[state_idx], setup_data, false)
    if options.verbose
      println("  PT2 correction: $(ΔE[state_idx]) Ha")
      println("  Total energy (VAR+PT2): $(E_variational[state_idx] + ΔE[state_idx]) Ha")
      println("="^70)
    end
  end
  options.epsilon_h = save_epsilon_h
  return ΔE  
end

"""
    run_heatbath_ci!(ctx::Union{FCIContext, HCIContext}, options::HCIOptions) 
      -> (Vector{Float64}, Matrix{Float64}, Vector{Determinant}, Vector{Float64})

Run Heat-Bath CI calculation with support for multiple states.

# Arguments
- `ctx`: FCI context
- `options`: Heat-Bath CI options (including nstates for multi-state)

# Returns
- `energies`: Vector of length nstates with total energies (electronic + nuclear)
- `coefficients`: Matrix (n_dets × nstates) with CI coefficients for all states
- `variational_dets`: Vector of determinants in final variational space
- `pt2_result`: PT2 correction results for all states

# Notes
- For nstates=1 (default), uses single-state selection strategy
- For nstates>1, uses multi-state selection with state-maximum probability
"""
function run_heatbath_ci!(ctx::Union{FCIContext{OPattern}, HCIContext{OPattern}}, options::HCIOptions)::Tuple{Vector{Scalar}, Matrix{Scalar}, Vector{Determinant{OPattern}}, Vector{Float64}} where OPattern
  if options.verbose
    println("\n" * "="^70)
    println("Heat-Bath Configuration Interaction (HCI)")
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
  
  # Setup
  # Pre-compute and store sorted double excitation matrix elements
  if options.verbose
    println("\nSetup - Pre-computing double excitation matrix elements")
    println("  Computing and sorting |H(rs ← pq)| for all orbital pairs...")
  end
  
  setup_data = setup_hci!(ctx)
  
  if options.verbose
    n_pairs = length(setup_data.double_excitations)
    total_triplets = sum(length(v) for v in values(setup_data.double_excitations))
    println("  Stored $(n_pairs) (p,q) pairs with $(total_triplets) total (r,s) triplets")
    println("  Maximum |H_doub|: $(setup_data.h_doub_max)")
  end
  
  # Enhanced initial guess using small-space Hamiltonian (if enabled)
  variational_dets = Determinant{OPattern}[]
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
  for iter in 1:options.max_iter
    if options.verbose
      println("\nHCI Iteration $iter:")
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
    new_dets_dict = Dict{Determinant{OPattern}, Scalar}()  # To hold selected new determinants
    pt2_corrections = zeros(Float64, options.nstates)
    for state = 1:options.nstates
      new_dets, pt2_corrections[state] = heatbath_selection(selected_ctx, @view(coeffs_matrix[:,state]), 
                                          options, E_electronic_vec[state], setup_data)
      # Merge new determinants from all states, taking maximum weight (for target_selection)
      mergewith!(max, new_dets_dict, new_dets)
    end
    if options.verbose
      if options.nstates == 1
        println("  PT2 correction: $(pt2_corrections[1]) Hartree")
      else
        println("  PT2 corrections:")
        for state in 1:options.nstates
          println("    State $state: $(pt2_corrections[state]) Hartree")
        end
      end
    end
    new_dets = collect(keys(new_dets_dict))
    target_size = options.target_selection - n_selected(selected_ctx)
    if options.target_selection > 0 && length(new_dets_dict) > target_size
      # Select top determinants by weight
      weights = collect(values(new_dets_dict))
      sorted_indices = sortperm(weights, rev=true)
      selected_indices = sorted_indices[1:target_size]
      new_dets = new_dets[selected_indices]
    end
    n_new = length(new_dets)
    
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
    @warn "HCI did not converge in $(options.max_iter) iterations"
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
    println("HCI Complete!")
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

  # Compute PT2 correction if requested
  pt2_result = Float64[]
  if options.compute_pt2
    pt2_result = compute_pt2_correction!(selected_ctx, coeffs_final_matrix, 
                                         E_electronic_vec, setup_data, options)

    E_total_with_pt2 = E_final_vec .+ pt2_result

    if options.verbose
      println("\nFinal Energies (Ground State with PT2):")
      println("  Variational:     $(E_final_vec[1]) Ha")
      println("  PT2 correction:  $(pt2_result[1]) Ha")
      println("  Total (VAR+PT2): $(E_total_with_pt2[1]) Ha")
      if options.nstates > 1
        println("\nFinal Energies (Excited States):")
        for state in 2:options.nstates
          println("  State $state: $(E_total_with_pt2[state]) Ha")
        end
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
function run_heatbath_ci!(ctx::HCIContext{OPattern})::Tuple{Vector{Scalar}, Matrix{Scalar}, Vector{Determinant{OPattern}}, Vector{Float64}} where OPattern
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
    shift = selected_ctx.base_context.options.shift,
    convergence_threshold = conv_tol,
    verbose = false
  )
  return real.(eigenvalues), real.(eigenvectors)
  
end