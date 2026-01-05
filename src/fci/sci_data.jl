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
  function SelectedHamiltonianMatrix(hermitian::Bool=true)
    new([], hermitian)
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
      h_ji = compute_matrix_element_direct(det_i, det_j, context, occa, occb)
      if sel_ham.hermitian
        h_ij = h_ji
      else
        h_ij = compute_matrix_element_direct(det_j, det_i, context)
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
    get_diagonal_element(sel_ham::SelectedHamiltonianMatrix, idet::Int) -> Scalar

Get diagonal Hamiltonian element for selected determinant index.
"""
function get_diagonal_element(sel_ham::SelectedHamiltonianMatrix, idet::Int)
  @assert_devel idet <= length(sel_ham.rows) "Invalid determinant index"
  return getvalueat(sel_ham.rows[idet], 1)
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
    is_hermitian(ctx::SelectedCIContext) -> Bool

Get whether the Hamiltonian is Hermitian.
"""
is_hermitian(ctx::SelectedCIContext) = is_hermitian(ctx.base_context)

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

"""
    get_diagonal_element(selected_ctx::SelectedCIContext, idet::Int) -> Scalar

Get diagonal Hamiltonian element for selected determinant index.
"""
get_diagonal_element(selected_ctx::SelectedCIContext, idet::Int) =
  get_diagonal_element(selected_ctx.hamiltonian, idet)