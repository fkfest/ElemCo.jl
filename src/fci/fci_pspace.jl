# fci_pspace.jl - P-space (P-space) algorithm implementation
# Implements enhanced initial guess generation and preconditioning

using LinearAlgebra

"""
    project_to_pspace(r::FCIVector, pspace_data::PSpaceData) -> Vector{Scalar}

Extract P-space components from full CI vector.
Returns a vector of size n_pspace containing the coefficients at P-space determinants.
"""
function project_to_pspace(r::FCIVector, pspace_data::PSpaceData)
  r_pspace = Vector{Scalar}(undef, pspace_data.n_pspace)
  for i in 1:(pspace_data.n_pspace)
    addr = pspace_data.indices[i]
    r_pspace[i] = r.data[addr]
  end
  return r_pspace
end

"""
    build_pspace_hamiltonian_selected!(context::FCIContext)

Build P-space Hamiltonian using selected CI framework.
"""
function build_pspace_hamiltonian_selected!(context::FCIContext)
  # Get P-space data from context
  pspace = context.pspace_data
  n_pspace = length(pspace.determinants)

  # Create selected CI context with new interface
  selected_ctx = setup_selected_ci_from_determinants!(context, pspace.determinants)

  # Initialize P-space Hamiltonian matrix
  H_pspace = zeros(Scalar, n_pspace, n_pspace)

  # For each P-space determinant
  for i in 1:n_pspace
    # Create unit vector for determinant i
    unit_vec = zeros(Scalar, n_pspace)
    unit_vec[i] = one(Scalar)

    # Apply Hamiltonian using new interface
    result_vec = zeros(Scalar, n_pspace)
    contract_hamiltonian_selected!(result_vec, unit_vec, selected_ctx, one(Scalar))

    # Store column of Hamiltonian matrix
    H_pspace[:, i] = result_vec
  end

  # Store in P-space data structure
  pspace.hamiltonian = H_pspace

  return nothing
end

"""
    get_reference_determinant(context::Union{FCIContext, HCIContext}) -> Determinant

Return the reference determinant.
This is used as the starting point for P-space determinant selection.
"""
function get_reference_determinant(context::Union{FCIContext, HCIContext})::Determinant
  return context.reference_det
end

"""
    count_excitation_level(det1::Determinant, det2::Determinant) -> Int

Count the excitation level between two determinants.
Returns the number of orbital differences (alpha + beta).
"""
function count_excitation_level(det1::Determinant, det2::Determinant)::Int
  alpha_diff = count_ones(det1.alpha ⊻ det2.alpha)
  beta_diff = count_ones(det1.beta ⊻ det2.beta)
  return (alpha_diff + beta_diff) ÷ 2  # Each excitation changes 2 orbitals
end

"""
    determinant_from_address(context::FCIContext, addr::Address) -> Determinant

Convert a determinant address to alpha/beta orbital patterns.
Uses the addressing tables to reconstruct the determinant.
"""
function determinant_from_address(context::FCIContext, addr::Address)
  # Convert 1-based address to 0-based for decomposition
  addr_0based = addr - 1

  # Convert linear address to (addr_a, addr_b) - both will be 0-based
  addr_a = addr_0based % context.adr_a.adr_count
  addr_b = addr_0based ÷ context.adr_a.adr_count

  # Convert addresses to orbital patterns using existing addressing functions
  alpha_pattern = make_pattern(context.adr_a, Address(addr_a+1))
  beta_pattern = make_pattern(context.adr_b, Address(addr_b+1))

  return Determinant(alpha_pattern, beta_pattern)
end

"""
    address_from_determinant(context::FCIContext, det::Determinant) -> Address

Convert a determinant to its linear address in the CI vector.
"""
function address_from_determinant(context::FCIContext, det::Determinant)::Address
  addr_a = context.adr_a(det.alpha)
  addr_b = context.adr_b(det.beta)

  # Calculate 1-based linear address
  return Address(addr_a + (addr_b - 1) * context.adr_a.adr_count)
end

"""
    select_pspace_determinants!(context::FCIContext)

Select P-space determinants based on the configured selection method.
Populates context.pspace_data with selected determinants and their indices.
"""
function select_pspace_determinants!(context::FCIContext)
  opts = context.options
  pspace = context.pspace_data

  # Build HF reference determinant
  hf_ref = get_reference_determinant(context)
  pspace.reference_det = hf_ref

  # Get diagonal Hamiltonian elements for energy-based selection
  n_total = n_data(context.coeff)

  if context.options.print_level >= 2
    println("Selecting P-space determinants:")
    println("  Method: $(opts.pspace_selection_method)")
    println("  Max size: $(opts.max_pspace_size)")
    println("  Max excitation: $(opts.max_pspace_excitation)")
    println("  Energy threshold: $(opts.pspace_energy_threshold)")
  end

  # Create candidate list with energy and excitation level
  candidates = Tuple{Address, Scalar, Int}[]  # (address, energy, excitation_level)

  # Calculate HF energy once (used for thresholding and sorting)
  hf_addr = address_from_determinant(context, hf_ref)
  hf_energy::Scalar = context.diag_h.data[hf_addr]

  for addr in Address(1):n_total
    det = determinant_from_address(context, addr)

    # Calculate excitation level from HF reference
    excitation_level = count_excitation_level(hf_ref, det)

    # Filter by maximum excitation level
    if excitation_level > opts.max_pspace_excitation
      continue
    end

    # Get diagonal energy for this determinant
    diagonal_energy = context.diag_h.data[addr]  # addr is now 1-based

    # Apply energy threshold (relative to HF diagonal energy)
    if diagonal_energy - hf_energy > opts.pspace_energy_threshold
      continue
    end

    push!(candidates, (addr, diagonal_energy, excitation_level))
  end

  # Sort candidates based on selection method
  if opts.pspace_selection_method == :energy
    # Sort by diagonal energy (lowest first)
    sort!(candidates, by = x -> x[2])
  elseif opts.pspace_selection_method == :excitation
    # Sort by excitation level, then by energy
    sort!(candidates, by = x -> (x[3], x[2]))
  elseif opts.pspace_selection_method == :hybrid
    # Balanced approach: weight both energy and excitation level
    sort!(candidates, by = x -> (x[3] * 0.1 + (x[2] - hf_energy)))
  else
    error("Unknown P-space selection method: $(opts.pspace_selection_method)")
  end

  # Select top candidates up to max_pspace_size
  n_selected = min(length(candidates), opts.max_pspace_size)

  # Resize arrays
  resize!(pspace.determinants, n_selected)
  resize!(pspace.indices, n_selected)

  # Store selected determinants
  for i in 1:n_selected
    addr, energy, excitation_level = candidates[i]
    pspace.determinants[i] = determinant_from_address(context, addr)
    pspace.indices[i] = addr
  end

  pspace.n_pspace = n_selected

  if context.options.print_level >= 2
    println("  Selected $(n_selected) determinants for P-space")
    if n_selected > 0
      min_energy = candidates[1][2]
      max_energy = candidates[n_selected][2]
      println("  Energy range: $min_energy to $max_energy Hartree")
    end
  end

  return nothing
end

"""
    build_pspace_hamiltonian!(context::FCIContext)

Build the P-space Hamiltonian matrix using efficient Selected CI machinery.
This replaces the slow full-space approach with fast selected-space operations.
"""
function build_pspace_hamiltonian!(context::FCIContext)
  # Use the new efficient Selected CI implementation
  build_pspace_hamiltonian_selected!(context)
  return nothing
end

"""
    diagonalize_pspace_hamiltonian!(context::FCIContext)

Diagonalize the P-space Hamiltonian matrix to obtain eigenvalues and eigenvectors.
"""
function diagonalize_pspace_hamiltonian!(context::FCIContext)
  pspace = context.pspace_data
  n_pspace = pspace.n_pspace

  if n_pspace == 0 || isempty(pspace.hamiltonian)
    error("P-space Hamiltonian not built")
  end

  if context.options.print_level >= 2
    println("Diagonalizing P-space Hamiltonian")
  end

  # Full diagonalization 
  eigenvals, eigenvecs = eigen(Symmetric(pspace.hamiltonian))

  # Store results (eigenvalues are already sorted by eigen())
  pspace.eigenvalues = eigenvals
  pspace.eigenvectors = eigenvecs

  if context.options.print_level >= 2
    n_print = min(3, n_pspace)
    println("P-space eigenvalues ($(n_print) lowest):")
    for i in 1:n_print
      println("  E[$i] = $(eigenvals[i]) Hartree")
    end
  end

  return nothing
end

# ===========================================
# Small-Space Initial Guess
# ===========================================

"""
    select_small_space_determinants(context::FCIContext, target_size::Int, nstates::Int=1) -> Vector{Determinant}

Select determinants for small-space Hamiltonian diagonalization.
Uses the same hybrid selection method as traditional P-space.

# Arguments
- `context`: FCI context
- `target_size`: Target number of determinants (adaptive: max(100, sqrt(target_selection), 5*nstates))
- `nstates`: Number of states to compute (used for sizing)

# Returns
- Vector of selected determinants
"""
function select_small_space_determinants(context::FCIContext{OPattern}, target_size, nstates=1) where OPattern
  # Build HF reference determinant
  hf_ref = get_reference_determinant(context)
  
  # Get diagonal Hamiltonian elements for energy-based selection
  n_total = n_data(context.coeff)
  
  if context.options.print_level >= 2
    println("  Small-space selection (hybrid method):")
    println("    Target size: $target_size determinants")
    println("    Total determinants: $n_total")
  end
  
  # Create candidate list with energy and excitation level
  # Use similar criteria as traditional P-space but with larger threshold
  candidates = Tuple{Determinant{OPattern}, Scalar, Int}[]  # (determinant, energy, excitation_level)
  
  hf_addr = address_from_determinant(context, hf_ref)
  hf_energy::Scalar = context.diag_h.data[hf_addr]
  
  # Use larger energy threshold for small-space (more permissive)
  pspace_energy_threshold = 1.0  # Hartree (much larger than typical P-space)
  max_pspace_excitation = 4      # Upto quadruple excitations (for efficiency)
  
  for addr in Address(1):n_total
    det = determinant_from_address(context, addr)
    
    # Calculate excitation level from HF reference
    excitation_level = count_excitation_level(hf_ref, det)
    
    # Filter by maximum excitation level
    if excitation_level > max_pspace_excitation
      continue
    end
    
    # Get diagonal energy for this determinant
    diagonal_energy = context.diag_h.data[addr]
    
    # Apply energy threshold (relative to HF diagonal energy)
    if diagonal_energy - hf_energy > pspace_energy_threshold
      continue
    end
    
    push!(candidates, (det, diagonal_energy, excitation_level))
  end
  
  # Sort using hybrid method: weight both energy and excitation level
  sort!(candidates, by = x -> (x[3] * 0.1 + (x[2] - hf_energy)))
  
  # Select top candidates up to target_size
  n_selected = min(length(candidates), target_size)
  
  if context.options.print_level >= 2
    println("    Selected $n_selected determinants")
    if n_selected > 0
      min_energy = candidates[1][2]
      max_energy = candidates[n_selected][2]
      println("    Energy range: $(min_energy - hf_energy) to $(max_energy - hf_energy) Ha above HF")
    end
  end
  
  # Extract determinants
  selected_dets = [candidates[i][1] for i in 1:n_selected]
  
  return selected_dets
end

"""
    select_small_space_determinants(context::HCIContext, target_size::Int, nstates::Int=1) -> Vector{Determinant}

Select determinants for small-space Hamiltonian diagonalization.
For HCIContext, generates determinants by excitation level from reference,
then calculates diagonal energies only for the generated determinants.

# Arguments
- `context`: HCI context
- `target_size`: Target number of determinants (adaptive: max(100, sqrt(target_selection), 5*nstates))
- `nstates`: Number of states to compute (used for sizing)

# Returns
- Vector of selected determinants
"""
function select_small_space_determinants(context::HCIContext{OPattern}, target_size, nstates=1) where OPattern
  # Build HF reference determinant
  hf_ref = get_reference_determinant(context)
  
  if context.options.print_level >= 2
    println("  Small-space selection (excitation-based for HCI):")
    println("    Target size: $target_size determinants")
  end
  
  # Generate determinants by excitation level (up to quadruples)
  max_pspace_excitation = 4
  
  # Extract occupied and virtual orbitals from reference
  n_alpha, n_beta = context.n_elec
  n_orb = context.n_orb
  
  alpha_occ = BufVec(zeros(Int, n_orb))
  alpha_virt = BufVec(zeros(Int, n_orb))
  occupied_and_virtual_orbitals!(alpha_occ, alpha_virt, hf_ref.alpha, n_orb)
  beta_occ = BufVec(zeros(Int, n_orb))
  beta_virt = BufVec(zeros(Int, n_orb))
  occupied_and_virtual_orbitals!(beta_occ, beta_virt, hf_ref.beta, n_orb)
  
  # Generate determinants by excitation level
  candidates_by_level = Vector{Determinant{OPattern}}[]
  
  # Level 0: HF reference
  push!(candidates_by_level, [hf_ref])
  
  if context.options.print_level >= 2
    println("    Level 0 (reference): 1 determinant")
  end
  
  # Level 1: Singles
  if max_pspace_excitation >= 1
    singles = Determinant{OPattern}[]
    for i in alpha_occ, a in alpha_virt
      push!(singles, single_excitation_alpha(hf_ref, i, a))
    end
    for i in beta_occ, a in beta_virt
      push!(singles, single_excitation_beta(hf_ref, i, a))
    end
    push!(candidates_by_level, singles)
    if context.options.print_level >= 2
      println("    Level 1 (singles): $(length(singles)) determinants")
    end
  end
  
  # Level 2: Doubles
  if max_pspace_excitation >= 2
    doubles = Determinant{OPattern}[]
    # Alpha-alpha doubles
    for i in 1:length(alpha_occ), j in 1:(i-1)
      for a in 1:length(alpha_virt), b in 1:(a-1)
        det = double_excitation_alpha(hf_ref, alpha_occ[i], alpha_occ[j], alpha_virt[a], alpha_virt[b])
        push!(doubles, det)
      end
    end
    # Beta-beta doubles
    for i in 1:length(beta_occ), j in 1:(i-1)
      for a in 1:length(beta_virt), b in 1:(a-1)
        det = double_excitation_beta(hf_ref, beta_occ[i], beta_occ[j], beta_virt[a], beta_virt[b])
        push!(doubles, det)
      end
    end
    # Alpha-beta mixed doubles
    for i in alpha_occ, a in alpha_virt
      for j in beta_occ, b in beta_virt
        det = double_excitation_mixed(hf_ref, i, j, a, b)
        push!(doubles, det)
      end
    end
    push!(candidates_by_level, doubles)
    if context.options.print_level >= 2
      println("    Level 2 (doubles): $(length(doubles)) determinants")
    end
  end
  
  # Level 3 and 4: Can be added if needed, but typically doubles are sufficient
  # For now, we'll stop at doubles for efficiency
  
  # Flatten and calculate diagonal energies
  candidates = Tuple{Determinant{OPattern}, Scalar, Int}[]  # (determinant, energy, excitation_level)
  
  for (level, dets) in enumerate(candidates_by_level)
    for det in dets
      # Calculate diagonal energy on-the-fly using compute_diagonal_element
      diagonal_energy = compute_diagonal_element(det, context)
      push!(candidates, (det, diagonal_energy, level-1))
    end
  end
  
  # Calculate HF energy for sorting
  hf_energy = compute_diagonal_element(hf_ref, context)
  
  # Sort using hybrid method: weight both energy and excitation level
  sort!(candidates, by = x -> (x[3] * 0.1 + (x[2] - hf_energy)))
  
  # Select top candidates up to target_size
  n_selected = min(length(candidates), target_size)
  
  if context.options.print_level >= 2
    println("    Total candidates: $(length(candidates))")
    println("    Selected: $n_selected determinants")
    if n_selected > 0
      min_energy = candidates[1][2]
      max_energy = candidates[n_selected][2]
      println("    Energy range: $(min_energy - hf_energy) to $(max_energy - hf_energy) Ha above HF")
    end
  end
  
  # Extract determinants
  selected_dets = [candidates[i][1] for i in 1:n_selected]
  
  return selected_dets
end

"""
    build_small_space_hamiltonian(context::Union{FCIContext, HCIContext}, determinants::Vector{Determinant}) -> Matrix{Scalar}

Build Hamiltonian matrix for small space of determinants.
Uses Selected CI framework for efficient matrix element computation.

# Arguments
- `context`: FCI or HCIcontext
- `determinants`: Vector of determinants spanning the small space

# Returns
- Hamiltonian matrix H[i,j] = ⟨det_i|H|det_j⟩
"""
function build_small_space_hamiltonian(context::Union{FCIContext,HCIContext}, determinants)::Matrix{Scalar}
  n_small = length(determinants)
  
  if context.options.print_level >= 2
    println("  Building small-space Hamiltonian ($n_small × $n_small)")
  end
  
  # Create selected CI context
  selected_ctx = setup_selected_ci_from_determinants!(context, determinants)
  
  # Build Hamiltonian matrix using Selected CI machinery
  H_small = hamiltonian_matrix(selected_ctx)
  
  return H_small
end

"""
    SmallSpaceResult

Result from small-space Hamiltonian diagonalization.
"""
struct SmallSpaceResult{OPattern}
  determinants::Vector{Determinant{OPattern}}    # Determinants in small space
  eigenvalues::Vector{Float64}         # Eigenvalues (nstates lowest)
  eigenvectors::Matrix{Float64}        # Eigenvectors in small-space basis (n_small × nstates)
  n_small::Int                         # Size of small space
  nstates::Int                         # Number of states computed
end

"""
    initialize_multistate_from_small_space(context::Union{FCIContext, HCIContext}, target_selection::Int, nstates::Int) -> SmallSpaceResult

Initialize multi-state HCI using small-space Hamiltonian diagonalization.
This provides better initial guesses for all states, preventing missed excited states.

# Algorithm
1. Select small space: max(100, sqrt(target_selection), 5*nstates) determinants
2. Build Hamiltonian in small space
3. Diagonalize to get nstates lowest eigenstates
4. Return determinants and eigenvectors as initial guess for HCI

# Arguments
- `context`: FCI or HCI context
- `target_selection`: Target HCI variational space size (for adaptive sizing)
- `nstates`: Number of states to compute

# Returns
- `SmallSpaceResult` containing determinants, eigenvalues, and eigenvectors
"""
function initialize_multistate_from_small_space(context::Union{FCIContext, HCIContext{OPattern}},
                                target_selection::Int, nstates::Int)::SmallSpaceResult{OPattern} where OPattern
  
  if context.options.print_level >= 1
    println("\nSmall-Space Initial Guess Generation")
  end
  
  # 1. Determine small-space size (adaptive)
  small_space_size = max(100, trunc(Int,sqrt(target_selection)), 5 * nstates)
  
  if context.options.print_level >= 1
    println("  Adaptive sizing: max(100, sqrt($target_selection), 5×$nstates) = $small_space_size")
  end
  
  # 2. Select determinants using hybrid method (same as traditional P-space)
  small_space_dets = select_small_space_determinants(context, small_space_size, nstates)
  n_small = length(small_space_dets)
  
  if n_small < nstates
    error("Small-space size ($n_small) < nstates ($nstates). Cannot compute $nstates states.")
  end
  
  # 3. Build Hamiltonian in small space
  H_small = build_small_space_hamiltonian(context, small_space_dets)
  
  # 4. Diagonalize for nstates lowest eigenstates
  if context.options.print_level >= 2
    println("  Diagonalizing small-space Hamiltonian for $nstates states")
  end
  
  eigenvals, eigenvecs = eigen(Hermitian(H_small))

  # Extract nstates lowest states
  eigenvalues_selected = eigenvals[1:nstates]
  eigenvectors_selected = eigenvecs[:, 1:nstates]
  
  if context.options.print_level >= 1
    println("  Small-space energies (electronic):")
    for (i, E) in enumerate(eigenvalues_selected)
      E_total = E + context.fcidump.int0
      println("    State $i: $E_total Hartree (electronic: $E)")
    end
  end
  
  return SmallSpaceResult{OPattern}(small_space_dets, eigenvalues_selected, eigenvectors_selected,
                          n_small, nstates)
end

"""
    setup_pspace!(context::FCIContext, n_states::Int=1)

Complete P-space setup: select determinants, build Hamiltonian, and diagonalize.
This is the main entry point for P-space initialization.

Supports two modes:
1. Traditional: Excitation-based or energy-based determinant selection
2. HBCI: Uses Heat-Bath CI to select important determinants (more efficient)

# Arguments
- `context`: FCI context
- `n_states`: Number of states to compute in subsequent Davidson (for HBCI multi-state P-space)
"""
function setup_pspace!(context::FCIContext, n_states::Int=1)
  if context.options.print_level >= 1
    println("Setting up P-space for enhanced initial guess")
    if n_states > 1 && context.options.pspace_selection_method == :hci
      println("  Multi-state P-space: computing $n_states roots in HBCI")
    end
  end

  # Diagonal Hamiltonian should already be computed during init_hamiltonian_terms!
  if all(x -> x == 0, context.diag_h.data)
    error("Diagonal Hamiltonian not initialized. Call init_hamiltonian_terms! first.")
  end

  # Check if HBCI-based P-space selection is enabled
  if context.options.pspace_selection_method == :hci
    # Use Heat-Bath CI for P-space selection
    # Pass n_states so HCI computes the same number of roots
    setup_pspace_hci!(context, n_states)
  else
    # Traditional P-space selection (single-state only for now)
    select_pspace_determinants!(context)
  end

  if context.pspace_data.n_pspace == 0
    if context.options.print_level >= 1
      println("Warning: No P-space determinants selected, using standard initial guess")
    end
    return
  end

  # Build and diagonalize P-space Hamiltonian
  build_pspace_hamiltonian!(context)
  diagonalize_pspace_hamiltonian!(context)

  if context.options.print_level >= 1
    println("P-space setup complete ($(context.pspace_data.n_pspace) determinants)")
  end

  return nothing
end

"""
    setup_pspace_hci!(context::FCIContext, n_states::Int=1)

Use Heat-Bath CI to select P-space determinants.
This provides a more efficient and targeted selection compared to traditional methods.

The selected determinants from HCI variational space become the P-space for
subsequent full FCI Davidson iterations.

# Arguments
- `context`: FCI context
- `n_states`: Number of states to compute (HCI will compute the same number of roots)
"""
function setup_pspace_hci!(context::FCIContext, n_states::Int=1)
  opts = context.options
  
  if context.options.print_level >= 1
    println("  Using Heat-Bath CI for P-space selection")
    println("  Target size: $(opts.max_pspace_size)")
    println("  HCI ε₁: $(opts.pspace_hci_epsilon)")
    if n_states > 1
      println("  HCI nstates: $n_states (matching FCI)")
    end
  end
  
  # Configure HCI options for P-space generation
  # CRITICAL: Use same nstates as the final FCI calculation for multi-state
  hci_options = HCIOptions(
    target_selection = opts.max_pspace_size,
    epsilon = opts.pspace_hci_epsilon,
    tol = 1e-6,  # Convergence threshold for HCI iterations
    max_iter = 10,
    shift = opts.shift,
    compute_pt2 = false,  # Don't need PT2 for P-space
    verbose = false,  # Keep HCI output minimal
    nstates = n_states  # Match the number of states in the final FCI calculation
  )

  # Run HCI to get selected determinants
  # Returns: E_vec (Vector{Float64}), coeffs_matrix (Matrix{Float64}), dets, pt2_result
  E_hci_vec, coeffs_hci_matrix, dets_hci, _ = run_heatbath_ci!(context, hci_options)
  
  if context.options.print_level >= 1
    println("  HBCI selected $(length(dets_hci)) determinants")
    if n_states == 1
      println("  HBCI energy: $(E_hci_vec[1]) Hartree")
    else
      println("  HBCI energies:")
      for (i, E) in enumerate(E_hci_vec)
        println("    State $i: $E Hartree")
      end
    end
  end
  
  # Convert HBCI determinants to P-space indices
  n_selected = length(dets_hci)
  
  # Limit to requested max_pspace_size if HBCI selected more
  if n_selected > opts.max_pspace_size
    @warn "HBCI selected $n_selected determinants, exceeding requested max_pspace_size $(opts.max_pspace_size). Truncating."
    n_selected = opts.max_pspace_size
  end
  
  # Resize P-space storage arrays
  resize!(context.pspace_data.indices, n_selected)
  resize!(context.pspace_data.determinants, n_selected)
  
  # Store selected determinants and their addresses
  for i in 1:n_selected
    det = dets_hci[i]
    addr = address_from_determinant(context, det)
    context.pspace_data.determinants[i] = det
    context.pspace_data.indices[i] = addr
  end
  
  context.pspace_data.n_pspace = n_selected
  
  # Resize eigenvalue/eigenvector storage (will be filled by diagonalize_pspace_hamiltonian!)
  # For now, pre-populate with HBCI results
  context.pspace_data.eigenvalues = zeros(Scalar, n_selected)
  context.pspace_data.eigenvectors = zeros(Scalar, n_selected, n_selected)
  
  # Store HBCI eigenvector coefficients for ground state as first eigenvector (good initial guess)
  # coeffs_hbci_matrix is (n_dets × nstates), extract ground state coefficients
  if size(coeffs_hci_matrix, 1) == n_selected
    context.pspace_data.eigenvalues[1] = E_hci_vec[1] - context.fcidump.int0  # Electronic energy
    context.pspace_data.eigenvectors[:, 1] = coeffs_hci_matrix[:, 1]  # Ground state coefficients
  end
  
  return nothing
end

# ===========================================
# PSpace Enhanced Initial Guess Generation
# ===========================================

"""
    project_pspace_to_fullspace!(v_full::FCIVector, v_pspace::Vector{Scalar}, 
                                 pspace_data::PSpaceData)

Project P-space eigenvector onto full CI space.
Zeros coefficients for determinants not in P-space and normalizes.
"""
function project_pspace_to_fullspace!(v_full::FCIVector, v_pspace::Vector{Scalar}, pspace_data::PSpaceData)
  # Zero the full vector
  fill!(v_full.data, 0.0)

  # Project P-space coefficients onto full space
  for i in 1:(pspace_data.n_pspace)
    addr = pspace_data.indices[i]
    v_full.data[addr] = v_pspace[i]  # addr is now 1-based
  end

  # Normalize the resulting vector
  normalize!(v_full)

  return nothing
end

"""
    generate_pspace_initial_guess!(context::FCIContext, guess_vectors::Vector{FCIVector},
                                  n_states::Int)

Generate high-quality initial guess vectors using P-space eigenvectors.
This replaces the diagonal-based initial guess with P-space enhanced vectors.
"""
function generate_pspace_initial_guess!(context::FCIContext, guess_vectors, n_states::Int)
  pspace = context.pspace_data

  if pspace.n_pspace == 0 || isempty(pspace.eigenvectors)
    if context.options.print_level >= 2
      println("P-space not available, using diagonal-based initial guess")
    end
    return false  # Fall back to diagonal guess
  end

  if context.options.print_level >= 2
    println("Generating P-space enhanced initial guess for $n_states states")
  end

  # Use P-space eigenvectors as initial guesses
  n_available = min(n_states, pspace.n_pspace, size(pspace.eigenvectors, 2))

  for i in 1:n_available
    pspace_eigenvec = pspace.eigenvectors[:, i]  # i-th eigenstate
    project_pspace_to_fullspace!(guess_vectors[i], pspace_eigenvec, pspace)

    if context.options.print_level >= 3
      pspace_energy = pspace.eigenvalues[i]
      println("  Initial guess $i: P-space energy = $pspace_energy")
    end
  end

  # If we need more states than P-space provides, fill remaining with diagonal guess
  if n_available < n_states
    if context.options.print_level >= 2
      println("  Using diagonal guess for remaining $(n_states - n_available) states")
    end
    return false  # Partial success - caller should handle remaining states
  end

  return true  # Complete success
end
