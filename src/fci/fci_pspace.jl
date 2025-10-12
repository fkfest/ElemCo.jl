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
    build_hf_reference_determinant(context::FCIContext) -> Determinant

Construct the Hartree-Fock reference determinant from the FCIDUMP data.
This is used as the starting point for P-space determinant selection.
"""
function build_hf_reference_determinant(context::FCIContext)::Determinant
  n_elec = context.fcidump.n_elec
  n_orb = context.fcidump.n_orb
  n_spin = context.fcidump.n_spin

  # Number of alpha and beta electrons
  n_elec_a = (n_elec + n_spin) ÷ 2
  n_elec_b = (n_elec - n_spin) ÷ 2

  # Build HF determinant: occupy lowest n_elec_a/n_elec_b orbitals
  alpha_pattern = OrbPattern(0)
  beta_pattern = OrbPattern(0)

  for i in 0:(n_elec_a - 1)
    alpha_pattern |= (OrbPattern(1) << i)
  end

  for i in 0:(n_elec_b - 1)
    beta_pattern |= (OrbPattern(1) << i)
  end

  return Determinant(alpha_pattern, beta_pattern)
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
function determinant_from_address(context::FCIContext, addr::Address)::Determinant
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
  pspace_opts = context.options.pspace_options
  pspace = context.pspace_data

  # Build HF reference determinant
  hf_ref = build_hf_reference_determinant(context)
  pspace.reference_det = hf_ref

  # Get diagonal Hamiltonian elements for energy-based selection
  n_total = n_data(context.coeff)

  if context.options.print_level >= 2
    println("Selecting P-space determinants:")
    println("  Method: $(pspace_opts.selection_method)")
    println("  Max size: $(pspace_opts.max_size)")
    println("  Max excitation: $(pspace_opts.max_excitation)")
    println("  Energy threshold: $(pspace_opts.energy_threshold)")
  end

  # Create candidate list with energy and excitation level
  candidates = Tuple{Address, Scalar, Int}[]  # (address, energy, excitation_level)

  for addr in Address(1):n_total
    det = determinant_from_address(context, addr)

    # Calculate excitation level from HF reference
    excitation_level = count_excitation_level(hf_ref, det)

    # Filter by maximum excitation level
    if excitation_level > pspace_opts.max_excitation
      continue
    end

    # Get diagonal energy for this determinant
    diagonal_energy = context.diag_h.data[addr]  # addr is now 1-based

    # Apply energy threshold (relative to HF diagonal energy)
    hf_addr = address_from_determinant(context, hf_ref)
    hf_energy = context.diag_h.data[hf_addr]  # hf_addr is now 1-based

    if diagonal_energy - hf_energy > pspace_opts.energy_threshold
      continue
    end

    push!(candidates, (addr, diagonal_energy, excitation_level))
  end

  # Sort candidates based on selection method
  if pspace_opts.selection_method == :energy
    # Sort by diagonal energy (lowest first)
    sort!(candidates, by = x -> x[2])
  elseif pspace_opts.selection_method == :excitation
    # Sort by excitation level, then by energy
    sort!(candidates, by = x -> (x[3], x[2]))
  elseif pspace_opts.selection_method == :hybrid
    # Balanced approach: weight both energy and excitation level
    hf_energy = context.diag_h.data[address_from_determinant(context, hf_ref)]  # address is now 1-based
    sort!(candidates, by = x -> (x[3] * 0.1 + (x[2] - hf_energy)))
  else
    error("Unknown P-space selection method: $(pspace_opts.selection_method)")
  end

  # Select top candidates up to max_size
  n_selected = min(length(candidates), pspace_opts.max_size)

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
    select_small_space_determinants(context::FCIContext, target_size::Int, n_roots::Int=1) -> Vector{Determinant}

Select determinants for small-space Hamiltonian diagonalization.
Uses the same hybrid selection method as traditional P-space.

# Arguments
- `context`: FCI context
- `target_size`: Target number of determinants (adaptive: max(100, target_selection÷10, 5*n_roots))
- `n_roots`: Number of states to compute (used for sizing)

# Returns
- Vector of selected determinants
"""
function select_small_space_determinants(context::FCIContext, target_size::Int, n_roots::Int=1)::Vector{Determinant}
  # Build HF reference determinant
  hf_ref = build_hf_reference_determinant(context)
  
  # Get diagonal Hamiltonian elements for energy-based selection
  n_total = n_data(context.coeff)
  
  if context.options.print_level >= 2
    println("  Small-space selection (hybrid method):")
    println("    Target size: $target_size determinants")
    println("    Total determinants: $n_total")
  end
  
  # Create candidate list with energy and excitation level
  # Use similar criteria as traditional P-space but with larger threshold
  candidates = Tuple{Determinant, Scalar, Int}[]  # (determinant, energy, excitation_level)
  
  hf_addr = address_from_determinant(context, hf_ref)
  hf_energy = context.diag_h.data[hf_addr]
  
  # Use larger energy threshold for small-space (more permissive)
  energy_threshold = 1.0  # Hartree (much larger than typical P-space)
  max_excitation = 2      # Singles and doubles only (for efficiency)
  
  for addr in Address(1):n_total
    det = determinant_from_address(context, addr)
    
    # Calculate excitation level from HF reference
    excitation_level = count_excitation_level(hf_ref, det)
    
    # Filter by maximum excitation level
    if excitation_level > max_excitation
      continue
    end
    
    # Get diagonal energy for this determinant
    diagonal_energy = context.diag_h.data[addr]
    
    # Apply energy threshold (relative to HF diagonal energy)
    if diagonal_energy - hf_energy > energy_threshold
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
    build_small_space_hamiltonian(context::FCIContext, determinants::Vector{Determinant}) -> Matrix{Scalar}

Build Hamiltonian matrix for small space of determinants.
Uses Selected CI framework for efficient matrix element computation.

# Arguments
- `context`: FCI context
- `determinants`: Vector of determinants spanning the small space

# Returns
- Hamiltonian matrix H[i,j] = ⟨det_i|H|det_j⟩
"""
function build_small_space_hamiltonian(context::FCIContext, determinants::Vector{Determinant})::Matrix{Scalar}
  n_small = length(determinants)
  
  if context.options.print_level >= 2
    println("  Building small-space Hamiltonian ($n_small × $n_small)")
  end
  
  # Create selected CI context
  selected_ctx = setup_selected_ci_from_determinants!(context, determinants)
  
  # Build Hamiltonian matrix using Selected CI machinery
  H_small = zeros(Scalar, n_small, n_small)
  
  for i in 1:n_small
    # Create unit vector for determinant i
    unit_vec = zeros(Scalar, n_small)
    unit_vec[i] = one(Scalar)
    
    # Apply Hamiltonian: H * e_i gives column i
    result_vec = zeros(Scalar, n_small)
    contract_hamiltonian_selected!(result_vec, unit_vec, selected_ctx, one(Scalar))
    
    # Store column of Hamiltonian matrix
    H_small[:, i] = result_vec
  end
  
  return H_small
end

"""
    SmallSpaceResult

Result from small-space Hamiltonian diagonalization.
"""
struct SmallSpaceResult
  determinants::Vector{Determinant}    # Determinants in small space
  eigenvalues::Vector{Float64}         # Eigenvalues (n_roots lowest)
  eigenvectors::Matrix{Float64}        # Eigenvectors in small-space basis (n_small × n_roots)
  n_small::Int                         # Size of small space
  n_roots::Int                         # Number of states computed
end

"""
    initialize_multistate_from_small_space(context::FCIContext, target_selection::Int, n_roots::Int) -> SmallSpaceResult

Initialize multi-state HBCI using small-space Hamiltonian diagonalization.
This provides better initial guesses for all states, preventing missed excited states.

# Algorithm
1. Select small space: max(100, target_selection÷10, 5*n_roots) determinants
2. Build Hamiltonian in small space
3. Diagonalize to get n_roots lowest eigenstates
4. Return determinants and eigenvectors as initial guess for HBCI

# Arguments
- `context`: FCI context
- `target_selection`: Target HBCI variational space size (for adaptive sizing)
- `n_roots`: Number of states to compute

# Returns
- `SmallSpaceResult` containing determinants, eigenvalues, and eigenvectors
"""
function initialize_multistate_from_small_space(
  context::FCIContext,
  target_selection::Int,
  n_roots::Int
)::SmallSpaceResult
  
  if context.options.print_level >= 1
    println("\nSmall-Space Initial Guess Generation")
  end
  
  # 1. Determine small-space size (adaptive)
  small_space_size = max(100, target_selection ÷ 10, 5 * n_roots)
  
  if context.options.print_level >= 1
    println("  Adaptive sizing: max(100, $target_selection÷10, 5×$n_roots) = $small_space_size")
  end
  
  # 2. Select determinants using hybrid method (same as traditional P-space)
  small_space_dets = select_small_space_determinants(context, small_space_size, n_roots)
  n_small = length(small_space_dets)
  
  if n_small < n_roots
    error("Small-space size ($n_small) < n_roots ($n_roots). Cannot compute $n_roots states.")
  end
  
  # 3. Build Hamiltonian in small space
  H_small = build_small_space_hamiltonian(context, small_space_dets)
  
  # 4. Diagonalize for n_roots lowest eigenstates
  if context.options.print_level >= 2
    println("  Diagonalizing small-space Hamiltonian for $n_roots states")
  end
  
  eigenvals, eigenvecs = eigen(Hermitian(H_small))
  
  # Extract n_roots lowest states
  eigenvalues_selected = eigenvals[1:n_roots]
  eigenvectors_selected = eigenvecs[:, 1:n_roots]
  
  if context.options.print_level >= 1
    println("  Small-space energies (electronic):")
    for (i, E) in enumerate(eigenvalues_selected)
      E_total = E + context.fcidump.e_nuc
      println("    State $i: $E_total Hartree (electronic: $E)")
    end
  end
  
  return SmallSpaceResult(
    small_space_dets,
    eigenvalues_selected,
    eigenvectors_selected,
    n_small,
    n_roots
  )
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
    if n_states > 1 && context.options.pspace_options.use_hbci
      println("  Multi-state P-space: computing $n_states roots in HBCI")
    end
  end

  # Diagonal Hamiltonian should already be computed during init_hamiltonian_terms!
  if all(x -> x == 0, context.diag_h.data)
    error("Diagonal Hamiltonian not initialized. Call init_hamiltonian_terms! first.")
  end

  # Check if HBCI-based P-space selection is enabled
  if context.options.pspace_options.use_hbci
    # Use Heat-Bath CI for P-space selection
    # Pass n_states so HBCI computes the same number of roots
    setup_pspace_hbci!(context, n_states)
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
    setup_pspace_hbci!(context::FCIContext, n_states::Int=1)

Use Heat-Bath CI to select P-space determinants.
This provides a more efficient and targeted selection compared to traditional methods.

The selected determinants from HBCI variational space become the P-space for
subsequent full FCI Davidson iterations.

# Arguments
- `context`: FCI context
- `n_states`: Number of states to compute (HBCI will compute the same number of roots)
"""
function setup_pspace_hbci!(context::FCIContext, n_states::Int=1)
  pspace_opts = context.options.pspace_options
  
  if context.options.print_level >= 1
    println("  Using Heat-Bath CI for P-space selection")
    println("  Target size: $(pspace_opts.max_size)")
    println("  HBCI ε₁: $(pspace_opts.hbci_epsilon)")
    if n_states > 1
      println("  HBCI n_roots: $n_states (matching FCI)")
    end
  end
  
  # Configure HBCI options for P-space generation
  # CRITICAL: Use same n_roots as the final FCI calculation for multi-state
  hbci_options = HeatBathCIOptions(
    target_selection = pspace_opts.max_size,
    epsilon_1 = pspace_opts.hbci_epsilon,
    epsilon_2 = 1e-8,  # Convergence threshold for HBCI iterations
    max_iterations = 10,
    use_stochastic = false,
    use_setup_phase = pspace_opts.hbci_use_setup_phase,
    compute_pt2 = false,  # Don't need PT2 for P-space
    verbose = false,  # Keep HBCI output minimal
    n_roots = n_states  # Match the number of states in the final FCI calculation
  )
  
  # Run HBCI to get selected determinants
  # Returns: E_vec (Vector{Float64}), coeffs_matrix (Matrix{Float64}), dets, pt2_result
  E_hbci_vec, coeffs_hbci_matrix, dets_hbci, _ = run_heatbath_ci!(context, hbci_options)
  
  if context.options.print_level >= 1
    println("  HBCI selected $(length(dets_hbci)) determinants")
    if n_states == 1
      println("  HBCI energy: $(E_hbci_vec[1]) Hartree")
    else
      println("  HBCI energies:")
      for (i, E) in enumerate(E_hbci_vec)
        println("    State $i: $E Hartree")
      end
    end
  end
  
  # Convert HBCI determinants to P-space indices
  n_selected = length(dets_hbci)
  
  # Limit to requested max_size if HBCI selected more
  if n_selected > pspace_opts.max_size
    @warn "HBCI selected $n_selected determinants, exceeding requested max_size $(pspace_opts.max_size). Truncating."
    n_selected = pspace_opts.max_size
  end
  
  # Resize P-space storage arrays
  resize!(context.pspace_data.indices, n_selected)
  resize!(context.pspace_data.determinants, n_selected)
  
  # Store selected determinants and their addresses
  for i in 1:n_selected
    det = dets_hbci[i]
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
  # coeffs_hbci_matrix is (n_dets × n_roots), extract ground state coefficients
  if size(coeffs_hbci_matrix, 1) == n_selected
    context.pspace_data.eigenvalues[1] = E_hbci_vec[1] - context.fcidump.e_nuc  # Electronic energy
    context.pspace_data.eigenvectors[:, 1] = coeffs_hbci_matrix[:, 1]  # Ground state coefficients
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
function project_pspace_to_fullspace!(
  v_full::FCIVector,
  v_pspace::Vector{Scalar},
  pspace_data::PSpaceData,
)
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
function generate_pspace_initial_guess!(
  context::FCIContext,
  guess_vectors::Vector{FCIVector},
  n_states::Int,
)
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
