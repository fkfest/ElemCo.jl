
"""
Selected CI implementation

This provides:
1. Efficient P-space calculations with O(N²) scaling and O(N) memory
2. Standalone Selected CI solver capability  
3. Direct integration with Davidson solvers
4. On-the-fly matrix element evaluation using Slater-Condon rules
"""

using LinearAlgebra

include("sci_data.jl")
include("sci_excitations.jl")
include("sci_hmatrix.jl")
include("sci_pchb.jl")
include("sci_fock.jl")
include("sci_hb_selection.jl")


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
# Main HCI Iteration Loop
# ===========================================

"""
    run_heatbath_ci!(ctx::Union{FCIContext, HCIContext}, options::HCIOptions) 
      -> (Vector{Float64}, Matrix{Float64}, Vector{Determinant}, Vector{(Float64, Float64)})

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
function run_heatbath_ci!(ctx::Union{FCIContext{OPattern}, HCIContext{OPattern}}, options::HCIOptions) where OPattern
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
    n_pairs = length(setup_data.double_excitations_aa)
    total_triplets = sum(length(v) for v in values(setup_data.double_excitations_aa))
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
    pt2_corrections = Array{Tuple{Float64, Float64}}(undef, options.nstates)
    for state = 1:options.nstates
      new_dets, pt2_corrections[state] = heatbath_selection(selected_ctx, @view(coeffs_matrix[:,state]), 
                                          options, E_electronic_vec[state], setup_data)
      # Merge new determinants from all states, taking maximum weight (for target_selection)
      mergewith!(max, new_dets_dict, new_dets)
    end
    if options.verbose
      if options.nstates == 1
        println("  PT2 correction: $(pt2_corrections[1][1]) ± $(pt2_corrections[1][2]) Hartree")
      else
        println("  PT2 corrections:")
        for state in 1:options.nstates
          println("    State $state: $(pt2_corrections[state][1]) ± $(pt2_corrections[state][2]) Hartree")
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
  pt2_result = Array{Tuple{Float64, Float64}}(undef, 0)
  if options.compute_pt2
    pt2_result = compute_pt2_correction!(selected_ctx, coeffs_final_matrix, 
                                         E_electronic_vec, setup_data, options)

    E_total_with_pt2 = E_final_vec .+ [e[1] for e in pt2_result]

    if options.verbose
      println("\nFinal Energies (Ground State with PT2):")
      println("  Variational:     $(E_final_vec[1]) Ha")
      println("  PT2 correction:  $(pt2_result[1][1]) ± $(pt2_result[1][2]) Ha")
      println("  Total (VAR+PT2): $(E_total_with_pt2[1]) ± $(pt2_result[1][2]) Ha")
      if options.nstates > 1
        println("\nFinal Energies (Excited States):")
        for state in 2:options.nstates
          println("  State $state: $(E_total_with_pt2[state]) ± $(pt2_result[state][2]) Ha")
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
function run_heatbath_ci!(ctx::HCIContext{OPattern}) where OPattern
  return run_heatbath_ci!(ctx, ctx.options)
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
  ΔE = Array{Tuple{Float64, Float64}}(undef, nstates)
  save_epsilon_h = options.epsilon_h
  options.epsilon_h = options.epsilon_pt2
  save_epsilon_c = options.epsilon_c
  options.epsilon_c = options.epsilon_pt2_c < 0.0 ? options.epsilon_pt2/2 : options.epsilon_pt2_c
  for state_idx in 1:nstates
    if options.verbose
      println("\nState $state_idx:")
    end
    # sort coefficients for current state
    if options.sort4pt2
      sort_indices = sortperm(@view(coefficients[:, state_idx]), rev=true, by=abs)
    else
      sort_indices = nothing
    end
    sort_indices = sortperm(@view(coefficients[:, state_idx]), rev=true, by=abs)
    _, ΔE[state_idx] = heatbath_selection(selected_ctx, @view(coefficients[:, state_idx]), options,
                               E_variational[state_idx], setup_data, sort_indices, false)
    if options.verbose
      println("  PT2 correction: $(ΔE[state_idx][1]) ± $(ΔE[state_idx][2]) Ha")
      println("  Total energy (VAR+PT2): $(E_variational[state_idx] + ΔE[state_idx][1]) ± $(ΔE[state_idx][2]) Ha")
      println("="^70)
    end
  end
  options.epsilon_h = save_epsilon_h
  options.epsilon_c = save_epsilon_c
  return ΔE  
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