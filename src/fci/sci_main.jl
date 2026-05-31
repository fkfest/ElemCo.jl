
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
    setup_selected_ci_from_determinants!(context::Union{FCIContext, CIPHIContext}, determinants::Vector{Determinant}, hamiltonian=SelectedHamiltonianMatrix()) -> SelectedCIContext

Create SelectedCIContext from list of determinants.
"""
function setup_selected_ci_from_determinants!(context::Union{FCIContext{O, T}, CIPHIContext{O, T}}, determinants, 
                                              hamiltonian::Union{Nothing,SelectedHamiltonianMatrix}=nothing) where {O, T}
  if isnothing(hamiltonian) 
    hamiltonian = SelectedHamiltonianMatrix{T}(is_hermitian(context))
  end
  return SelectedCIContext(context, determinants, hamiltonian)
end

"""
    setup_selected_ci_from_addresses!(context::FCIContext, addresses::Vector{Address}) -> SelectedCIContext

Create SelectedCIContext from list of addresses.
"""
function setup_selected_ci_from_addresses!(context::FCIContext{O, T}, addresses::Vector{Address}) where {O, T}
  determinants = [determinant_from_address(context, addr) for addr in addresses]
  return SelectedCIContext(context, determinants, SelectedHamiltonianMatrix{T}(is_hermitian(context)))
end

"""
    project_selected_to_full!(v_full::FCIVector, v_selected::AbstractVector, 
                             selected_ctx::SelectedCIContext)

Project selected CI vector onto full CI space.
"""
function project_selected_to_full!(v_full::FCIVector, v_selected::AbstractVector{T},
                                   selected_ctx::SelectedCIContext) where T
  fill!(v_full.data, zero(T))

  for i in 1:n_selected(selected_ctx)
    addr = selected_ctx.selected_dets.addresses[i]
    v_full.data[addr] = v_selected[i]
  end

  normalize!(v_full)
end

"""
    extract_full_to_selected!(v_selected::AbstractVector, v_full::FCIVector, 
                             selected_ctx::SelectedCIContext)

Extract selected CI coefficients from full CI vector.
"""
function extract_full_to_selected!(v_selected::AbstractVector, v_full::FCIVector, selected_ctx::SelectedCIContext)
  for i in 1:n_selected(selected_ctx)
    addr = selected_ctx.selected_dets.addresses[i]
    v_selected[i] = v_full.data[addr]
  end
end

function make_selected_1rdms!(rdm_a::AbstractMatrix{T}, rdm_b::AbstractMatrix{T},
                              determinants::AbstractVector{Determinant{OPattern}},
                              coefficients::AbstractVector, n_orb::Integer) where {T, OPattern}
  @assert length(determinants) == length(coefficients)
  @assert size(rdm_a) == (n_orb, n_orb)
  @assert size(rdm_b) == (n_orb, n_orb)

  fill!(rdm_a, zero(T))
  fill!(rdm_b, zero(T))

  det_index = Dict{Determinant{OPattern}, Int}()
  for i in eachindex(determinants)
    det_index[determinants[i]] = i
  end

  occa = Int[]
  virta = Int[]
  occb = Int[]
  virtb = Int[]

  for i in eachindex(determinants)
    det = determinants[i]
    coeff_i = coefficients[i]

    occupied_and_virtual_orbitals!(occa, virta, det.alpha, n_orb)
    occupied_and_virtual_orbitals!(occb, virtb, det.beta, n_orb)

    for orb in occa
      rdm_a[orb, orb] += conj(coeff_i) * coeff_i
    end
    for orb in occb
      rdm_b[orb, orb] += conj(coeff_i) * coeff_i
    end

    for orb_i in occa, orb_a in virta
      det_j = single_excitation_alpha(det, orb_i, orb_a)
      j = get(det_index, det_j, 0)
      if j != 0
        phase = calculate_excitation_phase(det.alpha, orb_i, orb_a)
        rdm_a[orb_a, orb_i] += phase * conj(coefficients[j]) * coeff_i
      end
    end

    for orb_i in occb, orb_a in virtb
      det_j = single_excitation_beta(det, orb_i, orb_a)
      j = get(det_index, det_j, 0)
      if j != 0
        phase = calculate_excitation_phase(det.beta, orb_i, orb_a)
        rdm_b[orb_a, orb_i] += phase * conj(coefficients[j]) * coeff_i
      end
    end
  end

  return rdm_a, rdm_b
end



# ===========================================
# Main CIPHI Iteration Loop
# ===========================================

"""
    run_ciphi!(ctx::Union{FCIContext, CIPHIContext}, options::CIPHIOptions; 
                     initial_dets=nothing, initial_coeffs=nothing) 
      -> (Vector{Float64}, Matrix{Float64}, Vector{Determinant}, Vector{(Float64, Float64)})

Run CIPHI (CIΦ - Selected CI via Perturbation, Heat-Bath and Iterations) calculation with support for multiple states.

# Arguments
- `ctx`: FCI context
- `options`: CIPHI options (including nstates for multi-state)

# Keyword Arguments
- `initial_dets::Union{Nothing, Vector{<:AbstractDeterminant}}`: Starting determinants from previous calculation
- `initial_coeffs::Union{Nothing, AbstractVecOrMat{Float64}}`: Starting CI coefficients (optional, for warm start)

# Returns
- `energies`: Vector of length nstates with total energies (electronic + nuclear)
- `coefficients`: Matrix (n_dets × nstates) with CI coefficients for all states
- `variational_dets`: Vector of determinants in final variational space
- `pt2_result`: PT2 correction results for all states

# Notes
- For nstates=1 (default), uses single-state selection strategy
- For nstates>1, uses multi-state selection with state-maximum probability
- If `initial_dets` is provided, these determinants are used as the starting variational space
"""
function run_ciphi!(ctx::Union{FCIContext{OPattern, T}, CIPHIContext{OPattern, T}}, options::CIPHIOptions;
                          initial_dets::Union{Nothing, Vector{<:AbstractDeterminant}}=nothing,
                          initial_coeffs::Union{Nothing, AbstractVecOrMat}=nothing) where {OPattern, T}
  if options.verbose
    println("\n" * "="^70)
    println("CIPHI - Selected CI via Perturbation, Heat-Bath and Iterations")
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
  
  setup_data = setup_ciphi!(ctx)
  
  if options.verbose
    n_pairs = length(setup_data.double_excitations_aa)
    total_triplets = sum(length(v) for v in values(setup_data.double_excitations_aa))
    println("  Stored $(n_pairs) (p,q) pairs with $(total_triplets) total (r,s) triplets")
    println("  Maximum |H_doub|: $(setup_data.h_doub_max)")
  end
  
  # Enhanced initial guess using small-space Hamiltonian (if enabled)
  variational_dets = Determinant{OPattern}[]
  E_init_vec = Float64[]

  if !isnothing(initial_dets) && length(initial_dets) > 0
    # Start from provided determinants (restart from previous calculation)
    if options.verbose
      println("\nInitialization (Restart from stored determinants)")
      println("  Loading $(length(initial_dets)) determinants from previous calculation")
    end
    
    # Convert to native Determinant type if needed
    for det in initial_dets
      push!(variational_dets, Determinant{OPattern}(det.alpha, det.beta))
    end
    
    # Get initial energies from diagonalization
    selected_ctx = SelectedCIContext(ctx, variational_dets, SelectedHamiltonianMatrix{T}(is_hermitian(ctx)))
    
    # Use provided coefficients as warm start if available
    prev_coeffs = nothing
    if !isnothing(initial_coeffs) && length(initial_coeffs) > 0
      # Convert to matrix if vector
      prev_coeffs = initial_coeffs isa AbstractMatrix ? initial_coeffs : reshape(initial_coeffs, :, 1)
      if options.verbose
        println("  Using stored CI coefficients as warm start")
      end
    end
    
    E_electronic_vec, coeffs_init = diagonalize_selected_space(selected_ctx, 
                                                               nstates=options.nstates,
                                                               previous_vectors=prev_coeffs)
    E_init_vec = E_electronic_vec .+ ctx.fcidump.int0
    
    if options.verbose
      println("  Initial energies from restart:")
      for (i, E) in enumerate(E_init_vec)
        println("    State $i: $E Hartree")
      end
    end
  elseif options.use_small_space_guess && options.nstates > 1
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
    selected_ctx = SelectedCIContext(ctx, variational_dets, SelectedHamiltonianMatrix{T}(is_hermitian(ctx)))
    
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
    selected_ctx = SelectedCIContext(ctx, variational_dets, SelectedHamiltonianMatrix{T}(is_hermitian(ctx)))
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
  # Initialize previous_eigenvectors with stored coefficients if available (for warm start)
  previous_eigenvectors = nothing
  if !isnothing(initial_coeffs) && length(initial_coeffs) > 0
    prev_coeffs = initial_coeffs isa AbstractMatrix ? initial_coeffs : reshape(initial_coeffs, :, 1)
    previous_eigenvectors = prev_coeffs
  end
  
  # Skip iterative selection if pt2_only mode (just compute PT2 on loaded determinants)
  if options.pt2_only
    if isnothing(initial_dets) || length(initial_dets) == 0
      error("pt2_only mode requires starting determinants (use with restart)")
    end
    if options.verbose
      println("\nPT2-only mode: Skipping variational CIPHI iterations")
    end
    # Use stored coefficients as previous eigenvectors for final diagonalization
    # Go directly to final diagonalization and PT2
    @goto final_diagonalization
  end
  
  if options.verbose
    println("\nIterative perturbative selection")
  end
  nsteps = max(options.nsteps, 1)

  E_prev_vec = zero(E_init_vec)
  converged = false
  res_tol = options.res_tol * 100  # Looser tolerance for main loop diagonalization
  for iter in 1:options.max_iter
    if options.verbose
      println("\nCIPHI Iteration $iter:")
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
      if nsteps <= 1
        converged = true
        break
      else
        nsteps -= 1
        set_n_old_dets!(selected_ctx, 0)
        if options.verbose
          println("  Refresh perturbative selection: Remaining steps: $nsteps")
        end
      end
    end

    if n_selected(selected_ctx) >= options.target_selection
      if options.verbose
        println("  ✓ Reached target selection size: $(options.target_selection) determinants")
      end
      converged = true
      break
    end
    
    # 3. Generate candidates and compute probabilities
    new_dets_dict = Dict{Determinant{OPattern}, Float64}()  # To hold selected new determinants
    pt2_corrections = Array{Tuple{Float64, Float64}}(undef, options.nstates)
    for state = 1:options.nstates
      new_dets, (pt2_raw, negl_raw) = heatbath_selection(selected_ctx, @view(coeffs_matrix[:,state]), 
                                          options, E_electronic_vec[state], setup_data)
      pt2_corrections[state] = (real(pt2_raw), real(negl_raw))
      # Merge new determinants from all states, taking maximum weight (for target_selection)
      mergewith!(max, new_dets_dict, new_dets)
    end
    if options.verbose && is_hermitian(selected_ctx)
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
      if nsteps <= 1
        if options.verbose
          println("  No new determinants selected. Converged.")
        end
        converged = true
        break
      else
        nsteps -= 1
        set_n_old_dets!(selected_ctx, 0)
        if options.verbose
          println("  Refresh perturbative selection: Remaining steps: $nsteps")
        end
      end
    end
    
    # 5. Update variational space
    extend!(selected_ctx, new_dets)
    
    E_prev_vec = copy(E_current_vec)
    # Save eigenvectors for next iteration as warm start
    previous_eigenvectors = coeffs_matrix[:, 1:options.nstates]
  end
  
  if !converged
    @warn "CIPHI did not converge in $(options.max_iter) iterations"
  end
  
  @label final_diagonalization
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
    println("CIPHI Complete!")
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
    run_ciphi!(ctx::CIPHIContext; initial_dets=nothing, initial_coeffs=nothing) 
      -> (energies, coeffs, determinants, pt2_result)

Convenience method for CIPHIContext that uses bundled options.

This lightweight interface avoids FCIContext initialization overhead:
- No full-space address tables computed upfront
- No full-space diagonal H computed upfront  
- Initialization time proportional to N_orbitals, not N_determinants

# Keyword Arguments
- `initial_dets`: Starting determinants from previous calculation (for restart)
- `initial_coeffs`: Starting CI coefficients (optional, for warm start)

# Returns
Same as run_ciphi!(ctx, options): (energies, coeffs, determinants, pt2_result)
"""
function run_ciphi!(ctx::CIPHIContext{OPattern}; 
                          initial_dets::Union{Nothing, Vector{<:AbstractDeterminant}}=nothing,
                          initial_coeffs::Union{Nothing, AbstractVecOrMat}=nothing) where OPattern
  return run_ciphi!(ctx, ctx.options; initial_dets=initial_dets, initial_coeffs=initial_coeffs)
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
                                 coefficients::AbstractMatrix, E_variational::Vector{Float64},
                                 setup_data::CIPHISetupData, options::CIPHIOptions)
  
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
  nstates = options.nstates
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
    if is_hermitian(selected_ctx)
      # For Hermitian case, only need right coefficients
      _, (pt2_raw, negl_raw) = heatbath_selection(selected_ctx, @view(coefficients[:, state_idx]),
                                            options, E_variational[state_idx], setup_data,
                                            nothing, sort_indices, false; pt2_correct=true)
      ΔE[state_idx] = (real(pt2_raw), real(negl_raw))
    else
      left_idx = state_idx + nstates
      _, (pt2_raw, negl_raw) = heatbath_selection(selected_ctx, @view(coefficients[:, state_idx]),
                                            options, E_variational[state_idx], setup_data,
                                            @view(coefficients[:, left_idx]), sort_indices, false; 
                                            pt2_correct=true)
      ΔE[state_idx] = (real(pt2_raw), real(negl_raw))
    end
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
                               previous_vectors::Union{Nothing,AbstractMatrix}=nothing,
                               conv_tol::Float64=1e-6) 
      -> (Vector{Float64}, Matrix{T})

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
- For non-Hermitian Hamiltonians (similarity-transformed integrals), both right and left
  eigenvectors are computed and returned in the same matrix (right vectors first, then left).
"""
function diagonalize_selected_space(selected_ctx::SelectedCIContext{OPattern, T}; 
                                   nstates::Int=1,
                                   previous_vectors::Union{Nothing,AbstractMatrix}=nothing,
                                   conv_tol::Float64=1e-6) where {OPattern, T}
  n_dets = n_selected(selected_ctx)
  dets = determinants(selected_ctx)
  nstates = min(nstates, n_dets)  # Can't compute more roots than determinants
  
  # For small spaces, use direct diagonalization (faster startup)
  if n_dets < 1000
    H_matrix = hamiltonian_matrix(selected_ctx)
    
    nval = min(nstates+5, n_dets)
    if is_hermitian(selected_ctx)
      # Use Hermitian eigenvalue solver
      eigenvalues, eigenvectors = eigen(Hermitian(H_matrix), 1:nval)
      eigenvectors = eigenvectors[:, 1:nstates]
    else
      eigenvalues, eigenvectors = eigen(H_matrix)
      left_eigenvectors = inv(eigenvectors')
      eigenvectors = hcat(eigenvectors[:, 1:nstates], left_eigenvectors[:, 1:nstates])
      if !(T <:Complex)
        eigenvectors = real.(eigenvectors)
      end
    end
    return real.(eigenvalues[1:nstates]), eigenvectors
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
    initial_guesses = zeros(T, n_dets, n_use_prev)
    
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
    return real.(eigenvalues), eigenvectors
  end
  # No previous vectors: use determinant with lowest diagonal element
  diagonal = [real(compute_diagonal_element(det, selected_ctx.base_context)) 
              for det in dets]
  min_idx = argmin(diagonal)
  initial_guess = zeros(T, n_dets, 1)
  initial_guess[min_idx, 1] = one(T)
  
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
  return real.(eigenvalues), eigenvectors
  
end