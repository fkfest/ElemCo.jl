# fci_davidson.jl - Davidson diagonalization algorithms

"""
Davidson Diagonalization Module

This module contains the Davidson iterative diagonalization algorithm for FCI calculations,
including the Jacobi-Davidson preconditioner with P-Q space coupling.

Main entry point: `davidson_fci!`
"""

"""
    inject_pspace_to_full!(c::FCIVector, c_pspace::Vector{Scalar}, 
                          pspace_data::PSpaceData)

Map P-space correction vector back to full CI space.
Sets coefficients at P-space determinant positions.
"""
function inject_pspace_to_full!(c::FCIVector, c_pspace::AbstractVector, pspace_data::PSpaceData)
  for i in 1:(pspace_data.n_pspace)
    addr = pspace_data.indices[i]
    c.data[addr] = c_pspace[i]
  end
  return nothing
end

"""
    solve_projected_pspace_system(r_P::Vector{Scalar}, u_P::Vector{Scalar}, 
                                  theta::Scalar, pspace_data::PSpaceData, 
                                  shift::Scalar, alpha::Scalar)

Solve the projected P-space correction equation with Q-space coupling:
    [(I - u_P*u_P^T)(H_PP - θI)(I - u_P*u_P^T) + α*u_P*u_P^T] t_P = r_P

This is part of the Jacobi-Davidson P-Q coupled preconditioner algorithm.
The α term accounts for energy shift from Q-space component of Ritz vector.

Arguments:
- r_P: Residual vector in P-space
- u_P: Current Ritz vector in P-space (should be normalized)
- theta: Current Ritz value (approximate eigenvalue)
- pspace_data: P-space data structure with Hamiltonian
- shift: Level shift for regularization
- alpha: Q-space energy shift term (α = u_Q^T(H_QQ - E)u_Q)

Returns: Correction vector t_P (orthogonal to u_P)
"""
function solve_projected_pspace_system(r_P::AbstractVector, u_P::AbstractVector, 
                                       theta, pspace_data::PSpaceData, shift, alpha)
  n_p = pspace_data.n_pspace
  H_P = pspace_data.hamiltonian
  
  # Build operator: A = H_P - (θ - shift)*I
  A = copy(H_P)
  for i in 1:n_p
    A[i, i] -= (theta - shift)
  end
  
  # Apply left projection: A_left = (I - u_P*u_P^T) * A
  A -= u_P * (u_P' * A) 
  # Apply right projection: A_proj = A_left * (I - u_P*u_P^T)
  A -= (A * u_P) * u_P' 
  
  # Add α u_P*u_P^T
  A += alpha * u_P * u_P'

  # Solve: A_proj * t_P = r_P
  t_P = A \ r_P
  
  return t_P
end

"""
    jacobi_davidson_correction!(t::FCIVector, r::FCIVector, u::FCIVector,
                               theta::Scalar, context::FCIContext, shift::Scalar)

Compute Jacobi-Davidson correction vector using P-space enhanced preconditioning:
    (I - u*u^T) * (H - θ*I) * (I - u*u^T) * t = -r

This solves the projected correction equation to ensure the result is orthogonal
to the current approximate eigenvector u, preventing linear dependency in the
Davidson subspace.

Algorithm:
1. Project residual to ensure orthogonality: r_proj = (I - u*u^T) * r
2. If P-space available:
   - Extract P-space components of r_proj and u
   - Solve projected P-space system with double projection
   - Inject P-space correction into full space
   - Apply projected Davidson-Jacobi to Q-space
3. Otherwise:
   - Apply projected Davidson-Jacobi to all determinants
4. Final projection to enforce orthogonality

Arguments:
- t: Output correction vector (modified in-place)
- r: Residual vector
- u: Current Ritz vector (approximate eigenvector, should be normalized)
- theta: Current Ritz value (approximate eigenvalue)
- context: FCI context with Hamiltonian and P-space data
- shift: Level shift for regularization

Result: t is orthogonal to u by construction (dot(t, u) ≈ 0)
"""
function jacobi_davidson_correction!(t::FCIVector, r::FCIVector, u::FCIVector,
                                     theta, context::FCIContext{O, T}, shift) where {O, T}
  # Corrected Jacobi-Davidson with proper P-Q space coupling
  debug = context.options.print_level >= 2
  
  if debug
    println("\n=== Jacobi-Davidson Correction (P-Q Coupled) ===")
    println("Residual norm: ", norm(r))
    println("Ritz vector norm: ", norm(u))
    println("Theta: $theta")
    println("Shift: $shift")
  end
  
  # Step 1: Project residual to ensure orthogonality (eliminates u_P(u_Q^T r_Q) term)
  r_perp = copy(r)
  orthogonalize_against!(r_perp, u)  # r_perp = (I - uu^T)r
  
  # Step 2: Initial Q-space estimate with diagonal preconditioner
  diag_h = context.diag_h
  t_Q = zero(r_perp)
  
  ThrNeglect = context.options.thr_negligible
  for i in 1:n_data(r_perp)
    denominator = theta - diag_h.data[i] - shift
    if abs(denominator) > ThrNeglect
      t_Q.data[i] = r_perp.data[i] / denominator
    else
      t_Q.data[i] = 0.0
    end
  end
  
  # Zero out P-space part of t_Q (keep it pure Q-space)
  use_pspace = context.pspace_data.n_pspace > 0
  
  if use_pspace
    pspace_data = context.pspace_data
    
    for i in 1:pspace_data.n_pspace
      addr = pspace_data.indices[i]
      t_Q.data[addr] = 0.0
    end
    
    # Step 3: Extract u_Q (Q-space component of Ritz vector)
    u_Q = copy(u)
    for i in 1:pspace_data.n_pspace
      addr = pspace_data.indices[i]
      u_Q.data[addr] = 0.0
    end
    
    # Step 4: Compute α = u_Q^T (H_QQ - E) u_Q using diagonal
    # α = Σ_{i∈Q} u_Q[i]² (diag_H[i] - θ)
    alpha = zero(T)
    for i in 1:length(u_Q.data)
      if abs(u_Q.data[i]) > 1e-15  # Only Q-space indices (where u_Q is non-zero)
        alpha += u_Q.data[i]^2 * (diag_h.data[i] - theta + shift)
      end
    end
    
    # Step 5: Compute β = u_Q^T (H_QQ - E) (I - u_Q u_Q^T) t_Q using diagonal
    # First compute (I - u_Q u_Q^T) t_Q
    t_Q_perp = copy(t_Q)
    orthogonalize_against!(t_Q_perp, u_Q)
    
    # β = Σ_{i∈Q} u_Q[i] (diag_H[i] - θ) t_Q_perp[i]
    beta = zero(T)
    for i in 1:length(u_Q.data)
      if abs(u_Q.data[i]) > 1e-15
        beta += u_Q.data[i] * (diag_h.data[i] - theta + shift) * t_Q_perp.data[i]
      end
    end
    
    # Step 6: Compute overlap u_Q^T t_Q
    overlap_uQ_tQ = dot(u_Q, t_Q)
    
    if debug
      println("\nP-Q coupling parameters:")
      println("  α (Q-space shift): ", alpha)
      println("  β (Q-space coupling): ", beta)
      println("  u_Q^T t_Q: ", overlap_uQ_tQ)
    end
    
    # Step 7: Extract P-space components
    r_P = project_to_pspace(r_perp, pspace_data)
    u_P = project_to_pspace(u, pspace_data)
    
    # Step 8: Build enhanced RHS for P-space
    # RHS = -r_P + (I - u_P u_P^T)((H_PP - E)u_P)(u_Q^T t_Q) + β u_P
    H_PP = pspace_data.hamiltonian
    H_PP_uP = H_PP * u_P - (theta - shift) * u_P  # (H_PP - E) u_P
    
    # Project it: (I - u_P u_P^T) (H_PP_uP)
    orthogonalize_against!(H_PP_uP, u_P)
    
    # Build complete RHS
    RHS_P = -r_P
    RHS_P += H_PP_uP * overlap_uQ_tQ  # Q-space correction coupling
    RHS_P += beta * u_P  # Q-space Hamiltonian coupling
    
    if debug
      println("\nP-space system:")
      println("  r_P norm: ", norm(r_P))
      println("  u_P norm: ", norm(u_P))
      println("  RHS_P norm: ", norm(RHS_P))
    end
    
    # Step 9: Solve modified P-space system with α correction
    # [(I - u_P u_P^T)(H_PP - E)(I - u_P u_P^T) + α u_P u_P^T] t_P = RHS_P
    t_pspace = solve_projected_pspace_system(RHS_P, u_P, theta, pspace_data, shift, alpha)
    
    if debug
      println("  t_P norm: ", norm(t_pspace))
      println("  dot(t_P, u_P): ", abs(LinearAlgebra.dot(t_pspace, u_P)))
    end
    
    # Step 10: Combine P and Q solutions
    # Start with Q-space diagonal solution
    copy!(t.data, t_Q.data)
    
    # Inject refined P-space solution (overwrites P-space part)
    inject_pspace_to_full!(t, t_pspace, pspace_data)
  else
    # No P-space: just use diagonal preconditioner
    copy!(t.data, t_Q.data)
  end
  
  # Step 11: Final full-space projection to ensure orthogonality
  if debug
    println("\nBefore final projection:")
    println("  t norm: ", norm(t))
    println("  dot(t, u): ", abs(dot(t, u)))
  end
  
  orthogonalize_against!(t, u)
  
  if debug
    println("After final projection:")
    println("  t norm: ", norm(t))
    println("  dot(t, u): ", abs(dot(t, u)))
    println("=" ^ 50)
  end
  
  return nothing
end

"""
    update_ci_vector!(c::FCIVector, r::FCIVector, diag_h::FCIVector,
                     energy::Scalar, context::FCIContext, shift::Scalar,
                     u::Union{FCIVector,Nothing}=nothing)

Update CI vector using Davidson preconditioner with optional Jacobi-Davidson projection.

Two modes available:
1. Jacobi-Davidson (jacobi_davidson=true, default): Uses projection to prevent linear dependency
   Requires current Ritz vector u to be provided
2. Simple preconditioner (jacobi_davidson=false)

When P-space is available: uses exact solution in P-space + diagonal preconditioner in Q-space.
When P-space is not available: falls back to standard diagonal preconditioner.

Arguments:
- c: Output correction vector
- r: Residual vector
- diag_h: Diagonal Hamiltonian elements
- energy: Current energy estimate
- context: FCI context
- shift: Level shift for regularization
- u: Current Ritz vector (normalized approximate eigenvector). Required for Jacobi-Davidson mode.
"""
function update_ci_vector!(c::FCIVector, r::FCIVector, diag_h::FCIVector,
                           energy, context::FCIContext, shift,
                           u::Union{FCIVector, Nothing} = nothing)
  # Decide which method to use
  jacobi_davidson = context.options.jacobi_davidson && u !== nothing

  if jacobi_davidson
    # Use Jacobi-Davidson correction with projection (prevents linear dependency)
    jacobi_davidson_correction!(c, r, u, energy, context, shift)
  else
    # Fall back to standard diagonal preconditioner
    # Note: Simple P-space preconditioner (without projection) is not recommended
    # as it can cause linear dependency issues. Use Jacobi-Davidson instead.
    
    # Standard Davidson-Jacobi preconditioning: c = r / (E - H_ii)
    ThrNeglect = context.options.thr_negligible
    for i in 1:n_data(r)
      denominator = energy - diag_h.data[i] - shift
      if abs(denominator) > ThrNeglect
        c.data[i] = r.data[i] / denominator
      else
        c.data[i] = 0.0
      end
    end
  end
end

"""
    refresh_davidson_subspace!(V, HV, eigenvecs, k::Int, n_keep::Int)

Refresh Davidson subspace by keeping the most important eigenvectors.
This is called when the subspace becomes too large.

# Arguments
- `V`: Array of subspace vectors
- `HV`: Array of Hamiltonian-vector products  
- `eigenvecs`: Current eigenvectors
- `k`: Current subspace size
- `n_keep`: Number of vectors to keep after refresh

# Returns
- New subspace size after refresh
"""
function refresh_davidson_subspace!(V::Vector{<:FCIVector{OPattern}}, HV::Vector{<:FCIVector{OPattern}},
                                    eigenvecs::AbstractMatrix, k::Int, n_keep::Int) where OPattern
  # Transform the V and HV to the eigenvector basis
  # Keep the first n_keep eigenvectors (lowest energy ones)

  # Create temporary storage for transformed vectors
  V_new = [zero(V[1]) for _ in 1:n_keep]
  HV_new = [zero(V[1]) for _ in 1:n_keep]

  # Transform to eigenvector basis and keep only n_keep vectors
  for i in 1:n_keep
    clear!(V_new[i])
    clear!(HV_new[i])

    # V_new[i] = Σⱼ V[j] * eigenvecs[j,i]
    for j in 1:k
      add!(V_new[i], V[j], eigenvecs[j, i])
    end

    # HV_new[i] = Σⱼ HV[j] * eigenvecs[j,i] 
    for j in 1:k
      add!(HV_new[i], HV[j], eigenvecs[j, i])
    end
  end

  # Gram-Schmidt orthogonalization of V_new, applying same transformations to HV_new.
  # This maintains HV_new[i] = H * V_new[i] while ensuring V_new is orthonormal.
  # For Hermitian case, eigenvecs are unitary so this is essentially a no-op.
  # For non-Hermitian case (ST), eigenvecs are not orthogonal and this step is essential.
  for i in 1:n_keep
    for j in 1:(i - 1)
      overlap = dot(V_new[j], V_new[i])
      add!(V_new[i], V_new[j], -overlap)
      add!(HV_new[i], HV_new[j], -overlap)
    end
    nrm = norm(V_new[i])
    V_new[i].data .*= inv(nrm)
    HV_new[i].data .*= inv(nrm)
  end

  # Copy transformed vectors back
  for i in 1:n_keep
    copy!(V[i].data, V_new[i].data)
    copy!(HV[i].data, HV_new[i].data)
  end

  println("  🔄 Subspace refreshed: $k vectors → $n_keep vectors")
  return n_keep
end

"""
    davidson_fci!(context::FCIContext, n_states::Union{Int, Nothing} = nothing) 
    -> Tuple{Vector{Scalar}, Vector{FCIVector}}

Unified Davidson diagonalization for FCI with subspace refresh capability.
Always returns arrays of energies and states for type stability.

# Arguments
- `context`: FCI context containing system information
- `n_states`: Number of electronic states to compute (default: uses context.options.nstates)

# Returns
- `Tuple{Vector{Scalar}, Vector{FCIVector}}`: Arrays of energies and corresponding eigenvectors
"""
function davidson_fci!(context::FCIContext{OPattern, T}, n_states::Union{Int, Nothing} = nothing) where {OPattern, T}
  # Use nstates from options if n_states not provided
  n_states = isnothing(n_states) ? context.options.nstates : n_states
  hermitian = is_hermitian(context)

  # Scale max_iter for multi-state calculations (excited states need more iterations)
  max_iter = context.options.max_iter
  if n_states > 1
    # Increase max_iter for multi-state: 50→75 for n_states=2-3, 50→100 for n_states=4-5, etc.
    max_iter = max_iter + (n_states - 1) * 25
    println("Multi-state calculation: max_iter scaled to $max_iter (from $(context.options.max_iter))")
  end
  
  conv_tol = context.options.conv_tol
  subspace_size = context.options.subspace_size * n_states  # Scale subspace size with number of states

  # Validate input
  if n_states < 1
    error("Number of states must be at least 1")
  end

  # Adaptive parameters - treat single-state as special case of multi-state
  n_keep = max(n_states + 1, subspace_size ÷ 3)
  max_refreshes = 15

  refresh_count = 0
  n_total = n_data(context.coeff)

  # Diagonal Hamiltonian is computed during init_hamiltonian_terms!
  # Verify it has been computed
  if all(x -> x == 0, context.diag_h.data)
    error("Diagonal Hamiltonian not initialized. Call init_hamiltonian_terms! first.")
  end

  # Setup P-space for enhanced initial guess
  # Pass n_states so HBCI can compute the same number of roots
  setup_pspace!(context, n_states)

  # Print unified header
  mode_name = n_states == 1 ? "Single-State" : "Multi-State"
  println("Starting $mode_name Davidson FCI diagonalization")
  println("System: $(context.n_orb) orbitals, $(context.n_elec) electrons")
  println("Determinant space: $(n_total) determinants")
  println("Computing $n_states electronic state$(n_states > 1 ? "s" : "")")
  println(
    "Subspace settings: max_size=$subspace_size, refresh_keep=$n_keep, max_refreshes=$max_refreshes",
  )

  # Davidson subspace vectors
  n_spin = context.n_elec[1] - context.n_elec[2]
  n_elec = context.n_elec[1] + context.n_elec[2]
  V = [
    FCIVector{OPattern, T}(n_elec, context.n_orb, n_spin) for
    _ in 1:subspace_size
  ]
  HV = [
    FCIVector{OPattern, T}(n_elec, context.n_orb, n_spin) for
    _ in 1:subspace_size
  ]

  # Initialize guess vectors - try P-space enhanced, supplemented with diagonal guesses
  # For multi-state, we need MORE initial vectors to span excited states properly
  # Use n_states * 2 initial vectors to give Davidson flexibility
  k_initial = n_states > 1 ? min(n_states * 2, subspace_size) : min(n_states + 1, subspace_size)
  k = k_initial
  guess_vectors = [V[i] for i in 1:k]

  # Try P-space enhanced initial guess
  pspace_success = generate_pspace_initial_guess!(context, guess_vectors, min(k, n_states))

  # Always supplement with diagonal-based guesses, especially for excited states
  # This ensures we have initial vectors spanning determinants outside P-space
  diag_sorted = sort([(real(context.diag_h.data[i]), i) for i in 1:length(context.diag_h.data)])
  
  if pspace_success && n_states > 1
    # P-space guesses provided for first n_states vectors
    # Add diagonal guesses for remaining vectors
    println("P-space enhanced initial guess for $n_states states")
    println("Supplementing with $(k - n_states) additional diagonal-based vectors")
    
    for i in (n_states + 1):k
      clear!(V[i])
      V[i].data[diag_sorted[i][2]] = 1.0
      normalize!(V[i])
      
      # Orthogonalize against previous vectors to avoid linear dependency
      for j in 1:(i-1)
        overlap = dot(V[j], V[i])
        add!(V[i], V[j], -overlap)
      end
      normalize!(V[i])
      
      println("Supplemental guess $i: H_diag = $(diag_sorted[i][1] + context.fcidump.int0) Hartree")
    end
  elseif !pspace_success
    # Fall back entirely to diagonal-based initial guess
    println("Using diagonal-based initial guess for all $k vectors")
    
    for i in 1:k
      clear!(V[i])
      V[i].data[diag_sorted[i][2]] = 1.0
      normalize!(V[i])

      if n_states == 1
        println("Initial guess energy: $(diag_sorted[i][1] + context.fcidump.int0) Hartree")
      else
        println("Initial guess $i: H_diag = $(diag_sorted[i][1] + context.fcidump.int0) Hartree")
      end
    end
  end

  # Compute HV for all initial vectors
  for i in 1:k
    clear!(HV[i])
    contract_hamiltonian!(context, HV[i], V[i], 1.0)
  end

  # Unified convergence tracking (always use arrays, even for single state)
  converged_states = fill(false, n_states)
  energies = zeros(T, n_states)
  old_energies = fill(T(Inf), n_states)

  iter = 0
  for iter in 1:max_iter
    # Build subspace Hamiltonian matrix
    T_mat = zeros(T, k, k)
    @inbounds for i in 1:k, j in 1:k
      T_mat[i, j] = dot(V[i], HV[j])
    end

    # Diagonalize subspace
    eigenvals, eigenvecs = _eigen_subspace(T_mat, hermitian)

    # Unified convergence check (always use arrays, even for single state)
    energies[1:n_states] .= eigenvals[1:n_states]

    # Compute residual norms for all states to check TRUE convergence
    residual_norms = zeros(Float64, n_states)
    for istate in 1:n_states
      # Form current eigenvector for this state
      clear!(context.coeff)
      for j in 1:k
        add!(context.coeff, V[j], eigenvecs[j, istate])
      end
      
      # Form residual: r = H|ψ⟩ - E|ψ⟩
      clear!(context.resid)
      for j in 1:k
        add!(context.resid, HV[j], eigenvecs[j, istate])
      end
      add!(context.resid, context.coeff, -energies[istate])
      
      residual_norms[istate] = norm(context.resid)
    end

    all_converged = true
    print("Iteration $iter:")
    if iter > 1
      for istate in 1:n_states
        if n_states > 1
          println()
          print("  State $istate: ")
        end
        energy_change = abs(energies[istate] - old_energies[istate])
        res_norm = residual_norms[istate]
        
        # DUAL convergence criteria: BOTH energy AND residual must converge
        energy_converged = energy_change < conv_tol
        residual_converged = res_norm < context.options.res_tol
        converged_states[istate] = energy_converged && residual_converged
        
        status = converged_states[istate] ? "✓" : " "
        print("E = $(energies[istate] + context.fcidump.int0) Hartree, ΔE = $energy_change")
        if !energy_converged
          print(" (E)")  # Energy not converged
        end
        if !residual_converged
          print(" (R=$(Printf.@sprintf("%.2e", res_norm)))")  # Residual not converged, show value
        end
        print(" $status")
        
        if !converged_states[istate]
          all_converged = false
        end
      end
    else
      for istate in 1:n_states
        if n_states > 1
          println()
          print("  State $istate: ")
        end
        res_norm = residual_norms[istate]
        print("E = $(energies[istate] + context.fcidump.int0) Hartree")
        print(" (R=$(Printf.@sprintf("%.2e", res_norm)))")
      end
      all_converged = false
    end
    println()

    old_energies .= energies

    if all_converged
      break
    end

    # Expand subspace if not converged
    if !all_converged
      # Check if we need to refresh the subspace
      if k >= subspace_size
        if refresh_count < max_refreshes
          # Perform subspace refresh (unified approach)
          k = refresh_davidson_subspace!(V, HV, eigenvecs, k, n_keep)
          refresh_count += 1
          println("  Refresh $refresh_count/$max_refreshes completed, continuing with $k vectors")
        else
          println("  Maximum refreshes ($max_refreshes) reached, stopping iteration")
          break
        end
      end

      # Unified subspace expansion: add residual vectors for ALL unconverged states
      # This improves multi-state convergence by expanding the subspace more aggressively
      vectors_added = 0
      k_start = k  # Save initial k before adding vectors
      
      # Store current eigenvector coefficients and Ritz vectors for each state BEFORE expansion
      # This prevents bounds errors when k changes during the loop
      state_data = Tuple{Int, FCIVector{OPattern, T}, FCIVector{OPattern, T}, Float64, Float64}[]  # (istate, coeff, resid, energy, res_norm)
     
      n_spin = context.n_elec[1] - context.n_elec[2]
      n_elec = context.n_elec[1] + context.n_elec[2]
      for istate in 1:n_states
        if !converged_states[istate]
          # Form current eigenvector for this state using the CURRENT subspace size k_start
          coeff_state = FCIVector{OPattern, T}(n_elec, context.n_orb, n_spin)
          clear!(coeff_state)
          for j in 1:k_start
            add!(coeff_state, V[j], eigenvecs[j, istate])
          end

          # Form residual: r = H|ψ⟩ - E|ψ⟩
          resid_state = FCIVector{OPattern, T}(n_elec, context.n_orb, n_spin)
          clear!(resid_state)
          for j in 1:k_start
            add!(resid_state, HV[j], eigenvecs[j, istate])
          end
          add!(resid_state, coeff_state, -energies[istate])

          # Check residual norm
          residual_norm = norm(resid_state)
          if residual_norm < conv_tol
            converged_states[istate] = true
          else
            push!(state_data, (istate, coeff_state, resid_state, real(energies[istate]), residual_norm))
          end
        end
      end
      
      # Add residual vectors for unconverged states (up to subspace limit)
      for (istate, coeff_state, resid_state, energy_state, residual_norm) in state_data
        if k >= subspace_size
          break  # Subspace full, will refresh on next iteration
        end

        # Create new vector using Davidson preconditioner
        new_idx = k + 1
        clear!(V[new_idx])
        
        # Normalize current Ritz vector for Jacobi-Davidson correction
        u_normalized = copy(coeff_state)
        normalize!(u_normalized)
        
        update_ci_vector!(V[new_idx], resid_state, context.diag_h, energy_state,
                          context, context.options.shift, u_normalized)

        # Orthogonalize against all existing vectors (including newly added ones)
        for j in 1:k
          overlap = dot(V[j], V[new_idx])
          add!(V[new_idx], V[j], -overlap)
        end

        # Check if vector became zero after orthogonalization
        vec_norm = norm(V[new_idx])
        if vec_norm < context.options.thr_negligible
          if n_states == 1
            println("  Warning: Vector became zero after orthogonalization - stopping expansion")
          else
            println("  Warning: Vector for state $istate became zero after orthogonalization")
          end
          continue
        end

        normalize!(V[new_idx])

        # Apply Hamiltonian to new vector
        clear!(HV[new_idx])
        contract_hamiltonian!(context, HV[new_idx], V[new_idx], 1.0)

        k = new_idx
        vectors_added += 1

        if n_states == 1
          println("  Residual norm: $residual_norm")
        else
          println("  Added residual vector for state $istate (norm: $residual_norm)")
        end
      end

      if vectors_added == 0 && !all_converged
        println("  No new vectors added, stopping iteration")
        break
      end
    end
  end

  # Unified final results processing
  n_converged = sum(converged_states)
  if n_states == 1
    if converged_states[1]
      println("✅ Single-State Davidson converged in $iter iterations!")
    else
      println("❌ Single-State Davidson did not converge in $max_iter iterations")
    end
  else
    if n_converged == n_states
      println("✅ Multi-State Davidson converged for all $n_states states in $iter iterations!")
    else
      println(
        "❌ Multi-State Davidson converged for only $n_converged/$n_states states in $iter iterations",
      )
    end
  end

  # Build final subspace and extract eigenvectors (unified approach)
  T_mat = zeros(T, k, k)
  @inbounds for i in 1:k, j in 1:k
    T_mat[i, j] = dot(V[i], HV[j])
  end
  eigenvals, eigenvecs = _eigen_subspace(T_mat, hermitian)

  # Extract final states (unified approach)
  final_energies = eigenvals[1:n_states] .+ context.fcidump.int0
  final_states = Vector{FCIVector{OPattern, T}}(undef, n_states)

  n_spin = context.n_elec[1] - context.n_elec[2]
  n_elec = context.n_elec[1] + context.n_elec[2]
  for istate in 1:n_states
    final_states[istate] =
      FCIVector{OPattern, T}(n_elec, context.n_orb, n_spin)
    clear!(final_states[istate])

    for j in 1:k
      add!(final_states[istate], V[j], eigenvecs[j, istate])
    end
    normalize!(final_states[istate])

    if n_states > 1
      println("Final State $istate: E = $(final_energies[istate]) Hartree")
    end
  end

  # Set ground state in context for backward compatibility
  copy!(context.coeff.data, final_states[1].data)

  # Always return arrays for type stability
  return (final_energies, final_states)
end

"""
    davidson_selected_ci!(selected_ctx, initial_guesses; kwargs...)

Davidson diagonalization in selected CI space.

This is a specialized Davidson algorithm for selected CI calculations where
the Hamiltonian matrix elements are computed on-the-fly. It uses the same
Davidson algorithm as `davidson_fci!` but operates on a selected subset of
determinants rather than the full CI space.

Automatically detects whether the Hamiltonian is Hermitian or non-Hermitian
based on the context (similarity-transformed integrals are non-Hermitian).

# Arguments
- `selected_ctx::SelectedCIContext`: Selected CI context containing determinants
- `initial_guesses::Matrix{Scalar}`: Initial guess vector in selected CI basis (n_selected × n_guess)

# Keyword Arguments
- `nstates::Int=1`: Number of lowest eigenvalues to compute
- `max_iterations::Int=50`: Maximum number of Davidson iterations
- `convergence_threshold::Float64=1e-8`: Energy convergence threshold
- `shift::Float64=0.1`: (square of imaginary) level shift for preconditioner
- `max_subspace::Int=30`: Maximum subspace size before refresh
- `verbose::Bool=false`: Print iteration information

# Returns
- `Tuple{Vector{Float64}, Matrix{Float64}}`: Eigenvalues and eigenvectors
  - eigenvalues: Vector of lowest `nstates` eigenvalues
  - eigenvectors: Matrix of eigenvectors (n_selected × nstates(times 2 for non-Hermitian))

# Algorithm
Uses the Davidson iterative diagonalization method:
1. Build subspace by expanding with correction vectors
2. Project Hamiltonian onto subspace
3. Diagonalize small projected Hamiltonian (Hermitian or non-Hermitian)
4. Compute residuals and check convergence
5. Add correction vectors to expand subspace
6. Refresh subspace when it becomes too large

For non-Hermitian Hamiltonians (similarity-transformed integrals):
- Uses standard eigen() instead of eigen(Hermitian())
- Preconditioner uses complex shift to handle non-real eigenvalues
- The left eigenvectors are returned in the same array as right eigenvectors (after the right ones)

The key difference from `davidson_fci!` is that matrix elements are computed
on-the-fly using `contract_hamiltonian_selected!`, maintaining O(N_selected)
memory usage rather than O(N_full).
"""
function davidson_selected_ci!(selected_ctx::SelectedCIContext, initial_guesses::AbstractMatrix;
                               nstates::Int = 1,
                               max_iterations::Int = 50,
                               convergence_threshold::Float64 = 1e-8,
                               shift::Float64 = 0.1,
                               max_subspace::Int = 30,
                               verbose::Bool = false)
  n_dets = n_selected(selected_ctx)
  
  n_guess::Int = size(initial_guesses, 2)
  
  # Validate input
  nstates >= 1 || error("nstates must be at least 1")
  nstates <= n_dets || error("nstates cannot exceed n_selected")
  size(initial_guesses, 1) == n_dets || error("initial_guess must have ", n_dets, " rows")
  max_subspace >= 2 * nstates || error("max_subspace must be at least 2*nstates")

  # Detect if Hamiltonian is Hermitian
  hermitian = is_hermitian(selected_ctx)
  
  # Scale subspace size with number of roots
  max_subspace = max(max_subspace, 3 * nstates)
  n_keep = max(nstates + 2, max_subspace ÷ 3)
  
  if verbose
    println("\nStarting Davidson Selected CI diagonalization")
    println("Selected space: $n_dets determinants")
    println("Hamiltonian type: $(hermitian ? "Hermitian" : "non-Hermitian")")
    println("Computing $nstates eigenstate$(nstates > 1 ? "s" : "")")
    println("Initial guesses provided: $n_guess")
    println("Subspace settings: max=$max_subspace, keep=$n_keep")
  end

  ThrNeglect = selected_ctx.base_context.options.thr_negligible
  
  Ts = eltype(initial_guesses)
  # Precompute diagonal elements for preconditioner
  diagonal = zeros(Ts, n_dets)
  for i in 1:n_dets
    det_i = selected_ctx.selected_dets.determinants[i]
    diagonal[i] = diagonal_matrix_element(det_i, selected_ctx.base_context)
  end
  
  # Davidson subspace vectors
  V = [zeros(Ts, n_dets) for _ in 1:max_subspace]
  HV = [zeros(Ts, n_dets) for _ in 1:max_subspace]
  
  # Initialize with guess vector(s)
  k = min(max(nstates, n_guess), max_subspace)
  
  # First vectors from provided initial guesses
  n_use_guess = min(n_guess, k)
  for i in 1:n_use_guess
    V[i] .= @view(initial_guesses[:, i])
    # Normalize
    norm_val = norm(V[i])
    if norm_val > ThrNeglect
      V[i] ./= norm_val
    else
      # If guess is zero, use diagonal-based guess
      min_idx = argmin(diagonal)
      V[i] .= 0.0
      V[i][min_idx] = 1.0
    end
    
    # Orthogonalize against previous vectors
    for j in 1:(i-1)
      overlap = dot(V[j], V[i])
      V[i] .-= overlap .* V[j]
    end
    norm_val = norm(V[i])
    if norm_val > ThrNeglect
      V[i] ./= norm_val
    end
  end
  
  # Additional vectors if needed: perturb or use random
  for i in (n_use_guess+1):k
    if i <= nstates + 1 && n_use_guess > 0
      # Perturb around first guess with different random seeds
      V[i] .= initial_guesses[:, 1] .+ 0.01 * randn(n_dets)
    else
      # Random orthogonal vectors
      V[i] .= randn(n_dets)
    end
    
    # Gram-Schmidt orthogonalization
    for j in 1:(i-1)
      overlap = dot(V[j], V[i])
      V[i] .-= overlap .* V[j]
    end
    V[i] ./= norm(V[i])
  end
  
  # Compute initial HV vectors
  for i in 1:k
    contract_hamiltonian_selected!(HV[i], V[i], selected_ctx, 1.0)
  end
  
  # Iteration loop
  converged = false
  iteration = 0
  max_residual = Inf  # Initialize outside loop for warning message
  eigenvalues = zeros(Float64, nstates)
  eigenvectors = zeros(Ts, n_dets, nstates)
  sub_vecs = zeros(Ts, 1, 1)  # Placeholder

  for iter in 1:max_iterations
    iteration = iter
    
    # Build subspace Hamiltonian matrix
    H_sub = zeros(Ts, k, k)
    
    if hermitian
      for i in 1:k
        for j in 1:i
          H_sub[i,j] = dot(V[i], HV[j])
          H_sub[j,i] = conj(H_sub[i,j])
        end
      end
      # Solve eigenvalue problem for Hermitian matrix
      sub_vals, sub_vecs = eigen(Hermitian(H_sub))
    else
      # For non-Hermitian case, compute full matrix
      for i in 1:k
        for j in 1:k
          H_sub[i,j] = dot(V[i], HV[j])
        end
      end
      # Solve eigenvalue problem for non-Hermitian matrix
      # Use _eigen_subspace to rotate complex eigenvectors to real for real T
      sub_vals, sub_vecs = _eigen_subspace(H_sub, false)
    end
    
    # Extract lowest nstates eigenvalues
    eigenvalues .= real.(sub_vals[1:nstates])
    
    # For non-Hermitian case, check if eigenvalues have significant imaginary parts
    if !hermitian && iter == 1
      max_imag = maximum(abs.(imag.(sub_vals[1:nstates])))
      if max_imag > 1e-6
        @warn "Non-Hermitian Hamiltonian has eigenvalues with significant imaginary parts (max: $max_imag)"
      end
    end
    
    # Compute Ritz vectors and residuals
    max_residual = 0.0
    residuals = [zeros(Ts, n_dets) for _ in 1:nstates]
    
    for iroot in 1:nstates
      # Ritz vector: linear combination of subspace vectors
      eigenvectors[:, iroot] .= 0.0
      for i in 1:k
        eigenvectors[:, iroot] .+= sub_vecs[i, iroot] .* V[i]
      end
      
      # Residual: r = H*ψ - E*ψ
      residuals[iroot] .= 0.0
      for i in 1:k
        residuals[iroot] .+= sub_vecs[i, iroot] .* HV[i]
      end
      residuals[iroot] .-= eigenvalues[iroot] .* eigenvectors[:, iroot]
      
      res_norm = norm(residuals[iroot])
      max_residual = max(max_residual, res_norm)
    end
    
    if verbose
      if nstates == 1
        E_total = real(eigenvalues[1]) + selected_ctx.base_context.fcidump.int0
        println("Iteration $iter: E = $(E_total) Hartree, |r| = $(max_residual)")
      else
        println("Iteration $iter:")
        for iroot in 1:nstates
          E_total = real(eigenvalues[iroot]) + selected_ctx.base_context.fcidump.int0
          res_norm = norm(residuals[iroot])
          println("  State $iroot: E = $(E_total) Hartree, |r| = $(res_norm)")
        end
      end
    end
    
    # Check convergence
    if max_residual < convergence_threshold
      converged = true
      if verbose
        println("✓ Davidson converged in $iteration iterations!")
      end
      break
    end
    
    # Generate correction vectors using preconditioner
    n_new = 0
    for iroot in 1:nstates
      if k + n_new >= max_subspace
        break  # Subspace is full
      end
      
      # Preconditioned residual: r * conj(d) / (|d|² + σ)
      # Tikhonov-regularized Davidson preconditioner. For real d: r * d/(d²+σ).
      # For complex d: uses |d|² instead of d² to avoid near-zero denominators.
      correction = zeros(Ts, n_dets)
      for i in 1:n_dets
        denom = eigenvalues[iroot] - diagonal[i]
        correction[i] = residuals[iroot][i] * conj(denom) / (abs2(denom) + shift)
      end
      
      # Gram-Schmidt orthogonalization
      for j in 1:k+n_new
        overlap = dot(V[j], correction)
        correction .-= overlap .* V[j]
      end
      
      correction_norm = norm(correction)
      if correction_norm < ThrNeglect
        continue  # Skip if correction is too small
      end
      
      correction ./= correction_norm
      
      # Add to subspace
      n_new += 1
      V[k + n_new] .= correction
      contract_hamiltonian_selected!(HV[k + n_new], V[k + n_new], selected_ctx, 1.0)
    end
    
    k += n_new
    
    # Check if subspace needs refresh
    if k >= max_subspace || n_new == 0
      if verbose
        println("  Refreshing subspace (k=$k → $n_keep)")
      end
      
      # Keep the best n_keep Ritz vectors
      # The Ritz vectors are already computed and stored in eigenvectors
      # We just need to copy them to V and compute their HV products
      for i in 1:min(n_keep, size(eigenvectors, 2))
        V[i] .= eigenvectors[:, i]
        contract_hamiltonian_selected!(HV[i], V[i], selected_ctx, 1.0)
      end
      
      k = min(n_keep, size(eigenvectors, 2))
    end
  end
  
  if !converged
    @warn "Davidson did not converge in $max_iterations iterations (max residual: $max_residual)"
  end
 
  if !hermitian
    # calculate the left eigenvectors for non-Hermitian case
    left_eigenvectors = zeros(Ts, n_dets, nstates)
    left_sub_vecs = inv(sub_vecs')
    for iroot in 1:nstates
      # Ritz vector: linear combination of subspace vectors
      for i in 1:k
        left_eigenvectors[:, iroot] .+= left_sub_vecs[i, iroot] .* V[i]
      end
    end
    eigenvectors = hcat(eigenvectors, left_eigenvectors)
  end
  return (eigenvalues, eigenvectors)
end
