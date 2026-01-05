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

Compute ⟨det_j|Ĥ|det_i⟩ directly using orbital excitation analysis.
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
                                        selected_ctx::SelectedCIContext, prefactor)
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
                               input::Vector{Scalar}, prefactor)
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
# Diagonal Element Computation
# ===========================================

"""
    compute_diagonal_element(det::Determinant, ctx) -> Scalar

Compute diagonal matrix element ⟨det|H|det⟩ for a single determinant using HEvalData.
Works with both FCIContext and HCIContext.
"""
@pib function compute_diagonal_element(det::Determinant, ctx::Union{FCIContext, HCIContext})::Scalar
  spaces = ctx.heval_data.spaces_buf
  set_occupied_orbspaces!(spaces, det)
  return calc_diagonalH(ctx.heval_data, spaces.occa, spaces.occb)
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
