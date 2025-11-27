# ===========================================
# Heat-Bath Selection 
# ===========================================

"""
    heatbath_selection(selected_ctx::SelectedCIContext,
                       variational_coeffs::AbstractVector{Scalar},
                       options::HCIOptions,
                       E_current::Float64,
                       setup_data::HCISetupData,
                       detorder::Union{Nothing, Vector{Int}}=nothing, store_dets::Bool=true) 
                       -> (VecDict{Determinant, Scalar}, (Float64, Float64))

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
                            setup_data::HCISetupData,
                            coeffs_dg::Union{Nothing, AbstractVector{Scalar}}=nothing,
                            detorder::Union{Nothing, Vector{Int}}=nothing, store_dets::Bool=true)
  t0 = time_ns()
  shift = options.pt2_shift
  variational_dets = determinants(selected_ctx)
  ctx = selected_ctx.base_context
  ThrNeglect = options.thr_negligible
  # Get HB candidates determinants from variational space together with the H*c values
  DetType = eltype(variational_dets)
  candidates = Dict{DetType, ExcVals}()
  newcandidates = VecDict{DetType, ExcVals}()  # Temporary storage for new candidates
  fockd_buf = FockDiagonal(ctx.n_orb) # Reusable Fock diagonal buffer
  spaces_buf = OrbSpaces(ctx.n_orb)  # Reusable orbital spaces buffer
  pt_negl = 0.0
  # Use efficient threshold-based excitation generation
  eps_h = options.epsilon_h > -0.1 ? options.epsilon_h : options.epsilon/10.0
  eps_c = options.epsilon_c > -0.1 ? options.epsilon_c : options.epsilon_h
  # first we run through new determinants
  # for the old determinants we only update those that were already added by the new ones.
  # this avoids adding the same determinants over and over again that will be neglected anyway by PT2
  # this has a very small effect on the final energy (coming only from the change of the correlation energy
  # in the denominator in PT2 step), but speeds up the selection significantly in later iterations.
  n_olddet = n_old_dets(selected_ctx)
  det_indices = Iterators.flatten(((n_olddet+1):length(variational_dets), 1:n_olddet))
  for idx in det_indices
    if isnothing(detorder)
      # use natural order
      i = idx
    else
      # use provided order
      i = detorder[idx]
    end
    det = variational_dets[i]
    c_I = variational_coeffs[i]
    if abs(c_I) < ThrNeglect
      continue  # Skip negligible coefficients
    end
    if isnothing(coeffs_dg)
      c_I_dg = nothing
    else
      c_I_dg = coeffs_dg[i]
    end
    eps = eps_h / abs(c_I)
    ecorr = E_current - get_diagonal_element(selected_ctx, i)
    if i <= n_olddet
      eps_c = Inf  # Skip adding new contributions for old determinants
    end
    pt_negl += generate_excitations!(candidates, newcandidates, det, c_I, c_I_dg, ecorr, ctx, setup_data,
                          spaces_buf, fockd_buf, eps, eps_c, shift)
    mergewith!(+, candidates, newcandidates)
  end
  newcandidates = nothing  # Free memory
  # Remove determinants already in variational space
  setdiff4dict!(candidates, variational_dets)
  if options.verbose
    println("  Generated $(length(candidates)) candidate determinants")
  end
  t0 = print_time(options.print_level, t0, "candidate determinants", 1)

  # Select determinants above threshold or until target reached
  new_dets = VecDict{DetType, Scalar}()
  pt2_correction = 0.0
  t2sq = 0.0
  # use square of epsilon_p to match probability definition (T_2^2)
  eps_p = options.epsilon_p > -0.1 ? options.epsilon_p : options.epsilon
  epsilon = eps_p^2
  for (det_J, ev) in candidates
    c_J = ev.coef
    hval_J = ev.hval
    # Selection probability: |Σ c_I H_IJ|² / ΔE²
    c_J² = abs2(c_J)
    pt2_correction += c_J * hval_J  # Perturbative energy contribution
    t2sq += c_J²
    if store_dets && c_J² >= epsilon
      new_dets[det_J] = c_J²
    end
  end
  t0 = print_time(options.print_level, t0, "perturbative selection", 1)
  # Apply renormalization if requested
  if options.renorm_pt2
    renorm_factor = 1.0 / (1.0 + t2sq)
    pt2_correction *= renorm_factor
    if options.verbose
      println("  Renormalized PT2 correction by factor $renorm_factor")
    end
  end
  pt2_correction += pt_negl
  return new_dets, (pt2_correction, pt_negl)
end

"""
    modify_excitation!(excitations::Dict{Determinant, ExcVals},
                       det::Determinant, vals::ExcVals) -> Float64

Modify existing excitation entry in the excitations dictionary by adding new ExcVals.
If the determinant is not present, returns the energy contribution from the new ExcVals.
"""
function modify_excitation!(excitations::Dict{Determinant{OPattern}, ExcVals},
                 det::Determinant{OPattern}, vals::ExcVals) where OPattern
  v1 = get(excitations, det, nothing)
  en = 0.0
  if !isnothing(v1)
    @inbounds excitations[det] = v1 + vals
  else
    en = vals.coef * vals.hval
  end
  return en
end

"""
    generate_excitations!(excitations::VecDict{Determinant, ExcVals},
                          det::Determinant, coef::Scalar, 
                          coef_dg::Union{Scalar, Nothing}, ecorr::Scalar,
                          ctx::FCIContext,
                          setup_data::HCISetupData, spaces::OrbSpaces, fockd::FockDiagonal,
                          epsilon::Float64, shift) -> Int

Generate only excitations with |H| > epsilon using pre-computed data.

This is the efficient excitation generation from Holmes et al. (2016):
1. For doubles: Use pre-sorted lists to stop early when |H| < epsilon
2. For singles: Compute on-the-fly and discard if |H| < epsilon

Additionally, `H*coef/denom` and `coef*H` is computed and stored during generation for efficiency.

Works with both FCIContext and HCIContext.

`spaces` is an OrbSpaces object used for temporary storage of occupied and virtual orbitals.
`fockd` is the Fock diagonal object used to store Fock matrix diagonals for the determinant.
`coef_dg` is the bra coefficient of the determinant (dagger) if needed, otherwise `nothing`.
"""
function generate_excitations!(excitations::Dict{Determinant{OPattern}, ExcVals},
                               newexcitations::VecDict{Determinant{OPattern}, ExcVals},
                               det::Determinant{OPattern}, coef::Scalar, 
                               coef_dg::Union{Scalar, Nothing}, ecorr::Scalar,
                               ctx::Union{FCIContext{OPattern}, HCIContext{OPattern}},
                               setup_data::HCISetupData,
                               spaces::OrbSpaces, fockd::FockDiagonal,
                               epsilon::Float64, epsilon_c::Float64, shift) where OPattern
  n_orb = ctx.n_orb
  empty!(newexcitations)
  set_orbspaces!(spaces, det)
  alpha_occ = spaces.occa
  alpha_virt = spaces.virta
  beta_occ = spaces.occb
  beta_virt = spaces.virtb

  # Pre-compute diagonal Fock matrix elements for the current determinant
  calc_fock_diagonal4det!(fockd, ctx, alpha_occ, beta_occ)
  fa = fockd.alpha
  fb = fockd.beta
  
  pt = 0.0

  # ===========================================
  # 1. Generate double excitations using pre-computed lists
  # ===========================================
  
  # Check if we should skip all double excitations
  if epsilon <= setup_data.h_doub_max
    # Alpha-alpha double excitations
    pt += add_excitations!(excitations, newexcitations, det, coef, coef_dg, double_alpha_excitation_phase, alpha_occ,
                     setup_data.double_excitations_aa, fa, ecorr, shift, epsilon, epsilon_c)
    # Beta-beta double excitations
    pt += add_excitations!(excitations, newexcitations, det, coef, coef_dg, double_beta_excitation_phase, beta_occ,
                     setup_data.double_excitations_bb, fb, ecorr, shift, epsilon, epsilon_c)
    # Mixed double excitations (alpha-beta) (use int2ab pre-computed lists, i.e., no exchange)
    pt += add_excitations!(excitations, newexcitations, det, coef, coef_dg, alpha_occ, beta_occ, n_orb,
                     setup_data.double_excitations_ab, fa, fb, ecorr, shift, epsilon, epsilon_c)
  end
  # ===========================================
  # 2. Generate single excitations with on-the-fly filtering using Fock elements
  # ===========================================

  # Alpha single excitations
  pt += add_excitations!(excitations, newexcitations, det, coef, coef_dg, single_alpha_excitation_phase,
                  alpha_occ, alpha_virt, beta_occ,
                  ctx.int1a, ctx.heval_data.h1e2_aa, ctx.heval_data.h1e2_ab,
                  setup_data.singles_denoma,
                  fa, ecorr, shift, epsilon, epsilon_c)
  # Beta single excitations
  pt += add_excitations!(excitations, newexcitations, det, coef, coef_dg, single_beta_excitation_phase,
                  beta_occ, beta_virt, alpha_occ,
                  ctx.int1b, ctx.heval_data.h1e2_bb, ctx.heval_data.h1e2_ba,
                  setup_data.singles_denomb,
                  fb, ecorr, shift, epsilon, epsilon_c)

  return pt
end

"""
    add_excitations!(excitations::VecDict{Determinant, ExcVals}, det::Determinant,
                    coef::Scalar, coef_dg::Union{Scalar, Nothing}, double_excitation_phase::Function,
                    occ, pchb_excitations, fdiag, ecorr, shift, epsilon::Float64, epsilon_c::Float64)

Generate same-spin double excitations from occupied orbitals using pre-computed
Heat-Bath lists, adding only those with |H| > epsilon and storing in excitations dictionary
together with their weighted matrix elements.
"""
function add_excitations!(excitations::Dict{Determinant{OPattern}, ExcVals}, 
                          newexcitations::VecDict{Determinant{OPattern}, ExcVals},
                          det::Determinant{OPattern},
                          coef::Scalar, coef_dg::Union{Scalar, Nothing}, double_excitation_phase::Function, 
                          occ, pchb_excitations,
                          fdiag, ecorr, shift, epsilon::Float64, epsilon_c::Float64) where OPattern
  pt = 0.0
  for idx_i in eachindex(occ)
    i = occ[idx_i]
    for idx_j in (idx_i+1):length(occ)
      j = occ[idx_j]
      # Look up pre-sorted list for (i,j) pair
      pq_key = trip_index(i, j)
      for entry in pchb_excitations[pq_key]
        a, b, h_val = entry.a, entry.b, entry.value
        # Stop when matrix element falls below threshold
        if abs(h_val) < epsilon
          break
        end
        
        # Check if r and s are virtual (not occupied)
        if !(a in occ) && !(b in occ)
          new_det, phase = double_excitation_phase(det, i, j, a, b)
          @assert_devel count_ones(new_det.alpha) == count_ones(det.alpha) "Alpha electron count mismatch $i $j -> $a $b from $(bitstring(det.alpha))"
          @assert_devel count_ones(new_det.beta) == count_ones(det.beta) "Beta electron count mismatch $i $j -> $a $b from $(bitstring(det.beta))"

          denom = fdiag[a] + fdiag[b] - fdiag[i] - fdiag[j] + entry.denom - ecorr
          fac = -denom/(denom^2 + shift)
          hc = coef * h_val * phase
          c = hc * fac
          if !isnothing(coef_dg)
            hc = coef_dg * entry.dagger * phase
          end
          if abs(c) < epsilon_c
            # negligible contribution, modify existing entry if present
            pt += modify_excitation!(excitations, new_det, ExcVals(c, hc))
          else
            # significant contribution, add new entry if not present
            newexcitations[new_det] = ExcVals(c, hc)
          end
        end
      end
    end
  end
  return pt
end

"""
    add_excitations!(excitations::VecDict{Determinant, ExcVals}, det::Determinant, coef::Scalar,
                    coef_dg::Union{Scalar, Nothing}, alpha_occ, beta_occ, n_orb, 
                    pchb_excitations, fa, fb, ecorr, shift, epsilon::Float64, epsilon_c::Float64)

Generate mixed-spin double excitations from occupied alpha and beta orbitals
using pre-computed Heat-Bath lists, adding only those with |H| > epsilon.
"""
function add_excitations!(excitations::Dict{Determinant{OPattern}, ExcVals}, 
                          newexcitations::VecDict{Determinant{OPattern}, ExcVals},
                          det::Determinant{OPattern},
                          coef::Scalar, coef_dg::Union{Scalar, Nothing}, alpha_occ, beta_occ, 
                          n_orb, pchb_excitations,
                          fa, fb, ecorr, shift, epsilon::Float64, epsilon_c::Float64) where OPattern
  pt = 0.0
  for i_alpha in alpha_occ
    for i_beta in beta_occ
      pq_key = trip_index(i_alpha, i_beta, n_orb)
      for entry in pchb_excitations[pq_key]
        a, b, h_val = entry.a, entry.b, entry.value
        if abs(h_val) < epsilon
          break
        end
        # r is alpha virtual, s is beta virtual
        if !(a in alpha_occ) && !(b in beta_occ)
          new_det, phase = double_alpha_beta_excitation_phase(det, i_alpha, i_beta, a, b)
          denom = fa[a] + fb[b] - fa[i_alpha] - fb[i_beta] + entry.denom - ecorr
          fac = -denom/(denom^2 + shift)
          hc = coef * h_val * phase
          c = hc * fac
          if !isnothing(coef_dg)
            hc = coef_dg * entry.dagger * phase
          end
          if abs(c) < epsilon_c
            # negligible contribution, modify existing entry if present
            pt += modify_excitation!(excitations, new_det, ExcVals(c, hc))
          else
            # significant contribution, add new entry if not present
            newexcitations[new_det] = ExcVals(c, hc)
          end
        end
      end
    end
  end
  return pt
end

"""
    add_excitations!(excitations::VecDict{Determinant, ExcVals}, det::Determinant,
                    coef::Scalar, coef_dg::Union{Scalar, Nothing}, single_excitation_phase::Function, 
                    occs, virts, occs_opp, int1, h1e2_same, h1e2_opp,
                    fdiag, ecorr, shift, epsilon::Float64, epsilon_c::Float64)

Add single excitations to the excitations dictionary.
"""
function add_excitations!(excitations::Dict{Determinant{OPattern}, ExcVals}, 
                          newexcitations::VecDict{Determinant{OPattern}, ExcVals},
                          det::Determinant{OPattern},
                          coef::Scalar, coef_dg::Union{Scalar, Nothing}, single_excitation_phase::Function, 
                          occs, virts, occs_opp, int1, h1e2_same, h1e2_opp, singles_denom,
                          fdiag, ecorr, shift, 
                          epsilon::Float64, epsilon_c::Float64) where OPattern
  pt = 0.0
  for i in occs
    for a in virts
      # Compute Fock matrix element f_ai
      h_val = compute_fock_element(int1, h1e2_same, h1e2_opp, occs, occs_opp, a, i)
      if abs(h_val) >= epsilon
        new_det, phase = single_excitation_phase(det, i, a)
        denom = fdiag[a] - fdiag[i] + singles_denom[a, i] - ecorr
        fac = -denom/(denom^2 + shift)
        hc = coef * h_val * phase
        c = hc * fac
        if !isnothing(coef_dg)
          h_val_dg = compute_fock_element(int1, h1e2_same, h1e2_opp, occs, occs_opp, i, a)
          hc = coef_dg * h_val_dg * phase
        end
        if abs(c) < epsilon_c
          # negligible contribution, modify existing entry if present
          pt += modify_excitation!(excitations, new_det, ExcVals(c, hc))
        else
          # significant contribution, add new entry if not present
          newexcitations[new_det] = ExcVals(c, hc)
        end
      end
    end
  end
  return pt
end