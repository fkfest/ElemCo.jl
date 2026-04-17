"""
Orbital localization methods.

Implements Intrinsic Bond Orbitals (IBO) (Reference: G. Knizia, JCTC 2013, 9, 4834-4843),
Pipek-Mezey localization with Mulliken charges,
and Foster-Boys localization
for occupied orbital localization
and orthogonal Projected Atomic Orbitals (OPAO) for virtual orbital localization.
"""
module OrbLocalization
using LinearAlgebra
using Printf
using ..ElemCo.ECInfos
using ..ElemCo.MSystems
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.OrbTools
using ..ElemCo.TensorTools
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions

export compute_localization_rotations, localize_orbitals

"""
    compute_iaos(EC::ECInfo, cMO_occ::AbstractMatrix)

Compute Intrinsic Atomic Orbitals (IAOs) following Knizia, JCTC 2013, 9, 4834.

The IAOs are constructed from a minimal basis and the occupied MOs.
The minimal basis is determined by `EC.options.loc.minao` (default: `"minao"`).
All nmin IAOs are returned (not just nocc), ensuring each atom contributes
IAOs that can capture bonding character in the IBO charge analysis.
Ghost atom minimal basis functions are excluded.

Returns `(C_iao, iao_atoms, natom)` where:
- `C_iao`: IAO coefficient matrix in AO basis (nAO × nmin), S-orthonormal
- `iao_atoms`: vector mapping each IAO to its atom index (1-based, contiguous)
- `natom`: number of unique (non-ghost) atoms represented in the IAOs
"""
function compute_iaos(EC::ECInfo, cMO_occ::AbstractMatrix{T}) where T
  bao = generate_basis(EC, "ao")
  minao_basis = EC.options.loc.minao
  bminao = generate_basis(EC, basisset=minao_basis)

  S = overlap(bao)             # nAO × nAO
  S12_full = overlap(bminao, bao)   # nmin_full × nAO
  S11_full = overlap(bminao)        # nmin_full × nmin_full

  # Exclude minimal basis functions on ghost atoms
  aos_min_full = ao_list(bminao)
  ghost_mask = [is_dummy(EC.system[Int(ao.icentre)]) for ao in aos_min_full]
  real_idx = findall(.!ghost_mask)

  S12 = Matrix{T}(S12_full[real_idx, :])   # nmin × nAO
  S11 = Matrix{T}(S11_full[real_idx, real_idx])  # nmin × nmin

  # s12_T = S12' is nAO × nmin
  s12_T = S12'  # nAO × nmin

  # Step 1: depolarized MO coefficients in minbas
  # ctild_minbas = S11^{-1} S12 C_occ  (nmin × nocc)
  ctild_minbas = S11 \ (S12 * cMO_occ)

  # Step 2: p12 = S^{-1} s12_T  (nAO × nmin) — minbas functions in AO representation
  p12 = S \ s12_T

  # Step 3: depolarized MOs in AO representation
  # ctild = S^{-1} s12_T ctild_minbas = p12 ctild_minbas  (nAO × nocc)
  ctild = p12 * ctild_minbas

  # Step 4: Löwdin-orthogonalize ctild under S-metric
  StC = ctild' * S * ctild  # nocc × nocc
  evals, evecs = eigen(Hermitian(StC))
  evals_inv_sqrt = [e > 1e-12 ? 1.0/sqrt(e) : 0.0 for e in evals]
  ctild = ctild * (evecs * Diagonal(evals_inv_sqrt) * evecs')  # nAO × nocc

  # Step 5: Knizia's eq. (10) — construct IAOs
  # P_occ = C_occ C_occ^T S  (projector onto occupied space)
  # P_ctild = ctild ctild^T S  (projector onto depolarized space)
  # A = (I + 2 P_occ P_ctild - P_occ - P_ctild) p12  (nAO × nmin)
  P_occ_S = cMO_occ * (cMO_occ' * S)  # nAO × nAO (= P_occ acting via S)
  P_ctild_S = ctild * (ctild' * S)     # nAO × nAO

  A = p12 + 2 * P_occ_S * (P_ctild_S * p12) - P_occ_S * p12 - P_ctild_S * p12

  # Step 6: Löwdin-orthogonalize IAOs under S-metric
  StA = A' * S * A  # nmin × nmin
  evals2, evecs2 = eigen(Hermitian(StA))
  evals2_inv_sqrt = [e > 1e-12 ? 1.0/sqrt(e) : 0.0 for e in evals2]
  C_iao = A * (evecs2 * Diagonal(evals2_inv_sqrt) * evecs2')  # nAO × nmin

  # Atom assignment: each IAO inherits the atom of its parent minbas function
  iao_min_atoms_raw = [Int(aos_min_full[j].icentre) for j in real_idx]
  unique_atoms = sort(unique(iao_min_atoms_raw))
  atom_remap = Dict(a => idx for (idx, a) in enumerate(unique_atoms))
  iao_atoms = [atom_remap[a] for a in iao_min_atoms_raw]
  natom = length(unique_atoms)

  return C_iao, iao_atoms, natom
end

"""
    compute_ao_atoms(EC::ECInfo)

Compute the atom assignment for each AO basis function.

Returns `(ao_atoms, natom)` where:
- `ao_atoms`: vector mapping each AO to its atom index (1-based, 0 for ghost atoms)
- `natom`: number of unique non-ghost atoms
"""
function compute_ao_atoms(EC::ECInfo)
  bao = generate_basis(EC, "ao")
  aos = ao_list(bao)
  ao_atoms_raw = [Int(ao.icentre) for ao in aos]
  ghost_mask = [is_dummy(EC.system[a]) for a in ao_atoms_raw]
  unique_real = sort(unique(ao_atoms_raw[.!ghost_mask]))
  atom_remap = Dict(a => idx for (idx, a) in enumerate(unique_real))
  ao_atoms = [ghost_mask[i] ? 0 : atom_remap[ao_atoms_raw[i]] for i in eachindex(ao_atoms_raw)]
  natom = length(unique_real)
  return ao_atoms, natom
end

"""
    _jacobi_localization!(R, charges, natom, nocc,
                          compute_qAij, rotate_workspace!, update_charges!;
                          exponent, maxiter, tol, method_name)

Shared Jacobi sweep engine for orbital localization.

Maximizes ``\\sum_i \\sum_A (q_A^i)^p`` over 2×2 rotations in the occupied space.
The method-specific operations (charge computation, workspace rotation, charge updates)
are provided via typed callback arguments for both type stability and code reuse.

# Arguments
- `R`: nocc × nocc rotation accumulator (modified in-place)
- `charges`: natom × nocc partial charge matrix (modified in-place)
- `natom`, `nocc`: dimensions
- `compute_qAij(i, j, A)`: returns off-diagonal charge ``q_A^{ij}`` for orbitals i,j on atom A
- `rotate_workspace!(i, j, c, s)`: applies 2×2 rotation to method-specific workspace columns
- `update_charges!(charges, i, j)`: recomputes charges for columns i and j after rotation
"""
function _jacobi_localization!(
    R::Matrix{T}, charges::Matrix{<:Real},
    natom::Int, nocc::Int,
    compute_qAij::F1, rotate_workspace!::F2, update_charges!::F3;
    exponent::Int=4, maxiter::Int=500, tol::Float64=1e-10,
    method_name::String="Localization"
) where {T, F1, F2, F3}

  func_val = sum(charges .^ exponent)

  for iter in 1:maxiter
    max_rotation = zero(real(T))
    for i in 1:nocc, j in i+1:nocc
      if exponent == 2
        Aij = zero(real(T))
        Bij = zero(real(T))
        for A in 1:natom
          qAi = charges[A, i]
          qAj = charges[A, j]
          qAij = compute_qAij(i, j, A)
          diff = qAi - qAj
          Aij += diff^2 - 4 * qAij^2
          Bij += 4 * diff * qAij
        end
        angle = atan(Bij, Aij) / 4
      elseif exponent == 4
        Aij = zero(real(T))
        Bij = zero(real(T))
        for A in 1:natom
          qAi = charges[A, i]
          qAj = charges[A, j]
          qAij = compute_qAij(i, j, A)
          Bij += 4 * qAij * (qAi^3 - qAj^3)
          Aij += qAi^4 + qAj^4 - 6 * (qAi^2 + qAj^2) * qAij^2 - qAi^3 * qAj - qAi * qAj^3
        end
        angle = atan(Bij, Aij) / 4
      else
        p = exponent
        grad = zero(real(T))
        hess = zero(real(T))
        for A in 1:natom
          qAi = charges[A, i]
          qAj = charges[A, j]
          qAij = compute_qAij(i, j, A)
          qAi_pm1 = qAi > 0 ? qAi^(p - 1) : zero(real(T))
          qAj_pm1 = qAj > 0 ? qAj^(p - 1) : zero(real(T))
          qAi_pm2 = qAi > 0 ? qAi^(p - 2) : zero(real(T))
          qAj_pm2 = qAj > 0 ? qAj^(p - 2) : zero(real(T))
          grad += 2 * p * qAij * (qAi_pm1 - qAj_pm1)
          hess += p * (4 * (p - 1) * qAij^2 * (qAi_pm2 + qAj_pm2) -
                       2 * (qAi - qAj) * (qAi_pm1 - qAj_pm1))
        end
        if hess < 0
          angle = clamp(-grad / hess, -π/4, π/4)
        elseif abs(grad) > 1e-14
          angle = sign(grad) * min(abs(grad) * 0.1, π/4)
        else
          continue
        end
      end

      abs(angle) < 1e-14 && continue
      max_rotation = max(max_rotation, abs(angle))

      c = cos(angle)
      s = sin(angle)
      rotate_workspace!(i, j, c, s)
      for k in 1:nocc
        ri = R[k, i]; rj = R[k, j]
        R[k, i] = c * ri + s * rj
        R[k, j] = -s * ri + c * rj
      end
      update_charges!(charges, i, j)
    end

    func_val_new = sum(charges .^ exponent)
    delta = func_val_new - func_val
    func_val = func_val_new

    if max_rotation < tol
      println("$method_name localization converged in $iter sweeps (max_rotation=$(Printf.@sprintf("%.2e", max_rotation)))")
      break
    end
    # Functional-based convergence: catches degenerate oscillations (e.g., core/lone-pair pairs
    # on the same atom where the atan2 formula gives ±π/8 indefinitely without changing the functional)
    if iter > 2 && abs(delta) < tol * max(abs(func_val), one(real(T)))
      println("$method_name localization converged in $iter sweeps (functional change=$(Printf.@sprintf("%.2e", delta)))")
      break
    end
    if iter == maxiter
      @warn "$method_name localization did not converge in $maxiter sweeps (max_rotation=$max_rotation)"
    end
  end
  return nothing
end

"""
    localize_ibo(cMO_occ::AbstractMatrix, S::AbstractMatrix, C_iao::AbstractMatrix, 
                 iao_atoms::Vector{Int}, natom::Int; exponent=4, maxiter=500, tol=1e-10)

Localize occupied orbitals using the IBO criterion.

Maximizes ``\\sum_i \\sum_A (q_A^i)^p`` where ``q_A^i = \\sum_{\\mu \\in A} |\\langle iao_\\mu | \\phi_i \\rangle|^2``
using 2×2 Jacobi rotations.

Returns the rotation matrix `R_occ` (nocc × nocc) that transforms canonical to localized occupied MOs:
`C_occ_loc = C_occ * R_occ`.
"""
function localize_ibo(cMO_occ::AbstractMatrix{T}, S::AbstractMatrix, C_iao::AbstractMatrix,
                      iao_atoms::Vector{Int}, natom::Int;
                      exponent::Int=4, maxiter::Int=500, 
                      tol::Float64=1e-10) where T
  nocc = size(cMO_occ, 2)
  niao = size(C_iao, 2)

  # Compute IAO representation of occupied MOs: Q[μ,i] = ⟨IAO_μ|ϕ_i⟩ = (C_iao^T S C_occ)[μ,i]
  Q = C_iao' * S * cMO_occ  # niao × nocc

  # Current rotation matrix (accumulates Jacobi rotations)
  R = Matrix{T}(I, nocc, nocc)

  # Early exit: if only 1 unique atom, the IBO functional is constant
  if natom <= 1
    println("IBO localization: only $natom unique atom(s), returning identity rotation")
    return R
  end

  # Compute partial charges: q_A^i = Σ_{μ∈A} |Q[μ,i]|²  (skip ghost atoms with iao_atoms[mu]==0)
  charges = zeros(real(T), natom, nocc)
  for i in 1:nocc, mu in 1:niao
    A = iao_atoms[mu]
    A == 0 && continue
    charges[A, i] += abs2(Q[mu, i])
  end

  # IBO-specific callbacks for the shared Jacobi sweep
  compute_qAij = (i, j, A) -> begin
    qAij = zero(real(T))
    for mu in 1:niao
      iao_atoms[mu] == A || continue
      qAij += real(conj(Q[mu, i]) * Q[mu, j])
    end
    qAij
  end

  rotate_workspace! = (i, j, c, s) -> begin
    for mu in 1:niao
      qi = Q[mu, i]; qj = Q[mu, j]
      Q[mu, i] = c * qi + s * qj
      Q[mu, j] = -s * qi + c * qj
    end
  end

  update_charges! = (charges, i, j) -> begin
    for A in 1:natom
      charges[A, i] = zero(real(T))
      charges[A, j] = zero(real(T))
    end
    for mu in 1:niao
      A = iao_atoms[mu]
      A == 0 && continue
      charges[A, i] += abs2(Q[mu, i])
      charges[A, j] += abs2(Q[mu, j])
    end
  end

  _jacobi_localization!(R, charges, natom, nocc,
    compute_qAij, rotate_workspace!, update_charges!;
    exponent, maxiter, tol, method_name="IBO")

  # Post-convergence refinement: break degeneracy by maximizing AO compactness.
  # For orbital pairs with identical atom-resolved charges (e.g., lone pairs on the
  # same atom), the IBO functional is exactly flat and the rotation angle is arbitrary.
  # We refine by maximizing ∑_i ∑_μ |Q_{μi}|⁴ within each degenerate subgroup,
  # which prefers orbitals localized on the fewest IAOs (e.g., σ/π separation).
  fill!(charges, zero(real(T)))
  for i in 1:nocc, mu in 1:niao
    A = iao_atoms[mu]; A == 0 && continue
    charges[A, i] += abs2(Q[mu, i])
  end

  charge_tol = 1e-6
  groups = Vector{Vector{Int}}()
  assigned = falses(nocc)
  for i in 1:nocc
    assigned[i] && continue
    group = [i]
    for j in i+1:nocc
      assigned[j] && continue
      if maximum(abs.(charges[:, i] .- charges[:, j])) < charge_tol
        push!(group, j)
        assigned[j] = true
      end
    end
    assigned[i] = true
    length(group) > 1 && push!(groups, group)
  end

  if !isempty(groups)
    for group in groups
      ng = length(group)
      for _sweep in 1:50
        max_rot = zero(real(T))
        for gi in 1:ng, gj in gi+1:ng
          ii = group[gi]
          jj = group[gj]
          # Compute optimal angle for AO compactness: maximize ∑_μ |Q_μi|^4 + |Q_μj|^4
          PP = zero(real(T))
          QQ = zero(real(T))
          PQ = zero(real(T))
          for mu in 1:niao
            p_mu = (abs2(Q[mu, ii]) - abs2(Q[mu, jj])) / 2
            w_mu = real(conj(Q[mu, ii]) * Q[mu, jj])
            PP += p_mu^2
            QQ += w_mu^2
            PQ += p_mu * w_mu
          end
          angle = atan(2 * PQ, PP - QQ) / 4
          abs(angle) < 1e-14 && continue
          max_rot = max(max_rot, abs(angle))
          c = cos(angle)
          s = sin(angle)
          for mu in 1:niao
            qi = Q[mu, ii]
            qj = Q[mu, jj]
            Q[mu, ii] = c * qi + s * qj
            Q[mu, jj] = -s * qi + c * qj
          end
          for k in 1:nocc
            ri = R[k, ii]
            rj = R[k, jj]
            R[k, ii] = c * ri + s * rj
            R[k, jj] = -s * ri + c * rj
          end
        end
        max_rot < 1e-10 && break
      end
    end
    println("IBO: refined $(length(groups)) degenerate group(s) by AO compactness")
  end

  return R
end

"""
    localize_pm(cMO_occ::AbstractMatrix, S::AbstractMatrix,
                ao_atoms::Vector{Int}, natom::Int; exponent=2, maxiter=500, tol=1e-10)

Localize occupied orbitals using the Pipek-Mezey criterion with Mulliken charges.

Maximizes ``\\sum_i \\sum_A (q_A^i)^p`` where 
``q_A^i = \\sum_{\\mu \\in A} \\sum_\\nu C_{\\mu i} S_{\\mu\\nu} C_{\\nu i}``
(Mulliken partial charges) using 2×2 Jacobi rotations.

# Arguments
- `cMO_occ`: occupied MO coefficients (nAO × nocc)
- `S`: AO overlap matrix (nAO × nAO)
- `ao_atoms`: vector mapping each AO to its atom index (1-based, 0 for ghost)
- `natom`: number of unique (non-ghost) atoms
- `exponent`: power of charges in functional (default 2, the standard PM functional)
- `maxiter`: maximum number of Jacobi sweeps
- `tol`: convergence threshold for rotation angles

Returns the rotation matrix `R_occ` (nocc × nocc) that transforms canonical to localized occupied MOs:
`C_occ_loc = C_occ * R_occ`.
"""
function localize_pm(cMO_occ::AbstractMatrix{T}, S::AbstractMatrix,
                     ao_atoms::Vector{Int}, natom::Int;
                     exponent::Int=2, maxiter::Int=500,
                     tol::Float64=1e-10) where T
  nocc = size(cMO_occ, 2)
  nao = size(cMO_occ, 1)

  # Work with C (MO coefficients) and P = S*C (overlap-weighted coefficients)
  C = copy(cMO_occ)  # nAO × nocc
  P = S * C           # nAO × nocc

  # Current rotation matrix (accumulates Jacobi rotations)
  R = Matrix{T}(I, nocc, nocc)

  if natom <= 1
    println("PM localization: only $natom unique atom(s), returning identity rotation")
    return R
  end

  # Compute Mulliken partial charges: q_A^i = Σ_{μ∈A} Re(P[μ,i] * C̄[μ,i])
  charges = zeros(real(T), natom, nocc)
  for i in 1:nocc, mu in 1:nao
    A = ao_atoms[mu]
    A == 0 && continue
    charges[A, i] += real(P[mu, i] * conj(C[mu, i]))
  end

  # PM-specific callbacks for the shared Jacobi sweep
  compute_qAij = (i, j, A) -> begin
    qAij = zero(real(T))
    for mu in 1:nao
      ao_atoms[mu] == A || continue
      qAij += real(P[mu, i] * conj(C[mu, j]) + P[mu, j] * conj(C[mu, i])) / 2
    end
    qAij
  end

  rotate_workspace! = (i, j, c, s) -> begin
    for mu in 1:nao
      ci = C[mu, i]; cj = C[mu, j]
      C[mu, i] = c * ci + s * cj
      C[mu, j] = -s * ci + c * cj
      pi = P[mu, i]; pj = P[mu, j]
      P[mu, i] = c * pi + s * pj
      P[mu, j] = -s * pi + c * pj
    end
  end

  update_charges! = (charges, i, j) -> begin
    for A in 1:natom
      charges[A, i] = zero(real(T))
      charges[A, j] = zero(real(T))
    end
    for mu in 1:nao
      A = ao_atoms[mu]
      A == 0 && continue
      charges[A, i] += real(P[mu, i] * conj(C[mu, i]))
      charges[A, j] += real(P[mu, j] * conj(C[mu, j]))
    end
  end

  _jacobi_localization!(R, charges, natom, nocc,
    compute_qAij, rotate_workspace!, update_charges!;
    exponent, maxiter, tol, method_name="PM")

  return R
end

"""
    localize_boys(cMO_occ::AbstractMatrix, S::AbstractMatrix,
                  Dx::AbstractMatrix, Dy::AbstractMatrix, Dz::AbstractMatrix;
                  maxiter=500, tol=1e-10)

Localize occupied orbitals using the Foster-Boys criterion.

Maximizes ``\\sum_i |\\langle i | \\mathbf{r} | i \\rangle|^2 = \\sum_i \\sum_{\\alpha=x,y,z} (d_\\alpha^i)^2``
where ``d_\\alpha^i = \\langle i | r_\\alpha | i \\rangle`` are the orbital dipole moments,
using 2×2 Jacobi rotations.

# Arguments
- `cMO_occ`: occupied MO coefficients (nAO × nocc)
- `S`: AO overlap matrix (nAO × nAO)
- `Dx`, `Dy`, `Dz`: AO dipole integral matrices ``\\langle \\mu | r_\\alpha | \\nu \\rangle``
- `maxiter`: maximum number of Jacobi sweeps
- `tol`: convergence threshold for rotation angles

Returns the rotation matrix `R_occ` (nocc × nocc) that transforms canonical to localized occupied MOs:
`C_occ_loc = C_occ * R_occ`.
"""
function localize_boys(cMO_occ::AbstractMatrix{T}, S::AbstractMatrix,
                       Dx::AbstractMatrix, Dy::AbstractMatrix, Dz::AbstractMatrix;
                       maxiter::Int=500, tol::Float64=1e-10) where T
  nocc = size(cMO_occ, 2)
  nao = size(cMO_occ, 1)

  # Work with C (MO coefficients) and DC_α = D_α * C (dipole-weighted coefficients)
  C = copy(cMO_occ)
  DCx = Dx * C
  DCy = Dy * C
  DCz = Dz * C

  R = Matrix{T}(I, nocc, nocc)

  # Treat x, y, z as 3 "atoms" for the Jacobi framework
  natom = 3

  # Compute orbital dipole moments: d_α^i = Σ_μ C[μ,i] * DC_α[μ,i]
  charges = zeros(real(T), natom, nocc)
  for i in 1:nocc
    charges[1, i] = real(dot(view(C, :, i), view(DCx, :, i)))
    charges[2, i] = real(dot(view(C, :, i), view(DCy, :, i)))
    charges[3, i] = real(dot(view(C, :, i), view(DCz, :, i)))
  end

  compute_qAij = (i, j, A) -> begin
    DC = A == 1 ? DCx : A == 2 ? DCy : DCz
    real(dot(view(C, :, i), view(DC, :, j)))
  end

  rotate_workspace! = (i, j, c, s) -> begin
    for mu in 1:nao
      ci = C[mu, i]; cj = C[mu, j]
      C[mu, i] = c * ci + s * cj
      C[mu, j] = -s * ci + c * cj
      dxi = DCx[mu, i]; dxj = DCx[mu, j]
      DCx[mu, i] = c * dxi + s * dxj
      DCx[mu, j] = -s * dxi + c * dxj
      dyi = DCy[mu, i]; dyj = DCy[mu, j]
      DCy[mu, i] = c * dyi + s * dyj
      DCy[mu, j] = -s * dyi + c * dyj
      dzi = DCz[mu, i]; dzj = DCz[mu, j]
      DCz[mu, i] = c * dzi + s * dzj
      DCz[mu, j] = -s * dzi + c * dzj
    end
  end

  update_charges! = (charges, i, j) -> begin
    charges[1, i] = real(dot(view(C, :, i), view(DCx, :, i)))
    charges[2, i] = real(dot(view(C, :, i), view(DCy, :, i)))
    charges[3, i] = real(dot(view(C, :, i), view(DCz, :, i)))
    charges[1, j] = real(dot(view(C, :, j), view(DCx, :, j)))
    charges[2, j] = real(dot(view(C, :, j), view(DCy, :, j)))
    charges[3, j] = real(dot(view(C, :, j), view(DCz, :, j)))
  end

  _jacobi_localization!(R, charges, natom, nocc,
    compute_qAij, rotate_workspace!, update_charges!;
    exponent=2, maxiter, tol, method_name="Boys")

  return R
end

"""
    compute_opao_rotation(cMO_virt::AbstractMatrix, S::AbstractMatrix; tol=1e-8)

Compute the rotation matrix for virtual orbitals based on orthogonalized PAOs.

PAOs (Projected Atomic Orbitals) are constructed by projecting AO basis functions
onto the virtual space: ``C_{\\text{PAO}} = C_{\\text{virt}} C_{\\text{virt}}^T S``.
The PAO overlap matrix is decomposed via ALPACA (pivoted Cholesky with rank control)
to get orthogonal PAOs, which define a rotation of the virtual space.

Returns `R_virt` (nvirt × nvirt) such that `C_virt_loc = C_virt * R_virt`.
"""
function compute_opao_rotation(cMO_virt::AbstractMatrix{T},
                               S::AbstractMatrix; tol::Float64=1e-8) where T
  nao = size(S, 1)
  nvirt = size(cMO_virt, 2)

  # PAO coefficients: project AO basis functions onto virtual space
  # C_PAO = C_virt C_virt^T S  (nAO × nAO, rank = nvirt)
  # PAO_μ = Σ_a |ϕ_a⟩⟨ϕ_a|χ_μ⟩
  C_PAO = cMO_virt * (cMO_virt' * S)

  # PAO overlap: S_PAO = C_PAO^T S C_PAO  (nAO × nAO, rank = nvirt)
  S_PAO = Hermitian(C_PAO' * S * C_PAO)

  # Orthogonalize PAOs using ALPACA with explicit rank control
  M = sqrtinvchol(S_PAO; tol=tol, max_rank=nvirt)  # nAO × nvirt
  n_opao = size(M, 2)
  if n_opao != nvirt
    @warn "OPAO dimension ($n_opao) differs from virtual dimension ($nvirt)"
  end

  # Orthogonal PAOs: C_OPAO = C_PAO * M  (nAO × nvirt, orthonormal under S)
  C_OPAO = C_PAO * M

  # Virtual rotation: express canonical virtuals in OPAO basis
  # R_virt = C_virt^T S C_OPAO  (nvirt × nvirt)
  R_virt = cMO_virt' * S * C_OPAO

  return R_virt
end

"""
    compute_localization_rotations(EC::ECInfo; exponent=4)

Compute orbital localization rotation matrices for occupied (IBO) and virtual (OPAO) orbitals.

Returns `(R_occ, R_virt)` where:
- `R_occ`: nocc × nocc rotation matrix (canonical → IBO-localized occupied)
- `R_virt`: nvirt × nvirt rotation matrix (canonical → OPAO-localized virtual)

The rotation matrices are orthogonal/unitary and can be used to transform
amplitude matrices before SVD decomposition.
"""
function compute_localization_rotations(EC::ECInfo; exponent::Int=4)
  println("Computing orbital localization rotations...")

  cMO, _, _ = fetch_orbitals(EC)
  if is_restricted(cMO)
    cMO_full = cMO[1]
  else
    error("Orbital localization currently only supports restricted orbitals")
  end

  SP = EC.space
  cMO_occ = cMO_full[:, SP['o']]
  cMO_virt = cMO_full[:, SP['v']]

  # Get ALL occupied MOs (including frozen core) for IAO and PAO construction
  space_save, space_b4freeze = restore_system_space!(EC; verbose=false)
  cMO_all_occ = cMO_full[:, space_b4freeze['o']]
  restore_space!(EC, space_save)

  bao = generate_basis(EC, "ao")
  S = overlap(bao)

  # IAOs from ALL occupied (including frozen core) for proper atom assignment
  # Then IBO rotates only the active occupied MOs
  println("  Computing IAOs...")
  C_iao, iao_atoms, natom = compute_iaos(EC, cMO_all_occ)
  println("  Localizing occupied orbitals (IBO, exponent=$exponent)...")
  R_occ = localize_ibo(cMO_occ, S, C_iao, iao_atoms, natom; exponent=exponent)

  # OPAO for virtual orbitals
  println("  Computing orthogonal PAO rotation for virtual orbitals...")
  R_virt = compute_opao_rotation(cMO_virt, S)

  nocc = size(R_occ, 1)
  nvirt = size(R_virt, 1)
  println("  Localization done: $nocc occupied, $nvirt virtual orbitals")

  return R_occ, R_virt
end

"""
    localize_orbitals(EC::ECInfo)

Localize the current orbitals using IBO, PM, or Boys (occupied) and optionally OPAO (virtual).

Reads orbitals from the wavefunction dump, applies localization to the
occupied orbitals and optionally OPAO rotation to the virtual orbitals,
then stores the localized orbitals back to the dump file.

Options are read from `EC.options.loc`:
- `method::String`: `"ibo"` (default), `"pm"` (Pipek-Mezey with Mulliken charges),
  or `"boys"` (Foster-Boys, maximizes sum of squared orbital dipole moments).
- `virtual::Bool`: if `true` (default), also localize virtual orbitals via OPAO.
- `exponent::Int`: localization exponent, 2 for Pipek-Mezey, 4 for fourth-moment (default).

# Examples
```julia
@dfhf
@localize              # localize occupied (IBO) + virtual (OPAO)
@set loc method="pm"
@localize              # Pipek-Mezey localization (occupied) + OPAO (virtual)
@set loc method="boys"
@localize              # Foster-Boys localization (occupied) + OPAO (virtual)
@set loc virtual=false
@localize              # localize only occupied
```
"""
function localize_orbitals(EC::ECInfo)
  localize_virtual = EC.options.loc.virtual
  exponent = EC.options.loc.exponent
  method = lowercase(EC.options.loc.method)
  use_pm = method == "pm"
  use_boys = method == "boys"
  method_name = use_boys ? "Boys" : use_pm ? "PM" : "IBO"

  println("Localizing orbitals ($method_name" * (localize_virtual ? "+OPAO" : "") * ", exponent=$exponent)...")

  # Fetch current orbitals, energies, and occupations
  use_start = EC.options.wf.start != ""
  cMO, type, basis = fetch_orbitals(EC; start=use_start)
  energies = fetch_orbital_energies(EC)
  occupations = fetch_orbital_occupations(EC)
  has_basis = !isempty(basis)

  if !is_restricted(cMO)
    error("Orbital localization currently only supports restricted orbitals")
  end

  cMO_full = cMO[1]

  # Determine core and valence orbitals using freeze settings
  space_save, space_b4freeze = restore_system_space!(EC; verbose=false)
  all_occ_range = space_b4freeze['o']
  valence_occ_range = EC.space['o']  # after freeze: valence only
  core_range = setdiff(all_occ_range, valence_occ_range)
  virt_range = EC.space['v']
  restore_space!(EC, space_save)

  localize_core = EC.options.loc.localize_core
  ncore = length(core_range)
  if ncore > 0
    if localize_core
      println("  Localizing $ncore core orbital(s) separately: $core_range")
    else
      println("  Keeping $ncore core orbital(s) frozen: $core_range")
    end
  end

  cMO_occ = cMO_full[:, valence_occ_range]
  cMO_core = cMO_full[:, core_range]
  cMO_virt = cMO_full[:, virt_range]
  cMO_all_occ = cMO_full[:, all_occ_range]

  bao = generate_basis(EC, "ao")
  S = overlap(bao)

  # Localize occupied orbitals
  if use_boys
    # Foster-Boys: maximize sum of squared orbital dipole moments
    Dx, Dy, Dz = dipole(bao)
    println("  Localizing occupied orbitals (Boys)...")
    R_occ = localize_boys(cMO_occ, S, Dx, Dy, Dz)
    if localize_core && ncore > 0
      println("  Localizing core orbitals (Boys)...")
      R_core = localize_boys(cMO_core, S, Dx, Dy, Dz)
    end
  elseif use_pm
    # Pipek-Mezey with Mulliken charges
    ao_atoms, natom = compute_ao_atoms(EC)
    println("  Localizing occupied orbitals (PM, exponent=$exponent)...")
    R_occ = localize_pm(cMO_occ, S, ao_atoms, natom; exponent=exponent)
    if localize_core && ncore > 0
      println("  Localizing core orbitals (PM, exponent=$exponent)...")
      R_core = localize_pm(cMO_core, S, ao_atoms, natom; exponent=exponent)
    end
  else
    # IBO: use IAOs from ALL occupied (including frozen core) for proper atom assignment
    println("  Computing IAOs...")
    C_iao, iao_atoms, natom = compute_iaos(EC, cMO_all_occ)
    println("  Localizing occupied orbitals (IBO, exponent=$exponent)...")
    R_occ = localize_ibo(cMO_occ, S, C_iao, iao_atoms, natom; exponent=exponent)
    if localize_core && ncore > 0
      println("  Localizing core orbitals (IBO, exponent=$exponent)...")
      R_core = localize_ibo(cMO_core, S, C_iao, iao_atoms, natom; exponent=exponent)
    end
  end

  # Apply occupied rotation (only to valence)
  cMO_full_loc = copy(cMO_full)
  cMO_full_loc[:, valence_occ_range] = cMO_occ * R_occ

  # Apply core rotation if requested
  if localize_core && ncore > 0
    cMO_full_loc[:, core_range] = cMO_core * R_core
  end

  # Rotate orbital energies: ε_loc_p = diag(R^T diag(ε) R)_p
  energies_a = copy(energies[1])
  if !isempty(energies_a)
    ε_occ = energies_a[valence_occ_range]
    energies_a[valence_occ_range] = diag(R_occ' * Diagonal(ε_occ) * R_occ)
    if localize_core && ncore > 0
      ε_core = energies_a[core_range]
      energies_a[core_range] = diag(R_core' * Diagonal(ε_core) * R_core)
    end
  end

  if localize_virtual
    # OPAO for virtual orbitals
    println("  Computing orthogonal PAO rotation for virtual orbitals...")
    R_virt = compute_opao_rotation(cMO_virt, S)
    cMO_full_loc[:, virt_range] = cMO_virt * R_virt
    if !isempty(energies_a)
      ε_virt = energies_a[virt_range]
      energies_a[virt_range] = diag(R_virt' * Diagonal(ε_virt) * R_virt)
    end
  end

  energies_loc = (energies_a, energies[2])

  nocc = length(valence_occ_range)
  nvirt = length(virt_range)
  println("  Localization done: $nocc occupied" * (localize_virtual ? ", $nvirt virtual" : "") * " orbitals")

  # Store localized orbitals
  loc_type = method_name * (localize_virtual ? "+OPAO" : "")
  cMO_loc = SpinMatrix(cMO_full_loc)
  if has_basis
    dump_orbitals(EC, cMO_loc; basis=basis, type=loc_type, energies=energies_loc, occupations=occupations)
  else
    dump_rotations(EC, cMO_loc; type=loc_type, energies=energies_loc, occupations=occupations)
  end

  return nothing
end

end # module
