# triples routines


"""
    PseudoCanonicalTransform

Holds transformation matrices for pseudo-canonicalization of amplitudes and integrals.

For biorthogonal systems, the Fock matrix is non-Hermitian and requires separate
left (L) and right (R) eigenvector matrices: F = R * Diagonal(ϵ) * L^†.

The transformation convention is:
- Lower indices (first half of index string) → transformed using Left eigenvectors
- Upper indices (second half of index string) → transformed using Right eigenvectors
"""
struct PseudoCanonicalTransform
  "Whether any transformation is needed"
  need_transform::Bool
  "Right eigenvectors for occupied orbitals (SpinMatrix with α and β)"
  Ro::SpinMatrix{Float64}
  "Left eigenvectors for occupied orbitals (SpinMatrix with α and β)"
  Lo::SpinMatrix{Float64}
  "Right eigenvectors for virtual orbitals (SpinMatrix with α and β)"
  Rv::SpinMatrix{Float64}
  "Left eigenvectors for virtual orbitals (SpinMatrix with α and β)"
  Lv::SpinMatrix{Float64}
  "Pseudo-canonical occupied orbital energies (α and β)"
  ϵo::SpinVector{Float64}
  "Pseudo-canonical virtual orbital energies (α and β)"
  ϵv::SpinVector{Float64}
end

"""
    PseudoCanonicalTransform(EC::ECInfo; restricted::Bool=true)

Construct a PseudoCanonicalTransform by checking and diagonalizing Fock matrix blocks.

If the Fock matrix is already diagonal (within threshold), returns identity transforms.
For non-diagonal blocks, computes the eigenvectors for pseudo-canonicalization.
"""
function PseudoCanonicalTransform(EC::ECInfo; restricted::Bool=true)
  thr = EC.options.cc.fock_diag_thr
  hermitian = !is_similarity_transformed(EC.fd) 
  if restricted
    is_diag_oo, is_diag_vv, F_oo, F_vv = check_fock_diagonal_blocks(EC)
    # Check occupied block
    ϵo, Ro, Lo, rotate_oo = compute_pseudocanonical_transform(F_oo; skip=is_diag_oo, thr, hermitian)
    # Check virtual block
    ϵv, Rv, Lv, rotate_vv = compute_pseudocanonical_transform(F_vv; skip=is_diag_vv, thr, hermitian)
    
    return PseudoCanonicalTransform(
      rotate_oo || rotate_vv,
      SpinMatrix(Ro), SpinMatrix(Lo),
      SpinMatrix(Rv), SpinMatrix(Lv),
      SpinVector(ϵo), SpinVector(ϵv)
    )
  else
    # Unrestricted case
    is_diag_oo_a, is_diag_vv_a, F_oo_a, F_vv_a = check_fock_diagonal_blocks(EC, :α)
    is_diag_oo_b, is_diag_vv_b, F_oo_b, F_vv_b = check_fock_diagonal_blocks(EC, :β)
    
    # Alpha occupied
    ϵo_α, Ro_α, Lo_α, rotate_oo_a = compute_pseudocanonical_transform(F_oo_a; skip=is_diag_oo_a, thr, hermitian)
    # Alpha virtual
    ϵv_α, Rv_α, Lv_α, rotate_vv_a = compute_pseudocanonical_transform(F_vv_a; skip=is_diag_vv_a, thr, hermitian)
    # Beta occupied
    ϵo_β, Ro_β, Lo_β, rotate_oo_b = compute_pseudocanonical_transform(F_oo_b; skip=is_diag_oo_b, thr, hermitian)
    # Beta virtual
    ϵv_β, Rv_β, Lv_β, rotate_vv_b = compute_pseudocanonical_transform(F_vv_b; skip=is_diag_vv_b, thr, hermitian)
    
    return PseudoCanonicalTransform(
      rotate_oo_a || rotate_vv_a || rotate_oo_b || rotate_vv_b,
      SpinMatrix(Ro_α, Ro_β), SpinMatrix(Lo_α, Lo_β),
      SpinMatrix(Rv_α, Rv_β), SpinMatrix(Lv_α, Lv_β),
      SpinVector(ϵo_α, ϵo_β), SpinVector(ϵv_α, ϵv_β)
    )
  end
end

"""
    check_fock_diagonal_blocks(EC::ECInfo, spin::Symbol=:α)

Check if the Fock matrix occ-occ and virt-virt blocks are diagonal.
Returns `(is_diagonal_oo, is_diagonal_vv, F_oo, F_vv)`.
"""
function check_fock_diagonal_blocks(EC::ECInfo, spin::Symbol=:α)
  thr = EC.options.cc.fock_diag_thr
  SP = EC.space
  if spin == :α
    fock = load2idx(EC, "f_mm")
    o_space = SP['o']
    v_space = SP['v']
  else
    fock = load2idx(EC, "f_MM")
    o_space = SP['O']
    v_space = SP['V']
  end
  
  F_oo = fock[o_space, o_space]
  F_vv = fock[v_space, v_space]
  
  # Check off-diagonal elements
  max_offdiag_oo = maximum(abs.(F_oo - Diagonal(diag(F_oo))); init=0.0)
  max_offdiag_vv = maximum(abs.(F_vv - Diagonal(diag(F_vv))); init=0.0)
  is_diagonal_oo = max_offdiag_oo < thr
  is_diagonal_vv = max_offdiag_vv < thr
  if thr < 0.0
    # No check requested
    println("Skipping Fock matrix diagonality check (fock_diag_thr < 0)")
    is_diagonal_oo = true
    is_diagonal_vv = true
  end 
  if !is_diagonal_oo
    println("Max off-diagonal in Fock $spin occ-occ: $max_offdiag_oo (thr=$thr)")
  end
  if !is_diagonal_vv
    println("Max off-diagonal in Fock $spin virt-virt: $max_offdiag_vv (thr=$thr)")
  end
  
  return is_diagonal_oo, is_diagonal_vv, F_oo, F_vv
end

"""
    compute_pseudocanonical_transform(F_block::Matrix; skip::Bool=false, thr::Float64=1e-6, hermitian::Bool=true)

Diagonalize a potentially non-Hermitian Fock block.
Returns `(ϵ, Cr, Cl, is_beneficial)`: eigenvalues, right/left eigenvector matrices,
and a flag indicating whether the transformation is beneficial.

If `skip=true`, skips the diagonalization and returns identity transforms.
If `hermitian=true`, assumes the Fock block is Hermitian and uses `eigen(Hermitian(...))`.
The transformation is considered beneficial if the transformed Fock block is significantly 
different from the original. 

Uses `rotate_eigenvectors_to_real` for complex pairs.
"""
function compute_pseudocanonical_transform(F_block::Matrix; skip::Bool=false, thr::Float64=1e-6, hermitian::Bool=true)
  if skip
    ϵ = diag(F_block)
    Ctr = Matrix{Float64}(I, size(F_block))
    return ϵ, Ctr, Ctr, false
  end
  if hermitian
    eigvals, eigvecs = eigen(Hermitian(F_block))
    eigvecs_left = eigvecs_right = eigvecs
    ϵ = eigvals
  else
    # Diagonalize (general eigenvalue problem for non-Hermitian)
    eigvals, eigvecs_right = eigen(F_block)
  
    # Handle complex eigenvalues - rotate complex conjugate pairs to real
    eigvecs_right, ϵ = rotate_eigenvectors_to_real(eigvecs_right, eigvals)
  
    # Compute left eigenvectors: L = (R^{-1})^T
    eigvecs_left = (inv(eigvecs_right))'
  
    # Balance norms of left and right eigenvectors
    eigvecs_right, eigvecs_left = balance_norms!(eigvecs_right, eigvecs_left)
  end
  
  # Check if transformation is beneficial
  # Compute transformed Fock: L' * F * R
  F_transformed = eigvecs_left' * F_block * eigvecs_right
  max_diff = maximum(abs.(F_block - F_transformed); init=0.0)
  
  # Transformation is beneficial if the resulting Fock is significantly different
  is_beneficial = max_diff > thr
  if !is_beneficial 
    # Transformation not beneficial, use identity
    ϵ = diag(F_block)
    eigvecs_left = eigvecs_right = Matrix{Float64}(I, size(F_block))
  end
  return ϵ, eigvecs_right, eigvecs_left, is_beneficial
end

"""
    pseudocan_transform!(pct::PseudoCanonicalTransform, arr::AbstractArray, indices::String; 
                      conjugate::Bool=false)

Transform an array to the pseudo-canonical basis in-place using the transformation matrices.

The `indices` string specifies the index type for each dimension:
- `'o'`/`'O'`: occupied α/β
- `'v'`/`'V'`: virtual α/β

The transformation convention (canonical order):
- First half of indices = lower indices (subscript/bra) → Left eigenvectors
- Second half of indices = upper indices (superscript/ket) → Right eigenvectors

For `conjugate=true`, the L/R roles are swapped (used for Lagrange multipliers).

Supported array dimensions: 2 and 4.

# Example
```julia
# Transform T2 amplitudes stored as T_{ab}^{ij} in [a,b,i,j] order
pseudocan_transform!(pct, T2, "vvoo")

# Transform U2 Lagrange multipliers (use conjugate)
pseudocan_transform!(pct, U2, "vvoo"; conjugate=true)

# Transform mixed-spin integrals v_{aB}^{iJ}
pseudocan_transform!(pct, int_aB_iJ, "vVoO")
```
"""
function pseudocan_transform!(pct::PseudoCanonicalTransform, arr::AbstractArray, indices::String;
                           conjugate::Bool=false)
  if !pct.need_transform
    return arr
  end
  
  ndim = ndims(arr)
  @assert length(indices) == ndim "indices string length must match array dimensions"
  @assert ndim ∈ (2, 4) "only 2-index and 4-index arrays are supported"
  
  half = ndim ÷ 2
  
  # Build list of transformation matrices for each index
  # First half: lower indices → L (or R if conjugate)
  # Second half: upper indices → R (or L if conjugate)
  transforms = Vector{Matrix{Float64}}(undef, ndim)
  
  for (i, idx) in enumerate(indices)
    is_first_half = i <= half
    # For first half (lower): use L, for second half (upper): use R
    # If conjugate, swap L↔R
    use_right = !is_first_half ⊻ conjugate  # XOR to swap if conjugate
    
    if idx == 'o'
      transforms[i] = use_right ? pct.Ro.α : pct.Lo.α
    elseif idx == 'O'
      transforms[i] = use_right ? pct.Ro.β : pct.Lo.β
    elseif idx == 'v'
      transforms[i] = use_right ? pct.Rv.α : pct.Lv.α
    elseif idx == 'V'
      transforms[i] = use_right ? pct.Rv.β : pct.Lv.β
    else
      error("Unknown index type '$idx'. Use 'o', 'O', 'v', or 'V'.")
    end
  end
  
  if ndim == 2
    transform_2idx!(arr, transforms[1], transforms[2])
  else  # ndim == 4
    transform_4idx!(arr, transforms[1], transforms[2], transforms[3], transforms[4])
  end
  
  return arr
end

"""
    transform_2idx!(arr::AbstractMatrix, U1::Matrix, U2::Matrix)

Transform a 2-index array in-place using @mtensor.
Implements: arr[p',q'] = U1[p,p'] * U2[q,q'] * arr[p,q]
"""
function transform_2idx!(arr::AbstractMatrix, U1::Matrix, U2::Matrix)
  @mtensor tmp[p, q] := U1[r, p] * arr[r, q]
  @mtensor arr[p, q] = tmp[p, r] * U2[r, q]
  return arr
end

"""
    transform_4idx!(arr::AbstractArray{T,4}, U1::Matrix, U2::Matrix, U3::Matrix, U4::Matrix) where T

Transform a 4-index array in-place using @mtensor.
Implements: arr[p',q',r',s'] = U1[p,p'] * U2[q,q'] * U3[r,r'] * U4[s,s'] * arr[p,q,r,s]
"""
function transform_4idx!(arr::AbstractArray{T,4}, U1::Matrix, U2::Matrix, 
                         U3::Matrix, U4::Matrix) where T
  @mtensor tmp[a, b, c, d] := U1[p, a] * arr[p, b, c, d]
  @mtensor arr[a, b, c, d] = tmp[a, p, c, d] * U2[p, b]
  @mtensor tmp[a, b, c, d] = arr[a, b, p, d] * U3[p, c]
  @mtensor arr[a, b, c, d] = tmp[a, b, c, p] * U4[p, d]
  return arr
end

"""
    get_pseudo_orbital_energies(pct::PseudoCanonicalTransform, spin::Symbol=:α)

Get pseudo-canonical orbital energies from the transform object.
Returns `(ϵo, ϵv)` for the specified spin.
"""
function get_pseudo_orbital_energies(pct::PseudoCanonicalTransform, spin::Symbol=:α)
  if spin == :α
    return pct.ϵo.α, pct.ϵv.α
  else
    return pct.ϵo.β, pct.ϵv.β
  end
end

"""
    calc_pertT(EC::ECInfo, method::ECMethod; save_t3=false)

  Calculate (T) correction for [Λ][U]CCSD(T)

  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`. 
  If `save_t3` is true, the T3 amplitudes are saved in `T_vvvooo` file (only for closed-shell).
"""
function calc_pertT(EC::ECInfo, method::ECMethod; save_t3=false)
  if is_unrestricted(method) || has_prefix(method, "R")
    # unrestricted/restricted-open-shell branch
    if has_prefix(method, "Λ")
      @assert !save_t3 "Saving perturbative triples not implemented for ΛUCCSD(T)"
      return calc_ΛpertT_unrestricted(EC)
    else
      @assert !save_t3 "Saving perturbative triples not implemented for UCCSD(T)"
      return calc_pertT_unrestricted(EC)
    end
  else
    # closed-shell branch
    if has_prefix(method, "Λ")
      @assert !save_t3 "Saving perturbative triples not implemented for ΛCCSD(T)"
      return calc_ΛpertT_closed_shell(EC)
    else 
      return calc_pertT_closed_shell(EC; save_t3)
    end
  end
end

"""
    calc_pertT_closed_shell(EC::ECInfo; save_t3=false)

  Calculate (T) correction for closed-shell CCSD.
  If the Fock matrix is not diagonal in the occ-occ and virt-virt blocks,
  performs pseudo-canonicalization before the (T) calculation.

  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_pertT_closed_shell(EC::ECInfo; save_t3=false)
  # Build pseudo-canonical transformation
  pct = PseudoCanonicalTransform(EC; restricted=true)
  if pct.need_transform
    println("Fock matrix not diagonal - performing pseudo-canonicalization for CCSD(T)")
  end
  
  # Load and  (if needed) transform amplitudes
  T1 = load2idx(EC,"T_vo")
  pseudocan_transform!(pct, T1, "vo")
  T2 = load4idx(EC,"T_vvoo")
  pseudocan_transform!(pct, T2, "vvoo")

  # Load and (if needed) transform integrals
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC,"oovv"),[3,4,1,2])
  pseudocan_transform!(pct, vv_oo, "vvoo"; conjugate=true)
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC,"vvvo")
  pseudocan_transform!(pct, vvvo, "vvvo")
  # ``v_{ia}^{jk}``
  ovoo = ints2(EC,"ovoo")
  pseudocan_transform!(pct, ovoo, "ovoo")

  # Transform Fock ov block: f_{ia} as [i,a] → "ov"
  fov = load2idx(EC, "f_mm")[EC.space['o'], EC.space['v']]
  pseudocan_transform!(pct, fov, "ov")

  nocc = n_occ_orbs(EC)
  nvir = n_virt_orbs(EC)
  ϵo, ϵv = get_pseudo_orbital_energies(pct)

  X = zeros(nvir, nvir, nvir)
  Kijk = zeros(nvir, nvir, nvir)

  Enb3 = 0.0
  IntX = zeros(nvir, nocc)
  IntY = zeros(nvir, nocc)
  if save_t3
    t3file, T3 = newmmap(EC,"T_vvvooo",(nvir,nvir,nvir,uppertriangular_index(nocc,nocc,nocc)))
  end
  for k = 1:nocc 
    for j = 1:k
      prefac = (j == k) ? 1.0 : 2.0
      for i = 1:j
        fac = prefac 
        if i == j 
          if j == k
            continue
          end 
          fac = 1.0
        end
        v!T2ij = @mview T2[:,:,i,j]
        v!T2ik = @mview T2[:,:,i,k]
        v!T2jk = @mview T2[:,:,j,k]
        v!T2i = @mview T2[:,:,:,i]
        v!T2j = @mview T2[:,:,:,j]
        v!T2k = @mview T2[:,:,:,k]
        v!vvvk = @mview vvvo[:,:,:,k]
        v!vvvj = @mview vvvo[:,:,:,j]
        v!vvvi = @mview vvvo[:,:,:,i]
        v!ovjk = @mview ovoo[:,:,j,k]
        v!ovkj = @mview ovoo[:,:,k,j]
        v!ovik = @mview ovoo[:,:,i,k]
        v!ovki = @mview ovoo[:,:,k,i]
        v!ovij = @mview ovoo[:,:,i,j]
        v!ovji = @mview ovoo[:,:,j,i]
        @mtensor begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = v!T2ij[a,d] * v!vvvk[b,c,d]
          Kijk[a,b,c] += v!T2ij[d,b] * v!vvvk[a,c,d]
          Kijk[a,b,c] += v!T2ik[a,d] * v!vvvj[c,b,d]
          Kijk[a,b,c] += v!T2ik[d,c] * v!vvvj[a,b,d]
          Kijk[a,b,c] += v!T2jk[b,d] * v!vvvi[c,a,d]
          Kijk[a,b,c] += v!T2jk[d,c] * v!vvvi[b,a,d]

          Kijk[a,b,c] -= v!T2i[b,a,l] * v!ovjk[l,c]
          Kijk[a,b,c] -= v!T2j[a,b,l] * v!ovik[l,c]
          Kijk[a,b,c] -= v!T2i[c,a,l] * v!ovkj[l,b]
          Kijk[a,b,c] -= v!T2k[a,c,l] * v!ovij[l,b]
          Kijk[a,b,c] -= v!T2j[c,b,l] * v!ovki[l,a]
          Kijk[a,b,c] -= v!T2k[b,c,l] * v!ovji[l,a]
        end
        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        if save_t3
          ijk = uppertriangular_index(i,j,k)
          T3[:,:,:,ijk] = Kijk
          for abc ∈ CartesianIndices(Kijk)
            a,b,c = Tuple(abc)
            T3[abc,ijk] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
          end
        end
        @mtensor  X[a,b,c] = 4.0*Kijk[a,b,c] - 2.0*Kijk[a,c,b] - 2.0*Kijk[c,b,a] - 2.0*Kijk[b,a,c] + Kijk[c,a,b] + Kijk[b,c,a]
        for abc ∈ CartesianIndices(X)
          a,b,c = Tuple(abc)
          X[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
        end

        @mtensor Enb3 += fac * (Kijk[a,b,c] * X[a,b,c])
        
        v!vv_jk = @mview vv_oo[:,:,j,k]
        v!vv_ik = @mview vv_oo[:,:,i,k]
        v!vv_ij = @mview vv_oo[:,:,i,j]
        v!IntX_i = @mview IntX[:,i]
        v!IntX_j = @mview IntX[:,j]
        v!IntX_k = @mview IntX[:,k]
        v!IntY_i = @mview IntY[:,i]
        v!IntY_j = @mview IntY[:,j]
        v!IntY_k = @mview IntY[:,k]
        @mtensor v!IntX_i[a] += fac * (X[a,b,c] * v!vv_jk[b,c])
        @mtensor v!IntX_j[b] += fac * (X[a,b,c] * v!vv_ik[a,c])
        @mtensor v!IntX_k[c] += fac * (X[a,b,c] * v!vv_ij[a,b])
        @mtensor v!IntY_i[a] += fac * (X[a,b,c] * v!T2jk[b,c])
        @mtensor v!IntY_j[b] += fac * (X[a,b,c] * v!T2ik[a,c])
        @mtensor v!IntY_k[c] += fac * (X[a,b,c] * v!T2ij[a,b])
      end 
    end
  end
  if save_t3
    closemmap(EC,t3file,T3)
  end
  # singles contribution
  @mtensor En3 = T1[a,i] * IntX[a,i]
  # fock contribution
  @mtensor En3 += fov[i,a] * IntY[a,i]
  En3 += Enb3
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_ΛpertT_closed_shell(EC::ECInfo)

  Calculate (T) correction for closed-shell ΛCCSD(T).

  The amplitudes are stored in `T_vvoo` file, 
  and the Lagrangian multipliers are stored in `U_vvoo` file.
  If the Fock matrix is not diagonal in the occ-occ and virt-virt blocks,
  performs pseudo-canonicalization before the (T) calculation.
  Return ( `"ET3"`=(T) energy, `"ET3b"`=[T] energy) `OutDict`.
"""
function calc_ΛpertT_closed_shell(EC::ECInfo)
  # Build pseudo-canonical transformation
  pct = PseudoCanonicalTransform(EC; restricted=true)
  if pct.need_transform
    println("Fock matrix not diagonal - performing pseudo-canonicalization for ΛCCSD(T)")
  end
  
  nocc = n_occ_orbs(EC)
  nvir = n_virt_orbs(EC)
  ϵo, ϵv = get_pseudo_orbital_energies(pct)
  
  # Load and  (if needed) transform amplitudes
  T1 = load2idx(EC, "T_vo")
  pseudocan_transform!(pct, T1, "vo")
  T2 = load4idx(EC, "T_vvoo")
  pseudocan_transform!(pct, T2, "vvoo")
  
  # U1, U2: Lagrange multipliers use conjugate transformation
  U1 = load2idx(EC, "U_vo")
  pseudocan_transform!(pct, U1, "vo"; conjugate=true)
  U2 = contra2covariant(load4idx(EC, "U_vvoo"))
  pseudocan_transform!(pct, U2, "vvoo"; conjugate=true)
  
  # Load and (if needed) transform integrals
  # v^{ab}_{ij} stored as [a,b,i,j] → "vvoo"
  vv_oo = permutedims(ints2(EC, "oovv"), [3,4,1,2])
  pseudocan_transform!(pct, vv_oo, "vvoo"; conjugate=true)
  # v_{ab}^{ck} as [a,b,c,k] → "vvvo"
  vvvo = ints2(EC, "vvvo")
  pseudocan_transform!(pct, vvvo, "vvvo")
  # v_{ia}^{jk} as [i,a,j,k] → "ovoo"
  ovoo = ints2(EC, "ovoo")
  pseudocan_transform!(pct, ovoo, "ovoo")
  
  # Load and transform integrals for U contractions (conjugate transformation)
  # v^{ab}_{ck} as [a,b,c,k] → "vvvo"
  vv_vo = permutedims(ints2(EC, "vovv"), [3,4,1,2])
  pseudocan_transform!(pct, vv_vo, "vvvo"; conjugate=true)
  # v^{ia}_{jk} as [i,a,j,k] → "ovoo"
  # v_{jk}^{ia}, reordered to v^{ia}_{jk}
  ov_oo = permutedims(ints2(EC, "ooov"), [3,4,1,2])
  pseudocan_transform!(pct, ov_oo, "ovoo"; conjugate=true)

  # Transform Fock ov block: f_{ia} as [i,a] → "ov"
  fov = load2idx(EC, "f_mm")[EC.space['o'], EC.space['v']]
  pseudocan_transform!(pct, fov, "ov")
  
  Enb3 = 0.0
  IntX = zeros(nvir, nocc)
  IntY = zeros(nvir, nocc)
  for k = 1:nocc 
    for j = 1:k
      prefac = (j == k) ? 1.0 : 2.0
      for i = 1:j
        fac = prefac 
        if i == j 
          if j == k
            continue
          end 
          fac = 1.0
        end
        v!T2ij = @mview T2[:,:,i,j]
        v!T2ik = @mview T2[:,:,i,k]
        v!T2jk = @mview T2[:,:,j,k]
        v!T2i = @mview T2[:,:,:,i]
        v!T2j = @mview T2[:,:,:,j]
        v!T2k = @mview T2[:,:,:,k]
        v!vvvk = @mview vvvo[:,:,:,k]
        v!vvvj = @mview vvvo[:,:,:,j]
        v!vvvi = @mview vvvo[:,:,:,i]
        v!ovjk = @mview ovoo[:,:,j,k]
        v!ovkj = @mview ovoo[:,:,k,j]
        v!ovik = @mview ovoo[:,:,i,k]
        v!ovki = @mview ovoo[:,:,k,i]
        v!ovij = @mview ovoo[:,:,i,j]
        v!ovji = @mview ovoo[:,:,j,i]
        @mtensor begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] := v!T2ij[a,d] * v!vvvk[b,c,d]
          Kijk[a,b,c] += v!T2ij[d,b] * v!vvvk[a,c,d]
          Kijk[a,b,c] += v!T2ik[a,d] * v!vvvj[c,b,d]
          Kijk[a,b,c] += v!T2ik[d,c] * v!vvvj[a,b,d]
          Kijk[a,b,c] += v!T2jk[b,d] * v!vvvi[c,a,d]
          Kijk[a,b,c] += v!T2jk[d,c] * v!vvvi[b,a,d]

          Kijk[a,b,c] -= v!T2i[b,a,l] * v!ovjk[l,c]
          Kijk[a,b,c] -= v!T2j[a,b,l] * v!ovik[l,c]
          Kijk[a,b,c] -= v!T2i[c,a,l] * v!ovkj[l,b]
          Kijk[a,b,c] -= v!T2k[a,c,l] * v!ovij[l,b]
          Kijk[a,b,c] -= v!T2j[c,b,l] * v!ovki[l,a]
          Kijk[a,b,c] -= v!T2k[b,c,l] * v!ovji[l,a]
        end
        @mtensor  X[a,b,c] := 4.0*Kijk[a,b,c] - 2.0*Kijk[a,c,b] - 2.0*Kijk[c,b,a] - 2.0*Kijk[b,a,c] + Kijk[c,a,b] + Kijk[b,c,a]

        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        for abc ∈ CartesianIndices(X)
          a,b,c = Tuple(abc)
          X[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
        end

        v!U2ij = @mview U2[:,:,i,j]
        v!U2ik = @mview U2[:,:,i,k]
        v!U2jk = @mview U2[:,:,j,k]
        v!U2i = @mview U2[:,:,:,i]
        v!U2j = @mview U2[:,:,:,j]
        v!U2k = @mview U2[:,:,:,k]
        v!vv_vk = @mview vv_vo[:,:,:,k]
        v!vv_vj = @mview vv_vo[:,:,:,j]
        v!vv_vi = @mview vv_vo[:,:,:,i]
        v!ov_jk = @mview ov_oo[:,:,j,k]
        v!ov_kj = @mview ov_oo[:,:,k,j]
        v!ov_ik = @mview ov_oo[:,:,i,k]
        v!ov_ki = @mview ov_oo[:,:,k,i]
        v!ov_ij = @mview ov_oo[:,:,i,j]
        v!ov_ji = @mview ov_oo[:,:,j,i]
        @mtensor begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = v!U2ij[a,d] * v!vv_vk[b,c,d]
          Kijk[a,b,c] += v!U2ij[d,b] * v!vv_vk[a,c,d]
          Kijk[a,b,c] += v!U2ik[a,d] * v!vv_vj[c,b,d]
          Kijk[a,b,c] += v!U2ik[d,c] * v!vv_vj[a,b,d]
          Kijk[a,b,c] += v!U2jk[b,d] * v!vv_vi[c,a,d]
          Kijk[a,b,c] += v!U2jk[d,c] * v!vv_vi[b,a,d]

          Kijk[a,b,c] -= v!U2i[b,a,l] * v!ov_jk[l,c]
          Kijk[a,b,c] -= v!U2j[a,b,l] * v!ov_ik[l,c]
          Kijk[a,b,c] -= v!U2i[c,a,l] * v!ov_kj[l,b]
          Kijk[a,b,c] -= v!U2k[a,c,l] * v!ov_ij[l,b]
          Kijk[a,b,c] -= v!U2j[c,b,l] * v!ov_ki[l,a]
          Kijk[a,b,c] -= v!U2k[b,c,l] * v!ov_ji[l,a]
        end
        @mtensor Enb3 += fac * (Kijk[a,b,c] * X[a,b,c])
        
        v!vv_jk = @mview vv_oo[:,:,j,k]
        v!vv_ik = @mview vv_oo[:,:,i,k]
        v!vv_ij = @mview vv_oo[:,:,i,j]
        v!IntX_i = @mview IntX[:,i]
        v!IntX_j = @mview IntX[:,j]
        v!IntX_k = @mview IntX[:,k]
        v!IntY_i = @mview IntY[:,i]
        v!IntY_j = @mview IntY[:,j]
        v!IntY_k = @mview IntY[:,k]
        @mtensor v!IntX_i[a] += fac * (X[a,b,c] * v!vv_jk[b,c])
        @mtensor v!IntX_j[b] += fac * (X[a,b,c] * v!vv_ik[a,c])
        @mtensor v!IntX_k[c] += fac * (X[a,b,c] * v!vv_ij[a,b])
        @mtensor v!IntY_i[a] += fac * (X[a,b,c] * v!U2jk[b,c])
        @mtensor v!IntY_j[b] += fac * (X[a,b,c] * v!U2ik[a,c])
        @mtensor v!IntY_k[c] += fac * (X[a,b,c] * v!U2ij[a,b])
      end 
    end
  end
  # singles contribution
  @mtensor En3 = 0.5 * (U1[a,i] * IntX[a,i])
  # fock contribution
  @mtensor En3 += fov[i,a] * IntY[a,i]
  En3 += Enb3
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_pertT_unrestricted(EC::ECInfo)

  Calculate (T) correction for UCCSD.

  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_pertT_unrestricted(EC::ECInfo)
  # Build pseudo-canonical transformation
  pct = PseudoCanonicalTransform(EC; restricted=false)
  if pct.need_transform
    println("Fock matrix not diagonal - performing pseudo-canonicalization for UCCSD(T)")
  end

  T1a = load2idx(EC,"T_vo")
  pseudocan_transform!(pct, T1a, "vo")
  T2 = load4idx(EC,"T_vvoo")
  pseudocan_transform!(pct, T2, "vvoo")
  En3a, Enb3a = values(calc_pertT_samespin(EC, T1a, T2, pct, :α))

  T1b = load2idx(EC,"T_VO")
  pseudocan_transform!(pct, T1b, "VO")
  T2ba = permutedims(load4idx(EC,"T_vVoO"), [2,1,4,3])
  pseudocan_transform!(pct, T2ba, "VvOo")
  En3ab, Enb3ab = values(calc_pertT_mixedspin(EC, T1a, T2, T1b, T2ba, pct, :α))
  T2ba = nothing

  T2 = load4idx(EC,"T_VVOO")
  pseudocan_transform!(pct, T2, "VVOO")
  En3b, Enb3b = values(calc_pertT_samespin(EC, T1b, T2, pct, :β))

  T2ab = load4idx(EC,"T_vVoO")
  pseudocan_transform!(pct, T2ab, "vVoO")
  En3ba, Enb3ba = values(calc_pertT_mixedspin(EC, T1b, T2, T1a, T2ab, pct, :β))

  En3 = En3a + En3b + En3ab + En3ba 
  Enb3 = Enb3a + Enb3b + Enb3ab + Enb3ba
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_ΛpertT_unrestricted(EC::ECInfo)

  Calculate (T) correction for ΛUCCSD(T).

  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_ΛpertT_unrestricted(EC::ECInfo)
  # Build pseudo-canonical transformation
  pct = PseudoCanonicalTransform(EC; restricted=false)
  if pct.need_transform
    println("Fock matrix not diagonal - performing pseudo-canonicalization for ΛUCCSD(T)")
  end
  
  U1a = load2idx(EC,"U_vo")
  pseudocan_transform!(pct, U1a, "vo"; conjugate=true)
  U1b = load2idx(EC,"U_VO")
  pseudocan_transform!(pct, U1b, "VO"; conjugate=true)
  T2 = load4idx(EC,"T_vvoo")
  pseudocan_transform!(pct, T2, "vvoo")
  U2 = load4idx(EC,"U_vvoo")
  pseudocan_transform!(pct, U2, "vvoo"; conjugate=true)
  En3a, Enb3a = values(calc_ΛpertT_samespin(EC, T2, U1a, U2, pct, :α))

  T2ba = permutedims(load4idx(EC,"T_vVoO"), [2,1,4,3])
  pseudocan_transform!(pct, T2ba, "VvOo")
  U2ba = permutedims(load4idx(EC,"U_vVoO"), [2,1,4,3])
  pseudocan_transform!(pct, U2ba, "VvOo"; conjugate=true)
  En3ab, Enb3ab = values(calc_ΛpertT_mixedspin(EC, T2, T2ba, U1a, U2, U1b, U2ba, pct, :α))
  T2ba = nothing
  U2ba = nothing

  T2 = load4idx(EC,"T_VVOO")
  pseudocan_transform!(pct, T2, "VVOO")
  U2 = load4idx(EC,"U_VVOO")
  pseudocan_transform!(pct, U2, "VVOO"; conjugate=true)
  En3b, Enb3b = values(calc_ΛpertT_samespin(EC, T2, U1b, U2, pct, :β))

  T2ab = load4idx(EC,"T_vVoO")
  pseudocan_transform!(pct, T2ab, "vVoO")
  U2ab = load4idx(EC,"U_vVoO")
  pseudocan_transform!(pct, U2ab, "vVoO"; conjugate=true)
  En3ba, Enb3ba = values(calc_ΛpertT_mixedspin(EC, T2, T2ab, U1b, U2, U1a, U2ab, pct, :β))

  En3 = En3a + En3b + En3ab + En3ba 
  Enb3 = Enb3a + Enb3b + Enb3ab + Enb3ba
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_pertT_samespin(EC::ECInfo, T1, T2, pct, spin::Symbol)

  Calculate same-spin (T) correction for UCCSD(T) (i.e., ααα or βββ).
  `spin` ∈ (:α,:β)

  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_pertT_samespin(EC::ECInfo, T1, T2, pct, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  o = space4spin('o', spin==:α)
  v = space4spin('v', spin==:α)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  pseudocan_transform!(pct, vv_oo, v*v*o*o; conjugate=true)
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  pseudocan_transform!(pct, vvvo, v*v*v*o)
  # ``0.5(v_{ai}^{kj} - v_{ai}^{jk})``
  vooo = 0.5*ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  pseudocan_transform!(pct, vooo, v*o*o*o)
  m = space4spin('m', spin==:α)
  fov = load2idx(EC,"f_"*m*m)[SP[o],SP[v]]
  pseudocan_transform!(pct, fov, o*v)
  nocc = length(SP[o])
  nvir = length(SP[v])
  ϵo, ϵv = get_pseudo_orbital_energies(pct, spin)

  T = zeros(nvir, nvir, nvir)
  Kijk = zeros(nvir, nvir, nvir)

  Enb3 = 0.0
  IntX = zeros(nvir, nocc)
  IntY = zeros(nvir, nocc)
  for k = 3:nocc 
    for j = 1:k-1
      for i = 1:j-1
        v!T2ij = @mview T2[:,:,i,j]
        v!T2ik = @mview T2[:,:,i,k]
        v!T2jk = @mview T2[:,:,j,k]
        v!T2i = @mview T2[:,:,:,i]
        v!T2j = @mview T2[:,:,:,j]
        v!T2k = @mview T2[:,:,:,k]
        v!vvvk = @mview vvvo[:,:,:,k]
        v!vvvj = @mview vvvo[:,:,:,j]
        v!vvvi = @mview vvvo[:,:,:,i]
        v!vokj = @mview vooo[:,:,k,j]
        v!voki = @mview vooo[:,:,k,i]
        v!voij = @mview vooo[:,:,i,j]
        @mtensor begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = v!T2ij[a,d] * v!vvvk[b,c,d]
          Kijk[a,b,c] += v!T2ik[a,d] * v!vvvj[c,b,d]
          Kijk[a,b,c] += v!T2jk[d,c] * v!vvvi[b,a,d]

          Kijk[a,b,c] -= v!T2i[b,a,l] * v!vokj[c,l]
          Kijk[a,b,c] -= v!T2j[a,b,l] * v!voki[c,l]
          Kijk[a,b,c] -= v!T2k[b,c,l] * v!voij[a,l]
        end
        # antisymmetrize K = A(a,b,c) Kijk[a,b,c]
        @mtensor  T[a,b,c] = Kijk[a,b,c] - Kijk[c,b,a]
        @mtensor Kijk[a,b,c] = T[a,b,c] - T[b,a,c] - T[a,c,b]
        T .= Kijk
        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        for abc ∈ CartesianIndices(T)
          a,b,c = Tuple(abc)
          T[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
        end

        @mtensor Enb3 += 1/6*(Kijk[a,b,c] * T[a,b,c])
        
        v!vv_jk = @mview vv_oo[:,:,j,k]
        v!vv_ik = @mview vv_oo[:,:,i,k]
        v!vv_ij = @mview vv_oo[:,:,i,j]
        v!IntX_i = @mview IntX[:,i]
        v!IntX_j = @mview IntX[:,j]
        v!IntX_k = @mview IntX[:,k]
        v!IntY_i = @mview IntY[:,i]
        v!IntY_j = @mview IntY[:,j]
        v!IntY_k = @mview IntY[:,k]
        @mtensor v!IntX_i[a] += T[a,b,c] * v!vv_jk[b,c]
        @mtensor v!IntX_j[b] += T[a,b,c] * v!vv_ik[a,c]
        @mtensor v!IntX_k[c] += T[a,b,c] * v!vv_ij[a,b]
        @mtensor v!IntY_i[a] += T[a,b,c] * v!T2jk[b,c]
        @mtensor v!IntY_j[b] += T[a,b,c] * v!T2ik[a,c]
        @mtensor v!IntY_k[c] += T[a,b,c] * v!T2ij[a,b]
      end 
    end
  end
  # singles contribution
  @mtensor En3 = T1[a,i] * IntX[a,i]
  # fock contribution
  @mtensor En3 += 0.5 * (fov[i,a] * IntY[a,i])
  En3 += Enb3
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_pertT_mixedspin(EC::ECInfo, T1, T2, T1os, T2mix, pct, spin::Symbol)

  Calculate mixed-spin (T) correction for UCCSD(T) (i.e., ααβ or ββα).

  `spin` ∈ (:α,:β)
  `T1` and `T2` are same-`spin` amplitudes, `T1os` are opposite-`spin` amplitudes,
  and `T2mix` are mixed-spin amplitudes with the *second* electron being `spin`,
  i.e., Tβα for `spin == :α` and Tαβ for `spin == :β`.
  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_pertT_mixedspin(EC::ECInfo, T1, T2, T1os, T2mix, pct, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  isα = (spin == :α)
  o = space4spin('o', isα)
  v = space4spin('v', isα)
  O = space4spin('o', !isα)
  V = space4spin('v', !isα)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  pseudocan_transform!(pct, vv_oo, v*v*o*o; conjugate=true)
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  pseudocan_transform!(pct, vvvo, v*v*v*o)
  # ``v_{ai}^{kj} - v_{ai}^{jk}``
  vooo = ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  pseudocan_transform!(pct, vooo, v*o*o*o)
  if isα
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, o*O*v*V),[3,4,1,2])
    pseudocan_transform!(pct, vV_oO, v*V*o*O; conjugate=true)
    # ``v_{aB}^{cK}``
    vVvO = ints2(EC, v*V*v*O)
    pseudocan_transform!(pct, vVvO, v*V*v*O)
    # ``v_{Ab}^{Ck}``
    VvVo = permutedims(ints2(EC, v*V*o*V),[2,1,4,3])
    pseudocan_transform!(pct, VvVo, V*v*V*o)
    # ``v_{aI}^{kJ}``
    vOoO = ints2(EC, v*O*o*O)
    pseudocan_transform!(pct, vOoO, v*O*o*O)
    # ``v_{Ai}^{Kj}``
    VoOo = permutedims(ints2(EC, o*V*o*O),[2,1,4,3])
    pseudocan_transform!(pct, VoOo, V*o*O*o)
  else
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, O*o*V*v),[4,3,2,1])
    pseudocan_transform!(pct, vV_oO, v*V*o*O; conjugate=true)
    # ``v_{aB}^{cK}``
    vVvO = permutedims(ints2(EC, V*v*O*v),[2,1,4,3])
    pseudocan_transform!(pct, vVvO, v*V*v*O)
    # ``v_{Ab}^{Ck}``
    VvVo = ints2(EC, V*v*V*o)
    pseudocan_transform!(pct, VvVo, V*v*V*o)
    # ``v_{aI}^{kJ}``
    vOoO = permutedims(ints2(EC, O*v*O*o),[2,1,4,3])
    pseudocan_transform!(pct, vOoO, v*O*o*O)
    # ``v_{Ai}^{Kj}``
    VoOo = ints2(EC, V*o*O*o)
    pseudocan_transform!(pct, VoOo, V*o*O*o)
  end
  m = space4spin('m', isα)
  fov = load2idx(EC,"f_"*m*m)[SP[o],SP[v]]
  pseudocan_transform!(pct, fov, o*v)
  M = space4spin('m', !isα)
  fOV = load2idx(EC,"f_"*M*M)[SP[O],SP[V]]
  pseudocan_transform!(pct, fOV, O*V)

  nocc = length(SP[o])
  nOcc = length(SP[O])
  nvir = length(SP[v])
  nVir = length(SP[V])
  ϵo, ϵv = get_pseudo_orbital_energies(pct, spin)
  opspin = isα ? :β : :α
  ϵO, ϵV = get_pseudo_orbital_energies(pct, opspin)

  T = zeros(nvir, nvir, nVir)
  Kijk = zeros(nvir, nvir, nVir)

  Enb3 = 0.0
  IntX = zeros(nvir, nocc)
  IntY = zeros(nvir, nocc)
  IntXos = zeros(nVir, nOcc)
  IntYos = zeros(nVir, nOcc)
  for K = 1:nOcc 
    T2K = T2mix[:,:,K,:]
    for j = 2:nocc
      for i = 1:j-1
        v!T2ij = @mview T2[:,:,i,j]
        v!T2Ki = @mview T2K[:,:,i]
        v!T2Kj = @mview T2K[:,:,j]
        v!T2i = @mview T2[:,:,:,i]
        v!T2j = @mview T2[:,:,:,j]
        v!T2mixi = @mview T2mix[:,:,:,i]
        v!T2mixj = @mview T2mix[:,:,:,j]
        v!vVvK = @mview vVvO[:,:,:,K]
        v!vvvi = @mview vvvo[:,:,:,i]
        v!vvvj = @mview vvvo[:,:,:,j]
        v!VvVi = @mview VvVo[:,:,:,i]
        v!VvVj = @mview VvVo[:,:,:,j]
        v!voij = @mview vooo[:,:,i,j]
        v!vOjK = @mview vOoO[:,:,j,K]
        v!vOiK = @mview vOoO[:,:,i,K]
        v!VoKj = @mview VoOo[:,:,K,j]
        v!VoKi = @mview VoOo[:,:,K,i]
        @mtensor begin
          # K_{abC}^{ijK} = v_{bC}^{dK} T^{ij}_{ad} + ...
          Kijk[a,b,C] = v!T2ij[a,d] * v!vVvK[b,C,d]
          Kijk[a,b,C] += v!T2Kj[C,d] * v!vvvi[b,a,d]
          Kijk[a,b,C] += v!T2Ki[C,d] * v!vvvj[a,b,d]
          Kijk[a,b,C] += v!T2Kj[D,b] * v!VvVi[C,a,D]
          Kijk[a,b,C] += v!T2Ki[D,a] * v!VvVj[C,b,D]
          Kijk[a,b,C] -= T2K[C,b,l] * v!voij[a,l]
          Kijk[a,b,C] -= v!T2mixi[C,a,L] * v!vOjK[b,L]
          Kijk[a,b,C] -= v!T2mixj[C,b,L] * v!vOiK[a,L]
        end
        # antisymmetrize ΔK = A(a,b) ΔKijk[a,b,C]
        Kijk -= permutedims(Kijk,[2,1,3])
        @mtensor begin
          Kijk[a,b,C] -= v!T2i[b,a,l] * v!VoKj[C,l]
          Kijk[a,b,C] -= v!T2j[a,b,l] * v!VoKi[C,l]
        end
        T .= Kijk
        ϵoijK = ϵo[i] + ϵo[j] + ϵO[K]
        for abC ∈ CartesianIndices(T)
          a,b,C = Tuple(abC)
          T[abC] /= ϵoijK - ϵv[a] - ϵv[b] - ϵV[C]
        end

        @mtensor Enb3 += 0.5 * (Kijk[a,b,C] * T[a,b,C])
        
        v!vV_jK = @mview vV_oO[:,:,j,K]
        v!vV_iK = @mview vV_oO[:,:,i,K]
        v!vv_ij = @mview vv_oo[:,:,i,j]
        v!IntX_i   = @mview IntX[:,i]
        v!IntX_j   = @mview IntX[:,j]
        v!IntXos_K = @mview IntXos[:,K]
        v!IntY_i   = @mview IntY[:,i]
        v!IntY_j   = @mview IntY[:,j]
        v!IntYos_K = @mview IntYos[:,K]
        @mtensor v!IntX_i[a] += T[a,b,C] * v!vV_jK[b,C]
        @mtensor v!IntX_j[b] += T[a,b,C] * v!vV_iK[a,C]
        @mtensor v!IntXos_K[C] += T[a,b,C] * v!vv_ij[a,b]
        @mtensor v!IntY_i[a] += T[a,b,C] * v!T2Kj[C,b]
        @mtensor v!IntY_j[b] += T[a,b,C] * v!T2Ki[C,a]
        @mtensor v!IntYos_K[C] += T[a,b,C] * v!T2ij[a,b]
      end 
    end
  end
  # singles contribution
  @mtensor En3 = T1[a,i] * IntX[a,i]
  @mtensor En3 += T1os[A,I] * IntXos[A,I]
  # fock contribution
  @mtensor En3 += fov[i,a] * IntY[a,i]
  @mtensor En3 += 0.5 * (fOV[I,A] * IntYos[A,I])
  En3 += Enb3
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_ΛpertT_samespin(EC::ECInfo, T2, U1, U2, pct, spin::Symbol)

  Calculate same-spin (T) correction for ΛUCCSD(T) (i.e., ααα or βββ).
  `spin` ∈ (:α,:β)

  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_ΛpertT_samespin(EC::ECInfo, T2, U1, U2, pct, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  o = space4spin('o', spin==:α)
  v = space4spin('v', spin==:α)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  pseudocan_transform!(pct, vv_oo, v*v*o*o; conjugate=true)
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  pseudocan_transform!(pct, vvvo, v*v*v*o)
  # ``0.5(v_{ai}^{kj} - v_{ai}^{jk})``
  vooo = 0.5*ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  pseudocan_transform!(pct, vooo, v*o*o*o)
  # ``v_{ck}^{ab}``, reordered to ``v^{ab}_{ck}``
  vv_vo = permutedims(ints2(EC, v*o*v*v), [3,4,1,2])
  pseudocan_transform!(pct, vv_vo, v*v*v*o; conjugate=true)
  # ``0.5(v_{kj}^{ai} - v_{jk}^{ai})``, reordered to ``\bar v^{ai}_{kj}``
  vo_oo = 0.5*permutedims(ints2(EC, o*o*v*o), [3,4,1,2])
  vo_oo -= permutedims(vo_oo,[1,2,4,3])
  pseudocan_transform!(pct, vo_oo, v*o*o*o; conjugate=true)
  m = space4spin('m', spin==:α)
  fov = load2idx(EC,"f_"*m*m)[SP[o],SP[v]]
  pseudocan_transform!(pct, fov, o*v)
  nocc = length(SP[o])
  nvir = length(SP[v])
  ϵo, ϵv = get_pseudo_orbital_energies(pct, spin)

  T = zeros(nvir, nvir, nvir)
  Kijk = zeros(nvir, nvir, nvir)

  Enb3 = 0.0
  IntX = zeros(nvir, nocc)
  IntY = zeros(nvir, nocc)
  for k = 3:nocc 
    for j = 1:k-1
      for i = 1:j-1
        v!T2ij = @mview T2[:,:,i,j]
        v!T2ik = @mview T2[:,:,i,k]
        v!T2jk = @mview T2[:,:,j,k]
        v!T2i = @mview T2[:,:,:,i]
        v!T2j = @mview T2[:,:,:,j]
        v!T2k = @mview T2[:,:,:,k]
        v!vvvk = @mview vvvo[:,:,:,k]
        v!vvvj = @mview vvvo[:,:,:,j]
        v!vvvi = @mview vvvo[:,:,:,i]
        v!vokj = @mview vooo[:,:,k,j]
        v!voki = @mview vooo[:,:,k,i]
        v!voij = @mview vooo[:,:,i,j]
        @mtensor begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = v!T2ij[a,d] * v!vvvk[b,c,d]
          Kijk[a,b,c] += v!T2ik[a,d] * v!vvvj[c,b,d]
          Kijk[a,b,c] += v!T2jk[d,c] * v!vvvi[b,a,d]

          Kijk[a,b,c] -= v!T2i[b,a,l] * v!vokj[c,l]
          Kijk[a,b,c] -= v!T2j[a,b,l] * v!voki[c,l]
          Kijk[a,b,c] -= v!T2k[b,c,l] * v!voij[a,l]
        end
        # antisymmetrize K = A(a,b,c) Kijk[a,b,c]
        @mtensor  T[a,b,c] = Kijk[a,b,c] - Kijk[c,b,a]
        @mtensor Kijk[a,b,c] = T[a,b,c] - T[b,a,c] - T[a,c,b]
        T .= Kijk
        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        for abc ∈ CartesianIndices(T)
          a,b,c = Tuple(abc)
          T[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
        end

        v!U2ij = @mview U2[:,:,i,j]
        v!U2ik = @mview U2[:,:,i,k]
        v!U2jk = @mview U2[:,:,j,k]
        v!U2i = @mview U2[:,:,:,i]
        v!U2j = @mview U2[:,:,:,j]
        v!U2k = @mview U2[:,:,:,k]
        v!vv_vk = @mview vv_vo[:,:,:,k]
        v!vv_vj = @mview vv_vo[:,:,:,j]
        v!vv_vi = @mview vv_vo[:,:,:,i]
        v!vo_kj = @mview vo_oo[:,:,k,j]
        v!vo_ki = @mview vo_oo[:,:,k,i]
        v!vo_ij = @mview vo_oo[:,:,i,j]
        @mtensor begin
          # K^{abc}_{ijk} = v_{dk}^{bc} Λ_{ij}^{ad} + ...
          Kijk[a,b,c] := v!U2ij[a,d] * v!vv_vk[b,c,d]
          Kijk[a,b,c] += v!U2ik[a,d] * v!vv_vj[c,b,d]
          Kijk[a,b,c] += v!U2jk[d,c] * v!vv_vi[b,a,d]

          Kijk[a,b,c] -= v!U2i[b,a,l] * v!vo_kj[c,l]
          Kijk[a,b,c] -= v!U2j[a,b,l] * v!vo_ki[c,l]
          Kijk[a,b,c] -= v!U2k[b,c,l] * v!vo_ij[a,l]
        end
        # antisymmetrize K = A(a,b,c) Kijk[a,b,c]
        @mtensor  X[a,b,c] := Kijk[a,b,c] - Kijk[c,b,a]
        @mtensor Kijk[a,b,c] = X[a,b,c] - X[b,a,c] - X[a,c,b]

        @mtensor Enb3 += 1/6*(Kijk[a,b,c] * T[a,b,c])
        
        v!vv_jk = @mview vv_oo[:,:,j,k]
        v!vv_ik = @mview vv_oo[:,:,i,k]
        v!vv_ij = @mview vv_oo[:,:,i,j]
        v!IntX_i = @mview IntX[:,i]
        v!IntX_j = @mview IntX[:,j]
        v!IntX_k = @mview IntX[:,k]
        v!IntY_i = @mview IntY[:,i]
        v!IntY_j = @mview IntY[:,j]
        v!IntY_k = @mview IntY[:,k]
        @mtensor v!IntX_i[a] += T[a,b,c] * v!vv_jk[b,c]
        @mtensor v!IntX_j[b] += T[a,b,c] * v!vv_ik[a,c]
        @mtensor v!IntX_k[c] += T[a,b,c] * v!vv_ij[a,b]
        @mtensor v!IntY_i[a] += T[a,b,c] * v!U2jk[b,c]
        @mtensor v!IntY_j[b] += T[a,b,c] * v!U2ik[a,c]
        @mtensor v!IntY_k[c] += T[a,b,c] * v!U2ij[a,b]
      end 
    end
  end
  # singles contribution
  @mtensor En3 = U1[a,i] * IntX[a,i]
  # fock contribution
  @mtensor En3 += 0.5 * (fov[i,a] * IntY[a,i])
  En3 += Enb3
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end

"""
    calc_ΛpertT_mixedspin(EC::ECInfo, T2, T2mix, U1, U2, U1os, U2mix, pct, spin::Symbol)

  Calculate mixed-spin (T) correction for ΛUCCSD(T) (i.e., ααβ or ββα).

  `spin` ∈ (:α,:β)
  `U1` and `U2`/`T2` are same-`spin` Lagrange multipliers/amplitudes,
  `U1os` are opposite-`spin` Lagrange multipliers,
  and `U2mix`/`T2mix` are mixed-spin Lagrange multipliers/amplitudes 
  with the *second* electron being `spin`,
  i.e., Tβα for `spin == :α` and Tαβ for `spin == :β`.
  Return ( `"ET3"`=(T)-energy, `"ET3b"`=[T]-energy)) `OutDict`.
"""
function calc_ΛpertT_mixedspin(EC::ECInfo, T2, T2mix, U1, U2, U1os, U2mix, pct, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  isα = (spin == :α)
  o = space4spin('o', isα)
  v = space4spin('v', isα)
  O = space4spin('o', !isα)
  V = space4spin('v', !isα)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  pseudocan_transform!(pct, vv_oo, v*v*o*o; conjugate=true)
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  pseudocan_transform!(pct, vvvo, v*v*v*o)
  # ``v_{ai}^{kj} - v_{ai}^{jk}``
  vooo = ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  pseudocan_transform!(pct, vooo, v*o*o*o)
  # ``v_{ck}^{ab}``, reordered to ``v^{ab}_{ck}``
  vv_vo = permutedims(ints2(EC, v*o*v*v), [3,4,1,2])
  pseudocan_transform!(pct, vv_vo, v*v*v*o; conjugate=true)
  # ``v_{kj}^{ai} - v_{jk}^{ai}``, reordered to ``\bar v^{ai}_{kj}``
  vo_oo = permutedims(ints2(EC, o*o*v*o), [3,4,1,2])
  vo_oo -= permutedims(vo_oo,[1,2,4,3])
  pseudocan_transform!(pct, vo_oo, v*o*o*o; conjugate=true)
  if isα
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, o*O*v*V),[3,4,1,2])
    pseudocan_transform!(pct, vV_oO, v*V*o*O; conjugate=true)
    # ``v_{aB}^{cK}``
    vVvO = ints2(EC, v*V*v*O)
    pseudocan_transform!(pct, vVvO, v*V*v*O)
    # ``v_{Ab}^{Ck}``
    VvVo = permutedims(ints2(EC, v*V*o*V),[2,1,4,3])
    pseudocan_transform!(pct, VvVo, V*v*V*o)
    # ``v_{aI}^{kJ}``
    vOoO = ints2(EC, v*O*o*O)
    pseudocan_transform!(pct, vOoO, v*O*o*O)
    # ``v_{Ai}^{Kj}``
    VoOo = permutedims(ints2(EC, o*V*o*O),[2,1,4,3])
    pseudocan_transform!(pct, VoOo, V*o*O*o)
    # ``v_{cK}^{aB}``, reordered to ``v^{aB}_{cK}``
    vV_vO = permutedims(ints2(EC, v*O*v*V), [3,4,1,2])
    pseudocan_transform!(pct, vV_vO, v*V*v*O; conjugate=true)
    # ``v_{Ck}^{Ab}``, reordered to ``v^{Ab}_{Ck}``
    Vv_Vo = permutedims(ints2(EC, o*V*v*V),[4,3,2,1])
    pseudocan_transform!(pct, Vv_Vo, V*v*V*o; conjugate=true)
    # ``v_{kJ}^{aI}``, reordered to ``v^{aI}_{kJ}``
    vO_oO = permutedims(ints2(EC, o*O*v*O), [3,4,1,2])
    pseudocan_transform!(pct, vO_oO, v*O*o*O; conjugate=true)
    # ``v_{Kj}^{Ai}``, reordered to ``v^{Ai}_{Kj}``
    Vo_Oo = permutedims(ints2(EC, o*O*o*V),[4,3,2,1])
    pseudocan_transform!(pct, Vo_Oo, V*o*O*o; conjugate=true)
  else
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, O*o*V*v),[4,3,2,1])
    pseudocan_transform!(pct, vV_oO, v*V*o*O; conjugate=true)
    # ``v_{aB}^{cK}``
    vVvO = permutedims(ints2(EC, V*v*O*v),[2,1,4,3])
    pseudocan_transform!(pct, vVvO, v*V*v*O)
    # ``v_{Ab}^{Ck}``
    VvVo = ints2(EC, V*v*V*o)
    pseudocan_transform!(pct, VvVo, V*v*V*o)
    # ``v_{aI}^{kJ}``
    vOoO = permutedims(ints2(EC, O*v*O*o),[2,1,4,3])
    pseudocan_transform!(pct, vOoO, v*O*o*O)
    # ``v_{Ai}^{Kj}``
    VoOo = ints2(EC, V*o*O*o)
    pseudocan_transform!(pct, VoOo, V*o*O*o)
    # ``v_{cK}^{aB}``, reordered to ``v^{aB}_{cK}``
    vV_vO = permutedims(ints2(EC, O*v*V*v),[4,3,2,1])
    pseudocan_transform!(pct, vV_vO, v*V*v*O; conjugate=true)
    # ``v_{Ck}^{Ab}``, reordered to ``v^{Ab}_{Ck}``
    Vv_Vo = permutedims(ints2(EC, V*o*V*v),[3,4,1,2])
    pseudocan_transform!(pct, Vv_Vo, V*v*V*o; conjugate=true)
    # ``v_{kJ}^{aI}``, reordered to ``v^{aI}_{kJ}``
    vO_oO = permutedims(ints2(EC, O*o*O*v),[4,3,2,1])
    pseudocan_transform!(pct, vO_oO, v*O*o*O; conjugate=true)
    # ``v_{Kj}^{Ai}``, reordered to ``v^{Ai}_{Kj}``
    Vo_Oo = permutedims(ints2(EC, O*o*V*o),[3,4,1,2])
    pseudocan_transform!(pct, Vo_Oo, V*o*O*o; conjugate=true)
  end
  m = space4spin('m', isα)
  fov = load2idx(EC,"f_"*m*m)[SP[o],SP[v]]
  pseudocan_transform!(pct, fov, o*v)
  M = space4spin('m', !isα)
  fOV = load2idx(EC,"f_"*M*M)[SP[O],SP[V]]
  pseudocan_transform!(pct, fOV, O*V)

  nocc = length(SP[o])
  nOcc = length(SP[O])
  nvir = length(SP[v])
  nVir = length(SP[V])
  ϵo, ϵv = get_pseudo_orbital_energies(pct, spin)
  opspin = isα ? :β : :α
  ϵO, ϵV = get_pseudo_orbital_energies(pct, opspin)

  T = zeros(nvir, nvir, nVir)
  Kijk = zeros(nvir, nvir, nVir)

  Enb3 = 0.0
  IntX = zeros(nvir, nocc)
  IntY = zeros(nvir, nocc)
  IntXos = zeros(nVir, nOcc)
  IntYos = zeros(nVir, nOcc)
  for K = 1:nOcc 
    T2K = T2mix[:,:,K,:]
    U2K = U2mix[:,:,K,:]
    for j = 2:nocc
      for i = 1:j-1
        v!T2ij = @mview T2[:,:,i,j]
        v!T2Ki = @mview T2K[:,:,i]
        v!T2Kj = @mview T2K[:,:,j]
        v!T2i = @mview T2[:,:,:,i]
        v!T2j = @mview T2[:,:,:,j]
        v!T2mixi = @mview T2mix[:,:,:,i]
        v!T2mixj = @mview T2mix[:,:,:,j]
        v!vVvK = @mview vVvO[:,:,:,K]
        v!vvvi = @mview vvvo[:,:,:,i]
        v!vvvj = @mview vvvo[:,:,:,j]
        v!VvVi = @mview VvVo[:,:,:,i]
        v!VvVj = @mview VvVo[:,:,:,j]
        v!voij = @mview vooo[:,:,i,j]
        v!vOjK = @mview vOoO[:,:,j,K]
        v!vOiK = @mview vOoO[:,:,i,K]
        v!VoKj = @mview VoOo[:,:,K,j]
        v!VoKi = @mview VoOo[:,:,K,i]
        @mtensor begin
          # K_{abC}^{ijK} = v_{bC}^{dK} T^{ij}_{ad} + ...
          Kijk[a,b,C] = v!T2ij[a,d] * v!vVvK[b,C,d]
          Kijk[a,b,C] += v!T2Kj[C,d] * v!vvvi[b,a,d]
          Kijk[a,b,C] += v!T2Ki[C,d] * v!vvvj[a,b,d]
          Kijk[a,b,C] += v!T2Kj[D,b] * v!VvVi[C,a,D]
          Kijk[a,b,C] += v!T2Ki[D,a] * v!VvVj[C,b,D]
          Kijk[a,b,C] -= T2K[C,b,l] * v!voij[a,l]
          Kijk[a,b,C] -= v!T2mixi[C,a,L] * v!vOjK[b,L]
          Kijk[a,b,C] -= v!T2mixj[C,b,L] * v!vOiK[a,L]
        end
        # antisymmetrize ΔK = A(a,b) ΔKijk[a,b,C]
        Kijk -= permutedims(Kijk,[2,1,3])
        @mtensor begin
          Kijk[a,b,C] -= v!T2i[b,a,l] * v!VoKj[C,l]
          Kijk[a,b,C] -= v!T2j[a,b,l] * v!VoKi[C,l]
        end
        T .= Kijk
        ϵoijK = ϵo[i] + ϵo[j] + ϵO[K]
        for abC ∈ CartesianIndices(T)
          a,b,C = Tuple(abC)
          T[abC] /= ϵoijK - ϵv[a] - ϵv[b] - ϵV[C]
        end

        v!U2ij = @mview U2[:,:,i,j]
        v!U2Ki = @mview U2K[:,:,i]
        v!U2Kj = @mview U2K[:,:,j]
        v!U2i = @mview U2[:,:,:,i]
        v!U2j = @mview U2[:,:,:,j]
        v!U2mixi = @mview U2mix[:,:,:,i]
        v!U2mixj = @mview U2mix[:,:,:,j]
        v!vV_vK = @mview vV_vO[:,:,:,K]
        v!vv_vi = @mview vv_vo[:,:,:,i]
        v!vv_vj = @mview vv_vo[:,:,:,j]
        v!Vv_Vi = @mview Vv_Vo[:,:,:,i]
        v!Vv_Vj = @mview Vv_Vo[:,:,:,j]
        v!vo_ij = @mview vo_oo[:,:,i,j]
        v!vO_jK = @mview vO_oO[:,:,j,K]
        v!vO_iK = @mview vO_oO[:,:,i,K]
        v!Vo_Kj = @mview Vo_Oo[:,:,K,j]
        v!Vo_Ki = @mview Vo_Oo[:,:,K,i]
        @mtensor begin
          # K^{abC}_{ijK} = v_{dK}^{bC} Λ_{ij}^{ad} + ...
          Kijk[a,b,C] = v!U2ij[a,d] * v!vV_vK[b,C,d]
          Kijk[a,b,C] += v!U2Kj[C,d] * v!vv_vi[b,a,d]
          Kijk[a,b,C] += v!U2Ki[C,d] * v!vv_vj[a,b,d]
          Kijk[a,b,C] += v!U2Kj[D,b] * v!Vv_Vi[C,a,D]
          Kijk[a,b,C] += v!U2Ki[D,a] * v!Vv_Vj[C,b,D]
          Kijk[a,b,C] -= U2K[C,b,l] * v!vo_ij[a,l]
          Kijk[a,b,C] -= v!U2mixi[C,a,L] * v!vO_jK[b,L]
          Kijk[a,b,C] -= v!U2mixj[C,b,L] * v!vO_iK[a,L]
        end
        # antisymmetrize ΔK = A(a,b) ΔKijk[a,b,C]
        Kijk -= permutedims(Kijk,[2,1,3])
        @mtensor begin
          Kijk[a,b,C] -= v!U2i[b,a,l] * v!Vo_Kj[C,l]
          Kijk[a,b,C] -= v!U2j[a,b,l] * v!Vo_Ki[C,l]
        end

        @mtensor Enb3 += 0.5 * (Kijk[a,b,C] * T[a,b,C])
        
        v!vV_jK = @mview vV_oO[:,:,j,K]
        v!vV_iK = @mview vV_oO[:,:,i,K]
        v!vv_ij = @mview vv_oo[:,:,i,j]
        v!IntX_i = @mview IntX[:,i]
        v!IntX_j = @mview IntX[:,j]
        v!IntXos_K = @mview IntXos[:,K]
        v!IntY_i = @mview IntY[:,i]
        v!IntY_j = @mview IntY[:,j]
        v!IntYos_K = @mview IntYos[:,K]
        @mtensor v!IntX_i[a] += T[a,b,C] * v!vV_jK[b,C]
        @mtensor v!IntX_j[b] += T[a,b,C] * v!vV_iK[a,C]
        @mtensor v!IntXos_K[C] += T[a,b,C] * v!vv_ij[a,b]
        @mtensor v!IntY_i[a] += T[a,b,C] * v!U2Kj[C,b]
        @mtensor v!IntY_j[b] += T[a,b,C] * v!U2Ki[C,a]
        @mtensor v!IntYos_K[C] += T[a,b,C] * v!U2ij[a,b]
      end 
    end
  end
  # singles contribution
  @mtensor En3 = U1[a,i] * IntX[a,i]
  @mtensor En3 += U1os[A,I] * IntXos[A,I]
  # fock contribution
  @mtensor En3 += fov[i,a] * IntY[a,i]
  @mtensor En3 += 0.5 * (fOV[I,A] * IntYos[A,I])
  En3 += Enb3
  return OutDict("ET3"=>En3, "ET3b"=>Enb3)
end
