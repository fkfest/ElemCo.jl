# fci_ops.jl - Translation of FciOps.h and FciOps.cpp

"""
FCI Hamiltonian operators implementation.
"""

"""
    SubstResult

Result of string substitution operation c^k c_l.
"""
mutable struct SubstResult
  k::FCIUInt      # creation orbital
  l::FCIUInt      # annihilation orbital
  sign::Int8      # fermionic sign
  str::OrbPattern    # resulting orbital pattern
  i_str::Address     # address of resulting string

  SubstResult() = new(0, 0, 0, 0, 0)
  SubstResult(k, l, sign, str, i_str) = new(k, l, sign, str, i_str)
end

"""
    StrInfo

Information about string substitutions for a block of strings.
"""
mutable struct StrInfo
  n_subst::FCIUInt
  subst::Vector{SubstResult}

  StrInfo() = new(0, SubstResult[])
  StrInfo(max_subst::Integer) = new(0, [SubstResult() for _ in 1:max_subst])
end

# Display functions for debugging
function Base.show(io::IO, s::SubstResult)
  print(io, "-> c^$(s.k) c_$(s.l) -> $(s.sign)$(fmt_pat(s.str, 9))")
end

"""
    occupied_orbitals!(orbs, pattern::OrbPattern, n_orb::Int)

Get list of occupied orbital indices and store in `orbs`.
"""
function occupied_orbitals!(orbs, pattern::OrbPattern, n_orb::Int)
  empty!(orbs)
  @inbounds @simd for i in 1:n_orb
    if (pattern >>> (i-1)) & one(pattern) != zero(pattern)
      push!(orbs, i)
    end
  end
  return orbs
end

"""
    virtual_orbitals!(orbs, pattern::OrbPattern, n_orb::Int)

Get list of virtual (unoccupied) orbital indices and store in `orbs`.
"""
function virtual_orbitals!(orbs, pattern::OrbPattern, n_orb::Int)
  empty!(orbs)
  @inbounds @simd for i in 1:n_orb
    if (pattern >>> (i-1)) & one(pattern) == zero(pattern)
      push!(orbs, i)
    end
  end
  return orbs
end

"""
    occupied_and_virtual_orbitals!(occ, virt, pattern::OrbPattern, n_orb::Int)

Get lists of occupied and virtual orbital indices and store in `occ` and `virt`.
"""
function occupied_and_virtual_orbitals!(occ, virt, pattern::OrbPattern, n_orb::Int)
  empty!(occ)
  empty!(virt)
  @inbounds @simd for i in 1:n_orb
    if (pattern >>> (i-1)) & one(pattern) != zero(pattern)
      push!(occ, i)
    else
      push!(virt, i)
    end
  end
  return
end

"""
    form_string_substs_for_spin!(result::Vector{SubstResult}, 
                                op_matrix_1e, adr::OrbStringAdrTable,
                                I::OrbPattern) -> Int

Form sparse list of all |K> which can be reached by applying c^k_l on string |I>.

Returns number of valid substitutions found.
"""
function form_string_substs_for_spin!(result::Vector{SubstResult}, op_matrix_1e,
                                      adr::OrbStringAdrTable, I::OrbPattern)::Int
  n_orb_val = n_orb(adr)
  n_entries = 0

  # Ensure result vector is large enough
  if length(result) < n_orb_val * n_orb_val
    resize!(result, n_orb_val * n_orb_val)
  end

  for l in 0:(n_orb_val - 1)
    mask_l = OrbPattern(1) << l
    if (I & mask_l) == 0
      continue  # c_l annihilates |I>
    end

    J = I & ~mask_l
    sign1 = string_parity_before_pos(J, l)

    for k in 0:(n_orb_val - 1)
      mask_k = OrbPattern(1) << k
      K = J | mask_k

      # Skip if c^k annihilates c_l|I> or matrix element is zero
      if (J & mask_k) != 0
        continue
      end

      if op_matrix_1e !== nothing && abs(op_matrix_1e[k + 1, l + 1]) < ThrNeglect
        continue
      end

      n_entries += 1
      r = result[n_entries]
      r.k = FCIUInt(k + 1)  # Store 1-based orbital index
      r.l = FCIUInt(l + 1)  # Store 1-based orbital index
      combined_parity = (sign1 ⊻ string_parity_before_pos(K, k)) & 1  # Only 0 or 1
      r.sign = Int8(combined_parity == 0 ? 1 : -1)  # Convert parity to sign explicitly
      r.str = K
      r.i_str = adr(K)  # adr() now returns 1-based address
    end
  end

  return n_entries
end

"""
    apply_cop1(input_pattern::OrbPattern, iorb::Integer, create_or_destroy::Integer) -> Tuple{OrbPattern, Int8}

Apply orbital creation (+1) or destruction (-1) operator on a determinant string.
Returns (output_pattern, sign) where:
- sign = +1: operation has positive parity
- sign = -1: operation has negative parity  
- sign = 0: operation annihilates input
"""
function apply_cop1(input_pattern::OrbPattern, iorb::Integer, create_or_destroy::Integer)::Tuple{OrbPattern, Int8}
  mask_l = OrbPattern(1) << iorb

  if create_or_destroy == -1  # destruction
    if (input_pattern & mask_l) == 0
      return (OrbPattern(0), Int8(0))  # c_l annihilates |I>
    end
    output_pattern = input_pattern & ~mask_l
  else  # creation
    if (input_pattern & mask_l) != 0
      return (OrbPattern(0), Int8(0))  # c^l annihilates |I>
    end
    output_pattern = input_pattern | mask_l
  end

  parity = string_parity_before_pos(output_pattern, iorb) & 1  # Only 0 or 1
  sign = Int8(parity == 0 ? 1 : -1)  # Convert parity to sign explicitly

  return (output_pattern, sign)
end

"""
    block_contract_cc1!(data_k::AbstractArray{Scalar, 3}, info1::Vector{StrInfo},
                        coeffs::AbstractMatrix{Scalar},
                        direction::Char, c_sum::Ref{Scalar}, prefactor::Scalar)

Block contraction for c^k c_l operators.
Direction: 'c' for contraction, 'R' for residual formation.
"""
function block_contract_cc1!(data_k::AbstractArray{Scalar, 3}, info1::Vector{StrInfo},
                             coeffs::AbstractMatrix{Scalar}, direction::Char,
                             c_sum::Ref{Scalar}, prefactor::Scalar)
  @assert direction == 'c' || direction == 'R'
  n_pair, n_blk1, n_blk2 = size(data_k)
  for i_blk1 in 1:n_blk1
    info1_data = info1[i_blk1]

    # Loop over k,l substitutions
    for i_subst in 1:Int(info1_data.n_subst)
      s = info1_data.subst[i_subst]
      pf = s.sign * prefactor

      # Symmetric packing: k >= l (convert to 0-based for packing calculation)
      k_ = max(Int(s.k) - 1, Int(s.l) - 1)  # s.k and s.l are 1-based, convert to 0-based
      l_ = min(Int(s.k) - 1, Int(s.l) - 1)  # for symmetric packing formula
      kl = (k_ * (k_ + 1)) ÷ 2 + l_ + 1

      if direction == 'c'
        for i_blk2 in 1:n_blk2
          t = pf * coeffs[Int(s.i_str), i_blk2]
          data_k[kl, i_blk1, i_blk2] += t
          c_sum[] += t * t
        end
      else
        for i_blk2 in 1:n_blk2
          coeffs[Int(s.i_str), i_blk2] += pf * data_k[kl, i_blk1, i_blk2]
        end
      end
    end
  end
end

"""
    block_contract_cc1_nosym!(data_k, info_1, coeffs, c_sum, prefactor)

Helper function for 2-RDM calculation. Performs block contraction for non-symmetric pairs.

Forms: `data_k[k,l,iBlk1,iBlk2] += <K_1|c†_k c_l|J_1> * coeffs[J_1,K_2] * sign * prefactor`

# Arguments
- `data_k`: Output array [n_orb, n_orb, n_blk1, n_blk2] 
- `info_1`: Array of StrInfo for spin 1 substitutions
- `coeffs`: CI coefficients as AbstractMatrix `[n_str_1, n_blk_2]` (can be StridedView)
- `c_sum`: Accumulator for sum of contributions (for screening)
- `prefactor`: Multiplicative prefactor (typically ±1)
"""
function block_contract_cc1_nosym!(data_k::AbstractArray{Scalar, 4}, info_1::AbstractVector{StrInfo},
                                   coeffs::AbstractMatrix{Scalar},
                                   c_sum::Ref{Scalar}, prefactor::Scalar)
  n_orb, n_orb, n_blk1, n_blk2 = size(data_k)
  for i_blk1 in 1:n_blk1
    info = info_1[i_blk1]
    n_subst = Int(info.n_subst)
    
    # Loop over k,l substitutions
    for i_subst in 1:n_subst
      s = info.subst[i_subst]
      pf = s.sign * prefactor
     
      k = Int(s.k)
      l = Int(s.l)
      
      for i_blk2 in 1:n_blk2
        t = pf * coeffs[Int(s.i_str), i_blk2]
        data_k[k, l, i_blk1, i_blk2] += t
        c_sum[] += t * t
      end
    end
  end
end


"""
    symmetrize_ci_vector!(x::FCIVector)

Symmetrize CI vector for systems with equal alpha and beta electrons.
"""
function symmetrize_ci_vector!(x::FCIVector)
  @assert x.n_elec_a == x.n_elec_b "Can only symmetrize vectors with equal alpha/beta electrons"

  pf = 1.0

  # Symmetrize: x[i_a, i_b] = pf * x[i_b, i_a]  
  n_a = Int(x.n_str_a)
  n_b = Int(x.n_str_b)
  for i_a in 1:n_a
    for i_b in 1:min(i_a - 1, n_b)
      avg = 0.5 * (x[i_a, i_b] + pf * x[i_b, i_a])
      x[i_a, i_b] = avg
      x[i_b, i_a] = pf * avg
    end
  end
end

"""
    pair_index(i::Int, j::Int) -> Int

Get the pair index for a given pair of orbital indices.
"""
@inline function pair_index(i::Int, j::Int)
  ii = i
  jj = j
  if ii < jj
    ii, jj = jj, ii
  end
  ii0 = ii - 1
  jj0 = jj - 1
  return (ii0 * (ii0 + 1)) ÷ 2 + jj0 + 1
end

"""
    convert_op2_to_pair_matrix(op::AbstractArray{Scalar}, n_orb::Int)

Convert 4D integral tensor to pair matrix format.
"""
function convert_op2_to_pair_matrix(op::AbstractArray{Scalar}, n_orb::Int)
  n_pairs = n_orb * (n_orb + 1) ÷ 2
  mat = zeros(Scalar, n_pairs, n_pairs)
  if ndims(op) == 4
    @inbounds for i in 1:n_orb
      for j in 1:n_orb
        ij = pair_index(i, j)
        for k in 1:n_orb
          for l in 1:n_orb
            kl = pair_index(k, l)
            mat[ij, kl] = op[i, j, k, l]
          end
        end
      end
    end
  elseif ndims(op) == 2
    @assert size(op, 1) == n_pairs && size(op, 2) == n_pairs "Integral matrix size mismatch"
    mat .= op
  else
    error("Unsupported integral tensor dimensions: $(ndims(op))")
  end
  return mat
end

"""
    convert_op2_to_pair_matrix_same_spin(op::AbstractArray{Scalar}, n_orb::Int)

Convert 4D integral tensor to pair matrix format for same-spin interactions.
"""
function convert_op2_to_pair_matrix_same_spin(op::AbstractArray{Scalar}, n_orb::Int)
  n_pairs = n_orb * (n_orb + 1) ÷ 2
  mat = zeros(Scalar, n_pairs, n_pairs)
  if ndims(op) == 4
    @inbounds for i in 1:n_orb
      for j in 1:n_orb
        ij = pair_index(i, j)
        for k in 1:n_orb
          for l in 1:n_orb
            kl = pair_index(k, l)
            mat[ij, kl] = op[i, j, k, l]    # Raw (ij|kl)
          end
        end
      end
    end
  else
    error("Same-spin pair matrix requires 4D integral tensor")
  end
  return mat
end

"""
    convert_op2_to_pair_matrix_opposite_spin(op::AbstractArray{Scalar}, n_orb::Int)

Convert 4D integral tensor to pair matrix format for opposite-spin interactions.
Stores pure Coulomb integrals: `(ij|kl)` without exchange correction.
"""
function convert_op2_to_pair_matrix_opposite_spin(op::AbstractArray{Scalar}, n_orb::Int)
  n_pairs = n_orb * (n_orb + 1) ÷ 2
  mat = zeros(Scalar, n_pairs, n_pairs)
  if ndims(op) == 4
    @inbounds for i in 1:n_orb
      for j in 1:n_orb
        ij = pair_index(i, j)
        for k in 1:n_orb
          for l in 1:n_orb
            kl = pair_index(k, l)
            mat[ij, kl] = op[i, j, k, l]    # (ij|kl)
          end
        end
      end
    end
  else
    error("Opposite-spin pair matrix requires 4D integral tensor")
  end
  return mat
end

"""
    contract!(term::HamiltonianTerm, r::FCIVector, c::FCIVector, prefactor::Scalar)

Apply Hamiltonian term: |r⟩ += prefactor * H |c⟩
"""
function contract! end

"""
    HamiltonianTerm1e <: HamiltonianTerm

One-electron Hamiltonian term: h_ij c^i c_j
"""
mutable struct HamiltonianTerm1e <: HamiltonianTerm
  n_orb::FCIUInt
  spatial::Bool  # true if same matrix for both spins
  base_factor::Scalar
  op1_matrix_a::Matrix{Scalar}  # Alpha spin matrix
  op1_matrix_b::Matrix{Scalar}  # Beta spin matrix

  function HamiltonianTerm1e(n_orb::Integer, op1_matrix_a::AbstractMatrix{Scalar},
                             op1_matrix_b::Union{AbstractMatrix{Scalar}, Nothing} = nothing)
    spatial = (op1_matrix_b === nothing)
    op1_b = spatial ? op1_matrix_a : op1_matrix_b
    new(FCIUInt(n_orb), spatial, 1.0, Matrix(op1_matrix_a), Matrix(op1_b))
  end
end

"""
    HamiltonianTerm2e <: HamiltonianTerm

Two-electron Hamiltonian term: ``\\[1/2\\] (ij|kl) E^i_j E^k_l``

WARNING: Uses (ij|kl) integrals, NOT ⟨ij|kl⟩ and includes factor 1/2.
"""
mutable struct HamiltonianTerm2e <: HamiltonianTerm
  n_orb::FCIUInt
  spatial::Bool
  base_factor::Scalar
  op2_matrix_aa::Matrix{Scalar}
  op2_matrix_bb::Matrix{Scalar}
  op2_matrix_ab::Matrix{Scalar}

  function HamiltonianTerm2e(n_orb::Integer, op2_matrix_aa::AbstractArray{Scalar},
                             op2_matrix_bb::Union{AbstractArray{Scalar}, Nothing} = nothing,
                             op2_matrix_ab::Union{AbstractArray{Scalar}, Nothing} = nothing)
    n_orb_int = Int(n_orb)
    spatial = (op2_matrix_bb === nothing)
    mat_aa = convert_op2_to_pair_matrix(op2_matrix_aa, n_orb_int)
    mat_bb = spatial ? mat_aa : convert_op2_to_pair_matrix(op2_matrix_bb, n_orb_int)
    mat_ab = spatial ? mat_aa : convert_op2_to_pair_matrix(op2_matrix_ab, n_orb_int)
    new(FCIUInt(n_orb_int), spatial, 0.5, mat_aa, mat_bb, mat_ab)
  end
end

"""
    apply_1e_op!(result::AbstractVector{Scalar}, coeffs::AbstractVector{Scalar}, 
                prefactor::Scalar, op_matrix_1e::AbstractMatrix{Scalar},
                adr1::OrbStringAdrTable, adr2::OrbStringAdrTable,
                st1::Integer, st2::Integer)

Apply 1-electron operator on specified spin branch.
"""
function apply_1e_op!(result::AbstractVector{Scalar}, coeffs::AbstractVector{Scalar},
                      prefactor::Scalar, op_matrix_1e::AbstractMatrix{Scalar},
                      adr1::OrbStringAdrTable, adr2::OrbStringAdrTable, st1::Integer, st2::Integer)
  if n_elec(adr1) == 0
    return
  end

  n_orb_val = adr1.n_orb

  # Create intermediate addressing table for N-1 electrons
  adr_k1 = OrbStringAdrTable(n_elec(adr1) - 1, n_orb_val)

  # Pre-compute addressing and signs for all intermediate states
  addr_k1 = Vector{Address}(undef, n_orb_val * n_str(adr_k1))
  signs = Vector{Int8}(undef, n_orb_val * n_str(adr_k1))

  # Parallel computation of intermediate state connections
  Threads.@threads for i_str1 in 1:n_str(adr_k1)
    k1 = make_pattern(adr_k1, Address(i_str1))
    for k in 0:(n_orb_val - 1)
      i_kk = k + 1 + n_orb_val * (i_str1 - 1)
      i1, sign = apply_cop1(k1, k, +1)
      signs[i_kk] = sign
      if sign != 0
        addr_k1[i_kk] = adr1(i1)
      else
        addr_k1[i_kk] = Address(0)
      end
    end
  end

  # Block size for efficient vectorization
  n_tgt_blk_k = 64

  # Process each beta string separately (parallel over beta strings)
  Threads.@threads for i_str2 in 1:n_str(adr2)
    coeffs_beta = @view(coeffs[(st2 * (i_str2 - 1) + 1):end])
    result_beta = @view(result[(st2 * (i_str2 - 1) + 1):end])

    # Allocate temporary arrays for this thread
    input_k = zeros(Scalar, n_orb_val * n_tgt_blk_k)
    output_k = zeros(Scalar, n_orb_val * n_tgt_blk_k)

    # Process intermediate states in blocks
    for i_block_beg_k1 in 1:n_tgt_blk_k:n_str(adr_k1)
      n_blk_k = min(i_block_beg_k1 + n_tgt_blk_k - 1, n_str(adr_k1)) - i_block_beg_k1 + 1

      block_start = n_orb_val * (i_block_beg_k1 - 1)
      addr_k1_block = view(addr_k1, (block_start + 1):(block_start + n_orb_val * n_blk_k))
      signs_block = view(signs, (block_start + 1):(block_start + n_orb_val * n_blk_k))

      # Gather input coefficients with signs
      for i_kk in 1:(n_orb_val * n_blk_k)
        adr_idx = addr_k1_block[i_kk]
        input_k[i_kk] = signs_block[i_kk] * coeffs_beta[st1 * adr_idx + 1]
      end

      # Matrix multiplication: output_k = op_matrix_1e * input_k
      input_k_mat = reshape(view(input_k, 1:(n_orb_val * n_blk_k)), n_orb_val, n_blk_k)
      output_k_mat = reshape(view(output_k, 1:(n_orb_val * n_blk_k)), n_orb_val, n_blk_k)

      mul!(output_k_mat, op_matrix_1e, input_k_mat, prefactor, 0.0)

      # Scatter output coefficients with signs
      for i_kk in 1:(n_orb_val * n_blk_k)
        adr_idx = addr_k1_block[i_kk]
        result_beta[st1 * adr_idx + 1] += signs_block[i_kk] * output_k[i_kk]
      end
    end
  end
end

"""
    contract!(term::HamiltonianTerm1e, r::FCIVector, c::FCIVector, prefactor::Scalar)

Apply one-electron Hamiltonian term.
"""
function contract!(term::HamiltonianTerm1e, r::FCIVector, c::FCIVector, prefactor::Scalar)
  @assert compatible(r, c) "Incompatible FCI vectors"

  base_prefactor = term.base_factor * prefactor
  if abs(base_prefactor) < ThrNeglect
    return
  end

  # Apply on alpha strings, leaving beta invariant
  apply_1e_op!(vec(r.data), vec(c.data), base_prefactor, term.op1_matrix_a, r.adr_a, r.adr_b,
               1, Int(r.n_str_a))

  # Apply on beta strings, leaving alpha invariant
  apply_1e_op!(vec(r.data), vec(c.data), base_prefactor, term.op1_matrix_b, r.adr_b, r.adr_a,
               Int(r.n_str_a), 1)
end

"""
    add_1rdm_for_spin!(rdm::AbstractMatrix{Scalar}, coeff_l::AbstractVector{Scalar}, 
                      coeff_r::AbstractVector{Scalar}, adr1::OrbStringAdrTable, 
                      adr2::OrbStringAdrTable, st1::Integer, st2::Integer)

Add contribution to 1-RDM for one spin.
"""
function add_1rdm_for_spin!(rdm::AbstractMatrix{Scalar}, 
                            coeff_l::AbstractVector{Scalar}, coeff_r::AbstractVector{Scalar},
                            adr1::OrbStringAdrTable, adr2::OrbStringAdrTable,
                            st1::Integer, st2::Integer)
  n_orb_val = adr1.n_orb

  # Pre-allocate substitution buffer (reused for each string)
  subst_buffer = [SubstResult() for _ in 1:(n_orb_val * n_orb_val)]

  # Iterate through string configurations
  for i_str1 in 1:n_str(adr1)
    i1 = make_pattern(adr1, Address(i_str1))
    n_subst = form_string_substs_for_spin!(subst_buffer, nothing, adr1, i1)

    for i_s in 1:n_subst
      s = subst_buffer[i_s]
      i_adr_r1 = st1 * (s.i_str - 1) + 1
      i_adr_c1 = st1 * (i_str1 - 1) + 1

      tkl = 0.0
      for i_str2 in 1:n_str(adr2)
        idx_l = i_adr_r1 + st2 * (i_str2 - 1)
        idx_r = i_adr_c1 + st2 * (i_str2 - 1)
        tkl += coeff_l[idx_l] * coeff_r[idx_r]
      end

      rdm[s.k, s.l] += s.sign * tkl  # s.k and s.l are already 1-based
    end
  end
end

"""
    make_1rdms!(rdm_a::Matrix{Scalar}, rdm_b::Matrix{Scalar},
                coeff_l::FCIVector, coeff_r::FCIVector)

Compute 1-RDMs for alpha and beta spins as transition density matrices.

Computes:
- Γ_α[r,s] = <coeff_l|c†_{rα} c_{sα}|coeff_r>
- Γ_β[r,s] = <coeff_l|c†_{rβ} c_{sβ}|coeff_r>

For a single state (coeff_l == coeff_r), this gives the regular 1-RDM.
For different states, this gives the transition density matrix.

# Arguments
- `rdm_a`: Pre-allocated n_orb × n_orb matrix for alpha 1-RDM
- `rdm_b`: Pre-allocated n_orb × n_orb matrix for beta 1-RDM
- `coeff_l`: Left FCI vector (bra state)
- `coeff_r`: Right FCI vector (ket state)
"""
function make_1rdms!(rdm_a::Matrix{Scalar}, rdm_b::Matrix{Scalar}, 
                     coeff_l::FCIVector, coeff_r::FCIVector)
  # Verify compatibility
  @assert coeff_l.n_orb == coeff_r.n_orb "Vectors must have same n_orb"
  @assert coeff_l.n_elec_a == coeff_r.n_elec_a "Vectors must have same n_elec_a"
  @assert coeff_l.n_elec_b == coeff_r.n_elec_b "Vectors must have same n_elec_b"
  
  n_orb = Int(coeff_l.n_orb)
  @assert size(rdm_a) == (n_orb, n_orb) "rdm_a must be n_orb × n_orb"
  @assert size(rdm_b) == (n_orb, n_orb) "rdm_b must be n_orb × n_orb"
  
  # Initialize RDMs to zero
  fill!(rdm_a, 0.0)
  fill!(rdm_b, 0.0)
  
  # Compute alpha RDM
  # Loop over beta strings (outer), alpha string substitutions (inner)
  add_1rdm_for_spin!(rdm_a, vec(coeff_l.data), vec(coeff_r.data), coeff_l.adr_a, coeff_l.adr_b,
                     1, Int(coeff_l.n_str_a))
  
  # Compute beta RDM
  # Loop over alpha strings (outer), beta string substitutions (inner)
  add_1rdm_for_spin!(rdm_b, vec(coeff_l.data), vec(coeff_r.data), coeff_l.adr_b, coeff_l.adr_a,
                     Int(coeff_l.n_str_a), 1)
end

"""
    make_1rdms!(rdm_a::Matrix{Scalar}, rdm_b::Matrix{Scalar}, coeff::FCIVector)

Convenience method for computing 1-RDM of a single state (self-transition).
"""
function make_1rdms!(rdm_a::Matrix{Scalar}, rdm_b::Matrix{Scalar}, coeff::FCIVector)
  make_1rdms!(rdm_a, rdm_b, coeff, coeff)
end

"""
    make_2rdm!(rdm2::Array{Scalar, 4}, coeff::FCIVector, spin_phase::Scalar = 1.0)

Compute 2-particle reduced density matrix (2-RDM).

Computes: `Γ[r,s,t,u] = <coeff|E^r_s E^t_u|coeff>`
where `E^r_s = c†_r c_s` is the singlet excitation operator.

If `spin_phase = -1`, computes spin-adapted version with
`Ẽ^r_s = a†_{rα} a_{sα} - a†_{rβ} a_{sβ}`

The algorithm:
1. Loop over blocks of alpha/beta strings
2. Form intermediate matrices: `Inp[kl,K] = <K|c†_k c_l|J> c[J,K]`
3. Contract to 2-RDM: `Rdm2[rs,tu] += Inp[rs,K] * Inp[tu,K]`
4. Transpose first two indices for correct ordering

# Arguments
- `rdm2`: Pre-allocated 4D array [n_orb, n_orb, n_orb, n_orb]
- `coeff`: FCI vector
- `spin_phase`: Phase for spin-adapted operators (default: 1.0)

# Notes
- Computational cost: O(N_det × n_orb^4)
"""
function make_2rdm!(rdm2::Array{Scalar, 4}, coeff::FCIVector, spin_phase::Scalar = 1.0)
  n_orb = Int(coeff.n_orb)
  n_pairs_n = n_orb * n_orb
  
  @assert size(rdm2) == (n_orb, n_orb, n_orb, n_orb) "rdm2 must be n_orb × n_orb × n_orb × n_orb"
  
  adr_a = coeff.adr_a
  adr_b = coeff.adr_b
  
  n_tgt_blk_k = 64
  n_tgt_blk_kb = 64
  
  # Flatten rdm2 to matrix form [rs, tu]
  rdm2_flat = reshape(rdm2, n_pairs_n, n_pairs_n)
  fill!(rdm2_flat, 0.0)
  
  # Flatten coefficients
  coeffs = vec(coeff.data)
  n_str_a = Int(coeff.n_str_a)
  n_str_b = Int(coeff.n_str_b)
  
  input = zeros(Scalar, n_pairs_n * n_tgt_blk_k * n_tgt_blk_kb)
  # Pre-allocate StrInfo arrays with substitution buffers
  max_subst = n_orb * n_orb
  info_a_pool = [StrInfo(max_subst) for _ in 1:n_tgt_blk_k]
  info_b_pool = [StrInfo(max_subst) for _ in 1:n_tgt_blk_kb]
  
  # Loop over beta string blocks
  for block_b_start in 1:n_tgt_blk_kb:n_str_b
    block_b_end = min(block_b_start + n_tgt_blk_kb - 1, n_str_b)
    n_blk_b = block_b_end - block_b_start + 1
    
    # Get beta string substitutions for this block
    info_b = @view info_b_pool[1:n_blk_b]
    for (idx, str_idx) in enumerate(block_b_start:block_b_end)
      pattern = make_pattern(adr_b, Address(str_idx))
      n_subst = form_string_substs_for_spin!(info_b[idx].subst, nothing, adr_b, pattern)
      info_b[idx].n_subst = FCIUInt(n_subst)
    end
    
    # Loop over alpha string blocks
    for block_a_start in 1:n_tgt_blk_k:n_str_a
      block_a_end = min(block_a_start + n_tgt_blk_k - 1, n_str_a)
      n_blk_a = block_a_end - block_a_start + 1
      
      # Get alpha string substitutions for this block
      info_a = @view info_a_pool[1:n_blk_a]
      for (idx, str_idx) in enumerate(block_a_start:block_a_end)
        pattern = make_pattern(adr_a, Address(str_idx))
        n_subst = form_string_substs_for_spin!(info_a[idx].subst, nothing, adr_a, pattern)
        info_a[idx].n_subst = FCIUInt(n_subst)
      end
      
      block_len = n_pairs_n * n_blk_a * n_blk_b
      input[1:block_len] .= 0.0

      c_sum = Ref{Scalar}(0.0)
      
      # Contract alpha strings:
      # Inp[kl,K] = <K_α|c†_k c_l|J_α> c[J_α,K_β]
      input_alpha = StridedView(input, (n_orb, n_orb, n_blk_a, n_blk_b), (1, n_orb, n_pairs_n, n_pairs_n * n_blk_a))
      beta_offset = (block_b_start - 1) * n_str_a
      coeff_beta = StridedView(coeffs, (n_str_a, n_blk_b), (1, n_str_a), beta_offset)
      block_contract_cc1_nosym!(input_alpha, info_a, coeff_beta, c_sum, 1.0)
      
      # Contract beta strings (add contribution):
      # Inp[kl,K] += spin_phase * <K_β|c†_k c_l|J_β> c[K_α,J_β]
      input_beta = StridedView(input, (n_orb, n_orb, n_blk_b, n_blk_a), (1, n_orb, n_pairs_n * n_blk_a, n_pairs_n))
      alpha_offset = block_a_start - 1
      coeff_alpha = StridedView(coeffs, (n_str_b, n_blk_a), (n_str_a, 1), alpha_offset)
      block_contract_cc1_nosym!(input_beta, info_b, coeff_alpha, c_sum, spin_phase)
      
      # Contract to 2-RDM if contributions are significant
      if c_sum[] > ThrNeglect
        # Rdm2[rs,tu] += Inp[rs,K] * Inp[tu,K]^T
        # This is: rdm2 += inp_k * inp_k^T (rank-k update)
        inp_k_mat = reshape(@view(input[1:block_len]), n_pairs_n, n_blk_a * n_blk_b)
        mul!(rdm2_flat, inp_k_mat, inp_k_mat', 1.0, 1.0)
      end
    end
  end
  
  # Symmetrize (we only computed upper triangle)
  for tu in 1:n_pairs_n
    for rs in 1:tu-1
      rdm2_flat[tu, rs] = rdm2_flat[rs, tu]
    end
  end
  
  # Transpose first two indices: computed <E^s_r E^t_u>, need <E^r_s E^t_u>
  for tu in 1:n_pairs_n
    transpose_inplace_sqr!(view(rdm2_flat, :, tu), n_orb)
  end
end

"""
    transpose_inplace_sqr!(mat_view::AbstractVector, n::Int)

Transpose a square matrix in-place stored as a flattened vector.

The matrix is n × n stored in column-major order in mat_view.
Transposes indices: mat[i,j] ↔ mat[j,i].
"""
function transpose_inplace_sqr!(mat_view::AbstractVector, n::Int)
  @assert length(mat_view) == n * n "Vector length must equal n^2"
  
  for i in 1:n
    for j in 1:i-1
      idx1 = (j - 1) * n + i  # mat[i,j] in column-major
      idx2 = (i - 1) * n + j  # mat[j,i] in column-major
      mat_view[idx1], mat_view[idx2] = mat_view[idx2], mat_view[idx1]
    end
  end
end

"""
    contract!(term::HamiltonianTerm2e, r::FCIVector, c::FCIVector, prefactor::Scalar)

Apply two-electron Hamiltonian term.
"""
function contract!(term::HamiltonianTerm2e, r::FCIVector, c::FCIVector, prefactor::Scalar)
  @assert compatible(r, c) "Incompatible FCI vectors"

  base_prefactor = term.base_factor * prefactor
  if abs(base_prefactor) < ThrNeglect
    return
  end

  n_str_a = Int(r.n_str_a)
  n_str_b = Int(r.n_str_b)
  n_orb = Int(term.n_orb)
  n_pairs = n_orb * (n_orb + 1) ÷ 2

  # Data is stored in [alpha, beta] order
  coeff = reshape(c.data, :)
  resid = reshape(r.data, :)

  use_symmetry_ab = term.spatial && (n_spin(r) == 0) && r.is_spin_projected
  if use_symmetry_ab
    symmetrize_ci_vector!(c)
    coeff = reshape(c.data, :)
  end

  adr_a = r.adr_a
  adr_b = r.adr_b

  n_tgt_blk_k = 64
  n_tgt_blk_kb = 64
  # Pre-allocate input and output buffers
  n_block_cols = n_tgt_blk_k * n_tgt_blk_kb
  spatial_block_len = n_pairs * n_block_cols
  block_len = spatial_block_len * (term.spatial ? 1 : 2)
  input = zeros(Scalar, block_len)
  output = similar(input)
  # Pre-allocate StrInfo arrays with pre-allocated subst vectors
  # Each StrInfo gets its own subst buffer to avoid allocations in loops
  max_subst = n_orb * n_orb
  info_a = [StrInfo(max_subst) for _ in 1:n_tgt_blk_k]
  info_b = [StrInfo(max_subst) for _ in 1:n_tgt_blk_kb]

  dummy_ref = Ref{Scalar}(0.0)

  for block_b_start in 1:n_tgt_blk_kb:n_str_b
    block_b_end = min(block_b_start + n_tgt_blk_kb - 1, n_str_b)
    n_blk_b = block_b_end - block_b_start + 1

    for (idx, str_idx) in enumerate(block_b_start:block_b_end)
      pattern = make_pattern(adr_b, Address(str_idx))
      # Fill pre-allocated buffer directly, no copying needed
      n_subst = form_string_substs_for_spin!(info_b[idx].subst, nothing, adr_b, pattern)
      info_b[idx].n_subst = FCIUInt(n_subst)
    end

    for block_a_start in 1:n_tgt_blk_k:n_str_a
      block_a_end = min(block_a_start + n_tgt_blk_k - 1, n_str_a)
      n_blk_a = block_a_end - block_a_start + 1

      scale_factor = base_prefactor
      if use_symmetry_ab
        if block_a_start > block_b_start
          continue
        elseif block_a_start < block_b_start
          scale_factor *= 2.0  # symmetry correction
        end
      end

      for (idx, str_idx) in enumerate(block_a_start:block_a_end)
        pattern = make_pattern(adr_a, Address(str_idx))
        # Fill pre-allocated buffer directly, no copying needed
        n_subst = form_string_substs_for_spin!(info_a[idx].subst, nothing, adr_a, pattern)
        info_a[idx].n_subst = FCIUInt(n_subst)
      end

      n_block_cols = n_blk_a * n_blk_b
      spatial_block_len = n_pairs * n_block_cols
      block_len = spatial_block_len * (term.spatial ? 1 : 2)

      input[1:block_len] .= 0.0
      # In spatial case, alpha and beta point to same memory
      io_beta_offset = term.spatial ? 0 : spatial_block_len

      csum = Ref{Scalar}(0.0)

      beta_offset = (block_b_start - 1) * n_str_a
      coeff_beta = StridedView(coeff, (n_str_a, n_blk_b), (1, n_str_a), beta_offset)
      input_alpha = StridedView(input, (n_pairs, n_blk_a, n_blk_b), (1, n_pairs, n_pairs * n_blk_a))
      block_contract_cc1!(input_alpha, info_a, coeff_beta, 'c', csum, 1.0)

      alpha_offset = block_a_start - 1
      coeff_alpha = StridedView(coeff, (n_str_b, n_blk_a), (n_str_a, 1), alpha_offset)
      input_beta = StridedView(input, (n_pairs, n_blk_b, n_blk_a), (1, n_pairs * n_blk_a, n_pairs),
                               io_beta_offset)
      block_contract_cc1!(input_beta, info_b, coeff_alpha, 'c', csum, 1.0)

      if csum[] > ThrNeglect
        input_a_mat = reshape(@view(input[1:spatial_block_len]), n_pairs, n_block_cols)
        output_a_mat = reshape(@view(output[1:spatial_block_len]), n_pairs, n_block_cols)

        if term.spatial
          # Spatial case
          mul!(output_a_mat, term.op2_matrix_aa, input_a_mat)
        else
          # Non-spatial case: separate matrices
          mul!(output_a_mat, term.op2_matrix_aa, input_a_mat)

          input_b_mat = reshape(@view(input[(1 + io_beta_offset):block_len]), n_pairs, n_block_cols)
          output_b_mat = reshape(@view(output[(1 + io_beta_offset):block_len]), n_pairs, n_block_cols)
          mul!(output_b_mat, term.op2_matrix_bb, input_b_mat)
          mul!(output_a_mat, term.op2_matrix_ab, input_b_mat, 1.0, 1.0)
          mul!(output_b_mat, term.op2_matrix_ab', input_a_mat, 1.0, 1.0)
        end

        resid_beta = StridedView(resid, (n_str_a, n_blk_b), (1, n_str_a), beta_offset)
        output_alpha = StridedView(output, (n_pairs, n_blk_a, n_blk_b), (1, n_pairs, n_pairs * n_blk_a))
        block_contract_cc1!(output_alpha, info_a, resid_beta, 'R', dummy_ref, scale_factor)

        resid_alpha = StridedView(resid, (n_str_b, n_blk_a), (n_str_a, 1), alpha_offset)
        output_beta = StridedView(output, (n_pairs, n_blk_b, n_blk_a), (1, n_pairs * n_blk_a, n_pairs),
                                  io_beta_offset)
        block_contract_cc1!(output_beta, info_b, resid_alpha, 'R', dummy_ref, scale_factor)
      end
    end
  end

  resid_matrix = reshape(resid, n_str_a, n_str_b)
  # No permutedims needed since data is now stored in [alpha, beta] order
  r.data .= resid_matrix

  if use_symmetry_ab
    symmetrize_ci_vector!(r)
  end
end