"""
    DiagonalHEvalData

Data for evaluating diagonal Hamiltonian elements.
"""
mutable struct DiagonalHEvalData
  jkaa::Matrix{Scalar}      # JKAA: 0.5[(ii|jj) - (ij|ij)]
  jkbb::Matrix{Scalar}      # JKBB: 0.5[(ii|jj) - (ij|ij)] for beta
  jab::Matrix{Scalar}      # JAB: (ii|jj) mixed
  hbar_a::Vector{Scalar}   # ⟨i|ModCoreH|i⟩ for alpha
  hbar_b::Vector{Scalar}   # ⟨i|ModCoreH|i⟩ for beta
  n_orb::FCIUInt
  ibuf::Vector{Int}        # Buffer for indices

  function DiagonalHEvalData()
    new(
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      Scalar[],
      Scalar[],
      0,
      Int[]
    )
  end
end

"""
    init_ab!(eval_data::DiagonalHEvalData, int2e_aa, int2e_bb, int2e_ab,
            core_h_a, core_h_b, n_orb::Integer)

Initialize diagonal evaluation data for alpha/beta spins.
"""
function init_ab!(eval_data::DiagonalHEvalData, int2e_aa, int2e_bb, int2e_ab,
                  core_h_a, core_h_b, n_orb::Integer)
  eval_data.n_orb = FCIUInt(n_orb)
  eval_data.ibuf = zeros(Int, 2*n_orb)

  # Resize matrices to match n_orb
  eval_data.jkaa = zeros(Scalar, n_orb, n_orb)
  eval_data.jkbb = zeros(Scalar, n_orb, n_orb)
  eval_data.jab = zeros(Scalar, n_orb, n_orb)

  # Resize vectors to match n_orb
  eval_data.hbar_a = zeros(Scalar, n_orb)
  eval_data.hbar_b = zeros(Scalar, n_orb)

  get_diagonal_pair_aints_2e!(eval_data.jkaa, int2e_aa, n_orb)
  get_diagonal_pair_aints_2e!(eval_data.jkbb, int2e_bb, n_orb)
  get_diagonal_pair_ints_2e!(eval_data.jab, int2e_ab, n_orb)

  get_diagonal_pair_ints_1e!(eval_data.hbar_a, core_h_a, n_orb)
  get_diagonal_pair_ints_1e!(eval_data.hbar_b, core_h_b, n_orb)
end

"""
    get_diagonal_pair_aints_2e!(jkaa::AbstractMatrix{Scalar}, int2e::AbstractArray{Scalar}, n_orb::Integer)

Extract diagonal pair antisymmetrized integrals for 2-electron terms.
"""
function get_diagonal_pair_aints_2e!(jkaa::AbstractMatrix{Scalar}, int2e::AbstractArray{Scalar}, n_orb::Integer)
  @assert size(jkaa) == (n_orb, n_orb) "jkaa matrix wrong size"
  @assert size(int2e) == (n_orb, n_orb, n_orb, n_orb) "int2e dimension mismatch"

  @inbounds for i in 1:n_orb
    for j in 1:i
      jij = 0.5 * (int2e[i, i, j, j] - int2e[i, j, i, j])  # (ii|jj) - (ij|ij)
      jkaa[i, j] = jij
      jkaa[j, i] = jij
    end
  end
end

"""
    get_diagonal_pair_ints_2e!(jab::AbstractMatrix{Scalar}, int2e::AbstractArray{Scalar}, n_orb::Integer)

Extract diagonal pair integrals for 2-electron terms.
"""
function get_diagonal_pair_ints_2e!(jab::AbstractMatrix{Scalar}, int2e::AbstractArray{Scalar}, n_orb::Integer)
  @assert size(jab) == (n_orb, n_orb) "jab matrix wrong size"
  @assert size(int2e) == (n_orb, n_orb, n_orb, n_orb) "int2e dimension mismatch"

  @inbounds for i in 1:n_orb
    for j in 1:i
      jab[i, j] = int2e[i, i, j, j]       # (ii|jj) - raw integral
      jab[j, i] = int2e[j, j, i, i]       # (jj|ii)
    end
  end
end

"""
    get_diagonal_pair_ints_1e!(hbar::AbstractVector{Scalar}, core_h, n_orb::Integer)

Extract diagonal pair integrals for 1-electron terms.
"""
function get_diagonal_pair_ints_1e!(hbar::AbstractVector{Scalar}, core_h, n_orb::Integer)
  @assert length(hbar) == n_orb "hbar vector wrong size"
  @inbounds for i in 1:n_orb
    hbar[i] = core_h[i, i]
  end
end

"""
    (eval_data::DiagonalHEvalData)(str_a::OrbPattern, str_b::OrbPattern) -> Scalar

Evaluate diagonal Hamiltonian element ⟨Ψ|H|Ψ⟩ for determinant |str_a, str_b⟩.
"""
function (eval_data::DiagonalHEvalData)(str_a::OrbPattern, str_b::OrbPattern)::Scalar
  f_elem = 0.0
  n_orb_val = Int(eval_data.n_orb)
  ibuf = eval_data.ibuf
  occa = BufVec(@view(ibuf[1:n_orb_val]))
  occb = BufVec(@view(ibuf[n_orb_val+1:end]))
  occupied_orbitals!(occa, str_a, n_orb_val)
  occupied_orbitals!(occb, str_b, n_orb_val)

  # One-electron contributions
  @inbounds @simd for i in occa
    f_elem += eval_data.hbar_a[i]
  end
  @inbounds @simd for i in occb
    f_elem += eval_data.hbar_b[i]
  end

  # Two-electron contributions
  @inbounds @simd for ia in occa
    for ja in occa
      f_elem += eval_data.jkaa[ia, ja] 
    end
    for jb in occb
      f_elem += eval_data.jab[ia, jb]
    end
  end
  @inbounds @simd for ib in occb
    for jb in occb
      f_elem += eval_data.jkbb[ib, jb]
    end
  end
  return f_elem
end

"""
    absorb_1e!(int2::Array{Scalar, 4}, n_orb::Integer, n_elec::Integer, 
               core_h_x::Matrix{Scalar}, core_h_y::Matrix{Scalar})

Absorb one-electron operators into two-electron operator.
"""
function absorb_1e!(int2::Array{Scalar, 4}, n_orb::Integer, n_elec::Integer,
                     core_h_x::Matrix{Scalar}, core_h_y::Matrix{Scalar})
  @assert size(core_h_x) == size(core_h_y) == (n_orb, n_orb) "core_h_x and core_h_y size mismatch"
  @assert size(int2) == (n_orb, n_orb, n_orb, n_orb) "int2 size mismatch"
  
  f_scale = 1.0 / n_elec
  @inbounds for k in 1:n_orb
    for i in 1:n_orb
      for j in 1:n_orb
        # Absorb 1e terms into 2e integrals
        int2[k, k, i, j] += f_scale * core_h_y[j, i]
        int2[i, j, k, k] += f_scale * core_h_x[j, i]
      end
    end
  end
end

"""
    calc_mod_core_h!(mod_core_h::Matrix{Scalar}, int2::Array{Scalar, 4}, n_orb::Integer, c1_integrals::Bool)

Calculate modified core Hamiltonian by absorbing two-electron contributions arising from 
changed order of creation/annihilation operators.
"""
function calc_mod_core_h!(mod_core_h::Matrix{Scalar}, int2::Array{Scalar, 4},
                          n_orb::Integer, c1_integrals::Bool)
  @assert size(mod_core_h) == (n_orb, n_orb) "mod_core_h size mismatch"
  @assert size(int2) == (n_orb, n_orb, n_orb, n_orb) "int2 size mismatch"

  fill!(mod_core_h, 0.0)
  if !c1_integrals
    # Use broadcasting for efficient calculation
    # mod_core_h[m, n] -= 0.5 * sum_i(int2[m, i, n, i])
    @inbounds for i in 1:n_orb
      mod_core_h .-= 0.5 .* view(int2, :, i, :, i)
    end
  end
end

"""
    init_hamiltonian_terms!(context::FCIContext)

Initialize Hamiltonian terms for the FCI calculation and compute diagonal Hamiltonian.
"""
function init_hamiltonian_terms!(context::FCIContext)
  n_orb = context.n_orb
  n_elec = context.n_elec[1] + context.n_elec[2]

  if context.fcidump.uhf
    # UHF case: Handle all three spin-separated integral tensors properly
    
    # Create modified copies of all three integral tensors
    int2aa_modified = copy(context.fcidump.int2aa)
    int2bb_modified = copy(context.fcidump.int2bb)
    int2ab_modified = copy(context.fcidump.int2ab)
    
    # Compute diagonal with unmodified integrals
    make_diagonal_h!(context, context.diag_h, int2aa_modified, int2bb_modified, int2ab_modified,
                     context.fcidump.int1a, context.fcidump.int1b)
    # Calculate modified core Hamiltonian for alpha spin using int2aa
    # mod_core_h_a[m,n] = int1a[m,n] - 0.5 * sum_i(int2aa[m,i,n,i])
    calc_mod_core_h!(context.mod_core_h_a, int2aa_modified, n_orb, false)
    
    # Calculate modified core Hamiltonian for beta spin using int2bb
    # mod_core_h_b[m,n] = int1b[m,n] - 0.5 * sum_i(int2bb[m,i,n,i])
    calc_mod_core_h!(context.mod_core_h_b, int2bb_modified, n_orb, false)
    
    # Add original core Hamiltonian for each spin
    context.mod_core_h_a .+= context.fcidump.int1a
    context.mod_core_h_b .+= context.fcidump.int1b
    
    if context.absorb_1e
      # Absorb 1e terms into 2e integrals for each spin block
      # For UHF, we need to absorb differently:
      # - int2aa absorbs alpha contributions
      # - int2bb absorbs beta contributions
      # - int2ab absorbs mixed contributions

      # Absorb alpha 1e terms into int2aa
      absorb_1e!(int2aa_modified, n_orb, n_elec, context.mod_core_h_a, context.mod_core_h_a)

      # Absorb beta 1e terms into int2bb
      absorb_1e!(int2bb_modified, n_orb, n_elec, context.mod_core_h_b, context.mod_core_h_b)

      # Absorb mixed 1e terms into int2ab (alpha-beta coupling)
      absorb_1e!(int2ab_modified, n_orb, n_elec, context.mod_core_h_a, context.mod_core_h_b)
      
      # Use 2e term with all three UHF integral tensors (1e terms absorbed)
      h2_term = HamiltonianTerm2e(n_orb, int2aa_modified, int2bb_modified, int2ab_modified)
      push!(context.hamiltonian_terms, h2_term)
    else
      # Use separate 1e and 2e terms with all three UHF integral tensors
      h1_term = HamiltonianTerm1e(n_orb, context.mod_core_h_a, context.mod_core_h_b)
      h2_term = HamiltonianTerm2e(n_orb, int2aa_modified, int2bb_modified, int2ab_modified)
      push!(context.hamiltonian_terms, h1_term)
      push!(context.hamiltonian_terms, h2_term)
    end
  else
    # RHF case: Use spatial integrals
    int2_modified = copy(context.fcidump.int2)

    # Compute diagonal with unmodified integrals
    make_diagonal_h!(context, context.diag_h, int2_modified, int2_modified, int2_modified, 
                     context.fcidump.int1, context.fcidump.int1)
    calc_mod_core_h!(context.mod_core_h_a, int2_modified, n_orb, false)
    calc_mod_core_h!(context.mod_core_h_b, int2_modified, n_orb, false)

    # Add original core Hamiltonian
    context.mod_core_h_a .+= context.fcidump.int1
    context.mod_core_h_b .+= context.fcidump.int1

    if context.absorb_1e
      # Absorb 1e terms into 2e integrals
      absorb_1e!(int2_modified, n_orb, n_elec, context.mod_core_h_a, context.mod_core_h_a)
      # Use only 2e term (1e terms absorbed)
      h2_term = HamiltonianTerm2e(n_orb, int2_modified)
      push!(context.hamiltonian_terms, h2_term)
    else
      # Use separate 1e and 2e terms
      h1_term = HamiltonianTerm1e(n_orb, context.mod_core_h_a, context.mod_core_h_b)
      h2_term = HamiltonianTerm2e(n_orb, int2_modified)
      push!(context.hamiltonian_terms, h1_term)
      push!(context.hamiltonian_terms, h2_term)
    end
  end
end

"""
    make_diagonal_h!(context::FCIContext, diag_h::FCIVector, int2aa::Array{Scalar,4}, 
                     int2bb::Array{Scalar,4}, int2ab::Array{Scalar,4}, 
                     core_a::Matrix{Scalar}, core_b::Matrix{Scalar})

Compute diagonal Hamiltonian matrix elements with given integrals.

# Arguments
- `context`: FCIContext containing orbital and electron information
- `diag_h`: Output vector for diagonal elements
- `int2aa`: Alpha-alpha two-electron integrals
- `int2bb`: Beta-beta two-electron integrals
- `int2ab`: Alpha-beta two-electron integrals
- `core_a`: Alpha one-electron operator
- `core_b`: Beta one-electron operator
"""
function make_diagonal_h!(context::FCIContext, diag_h::FCIVector,
                          int2aa::Array{Scalar,4}, int2bb::Array{Scalar,4}, int2ab::Array{Scalar,4},
                          core_a::Matrix{Scalar}, core_b::Matrix{Scalar})
  n_orb = context.n_orb

  eval_data = DiagonalHEvalData()
  # Initialize diagonal evaluation data with provided integrals
  init_ab!(eval_data, int2aa, int2bb, int2ab, core_a, core_b, n_orb)

  n_str_a = Int(diag_h.n_str_a)
  n_str_b = Int(diag_h.n_str_b)

  for idx_b in 1:n_str_b
    str_b = make_pattern(diag_h.adr_b, Address(idx_b))
    for idx_a in 1:n_str_a
      str_a = make_pattern(diag_h.adr_a, Address(idx_a))
      diag_h[idx_a,idx_b] = Scalar(eval_data(str_a, str_b))
    end
  end
end

"""
    contract_hamiltonian!(context::FCIContext, r::FCIVector, c::FCIVector, prefactor::Scalar)

Apply full Hamiltonian: |r⟩ += prefactor * H |c⟩
"""
function contract_hamiltonian!(context::FCIContext, r::FCIVector, c::FCIVector, prefactor::Scalar)
  for term in context.hamiltonian_terms
    contract!(term, r, c, prefactor)
  end
end

"""
    run_fci!(context::FCIContext) -> Vector{Scalar}

Run full FCI calculation.
"""
function run_fci!(context::FCIContext)
  println("="^80)
  println("FCI CALCULATION")
  println("="^80)

  # Run Davidson diagonalization
  energies, eigenvectors = davidson_fci!(context)
  energy = energies[1]  # Ground state energy

  println("\n" * "="^80)
  println("FCI RESULTS")
  println("="^80)
  println("Ground state energy: $(energy) Hartree")
  println("Electronic energy:   $(energy - context.fcidump.int0) Hartree")
  
  # Compute RDMs if requested
  if context.options.compute_rdms
    println("\nComputing 1-RDMs...")
    make_1rdms!(context.rdm1_a, context.rdm1_b, context.coeff)
    
    # Verify trace
    n_elec = context.n_elec[1] + context.n_elec[2]
    trace_1rdm = tr(context.rdm1_a) + tr(context.rdm1_b)
    println("  1-RDM trace: $(trace_1rdm) (expected: $(n_elec))")
    
    if abs(trace_1rdm - n_elec) > 1e-8
      @warn "1-RDM trace error: $(abs(trace_1rdm - n_elec))"
    else
      println("  ✓ 1-RDM trace verified")
    end
    
    # Compute 2-RDM if requested
    if context.options.compute_2rdm
      n_orb = context.n_orb
      println("\nComputing 2-RDM...")
      
      # Allocate 2-RDM array if not already allocated
      if context.rdm2 === nothing
        context.rdm2 = zeros(Scalar, n_orb, n_orb, n_orb, n_orb)
      end
      
      # Compute 2-RDM
      make_2rdm!(context.rdm2, context.coeff, 1.0)
      
      # Verify energy from 2-RDM
      int2 = context.fcidump.int2
      e_2rdm = 0.0
      for l in 1:n_orb, k in 1:n_orb, j in 1:n_orb, i in 1:n_orb
        e_2rdm += 0.5 * context.rdm2[i,j,k,l] * int2[i,j,k,l]
      end
      
      # Add 1-electron contribution if absorb_1e is true
      if context.absorb_1e
        e_1rdm = tr(context.rdm1_a * context.mod_core_h_a) + tr(context.rdm1_b * context.mod_core_h_b)
        e_total_rdm = e_1rdm + e_2rdm + context.fcidump.int0
      else
        e_total_rdm = e_2rdm + context.fcidump.int0
      end
      
      println("  Energy from 2-RDM: $(e_total_rdm) Hartree")
      println("  FCI energy:        $(energy) Hartree")
      println("  Difference:        $(abs(e_total_rdm - energy)) Hartree")
      
      if abs(e_total_rdm - energy) > 1e-6
        @warn "2-RDM energy check failed! Difference: $(abs(e_total_rdm - energy))"
      else
        println("  ✓ 2-RDM energy verified")
      end
    end
  end
  
  println("="^80)

  return energies
end