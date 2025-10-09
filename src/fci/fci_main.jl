"""
    FCIContext

Main FCI calculation context.
"""
mutable struct FCIContext
  fcidump::FCIDump
  options::FCIOptions
  adr_a::OrbStringAdrTable
  adr_b::OrbStringAdrTable
  coeff::FCIVector
  resid::FCIVector
  diag_h::FCIVector
  absorb_1e::Bool
  hamiltonian_terms::Vector{HamiltonianTerm}
  mod_core_h_a::Matrix{Scalar}
  mod_core_h_b::Matrix{Scalar}
  basis_a::Matrix{Scalar}
  basis_b::Matrix{Scalar}
  rdm1_a::Matrix{Scalar}
  rdm1_b::Matrix{Scalar}
  rdm2::Union{Nothing, Array{Scalar, 4}}  # 2-RDM (optional)
  energy_fci::Scalar
  energy_ptrace::Scalar
  method_name::String
  pspace_data::PSpaceData             # P-space calculation data

  function FCIContext(fcidump::FCIDump, options::FCIOptions = FCIOptions())
    n_elec = fcidump.n_elec
    n_orb = fcidump.n_orb
    n_spin = fcidump.n_spin

    # Initialize FCI vectors
    coeff = FCIVector(n_elec, n_orb, n_spin, false)
    resid = FCIVector(n_elec, n_orb, n_spin, false)
    diag_h = FCIVector(n_elec, n_orb, n_spin, false)

    # Select integrals based on RHF vs UHF
    if fcidump.is_uhf
      # UHF: Use spin-separated integrals
      if fcidump.h1a === nothing || fcidump.h1b === nothing
        error("UHF calculation requires h1a, h1b integrals")
      end
      if fcidump.h2aa === nothing || fcidump.h2bb === nothing || fcidump.h2ab === nothing
        error("UHF calculation requires h2aa, h2bb, h2ab integrals")
      end
      
      mod_core_h_a = copy(fcidump.h1a)
      mod_core_h_b = copy(fcidump.h1b)
    else
      # RHF: Use spatial integrals
      mod_core_h_a = copy(fcidump.h1)
      mod_core_h_b = copy(fcidump.h1)
    end

    context = new(
      fcidump,
      options,
      coeff.adr_a,
      coeff.adr_b,
      coeff,
      resid,
      diag_h,
      true,
      HamiltonianTerm[],  # Set absorb_1e = true by default
      mod_core_h_a,
      mod_core_h_b,
      Matrix{Scalar}(I, n_orb, n_orb),
      Matrix{Scalar}(I, n_orb, n_orb),
      zeros(Scalar, n_orb, n_orb),
      zeros(Scalar, n_orb, n_orb),
      nothing,  # rdm2 - not computed by default
      0.0,
      0.0,
      fcidump.is_uhf ? "UHF-FCI" : "FCI",
      PSpaceData(),
    )  # Initialize empty P-space data

    # Initialize Hamiltonian terms
    init_hamiltonian_terms!(context)

    return context
  end
end

"""
    DiagonalHEvalData

Data for evaluating diagonal Hamiltonian elements.
"""
mutable struct DiagonalHEvalData
  jaa::Matrix{Scalar}      # JAA: (ii|jj)
  kaa::Matrix{Scalar}      # KAA: (ij|ij)
  kbb::Matrix{Scalar}      # KBB: (ij|ij) for beta
  jbb::Matrix{Scalar}      # JBB: (ii|jj) for beta
  jab::Matrix{Scalar}      # JAB: (ii|jj) mixed
  kab::Matrix{Scalar}      # KAB: (ij|ij) mixed
  hbar_a::Vector{Scalar}   # ⟨i|ModCoreH|i⟩ for alpha
  hbar_b::Vector{Scalar}   # ⟨i|ModCoreH|i⟩ for beta
  n_orb::FCIUInt

  function DiagonalHEvalData()
    new(
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      zeros(Scalar, 0, 0),
      Scalar[],
      Scalar[],
      0,
    )
  end
end

"""
    init_ab!(eval_data::DiagonalHEvalData, int2e_aa, int2e_bb, int2e_ab,
            core_h_a, core_h_b, n_orb::Integer, n_pairs::Integer, c1_integrals::Bool)

Initialize diagonal evaluation data for alpha/beta spins.
"""
function init_ab!(eval_data::DiagonalHEvalData, int2e_aa, int2e_bb, int2e_ab,
                  core_h_a, core_h_b, n_orb::Integer, n_pairs::Integer, c1_integrals::Bool)
  eval_data.n_orb = FCIUInt(n_orb)

  # Resize matrices to match n_orb
  eval_data.jaa = zeros(Scalar, n_orb, n_orb)
  eval_data.kaa = zeros(Scalar, n_orb, n_orb)
  eval_data.jbb = zeros(Scalar, n_orb, n_orb)
  eval_data.kbb = zeros(Scalar, n_orb, n_orb)
  eval_data.jab = zeros(Scalar, n_orb, n_orb)
  eval_data.kab = zeros(Scalar, n_orb, n_orb)

  # Resize vectors to match n_orb
  eval_data.hbar_a = zeros(Scalar, n_orb)
  eval_data.hbar_b = zeros(Scalar, n_orb)

  get_diagonal_pair_ints_2e!(eval_data.jaa, eval_data.kaa, int2e_aa, n_pairs, n_orb)
  get_diagonal_pair_ints_2e!(eval_data.jbb, eval_data.kbb, int2e_bb, n_pairs, n_orb)
  get_diagonal_pair_ints_2e!(eval_data.jab, eval_data.kab, int2e_ab, n_pairs, n_orb)

  get_diagonal_pair_ints_1e!(eval_data.hbar_a, eval_data.kaa, core_h_a, n_orb, c1_integrals)
  get_diagonal_pair_ints_1e!(eval_data.hbar_b, eval_data.kbb, core_h_b, n_orb, c1_integrals)
end

"""
    (eval_data::DiagonalHEvalData)(str_a::OrbPattern, str_b::OrbPattern) -> Scalar

Evaluate diagonal Hamiltonian element ⟨Ψ|H|Ψ⟩ for determinant |str_a, str_b⟩.
"""
function (eval_data::DiagonalHEvalData)(str_a::OrbPattern, str_b::OrbPattern)::Scalar
  f_elem = 0.0
  n_orb_val = Int(eval_data.n_orb)

  # One-electron contributions
  if !isempty(eval_data.hbar_a)
    for i in 0:(n_orb_val - 1)
      ni_a = Int((str_a >> i) & 1)
      ni_b = Int((str_b >> i) & 1)
      if ni_a > 0
        f_elem += eval_data.hbar_a[i + 1]
      end
      if ni_b > 0
        f_elem += eval_data.hbar_b[i + 1]
      end
    end
  end

  # Two-electron contributions
  for i in 0:(n_orb_val - 1)
    ni_a = Int((str_a >> i) & 1)
    ni_b = Int((str_b >> i) & 1)

    if ni_a + ni_b == 0
      continue
    end

    for j in 0:(n_orb_val - 1)
      nj_a = Int((str_a >> j) & 1)
      nj_b = Int((str_b >> j) & 1)

      f_elem += ni_a * nj_b * eval_data.jab[i + 1, j + 1]
      f_elem += (ni_a * nj_a) * 0.5 * eval_data.jaa[i + 1, j + 1]
      f_elem += (ni_b * nj_b) * 0.5 * eval_data.jbb[i + 1, j + 1]

      f_elem += 0.5 * eval_data.kaa[i + 1, j + 1] * (ni_a * (1 - nj_a))
      f_elem += 0.5 * eval_data.kbb[i + 1, j + 1] * (ni_b * (1 - nj_b))
    end
  end

  return f_elem
end

"""
    absorb_1e!(h2::Array{Scalar, 4}, n_orb::Integer, n_elec::Integer, 
               core_h_x::Matrix{Scalar}, core_h_y::Matrix{Scalar})

Absorb one-electron operators into two-electron operator.
"""
function absorb_1e!(h2::Array{Scalar, 4}, n_orb::Integer, n_elec::Integer,
                     core_h_x::Matrix{Scalar}, core_h_y::Matrix{Scalar})
  f_scale = 1.0 / n_elec

  for k in 1:n_orb
    for i in 1:n_orb
      for j in 1:n_orb
        # Absorb 1e terms into 2e integrals
        h2[k, k, i, j] += f_scale * core_h_y[j, i]
        h2[i, j, k, k] += f_scale * core_h_x[j, i]
      end
    end
  end
end

"""
    calc_mod_core_h!(mod_core_h::Matrix{Scalar}, h2::Array{Scalar, 4}, n_orb::Integer, c1_integrals::Bool)

Calculate modified core Hamiltonian by absorbing two-electron contributions.
"""
function calc_mod_core_h!(mod_core_h::Matrix{Scalar}, h2::Array{Scalar, 4},
                          n_orb::Integer, c1_integrals::Bool)
  fill!(mod_core_h, 0.0)

  if !c1_integrals
    # Use broadcasting for efficient calculation
    # mod_core_h[m, n] -= 0.5 * sum_i(h2[m, i, n, i])
    @inbounds for i in 1:n_orb
      mod_core_h .-= 0.5 .* view(h2, :, i, :, i)
    end
  end
end

"""
    init_hamiltonian_terms!(context::FCIContext)

Initialize Hamiltonian terms for the FCI calculation and compute diagonal Hamiltonian.
"""
function init_hamiltonian_terms!(context::FCIContext)
  n_orb = context.fcidump.n_orb
  n_elec = context.fcidump.n_elec

  if context.fcidump.is_uhf
    # UHF case: Handle all three spin-separated integral tensors properly
    
    # Create modified copies of all three integral tensors
    h2aa_modified = copy(context.fcidump.h2aa)
    h2bb_modified = copy(context.fcidump.h2bb)
    h2ab_modified = copy(context.fcidump.h2ab)
    
    # Calculate modified core Hamiltonian for alpha spin using h2aa and h2ab
    # mod_core_h_a[m,n] = h1a[m,n] - 0.5 * sum_i(h2aa[m,i,n,i])
    calc_mod_core_h!(context.mod_core_h_a, h2aa_modified, n_orb, false)
    
    # Calculate modified core Hamiltonian for beta spin using h2bb and h2ab
    # mod_core_h_b[m,n] = h1b[m,n] - 0.5 * sum_i(h2bb[m,i,n,i])
    calc_mod_core_h!(context.mod_core_h_b, h2bb_modified, n_orb, false)
    
    # Add original core Hamiltonian for each spin
    context.mod_core_h_a .+= context.fcidump.h1a
    context.mod_core_h_b .+= context.fcidump.h1b
    
    if context.absorb_1e
      # Absorb 1e terms into 2e integrals for each spin block
      # For UHF, we need to absorb differently:
      # - h2aa absorbs alpha contributions
      # - h2bb absorbs beta contributions  
      # - h2ab absorbs mixed contributions
      
      # Absorb alpha 1e terms into h2aa
      absorb_1e!(h2aa_modified, n_orb, n_elec, context.mod_core_h_a, context.mod_core_h_a)
      
      # Absorb beta 1e terms into h2bb
      absorb_1e!(h2bb_modified, n_orb, n_elec, context.mod_core_h_b, context.mod_core_h_b)
      
      # Absorb mixed 1e terms into h2ab (alpha-beta coupling)
      absorb_1e!(h2ab_modified, n_orb, n_elec, context.mod_core_h_a, context.mod_core_h_b)
      
      # Compute diagonal with modified integrals (1e terms absorbed, no separate core)
      make_diagonal_h!(context, context.diag_h, h2aa_modified, h2bb_modified, h2ab_modified, nothing, nothing)
      
      # Use 2e term with all three UHF integral tensors (1e terms absorbed)
      h2_term = HamiltonianTerm2e(n_orb, h2aa_modified, h2bb_modified, h2ab_modified)
      push!(context.hamiltonian_terms, h2_term)
    else
      # For absorb_1e=false, use original integrals for diagonal with separate core Hamiltonian
      make_diagonal_h!(context, context.diag_h, context.fcidump.h2aa, context.fcidump.h2bb, 
                       context.fcidump.h2ab, context.mod_core_h_a, context.mod_core_h_b)
      
      # Use separate 1e and 2e terms with all three UHF integral tensors
      h1_term = HamiltonianTerm1e(n_orb, context.mod_core_h_a, context.mod_core_h_b)
      h2_term = HamiltonianTerm2e(n_orb, h2aa_modified, h2bb_modified, h2ab_modified)
      push!(context.hamiltonian_terms, h1_term)
      push!(context.hamiltonian_terms, h2_term)
    end
  else
    # RHF case: Use spatial integrals (existing code)
    h2_modified = copy(context.fcidump.h2)

    # Calculate modified core Hamiltonian (includes 2e contributions)
    calc_mod_core_h!(context.mod_core_h_a, h2_modified, n_orb, false)
    calc_mod_core_h!(context.mod_core_h_b, h2_modified, n_orb, false)

    # Add original core Hamiltonian
    context.mod_core_h_a .+= context.fcidump.h1
    context.mod_core_h_b .+= context.fcidump.h1

    if context.absorb_1e
      # Absorb 1e terms into 2e integrals
      absorb_1e!(h2_modified, n_orb, n_elec, context.mod_core_h_a, context.mod_core_h_a)

      # Compute diagonal with modified integrals (1e terms absorbed, no separate core)
      make_diagonal_h!(context, context.diag_h, h2_modified, h2_modified, h2_modified, nothing, nothing)

      # Use only 2e term (1e terms absorbed)
      h2_term = HamiltonianTerm2e(n_orb, h2_modified)
      push!(context.hamiltonian_terms, h2_term)
    else
      # For absorb_1e=false, compute diagonal with original integrals and separate core
      make_diagonal_h!(context, context.diag_h, context.fcidump.h2, context.fcidump.h2, 
                       context.fcidump.h2, context.mod_core_h_a, context.mod_core_h_b)

      # Use separate 1e and 2e terms
      h1_term = HamiltonianTerm1e(n_orb, context.mod_core_h_a, context.mod_core_h_b)
      h2_term = HamiltonianTerm2e(n_orb, h2_modified)
      push!(context.hamiltonian_terms, h1_term)
      push!(context.hamiltonian_terms, h2_term)
    end
  end
end

"""
    make_diagonal_h!(context::FCIContext, diag_h::FCIVector, h2aa::Array{Scalar,4}, 
                     h2bb::Array{Scalar,4}, h2ab::Array{Scalar,4}, 
                     core_a::Union{Matrix{Scalar},Nothing}, core_b::Union{Matrix{Scalar},Nothing})

Compute diagonal Hamiltonian matrix elements with given integrals.

# Arguments
- `context`: FCIContext containing orbital and electron information
- `diag_h`: Output vector for diagonal elements
- `h2aa`: Alpha-alpha two-electron integrals
- `h2bb`: Beta-beta two-electron integrals  
- `h2ab`: Alpha-beta two-electron integrals
- `core_a`: Alpha one-electron operator (nothing if absorbed into h2)
- `core_b`: Beta one-electron operator (nothing if absorbed into h2)
"""
function make_diagonal_h!(context::FCIContext, diag_h::FCIVector,
                          h2aa::Array{Scalar,4}, h2bb::Array{Scalar,4}, h2ab::Array{Scalar,4},
                          core_a::Union{Matrix{Scalar},Nothing}, core_b::Union{Matrix{Scalar},Nothing})
  n_orb = Int(context.fcidump.n_orb)
  n_pairs = n_orb * (n_orb + 1) ÷ 2

  eval_data = DiagonalHEvalData()
  eval_data.jaa = zeros(Scalar, n_orb, n_orb)
  eval_data.kaa = zeros(Scalar, n_orb, n_orb)
  eval_data.jbb = zeros(Scalar, n_orb, n_orb)
  eval_data.kbb = zeros(Scalar, n_orb, n_orb)
  eval_data.jab = zeros(Scalar, n_orb, n_orb)
  eval_data.kab = zeros(Scalar, n_orb, n_orb)

  if core_a === nothing || core_b === nothing
    # 1e terms absorbed into 2e integrals
    eval_data.hbar_a = Scalar[]
    eval_data.hbar_b = Scalar[]
  else
    # Separate 1e contribution
    eval_data.hbar_a = zeros(Scalar, n_orb)
    eval_data.hbar_b = zeros(Scalar, n_orb)
  end

  # Initialize diagonal evaluation data with provided integrals
  init_ab!(eval_data, h2aa, h2bb, h2ab, core_a, core_b, n_orb, n_pairs, false)  # C1_Integrals = false

  n_str_a = Int(diag_h.n_str_a)
  n_str_b = Int(diag_h.n_str_b)

  Threads.@threads for idx_b in 1:n_str_b
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
    run_fci!(context::FCIContext) -> Scalar

Run full FCI calculation.
"""
function run_fci!(context::FCIContext)::Scalar
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
  println("Electronic energy:   $(energy - context.fcidump.e_nuc) Hartree")
  
  # Compute RDMs if requested
  if context.options.compute_rdms
    println("\nComputing 1-RDMs...")
    make_1rdms!(context.rdm1_a, context.rdm1_b, context.coeff)
    
    # Verify trace
    n_elec = context.fcidump.n_elec
    trace_1rdm = tr(context.rdm1_a) + tr(context.rdm1_b)
    println("  1-RDM trace: $(trace_1rdm) (expected: $(n_elec))")
    
    if abs(trace_1rdm - n_elec) > 1e-8
      @warn "1-RDM trace error: $(abs(trace_1rdm - n_elec))"
    else
      println("  ✓ 1-RDM trace verified")
    end
    
    # Compute 2-RDM if requested
    if context.options.compute_2rdm
      n_orb = context.fcidump.n_orb
      println("\nComputing 2-RDM...")
      
      # Allocate 2-RDM array if not already allocated
      if context.rdm2 === nothing
        context.rdm2 = zeros(Scalar, n_orb, n_orb, n_orb, n_orb)
      end
      
      # Compute 2-RDM
      make_2rdm!(context.rdm2, context.coeff, 1.0)
      
      # Verify energy from 2-RDM
      h2 = context.fcidump.h2
      e_2rdm = 0.0
      for l in 1:n_orb, k in 1:n_orb, j in 1:n_orb, i in 1:n_orb
        e_2rdm += 0.5 * context.rdm2[i,j,k,l] * h2[i,j,k,l]
      end
      
      # Add 1-electron contribution if absorb_1e is true
      if context.absorb_1e
        e_1rdm = tr(context.rdm1_a * context.mod_core_h_a) + tr(context.rdm1_b * context.mod_core_h_b)
        e_total_rdm = e_1rdm + e_2rdm + context.fcidump.e_nuc
      else
        e_total_rdm = e_2rdm + context.fcidump.e_nuc
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

  return energy
end

"""
    eval_diag_h(data::DiagonalHEvalData, str_a::BitVector, str_b::BitVector) -> Scalar

Evaluate diagonal Hamiltonian element ⟨Ψ|H|Ψ⟩ for bitstring representation of determinants.
"""
function eval_diag_h(data::DiagonalHEvalData, str_a::BitVector, str_b::BitVector)
  h = 0.0
  n_orb = length(str_a)

  for i in 1:n_orb
    ni_a = str_a[i] ? 1.0 : 0.0
    ni_b = str_b[i] ? 1.0 : 0.0

    if ni_a + ni_b == 0.0
      continue
    end

    if !isempty(data.hbar_a)
      h += ni_a * data.hbar_a[i] + ni_b * data.hbar_b[i]
    end

    for j in 1:n_orb
      nj_a = str_a[j] ? 1.0 : 0.0
      nj_b = str_b[j] ? 1.0 : 0.0

      h += ni_a * nj_b * data.jab[i, j]
      h += 0.5 * ni_a * nj_a * data.jaa[i, j]
      h += 0.5 * ni_b * nj_b * data.jbb[i, j]
      h += 0.5 * data.kaa[i, j] * (ni_a * (1.0 - nj_a))
      h += 0.5 * data.kbb[i, j] * (ni_b * (1.0 - nj_b))
    end
  end

  return h
end
