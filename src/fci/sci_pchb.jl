# ===========================================
# CIPHI - CI via Perturbative and Heat-Bath Iterative selection
# ===========================================

struct PCHBEntry{T}
  a::Int
  b::Int
  value::T
  dagger::T
  denom::T
end

"""
    CIPHISetupData

Setup data: Pre-computed and sorted double excitation matrix elements.

For each pair of orbitals {p,q}, stores a list of PCHBEntry {r,s,H(rs←pq),H(pq→rs),denom},
sorted by |H| in decreasing order. This enables efficient generation of only
important excitations during iterative selection.

Following Holmes et al. (2016), Algorithm Step IIa.
"""
struct CIPHISetupData{T}
  double_excitations_aa::Vector{Vector{PCHBEntry{T}}}  # alpha-alpha
  double_excitations_bb::Vector{Vector{PCHBEntry{T}}}  # beta-beta
  double_excitations_ab::Vector{Vector{PCHBEntry{T}}}  # alpha-beta mixed
  h_doub_max::Float64              # Maximum |H(rs ← pq)| over all excitations
  # Integrals for Epstein-Nesbet singles denominator -v_{ia}^{ia} + v_{ia}^{ai}
  singles_denoma::Matrix{T}
  singles_denomb::Matrix{T}
end

CIPHISetupData{T}() where T = CIPHISetupData{T}(
    Vector{PCHBEntry{T}}[], Vector{PCHBEntry{T}}[], Vector{PCHBEntry{T}}[],
    0.0, zeros(T, 0, 0), zeros(T, 0, 0))
CIPHISetupData() = CIPHISetupData{Float64}()

# RHF constructor
function CIPHISetupData(double_exc::Vector{Vector{PCHBEntry{T}}}, 
                      double_exc_ab::Vector{Vector{PCHBEntry{T}}},
                      h_max::Float64, singles_denom::Matrix{T}) where T
  CIPHISetupData{T}(double_exc, double_exc, double_exc_ab, h_max, singles_denom, singles_denom)
end

"""
    setup_ciphi!(ctx::Union{FCIContext, CIPHIContext}) -> CIPHISetupData

Setup: Pre-compute and store sorted double excitation matrix elements.

For each pair of orbitals {p,q}, computes H(rs ← pq) for all distinct {r,s} pairs
that don't include {p,q}, and stores them sorted by |H| in decreasing order.

This enables efficient generation of only important excitations,
avoiding computation of matrix elements that would be below threshold.

Algorithm from Holmes et al. (2016), IIa:
- Time complexity: O(M^4 log M)
- Space complexity: O(M^4)
where M is the number of orbitals.
"""
function setup_ciphi!(ctx::Union{FCIContext, CIPHIContext})
  is_uhf = ctx.fcidump.uhf
  
  if !is_uhf
    # RHF case: use standard int2 integrals
    return setup_ciphi_rhf!(ctx)
  else
    # UHF case: use spin-separated integrals
    return setup_ciphi_uhf!(ctx)
  end
end

"""
    trip_index(p, q) -> Int

Compute unique index for orbital pair (p, q) with p < q.
"""
function trip_index(p, q)
  @assert_devel p < q "trip_index requires p < q"
  return p + (q - 1) * (q - 2) ÷ 2
end

"""
    trip_index(p, q, n) -> Int

Compute unique index for orbital pair (p, q),
with n orbitals per spin.
"""
function trip_index(p, q, n)
  return p + (q - 1) * n
end

function doubles_denom(int2, i, j, a, b)
  denom = int2[i, j, i, j] - int2[i, j, j, i] +
          int2[a, b, a, b] - int2[a, b, b, a] -
          int2[a, j, a, j] + int2[a, j, j, a] -
          int2[b, i, b, i] + int2[b, i, i, b] -
          int2[a, i, a, i] + int2[a, i, i, a] -
          int2[b, j, b, j] + int2[b, j, j, b]
  return denom
end

function doubles_denom_ab(int2ab, int2aa, int2bb, iα, iβ, aα, aβ)
  denom = int2ab[iα, iβ, iα, iβ] +
          int2ab[aα, aβ, aα, aβ] -
          int2ab[aα, iβ, aα, iβ] -
          int2ab[iα, aβ, iα, aβ] -
          int2aa[aα, iα, aα, iα] + int2aa[aα, iα, iα, aα] -
          int2bb[aβ, iβ, aβ, iβ] + int2bb[aβ, iβ, iβ, aβ]
  return denom
end


function gen_pchb_list(n_orb::Int, int2::AbstractArray{T,4}, ThrNeglect::Float64=1e-10, use_mp2_denom::Bool=false) where T
  double_exc_lists = Vector{PCHBEntry{T}}[]
  h_doub_max = 0.0
  
  # Loop over all pairs of orbitals {p, q}
  for q in 2:n_orb
    for p in 1:(q-1)  # Only consider p < q to avoid duplicates
      # List of triplets {r, s, |H(rs ← pq)|} for this (p,q) pair
      entries = PCHBEntry[]
      
      # Loop over all distinct pairs of orbitals {r, s} that don't include {p, q}
      for s in 2:n_orb
        if s == p || s == q
          continue
        end
        for r in 1:(s-1)  # Only consider r < s to avoid duplicates
          if r == p || r == q
            continue
          end
          
          # Compute antisymmetrized two-electron integral <pq||rs>
          # Matrix element for double excitation p,q → r,s is v_pq^rs - v_pq^sr
          h_val = int2[r, s, p, q] - int2[r, s, q, p]

          if abs(h_val) > ThrNeglect # Skip negligible matrix elements
            if use_mp2_denom
              denom = zero(T)
            else
              denom = doubles_denom(int2, p, q, r, s)
            end
            h_val_dagger = int2[p, q, r, s] - int2[p, q, s, r]
            push!(entries, PCHBEntry{T}(r, s, h_val, h_val_dagger, denom))
            h_doub_max = max(h_doub_max, abs(h_val))
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(entries, by=x->abs(x.value), rev=true)
      
      # Store sorted list for this (p,q) pair
      push!(double_exc_lists, entries)
    end
  end
  return double_exc_lists, h_doub_max
end

function gen_pchb_list_ab(n_orb::Int, int2ab::AbstractArray{T,4}, int2aa::AbstractArray{T,4}, 
                          int2bb::AbstractArray{T,4}, ThrNeglect::Float64=1e-10, use_mp2_denom::Bool=false) where T
  double_exc_ab_lists = Vector{PCHBEntry{T}}[]
  h_doub_max = 0.0
  
  # Loop over all pairs of orbitals {p, q}
  # For mixed excitations, we don't need antisymmetrization (different spins)
  for q in 1:n_orb
    for p in 1:n_orb
      entries = PCHBEntry{T}[]
      for r in 1:n_orb
        if r == p; continue; end  # Alpha r cannot equal alpha p
        for s in 1:n_orb
          if s == q; continue; end  # Beta s cannot equal beta q
          
          # Mixed integral v_pq^rs (αβ) (no antisymmetrization for different spins)
          h_val = int2ab[r, s, p, q]
          if abs(h_val) > ThrNeglect
            if use_mp2_denom
              denom = zero(T)
            else
              denom = doubles_denom_ab(int2ab, int2aa, int2bb, p, q, r, s)
            end
            h_val_dagger = int2ab[p, q, r, s]
            push!(entries, PCHBEntry{T}(r, s, h_val, h_val_dagger, denom))
            h_doub_max = max(h_doub_max, abs(h_val))
          end
        end
      end
      
      # Sort triplets by |H| in decreasing order
      sort!(entries, by=x->abs(x.value), rev=true)
      push!(double_exc_ab_lists, entries)
    end
  end
  return double_exc_ab_lists, h_doub_max
end

function gen_singles_denom(int2::AbstractArray{T,4}) where T
  n_orb = size(int2, 1)
  denom = zeros(T, n_orb, n_orb)
  @inbounds for i in 2:n_orb
    for j in 1:i-1
      jij = int2[i, j, i, j] - int2[i, j, j, i]  # v_ij^ij - v_ij^ji
      denom[i, j] = -jij
      denom[j, i] = -jij
    end
  end
  return denom
end

"""
    setup_ciphi_rhf!(ctx::Union{FCIContext, CIPHIContext}) -> CIPHISetupData

Setup for RHF systems using spatial orbital integrals.
"""
function setup_ciphi_rhf!(ctx::Union{FCIContext{O,T}, CIPHIContext{O,T}}) where {O, T}
  n_orb = ctx.n_orb
  int2 = ctx.fcidump.int2
  thr_negligible = ctx.options.thr_negligible
  use_mp2_denom = false
  if ctx isa CIPHIContext
    use_mp2_denom = ctx.options.use_mp2
  end
  # Dictionary to store sorted lists for each (p,q) pair
  double_exc_lists, h_doub_max = gen_pchb_list(n_orb, int2, thr_negligible, use_mp2_denom)
  double_exc_ab_lists, h_doub_max_ab = gen_pchb_list_ab(n_orb, int2, int2, int2, thr_negligible, use_mp2_denom)
  h_doub_max = max(h_doub_max, h_doub_max_ab)
  if use_mp2_denom
    sdenom = zeros(T, n_orb, n_orb)
  else
    sdenom = gen_singles_denom(int2)
  end
  return CIPHISetupData(double_exc_lists, double_exc_ab_lists, h_doub_max, sdenom)
end

"""
    setup_ciphi_uhf!(ctx::Union{FCIContext, CIPHIContext}) -> CIPHISetupData

Setup for UHF systems using spin-separated integrals.
Handles three types of double excitations:
- Alpha-alpha (using int2aa)
- Beta-beta (using int2bb)
- Mixed alpha-beta (using int2ab)
"""
function setup_ciphi_uhf!(ctx::Union{FCIContext{O,T}, CIPHIContext{O,T}}) where {O, T}
  n_orb = ctx.n_orb
  int2aa = ctx.fcidump.int2aa
  int2bb = ctx.fcidump.int2bb
  int2ab = ctx.fcidump.int2ab
  thr_negligible = ctx.options.thr_negligible 
  use_mp2_denom = false
  if ctx isa CIPHIContext
    use_mp2_denom = ctx.options.use_mp2
  end
  # Three dictionaries for the three types of double excitations
  double_exc_aa, h_doub_max_aa = gen_pchb_list(n_orb, int2aa, thr_negligible, use_mp2_denom)
  double_exc_bb, h_doub_max_bb = gen_pchb_list(n_orb, int2bb, thr_negligible, use_mp2_denom)
  double_exc_ab, h_doub_max_ab = gen_pchb_list_ab(n_orb, int2ab, int2aa, int2bb, thr_negligible, use_mp2_denom)
  h_doub_max = max(h_doub_max_aa, h_doub_max_bb, h_doub_max_ab)
  if use_mp2_denom
    sdenom_a = zeros(T, n_orb, n_orb)
    sdenom_b = zeros(T, n_orb, n_orb)
  else
    sdenom_a = gen_singles_denom(int2aa)
    sdenom_b = gen_singles_denom(int2bb)
  end
  return CIPHISetupData(double_exc_aa, double_exc_bb, double_exc_ab, h_doub_max, sdenom_a, sdenom_b)
end


"""
    ExcVals

Holds excitation values: coefficient and Hamiltonian matrix element required to calculate 
the contribution to the perturbative energy.
"""
struct ExcVals{T}
  coef::T
  hval::T
end

function Base.:+(ev1::ExcVals{T}, ev2::ExcVals{T}) where T
  return ExcVals{T}(ev1.coef + ev2.coef, ev1.hval + ev2.hval)
end