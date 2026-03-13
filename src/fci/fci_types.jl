const Address = UInt64          # addressing patterns and vector elements

const FCIUInt = UInt32

# Utility functions for bit manipulation

"""
    string_parity_before_pos(pat::OPattern, ipos::Integer) where OPattern -> UInt

Return 0 if number of bits SET in bits [0..ipos) is even, 1 if it is odd.
"""
function string_parity_before_pos(pat::OPattern, ipos::Integer)::UInt where OPattern
  # mask out bits at ipos and above
  tmp = pat & ((OPattern(1) << ipos) - 1)

  # count bit parity
  tmp ⊻= (tmp >> 32)  # only last 16 binary digits used
  tmp ⊻= (tmp >> 16)  # only last 8 binary digits used  
  tmp ⊻= (tmp >> 8)   # ...
  tmp ⊻= (tmp >> 4)
  tmp ⊻= (tmp >> 2)
  return UInt((tmp ⊻ (tmp >> 1)) & 1)
end

"""
    binomial_coefficient(N::Integer, k::Integer) -> UInt64

Calculate binomial coefficient C(N,k) = N! / (k! * (N-k)!)
"""
function binomial_coefficient(N::Integer, k::Integer)::UInt64
  if k > N || k < 0
    return UInt64(0)
  end

  result = UInt64(1)
  denominator = UInt64(1)

  for i in 0:(k - 1)
    result *= UInt64(N - i)
    denominator *= UInt64(i + 1)
  end

  return result ÷ denominator
end

"""
    sym_dof(N::Integer, ndim::Integer) -> UInt64

Number of symmetric degrees of freedom for dimension ndim.
"""
function sym_dof(N::Integer, ndim::Integer)::UInt64
  return binomial_coefficient(N, ndim)
end

"""
    fmt_pat(pat::OPattern, n_max_orb::Integer) where OPattern -> String

Format orbital pattern as string.
"""
function fmt_pat(pat::OPattern, n_max_orb::Integer)::String where OPattern
  result = ""
  for i in 0:(n_max_orb - 1)
    if (pat & (OPattern(1) << i)) != 0
      result *= "1"
    else
      result *= "0"
    end
  end
  return reverse(result)  # Most significant bit first
end

"""
    fmt_det(pat_a::OPattern, pat_b::OPattern, n_max_orb::Integer) where OPattern -> String

Format determinant (alpha and beta patterns) as string.
"""
function fmt_det(pat_a::OPattern, pat_b::OPattern, n_max_orb::Integer)::String where OPattern
  return fmt_pat(pat_a, n_max_orb) * "|" * fmt_pat(pat_b, n_max_orb)
end

using ..ElemCo.AbstractEC: AbstractDeterminant

"""
    Determinant

Represents a single determinant with alpha and beta orbital occupation patterns.
Used for selected space determinant storage and manipulation.
"""
struct Determinant{OPattern} <: AbstractDeterminant
  alpha::OPattern    # Alpha electron orbital pattern
  beta::OPattern     # Beta electron orbital pattern
end

Determinant{OPattern}() where OPattern = Determinant(OPattern(0), OPattern(0))
function Determinant{OPattern}(occa::Union{AbstractArray, UnitRange}, occb::Union{AbstractArray, UnitRange}) where OPattern
  alpha = OPattern(0)
  beta = OPattern(0)
  for i in occa
    alpha |= OPattern(1) << (i - 1)
  end
  for i in occb
    beta |= OPattern(1) << (i - 1)
  end
  return Determinant(alpha, beta)
end

"""
    OrbSpaces

Holds occupied and virtual orbital indices for alpha and beta spins.
"""
struct OrbSpaces{A <: AbstractVector{Int}}
  occa::A
  virta::A
  occb::A
  virtb::A
  norb::Int
end

OrbSpaces() = OrbSpaces(Int[], Int[], Int[], Int[], 0)

"""
    OrbSpaces(n_orb::Int) -> OrbSpaces

Create OrbSpaces with pre-allocated buffers for occupied and virtual orbitals.
"""
function OrbSpaces(n_orb::Int)
  occa = BufVec(zeros(Int, n_orb))
  virta = BufVec(zeros(Int, n_orb))
  occb = BufVec(zeros(Int, n_orb))
  virtb = BufVec(zeros(Int, n_orb))
  return OrbSpaces(occa, virta, occb, virtb, n_orb)
end

"""
    OrbSpaces(n_orb::Int, buf::AbstractVector{Int}) -> OrbSpaces

Create OrbSpaces using provided buffer for occupied and virtual orbitals.
Buffer size must be at least n_orb * 4.
"""
function OrbSpaces(n_orb::Int, buf::AbstractVector{Int})
  @assert length(buf) >= n_orb * 4 "Buffer size insufficient for OrbSpaces"
  occa = BufVec(@view(buf[1:n_orb]))
  virta = BufVec(@view(buf[n_orb+1:n_orb*2]))
  occb = BufVec(@view(buf[n_orb*2+1:n_orb*3]))
  virtb = BufVec(@view(buf[n_orb*3+1:n_orb*4]))
  return OrbSpaces(occa, virta, occb, virtb, n_orb)
end

"""
    PSpaceData

Container for P-space determinants, Hamiltonian matrix, and eigenvectors.
Contains all data needed for P-space enhanced initial guess generation.
"""
mutable struct PSpaceData{OPattern, T}
  determinants::Vector{Determinant{OPattern}}     # P-space determinants
  indices::Vector{Address}              # Indices of P-space dets in full space
  hamiltonian::Matrix{T}          # P-space Hamiltonian matrix H_ij
  eigenvalues::Vector{T}          # P-space eigenvalues
  eigenvectors::Matrix{T}         # P-space eigenvectors (columns)
  n_pspace::Int                        # Actual P-space size
  reference_det::Determinant{OPattern}        # HF reference determinant

  function PSpaceData{OPattern, T}() where {OPattern, T}
    new{OPattern, T}(
      Determinant{OPattern}[],
      Address[],
      Matrix{T}(undef, 0, 0),
      T[],
      Matrix{T}(undef, 0, 0),
      0,
      Determinant(OPattern(0), OPattern(0)),
    )
  end
end

PSpaceData{OPattern}() where OPattern = PSpaceData{OPattern, Float64}()

"""
    HEvalData

Precomputed data for evaluating diagonal Hamiltonian and single excitations elements.

1. The jkaa, jkbb, jab, ha, hb arrays store precomputed Coulomb and exchange contributions
   needed for efficient diagonal Hamiltonian element evaluation.

   - `jkaa[i, j] = 0.5 * [(ii|jj) - (ij|ji)]` for alpha spin
   - `jkbb[i, j] = 0.5 * [(ii|jj) - (ij|ji)]` for beta spin
   - `jab[i, j] = (ii|jj)` mixed alpha-beta
   - `ha[i] = ⟨i|h|i⟩` for alpha spin
   - `hb[i] = ⟨i|h|i⟩` for beta spin
2. The h1e2 arrays contain combinations of two-electron integrals that appear
   frequently in single excitation matrix elements and Fock matrix construction.
    For example, for alpha-alpha excitations:
   `h1e2[i, p, q] = v_{pi}^{qi} - v_{pi}^{iq} = (pq|ii) - (pi|iq)`

    These arrays enable efficient computation of single excitation matrix elements via `compute_fock_element`.
"""
struct HEvalData{T}
  # Precomputed data for evaluating diagonal Hamiltonian elements
  jkaa::Matrix{T}      # JKAA: 0.5[(ii|jj) - (ij|ji)]
  jkbb::Matrix{T}      # JKBB: 0.5[(ii|jj) - (ij|ji)] for beta
  jab::Matrix{T}      # JAB: (ii|jj) mixed
  ha::Vector{T}   # ⟨i|h|i⟩ for alpha
  hb::Vector{T}   # ⟨i|h|i⟩ for beta

  # Precomputed h1e2 terms for efficient Fock element calculation.
  h1e2_aa::Array{T, 3}       # RHF and UHF: alpha-alpha
  h1e2_bb::Array{T, 3}       # UHF: beta-beta (for RHF: reference to h1e2_aa)
  h1e2_ab::Array{T, 3}       # UHF and RHF: alpha-beta (no exchange)
  h1e2_ba::Array{T, 3}       # UHF: beta-alpha (no exchange) (for RHF: reference to h1e2_ab)

  is_uhf::Bool
  n_orb::Int
  spaces_buf::OrbSpaces           # Buffer for indices for diagonal element calculations

  function HEvalData{T}() where T
    mat = zeros(T, 0, 0)
    ten = zeros(T, 0, 0, 0)
    new{T}(mat, mat, mat, T[], T[],
        ten, ten, ten, ten, false, 0, OrbSpaces())
  end
  
  # RHF constructor
  function HEvalData(jk::Matrix{T}, jab::Matrix{T}, ha::Vector{T}, 
                     h1e2::Array{T, 3}, h1e2_ab::Array{T, 3}) where T
    n_orb = size(jk, 1)
    new{T}(jk, jk, jab, ha, ha,
        h1e2, h1e2, h1e2_ab, h1e2_ab,
        false, n_orb, OrbSpaces(n_orb))
  end
  
  # UHF constructor
  function HEvalData(jkaa::Matrix{T}, jkbb::Matrix{T}, jab::Matrix{T},
                     ha::Vector{T}, hb::Vector{T},
                     h1e2_aa::Array{T, 3}, h1e2_bb::Array{T, 3},
                     h1e2_ab::Array{T, 3}, h1e2_ba::Array{T, 3}) where T
    n_orb = size(jkaa, 1)
    new{T}(jkaa, jkbb, jab, ha, hb, 
        h1e2_aa, h1e2_bb, h1e2_ab, h1e2_ba, 
        true, n_orb, OrbSpaces(n_orb))
  end
end

HEvalData() = HEvalData{Float64}()

function HEvalData(int2e::AbstractArray{T, 4}, core_h::AbstractArray{T, 2}) where T
  # RHF case
  n_orb = size(core_h, 1)
  
  # Precompute JK and JAB matrices
  jk = get_diagonal_pair_antisym_ints(int2e)
  jab = get_diagonal_pair_ints(int2e)
  
  # One-electron integrals
  ha = diag(core_h)

  # Precompute h1e2 arrays
  h1e2 = zeros(T, n_orb, n_orb, n_orb)
  h1e2_ab = zeros(T, n_orb, n_orb, n_orb)
  for i in 1:n_orb
    for p in 1:n_orb, q in 1:n_orb
      h_piqi = int2e[p,i,q,i]
      h1e2[i,p,q] = h_piqi - int2e[p,i,i,q]
      h1e2_ab[i,p,q] = h_piqi
    end
  end
  return HEvalData(jk, jab, ha, h1e2, h1e2_ab)
end

function HEvalData(int2e_aa::AbstractArray{T, 4}, int2e_bb::AbstractArray{T, 4}, int2e_ab::AbstractArray{T, 4}, 
                   core_h_a::AbstractArray{T, 2}, core_h_b::AbstractArray{T, 2}) where T
  # UHF case
  n_orb = size(core_h_a, 1)
  
  # Precompute JK and JAB matrices
  jkaa = get_diagonal_pair_antisym_ints(int2e_aa)
  jkbb = get_diagonal_pair_antisym_ints(int2e_bb)
  jab = get_diagonal_pair_ints(int2e_ab)
  
  # One-electron integrals
  ha = diag(core_h_a)
  hb = diag(core_h_b)
  
  # Precompute h1e2 arrays
  h1e2_aa = zeros(T, n_orb, n_orb, n_orb)
  h1e2_bb = zeros(T, n_orb, n_orb, n_orb)
  h1e2_ab = zeros(T, n_orb, n_orb, n_orb)
  h1e2_ba = zeros(T, n_orb, n_orb, n_orb)
  for i in 1:n_orb
    for p in 1:n_orb, q in 1:n_orb
      # Alpha-alpha
      h1e2_aa[i,p,q] = int2e_aa[p,i,q,i] - int2e_aa[p,i,i,q]
      # Beta-beta
      h1e2_bb[i,p,q] = int2e_bb[p,i,q,i] - int2e_bb[p,i,i,q]
      # Alpha-beta
      h1e2_ab[i,p,q] = int2e_ab[p,i,q,i]
      h1e2_ba[i,p,q] = int2e_ab[i,p,i,q]
    end
  end
  return HEvalData(jkaa, jkbb, jab, ha, hb,
                   h1e2_aa, h1e2_bb, h1e2_ab, h1e2_ba)
end