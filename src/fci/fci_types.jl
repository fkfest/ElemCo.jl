const OrbPattern = UInt64       # orbital occupation patterns
const Address = UInt64          # addressing patterns and vector elements  
const Scalar = Float64          # scalar values for CI coefficients

const FCIUInt = UInt32

# Threshold for neglecting small values
const ThrNeglect = 1e-16

# Utility functions for bit manipulation

"""
    string_parity_before_pos(pat::OrbPattern, ipos::Integer) -> UInt

Return 0 if number of bits SET in bits [0..ipos) is even, 1 if it is odd.
"""
function string_parity_before_pos(pat::OrbPattern, ipos::Integer)::UInt
  # mask out bits at ipos and above
  tmp = pat & ((OrbPattern(1) << ipos) - 1)

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
    fmt_pat(pat::OrbPattern, n_max_orb::Integer) -> String

Format orbital pattern as string.
"""
function fmt_pat(pat::OrbPattern, n_max_orb::Integer)::String
  result = ""
  for i in 0:(n_max_orb - 1)
    if (pat & (OrbPattern(1) << i)) != 0
      result *= "1"
    else
      result *= "0"
    end
  end
  return reverse(result)  # Most significant bit first
end

"""
    fmt_det(pat_a::OrbPattern, pat_b::OrbPattern, n_max_orb::Integer) -> String

Format determinant (alpha and beta patterns) as string.
"""
function fmt_det(pat_a::OrbPattern, pat_b::OrbPattern, n_max_orb::Integer)::String
  return fmt_pat(pat_a, n_max_orb) * "|" * fmt_pat(pat_b, n_max_orb)
end

# ===========================================
# PSpace (P-space) Algorithm Data Structures
# ===========================================

"""
    Determinant

Represents a single determinant with alpha and beta orbital occupation patterns.
Used for selected space determinant storage and manipulation.
"""
struct Determinant
  alpha::OrbPattern    # Alpha electron orbital pattern
  beta::OrbPattern     # Beta electron orbital pattern

  Determinant(alpha::OrbPattern, beta::OrbPattern) = new(alpha, beta)
end

Determinant() = Determinant(OrbPattern(0), OrbPattern(0))
function Determinant(occa::Union{AbstractArray, UnitRange}, occb::Union{AbstractArray, UnitRange})
  alpha = OrbPattern(0)
  beta = OrbPattern(0)
  for i in occa
    alpha |= OrbPattern(1) << (i - 1)
  end
  for i in occb
    beta |= OrbPattern(1) << (i - 1)
  end
  return Determinant(alpha, beta)
end
"""
    PSpaceData

Container for P-space determinants, Hamiltonian matrix, and eigenvectors.
Contains all data needed for P-space enhanced initial guess generation.
"""
mutable struct PSpaceData
  determinants::Vector{Determinant}     # P-space determinants
  indices::Vector{Address}              # Indices of P-space dets in full space
  hamiltonian::Matrix{Scalar}          # P-space Hamiltonian matrix H_ij
  eigenvalues::Vector{Scalar}          # P-space eigenvalues
  eigenvectors::Matrix{Scalar}         # P-space eigenvectors (columns)
  n_pspace::Int                        # Actual P-space size
  reference_det::Determinant           # HF reference determinant

  function PSpaceData()
    new(
      Determinant[],
      Address[],
      Matrix{Scalar}(undef, 0, 0),
      Scalar[],
      Matrix{Scalar}(undef, 0, 0),
      0,
      Determinant(OrbPattern(0), OrbPattern(0)),
    )
  end
end

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
struct HEvalData
  # Precomputed data for evaluating diagonal Hamiltonian elements
  jkaa::Matrix{Scalar}      # JKAA: 0.5[(ii|jj) - (ij|ji)]
  jkbb::Matrix{Scalar}      # JKBB: 0.5[(ii|jj) - (ij|ji)] for beta
  jab::Matrix{Scalar}      # JAB: (ii|jj) mixed
  ha::Vector{Scalar}   # ⟨i|h|i⟩ for alpha
  hb::Vector{Scalar}   # ⟨i|h|i⟩ for beta

  # Precomputed h1e2 terms for efficient Fock element calculation.
  h1e2_aa::Array{Scalar, 3}       # RHF and UHF: alpha-alpha
  h1e2_bb::Array{Scalar, 3}       # UHF: beta-beta
  h1e2_ab::Array{Scalar, 3}       # UHF and RHF: alpha-beta (no exchange)
  h1e2_ba::Array{Scalar, 3}       # UHF: beta-alpha (no exchange)

  is_uhf::Bool
  n_orb::Int
  ibuf::Vector{Int}        # Buffer for indices

  function HEvalData()
    mat = zeros(Scalar, 0, 0)
    ten = zeros(Scalar, 0, 0, 0)
    new(mat, mat, mat, Scalar[], Scalar[],
        ten, ten, ten, ten, false, 0, Int[])
  end
  
  # RHF constructor
  function HEvalData(jk::Matrix{Scalar}, jab::Matrix{Scalar}, ha::Vector{Scalar}, 
                     h1e2::Array{Scalar, 3}, h1e2_ab::Array{Scalar, 3})
    n_orb = size(jk, 1)
    new(jk, jk, jab, ha, ha,
        h1e2, zeros(Scalar, 0, 0, 0), h1e2_ab, zeros(Scalar, 0, 0, 0),
        false, n_orb, zeros(Int, 4*n_orb))
  end
  
  # UHF constructor
  function HEvalData(jkaa::Matrix{Scalar}, jkbb::Matrix{Scalar}, jab::Matrix{Scalar},
                     ha::Vector{Scalar}, hb::Vector{Scalar},
                     h1e2_aa::Array{Scalar, 3}, h1e2_bb::Array{Scalar, 3},
                     h1e2_ab::Array{Scalar, 3}, h1e2_ba::Array{Scalar, 3})
    n_orb = size(jkaa, 1)
    new(jkaa, jkbb, jab, ha, hb, 
        h1e2_aa, h1e2_bb, h1e2_ab, h1e2_ba, 
        true, n_orb, zeros(Int, 4*n_orb))
  end
end

function HEvalData(int2e, core_h)
  # RHF case
  n_orb = size(core_h, 1)
  
  # Precompute JK and JAB matrices
  jk = get_diagonal_pair_antisym_ints(int2e)
  jab = get_diagonal_pair_ints(int2e)
  
  # One-electron integrals
  ha = diag(core_h)
  
  # Precompute h1e2 array for alpha-beta excitations
  h1e2 = zeros(Scalar, n_orb, n_orb, n_orb)
  h1e2_ab = zeros(Scalar, n_orb, n_orb, n_orb)
  for i in 1:n_orb
    for p in 1:n_orb, q in 1:n_orb
      h_pqii = int2e[p,q,i,i]
      h1e2[i,p,q] = h_pqii - int2e[p,i,i,q]
      h1e2_ab[i,p,q] = h_pqii
    end
  end
  return HEvalData(jk, jab, ha, h1e2, h1e2_ab)
end

function HEvalData(int2e_aa, int2e_bb, int2e_ab, core_h_a, core_h_b)
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
  h1e2_aa = zeros(Scalar, n_orb, n_orb, n_orb)
  h1e2_bb = zeros(Scalar, n_orb, n_orb, n_orb)
  h1e2_ab = zeros(Scalar, n_orb, n_orb, n_orb)
  h1e2_ba = zeros(Scalar, n_orb, n_orb, n_orb)
  for i in 1:n_orb
    for p in 1:n_orb, q in 1:n_orb
      # Alpha-alpha
      h1e2_aa[i,p,q] = int2e_aa[p,q,i,i] - int2e_aa[p,i,i,q]
      # Beta-beta
      h1e2_bb[i,p,q] = int2e_bb[p,q,i,i] - int2e_bb[p,i,i,q]
      # Alpha-beta
      h1e2_ab[i,p,q] = int2e_ab[p,q,i,i]
      h1e2_ba[i,p,q] = int2e_ab[i,i,p,q]
    end
  end
  return HEvalData(jkaa, jkbb, jab, ha, hb,
                   h1e2_aa, h1e2_bb, h1e2_ab, h1e2_ba)
end