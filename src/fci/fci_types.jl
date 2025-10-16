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
