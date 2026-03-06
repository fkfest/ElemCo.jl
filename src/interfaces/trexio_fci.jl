# ============================================================================
# Determinant I/O for CIPHI/FCI wave functions
# ============================================================================
# This file is included inside TrexioInterface module

"""
    SimpleDeterminant{OPattern}

Simple determinant type for TREXIO I/O. Holds alpha and beta occupation patterns.
This is returned by `read_trexio_determinants` when reading from files.
Can be converted to/from FCI.Determinant when FCI module is available.
"""
struct SimpleDeterminant{OPattern} <: AbstractDeterminant
  alpha::OPattern
  beta::OPattern
end

export SimpleDeterminant

"""
    write_trexio_determinants(trexio::TrexioFile, determinants::Vector{<:AbstractDeterminant}, 
                              coefficients::AbstractVector{<:Number})

Write determinants and CI coefficients to TREXIO file using extended format with 
separate alpha/beta bit fields.

# Arguments
- `trexio::TrexioFile`: Open TREXIO file handle (must be opened with write/update mode)
- `determinants::Vector{<:AbstractDeterminant}`: Vector of determinants (must have `alpha` and `beta` fields)
- `coefficients::AbstractVector{Float64}`: CI coefficients for each determinant

# Storage Format
- `determinant.num`: Number of determinants
- `determinant.n_int`: Number of 64-bit integers per spin pattern (ceil(mo.num/64))
- `determinant.alpha`: Alpha spin patterns as Int64 bit fields [n_int, n_dets]
- `determinant.beta`: Beta spin patterns as Int64 bit fields [n_int, n_dets]
- `determinant.coefficient`: CI coefficients [n_dets]

The number of orbitals is determined from `mo.num` in the TREXIO file.
Bit `i` in the pattern indicates orbital `i+1` is occupied (0-indexed bits).

# Example
```julia
using ElemCo
open_trexio("output.h5", "w") do trexio
    # ... write orbitals first to set mo.num ...
    write_trexio_determinants(trexio, variational_dets, ci_coefficients)
end
```
"""
function write_trexio_determinants(trexio::TrexioFile, 
                                   determinants::Vector{D}, 
                                   coefficients::AbstractVector{<:Number}) where {D <: AbstractDeterminant}
  n_dets = length(determinants)
  @assert n_dets == length(coefficients) "Number of determinants ($n_dets) must match coefficients ($(length(coefficients)))"
  
  if n_dets == 0
    return TREXIO.TREXIO_SUCCESS
  end
  
  # Get mo.num to determine n_int
  n_orb, status = TREXIO.trexio_read_mo_num(trexio)
  if status != TREXIO.TREXIO_SUCCESS
    error("mo.num must be written before determinants (status: $status)")
  end
  
  # Calculate n_int based on number of orbitals
  n_int = cld(n_orb, 64)  # ceiling division
  
  # Verify pattern type is large enough (get type from first determinant)
  OPattern = typeof(determinants[1].alpha)
  pattern_bits = sizeof(OPattern) * 8
  if n_orb > pattern_bits
    error("OPattern type ($OPattern with $pattern_bits bits) too small for $n_orb orbitals")
  end
  
  # Write number of determinants
  status = TREXIO.trexio_write_determinant_num(trexio, n_dets)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write determinant.num with status $status"
  
  # Write n_int
  status = TREXIO.trexio_write_determinant_n_int(trexio, n_int)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write determinant.n_int with status $status"
  
  # Pack determinants into Int64 arrays
  alpha_patterns = zeros(Int64, n_int, n_dets)
  beta_patterns = zeros(Int64, n_int, n_dets)
  
  for (i, det) in enumerate(determinants)
    _pack_pattern!(view(alpha_patterns, :, i), det.alpha, n_int)
    _pack_pattern!(view(beta_patterns, :, i), det.beta, n_int)
  end
  
  # Write alpha and beta patterns
  status = TREXIO.trexio_write_determinant_alpha(trexio, alpha_patterns)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write determinant.alpha with status $status"
  
  status = TREXIO.trexio_write_determinant_beta(trexio, beta_patterns)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write determinant.beta with status $status"
  
  # Write coefficients using standard TREXIO field
  status = TREXIO.trexio_write_determinant_coefficient(trexio, Vector{Float64}(coefficients))
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write determinant.coefficient with status $status"
  
  return TREXIO.TREXIO_SUCCESS
end

"""
    _pack_pattern!(dest::AbstractVector{Int64}, pattern::OPattern, n_int::Int) where OPattern

Pack an orbital pattern into an array of Int64 values.
Orbital `i` (1-indexed) is stored in bit `(i-1) % 64` of integer `(i-1) ÷ 64 + 1`.
"""
function _pack_pattern!(dest::AbstractVector{Int64}, pattern::OPattern, n_int::Int) where OPattern
  p = UInt128(pattern)  # promote to large enough type for shifting
  for k in 1:n_int
    dest[k] = Int64(p & 0xffffffffffffffff)
    p >>= 64
  end
end

"""
    _unpack_pattern(src::AbstractVector{Int64}, ::Type{OPattern}) where OPattern -> OPattern

Unpack an array of Int64 values into an orbital pattern.
"""
function _unpack_pattern(src::AbstractVector{Int64}, ::Type{OPattern}) where OPattern
  result = OPattern(0)
  for k in length(src):-1:1
    result = (result << 64) | OPattern(UInt64(src[k]))
  end
  return result
end

"""
    read_trexio_determinants(trexio::TrexioFile; OPattern::Type=UInt64) 
      -> (Vector{SimpleDeterminant{OPattern}}, Vector{Float64})

Read determinants and CI coefficients from TREXIO file.

# Arguments
- `trexio::TrexioFile`: Open TREXIO file handle
- `OPattern::Type`: Orbital pattern type (default: UInt64, use UInt128 for >64 orbitals)

# Returns
- `determinants::Vector{SimpleDeterminant{OPattern}}`: Vector of determinants
- `coefficients::Vector{Float64}`: CI coefficients

# Example
```julia
using ElemCo
open_trexio("input.h5", "r") do trexio
    dets, coeffs = read_trexio_determinants(trexio)
    # For systems with >64 orbitals:
    # dets, coeffs = read_trexio_determinants(trexio; OPattern=UInt128)
end
```
"""
function read_trexio_determinants(trexio::TrexioFile; OPattern::Type=UInt64)
  # Check if determinants exist using extended format
  if !TREXIO.trexio_has_determinant_alpha(trexio)
    return SimpleDeterminant{OPattern}[], Float64[]
  end
  
  # Read n_int
  n_int, status = TREXIO.trexio_read_determinant_n_int(trexio)
  if status != TREXIO.TREXIO_SUCCESS
    error("Failed to read determinant.n_int with status $status")
  end
  
  # Verify OPattern is large enough
  pattern_bits = sizeof(OPattern) * 8
  required_bits = n_int * 64
  if required_bits > pattern_bits
    error("OPattern type ($OPattern with $pattern_bits bits) too small for n_int=$n_int ($required_bits bits required). Use UInt128 or larger.")
  end
  
  # Read number of determinants
  n_dets, status = TREXIO.trexio_read_determinant_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read determinant.num with status $status"
  
  # Read alpha and beta patterns
  alpha_patterns, status = TREXIO.trexio_read_determinant_alpha(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read determinant.alpha with status $status"
  
  beta_patterns, status = TREXIO.trexio_read_determinant_beta(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read determinant.beta with status $status"
  
  # Read coefficients
  coefficients, status = TREXIO.trexio_read_determinant_coefficient(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read determinant.coefficient with status $status"
  
  # Unpack determinants
  determinants = Vector{SimpleDeterminant{OPattern}}(undef, n_dets)
  for i in 1:n_dets
    alpha = _unpack_pattern(view(alpha_patterns, :, i), OPattern)
    beta = _unpack_pattern(view(beta_patterns, :, i), OPattern)
    determinants[i] = SimpleDeterminant(alpha, beta)
  end
  
  return determinants, coefficients
end

"""
    has_trexio_determinants(trexio::TrexioFile) -> Bool

Check if determinants are stored in the TREXIO file (extended format with separate alpha/beta).

Returns `true` if determinant data with extended alpha/beta format is found.
"""
function has_trexio_determinants(trexio::TrexioFile)
  return TREXIO.trexio_has_determinant_alpha(trexio) && 
         TREXIO.trexio_has_determinant_beta(trexio) &&
         TREXIO.trexio_has_determinant_n_int(trexio)
end
