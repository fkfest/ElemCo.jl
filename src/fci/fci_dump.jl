"""
FCI_DUMP molecular data structure.
"""

"""
    FCIDump

Container for molecular data for FCI calculations.
TODO: Use QFDump instead!
"""
mutable struct FCIDump
  n_elec::Int
  n_orb::Int
  n_alpha::Int
  n_beta::Int
  n_spin::Int
  e_nuc::Scalar
  
  # RHF integrals (spatial orbitals)
  h1::Matrix{Scalar}
  h2::Array{Scalar, 4}
  
  # UHF integrals (spin-separated)
  h1a::Union{Matrix{Scalar}, Nothing}
  h1b::Union{Matrix{Scalar}, Nothing}
  h2aa::Union{Array{Scalar, 4}, Nothing}
  h2bb::Union{Array{Scalar, 4}, Nothing}
  h2ab::Union{Array{Scalar, 4}, Nothing}
  
  is_uhf::Bool
  orbital_energies::Vector{Scalar}

  function FCIDump()
    new(0, 0, 0, 0, 0, 0.0,
        zeros(Scalar, 0, 0), zeros(Scalar, 0, 0, 0, 0),
        nothing, nothing, nothing, nothing, nothing,
        false, Scalar[])
  end
end

"""
    get_diagonal_pair_ints_2e!(jaa::AbstractMatrix{Scalar}, kaa::AbstractMatrix{Scalar},
                               int2e::AbstractArray{Scalar}, n_pairs::Integer, n_orb::Integer)

Extract diagonal pair integrals for 2-electron terms.
"""
function get_diagonal_pair_ints_2e!(jaa::AbstractMatrix{Scalar}, kaa::AbstractMatrix{Scalar},
                                    int2e::AbstractArray{Scalar}, n_pairs::Integer, n_orb::Integer)
  @assert size(jaa) == (n_orb, n_orb) "jaa matrix wrong size"
  @assert size(kaa) == (n_orb, n_orb) "kaa matrix wrong size"

  if ndims(int2e) == 4
    @assert size(int2e, 1) == n_orb "int2e dimension mismatch"
    @assert size(int2e, 2) == n_orb "int2e dimension mismatch"
    @assert size(int2e, 3) == n_orb "int2e dimension mismatch"
    @assert size(int2e, 4) == n_orb "int2e dimension mismatch"

    @inbounds for i0 in 0:(n_orb - 1)
      i = i0 + 1
      for j0 in 0:i0
        j = j0 + 1

        kij = int2e[i, j, i, j]  # (ij|ij)
        jij = int2e[i, i, j, j]  # (ii|jj)

        kaa[i, j] = kij        # (ij|ij) - raw integral
        kaa[j, i] = kij        # (ij|ij) - symmetric

        jaa[i, j] = jij        # (ii|jj) - raw integral  
        jaa[j, i] = int2e[j, j, i, i]  # (jj|ii)
      end
    end
  else
    for i in 1:n_orb
      for j in 1:i
        ij = i * (i - 1) ÷ 2 + j
        ii = i * (i - 1) ÷ 2 + i
        jj = j * (j - 1) ÷ 2 + j

        val_k = int2e[ij, ij]  # (ij|ij)
        val_j_ii = int2e[ii, jj]  # (ii|jj)

        # Store corrected combination for same-spin 
        kaa[i, j] = val_k - val_j_ii  # (ij|ij) - (ii|jj)
        kaa[j, i] = kaa[i, j]

        # BUGFIX: Store (ij|ij) for opposite-spin, not (ii|jj)
        jaa[i, j] = val_k      # (ij|ij) - CORRECTED  
        jaa[j, i] = int2e[ji, ji]  # (ji|ji)
      end
    end
  end
end

"""
    get_diagonal_pair_ints_1e!(hbar::AbstractVector{Scalar}, kaa_matrix::AbstractMatrix{Scalar},
                               core_h::AbstractMatrix{Scalar}, n_orb::Integer, c1_integrals::Bool)

Extract diagonal pair integrals for 1-electron terms.
"""
function get_diagonal_pair_ints_1e!(hbar::AbstractVector{Scalar}, kaa_matrix::AbstractMatrix{Scalar},
                                    core_h::Union{AbstractMatrix{Scalar}, Nothing},
                                    n_orb::Integer, c1_integrals::Bool)
  if core_h === nothing
    # When absorb_1e is true, there are no 1e contributions to store
    fill!(hbar, 0.0)
    return
  end

  @assert length(hbar) == n_orb "hbar vector wrong size"

  for i in 1:n_orb
    hbar[i] = core_h[i, i]
    if !c1_integrals
      for k in 1:n_orb
        hbar[i] -= 0.5 * kaa_matrix[i, k]
      end
    end
  end
end
