"""
    FockDiagonal

Holds diagonal of the Fock matrix for alpha and beta spins.
"""
struct FockDiagonal
  """ alpha f_p^p vector """
  alpha::Vector{Float64}
  """ beta f_P^P vector """
  beta::Vector{Float64}
end

function FockDiagonal(n_orb::Int)
  alpha = zeros(Float64, n_orb)
  beta = zeros(Float64, n_orb)
  return FockDiagonal(alpha, beta)
end

"""
    calc_fock_diagonal4det!(fock::FockDiagonal, ctx::Union{FCIContext, CIPHIContext}, occa, occb)

Compute diagonal of the Fock matrix for a given determinant defined by occupied orbitals `occa` and `occb`.

`fock` is modified in-place and returned.
"""
function calc_fock_diagonal4det!(fock::FockDiagonal, ctx::Union{FCIContext, CIPHIContext}, occa, occb)
  for m in eachindex(fock.alpha)
    fock.alpha[m] = compute_fock_element(ctx.int1a, ctx.heval_data.h1e2_aa, ctx.heval_data.h1e2_ab, occa, occb, m, m)
    fock.beta[m] = compute_fock_element(ctx.int1b, ctx.heval_data.h1e2_bb, ctx.heval_data.h1e2_ba, occb, occa, m, m)
  end
  return fock
end

"""
    sum_h1e2(h1e2, occ::AbstractVector, a, i) -> Float64

Compute Σ_j h1e2[j, a, i] over occupied orbitals j.
"""
@pib function sum_h1e2(h1e2, occ::AbstractVector, a, i)
  total = zero(eltype(h1e2))
  @inbounds @simd for j in occ
    total += h1e2[j, a, i]
  end
  return total
end

"""
    compute_fock_element(int1, h1e2_same, h1e2_opp, occ_same::AbstractVector, occ_opp::AbstractVector,
                        a::Int, i::Int) -> Float64

Compute Fock matrix element f_ai
f_ai = h_ai + Σ_j (v_aijj - v_ajji)_SS + Σ_j (v_aijj)_OS
where SS = same spin, OS = opposite spin. 
"""
@pib function compute_fock_element(int1, h1e2_same, h1e2_opp, occ_same::AbstractVector, occ_opp::AbstractVector,
                              a::Int, i::Int)
  # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
  return int1[a, i] + sum_h1e2(h1e2_same, occ_same, a, i) + sum_h1e2(h1e2_opp, occ_opp, a, i)
end

"""
    sum_h1e2(h1e2, str::OPattern, a, i) where OPattern -> Float64

Compute Σ_j h1e2[j, a, i] over occupied orbitals j.
"""
@pib function sum_h1e2(h1e2, str::OPattern, a, i) where OPattern
  total = zero(eltype(h1e2))
  @inbounds @simd for k in axes(h1e2, 1)
    if (str >>> (k-1)) & one(str) != zero(str)
      total += h1e2[k, a, i]
    end
  end
  return total
end
"""
    compute_fock_element(int1, h1e2_same, h1e2_opp, str_same::OPattern, str_opp::OPattern,
                        a::Int, i::Int) where OPattern -> Float64

Compute Fock matrix element f_ai
f_ai = h_ai + Σ_j (v_aijj - v_ajji)_SS + Σ_j (v_aijj)_OS
where SS = same spin, OS = opposite spin. 
"""
@pib function compute_fock_element(int1, h1e2_same, h1e2_opp, str_same::OPattern, str_opp::OPattern,
                              a::Int, i::Int) where OPattern
  # f_ai = h1_ai + Σ_j_same h1e2_same[j,a,i] + Σ_j_opp h1e2_ab[j,a,i]
  return int1[a, i] + sum_h1e2(h1e2_same, str_same, a, i) + sum_h1e2(h1e2_opp, str_opp, a, i)
end