# Shared preamble for the `pm_store*` test items (`include`d by each of them; NOT a test file
# itself — the runner only discovers `*_test.jl`). The ± store tests were one 960-line item, which
# made a focused run of them serial; they are now several items that schedule in parallel, and this
# holds what all of them need: the module imports and the synthetic-store helpers.

using ElemCo
using ElemCo.QMTensors: uppertriangular_index, calc_tri_sym_antisym!
using ElemCo.TensorTools: detri_int2, @tensor, @mview, newmmap, closemmap, mmap3idx, load4idx, load2idx
using ElemCo.PMStore
using ElemCo.FockFactory: ao_JK!, ao_J2K!, add_coulomb!, add_exchange!
using ElemCo.IntegralTools: transform_int2, transform_int2_Q, pm_transform_int2, pm_transform_int2_n5, pm_transform
using ElemCo.ECInfos: delete_file!, file_exists
using ElemCo.MSystems: parse_geometry
using ElemCo.BasisSets: generate_basis, n_ao
using ElemCo.Integrals: calc_2e4idx_tri!, eri_2e4idx_sph!, schwarz_bounds,
                        eri_2e4idx_tri_batch!, ket_shell_blocks, buffer_size_4idx
using ElemCo.IntegralTools: pm_integrals!, save_ao_1e_integrals!
using LinearAlgebra
using Random

# Fold a full density D[ρ,σ] into the ± triangular vectors (plan §2.3): the ½ on the
# ρ=σ diagonal is what makes the identity exact — do not touch it without re-deriving.
function pack_pm_density(D)
  n = size(D, 1); ntri = n*(n+1)÷2; T = eltype(D)
  Ds = zeros(T, ntri); Da = zeros(T, ntri)
  for σ in 1:n, ρ in 1:σ
    idx = uppertriangular_index(ρ, σ)
    if ρ == σ
      Ds[idx] = D[ρ,ρ] / 2
    else
      Ds[idx] = (D[ρ,σ] + D[σ,ρ]) / 2
      Da[idx] = (D[ρ,σ] - D[σ,ρ]) / 2
    end
  end
  return Ds, Da
end

# A random packed int2 with the ONE symmetry a *stored* physical AO integral must carry:
# the diagonal ket slabs (ρ=σ) are μν-symmetric — ⟨μν|ρρ⟩=⟨νμ|ρρ⟩ — so detri_int2 then
# reconstructs a fully exchange-symmetric ⟨μν|ρσ⟩=⟨νμ|σρ⟩ (off-diagonal slabs unconstrained).
# `transpose` not `adjoint`: exchange is the conjugation-free swap (plan §2.2).
function exch_int2(n, ::Type{T}) where T
  int2 = randn(T, n, n, n*(n+1)÷2)
  for ρ in 1:n
    d = uppertriangular_index(ρ, ρ)
    @views int2[:, :, d] .= (int2[:, :, d] .+ transpose(int2[:, :, d])) ./ 2
  end
  return int2
end

# The FULL physical symmetry: exchange (via exch_int2) AND hermiticity ⟨μν|ρσ⟩=conj(⟨ρσ|μν⟩).
# Used only for the supermatrix-hermiticity check; the other checks use exch_int2 directly.
function herm_int2(int2raw)
  n = size(int2raw, 1); T = eltype(int2raw)
  G = detri_int2(int2raw, n, 1:n, 1:n, 1:n, 1:n)          # exchange-symmetric dense
  Gh = (G .+ conj(permutedims(G, (3,4,1,2)))) ./ 2         # hermitize (keeps exchange sym)
  out = zeros(T, n, n, n*(n+1)÷2)
  for σ in 1:n, ρ in 1:σ
    @views out[:, :, uppertriangular_index(ρ, σ)] .= Gh[:, :, ρ, σ]
  end
  return out
end

# place a synthetic physical ao_int2 into a fresh EC's scratch and build the ± store from it
function build_store(n, ::Type{T}, maxcols; int2=nothing) where T
  EC = ElemCo.ECInfo{T}()
  int2 === nothing && (int2 = herm_int2(exch_int2(n, T)))  # default: exchange + hermitian
  f, arr = newmmap(EC, "ao_int2", (n, n, n*(n+1)÷2), T; description="int2 ao")
  arr .= int2; closemmap(EC, f, arr)
  pm_from_joint!(EC; maxcols=maxcols)
  return EC, int2
end
