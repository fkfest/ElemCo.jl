@testitem "pm_store" tags=[:cc, :quick] begin
# Phase-0 harness for the PM (±) supermatrix AO-integral store — see dev/pm_ao_store_plan.md §2.
# Pins the normative conventions the persisted-± kernels will rely on:
#   Vs[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ + ⟨νμ|ρσ⟩   (row diagonal μ=ν carries ×2)
#   Va[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ − ⟨νμ|ρσ⟩   (row/col diagonals = 0)
# via `calc_tri_sym_antisym!`, and the contraction identity
#   K[μ,ν] = Σ_ρσ ⟨μν|ρσ⟩ D[ρ,σ] = (Vs·Ds + Va·Da)[tri(μν)],  K[ν,μ] = (Vs·Ds − Va·Da)[tri(μν)]
# with Ds/Da the ½-diagonal ±-folded density. Every check runs for Float64 AND ComplexF64
# (real supermatrices are symmetric, complex ones Hermitian — the derivation uses only the
# conjugation-free exchange symmetry, so the identity holds verbatim over ℂ).
using ElemCo
using ElemCo.QMTensors: uppertriangular_index, calc_tri_sym_antisym!
using ElemCo.TensorTools: detri_int2, @tensor
using ElemCo.MSystems: parse_geometry
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

@testset "PM convention harness (dense reference)" begin
  Random.seed!(20260720)
  for T in (Float64, ComplexF64), n in (6, 9, 13)
    ntri = n*(n+1)÷2
    int2 = exch_int2(n, T)                                 # physical (exchange-valid) packed AO ints
    Vs = zeros(T, ntri, ntri); Va = zeros(T, ntri, ntri)
    calc_tri_sym_antisym!(Vs, Va, int2)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)            # dense ⟨μν|ρσ⟩ (exchange applied)

    # (a) inversion: recover the stored μν/νμ slabs from Vs/Va
    maxinv = 0.0
    for σ in 1:n, ρ in 1:σ
      col = uppertriangular_index(ρ, σ)
      for ν in 1:n, μ in 1:ν
        row = uppertriangular_index(μ, ν)
        if μ == ν
          maxinv = max(maxinv, abs(Vs[row,col]/2 - int2[μ,μ,col]))
        else
          maxinv = max(maxinv, abs((Vs[row,col]+Va[row,col])/2 - int2[μ,ν,col]))
          maxinv = max(maxinv, abs((Vs[row,col]-Va[row,col])/2 - int2[ν,μ,col]))
        end
      end
    end
    @test maxinv < 1e-13

    # (b) contraction identity §2.3 vs a dense reference with a NONSYMMETRIC density
    D = randn(T, n, n)
    @tensor Kref[μ,ν] := G[μ,ν,ρ,σ] * D[ρ,σ]
    Ds, Da = pack_pm_density(D)
    VsDs = Vs * Ds; VaDa = Va * Da
    Kpm = zeros(T, n, n)
    for ν in 1:n, μ in 1:ν
      row = uppertriangular_index(μ, ν)
      if μ == ν
        Kpm[μ,μ] = VsDs[row]
      else
        Kpm[μ,ν] = VsDs[row] + VaDa[row]
        Kpm[ν,μ] = VsDs[row] - VaDa[row]
      end
    end
    @test maximum(abs.(Kpm .- Kref)) < 1e-13

    # (c) supermatrix symmetry/hermiticity for a physical (hermitian) int2: V = V'
    inth = herm_int2(int2)
    Vsh = zeros(T, ntri, ntri); Vah = zeros(T, ntri, ntri)
    calc_tri_sym_antisym!(Vsh, Vah, inth)
    @test maximum(abs.(Vsh .- Vsh')) < 1e-13               # ' = adjoint: symmetric (real) / Hermitian (ℂ)
    @test maximum(abs.(Vah .- Vah')) < 1e-13
  end
end

# Baseline: the existing on-the-fly ± kext (cc.use_pm_kext) must agree with the standard
# GEMM kext on the AO-direct path — documents correctness of the reused D-prep/scatter
# before the persisted store replaces the per-iteration ± build (plan Phase 0, gate).
@testset "AO-direct kext: pm ↔ standard" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"
  fresh() = (e = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>"sto-3g")));
             e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)

  EC = fresh(); @ints; @hf; @set cc use_pm_kext=false
  e_std = @cc ccsd
  EC = fresh(); @ints; @hf; @set cc use_pm_kext=true
  e_pm = @cc ccsd
  @test abs(e_std["CCSD"] - e_pm["CCSD"]) < 1e-10
end

end # @testitem
