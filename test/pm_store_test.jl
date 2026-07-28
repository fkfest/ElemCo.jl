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
using ElemCo.TensorTools: detri_int2, @tensor, @mview, newmmap, closemmap, mmap3idx, load4idx, load2idx
using ElemCo.PMStore
using ElemCo.FockFactory: ao_JK!, ao_J2K!, add_coulomb!, add_exchange!
using ElemCo.IntegralTools: transform_int2, transform_int2_Q, pm_transform_int2, pm_transform_int2_n5, pm_transform
using ElemCo.ECInfos: delete_file!, file_exists
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

# Phase-1 gate: the persisted ± store round-trips exactly. Build a physical (exchange +
# hermitian) int2, write it as "ao_int2", pm_from_joint!, reopen, reconstruct the full
# supermatrices from the stored lower block-triangle (+ Hermitian mirror) and compare to
# calc_tri_sym_antisym! of the joint tensor. Real AND complex; several block sizes.
# place a synthetic physical ao_int2 into a fresh EC's scratch and build the ± store from it
function build_store(n, ::Type{T}, maxcols; int2=nothing) where T
  EC = ElemCo.ECInfo{T}()
  int2 === nothing && (int2 = herm_int2(exch_int2(n, T)))  # default: exchange + hermitian
  f, arr = newmmap(EC, "ao_int2", (n, n, n*(n+1)÷2), T; description="int2 ao")
  arr .= int2; closemmap(EC, f, arr)
  pm_from_joint!(EC; maxcols=maxcols)
  return EC, int2
end

@testset "PM store round-trip (build ↔ reconstruct)" begin
  # reconstruct the full npp×npp Vs/Va from the stored panels (+ Hermitian mirror)
  function reconstruct(pm, ::Type{T}) where T
    Vs = zeros(T, pm.npp, pm.npp); Va = zeros(T, pm.npp, pm.npp)
    for J in 1:pm_nblocks(pm)
      cJ = pm.pairblocks[J]; r0 = first(cJ); lc = last(cJ)
      Ps = spanel(pm, J); Pa = apanel(pm, J)
      for (jj, c) in enumerate(cJ), (ii, r) in enumerate(r0:pm.npp)
        Vs[r,c] = Ps[ii,jj]; Va[r,c] = Pa[ii,jj]
        if r > lc                                          # sub-diagonal → fill upper mirror
          Vs[c,r] = conj(Ps[ii,jj]); Va[c,r] = conj(Pa[ii,jj])
        end
      end
    end
    return Vs, Va
  end

  # multi-block configs (real n⁴/4 saving) + one single-block edge (nb=1 ⇒ full square)
  for T in (Float64, ComplexF64), (n, maxcols) in ((8, 8), (12, 30), (15, 40), (10, 400))
    EC, int2 = build_store(n, T, maxcols)
    ntri = n*(n+1)÷2
    Vsr = zeros(T, ntri, ntri); Var = zeros(T, ntri, ntri)
    calc_tri_sym_antisym!(Vsr, Var, int2)                  # joint reference
    pm = open_pm_store(EC)
    @test pm.nao == n && pm.npp == ntri
    if pm_nblocks(pm) > 1
      @test length(pm.smap) < ntri^2                       # lower block-triangle saves storage
    else
      @test length(pm.smap) == ntri^2                      # single block = full square (correct)
    end
    Vs, Va = reconstruct(pm, T)
    @test maximum(abs.(Vs .- Vsr)) < 1e-13
    @test maximum(abs.(Va .- Var)) < 1e-13
    # physicist spot-check: unpack ⟨μν|ρσ⟩ from the store vs detri_int2
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)
    maxg = 0.0
    for σ in 1:n, ρ in 1:σ, ν in 1:n, μ in 1:ν
      col = uppertriangular_index(ρ,σ); row = uppertriangular_index(μ,ν)
      if μ == ν
        maxg = max(maxg, abs(Vs[row,col]/2 - G[μ,μ,ρ,σ]))
      else
        maxg = max(maxg, abs((Vs[row,col]+Va[row,col])/2 - G[μ,ν,ρ,σ]))
      end
    end
    @test maxg < 1e-13
    # pm_matmul! primitive: V·X panel-loop == dense reference (both ± matrices)
    Xr = randn(T, ntri, 4)
    outs = zeros(T, ntri, 4); outa = zeros(T, ntri, 4)
    pm_matmul!(outs, pm, :s, Xr); pm_matmul!(outa, pm, :a, Xr)
    @test maximum(abs.(outs .- Vsr*Xr)) < 1e-12
    @test maximum(abs.(outa .- Var*Xr)) < 1e-12
    close_pm_store!(EC, pm)
  end
end

# Phase-2 kernel gate: the persisted ± kext pm_K2! reproduces calc_K2 — the kext of the MO/dump
# path, which `cc_kext!` uses interchangeably with pm_K2! on the same `D2`/`tripp`, so this pins
# exactly the invariant the two integral paths rely on. Holds for ANY D2. Real and complex.
@testset "pm_K2! ↔ calc_K2" begin
  pm_K2! = ElemCo.CoupledCluster.pm_K2!
  calc_K2 = ElemCo.CoupledCluster.calc_K2
  for T in (Float64, ComplexF64), (n, nocc, maxcols) in ((9, 3, 12), (12, 4, 30))
    EC, int2 = build_store(n, T, maxcols)
    tripp = ElemCo.QMTensors.uppertriangular_cut(n); ntri = length(tripp)
    D2 = randn(T, ntri, nocc, nocc)                        # arbitrary density (kernel identity is D-agnostic)
    pm = open_pm_store(EC)
    Kpm = pm_K2!(pm, D2, tripp)
    close_pm_store!(EC, pm)
    Kref = calc_K2(int2, D2, tripp)
    @test maximum(abs.(Kpm .- Kref)) < 1e-11
  end
end

# The fused, multi-threaded ± fold and 4-quadrant scatter are shared by pm_K2! and pm_K2ab!,
# so they need their own gate against the explicit cut-by-cut reference they replaced. Bit-for-bit
# (not "to round-off"): both forms perform the same two flops per element, in the same order.
# Includes odd/even nocc, nocc=1, and the pq/ij diagonals where the four quadrants alias.
@testset "pm_fold_ij! / pm_scatter_K2! (fused ± fold and unscatter)" begin
  pm_fold_ij! = ElemCo.CoupledCluster.pm_fold_ij!
  pm_scatter_K2! = ElemCo.CoupledCluster.pm_scatter_K2!
  pm_scatter_K2ab! = ElemCo.CoupledCluster.pm_scatter_K2ab!
  ucut = ElemCo.QMTensors.uppertriangular_cut
  scut = ElemCo.QMTensors.swapped_uppertriangular_cut
  for T in (Float64, ComplexF64), (n, nocc) in ((9, 4), (7, 3), (5, 1), (11, 6))
    tripp = ucut(n); tripp_swap = scut(n); ntri = length(tripp)
    trioo = ucut(nocc); trioo_swap = scut(nocc); m = length(trioo)
    D2 = randn(T, ntri, nocc, nocc)
    # --- fold: ½(D2[:,i,j] ± D2[:,j,i]) ---
    Ds = Matrix{T}(undef, ntri, m); Da = Matrix{T}(undef, ntri, m)
    pm_fold_ij!(Ds, Da, D2, trioo)
    Dpp = D2[:, trioo]; Dsw = D2[:, trioo_swap]
    @test Ds == 0.5 .* (Dpp .+ Dsw)                        # bit-for-bit
    @test Da == 0.5 .* (Dpp .- Dsw)
    @test all(iszero, Da[:, [k for k in 1:m if trioo[k][1] == trioo[k][2]]])   # ij-diagonal is 0
    # --- scatter: the four quadrants. aK2 must carry the physical zeros on both diagonals,
    #     which is where the quadrants overlap (V_a has zero pp rows, D_a zero ii columns).
    sK2 = randn(T, ntri, m); aK2 = randn(T, ntri, m)
    for q in 1:n; aK2[ElemCo.QMTensors.uppertriangular_index(q, q), :] .= zero(T); end
    for k in 1:m; trioo[k][1] == trioo[k][2] && (aK2[:, k] .= zero(T)); end
    Kref = Array{T,4}(undef, n, n, nocc, nocc)
    @views Kref[tripp, trioo] .= sK2 .+ aK2
    @views Kref[tripp_swap, trioo_swap] .= sK2 .+ aK2
    @views Kref[tripp, trioo_swap] .= sK2 .- aK2
    @views Kref[tripp_swap, trioo] .= sK2 .- aK2
    K = fill(T(NaN), n, n, nocc, nocc)
    pm_scatter_K2!(K, sK2, aK2, trioo)
    @test K == Kref                                        # bit-for-bit, and no element left unwritten
    # --- αβ scatter: both pq orders of a pair-nonsymmetric density ---
    nij = 3
    sK = randn(T, ntri, nij); aK = randn(T, ntri, nij)
    for q in 1:n; aK[ElemCo.QMTensors.uppertriangular_index(q, q), :] .= zero(T); end
    Kabref = Array{T,3}(undef, n, n, nij)
    @views Kabref[tripp, :] .= sK .+ aK
    @views Kabref[tripp_swap, :] .= sK .- aK
    Kab = fill(T(NaN), n, n, nij)
    pm_scatter_K2ab!(Kab, sK, aK)
    @test Kab == Kabref
  end
end

# Phase-4 gate: pm_to_joint! reconstructs the joint "ao_int2" file exactly from the ± store
# (serves joint-format consumers — the AO→MO transform — once ao_int2 is retired).
@testset "pm_to_joint! (inverse builder)" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((10, 12), (13, 300))
    EC, int2 = build_store(n, T, maxcols)
    delete_file!(EC, "ao_int2")                            # only the ± store remains
    @test !file_exists(EC, "ao_int2")
    ElemCo.PMStore.pm_to_joint!(EC)
    @test file_exists(EC, "ao_int2")
    f, rec = mmap3idx(EC, "ao_int2", T)
    @test maximum(abs.(rec .- int2)) < 1e-13
    close(f)
  end
end

# Phase-3 kernel gate: the PM Fock kernels reproduce the streaming joint-store kernels for
# ARBITRARY (nonsymmetric!) densities — pins the elementwise two-role sweep incl. the
# uniform ½-degeneracy weights and the conj mirror role. Real and complex, several blockings.
# ao_JK!/ao_J2K! on the ± store reproduce a DENSE reference for arbitrary (nonsymmetric!)
# densities — pins the two-role band sweep + conj mirror. Real and complex.
@testset "ao_JK!/ao_J2K! (± store) ↔ dense" begin
  # dense physicist references: J[p,q]=Σ⟨pr|qs⟩Dj[r,s], K[p,q]=Σ⟨pr|sq⟩Dk[r,s]
  jdense(G, D) = (@tensor Jd[p,q] := G[p,r,q,s] * D[r,s]; Jd)
  kdense(G, D) = (@tensor Kd[p,q] := G[p,r,s,q] * D[r,s]; Kd)
  for T in (Float64, ComplexF64), (n, maxcols) in ((9, 9), (13, 30), (11, 200))
    EC, int2 = build_store(n, T, maxcols)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)            # dense ⟨pq|rs⟩ reference
    pm = open_pm_store(EC)
    Dj = randn(T, n, n); Dk = randn(T, n, n)               # nonsymmetric on purpose
    J2 = zeros(T, n, n); K2 = zeros(T, n, n)
    ao_JK!(J2, K2, pm, Dj, Dk)             # dispatch on PMSupermatrices
    @test maximum(abs.(J2 .- jdense(G, Dj))) < 1e-12
    @test maximum(abs.(K2 .- kdense(G, Dk))) < 1e-12
    # UHF variant: shared Coulomb + two exchanges in one pass
    Da = randn(T, n, n); Db = randn(T, n, n); Dt = Da .+ Db
    J2 .= 0; Ka2 = zeros(T, n, n); Kb2 = zeros(T, n, n)
    ao_J2K!(J2, Ka2, Kb2, pm, Dt, Da, Db)
    @test maximum(abs.(J2 .- jdense(G, Dt))) < 1e-12
    @test maximum(abs.(Ka2 .- kdense(G, Da))) < 1e-12
    @test maximum(abs.(Kb2 .- kdense(G, Db))) < 1e-12
    close_pm_store!(EC, pm)
  end

  # theory pin: K_Fock = Σᵢ kext(Cᵢ⊗Cᵢ). The derivation uses T₁ (the real-orbital
  # e1-bra↔ket swap) — an INDEPENDENT symmetry beyond exchange+hermiticity — so it needs a
  # FULLY 8-fold-symmetric integral: build one from a chemist-form (μρ|νσ) with symmetric
  # pairs (as every real physical AO integral is). Complex integrals lack T₁ — the identity
  # is real-only (ao_JK! itself never relies on it; see the general checks above).
  begin
    n = 8; ntri = n*(n+1)÷2
    Gc = randn(n, n, n, n)
    Gc .+= permutedims(Gc, (2,1,3,4)); Gc .+= permutedims(Gc, (1,2,4,3))
    Gc .+= permutedims(Gc, (3,4,1,2))                      # (ab|cd): a↔b, c↔d, (ab)↔(cd)
    int8 = zeros(n, n, ntri)
    for σq in 1:n, ρq in 1:σq, νq in 1:n, μq in 1:n
      int8[μq, νq, uppertriangular_index(ρq, σq)] = Gc[μq, ρq, νq, σq]   # ⟨μν|ρσ⟩ = (μρ|νσ)
    end
    EC, _ = build_store(8, Float64, 12; int2=int8)
    pm = open_pm_store(EC)
    nocc = 3; C = randn(n, nocc); D = C*C'
    Jd = zeros(n, n); Kd = zeros(n, n)
    ao_JK!(Jd, Kd, pm, D, D)
    Es = zeros(pm.npp, nocc)                               # ±-fold of each Cᵢ⊗Cᵢ (Ds only)
    for i in 1:nocc, σq in 1:n, ρq in 1:σq
      idx = uppertriangular_index(ρq, σq)
      Es[idx, i] = ρq == σq ? C[ρq,i]*C[ρq,i]/2 : C[ρq,i]*C[σq,i]
    end
    W = zeros(pm.npp, nocc)
    pm_matmul!(W, pm, :s, Es)
    Kkext = zeros(n, n)
    for i in 1:nocc, νq in 1:n, μq in 1:νq
      row = uppertriangular_index(μq, νq)
      Kkext[μq,νq] += W[row,i]
      μq < νq && (Kkext[νq,μq] += W[row,i])
    end
    @test maximum(abs.(Kkext .- Kd)) < 1e-10
    close_pm_store!(EC, pm)
  end
end

# symmetric-density (HF) fast path: for a real symmetric density the mirror role is redundant,
# so ao_JK!/ao_J2K! with `hermitian=true` (mirror-free sweep + symmetrize, with the diagonal-tile
# split) must reproduce the general two-role path exactly. Several blockings incl. single-block;
# also pins that the guard falls through to the general path for a complex (Hermitian) density.
@testset "ao_JK!/ao_J2K! symmetric-D fast path" begin
  for (n, maxcols) in ((9, 9), (13, 30), (12, 16), (11, 200))
    EC, _ = build_store(n, Float64, maxcols)
    pm = open_pm_store(EC)
    D = (C = randn(n, 4); C*C')                            # real symmetric (HF-like) density
    Jg = zeros(n,n); Kg = zeros(n,n); ao_JK!(Jg, Kg, pm, D, D)                  # general
    Jf = zeros(n,n); Kf = zeros(n,n); ao_JK!(Jf, Kf, pm, D, D; hermitian=true)  # fast
    @test maximum(abs.(Jf .- Jg)) < 1e-11
    @test maximum(abs.(Kf .- Kg)) < 1e-11
    Da = (C = randn(n,3); C*C'); Db = (C = randn(n,2); C*C'); Dt = Da .+ Db
    Jg2 = zeros(n,n); Kag = zeros(n,n); Kbg = zeros(n,n); ao_J2K!(Jg2, Kag, Kbg, pm, Dt, Da, Db)
    Jf2 = zeros(n,n); Kaf = zeros(n,n); Kbf = zeros(n,n); ao_J2K!(Jf2, Kaf, Kbf, pm, Dt, Da, Db; hermitian=true)
    @test maximum(abs.(Jf2 .- Jg2)) < 1e-11
    @test maximum(abs.(Kaf .- Kag)) < 1e-11
    @test maximum(abs.(Kbf .- Kbg)) < 1e-11
    close_pm_store!(EC, pm)
  end
  # complex Hermitian density: the real-only fast path must fall through → identical to general
  EC, _ = build_store(9, ComplexF64, 30)
  pm = open_pm_store(EC)
  D = (C = randn(ComplexF64, 9, 4); C*C')                 # Hermitian
  Jg = zeros(ComplexF64,9,9); Kg = zeros(ComplexF64,9,9); ao_JK!(Jg, Kg, pm, D, D)
  Jf = zeros(ComplexF64,9,9); Kf = zeros(ComplexF64,9,9); ao_JK!(Jf, Kf, pm, D, D; hermitian=true)
  @test maximum(abs.(Jf .- Jg)) < 1e-11
  @test maximum(abs.(Kf .- Kg)) < 1e-11
  close_pm_store!(EC, pm)
end

# Phase-5 gate: the PM-native dressing sweeps reproduce the dense einsum references —
# all 8 occ-early intermediates (3 closed/same-spin + 5 opposite-spin), real+complex,
# single/multi-block, Lo ≠ Ro (dressed) and nocca ≠ noccb (asymmetric).
@testset "pm_occ_early / pm_os_sweep ↔ dense" begin
  pm_occ_early = ElemCo.CoupledCluster.pm_occ_early
  pm_os_sweep = ElemCo.CoupledCluster.pm_os_sweep
  for T in (Float64, ComplexF64), (n, maxcols) in ((8, 8), (12, 30), (14, 300))
    nocc = 3; na = 3; nb = 2
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)
    Lo = randn(T, n, nocc); Ro = randn(T, n, nocc)
    @tensor r_ooAA[i,j,ρ,σ] := G[μ,ν,ρ,σ] * Lo[μ,i] * Lo[ν,j]
    @tensor r_AooA[μ,i,j,σ] := G[μ,ν,ρ,σ] * Lo[ν,i] * Ro[ρ,j]
    @tensor r_oAoA[i,ν,j,σ] := G[μ,ν,ρ,σ] * Lo[μ,i] * Ro[ρ,j]
    a, b, c = pm_occ_early(pm, Lo, Ro)
    @test maximum(abs.(a .- r_ooAA)) < 1e-13
    @test maximum(abs.(b .- r_AooA)) < 1e-13
    @test maximum(abs.(c .- r_oAoA)) < 1e-13
    La = randn(T, n, na); Ra = randn(T, n, na); Lb = randn(T, n, nb); Rb = randn(T, n, nb)
    @tensor r_oOAA[i,J,ρ,σ] := G[μ,ν,ρ,σ] * La[μ,i] * Lb[ν,J]
    @tensor r_AOoA[μ,I,k,σ] := G[μ,ν,ρ,σ] * Lb[ν,I] * Ra[ρ,k]
    @tensor r_oAoA2[i,ν,k,σ] := G[μ,ν,ρ,σ] * La[μ,i] * Ra[ρ,k]
    @tensor r_oAAO[i,ν,ρ,J] := G[μ,ν,ρ,σ] * La[μ,i] * Rb[σ,J]
    @tensor r_AOAO[μ,I,ρ,J] := G[μ,ν,ρ,σ] * Lb[ν,I] * Rb[σ,J]
    o1, o2, o3, o4, o5 = pm_os_sweep(pm, La, Ra, Lb, Rb)
    @test maximum(abs.(o1 .- r_oOAA)) < 1e-13
    @test maximum(abs.(o2 .- r_AOoA)) < 1e-13
    @test maximum(abs.(o3 .- r_oAoA2)) < 1e-13
    @test maximum(abs.(o4 .- r_oAAO)) < 1e-13
    @test maximum(abs.(o5 .- r_AOAO)) < 1e-13
    close_pm_store!(EC, pm)
  end
end

# Half-transformed store: out-of-core gather build + σ-column reader. `ht_column!` must reconstruct
# both the μ→occ (A-role) and ν→occ (B-role, exchange symmetry) dense σ-slabs from the pair-packed
# file. Multi-block stores (small maxcols) exercise the native + mirror-from-earlier-panels gather.
@testset "pm_half_trans / HTStore ↔ dense" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((8, 8), (12, 30), (14, 300))
    m = 4
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)
    C = randn(T, n, m)
    @tensor Aref[i,ν,ρ,σ] := G[μ,ν,ρ,σ] * C[μ,i]        # A-role: μ→occ, ν kept
    @tensor Bref[i,μ,ρ,σ] := G[μ,ν,ρ,σ] * C[ν,i]        # B-role: ν→occ, μ kept
    pm_half_trans(EC, pm, C, "ht_test")
    ht = open_ht_store(EC, "ht_test")
    @test ht.nao == n && ht.m == m
    Aσ = zeros(T, m*n, n); Bσ = zeros(T, m*n, n); eA = 0.0; eB = 0.0
    for σ in 1:n
      ht_column!(Aσ, Bσ, ht, σ)
      eA = max(eA, maximum(abs, reshape(Aσ, m, n, n) .- @view Aref[:,:,:,σ]))
      eB = max(eB, maximum(abs, reshape(Bσ, m, n, n) .- @view Bref[:,:,:,σ]))
    end
    @test eA < 1e-13
    @test eB < 1e-13
    close_ht_store!(EC, ht); delete_ht_store!(EC, "ht_test")
    close_pm_store!(EC, pm)
  end
end

# `ht_jk_columns!`: the single-pass generalized-J/K contraction of both slab roles with two AO
# densities, the kernel the Λ residual's dD1 v,o term (`dD1_ht_vo`) runs on. It reads the pair-packed
# map directly (each element ONCE, vs `ht_column!`'s two passes), so it must reproduce exactly what
# the `ht_column!` slabs give — including on the pair diagonal, where the two ket orders coincide.
# `DJ`/`DK` are deliberately DIFFERENT and NON-symmetric (the whole point of the dD1 term), and the
# complex cases pin that no conjugation sneaks in (real alone cannot distinguish that).
@testset "ht_jk_columns! (single-pass J/K columns) ↔ ht_column!" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((8, 8), (12, 30), (14, 300))
    m = 4
    EC, int2 = build_store(n, T, maxcols)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)
    C = randn(T, n, m); DJ = randn(T, n, n); DK = randn(T, n, n)
    pm = open_pm_store(EC); pm_half_trans(EC, pm, C, "ht_jk"); close_pm_store!(EC, pm)
    ht = open_ht_store(EC, "ht_jk")
    # dense reference: t[i,σ] = Σ_xy ( ⟨x i|y σ⟩ DJ[x,y] − ⟨i x|y σ⟩ DK[x,y] )
    @tensor tref[i,σ] := (G[x,ν,y,σ] * C[ν,i]) * DJ[x,y] - (G[ν,x,y,σ] * C[ν,i]) * DK[x,y]
    t = Matrix{T}(undef, m, n)
    ht_jk_columns!(t, ht, DJ, DK)
    @test maximum(abs, t .- tref) < 1e-12 * max(1.0, maximum(abs, tref))
    # and bit-for-structure agreement with the ht_column! route it replaces
    tcol = zeros(T, m, n); Aσ = zeros(T, m*n, n); Bσ = zeros(T, m*n, n)
    for σ in 1:n
      ht_column!(Aσ, Bσ, ht, σ)
      tv = view(tcol, :, σ)
      mul!(tv, reshape(Bσ, m, n*n), vec(DJ))
      mul!(tv, reshape(Aσ, m, n*n), vec(DK), -one(T), one(T))
    end
    @test maximum(abs, t .- tcol) < 1e-12 * max(1.0, maximum(abs, tref))
    close_ht_store!(EC, ht); delete_ht_store!(EC, "ht_jk")
  end
end

# Open-shell half-transformed store: the per-spin build (`ht_build_dress_unrestricted!` → `ht_oAAA_a`/
# `ht_oAAA_b` stores + `ht_ooAA_a`/`ht_ooAA_b`/`ht_oOAA` doubly-occ blocks) + the fused single-pass reader
# `ht_occ_early_unrestricted` (each store read once) must reproduce ALL the dense occ-early intermediates
# the unrestricted dressing needs — the ssa/ssb same-spin triples + the five opposite-spin blocks.
@testset "ht_occ_early_unrestricted (open-shell) ↔ dense" begin
  ht_build_dress_unrestricted! = ElemCo.CoupledCluster.ht_build_dress_unrestricted!
  ht_occ_early_unrestricted = ElemCo.CoupledCluster.ht_occ_early_unrestricted
  for T in (Float64, ComplexF64), (n, maxcols) in ((8, 8), (12, 30), (14, 300))
    na = 3; nb = 2
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)
    La = randn(T, n, na); Ra = randn(T, n, na); Lb = randn(T, n, nb); Rb = randn(T, n, nb)
    ht_build_dress_unrestricted!(EC, pm, La, Lb)      # La/Lb are the (T1-independent) bra-occ coeffs
    close_pm_store!(EC, pm)
    ssa_in, ssb_in, os_in = ht_occ_early_unrestricted(EC, Ra, Rb)
    # same-spin α triple (v_ooAA, v_AooA, v_oAoA)
    @tensor ra_ooAA[i,j,ρ,σ] := G[μ,ν,ρ,σ] * La[μ,i] * La[ν,j]
    @tensor ra_AooA[μ,i,j,σ] := G[μ,ν,ρ,σ] * La[ν,i] * Ra[ρ,j]
    @tensor ra_oAoA[i,ν,j,σ] := G[μ,ν,ρ,σ] * La[μ,i] * Ra[ρ,j]
    @test maximum(abs.(ssa_in[1] .- ra_ooAA)) < 1e-13
    @test maximum(abs.(ssa_in[2] .- ra_AooA)) < 1e-13
    @test maximum(abs.(ssa_in[3] .- ra_oAoA)) < 1e-13
    # same-spin β triple
    @tensor rb_ooAA[i,j,ρ,σ] := G[μ,ν,ρ,σ] * Lb[μ,i] * Lb[ν,j]
    @tensor rb_AooA[μ,i,j,σ] := G[μ,ν,ρ,σ] * Lb[ν,i] * Rb[ρ,j]
    @tensor rb_oAoA[i,ν,j,σ] := G[μ,ν,ρ,σ] * Lb[μ,i] * Rb[ρ,j]
    @test maximum(abs.(ssb_in[1] .- rb_ooAA)) < 1e-13
    @test maximum(abs.(ssb_in[2] .- rb_AooA)) < 1e-13
    @test maximum(abs.(ssb_in[3] .- rb_oAoA)) < 1e-13
    # opposite-spin five (v_oOAA, v_AOoA, v_oAoA(αα), v_oAAO, v_AOAO)
    @tensor r_oOAA[i,J,ρ,σ] := G[μ,ν,ρ,σ] * La[μ,i] * Lb[ν,J]
    @tensor r_AOoA[μ,I,k,σ] := G[μ,ν,ρ,σ] * Lb[ν,I] * Ra[ρ,k]
    @tensor r_oAAO[i,ν,ρ,J] := G[μ,ν,ρ,σ] * La[μ,i] * Rb[σ,J]
    @tensor r_AOAO[μ,I,ρ,J] := G[μ,ν,ρ,σ] * Lb[ν,I] * Rb[σ,J]
    @test maximum(abs.(os_in[1] .- r_oOAA)) < 1e-13
    @test maximum(abs.(os_in[2] .- r_AOoA)) < 1e-13
    @test os_in[3] === ssa_in[3]                       # os reuses the αα v_oAoA (shared, not recomputed)
    @test maximum(abs.(os_in[4] .- r_oAAO)) < 1e-13
    @test maximum(abs.(os_in[5] .- r_AOAO)) < 1e-13
    delete_ht_store!(EC, "ht_oAAA_a"); delete_ht_store!(EC, "ht_oAAA_b")
    for k in ("ht_ooAA_a", "ht_ooAA_b", "ht_oOAA")
      file_exists(EC, k) && delete_file!(EC, k)
    end
  end
end

# MO-blocks engine (ht_mo_block / save_mo_block!): build 4-index MO integral blocks that carry the
# store's occupied bra index, one sweep per block. Each must match the dense physicist reference. The 5
# directly-mapped blocks are element-type-generic (real + complex); the vvvo block uses the bra↔ket
# Hermiticity swap and is real-only from a bra-store (must throw for complex).
@testset "ht_mo_block / save_mo_block! ↔ dense" begin
  save_mo_block! = ElemCo.CoupledCluster.save_mo_block!
  for T in (Float64, ComplexF64), (n, maxcols) in ((8, 8), (12, 30), (14, 300))
    no = 3; nv = 5
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    G = detri_int2(int2, n, 1:n, 1:n, 1:n, 1:n)
    Co = randn(T, n, no); Cv = randn(T, n, nv)
    pm_half_trans(EC, pm, Co, "ht_e")                      # store's occupied bra = Co
    close_pm_store!(EC, pm)
    @tensor ovoo_r[i,a,j,k] := G[μ,ν,ρ,σ]*Co[μ,i]*Cv[ν,a]*Co[ρ,j]*Co[σ,k]   # ⟨ia|jk⟩
    @tensor ooov_r[i,j,k,a] := G[μ,ν,ρ,σ]*Co[μ,i]*Co[ν,j]*Co[ρ,k]*Cv[σ,a]   # ⟨ij|ka⟩
    @tensor vovv_r[a,i,b,c] := G[μ,ν,ρ,σ]*Cv[μ,a]*Co[ν,i]*Cv[ρ,b]*Cv[σ,c]   # ⟨ai|bc⟩
    @tensor voov_r[a,i,j,b] := G[μ,ν,ρ,σ]*Cv[μ,a]*Co[ν,i]*Co[ρ,j]*Cv[σ,b]   # ⟨ai|jb⟩
    @tensor vovo_r[a,i,b,j] := G[μ,ν,ρ,σ]*Cv[μ,a]*Co[ν,i]*Cv[ρ,b]*Co[σ,j]   # ⟨ai|bj⟩
    @tensor vvvo_r[a,b,c,k] := G[μ,ν,ρ,σ]*Cv[μ,a]*Cv[ν,b]*Cv[ρ,c]*Co[σ,k]   # ⟨ab|ck⟩
    for (name, ref) in (("ovoo",ovoo_r), ("ooov",ooov_r), ("vovv",vovv_r), ("voov",voov_r), ("vovo",vovo_r))
      save_mo_block!(EC, name, "ht_e", Co, Cv)
      @test maximum(abs.(load4idx(EC, name) .- ref)) < 1e-11
    end
    if T <: Complex
      @test_throws Exception save_mo_block!(EC, "vvvo", "ht_e", Co, Cv)   # swap-block real-only
    else
      save_mo_block!(EC, "vvvo", "ht_e", Co, Cv)
      @test maximum(abs.(load4idx(EC, "vvvo") .- vvvo_r)) < 1e-11
    end
    for k in ("ovoo", "ooov", "vovv", "voov", "vovo", "vvvo")
      file_exists(EC, k) && delete_file!(EC, k)
    end
    delete_ht_store!(EC, "ht_e")
  end
end

# On-the-fly AO→MO transform straight from the ± store (pm_transform_int2, used by
# generate_mo_dump) reproduces the joint-int2 transform kernels for arbitrary RECTANGULAR
# coefficients — RHF (triangular ket) and UHF αβ (full dense) — real + complex, several
# blockings. The full joint int2 is never reconstructed.
@testset "pm_transform_int2 (on-the-fly AO→MO) ↔ joint transform" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((9, 9), (12, 16), (11, 200))
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    intmem = Array{T,3}(int2)                          # in-memory joint for the reference kernels
    na, nb = 4, 3
    Ca = randn(T, n, na); Cb = randn(T, n, nb)         # rectangular, different per spin
    ref_aa = transform_int2(intmem, Ca, Ca, Ca, Ca)                              # RHF (triangular)
    new_aa = pm_transform_int2(EC, pm, Ca, Ca, Ca, Ca, "t_aa"; triangular=true)
    @test maximum(abs.(collect(new_aa) .- ref_aa)) < 1e-11
    ref_ab = transform_int2_Q(intmem, Ca, Cb, Ca, Cb)                            # UHF αβ (full dense)
    new_ab = pm_transform_int2(EC, pm, Ca, Cb, Ca, Cb, "t_ab"; triangular=false)
    @test maximum(abs.(collect(new_ab) .- ref_ab)) < 1e-11
    # ket-output blocking (membudget=1 ⇒ one ket pair per pm_matmul! pass) must be identical
    blk_aa = pm_transform_int2(EC, pm, Ca, Ca, Ca, Ca, "t_aab"; triangular=true, membudget=1)
    blk_ab = pm_transform_int2(EC, pm, Ca, Cb, Ca, Cb, "t_abb"; triangular=false, membudget=1)
    @test maximum(abs.(collect(blk_aa) .- ref_aa)) < 1e-11
    @test maximum(abs.(collect(blk_ab) .- ref_ab)) < 1e-11
    close_pm_store!(EC, pm)
  end
end

@testset "pm_transform_int2_n5 (batched direct-from-± N⁵) ↔ joint transform" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((9, 9), (12, 16), (11, 200))
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    intmem = Array{T,3}(int2)
    na, nb = 4, 3
    Ca = randn(T, n, na); Cb = randn(T, n, nb)
    ref_aa = transform_int2(intmem, Ca, Ca, Ca, Ca)                              # RHF (triangular)
    new_aa = pm_transform_int2_n5(EC, pm, Ca, Ca, Ca, Ca, "n_aa"; triangular=true)
    @test maximum(abs.(collect(new_aa) .- ref_aa)) < 1e-11
    ref_ab = transform_int2_Q(intmem, Ca, Cb, Ca, Cb)                            # UHF αβ (full dense)
    new_ab = pm_transform_int2_n5(EC, pm, Ca, Cb, Ca, Cb, "n_ab"; triangular=false)
    @test maximum(abs.(collect(new_ab) .- ref_ab)) < 1e-11
    # p-block + slab-chunk blocking (tiny membudget ⇒ |pb|=1, chunk=1) must be identical
    blk_aa = pm_transform_int2_n5(EC, pm, Ca, Ca, Ca, Ca, "n_aab"; triangular=true, membudget=1)
    blk_ab = pm_transform_int2_n5(EC, pm, Ca, Cb, Ca, Cb, "n_abb"; triangular=false, membudget=1)
    @test maximum(abs.(collect(blk_aa) .- ref_aa)) < 1e-11
    @test maximum(abs.(collect(blk_ab) .- ref_ab)) < 1e-11
    close_pm_store!(EC, pm)
  end
end

# High-level ± "tensor" API: the pm_transform verb (routes pair-space vs N⁵), the eachslab
# streaming iterator (must reproduce pm_slab_sweep! exactly), and the pm_matvec! alias.
@testset "PM tensor verbs (pm_transform / eachslab / pm_matvec!)" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((9, 9), (12, 16), (11, 200))
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    intmem = Array{T,3}(int2)
    C = randn(T, n, 4)
    # pm_transform ≡ transform_int2 (RHF triangular) and ≡ transform_int2_Q (full)
    @test maximum(abs.(collect(pm_transform(EC, pm, C, "v_aa"; triangular=true)) .-
                       transform_int2(intmem, C, C, C, C))) < 1e-11
    Cb = randn(T, n, 3)
    @test maximum(abs.(collect(pm_transform(EC, pm, C, Cb, C, Cb, "v_ab"; triangular=false)) .-
                       transform_int2_Q(intmem, C, Cb, C, Cb))) < 1e-11
    # eachslab must reproduce pm_slab_sweep! bit-for-bit (an AO Coulomb build), and visit npp columns
    D = randn(T, n, n)
    Jsweep = zeros(T, n, n); w = SlabWork{T}(pm)
    pm_slab_sweep!(pm, w) do ρ, σ, native_lo, mirror_lo
      band_mul!(@mview(Jsweep[:,ρ]), w.G, native_lo, n, @mview(D[:,σ]))
      ρ < σ && band_tmul!(@mview(Jsweep[:,σ]), w.G, native_lo, n, @mview(D[:,ρ]))
      add_mirror_row!(Jsweep, ρ, w, D, σ, mirror_lo, false)
      ρ < σ && add_mirror_row!(Jsweep, σ, w, D, ρ, mirror_lo, true)
    end
    Jiter = zeros(T, n, n); nslab = 0
    for s in eachslab(pm)
      nslab += 1
      slab_bandmul!(@mview(Jiter[:,s.ρ]), s, @mview(D[:,s.σ]))
      s.ρ < s.σ && slab_bandtmul!(@mview(Jiter[:,s.σ]), s, @mview(D[:,s.ρ]))
      slab_mirror!(Jiter, s.ρ, s, D, s.σ)
      s.ρ < s.σ && slab_mirrort!(Jiter, s.σ, s, D, s.ρ)
    end
    @test nslab == pm.npp
    @test Jiter == Jsweep                      # bit-for-bit
    # pm_matvec! ≡ pm_matmul!
    X = randn(T, pm.npp, 3); o1 = zeros(T, pm.npp, 3); o2 = zeros(T, pm.npp, 3)
    pm_matvec!(o1, pm, :s, X); pm_matmul!(o2, pm, :s, X)
    @test o1 == o2
    close_pm_store!(EC, pm)
  end
end

# The per-slab Fock routines add_coulomb!/add_exchange! inside an `eachslab` loop (native band GEMV +
# ket-swap + Hermitian mirror). Fuse Coulomb J and exchange K over one sweep; check vs dense @tensor.
@testset "add_coulomb! / add_exchange! (per-slab Fock)" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((9, 9), (12, 16), (11, 200))
    EC, int2 = build_store(n, T, maxcols); pm = open_pm_store(EC)
    G = detri_int2(Array{T,3}(int2), n, 1:n, 1:n, 1:n, 1:n)
    D = randn(T, n, n); J = zeros(T, n, n); K = zeros(T, n, n)
    for s in eachslab(pm)                        # one reconstruction per slab, J + K fused
      add_coulomb!(J, s, D)
      add_exchange!(K, s, D)
    end
    @tensor Jref[μ,ρ] := G[μ,ν,ρ,σ]*D[ν,σ]
    @tensor Kref[μ,σ] := G[μ,ν,ρ,σ]*D[ν,ρ]
    @test maximum(abs.(J .- Jref)) < 1e-11
    @test maximum(abs.(K .- Kref)) < 1e-11
    close_pm_store!(EC, pm)
  end
end

# Fused ± generation: `ao_integrals` builds the store straight from the ERI generator
# (shell-aligned blocks, no jointly packed intermediate at any point). Validate the store contents
# against a reference obtained by running the ERI kernel into a plain triangular array, via
# pm_to_joint!, incl. a forced multi-block blocking.
@testset "fused ± generation ≡ direct ERI generation" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"
  mk() = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>"sto-3g")))
  # reference: the ERI kernel written straight into a jointly packed triangular array
  EC2 = mk()
  bao2 = ElemCo.IntegralTools.save_ao_1e_integrals!(EC2)
  nao2 = size(load2idx(EC2, "S_AA"), 1)
  ref = zeros(Float64, nao2, nao2, nao2*(nao2+1)÷2)
  ElemCo.Integrals.eri_2e4idx_tri!(ref, bao2)
  # fused, default blocking
  EC1 = mk()
  ElemCo.IntegralTools.ao_integrals(EC1)
  @test pm_exists(EC1)
  @test !file_exists(EC1, "ao_int2")                       # never created
  ElemCo.PMStore.pm_to_joint!(EC1)
  f1, j1 = mmap3idx(EC1, "ao_int2")
  @test maximum(abs.(j1 .- ref)) < 1e-13
  close(f1)
  # fused, forced small blocks (multi-block + single-shell batches)
  EC3 = mk()
  bao = ElemCo.IntegralTools.save_ao_1e_integrals!(EC3)
  ElemCo.IntegralTools.pm_integrals!(EC3, bao; maxcols=8)  # nao=7, npp=28 ⇒ several blocks
  pm3 = open_pm_store(EC3)
  @test pm_nblocks(pm3) > 1
  close_pm_store!(EC3, pm3)
  ElemCo.PMStore.pm_to_joint!(EC3)
  f3, j3 = mmap3idx(EC3, "ao_int2")
  @test maximum(abs.(j3 .- ref)) < 1e-13
  close(f3)
end

# Phase-2 acceptance: AO-direct energies through the PM-store kext match the derived-MO-dump
# reference, closed shell (CCSD/DCSD) and open shell (UCCSD, same-spin via PM + αβ raw).
# These runs also exercise the PM-native dressing sweeps end-to-end.
@testset "AO-direct kext via PM store ↔ derived MO dump" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"
  fresh() = (e = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>"sto-3g")));
             e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)
  for m in ("ccsd", "dcsd")
    key = uppercase(m)
    EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_std = @cc m
    EC = fresh(); @ints; @hf; e_pm = @cc m                 # default: AO-direct on the ± store
    @test pm_exists(EC)                                    # the ± store was built by @ints
    @test isempty(EC.fd)                                   # still AO-direct
    @test abs(e_std[key] - e_pm[key]) < 1e-10
  end
  # open-shell cation (ms2=1): UHF via ao_J2K!(pm,…) + frozen-core ao_core_ufock via the PM path
  EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_std = @cc ccsd
  EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_pm = @cc ccsd            # AO-direct on the ± store
  @test pm_exists(EC)
  @test abs(e_std["UCCSD"] - e_pm["UCCSD"]) < 1e-10
  # standalone MP2/UMP2 run AO-direct off the ± store (method gate admits MP2)
  EC = fresh(); @ints; @hf; e_pm_mp2 = @cc mp2
  @test pm_exists(EC) && isempty(EC.fd)                    # standalone MP2 stayed on the ± store
  EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_d_mp2 = @cc mp2
  @test abs(e_pm_mp2["MP2"] - e_d_mp2["MP2"]) < 1e-10      # ± store MP2 == derived-dump MP2
  EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_pm_ump2 = @cc mp2
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_d_ump2 = @cc mp2
  @test abs(e_pm_ump2["UMP2"] - e_d_ump2["UMP2"]) < 1e-10
  # closed-shell CCSD(T)/DCSD(T) run AO-direct off the ± store: the 3-external vvvo/ovoo blocks are built
  # from ht_oAAA (no MO dump), and the (T) energy matches the derived-MO-dump reference
  for m in ("ccsd(t)", "dcsd(t)")
    key = uppercase(m)                                       # "CCSD(T)" / "DCSD(T)"
    EC = fresh(); @ints; @hf; e_pm_t = @cc m
    @test pm_exists(EC) && isempty(EC.fd)                    # AO-direct (T): stayed on the ± store
    EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_ref_t = @cc m
    @test abs(e_pm_t[key] - e_ref_t[key]) < 1e-8
  end
  # closed-shell Λ runs AO-direct off the ± store: the full Λ residual (dressed d_vovv + dD1 Fock-vo +
  # Λ-kext) + the correlated 1-RDM reproduce the derived-MO-dump dipole; ΛCCSD(T)/ΛDCSD(T) energies too.
  EC = fresh(); EC.options.cc.properties = true; @ints; @hf; e_pm_l = @cc ccsd
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); EC.options.cc.properties = true; EC.options.int.ao_direct = false; @ints; @hf; e_ref_l = @cc ccsd
  @test abs(e_pm_l["mu"] - e_ref_l["mu"]) < 1e-8
  for m in ("Λccsd(t)", "Λdcsd(t)")
    key = uppercase(m)                                      # "ΛCCSD(T)" / "ΛDCSD(T)"
    EC = fresh(); @ints; @hf; e_pm_lt = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_ref_lt = @cc m
    @test abs(e_pm_lt[key] - e_ref_lt[key]) < 1e-8
  end
  # closed-shell EOM-CCSD runs AO-direct off the ± store: the CIS pre-pass reads voov/vovo built from
  # ht_oAAA, the doubles Jacobian reuses the Λ machinery; excitation energies match the derive path.
  EC = fresh(); @ints; @hf; e_pm_eom = @cc "eom-ccsd"
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_ref_eom = @cc "eom-ccsd"
  @test abs(e_pm_eom["ω1"] - e_ref_eom["ω1"]) < 1e-7
  # unrestricted (T) (water cation): the same-spin vvvo/vooo (per spin) and the five opposite-spin
  # 3-external blocks are built from the per-spin stores ht_oAAA_a/_b — the βα-looking reads resolve to
  # the same five αβ files. Energies match the derived-UHF-dump reference.
  for m in ("uccsd(t)", "udcsd(t)")
    key = uppercase(m)
    EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_pm_ut = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_ref_ut = @cc m
    @test abs(e_pm_ut[key] - e_ref_ut[key]) < 1e-8
  end
  # unrestricted Λ: the dressed 3-external blocks (d_vovv/d_VOVV/d_vOvV/d_oVvV), the per-spin and αβ
  # Λ-kext (pm_K2!/pm_K2ab! on the AO-folded Λ2), and the dD1 term as the v,o block of the UHF
  # generalized Fock J(Dα+Dβ)−K(Dσ). ΛUCCSD energy and the unrestricted correlated dipole match derive.
  EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_pm_ul = @cc "Λuccsd"
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_ref_ul = @cc "Λuccsd"
  @test abs(e_pm_ul["ΛUCCSD"] - e_ref_ul["ΛUCCSD"]) < 1e-8
  EC = fresh(); @set wf charge=1 ms2=1; EC.options.cc.properties = true; @uhf; e_pm_up = @cc ccsd
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); @set wf charge=1 ms2=1; EC.options.cc.properties = true
  EC.options.int.ao_direct = false; @uhf; e_ref_up = @cc ccsd
  @test abs(e_pm_up["mu"] - e_ref_up["mu"]) < 1e-8
  # unrestricted Λ(T) and unrestricted EOM: the Λ-specific conjugate mixed blocks and the unrestricted
  # CIS blocks are built from the per-spin stores too, so these run AO-direct as well
  for m in ("Λuccsd(t)", "Λudcsd(t)")
    key = uppercase(m)
    EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_pm_ult = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_ref_ult = @cc m
    @test abs(e_pm_ult[key] - e_ref_ult[key]) < 1e-8
  end
  EC = fresh(); @set wf charge=1 ms2=1; @uhf; e_pm_ueom = @cc "eom-uccsd"
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); @set wf charge=1 ms2=1; EC.options.int.ao_direct = false; @uhf; e_ref_ueom = @cc "eom-uccsd"
  @test abs(e_pm_ueom["ω1"] - e_ref_ueom["ω1"]) < 1e-7
  # doubles-only methods (CCD/DCD and the quasi-variational QV-CCD/QV-DCD). They carry no singles, so
  # the MO path uses the bare-integral `pseudo_dressed_ints`; AO-direct gets the same bare blocks from
  # the dressing inside calc_cc_resid (empty T1) plus d_vvoo transposed from d_oovv.
  for m in ("ccd", "dcd", "qv-ccd", "qv-dcd")
    key = uppercase(m)
    EC = fresh(); @ints; @hf; e_pm_qv = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_ref_qv = @cc m
    @test abs(e_pm_qv[key] - e_ref_qv[key]) < 1e-8
  end
  # Λ for the doubles-only methods: the Λ dressing is called with EMPTY Lagrange singles, so the AO
  # path writes the BARE blocks under the same `d_*` names that `pseudo_dressed_ints` writes on the MO
  # path (incl. the 3-external d_vovv). Closed shell and the unrestricted (cation) counterparts.
  for (m, key, open) in (("Λccd", "ΛCCD", false), ("Λdcd", "ΛDCD", false),
                         ("Λuccd", "ΛUCCD", true), ("Λudcd", "ΛUDCD", true))
    EC = fresh(); @ints
    if open; @set wf charge=1 ms2=1; @uhf; else; @hf; end
    e_pm_lam = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); EC.options.int.ao_direct = false; @ints
    if open; @set wf charge=1 ms2=1; @uhf; else; @hf; end
    e_ref_lam = @cc m
    @test abs(e_pm_lam[key] - e_ref_lam[key]) < 1e-8
  end
  # correlated properties without singles go through the same no-singles Λ (the 1-RDM is
  # amplitude-only): the CCD dipole matches the derived-MO-dump one.
  EC = fresh(); EC.options.cc.properties = true; @ints; @hf; e_pm_dp = @cc ccd
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); EC.options.cc.properties = true; EC.options.int.ao_direct = false
  @ints; @hf; e_ref_dp = @cc ccd
  @test abs(e_pm_dp["mu"] - e_ref_dp["mu"]) < 1e-8
  # the orbital-optimized variants re-transform the integrals every macro-iteration: AO-direct folds
  # the rotation into the coefficients (ao_rotate_ints) instead of re-transforming an MO dump, and
  # rebuilds the half-transformed stores for the ROTATED occupied space — their bra IS the occupied
  # space, so unlike the T1 dressing they cannot be built once. `d_mmmo` stays in the AO basis.
  # Note both `E` and `HF(rotated)` are compared: the rotated reference energy is the sensitive one
  # (a stale store leaves E right at R=I and only drifts once the orbitals actually rotate).
  for m in ("oqv-ccd", "oqv-dcd")
    key = uppercase(m)
    EC = fresh(); @ints; @hf; e_oqv = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_oqv_ref = @cc m
    @test abs(e_oqv[key] - e_oqv_ref[key]) < 1e-8
    @test abs(e_oqv["HF"] - e_oqv_ref["HF"]) < 1e-8
  end
  # ... but the orbital-optimized methods provide NO Λ equations, so they cannot be combined with a
  # Λ prefix, `cc.properties` or `wf.natorb` — that is rejected rather than silently rerouted.
  # (Historically it was allowed and produced a dipole wrong in the 3rd decimal at an unchanged
  # energy: `ao_rotate_ints` rebuilds the half-transformed store from the ROTATED occupied orbitals
  # without persisting the rotated coefficients, so `dress_lambda_ints!` — which reads the STORED
  # ones — mixed rotated bras with unrotated kets.)
  for m_oqvp in ("oqv-ccd", "oqv-ccsd")
    EC = fresh(); EC.options.cc.properties = true; @ints; @hf
    @test_throws ErrorException ElemCo.ccdriver(EC, m_oqvp; fcidump="")
    EC = fresh(); EC.options.wf.natorb = "natorb"; @ints; @hf
    @test_throws ErrorException ElemCo.ccdriver(EC, m_oqvp; fcidump="")
  end
  # EOM needs singles — also an error, not a reroute
  EC = fresh(); @ints; @hf
  @test_throws ErrorException ElemCo.ccdriver(EC, "eom-ccd"; fcidump="")
end

# An UNRESTRICTED residual on RESTRICTED (RHF) orbitals runs AO-direct too: `ao_cc_setup!` follows
# the residual's spin treatment (passed by the driver), not the stored orbitals, and unrestricts the
# latter (β = α) exactly as `uhf` does with a restricted guess. Only the opposite combination — a
# closed-shell residual on UHF orbitals — still derives an MO dump, since that is a different
# calculation (the derive path turns it into UCCSD on the UHF dump).
@testset "AO-direct unrestricted residual on restricted orbitals" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"
  fresh() = (e = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>"sto-3g")));
             e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)
  # UHF-form methods on RHF orbitals: the per-spin reference, half-transformed stores (also used by
  # the (T) 3-external blocks) and the unrestricted Λ machinery all run on the duplicated orbitals.
  for (m, key) in (("uccsd", "UCCSD"), ("uccsd(t)", "UCCSD(T)"), ("Λuccsd", "ΛUCCSD"), ("uccd", "UCCD"))
    EC = fresh(); @ints; @hf; e_ao = @cc m
    @test pm_exists(EC) && isempty(EC.fd)
    EC = fresh(); EC.options.int.ao_direct = false; @ints; @hf; e_ref = @cc m
    @test abs(e_ao[key] - e_ref[key]) < 1e-8
  end
  # ... and the physical identity: a UHF-form calculation on RHF orbitals IS the closed-shell one.
  EC = fresh(); @ints; @hf; e_u = @cc uccsd
  EC = fresh(); @ints; @hf; e_cs = @cc ccsd
  @test abs(e_u["UCCSD"] - e_cs["CCSD"]) < 1e-8
  # open-shell occupations on restricted orbitals (ionized reference on the neutral's RHF orbitals):
  # the residual is unrestricted although the orbitals are not, which used to force the derive path
  EC = fresh(); @ints; @hf; @set wf charge=1 ms2=1; e_ao = @cc ccsd
  @test pm_exists(EC) && isempty(EC.fd)
  EC = fresh(); EC.options.int.ao_direct = false
  @ints; @hf; @set wf charge=1 ms2=1; e_ref = @cc ccsd
  @test abs(e_ao["UCCSD"] - e_ref["UCCSD"]) < 1e-8
  # the other direction still derives: with UHF orbitals of a closed-shell molecule, `ccsd` must NOT
  # run the closed-shell residual AO-direct. The derive path detects the UHF dump and runs UCCSD —
  # so the returned key is the route marker (AO-direct would have produced "CCSD").
  EC = fresh(); @uhf; e_uhf_orbs = @cc ccsd
  @test haskey(e_uhf_orbs, "UCCSD") && !haskey(e_uhf_orbs, "CCSD")
  @test abs(e_uhf_orbs["UCCSD"] - e_cs["CCSD"]) < 1e-8   # UHF collapses to RHF for this molecule
end

# AO-direct with DELETED (linearly-dependent) orbitals. `freeze_orbitals!` already removes the
# "Deleted"-class orbitals from the correlated space and `ao_direct_orbitals` drops exactly that
# redundant tail, so the AO-direct path needs no special casing — only the driver gate had to stop
# excluding it. H2 with a duplicated s contraction makes the AO overlap exactly rank-deficient
# (one redundant function per H), so the count is structural and platform-independent.
@testset "AO-direct with redundant (deleted) orbitals" begin
  hbasis = "{
    s, H, 13.01, 1.962, 0.4446, 0.122
    c, 1.4, 0.019685, 0.137977, 0.478148, 0.50124
    c, 4.4, 1.0
    c, 4.4, 1.0
    p, H, 0.727
    c, 1.1, 1.0}"
  geometry = "bohr
       H1 0.0 0.0 0.0
       H2 0.0 0.0 1.4"
  fresh() = (e = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>hbasis)));
             e.options.wf.dump = joinpath(e.scr, "wf.h5"); e.options.scf.redthr = 1.e-6; e)
  for (m, key, open) in (("ccsd", "CCSD", false), ("ccsd(t)", "CCSD(T)", false),
                         ("Λccsd", "ΛCCSD", false), ("eom-ccsd", "ω1", false),
                         ("uccsd", "UCCSD", true), ("Λuccsd", "ΛUCCSD", true))
    EC = fresh(); @ints
    if open; @set wf charge=-1 ms2=1; @uhf; else; @hf; end
    @test ElemCo.OrbTools.n_deleted_orbitals(EC) == 2       # the two linearly-dependent orbitals
    e_ao = @cc m
    @test pm_exists(EC) && isempty(EC.fd)                   # ran AO-direct on the ± store
    EC = fresh(); EC.options.int.ao_direct = false; @ints
    if open; @set wf charge=-1 ms2=1; @uhf; else; @hf; end
    e_ref = @cc m
    @test abs(e_ao[key] - e_ref[key]) < 1e-8
  end
  # ... and the physical invariant (as in oqv_restart_redundant): the duplicated functions add
  # nothing, so the AO-direct energy in the redundant basis must equal the non-redundant one.
  hbasis_nonredund = "{
    s, H, 13.01, 1.962, 0.4446, 0.122
    c, 1.4, 0.019685, 0.137977, 0.478148, 0.50124
    c, 4.4, 1.0
    p, H, 0.727
    c, 1.1, 1.0}"
  EC = fresh(); @ints; @hf; e_red = @cc ccsd
  EC = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>hbasis_nonredund)))
  EC.options.wf.dump = joinpath(EC.scr, "wf.h5"); EC.options.scf.redthr = 1.e-6
  @ints; @hf; e_non = @cc ccsd
  @test ElemCo.OrbTools.n_deleted_orbitals(EC) == 0
  @test abs(e_red["CCSD"] - e_non["CCSD"]) < 1e-8
end

end # @testitem
