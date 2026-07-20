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
using ElemCo.TensorTools: detri_int2, @tensor, newmmap, closemmap, mmap3idx
using ElemCo.PMStore
using ElemCo.FockFactory: ao_JK!, ao_J2K!
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

# Phase-2 kernel gate: the persisted kext pm_K2! reproduces the streaming ± kext
# calc_pm_K2! bit-for-bit (both apply the same ij-fold + 4-quadrant scatter to the same
# Vs/Va — persisted vs re-folded). Holds for ANY D2. Real and complex.
@testset "pm_K2! ↔ calc_pm_K2!" begin
  pm_K2! = ElemCo.CoupledCluster.pm_K2!
  calc_pm_K2! = ElemCo.CoupledCluster.calc_pm_K2!
  for T in (Float64, ComplexF64), (n, nocc, maxcols) in ((9, 3, 12), (12, 4, 30))
    EC, int2 = build_store(n, T, maxcols)
    tripp = ElemCo.QMTensors.uppertriangular_cut(n); ntri = length(tripp)
    D2 = randn(T, ntri, nocc, nocc)                        # arbitrary density (kernel identity is D-agnostic)
    pm = open_pm_store(EC)
    Kpm = pm_K2!(pm, D2, tripp)
    close_pm_store!(EC, pm)
    Kref = calc_pm_K2!(int2, D2, tripp)
    @test maximum(abs.(Kpm .- Kref)) < 1e-11
  end
end

# Phase-3 kernel gate: the PM Fock kernels reproduce the streaming joint-store kernels for
# ARBITRARY (nonsymmetric!) densities — pins the elementwise two-role sweep incl. the
# uniform ½-degeneracy weights and the conj mirror role. Real and complex, several blockings.
@testset "pm_JK!/pm_J2K! ↔ ao_JK!/ao_J2K!" begin
  for T in (Float64, ComplexF64), (n, maxcols) in ((9, 9), (13, 30), (11, 200))
    EC, int2 = build_store(n, T, maxcols)
    pm = open_pm_store(EC)
    Dj = randn(T, n, n); Dk = randn(T, n, n)               # nonsymmetric on purpose
    J1 = zeros(T, n, n); K1 = zeros(T, n, n)
    ao_JK!(J1, K1, int2, Dj, Dk)                           # joint-store reference
    J2 = zeros(T, n, n); K2 = zeros(T, n, n)
    pm_JK!(J2, K2, pm, Dj, Dk)
    @test maximum(abs.(J2 .- J1)) < 1e-12
    @test maximum(abs.(K2 .- K1)) < 1e-12
    # UHF variant: shared Coulomb + two exchanges in one pass
    Da = randn(T, n, n); Db = randn(T, n, n); Dt = Da .+ Db
    J1 .= 0; Ka1 = zeros(T, n, n); Kb1 = zeros(T, n, n)
    ao_J2K!(J1, Ka1, Kb1, int2, Dt, Da, Db)
    J2 .= 0; Ka2 = zeros(T, n, n); Kb2 = zeros(T, n, n)
    pm_J2K!(J2, Ka2, Kb2, pm, Dt, Da, Db)
    @test maximum(abs.(J2 .- J1)) < 1e-12
    @test maximum(abs.(Ka2 .- Ka1)) < 1e-12
    @test maximum(abs.(Kb2 .- Kb1)) < 1e-12
    close_pm_store!(EC, pm)
  end

  # theory pin: K_Fock = Σᵢ kext(Cᵢ⊗Cᵢ). The derivation uses T₁ (the real-orbital
  # e1-bra↔ket swap) — an INDEPENDENT symmetry beyond exchange+hermiticity — so it needs a
  # FULLY 8-fold-symmetric integral: build one from a chemist-form (μρ|νσ) with symmetric
  # pairs (as every real physical AO integral is). Complex integrals lack T₁ — the identity
  # is real-only (pm_JK! itself never relies on it; see the general checks above).
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
    pm_JK!(Jd, Kd, pm, D, D)
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

# Phase-2 acceptance: AO-direct energies through the PM-store kext match the standard GEMM
# kext, closed shell (CCSD/DCSD) and open shell (UCCSD, same-spin via PM + αβ raw).
@testset "AO-direct kext via PM store ↔ standard" begin
  geometry = "
    O   0.000000000   0.000000000  -0.130186067
    H1  0.000000000   1.489124508   1.033245507
    H2  0.000000000  -1.489124508   1.033245507"
  fresh() = (e = ElemCo.ECInfo(system=parse_geometry(geometry, Dict("ao"=>"sto-3g")));
             e.options.wf.dump = joinpath(e.scr, "wf.h5"); e)
  for m in ("ccsd", "dcsd")
    key = uppercase(m)
    EC = fresh(); @ints; ehf_std = @hf; e_std = @cc m
    EC = fresh(); @set int ao_pm=true; @ints; ehf_pm = @hf; e_pm = @cc m
    @test pm_exists(EC)                                    # the ± store was built by @ints
    @test isempty(EC.fd)                                   # still AO-direct
    # AO-HF via the PM Fock builder (gen_fock dispatch) == joint-store HF
    @test abs(ehf_std["HF"] - ehf_pm["HF"]) < 1e-11
    @test abs(e_std[key] - e_pm[key]) < 1e-10
  end
  # open-shell cation (ms2=1): UHF via pm_J2K! + frozen-core ao_core_ufock via the PM path
  EC = fresh(); @set wf charge=1 ms2=1; ehf_std = @uhf; e_std = @cc ccsd
  EC = fresh(); @set wf charge=1 ms2=1; @set int ao_pm=true; ehf_pm = @uhf; e_pm = @cc ccsd
  @test pm_exists(EC)
  @test abs(ehf_std["UHF"] - ehf_pm["UHF"]) < 1e-11
  @test abs(e_std["UCCSD"] - e_pm["UCCSD"]) < 1e-10
end

end # @testitem
