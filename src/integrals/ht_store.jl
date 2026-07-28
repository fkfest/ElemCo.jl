# Half-transformed AO-integral store (`⟨iν|ρσ⟩` on disk), built from a ± supermatrix store.
#
# The one-index(→occ) half-transform `Σ_μ ⟨μν|ρσ⟩ C[μ,i]` is the universal AO-direct CC intermediate
# (dressing, λ, EOM, (T)): the bra-occ transform is T1-independent, so it is built ONCE (before the CC
# iterations) and read every iteration. Stored PAIR-PACKED — `A[(i,ν), o, tri(ρσ)]`, `o=1` ket order
# `(ρ,σ)`, `o=2` ket order `(σ,ρ)` — the only layout that is write-once under the out-of-core gather
# (each output pair-block owns its `tri` range outright). By exchange symmetry a single file serves both
# the μ→occ ("A") and ν→occ ("B") roles the dressing needs ([`ht_column!`](@ref) resolves both).
#
# This file is `include`d into `module PMStore` (it uses `SlabWork`/`reconstruct_*_both!`/`band_htrans!`/
# `spanel`/`apanel`/`pm_layout` directly).

const HT_META_SUFFIX = "_meta"

"""
    HTStore{T}

Open handle to a half-transformed AO-integral store built by [`pm_half_trans`](@ref): the mmapped
`map[m·nao, 2, npp]` with `map[(i,ν), 1, tri(ρσ)] = Σ_μ ⟨μν|ρσ⟩ C[μ,i]` (ket order `(ρ,σ)`) and
`map[(i,ν), 2, tri(ρσ)] = Σ_μ ⟨μν|σρ⟩ C[μ,i]` (ket order `(σ,ρ)`). Read dense σ-columns with
[`ht_column!`](@ref); the pair-packed layout never leaks to consumers.
"""
struct HTStore{T}
  nao::Int
  m::Int                 # number of transformed columns (e.g. nocc)
  npp::Int
  io::IOStream
  map::Array{T,3}        # [m·nao, 2, npp]
end

"True if the half-transformed store `key` (data + `key*\"_meta\"`) exists on file."
ht_exists(EC::ECInfo, key::AbstractString) = file_exists(EC, key) && file_exists(EC, key * HT_META_SUFFIX)

"""    open_ht_store(EC, key) -> HTStore    — memory-map an existing half-transformed store (read-only)."""
function open_ht_store(EC::ECInfo{T}, key::AbstractString) where {T}
  meta = Int.(load(EC, key * HT_META_SUFFIX, Val(1), Int))
  nao, m = meta[1], meta[2]
  io, map = mmap3idx(EC, key)
  return HTStore{T}(nao, m, nao * (nao + 1) ÷ 2, io, map)
end

close_ht_store!(EC::ECInfo, ht::HTStore) = closemmap(EC, ht.io, ht.map)

"Delete a half-transformed store (data + metadata)."
function delete_ht_store!(EC::ECInfo, key::AbstractString)
  file_exists(EC, key) && delete_file!(EC, key)
  file_exists(EC, key * HT_META_SUFFIX) && delete_file!(EC, key * HT_META_SUFFIX)
  return
end

# [`reconstruct_mirror_both!`](@ref) reading from an in-RAM row-slice sub-block `Ss/As[r, jc]` (row `r`
# = the r-th block-J pair, col `jc` = panel-Jp column) instead of the mmapped panel — for the pread path.
@inline function reconstruct_mirror_sub!(w::SlabWork, Ss, As, r::Int, cJp, ntilep::Int)
  G = w.G; Gt = w.Gt; lutμ = w.lutμ; lutν = w.lutν
  @inbounds for jc in 1:ntilep
    c = cJp[jc]; u = lutμ[c]; v = lutν[c]; s = conj(Ss[r, jc]); a = conj(As[r, jc])
    G[u,v] = (s+a)/2; G[v,u] = (s-a)/2; Gt[u,v] = (s-a)/2; Gt[v,u] = (s+a)/2
  end
end

"""
    pm_half_trans(EC, pm, C, key) -> key

Build the half-transformed store `key` (data + `key*"_meta"`) from the ± store `pm` and bra
coefficients `C[nao, m]`: `Σ_μ ⟨μν|ρσ⟩ C[μ,i]` for every ket pair, in both ket orders. Written by an
**out-of-core gather** — one output pair-block `J` at a time: the native contribution from panel `J`'s
own columns (sequential mmap), plus the Hermitian-mirror contribution of block-`J` pairs as sub-panel
**rows** of every earlier panel, read via `pread`+`WILLNEED` (contiguous per-column runs). Each ± element
is read ≈ twice; only [`band_htrans!`](@ref)-transformed blocks are held in RAM, each output block written
once. Reuses the exact `eachslab(:both)` reconstruction, reordered by output block (so every
`reconstruct_*_both!`+`band_htrans!` is self-contained).
"""
function pm_half_trans(EC::ECInfo, pm::PMSupermatrices{Te}, C::AbstractMatrix, key::AbstractString) where {Te}
  n = pm.nao; m = size(C, 2); npp = pm.npp; nb = pm_nblocks(pm); sz = sizeof(Te)
  size(C, 1) == n || error("pm_half_trans: size(C,1)=$(size(C,1)) must be nao=$n")
  io, A = newmmap(EC, key, (m * n, 2, npp), Te)
  w = SlabWork{Te}(pm); hν = zeros(Te, m, n); hμ = zeros(Te, m, n)
  σ0(J) = J == 1 ? 0 : last(pm.σblocks[J-1])         # first μ below the panel's band
  σend(J) = last(pm.σblocks[J])
  hdr_s = mioheadersize(pm.sio)                      # mio header bytes preceding the mmapped ± data
  hdr_a = mioheadersize(pm.aio)                      # (absolute-offset base for the mirror preads)
  mx = maximum(length, pm.pairblocks)                # reusable mirror row-slice buffers (≤ block size)
  Ss = Matrix{Te}(undef, mx, mx); As = Matrix{Te}(undef, mx, mx)
  @inbounds for J in 1:nb
    cJ = pm.pairblocks[J]; r0 = first(cJ); ntile = length(cJ)
    Ps = spanel(pm, J); Pa = apanel(pm, J); lo = σ0(J)
    for jc in 1:ntile                                # NATIVE: block J's own columns (band (σ0(J), nao])
      reconstruct_slab_both!(w, Ps, Pa, jc, r0)
      band_htrans!(hν, C, w.G,  lo, n)               # μ→occ, ket (ρ,σ)  = order 1
      band_htrans!(hμ, C, w.Gt, lo, n)               # μ→occ, ket (σ,ρ)  = order 2
      p = cJ[jc]; @views A[:, 1, p] .= vec(hν); @views A[:, 2, p] .= vec(hμ)
    end
    nsub = ntile                                     # block-J pairs form a contiguous row range per panel
    for Jp in 1:J-1                                  # MIRROR: block J's pairs as sub-panel rows of Jp
      cJp = pm.pairblocks[Jp]; r0p = first(cJp); ntilep = length(cJp); nrowp = npp - r0p + 1
      base = (pm.offsets[Jp] - 1 + (first(cJ) - r0p)) * sz   # byte of (row=first block-J pair, col 1)
      GC.@preserve Ss As begin
        for jc in 1:ntilep                           # prefetch every column's contiguous row-run
          off = base + (jc - 1) * nrowp * sz
          mioprefetch(pm.sio, hdr_s + off, nsub * sz); mioprefetch(pm.aio, hdr_a + off, nsub * sz)
        end
        for jc in 1:ntilep
          off = base + (jc - 1) * nrowp * sz
          miopread!(pm.sio, pointer(Ss, (jc - 1) * mx + 1), nsub * sz, hdr_s + off)
          miopread!(pm.aio, pointer(As, (jc - 1) * mx + 1), nsub * sz, hdr_a + off)
        end
      end
      lop = σ0(Jp); hip = σend(Jp)
      for r in 1:nsub
        reconstruct_mirror_sub!(w, Ss, As, r, cJp, ntilep)
        band_htrans!(hν, C, w.G,  lop, hip); band_htrans!(hμ, C, w.Gt, lop, hip)
        p = cJ[r]; @views A[:, 1, p] .+= vec(hν); @views A[:, 2, p] .+= vec(hμ)
      end
    end
  end
  closemmap(EC, io, A)
  save!(EC, key * HT_META_SUFFIX, Int[n, m]; description="tmp half-transformed store meta (nao, m)")
  return key
end

"""
    ht_column_B!(Bσ, ht, σ) -> Bσ

  B-role half of [`ht_column!`](@ref): the slab `Bσ[(i,μ), ρ] = Σ_ν⟨μν|ρσ⟩C[ν,i]` alone. Like
  [`ht_column_A!`](@ref) it touches each stored element exactly once, so a full σ-sweep reads the store
  once rather than twice.

  A block builder does NOT need this: the B ordering is reachable from the A slab by particle exchange
  (see [`ht_mo_block`](@ref CoupledCluster.ht_mo_block)), and a builder is free to choose its output
  permutation. It is for consumers whose output layout is fixed by someone else and for which the B
  order is the one that writes contiguously — `save_ao_AAAo!` streams `⟨μν|ρi⟩` into `[μ,ν,ρ,i]` and
  gets a contiguous `[:,ν,:,:]` slice per column from the B slab, where the A slab would leave it
  writing single strided elements.
"""
function ht_column_B!(Bσ::AbstractMatrix, ht::HTStore, σ::Int)
  n = ht.nao
  colrng = uppertriangular_range(σ)
  @views Bσ[:, 1:σ] .= ht.map[:, 2, colrng]
  @inbounds for ρ in σ+1:n
    @views Bσ[:, ρ] .= ht.map[:, 1, uppertriangular_index(σ, ρ)]
  end
  return Bσ
end

"""
    ht_column_A!(Aσ, ht, σ) -> Aσ

  A-role half of [`ht_column!`](@ref): the slab `Aσ[(i,ν), ρ] = Σ_μ⟨μν|ρσ⟩C[μ,i]` alone.

  Each stored element is the A entry of exactly ONE column, so a full σ-sweep with this reader touches
  the store exactly once — half the bytes of the two-slab `ht_column!`. Consumers that need the other
  ket order do NOT need the B slab: by the particle-exchange symmetry `⟨μν|ρσ⟩ = ⟨νμ|σρ⟩` the B-role
  block is the A-role block with the two ket coefficients swapped and the output permuted `(2,1,4,3)`
  (see [`ht_mo_block`](@ref CoupledCluster.ht_mo_block)). Only the per-column dressing sweeps, which
  contract both roles into the SAME output, genuinely need both slabs.
"""
function ht_column_A!(Aσ::AbstractMatrix, ht::HTStore, σ::Int)
  n = ht.nao
  colrng = uppertriangular_range(σ)
  @views Aσ[:, 1:σ] .= ht.map[:, 1, colrng]
  @inbounds for ρ in σ+1:n
    @views Aσ[:, ρ] .= ht.map[:, 2, uppertriangular_index(σ, ρ)]
  end
  return Aσ
end

"""
    ht_column!(Aσ, Bσ, ht, σ) -> (Aσ, Bσ)

Fill the dense σ-column slabs (both `[m·nao, nao]`) for all `ρ`:
- `Aσ[(i,ν), ρ] = Σ_μ ⟨μν|ρσ⟩ C[μ,i]` (A-role, μ→occ) — feeds `v_oAoA`/`v_ooAA`;
- `Bσ[(i,μ), ρ] = Σ_ν ⟨μν|ρσ⟩ C[ν,i]` (B-role, ν→occ, via exchange symmetry) — feeds `v_AooA`.

The two roles read the SAME pairs with swapped ket orders (`Aσ`: o1 for the contiguous σ-column
`ρ≤σ`, o2 for the row picks `ρ>σ`; `Bσ`: the opposite). The pair-packing is invisible to the caller.
"""
function ht_column!(Aσ::AbstractMatrix, Bσ::AbstractMatrix, ht::HTStore, σ::Int)
  n = ht.nao
  colrng = uppertriangular_range(σ)                  # tri(1,σ)..tri(σ,σ) — contiguous
  @views Aσ[:, 1:σ] .= ht.map[:, 1, colrng]          # ρ≤σ: A o1 / B o2
  @views Bσ[:, 1:σ] .= ht.map[:, 2, colrng]
  @inbounds for ρ in σ+1:n                           # ρ>σ: row-σ picks, swapped orders
    p = uppertriangular_index(σ, ρ)
    @views Aσ[:, ρ] .= ht.map[:, 2, p]
    @views Bσ[:, ρ] .= ht.map[:, 1, p]
  end
  return Aσ, Bσ
end

"""
    ht_jk_columns!(t, ht, DJ, DK) -> t

Contract every σ-column's B- and A-role slabs with the AO densities `DJ` / `DK` over **both** of their
free AO slots, for all σ in one go (`t` is `[m, nao]` and is overwritten):

    t[i,σ] = Σ_xy ( B_σ[i,x,y]·DJ[x,y] − A_σ[i,x,y]·DK[x,y] )
           = Σ_xy ( ⟨x i|y σ⟩·DJ[x,y]  − ⟨i x|y σ⟩·DK[x,y] )

with the slabs exactly as [`ht_column!`](@ref) defines them (σ = ket-2 in both). That is the
`[bra-occ, AO-ket]` block of a generalized `J(DJ) − K(DK)` Fock — the contraction the Λ residual's
`dD1` generalized-Fock term needs (`CoupledCluster.dD1_ht_vo`), which wants only that block and so
never has to build an `nao×nao` Fock.

**One sequential pass, each stored element read EXACTLY once** — half of what a full `ht_column!`
sweep reads (that gathers every pair twice: once as a column entry, once as a row pick) and with no
dense per-column slab materialised at all. The four contributions of a stored pair `p = tri(a,b)`,
`a ≤ b`, are read straight off `ht_column!`'s two cases:

    map[:,1,p] = A_{σ=b}[:,y=a] = B_{σ=a}[:,y=b]     (ket order (a,b))
    map[:,2,p] = A_{σ=a}[:,y=b] = B_{σ=b}[:,y=a]     (ket order (b,a))

so each `map[:,o,p]` — a contiguous `m×nao` block, hence a BLAS `gemv` — is used twice, against one
column of `DJ`/`DK`. On the pair diagonal `a == b` the two ket orders coincide with `ht_column!`'s
single `ρ ≤ σ` case, which is why only the `σ=b` pair of contributions is applied there.

Threaded over contiguous `b`-ranges holding equal numbers of pairs (so every task still streams the
map sequentially), each with a private accumulator: race-free and deterministic. `DJ`/`DK` need not
be symmetric, and no conjugation is applied anywhere, so this is element-type generic.
"""
function ht_jk_columns!(t::AbstractMatrix{T}, ht::HTStore, DJ::AbstractMatrix{T},
                        DK::AbstractMatrix{T}) where {T}
  n = ht.nao; m = ht.m
  size(t) == (m, n) || error("ht_jk_columns!: t is $(size(t)), must be $((m, n))")
  size(DJ) == (n, n) && size(DK) == (n, n) ||
    error("ht_jk_columns!: densities are $(size(DJ))/$(size(DK)), must be $((n, n))")
  M4 = reshape(ht.map, m, n, 2, ht.npp)              # [i, x, ketorder, pair] — contiguous m×n slices
  nch = max(1, min(Threads.nthreads(), n))
  bnd = [round(Int, n*sqrt(k/nch)) for k in 0:nch]   # equal pair count per chunk (Σ_{b≤B} b ∝ B²)
  parts = [zeros(T, m, n) for _ in 1:nch]
  @sync for c in 1:nch
    Threads.@spawn begin
      tl = parts[c]
      @inbounds for b in bnd[c]+1:bnd[c+1]
        tb = view(tl, :, b); p0 = b*(b-1)÷2          # tri(a,b) = a + b(b-1)/2
        for a in 1:b
          M1 = view(M4, :, :, 1, p0+a); M2 = view(M4, :, :, 2, p0+a)
          mul!(tb, M2, view(DJ, :, a),  one(T), one(T))   # + B_{σ=b}[:,y=a]·DJ[:,a]
          mul!(tb, M1, view(DK, :, a), -one(T), one(T))   # − A_{σ=b}[:,y=a]·DK[:,a]
          a == b && continue                              # diagonal pair: the σ=a case IS the σ=b one
          ta = view(tl, :, a)
          mul!(ta, M1, view(DJ, :, b),  one(T), one(T))   # + B_{σ=a}[:,y=b]·DJ[:,b]
          mul!(ta, M2, view(DK, :, b), -one(T), one(T))   # − A_{σ=a}[:,y=b]·DK[:,b]
        end
      end
    end
  end
  t .= parts[1]
  for c in 2:nch
    t .+= parts[c]
  end
  return t
end

# ---- consumer-side sweeps over the stores -------------------------------------------------
#
# These read the ± / half-transformed stores and return occupied-contracted intermediates. They
# are pure store operations — no MO space, orbital or method concepts — so they live here rather
# than with the coupled-cluster code that consumes them.

"""
    pm_occ_early(pm::PMSupermatrices, Lo, Ro) -> (v_ooAA, v_AooA, v_oAoA)

  Occ-early half-transform on the persisted ± supermatrix store — three intermediates at half the
  integral streaming (each stored element read once, ≈ n⁴/4; flop parity). One
  [`eachslab`](@ref PMStore.eachslab)`(pm; roles=:both)` pass does the shared bra half-transform pair
  ([`pm_bra_half!`](@ref PMStore.pm_bra_half!) → `hν[i,ν] = Σ_μ ⟨μν|ρσ⟩ Lo[μ,i]` (μ→occ),
  `hμ[i,μ] = Σ_ν ⟨μν|ρσ⟩ Lo[ν,i]` (ν→occ)) and, fused in-cache, the `v_ooAA` GEMM; it also ACCUMULATES
  the half-transforms into `A[(i,ν),ρ,σ]` / `B[(μ,i),ρ,σ]` (the bra index fused LEADING so each `σ`-slice
  is a BLAS-contiguous matrix). The ket-contracted `v_oAoA`/`v_AooA` are then ONE BLAS GEMM per `σ` off
  `A`/`B` — batching the ket ρ-sum instead of a per-slab rank-1 update (memory-bound at BLAS-2). This is
  ≈1.1–1.6× the old per-slab outer-product sweep and grows with `nao`. `A`/`B` cost `2·nocc·nao³` in RAM
  (fine up to a few hundred AOs; an mmap fallback for larger systems is a follow-up).
"""
function pm_occ_early(pm::PMSupermatrices{Te}, Lo, Ro) where Te
  n = pm.nao; nocc = size(Lo, 2)
  v_ooAA = zeros(Te, nocc,nocc,n,n)
  A = zeros(Te, nocc*n, n, n)              # [(i,ν),ρ,σ]  μ→occ half-transforms (feed v_oAoA)
  B = zeros(Te, n*nocc, n, n)              # [(μ,i),ρ,σ]  ν→occ half-transforms (feed v_AooA)
  hν = zeros(Te, nocc, n); hμ = zeros(Te, nocc, n)         # shared bra half-transforms (μ→occ / ν→occ)
  hνT = zeros(Te, n, nocc); hμT = zeros(Te, n, nocc)        # transposes for the B (μ-leading) accumulation
  for s in eachslab(pm; roles=:both)
    pm_bra_half!(hν, hμ, s, Lo)                            # ONE band-GEMM pair, reused by all outputs
    ρ, σ = s.ρ, s.σ; permutedims!(hμT, hμ, (2,1))
    @mview(A[:,ρ,σ]) .+= vec(hν); @mview(B[:,ρ,σ]) .+= vec(hμT)               # accumulate the ket column (ρ,σ)
    v!ooρσ = @mview v_ooAA[:,:,ρ,σ]; @mtensor v!ooρσ[i,j] += hν[i,ν] * Lo[ν,j] # both bra → occ (fused, in cache)
    if ρ < σ                                              # ket order (σ,ρ)
      permutedims!(hνT, hν, (2,1))
      @mview(A[:,σ,ρ]) .+= vec(hμ); @mview(B[:,σ,ρ]) .+= vec(hνT)
      v!ooσρ = @mview v_ooAA[:,:,σ,ρ]; @mtensor v!ooσρ[i,j] += hμ[i,ν] * Lo[ν,j]
    end
  end
  v_oAoA = zeros(Te, nocc*n, nocc, n); v_AooA = zeros(Te, n*nocc, nocc, n)    # [(i,ν),j,σ] / [(μ,i),j,σ]
  @inbounds for σ in 1:n                                                      # batch the ket ρ-sum: BLAS GEMM/σ
    mul!(view(v_oAoA,:,:,σ), view(A,:,:,σ), Ro)                              # plain `view` slices ⇒ BLAS-3
    mul!(view(v_AooA,:,:,σ), view(B,:,:,σ), Ro)                              # (reshape-of-@mview drops to generic)
  end
  return v_ooAA, reshape(v_AooA, n,nocc,nocc,n), reshape(v_oAoA, nocc,n,nocc,n)
end

"""
    pm_os_sweep(pm::PMSupermatrices, La_o, Ra_o, Lb_o, Rb_o) -> (v_oOAA, v_AOoA, v_oAoA, v_oAAO, v_AOAO)

  The opposite-spin occ-early sweep on the ± store (the five intermediates of [`ao_os_blocks`](@ref)
  at half the streaming), as a [`eachslab`](@ref PMStore.eachslab)`(pm; roles=:both)` sweep like
  [`pm_occ_early`](@ref) but with TWO coefficient sets: [`pm_bra_half!`](@ref PMStore.pm_bra_half!)
  gives the shared half-transforms of `La_o` (`hνa`/`hμa`) and `Lb_o` (`hνb`/`hμb`), reused by the
  five `@mtensor` outputs (the ket-contracted `v_oAAO`/`v_AOAO` take the `Rb_o` row of the kept ket
  order). `hνbT`/`hμbT` are the `b` transposes the AO-first outputs (`v_AOoA`/`v_AOAO`) need.
"""
function pm_os_sweep(pm::PMSupermatrices{Te}, La_o, Ra_o, Lb_o, Rb_o) where Te
  n = pm.nao; na = size(La_o,2); nb = size(Lb_o,2)
  v_oOAA = zeros(Te, na,nb,n,n); v_AOoA = zeros(Te, n,nb,na,n); v_oAoA = zeros(Te, na,n,na,n)
  v_oAAO = zeros(Te, na,n,n,nb); v_AOAO = zeros(Te, n,nb,n,nb)
  hνa = zeros(Te,na,n); hμa = zeros(Te,na,n)               # La half-transforms (μ→a / ν→a)
  hνb = zeros(Te,nb,n); hμb = zeros(Te,nb,n)               # Lb half-transforms
  hνbT = zeros(Te,n,nb); hμbT = zeros(Te,n,nb)             # b transposes (for the AO-first outputs)
  for s in eachslab(pm; roles=:both)
    pm_bra_half!(hνa, hμa, s, La_o); pm_bra_half!(hνb, hμb, s, Lb_o)
    ρ, σ = s.ρ, s.σ; v!Raρ = @mview Ra_o[ρ,:]; v!Rbσ = @mview Rb_o[σ,:]
    permutedims!(hνbT, hνb, (2,1)); permutedims!(hμbT, hμb, (2,1))
    v!oOAAρσ = @mview v_oOAA[:,:,ρ,σ]; @mtensor v!oOAAρσ[i,J]   += hνa[i,ν]  * Lb_o[ν,J]
    v!AOoAσ  = @mview v_AOoA[:,:,:,σ]; @mtensor v!AOoAσ[μ,I,k]  += hμbT[μ,I] * v!Raρ[k]
    v!oAoAσ  = @mview v_oAoA[:,:,:,σ]; @mtensor v!oAoAσ[i,ν,k]  += hνa[i,ν]  * v!Raρ[k]
    v!oAAOρ  = @mview v_oAAO[:,:,ρ,:]; @mtensor v!oAAOρ[i,ν,J]  += hνa[i,ν]  * v!Rbσ[J]
    v!AOAOρ  = @mview v_AOAO[:,:,ρ,:]; @mtensor v!AOAOρ[μ,I,J]  += hμbT[μ,I] * v!Rbσ[J]
    if ρ < σ                                              # ket order (σ,ρ)
      v!Raσ = @mview Ra_o[σ,:]; v!Rbρ = @mview Rb_o[ρ,:]
      v!oOAAσρ = @mview v_oOAA[:,:,σ,ρ]; @mtensor v!oOAAσρ[i,J]   += hμa[i,ν]  * Lb_o[ν,J]
      v!AOoAρ  = @mview v_AOoA[:,:,:,ρ]; @mtensor v!AOoAρ[μ,I,k]  += hνbT[μ,I] * v!Raσ[k]
      v!oAoAρ  = @mview v_oAoA[:,:,:,ρ]; @mtensor v!oAoAρ[i,ν,k]  += hμa[i,ν]  * v!Raσ[k]
      v!oAAOσ  = @mview v_oAAO[:,:,σ,:]; @mtensor v!oAAOσ[i,ν,J]  += hμa[i,ν]  * v!Rbρ[J]
      v!AOAOσ  = @mview v_AOAO[:,:,σ,:]; @mtensor v!AOAOσ[μ,I,J]  += hνbT[μ,I] * v!Rbρ[J]
    end
  end
  return v_oOAA, v_AOoA, v_oAoA, v_oAAO, v_AOAO
end

"""
    ht_build_oo!(EC, ht, C2, ookey)

  Build the doubly-bra-occ intermediate `v[i,j,ρ,σ] = Σ_μν⟨μν|ρσ⟩ C1[μ,i] C2[ν,j]` (T1-independent)
  from an open half-transformed store `ht` (whose bra-1 transform used `C1`) and a second occupied bra
  `C2`, saved (mmapped) under `ookey` with shape `(ht.m, size(C2,2), nao, nao)`. Reads each A-role
  σ-column ([`ht_column_A!`](@ref PMStore.ht_column_A!) — the B slab is not needed, so this touches
  each stored element once) and contracts the free `ν` slot with `C2`. For the
  closed shell `C2 == C1` (→ `v_ooAA`); for the opposite-spin cross block `C1 = La_o`, `C2 = Lb_o`
  (→ `v_oOAA`).
"""
function ht_build_oo!(EC::ECInfo{T}, ht::HTStore, C2::AbstractMatrix, ookey::AbstractString) where {T}
  n = ht.nao; m1 = ht.m; m2 = size(C2, 2)
  voio, v_oo = newmmap(EC, ookey, (m1, m2, n, n), T)
  Aσ = zeros(T, m1*n, n)                                 # A-role only → one pass over the store
  for σ in 1:n
    ht_column_A!(Aσ, ht, σ)
    A3 = reshape(Aσ, m1, n, n)
    voσ = @view v_oo[:, :, :, σ]
    @mtensor voσ[i,j,ρ] = A3[i,ν,ρ] * C2[ν,j]            # second bra → occ (contract ν)
  end
  closemmap(EC, voio, v_oo)
  return
end

"""
    ht_build_dress!(EC, pm, C; key="ht_oAAA", ookey="ht_ooAA")

  Build the persistent half-transformed store for the AO dressing (called ONCE per orbital set in
  [`ao_cc_setup!`](@ref)): `key` = `Σ_μ⟨μν|ρσ⟩C[μ,i]` in both ket orders ([`pm_half_trans`](@ref
  PMStore.pm_half_trans)) plus the T1-independent `ookey` = `v_ooAA[i,j,ρ,σ] = Σ_μν⟨μν|ρσ⟩C[μ,i]C[ν,j]`
  ([`ht_build_oo!`](@ref)). `ao_dressed_ints` then reads these each iteration instead of re-streaming the
  ± store — one ± sweep per CC run, not per iteration. `C` is the (T1-independent) active occupied bra
  coefficients. The `key`/`ookey` arguments let the unrestricted build reuse this per spin.
"""
function ht_build_dress!(EC::ECInfo{T}, pm::PMSupermatrices, C::AbstractMatrix;
                         key::AbstractString="ht_oAAA", ookey::AbstractString="ht_ooAA") where {T}
  pm_half_trans(EC, pm, C, key)
  ht = open_ht_store(EC, key)
  ht_build_oo!(EC, ht, C, ookey)
  close_ht_store!(EC, ht)
  return
end

"""
    ht_build_dress_unrestricted!(EC, pm, La_o, Lb_o)

  Unrestricted analogue of [`ht_build_dress!`](@ref): build the per-spin half-transformed stores plus the
  three T1-independent doubly-occ blocks the open-shell dressing reads each iteration. The bra-occ
  transforms `La_o`/`Lb_o` are T1-independent, so this runs ONCE per orbital set in [`ao_cc_setup!`](@ref):

  - `"ht_oAAA_a"` / `"ht_oAAA_b"` : the α / β half-transformed stores ([`pm_half_trans`](@ref));
  - `"ht_ooAA_a"` = `v_ooAA(αα)` , `"ht_ooAA_b"` = `v_ooAA(ββ)` : the same-spin doubly-occ blocks;
  - `"ht_oOAA"`   = `v_oOAA[i,J,ρ,σ] = Σ_μν⟨μν|ρσ⟩La_o[μ,i]Lb_o[ν,J]` : the opposite-spin cross block
    (built off the α store's A-role columns × `Lb_o`, so no separate ± sweep).
"""
function ht_build_dress_unrestricted!(EC::ECInfo{T}, pm::PMSupermatrices,
                                      La_o::AbstractMatrix, Lb_o::AbstractMatrix) where {T}
  pm_half_trans(EC, pm, La_o, "ht_oAAA_a")
  hta = open_ht_store(EC, "ht_oAAA_a")
  ht_build_oo!(EC, hta, La_o, "ht_ooAA_a")               # v_ooAA (αα)
  ht_build_oo!(EC, hta, Lb_o, "ht_oOAA")                 # v_oOAA (αβ cross), same α store
  close_ht_store!(EC, hta)
  pm_half_trans(EC, pm, Lb_o, "ht_oAAA_b")
  htb = open_ht_store(EC, "ht_oAAA_b")
  ht_build_oo!(EC, htb, Lb_o, "ht_ooAA_b")               # v_ooAA (ββ)
  close_ht_store!(EC, htb)
  return
end

"""
    ht_occ_early(EC, Ro; key="ht_oAAA", ookey="ht_ooAA") -> (v_ooAA, v_AooA, v_oAoA)

  The occ-early intermediates read from a prebuilt half-transformed store ([`ht_build_dress!`](@ref)):
  `v_ooAA` is loaded (T1-independent); the T1-dependent ket contractions `v_oAoA[i,ν,j,σ] = Σ_ρ
  Aσ[(i,ν),ρ]·Ro[ρ,j]` and `v_AooA[μ,i,j,σ] = Σ_ρ Bσ[(i,μ),ρ]·Ro[ρ,j]` are one BLAS GEMM per σ off the
  A-/B-role slabs from [`ht_column!`](@ref PMStore.ht_column!). Bit-identical to `pm_occ_early` but
  without re-streaming the ± store. `key`/`ookey` select the store (per-spin in the unrestricted path).
"""
function ht_occ_early(EC::ECInfo{T}, Ro::AbstractMatrix;
                      key::AbstractString="ht_oAAA", ookey::AbstractString="ht_ooAA") where {T}
  ht = open_ht_store(EC, key)
  n = ht.nao; m = ht.m; nj = size(Ro, 2)
  v_ooAA = load4idx(EC, ookey)
  v_oAoA_f = zeros(T, m*n, nj, n)                         # [(i,ν), j, σ]  fused-leading ⇒ BLAS σ-slices
  v_AooA   = zeros(T, n, m, nj, n)                        # [μ, i, j, σ]
  Aσ = zeros(T, m*n, n); Bσ = zeros(T, m*n, n); vAtmp = zeros(T, m*n, nj)
  @inbounds for σ in 1:n
    ht_column!(Aσ, Bσ, ht, σ)
    mul!(view(v_oAoA_f, :, :, σ), Aσ, Ro)                 # v_oAoA[(iν),j] = Aσ[(iν),ρ]·Ro[ρ,j]
    mul!(vAtmp, Bσ, Ro)                                   # [(iμ),j]
    permutedims!(view(v_AooA, :, :, :, σ), reshape(vAtmp, m, n, nj), (2, 1, 3))  # [i,μ,j] → [μ,i,j]
  end
  close_ht_store!(EC, ht)
  return v_ooAA, v_AooA, reshape(v_oAoA_f, m, n, nj, n)
end

"""
    ht_occ_early_unrestricted(EC, Ra_o, Rb_o) -> (ssa_in, ssb_in, os_in)

  All the open-shell occ-early intermediates the unrestricted dressing needs, read from the prebuilt
  per-spin half-transformed stores ([`ht_build_dress_unrestricted!`](@ref)) in ONE pass per store — the
  fused replacement for two [`ht_occ_early`](@ref) calls plus a `pm_os_sweep`. Reading each store once (its
  A-/B-role slabs both come from one [`ht_column!`](@ref PMStore.ht_column!) call) and reusing the shared
  products removes the redundant re-reads/re-GEMMs of the separate passes (6 GEMMs + 2 reads per σ-column
  vs 8 + 4). The doubly-occ blocks (`v_ooAA(αα)`/`v_ooAA(ββ)`/`v_oOAA`) are T1-independent and loaded.

  Per α column `c` (`Aa`/`Ba` = A-/B-role), particle symmetry `⟨μν|ρσ⟩=⟨νμ|σρ⟩` routing every kept-AO to
  the fixed second-ket column: `Aa·Ra_o`→`v_oAoA(αα)` (shared by `ssa` AND `os`); `Ba·Ra_o`→`v_AooA(αα)`;
  `Ba·Rb_o`→`v_oAAO`. Per β column: `Ab·Rb_o`→`v_oAoA(ββ)` (`ssb`) AND `v_AOAO` (`os`); `Bb·Rb_o`→
  `v_AooA(ββ)`; `Bb·Ra_o`→`v_AOoA`. Returns the tuples for `ao_ss_finish`(α), `ao_ss_finish`(β),
  `ao_os_finish`. `Ra_o`/`Rb_o` are the T1-dependent occupied ket coefficients.
"""
function ht_occ_early_unrestricted(EC::ECInfo{T}, Ra_o::AbstractMatrix, Rb_o::AbstractMatrix) where {T}
  hta = open_ht_store(EC, "ht_oAAA_a"); htb = open_ht_store(EC, "ht_oAAA_b")
  n = hta.nao; na = hta.m; nb = htb.m
  v_ooAA_a = load4idx(EC, "ht_ooAA_a"); v_ooAA_b = load4idx(EC, "ht_ooAA_b")  # same-spin, T1-independent
  v_oOAA   = load4idx(EC, "ht_oOAA")                      # [i,J,ρ,σ]  both bra occ, T1-independent
  v_oAoA_a_f = zeros(T, na*n, na, n)                      # [(i,ν),k,σ]  αα (shared by ssa & os), fused-leading
  v_oAoA_b_f = zeros(T, nb*n, nb, n)                      # [(I,ν),L,σ]  ββ (ssb & os v_AOAO), fused-leading
  v_AooA_a = zeros(T, n, na, na, n)                       # [μ,i,j,σ]   ssa
  v_AooA_b = zeros(T, n, nb, nb, n)                       # [μ,I,J,σ]   ssb
  v_oAAO = zeros(T, na, n, n, nb)                         # [i,ν,ρ,J]   os
  v_AOoA = zeros(T, n, nb, na, n)                         # [μ,I,k,σ]   os
  v_AOAO = zeros(T, n, nb, n, nb)                         # [μ,I,ρ,J]   os
  Aa = zeros(T, na*n, n); Ba = zeros(T, na*n, n)          # α A-/B-role columns (one ht_column! each)
  Ab = zeros(T, nb*n, n); Bb = zeros(T, nb*n, n)          # β A-/B-role columns
  wa = zeros(T, na*n, na); ua = zeros(T, na*n, nb)        # α scratch: Ba·Ra_o (→v_AooA_a), Ba·Rb_o (→v_oAAO)
  wb = zeros(T, nb*n, nb); ub = zeros(T, nb*n, na)        # β scratch: Bb·Rb_o (→v_AooA_b), Bb·Ra_o (→v_AOoA)
  @inbounds for c in 1:n
    ht_column!(Aa, Ba, hta, c)                            # α store read (both roles)
    mul!(view(v_oAoA_a_f, :, :, c), Aa, Ra_o)            # v_oAoA(αα)[(i,ν),k]  — shared with os
    mul!(wa, Ba, Ra_o); permutedims!(view(v_AooA_a, :, :, :, c), reshape(wa, na, n, na), (2, 1, 3))  # →v_AooA(αα)
    mul!(ua, Ba, Rb_o); @views v_oAAO[:, :, c, :] .= reshape(ua, na, n, nb)                          # →v_oAAO (kept ρ=c)
    ht_column!(Ab, Bb, htb, c)                            # β store read (both roles)
    mul!(view(v_oAoA_b_f, :, :, c), Ab, Rb_o)           # v_oAoA(ββ)[(I,ν),L]  — reused below for v_AOAO
    permutedims!(view(v_AOAO, :, :, c, :), reshape(view(v_oAoA_b_f, :, :, c), nb, n, nb), (2, 1, 3))  # →v_AOAO (kept ρ=c)
    mul!(wb, Bb, Rb_o); permutedims!(view(v_AooA_b, :, :, :, c), reshape(wb, nb, n, nb), (2, 1, 3))  # →v_AooA(ββ)
    mul!(ub, Bb, Ra_o); permutedims!(view(v_AOoA, :, :, :, c), reshape(ub, nb, n, na), (2, 1, 3))    # →v_AOoA (kept σ=c)
  end
  close_ht_store!(EC, hta); close_ht_store!(EC, htb)
  v_oAoA_a = reshape(v_oAoA_a_f, na, n, na, n); v_oAoA_b = reshape(v_oAoA_b_f, nb, n, nb, n)
  ssa_in = (v_ooAA_a, v_AooA_a, v_oAoA_a)
  ssb_in = (v_ooAA_b, v_AooA_b, v_oAoA_b)
  os_in  = (v_oOAA, v_AOoA, v_oAoA_a, v_oAAO, v_AOAO)     # os reuses the αα v_oAoA
  return ssa_in, ssb_in, os_in
end
