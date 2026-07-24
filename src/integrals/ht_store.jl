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

"""
    pm_half_trans(EC, pm, C, key) -> key

Build the half-transformed store `key` (data + `key*"_meta"`) from the ± store `pm` and bra
coefficients `C[nao, m]`: `Σ_μ ⟨μν|ρσ⟩ C[μ,i]` for every ket pair, in both ket orders. Written by an
**out-of-core gather** — one output pair-block `J` at a time: the native contribution from panel `J`'s
own columns, plus the Hermitian-mirror contribution of block-`J` pairs as sub-panel **rows** of every
earlier panel. Each ± element is read ≈ twice; only [`band_htrans!`](@ref)-transformed blocks are held
in RAM, and each output block is written once. Reuses the exact `eachslab(:both)` reconstruction, just
reordered by output block (so every `reconstruct_*_both!`+`band_htrans!` is self-contained).
"""
function pm_half_trans(EC::ECInfo, pm::PMSupermatrices{Te}, C::AbstractMatrix, key::AbstractString) where {Te}
  n = pm.nao; m = size(C, 2); npp = pm.npp; nb = pm_nblocks(pm)
  size(C, 1) == n || error("pm_half_trans: size(C,1)=$(size(C,1)) must be nao=$n")
  io, A = newmmap(EC, key, (m * n, 2, npp), Te)
  w = SlabWork{Te}(pm); hν = zeros(Te, m, n); hμ = zeros(Te, m, n)
  σ0(J) = J == 1 ? 0 : last(pm.σblocks[J-1])         # first μ below the panel's band
  σend(J) = last(pm.σblocks[J])
  @inbounds for J in 1:nb
    cJ = pm.pairblocks[J]; r0 = first(cJ); ntile = length(cJ)
    Ps = spanel(pm, J); Pa = apanel(pm, J); lo = σ0(J)
    for jc in 1:ntile                                # NATIVE: block J's own columns (band (σ0(J), nao])
      reconstruct_slab_both!(w, Ps, Pa, jc, r0)
      band_htrans!(hν, C, w.G,  lo, n)               # μ→occ, ket (ρ,σ)  = order 1
      band_htrans!(hμ, C, w.Gt, lo, n)               # μ→occ, ket (σ,ρ)  = order 2
      p = cJ[jc]; @views A[:, 1, p] .= vec(hν); @views A[:, 2, p] .= vec(hμ)
    end
    for Jp in 1:J-1                                  # MIRROR: block J's pairs as sub-panel rows of Jp
      cJp = pm.pairblocks[Jp]; r0p = first(cJp); ntilep = length(cJp)
      Psp = spanel(pm, Jp); Pap = apanel(pm, Jp); lop = σ0(Jp); hip = σend(Jp)
      for p in cJ
        reconstruct_mirror_both!(w, Psp, Pap, p - r0p + 1, cJp, ntilep)
        band_htrans!(hν, C, w.G,  lop, hip)          # band (σ0(Jp), σend(Jp)] of the μ-tiling
        band_htrans!(hμ, C, w.Gt, lop, hip)
        @views A[:, 1, p] .+= vec(hν); @views A[:, 2, p] .+= vec(hμ)
      end
    end
  end
  closemmap(EC, io, A)
  save!(EC, key * HT_META_SUFFIX, Int[n, m]; description="half-transformed store meta (nao, m)")
  return key
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
