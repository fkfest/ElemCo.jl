"""
Persisted ± (plus/minus) supermatrix store for the exact AO integrals — see
`dev/pm_ao_store_plan.md`. The two symmetric-pair-space supermatrices

    Vs[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ + ⟨νμ|ρσ⟩      (row/col diagonal μ=ν carries ×2)
    Va[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ − ⟨νμ|ρσ⟩      (row/col diagonals = 0)

(exactly the `QMTensors.calc_tri_sym_antisym!` convention) are each symmetric for real
integrals and Hermitian for complex ones. Only the lower **block**-triangle is stored, as
dense column panels — so every consumer is a plain zero-copy GEMM (`'N'` + adjoint flag),
never a packed-format special case. Total on disk ≈ n⁴/4, half of the joint `ao_int2`.

Files: `ao_pm_s`, `ao_pm_a` (1-D panel concatenations, ket-2-orbital `σ`-aligned blocks) and
`ao_pm_meta` (the `σ` block breakpoints, so reader and writer never disagree on the layout).

THE CONJUGATION RULE (plan §2.2): pair-internal swaps (μ↔ν, ρ↔σ) are the ± signs and NEVER
conjugate; the block mirror (bra-pair↔ket-pair, the unstored triangle) conjugates with no
sign. In code: mirror-role GEMMs use `adjoint`, mirror-role scatters use `conj` — both no-ops
for real `T` (one code path).
"""
module PMStore
using LinearAlgebra
using ..ElemCo.QMTensors
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.Utils

export PMSupermatrices, pm_from_joint!, pm_to_joint!, open_pm_store, close_pm_store!,
       pm_exists, delete_pm_store!, pm_nblocks, spanel, apanel, diagtile, subpanel,
       pm_matmul!, pm_JK!, pm_J2K!, band_htrans!, pair_luts,
       PMWriter, pm_writer, pm_write_block!, pm_close_writer!

const PM_S_FILE = "ao_pm_s"
const PM_A_FILE = "ao_pm_a"
const PM_META_FILE = "ao_pm_meta"

"""
    PMSupermatrices{T}

Handle to an open ± supermatrix store (both files memory-mapped). Panels are indexed by
block `J`; `σblocks[J]` is the ket-2 orbital range, `pairblocks[J]` the (contiguous) packed
bra/ket pair range `c_J` it owns as columns, and `offsets[J]` the 1-based element start of
panel `J` in each flat map. Panel `J` is the dense `(npp − first(c_J) + 1) × |c_J|` block
`V[first(c_J):npp, c_J]` (column-major): its top `|c_J|` rows are the diagonal tile
`V[c_J,c_J]`, the rest the sub-diagonal `V[last(c_J)+1:npp, c_J]`.
"""
struct PMSupermatrices{T}
  nao::Int
  npp::Int
  σblocks::Vector{UnitRange{Int}}
  pairblocks::Vector{UnitRange{Int}}
  offsets::Vector{Int}
  sio::IOStream
  smap::Vector{T}
  aio::IOStream
  amap::Vector{T}
end

pm_nblocks(pm::PMSupermatrices) = length(pm.σblocks)

# ---- layout ---------------------------------------------------------------------------

"""
    pm_layout(nao, breakpoints) -> (σblocks, pairblocks, offsets, totlen)

Reconstruct the full panel layout from the `σ`-block breakpoints (the last `σ` of each
block; `breakpoints[end] == nao`). Pure/closed-form: writer and reader call it identically.
"""
function pm_layout(nao::Int, breakpoints::Vector{Int})
  npp = nao*(nao+1)÷2
  σblocks = UnitRange{Int}[]
  a = 1
  for b in breakpoints
    push!(σblocks, a:b); a = b + 1
  end
  pairblocks = [uppertriangular_index(1, first(rb)) : uppertriangular_index(last(rb), last(rb))
                for rb in σblocks]
  offsets = Vector{Int}(undef, length(σblocks)); tot = 0
  for J in eachindex(σblocks)
    offsets[J] = tot + 1
    r0 = first(pairblocks[J]); ncol = length(pairblocks[J])
    tot += (npp - r0 + 1) * ncol
  end
  return σblocks, pairblocks, offsets, tot
end

"""
    pm_breakpoints(nao; maxcols) -> Vector{Int}

`σ`-aligned block breakpoints: greedily grow a block until its column count `Σσ` would
exceed `maxcols` (each `σ` contributes `σ` columns). `maxcols ≥ nao` guarantees every single
`σ` fits, so no block is empty.
"""
function pm_breakpoints(nao::Int; maxcols::Int)
  @assert maxcols >= nao "maxcols ($maxcols) must be ≥ nao ($nao) so each σ column-block fits"
  bp = Int[]; cols = 0
  for σ in 1:nao
    if cols + σ > maxcols && cols > 0
      push!(bp, σ - 1); cols = 0
    end
    cols += σ
  end
  push!(bp, nao)
  return bp
end

"""
    pm_default_maxcols(EC, nao, ::Type{T}) -> Int

Column ceiling for a block: the build buffer is `2·npp·maxcols·sizeof(T)`; keep it within a
fraction of the memory budget, but never below `nao` (a single σ) and cap for cache/GEMM.
"""
function pm_default_maxcols(EC::ECInfo, nao::Int, ::Type{T}) where T
  npp = nao*(nao+1)÷2
  cap = fld(available_memory(EC), 4 * npp * sizeof(T))
  return clamp(cap, nao, 2048)
end

# ---- panel accessors (zero-copy views) ------------------------------------------------

@inline function _panel(map::Vector{T}, pm::PMSupermatrices{T}, J::Int) where T
  r0 = first(pm.pairblocks[J]); ncol = length(pm.pairblocks[J]); nrow = pm.npp - r0 + 1
  off = pm.offsets[J]
  return reshape(view(map, off : off + nrow*ncol - 1), nrow, ncol)
end

"Symmetric-panel `J`: dense `V_s[first(c_J):npp, c_J]` view (no copy)."
spanel(pm::PMSupermatrices, J::Int) = _panel(pm.smap, pm, J)
"Antisymmetric-panel `J`: dense `V_a[first(c_J):npp, c_J]` view (no copy)."
apanel(pm::PMSupermatrices, J::Int) = _panel(pm.amap, pm, J)

"Diagonal tile `V[c_J, c_J]` of a panel `P` (its top `|c_J|` rows)."
diagtile(P::AbstractMatrix, pm::PMSupermatrices, J::Int) = @view P[1:length(pm.pairblocks[J]), :]
"Sub-diagonal block `V[last(c_J)+1:npp, c_J]` of a panel `P` (its remaining rows)."
subpanel(P::AbstractMatrix, pm::PMSupermatrices, J::Int) = @view P[length(pm.pairblocks[J])+1:end, :]

# ---- open / close / exists ------------------------------------------------------------

pm_exists(EC::ECInfo) = file_exists(EC, PM_S_FILE) && file_exists(EC, PM_A_FILE) && file_exists(EC, PM_META_FILE)

"""
    open_pm_store(EC) -> PMSupermatrices

Memory-map an existing ± store (read-only) and rebuild the layout from `ao_pm_meta`.
"""
function open_pm_store(EC::ECInfo{T}) where T
  breakpoints = Int.(load(EC, PM_META_FILE, Val(1), Int))
  nao = breakpoints[end]
  σblocks, pairblocks, offsets, _ = pm_layout(nao, breakpoints)
  sio, smap = mmap1idx(EC, PM_S_FILE, T)
  aio, amap = mmap1idx(EC, PM_A_FILE, T)
  return PMSupermatrices{T}(nao, nao*(nao+1)÷2, σblocks, pairblocks, offsets, sio, smap, aio, amap)
end

function close_pm_store!(EC::ECInfo, pm::PMSupermatrices)
  closemmap(EC, pm.sio, pm.smap)
  closemmap(EC, pm.aio, pm.amap)
  return
end

function delete_pm_store!(EC::ECInfo)
  for f in (PM_S_FILE, PM_A_FILE, PM_META_FILE)
    file_exists(EC, f) && delete_file!(EC, f)
  end
  return
end

# ---- builder / writer API -------------------------------------------------------------

"""
    PMWriter{T}

Write handle for building a ± store block by block: the panel layout (from the `σ`-block
`breakpoints`) plus the two open output mmaps. Fill with [`pm_write_block!`](@ref) (any
block order), finalize with [`pm_close_writer!`](@ref).
"""
struct PMWriter{T}
  nao::Int
  npp::Int
  breakpoints::Vector{Int}
  pairblocks::Vector{UnitRange{Int}}
  offsets::Vector{Int}
  sio::IOStream
  smap::Vector{T}
  aio::IOStream
  amap::Vector{T}
end

"Open a ± store for writing with the given `σ`-block `breakpoints` (any valid blocking)."
function pm_writer(EC::ECInfo{T}, nao::Int, breakpoints::Vector{Int}) where T
  _, pairblocks, offsets, totlen = pm_layout(nao, breakpoints)
  sio, smap = newmmap(EC, PM_S_FILE, (totlen,), T; description="PM ± AO ints (symmetric)")
  aio, amap = newmmap(EC, PM_A_FILE, (totlen,), T; description="PM ± AO ints (antisymmetric)")
  return PMWriter{T}(nao, nao*(nao+1)÷2, breakpoints, pairblocks, offsets, sio, smap, aio, amap)
end

"""
    pm_write_block!(w::PMWriter, J, S, A)

Write the ±-folded block `J`: `S`/`A` hold the **full-height** `npp × |c_J|` symmetric/
antisymmetric combinations for the ket columns `c_J`; only the lower rows `≥ first(c_J)`
are stored (the upper rows are the Hermitian mirror owned by earlier panels).
"""
function pm_write_block!(w::PMWriter, J::Int, S::AbstractMatrix, A::AbstractMatrix)
  cJ = w.pairblocks[J]; ncol = length(cJ); r0 = first(cJ); nrow = w.npp - r0 + 1
  off = w.offsets[J]
  for j in 1:ncol                                               # write lower rows column-major
    dst = off + (j-1)*nrow
    copyto!(view(w.smap, dst:dst+nrow-1), @view S[r0:w.npp, j])
    copyto!(view(w.amap, dst:dst+nrow-1), @view A[r0:w.npp, j])
  end
  return
end

"Flush both files and store the layout metadata (`ao_pm_meta`)."
function pm_close_writer!(EC::ECInfo, w::PMWriter)
  closemmap(EC, w.sio, w.smap)
  closemmap(EC, w.aio, w.amap)
  save!(EC, PM_META_FILE, w.breakpoints; description="PM store σ-block breakpoints")
  return
end

"""
    pm_from_joint!(EC; maxcols=pm_default_maxcols(...))

One-time build of the ± store from the joint triangular `ao_int2`. Streams the packed ket
columns block by block (each `c_J` is a contiguous packed range ⇒ sequential mmap read),
folds each `nao×nao` slab into its ±-symmetrized bra pairs with `calc_tri_sym_antisym!`, and
writes the panels via a [`PMWriter`](@ref). Sequential read, sequential write, one pass.
(For generation *without* a joint intermediate see `IntegralTools.pm_integrals!`.)
"""
function pm_from_joint!(EC::ECInfo{T}; maxcols::Int=0) where T
  @assert file_exists(EC, "ao_int2") "no ao_int2 to build the PM store from"
  aofile, int2 = mmap3idx(EC, "ao_int2")
  nao = size(int2, 1); npp = nao*(nao+1)÷2
  maxcols == 0 && (maxcols = pm_default_maxcols(EC, nao, T))
  w = pm_writer(EC, nao, pm_breakpoints(nao; maxcols=maxcols))
  colcap = maximum(length, w.pairblocks)
  fullS = zeros(T, npp, colcap); fullA = zeros(T, npp, colcap)   # reused full-height ± buffers
  for J in eachindex(w.pairblocks)
    cJ = w.pairblocks[J]; ncol = length(cJ)
    Ssub = @view fullS[:, 1:ncol]; Asub = @view fullA[:, 1:ncol]
    calc_tri_sym_antisym!(Ssub, Asub, @view int2[:, :, cJ])     # full-height ± for these ket columns
    pm_write_block!(w, J, Ssub, Asub)
  end
  close(aofile)
  pm_close_writer!(EC, w)
  return
end

"""
    pm_to_joint!(EC)

Inverse of [`pm_from_joint!`](@ref): reconstruct the joint triangular `"ao_int2"` file
(`int2[μ,ν,tri(ρσ)] = ⟨μν|ρσ⟩`) from the ± supermatrix store. One panel-major pass; each
stored element writes its two native slab positions (± inversion) and — sub-panel rows —
its two Hermitian-mirror positions (`conj`; the mirrors of diagonal-tile elements are
their own stored partners). Used to serve joint-format consumers (e.g. the AO→MO
transform) when only the ± store is on disk.
"""
function pm_to_joint!(EC::ECInfo{T}) where T
  pm = open_pm_store(EC)
  n = pm.nao
  lutμ, lutν = pair_luts(n)
  jfile, jint2 = newmmap(EC, "ao_int2", (n, n, pm.npp), T; description="int2 ao")
  @inbounds for Jb in 1:pm_nblocks(pm)
    cJ = pm.pairblocks[Jb]; r0 = first(cJ); lc = last(cJ)
    Ps = spanel(pm, Jb); Pa = apanel(pm, Jb)
    for (jc, c) in enumerate(cJ)
      ρ = lutμ[c]; σ = lutν[c]
      for k in 1:size(Ps, 1)
        r = r0 + k - 1
        x = lutμ[r]; y = lutν[r]
        g1 = (Ps[k,jc] + Pa[k,jc]) / 2       # ⟨xy|ρσ⟩
        g2 = (Ps[k,jc] - Pa[k,jc]) / 2       # ⟨yx|ρσ⟩
        jint2[x, y, c] = g1
        jint2[y, x, c] = g2
        if r > lc                            # Hermitian mirror → slab of ket pair r
          jint2[ρ, σ, r] = conj(g1)          # ⟨ρσ|xy⟩
          jint2[σ, ρ, r] = conj(g2)          # ⟨σρ|xy⟩ = ⟨ρσ|yx⟩
        end
      end
    end
  end
  closemmap(EC, jfile, jint2)
  close_pm_store!(EC, pm)
  return
end

# ---- contraction primitive ------------------------------------------------------------

"""
    pm_matmul!(out, pm, which, X) -> out

Left-multiply by a stored ± supermatrix: `out .= V · X`, `V = Vs` (`which===:s`) or `Va`
(`which===:a`), reconstructed on the fly from its lower block-triangle. `X`, `out` are
`npp × m`. Per panel: the diagonal tile once, the sub-diagonal in its own role (`'N'`) and —
by hermiticity — its mirror role via `adjoint` (BLAS `'C'`/`'T'`, no copy; a no-op transpose
for real `T`). Every stored element fuels exactly its own and (sub-diagonal) its mirror
product, so `out = V·X` for the full Hermitian `V`. Zero-copy panel GEMMs.
"""
function pm_matmul!(out::AbstractMatrix{T}, pm::PMSupermatrices{T}, which::Symbol,
                    X::AbstractMatrix{T}) where T
  @assert size(out, 1) == pm.npp && size(X, 1) == pm.npp "row dims must equal npp=$(pm.npp)"
  fill!(out, zero(T))
  for J in 1:pm_nblocks(pm)
    cJ = pm.pairblocks[J]; lc = last(cJ)
    P = which === :s ? spanel(pm, J) : apanel(pm, J)
    Pd = diagtile(P, pm, J); Pb = subpanel(P, pm, J)
    Xc = @view X[cJ, :]; outc = @view out[cJ, :]
    mul!(outc, Pd, Xc, one(T), one(T))                       # V[c_J,c_J] · X[c_J]  (diagonal tile)
    if lc < pm.npp
      Xb = @view X[lc+1:pm.npp, :]; outb = @view out[lc+1:pm.npp, :]
      mul!(outb, Pb, Xc, one(T), one(T))                     # V[below,c_J] · X[c_J]        ('N')
      mul!(outc, Pb', Xb, one(T), one(T))                    # V[c_J,below] · X[below]  (mirror, adjoint)
    end
  end
  return out
end

# ---- Fock kernels ---------------------------------------------------------------------

"Row-pair decode: `lutμ[tri(μ,ν)] = μ`, `lutν[tri(μ,ν)] = ν` for `μ ≤ ν`."
function pair_luts(nao::Int)
  npp = nao*(nao+1)÷2
  lutμ = Vector{Int}(undef, npp); lutν = Vector{Int}(undef, npp)
  for ν in 1:nao, μ in 1:ν
    r = uppertriangular_index(μ, ν)
    lutμ[r] = μ; lutν[r] = ν
  end
  return lutμ, lutν
end

"""
    unpack_pm_column!(G, Ps, Pa, jc, r0, lutμ, lutν)

Reconstruct column `jc` (ket pair fixed) of a panel into the dense slab `G[μ,ν] = ⟨μν|ρσ⟩`
via the ± inversion `⟨μν|ρσ⟩ = (Vs+Va)/2`, `⟨νμ|ρσ⟩ = (Vs−Va)/2` (the stored ×2 row
diagonals make the `μ=ν` case come out right with no special-casing). The written positions
are exactly the **L-band** `max(μ,ν) ∈ (σ₀, n]` of the panel's stored bra pairs (`σ₀` = last
ket orbital of the previous block); the untouched corner is never read by the band-restricted
kernels below. The missing bra pairs are the Hermitian mirrors owned by earlier panels.
"""
@inline function unpack_pm_column!(G, Ps, Pa, jc::Int, r0::Int, lutμ::Vector{Int}, lutν::Vector{Int})
  @inbounds for k in 1:size(Ps,1)
    x = lutμ[r0+k-1]; y = lutν[r0+k-1]
    s = Ps[k,jc]; a = Pa[k,jc]
    G[x,y] = (s+a)/2; G[y,x] = (s-a)/2
  end
  return
end

"""
    band_mul!(y, G, lo, hi, x)

`y += L·x` where `L` is the L-band `max(row,col) ∈ (lo,hi]` of `G` (all other positions
ignored): two rectangle GEMVs — the bottom band `G[lo+1:hi, 1:hi]` and the right band
`G[1:lo, lo+1:hi]` — exactly covering the band, so no zero-padding flops.
"""
@inline function band_mul!(y, G, lo::Int, hi::Int, x)
  @views mul!(y[lo+1:hi], G[lo+1:hi, 1:hi], x[1:hi], true, true)
  lo > 0 && @views mul!(y[1:lo], G[1:lo, lo+1:hi], x[lo+1:hi], true, true)
  return
end

"`y += transpose(L)·x` for the same L-band (exchange partner — conjugation-free)."
@inline function band_tmul!(y, G, lo::Int, hi::Int, x)
  @views mul!(y[1:hi], transpose(G[lo+1:hi, 1:hi]), x[lo+1:hi], true, true)
  lo > 0 && @views mul!(y[lo+1:hi], transpose(G[1:lo, lo+1:hi]), x[1:lo], true, true)
  return
end

"""
    band_htrans!(H, L, G, lo, hi)

Half-transform over the L-band: `H[i,y] = Σ_x G[x,y] L[x,i]` restricted to
`max(x,y) ∈ (lo,hi]` (two rectangle GEMMs, no zero-padding flops; columns `> hi` of `H`
are zeroed). Plain contraction — no conjugation of `L` (the coefficient convention of the
dressing half-transforms).
"""
function band_htrans!(H, L, G, lo::Int, hi::Int)
  fill!(H, zero(eltype(H)))
  @views mul!(H[:, 1:hi], transpose(L[lo+1:hi, :]), G[lo+1:hi, 1:hi], true, true)
  lo > 0 && @views mul!(H[:, lo+1:hi], transpose(L[1:lo, :]), G[1:lo, lo+1:hi], true, true)
  return H
end

"""
    pm_JK!(J, K, pm, Dj, Dk)

Accumulate (added to `J`/`K`) the Coulomb and exchange AO matrices
``J_{pq} \\mathrel{+}= Σ_{rs} ⟨pr|qs⟩ D^J_{rs}``, ``K_{pq} \\mathrel{+}= Σ_{rs} ⟨pr|sq⟩ D^K_{rs}``
directly from the ± supermatrix store — the `FockFactory.ao_JK!` analogue at **half the
streaming I/O** (each stored element read once, ≈ n⁴/4; requires the physical hermiticity
`⟨μν|ρσ⟩ = conj(⟨ρσ|μν⟩)` the store presumes). Per stored ket-pair column the ± inversion
reconstructs the slab pieces `Gsub` (sub-panel bra pairs) and `Gtile` (diagonal-tile band),
then BLAS `mul!` applies the same slab identities as `ao_JK!` — the flop count is identical,
only the reads halve:
- **native role** (this column as ket `⟨··|ρσ⟩`): `J[:,ρ] += G·Dj[:,σ]`, `K[:,σ] += G·Dk[:,ρ]`
  and for `ρ<σ` the `transpose(G)` partner (exchange — conjugation-free) — on both pieces;
- **Hermitian mirror role** (this column as bra `⟨ρσ|··⟩`, `Gsub` only — the diagonal tile
  is stored full, its mirrors are their own stored elements): the `conj`-wrapped GEMVs
  `J[ρ,:] += conj(Gsub·conj(Dj[σ,:]))` etc. (no-op conj for real `T`).
`Dj`, `Dk` need not be symmetric. O(nao²) working memory.
"""
function pm_JK!(J::AbstractMatrix, K::AbstractMatrix, pm::PMSupermatrices{T},
                Dj::AbstractMatrix, Dk::AbstractMatrix) where T
  n = pm.nao
  lutμ, lutν = pair_luts(n)
  TF = promote_type(T, eltype(Dj), eltype(Dk))
  G = zeros(TF, n, n)
  tv = zeros(TF, n); tw = zeros(TF, n)                     # scratch for the conj-wrapped mirror GEMVs
  # mirror-role GEMV: out[i,:] += conj(L·conj(D[j,:])) (tmul: transpose(L)) — plain BLAS between conj's
  function mirror!(out, i, D, j, lo, tra::Bool)
    @views tv .= conj.(D[j, :])
    fill!(tw, zero(TF))
    tra ? band_tmul!(tw, G, lo, n, tv) : band_mul!(tw, G, lo, n, tv)
    @views out[i, :] .+= conj.(tw)
    return
  end
  @inbounds for Jb in 1:pm_nblocks(pm)
    cJ = pm.pairblocks[Jb]; r0 = first(cJ)
    σ0 = Jb == 1 ? 0 : last(pm.σblocks[Jb-1])              # native band = (σ0, n]; mirror band = (σend, n]
    σend = last(pm.σblocks[Jb])
    Ps = spanel(pm, Jb); Pa = apanel(pm, Jb)
    for (jc, c) in enumerate(cJ)
      ρ = lutμ[c]; σ = lutν[c]
      unpack_pm_column!(G, Ps, Pa, jc, r0, lutμ, lutν)
      # native role: ao_JK!'s slab identities on the stored band (transpose = exchange, no conj)
      @views begin
        band_mul!(J[:,ρ], G, σ0, n, Dj[:,σ])               # J[:,ρ] += ⟨··|ρσ⟩ · Dj[:,σ]
        band_mul!(K[:,σ], G, σ0, n, Dk[:,ρ])               # K[:,σ] += ⟨··|ρσ⟩ · Dk[:,ρ]
        if ρ < σ
          band_tmul!(J[:,σ], G, σ0, n, Dj[:,ρ])            # ket-swapped slab ⟨··|σρ⟩ = Lᵀ
          band_tmul!(K[:,ρ], G, σ0, n, Dk[:,σ])
        end
      end
      # Hermitian mirror role (sub-panel band only — the diagonal tile's mirrors are their
      # own stored elements): this column's stored rows as ⟨ρσ|··⟩ bras
      mirror!(J, ρ, Dj, σ, σend, false)                    # J[ρ,x] += Σ_y ⟨ρσ|xy⟩ Dj[σ,y]
      mirror!(K, ρ, Dk, σ, σend, true)                     # K[ρ,y] += Σ_x ⟨ρσ|xy⟩ Dk[σ,x]
      if ρ < σ
        mirror!(J, σ, Dj, ρ, σend, true)                   # J[σ,x] += Σ_y ⟨σρ|xy⟩ Dj[ρ,y]
        mirror!(K, σ, Dk, ρ, σend, false)                  # K[σ,y] += Σ_x ⟨σρ|xy⟩ Dk[ρ,x]
      end
    end
  end
  return J, K
end

"""
    pm_J2K!(J, Ka, Kb, pm, Dt, Da, Db)

Like [`pm_JK!`](@ref) but builds the shared Coulomb `J` from the total density `Dt` and
both same-spin exchange matrices `Ka`,`Kb` (from `Da`,`Db`) in the same single streaming
pass (the `FockFactory.ao_J2K!` analogue at half the I/O).
"""
function pm_J2K!(J::AbstractMatrix, Ka::AbstractMatrix, Kb::AbstractMatrix,
                 pm::PMSupermatrices{T}, Dt::AbstractMatrix,
                 Da::AbstractMatrix, Db::AbstractMatrix) where T
  n = pm.nao
  lutμ, lutν = pair_luts(n)
  TF = promote_type(T, eltype(Dt), eltype(Da), eltype(Db))
  G = zeros(TF, n, n)
  tv = zeros(TF, n); tw = zeros(TF, n)
  function mirror!(out, i, D, j, lo, tra::Bool)
    @views tv .= conj.(D[j, :])
    fill!(tw, zero(TF))
    tra ? band_tmul!(tw, G, lo, n, tv) : band_mul!(tw, G, lo, n, tv)
    @views out[i, :] .+= conj.(tw)
    return
  end
  @inbounds for Jb in 1:pm_nblocks(pm)
    cJ = pm.pairblocks[Jb]; r0 = first(cJ)
    σ0 = Jb == 1 ? 0 : last(pm.σblocks[Jb-1])
    σend = last(pm.σblocks[Jb])
    Ps = spanel(pm, Jb); Pa = apanel(pm, Jb)
    for (jc, c) in enumerate(cJ)
      ρ = lutμ[c]; σ = lutν[c]
      unpack_pm_column!(G, Ps, Pa, jc, r0, lutμ, lutν)
      @views begin
        band_mul!(J[:,ρ],  G, σ0, n, Dt[:,σ])
        band_mul!(Ka[:,σ], G, σ0, n, Da[:,ρ])
        band_mul!(Kb[:,σ], G, σ0, n, Db[:,ρ])
        if ρ < σ
          band_tmul!(J[:,σ],  G, σ0, n, Dt[:,ρ])
          band_tmul!(Ka[:,ρ], G, σ0, n, Da[:,σ])
          band_tmul!(Kb[:,ρ], G, σ0, n, Db[:,σ])
        end
      end
      mirror!(J, ρ, Dt, σ, σend, false)
      mirror!(Ka, ρ, Da, σ, σend, true)
      mirror!(Kb, ρ, Db, σ, σend, true)
      if ρ < σ
        mirror!(J, σ, Dt, ρ, σend, true)
        mirror!(Ka, σ, Da, ρ, σend, false)
        mirror!(Kb, σ, Db, ρ, σend, false)
      end
    end
  end
  return J, Ka, Kb
end

end # module PMStore
