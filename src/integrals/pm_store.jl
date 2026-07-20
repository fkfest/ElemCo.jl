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

export PMSupermatrices, pm_from_joint!, open_pm_store, close_pm_store!, pm_exists,
       delete_pm_store!, pm_nblocks, spanel, apanel, diagtile, subpanel, pm_matmul!

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

# ---- builder --------------------------------------------------------------------------

"""
    pm_from_joint!(EC; maxcols=pm_default_maxcols(...))

One-time build of the ± store from the joint triangular `ao_int2`. Streams the packed ket
columns block by block (each `c_J` is a contiguous packed range ⇒ sequential mmap read),
folds each `nao×nao` slab into its ±-symmetrized bra pairs with `calc_tri_sym_antisym!`, and
writes only the lower rows `≥ first(c_J)` (the upper rows are the Hermitian mirror already
owned by earlier panels). Sequential read, sequential write, one pass.
"""
function pm_from_joint!(EC::ECInfo{T}; maxcols::Int=0) where T
  @assert file_exists(EC, "ao_int2") "no ao_int2 to build the PM store from"
  aofile, int2 = mmap3idx(EC, "ao_int2")
  nao = size(int2, 1); npp = nao*(nao+1)÷2
  maxcols == 0 && (maxcols = pm_default_maxcols(EC, nao, T))
  breakpoints = pm_breakpoints(nao; maxcols=maxcols)
  σblocks, pairblocks, offsets, totlen = pm_layout(nao, breakpoints)
  sio, smap = newmmap(EC, PM_S_FILE, (totlen,), T; description="PM ± AO ints (symmetric)")
  aio, amap = newmmap(EC, PM_A_FILE, (totlen,), T; description="PM ± AO ints (antisymmetric)")
  colcap = maximum(length, pairblocks)
  fullS = zeros(T, npp, colcap); fullA = zeros(T, npp, colcap)   # reused full-height ± buffers
  for J in eachindex(σblocks)
    cJ = pairblocks[J]; ncol = length(cJ); r0 = first(cJ); nrow = npp - r0 + 1
    Ssub = @view fullS[:, 1:ncol]; Asub = @view fullA[:, 1:ncol]
    calc_tri_sym_antisym!(Ssub, Asub, @view int2[:, :, cJ])     # full-height ± for these ket columns
    off = offsets[J]
    for j in 1:ncol                                             # write lower rows column-major
      dst = off + (j-1)*nrow
      copyto!(view(smap, dst:dst+nrow-1), @view Ssub[r0:npp, j])
      copyto!(view(amap, dst:dst+nrow-1), @view Asub[r0:npp, j])
    end
  end
  closemmap(EC, sio, smap); closemmap(EC, aio, amap)
  close(aofile)
  save!(EC, PM_META_FILE, breakpoints; description="PM store σ-block breakpoints")
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

end # module PMStore
