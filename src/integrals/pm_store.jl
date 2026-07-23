"""
Persisted ± (plus/minus) supermatrix store for the exact AO integrals.

The two symmetric-pair-space supermatrices

    Vs[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ + ⟨νμ|ρσ⟩      (row/col diagonal μ=ν carries ×2)
    Va[tri(μν),tri(ρσ)] = ⟨μν|ρσ⟩ − ⟨νμ|ρσ⟩      (row/col diagonals = 0)

(exactly the `QMTensors.calc_tri_sym_antisym!` convention) are each symmetric for real
integrals and Hermitian for complex ones. Only the lower **block**-triangle is stored, as
dense column panels — so kext consumer is a plain zero-copy GEMM (`'N'` + adjoint flag),
never a packed-format special case. Total on disk ≈ n⁴/4.

Dimensions: `μ,ν,ρ,σ` are AO indices running `1:nao`; `tri(μν)` packs an AO pair `μ≤ν` into
`1:npp` with `npp = nao·(nao+1)/2`. So each supermatrix is `npp × npp` in pair space, and the
`n⁴/4` size estimates use `n = nao` (`npp² ≈ n⁴/4`).

Files: `ao_pm_s`, `ao_pm_a` (1-D panel concatenations, ket-2-orbital `σ`-aligned blocks) and
`ao_pm_meta` (the `σ` block breakpoints, so reader and writer never disagree on the layout).

THE CONJUGATION RULE: pair-internal swaps (μ↔ν, ρ↔σ) are the ± signs and NEVER
conjugate; the block mirror (bra-pair↔ket-pair, the unstored triangle) conjugates with no
sign. In code: mirror-role GEMMs use `adjoint`, mirror-role scatters use `conj` — both no-ops
for real `T` (one code path).

Entry points: build with [`pm_from_joint!`](@ref) (or `IntegralTools.pm_integrals!` for the
fused, no-joint build); open a built store with [`open_pm_store`](@ref); contract with
[`pm_matmul!`](@ref) or the `FockFactory.ao_JK!(…, ::PMSupermatrices, …)` methods. See
[`PMSupermatrices`](@ref) for the handle type, its fields, and usage examples.
"""
module PMStore
using LinearAlgebra
using ..ElemCo.QMTensors
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.Utils

export PMSupermatrices, pm_from_joint!, pm_to_joint!, open_pm_store, close_pm_store!,
       pm_exists, delete_pm_store!, pm_nblocks, spanel, apanel, diagtile, subpanel,
       pm_matmul!, pm_matvec!, band_htrans!, pair_luts,
       SlabWork, reconstruct_slab!, band_mul!, band_tmul!, add_mirror_row!, pm_slab_sweep!,
       PMSlab, eachslab, slab_bandmul!, slab_bandtmul!, slab_mirror!, slab_mirrort!, @pmtensor,
       BothSlab, pm_bra_half!,
       PMWriter, pm_writer, pm_write_block!, pm_close_writer!

const PM_S_FILE = "ao_pm_s"
const PM_A_FILE = "ao_pm_a"
const PM_META_FILE = "ao_pm_meta"

"""
    PMSupermatrices{T}

Open handle to a memory-mapped ± supermatrix AO-integral store (see the module docstring for
the on-disk format). In the usual flow you never touch this directly — `ao_integrals(EC)`
builds the store and the consumers (`FockFactory.ao_JK!`, `cc.pm_K2!`, the dressing sweeps)
dispatch on it. To use it by hand: [`open_pm_store`](@ref) it, contract, [`close_pm_store!`](@ref).

# Notation
- `nao` — number of AOs (basis size); AO indices `μ,ν,ρ,σ` run `1:nao`.
- `npp` — number of packed AO pairs, `nao·(nao+1)÷2`. This is the **pair-space** dimension:
  each supermatrix `Vs`,`Va` is conceptually `npp × npp`, its rows/cols indexed by packed pairs.
- `tri(μν)` — the packed index of AO pair `(μ,ν)`, `μ≤ν` (`QMTensors.uppertriangular_index`);
  runs `1:npp`. [`pair_luts`](@ref)`(nao)` gives the inverse (packed index → `μ`, `ν`).
- **panel / block `J`** — the store keeps only the lower block-triangle of `Vs`/`Va`, cut into
  column blocks. Panel `J` is a dense matrix: the columns `c_J` (a contiguous ket-pair range)
  for every bra-pair row `≥ first(c_J)` (the rows above are the unstored Hermitian mirror).

# Fields
- `nao`, `npp` — as above.
- `σblocks[J]` — the ket-2 orbital (`σ`) range that defines block `J`.
- `pairblocks[J]` — `c_J`, the contiguous packed **pair** range panel `J` owns as its columns.
- `offsets[J]` — 1-based start of panel `J` inside each flat memory map.
- `sio,smap` / `aio,amap` — open `IOStream` + mmapped flat vector for `Vs` / `Va`; the panels
  are reshaped views into these (via [`spanel`](@ref) / [`apanel`](@ref)).

Panel `J` has shape `(npp − first(c_J) + 1) × |c_J|` (column-major): its top `|c_J|` rows are
the diagonal tile `V[c_J, c_J]` ([`diagtile`](@ref)), the rest the sub-diagonal
`V[last(c_J)+1:npp, c_J]` ([`subpanel`](@ref)).

# Example — kext contraction `K2[tri(pq)] = Σ_rs ⟨pq|rs⟩ D2[rs]`
Its pair-space form is `K2 = Vs·Ds + Va·Da`, where `Ds`,`Da` are the ± fold of the density
over ket pairs (both `npp × m`, `m` right-hand sides — e.g. `nocc²` for the CC 4-external term):
```julia
pm = open_pm_store(EC)                        # T inferred from the stored files
Ks = zeros(eltype(pm), pm.npp, m)             # eltype(pm) === T
Ka = zeros(eltype(pm), pm.npp, m)
pm_matmul!(Ks, pm, :s, Ds)                    # Ks = Vs · Ds   (zero-copy panel GEMMs)
pm_matmul!(Ka, pm, :a, Da)                    # Ka = Va · Da
K2 = Ks .+ Ka                                 # unpack tri(pq) → K2[p,q] with pair_luts(pm.nao)
close_pm_store!(EC, pm)
```

# Example — low-level panel access (zero-copy views into the mmap)
```julia
for J in 1:pm_nblocks(pm)
  S = spanel(pm, J)                           # Vs panel: (npp-first(c_J)+1) × |c_J|
  Sdiag = diagtile(S, pm, J)                  #   top |c_J| rows = Vs[c_J, c_J]
  Ssub  = subpanel(S, pm, J)                  #   remaining rows = Vs[below, c_J]
end
```
"""
struct PMSupermatrices{T}
  nao::Int                      # number of AOs (basis size)
  npp::Int                      # number of packed AO pairs = nao*(nao+1)÷2 (pair-space dimension)
  σblocks::Vector{UnitRange{Int}}     # σblocks[J] = ket-2 orbital range defining block J
  pairblocks::Vector{UnitRange{Int}}  # pairblocks[J] = c_J, the packed pair columns of panel J
  offsets::Vector{Int}          # offsets[J] = 1-based start of panel J in each flat map
  sio::IOStream                 # open handle + mmapped flat vector of the Vs (symmetric) panels
  smap::Vector{T}
  aio::IOStream                 # open handle + mmapped flat vector of the Va (antisymmetric) panels
  amap::Vector{T}
end

pm_nblocks(pm::PMSupermatrices) = length(pm.σblocks)
Base.eltype(::PMSupermatrices{T}) where {T} = T   # so Fock builders promote the same as for a raw int2

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

"GEMM-friendly σ-block column width (see [`pm_default_maxcols`](@ref))."
const PM_BLOCK_COLS = 512

"""
    pm_default_maxcols(nao) -> Int

Default σ-block column ceiling for the ± store layout: `clamp(PM_BLOCK_COLS, nao, npp)`.

**Deterministic — a function of `nao` only, NOT of available memory.** The layout is persisted
in `ao_pm_meta` and rebuilt from it by every reader ([`open_pm_store`](@ref)), so a
memory-dependent width would make the on-disk block structure non-reproducible (it would differ
between the writing machine/run and any other) and size the build buffer to volatile free memory.
A fixed width is reproducible and keeps the build buffer (`2·npp·maxcols·sizeof(T)`) predictable.
The clamp bounds: `≥ nao` is required (a single σ contributes up to `nao` columns), and `≤ npp`
means a tiny store (`npp < PM_BLOCK_COLS`, i.e. `nao ≤ 31`) is a single block (the full square) —
the block-triangle saving needs ≥2 blocks anyway. Between the two, `PM_BLOCK_COLS` batches enough
columns per panel for efficient BLAS-3.
"""
pm_default_maxcols(nao::Int) = clamp(PM_BLOCK_COLS, nao, nao*(nao+1)÷2)

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
    pm_from_joint!(EC; maxcols=pm_default_maxcols(nao))

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
  maxcols == 0 && (maxcols = pm_default_maxcols(nao))
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

# ---- panel decode + half-transform helpers (shared by the Fock and dressing consumers) -

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
    band_htrans!(H, L, G, lo, hi)

Half-transform over the L-band: `H[i,y] = Σ_x G[x,y] L[x,i]` restricted to
`max(x,y) ∈ (lo,hi]` (two rectangle GEMMs, no zero-padding flops; columns `> hi` of `H`
are zeroed). Plain contraction — no conjugation of `L` (the coefficient convention of the
dressing half-transforms). The reconstructed slab `G` over the panel's L-band (via
the Fock builders' `reconstruct_slab!`) is the input.
"""
function band_htrans!(H, L, G, lo::Int, hi::Int)
  fill!(H, zero(eltype(H)))
  @views mul!(H[:, 1:hi], transpose(L[lo+1:hi, :]), G[lo+1:hi, 1:hi], true, true)
  lo > 0 && @views mul!(H[:, lo+1:hi], transpose(L[1:lo, :]), G[1:lo, lo+1:hi], true, true)
  return H
end

# ---- per-column slab reconstruction sweep ---------------------------------------------
#
# Any consumer that needs the dense AO slabs `G[μ,ν] = ⟨μν|ρσ⟩` back from the ± store (the Fock
# builders, and any other per-slab AO transform) drives it through this sweep. A stored ket
# column `(ρσ)` holds, over its panel's **L-band** bra pairs, the ± combinations
# `Vs,Va = ⟨μν|ρσ⟩ ± ⟨νμ|ρσ⟩`. `reconstruct_slab!` inverts them to `G[μ,ν]` on that band;
# `band_mul!`/`band_tmul!` contract the band with a vector without touching the unstored corner.
# `pm_slab_sweep!` walks the whole store, reconstructs each column's slab, and hands it to a
# consumer callback. Because only the lower block-triangle is stored, each column is used in TWO
# roles (this keeps the read at n⁴/4, half the joint store):
#   • NATIVE  — the column IS the ket `⟨··|ρσ⟩`: contract `G`'s L-band `(native_lo, nao]`.
#   • MIRROR  — by hermiticity the same column is the bra `⟨ρσ|··⟩`: contract the SUB-PANEL band
#               `(mirror_lo, nao]` (the diagonal tile's mirrors are its own stored elements).
# `FockFactory.ao_JK!`/`ao_J2K!` are the worked reference consumers.

"""
    SlabWork{TF}(pm::PMSupermatrices) -> SlabWork{TF}

Reusable scratch for a [`pm_slab_sweep!`](@ref): the reconstructed `nao×nao` slab `G`, two
length-`nao` buffers for the mirror-role GEMVs, and the packed-pair lookup tables. `TF` is the
working element type — take `TF = promote_type(eltype(pm), eltype(data))` so a real store still
contracts against complex data (a real store gives `TF = eltype(pm) = T`).
"""
struct SlabWork{TF}
  G::Matrix{TF}
  Gt::Matrix{TF}          # transpose of the slab; filled only by the roles=:both reconstruction
  tv::Vector{TF}
  tw::Vector{TF}
  lutμ::Vector{Int}
  lutν::Vector{Int}
end
function SlabWork{TF}(pm::PMSupermatrices) where {TF}
  n = pm.nao
  lutμ, lutν = pair_luts(n)
  return SlabWork{TF}(zeros(TF, n, n), zeros(TF, n, n), zeros(TF, n), zeros(TF, n), lutμ, lutν)
end

"""
    reconstruct_slab!(w, Ps, Pa, jc, r0)

Fill `w.G[μ,ν] = ⟨μν|ρσ⟩` for stored column `jc` (ket pair `(ρ,σ)`) of the ± panels
`Ps = spanel(pm,J)`, `Pa = apanel(pm,J)` (`r0 = first(pairblocks[J])`), via the inversion
`⟨μν|ρσ⟩=(Vs+Va)/2`, `⟨νμ|ρσ⟩=(Vs−Va)/2` (the stored ×2 row diagonals make `μ=ν` come out right
with no special-casing). Only the panel's L-band bra pairs are written; the untouched corner is
never read by the band GEMVs. Normally called for you by [`pm_slab_sweep!`](@ref).
"""
@inline function reconstruct_slab!(w::SlabWork, Ps, Pa, jc::Int, r0::Int)
  G = w.G; lutμ = w.lutμ; lutν = w.lutν
  @inbounds for k in 1:size(Ps,1)
    x = lutμ[r0+k-1]; y = lutν[r0+k-1]
    s = Ps[k,jc]; a = Pa[k,jc]
    G[x,y] = (s+a)/2; G[y,x] = (s-a)/2
  end
  return
end

"""
    band_mul!(y, G, lo, hi, x)   /   band_tmul!(y, G, lo, hi, x)

`y += L·x` (resp. `transpose(L)·x`) where `L` is the **L-band** `max(row,col) ∈ (lo,hi]` of the
reconstructed slab `G`: two rectangle GEMVs (bottom band + right band) covering the band exactly
— no zero-padding flops, and the unstored corner (`row,col ≤ lo`) is never read. `transpose`
(not `adjoint`): swapping the ket pair is exchange, conjugation-free.
"""
@inline function band_mul!(y, G, lo::Int, hi::Int, x)
  @views mul!(y[lo+1:hi], G[lo+1:hi, 1:hi], x[1:hi], true, true)
  lo > 0 && @views mul!(y[1:lo], G[1:lo, lo+1:hi], x[lo+1:hi], true, true)
  return
end
@inline function band_tmul!(y, G, lo::Int, hi::Int, x)
  @views mul!(y[1:hi], transpose(G[lo+1:hi, 1:hi]), x[lo+1:hi], true, true)
  lo > 0 && @views mul!(y[lo+1:hi], transpose(G[1:lo, lo+1:hi]), x[1:lo], true, true)
  return
end

"""
    add_mirror_row!(out, i, w, D, j, lo, transposed)

Mirror-role GEMV: `out[i,:] += Σ conj(G)·D[j,·]` over the L-band `(lo, nao]` — the current slab
read as the bra `⟨ρσ|··⟩`. Runs as `conj(band · conj(D[j,:]))` so the BLAS happens between the
`conj`s (both no-ops for real `TF`); `transposed` selects [`band_tmul!`](@ref) over
[`band_mul!`](@ref). Uses the `w.tv`/`w.tw` scratch of the [`SlabWork`](@ref).
"""
@inline function add_mirror_row!(out, i::Int, w::SlabWork, D, j::Int, lo::Int, transposed::Bool)
  n = size(w.G, 1)
  @views w.tv .= conj.(D[j, :])
  fill!(w.tw, zero(eltype(w.tw)))
  transposed ? band_tmul!(w.tw, w.G, lo, n, w.tv) : band_mul!(w.tw, w.G, lo, n, w.tv)
  @views out[i, :] .+= conj.(w.tw)
  return
end

"""
    pm_slab_sweep!(f, pm, w)

Visit each stored ket-pair column of the ± store `pm` exactly once: reconstruct its dense slab
`⟨μν|ρσ⟩` into `w.G` (a [`SlabWork`](@ref)) and call `f(ρ, σ, native_lo, mirror_lo)`. `(ρ,σ)` is
the ket pair; `native_lo`/`mirror_lo` are the L-band lower bounds for the native/mirror roles
(see the section comment above). `f` contracts `w.G` with [`band_mul!`](@ref)/[`band_tmul!`](@ref)
(native role) and/or [`add_mirror_row!`](@ref) (mirror role).

# Example — an AO Coulomb build `J[μ,ρ] = Σ_νσ ⟨μν|ρσ⟩ D[ν,σ]`
```julia
w = SlabWork{eltype(pm)}(pm)                 # real store: TF = eltype(pm) = T
J = zeros(eltype(pm), pm.nao, pm.nao)
pm_slab_sweep!(pm, w) do ρ, σ, native_lo, mirror_lo
  band_mul!(view(J,:,ρ), w.G, native_lo, pm.nao, view(D,:,σ))   # native: ⟨··|ρσ⟩ · D[:,σ]
  ρ < σ && band_tmul!(view(J,:,σ), w.G, native_lo, pm.nao, view(D,:,ρ))
  add_mirror_row!(J, ρ, w, D, σ, mirror_lo, false)              # mirror: ⟨ρσ|··⟩ · D[σ,:]
  ρ < σ && add_mirror_row!(J, σ, w, D, ρ, mirror_lo, true)
end
```
(`FockFactory.ao_JK!` wraps exactly this — Coulomb + exchange — into `add_coulomb!`/`add_exchange!`.)
"""
function pm_slab_sweep!(f, pm::PMSupermatrices, w::SlabWork)
  @inbounds for Jb in 1:pm_nblocks(pm)
    cJ = pm.pairblocks[Jb]; r0 = first(cJ)
    native_lo = Jb == 1 ? 0 : last(pm.σblocks[Jb-1])       # native band = (native_lo, nao]
    mirror_lo = last(pm.σblocks[Jb])                        # mirror band = (mirror_lo, nao] (sub-panel)
    Ps = spanel(pm, Jb); Pa = apanel(pm, Jb)
    for (jc, c) in enumerate(cJ)
      ρ = w.lutμ[c]; σ = w.lutν[c]
      reconstruct_slab!(w, Ps, Pa, jc, r0)
      f(ρ, σ, native_lo, mirror_lo)
    end
  end
  return
end

# ======================= high-level "PM tensor" verbs / streaming ============================
# A thin, tensor-like API over the primitives above: the pair-space matvec `pm_matvec!`, the slab
# iterator `eachslab` + `pm_bra_half!`, and the per-slab einsum macro `@pmtensor` (below). The
# full 4-index AO→MO transform is the `pm_transform` verb in IntegralTools.

"""
    pm_matvec!(out, pm, which, X) -> out

  Pair-space matvec `out = Vs·X` (`which=:s`) or `out = Va·X` (`which=:a`) over packed AO pairs —
  the verb name for the primitive [`pm_matmul!`](@ref).
"""
@inline pm_matvec!(out, pm::PMSupermatrices, which::Symbol, X) = pm_matmul!(out, pm, which, X)

"""
    PMSlab{TF}

  One reconstructed ket-pair slab yielded by [`eachslab`](@ref): `Gslab(s)[μ,ν] = ⟨μν|ρσ⟩` for the
  ket pair `(s.ρ, s.σ)`, with the native/mirror L-band lower bounds `s.native_lo`/`s.mirror_lo`
  (band = `(lo, s.nao]`). The slab aliases the iterator's single reusable scratch — it is valid
  ONLY until the next iteration; do not `collect`/retain it. Contract it with [`slab_bandmul!`](@ref)
  / [`slab_bandtmul!`](@ref) (native role) and [`slab_mirror!`](@ref) / [`slab_mirrort!`](@ref)
  (Hermitian mirror role).
"""
struct PMSlab{TF}
  w::SlabWork{TF}
  ρ::Int
  σ::Int
  native_lo::Int
  mirror_lo::Int
  nao::Int
end
"""    Gslab(s::PMSlab) -> Matrix — the reconstructed dense slab `⟨μν|ρσ⟩` (aliases scratch)."""
@inline Gslab(s::PMSlab) = s.w.G

struct SlabIterator{TF}
  pm::PMSupermatrices
  w::SlabWork{TF}
end
"""
    eachslab(pm; TF=eltype(pm)) -> iterator of PMSlab

  The iterator form of [`pm_slab_sweep!`](@ref): `for slab in eachslab(pm) …` reconstructs each
  stored ket-pair slab `⟨μν|ρσ⟩` into shared scratch and yields a [`PMSlab`](@ref), in the exact
  same visit order (native columns of a σ-block in turn). Lets a custom AO-store contraction be
  written as an ordinary loop with the band helpers ([`slab_bandmul!`](@ref) etc.) instead of a
  `pm_slab_sweep!` callback. The yielded slab aliases one reusable buffer — single pass, do not retain.

  `roles=:both` instead yields ALL ket pairs (native columns + the Hermitian-mirror sub-panel rows)
  as [`BothSlab`](@ref)s carrying `G` and `Gt`, for [`pm_bra_half!`](@ref) — the dressing sweep.

# Example — AO Coulomb build `J[μ,ρ] = Σ_νσ ⟨μν|ρσ⟩ D[ν,σ]`
```julia
J = zeros(eltype(pm), pm.nao, pm.nao)
for s in eachslab(pm)
  slab_bandmul!(view(J,:,s.ρ), s, view(D,:,s.σ))            # native: ⟨··|ρσ⟩ · D[:,σ]
  s.ρ < s.σ && slab_bandtmul!(view(J,:,s.σ), s, view(D,:,s.ρ))
  slab_mirror!(J, s.ρ, s, D, s.σ)                           # mirror: ⟨ρσ|··⟩ · D[σ,:]
  s.ρ < s.σ && slab_mirrort!(J, s.σ, s, D, s.ρ)
end
```
"""
eachslab(pm::PMSupermatrices; TF=eltype(pm), roles::Symbol=:native) =
  roles === :both   ? BothSlabIterator{TF}(pm, SlabWork{TF}(pm)) :
  roles === :native ? SlabIterator{TF}(pm, SlabWork{TF}(pm)) :
  error("eachslab: roles must be :native or :both, got :$roles")

Base.IteratorSize(::Type{<:SlabIterator}) = Base.SizeUnknown()
Base.eltype(::Type{SlabIterator{TF}}) where {TF} = PMSlab{TF}
Base.iterate(it::SlabIterator) = _slab_step(it, 1, 1)
Base.iterate(it::SlabIterator, state) = _slab_step(it, state[1], state[2])
@inline function _slab_step(it::SlabIterator{TF}, Jb::Int, jc::Int) where {TF}
  pm = it.pm; w = it.w
  @inbounds while Jb <= pm_nblocks(pm)
    cJ = pm.pairblocks[Jb]
    if jc <= length(cJ)
      native_lo = Jb == 1 ? 0 : last(pm.σblocks[Jb-1])
      mirror_lo = last(pm.σblocks[Jb])
      c = cJ[jc]
      reconstruct_slab!(w, spanel(pm, Jb), apanel(pm, Jb), jc, first(cJ))
      return PMSlab{TF}(w, w.lutμ[c], w.lutν[c], native_lo, mirror_lo, pm.nao), (Jb, jc + 1)
    end
    Jb += 1; jc = 1
  end
  return nothing
end

# band helpers reading the band bounds off the slab (native role = this column as ket ⟨··|ρσ⟩;
# mirror role = the Hermitian mirror ⟨ρσ|··⟩, conj-wrapped). `x`/`D` are the density operands.
"""    slab_bandmul!(y, s::PMSlab, x) — native role: `y += band(⟨··|ρσ⟩)·x` (ket order ρσ)."""
@inline slab_bandmul!(y, s::PMSlab, x)  = band_mul!(y, s.w.G, s.native_lo, s.nao, x)
"""    slab_bandtmul!(y, s::PMSlab, x) — native role, transposed: `y += band(⟨··|ρσ⟩)ᵀ·x` (ket order σρ)."""
@inline slab_bandtmul!(y, s::PMSlab, x) = band_tmul!(y, s.w.G, s.native_lo, s.nao, x)
"""    slab_mirror!(out, i, s::PMSlab, D, j) — Hermitian mirror role: `out[i,:] += conj(band)·conj(D[j,:])`."""
@inline slab_mirror!(out, i::Int, s::PMSlab, D, j::Int)  = add_mirror_row!(out, i, s.w, D, j, s.mirror_lo, false)
"""    slab_mirrort!(out, i, s::PMSlab, D, j) — Hermitian mirror role, transposed band."""
@inline slab_mirrort!(out, i::Int, s::PMSlab, D, j::Int) = add_mirror_row!(out, i, s.w, D, j, s.mirror_lo, true)

# ---- eachslab(pm; roles=:both) + pm_bra_half!: bra half-transforms (the dressing) -----------
# roles=:both yields ALL ket pairs (native columns AND the Hermitian-mirror sub-panel rows), each
# as a BothSlab carrying the reconstructed slab `G` AND its transpose `Gt` over the band `(lo,hi]`,
# so `pm_bra_half!` gets BOTH bra half-transforms (one from G, one from Gt) from one coeff set — the
# shared expensive step the fused dressing outputs reuse (compute it once, not once per output).

# native column jc: fill G/Gt for the panel's L-band bra pairs, ket pair = column jc
@inline function reconstruct_slab_both!(w::SlabWork, Ps, Pa, jc::Int, r0::Int)
  G = w.G; Gt = w.Gt; lutμ = w.lutμ; lutν = w.lutν
  @inbounds for k in 1:size(Ps,1)
    x = lutμ[r0+k-1]; y = lutν[r0+k-1]; s = Ps[k,jc]; a = Pa[k,jc]
    G[x,y] = (s+a)/2; G[y,x] = (s-a)/2; Gt[x,y] = (s-a)/2; Gt[y,x] = (s+a)/2
  end
end
# mirror sub-panel row k: the tile's ket pairs are the bra pairs (conj); ket pair = row k's pair
@inline function reconstruct_mirror_both!(w::SlabWork, Ps, Pa, k::Int, cJ, ntile::Int)
  G = w.G; Gt = w.Gt; lutμ = w.lutμ; lutν = w.lutν
  @inbounds for jc in 1:ntile
    c = cJ[jc]; u = lutμ[c]; v = lutν[c]; s = conj(Ps[k,jc]); a = conj(Pa[k,jc])
    G[u,v] = (s+a)/2; G[v,u] = (s-a)/2; Gt[u,v] = (s-a)/2; Gt[v,u] = (s+a)/2
  end
end

"""
    BothSlab{TF}

  One ket-pair slab yielded by [`eachslab`](@ref)`(pm; roles=:both)`: the reconstructed `⟨μν|ρσ⟩`
  (`s.w.G`) and its transpose (`s.w.Gt`) over the band `(s.lo, s.hi]`, for ket pair `(s.ρ, s.σ)`.
  Aliases the iterator scratch — valid only until the next iteration. Contract with [`pm_bra_half!`](@ref).
"""
struct BothSlab{TF}
  w::SlabWork{TF}
  ρ::Int
  σ::Int
  lo::Int
  hi::Int
end

struct BothSlabIterator{TF}
  pm::PMSupermatrices
  w::SlabWork{TF}
end
Base.IteratorSize(::Type{<:BothSlabIterator}) = Base.SizeUnknown()
Base.eltype(::Type{BothSlabIterator{TF}}) where {TF} = BothSlab{TF}
Base.iterate(it::BothSlabIterator) = _both_step(it, 1, 1, 1)
Base.iterate(it::BothSlabIterator, st) = _both_step(it, st[1], st[2], st[3])
# state = (Jb, phase, idx): phase 1 = native columns (band (σ0,nao]), phase 2 = mirror rows (band (σ0,σend])
@inline function _both_step(it::BothSlabIterator{TF}, Jb::Int, phase::Int, idx::Int) where {TF}
  pm = it.pm; w = it.w
  @inbounds while Jb <= pm_nblocks(pm)
    cJ = pm.pairblocks[Jb]; r0 = first(cJ); ntile = length(cJ)
    σ0 = Jb == 1 ? 0 : last(pm.σblocks[Jb-1]); σend = last(pm.σblocks[Jb])
    Ps = spanel(pm, Jb); Pa = apanel(pm, Jb)
    if phase == 1
      if idx <= ntile
        c = cJ[idx]
        reconstruct_slab_both!(w, Ps, Pa, idx, r0)
        return BothSlab{TF}(w, w.lutμ[c], w.lutν[c], σ0, pm.nao), (Jb, 1, idx + 1)
      end
      phase = 2; idx = ntile + 1
    end
    if idx <= size(Ps, 1)
      r = r0 + idx - 1
      reconstruct_mirror_both!(w, Ps, Pa, idx, cJ, ntile)
      return BothSlab{TF}(w, w.lutμ[r], w.lutν[r], σ0, σend), (Jb, 2, idx + 1)
    end
    Jb += 1; phase = 1; idx = 1
  end
  return nothing
end

"""
    pm_bra_half!(hν, hμ, s::BothSlab, C) -> (hν, hμ)

  Half-transform the bra pair of slab `s` with coefficients `C` (`[nao, m]`), filling
  `hν[i,ν] = Σ_μ ⟨μν|ρσ⟩ C[μ,i]` (μ→i, ν kept AO) and `hμ[i,μ] = Σ_ν ⟨μν|ρσ⟩ C[ν,i]` (ν→i, μ kept AO)
  — the two band GEMMs (`band_htrans!` on `G` and `Gt`) that the fused dressing outputs SHARE (compute
  once, reuse for every output). `hν`,`hμ` are caller-owned `[m, nao]` buffers (allocate once outside
  the loop). Cheap second-stage contractions (`hν`/`hμ` with the remaining coeffs) then form the outputs.
"""
@inline function pm_bra_half!(hν, hμ, s::BothSlab, C)
  band_htrans!(hν, C, s.w.G,  s.lo, s.hi)
  band_htrans!(hμ, C, s.w.Gt, s.lo, s.hi)
  return hν, hμ
end

# ---- @pmtensor: a per-slab einsum, the workhorse inside an `eachslab` loop --------------------
# `@pmtensor out[…] += s[μ,ν,ρ,σ] * D[…]` where `s` is the current PMSlab. Emits the native band GEMV
# + the ket-swap + the two Hermitian-mirror rows automatically (exactly add_coulomb!/add_exchange!),
# so the caller writes the physics per slab and fuses as many terms as it likes in one sweep. `D`
# contracting the bra index ν with ket label σ is Coulomb; with ket label ρ is exchange.
function _pmtensor_gen(ex)
  (ex isa Expr && ex.head === :(+=)) || error("@pmtensor: expected `out[…] += s[μ,ν,ρ,σ] * D[…]`")
  lhs, rhs = ex.args
  (lhs isa Expr && lhs.head === :ref) || error("@pmtensor: the output must be indexed, got `$lhs`")
  factors = (rhs isa Expr && rhs.head === :call && rhs.args[1] === :*) ? rhs.args[2:end] : Any[rhs]
  slabv = nothing; sidx = nothing; dens = nothing; didx = nothing
  for f in factors
    (f isa Expr && f.head === :ref) || error("@pmtensor: factors must be indexed, got `$f`")
    idx = f.args[2:end]
    if length(idx) == 4 && slabv === nothing
      slabv = f.args[1]; sidx = idx
    elseif length(idx) == 2
      dens = f.args[1]; didx = idx
    else
      error("@pmtensor: expected a 4-index slab `s[μ,ν,ρ,σ]` and one 2-index density `D[·,·]`")
    end
  end
  (slabv !== nothing && dens !== nothing) || error("@pmtensor: need `s[μ,ν,ρ,σ] * D[·,·]`")
  μ, ν, ρ, σ = sidx
  coulomb = didx[1] === ν && didx[2] === σ                # out[μ,ρ] += Σ_ν ⟨μν|ρσ⟩ D[ν,σ]
  exch    = didx[1] === ν && didx[2] === ρ                # out[μ,σ] += Σ_ν ⟨μν|ρσ⟩ D[ν,ρ]
  (coulomb || exch) || error("@pmtensor: density $didx must be [$ν,$σ] (Coulomb) or [$ν,$ρ] (exchange)")
  s = esc(slabv); D = esc(dens); O = esc(lhs.args[1])
  coulomb ? quote
      slab_bandmul!(view($O,:,$s.ρ), $s, view($D,:,$s.σ))
      $s.ρ < $s.σ && slab_bandtmul!(view($O,:,$s.σ), $s, view($D,:,$s.ρ))
      slab_mirror!($O, $s.ρ, $s, $D, $s.σ); $s.ρ < $s.σ && slab_mirrort!($O, $s.σ, $s, $D, $s.ρ)
    end : quote
      slab_bandmul!(view($O,:,$s.σ), $s, view($D,:,$s.ρ))
      $s.ρ < $s.σ && slab_bandtmul!(view($O,:,$s.ρ), $s, view($D,:,$s.σ))
      slab_mirrort!($O, $s.ρ, $s, $D, $s.σ); $s.ρ < $s.σ && slab_mirror!($O, $s.σ, $s, $D, $s.ρ)
    end
end

"""
    @pmtensor out[…] += s[μ,ν,ρ,σ] * D[…]

  A per-slab einsum for use inside `for s in eachslab(pm) … end`: contract the current slab
  (`s[μ,ν,ρ,σ]` = `⟨μν|ρσ⟩`) with a density `D`, accumulating into `out`. The macro emits the
  native band GEMV, the ket-swap, and both Hermitian-mirror rows (i.e. [`slab_bandmul!`](@ref) /
  [`slab_bandtmul!`](@ref) / [`slab_mirror!`](@ref) / [`slab_mirrort!`](@ref)) — the full two-role
  contribution of that slab, identical to `FockFactory`'s `add_coulomb!`/`add_exchange!`. Stack
  several `@pmtensor` lines in one loop to fuse them over a single reconstruction (as `ao_JK!` does).
  `D[ν,σ]` (ket label σ) ⇒ Coulomb `out[μ,ρ]`; `D[ν,ρ]` (ket label ρ) ⇒ exchange `out[μ,σ]`.
"""
macro pmtensor(ex)
  return _pmtensor_gen(ex)
end

end # module PMStore
