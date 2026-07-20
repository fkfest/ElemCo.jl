# PM supermatrix AO-integral store — implementation plan

Branch: `pm-ao-store` (based on `non-df-ao-integrals` @ acf2365f).
Status: **plan — not implemented**. Each phase ends with a hard validation gate;
do not start a phase before the previous gate is green.

## 1. Context and goal

The AO-direct path stores exact AO ERIs as `"ao_int2"`: `int2[μ,ν,tri(ρσ)] = ⟨μν|ρσ⟩`
(physicist notation, ket-pair triangular, joint symmetry `⟨μν|ρσ⟩=⟨νμ|σρ⟩` implicit),
size ≈ n⁴/2 elements. The kext 4-external contraction (`calc_K2`) contracts it at
n⁴nocc² flops. The ± (plus/minus) factorization halves those flops, but the existing
`cc.use_pm_kext` path (`calc_pm_K2!`) builds the ± combinations **per CC iteration**
from the raw store — O(n⁴) bandwidth-bound, partly strided packing work per iteration —
and measured *slower* than the standard GEMM on almost all sizes (roofline: it trades
peak-rate GEMM flops for bandwidth-rate packing; break-even needs ≳25 correlated
occupied orbitals).

**This plan persists the ± supermatrices once, at integral-build time.** Result:

| | flops (kext) | streamed/iter (kext) | disk | Fock build | dressing | MO transform |
|---|---|---|---|---|---|---|
| now (`ao_int2`) | n⁴nocc² | n⁴/2 | n⁴/2 | n⁴/2 read | n⁴/2 read | n⁵ compute |
| PM store | **n⁴nocc²/2** | **n⁴/4** | **n⁴/4** | **n⁴/4 read (~2×)** | = (adapter) or n⁴/4 (native) | = (adapter) |

The two supermatrices are symmetric matrices over the packed pair space; only their
lower **block**-triangles are stored as dense column panels, so every kernel is a plain
zero-copy GEMM (`'N'`/`'T'` flags), never a packed-format special case.

## 2. Format specification (normative)

### 2.1 Pair space

One pair space for both matrices: `P = {(p,q): 1 ≤ p ≤ q ≤ n}`, packed index
`tri(p,q) = q(q−1)/2 + p` (= `QMTensors.uppertriangular_index`), `npp = n(n+1)/2`.
The antisymmetric matrix uses the **same** indexing with exact zeros on diagonal-pair
rows/columns (waste ~2/n of its storage — accepted for unified indexing/blocking).

### 2.2 The supermatrices

Exactly the `calc_tri_sym_antisym!` (src/tools/utensors.jl:247) output convention:

```
Vs[tri(μν), tri(ρσ)] = ⟨μν|ρσ⟩ + ⟨νμ|ρσ⟩     (μ<ν)      "s" = symmetric
Vs[tri(μμ), tri(ρσ)] = 2⟨μμ|ρσ⟩                          (row diagonal: factor 2)
Va[tri(μν), tri(ρσ)] = ⟨μν|ρσ⟩ − ⟨νμ|ρσ⟩     (μ<ν)      "a" = antisymmetric
Va[tri(μμ), ·] = 0,   Va[·, tri(ρρ)] = 0
```

By exchange `⟨νμ|ρσ⟩ = ⟨μν|σρ⟩`, the bra-symmetrization equals the ket-symmetrization —
the convention is self-consistent on both index sides (both diagonals carry the factor 2
automatically). **Both matrices are symmetric**: `V[r,c] = V[c,r]` (hermiticity + exchange;
real orbitals — the store is real-only, assert it).

Inversion (the unpack used by every pair-splitting consumer):

```
⟨μν|ρσ⟩ = (Vs + Va)[tri(μν), tri(ρσ)] / 2    (μ<ν)
⟨νμ|ρσ⟩ = (Vs − Va)[tri(μν), tri(ρσ)] / 2    (μ<ν)
⟨μμ|ρσ⟩ =  Vs[tri(μμ), tri(ρσ)] / 2
```

### 2.3 The contraction identity (kernel-level ground truth)

For any density-like `D[ρ,σ]` (full n×n), define packed ± vectors

```
Ds[tri(ρσ)] = (D[ρ,σ]+D[σ,ρ])/2  (ρ<σ),   Ds[tri(ρρ)] = D[ρ,ρ]/2
Da[tri(ρσ)] = (D[ρ,σ]−D[σ,ρ])/2  (ρ<σ),   Da[tri(ρρ)] = 0
```

and `K[μ,ν] := Σ_{ρσ} ⟨μν|ρσ⟩ D[ρ,σ]` (full sums). Then **exactly**:

```
(Vs·Ds)[tri(μν)] = ½(K[μ,ν] + K[ν,μ])   (μ<ν);   (Vs·Ds)[tri(μμ)] = K[μ,μ]
(Va·Da)[tri(μν)] = ½(K[μ,ν] − K[ν,μ])   (μ<ν)
⇒  K[μ,ν] = (Vs·Ds + Va·Da)[tri(μν)],  K[ν,μ] = (Vs·Ds − Va·Da)[tri(μν)],  K[μ,μ] = (Vs·Ds)[tri(μμ)]
```

No stray ½ anywhere — the weights are absorbed by the conventions above. Every kernel
in this plan is an instance of this identity; Phase 0 encodes it as a unit test.

Consistency with existing code: `calc_D2(...; scalepp=true)` (cc.jl:1192–1195) already
produces the `/2` rs-diagonal; `calc_pm_K2!`'s `D2s/D2a = 0.5·(D2[:,trioo] ± D2[:,trioo_swap])`
are **exactly** `Ds/Da` over the rs pairs, because the kext density obeys
`D[ρσ,ij] = D[σρ,ji]` — the ij-symmetrization *is* the rs-± fold. The existing D-prep
and 4-quadrant output scatter of `calc_pm_K2!` (cc.jl:2428) carry over verbatim.

### 2.4 Block-column panel layout

Blocks are **σ-aligned**: breakpoints `0 = σ₀ < σ₁ < … < σ_nb = n`; block `J` covers ket
orbitals `σ ∈ σ_{J-1}+1 : σ_J`, i.e. the contiguous packed-pair range
`c_J = tri(1,σ_{J-1}+1) : tri(σ_J,σ_J)`. σ-alignment guarantees each block is "all pairs
whose larger member is in a σ range" — the same column structure every existing sweep
(`ao_occ_early` etc.) already walks, and the strict-pair subset stays contiguous.

Stored data per matrix = the lower block-trapezoid **column panels**:

```
panel_J = V[ r ∈ first(c_J):npp ,  c ∈ c_J ]      (dense, column-major)
        = [ T_J  ]   T_J = diagonal tile  V[c_J, c_J]        (stored FULL, applied once)
          [ B_J  ]   B_J = subdiagonal    V[c_J-end+1:npp, c_J]
```

File = panels concatenated `J = 1..nb` (ascending; this is also the streaming order).
Total = Σ_J |rows_J|·|c_J| ≈ npp²/2·(1 + O(1/nb)) per matrix ⇒ **both ≈ n⁴/4**.

Panel column-count target: from `available_memory(EC)` as a **ceiling** with a moderate
cap — do *not* maximize to the budget (measured lesson from the occ-early batching:
zero-padding/oversized blocks are slower; sweep before hard-coding).

### 2.5 Files & metadata

- `"ao_pm_s"`, `"ao_pm_a"`: 1-D `newmmap` files (TensorTools; see integral_tools.jl:380
  for the `ao_int2` pattern), panels as subrange `reshape` views — zero-copy GEMM operands.
- `"ao_pm_meta"`: the σ breakpoints (Vector{Int}) via `save!`, so reader and writer can
  never disagree on the blocking. `open` recomputes offsets from it (closed form).
- Register descriptions; extend `delete_ao_integrals!` (integral_tools.jl:423) so
  geometry/basis changes invalidate the PM files together with `ao_int2`/`S_AA`/`h_AA`.

## 3. Struct and API

New file `src/integrals/pm_store.jl`, `module PMStore`, included in src/ElemCo.jl
**before** `scf/fockfactory.jl` (line 41) — PMStore needs only QMTensors, TensorTools,
ECInfos, Utils, Buffers; FockFactory, IntegralTools and CoupledCluster then `using` it
(clean dependency direction; IntegralTools comes later at line 43, so it can call the
builder for fused generation).

```julia
"""
    PMSupermatrices{T}

Persisted ± AO-integral supermatrices ... (put the §2 conventions verbatim in this
docstring — it is the single normative reference all kernels cite).
"""
struct PMSupermatrices{T}
  n::Int                              # AOs
  npp::Int                            # n(n+1)/2
  σblocks::Vector{UnitRange{Int}}     # ket-orbital range per block
  pairblocks::Vector{UnitRange{Int}}  # packed-pair range c_J per block
  offsets::Vector{Int}                # panel start offsets (elements) in each file
  sfile::IOStream; smap::Vector{T}    # mmapped ao_pm_s
  afile::IOStream; amap::Vector{T}    # mmapped ao_pm_a
end

pm_blocks(n::Int; maxcols::Int) -> (σblocks, pairblocks, offsets, totallen)
create_pm_store(EC; maxcols)    -> PMSupermatrices   # newmmap both files + meta
open_pm_store(EC)               -> PMSupermatrices   # mmap existing + read meta
close_pm_store!(EC, pm)
pm_exists(EC) -> Bool

# panel accessors (all zero-copy views)
spanel(pm, J) / apanel(pm, J)   -> Matrix view  (rows first(c_J):npp × cols c_J)
diagtile(P, pm, J)              -> @view of T_J  (first |c_J| rows)
subpanel(P, pm, J)              -> @view of B_J  (remaining rows)

# builder (Phase 1) and consumers (Phases 2–5)
pm_from_joint!(EC)                          # one-time build from "ao_int2"
pm_K2!(pm, D2, tripp) -> K2pq               # kext (replaces per-iteration calc_pm_K2!)
pm_JK!(J, K, pm, Dj, Dk); pm_J2K!(...)      # Fock kernels
gen_fock(EC, pm::PMSupermatrices, h1, CMOl, CMOr); gen_ufock(...)   # dispatch overloads
pm_joint_slabs!(dest, pm, σrange)           # adapter: reconstruct ao_int2-format slabs
```

Style requirement: every kernel ≤ ~60 lines, one comment per GEMM stating which term of
the §2.3 identity it computes, no packed-index arithmetic inside inner loops (views +
precomputed ranges only).

## 4. Phases

### Phase 0 — convention pinning + dense harness *(small, do first)*

1. New testitem `test/pm_store_test.jl` (tag it into the existing runner scheme):
   build a random **joint-symmetric** packed `int2` (symmetrize diagonal slabs;
   see test/ao_integrals_test.jl for the dense-reference style), form Vs/Va with
   `calc_tri_sym_antisym!`, and assert: symmetry `V[r,c]=V[c,r]`, the inversion
   formulas, and the §2.3 contraction identity vs a dense `detri_int2` reference
   (random nonsymmetric D), at n ∈ {6, 9, 13}, to 1e-13.
2. Baseline check: run one AO-direct CCSD with `cc.use_pm_kext=true` vs `false` —
   energies must agree (documents that the current pm path is correct before reuse).

**Gate 0**: new testitem green; energies agree to 1e-10.

### Phase 1 — PMStore module + builder

1. `src/integrals/pm_store.jl` per §3; include at src/ElemCo.jl:41 (before fockfactory).
2. `pm_blocks`: σ-aligned, balanced pair counts, `maxcols` from memory ceiling + cap.
   Edge cases: block of a single σ; σ=1 block (no strict pairs); nb=1 (tiny systems).
3. `pm_from_joint!(EC)`: stream `ao_int2` σ-column chunks (`mmap3idx` + 
   `uppertriangular_range`, exactly like `ao_occ_early`'s read), per chunk call
   `calc_tri_sym_antisym!` → full-height [npp × cols] pieces, **write only rows
   ≥ first(c_J)** (the discarded upper rows are the hermitian partners already stored
   in earlier panels). Sequential read, sequential write, one pass.
4. Round-trip test in pm_store_test.jl: joint → PM → reconstruct via inversion → equal
   to `detri_int2` dense reference (1e-14); file sizes ≈ npp(npp+1) elements total.

**Gate 1**: round-trip green; `ao_integrals` testitem still 74/74 (nothing consumes PM yet).

### Phase 2 — kext on the PM store *(the payoff)*

1. `pm_K2!(pm, D2, tripp)`: copy `calc_pm_K2!`'s D-prep (D2s/D2a; unchanged — see §2.3)
   and 4-quadrant output scatter; replace the middle (per-iteration
   `calc_tri_sym_antisym!` + flat GEMMs) with the panel loop:
   ```
   for J: (three GEMMs per matrix, all zero-copy on the mmap)
     sK2[c_J,:]    += T_J  · Ds[c_J,:]        # diagonal tile — once, 'N' only
     sK2[below,:]  += B_J  · Ds[c_J,:]        # 'N'
     sK2[c_J,:]    += B_Jᵀ · Ds[below,:]      # 'T' (transpose FLAG, no copy)
   ```
   (Va: identical with aK2/Da. Panels streamed in file order — each element read once.)
2. Wire in `cc_kext!` (cc.jl:2258/2316/2346): if `pm_exists(EC)` use `pm_K2!`; the old
   on-the-fly `calc_pm_K2!` stays as cross-check. New option `int.ao_pm::Bool=false`:
   when true, `ao_integrals` also builds the PM store (via `pm_from_joint!` for now).
3. Tests: AO-direct CCSD/DCSD energies PM vs standard to 1e-10 (closed + open shell —
   the AO-direct open-shell branch reuses the same spin-free store for αα/ββ/αβ; verify
   all three); pm_K2! vs calc_K2 vs calc_pm_K2! on random amplitudes to 1e-11.
4. Benchmark (report in the PR): kext wall time standard vs PM at
   nao ∈ {60,100,140} × nocc ∈ {5,10,20}. **Acceptance: PM ≥ standard everywhere**
   (unlike streaming-pm, the persisted form also wins the bandwidth-bound regime —
   n⁴/4 vs n⁴/2 streamed), and ≥1.5× where nocc ≥ 10.

**Gate 2**: full suite 811/811 + both new testitems; benchmark table meets acceptance.

### Phase 3 — Fock builders

1. `pm_JK!(J, K, pm, Dj, Dk)`: walk panels; per panel unpack (inversion formulas) a
   dense slab piece and apply the `ao_JK!` (fockfactory.jl:161) slab identities in
   **both roles** of each stored element — (bra r, ket c) and, by hermiticity,
   (bra c, ket r). Invariant to assert in the test: every logical ⟨μν|ρσ⟩ contributes
   exactly once. BLAS-2 (`mul!` on views) like `ao_JK!`; O(panel) scratch via Buffers.
   Write it against the Phase-0 harness first: `pm_JK!` vs `ao_JK!` on random density,
   1e-13, several n and blockings. Same for `pm_J2K!`.
2. Dispatch overloads `gen_fock(EC, pm::PMSupermatrices, h1, CMOl, CMOr)`/`gen_ufock`
   (fockfactory.jl:219/239 siblings); switch `ao_core_fock`/`ao_core_ufock` (cc.jl)
   and the AO-HF fockbuilder closures (hf.jl:331/371) to the PM store when present.
3. Benchmark: AO-HF iteration time (bandwidth-bound → expect ~2× at larger nao).

**Gate 3**: HF/UHF energies identical to 1e-11 vs joint-store path; suite green.

### Phase 4 — MO transform (adapter)

1. `pm_joint_slabs!(dest, pm, σrange)`: reconstruct `ao_int2`-format slabs for a σ-chunk.
   Access pattern: native panel columns + row-slices of earlier panels (regular strided
   reads — contiguous runs of |c_J| per column; total I/O n⁴/2 per full sweep, same as
   today — irrelevant here, the transform is n⁵ compute-bound).
2. Feed `generate_mo_dump`'s use of the `transform_int2` family (integral_tools.jl:463–642)
   through the adapter — either refactor their int2 access to chunk iteration, or (zero-risk
   fallback) materialize a temporary joint file for the transform and delete it after.
   Choose whichever keeps the transforms untouched; document the choice.

**Gate 4**: derive-path tests (UCCSD(T)/λ/EOM routes in ao_integrals testitem) green,
energies identical to 1e-10.

### Phase 5 — dressing sweeps (decision gate, then maybe native)

1. **Ship first with the adapter**: `ao_occ_early`/`ao_ss_blocks`/`ao_os_blocks` read
   per-σ chunks from `pm_joint_slabs!` instead of the raw mmap (their loop structure
   already processes per-σ slabs — the change is the input source only). I/O = today's
   n⁴/2; flops unchanged; the freshly verified sweeps stay intact.
2. Profile a representative AO-direct CCSD (nao ≥ 100): if the dressing is I/O-bound,
   implement the native tile-driven sweep (n⁴/4): per panel, unpack the ± pieces,
   half-transform, and accumulate contributions in both hermiticity roles into the
   in-RAM `v_ooAA/v_AooA/v_oAoA` accumulators (they are unordered sums — the
   deferred-contribution pattern already used for the ρ>σ batch). This is the most
   intricate step of the whole plan; **dense-reference-first is mandatory**
   (template: the ao_os_blocks rewrite validated every intermediate against
   `detri_int2` before touching cc.jl).

**Gate 5**: suite green; document the profile numbers and the Option-A/B decision.

### Phase 6 — retire the joint store

1. Fused generation: `ao_integrals` (integral_tools.jl:370) writes the PM panels
   directly from `eri_2e4idx_tri!` output chunks (the ± fold is slab-local), stops
   writing `ao_int2`; `int.ao_pm` default flips to true; keep a debug option to write
   the joint format. Disk: n⁴/4 total.
2. Invalidation/freshness: `ensure_ao_integrals!`, `delete_ao_integrals!`, `@setupEC`/
   `@dummy` hooks cover the PM files; `EC.ao_direct` checks `pm_exists` where it now
   checks `file_exists(EC,"ao_int2")` (drivers.jl:299/521, ElemCo.jl:886, cc.jl:1754).
3. Deprecate `cc.use_pm_kext` (the on-the-fly path) after one release note; docs +
   CHANGELOG (fold into the pending Step-5 docs task of the AO feature).

**Gate 6**: full suite green with the PM-only store; disk measured ≈ half.

## 5. Attention list (bug sources, in priority order)

1. **Diagonal-pair weights.** Every kernel must cite §2.2/§2.3. The three factor
   conventions (Vs row/col diagonal = 2×, Ds diagonal = ½×, Va diagonals = 0) are the
   single most likely source of silent errors. The Phase-0 identity test exists to
   catch exactly this — run it against every new kernel.
2. **Diagonal tile applied once.** The 'T' role runs on `B_J` only, never `T_J`.
3. **Random-D validation.** Always validate kernels with *nonsymmetric* random D and
   nocca≠noccb-style asymmetric sizes — symmetric test data hides transposition bugs
   (lesson from the non-Hermitian dressed-Fock bug).
4. **Dense-reference-first.** For every kernel: standalone check against `detri_int2`
   dense einsum *before* wiring into ElemCo paths (this caught real bugs in both the
   occ-early and os-blocks rewrites).
5. **`@mtensor` tensor-product into a `@view` errors** ("output aliased") — a pure
   outer product must target a full array (zero-tail trick) or an explicit loop;
   contractions into views are fine.
6. **Memory budget is a ceiling, not a target** for panel widths — sweep small/medium/
   large before fixing the default (measured: oversized blocks lose).
7. **Threading**: GEMMs are BLAS-threaded; unpack kernels Julia-threaded (pattern:
   `calc_tri_sym_antisym!`); never nest both. The test runner pins BLAS threads.
8. **Real-only**: `@assert T <: Real` at create with a clear message. (Complex would
   need a conjugate fold: 'C' roles + conj in unpack — out of scope, leave a comment.)
9. **mmap lifecycle**: mirror `cc_kext!`'s open/close discipline; two consumers must
   not double-close; `closemmap` on create, plain `close` on read (see ao_int2 usage).
10. **σ-aligned blocking edge cases**: single-σ blocks, first block (σ=1 has one pair,
    no strict pair), nb=1, nao smaller than maxcols.
11. **Open-shell αβ**: the AO-direct branch uses the spin-free store for αα/ββ/αβ —
    make sure the αβ kext D2 path (calc_D2ab) is exercised in the PM tests.
12. **Do not regress the standard path**: `calc_K2` stays the fallback whenever no PM
    store exists (external FCIDUMP runs, `int.ao_pm=false`).

## 6. Reference map (read these before coding)

| what | where |
|---|---|
| ± fold kernel (reuse verbatim) | `calc_tri_sym_antisym!` src/tools/utensors.jl:247 |
| existing pm kext (D-prep + scatter to reuse) | `calc_pm_K2!` src/cc/cc.jl:2428 |
| standard kext (fallback + benchmark baseline) | `calc_K2` src/cc/cc.jl:2477; call sites cc.jl:2258/2316/2346 |
| kext density + `scalepp` ½-diagonal | `calc_D2` src/cc/cc.jl:1166 (1192–1195) |
| Fock slab kernels (identities to mirror) | `ao_JK!`/`ao_J2K!` src/scf/fockfactory.jl:161/187 |
| explicit-int2 Fock entry points | `gen_fock`/`gen_ufock` fockfactory.jl:219/239; hf.jl:331/371 |
| occ-early sweep pattern (read/batch template) | `ao_occ_early` cc.jl:1639; `ao_os_blocks` cc.jl:1872 |
| generation + invalidation | `ao_integrals` integral_tools.jl:370; `delete_ao_integrals!` :423 |
| MO transform family | integral_tools.jl:463–642 |
| mmap API | `newmmap`/`mmap3idx`/`closemmap` src/tools/tensortools.jl:114+ |
| pair indexing | `uppertriangular_index/range` src/tools/utensors.jl:73/106 (exported via QMTensors) |
| include point for the new module | src/ElemCo.jl:41 (before fockfactory.jl) |
| dense-reference test style | test/ao_integrals_test.jl (detri_int2-based checks) |

## 7. Success criteria (overall)

- Full suite 811/811 (+ new pm_store testitem) at every gate.
- kext: PM ≥ standard wall time at **all** benchmarked sizes; ≥1.5× at nocc ≥ 10.
- AO-HF iteration: measurably faster at nao ≥ 100 (bandwidth ÷2).
- Disk after Phase 6: ≈ half of today's `ao_int2`.
- No performance or accuracy regression on any non-AO path (FCIDUMP/DF suites).
