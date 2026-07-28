"""
AO-direct MO integral blocks.

Everything that turns the persisted stores into the MO blocks the coupled-cluster equations read:
the half-transformed → MO block engine (`ht_mo_block` and the `HT_MO_BLOCK_SPEC` table naming, per
block, which store and which coefficient goes on each free slot), the dressed same-/opposite-spin
block assembly, and the ± `kext` 4-external kernels.

Unlike the sweeps in `PMStore` (`ht_occ_early`, `pm_occ_early`, …), these know about the MO spaces
(`EC.space`, the active orbitals) and the method's index conventions, so they belong on the
coupled-cluster side. Included into `CoupledCluster`; not a module of its own.
"""

"""
    ht_mo_block(EC, htkey, role, CX, CY, CZ) -> W

  General half-transformed-store → MO-block kernel: build a 4-index MO integral block that carries the
  store's (occupied) bra index `i` plus three MO indices transformed by the coefficient matrices
  `CX`/`CY`/`CZ`, in ONE sweep over the store `htkey` (`Σ_μ⟨μν|ρσ⟩C[μ,i]` in both ket orders, read
  column-by-column with [`ht_column!`](@ref PMStore.ht_column!)). With

      W[i,X,Y,Z] = Σ_μνρσ ⟨μν|ρσ⟩ C[μ,i] CX[ν,X] CY[ρ,Y] CZ[σ,Z]   (`role=:A`, occ i on bra-1)
      W[X,i,Y,Z] = Σ_μνρσ ⟨μν|ρσ⟩ CX[μ,X] C[ν,i] CY[ρ,Y] CZ[σ,Z]   (`role=:B`, occ i on bra-2)

  every block a CC method needs (which always has ≥1 occupied index) is one call plus a fixed output
  permutation — see [`save_mo_block!`](@ref). Per σ-column two GEMMs transform the free bra (`ν`/`μ`)
  and ket-1 (`ρ`) indices into the `[(i,X),Y]` intermediate; stacking those over σ and one final GEMM
  contracts ket-2 (`σ`) into `Z`. Generic over the element type (the store applies plain `C`, matching
  the AO-direct/FCIDUMP `detri` convention). Cost: nocc·nao³ read + GEMMs; peak RAM ≈ the output block.
"""
function ht_mo_block(EC::ECInfo, htkey::AbstractString, role::Symbol,
                     CX::AbstractMatrix, CY::AbstractMatrix, CZ::AbstractMatrix)
  role === :A || role === :B || error("ht_mo_block: role must be :A or :B (got $role)")
  ht = open_ht_store(EC, htkey)
  T = eltype(ht.map); n = ht.nao; m = ht.m
  nX = size(CX, 2); nY = size(CY, 2); nZ = size(CZ, 2)
  size(CX,1) == n && size(CY,1) == n && size(CZ,1) == n ||
    error("ht_mo_block: coefficient rows must equal nao=$n")
  Aσ = zeros(T, m*n, n); Bσ = zeros(T, m*n, n)
  IXY = zeros(T, m*nX*nY, n)                        # [(i,X,Y), σ]  stacked ket-2 columns
  iXρ = zeros(T, m, nX, n); iXY = zeros(T, m, nX, nY)
  @inbounds for σ in 1:n
    ht_column!(Aσ, Bσ, ht, σ)
    S = reshape(role === :A ? Aσ : Bσ, m, n, n)    # [i, freeAO(ν|μ), ρ]
    @mtensor iXρ[i,X,ρ] = S[i,f,ρ] * CX[f,X]        # free bra → X
    @mtensor iXY[i,X,Y] = iXρ[i,X,ρ] * CY[ρ,Y]      # ket-1 → Y
    @views IXY[:, σ] .= vec(iXY)
  end
  close_ht_store!(EC, ht)
  W = reshape(IXY * CZ, m, nX, nY, nZ)             # ket-2 σ → Z (one BLAS-3 GEMM)
  return role === :A ? W : permutedims(W, (2, 1, 3, 4))
end

const HT_MO_BLOCK_SPEC = Dict{String,NamedTuple{(:store,:role,:X,:Y,:Z,:perm,:swap),
                                                Tuple{Symbol,Symbol,Char,Char,Char,NTuple{4,Int},Bool}}}(
  # closed-shell / same-spin α
  "ovoo" => (store=:a, role=:A, X='v', Y='o', Z='o', perm=(1,2,3,4), swap=false),  # ⟨ia|jk⟩
  "ooov" => (store=:a, role=:A, X='o', Y='o', Z='v', perm=(1,2,3,4), swap=false),  # ⟨ij|ka⟩
  "vovv" => (store=:a, role=:B, X='v', Y='v', Z='v', perm=(1,2,3,4), swap=false),  # ⟨ai|bc⟩
  "voov" => (store=:a, role=:B, X='v', Y='o', Z='v', perm=(1,2,3,4), swap=false),  # ⟨ai|jb⟩
  "vovo" => (store=:a, role=:B, X='v', Y='v', Z='o', perm=(1,2,3,4), swap=false),  # ⟨ai|bj⟩
  "vooo" => (store=:a, role=:B, X='v', Y='o', Z='o', perm=(1,2,3,4), swap=false),  # ⟨ai|kj⟩
  "vvvo" => (store=:a, role=:B, X='v', Y='v', Z='v', perm=(3,4,1,2), swap=true),   # ⟨ab|ck⟩ = conj⟨ck|ab⟩
  # same-spin β (the α entries with every space flipped, off the β store)
  "VOOO" => (store=:b, role=:B, X='V', Y='O', Z='O', perm=(1,2,3,4), swap=false),  # ⟨AI|KJ⟩
  "VVVO" => (store=:b, role=:B, X='V', Y='V', Z='V', perm=(3,4,1,2), swap=true),   # ⟨AB|CK⟩
  "VOVV" => (store=:b, role=:B, X='V', Y='V', Z='V', perm=(1,2,3,4), swap=false),  # ⟨AI|BC⟩
  "oovo" => (store=:a, role=:A, X='o', Y='v', Z='o', perm=(1,2,3,4), swap=false),  # ⟨ij|ak⟩
  "OOVO" => (store=:b, role=:A, X='O', Y='V', Z='O', perm=(1,2,3,4), swap=false),  # ⟨IJ|AK⟩
  "VOOV" => (store=:b, role=:B, X='V', Y='O', Z='V', perm=(1,2,3,4), swap=false),  # ⟨AI|JB⟩
  "VOVO" => (store=:b, role=:B, X='V', Y='V', Z='O', perm=(1,2,3,4), swap=false),  # ⟨AI|BJ⟩
  # opposite spin (index 1,3 = α = electron 1; index 2,4 = β = electron 2)
  "vOoO" => (store=:b, role=:B, X='v', Y='o', Z='O', perm=(1,2,3,4), swap=false),  # ⟨aI|kJ⟩
  "oVoO" => (store=:a, role=:A, X='V', Y='o', Z='O', perm=(1,2,3,4), swap=false),  # ⟨iB|kJ⟩
  "vOvV" => (store=:b, role=:B, X='v', Y='v', Z='V', perm=(1,2,3,4), swap=false),  # ⟨aI|bC⟩
  "oVvV" => (store=:a, role=:A, X='V', Y='v', Z='V', perm=(1,2,3,4), swap=false),  # ⟨iB|aC⟩
  "oOvO" => (store=:a, role=:A, X='O', Y='v', Z='O', perm=(1,2,3,4), swap=false),  # ⟨iJ|aK⟩
  "oOoV" => (store=:a, role=:A, X='O', Y='o', Z='V', perm=(1,2,3,4), swap=false),  # ⟨iJ|kB⟩
  "vOoV" => (store=:b, role=:B, X='v', Y='o', Z='V', perm=(1,2,3,4), swap=false),  # ⟨aI|kB⟩
  "oVvO" => (store=:a, role=:A, X='V', Y='v', Z='O', perm=(1,2,3,4), swap=false),  # ⟨iB|aJ⟩
  "vVvO" => (store=:b, role=:B, X='v', Y='v', Z='V', perm=(3,4,1,2), swap=true),   # ⟨aB|cK⟩ = conj⟨cK|aB⟩
  "vVoV" => (store=:a, role=:B, X='V', Y='V', Z='v', perm=(4,3,2,1), swap=true),   # ⟨aB|iD⟩ = conj⟨Di|Ba⟩
)

"""
    save_mo_block!(EC, name, htkey, Co, Cv)

  Build the closed-shell bare MO block `name` (e.g. `"vvvo"`, `"ovoo"`, `"vovv"`, `"ooov"`, `"voov"`,
  `"vovo"`) from the occupied-bra store `htkey` via [`ht_mo_block`](@ref) and save it (mmapped, `"tmp"`
  so it is reclaimed at end-of-run) under its plain space name, in the exact index order the consumers
  read. `Co`/`Cv` are the occupied/virtual MO coefficients. The store's own bra index supplies the
  block's stored occupied index. Blocks flagged `swap` use the bra↔ket Hermiticity relation and are
  real-only from this bra-store (see [`ht_mo_block`](@ref)); complex needs the (future) ket-transformed
  store.
"""
function save_mo_block!(EC::ECInfo, name::AbstractString, htkeys, coefs)
  spec = get(HT_MO_BLOCK_SPEC, name, nothing)
  isnothing(spec) && error("save_mo_block!: no spec for block \"$name\"")
  htkey = htkeys[spec.store]
  spec.swap && eltype(coefs[spec.X]) <: Complex &&
    error("save_mo_block!: block \"$name\" uses a bra↔ket Hermiticity swap, which is real-only from a " *
          "bra-transformed store; complex needs a ket-transformed store (pending complex-AO integrals)")
  W = ht_mo_block(EC, htkey, spec.role, coefs[spec.X], coefs[spec.Y], coefs[spec.Z])
  spec.perm == (1,2,3,4) || (W = permutedims(W, spec.perm))
  spec.swap && (W = conj(W))                       # Hermiticity conj (no-op for real)
  save!(EC, name, W; description="tmp mo block ($name)")
  return name
end

"""
    build_ht_mo_blocks!(EC, names)

  Build the closed-shell bare MO blocks `names` (e.g. `("vvvo","ovoo")` for (T), plus `"vovv"`/`"ooov"`
  for Λ(T)) that the AO-direct triples / λ-triples read, from the occupied-bra half-transformed store
  `"ht_oAAA"` (still on file from [`ao_cc_setup!`](@ref)) and the active-space MO coefficients. Each
  block is saved (mmapped, `"tmp"`) under its plain name via [`save_mo_block!`](@ref); the consumers
  pick them up with `load4idx`.
"""
function build_ht_mo_blocks!(EC::ECInfo, names; Rv=nothing)
  ht_exists(EC, "ht_oAAA") ||
    error("build_ht_mo_blocks!: half-transformed store \"ht_oAAA\" not found — AO-direct (T) needs the " *
          "± supermatrix store; it is built in ao_cc_setup!")
  cMO = ao_direct_orbitals(EC); SP = EC.space
  Co = cMO[:, SP['o']]; Cv = cMO[:, SP['v']]
  # `Rv`: fold a virtual-space rotation (the pseudo-canonicalization of (T)) straight into the
  # coefficients, so the blocks come out with their virtual indices already in that basis instead of
  # being built and then rotated — rotating three virtual indices of a `vvvo` costs ~3·nv⁴·no, more
  # than building the block. The OCCUPIED index of these blocks comes from the store's bra (fixed
  # when `ht_oAAA` was built), so it is left plain and rotated afterwards by
  # `pseudocan_transform!(...; spaces=:occ)` — an occ-occ contraction, which is cheap.
  isnothing(Rv) || (Cv = Cv * Rv)
  for name in names
    save_mo_block!(EC, name, "ht_oAAA", Co, Cv)
  end
  return
end

"""
    build_ht_mo_blocks_unrestricted!(EC, names)

  Unrestricted analogue of [`build_ht_mo_blocks!`](@ref): build the bare MO blocks `names` (same-spin
  `vvvo`/`vooo`/`VVVO`/`VOOO` and the opposite-spin `vVvO`/`vVoV`/`vOoO`/`oVoO` the unrestricted (T)
  reads) from the per-spin half-transformed stores `"ht_oAAA_a"`/`"ht_oAAA_b"` (built in
  [`ao_cc_setup!`](@ref)) and the per-spin MO coefficients. Each block's spec says which store supplies
  its occupied index and which spin's coefficients go on the free slots.
"""
function build_ht_mo_blocks_unrestricted!(EC::ECInfo, names; Rv=nothing)
  (ht_exists(EC, "ht_oAAA_a") && ht_exists(EC, "ht_oAAA_b")) ||
    error("build_ht_mo_blocks_unrestricted!: per-spin half-transformed stores not found — AO-direct " *
          "unrestricted (T)/Λ needs the ± supermatrix store; they are built in ao_cc_setup!")
  cMOsm = load_orbitals(EC); SP = EC.space
  cMOa = Matrix(cMOsm.α); cMOb = Matrix(cMOsm.β)
  coefs = Dict('o' => cMOa[:, SP['o']], 'v' => cMOa[:, SP['v']],
               'O' => cMOb[:, SP['O']], 'V' => cMOb[:, SP['V']])
  # per-spin virtual rotation folded into the coefficients (see `build_ht_mo_blocks!`): the blocks
  # come out with their virtual indices already pseudo-canonical, so only their occupied indices
  # have to be rotated afterwards (`pseudocan_transform!(...; spaces=:occ)`).
  if !isnothing(Rv)
    coefs['v'] = coefs['v'] * Rv.α
    coefs['V'] = coefs['V'] * Rv.β
  end
  htkeys = Dict(:a => "ht_oAAA_a", :b => "ht_oAAA_b")
  for name in names
    save_mo_block!(EC, name, htkeys, coefs)
  end
  return
end

"""
    ao_ss_blocks(pm, Lo, Lv, Ro, Rv) -> NamedTuple

  Same-spin occ-early pass (the closed-shell [`ao_dressed_ints`](@ref) kernel, reused per spin):
  one [`pm_occ_early`](@ref) sweep over the ± supermatrix store builds three occupied-contracted
  intermediates, then only the two remaining AO indices are transformed into the needed spaces.
  Returns the dressed `oooo/oovo/oovv/voov/vovo/vooo` blocks (bra columns from the dressed
  `Lo,Lv`, ket from `Ro,Rv`). No `nao⁴` tensor and no all-virtual block is ever formed.
"""
function ao_ss_blocks(pm::PMSupermatrices, Lo, Lv, Ro, Rv; membytes::Int=typemax(Int))
  v_ooAA, v_AooA, v_oAoA = pm_occ_early(pm, Lo, Ro)
  return ao_ss_finish(v_ooAA, v_AooA, v_oAoA, Lv, Ro, Rv)
end

"Transform the remaining AO indices of the occ-early intermediates into the dressed same-spin blocks."
function ao_ss_finish(v_ooAA, v_AooA, v_oAoA, Lv, Ro, Rv)
  @mtensor v_oooA[i,j,k,σ] := v_ooAA[i,j,ρ,σ] * Ro[ρ,k]
  @mtensor v_oovA[i,j,a,σ] := v_ooAA[i,j,ρ,σ] * Rv[ρ,a]
  @mtensor d_oooo[i,j,k,l] := v_oooA[i,j,k,σ] * Ro[σ,l]
  @mtensor d_oovo[i,j,a,k] := v_oovA[i,j,a,σ] * Ro[σ,k]
  @mtensor d_oovv[i,j,a,b] := v_oovA[i,j,a,σ] * Rv[σ,b]
  @mtensor v_vooA[a,i,j,σ] := v_AooA[μ,i,j,σ] * Lv[μ,a]              # shared by d_voov and d_vooo
  @mtensor d_voov[a,i,j,b] := v_vooA[a,i,j,σ] * Rv[σ,b]
  @mtensor d_vooo[a,i,j,k] := v_vooA[a,i,j,σ] * Ro[σ,k]
  @mtensor d_vovo[a,i,b,j] := (v_oAoA[i,ν,j,σ] * Lv[ν,a]) * Rv[σ,b]   # ⟨ai|bj⟩ = ⟨ia|jb⟩ (electron exchange)
  return (oooo=d_oooo, oovo=d_oovo, oovv=d_oovv, voov=d_voov, vovo=d_vovo, vooo=d_vooo)
end

"""
    ao_os_blocks(pm, La_o,La_v,Ra_o,Ra_v, Lb_o,Lb_v,Rb_o,Rb_v) -> NamedTuple

  Opposite-spin (αβ) counterpart of [`ao_ss_blocks`](@ref): one [`pm_os_sweep`](@ref) over the ±
  supermatrix store builds the occupied-contracted intermediates for both spins, then only the
  remaining AO indices are transformed into the needed spaces. No `nao⁴` tensor is formed.
"""
function ao_os_blocks(pm::PMSupermatrices, La_o, La_v, Ra_o, Ra_v,
                                           Lb_o, Lb_v, Rb_o, Rb_v; membytes::Int=typemax(Int))
  v_oOAA, v_AOoA, v_oAoA, v_oAAO, v_AOAO = pm_os_sweep(pm, La_o, Ra_o, Lb_o, Rb_o)
  return ao_os_finish(v_oOAA, v_AOoA, v_oAoA, v_oAAO, v_AOAO, La_v, Ra_o, Ra_v, Lb_v, Rb_o, Rb_v)
end

"Transform the remaining AO indices of the five opposite-spin intermediates into the dressed αβ blocks."
function ao_os_finish(v_oOAA, v_AOoA, v_oAoA, v_oAAO, v_AOAO, La_v, Ra_o, Ra_v, Lb_v, Rb_o, Rb_v)
  # shared half-transforms: each bra-virtual contraction feeds two ket blocks (β-occ O and β-virt V)
  @mtensor v_oOoA[i,J,k,σ] := v_oOAA[i,J,ρ,σ] * Ra_o[ρ,k]   # → d_oOoO, d_oOoV
  @mtensor v_oOvA[i,J,a,σ] := v_oOAA[i,J,ρ,σ] * Ra_v[ρ,a]   # → d_oOvO, d_oOvV
  @mtensor v_vOoA[a,I,k,σ] := v_AOoA[μ,I,k,σ] * La_v[μ,a]   # → d_vOoO, d_vOoV
  @mtensor v_oVoA[i,B,k,σ] := v_oAoA[i,ν,k,σ] * Lb_v[ν,B]   # → d_oVoO, d_oVoV
  @mtensor d_oOoO[i,J,k,L] := v_oOoA[i,J,k,σ] * Rb_o[σ,L]
  @mtensor d_oOoV[i,J,k,B] := v_oOoA[i,J,k,σ] * Rb_v[σ,B]
  @mtensor d_oOvO[i,J,a,L] := v_oOvA[i,J,a,σ] * Rb_o[σ,L]
  @mtensor d_oOvV[i,J,a,B] := v_oOvA[i,J,a,σ] * Rb_v[σ,B]
  @mtensor d_vOoO[a,I,k,L] := v_vOoA[a,I,k,σ] * Rb_o[σ,L]
  @mtensor d_vOoV[a,I,k,B] := v_vOoA[a,I,k,σ] * Rb_v[σ,B]
  @mtensor d_oVoO[i,B,k,L] := v_oVoA[i,B,k,σ] * Rb_o[σ,L]
  @mtensor d_oVoV[i,B,k,D] := v_oVoA[i,B,k,σ] * Rb_v[σ,D]
  @mtensor d_oVvO[i,B,a,J] := (v_oAAO[i,ν,ρ,J] * Lb_v[ν,B]) * Ra_v[ρ,a]
  @mtensor d_vOvO[a,I,b,J] := (v_AOAO[μ,I,ρ,J] * La_v[μ,a]) * Ra_v[ρ,b]
  return (oOoO=d_oOoO, oOoV=d_oOoV, oOvO=d_oOvO, oOvV=d_oOvV, vOoO=d_vOoO,
          vOoV=d_vOoV, oVoO=d_oVoO, oVoV=d_oVoV, oVvO=d_oVvO, vOvO=d_vOvO)
end

"""
    pm_K2!(pm, D2, tripp)

  kext K2 from the persisted ± supermatrix store — the amortized replacement for the
  per iteration. Reuses the same ij/rs ±-fold of the density and
  4-quadrant output scatter, but obtains the ± integral action as zero-copy panel GEMMs
  ``s\\!K2 = V_s·D_s`` / ``a\\!K2 = V_a·D_a`` ([`pm_matmul!`](@ref)) over the stored lower
  block-triangle — halved flops and streaming, no per-iteration ± build. `D2[tri(pq),i,j]`
  must already carry the ½ rs-diagonal (`calc_D2(...; scalepp=true)`, as `cc_kext!` passes).

  The ±-fold of the density ([`pm_fold_ij!`](@ref)) and the 4-quadrant unpacking of the products
  ([`pm_scatter_K2!`](@ref)) are pure memory traffic around the two GEMMs; both run as one fused
  multi-threaded pass over the `i ≤ j` pairs instead of the `CartesianIndex`-cut gather/scatter
  broadcasts.

  ``K^{ij}_{pq} = v_{pq}^{rs} D^{ij}_{rs}``

  Return K2pq::Array{4}.
"""
function pm_K2!(pm::PMSupermatrices{T}, D2::AbstractArray{T,3}, tripp) where {T<:Number}
  norb = pm.nao; nocc = size(D2, 2); npp = pm.npp
  @assert npp == length(tripp) "PM store nao ($(pm.nao)) inconsistent with kext norb"
  trioo = uppertriangular_cut(nocc)      # the (i ≤ j) list driving fold, GEMM and scatter
  ntri_oo = length(trioo)
  # ij-±-folded density, ket rs kept raw (½ rs-diagonal already in D2 via scalepp): [npp × ntri_oo]
  Ds = Matrix{T}(undef, npp, ntri_oo); Da = Matrix{T}(undef, npp, ntri_oo)
  pm_fold_ij!(Ds, Da, D2, trioo)
  sK2 = Matrix{T}(undef, npp, ntri_oo); aK2 = Matrix{T}(undef, npp, ntri_oo)
  pm_matmul!(sK2, pm, :s, Ds)     # sK2[pq,ij] = Σ_rs V_s[pq,rs] D_s[rs,ij]
  pm_matmul!(aK2, pm, :a, Da)     # aK2[pq,ij] = Σ_rs V_a[pq,rs] D_a[rs,ij]
  K2pq = Array{T,4}(undef, norb, norb, nocc, nocc)
  pm_scatter_K2!(K2pq, sK2, aK2, trioo)
  return K2pq
end

"""
    pm_fold_ij!(Ds, Da, D2, trioo)

  ±-fold of the kext density over the *occupied* pair: `Ds/Da[rs,ij] = ½(D2[rs,i,j] ± D2[rs,j,i])`
  for the `i ≤ j` list `trioo`, in a single fused pass (`Ds`, `Da` are `npp × length(trioo)`).

  Both source columns and both destination columns are contiguous in `rs`, so this is four
  streaming vectors; the previous `D2[:,trioo] ± D2[:,trioo_swap]` form instead paid two
  `CartesianIndex` gathers plus two broadcasts over a pair of `npp × ntri_oo` temporaries.

  Threaded over `ij`: iteration `ij` writes only column `ij` of `Ds`/`Da`, and each unordered
  pair `{i,j}` occurs once, so the writes are disjoint (`i == j` merely reads one column twice
  and yields `Da[:,ij] = 0`, as the ± fold requires).
"""
function pm_fold_ij!(Ds::AbstractMatrix{T}, Da::AbstractMatrix{T}, D2::AbstractArray{T,3},
                     trioo) where {T<:Number}
  npp = size(D2, 1); nocc = size(D2, 2)
  D2r = reshape(D2, npp, nocc*nocc)      # flat ij column, so the column offsets hoist
  Threads.@threads for ij in eachindex(trioo)
    @inbounds begin
      i = trioo[ij][1]; j = trioo[ij][2]
      cij = i + (j-1)*nocc; cji = j + (i-1)*nocc
      @simd ivdep for rs in 1:npp
        d1 = D2r[rs,cij]; d2 = D2r[rs,cji]
        Ds[rs,ij] = 0.5*(d1 + d2)
        Da[rs,ij] = 0.5*(d1 - d2)
      end
    end
  end
  return
end

"""
    pm_scatter_K2!(K2pq, sK2, aK2, trioo)

  Unpack the ± kext products into `K2pq[p,q,i,j]`, both `pq` orders and both `ij` orders:
  `K2pq[p,q,i,j] = K2pq[q,p,j,i] = sK2[pq,ij] + aK2[pq,ij]` and
  `K2pq[p,q,j,i] = K2pq[q,p,i,j] = sK2[pq,ij] - aK2[pq,ij]` (`p ≤ q`, `ij` running over `trioo`).

  One fused pass over the output replaces the four `K2pq[cut,cut] .= sK2 .± aK2` broadcasts,
  which re-read both products (and re-evaluated `sK2 .± aK2`) four times over.

  The `pq`- and `ij`-diagonals are where the four quadrants alias, and they are written *once*,
  explicitly: on `p == q` the two `pq` orders coincide, on `i == j` the two `ij` orders coincide.
  Both carry `aK2 = 0` exactly (`V_a` has zero `pp` rows; `D_a` has zero `ii` columns), so the
  single value `s - a` written there equals `s + a`, matching the previous last-write-wins.
  With no two writes of an iteration hitting one address, `@simd ivdep` is sound and the
  `Threads.@threads` over `ij` is race-free: iteration `ij` owns exactly the `(i,j)` and `(j,i)`
  planes of `K2pq`, and distinct `ij` are distinct unordered pairs, hence disjoint planes.
"""
function pm_scatter_K2!(K2pq::AbstractArray{T,4}, sK2::AbstractMatrix{T}, aK2::AbstractMatrix{T},
                        trioo) where {T<:Number}
  norb = size(K2pq, 1)
  Threads.@threads for ij in eachindex(trioo)
    @inbounds begin
      i = trioo[ij][1]; j = trioo[ij][2]
      Mij = @view K2pq[:,:,i,j]
      if i == j                                  # single plane, aK2[:,ij] = 0: symmetric fill
        for q in 1:norb
          pq0 = q*(q-1)÷2
          @simd ivdep for p in 1:q-1
            v = sK2[pq0+p,ij] - aK2[pq0+p,ij]
            Mij[p,q] = v; Mij[q,p] = v
          end
          Mij[q,q] = sK2[pq0+q,ij] - aK2[pq0+q,ij]
        end
      else
        Mji = @view K2pq[:,:,j,i]                # distinct plane: 4 distinct addresses per (p<q)
        for q in 1:norb
          pq0 = q*(q-1)÷2
          @simd ivdep for p in 1:q-1
            s = sK2[pq0+p,ij]; a = aK2[pq0+p,ij]
            Mij[p,q] = s + a; Mji[q,p] = s + a
            Mij[q,p] = s - a; Mji[p,q] = s - a
          end
          v = sK2[pq0+q,ij] - aK2[pq0+q,ij]      # p == q: the two pq orders coincide
          Mij[q,q] = v; Mji[q,q] = v
        end
      end
    end
  end
  return
end

"""
    pm_K2ab!(pm, D2ab_full, tripp)

  αβ kext from the persisted ± store: the full contraction
  ``K2ab_{pq}^{iJ} = Σ_{rs} ⟨pq|rs⟩ D^{iJ}_{rs}`` for the (non-`rs`-symmetric) αβ density.
  The `rs`-± fold is explicit — `Ds/Da = ½(D[pq] ± D[qp])`, with the ½ `rs`-diagonal already
  carried by `calc_D2ab(...; scalepp=true)` — then two [`pm_matmul!`](@ref PMStore.pm_matmul!)
  panel GEMMs and the ± unscatter to both `pq` orders ([`pm_scatter_K2ab!`](@ref)). Halved flops
  and streaming vs the two joint-store `calc_K2` passes.

  Because this fold transposes the *AO* pair (unlike the occupied-pair fold of [`pm_K2!`](@ref)),
  it is exactly [`calc_tri_sym_antisym!`](@ref QMTensors.calc_tri_sym_antisym!) with `fac = ½` —
  one fused, threaded, row-buffered pass instead of two `CartesianIndex` gathers plus two
  broadcasts over a pair of `npp × nanb` temporaries.
"""
function pm_K2ab!(pm::PMSupermatrices{T}, D2ab_full::AbstractArray{T,4}, tripp) where {T<:Number}
  norb = pm.nao
  na = size(D2ab_full, 3); nb = size(D2ab_full, 4); nij = na*nb
  ntri = length(tripp)
  @assert pm.npp == ntri "PM store nao ($(pm.nao)) inconsistent with kext norb"
  Ds = Matrix{T}(undef, ntri, nij); Da = Matrix{T}(undef, ntri, nij)
  calc_tri_sym_antisym!(Ds, Da, reshape(D2ab_full, norb, norb, nij), T(0.5))   # fac = ½: ± average
  sK = Matrix{T}(undef, ntri, nij); aK = Matrix{T}(undef, ntri, nij)
  pm_matmul!(sK, pm, :s, Ds)
  pm_matmul!(aK, pm, :a, Da)
  K2 = Array{T,4}(undef, norb, norb, na, nb)
  pm_scatter_K2ab!(reshape(K2, norb, norb, nij), sK, aK)
  return K2
end

"""
    pm_scatter_K2ab!(K2, sK, aK)

  Unpack the αβ ± kext products into both `pq` orders of the (pair-nonsymmetric) αβ `K2`:
  `K2[p,q,iJ] = sK[pq,iJ] + aK[pq,iJ]`, `K2[q,p,iJ] = sK[pq,iJ] - aK[pq,iJ]` for `p ≤ q`.
  `K2` is the output reshaped to `[p, q, iJ]` with the αβ pair flattened.

  One fused pass over the output replaces the two `K2[cut,:,:] .= sK .± aK` broadcasts. The
  `pq`-diagonal, where the two orders alias, is written once (there `aK = 0` exactly — `V_a` has
  zero `pp` rows — so `s - a` equals `s + a`, matching the previous last-write-wins). Threaded
  over `iJ`, which owns one full `norb × norb` plane, hence disjoint writes.
"""
function pm_scatter_K2ab!(K2::AbstractArray{T,3}, sK::AbstractMatrix{T},
                          aK::AbstractMatrix{T}) where {T<:Number}
  norb = size(K2, 1)
  Threads.@threads for u in 1:size(K2, 3)
    @inbounds begin
      M = @view K2[:,:,u]
      for q in 1:norb
        pq0 = q*(q-1)÷2
        @simd ivdep for p in 1:q-1
          s = sK[pq0+p,u]; a = aK[pq0+p,u]
          M[p,q] = s + a; M[q,p] = s - a
        end
        M[q,q] = sK[pq0+q,u] - aK[pq0+q,u]
      end
    end
  end
  return
end
