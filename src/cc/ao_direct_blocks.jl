"""
AO-direct MO integral blocks.

Everything that turns the persisted stores into the MO blocks the coupled-cluster equations read:
the half-transformed → MO block engine (`ht_mo_block`, with `ht_mo_block_spec` resolving a block name
into the store, coefficients and permutation that produce it), the dressed same-/opposite-spin block
assembly, and the ± `kext` 4-external kernels.

Unlike the sweeps in `PMStore` (`ht_occ_early`, `pm_occ_early`, …), these know about the MO spaces
(`EC.space`, the active orbitals) and the method's index conventions, so they belong on the
coupled-cluster side. Included into `CoupledCluster`; not a module of its own.
"""

"""
    ht_mo_block(EC, htkey, CX, CY, CZ) -> W

  General half-transformed-store → MO-block kernel: build a 4-index MO integral block that carries the
  store's (occupied) bra index `i` plus three MO indices transformed by the coefficient matrices
  `CX`/`CY`/`CZ`, in ONE sweep over the store `htkey` (`Σ_μ⟨μν|ρσ⟩C[μ,i]`, read column-by-column with
  [`ht_column_A!`](@ref PMStore.ht_column_A!)):

      W[i,X,Y,Z] = Σ_μνρσ ⟨μν|ρσ⟩ C[μ,i] CX[ν,X] CY[ρ,Y] CZ[σ,Z]   ( = ⟨iX|YZ⟩ )

  Every block a CC method needs (they all have ≥1 occupied index) is one call plus an output
  permutation — see [`ht_mo_block_spec`](@ref). Per σ-column two GEMMs transform the free bra `ν` and
  ket-1 `ρ` into the `[(i,X),Y]` intermediate; stacking those over σ and one final GEMM contracts ket-2
  `σ` into `Z`. Generic over the element type (the store applies plain `C`, matching the
  AO-direct/FCIDUMP `detri` convention). Cost: one pass over the store + GEMMs; peak RAM ≈ the output.

  There is no second "occupied on bra-2" kernel: by the particle-exchange symmetry `⟨μν|ρσ⟩=⟨νμ|σρ⟩`,

      ⟨Xi|YZ⟩ = ⟨iX|ZY⟩   ⟹   B(CX,CY,CZ) = permutedims(ht_mo_block(CX,CZ,CY), (2,1,4,3))

  i.e. such a block is THIS kernel with the two ket coefficients exchanged. That identity is exact for
  complex integrals and for a non-Hermitian (similarity-transformed, `CL≠CR`) transformation, because
  it never exchanges bra with ket — unlike the `swap` relation in [`ht_mo_block_spec`](@ref).
"""
function ht_mo_block(EC::ECInfo, htkey::AbstractString,
                     CX::AbstractMatrix, CY::AbstractMatrix, CZ::AbstractMatrix)
  ht = open_ht_store(EC, htkey)
  T = eltype(ht.map); n = ht.nao; m = ht.m
  nX = size(CX, 2); nY = size(CY, 2); nZ = size(CZ, 2)
  size(CX,1) == n && size(CY,1) == n && size(CZ,1) == n ||
    error("ht_mo_block: coefficient rows must equal nao=$n")
  Aσ = zeros(T, m*n, n)
  IXY = zeros(T, m*nX*nY, n)                        # [(i,X,Y), σ]  stacked ket-2 columns
  iXρ = zeros(T, m, nX, n); iXY = zeros(T, m, nX, nY)
  @inbounds for σ in 1:n
    ht_column_A!(Aσ, ht, σ)
    S = reshape(Aσ, m, n, n)                        # [i, ν, ρ]
    @mtensor iXρ[i,X,ρ] = S[i,ν,ρ] * CX[ν,X]        # free bra → X
    @mtensor iXY[i,X,Y] = iXρ[i,X,ρ] * CY[ρ,Y]      # ket-1 → Y
    @views IXY[:, σ] .= vec(iXY)
  end
  close_ht_store!(EC, ht)
  return reshape(IXY * CZ, m, nX, nY, nZ)           # ket-2 σ → Z (one BLAS-3 GEMM)
end

"""
    ht_mo_block_spec(name) -> (store, X, Y, Z, perm, swap)

  Resolve a block name (a 4-character space string `⟨s₁s₂|s₃s₄⟩`, lowercase = α, uppercase = β) into
  the [`ht_mo_block`](@ref) call that produces it: which per-spin store supplies the block's occupied
  index, which space goes on each free slot, and how to permute the result.

  The store holds its occupied index on a BRA slot, so where the block's occupied index sits decides
  everything. With `A ≡ ⟨iX|YZ⟩` and using `⟨pq|rs⟩ = ⟨qp|sr⟩` (particle exchange — always valid) and
  `⟨pq|rs⟩ = ⟨rs|pq⟩` (bra↔ket — `swap`, see below):

  | occ at | rewrite                    | A-form      | perm        | swap |
  |:------:|:---------------------------|:------------|:------------|:----:|
  | 1      | `⟨is₂\\|s₃s₄⟩`               | `(s₂,s₃,s₄)`| `(1,2,3,4)` | no   |
  | 2      | `⟨s₁i\\|s₃s₄⟩ = ⟨is₁\\|s₄s₃⟩` | `(s₁,s₄,s₃)`| `(2,1,4,3)` | no   |
  | 3      | `⟨s₁s₂\\|is₄⟩ = ⟨is₄\\|s₁s₂⟩` | `(s₄,s₁,s₂)`| `(3,4,1,2)` | YES  |
  | 4      | `⟨s₁s₂\\|s₃i⟩ = ⟨is₃\\|s₂s₁⟩` | `(s₃,s₂,s₁)`| `(4,3,2,1)` | YES  |

  The first occupied slot is taken, so a bra one wins whenever the block has one — the point being
  that rows 1–2 need only particle exchange, which holds for complex and for non-Hermitian
  (`CL≠CR`) transformations. Rows 3–4 have the occupied index on a KET slot, which no particle
  exchange can move to the bra, so they need the bra↔ket relation: valid only for Hermitian integrals
  with `CL=CR` and real coefficients (guarded in [`save_mo_blocks!`](@ref); complex would need a
  ket-transformed store).

  Blocks sharing an A-form differ only in `perm`/`swap` and so share one sweep — `vovv`/`vvvo`,
  `ovoo`/`vooo`, `VOVV`/`VVVO`, `vOvV`/`vVvO`, `oVvV`/`vVoV`.
"""
function ht_mo_block_spec(name::AbstractString)
  s = collect(name)
  length(s) == 4 && all(c -> c in ('o','O','v','V'), s) ||
    error("ht_mo_block_spec: \"$name\" is not a 4-character o/O/v/V space string")
  k = findfirst(c -> c == 'o' || c == 'O', s)
  isnothing(k) && error("ht_mo_block_spec: block \"$name\" has no occupied index — the " *
                        "half-transformed store cannot supply it")
  store = islowercase(s[k]) ? :a : :b              # spin of the store-supplied occupied index
  X, Y, Z, perm, swap =
    k == 1 ? (s[2], s[3], s[4], (1,2,3,4), false) :
    k == 2 ? (s[1], s[4], s[3], (2,1,4,3), false) :
    k == 3 ? (s[4], s[1], s[2], (3,4,1,2), true)  :
             (s[3], s[2], s[1], (4,3,2,1), true)
  return (store=store, X=X, Y=Y, Z=Z, perm=perm, swap=swap)
end

"""
    save_mo_blocks!(EC, names, htkeys, coefs)

  Build the bare MO blocks `names` and save them (mmapped, `"tmp"` so they are reclaimed at end-of-run)
  under their plain space names, in the index order the consumers read.

  Blocks that resolve to the same [`ht_mo_block_spec`](@ref) A-form share ONE store sweep and differ
  only by the output permutation — e.g. Λ(T)'s 16 unrestricted blocks come from 12 sweeps, including
  a single sweep for each of the four expensive 3-external pairs.
"""
function save_mo_blocks!(EC::ECInfo, names, htkeys, coefs)
  specs = Dict(String(n) => ht_mo_block_spec(n) for n in names)
  for (n, spec) in specs
    spec.swap && eltype(coefs[spec.X]) <: Complex &&
      error("save_mo_blocks!: block \"$n\" has its occupied index on a ket slot, so it needs the " *
            "bra↔ket Hermiticity relation, which is real-only from a bra-transformed store; " *
            "complex needs a ket-transformed store (pending complex-AO integrals)")
  end
  groups = Dict{NTuple{4,Any},Vector{String}}()
  for n in names
    spec = specs[String(n)]
    push!(get!(groups, (spec.store, spec.X, spec.Y, spec.Z), String[]), String(n))
  end
  for ((store, X, Y, Z), members) in groups
    W = ht_mo_block(EC, htkeys[store], coefs[X], coefs[Y], coefs[Z])
    for n in members
      spec = specs[n]
      B = spec.perm == (1,2,3,4) ? W : permutedims(W, spec.perm)
      spec.swap && (B = conj(B))                   # Hermiticity conj (a no-op for real)
      save!(EC, n, B; description="tmp mo block ($n)")
    end
  end
  return
end

"Closed-shell [`save_mo_blocks!`](@ref): the single store `htkey` with occupied/virtual coefficients."
save_mo_blocks!(EC::ECInfo, names, htkey::AbstractString, Co::AbstractMatrix, Cv::AbstractMatrix) =
  save_mo_blocks!(EC, names, Dict(:a => htkey), Dict('o' => Co, 'v' => Cv))

"""
    build_ht_mo_blocks!(EC, names; Rv=nothing)

  Build the closed-shell bare MO blocks `names` (e.g. `("vvvo","ovoo")` for (T), plus `"vovv"`/`"ooov"`
  for Λ(T)) that the AO-direct triples / λ-triples read, from the occupied-bra half-transformed store
  `"ht_oAAA"` (still on file from [`ao_cc_setup!`](@ref)) and the active-space MO coefficients. The
  consumers pick the blocks up with `load4idx`.

  `Rv` is an optional virtual-space rotation (e.g., for the pseudo-canonicalization)
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
  save_mo_blocks!(EC, names, "ht_oAAA", Co, Cv)
  return
end

"""
    build_ht_mo_blocks_unrestricted!(EC, names; Rv=nothing)

  Unrestricted analogue of [`build_ht_mo_blocks!`](@ref): build the bare MO blocks `names` (same-spin
  `vvvo`/`vooo`/`VVVO`/`VOOO` and the opposite-spin `vVvO`/`vVoV`/`vOoO`/`oVoO` the unrestricted (T)
  reads) from the per-spin half-transformed stores `"ht_oAAA_a"`/`"ht_oAAA_b"` (built in
  [`ao_cc_setup!`](@ref)) and the per-spin MO coefficients. Each block's [`ht_mo_block_spec`](@ref)
  says which store supplies its occupied index and which spin's coefficients go on the free slots.

  `Rv` is an optional per-spin virtual-space rotation (e.g., for the pseudo-canonicalization)
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
  save_mo_blocks!(EC, names, Dict(:a => "ht_oAAA_a", :b => "ht_oAAA_b"), coefs)
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
