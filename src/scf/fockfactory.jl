""" Fock builders (using FciDump or DF integrals) """
module FockFactory
try
  using MKL
catch
  #println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
#BLAS.set_num_threads(1)
using Buffers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions
using ..ElemCo.FciDumps
using ..ElemCo.Integrals
using ..ElemCo.OrbTools
using ..ElemCo.PMStore

export gen_fock, gen_ufock, gen_dffock, gen_pfock, gen_df3idx_fock
export gen_density_matrix, gen_frac_density_matrix

""" 
    gen_fock(EC::ECInfo)

  Calculate closed-shell fock matrix from FCIDump integrals. 
"""
function gen_fock(EC::ECInfo)
  @mtensor fock[p,q] := integ1(EC.fd,:α)[p,q] + 2.0*ints2(EC,":o:o",:α)[p,i,q,i] - ints2(EC,":oo:",:α)[p,i,i,q]
  if EC.options.wf.npositron > 0
    @mtensor fock[p,q] -= ints2(EC,":p:p",:p)[p,i,q,i] 
  end
  return fock
end

"""
    gen_pfock(EC::ECInfo)

  Calculate positron fock matrix from FCIDump integrals.
"""
function gen_pfock(EC::ECInfo)
  @mtensor pfock[p,q] := integ1(EC.fd,:p)[p,q] - 2.0*ints2(EC,"o:o:",:p)[i,p,i,q]
  return pfock
end

""" 
    gen_fock(EC::ECInfo, spincase::Symbol)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`}. 
"""
function gen_fock(EC::ECInfo, spincase::Symbol)
  @mtensor fock[p,q] := integ1(EC.fd,spincase)[p,q] 
  if spincase == :α
    if n_occb_orbs(EC) > 0 
      @mtensor fock[p,q] += ints2(EC,":O:O",:αβ)[p,i,q,i]
    end
    spo='o'
    spv='v'
    nocc = n_occ_orbs(EC)
  else
    if n_occ_orbs(EC) > 0 
      @mtensor fock[p,q] += ints2(EC,"o:o:",:αβ)[i,p,i,q]
    end
    spo='O'
    spv='V'
    nocc = n_occb_orbs(EC)
  end
  if nocc > 0
    @mtensor begin
      fock[p,q] += ints2(EC,":"*spo*":"*spo,spincase)[p,i,q,i]
      fock[p,q] -= ints2(EC,":"*spo*spo*":",spincase)[p,i,i,q]
    end
  end
  return fock
end

""" 
    gen_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occvec)

  Generate ``D_{\\mu\\nu}=C^l_{\\mu i} C^r_{\\nu i}`` with ``i`` defined by `occvec`.
  Only real part of ``D_{\\mu\\nu}`` is kept unless T is Complex.
""" 
function gen_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occvec) where T
  CMOlo = CMOl[:,occvec]
  CMOro = CMOr[:,occvec]
  @mtensor den[r,s] := CMOlo[r,i]*CMOro[s,i]
  T <: Complex && return den
  denr = real.(den)
  if sum(abs2,den) - sum(abs2,denr) > EC.options.scf.imagtol
    println("Large imaginary part in density matrix neglected!")
    println("Difference between squared norms:",sum(abs2,den)-sum(abs2,denr))
  end
  return denr
end

""" 
    gen_frac_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occupation)

  Generate ``D_{\\mu\\nu}=C^l_{\\mu i} C^r_{\\nu i} n_i`` with ``n_i`` provided in `occupation`.
  Only real part of ``D_{\\mu\\nu}`` is kept unless T is Complex.
""" 
function gen_frac_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occupation) where T  
  @assert length(occupation) == size(CMOr,2) "Wrong occupation vector length!"
  CMOrn = CMOr .* occupation'
  @mtensor den[r,s] := CMOl[r,i]*CMOrn[s,i]
  T <: Complex && return den
  denr = real.(den)
  if sum(abs2,den) - sum(abs2,denr) > EC.options.scf.imagtol
    println("Large imaginary part in density matrix neglected!")
    println("Difference between squared norms:",sum(abs2,den)-sum(abs2,denr))
  end
  return denr
end

""" 
    gen_fock(EC::ECInfo, den::AbstractMatrix)

  Calculate closed-shell fock matrix from FCIDump integrals and density matrix `den`. 
"""
function gen_fock(EC::ECInfo, den::AbstractMatrix)
  @mtensor begin 
    fock[p,q] := integ1(EC.fd,:α)[p,q] 
    fock[p,q] += ints2(EC,"::::",:α)[p,r,q,s] * den[r,s]
    fock[p,q] -= 0.5*(ints2(EC,"::::",:α)[p,r,s,q] * den[r,s])
  end
  return fock
end

""" 
    gen_fock(EC::ECInfo, CMOl::AbstractMatrix, CMOr::AbstractMatrix)

  Calculate closed-shell fock matrix from FCIDump integrals and orbitals `CMOl`, `CMOr`. 
"""
function gen_fock(EC::ECInfo, CMOl::AbstractMatrix, CMOr::AbstractMatrix)
  @assert EC.space['o'] == EC.space['O'] # closed-shell
  occ2 = EC.space['o']
  den = gen_density_matrix(EC, CMOl, CMOr, occ2)
  @mtensor begin 
    fock[p,q] := integ1(EC.fd,:α)[p,q] 
    fock[p,q] += 2.0*ints2(EC,"::::",:α)[p,r,q,s] * den[r,s]
    fock[p,q] -= ints2(EC,"::::",:α)[p,r,s,q] * den[r,s]
  end
  return fock
end

# ---- ao_JK!/ao_J2K! on the persisted ± supermatrix store ------------------------------
#
# Per ket pair (ρσ) the AO Fock build needs the slab identities  `J[:,ρ] += G·Dj[:,σ]`,
# `K[:,σ] += G·Dk[:,ρ]`  (plus the ket-swap `Gᵀ` for `ρ<σ`) with the full slab
# `G[μ,ν] = ⟨μν|ρσ⟩`. The ± store keeps only the lower block-triangle; `PMStore.pm_slab_sweep!`
# reconstructs each column's slab and lets us apply those identities in two roles — NATIVE (this
# column as the ket ⟨··|ρσ⟩, L-band `(native_lo, nao]`) and MIRROR (by hermiticity the bra
# ⟨ρσ|··⟩, sub-panel band `(mirror_lo, nao]`) — reading n⁴/4, half a jointly packed store. See the
# PMStore "per-column slab reconstruction sweep" section for the machinery.
#
# `add_coulomb!` / `add_exchange!` each perform both roles for one density. A closed-shell Fock
# is `add_coulomb! + add_exchange!`; the UHF `ao_J2K!` is `add_coulomb!` once (shared Coulomb from
# the total density) + `add_exchange!` twice (the two same-spin exchanges) in the same sweep.
# Requires the physical hermiticity `⟨μν|ρσ⟩ = conj(⟨ρσ|μν⟩)` the store presumes; `Dj`,`Dk` need
# not be symmetric. `O(nao²)` working memory.

"""
    add_coulomb!(J, s::PMSlab, D)

  Add the Coulomb contribution of the slab `s` (`⟨μν|ρσ⟩`, ket pair `(s.ρ,s.σ)`): the native role
  `J[:,ρ] += ⟨··|ρσ⟩·D[:,σ]` (+ the ket-swap `⟨··|σρ⟩ = Gᵀ` for `ρ<σ`) and the Hermitian mirror role
  into the rows `J[ρ,:]`/`J[σ,:]`. The simple per-slab routine [`ao_JK!`](@ref) calls in its `eachslab` loop.
"""
function add_coulomb!(J, s::PMSlab, D)
  # `view`, NOT `@mview`: these columns are BLAS `mul!` operands (see `band_mul!`), and the
  # `StridedView` that `@mview` returns is not `<:StridedVecOrMat`, so `mul!` would silently fall
  # back to the generic (non-BLAS) matvec — measured ~6× slower for a whole Fock build.
  slab_bandmul!(view(J,:,s.ρ), s, view(D,:,s.σ))               # J[:,ρ] += ⟨··|ρσ⟩ · D[:,σ]
  s.ρ < s.σ && slab_bandtmul!(view(J,:,s.σ), s, view(D,:,s.ρ)) # ket-swapped slab ⟨··|σρ⟩ = Gᵀ
  slab_mirror!(J, s.ρ, s, D, s.σ)                                    # mirror: J[ρ,:] += conj⟨ρσ|··⟩·conj(D[σ,:])
  s.ρ < s.σ && slab_mirrort!(J, s.σ, s, D, s.ρ)
  return
end

"""
    add_exchange!(K, s::PMSlab, D)

  Add the exchange contribution of the slab `s`. Same two roles as [`add_coulomb!`](@ref) with the
  `(ρ,σ)` output/density roles swapped (and the mirror transpose flag flipped): native
  `K[:,σ] += ⟨··|ρσ⟩·D[:,ρ]` (+ `Gᵀ` for `ρ<σ`), plus the mirror into rows `K[ρ,:]`/`K[σ,:]`.
"""
function add_exchange!(K, s::PMSlab, D)
  slab_bandmul!(view(K,:,s.σ), s, view(D,:,s.ρ))              # K[:,σ] += ⟨··|ρσ⟩ · D[:,ρ]  (`view`: see add_coulomb!)
  s.ρ < s.σ && slab_bandtmul!(view(K,:,s.ρ), s, view(D,:,s.σ))
  slab_mirrort!(K, s.ρ, s, D, s.σ)
  s.ρ < s.σ && slab_mirror!(K, s.σ, s, D, s.ρ)
  return
end

# ---- symmetric-density (HF) fast path ------------------------------------------------
#
# When the density is symmetric (real HF: `D = C·Cᵀ`), the Fock matrices are symmetric, so the
# mirror role is redundant: for real symmetric `D` the mirror contribution to row `J[ρ,·]` is
# exactly the transpose of the native contribution to column `J[·,ρ]`. We can therefore SKIP the
# mirror entirely and symmetrize at the end — ≈2× fewer GEMVs. The one subtlety is the diagonal
# tile (stored full, hence self-mirroring): a plain `J + Jᵀ` would double-count it. So we split
# the native contribution into the **sub-panel** band `(mirror_lo, nao]` (accumulated into `J`,
# then symmetrized) and the **diagonal-tile** band `(native_lo, mirror_lo]` (into `Jd`, symmetric
# already, added once): `J_final = J + Jᵀ + Jd`. Needs only the store's built-in Hermiticity +
# real symmetric `D` — no 8-fold assumption. Real only (`T<:Real`).

"[`add_coulomb!`](@ref) for symmetric real `D`, mirror-free: sub-panel band → `J`, diagonal-tile band → `Jd`."
function add_coulomb_sym!(J, Jd, s::PMSlab, D)
  G = s.w.G; ρ = s.ρ; σ = s.σ
  band_mul!(view(J,:,ρ),  G, s.mirror_lo, s.nao, view(D,:,σ))       # sub-panel (symmetrized later)
  ρ < σ && band_tmul!(view(J,:,σ), G, s.mirror_lo, s.nao, view(D,:,ρ))
  band_mul!(view(Jd,:,ρ), G, s.native_lo, s.mirror_lo, view(D,:,σ)) # diagonal tile (added once)
  ρ < σ && band_tmul!(view(Jd,:,σ), G, s.native_lo, s.mirror_lo, view(D,:,ρ))
  return
end

"[`add_exchange!`](@ref) for symmetric real `D`, mirror-free (the `(ρ,σ)` roles swapped vs Coulomb)."
function add_exchange_sym!(K, Kd, s::PMSlab, D)
  G = s.w.G; ρ = s.ρ; σ = s.σ
  band_mul!(view(K,:,σ),  G, s.mirror_lo, s.nao, view(D,:,ρ))
  ρ < σ && band_tmul!(view(K,:,ρ), G, s.mirror_lo, s.nao, view(D,:,σ))
  band_mul!(view(Kd,:,σ), G, s.native_lo, s.mirror_lo, view(D,:,ρ))
  ρ < σ && band_tmul!(view(Kd,:,ρ), G, s.native_lo, s.mirror_lo, view(D,:,σ))
  return
end

"In place `A .= A + Aᵀ + Ad` for the symmetric-D finalize (result is symmetric; `Ad` symmetric)."
function finalize_hermitian!(A::AbstractMatrix, Ad::AbstractMatrix)
  n = size(A, 1)
  @inbounds for j in 1:n, i in 1:j
    v = A[i,j] + A[j,i] + Ad[i,j]
    A[i,j] = v; A[j,i] = v
  end
  return A
end

"Symmetric real-`D` fast path for [`ao_JK!`](@ref) on the ± store: mirror-free sweep + symmetrize."
function ao_JK_sym!(J, K, pm::PMSupermatrices{T}, D) where {T<:Real}
  Jd = zeros(T, pm.nao, pm.nao); Kd = zeros(T, pm.nao, pm.nao)
  for s in eachslab(pm; TF=T)
    add_coulomb_sym!(J,  Jd, s, D)
    add_exchange_sym!(K, Kd, s, D)
  end
  finalize_hermitian!(J, Jd); finalize_hermitian!(K, Kd)
  return J, K
end

"Symmetric real-`D` fast path for [`ao_J2K!`](@ref): shared Coulomb + two exchanges, mirror-free."
function ao_J2K_sym!(J, Ka, Kb, pm::PMSupermatrices{T}, Dt, Da, Db) where {T<:Real}
  n = pm.nao
  Jd = zeros(T, n, n); Kad = zeros(T, n, n); Kbd = zeros(T, n, n)
  for s in eachslab(pm; TF=T)
    add_coulomb_sym!(J,  Jd,  s, Dt)
    add_exchange_sym!(Ka, Kad, s, Da)
    add_exchange_sym!(Kb, Kbd, s, Db)
  end
  finalize_hermitian!(J, Jd); finalize_hermitian!(Ka, Kad); finalize_hermitian!(Kb, Kbd)
  return J, Ka, Kb
end

"""
    ao_JK!(J, K, pm::PMSupermatrices, Dj, Dk)

  The [`ao_JK!`](@ref) Coulomb/exchange contraction from the persisted ± supermatrix store at
  **half the streaming I/O** (each stored element read once, ≈ n⁴/4). One [`PMStore.pm_slab_sweep!`](@ref)
  of the store; per ket column, add its Coulomb and exchange contributions. `Dj`, `Dk` need not
  be symmetric. Pass `hermitian=true` when `Dj === Dk` is a real symmetric density (HF) to take
  the mirror-free fast path ([`ao_JK_sym!`](@ref), ≈2× fewer GEMVs).
"""
function ao_JK!(J::AbstractMatrix, K::AbstractMatrix, pm::PMSupermatrices{T},
                Dj::AbstractMatrix, Dk::AbstractMatrix; hermitian::Bool=false) where T
  if hermitian && T <: Real && eltype(Dj) <: Real
    return ao_JK_sym!(J, K, pm, Dj)
  end
  TF = promote_type(T, eltype(Dj), eltype(Dk))
  for s in eachslab(pm; TF=TF)                             # one reconstruction per slab, J + K fused
    add_coulomb!(J, s, Dj)
    add_exchange!(K, s, Dk)
  end
  return J, K
end

"""
    ao_J2K!(J, Ka, Kb, pm::PMSupermatrices, Dt, Da, Db)

  The UHF [`ao_J2K!`](@ref) from the ± store: the shared Coulomb `J` (from the total density
  `Dt`) and both same-spin exchanges `Ka`,`Kb` (from `Da`,`Db`) in a single sweep — [`ao_JK!`](@ref)'s
  twin, one `add_coulomb!` + two `add_exchange!` per ket column. `hermitian=true` (real, all three
  densities symmetric) takes the mirror-free fast path ([`ao_J2K_sym!`](@ref)).
"""
function ao_J2K!(J::AbstractMatrix, Ka::AbstractMatrix, Kb::AbstractMatrix,
                 pm::PMSupermatrices{T}, Dt::AbstractMatrix,
                 Da::AbstractMatrix, Db::AbstractMatrix; hermitian::Bool=false) where T
  if hermitian && T <: Real && eltype(Dt) <: Real
    return ao_J2K_sym!(J, Ka, Kb, pm, Dt, Da, Db)
  end
  TF = promote_type(T, eltype(Dt), eltype(Da), eltype(Db))
  for s in eachslab(pm; TF=TF)                             # shared Coulomb + two same-spin K, one sweep
    add_coulomb!(J, s, Dt)                                 # shared Coulomb from the total density
    add_exchange!(Ka, s, Da)                               # same-spin exchange α
    add_exchange!(Kb, s, Db)                               # same-spin exchange β
  end
  return J, Ka, Kb
end

# Integral handle accepted by the AO Fock builders: the persisted ± supermatrix store.
const AOIntegrals = PMSupermatrices

"""
    gen_fock(EC::ECInfo, ints, h1::AbstractMatrix, CMOl::AbstractMatrix, CMOr::AbstractMatrix)

  Closed-shell AO Fock matrix `h1 + 2J − K` from explicitly given spin-free 2-e integrals
  `ints` and orbitals `CMOl`, `CMOr`. `ints` is the persisted ± supermatrix store
  ([`PMSupermatrices`](@ref PMStore.PMSupermatrices)), the only exact-AO integral
  representation — [`ao_JK!`](@ref) contracts it directly (the ± store halves the integral
  I/O). The contraction is basis-agnostic (physicists' notation), so feeding AO integrals + AO
  density yields the AO Fock; no dense `nao⁴` tensor is formed. `nao` is taken from the orbitals.
"""
function gen_fock(EC::ECInfo, ints::AOIntegrals, h1::AbstractMatrix,
                  CMOl::AbstractMatrix, CMOr::AbstractMatrix)
  @assert EC.space['o'] == EC.space['O'] # closed-shell
  den = gen_density_matrix(EC, CMOl, CMOr, EC.space['o'])
  nao = size(CMOl, 1)
  TF = promote_type(eltype(ints), eltype(den))
  J = zeros(TF, nao, nao); K = zeros(TF, nao, nao)
  ao_JK!(J, K, ints, den, den; hermitian = (CMOl === CMOr))   # den = C·Cᵀ symmetric ⇒ fast path
  return h1 .+ 2 .* J .- K
end

"""
    gen_ufock(EC::ECInfo, ints, h1::AbstractMatrix, cMOl::SpinMatrix, cMOr::SpinMatrix)

  UHF AO Fock matrix from explicitly given spin-free 2-e integrals `ints` (the ±
  store; [`ao_J2K!`](@ref) dispatches) and 1-e integrals `h1` (same for both spins). The shared
  Coulomb term (total density) is built once and both same-spin exchange terms in a single
  streaming pass; no dense `nao⁴` tensor is formed. `nao` is taken from the orbitals.
"""
function gen_ufock(EC::ECInfo, ints::AOIntegrals, h1::AbstractMatrix,
                   cMOl::SpinMatrix, cMOr::SpinMatrix)
  Da = gen_density_matrix(EC, cMOl.α, cMOr.α, EC.space['o'])
  Db = gen_density_matrix(EC, cMOl.β, cMOr.β, EC.space['O'])
  Dt = Da .+ Db
  nao = size(cMOl.α, 1)
  TF = promote_type(eltype(ints), eltype(Da))
  J = zeros(TF, nao, nao); Ka = zeros(TF, nao, nao); Kb = zeros(TF, nao, nao)
  # Da, Db (hence Dt) symmetric when left/right orbitals coincide ⇒ fast path
  ao_J2K!(J, Ka, Kb, ints, Dt, Da, Db; hermitian = (cMOl.α === cMOr.α && cMOl.β === cMOr.β))
  return SpinMatrix(h1 .+ J .- Ka, h1 .+ J .- Kb)
end

"""
    gen_fock(EC::ECInfo, spincase::Symbol, CMOl::AbstractMatrix, CMOr::AbstractMatrix)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`} and orbitals `CMOl`, `CMOr` and
  orbitals for the opposite-spin `CMOlOS` and `CMOrOS`.
"""
function gen_fock(EC::ECInfo, spincase::Symbol, CMOl::AbstractMatrix, CMOr::AbstractMatrix,
                  CMOlOS::AbstractMatrix, CMOrOS::AbstractMatrix)
  if spincase == :α
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['O'])
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[p,r,q,s]*denOS[r,s]
    spo = 'o'
  else
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['o'])
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[r,p,s,q]*denOS[r,s]
    spo = 'O'
  end
  den =  gen_density_matrix(EC, CMOl, CMOr, EC.space[spo])
  ints = ints2(EC,"::::",spincase)
  @mtensor fock[p,q] += ints[p,r,q,s] * den[r,s] 
  @mtensor fock[p,q] -= ints[p,r,s,q] * den[r,s]
  @mtensor fock[p,q] += integ1(EC.fd,spincase)[p,q] 
  return fock
end

""" 
    gen_fock(EC::ECInfo, spincase::Symbol, den::AbstractMatrix, denOS::AbstractMatrix)

  Calculate UHF fock matrix from FCIDump integrals and density matrices `den` (for `spincase`) 
  and `denOS` (opposite spin to `spincase`). 
"""
function gen_fock(EC::ECInfo, spincase::Symbol, 
                  den::AbstractMatrix, denOS::AbstractMatrix)
  if spincase == :α
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[p,r,q,s]*denOS[r,s]
  else
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[r,p,s,q]*denOS[r,s]
  end
  ints = ints2(EC,"::::",spincase)
  @mtensor fock[p,q] += ints[p,r,q,s] * den[r,s] 
  @mtensor fock[p,q] -= ints[p,r,s,q] * den[r,s]
  @mtensor fock[p,q] += integ1(EC.fd,spincase)[p,q] 
  return fock
end

""" 
    gen_ufock(EC::ECInfo, CMOl::SpinMatrix, CMOr::SpinMatrix)

  Calculate UHF fock matrix from FCIDump integrals and orbitals `cMOl`, `cMOr`
  with `cMOl[1]` and `cMOr[1]` - α-MO transformation coefficients and 
  `cMOl[2]` and `cMOr[2]` - β-MO transformation coefficients. 
"""
function gen_ufock(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix)
  return SpinMatrix(gen_fock(EC, :α, cMOl.α, cMOr.α, cMOl.β, cMOr.β), 
                    gen_fock(EC, :β, cMOl.β, cMOr.β, cMOl.α, cMOr.α))
end

"""
    gen_ufock(EC::ECInfo, den::SpinMatrix)

  Calculate UHF fock matrix from FCIDump integrals and density matrix `den`. 
"""
function gen_ufock(EC::ECInfo, den::SpinMatrix)
  return SpinMatrix(gen_fock(EC, :α, den.α, den.β), 
                    gen_fock(EC, :β, den.β, den.α))
end

""" 
    gen_dffock(EC::ECInfo, cMO::AbstractMatrix, bao, bfit)

  Compute closed-shell DF-HF Fock matrix (integral direct) in AO basis.
"""
function gen_dffock(EC::ECInfo{T}, cMO::AbstractMatrix, bao, bfit) where T
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  nA = size(CMO2, 1)
  nocc = size(CMO2, 2)
  nL = size(PL, 2)
  Pbatches = BasisBatcher(bao, bfit)
  maxP = max_batch_length(Pbatches)
  LoA = zeros(T, nL, nocc, nA)
  lenbuf = (nocc*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(T, lenbuf) cbuf(Cdouble, lencbuf) begin
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    oAP = alloc!(buf, nocc, nA, lenP)
    AAP = alloc!(buf, nA, nA, lenP)
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor oAP[j,ν,P] = AAP[μ,ν,P] * CMO2[μ,j]
    drop!(buf, AAP)
    M_PL = alloc!(buf, lenP, nL)
    M_PL .= @view PL[P,:]
    @mtensor LoA[L,j,ν] += oAP[j,ν,P] * M_PL[P,L]
    drop!(buf, oAP, M_PL)
  end
  @mtensor cL[L] := LoA[L,j,ν] * CMO2[ν,j]
  @mtensor fock[μ,ν] := hsmall[μ,ν] - LoA[L,j,μ]*LoA[L,j,ν] 
  @mtensor cP[P] := cL[L] * PL[P,L]
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    AAP = alloc!(buf, nA, nA, lenP)
    v!cP = @mview cP[P]
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor fock[μ,ν] += 2.0*v!cP[P]*AAP[μ,ν,P]
    drop!(buf, AAP)
  end
  end #buffer
  return fock
end

""" 
    gen_dffock(EC::ECInfo, cMO::Matrix{Float64}, cPO::Matrix{Float64}, bao, bfit)

  Compute closed-shell DF-HF electron and positron Fock matrices
  (integral direct) in AO basis.
"""
function gen_dffock(EC::ECInfo, cMO::Matrix{Float64}, cPO::Matrix{Float64}, bao, bfit)
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  hsmall_pos = load2idx(EC, "h_positron_AA")
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  occp = EC.space['p']
  CMO2 = cMO[:,occ2]
  CPO2 = cPO[:,occp]
  nA = size(CMO2, 1)
  nocc = size(CMO2, 2)
  noccp = size(CPO2, 2) # = 1
  nL = size(PL, 2)
  # FIXME: Why does one need 1000 here?
  Pbatches = BasisBatcher(bao, bfit, 1000)
  maxP = max_batch_length(Pbatches)
  LoA = zeros(nL, nocc, nA)
  LpA = zeros(nL, noccp, nA)
  J = zeros(nA, nA)
  Jp = zeros(nA, nA)
  fock_pos = hsmall_pos
  lenbuf = ((nocc+noccp)*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(lenbuf) cbuf(Cdouble, lencbuf) begin
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    oAP = alloc!(buf, nocc, nA, lenP)
    pAP = alloc!(buf, noccp, nA, lenP)
    AAP = alloc!(buf, nA, nA, lenP)
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor oAP[j,ν,P] = AAP[μ,ν,P] * CMO2[μ,j]
    @mtensor pAP[j,ν,P] = AAP[μ,ν,P] * CPO2[μ,j]
    drop!(buf, AAP)
    M_PL = alloc!(buf, lenP, nL)
    M_PL .= @view PL[P,:]
    @mtensor LoA[L,j,ν] += oAP[j,ν,P] * M_PL[P,L]
    @mtensor LpA[L,j,ν] += pAP[j,ν,P] * M_PL[P,L]
    reset!(buf)
  end
  @mtensor cL[L] := LoA[L,j,ν] * CMO2[ν,j]
  @mtensor cL_p[L] := LpA[L,j,ν] * CPO2[ν,j]
  @mtensor fock[μ,ν] := hsmall[μ,ν] - LoA[L,j,μ]*LoA[L,j,ν] 
  @mtensor cP[P] := cL[L] * PL[P,L]
  @mtensor cP_p[P] := cL_p[L] * PL[P,L]
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    AAP = alloc!(buf, nA, nA, lenP)
    v!cP = @mview cP[P]
    v!cP_p = @mview cP_p[P]
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor J[μ,ν] += v!cP[P]*AAP[μ,ν,P]
    @mtensor Jp[μ,ν] += v!cP_p[P]*AAP[μ,ν,P]
    @mtensor fock[μ,ν] += 2.0*J[μ,ν] - Jp[μ,ν]
    @mtensor fock_pos[μ,ν] -= 2.0*J[μ,ν]
    drop!(buf, AAP)
  end
  end #buffer
  return fock, fock_pos, Jp
end

""" 
    gen_dffock(EC::ECInfo{T}, cMO::SpinMatrix, bao, bfit)

  Compute unrestricted DF-HF Fock matrices `SpinMatrix(Fα, Fβ)` in AO basis (integral direct).
"""
function gen_dffock(EC::ECInfo{T}, cMO::SpinMatrix, bao, bfit) where T
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  # println(size(Ppq))
  occa = EC.space['o']
  occb = EC.space['O']
  CMOo = SpinMatrix(cMO[1][:,occa], cMO[2][:,occb])
  fock = SpinMatrix(hsmall)
  unrestrict!(fock)
  nA = size(CMOo[1], 1)
  nocc = size(CMOo[1], 2)
  nOcc = size(CMOo[2], 2)
  nL = size(PL, 2)
  Pbatches = BasisBatcher(bao, bfit)
  maxP = max_batch_length(Pbatches)
  LoA = zeros(T, nL, nocc, nA)
  LOA = zeros(T, nL, nOcc, nA)
  lenbuf = ((nocc+nOcc)*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(T, lenbuf) cbuf(Cdouble, lencbuf) begin
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    oAP = alloc!(buf, nocc, nA, lenP)
    OAP = alloc!(buf, nOcc, nA, lenP)
    AAP = alloc!(buf, nA, nA, lenP)
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor oAP[j,ν,P] = AAP[μ,ν,P] * CMOo[1][μ,j]
    if nOcc > 0
      @mtensor OAP[j,ν,P] = AAP[μ,ν,P] * CMOo[2][μ,j]
    end
    drop!(buf, AAP)
    M_PL = alloc!(buf, lenP, nL)
    M_PL .= @view PL[P,:]
    @mtensor LoA[L,j,ν] += oAP[j,ν,P] * M_PL[P,L]
    if nOcc > 0
      @mtensor LOA[L,j,ν] += OAP[j,ν,P] * M_PL[P,L]
    end
    reset!(buf)
  end
  @mtensor cL[L] := LoA[L,j,ν] * CMOo[1][ν,j]
  @mtensor cL[L] += LOA[L,j,ν] * CMOo[2][ν,j]
  @mtensor fock[1][μ,ν] -= LoA[L,j,μ]*LoA[L,j,ν] 
  @mtensor fock[2][μ,ν] -= LOA[L,j,μ]*LOA[L,j,ν] 
  @mtensor cP[P] := cL[L] * PL[P,L]
  coulfock = zeros(T, nA, nA)
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    AAP = alloc!(buf, nA, nA, lenP)
    v!cP = @mview cP[P]
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor coulfock[μ,ν] += v!cP[P]*AAP[μ,ν,P]
    drop!(buf, AAP)
  end
  fock[1] += coulfock
  fock[2] += coulfock
  end #buffer
  return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::AbstractMatrix)

  Compute closed-shell DF-HF Fock matrix in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo, cMO::AbstractMatrix)
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  CMO2d = permutedims(CMO2, [2,1])
  hsmall = load2idx(EC, "h_AA")
  AALfile, AAL = mmap3idx(EC, "AAL")
  nocc = size(CMO2, 2)
  nA = size(AAL, 1)
  nL = size(AAL, 3)
  fock = hsmall
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf((nocc*nA+1)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!AAL = @mview AAL[:,:,L]
    oAL = alloc!(buf, nocc, nA, lenL)
    @mtensor oAL[j,ν,L] = v!AAL[μ,ν,L] * CMO2[μ,j]
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = oAL[j,ν,L] * CMO2d[j,ν]
    @mtensor fock[μ,ν] += 2.0 * cL[L] * v!AAL[μ,ν,L]
    @mtensor fock[μ,ν] -= oAL[j,μ,L] * oAL[j,ν,L]
    drop!(buf, oAL, cL)
  end
  close(AALfile)
  end #buffer
  return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::AbstractMatrix, cPO::AbstractMatrix)

  Compute closed-shell DF-HF Fock matrix and the positron
  Fock matrix in AO basis  (using precalculated Cholesky-
  decomposed integrals and density matrices).
"""
function gen_dffock(EC::ECInfo, cMO::AbstractMatrix, cPO::AbstractMatrix)
  #TODO: rewrite with loops to reduce memory usage
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  CMO2p = cPO[:,1:1]
  hsmall = load2idx(EC,"h_AA")
  hsmall_pos = load2idx(EC,"h_positron_AA")
  AALfile, AAL = mmap3idx(EC, "AAL")
  # Electron
  @mtensor begin 
    μjL[p,j,L] := AAL[p,q,L] * CMO2[q,j]
    L[L] := μjL[p,j,L] * CMO2[p,j]
    J[p,q] := AAL[p,q,L] * L[L]
    K[p,q] := μjL[p,j,L] * μjL[q,j,L] 
  end
  # Positron
  @mtensor begin
    μjLpos[p,j,L] := AAL[p,q,L] * CMO2p[q,j]
    P[L] := μjLpos[p,j,L] * CMO2p[p,j]
    Jp[p,q] := AAL[p,q,L] * P[L] 
  end
  fock = hsmall + 2*J - K - Jp
  fock_pos = hsmall_pos - 2*J
  close(AALfile)
  return fock, fock_pos, Jp
  #return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::MOs)

  Compute unrestricted DF-HF Fock matrices [Fα, Fβ] in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo{T}, cMO::SpinMatrix) where T
  occa = EC.space['o']
  occb = EC.space['O']
  CMOo = SpinMatrix(cMO[1][:,occa], cMO[2][:,occb])
  CMOod = SpinMatrix(permutedims(CMOo[1], [2,1]), permutedims(CMOo[2], [2,1]))
  hsmall = load2idx(EC,"h_AA")
  fock = SpinMatrix(hsmall)
  unrestrict!(fock)
  AALfile, AAL = mmap3idx(EC, "AAL")
  nocc = size(CMOo[1], 2)
  nOcc = size(CMOo[2], 2)
  nA = size(AAL, 1)
  nL = size(AAL, 3)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(T, (nocc+nOcc)*nA*maxL + maxL) begin
  coulfock = zeros(T, nA, nA)
  for L in LBlks
    lenL = length(L)
    v!AAL = @mview AAL[:,:,L]
    oAL = alloc!(buf, nocc, nA, lenL)
    OAL = alloc!(buf, nOcc, nA, lenL)
    @mtensor oAL[j,ν,L] = v!AAL[μ,ν,L] * CMOo[1][μ,j]
    @mtensor OAL[j,ν,L] = v!AAL[μ,ν,L] * CMOo[2][μ,j]
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = oAL[j,ν,L] * CMOod[1][j,ν]
    if nOcc > 0
      @mtensor cL[L] += OAL[j,ν,L] * CMOod[2][j,ν]
    end
    @mtensor coulfock[μ,ν] += cL[L] * v!AAL[μ,ν,L]
    @mtensor fock[1][μ,ν] -= oAL[j,μ,L] * oAL[j,ν,L]
    @mtensor fock[2][μ,ν] -= OAL[j,μ,L] * OAL[j,ν,L]
    reset!(buf)
  end
  close(AALfile)
  fock[1] += coulfock
  fock[2] += coulfock
  end #buffer
  return fock
end

"""
    gen_df3idx_fock(EC::ECInfo, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, occ::AbstractVector{Int})

  Compute closed-shell Fock matrix from 1e-integrals `h1` and MO-basis 3-index integrals `mmL`.
  `occ` contains the indices of the occupied orbitals.

  ``F_p^q = h_p^q + 2 J_p^q - K_p^q``
  with ``J_p^q = \\sum_L v_p^{qL} c^L``, ``c^L = \\sum_i v_i^{iL}``
  and ``K_p^q = \\sum_{iL} v_p^{iL} v_i^{qL}``.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, 
                         occ::AbstractVector{Int}) where T
  norb = size(h1, 1)
  nL = size(mmL, 3)
  fock = copy(h1)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  nocc = length(occ)
  lenbuf = T <: Complex ? (2*nocc*norb + 1)*maxL : (nocc*norb + 1)*maxL
  @buffer buf(T, lenbuf) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    # Coulomb: cL = Σ_i mmL[i,i,L]
    cL = alloc!(buf, lenL)
    v!ooL = @view mmL[occ,occ,L]
    @mtensor cL[L] = v!ooL[i,i,L]
    @mtensor fock[p,q] += 2.0 * cL[L] * v!mmL[p,q,L]
    drop!(buf, cL)
    # Exchange: K_pq = Σ_i Σ_L B[p,i,L] * B[i,q,L]
    # (symmetric decomposition: first index is conjugated orbital, second is normal)
    piL = alloc!(buf, norb, nocc, lenL)
    piL .= @view(mmL[:,occ,L])
    if T <: Complex
      ipL = alloc!(buf, nocc, norb, lenL)
      ipL .= @view(mmL[occ,:,L])
      @mtensor fock[p,q] -= piL[p,i,L] * ipL[i,q,L]
      drop!(buf, ipL)
    else
      @mtensor fock[p,q] -= piL[p,i,L] * piL[q,i,L]
    end
    drop!(buf, piL)
  end
  end #buffer
  return fock
end

"""
    gen_df3idx_fock(EC::ECInfo, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, cMO_occ::AbstractMatrix)

  Compute closed-shell Fock matrix from 1e-integrals `h1` and MO-basis 3-index integrals `mmL`
  using occupied MO coefficients `cMO_occ` (rotation from original to occupied MOs).
  Uses symmetric decomposition convention: ``v_{pr}^{qs} = \\sum_L v_{p}^{qL} v_{r}^{sL}``.
  Lower index of ``v`` transforms with ``\\bar{C}`` (conjugated), upper index with ``C``.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, 
                         cMO_occ::AbstractMatrix) where T
  norb = size(h1, 1)
  nL = size(mmL, 3)
  nocc = size(cMO_occ, 2)
  fock = copy(h1)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  lenbuf = T <: Complex ? (2*nocc*norb + 1)*maxL : (nocc*norb + 1)*maxL
  @buffer buf(T, lenbuf) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    # half-transform: omL[j,q,L] = Σ_p mmL[p,q,L] * cMO_occ^*[p,j]
    omL = alloc!(buf, nocc, norb, lenL)
    @mtensor omL[j,q,L] = v!mmL[p,q,L] * conj(cMO_occ[p,j])
    # Coulomb: cL = Σ_{jq} oqL[j,q,L] * cMO_occ[q,j]
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = omL[j,q,L] * cMO_occ[q,j]
    @mtensor fock[p,q] += 2.0 * cL[L] * v!mmL[p,q,L]
    drop!(buf, cL)
    # Exchange: K_pq = Σ_{jL} moL[p,j,L])*omL[j,q,L]
    if T <: Complex
      moL = alloc!(buf, norb, nocc, lenL)
      @mtensor moL[p,j,L] = v!mmL[p,q,L] * cMO_occ[q,j]
      @mtensor fock[p,q] -= moL[p,j,L] * omL[j,q,L]
      drop!(buf, moL)
    else
      @mtensor fock[p,q] -= omL[j,p,L] * omL[j,q,L]
    end
    drop!(buf, omL)
  end
  end #buffer
  return fock
end

"""
    gen_df3idx_fock(EC::ECInfo, h1a, h1b, mmL, MML, occa, occb)

  Compute UHF Fock matrices from 1e-integrals `h1a`/`h1b` and MO-basis 
  3-index integrals `mmL` (α) and `MML` (β).

  Returns `SpinMatrix(Fα, Fβ)`.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1a::AbstractMatrix, h1b::AbstractMatrix,
                         mmL::AbstractArray{<:Number,3}, MML::AbstractArray{<:Number,3},
                         occa::AbstractVector{Int}, occb::AbstractVector{Int}) where T
  norb = size(h1a, 1)
  nL = size(mmL, 3)
  focka = copy(h1a)
  fockb = copy(h1b)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  nocca = length(occa)
  noccb = length(occb)
  lenbuf = T <: Complex ? (2*max(nocca, noccb)*norb + 1)*maxL : (max(nocca, noccb)*norb + 1)*maxL
  @buffer buf(T, lenbuf) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    v!MML = @mview MML[:,:,L]
    # Total Coulomb: cL_total = Σ_iα mmL[i,i,L] + Σ_Iβ MML[I,I,L]
    cL = alloc!(buf, lenL)
    v!ooL = @view mmL[occa,occa,L]
    v!OO_L = @view MML[occb,occb,L]
    @mtensor cL[L] = v!ooL[i,i,L]
    @mtensor cL[L] += v!OO_L[I,I,L]
    @mtensor focka[p,q] += cL[L] * v!mmL[p,q,L]
    @mtensor fockb[p,q] += cL[L] * v!MML[p,q,L]
    drop!(buf, cL)
    # α exchange
    if nocca > 0
      piL = alloc!(buf, norb, nocca, lenL)
      piL .= @view(mmL[:,occa,L])
      if T <: Complex
        ipL = alloc!(buf, nocca, norb, lenL)
        ipL .= @view(mmL[occa,:,L])
        @mtensor focka[p,q] -= piL[p,i,L] * ipL[i,q,L]
        drop!(buf, ipL)
      else
        @mtensor focka[p,q] -= piL[p,i,L] * piL[q,i,L]
      end
      drop!(buf, piL)
    end
    # β exchange
    if noccb > 0
      PIL = alloc!(buf, norb, noccb, lenL)
      PIL .= @view(MML[:,occb,L])
      if T <: Complex
        IPL = alloc!(buf, noccb, norb, lenL)
        IPL .= @view(MML[occb,:,L])
        @mtensor fockb[p,q] -= PIL[p,i,L] * IPL[i,q,L]
        drop!(buf, IPL)
      else
        @mtensor fockb[p,q] -= PIL[p,i,L] * PIL[q,i,L]
      end
      drop!(buf, PIL)
    end
  end
  end #buffer
  return SpinMatrix(focka, fockb)
end

"""
    gen_df3idx_fock(EC::ECInfo, h1a, h1b, mmL, MML, cMO_occa::AbstractMatrix, cMO_occb::AbstractMatrix)

  Compute UHF Fock matrices from 1e-integrals and MO-basis 3-index integrals
  using occupied MO coefficients (rotations from original to occupied MOs).
  Uses symmetric decomposition convention: ``v_{pr}^{qs} = \\sum_L v_{p}^{qL} v_{r}^{sL}``.
  Lower index of ``v`` transforms with ``\\bar{C}`` (conjugated), upper index with ``C``.

  Returns `SpinMatrix(Fα, Fβ)`.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1a::AbstractMatrix, h1b::AbstractMatrix,
                         mmL::AbstractArray{<:Number,3}, MML::AbstractArray{<:Number,3},
                         cMO_occa::AbstractMatrix, cMO_occb::AbstractMatrix) where T
  norb = size(h1a, 1)
  nL = size(mmL, 3)
  nocca = size(cMO_occa, 2)
  noccb = size(cMO_occb, 2)
  focka = copy(h1a)
  fockb = copy(h1b)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  lenbuf = T <: Complex ? (2*(nocca + noccb)*norb + 1)*maxL : ((nocca + noccb)*norb + 1)*maxL
  @buffer buf(T, lenbuf) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    v!MML = @mview MML[:,:,L]
    # half-transform α: omL[j,q,L] = Σ_p mmL[p,q,L] * cMO_occa[p,j]
    omL = alloc!(buf, nocca, norb, lenL)
    @mtensor omL[j,q,L] = v!mmL[p,q,L] * conj(cMO_occa[p,j])
    # half-transform β: OML[j,q,L] = Σ_p MML[p,q,L] * cMO_occb[p,j]
    OML = alloc!(buf, noccb, norb, lenL)
    if noccb > 0
      @mtensor OML[j,q,L] = v!MML[p,q,L] * conj(cMO_occb[p,j])
    end
    # Total Coulomb: cL = Σ_{jq} omL[j,q,L]*cMO_occa[q,j] + Σ_{jq} OML[j,q,L]*cMO_occb[q,j]
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = omL[j,q,L] * cMO_occa[q,j]
    if noccb > 0
      @mtensor cL[L] += OML[j,q,L] * cMO_occb[q,j]
    end
    @mtensor focka[p,q] += cL[L] * v!mmL[p,q,L]
    @mtensor fockb[p,q] += cL[L] * v!MML[p,q,L]
    drop!(buf, cL)
    # β exchange (drop OML before omL for LIFO order)
    if noccb > 0
      if T <: Complex
        MOL = alloc!(buf, norb, noccb, lenL)
        @mtensor MOL[p,j,L] = v!MML[p,q,L] * cMO_occb[q,j]
        @mtensor fockb[p,q] -= MOL[p,j,L] * OML[j,q,L]
        drop!(buf, MOL)
      else
        @mtensor fockb[p,q] -= OML[j,p,L] * OML[j,q,L]
      end
    end
    drop!(buf, OML)
    # α exchange
    if T <: Complex
      moL = alloc!(buf, norb, nocca, lenL)
      @mtensor moL[p,j,L] = v!mmL[p,q,L] * cMO_occa[q,j]
      @mtensor focka[p,q] -= moL[p,j,L] * omL[j,q,L]
      drop!(buf, moL)
    else
      @mtensor focka[p,q] -= omL[j,p,L] * omL[j,q,L]
    end
    drop!(buf, omL)
  end
  end #buffer
  return SpinMatrix(focka, fockb)
end

end #module
