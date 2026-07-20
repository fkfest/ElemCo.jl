"""
This module contains tools for generating integral dumps from AO integrals —
both density-fitted and exact (non-DF) — and transforming them AO→MO.
"""
module IntegralTools
using LinearAlgebra
using Buffers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.Integrals
using ..ElemCo.MSystems
using ..ElemCo.FockFactory
using ..ElemCo.PMStore
using ..ElemCo.TensorTools
using ..ElemCo.FciDumps
using ..ElemCo.OrbTools

export generate_AO_DF_integrals, generate_DF_integrals, generate_DF_Fock
export generate_3idx_integrals, contract_df_integrals!, transform_3idx!
export calc_system_df_integrals
export ao_integrals, ensure_ao_integrals!, save_ao_1e_integrals!, generate_mo_dump
export delete_ao_integrals!, invalidate_ao_1e_integrals!
export transform_int2, transform_int2_Q, transform_fcidump!

"""
    generate_AO_DF_integrals(EC::ECInfo, fitbasis="mpfit"; save3idx=true)

  Generate AO integrals using DF + Cholesky.
  If save3idx is true, save Cholesky-decomposed 3-index integrals, 
  otherwise save pseudo-square-root-inverse Cholesky decomposition.

  Return nuclear repulsion energy.
"""
function generate_AO_DF_integrals(EC::ECInfo, fitbasis="mpfit"; save3idx=true)
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, fitbasis)
  S_AA = overlap(bao)
  t_AA = kinetic(bao)
  v_AA = nuclear(bao)
  save!(EC, "S_AA", S_AA)
  save!(EC, "h_AA", t_AA + v_AA)  
  if EC.options.wf.npositron > 0
    save!(EC, "h_positron_AA", t_AA - v_AA)
  end
  PQ = eri_2e2idx(bfit)
  println("Number of fitting functions in $fitbasis: ", size(PQ, 2))
  M = sqrtinvchol(PQ, tol=EC.options.cholesky.thred, verbose=true)
  println("Number of fitting functions in $fitbasis after Cholesky: ", size(M, 2))
  if save3idx
    Pbatches = BasisBatcher(bao, bfit, EC.options.int.target_batch_length)
    lencbuf = buffer_size_3idx(Pbatches)
    maxP = max_batch_length(Pbatches)
    nA = size(S_AA, 1)
    nL = size(M, 2)
    AALfile, AAL = newmmap(EC, "AAL", (nA,nA,nL))
    @buffer buf(nA*nA*maxP + nL*maxP) cbuf(Cdouble, lencbuf) begin
    LBlks = get_spaceblocks(1:nL)
    first = true
    for Pblk in Pbatches
      P = range(Pblk)
      lenP = length(P)
      AAP = alloc!(buf, nA, nA, lenP)
      eri_2e3idx!(AAP, cbuf, Pblk)
      M_PL = alloc!(buf, lenP, nL)
      M_PL .= @view M[P,:]
      if first
        for L in LBlks
          v!M = @mview M_PL[:,L]
          v!AAL = @mview AAL[:,:,L]
          @mtensor v!AAL[p,q,L] = AAP[p,q,P] * v!M[P,L]
        end
        first = false
      else
        for L in LBlks
          v!M = @mview M_PL[:,L]
          v!AAL = @mview AAL[:,:,L]
          @mtensor v!AAL[p,q,L] += AAP[p,q,P] * v!M[P,L]
        end
      end
      drop!(buf, AAP, M_PL) 
    end
    closemmap(EC, AALfile, AAL)
    end #buffer
  else
    save!(EC, "C_PL", M)
  end
  return nuclear_repulsion(EC.system)
end

"""
    generate_3idx_integrals(EC::ECInfo, cMO::SpinMatrix, fitbasis="mpfit"; save3idx=true)

  Generate ``v_p^{qL}`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``
  and store in file `mmL`.
  If `save3idx` is false, no 3-index integrals are calculated, only save pseudo-square-root-inverse Cholesky decomposition.
"""
function generate_3idx_integrals(EC::ECInfo, cMO::SpinMatrix, fitbasis="mpfit"; save3idx=true)
  generate_AO_DF_integrals(EC, fitbasis; save3idx)
  if !save3idx
    return
  end
  AALfile, AAL = mmap3idx(EC, "AAL")
  nao = size(cMO[1], 1)
  nmo = size(cMO[1], 2)
  nL = size(AAL, 3)
  unrestricted = !is_restricted(cMO)
  mmLfile, mmL = newmmap(EC, "mmL", (nmo,nmo,nL))
  if unrestricted
    MMLfile, MML = newmmap(EC, "MML", (nmo,nmo,nL))
  end
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(nmo*nao*maxL) begin
  c_Am = cMO[1]
  c_AM = cMO[2]
  for L in LBlks
    lenL = length(L)
    v!AAL = @mview AAL[:,:,L]
    v!mmL = @mview mmL[:,:,L]
    mAL = alloc!(buf, nmo, nao, lenL)
    @mtensor mAL[p,ν,L] = c_Am[μ,p] * v!AAL[μ,ν,L]
    @mtensor v!mmL[p,q,L] = mAL[p,ν,L] * c_Am[ν,q]
    drop!(buf, mAL)
    if unrestricted
      v!MML = @mview MML[:,:,L]
      MAL = alloc!(buf, nmo, nao, lenL)
      @mtensor MAL[p,ν,L] = c_AM[μ,p] * v!AAL[μ,ν,L]
      @mtensor v!MML[p,q,L] = MAL[p,ν,L] * c_AM[ν,q]
      drop!(buf, MAL)
    end
  end
  close(AALfile)
  closemmap(EC, mmLfile, mmL)
  if unrestricted
    closemmap(EC, MMLfile, MML)
  end
  end #buffer
  return
end

"""
    generate_DF_integrals(EC::ECInfo, cMO::SpinMatrix; save3idx=true)

  Generate ``v_p^{qL}`` and ``f_p^q`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``.
  The ``v_p^{qL}`` are generated using `mpfit` fitting basis, and
  the ``f_p^q`` are generated using `jkfit` fitting basis.
  The integrals are stored in files `mmL` and `f_mm`.

  Return reference energy (calculated using `jkfit` fitting basis).
"""
function generate_DF_integrals(EC::ECInfo, cMO::SpinMatrix; save3idx=true)
  if isempty(EC.system)
    error("Molecular system not specified!")
  end
  # calculate fock matrix in AO basis (integral direct)
  EHF = generate_DF_Fock(EC, cMO)
  # calculate 3-index integrals
  generate_3idx_integrals(EC, cMO, "mpfit"; save3idx)
  return EHF
end

"""
    calc_system_df_integrals(EC::ECInfo)

  Calculate 3-index integrals for the `EC.system` and store them in `mmL` file.
  The routine is intended to be used in a combination with FDump integrals.
"""
function calc_system_df_integrals(EC::ECInfo)
  space_save, _ = restore_system_space!(EC)
  cMO = load_orbitals(EC)
  # correlated MOs
  SP = EC.space
  if is_restricted(cMO) && SP['o'] == SP['O']
    coMO = SpinMatrix(cMO[1][:,vcat(SP['o'],SP['v'])])
  else
    coMO = SpinMatrix(cMO[1][:,vcat(SP['o'],SP['v'])], cMO[2][:,vcat(SP['O'],SP['V'])])
  end
  generate_3idx_integrals(EC, coMO, "mpfit")
  restore_space!(EC, space_save)
end

"""
    generate_DF_Fock(EC::ECInfo, cMO::SpinMatrix; check_diagonal=false)

  Generate DF Fock matrix in MO basis.
  If `check_diagonal` is true, check the off-diagonal elements of the Fock matrix to be small.
  The Fock matrix is saved in files `f_mm`/`f_MM` and orbital energies in `e_m`/`e_M`.

  Return reference energy.
"""
function generate_DF_Fock(EC::ECInfo, cMO::SpinMatrix; check_diagonal=false)
  if isempty(EC.system)
    error("Molecular system not specified!")
  end
  occα = EC.space['o']
  occβ = EC.space['O']
  # calculate fock matrix in AO basis (integral direct)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "jkfit")
  h_AA = load2idx(EC, "h_AA")
  if is_restricted(cMO) && occα == occβ
    # restricted closed-shell
    fock = SpinMatrix(gen_dffock(EC, cMO[1], bao, bfit))
    nspin = 1
  else
    # unrestricted
    fock = gen_dffock(EC, cMO, bao, bfit)
    nspin = 2
  end
  EHF = 0.0
  for isp in 1:nspin
    fock_MO = cMO[isp]' * fock[isp] * cMO[isp]
    m = ('m','M')[isp]
    occ = (occα, occβ)[isp]
    save!(EC, "f_$m$m", fock_MO)
    eps = diag(fock_MO)
    println("Occupied orbital energies: ", eps[occ])
    save!(EC, "e_$m", eps)
    if nspin == 1
      save!(EC, "e_M", eps)
    end
    if check_diagonal
      # Checking off-diagonal elements of fock matrix
      maxoff = maximum(abs, fock_MO - Diagonal(fock_MO))
      if maxoff > 1e-8
        if EC.options.wf.ignore_error
          warnerror("The largest off-diagonal element of fock matrix is $maxoff > 1e-8")
        else
          error("The largest off-diagonal element of fock matrix is $maxoff > 1e-8 
          The error can be ignored by setting wf,ignore_error=true.")
        end
      end
    end
    hsmall = cMO[isp]' * h_AA * cMO[isp]
    EHF += sum(eps[occ]) + sum(diag(hsmall)[occ])
  end
  EHF /= nspin
  EHF += nuclear_repulsion(EC.system)
  return EHF
end

"""
    contract_df_integrals!(EC::ECInfo)

  Contract 3-index DF integrals from scratch file `mmL` into 4-index integrals
  and store them in `EC.fd.int2`.

  The 3-index integrals ``B_p^{qL}`` are stored in the scratch file `mmL` with
  shape `(norb, norb, naux)`. The 4-index integrals are computed as:
  ``v_{pr}^{qs} = \\sum_L B_p^{qL} B_r^{sL}``
  and stored in upper-triangular format for the last two indices (q ≤ s).

  After contraction, `EC.fd.df3idx` is set to `false`.
"""
function contract_df_integrals!(EC::ECInfo{T}) where T
  @assert EC.fd.df3idx "df3idx flag must be set"
  @assert !EC.fd.uhf "UHF contract_df_integrals! not implemented yet"
  norbs = headvar(EC.fd, "NORB", Int)

  println("Contracting 3-index DF integrals to 4-index...")

  mmLfile, mmL = mmap3idx(EC, "mmL")
  nL = size(mmL, 3)
  @assert size(mmL, 1) == norbs && size(mmL, 2) == norbs "mmL shape mismatch"

  ntri = norbs * (norbs + 1) ÷ 2
  filename2 = int2_npy_filename(EC.fd)
  int2_file, int2 = newmmap(EC, filename2, (norbs, norbs, ntri), description="int2")

  LBlks = get_spaceblocks(1:nL, 512)
  maxL = maximum(length, LBlks)
  @buffer buf(T, norbs*norbs*maxL) begin
  first = true
  for L in LBlks
    lenL = length(L)
    Lmm = alloc!(buf, lenL, norbs, norbs)
    v!mmL = @mview mmL[:,:,L]
    @mtensor Lmm[L,p,q] = v!mmL[p,q,L]
    # int2[p,r,tri(q,s)] = Σ_L mmL[p,q,L] * mmL[r,s,L]
    for s = 1:norbs
      v!Lmm_s = @mview Lmm[:,:,s]     # (L,norb)
      for q = 1:s
        v!Lmm_q = @mview Lmm[:,:,q]   # (L, norb)
        qs = uppertriangular_index(q, s)
        v!int2 = @mview int2[:,:,qs]    # (norb, norb)
        if first
          @mtensor v!int2[p,r] = v!Lmm_q[L,p] * v!Lmm_s[L,r]
        else
          @mtensor v!int2[p,r] += v!Lmm_q[L,p] * v!Lmm_s[L,r]
        end
      end
    end
    drop!(buf, Lmm)
    first = false
  end
  end #buffer
  flushmmap(EC, int2)
  EC.fd.int2 = int2
  close(mmLfile)

  EC.fd.df3idx = false
  println("  4-index integrals generated: ($norbs, $norbs, $ntri)")
end

"""
    transform_3idx!(EC::ECInfo, fname::String, U::AbstractMatrix)

  Transform 3-index integrals in-place: ``B_{pq}^{L} \\leftarrow U^\\dagger B U``.
  The integrals are memory-mapped from file `fname`.
"""
function transform_3idx!(EC::ECInfo{T}, fname::String, U::AbstractMatrix) where T
  mmLfile, mmL = mmap3idx(EC, fname; writable=true)
  nL = size(mmL, 3)
  norb = size(mmL, 1)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(T, norb*norb*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    mtL = alloc!(buf, norb, norb, lenL)
    @mtensor mtL[p,q',L] = v!mmL[p,q,L] * U[q,q']
    @mtensor v!mmL[p',q',L] = mtL[p,q',L] * conj(U[p,p'])
    drop!(buf, mtL)
  end
  end #buffer
  flushmmap(EC, mmL)
  close(mmLfile)
end

"""
    save_ao_1e_integrals!(EC::ECInfo) -> BasisSet

  Compute and store the 1-e integrals of the current system in the AO basis: the overlap
  ``S_{μν}`` and the core Hamiltonian ``h_{μν} = T_{μν} + V_{μν}`` under the standard
  `"S_AA"`/`"h_AA"` scratch keys.
  Cheap (2-index); (re)computed on demand by every consumer of the AO integral files.
  Return the AO basis.
"""
function save_ao_1e_integrals!(EC::ECInfo{T}) where T
  @assert !isempty(EC.system) "EC.system is not set up!"
  bao = generate_basis(EC, "ao")
  save!(EC, "S_AA", Matrix{T}(overlap(bao)))
  save!(EC, "h_AA", Matrix{T}(kinetic(bao) + nuclear(bao)))
  return bao
end

"""
    ao_integrals(EC::ECInfo) -> Float64

  Generate the exact (non-density-fitted) AO integrals of the current system and store
  them as files: the 2-e integrals `<μν|ρσ>` (physicists' notation,
  triangular `(ρ,σ)` packing) on the memory-mapped file `"ao_int2"`, and the 1-e
  integrals `"S_AA"`/`"h_AA"` (see [`save_ao_1e_integrals!`](@ref)).
  This is the entry point behind the `@ints` macro.

  The 2-e integrals are assembled batch-wise straight into the memory-mapped triangular
  `int2[μ,ν,tri(ρ,σ)] = ⟨μν|ρσ⟩` (see [`eri_2e4idx_tri!`](@ref)); the full `nao⁴`
  tensor is never materialized.

  If `EC.fd` currently holds (MO/FCIDUMP) integrals, they are discarded with a warning —
  a fresh AO integral generation supersedes any previously loaded/generated MO dump.

  Return the nuclear repulsion energy.
"""
function ao_integrals(EC::ECInfo{T}) where T
  @assert EC.options.wf.npositron == 0 "positrons are not supported with exact AO integrals (use density fitting)"
  if !isempty(EC.fd)
    @warn "ao_integrals (@ints) discards the MO/FCIDUMP integrals currently in EC.fd; " *
          "subsequent calculations will use the freshly generated AO integrals."
    EC.fd = FDump{T,3}()
  end
  bao = save_ao_1e_integrals!(EC)
  nao = size(load2idx(EC, "S_AA"), 1)
  ntri = nao*(nao+1)÷2
  int2_file, int2 = newmmap(EC, "ao_int2", (nao, nao, ntri), T; description="int2 ao")
  eri_2e4idx_tri!(int2, bao)
  closemmap(EC, int2_file, int2)
  return nuclear_repulsion(EC.system)
end

"""
    ensure_ao_integrals!(EC::ECInfo; method="@hf", alternative="@bohf")

  Make sure the exact AO integral files for the current system exist: the 2-e integrals
  (`"ao_int2"`) are generated with [`ao_integrals`](@ref) when the file is missing, and
  the cheap 1-e integrals (`"S_AA"`/`"h_AA"`) are always refreshed. The files are
  invalidated on geometry/basis changes by `@setupEC`.

  If `EC.fd` holds (MO/FCIDUMP) integrals, they are **discarded** with a warning:
  `method` (`@hf`/`@uhf`) runs on exact AO integrals, and a leftover MO dump would
  shadow the AO flow in subsequent correlated calculations. To run HF directly on
  FCIDUMP integrals, use `alternative` (`@bohf`/`@bouhf`) instead.
"""
function ensure_ao_integrals!(EC::ECInfo{T}; method="@hf", alternative="@bohf") where T
  if !isempty(EC.fd)
    @warn "$method runs on exact AO integrals and ignores the MO/FCIDUMP integrals " *
          "currently in EC.fd; EC.fd is cleared so that subsequent correlated " *
          "calculations use the AO integrals. To run HF on FCIDUMP integrals, " *
          "use $alternative instead."
    EC.fd = FDump{T,3}()
  end
  if !file_exists(EC, "ao_int2")
    ao_integrals(EC)
  else
    save_ao_1e_integrals!(EC)
  end
  return
end

"""
    delete_ao_integrals!(EC::ECInfo)

  Delete all exact AO integral scratch files (`"ao_int2"`, `"S_AA"`, `"h_AA"`) if present.
  Called when the geometry or basis changes — then even the 2-e AO integrals `(μν|ρσ)` and the
  overlap are invalid. For a pure nuclear charge/dummy change use [`invalidate_ao_1e_integrals!`](@ref),
  which keeps the (unchanged) 2-e integrals.
"""
function delete_ao_integrals!(EC::ECInfo)
  for f in ("ao_int2", "S_AA", "h_AA")
    file_exists(EC, f) && delete_file!(EC, f)
  end
  delete_pm_store!(EC)   # the ± supermatrices derive from ao_int2 — invalidate together
  return
end

"""
    invalidate_ao_1e_integrals!(EC::ECInfo)

  Invalidate only the 1-electron AO integral file `"h_AA"` (the core Hamiltonian `T + V`),
  which depends on the nuclear charges, so it is recomputed for the current system on demand.
  The exact 2-e integrals `"ao_int2"` and the overlap `"S_AA"` depend only on the basis
  functions (positions + basis), so they are **kept** — e.g. across a `charge`/`@dummy` change,
  where ghost atoms retain their basis functions and the ERIs are unchanged.
"""
function invalidate_ao_1e_integrals!(EC::ECInfo)
  file_exists(EC, "h_AA") && delete_file!(EC, "h_AA")
  return
end

# ==================================================================================================
#  4-index 2-electron integral transformation  v_{pq}^{rs} = v_{p'q'}^{r's'} Tl[p'p] Tl2[q'q] Tr[r'r] Tr2[s's]
#
#  Pure kernels (EC-free: plain arrays + a `membudget` byte budget — directly unit-testable) plus
#  EC-aware wrappers that memory-map the output and derive the budget from the machine/options. Lives
#  here (not in FciDumps) so the EC-aware entry points can call `newmmap`/`available_memory(EC)`
#  directly instead of threading allocator closures and budgets through every call site.
# ==================================================================================================

"""
    transform_int2_blocksize(ns, nin, np, nq, elsize; membudget)

  Choose the width of the output-4th-index block for the streaming 4-index integral transform so the
  per-block working set (the co-live intermediates `Z(nin³) + W(np·nin²) + Z2(np·nq·nin)`, each of
  `elsize` bytes) stays within `membudget` bytes. Capped at `ns` (a single block ⇒ the input is read
  once) and floored at 1. Blocking only tiles a loop, so the numerical result is independent of the
  choice — this purely trades peak memory against the number of passes over the input: small blocks
  (many passes) on a memory-starved node, a single pass when `membudget` is ample (e.g. a fat node).
"""
function transform_int2_blocksize(ns::Int, nin::Int, np::Int, nq::Int, elsize::Int; membudget::Int)
  per_unit = (nin*nin*nin + np*nin*nin + np*nq*nin) * elsize
  return clamp(membudget ÷ max(1, per_unit), 1, ns)
end

"""
    transform_int2_pqs_block!(buf, int2::Array{T,3}, Tl, Tl2, Tr2, tb) -> Z2

  Shared kernel for the streaming 4-index transform from the **triangular** input
  `int2[p',q',tri(r',s')] = <p'q'|r's'>` (r'≤s'). For the output-4th-index block `tb` (via `Tr2`),
  narrow the packed partner index `s'` into the block (correctly desymmetrizing the joint packing
  `<p'q'|r's'> = <q'p'|s'r'>`) and transform the two bra indices `p'→p`, `q'→q`, leaving the third
  input index `r'` untransformed. Returns `Z2[p,q,r',u]` (allocated from the buffer arena `buf`);
  the caller applies `Tr` to the remaining `r'` index. All heavy steps are BLAS-3 gemms.
"""
function transform_int2_pqs_block!(buf, int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                                   Tr2::AbstractArray, tb) where T
  nin = size(int2, 1); np = size(Tl, 2); nq = size(Tl2, 2); lent = length(tb)
  Z = alloc!(buf, nin, nin, nin, lent)
  fill!(Z, zero(T))
  for s = 1:nin
    off = strict_uppertriangular_range(s)          # packed columns (r',s), r' = 1:s-1
    if !isempty(off)
      Vc = @mview int2[:,:,off]                     # v_{p'q'}^{r',s}, r' = 1:s-1
      Tr2_s   = @mview Tr2[s, tb]                    # [u]
      Tr2_off = @mview Tr2[1:s-1, tb]               # [r', u]
      Zoff = @mview Z[:,:,1:s-1,:]
      @mtensor Zoff[p',q',r',u] += Vc[p',q',r'] * Tr2_s[u]     # partner s'=s  (r'<s', keep p',q')
      Zs = @mview Z[:,:,s,:]
      @mtensor Zs[p',q',u] += Vc[q',p',r'] * Tr2_off[r',u]     # partner s'=r'<s (r'>s', p'↔q' swap)
    end
    Vd = @mview int2[:,:, uppertriangular_index(s,s)]
    Tr2_d = @mview Tr2[s, tb]
    Zs = @mview Z[:,:,s,:]
    @mtensor Zs[p',q',u] += 0.5 * Vd[p',q'] * Tr2_d[u]          # diagonal, symmetrized 0.5(V+Vᵀ)
    @mtensor Zs[p',q',u] += 0.5 * Vd[q',p'] * Tr2_d[u]
  end
  W = alloc!(buf, np, nin, nin, lent)
  @mtensor W[p,q',r',u] = Z[p',q',r',u] * Tl[p',p]
  Z2 = alloc!(buf, np, nq, nin, lent)
  @mtensor Z2[p,q,r',u] = W[p,q',r',u] * Tl2[q',q]
  return Z2
end

"""
    transform_int2!(int2t, int2::Array{T,3}, Tl, Tl2, Tr, Tr2; membudget) -> int2t

  In-place triangular-output transform: write the transformed 2-e integrals into the preallocated
  `int2t` of size `(nout, nout, nout*(nout+1)÷2)` (in-memory or memory-mapped). Each packed output
  column is written exactly once, so `int2t` need not be zero-initialized. Streams the triangular
  integrals in blocks of the output partner index (via [`transform_int2_pqs_block!`](@ref)) and
  finishes with a matrix multiplication of the remaining index restricted to `r ≤ s` (so the
  triangular symmetry is exploited — only `r ≤ s` is formed). Requires the same-spin pattern
  `Tl≡Tl2`, `Tr≡Tr2` (implicit in triangular storage).
"""
function transform_int2!(int2t::AbstractArray{T,3}, int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                         Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  # General rectangular transform: input orbitals (`nin`, primed indices) → output orbitals (`nout`).
  nin = size(int2, 1)
  nout = size(Tl, 2)
  @assert size(Tl2,2) == nout && size(Tr,2) == nout && size(Tr2,2) == nout "transform_int2: all four transformation matrices must map onto the same number of output orbitals"
  @assert size(Tl,1) == nin && size(Tl2,1) == nin && size(Tr,1) == nin && size(Tr2,1) == nin "transform_int2: transformation matrices must have $nin rows (input orbitals)"
  @assert size(int2t) == (nout, nout, nout*(nout+1)÷2) "transform_int2!: output array must have size ($nout, $nout, $(nout*(nout+1)÷2))"
  bsz = transform_int2_blocksize(nout, nin, nout, nout, sizeof(T); membudget)
  oblks = get_spaceblocks(1:nout, bsz)
  maxlen = maximum(length, oblks; init=0)   # init=0 -> no-op for a reduce-to-empty (nout=0) transform
  @buffer buf(T, (nin*nin*nin + nout*nin*nin + nout*nout*nin)*maxlen) begin
  for tb in oblks
    Z2 = transform_int2_pqs_block!(buf, int2, Tl, Tl2, Tr2, tb)   # Z2[p,q,r',u]
    for (u, s) in enumerate(tb)
      v!int2t = @mview int2t[:,:,uppertriangular_range(s)]         # [nout,nout,s] = columns tri(1:s,s)
      Z2s = @mview Z2[:,:,:,u]                                     # [nout,nout,nin]
      v!Tr = @mview Tr[:,1:s]                                      # keep only r ≤ s (triangular)
      @mtensor v!int2t[p,q,r] = Z2s[p,q,r'] * v!Tr[r',r]           # BLAS-3, write column once
    end
    reset!(buf)
  end
  end #buffer
  return int2t
end

"""
    transform_int2_Q!(int2t, int2::Array{T,3}, Tl, Tl2, Tr, Tr2; membudget) -> int2t

  In-place full 4-index (dense) output transform from a **triangular** input. `int2t` must have size
  `(size(Tl,2), size(Tl2,2), size(Tr,2), size(Tr2,2))` (may be rectangular / spin-mixed, e.g. the
  `int2ab` block). `nin == size(int2,1)` must match the rows of all four matrices.
"""
function transform_int2_Q!(int2t::AbstractArray{T,4}, int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                           Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  nin = size(int2,1)
  np, nq, nr, ns = size(Tl,2), size(Tl2,2), size(Tr,2), size(Tr2,2)
  @assert size(Tl,1)==nin && size(Tl2,1)==nin && size(Tr,1)==nin && size(Tr2,1)==nin "transform_int2_Q!: transformation matrices must have $nin rows (input orbitals)"
  @assert size(int2t) == (np, nq, nr, ns) "transform_int2_Q!: output array must have size ($np, $nq, $nr, $ns)"
  bsz = transform_int2_blocksize(ns, nin, np, nq, sizeof(T); membudget)
  oblks = get_spaceblocks(1:ns, bsz)                # blocks of the 4th output index (via Tr2)
  maxlen = maximum(length, oblks; init=0)   # init=0 -> no-op for a reduce-to-empty (nout=0) transform
  # peak co-live intermediates per unit block: Z(nin³) + W(np·nin²) + Z2(np·nq·nin)
  @buffer buf(T, (nin*nin*nin + np*nin*nin + np*nq*nin)*maxlen) begin
  for tb in oblks
    Z2 = transform_int2_pqs_block!(buf, int2, Tl, Tl2, Tr2, tb)   # Z2[p,q,r',u]
    v!int2t = @mview int2t[:,:,:,tb]
    @mtensor v!int2t[p,q,r,u] = Z2[p,q,r',u] * Tr[r',r]           # write int2t[:,:,:,tb] ONCE
    reset!(buf)
  end
  end #buffer
  return int2t
end

"""
    transform_int2_Q!(int2t, int2::Array{T,4}, Tl, Tl2, Tr, Tr2; membudget) -> int2t

  In-place full 4-index (dense) output transform from a **dense** 4-index input (`int2t` of size
  `(norb, norb, norb, norb)`).
"""
function transform_int2_Q!(int2t::AbstractArray{T,4}, int2::Array{T,4}, Tl::AbstractArray, Tl2::AbstractArray,
                        Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  norb = size(int2,1)
  @assert size(int2t) == (norb, norb, norb, norb) "transform_int2_Q!: output array must have size ($norb, $norb, $norb, $norb)"
  bsz = transform_int2_blocksize(norb, norb, norb, norb, sizeof(T); membudget)
  oblks = get_spaceblocks(1:norb, bsz)
  maxlen = maximum(length, oblks; init=0)   # init=0 -> no-op for a reduce-to-empty (nout=0) transform
  @buffer buf(T, 2*norb*norb*norb*maxlen) begin
  for tb in oblks
    lent = length(tb)
    Tr2b = @mview Tr2[:, tb]                                # [t', u]
    Z = alloc!(buf, norb, norb, norb, lent)
    first = true
    for tpb in oblks
      v!int2 = @mview int2[:,:,:,tpb]
      v!Tr2b = @mview Tr2b[tpb,:]
      if first
        @mtensor Z[p',q',r',u] = v!int2[p',q',r',t'] * v!Tr2b[t',u]   # narrow t' into the output block
        first = false
      else
        @mtensor Z[p',q',r',u] += v!int2[p',q',r',t'] * v!Tr2b[t',u]   # narrow t' into the output block
      end
    end
    W = alloc!(buf, norb, norb, norb, lent)
    @mtensor W[p,q',r',u] = Z[p',q',r',u] * Tl[p',p]
    @mtensor Z[p,q,r',u] = W[p,q',r',u] * Tl2[q',q]          # reuse Z
    v!int2t = @mview int2t[:,:,:,tb]
    @mtensor v!int2t[p,q,r,u] = Z[p,q,r',u] * Tr[r',r]       # write int2t[:,:,:,tb] ONCE
    reset!(buf)
  end
  end #buffer
  return int2t
end

# --- in-memory convenience wrappers (EC-free; allocate a plain `zeros` output) --------------------

"""
    transform_int2(int2, Tl, Tl2, Tr, Tr2; membudget=available_memory()) -> int2t

  In-memory triangular-output transform (allocates a `zeros` output and calls
  [`transform_int2!`](@ref)). For large integrals prefer the memory-mapped `transform_int2(EC, …, key)`.
"""
function transform_int2(int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                        Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  nout = size(Tl, 2)
  int2t = zeros(T, nout, nout, nout*(nout+1)÷2)
  return transform_int2!(int2t, int2, Tl, Tl2, Tr, Tr2; membudget)
end
function transform_int2(int2::Array{T,4}, Tl::AbstractArray, Tl2::AbstractArray,
                        Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  return transform_int2_Q(int2, Tl, Tl2, Tr, Tr2; membudget)
end

"""
    transform_int2_Q(int2, Tl, Tl2, Tr, Tr2; membudget=available_memory()) -> int2t

  In-memory full 4-index (dense) transform (allocates a `zeros` output). For large integrals prefer
  the memory-mapped `transform_int2_Q(EC, …, key)`.
"""
function transform_int2_Q(int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                          Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  int2t = zeros(T, size(Tl,2), size(Tl2,2), size(Tr,2), size(Tr2,2))
  return transform_int2_Q!(int2t, int2, Tl, Tl2, Tr, Tr2; membudget)
end
function transform_int2_Q(int2::Array{T,4}, Tl::AbstractArray, Tl2::AbstractArray,
                          Tr::AbstractArray, Tr2::AbstractArray; membudget::Int = available_memory()) where T
  norb = size(int2,1)
  int2t = zeros(T, norb, norb, norb, norb)
  return transform_int2_Q!(int2t, int2, Tl, Tl2, Tr, Tr2; membudget)
end

# --- EC-aware wrappers: always memory-map the output under `key`; budget from available_memory(EC) -

"""
    transform_int2(EC::ECInfo, int2, Tl, Tl2, Tr, Tr2, key) -> int2t

  Triangular-output transform writing the result to a fresh memory-mapped scratch file named `key`
  ([`newmmap`](@ref)); the memory budget comes from [`available_memory`](@ref)`(EC)` (honoring
  `@set mem budget/fraction`). `key` must not name the scratch file currently backing `int2`.
"""
function transform_int2(EC::ECInfo, int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                        Tr::AbstractArray, Tr2::AbstractArray, key::AbstractString; description::AbstractString=key) where T
  nout = size(Tl, 2)
  int2t = newmmap(EC, key, (nout, nout, nout*(nout+1)÷2), T; description)[2]
  transform_int2!(int2t, int2, Tl, Tl2, Tr, Tr2; membudget=available_memory(EC))
  flushmmap(EC, int2t)
  return int2t
end

"""
    transform_int2_Q(EC::ECInfo, int2, Tl, Tl2, Tr, Tr2, key) -> int2t

  Full 4-index (dense) transform writing the result to a fresh memory-mapped scratch file named
  `key`; budget from [`available_memory`](@ref)`(EC)`. Accepts a triangular (`Array{T,3}`) or dense
  (`Array{T,4}`) input.
"""
function transform_int2_Q(EC::ECInfo, int2::Array{T,3}, Tl::AbstractArray, Tl2::AbstractArray,
                          Tr::AbstractArray, Tr2::AbstractArray, key::AbstractString; description::AbstractString=key) where T
  int2t = newmmap(EC, key, (size(Tl,2), size(Tl2,2), size(Tr,2), size(Tr2,2)), T; description)[2]
  transform_int2_Q!(int2t, int2, Tl, Tl2, Tr, Tr2; membudget=available_memory(EC))
  flushmmap(EC, int2t)
  return int2t
end
function transform_int2_Q(EC::ECInfo, int2::Array{T,4}, Tl::AbstractArray, Tl2::AbstractArray,
                          Tr::AbstractArray, Tr2::AbstractArray, key::AbstractString; description::AbstractString=key) where T
  norb = size(int2,1)
  int2t = newmmap(EC, key, (norb, norb, norb, norb), T; description)[2]
  transform_int2_Q!(int2t, int2, Tl, Tl2, Tr, Tr2; membudget=available_memory(EC))
  flushmmap(EC, int2t)
  return int2t
end

"""
    transform_int1(int1::AbstractArray, Tl::AbstractArray, Tr::AbstractArray) -> int1t

  Transform 1-e integrals to a new basis: `int1t[p,q] = int1[p',q'] Tl[p',p] Tr[q',q]`.
"""
function transform_int1(int1::AbstractArray, Tl::AbstractArray, Tr::AbstractArray)
  @mtensor int1t[p,q] := int1[p',q'] * Tl[p',p] * Tr[q',q]
  return int1t
end

"""
    transform_fcidump!(EC::ECInfo, fd::FDump, Tl::SpinMatrix, Tr::SpinMatrix)

  Transform the integrals of `fd` in place to a new basis using `Tl`, `Tr`. If `Tl`/`Tr` are
  unrestricted, an RHF dump is turned into a UHF dump. The transformed 2-e integrals are written to
  memory-mapped scratch files (their size bounded by [`available_memory`](@ref)`(EC)`); the 1-e
  integrals are transformed in memory. Intended as a **one-shot** rotation — the scratch keys are
  the block names, which must not already back `fd`'s current integrals (they never do for the
  in-memory / `mo_*`-backed dumps this is called on).
"""
function transform_fcidump!(EC::ECInfo, fd::FDump{T,N}, Tl::SpinMatrix, Tr::SpinMatrix) where {T<:Number,N}
  println("Transform integrals...")
  if !is_restricted(Tl) || !is_restricted(Tr)
    genuhfdump = true
  else
    genuhfdump = false
    @assert !fd.uhf # from uhf fcidump can generate only uhf fcidump
  end
  if fd.uhf
    fd.int2aa = transform_int2(EC, fd.int2aa, Tl[1], Tl[1], Tr[1], Tr[1], "int2aa")
    fd.int2bb = transform_int2(EC, fd.int2bb, Tl[2], Tl[2], Tr[2], Tr[2], "int2bb")
    fd.int2ab = transform_int2_Q(EC, fd.int2ab, Tl[1], Tl[2], Tr[1], Tr[2], "int2ab")
    fd.int1a = transform_int1(fd.int1a, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1b, Tl[2], Tr[2])
  elseif genuhfdump
    # change fcidump from rhf to uhf format
    fd.int2aa = transform_int2(EC, fd.int2, Tl[1], Tl[1], Tr[1], Tr[1], "int2aa")
    fd.int2bb = transform_int2(EC, fd.int2, Tl[2], Tl[2], Tr[2], Tr[2], "int2bb")
    fd.int2ab = transform_int2_Q(EC, fd.int2, Tl[1], Tl[2], Tr[1], Tr[2], "int2ab")
    fd.int1a = transform_int1(fd.int1, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1, Tl[2], Tr[2])
    fd.int2 = zeros(T, ntuple(i->0, Val(N)))
    fd.int1 = zeros(T, 0, 0)
    fd.head["IUHF"] = [1]
    fd.uhf = true
  else
    fd.int2 = transform_int2(EC, fd.int2, Tl[1], Tl[1], Tr[1], Tr[1], "int2")
    fd.int1 = transform_int1(fd.int1, Tl[1], Tr[1])
  end
  fd.modified = true
end

"""
    transform_fcidump!(fd::FDump, Tl::SpinMatrix, Tr::SpinMatrix)

  Like [`transform_fcidump!(EC, fd, Tl, Tr)`](@ref) but keeps the transformed 2-e integrals **in
  memory** (no `EC` / scratch files needed) — convenient for ad-hoc FDump manipulation. For large
  dumps prefer the `EC` method, which memory-maps the result.
"""
function transform_fcidump!(fd::FDump{T,N}, Tl::SpinMatrix, Tr::SpinMatrix) where {T<:Number,N}
  println("Transform integrals...")
  if !is_restricted(Tl) || !is_restricted(Tr)
    genuhfdump = true
  else
    genuhfdump = false
    @assert !fd.uhf # from uhf fcidump can generate only uhf fcidump
  end
  if fd.uhf
    fd.int2aa = transform_int2(fd.int2aa, Tl[1], Tl[1], Tr[1], Tr[1])
    fd.int2bb = transform_int2(fd.int2bb, Tl[2], Tl[2], Tr[2], Tr[2])
    fd.int2ab = transform_int2_Q(fd.int2ab, Tl[1], Tl[2], Tr[1], Tr[2])
    fd.int1a = transform_int1(fd.int1a, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1b, Tl[2], Tr[2])
  elseif genuhfdump
    # change fcidump from rhf to uhf format
    fd.int2aa = transform_int2(fd.int2, Tl[1], Tl[1], Tr[1], Tr[1])
    fd.int2bb = transform_int2(fd.int2, Tl[2], Tl[2], Tr[2], Tr[2])
    fd.int2ab = transform_int2_Q(fd.int2, Tl[1], Tl[2], Tr[1], Tr[2])
    fd.int1a = transform_int1(fd.int1, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1, Tl[2], Tr[2])
    fd.int2 = zeros(T, ntuple(i->0, Val(N)))
    fd.int1 = zeros(T, 0, 0)
    fd.head["IUHF"] = [1]
    fd.uhf = true
  else
    fd.int2 = transform_int2(fd.int2, Tl[1], Tl[1], Tr[1], Tr[1])
    fd.int1 = transform_int1(fd.int1, Tl[1], Tr[1])
  end
  fd.modified = true
end

"""
    generate_mo_dump(EC::ECInfo, cMO::AbstractMatrix) -> FDump

  Build an MO-basis [`FDump`](@ref) in `EC.fd` from the exact AO integral files
  (`"ao_int2"`/`"h_AA"`, see [`ao_integrals`](@ref)) and the (restricted/closed-shell)
  MO coefficients `cMO[μ,p]`. This is the non-DF analogue of `dfdump`.

  Exact (non-density-fitted) `O(N⁵)` four-index transformation, written straight onto a
  fresh memory-mapped scratch file (`"mo_int2"`, temporary — the dump is transient and
  re-derived on demand): the full MO tensor is never materialized in memory. `cMO` may
  be rectangular (`nao × nout` with `nout ≤ nao`): only those `nout` orbitals are kept
  (e.g. deleted virtuals and frozen virtuals excluded).

  `NELEC` is stored as the full (neutral) electron count — `charge` is applied later by
  `setup_space_fd!` — and frozen-core folding is left to the caller
  (`freeze_orbs_in_dump`), exactly as for a `dfdump`-generated dump.
"""
function generate_mo_dump(EC::ECInfo{T}, cMO::AbstractMatrix) where {T<:Number}
  @assert file_exists(EC, "ao_int2") "no AO integrals on file (\"ao_int2\"); generate them first (@ints / ao_integrals)"
  save_ao_1e_integrals!(EC)
  S = load2idx(EC, "S_AA")
  hAO = load2idx(EC, "h_AA")
  @assert size(cMO, 1) == size(S, 1) "cMO has $(size(cMO,1)) AOs but the system has $(size(S,1))"
  # the orbitals must be orthonormal in the AO basis for the transform to yield correct MO integrals
  @assert isapprox(cMO' * S * cMO, I, atol=1e-8) "cMO are not orthonormal in the AO basis"
  println("Transform AO integrals to MO basis...")
  C = Matrix{T}(cMO)
  nout = size(C, 2)
  aofile, aoint2 = mmap3idx(EC, "ao_int2")
  int2 = transform_int2(EC, aoint2, C, C, C, C, "mo_int2"; description="tmp")
  close(aofile)
  # NELEC/MS2 conventions follow `dfdump`: neutral electron count, `charge`/`ms2` from the
  # wf options are applied later by `setup_space_fd!`.
  nelec = EC.options.wf.nelec < 0 ? guess_nelec(EC.system) : EC.options.wf.nelec
  ms2 = EC.options.wf.ms2 < 0 ? mod(nelec, 2) : EC.options.wf.ms2
  fd = FDump{T,3}(nout, nelec; ms2)
  fd.int2 = int2
  fd.int1 = Matrix{T}(C' * hAO * C)
  fd.int0 = nuclear_repulsion(EC.system)
  EC.fd = fd
  return fd
end

"""
    generate_mo_dump(EC::ECInfo, cMO::SpinMatrix) -> FDump

  Build an MO-basis [`FDump`](@ref) in `EC.fd` from the exact AO integral files and the MO
  coefficients `cMO`. For a restricted `cMO` this builds a closed-shell (RHF) dump (see the
  matrix method); for an unrestricted `cMO` it builds an unrestricted (UHF) dump with the
  spin blocks `int2aa`/`int2bb` (`v_{pq}^{rs}` in each spin, triangular) and `int2ab`
  (`v_{pQ}^{rS}`, full 4-index), and per-spin 1-e integrals — the exact-AO analogue of the
  `rhf→uhf` branch of [`transform_fcidump!`](@ref). Both spins must have the same orbital
  count (a single `NORB`); each block may be rectangular (deleted / frozen-virtual orbitals
  dropped). Frozen-core folding is left to the caller ([`freeze_orbs_in_dump`](@ref)).
"""
function generate_mo_dump(EC::ECInfo{T}, cMO::SpinMatrix) where {T<:Number}
  is_restricted(cMO) && return generate_mo_dump(EC, cMO.α)
  @assert file_exists(EC, "ao_int2") "no AO integrals on file (\"ao_int2\"); generate them first (@ints / ao_integrals)"
  save_ao_1e_integrals!(EC)
  S = load2idx(EC, "S_AA")
  hAO = load2idx(EC, "h_AA")
  Ca = Matrix{T}(cMO.α); Cb = Matrix{T}(cMO.β)
  nout = size(Ca, 2)
  @assert size(Cb, 2) == nout "α and β must keep the same number of MOs (single NORB), got $(size(Ca,2)) and $(size(Cb,2))"
  @assert size(Ca, 1) == size(S, 1) && size(Cb, 1) == size(S, 1) "cMO AO dimension does not match the system"
  @assert isapprox(Ca' * S * Ca, I, atol=1e-8) && isapprox(Cb' * S * Cb, I, atol=1e-8) "cMO are not orthonormal in the AO basis"
  println("Transform AO integrals to UHF MO basis...")
  aofile, aoint2 = mmap3idx(EC, "ao_int2")
  int2aa = transform_int2(EC, aoint2, Ca, Ca, Ca, Ca, "mo_int2aa"; description="tmp")
  int2bb = transform_int2(EC, aoint2, Cb, Cb, Cb, Cb, "mo_int2bb"; description="tmp")
  int2ab = transform_int2_Q(EC, aoint2, Ca, Cb, Ca, Cb, "mo_int2ab"; description="tmp")
  close(aofile)
  nelec = EC.options.wf.nelec < 0 ? guess_nelec(EC.system) : EC.options.wf.nelec
  ms2 = EC.options.wf.ms2 < 0 ? mod(nelec, 2) : EC.options.wf.ms2
  fd = FDump{T,3}(nout, nelec; ms2, uhf=true)
  fd.int2aa = int2aa; fd.int2bb = int2bb; fd.int2ab = int2ab
  fd.int1a = Matrix{T}(Ca' * hAO * Ca)
  fd.int1b = Matrix{T}(Cb' * hAO * Cb)
  fd.int0 = nuclear_repulsion(EC.system)
  EC.fd = fd
  return fd
end

end #module
