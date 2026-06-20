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
using ..ElemCo.TensorTools
using ..ElemCo.FciDumps
using ..ElemCo.OrbTools

export generate_AO_DF_integrals, generate_DF_integrals, generate_DF_Fock
export generate_3idx_integrals, contract_df_integrals!, transform_3idx!
export calc_system_df_integrals
export generate_ao_fdump, ao_integrals, ensure_ao_integrals!
export ao_to_mo!, set_mo_basis!, set_ao_basis!, mmap_int2_allocator

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
    generate_ao_fdump(EC::ECInfo) -> FDump

  Build a non-orthogonal AO-basis [`FDump`](@ref) (`TFDump`) holding the exact
  (non-density-fitted) 2-e integrals `(μν|ρσ)` in physicists' notation, the AO core
  Hamiltonian `h_{μν} = T_{μν} + V_{μν}`, the nuclear repulsion energy, and the AO
  overlap `S_{μν}` (stored in `fd.overlap`, `AOBASIS` flag set).

  The 2-e integrals are assembled batch-wise straight into a memory-mapped triangular
  `int2` (see [`eri_2e4idx_tri!`](@ref)); the full `nao⁴` tensor is never materialized.
"""
function generate_ao_fdump(EC::ECInfo{T}) where T
  @assert !isempty(EC.system) "EC.system is not set up!"
  bao = generate_basis(EC, "ao")
  S = overlap(bao)
  nao = size(S, 1)
  hAO = kinetic(bao) + nuclear(bao)
  # Store the AO core Hamiltonian and overlap under the standard "h_AA"/"S_AA" keys (as the DF
  # path does), so the SCF starting-orbital guesses — `guess_hcore`/`guess_sad`, which read them
  # via `load(EC, "h_AA"/"S_AA")` — work for the non-DF AO-FDump path too.
  save!(EC, "h_AA", Matrix{T}(hAO))
  save!(EC, "S_AA", Matrix{T}(S))
  # Stream the exact AO integrals straight into a memory-mapped triangular
  # `int2[p,q,tri(r,s)] = ⟨pq|rs⟩` — never materializing the dense `nao⁴` tensor.
  ntri = nao*(nao+1)÷2
  int2_file, int2 = newmmap(EC, "ao_int2", (nao, nao, ntri), T; description="int2 ao")
  eri_2e4idx_tri!(int2, bao)
  flushmmap(EC, int2)
  Enuc = nuclear_repulsion(EC.system)
  # Store the full (neutral) electron count and a parity-consistent ms2, following the
  # FCIDUMP convention: `charge` and `ms2` from the wf options are applied later by
  # `setup_space_fd!` (which subtracts `charge` from the stored NELEC).
  nelec = EC.options.wf.nelec < 0 ? guess_nelec(EC.system) : EC.options.wf.nelec
  ms2 = mod(nelec, 2)
  return make_ao_fdump(int2, Matrix{T}(hAO), Enuc, Matrix{T}(S), nelec; ms2)
end

"""
    ao_integrals(EC::ECInfo) -> FDump

  Build the exact (non-density-fitted) AO-basis integral dump (see
  [`generate_ao_fdump`](@ref)) and store it in `EC.fd`. This is the non-DF analogue
  of `dfdump`; it is the entry point behind the `@ints` macro and is called
  automatically by `@hf`/`@uhf`/`@cc` when `EC.fd` does not already hold AO integrals.
"""
function ao_integrals(EC::ECInfo)
  EC.fd = generate_ao_fdump(EC)
  return EC.fd
end

"""
    ensure_ao_integrals!(EC::ECInfo; method="@hf", alternative="@bohf") -> FDump

  Make sure `EC.fd` holds exact AO integrals for the current system, (re)generating them
  with [`ao_integrals`](@ref) when it is empty or does not already hold AO integrals.

  If `EC.fd` currently holds **non-AO** integrals (e.g. an FCIDUMP / MO dump), they are
  **discarded** and a warning is issued: `method` (`@hf`/`@uhf`) always builds HF from
  freshly computed exact AO integrals, never from a loaded FCIDUMP. To run HF directly on
  existing FCIDUMP integrals, use `alternative` (`@bohf`/`@bouhf`) instead.
"""
function ensure_ao_integrals!(EC::ECInfo; method="@hf", alternative="@bohf")
  if isempty(EC.fd)
    ao_integrals(EC)
  elseif !is_ao_basis(EC.fd)
    @warn "$method discards the non-AO integrals currently in EC.fd and builds HF from " *
          "freshly generated exact AO integrals. To run HF on the existing FCIDUMP " *
          "integrals instead, use $alternative."
    ao_integrals(EC)
  end
  return EC.fd
end

"""
    ao_to_mo!(fd::FDump{T,3}, cMO::AbstractMatrix) -> FDump{T,3}

  Transform a non-orthogonal AO-basis [`FDump`](@ref) **in place** into a standard
  (orthonormal) MO-basis `TFDump` using the MO coefficients `cMO[μ,p]` (`nao × norb`).

  Exact (non-density-fitted) `O(N⁵)` four-index transformation: the AO `FDump` is a
  non-orthogonal "MO" dump, so transforming its integrals by `cMO` on both the left and
  the right yields the MO dump. Afterwards `fd` holds the MO integrals `<pq|rs>_MO`, the
  MO 1-e integrals `h_{pq} = cᵀ h_{μν} c` and the same nuclear repulsion energy; it is a
  regular MO fcidump (`ao_basis = false`, AO `overlap`/`AOBASIS` dropped) that BOHF/CC
  consume unchanged.

  `cMO` may be rectangular (`nao × nout` with `nout ≤ nao`): only those `nout` orbitals are
  kept, `NORB`/`ORBSYM` are updated accordingly, and the transform cost scales with `nout`
  (not `nao`). This is how deleted virtual orbitals are excluded — they carry
  no electrons, so no core-energy folding is needed. Frozen-core *occupied* orbitals are
  kept and frozen later by the correlated driver (`freeze_core!`).

  By default the MO `fd.int2` is allocated in memory; pass `alloc` 
  (e.g. [`mmap_int2_allocator`](@ref)) to write it straight onto a memory-mapped scratch file.
  [`set_mo_basis!`](@ref) does exactly that for the production swap.
"""
function ao_to_mo!(fd::FDump{T,3}, cMO::AbstractMatrix; alloc=dims->zeros(T, dims)) where {T<:Number}
  @assert fd.ao_basis "ao_to_mo! requires an AO-basis FDump"
  @assert !fd.uhf "open-shell AO→MO transform not yet implemented (closed-shell only)"
  @assert size(fd.overlap, 1) == size(cMO, 1) "cMO has $(size(cMO,1)) AOs but the AO-FDump has $(size(fd.overlap,1))"
  # the orbitals must be orthonormal in the AO basis for the transform to yield correct MO integrals
  @assert isapprox(cMO' * fd.overlap * cMO, I, atol=1e-8) "cMO are not orthonormal in the AO basis defined by fd.overlap"
  C = SpinMatrix(Matrix{T}(cMO))
  transform_fcidump!(fd, C, C; alloc=(key, dims)->alloc(dims))  # exact AO→MO 4-index transform
  # it is now a standard orthonormal MO fcidump — drop the AO-basis metadata and record the
  # (possibly reduced) orbital count, so passing a subset of `cMO` columns yields a smaller dump
  nout = size(cMO, 2)
  fd.head["NORB"] = [nout]
  osym = get(fd.head, "ORBSYM", Int[])
  fd.head["ORBSYM"] = length(osym) >= nout ? osym[1:nout] : ones(Int, nout)
  fd.ao_basis = false
  fd.overlap = zeros(T, 0, 0)
  fd.head["AOBASIS"] = [0]
  return fd
end

"""
    set_mo_basis!(EC::ECInfo, cMO::AbstractMatrix) -> FDump

  Switch `EC.fd` in place from the AO basis to the MO basis defined by `cMO[μ,p]`,
  overwriting the integrals exactly as `transform_fcidump!` does for a regular fcidump.

  The exact AO→MO transform is applied to `EC.fd` (see [`ao_to_mo!`](@ref)) and the
  resulting MO 2-e integrals are written **directly** onto a fresh memory-mapped scratch file
  (`"mo_int2"`, registered in `EC.files` with description `"int2 mo"`) — the transform fills the
  mmap in place, so the full `nout²·tri` MO tensor is never materialized in memory. The AO 2-e
  integrals stay on disk in their own scratch file (`"ao_int2"`, written by
  [`generate_ao_fdump`](@ref)), so the dump can be switched back with [`set_ao_basis!`](@ref).
  After this call `EC.fd` is an ordinary orthonormal MO fcidump that every correlated
  driver consumes unchanged; frozen-core / redundant-orbital handling is left to the
  driver (`freeze_core!` / `n_deleted_orbitals`).
"""
function set_mo_basis!(EC::ECInfo{T}, cMO::AbstractMatrix) where {T<:Number}
  # transform straight into the registered "mo_int2" scratch mmap (no intermediate in-memory copy)
  ao_to_mo!(EC.fd, cMO; alloc=dims->newmmap(EC, "mo_int2", dims, T; description="int2 mo")[2])
  flushmmap(EC, EC.fd.int2)
  return EC.fd
end

"""
    mmap_int2_allocator(EC::ECInfo) -> Function

  Build an output allocator for [`transform_fcidump!`](@ref) that parks each transformed 2-e
  integral block on its own fresh memory-mapped scratch file instead of an in-memory array.
  The returned closure `(key, dims) -> mmap` creates (and registers in `EC.files`) a scratch
  file named after the integral block (`key` ∈ `"int2"`, `"int2aa"`, `"int2bb"`, `"int2ab"`)
  and returns the (zero-initialized) mmaped array to be filled in place. Use it to transform
  large integrals without ever materializing the full result in memory.
"""
function mmap_int2_allocator(EC::ECInfo{T}) where {T<:Number}
  return (key, dims) -> newmmap(EC, key, dims, T; description=key)[2]
end

"""
    set_ao_basis!(EC::ECInfo) -> FDump

  Switch `EC.fd` back to the non-orthogonal AO basis, the inverse of
  [`set_mo_basis!`](@ref). The AO 2-e integrals are re-mmapped from their scratch file
  (`"ao_int2"`, written by [`generate_ao_fdump`](@ref)); the AO core Hamiltonian and
  overlap are rebuilt from the AO basis. No-op if `EC.fd` is already an AO dump.
"""
function set_ao_basis!(EC::ECInfo{T}) where {T<:Number}
  fd = EC.fd
  is_ao_basis(fd) && return fd
  @assert haskey(EC.files, "ao_int2") "no AO integrals on file (\"ao_int2\"); call ao_integrals(EC) first"
  _, aoint2 = mmap(EC, "ao_int2")       # dims + type recovered from the mmap header
  fd.int2 = aoint2
  bao = generate_basis(EC, "ao")
  fd.int1 = Matrix{T}(kinetic(bao) + nuclear(bao))
  fd.overlap = Matrix{T}(overlap(bao))
  fd.ao_basis = true
  fd.head["AOBASIS"] = [1]
  return fd
end

end #module
