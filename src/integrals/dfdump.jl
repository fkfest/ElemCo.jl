""" generate fcidump using df integrals and store in dumpfile """
module DfDump
using LinearAlgebra
using Buffers
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.Integrals
using ..ElemCo.OrbTools
using ..ElemCo.MSystems
using ..ElemCo.FockFactory
using ..ElemCo.FciDumps
using ..ElemCo.TensorTools
using ..ElemCo.DFTools
using ..ElemCo.Utils

export dfdump, setup_fcidump_if_needed!

"""
    prepare_mpfit(EC, bao, bfit)

Compute fitting matrix `M` and 3-index AO integrals `μνP` for density fitting.
"""
function prepare_mpfit(EC, bao, bfit)
  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  println("Number of fitting functions in mpfit: ", size(PQ, 2))
  println("Number of fitting functions in mpfit after Cholesky: ", size(M, 2))
  PQ = nothing
  μνP = eri_2e3idx(bao, bfit)
  return μνP, M
end

"""    
    _halftransform_mo_first!(B3_coeff_pairs, μνP, M)

MO-first half-transform: transform AO→MO first, then P→L.
More efficient when MO space is much smaller than AO space.
"""
function _halftransform_mo_first!(B3_coeff_pairs, μνP, M)
  nao = size(μνP, 1)
  nP = size(μνP, 3)
  max_norbs = maximum(size(pair[2], 2) for pair in B3_coeff_pairs)
  PBlks = get_spaceblocks(1:nP, 512)
  maxP = maximum(length, PBlks)
  tM = copy(transpose(M))
  @buffer buf(max_norbs * (nao + max_norbs) * maxP) begin
  for (B3, c) in B3_coeff_pairs
    norbs = size(c, 2)
    first = true
    for PBlk in PBlks
      lenP = length(PBlk)
      mmP = alloc!(buf, norbs, norbs, lenP)
      mνP = alloc!(buf, norbs, nao, lenP)
      v!μνP = @mview μνP[:,:,PBlk]
      @mtensor mνP[p,ν,P] = conj(c[μ,p]) * v!μνP[μ,ν,P]
      @mtensor mmP[p,q,P] = mνP[p,ν,P] * c[ν,q]
      drop!(buf, mνP)
      v!tM = @mview tM[:,PBlk]
      if first
        @mtensor B3[L,p,q] = mmP[p,q,P] * v!tM[L,P]
        first = false
      else
        @mtensor B3[L,p,q] += mmP[p,q,P] * v!tM[L,P]
      end
      drop!(buf, mmP)
    end
  end
  end #buffer
end

"""    _halftransform_df_first!(B3_coeff_pairs, μνP, M)

DF-first half-transform: transform P→L first, then AO→MO.
More efficient when MO space is comparable to AO space.
"""
function _halftransform_df_first!(B3_coeff_pairs, μνP, M)
  nao = size(μνP, 1)
  nL = size(M, 2)
  # check that all coeffs have the same number of orbitals
  norbs = size(B3_coeff_pairs[1][2], 2)
  @assert all(pair -> size(pair[2], 2) == norbs, B3_coeff_pairs) "All coefficient matrices must have the same number of orbitals"
  LBlks = get_spaceblocks(1:nL, 512)
  maxL = maximum(length, LBlks)
  @buffer buf((nao^2 + norbs*nao)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!M = @mview M[:,L]
    AAL = alloc!(buf, nao, nao, lenL)
    mAL = alloc!(buf, norbs, nao, lenL)
    @mtensor AAL[p,q,L] = μνP[p,q,P] * v!M[P,L]
    for (B3, c) in B3_coeff_pairs
      @mtensor mAL[p,ν,L] = c[μ,p] * AAL[μ,ν,L]
      v!B3 = @mview B3[L,:,:]
      @mtensor v!B3[L,p,q] = mAL[p,ν,L] * c[ν,q]
    end
    drop!(buf, mAL, AAL)
  end
  end #buffer
end

"""
    halftransform_3idx!(B3_coeff_pairs, μνP, M)

Half-transform AO 3-index integrals to MO basis.
For each `(B3, c)` pair, computes ``B3_{L,p,q} = \\sum_{\\mu\\nu} c_{\\mu,p} (\\mu\\nu P \\cdot M)_{\\mu,\\nu,L} c_{\\nu,q}``.
`B3` arrays must be pre-allocated as `Array{T,3}(undef, nL, norbs, norbs)`.

Uses MO-first route (AO→MO then P→L) when all MO sizes are at most half of AO size,
otherwise uses AO-first route (P→L then AO→MO).
"""
function halftransform_3idx!(B3_coeff_pairs, μνP, M)
  nao = size(μνP, 1)
  if all(pair -> 2 * size(pair[2], 2) <= nao, B3_coeff_pairs)
    _halftransform_mo_first!(B3_coeff_pairs, μνP, M)
  else
    _halftransform_df_first!(B3_coeff_pairs, μνP, M)
  end
end

"""
    contract_tri!(int2, B3)

Contract 3-index `B3[L,p,q]` into upper-triangular 4-index integrals:
``int2_{p,r,\\text{tri}(q,s)} = \\sum_L B3_{L,p,q} B3_{L,r,s}`` for ``q \\le s``.
"""
function contract_tri!(int2, B3)
  norbs = size(B3, 3)
  for s = 1:norbs
    B3s = @view B3[:,:,s]
    for q = 1:s
      B3q = @view B3[:,:,q]
      idx = uppertriangular_index(q, s)
      @views mul!(int2[:,:,idx], transpose(B3q), B3s)
    end
  end
end

"""
    contract_full!(int2, B3a, B3b)

Contract two 3-index arrays into full 4-index integrals:
``int2_{p,r,q,s} = \\sum_L B3a_{L,p,q} B3b_{L,r,s}``.
"""
function contract_full!(int2, B3a, B3b)
  norbs_q = size(B3a, 3)
  norbs_s = size(B3b, 3)
  for s = 1:norbs_s
    B3bs = @view B3b[:,:,s]
    for q = 1:norbs_q
      B3aq = @view B3a[:,:,q]
      @views mul!(int2[:,:,q,s], transpose(B3aq), B3bs)
    end
  end
end

"""
    generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::Matrix, full_spaces) where T

  Generate `int2`, `int1` and `int0` integrals for fcidump using density fitting.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::Matrix, full_spaces) where T
  @assert !fdump.uhf "Use generate_integrals(EC, fdump, cMO::SpinMatrix, full_spaces) for UHF"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")
  core_orbs = setdiff(full_spaces['o'], EC.space['o'])
  wocore = setdiff(1:size(cMO,2), core_orbs)

  μνP, M = prepare_mpfit(EC, bao, bfit)
  cMOval = cMO[:,wocore]
  nao = size(cMO, 1)
  norbs = length(wocore)
  println("norbs: ", norbs)
  filename2 = int2_npy_filename(fdump)
  int2_file, int2 = newmmap(EC, filename2, (norbs,norbs,(norbs+1)*norbs÷2), description="int2")
  nL = size(M,2)
  Lmm = Array{T,3}(undef, nL, norbs, norbs)
  halftransform_3idx!(((Lmm, cMOval),), μνP, M)
  μνP = nothing
  M = nothing
  contract_tri!(int2, Lmm)
  Lmm = nothing
  flushmmap(EC, int2)
  fdump.int2 = int2

  hAO = kinetic(bao) + nuclear(bao)
  cMO2 = cMO[:,full_spaces['o']]
  @mtensor hii = (cMO2[μ,i] * hAO[μ,ν]) * cMO2[ν,i]
  # fock matrix from fdump.int2
  ncore_orbs = length(core_orbs)
  spm = 1:norbs
  @assert core_orbs == 1:ncore_orbs "Only simple 1:ncore_orbs core orbitals implemented"
  spo = EC.space['o'] .- ncore_orbs
  @mtensor begin 
    fock[p,q] := 2.0*detri_int2(int2, norbs, spm, spo, spm, spo)[p,i,q,i] 
    fock[p,q] -= detri_int2(int2, norbs, spm, spo, spo, spm)[p,i,i,q]
  end
  space_save = save_space(EC)
  restore_space!(EC, full_spaces)
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  fock_jkfit = gen_dffock(EC, cMO, bao, jkfit)
  restore_space!(EC, space_save)
  fock_jkfitMO = cMO' * fock_jkfit * cMO
  filename1 = int1_npy_filename(fdump)
  int1_file, int1 = newmmap(EC, filename1, (norbs,norbs), description="int1")
  int1 .= fock_jkfitMO[wocore,wocore] - fock
  flushmmap(EC, int1)
  fdump.int1 = int1
  fdump.int0 = Enuc + hii + sum(diag(fock_jkfitMO)[core_orbs]) - sum(diag(int1)[spo])

  # reference energy
  eRef = Enuc + hii + sum(diag(fock_jkfitMO)[full_spaces['o']]) 
  println("Reference energy: ", eRef)
end

"""
    generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::Matrix, cPO::Matrix, full_spaces)

  Generate `int2`, `int1` and `int0` integrals for fcidump using density fitting.
  Generate both e-e, e-p and the 1-body e and p integrals.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::Matrix, cPO::Matrix, full_spaces) where T
  @assert !fdump.uhf "Use generate_integrals(EC, fdump, cMO::SpinMatrix, full_spaces) for UHF"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")

  μνP, M = prepare_mpfit(EC, bao, bfit)
  cMOval = cMO[:,1:end]
  cPOval = cPO[:,1:end]
  nao = size(cMO, 1)
  norbs = nao
  println("norbs: ", norbs)
  filename2 = int2_npy_filename(fdump)
  int2_file, int2 = newmmap(EC, filename2, (norbs,norbs,(norbs+1)*norbs÷2), description="int2")
  filename2ep = int2_npy_filename(fdump, :ep)
  int2ep_file, int2ep = newmmap(EC, filename2ep, (norbs,norbs,norbs,norbs), description="int2ep")
  nL = size(M,2)
  Lmm = Array{T,3}(undef, nL, norbs, norbs)
  Lmm_p = Array{T,3}(undef, nL, norbs, norbs)
  halftransform_3idx!(((Lmm, cMOval), (Lmm_p, cPOval)), μνP, M)
  μνP = nothing
  M = nothing
  contract_tri!(int2, Lmm)
  contract_full!(int2ep, Lmm, Lmm_p)
  Lmm = nothing
  Lmm_p = nothing
  flushmmap(EC, int2)
  flushmmap(EC, int2ep)
  fdump.int2 = int2
  fdump.int2ep = int2ep

  hAO = kinetic(bao) + nuclear(bao)
  hAO_p = kinetic(bao) - nuclear(bao)
  cMO2 = cMO[:,full_spaces['o']]
  @mtensor hii = (cMO2[μ,i] * hAO[μ,ν]) * cMO2[ν,i]
  cPO2 = cPO[:,full_spaces['p']]
  @mtensor hii_p = (cPO2[μ,i] * hAO_p[μ,ν]) * cPO2[ν,i]

  spm = 1:norbs
  spo = EC.space['o']
  sp_pos = EC.space['p']
  @mtensor begin 
    fock[p,q] := 2.0*detri_int2(int2, norbs, spm, spo, spm, spo)[p,i,q,i]
    fockjp[p,q] := int2ep[spm,sp_pos,spm,sp_pos][p,I,q,I] 
    fock[p,q] -= fockjp[p,q]
    fock[p,q] -= detri_int2(int2, norbs, spm, spo, spo, spm)[p,i,i,q]
    fock_p[p,q] := -2.0*int2ep[spo, spm, spo, spm][i,p,i,q]
  end
  space_save = save_space(EC)
  restore_space!(EC, full_spaces)
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  fock_jkfit, fock_jkfit_pos, jp = gen_dffock(EC, cMO, cPO, bao, jkfit)
  restore_space!(EC, space_save)
  fock_jkfitMO = cMO' * fock_jkfit  * cMO
  fock_jkfitMO_pos = cPO' * fock_jkfit_pos * cPO
  jpMO = cMO' * jp * cMO
  filename1 = int1_npy_filename(fdump)
  int1_file, int1 = newmmap(EC, filename1, (norbs,norbs), description="int1")
  int1 .= fock_jkfitMO - fock
  filename1p = int1_npy_filename(fdump, :p)
  int1p_file, int1p = newmmap(EC, filename1p, (norbs,norbs), description="int1p")
  int1p .= fock_jkfitMO_pos - fock_p
  flushmmap(EC, int1)
  flushmmap(EC, int1p)
  fdump.int1 = int1
  fdump.int1p = int1p
  int0 = Enuc + hii + .5*hii_p - sum(diag(int1)[spo]) - .5*sum(diag(int1p)[sp_pos])
  fdump.int0 = int0

  eRef_hii = Enuc+ hii

  eRef_fock = sum(diag(fock_jkfitMO)[full_spaces['o']])
  eRef_jp = sum(diag(jpMO)[full_spaces['o']])
  eRef_jp_pos = sum(diag(fock_jkfitMO_pos)[full_spaces['p']])

  eRef = eRef_hii + eRef_fock + eRef_jp + eRef_jp_pos

  println("Reference energy: ", eRef)

end

"""
    generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::SpinMatrix, full_spaces) where T

  Generate `int2aa`, `int2bb`, `int2ab`, `int1a`, `int1b` and `int0` integrals for fcidump using density fitting.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::SpinMatrix, full_spaces) where T
  @assert fdump.uhf "Use generate_integrals(EC, fdump, cMO, full_spaces) for RHF"
  @assert size(cMO.α) == size(cMO.β) "cMO.α and cMO.β must have the same size"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")
  core_orbs = setdiff(full_spaces['o'], EC.space['o'])
  @assert core_orbs == setdiff(full_spaces['O'], EC.space['O']) "Core space must be the same for α and β orbitals"
  wocore = setdiff(1:size(cMO, 2), core_orbs)

  μνP, M = prepare_mpfit(EC, bao, bfit)
  cMOaval = cMO[1][:,wocore]
  cMObval = cMO[2][:,wocore]
  nao = size(cMO, 1)
  norbs = length(wocore)
  println("norbs: ", norbs)
  filename2ab = int2_npy_filename(fdump, :αβ)
  int2ab_file, int2ab = newmmap(EC, filename2ab, (norbs,norbs,norbs,norbs), description="int2ab")
  filename2aa = int2_npy_filename(fdump, :α)
  int2aa_file, int2aa = newmmap(EC, filename2aa, (norbs,norbs,(norbs+1)*norbs÷2), description="int2aa")
  filename2bb = int2_npy_filename(fdump, :β)
  int2bb_file, int2bb = newmmap(EC, filename2bb, (norbs,norbs,(norbs+1)*norbs÷2), description="int2bb")
  nL = size(M,2)
  Lmm = Array{T,3}(undef, nL, norbs, norbs)
  LMM = Array{T,3}(undef, nL, norbs, norbs)
  halftransform_3idx!(((Lmm, cMOaval), (LMM, cMObval)), μνP, M)
  μνP = nothing
  M = nothing
  contract_tri!(int2aa, Lmm)
  contract_tri!(int2bb, LMM)
  contract_full!(int2ab, Lmm, LMM)
  Lmm = nothing
  LMM = nothing
  flushmmap(EC, int2ab)
  flushmmap(EC, int2aa)
  flushmmap(EC, int2bb)
  fdump.int2ab = int2ab
  fdump.int2aa = int2aa
  fdump.int2bb = int2bb
  hAO = kinetic(bao) + nuclear(bao)
  cMOao = cMO[1][:,full_spaces['o']]
  @mtensor haii = (cMOao[μ,i] * hAO[μ,ν]) * cMOao[ν,i]
  cMObo = cMO[2][:,full_spaces['O']]
  @mtensor hbii = (cMObo[μ,i] * hAO[μ,ν]) * cMObo[ν,i]
  # fock matrix from fdump.int2aa, fdump.int2bb, fdump.int2ab
  ncore_orbs = length(core_orbs)
  spm = 1:norbs
  @assert core_orbs == 1:ncore_orbs "Only simple 1:ncore_orbs core orbitals implemented"
  spo = EC.space['o'] .- ncore_orbs
  spO = EC.space['O'] .- ncore_orbs
  @mtensor begin 
    focka[p,q] := detri_int2(int2aa, norbs, spm, spo, spm, spo)[p,i,q,i] 
    focka[p,q] += int2ab[spm,spO,spm,spO][p,I,q,I] 
    focka[p,q] -= detri_int2(int2aa, norbs, spm, spo, spo, spm)[p,i,i,q]
    fockb[p,q] := detri_int2(int2bb, norbs, spm, spO, spm, spO)[p,I,q,I] 
    fockb[p,q] += int2ab[spo,spm,spo,spm][i,p,i,q] 
    fockb[p,q] -= detri_int2(int2bb, norbs, spm, spO, spO, spm)[p,I,I,q]
  end
  space_save = save_space(EC)
  restore_space!(EC, full_spaces)
  Enuc = generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  fock_jkfit = gen_dffock(EC, cMO, bao, jkfit)
  restore_space!(EC, space_save)
  fock_jkfitMOa = cMO[1]' * fock_jkfit[1] * cMO[1]
  fock_jkfitMOb = cMO[2]' * fock_jkfit[2] * cMO[2]
  filename1a = int1_npy_filename(fdump, :α)
  int1a_file, int1a = newmmap(EC, filename1a, (norbs,norbs), description="int1a")
  filename1b = int1_npy_filename(fdump, :β)
  int1b_file, int1b = newmmap(EC, filename1b, (norbs,norbs), description="int1b")
  int1a .= fock_jkfitMOa[wocore,wocore] - focka
  int1b .= fock_jkfitMOb[wocore,wocore] - fockb
  flushmmap(EC, int1a)
  flushmmap(EC, int1b)
  fdump.int1a = int1a
  fdump.int1b = int1b
  fdump.int0 = Enuc + 0.5*(haii + sum(diag(fock_jkfitMOa)[core_orbs]) - sum(diag(int1a)[spo]) 
                         + hbii + sum(diag(fock_jkfitMOb)[core_orbs]) - sum(diag(int1b)[spO]))

  # reference energy
  eRef = Enuc + 0.5*(haii + sum(diag(fock_jkfitMOa)[full_spaces['o']]) 
                   + hbii + sum(diag(fock_jkfitMOb)[full_spaces['O']]))
  println("Reference energy: ", eRef)
end

"""
    setup_fcidump_if_needed!(EC::ECInfo)

  (Re)generate the MO-integral FCIDUMP (`EC.fd`) with [`dfdump`](@ref) when it is missing, or when a
  same-session restart must rebuild it from the `start` orbitals.

  The latter is the `wf.dump==""` + `wf.start` reuse case: the cached `EC.fd` from an earlier call was
  built from different (e.g. pre-optimization) orbitals, so it must not be reused — otherwise the
  restarted amplitudes (in the stored, optimized basis) would be used with stale integrals and the
  calculation would re-optimize instead of resuming at the stored solution. Skipped when there is no
  molecular system (FCIDUMP-only), where the integrals come from a fixed file.
"""
function setup_fcidump_if_needed!(EC::ECInfo)
  if isempty(EC.fd) ||
     (EC.options.wf.dump == "" && EC.options.wf.start != "" && !isempty(EC.system))
    dfdump(EC)
  end
  return
end

"""
    dfdump(EC::ECInfo)

  Generate fcidump using df integrals and store in `IntOptions.fcidump`.
  If `IntOptions.fcidump` is empty, don't write to fcidump file, store in EC.fd.
"""
function dfdump(EC::ECInfo)
  println("Generating integrals")
  setup_space_system!(EC)
  dumpfile = EC.options.int.fcidump 
  if !EC.options.int.df
    error("Only density-fitted integrals implemented")
  end
  # classes describing the loaded orbital set; `nothing` means "use the dump's own classes". For a
  # basis-change restart the orbitals are completed to the full new basis and matching classes are
  # returned, so freezing operates on classes that describe the actual (completed) orbital set.
  completed_classes = nothing
  if EC.options.wf.npositron > 0
    cMO = load_orbitals(EC)
    cPO = load_positron_orbitals(EC)
    norbs_pos = size(cPO,2)
  else
    cMO, completed_classes = load_orbitals_for_correlation(EC)
  end
  norbs = size(cMO,2)
  space_save = save_space(EC)
  nocc_full = length(EC.space['o'])
  nvirt_full = length(EC.space['v'])
  EC.options.wf.npositron > 0 && n_redundant_orbitals(EC) > 0 &&
    error("Redundant (linearly-dependent) basis sets are not supported with positrons.")
  # frozen core, redundant orbitals, and (dump-deleted / explicit) virtuals, all by class/index
  cls = freeze_orbitals!(EC; classes=completed_classes)
  (cls.occ_a == cls.occ_b && cls.virt_a == cls.virt_b) ||
    error("FCIDUMP generation requires symmetric (restricted-like) freezing!")
  # total per-orbital frozen counts (chemical core + class-honored core, and frozen/deleted virt);
  # the region layout keeps frozen core at the lowest and deleted virtuals at the highest indices.
  ncore_orbs = nocc_full - length(EC.space['o'])
  nfrozvirt = nvirt_full - length(EC.space['v'])

  full_norb = length(space_save[':'])
  nelec = guess_nelec(EC.system) - 2*ncore_orbs
  npos = EC.options.wf.npositron
  norbs -= ncore_orbs + nfrozvirt
  ms2 = EC.options.wf.ms2
  ms2 = (ms2 < 0) ? mod(nelec,2) : ms2
  fdump = FDump{ec_eltype(EC),3}(norbs, nelec; ms2=ms2, uhf=!is_restricted(cMO), npos=npos)
  # if orbitals were frozen/deleted, record the full-space orbital index of each active orbital
  # (frozen core lowest, deleted virtuals highest) so user orbital lists given in the full MO space
  # can be translated to this active dump; leave empty for a non-reduced (identity) dump
  if ncore_orbs + nfrozvirt > 0
    fdump.orig_orbs = (ncore_orbs+1):(full_norb-nfrozvirt)
  end
  if fdump.uhf
    generate_integrals(EC, fdump, cMO[:,1:end-nfrozvirt], space_save)
  else
    if npos > 0
      @assert nfrozvirt == 0 "Frozen virtual orbitals not implemented for positron"
      generate_integrals(EC, fdump, cMO[1][:,1:end], 
        cPO[1][:,1:end], space_save)
    else
      generate_integrals(EC, fdump, cMO[1][:,1:end-nfrozvirt], space_save)
    end
  end
  restore_space!(EC, space_save)
  if length(dumpfile) > 0
    println("writing fcidump $dumpfile")
    write_fcidump(fdump, dumpfile; tol=-1.0)  
  else
    EC.fd = fdump
  end
  draw_endline()
  return
end

end
