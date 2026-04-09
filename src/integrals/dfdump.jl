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

export dfdump

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

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  println("Number of fitting functions in mpfit: ", size(PQ, 2))
  println("Number of fitting functions in mpfit after Cholesky: ", size(M, 2))
  PQ = nothing
  μνP = eri_2e3idx(bao, bfit)
  cMOval = cMO[:,wocore]
  nao = size(cMO, 1)
  norbs = length(wocore)
  println("norbs: ", norbs)
  filename2 = int2_npy_filename(fdump)
  int2_file, int2 = newmmap(EC, filename2, (norbs,norbs,(norbs+1)*norbs÷2), description="int2")
  nL = size(M,2)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(max(nao, norbs)^2*maxL+norbs*nao*maxL) begin
  first = true
  for L in LBlks
    lenL = length(L)
    v!M = @mview M[:,L]
    mAL = alloc!(buf, norbs, nao, lenL)
    AAL = alloc!(buf, nao, nao, lenL)
    @mtensor AAL[p,q,L] = μνP[p,q,P] * v!M[P,L]
    @mtensor mAL[p,ν,L] = cMOval[μ,p] * AAL[μ,ν,L]
    drop!(buf, AAL)
    Lmm = alloc!(buf, lenL, norbs, norbs)
    @mtensor Lmm[L,p,q] = mAL[p,ν,L] * cMOval[ν,q]
    # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
    if first
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        v!Lmm_q = @mview Lmm[:,:,q]
        v!Lmm_s = @mview Lmm[:,:,s]
        v!int2 = @mview int2[:,:,Iq]
        @mtensor v!int2[p,r,q] = v!Lmm_q[L,p,q] * v!Lmm_s[L,r]
      end
    else
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        v!Lmm_q = @mview Lmm[:,:,q]
        v!Lmm_s = @mview Lmm[:,:,s]
        v!int2 = @mview int2[:,:,Iq]
        @mtensor v!int2[p,r,q] += v!Lmm_q[L,p,q] * v!Lmm_s[L,r]
      end
    end
    drop!(buf, Lmm, mAL)
    first = false
  end
  end #buffer
  μνP = nothing
  M = nothing
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
    generate_integrals(EC::ECInfo, fdump::FDump{T,3}, cMO::Matrix, full_spaces)

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

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  println("Number of fitting functions in mpfit: ", size(PQ, 2))
  println("Number of fitting functions in mpfit after Cholesky: ", size(M, 2))
  PQ = nothing
  μνP = eri_2e3idx(bao, bfit)
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
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  # Buffer size needs space for: Lmm, Lmm_p, AAL, mAL, mAL_p
  @buffer buf(2*maxL*norbs^2 + nao^2*maxL + 2*norbs*nao*maxL) begin
  first = true
  for L in LBlks
    lenL = length(L)
    v!M = @mview M[:,L]
    Lmm = alloc!(buf, lenL, norbs, norbs)
    Lmm_p = alloc!(buf, lenL, norbs, norbs)
    AAL = alloc!(buf, nao, nao, lenL)
    mAL = alloc!(buf, norbs, nao, lenL)
    mAL_p = alloc!(buf, norbs, nao, lenL)
    @mtensor AAL[p,q,L] = μνP[p,q,P] * v!M[P,L]
    @mtensor mAL[p,ν,L] = cMOval[μ,p] * AAL[μ,ν,L]
    @mtensor Lmm[L,p,q] = mAL[p,ν,L] * cMOval[ν,q]
    @mtensor mAL_p[p,ν,L] = cPOval[μ,p] * AAL[μ,ν,L]
    @mtensor Lmm_p[L,p,q] = mAL_p[p,ν,L] * cPOval[ν,q]
    drop!(buf, AAL, mAL, mAL_p)
    # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
    if first
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        Lmm_q = @mview Lmm[:,:,q]
        Lmm_s = @mview Lmm[:,:,s]
        Lmm_p_s = @mview Lmm_p[:,:,s]
        @mtensor begin
          int2[:,:,Iq][p,r,q] = Lmm_q[L,p,q] * Lmm_s[L,r]
          int2ep[:,:,:,s][p,r,q] = Lmm[L,p,q] * Lmm_p_s[L,r]
        end
      end
    else
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        Lmm_q = @mview Lmm[:,:,q]
        Lmm_s = @mview Lmm[:,:,s]
        Lmm_p_s = @mview Lmm_p[:,:,s]
        @mtensor begin
          int2[:,:,Iq][p,r,q] += Lmm_q[L,p,q] * Lmm_s[L,r]
          int2ep[:,:,:,s][p,r,q] += Lmm[L,p,q] * Lmm_p_s[L,r]
        end
      end
    end
    drop!(buf, Lmm, Lmm_p)
    first = false
  end
  end #buffer
  μνP = nothing
  M = nothing
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
    generate_integrals(EC::ECInfo, fdump::TFDump, cMO::Matrix, full_spaces)

  Generate `int2`, `int1` and `int0` integrals for fcidump using density fitting.
  Generate both e-e, e-p and the 1-body e and p integrals.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::TFDump, cMO::Matrix, cPO::Matrix, full_spaces)
  @assert !fdump.uhf "Use generate_integrals(EC, fdump, cMO::SpinMatrix, full_spaces) for UHF"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  println("Number of fitting functions in mpfit: ", size(PQ, 2))
  println("Number of fitting functions in mpfit after Cholesky: ", size(M, 2))
  PQ = nothing
  μνP = eri_2e3idx(bao, bfit)
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
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  # Buffer size needs space for: Lmm, Lmm_p, AAL, mAL, mAL_p
  @buffer buf(2*maxL*norbs^2 + nao^2*maxL + 2*norbs*nao*maxL) begin
  first = true
  for L in LBlks
    lenL = length(L)
    v!M = @mview M[:,L]
    Lmm = alloc!(buf, lenL, norbs, norbs)
    Lmm_p = alloc!(buf, lenL, norbs, norbs)
    AAL = alloc!(buf, nao, nao, lenL)
    mAL = alloc!(buf, norbs, nao, lenL)
    mAL_p = alloc!(buf, norbs, nao, lenL)
    @mtensor AAL[p,q,L] = μνP[p,q,P] * v!M[P,L]
    @mtensor mAL[p,ν,L] = cMOval[μ,p] * AAL[μ,ν,L]
    @mtensor Lmm[L,p,q] = mAL[p,ν,L] * cMOval[ν,q]
    @mtensor mAL_p[p,ν,L] = cPOval[μ,p] * AAL[μ,ν,L]
    @mtensor Lmm_p[L,p,q] = mAL_p[p,ν,L] * cPOval[ν,q]
    drop!(buf, AAL, mAL, mAL_p)
    # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
    if first
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        Lmm_q = @mview Lmm[:,:,q]
        Lmm_s = @mview Lmm[:,:,s]
        Lmm_p_s = @mview Lmm_p[:,:,s]
        @mtensor begin
          int2[:,:,Iq][p,r,q] = Lmm_q[L,p,q] * Lmm_s[L,r]
          int2ep[:,:,:,s][p,r,q] = Lmm[L,p,q] * Lmm_p_s[L,r]
        end
      end
    else
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        Lmm_q = @mview Lmm[:,:,q]
        Lmm_s = @mview Lmm[:,:,s]
        Lmm_p_s = @mview Lmm_p[:,:,s]
        @mtensor begin
          int2[:,:,Iq][p,r,q] += Lmm_q[L,p,q] * Lmm_s[L,r]
          int2ep[:,:,:,s][p,r,q] += Lmm[L,p,q] * Lmm_p_s[L,r]
        end
      end
    end
    drop!(buf, Lmm, Lmm_p)
    first = false
  end
  end #buffer
  μνP = nothing
  M = nothing
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

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  PQ = nothing
  μνP = eri_2e3idx(bao, bfit)
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
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf((nao^2 + norbs*nao + 2*norbs^2)*maxL) begin
  first = true
  for L in LBlks
    lenL = length(L)
    v!M = @mview M[:,L]
    Lmm = alloc!(buf, lenL, norbs, norbs)
    LMM = alloc!(buf, lenL, norbs, norbs)
    AAL = alloc!(buf, nao, nao, lenL)
    MAL = mAL = alloc!(buf, norbs, nao, lenL)
    @mtensor AAL[p,q,L] = μνP[p,q,P] * v!M[P,L]
    @mtensor mAL[p,ν,L] = cMOaval[μ,p] * AAL[μ,ν,L]
    @mtensor Lmm[L,p,q] = mAL[p,ν,L] * cMOaval[ν,q]
    @mtensor MAL[p,ν,L] = cMObval[μ,p] * AAL[μ,ν,L]
    @mtensor LMM[L,p,q] = MAL[p,ν,L] * cMObval[ν,q]
    drop!(buf, AAL, mAL)
    # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
    if first
      for s = 1:norbs
        q = 1:s # only upper triangle
        v!LMM_s = @mview LMM[:,:,s]
        v!LMM_q = @mview LMM[:,:,q]
        v!Lmm_s = @mview Lmm[:,:,s]
        v!Lmm_q = @mview Lmm[:,:,q]
        Iq = uppertriangular_range(s)
        v!int2ab = @mview int2ab[:,:,:,s]
        v!int2aa = @mview int2aa[:,:,Iq]
        v!int2bb = @mview int2bb[:,:,Iq]
        @mtensor begin
          v!int2ab[p,r,q] = Lmm[L,p,q] * v!LMM_s[L,r]
          v!int2aa[p,r,q] = v!Lmm_q[L,p,q] * v!Lmm_s[L,r]
          v!int2bb[p,r,q] = v!LMM_q[L,p,q] * v!LMM_s[L,r]
        end
      end
    else
      for s = 1:norbs
        q = 1:s # only upper triangle
        v!LMM_s = @mview LMM[:,:,s]
        v!LMM_q = @mview LMM[:,:,q]
        v!Lmm_s = @mview Lmm[:,:,s]
        v!Lmm_q = @mview Lmm[:,:,q]
        Iq = uppertriangular_range(s)
        v!int2ab = @mview int2ab[:,:,:,s]
        v!int2aa = @mview int2aa[:,:,Iq]
        v!int2bb = @mview int2bb[:,:,Iq]
        @mtensor begin
          v!int2ab[p,r,q] += Lmm[L,p,q] * v!LMM_s[L,r]
          v!int2aa[p,r,q] += v!Lmm_q[L,p,q] * v!Lmm_s[L,r]
          v!int2bb[p,r,q] += v!LMM_q[L,p,q] * v!LMM_s[L,r]
        end
      end
    end
    drop!(buf, Lmm, LMM)
    first = false
  end
  end #buffer
  μνP = nothing
  M = nothing
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
  if EC.options.wf.npositron > 0
    cMO = load_orbitals(EC)
    cPO = load_positron_orbitals(EC)
    norbs_pos = size(cPO,2)
  else
    cMO = load_orbitals(EC)
  end
  norbs = size(cMO,2)
  space_save = save_space(EC)
  ncore_orbs = freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc)
  nfrozvirt = freeze_nvirt!(EC, EC.options.wf.freeze_nvirt)

  nelec = guess_nelec(EC.system) - 2*ncore_orbs
  npos = EC.options.wf.npositron
  norbs -= ncore_orbs + nfrozvirt
  ms2 = EC.options.wf.ms2
  ms2 = (ms2 < 0) ? mod(nelec,2) : ms2
  fdump = FDump{ec_eltype(EC),3}(norbs, nelec; ms2=ms2, uhf=!is_restricted(cMO), npos=npos)
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
