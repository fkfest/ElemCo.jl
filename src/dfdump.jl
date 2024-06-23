""" generate fcidump using df integrals and store in dumpfile """
module DfDump
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.Integrals
using ..ElemCo.OrbTools
using ..ElemCo.MSystem
using ..ElemCo.FockFactory
using ..ElemCo.FciDumps
using ..ElemCo.TensorTools
using ..ElemCo.DFTools
using ..ElemCo.Utils

export dfdump

"""
    generate_integrals(EC::ECInfo, fdump::TFDump, cMO::Matrix, full_spaces)

  Generate `int2`, `int1` and `int0` integrals for fcidump using density fitting.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::TFDump, cMO::Matrix, full_spaces)
  @assert !fdump.uhf "Use generate_integrals(EC, fdump, cMO::SpinMatrix, full_spaces) for UHF"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")
  core_orbs = setdiff(full_spaces['o'], EC.space['o'])
  wocore = setdiff(1:size(cMO,2), core_orbs)

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  PQ = nothing
  μνP = eri_2e3idx(bao, bfit)
  cMOval = cMO[:,wocore]
  nao = size(cMO, 1)
  norbs = length(wocore)
  println("norbs: ", norbs)
  filename2 = int2_npy_filename(fdump)
  int2_file, int2 = newmmap(EC, filename2, (norbs,norbs,(norbs+1)*norbs÷2), description="int2")
  nL = size(M,2)
  LBlks = get_auxblks(nL)
  maxL = maximum(length, LBlks)
  bufAAL = create_buf(nao^2*maxL)
  bufmmL = create_buf(norbs^2*maxL)
  first = true
  for L in LBlks
    nL = length(L)
    V_M = @view M[:,L]
    AAL = reshape_buf(bufAAL, nao, nao, nL)
    mmL = reshape_buf(bufmmL, norbs, norbs, nL)
    @tensoropt begin
      AAL[p,q,L] = μνP[p,q,P] * V_M[P,L]
      mmL[p,q,L] = cMOval[μ,p] * AAL[μ,ν,L] * cMOval[ν,q]
    end
    # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
    if first
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        @tensoropt int2[:,:,Iq][p,r,q] = mmL[:,q,:][p,q,L] * mmL[:,s,:][r,L]
      end
    else
      for s = 1:norbs
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        @tensoropt int2[:,:,Iq][p,r,q] += mmL[:,q,:][p,q,L] * mmL[:,s,:][r,L]
      end
    end
    first = false
  end
  bufAAL = bufmmL = nothing
  μνP = nothing
  M = nothing
  flushmmap(EC, int2)
  fdump.int2 = int2

  hAO = kinetic(bao) + nuclear(bao)
  cMO2 = cMO[:,full_spaces['o']]
  @tensoropt hii = cMO2[μ,i] * hAO[μ,ν] * cMO2[ν,i]
  # fock matrix from fdump.int2
  ncore_orbs = length(core_orbs)
  spm = 1:norbs
  @assert core_orbs == 1:ncore_orbs "Only simple 1:ncore_orbs core orbitals implemented"
  spo = EC.space['o'] .- ncore_orbs
  @tensoropt begin 
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
    generate_integrals(EC::ECInfo, fdump::TFDump, cMO::SpinMatrix, full_spaces)

  Generate `int2aa`, `int2bb`, `int2ab`, `int1a`, `int1b` and `int0` integrals for fcidump using density fitting.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::TFDump, cMO::SpinMatrix, full_spaces)
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
  LBlks = get_auxblks(nL)
  maxL = maximum(length, LBlks)
  bufAAL = create_buf(nao^2*maxL)
  bufmmL = create_buf(norbs^2*maxL)
  bufMML = create_buf(norbs^2*maxL)
  first = true
  for L in LBlks
    nL = length(L)
    V_M = @view M[:,L]
    AAL = reshape_buf(bufAAL, nao, nao, nL)
    mmL = reshape_buf(bufmmL, norbs, norbs, nL)
    MML = reshape_buf(bufMML, norbs, norbs, nL)
    @tensoropt begin
      AAL[p,q,L] = μνP[p,q,P] * V_M[P,L]
      mmL[p,q,L] = cMOaval[μ,p] * AAL[μ,ν,L] * cMOaval[ν,q]
      MML[p,q,L] = cMObval[μ,p] * AAL[μ,ν,L] * cMObval[ν,q]
    end
    # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
    if first
      for s = 1:norbs
        MML_s = MML[:,s,:]
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        @tensoropt begin
          int2ab[:,:,:,s][p,r,q] = mmL[p,q,L] * MML_s[r,L]
          int2aa[:,:,Iq][p,r,q] = mmL[:,q,:][p,q,L] * mmL[:,s,:][r,L]
          int2bb[:,:,Iq][p,r,q] = MML[:,q,:][p,q,L] * MML_s[r,L]
        end
      end
    else
      for s = 1:norbs
        MML_s = MML[:,s,:]
        q = 1:s # only upper triangle
        Iq = uppertriangular_range(s)
        @tensoropt begin
          int2ab[:,:,:,s][p,r,q] += mmL[p,q,L] * MML_s[r,L]
          int2aa[:,:,Iq][p,r,q] += mmL[:,q,:][p,q,L] * mmL[:,s,:][r,L]
          int2bb[:,:,Iq][p,r,q] += MML[:,q,:][p,q,L] * MML_s[r,L]
        end
      end
    end
    first = false
  end
  bufAAL = bufmmL = bufMML = nothing
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
  @tensoropt haii = cMOao[μ,i] * hAO[μ,ν] * cMOao[ν,i]
  cMObo = cMO[2][:,full_spaces['O']]
  @tensoropt hbii = cMObo[μ,i] * hAO[μ,ν] * cMObo[ν,i]
  # fock matrix from fdump.int2aa, fdump.int2bb, fdump.int2ab
  ncore_orbs = length(core_orbs)
  spm = 1:norbs
  @assert core_orbs == 1:ncore_orbs "Only simple 1:ncore_orbs core orbitals implemented"
  spo = EC.space['o'] .- ncore_orbs
  spO = EC.space['O'] .- ncore_orbs
  @tensoropt begin 
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
  cMO = load_orbitals(EC)
  norbs = size(cMO,2)
  space_save = save_space(EC)
  ncore_orbs = freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc)
  nfrozvirt = freeze_nvirt!(EC, EC.options.wf.freeze_nvirt)

  nelec = guess_nelec(EC.system) - 2*ncore_orbs
  norbs -= ncore_orbs + nfrozvirt
  ms2 = EC.options.wf.ms2
  ms2 = (ms2 < 0) ? mod(nelec,2) : ms2
  fdump = TFDump(norbs, nelec; ms2=ms2, uhf=!is_restricted(cMO))
  if fdump.uhf
    generate_integrals(EC, fdump, cMO[:,1:end-nfrozvirt], space_save)
  else
    generate_integrals(EC, fdump, cMO[1][:,1:end-nfrozvirt], space_save)
  end
  restore_space!(EC, space_save)
  if length(dumpfile) > 0
    println("writing fcidump $dumpfile")
    write_fcidump(fdump, dumpfile, -1.0)  
  else
    EC.fd = fdump
  end
  draw_endline()
  return
end

end
