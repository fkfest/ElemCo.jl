""" generate fcidump using df integrals and store in dumpfile """
module DfDump
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.BasisSets
using ..ElemCo.Wavefunctions
using ..ElemCo.Integrals
using ..ElemCo.OrbTools
using ..ElemCo.MSystem
using ..ElemCo.FockFactory
using ..ElemCo.FciDump
using ..ElemCo.TensorTools
using ..ElemCo.DFTools
using ..ElemCo.Utils

export dfdump

"""
    generate_integrals(EC::ECInfo, fdump::FDump, cMO, full_spaces)

  Generate `int2`, `int1` and `int0` integrals for fcidump using density fitting.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::FDump, cMO, full_spaces)
  @assert !fdump.uhf "Use generate_integrals(EC, fdump, cMOa, cMOb, full_spaces) for UHF"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")
  core_orbs = setdiff(full_spaces['o'], EC.space['o'])
  wocore = setdiff(1:size(cMO,2), core_orbs)

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  PQ = nothing
  μνP = eri_2e3idx(bao,bfit)
  @tensoropt μνL[p,q,L] := μνP[p,q,P] * M[P,L]
  μνP = nothing
  M = nothing
  cMOval = cMO[:,wocore]
  @tensoropt pqL[p,q,L] := cMOval[μ,p] * μνL[μ,ν,L] * cMOval[ν,q]
  μνL = nothing
  @assert fdump.triang # store only upper triangle
  # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
  norbs = length(wocore)
  println("norbs: ", norbs)
  fdump.int2 = zeros(norbs,norbs,(norbs+1)*norbs÷2)
  Threads.@threads for s = 1:size(pqL,1)
    q = 1:s # only upper triangle
    Iq = uppertriangular_range(s)
    @tensoropt fdump.int2[:,:,Iq][p,r,q] = pqL[:,q,:][p,q,L] * pqL[:,s,:][r,L]
  end
  pqL = nothing

  hAO = kinetic(bao) + nuclear(bao)
  cMO2 = cMO[:,full_spaces['o']]
  @tensoropt hii = cMO2[μ,i] * hAO[μ,ν] * cMO2[ν,i]
  # fock matrix from fdump.int2
  ncore_orbs = length(core_orbs)
  spm = 1:norbs
  @assert core_orbs == 1:ncore_orbs "Only simple 1:ncore_orbs core orbitals implemented"
  spo = EC.space['o'] .- ncore_orbs
  @tensoropt begin 
    fock[p,q] := 2.0*detri_int2(fdump.int2, norbs, spm, spo, spm, spo)[p,i,q,i] 
    fock[p,q] -= detri_int2(fdump.int2, norbs, spm, spo, spo, spm)[p,i,i,q]
  end
  space_save = save_space(EC)
  restore_space!(EC, full_spaces)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  fock_jkfit = gen_dffock(EC, cMO, bao, jkfit)
  restore_space!(EC, space_save)
  fock_jkfitMO = cMO' * fock_jkfit * cMO
  fdump.int1 = fock_jkfitMO[wocore,wocore] - fock
  Enuc = nuclear_repulsion(EC.system)
  fdump.int0 = Enuc + hii + sum(diag(fock_jkfitMO)[core_orbs]) - sum(diag(fdump.int1)[spo])

  # reference energy
  eRef = Enuc + hii + sum(diag(fock_jkfitMO)[full_spaces['o']]) 
  println("Reference energy: ", eRef)
end

"""
    generate_integrals(EC::ECInfo, fdump::FDump, cMOa, cMOb, full_spaces)

  Generate `int2aa`, `int2bb`, `int2ab`, `int1a`, `int1b` and `int0` integrals for fcidump using density fitting.

  `mpfit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::FDump, cMOa, cMOb, full_spaces)
  @assert fdump.uhf "Use generate_integrals(EC, fdump, cMO, full_spaces) for RHF"
  @assert size(cMOa) == size(cMOb) "cMOa and cMOb must have the same size"
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "mpfit")
  jkfit = generate_basis(EC, "jkfit")
  core_orbs = setdiff(full_spaces['o'], EC.space['o'])
  @assert core_orbs == setdiff(full_spaces['O'], EC.space['O']) "Core space must be the same for α and β orbitals"
  wocore = setdiff(1:size(cMOa,2), core_orbs)

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  PQ = nothing
  μνP = eri_2e3idx(bao,bfit)
  @tensoropt μνL[p,q,L] := μνP[p,q,P] * M[P,L]
  μνP = nothing
  M = nothing

  cMOaval = cMOa[:,wocore]
  cMObval = cMOb[:,wocore]
  @tensoropt pqLa[p,q,L] := cMOaval[μ,p] * μνL[μ,ν,L] * cMOaval[ν,q]
  @tensoropt pqLb[p,q,L] := cMObval[μ,p] * μνL[μ,ν,L] * cMObval[ν,q]
  μνL = nothing
  @assert fdump.triang # store only upper triangle for same-spin integrals
  # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
  norbs = length(wocore)
  println("norbs: ", norbs)
  @tensoropt fdump.int2ab[p,r,q,s] := pqLa[p,q,L] * pqLb[r,s,L]
  fdump.int2aa = zeros(norbs,norbs,(norbs+1)*norbs÷2)
  Threads.@threads for s = 1:size(pqLa,1)
    q = 1:s # only upper triangle
    Iq = uppertriangular_range(s)
    @tensoropt fdump.int2aa[:,:,Iq][p,r,q] = pqLa[:,q,:][p,q,L] * pqLa[:,s,:][r,L]
  end
  pqLa = nothing
  fdump.int2bb = zeros(norbs,norbs,(norbs+1)*norbs÷2)
  Threads.@threads for s = 1:size(pqLb,1)
    q = 1:s # only upper triangle
    Iq = uppertriangular_range(s)
    @tensoropt fdump.int2bb[:,:,Iq][p,r,q] = pqLb[:,q,:][p,q,L] * pqLb[:,s,:][r,L]
  end
  pqLb = nothing

  hAO = kinetic(bao) + nuclear(bao)
  cMOao = cMOa[:,full_spaces['o']]
  @tensoropt haii = cMOao[μ,i] * hAO[μ,ν] * cMOao[ν,i]
  cMObo = cMOb[:,full_spaces['O']]
  @tensoropt hbii = cMObo[μ,i] * hAO[μ,ν] * cMObo[ν,i]
  # fock matrix from fdump.int2aa, fdump.int2bb, fdump.int2ab
  ncore_orbs = length(core_orbs)
  spm = 1:norbs
  @assert core_orbs == 1:ncore_orbs "Only simple 1:ncore_orbs core orbitals implemented"
  spo = EC.space['o'] .- ncore_orbs
  spO = EC.space['O'] .- ncore_orbs
  @tensoropt begin 
    focka[p,q] := detri_int2(fdump.int2aa, norbs, spm, spo, spm, spo)[p,i,q,i] 
    focka[p,q] += fdump.int2ab[spm,spO,spm,spO][p,I,q,I] 
    focka[p,q] -= detri_int2(fdump.int2aa, norbs, spm, spo, spo, spm)[p,i,i,q]
    fockb[p,q] := detri_int2(fdump.int2bb, norbs, spm, spO, spm, spO)[p,I,q,I] 
    fockb[p,q] += fdump.int2ab[spo,spm,spo,spm][i,p,i,q] 
    fockb[p,q] -= detri_int2(fdump.int2bb, norbs, spm, spO, spO, spm)[p,I,I,q]
  end
  space_save = save_space(EC)
  restore_space!(EC, full_spaces)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  fock_jkfit = gen_dffock(EC, MOs(cMOa, cMOb), bao, jkfit)
  restore_space!(EC, space_save)
  fock_jkfitMOa = cMOa' * fock_jkfit[1] * cMOa
  fock_jkfitMOb = cMOb' * fock_jkfit[2] * cMOb
  fdump.int1a = fock_jkfitMOa[wocore,wocore] - focka
  fdump.int1b = fock_jkfitMOb[wocore,wocore] - fockb
  Enuc = nuclear_repulsion(EC.system)
  fdump.int0 = Enuc + 0.5*(haii + sum(diag(fock_jkfitMOa)[core_orbs]) - sum(diag(fdump.int1a)[spo]) 
                         + hbii + sum(diag(fock_jkfitMOb)[core_orbs]) - sum(diag(fdump.int1b)[spO]))

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
  fdump = FDump(norbs, nelec; ms2=ms2, uhf=!is_restricted_MO(cMO))
  if fdump.uhf
    generate_integrals(EC, fdump, cMO[1][:,1:end-nfrozvirt], cMO[2][:,1:end-nfrozvirt], space_save)
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
