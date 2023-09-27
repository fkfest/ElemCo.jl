""" generate fcidump using df integrals and store in dumpfile """
module DfDump
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
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

  `mp2fit` basis is used for `int2` integrals, and `jkfit` basis-correction is
  used for `int1` and `int0` integrals. 
  `full_spaces` is a dictionary with spaces without frozen orbitals.
"""
function generate_integrals(EC::ECInfo, fdump::FDump, cMO, full_spaces)
  @assert !fdump.uhf # TODO: uhf
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, "mp2fit")
  jkfit = generate_basis(EC.ms, "jkfit")
  core_orbs = setdiff(full_spaces['o'], EC.space['o'])
  wocore = setdiff(1:size(cMO,2), core_orbs)

  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thr, verbose = true)
  PQ = nothing
  μνP = ERI_2e3c(bao,bfit)
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
  Enuc = nuclear_repulsion(EC.ms)
  fdump.int0 = Enuc + hii + sum(diag(fock_jkfitMO)[core_orbs]) - sum(diag(fdump.int1)[spo])

  # reference energy
  eRef = Enuc + hii + sum(diag(fock_jkfitMO)[full_spaces['o']]) 
  println("Reference energy: ", eRef)
end

""" 
    dfdump(EC::ECInfo)

  Generate fcidump using df integrals and store in `IntOptions.fcidump`.
  If `IntOptions.fcidump` is empty, don't write to fcidump file, store in EC.fd.
"""
function dfdump(EC::ECInfo)
  println("generating integrals")
  setup_space_ms!(EC)
  dumpfile = EC.options.int.fcidump 
  if !EC.options.int.df
    error("Only density-fitted integrals implemented")
  end
  cMO = load_orbitals(EC)

  space_save = save_space(EC)
  ncore_orbs = freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc)
  nfrozvirt = freeze_nvirt!(EC, EC.options.wf.freeze_nvirt)

  nelec = guess_nelec(EC.ms) - 2*ncore_orbs
  norbs = size(cMO,2) - ncore_orbs - nfrozvirt
  fdump = FDump(norbs, nelec)
  generate_integrals(EC, fdump, cMO[:,1:end-nfrozvirt], space_save)
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
