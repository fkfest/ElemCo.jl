""" generate fcidump using df integrals and store in dumpfile """
module DfDump
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.FciDump
using ..ElemCo.TensorTools
using ..ElemCo.DFTools

export dfdump

"""
    generate_integrals(EC::ECInfo, fdump::FDump, cMO)

  Generate `int2`, `int1` and `int0` integrals for fcidump.
"""
function generate_integrals(EC::ECInfo, fdump::FDump, cMO)
  @assert !fdump.uhf # TODO: uhf
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, "mp2fit")
  hAO = kinetic(bao) + nuclear(bao)
  fdump.int1 = cMO' * hAO * cMO

  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thr, verbose = true)
  μνP = ERI_2e3c(bao,bfit)
  @tensoropt μνL[p,q,L] := μνP[p,q,P] * M[P,L]
  μνP = nothing
  M = nothing
  @tensoropt pqL[p,q,L] := cMO[μ,p] * μνL[μ,ν,L] * cMO[ν,q]
  μνL = nothing
  @assert fdump.triang # store only upper triangle
  # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
  fdump.int2 = zeros(size(cMO,1),size(cMO,1),(size(cMO,1)+1)*size(cMO,1)÷2)
  Threads.@threads for s = 1:size(pqL,1)
    q = 1:s # only upper triangle
    Iq = uppertriangular_range(s)
    @tensoropt fdump.int2[:,:,Iq][p,r,q] = pqL[:,q,:][p,q,L] * pqL[:,s,:][r,L]
  end
  fdump.int0 = nuclear_repulsion(EC.ms)
end

""" 
    dfdump(EC::ECInfo, cMO, dumpfile = "FCIDUMP")

  Generate fcidump using df integrals and store in dumpfile.
  If dumpfile is empty, don't write to fcidump file, store in EC.fd.
"""
function dfdump(EC::ECInfo, cMO, dumpfile = "FCIDUMP")
  println("generating fcidump $dumpfile")
  nelec = guess_nelec(EC.ms)
  fdump = FDump(size(cMO,2), nelec)
  generate_integrals(EC, fdump, cMO)
  if length(dumpfile) > 0
    println("writing fcidump $dumpfile")
    write_fcidump(fdump, dumpfile, -1.0)  
  else
    EC.fd = fdump
  end
end

end
