module DfDump
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.FciDump

export dfdump

function generate_basis(ms::MSys)
  # TODO: use element-specific basis!
  aobasis = lowercase(ms.atoms[1].basis["ao"].name)
  mp2fit = lowercase(ms.atoms[1].basis["mp2fit"].name)
  bao = BasisSet(aobasis,genxyz(ms,bohr=false))
  bfit = BasisSet(mp2fit,genxyz(ms,bohr=false))
  return bao,bfit
end

function generate_integrals(ms::MSys, EC::ECInfo, cMO)
  bao,bfit = generate_basis(ms)
  hAO = kinetic(bao) + nuclear(bao)
  hMO = cMO' * hAO * cMO

  PQ = ERI_2e2c(bfit)
  CPQ=cholesky(PQ, RowMaximum(), check = false, tol = EC.choltol)
  if CPQ.rank < size(PQ,1)
    redund = size(PQ,1) - CPQ.rank
    println("$redund DF vectors removed using Cholesky decomposition")
  end
  # (P|Q)^-1 = (P|Q)^-1 L ((P|Q)^-1 L)† = M M†
  # (P|Q) = L L†
  # LL† M = L
  Lp=CPQ.L[invperm(CPQ.p),1:CPQ.rank]
  M = CPQ \ Lp
  CPQ = nothing
  Lp = nothing
  μνP = ERI_2e3c(bao,bfit)
  @tensoropt μνL[p,q,L] := μνP[p,q,P] * M[P,L]
  μνP = nothing
  M = nothing
  @tensoropt pqL[p,q,L] := cMO[μ,p] * μνL[μ,ν,L] * cMO[ν,q]
  μνL = nothing
  # <pr|qs> = sum_L pqL[p,q,L] * pqL[r,s,L]
  prqs = zeros(size(cMO,1),size(cMO,1),(size(cMO,1)+1)*size(cMO,1)÷2)
  for s = 1:size(pqL,1), q = 1:s # only upper triangle
    I = uppertriangular(q,s)
    @tensoropt prqs[:,:,I][p,r] = pqL[:,q,:][p,L] * pqL[:,s,:][r,L]
  end
  Enuc = nuclear_repulsion(ms)
end

""" generate fcidump using df integrals """
function dfdump(ms::MSys, EC::ECInfo, cMO)

end

end