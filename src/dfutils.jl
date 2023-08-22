"""
This module contains various utils for density fitting.
"""
module DFUtils
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem

export generate_basis

"""
    generate_basis(ms::MSys, type = "ao")

  Generate basis sets for integral calculations.
  `type` can be `"ao"`, `"mp2fit"` or `"jkfit"`.
"""
function generate_basis(ms::MSys, type = "ao")
  # TODO: use element-specific basis!
  basis_name = lowercase(ms.atoms[1].basis[type].name)
  basis = BasisSet(basis_name,genxyz(ms,bohr=false))
  return basis
end

"""
    generate_3idx_integrals(EC::ECInfo, cMO)

  Generate ``(pq|P)``, ``f_{pq}`` and `E_{nuc}` with
  ``(pq|rs) = (pq|P) (P|rs)``.
"""
function generate_3idx_integrals(EC::ECInfo, cMO)
  @assert ndims(cMO) == 2 # TODO: uhf
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, "mp2fit")
  bjkfit = generate_basis(EC.ms, "jkfit")
  hAO = kinetic(bao) + nuclear(bao)
  hMO = cMO' * hAO * cMO

  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thr, verbose = true)
  μνP = ERI_2e3c(bao,bfit)
  @tensoropt μνL[p,q,L] := μνP[p,q,P] * M[P,L]
  μνP = nothing
  M = nothing
  @tensoropt pqL[p,q,L] := cMO[μ,p] * μνL[μ,ν,L] * cMO[ν,q]
  μνL = nothing
  save(EC,"pqP",pqL)
  return nuclear_repulsion(EC.ms)
end

"""
    generate_integrals(EC::ECInfo, fdump::FDump, cMO)

  Generate `int2`, `int1` and `int0` integrals for fcidump.
"""
end #module