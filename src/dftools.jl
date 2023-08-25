"""
This module contains various utils for density fitting.
"""
module DFTools
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.TensorTools

export generate_AO_DF_integrals, generate_3idx_integrals

"""
    generate_AO_DF_integrals(EC::ECInfo, fitbasis="mp2fit"; save3idx=true)

  Generate AO integrals using DF + Cholesky.
  If save3idx is true, save Cholesky-decomposed 3-index integrals, 
  otherwise save pseudo-square-root-inverse Cholesky decomposition.
"""
function generate_AO_DF_integrals(EC::ECInfo, fitbasis="mp2fit"; save3idx=true)
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, fitbasis)
  save(EC,"S_AA",overlap(bao))
  save(EC,"h_AA",kinetic(bao) + nuclear(bao))
  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thr, verbose = true)
  if save3idx
    pqP = ERI_2e3c(bao,bfit)
    @tensoropt pqL[p,q,L] := pqP[p,q,P] * M[P,L]
    save(EC,"AAL",pqL)
  else
    save(EC,"C_PL",M)
  end
  return nuclear_repulsion(EC.ms)
end

"""
    generate_3idx_integrals(EC::ECInfo, cMO)

  Generate ``v_p^{qL}``, ``f_p^q`` and `E_{nuc}` with
  ``v_{pr}^{qs} = v_p^{qL} 1_{LL'} v_r^{sL'}``.
"""
function generate_3idx_integrals(EC::ECInfo, cMO)
  @assert ndims(cMO) == 2 "unrestricted not implemented yet"
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

end #module