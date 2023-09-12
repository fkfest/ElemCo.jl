"""
This module contains various utils for density fitting.
"""
module DFTools
using LinearAlgebra, TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.FockFactory
using ..ElemCo.TensorTools

export generate_AO_DF_integrals, generate_DF_integrals

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
    generate_3idx_integrals(EC::ECInfo, cMO, fitbasis="mp2fit")

  Generate ``v_p^{qL}`` with
  ``v_{pr}^{qs} = v_p^{qL} 1_{LL'} v_r^{sL'}``
  and store in file `mmL`.
"""
function generate_3idx_integrals(EC::ECInfo, cMO, fitbasis="mp2fit")
  @assert ndims(cMO) == 2 "unrestricted not implemented yet"
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, fitbasis)

  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thr, verbose = true)
  μνP = ERI_2e3c(bao,bfit)
  @tensoropt μνL[p,q,L] := μνP[p,q,P] * M[P,L]
  μνP = nothing
  M = nothing
  @tensoropt pqL[p,q,L] := cMO[μ,p] * μνL[μ,ν,L] * cMO[ν,q]
  μνL = nothing
  save(EC,"mmL",pqL)
end

"""
    generate_DF_integrals(EC::ECInfo, cMO)

  Generate ``v_p^{qL}``, ``f_p^q`` and `E_{nuc}` with
  ``v_{pr}^{qs} = v_p^{qL} 1_{LL'} v_r^{sL'}``.
  The ``v_p^{qL}`` are generated using `mp2fit` fitting basis, and
  the ``f_p^q`` are generated using `jkfit` fitting basis.
  The integrals are stored in files `mmL` and `f_mm`.
"""
function generate_DF_integrals(EC::ECInfo, cMO)
  @assert ndims(cMO) == 2 "unrestricted not implemented yet"

  # calculate fock matrix in AO basis (integral direct)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, "jkfit")
  fock = gen_dffock(EC, cMO, bao, bfit)
  fock_MO = cMO' * fock * cMO
  save(EC,"f_mm",fock_MO)
  # calculate 3-index integrals
  generate_3idx_integrals(EC, cMO, "mp2fit")
end

end #module