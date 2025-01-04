"""
This module contains various utils for density fitting.
"""
module DFTools
using LinearAlgebra, TensorOperations
using Buffers
# using TSVD
using IterativeSolvers
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.Integrals
using ..ElemCo.MSystem
using ..ElemCo.FockFactory
using ..ElemCo.TensorTools

export generate_AO_DF_integrals, generate_DF_integrals

"""
    generate_AO_DF_integrals(EC::ECInfo, fitbasis="mpfit"; save3idx=true)

  Generate AO integrals using DF + Cholesky.
  If save3idx is true, save Cholesky-decomposed 3-index integrals, 
  otherwise save pseudo-square-root-inverse Cholesky decomposition.

  Return nuclear repulsion energy.
"""
function generate_AO_DF_integrals(EC::ECInfo, fitbasis="mpfit"; save3idx=true)
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, fitbasis)
  save!(EC,"S_AA",overlap(bao))
  save!(EC,"h_AA",kinetic(bao) + nuclear(bao))  
  if EC.options.wf.npositron > 0
    save!(EC,"h_positron_AA",kinetic(bao) - nuclear(bao))
  end
  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  if save3idx
    AAP = eri_2e3idx(bao,bfit)
    nA = size(AAP,1)
    nL = size(M,2)
    AALfile, AAL = newmmap(EC, "AAL", (nA,nA,nL))
    LBlks = get_spaceblocks(1:nL)
    for L in LBlks
      v!M = @view M[:,L]
      v!AAL = @view AAL[:,:,L]
      @tensoropt v!AAL[p,q,L] = AAP[p,q,P] * v!M[P,L]
    end
    closemmap(EC, AALfile, AAL)
  else
    save!(EC,"C_PL",M)
  end
  return nuclear_repulsion(EC.system)
end

"""
    generate_3idx_integrals(EC::ECInfo, cMO::SpinMatrix, fitbasis="mpfit")

  Generate ``v_p^{qL}`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``
  and store in file `mmL`.
"""
function generate_3idx_integrals(EC::ECInfo, cMO::SpinMatrix, fitbasis="mpfit")
  @assert is_restricted(cMO) "unrestricted not implemented yet"
  cMO1 = cMO[1]
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, fitbasis)

  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  μνP = eri_2e3idx(bao,bfit)
  nao = size(μνP, 1)
  nmo = size(cMO1, 2)
  nL = size(M, 2)
  mmLfile, mmL = newmmap(EC, "mmL", (nmo,nmo,nL))
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  buf = Buffer((nao+nmo)*nao*maxL)
  for L in LBlks
    v!M = @view M[:,L]
    v!mmL = @view mmL[:,:,L]
    AAL = alloc!(buf, nao, nao, nL)
    @mtensor AAL[μ,ν,L] = μνP[μ,ν,P] * v!M[P,L]
    mAL = alloc!(buf, nmo, nao, nL)
    n!mAL = neuralyze(mAL)
    @mtensor n!mAL[p,ν,L] = cMO1[μ,p] * AAL[μ,ν,L]
    @mtensor v!mmL[p,q,L] = mAL[p,ν,L] * cMO1[ν,q]
    drop!(buf, AAL, mAL)
  end
  closemmap(EC, mmLfile, mmL)
end

"""
    generate_DF_integrals(EC::ECInfo, cMO::SpinMatrix)

  Generate ``v_p^{qL}`` and ``f_p^q`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``.
  The ``v_p^{qL}`` are generated using `mpfit` fitting basis, and
  the ``f_p^q`` are generated using `jkfit` fitting basis.
  The integrals are stored in files `mmL` and `f_mm`.

  Return reference energy (calculated using `jkfit` fitting basis).
"""
function generate_DF_integrals(EC::ECInfo, cMO::SpinMatrix)
  @assert is_restricted(cMO) "unrestricted not implemented yet"
  if !system_exists(EC.system)
    error("Molecular system not specified!")
  end
  # calculate fock matrix in AO basis (integral direct)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "jkfit")
  fock = gen_dffock(EC, cMO[1], bao, bfit)
  fock_MO = cMO[1]' * fock * cMO[1]
  save!(EC,"f_mm",fock_MO)
  eps = diag(fock_MO)
  println("Occupied orbital energies: ", eps[EC.space['o']])
  save!(EC, "e_m", eps)
  save!(EC, "e_M", eps)
  occ = EC.space['o']
  hsmall = cMO[1]' * load2idx(EC,"h_AA") * cMO[1]
  EHF = sum(eps[occ]) + sum(diag(hsmall)[occ]) + nuclear_repulsion(EC.system)
  # calculate 3-index integrals
  generate_3idx_integrals(EC, cMO, "mpfit")
  return EHF
end


end #module
