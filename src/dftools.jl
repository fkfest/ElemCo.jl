"""
This module contains various utils for density fitting.
"""
module DFTools
using LinearAlgebra, TensorOperations
# using TSVD
using IterativeSolvers
using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.FockFactory
using ..ElemCo.TensorTools

export get_auxblks, generate_AO_DF_integrals, generate_DF_integrals

"""
    get_auxblks(naux, maxblocksize=128, strict=false)

  Generate ranges for block indices for auxiliary basis (for loop over blocks).

  If `strict` is true, the blocks will be of size `maxblocksize` (except for the last block).
  Otherwise the actual block size will be as close as possible to `blocksize` such that
  the resulting blocks are of similar size.
"""
function get_auxblks(naux, maxblocksize=128, strict=false)
  nauxblks = naux ÷ maxblocksize
  if nauxblks*maxblocksize < naux
    nauxblks += 1
  end
  if strict 
    auxblks = [ (i-1)*maxblocksize+1 : ((i == nauxblks) ? naux : i*maxblocksize) for i in 1:nauxblks ]
  else
    blocksize = naux ÷ nauxblks
    n_largeblks = mod(naux, nauxblks)
    auxblks = [ (i-1)*(blocksize+1)+1 : i*(blocksize+1) for i in 1:n_largeblks ]
    start = n_largeblks*(blocksize+1)+1
    for i = n_largeblks+1:nauxblks
      push!(auxblks, start:start+blocksize-1)
      start += blocksize
    end
  end
  return auxblks
end

"""
    generate_AO_DF_integrals(EC::ECInfo, fitbasis="mp2fit"; save3idx=true)

  Generate AO integrals using DF + Cholesky.
  If save3idx is true, save Cholesky-decomposed 3-index integrals, 
  otherwise save pseudo-square-root-inverse Cholesky decomposition.
"""
function generate_AO_DF_integrals(EC::ECInfo, fitbasis="mp2fit"; save3idx=true)
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, fitbasis)
  save!(EC,"S_AA",overlap(bao))
  save!(EC,"h_AA",kinetic(bao) + nuclear(bao))
  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  if save3idx
    AAP = ERI_2e3c(bao,bfit)
    nA = size(AAP,1)
    nL = size(M,2)
    AALfile, AAL = newmmap(EC, "AAL", Float64, (nA,nA,nL))
    LBlks = get_auxblks(nL)
    for L in LBlks
      V_M = @view M[:,L]
      V_AAL = @view AAL[:,:,L]
      @tensoropt V_AAL[p,q,L] = AAP[p,q,P] * V_M[P,L]
    end
    closemmap(EC, AALfile, AAL)
  else
    save!(EC,"C_PL",M)
  end
  return nuclear_repulsion(EC.ms)
end

"""
    generate_3idx_integrals(EC::ECInfo, cMO, fitbasis="mp2fit")

  Generate ``v_p^{qL}`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``
  and store in file `mmL`.
"""
function generate_3idx_integrals(EC::ECInfo, cMO, fitbasis="mp2fit")
  @assert ndims(cMO) == 2 "unrestricted not implemented yet"
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, fitbasis)

  PQ = ERI_2e2c(bfit)
  M = sqrtinvchol(PQ, tol = EC.options.cholesky.thred, verbose = true)
  μνP = ERI_2e3c(bao,bfit)
  nm = size(cMO,2)
  nL = size(M,2)
  mmLfile, mmL = newmmap(EC, "mmL", Float64, (nm,nm,nL))
  LBlks = get_auxblks(nL)
  for L in LBlks
    V_M = @view M[:,L]
    V_mmL = @view mmL[:,:,L]
    @tensoropt μνL[μ,ν,L] := μνP[μ,ν,P] * V_M[P,L]
    @tensoropt V_mmL[p,q,L] = cMO[μ,p] * μνL[μ,ν,L] * cMO[ν,q]
  end
  closemmap(EC, mmLfile, mmL)
end

"""
    generate_DF_integrals(EC::ECInfo, cMO)

  Generate ``v_p^{qL}`` and ``f_p^q`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``.
  The ``v_p^{qL}`` are generated using `mp2fit` fitting basis, and
  the ``f_p^q`` are generated using `jkfit` fitting basis.
  The integrals are stored in files `mmL` and `f_mm`.

  Return reference energy (calculated using `jkfit` fitting basis).
"""
function generate_DF_integrals(EC::ECInfo, cMO)
  @assert ndims(cMO) == 2 "unrestricted not implemented yet"
  if !ms_exists(EC.ms)
    error("Molecular system not specified!")
  end
  # calculate fock matrix in AO basis (integral direct)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  bao = generate_basis(EC.ms, "ao")
  bfit = generate_basis(EC.ms, "jkfit")
  fock = gen_dffock(EC, cMO, bao, bfit)
  fock_MO = cMO' * fock * cMO
  save!(EC,"f_mm",fock_MO)
  eps = diag(fock_MO)
  println("Occupied orbital energies: ", eps[EC.space['o']])
  save!(EC, "e_m", eps)
  save!(EC, "e_M", eps)
  occ = EC.space['o']
  hsmall = cMO' * load(EC,"h_AA") * cMO
  EHF = sum(eps[occ]) + sum(diag(hsmall)[occ]) + nuclear_repulsion(EC.ms)
  # calculate 3-index integrals
  generate_3idx_integrals(EC, cMO, "mp2fit")
  return EHF
end


end #module
