"""
This module contains various utils for density fitting.
"""
module DFTools
using LinearAlgebra, TensorOperations
using Buffers
# using TSVD
using IterativeSolvers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..ElemCo.Integrals
using ..ElemCo.MSystem
using ..ElemCo.FockFactory
using ..ElemCo.TensorTools

export generate_AO_DF_integrals, generate_DF_integrals, generate_DF_Fock

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
  S_AA = overlap(bao)
  t_AA = kinetic(bao)
  v_AA = nuclear(bao)
  save!(EC, "S_AA", S_AA)
  save!(EC, "h_AA", t_AA + v_AA)  
  if EC.options.wf.npositron > 0
    save!(EC, "h_positron_AA", t_AA - v_AA)
  end
  PQ = eri_2e2idx(bfit)
  M = sqrtinvchol(PQ, tol=EC.options.cholesky.thred, verbose=true)
  if save3idx
    Pbatches = BasisBatcher(bao, bfit, EC.options.int.target_batch_length)
    lencbuf = buffer_size_3idx(Pbatches)
    cbuf = Buffer{Cdouble}(lencbuf)
    maxP = max_batch_length(Pbatches)
    nA = size(S_AA, 1)
    nL = size(M, 2)
    AALfile, AAL = newmmap(EC, "AAL", (nA,nA,nL))
    buf = Buffer(nA*nA*maxP + nL*maxP)
    LBlks = get_spaceblocks(1:nL)
    first = true
    for Pblk in Pbatches
      P = range(Pblk)
      lenP = length(P)
      AAP = alloc!(buf, nA, nA, lenP)
      eri_2e3idx!(AAP, cbuf, Pblk)
      M_PL = alloc!(buf, lenP, nL)
      M_PL .= @view M[P,:]
      if first
        for L in LBlks
          v!M = @view M_PL[:,L]
          v!AAL = @view AAL[:,:,L]
          @mtensor v!AAL[p,q,L] = AAP[p,q,P] * v!M[P,L]
        end
        first = false
      else
        for L in LBlks
          v!M = @view M_PL[:,L]
          v!AAL = @view AAL[:,:,L]
          @mtensor v!AAL[p,q,L] += AAP[p,q,P] * v!M[P,L]
        end
      end
      drop!(buf, AAP, M_PL) 
    end
    closemmap(EC, AALfile, AAL)
  else
    save!(EC, "C_PL", M)
  end
  return nuclear_repulsion(EC.system)
end

"""
    generate_3idx_integrals(EC::ECInfo, cMO::SpinMatrix, fitbasis="mpfit"; save3idx=true)

  Generate ``v_p^{qL}`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``
  and store in file `mmL`.
  If `save3idx` is false, no 3-index integrals are calculated, only save pseudo-square-root-inverse Cholesky decomposition.
"""
function generate_3idx_integrals(EC::ECInfo, cMO::SpinMatrix, fitbasis="mpfit"; save3idx=true)
  generate_AO_DF_integrals(EC, fitbasis; save3idx)
  if !save3idx
    return
  end
  AALfile, AAL = mmap3idx(EC, "AAL")
  nao = size(cMO[1], 1)
  nmo = size(cMO[1], 2)
  nL = size(AAL, 3)
  unrestricted = !is_restricted(cMO)
  mmLfile, mmL = newmmap(EC, "mmL", (nmo,nmo,nL))
  if unrestricted
    MMLfile, MML = newmmap(EC, "MML", (nmo,nmo,nL))
  end
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  buf = Buffer(nmo*nao*maxL)
  c_Am = cMO[1]
  c_AM = cMO[2]
  for L in LBlks
    lenL = length(L)
    v!AAL = @view AAL[:,:,L]
    v!mmL = @view mmL[:,:,L]
    mAL = alloc!(buf, nmo, nao, lenL)
    @mtensor mAL[p,ν,L] = c_Am[μ,p] * v!AAL[μ,ν,L]
    @mtensor v!mmL[p,q,L] = mAL[p,ν,L] * c_Am[ν,q]
    drop!(buf, mAL)
    if unrestricted
      v!MML = @view MML[:,:,L]
      MAL = alloc!(buf, nmo, nao, lenL)
      @mtensor MAL[p,ν,L] = c_AM[μ,p] * v!AAL[μ,ν,L]
      @mtensor v!MML[p,q,L] = MAL[p,ν,L] * c_AM[ν,q]
      drop!(buf, MAL)
    end
  end
  close(AALfile)
  closemmap(EC, mmLfile, mmL)
  if unrestricted
    closemmap(EC, MMLfile, MML)
  end
  return
end

"""
    generate_DF_integrals(EC::ECInfo, cMO::SpinMatrix; save3idx=true)

  Generate ``v_p^{qL}`` and ``f_p^q`` with
  ``v_{pr}^{qs} = v_p^{qL} δ_{LL'} v_r^{sL'}``.
  The ``v_p^{qL}`` are generated using `mpfit` fitting basis, and
  the ``f_p^q`` are generated using `jkfit` fitting basis.
  The integrals are stored in files `mmL` and `f_mm`.

  Return reference energy (calculated using `jkfit` fitting basis).
"""
function generate_DF_integrals(EC::ECInfo, cMO::SpinMatrix; save3idx=true)
  if !system_exists(EC.system)
    error("Molecular system not specified!")
  end
  # calculate fock matrix in AO basis (integral direct)
  EHF = generate_DF_Fock(EC, cMO)
  # calculate 3-index integrals
  generate_3idx_integrals(EC, cMO, "mpfit"; save3idx)
  return EHF
end

"""
    generate_DF_Fock(EC::ECInfo, cMO::SpinMatrix; check_diagonal=false)

  Generate DF Fock matrix in MO basis.
  If `check_diagonal` is true, check the off-diagonal elements of the Fock matrix to be small.
  The Fock matrix is saved in files `f_mm`/`f_MM` and orbital energies in `e_m`/`e_M`.

  Return reference energy.
"""
function generate_DF_Fock(EC::ECInfo, cMO::SpinMatrix; check_diagonal=false)
  if !system_exists(EC.system)
    error("Molecular system not specified!")
  end
  occα = EC.space['o']
  occβ = EC.space['O']
  # calculate fock matrix in AO basis (integral direct)
  generate_AO_DF_integrals(EC, "jkfit"; save3idx=false)
  bao = generate_basis(EC, "ao")
  bfit = generate_basis(EC, "jkfit")
  h_AA = load2idx(EC, "h_AA")
  if is_restricted(cMO) && occα == occβ
    # restricted closed-shell
    fock = SpinMatrix(gen_dffock(EC, cMO[1], bao, bfit))
    nspin = 1
  else
    # unrestricted
    fock = gen_dffock(EC, cMO, bao, bfit)
    nspin = 2
  end
  EHF = 0.0
  for isp in 1:nspin
    fock_MO = cMO[isp]' * fock[isp] * cMO[isp]
    m = ('m','M')[isp]
    occ = (occα, occβ)[isp]
    save!(EC, "f_$m$m", fock_MO)
    eps = diag(fock_MO)
    println("Occupied orbital energies: ", eps[occ])
    save!(EC, "e_$m", eps)
    if nspin == 1
      save!(EC, "e_M", eps)
    end
    if check_diagonal
      # Checking off-diagonal elements of fock matrix
      maxoff = maximum(abs, fock_MO - Diagonal(fock_MO))
      if maxoff > 1e-8
        if EC.options.wf.ignore_error
          warn("The largest off-diagonal element of fock matrix is $maxoff > 1e-8")
        else
          error("The largest off-diagonal element of fock matrix is $maxoff > 1e-8 
          The error can be ignored by setting wf,ignore_error=true.")
        end
      end
    end
    hsmall = cMO[isp]' * h_AA * cMO[isp]
    EHF += sum(eps[occ]) + sum(diag(hsmall)[occ])
  end
  EHF /= nspin
  EHF += nuclear_repulsion(EC.system)
  return EHF
end

end #module
