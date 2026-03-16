""" Fock builders (using FciDump or DF integrals) """
module FockFactory
try
  using MKL
catch
  #println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
#BLAS.set_num_threads(1)
using Buffers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions
using ..ElemCo.FciDumps
using ..ElemCo.Integrals
using ..ElemCo.OrbTools

export gen_fock, gen_ufock, gen_dffock, gen_df3idx_fock
export gen_density_matrix, gen_frac_density_matrix

""" 
    gen_fock(EC::ECInfo)

  Calculate closed-shell fock matrix from FCIDump integrals. 
"""
function gen_fock(EC::ECInfo)
  @mtensor fock[p,q] := integ1(EC.fd,:α)[p,q] + 2.0*ints2(EC,":o:o",:α)[p,i,q,i] - ints2(EC,":oo:",:α)[p,i,i,q]
  return fock
end

""" 
    gen_fock(EC::ECInfo, spincase::Symbol)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`}. 
"""
function gen_fock(EC::ECInfo, spincase::Symbol)
  @mtensor fock[p,q] := integ1(EC.fd,spincase)[p,q] 
  if spincase == :α
    if n_occb_orbs(EC) > 0 
      @mtensor fock[p,q] += ints2(EC,":O:O",:αβ)[p,i,q,i]
    end
    spo='o'
    spv='v'
    nocc = n_occ_orbs(EC)
  else
    if n_occ_orbs(EC) > 0 
      @mtensor fock[p,q] += ints2(EC,"o:o:",:αβ)[i,p,i,q]
    end
    spo='O'
    spv='V'
    nocc = n_occb_orbs(EC)
  end
  if nocc > 0
    @mtensor begin
      fock[p,q] += ints2(EC,":"*spo*":"*spo,spincase)[p,i,q,i]
      fock[p,q] -= ints2(EC,":"*spo*spo*":",spincase)[p,i,i,q]
    end
  end
  return fock
end

""" 
    gen_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occvec)

  Generate ``D_{\\mu\\nu}=C^l_{\\mu i} C^r_{\\nu i}`` with ``i`` defined by `occvec`.
  Only real part of ``D_{\\mu\\nu}`` is kept unless T is Complex.
""" 
function gen_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occvec) where T
  CMOlo = CMOl[:,occvec]
  CMOro = CMOr[:,occvec]
  @mtensor den[r,s] := CMOlo[r,i]*CMOro[s,i]
  T <: Complex && return den
  denr = real.(den)
  if sum(abs2,den) - sum(abs2,denr) > EC.options.scf.imagtol
    println("Large imaginary part in density matrix neglected!")
    println("Difference between squared norms:",sum(abs2,den)-sum(abs2,denr))
  end
  return denr
end

""" 
    gen_frac_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occupation)

  Generate ``D_{\\mu\\nu}=C^l_{\\mu i} C^r_{\\nu i} n_i`` with ``n_i`` provided in `occupation`.
  Only real part of ``D_{\\mu\\nu}`` is kept unless T is Complex.
""" 
function gen_frac_density_matrix(EC::ECInfo{T}, CMOl::AbstractMatrix, CMOr::AbstractMatrix, occupation) where T  
  @assert length(occupation) == size(CMOr,2) "Wrong occupation vector length!"
  CMOrn = CMOr .* occupation'
  @mtensor den[r,s] := CMOl[r,i]*CMOrn[s,i]
  T <: Complex && return den
  denr = real.(den)
  if sum(abs2,den) - sum(abs2,denr) > EC.options.scf.imagtol
    println("Large imaginary part in density matrix neglected!")
    println("Difference between squared norms:",sum(abs2,den)-sum(abs2,denr))
  end
  return denr
end

""" 
    gen_fock(EC::ECInfo, den::AbstractMatrix)

  Calculate closed-shell fock matrix from FCIDump integrals and density matrix `den`. 
"""
function gen_fock(EC::ECInfo, den::AbstractMatrix)
  @mtensor begin 
    fock[p,q] := integ1(EC.fd,:α)[p,q] 
    fock[p,q] += ints2(EC,"::::",:α)[p,r,q,s] * den[r,s]
    fock[p,q] -= 0.5*(ints2(EC,"::::",:α)[p,r,s,q] * den[r,s])
  end
  return fock
end

""" 
    gen_fock(EC::ECInfo, CMOl::AbstractMatrix, CMOr::AbstractMatrix)

  Calculate closed-shell fock matrix from FCIDump integrals and orbitals `CMOl`, `CMOr`. 
"""
function gen_fock(EC::ECInfo, CMOl::AbstractMatrix, CMOr::AbstractMatrix)
  @assert EC.space['o'] == EC.space['O'] # closed-shell
  occ2 = EC.space['o']
  den = gen_density_matrix(EC, CMOl, CMOr, occ2)
  @mtensor begin 
    fock[p,q] := integ1(EC.fd,:α)[p,q] 
    fock[p,q] += 2.0*ints2(EC,"::::",:α)[p,r,q,s] * den[r,s]
    fock[p,q] -= ints2(EC,"::::",:α)[p,r,s,q] * den[r,s]
  end
  return fock
end

""" 
    gen_fock(EC::ECInfo, spincase::Symbol, CMOl::AbstractMatrix, CMOr::AbstractMatrix)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`} and orbitals `CMOl`, `CMOr` and
  orbitals for the opposite-spin `CMOlOS` and `CMOrOS`. 
"""
function gen_fock(EC::ECInfo, spincase::Symbol, CMOl::AbstractMatrix, CMOr::AbstractMatrix,
                  CMOlOS::AbstractMatrix, CMOrOS::AbstractMatrix)
  if spincase == :α
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['O'])
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[p,r,q,s]*denOS[r,s]
    spo = 'o'
  else
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['o'])
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[r,p,s,q]*denOS[r,s]
    spo = 'O'
  end
  den =  gen_density_matrix(EC, CMOl, CMOr, EC.space[spo])
  ints = ints2(EC,"::::",spincase)
  @mtensor fock[p,q] += ints[p,r,q,s] * den[r,s] 
  @mtensor fock[p,q] -= ints[p,r,s,q] * den[r,s]
  @mtensor fock[p,q] += integ1(EC.fd,spincase)[p,q] 
  return fock
end

""" 
    gen_fock(EC::ECInfo, spincase::Symbol, den::AbstractMatrix, denOS::AbstractMatrix)

  Calculate UHF fock matrix from FCIDump integrals and density matrices `den` (for `spincase`) 
  and `denOS` (opposite spin to `spincase`). 
"""
function gen_fock(EC::ECInfo, spincase::Symbol, 
                  den::AbstractMatrix, denOS::AbstractMatrix)
  if spincase == :α
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[p,r,q,s]*denOS[r,s]
  else
    @mtensor fock[p,q] := ints2(EC,"::::",:αβ)[r,p,s,q]*denOS[r,s]
  end
  ints = ints2(EC,"::::",spincase)
  @mtensor fock[p,q] += ints[p,r,q,s] * den[r,s] 
  @mtensor fock[p,q] -= ints[p,r,s,q] * den[r,s]
  @mtensor fock[p,q] += integ1(EC.fd,spincase)[p,q] 
  return fock
end

""" 
    gen_ufock(EC::ECInfo, CMOl::SpinMatrix, CMOr::SpinMatrix)

  Calculate UHF fock matrix from FCIDump integrals and orbitals `cMOl`, `cMOr`
  with `cMOl[1]` and `cMOr[1]` - α-MO transformation coefficients and 
  `cMOl[2]` and `cMOr[2]` - β-MO transformation coefficients. 
"""
function gen_ufock(EC::ECInfo, cMOl::SpinMatrix, cMOr::SpinMatrix)
  return SpinMatrix(gen_fock(EC, :α, cMOl.α, cMOr.α, cMOl.β, cMOr.β), 
                    gen_fock(EC, :β, cMOl.β, cMOr.β, cMOl.α, cMOr.α))
end

"""
    gen_ufock(EC::ECInfo, den::SpinMatrix)

  Calculate UHF fock matrix from FCIDump integrals and density matrix `den`. 
"""
function gen_ufock(EC::ECInfo, den::SpinMatrix)
  return SpinMatrix(gen_fock(EC, :α, den.α, den.β), 
                    gen_fock(EC, :β, den.β, den.α))
end

""" 
    gen_dffock(EC::ECInfo, cMO::AbstractMatrix, bao, bfit)

  Compute closed-shell DF-HF Fock matrix (integral direct) in AO basis.
"""
function gen_dffock(EC::ECInfo{T}, cMO::AbstractMatrix, bao, bfit) where T
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  nA = size(CMO2, 1)
  nocc = size(CMO2, 2)
  nL = size(PL, 2)
  Pbatches = BasisBatcher(bao, bfit)
  maxP = max_batch_length(Pbatches)
  LoA = zeros(T, nL, nocc, nA)
  lenbuf = (nocc*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(T, lenbuf) cbuf(Cdouble, lencbuf) begin
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    oAP = alloc!(buf, nocc, nA, lenP)
    AAP = alloc!(buf, nA, nA, lenP)
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor oAP[j,ν,P] = AAP[μ,ν,P] * CMO2[μ,j]
    drop!(buf, AAP)
    M_PL = alloc!(buf, lenP, nL)
    M_PL .= @view PL[P,:]
    @mtensor LoA[L,j,ν] += oAP[j,ν,P] * M_PL[P,L]
    drop!(buf, oAP, M_PL)
  end
  @mtensor cL[L] := LoA[L,j,ν] * CMO2[ν,j]
  @mtensor fock[μ,ν] := hsmall[μ,ν] - LoA[L,j,μ]*LoA[L,j,ν] 
  @mtensor cP[P] := cL[L] * PL[P,L]
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    AAP = alloc!(buf, nA, nA, lenP)
    v!cP = @mview cP[P]
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor fock[μ,ν] += 2.0*v!cP[P]*AAP[μ,ν,P]
    drop!(buf, AAP)
  end
  end #buffer
  return fock
end

""" 
    gen_dffock(EC::ECInfo{T}, cMO::SpinMatrix, bao, bfit)

  Compute unrestricted DF-HF Fock matrices `SpinMatrix(Fα, Fβ)` in AO basis (integral direct).
"""
function gen_dffock(EC::ECInfo{T}, cMO::SpinMatrix, bao, bfit) where T
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  # println(size(Ppq))
  occa = EC.space['o']
  occb = EC.space['O']
  CMOo = SpinMatrix(cMO[1][:,occa], cMO[2][:,occb])
  fock = SpinMatrix(hsmall)
  unrestrict!(fock)
  nA = size(CMOo[1], 1)
  nocc = size(CMOo[1], 2)
  nOcc = size(CMOo[2], 2)
  nL = size(PL, 2)
  Pbatches = BasisBatcher(bao, bfit)
  maxP = max_batch_length(Pbatches)
  LoA = zeros(T, nL, nocc, nA)
  LOA = zeros(T, nL, nOcc, nA)
  lenbuf = ((nocc+nOcc)*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(T, lenbuf) cbuf(Cdouble, lencbuf) begin
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    oAP = alloc!(buf, nocc, nA, lenP)
    OAP = alloc!(buf, nOcc, nA, lenP)
    AAP = alloc!(buf, nA, nA, lenP)
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor oAP[j,ν,P] = AAP[μ,ν,P] * CMOo[1][μ,j]
    if nOcc > 0
      @mtensor OAP[j,ν,P] = AAP[μ,ν,P] * CMOo[2][μ,j]
    end
    drop!(buf, AAP)
    M_PL = alloc!(buf, lenP, nL)
    M_PL .= @view PL[P,:]
    @mtensor LoA[L,j,ν] += oAP[j,ν,P] * M_PL[P,L]
    if nOcc > 0
      @mtensor LOA[L,j,ν] += OAP[j,ν,P] * M_PL[P,L]
    end
    reset!(buf)
  end
  @mtensor cL[L] := LoA[L,j,ν] * CMOo[1][ν,j]
  @mtensor cL[L] += LOA[L,j,ν] * CMOo[2][ν,j]
  @mtensor fock[1][μ,ν] -= LoA[L,j,μ]*LoA[L,j,ν] 
  @mtensor fock[2][μ,ν] -= LOA[L,j,μ]*LOA[L,j,ν] 
  @mtensor cP[P] := cL[L] * PL[P,L]
  coulfock = zeros(T, nA, nA)
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    AAP = alloc!(buf, nA, nA, lenP)
    v!cP = @mview cP[P]
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor coulfock[μ,ν] += v!cP[P]*AAP[μ,ν,P]
    drop!(buf, AAP)
  end
  fock[1] += coulfock
  fock[2] += coulfock
  end #buffer
  return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::AbstractMatrix)

  Compute closed-shell DF-HF Fock matrix in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo, cMO::AbstractMatrix)
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  CMO2d = permutedims(CMO2, [2,1])
  hsmall = load2idx(EC, "h_AA")
  AALfile, AAL = mmap3idx(EC, "AAL")
  nocc = size(CMO2, 2)
  nA = size(AAL, 1)
  nL = size(AAL, 3)
  fock = hsmall
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf((nocc*nA+1)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!AAL = @mview AAL[:,:,L]
    oAL = alloc!(buf, nocc, nA, lenL)
    @mtensor oAL[j,ν,L] = v!AAL[μ,ν,L] * CMO2[μ,j]
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = oAL[j,ν,L] * CMO2d[j,ν]
    @mtensor fock[μ,ν] += 2.0 * cL[L] * v!AAL[μ,ν,L]
    @mtensor fock[μ,ν] -= oAL[j,μ,L] * oAL[j,ν,L]
    drop!(buf, oAL, cL)
  end
  close(AALfile)
  end #buffer
  return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::AbstractMatrix, cPO::AbstractMatrix)

  Compute closed-shell DF-HF Fock matrix and the positron
  Fock matrix in AO basis  (using precalculated Cholesky-
  decomposed integrals and density matrices).
"""
function gen_dffock(EC::ECInfo, cMO::AbstractMatrix, cPO::AbstractMatrix)
  #TODO: rewrite with loops to reduce memory usage
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  CMO2p = cPO[:,1:1]
  hsmall = load2idx(EC,"h_AA")
  hsmall_pos = load2idx(EC,"h_positron_AA")
  μνL = load3idx(EC,"AAL")
  # Electron
  @mtensor begin 
    μjL[p,j,L] := μνL[p,q,L] * CMO2[q,j]
    L[L] := μjL[p,j,L] * CMO2[p,j]
    J[p,q] := μνL[p,q,L] * L[L]
    K[p,q] := μjL[p,j,L] * μjL[q,j,L] 
  end
  # Positron
  @mtensor begin
    μjLpos[p,j,L] := μνL[p,q,L] * CMO2p[q,j]
    P[L] := μjLpos[p,j,L] * CMO2p[p,j]
    Jp[p,q] := μνL[p,q,L] * P[L] 
  end
  fock = hsmall + 2*J - K - Jp
  fock_pos = hsmall_pos - 2*J
  return fock, fock_pos, Jp
  #return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::MOs)

  Compute unrestricted DF-HF Fock matrices [Fα, Fβ] in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo{T}, cMO::SpinMatrix) where T
  occa = EC.space['o']
  occb = EC.space['O']
  CMOo = SpinMatrix(cMO[1][:,occa], cMO[2][:,occb])
  CMOod = SpinMatrix(permutedims(CMOo[1], [2,1]), permutedims(CMOo[2], [2,1]))
  hsmall = load2idx(EC,"h_AA")
  fock = SpinMatrix(hsmall)
  unrestrict!(fock)
  AALfile, AAL = mmap3idx(EC, "AAL")
  nocc = size(CMOo[1], 2)
  nOcc = size(CMOo[2], 2)
  nA = size(AAL, 1)
  nL = size(AAL, 3)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(T, (nocc+nOcc)*nA*maxL + maxL) begin
  coulfock = zeros(T, nA, nA)
  for L in LBlks
    lenL = length(L)
    v!AAL = @mview AAL[:,:,L]
    oAL = alloc!(buf, nocc, nA, lenL)
    OAL = alloc!(buf, nOcc, nA, lenL)
    @mtensor oAL[j,ν,L] = v!AAL[μ,ν,L] * CMOo[1][μ,j]
    @mtensor OAL[j,ν,L] = v!AAL[μ,ν,L] * CMOo[2][μ,j]
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = oAL[j,ν,L] * CMOod[1][j,ν]
    if nOcc > 0
      @mtensor cL[L] += OAL[j,ν,L] * CMOod[2][j,ν]
    end
    @mtensor coulfock[μ,ν] += cL[L] * v!AAL[μ,ν,L]
    @mtensor fock[1][μ,ν] -= oAL[j,μ,L] * oAL[j,ν,L]
    @mtensor fock[2][μ,ν] -= OAL[j,μ,L] * OAL[j,ν,L]
    reset!(buf)
  end
  close(AALfile)
  fock[1] += coulfock
  fock[2] += coulfock
  end #buffer
  return fock
end

"""
    gen_df3idx_fock(EC::ECInfo, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, occ::AbstractVector{Int})

  Compute closed-shell Fock matrix from 1e-integrals `h1` and MO-basis 3-index integrals `mmL`.
  `occ` contains the indices of the occupied orbitals.

  ``F_{pq} = h_{pq} + 2 J_{pq} - K_{pq}``
  with ``J_{pq} = \\sum_L B_p^{qL} c_L``, ``c_L = \\sum_i B_i^{iL}``
  and ``K_{pq} = \\sum_{iL} B_p^{iL} B_i^{qL}``.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, 
                         occ::AbstractVector{Int}) where T
  norb = size(h1, 1)
  nL = size(mmL, 3)
  fock = copy(h1)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  nocc = length(occ)
  @buffer buf(T, (nocc*norb + 1)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    # Coulomb: cL = Σ_i mmL[i,i,L]
    cL = alloc!(buf, lenL)
    v!ooL = @view mmL[occ,occ,L]
    @mtensor cL[L] = v!ooL[i,i,L]
    @mtensor fock[p,q] += 2.0 * cL[L] * v!mmL[p,q,L]
    drop!(buf, cL)
    # Exchange: K_pq = Σ_i Σ_L mmL[p,i,L]*conj(mmL[q,i,L])
    piL = alloc!(buf, norb, nocc, lenL)
    piL .= @view(mmL[:,occ,L])
    @mtensor fock[p,q] -= piL[p,i,L] * conj(piL[q,i,L])
    drop!(buf, piL)
  end
  end #buffer
  return fock
end

"""
    gen_df3idx_fock(EC::ECInfo, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, cMO_occ::AbstractMatrix)

  Compute closed-shell Fock matrix from 1e-integrals `h1` and MO-basis 3-index integrals `mmL`
  using occupied MO coefficients `cMO_occ` (rotation from original to occupied MOs).
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1::AbstractMatrix, mmL::AbstractArray{<:Number,3}, 
                         cMO_occ::AbstractMatrix) where T
  norb = size(h1, 1)
  nL = size(mmL, 3)
  nocc = size(cMO_occ, 2)
  fock = copy(h1)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(T, (nocc*norb + 1)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    # half-transform: oqL[j,q,L] = Σ_p mmL[p,q,L] * cMO_occ[p,j]
    oqL = alloc!(buf, nocc, norb, lenL)
    @mtensor oqL[j,q,L] = v!mmL[p,q,L] * cMO_occ[p,j]
    # Coulomb: cL = Σ_{jq} oqL[j,q,L] * conj(cMO_occ[q,j])
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = oqL[j,q,L] * conj(cMO_occ[q,j])
    @mtensor fock[p,q] += 2.0 * cL[L] * v!mmL[p,q,L]
    drop!(buf, cL)
    # Exchange: K_pq = Σ_{jL} conj(oqL[j,p,L])*oqL[j,q,L]
    @mtensor fock[p,q] -= conj(oqL[j,p,L]) * oqL[j,q,L]
    drop!(buf, oqL)
  end
  end #buffer
  return fock
end

"""
    gen_df3idx_fock(EC::ECInfo, h1a, h1b, mmL, MML, occa, occb)

  Compute UHF Fock matrices from 1e-integrals `h1a`/`h1b` and MO-basis 
  3-index integrals `mmL` (α) and `MML` (β).

  Returns `SpinMatrix(Fα, Fβ)`.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1a::AbstractMatrix, h1b::AbstractMatrix,
                         mmL::AbstractArray{<:Number,3}, MML::AbstractArray{<:Number,3},
                         occa::AbstractVector{Int}, occb::AbstractVector{Int}) where T
  norb = size(h1a, 1)
  nL = size(mmL, 3)
  focka = copy(h1a)
  fockb = copy(h1b)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  nocca = length(occa)
  noccb = length(occb)
  @buffer buf(T, (max(nocca, noccb)*norb + 1)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    v!MML = @mview MML[:,:,L]
    # Total Coulomb: cL_total = Σ_iα mmL[i,i,L] + Σ_Iβ MML[I,I,L]
    cL = alloc!(buf, lenL)
    v!ooL = @view mmL[occa,occa,L]
    v!OO_L = @view MML[occb,occb,L]
    @mtensor cL[L] = v!ooL[i,i,L]
    @mtensor cL[L] += v!OO_L[I,I,L]
    @mtensor focka[p,q] += cL[L] * v!mmL[p,q,L]
    @mtensor fockb[p,q] += cL[L] * v!MML[p,q,L]
    drop!(buf, cL)
    # α exchange
    if nocca > 0
      piL = alloc!(buf, norb, nocca, lenL)
      piL .= @view(mmL[:,occa,L])
      @mtensor focka[p,q] -= piL[p,i,L] * conj(piL[q,i,L])
      drop!(buf, piL)
    end
    # β exchange
    if noccb > 0
      PIL = alloc!(buf, norb, noccb, lenL)
      PIL .= @view(MML[:,occb,L])
      @mtensor fockb[p,q] -= PIL[p,i,L] * conj(PIL[q,i,L])
      drop!(buf, PIL)
    end
  end
  end #buffer
  return SpinMatrix(focka, fockb)
end

"""
    gen_df3idx_fock(EC::ECInfo, h1a, h1b, mmL, MML, cMO_occa::AbstractMatrix, cMO_occb::AbstractMatrix)

  Compute UHF Fock matrices from 1e-integrals and MO-basis 3-index integrals
  using occupied MO coefficients (rotations from original to occupied MOs).

  Returns `SpinMatrix(Fα, Fβ)`.
"""
function gen_df3idx_fock(EC::ECInfo{T}, h1a::AbstractMatrix, h1b::AbstractMatrix,
                         mmL::AbstractArray{<:Number,3}, MML::AbstractArray{<:Number,3},
                         cMO_occa::AbstractMatrix, cMO_occb::AbstractMatrix) where T
  norb = size(h1a, 1)
  nL = size(mmL, 3)
  nocca = size(cMO_occa, 2)
  noccb = size(cMO_occb, 2)
  focka = copy(h1a)
  fockb = copy(h1b)
  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  @buffer buf(T, ((nocca + noccb)*norb + 1)*maxL) begin
  for L in LBlks
    lenL = length(L)
    v!mmL = @mview mmL[:,:,L]
    v!MML = @mview MML[:,:,L]
    # half-transform α: oqL[j,q,L] = Σ_p mmL[p,q,L] * cMO_occa[p,j]
    oqL = alloc!(buf, nocca, norb, lenL)
    @mtensor oqL[j,q,L] = v!mmL[p,q,L] * cMO_occa[p,j]
    # half-transform β: OqL[j,q,L] = Σ_p MML[p,q,L] * cMO_occb[p,j]
    OqL = alloc!(buf, noccb, norb, lenL)
    if noccb > 0
      @mtensor OqL[j,q,L] = v!MML[p,q,L] * cMO_occb[p,j]
    end
    # Total Coulomb: cL = Σ_{jq} oqL[j,q,L]*conj(cMO_occa[q,j]) + Σ_{jq} OqL[j,q,L]*conj(cMO_occb[q,j])
    cL = alloc!(buf, lenL)
    @mtensor cL[L] = oqL[j,q,L] * conj(cMO_occa[q,j])
    if noccb > 0
      @mtensor cL[L] += OqL[j,q,L] * conj(cMO_occb[q,j])
    end
    @mtensor focka[p,q] += cL[L] * v!mmL[p,q,L]
    @mtensor fockb[p,q] += cL[L] * v!MML[p,q,L]
    drop!(buf, cL)
    # β exchange (drop OqL before oqL for LIFO order)
    if noccb > 0
      @mtensor fockb[p,q] -= conj(OqL[j,p,L]) * OqL[j,q,L]
    end
    drop!(buf, OqL)
    # α exchange
    @mtensor focka[p,q] -= conj(oqL[j,p,L]) * oqL[j,q,L]
    drop!(buf, oqL)
  end
  end #buffer
  return SpinMatrix(focka, fockb)
end

end #module
