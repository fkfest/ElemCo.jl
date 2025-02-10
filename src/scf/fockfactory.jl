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
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions
using ..ElemCo.FciDumps
using ..ElemCo.Integrals
using ..ElemCo.OrbTools

export gen_fock, gen_ufock, gen_dffock
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
    gen_density_matrix(EC::ECInfo, CMOl::Matrix, CMOr::Matrix, occvec)

  Generate ``D_{μν}=C^l_{μi} C^r_{νi}`` with ``i`` defined by `occvec`.
  Only real part of ``D_{μν}`` is kept.
""" 
function gen_density_matrix(EC::ECInfo, CMOl::Matrix, CMOr::Matrix, occvec)
  CMOlo = CMOl[:,occvec]
  CMOro = CMOr[:,occvec]
  @mtensor den[r,s] := CMOlo[r,i]*CMOro[s,i]
  denr = real.(den)
  if sum(abs2,den) - sum(abs2,denr) > EC.options.scf.imagtol
    println("Large imaginary part in density matrix neglected!")
    println("Difference between squared norms:",sum(abs2,den)-sum(abs2,denr))
  end
  return denr
end

""" 
    gen_frac_density_matrix(EC::ECInfo, CMOl::Matrix, CMOr::Matrix, occupation)

  Generate ``D_{μν}=C^l_{μi} C^r_{νi} n_i`` with ``n_i`` provided in `occupation`.
  Only real part of ``D_{μν}`` is kept.
""" 
function gen_frac_density_matrix(EC::ECInfo, CMOl::Matrix, CMOr::Matrix, occupation)
  @assert length(occupation) == size(CMOr,2) "Wrong occupation vector length!"
  CMOrn = CMOr .* occupation'
  @mtensor den[r,s] := CMOl[r,i]*CMOrn[s,i]
  denr = real.(den)
  if sum(abs2,den) - sum(abs2,denr) > EC.options.scf.imagtol
    println("Large imaginary part in density matrix neglected!")
    println("Difference between squared norms:",sum(abs2,den)-sum(abs2,denr))
  end
  return denr
end

""" 
    gen_fock(EC::ECInfo, den::Matrix)

  Calculate closed-shell fock matrix from FCIDump integrals and density matrix `den`. 
"""
function gen_fock(EC::ECInfo, den::Matrix)
  @mtensor begin 
    fock[p,q] := integ1(EC.fd,:α)[p,q] 
    fock[p,q] += ints2(EC,"::::",:α)[p,r,q,s] * den[r,s]
    fock[p,q] -= 0.5*(ints2(EC,"::::",:α)[p,r,s,q] * den[r,s])
  end
  return fock
end

""" 
    gen_fock(EC::ECInfo, CMOl::Matrix, CMOr::Matrix)

  Calculate closed-shell fock matrix from FCIDump integrals and orbitals `CMOl`, `CMOr`. 
"""
function gen_fock(EC::ECInfo, CMOl::Matrix, CMOr::Matrix)
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
    gen_fock(EC::ECInfo, spincase::Symbol, CMOl::Matrix, CMOr::Matrix)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`} and orbitals `CMOl`, `CMOr` and
  orbitals for the opposite-spin `CMOlOS` and `CMOrOS`. 
"""
function gen_fock(EC::ECInfo, spincase::Symbol, CMOl::Matrix, CMOr::Matrix,
                  CMOlOS::Matrix, CMOrOS::Matrix)
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
    gen_fock(EC::ECInfo, spincase::Symbol, den::Matrix, denOS::Matrix)

  Calculate UHF fock matrix from FCIDump integrals and density matrices `den` (for `spincase`) 
  and `denOS` (opposite spin to `spincase`). 
"""
function gen_fock(EC::ECInfo, spincase::Symbol, 
                  den::Matrix, denOS::Matrix)
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
    gen_dffock(EC::ECInfo, cMO::Matrix{Float64}, bao, bfit)

  Compute closed-shell DF-HF Fock matrix (integral direct) in AO basis.
"""
function gen_dffock(EC::ECInfo, cMO::Matrix{Float64}, bao, bfit)
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
  LoA = zeros(nL, nocc, nA)
  lenbuf = (nocc*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(lenbuf) cbuf(Cdouble, lencbuf) begin
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
    gen_dffock(EC::ECInfo, cMO::SpinMatrix, bao, bfit)

  Compute unrestricted DF-HF Fock matrices `SpinMatrix(Fα, Fβ)` in AO basis (integral direct).
"""
function gen_dffock(EC::ECInfo, cMO::SpinMatrix, bao, bfit)
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
  LoA = zeros(nL, nocc, nA)
  LOA = zeros(nL, nOcc, nA)
  lenbuf = ((nocc+nOcc)*nA + max(nA*nA, nL))*maxP
  lencbuf = buffer_size_3idx(Pbatches)
  @buffer buf(lenbuf) cbuf(Cdouble, lencbuf) begin
  for Pblk in Pbatches
    P = range(Pblk)
    lenP = length(P)
    oAP = alloc!(buf, nocc, nA, lenP)
    OAP = alloc!(buf, nOcc, nA, lenP)
    AAP = alloc!(buf, nA, nA, lenP)
    eri_2e3idx!(AAP, cbuf, Pblk)
    @mtensor oAP[j,ν,P] = AAP[μ,ν,P] * CMOo[1][μ,j]
    @mtensor OAP[j,ν,P] = AAP[μ,ν,P] * CMOo[2][μ,j]
    drop!(buf, AAP)
    M_PL = alloc!(buf, lenP, nL)
    M_PL .= @view PL[P,:]
    @mtensor LoA[L,j,ν] += oAP[j,ν,P] * M_PL[P,L]
    @mtensor LOA[L,j,ν] += OAP[j,ν,P] * M_PL[P,L]
    reset!(buf)
  end
  @mtensor cL[L] := LoA[L,j,ν] * CMOo[1][ν,j]
  @mtensor cL[L] += LOA[L,j,ν] * CMOo[2][ν,j]
  @mtensor fock[1][μ,ν] -= LoA[L,j,μ]*LoA[L,j,ν] 
  @mtensor fock[2][μ,ν] -= LOA[L,j,μ]*LOA[L,j,ν] 
  @mtensor cP[P] := cL[L] * PL[P,L]
  coulfock = zeros(nA, nA)
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
    gen_dffock(EC::ECInfo, cMO::Matrix{Float64})

  Compute closed-shell DF-HF Fock matrix in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo, cMO::Matrix{Float64})
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
    gen_dffock(EC::ECInfo, cMO::Matrix{Float64}, cPO::Matrix{Float64})

  Compute closed-shell DF-HF Fock matrix and the positron
  Fock matrix in AO basis  (using precalculated Cholesky-
  decomposed integrals and density matrices).
"""
function gen_dffock(EC::ECInfo, cMO::Matrix{Float64}, cPO::Matrix{Float64})
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
function gen_dffock(EC::ECInfo, cMO::SpinMatrix)
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
  @buffer buf((nocc+nOcc)*nA*maxL + maxL) begin
  coulfock = zeros(nA, nA)
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

end #module
