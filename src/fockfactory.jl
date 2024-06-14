""" Fock builders (using FciDump or DF integrals) """
module FockFactory
try
  using MKL
catch
  #println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
#BLAS.set_num_threads(1)
using TensorOperations
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.Wavefunctions
using ..ElemCo.FciDump
using ..ElemCo.Integrals
using ..ElemCo.OrbTools

export gen_fock, gen_ufock, gen_dffock
export gen_density_matrix, gen_frac_density_matrix

""" 
    gen_fock(EC::ECInfo)

  Calculate closed-shell fock matrix from FCIDump integrals. 
"""
function gen_fock(EC::ECInfo)
  @tensoropt fock[p,q] := integ1(EC.fd,:α)[p,q] + 2.0*ints2(EC,":o:o",:α)[p,i,q,i] - ints2(EC,":oo:",:α)[p,i,i,q]
  return fock
end

""" 
    gen_fock(EC::ECInfo, spincase::Symbol)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`}. 
"""
function gen_fock(EC::ECInfo, spincase::Symbol)
  @tensoropt fock[p,q] := integ1(EC.fd,spincase)[p,q] 
  if spincase == :α
    if n_occb_orbs(EC) > 0 
      @tensoropt fock[p,q] += ints2(EC,":O:O",:αβ)[p,i,q,i]
    end
    spo='o'
    spv='v'
    nocc = n_occ_orbs(EC)
  else
    if n_occ_orbs(EC) > 0 
      @tensoropt fock[p,q] += ints2(EC,"o:o:",:αβ)[i,p,i,q]
    end
    spo='O'
    spv='V'
    nocc = n_occb_orbs(EC)
  end
  if nocc > 0
    @tensoropt begin
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
  @tensoropt den[r,s] := CMOlo[r,i]*CMOro[s,i]
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
  @tensoropt den[r,s] := CMOl[r,i]*CMOrn[s,i]
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
  @tensoropt begin 
    fock[p,q] := integ1(EC.fd,:α)[p,q] 
    fock[p,q] += ints2(EC,"::::",:α)[p,r,q,s] * den[r,s]
    fock[p,q] -= 0.5*ints2(EC,"::::",:α)[p,r,s,q] * den[r,s]
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
  @tensoropt begin 
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
    @tensoropt fock[p,q] := ints2(EC,"::::",:αβ)[p,r,q,s]*denOS[r,s]
    spo = 'o'
  else
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['o'])
    @tensoropt fock[p,q] := ints2(EC,"::::",:αβ)[r,p,s,q]*denOS[r,s]
    spo = 'O'
  end
  den =  gen_density_matrix(EC, CMOl, CMOr, EC.space[spo])
  ints = ints2(EC,"::::",spincase)
  @tensoropt fock[p,q] += ints[p,r,q,s] * den[r,s] 
  @tensoropt fock[p,q] -= ints[p,r,s,q] * den[r,s]
  @tensoropt fock[p,q] += integ1(EC.fd,spincase)[p,q] 
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
    @tensoropt fock[p,q] := ints2(EC,"::::",:αβ)[p,r,q,s]*denOS[r,s]
  else
    @tensoropt fock[p,q] := ints2(EC,"::::",:αβ)[r,p,s,q]*denOS[r,s]
  end
  ints = ints2(EC,"::::",spincase)
  @tensoropt fock[p,q] += ints[p,r,q,s] * den[r,s] 
  @tensoropt fock[p,q] -= ints[p,r,s,q] * den[r,s]
  @tensoropt fock[p,q] += integ1(EC.fd,spincase)[p,q] 
  return fock
end

""" 
    gen_ufock(EC::ECInfo, CMOl::MOs, CMOr::MOs)

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
  AAP = eri_2e3idx(bao,bfit)
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  # println(size(Ppq))
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  @tensoropt begin 
    AoP[p,j,P] := AAP[p,q,P] * CMO2[q,j]
    c_AoL[p,j,L] := AoP[p,j,P] * PL[P,L]
    cL[L] := c_AoL[p,j,L] * CMO2[p,j]
    fock[p,q] := hsmall[p,q] - c_AoL[p,j,L]*c_AoL[q,j,L]
    cP[P] := cL[L] * PL[P,L]
    fock[p,q] += 2.0*cP[P]*AAP[p,q,P]
  end
  return fock
end

""" 
    gen_dffock(EC::ECInfo, cMO::MOs, bao, bfit)

  Compute unrestricted DF-HF Fock matrices `SpinMatrix(Fα, Fβ)` in AO basis (integral direct).
"""
function gen_dffock(EC::ECInfo, cMO::SpinMatrix, bao, bfit)
  AAP = eri_2e3idx(bao, bfit)
  PL = load2idx(EC, "C_PL")
  hsmall = load2idx(EC, "h_AA")
  # println(size(Ppq))
  occa = EC.space['o']
  occb = EC.space['O']
  CMOo = SpinMatrix(cMO[1][:,occa], cMO[2][:,occb])
  fock = SpinMatrix(hsmall)
  unrestrict!(fock)
  cL = zeros(size(PL,2))
  for isp = 1:2 # loop over [α, β]
    @tensoropt begin 
      AoP[p,j,P] := AAP[p,q,P] * CMOo[isp][q,j]
      c_AoL[p,j,L] := AoP[p,j,P] * PL[P,L]
      cL[L] += c_AoL[p,j,L] * CMOo[isp][p,j]
      fock[isp][p,q] -= c_AoL[p,j,L]*c_AoL[q,j,L]
    end
  end
  @tensoropt begin
    cP[P] := cL[L] * PL[P,L]
    coulfock[p,q] := cP[P] * AAP[p,q,P]
  end
  fock[1] += coulfock
  fock[2] += coulfock
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
  μνL = load3idx(EC,"AAL")
  hsmall = load2idx(EC,"h_AA")
  @tensoropt begin 
    μjL[p,j,L] := μνL[p,q,L] * CMO2[q,j]
    L[L] := μjL[p,j,L] * CMO2[p,j]
    fock[p,q] := hsmall[p,q] - μjL[p,j,L]*μjL[q,j,L]
    fock[p,q] += 2.0*L[L]*μνL[p,q,L]
  end
  return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::MOs)

  Compute unrestricted DF-HF Fock matrices [Fα, Fβ] in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo, cMO::SpinMatrix)
  occa = EC.space['o']
  occb = EC.space['O']
  CMOo = [cMO[1][:,occa], cMO[2][:,occb]]
  hsmall = load2idx(EC,"h_AA")
  fock = SpinMatrix(hsmall)
  unrestrict!(fock)
  μνL = load3idx(EC,"AAL")
  L = zeros(size(μνL,3))
  for isp = 1:2 # loop over [α, β]
    @tensoropt begin 
      μjL[p,j,L] := μνL[p,q,L] * CMOo[isp][q,j]
      fock[isp][p,q] -= μjL[p,j,L]*μjL[q,j,L]
      L[L] += μjL[p,j,L] * CMOo[isp][p,j]
    end
  end
  @tensoropt coulfock[p,q] := L[L] * μνL[p,q,L]
  fock[1] += coulfock
  fock[2] += coulfock
  return fock
end

end #module
