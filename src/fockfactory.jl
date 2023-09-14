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
using ..ElemCo.TensorTools
using ..ElemCo.FciDump
using ..ElemCo.ECInts

export gen_fock, gen_ufock, gen_dffock, gen_density_matrix

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
    gen_density_matrix(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray, occvec)

  Generate ``D_{μν}=C^l_{μi} C^r_{νi}`` with i defined by occvec
  Only real part of D is kept.
""" 
function gen_density_matrix(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray, occvec)
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
    gen_fock(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray)

  Calculate closed-shell fock matrix from FCIDump integrals and orbitals `CMOl`, `CMOr`. 
"""
function gen_fock(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray)
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
    gen_fock(EC::ECInfo, spincase::Symbol, CMOl::AbstractArray, CMOr::AbstractArray)

  Calculate UHF fock matrix from FCIDump integrals for `spincase`∈{`:α`,`:β`} and orbitals `CMOl`, `CMOr` and
  orbitals for the opposite-spin `CMOlOS` and `CMOrOS`. 
"""
function gen_fock(EC::ECInfo, spincase::Symbol, CMOl::AbstractArray, CMOr::AbstractArray,
                  CMOlOS::AbstractArray, CMOrOS::AbstractArray)
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
    gen_ufock(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray)

  Calculate UHF fock matrix from FCIDump integrals and orbitals `cMOl`, `cMOr`
  with `cMOl[1]` and `cMOr[1]` - α-MO transformation coefficients and 
  `cMOl[2]` and `cMOr[2]` - β-MO transformation coefficients. 
"""
function gen_ufock(EC::ECInfo, cMOl::AbstractArray, cMOr::AbstractArray)
  return [gen_fock(EC, :α, cMOl[1], cMOr[1], cMOl[2], cMOr[2]), gen_fock(EC, :β, cMOl[2], cMOr[2], cMOl[1], cMOr[1])]
end

""" 
    gen_dffock(EC::ECInfo, cMO::AbstractArray, bao, bfit)

  Compute closed-shell DF-HF Fock matrix (integral direct) in AO basis.
"""
function gen_dffock(EC::ECInfo, cMO::AbstractArray, bao, bfit)
  @assert ndims(cMO) == 2 "Restricted orbitals only!"
  μνL = ERI_2e3c(bao,bfit)
  PL = load(EC,"C_PL")
  hsmall = load(EC,"h_AA")
  # println(size(Ppq))
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  @tensoropt begin 
    μjP[p,j,P] := μνL[p,q,P] * CMO2[q,j]
    cμjL[p,j,L] := μjP[p,j,P] * PL[P,L]
    cL[L] := cμjL[p,j,L] * CMO2[p,j]
    fock[p,q] := hsmall[p,q] - cμjL[p,j,L]*cμjL[q,j,L]
    cP[P] := cL[L] * PL[P,L]
    fock[p,q] += 2.0*cP[P]*μνL[p,q,P]
  end
  return fock
end

"""
    gen_dffock(EC::ECInfo, cMO::AbstractArray)

  Compute closed-shell DF-HF Fock matrix in AO basis
  (using precalculated Cholesky-decomposed integrals).
"""
function gen_dffock(EC::ECInfo, cMO::AbstractArray)
  @assert ndims(cMO) == 2 "Restricted orbitals only!"
  @assert EC.space['o'] == EC.space['O'] "Closed-shell only!"
  occ2 = EC.space['o']
  CMO2 = cMO[:,occ2]
  μνL = load(EC,"AAL")
  hsmall = load(EC,"h_AA")
  @tensoropt begin 
    μjL[p,j,L] := μνL[p,q,L] * CMO2[q,j]
    L[L] := μjL[p,j,L] * CMO2[p,j]
    fock[p,q] := hsmall[p,q] - μjL[p,j,L]*μjL[q,j,L]
    fock[p,q] += 2.0*L[L]*μνL[p,q,L]
  end
  return fock
end

end #module
