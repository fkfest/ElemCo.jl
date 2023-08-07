""" Fock builders (using FciDump integrals) """
module Focks
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

export gen_fock, gen_ufock, gen_density_matrix

""" 
    gen_fock(EC::ECInfo)

  Calculate closed-shell fock matrix from FCIDump integrals. 
"""
function gen_fock(EC::ECInfo)
  @tensoropt fock[p,q] := integ1(EC.fd,SCα)[p,q] + 2.0*ints2(EC,":o:o",SCα)[p,i,q,i] - ints2(EC,":oo:",SCα)[p,i,i,q]
  ϵ = diag(fock)
  ϵo = ϵ[EC.space['o']]
  ϵv = ϵ[EC.space['v']]
  println("Occupied orbital energies: ",ϵo)
  return fock, ϵo, ϵv
end

""" 
    gen_fock(EC::ECInfo, spincase::SpinCase)

  Calculate UHF fock matrix from FCIDump integrals. 
"""
function gen_fock(EC::ECInfo, spincase::SpinCase)
  @tensoropt fock[p,q] := integ1(EC.fd,spincase)[p,q] 
  if spincase == SCα
    if EC.noccb > 0 
      @tensoropt fock[p,q] += ints2(EC,":O:O",SCαβ)[p,i,q,i]
    end
    spo='o'
    spv='v'
    spin = "α"
    nocc = EC.nocc
  else
    if EC.nocc > 0 
      @tensoropt fock[p,q] += ints2(EC,"o:o:",SCαβ)[i,p,i,q]
    end
    spo='O'
    spv='V'
    spin = "β"
    nocc = EC.noccb
  end
  if nocc > 0
    @tensoropt begin
      fock[p,q] += ints2(EC,":"*spo*":"*spo,spincase)[p,i,q,i]
      fock[p,q] -= ints2(EC,":"*spo*spo*":",spincase)[p,i,i,q]
    end
  end
  ϵ = diag(fock)
  ϵo = ϵ[EC.space[spo]]
  ϵv = ϵ[EC.space[spv]]
  println("Occupied $spin orbital energies: ",ϵo)
  return fock, ϵo, ϵv
end

""" 
    gen_density_matrix(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray, occvec)

  Generate D_{μν}=C^l_{μi} C^r_{νi} with i defined by occvec
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
    fock[p,q] := integ1(EC.fd,SCα)[p,q] 
    fock[p,q] += 2.0*ints2(EC,"::::",SCα)[p,r,q,s] * den[r,s]
    fock[p,q] -= ints2(EC,"::::",SCα)[p,r,s,q] * den[r,s]
  end
  return fock
end

""" 
    gen_fock(EC::ECInfo, spincase::SpinCase, CMOl::AbstractArray, CMOr::AbstractArray)

  Calculate UHF fock matrix from FCIDump integrals for `spincase` and orbitals `CMOl`, `CMOr` and
  orbitals for the opposite-spin `CMOlOS` and `CMOrOS`. 
"""
function gen_fock(EC::ECInfo, spincase::SpinCase, CMOl::AbstractArray, CMOr::AbstractArray,
                  CMOlOS::AbstractArray, CMOrOS::AbstractArray)
  if spincase == SCα
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['O'])
    @tensoropt fock[p,q] := ints2(EC,"::::",SCαβ)[p,r,q,s]*denOS[r,s]
    spo = 'o'
  else
    denOS = gen_density_matrix(EC, CMOlOS, CMOrOS, EC.space['o'])
    @tensoropt fock[p,q] := ints2(EC,"::::",SCαβ)[r,p,s,q]*denOS[r,s]
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
  return [gen_fock(EC,SCα, cMOl[1],cMOr[1], cMOl[2],cMOr[2]), gen_fock(EC,SCβ, cMOl[2],cMOr[2], cMOl[1],cMOr[1])]
end

end #module
