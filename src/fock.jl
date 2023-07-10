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

export gen_fock, gen_ufock

""" calc closed-shell fock matrix """
function gen_fock(EC::ECInfo)
  @tensoropt fock[p,q] := integ1(EC.fd,SCα)[p,q] + 2.0*ints2(EC,":o:o",SCα)[p,i,q,i] - ints2(EC,":oo:",SCα)[p,i,i,q]
  ϵ = diag(fock)
  ϵo = ϵ[EC.space['o']]
  ϵv = ϵ[EC.space['v']]
  println("Occupied orbital energies: ",ϵo)
  return fock, ϵo, ϵv
end

""" calc uhf fock matrix """
function gen_fock(EC::ECInfo, spincase::SpinCase)
  @tensoropt fock[p,q] := integ1(EC.fd,spincase)[p,q] 
  if spincase == SCα
    @tensoropt fock[p,q] += ints2(EC,":O:O",SCαβ)[p,i,q,i]
    spo='o'
    spv='v'
    spin = "α"
  else
    @tensoropt fock[p,q] += ints2(EC,"o:o:",SCαβ)[i,p,i,q]
    spo='O'
    spv='V'
    spin = "β"
  end
  @tensoropt begin
    fock[p,q] += ints2(EC,":"*spo*":"*spo,spincase)[p,i,q,i]
    fock[p,q] -= ints2(EC,":"*spo*spo*":",spincase)[p,i,i,q]
  end
  ϵ = diag(fock)
  ϵo = ϵ[EC.space[spo]]
  ϵv = ϵ[EC.space[spv]]
  println("Occupied $spin orbital energies: ",ϵo)
  return fock, ϵo, ϵv
end

""" calc fock matrix for non-fcidump orbitals """
function gen_fock(EC::ECInfo, CMOl::AbstractArray, CMOr::AbstractArray)
  occ2 = EC.space['o']
  @assert EC.space['o'] == EC.space['O'] # closed-shell
  CMOl2 = CMOl[:,occ2]
  CMOr2 = CMOr[:,occ2]
  @tensoropt begin 
    fock[p,q] := integ1(EC.fd,SCα)[p,q] 
    fock[p,q] += 2.0*ints2(EC,"::::",SCα)[p,r,q,s] * (CMOl2[r,i]*CMOr2[s,i])
    fock[p,q] -= ints2(EC,"::::",SCα)[p,r,s,q] * CMOl2[r,i]*CMOr2[s,i]
  end
  return fock
end
""" calc UHF fock matrix for non-fcidump orbitals """
function gen_fock(EC::ECInfo, spincase::SpinCase, CMOl::AbstractArray, CMOr::AbstractArray,
                  CMOlOS::AbstractArray, CMOrOS::AbstractArray)
  if spincase == SCα
    CMOlOSo = CMOlOS[:,EC.space['O']]
    CMOrOSo = CMOrOS[:,EC.space['O']]
    @tensoropt fock[p,q] := ints2(EC,"::::",SCαβ)[p,r,q,s]*(CMOlOSo[r,i]*CMOrOSo[s,i])
    spo = 'o'
  else
    CMOlOSo = CMOlOS[:,EC.space['o']]
    CMOrOSo = CMOrOS[:,EC.space['o']]
    @tensoropt fock[p,q] := ints2(EC,"::::",SCαβ)[r,p,s,q]*(CMOlOSo[r,i]*CMOrOSo[s,i])
    spo = 'O'
  end
  occ = EC.space[spo]
  CMOlo = CMOl[:,occ]
  CMOro = CMOr[:,occ]
  @tensoropt begin 
    fock[p,q] += integ1(EC.fd,spincase)[p,q] 
    fock[p,q] += ints2(EC,"::::",spincase)[p,r,q,s] * (CMOlo[r,i]*CMOro[s,i])
    fock[p,q] -= ints2(EC,"::::",spincase)[p,r,s,q] * CMOlo[r,i]*CMOro[s,i]
  end
  return fock
end

function gen_ufock(EC::ECInfo, cMOl::AbstractArray, cMOr::AbstractArray)
  return [gen_fock(EC,SCα, cMOl[1],cMOr[1], cMOl[2],cMOr[2]), gen_fock(EC,SCβ, cMOl[2],cMOr[2], cMOl[1],cMOr[1])]
end

end #module
