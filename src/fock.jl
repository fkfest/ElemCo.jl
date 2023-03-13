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
using ..ECInfos
using ..TensorTools
using ..FciDump

export gen_fock

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


end #module
