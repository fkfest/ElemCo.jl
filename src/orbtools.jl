module OrbTools
using LinearAlgebra, TensorOperations

using ..ElemCo.ECInfos
using ..ElemCo.ECInts
using ..ElemCo.MSystem
using ..ElemCo.TensorTools

export guess_orb

"""
    guess_hcore(EC::ECInfo)

  Guess MO coefficients from core Hamiltonian.
"""
function guess_hcore(EC::ECInfo)
  hsmall = load(EC,"h_AA")
  sao = load(EC,"S_AA")
  ϵ,cMO = eigen(Hermitian(hsmall),Hermitian(sao))
  return cMO
end
  
"""
    guess_sad(EC::ECInfo)
  
  Guess MO coefficients from atomic densities.
"""
function guess_sad(EC::ECInfo)
  # minao = "ano-rcc-mb"
  minao = "ano-r0"
  # minao = "sto-6g"
  bminao = BasisSet(minao,genxyz(EC.ms,bohr=false))
  bao = generate_basis(EC.ms, "ao")
  smin2ao = overlap(bminao,bao)
  smin = overlap(bminao)
  eldist = electron_distribution(EC.ms,minao)
  sao = load(EC,"S_AA")
  denao = smin2ao' * diagm(eldist./diag(smin)) * smin2ao
  eigs,cMO = eigen(Hermitian(-denao),Hermitian(sao))
  return cMO
end

function guess_gwh(EC::ECInfo)
  error("not implemented yet")
end

"""
    guess_orb(EC::ECInfo, guess::Symbol)

  Calculate starting guess for MO coefficients.
  Type of initial guess for MO coefficients is given by `guess`.

  Possible values:
  - :HCORE from core Hamiltonian
  - :SAD from atomic densities
  - :GWH not implemented yet
  - :ORB from previous orbitals stored in file `EC.options.scf.orbsguess`
"""
function guess_orb(EC::ECInfo, guess::Symbol)
  if guess == :HCORE || guess == :hcore
    return guess_hcore(EC)
  elseif guess == :SAD || guess == :sad
    return guess_sad(EC)
  elseif guess == :GWH || guess == :gwh
    return guess_gwh(EC)
  elseif guess == :ORB || guess == :orb
    return load(EC,EC.options.scf.orbsguess)
  else
    error("unknown guess type")
  end
end


end #module