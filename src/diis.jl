"""
  DIIS module for iterative solvers
"""
module DIIS
using LinearAlgebra
using ..ElemCo.MIO
using ..ElemCo.ECInfos

export Diis, perform

"""
  DIIS object
"""
mutable struct Diis
  """ maximum number of DIIS vectors """
  maxdiis::Int
  """ threshold for residual norm to start DIIS """
  resthr::Float64
  """ files for DIIS vectors """
  ampfiles::Array{String}
  """ files for DIIS residuals """
  resfiles::Array{String}
  """ square weights for DIIS residuals components """
  weights::Array{Float64}
  """ next vector to be replaced """
  next::Int
  """ number of DIIS vectors """
  nDim::Int
  """ B matrix """
  bmat::Array{Float64}
  """
    Diis(EC::ECInfo, weights = Float64[]; maxdiis::Int = EC.options.diis.maxdiis, resthr::Float64 = EC.options.diis.resthr)
  
  Create DIIS object. `weights` is an array of square weights for DIIS residuals components.
  """
  function Diis(EC::ECInfo, weights = Float64[]; maxdiis::Int = EC.options.diis.maxdiis, resthr::Float64 = EC.options.diis.resthr)
    ampfiles = [ joinpath(EC.scr, "amp"*string(i)*EC.ext) for i in 1:maxdiis ]
    resfiles = [ joinpath(EC.scr, "res"*string(i)*EC.ext) for i in 1:maxdiis ]
    for i in 1:maxdiis
      add_file!(EC, "amp"*string(i), "tmp", overwrite=true)
      add_file!(EC, "res"*string(i), "tmp", overwrite=true)
    end
    new(maxdiis,resthr,ampfiles,resfiles,weights,1,0,zeros(maxdiis+1,maxdiis+1))
  end
end

"""
    saveamps(diis::Diis, vecs, ipos)

  Save vectors to file (replacing previous vectors at position `ipos`).
"""
function saveamps(diis::Diis, vecs, ipos)
  miosave(diis.ampfiles[ipos],vecs...)
end

"""
    saveres(diis::Diis, vecs, ipos)

  Save residuals to file (replacing previous residuals at position `ipos`).
"""
function saveres(diis::Diis, vecs, ipos)
  miosave(diis.resfiles[ipos], vecs...)
end

"""
    loadvecs(file)

  Load vectors from file.
"""
function loadvecs(file)
  return mioload(file, array_of_arrays = true)
end

"""
    loadamps(diis::Diis, ipos)

  Load vectors from file at position `ipos`.
"""
function loadamps(diis::Diis, ipos)
  return loadvecs(diis.ampfiles[ipos])
end

"""
    loadres(diis::Diis, ipos)

  Load residuals from file at position `ipos`.
"""
function loadres(diis::Diis, ipos)
  return loadvecs(diis.resfiles[ipos])
end

"""
    combine(diis::Diis, vecfiles, coeffs)

  Combine vectors from files with coefficients.
"""
function combine(diis::Diis, vecfiles, coeffs)
  outvecs = coeffs[1] * loadvecs(vecfiles[1])
  for i in 2:diis.nDim
    outvecs += coeffs[i] * loadvecs(vecfiles[i])
  end
  return outvecs
end

"""
    weighted_dot(diis::Diis, vecs1, vecs2)

  Compute weighted (with diis.weights) dot product of vectors.
"""
function weighted_dot(diis::Diis, vecs1, vecs2)
  if length(diis.weights) == 0
    return vecs1 ⋅ vecs2
  end
  @assert length(vecs1) == length(diis.weights)
  dot = 0.0
  for i in eachindex(vecs1)
    dot += diis.weights[i] * (vecs1[i] ⋅ vecs2[i])
  end
  return dot
end

"""
    perform(diis::Diis, Amps, Res)

  Perform DIIS.
"""
function perform(diis::Diis, Amps, Res)
  if diis.nDim < diis.maxdiis
    diis.nDim += 1
  end
  ithis = diis.next
  nDim = diis.nDim
  saveamps(diis,Amps,ithis)
  saveres(diis,Res,ithis)
  thisResDot = weighted_dot(diis, Res, Res)
  # update B matrix
  for i in 1:nDim
    if i != ithis
      resi = loadres(diis,i)
      dot = weighted_dot(diis, Res, resi)
      diis.bmat[i,ithis] = dot
      diis.bmat[ithis,i] = dot
    else
      diis.bmat[ithis,ithis] = thisResDot
    end
    diis.bmat[i,nDim+1] = -1.0
    diis.bmat[nDim+1,i] = -1.0
  end
  diis.bmat[nDim+1,nDim+1] = 0.0
  rhs = zeros(nDim+1)
  rhs[nDim+1] = -1.0

  bmat = diis.bmat[1:nDim+1,1:nDim+1]
  # display(bmat)
  # display(rhs)
  # TODO use svd?
  coeffs = bmat\rhs
  # print("coeffs: ")
  # display(coeffs)

  if nDim == 1 && diis.next == 1 && thisResDot > diis.resthr
    # very bad residual, wait with diis...
    diis.nDim = 0
    return Amps
  elseif nDim < diis.maxdiis
    diis.next = nDim+1
  else
    # vector with the smallest coef
    coefmin = 1.e10
    for i in 1:nDim
      if abs(coeffs[i]) < coefmin && ( i != ithis || nDim == 1 )
        coefmin = abs(coeffs[i])
        diis.next = i
      end
    end
  end
  return combine(diis,diis.ampfiles,coeffs)
end

end