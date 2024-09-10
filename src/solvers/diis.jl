"""
  DIIS module for iterative solvers

  This module provides the DIIS (Direct Inversion in the Iterative Subspace) method for iterative solvers.

  The DIIS method is used to accelerate the convergence of iterative solvers by combining 
  previous solutions to the problem to minimize the residual.
  The vectors and residuals are stored in files as `Vector{Vector{Float64}}`.

# Usage
```julia
diis = Diis(EC)
for it = 1:maxit
  # compute Vec = [Vec1,Vec2,...] and Res = [Res1,Res2,...]
  # ...
  perform!(diis, Vec, Res)
  # ...
end
```

One can also provide a tuple of custom dot-product functions for the residuals components
as `customdots` argument in [`perform!`](@ref) function.
"""
module DIIS
using LinearAlgebra
using ..ElemCo.MIO
using ..ElemCo.ECInfos

export Diis, perform!

"""
  DIIS object
"""
mutable struct Diis
  """ maximum number of DIIS vectors """
  maxdiis::Int
  """ threshold for residual norm to start DIIS """
  resthr::Float64
  """ use CROP-DIIS instead of the standard DIIS """
  cropdiis::Bool
  """ files for DIIS vectors """
  ampfiles::Vector{String}
  """ files for DIIS residuals """
  resfiles::Vector{String}
  """ square weights for DIIS residuals components """
  weights::Vector{Float64}
  """ next vector to be replaced """
  next::Int
  """ number of DIIS vectors """
  nDim::Int
  """ B matrix """
  bmat::Matrix{Float64}
  """
    Diis(EC::ECInfo, weights = Float64[]; maxdiis::Int = EC.options.diis.maxdiis, resthr::Float64 = EC.options.diis.resthr)
  
  Create DIIS object. `weights` is an array of square weights for DIIS residuals components.
  """
  function Diis(EC::ECInfo, weights = Float64[]; 
                maxdiis::Int = -1, resthr::Float64 = EC.options.diis.resthr,
                cropdiis::Bool = EC.options.diis.crop)
    if maxdiis < 0
      maxdiis = cropdiis ? EC.options.diis.maxcrop : EC.options.diis.maxdiis
    end
    ampfiles = [ joinpath(EC.scr, "amp"*string(i)*EC.ext) for i in 1:maxdiis ]
    resfiles = [ joinpath(EC.scr, "res"*string(i)*EC.ext) for i in 1:maxdiis ]
    for i in 1:maxdiis
      add_file!(EC, "amp"*string(i), "tmp", overwrite=true)
      add_file!(EC, "res"*string(i), "tmp", overwrite=true)
    end
    new(maxdiis,resthr,cropdiis,ampfiles,resfiles,weights,1,0,zeros(maxdiis+1,maxdiis+1))
  end
end

"""
    saveamps(diis::Diis, vecs, ipos)

  Save vectors to file (replacing previous vectors at position `ipos`).
"""
function saveamps(diis::Diis, vecs, ipos)
  miosave(diis.ampfiles[ipos], vecs...)
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

  Load vectors from file as `Vector{Vector{Float64}}`.
"""
function loadvecs(file)
  return mioload(file, Val(1))
end

"""
    loadamps(diis::Diis, ipos)

  Load vectors from file at position `ipos` as `Vector{Vector{Float64}}`.
"""
function loadamps(diis::Diis, ipos)
  return loadvecs(diis.ampfiles[ipos])
end

"""
    loadres(diis::Diis, ipos)

  Load residuals from file at position `ipos` as `Vector{Vector{Float64}}`.
"""
function loadres(diis::Diis, ipos)
  return loadvecs(diis.resfiles[ipos])
end

"""
    tuple_reshape(vecs, tens)

  Reshape vectors `vecs` to the shape of tensors `tens`.

  Returns `Tuple` of reshaped vectors.
"""
function tuple_reshape(vecs, tens)
  return Tuple([ reshape(vecs[i], size(tens[i])) for i in eachindex(tens) ])
end

"""
    combine(diis::Diis, vecfiles, coeffs)

  Combine vectors from files with coefficients.
"""
function combine(diis::Diis, vecfiles, coeffs)
  outvecs = loadvecs(vecfiles[1])
  for v in outvecs
    v .*= coeffs[1]
  end
  for i in 2:diis.nDim
    vect = loadvecs(vecfiles[i])
    coef = coeffs[i]
    for j in eachindex(vect)
      outvecs[j] .+= coef * vect[j]
    end
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
  dot::Float64 = 0.0
  for i in eachindex(vecs1)
    dot += diis.weights[i] * (vec(vecs1[i]) ⋅ vec(vecs2[i]))
  end
  return dot
end

"""
    custom_dot(diis::Diis, customdots, tens, vecs)

  Compute weighted (with diis.weights) dot product of vectors
  using custom dot-product functions `customdots::Tuple`.
  `vecs` are reshaped to the shape of tensors `tens`.
"""
function custom_dot(diis::Diis, customdots, tens, vecs)
  if length(diis.weights) == 0
    weights = ones(length(tens))
  else
    weights = diis.weights
    @assert length(tens) == length(weights)
  end
  @assert length(tens) == length(customdots)
  @assert length(tens) == length(vecs)
  dot::Float64 = 0.0
  for i in eachindex(tens)
    # f = customdots[i]
    dot += weights[i] * dispatch(customdots[i], tens[i], vecs[i])
  end
  return dot
end

dispatch(f::Function, t, v) = f(t, reshape(v, size(t)))::Float64

@doc raw"""
    update_Bmat(diis::Diis, nDim, Res, ithis, customdots=())

  Update B matrix with new residual (at the position `ithis`).

  B matrix is defined as:
```math
{\bf B} = \begin{pmatrix}
\langle {\bf R}_1, {\bf R}_1 \rangle & \langle {\bf R}_1, {\bf R}_2 \rangle & \cdots & \langle {\bf R}_1, {\bf R}_{\rm nDim} \rangle & -1 \\
\langle {\bf R}_2, {\bf R}_1 \rangle & \langle {\bf R}_2, {\bf R}_2 \rangle & \cdots & \langle {\bf R}_2, {\bf R}_{\rm nDim} \rangle & -1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\langle {\bf R}_{\rm nDim}, {\bf R}_1 \rangle & \langle {\bf R}_{\rm nDim}, {\bf R}_2 \rangle & \cdots & \langle {\bf R}_{\rm nDim}, {\bf R}_{\rm nDim} \rangle & -1 \\
-1 & -1 & \cdots & -1 & 0
\end{pmatrix}
```
  Returns the dot product of the new residual with itself, ``\langle {\bf R}_{\rm ithis}, {\bf R}_{\rm ithis} \rangle``.
"""
function update_Bmat(diis::Diis, nDim, Res, ithis, customdots=())
  if length(customdots) == 0
    thisResDot = weighted_dot(diis, Res, Res)
  else
    thisResDot = custom_dot(diis, customdots, Res, Res)
  end
  for i in 1:nDim
    if i != ithis
      resi = loadres(diis, i)
      if length(customdots) == 0
        dot = weighted_dot(diis, Res, resi)
      else
        dot = custom_dot(diis, customdots, Res, resi)
      end
      diis.bmat[i,ithis] = dot
      diis.bmat[ithis,i] = dot
    else
      diis.bmat[ithis,ithis] = thisResDot
    end
    diis.bmat[i,nDim+1] = -1.0
    diis.bmat[nDim+1,i] = -1.0
  end
  diis.bmat[nDim+1,nDim+1] = 0.0
  return thisResDot
end

"""
    perform!(diis::Diis, Amps, Res, customdots=())

  Perform DIIS.

  `Amps` is an array of vectors and `Res` is an array of residuals.
  The vectors `Amps` will be replaced by the DIIS optimized vectors.
  `customdots` is a tuple of functions for each residual component to calculate
  the dot-product. The functions should have the signature `f(ten1::Array{T,N}, ten2::Array{T,N})`.
"""
function perform!(diis::Diis, Amps, Res, customdots=())
  if diis.nDim < diis.maxdiis
    diis.nDim += 1
  end
  ithis = diis.next
  nDim = diis.nDim
  saveamps(diis, Amps, ithis)
  saveres(diis, Res, ithis)
  thisResDot = update_Bmat(diis, nDim, Res, ithis, customdots)
  rhs = zeros(nDim+1)
  rhs[nDim+1] = -1.0

  bmat = diis.bmat[1:nDim+1,1:nDim+1]
  # display(bmat)
  # display(rhs)
  coeffs = svd(bmat)\rhs
  # print("coeffs: ")
  # display(coeffs)

  if nDim == 1 && diis.next == 1 && thisResDot > diis.resthr
    # very bad residual, wait with diis...
    diis.nDim = 0
    return Amps
  elseif nDim < diis.maxdiis
    diis.next = nDim+1
  elseif diis.cropdiis
    # the oldest vector is replaced
    diis.next += 1
    if diis.next > diis.maxdiis
      diis.next = 1
    end
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
  if diis.cropdiis
    Opt = combine(diis, diis.resfiles, coeffs)
    saveres(diis, Opt, ithis)
    # reshape Opt to the shape of Res
    Optr = tuple_reshape(Opt, Res)
    optres2 = update_Bmat(diis, nDim, Optr, ithis, customdots)
    Opt = Optr = nothing
    # println("DIIS: ", thisResDot, " -> ", optres2)
    Opt = combine(diis, diis.ampfiles, coeffs)
    saveamps(diis, Opt, ithis)
  else
    Opt = combine(diis, diis.ampfiles, coeffs)
  end
  # replace Amps with Opt vectors keeping the shape of Amps
  for i in eachindex(Amps)
    Amps[i][:] = Opt[i]
  end
  return Amps
end

end
