
"""
    Davidson solver module
"""
module DavidsonSolver
using LinearAlgebra
using ..ElemCo.MIO
using ..ElemCo.ECInfos

export Davidson, perform!
export addvecs!, diagonalize, orthogonalize!, dav_normalize!

"""
  Davidson object
"""
mutable struct Davidson
    """ maximum number of trial vectors """
  maxdav::Int
  """ number of states """
  nstates::Int
  """ files for trial vectors """
  tvecfiles::Vector{String}
  """ files for product vectors """
  prodfiles::Vector{String}
  """ size of the Davidson space 
  (i.e., number of trial vectors with corresponding product vectors) """
  nDim::Int
  """ number of added trial vectors (≥ `nDim`) """
  nDimTrial::Int
  """ effective matrix """
  hmat::Matrix{Float64}
  """ overlap matrix (length=0 for hermitian Davidson algorithm) """
  smat::Matrix{Float64}
  """
    Davidson(EC::ECInfo, nstates=1; maxdav::Int = EC.options.davidson.maxdav, hermitian::Bool = true)
  
  Create Davidson object for `nstates` states.
  """
  function Davidson(EC::ECInfo, nstates=1; maxdav::Int = -1, hermitian::Bool = true)
    if maxdav < 0
      maxdav = EC.options.davidson.maxdav
    end
    maxdav_tot = maxdav*nstates
    tvecfiles = [ joinpath(EC.scr, "tvec"*string(i)*EC.ext) for i in 1:maxdav_tot ]
    prodfiles = [ joinpath(EC.scr, "prod"*string(i)*EC.ext) for i in 1:maxdav_tot ]
    for i in 1:maxdav_tot
      add_file!(EC, "tvec"*string(i), "tmp", overwrite=true)
      add_file!(EC, "prod"*string(i), "tmp", overwrite=true)
    end
    if hermitian
      new(maxdav, nstates, tvecfiles, prodfiles, 0, 0, zeros(maxdav_tot,maxdav_tot), zeros(0,0))
    else
      new(maxdav, nstates, tvecfiles, prodfiles, 0, 0, zeros(maxdav_tot,maxdav_tot), zeros(maxdav_tot,maxdav_tot))
    end
  end
end

"""
    hermitian(dav::Davidson)

  Check if Davidson object is hermitian.
"""
hermitian(dav::Davidson) = length(dav.smat) == 0

"""
    savetvecs(dav::Davidson, vecs)
  
  Save trial vectors to next file.
"""
function savetvecs(dav::Davidson, vecs)
  miosave(dav.tvecfiles[dav.nDimTrial], vecs...)
end

"""
    saveprods(dav::Davidson, vecs)
  
  Save vectors to next file.
"""
function saveprods(dav::Davidson, vecs)
  miosave(dav.prodfiles[dav.nDim], vecs...)
end

"""
    loadvecs(file)

  Load vectors from file as `Vector{Vector{Float64}}`.
"""
function loadvecs(file)
  return mioload(file, Val(1))
end

"""
    loadtvecs(dav::Davidson, ipos)

  Load trial vectors from file at position `ipos` as `Vector{Vector{Float64}}`.
"""
function loadtvecs(dav::Davidson, ipos)
  return loadvecs(dav.tvecfiles[ipos])
end

"""
    loadprods(dav::Davidson, ipos)

  Load product vectors from file at position `ipos` as `Vector{Vector{Float64}}`.
"""
function loadprods(dav::Davidson, ipos)
  return loadvecs(dav.prodfiles[ipos])
end

"""
    combine(dav::Davidson, vecfiles, coeffs)

  Combine vectors from files with coefficients.
"""
function combine(dav::Davidson, vecfiles, coeffs)
  outvecs = loadvecs(vecfiles[1])
  for v in outvecs
    v .*= coeffs[1]
  end
  for i in 2:dav.nDim
    vect = loadvecs(vecfiles[i])
    coef = coeffs[i]
    for j in eachindex(vect)
      outvecs[j] .+= coef * vect[j]
    end
  end
  return outvecs
end

"""
    combine!(dav::Davidson, outvec, vecfiles, coeffs)

  Combine vectors from files with coefficients.

  `outvec` is the output vector.
"""
function combine!(dav::Davidson, outvec, vecfiles, coeffs)
  outvecs = combine(dav, vecfiles, coeffs)
  for i in eachindex(outvec)
    outvec[i][:] = outvecs[i]
  end
end

"""
    update_Heff!(dav::Davidson, tvecs, prods)

  Update effective Hamiltonian matrix.

  `tvecs` and `prods` are trial and product vectors (for one state) 
  for the current iteration of Davidson algorithm (stored at dav.nDim position).
"""
function update_Heff!(dav::Davidson, tvecs, prods)
  for i in 1:dav.nDim-1
    vec = loadprods(dav, i)
    res = tvecs ⋅ vec
    dav.hmat[dav.nDim,i] = res
    if hermitian(dav)
      dav.hmat[i,dav.nDim] = res
    else
      vec = loadtvecs(dav, i)
      res = vec ⋅ prods 
      dav.hmat[i,dav.nDim] = res
      # update S matrix
      res = vec ⋅ tvecs
      dav.smat[i,dav.nDim] = res
      dav.smat[dav.nDim,i] = res
    end
  end
  if !hermitian(dav)
    thisDot = tvecs ⋅ tvecs
    dav.smat[dav.nDim,dav.nDim] = thisDot 
  end
  thisDot = tvecs ⋅ prods
  dav.hmat[dav.nDim,dav.nDim] = thisDot
  return thisDot
end

"""
    update_Heff!(dav::Davidson, prods)

  Update effective Hamiltonian matrix (symmetrically).

  `prods` are product vectors (for one state) 
  for the current iteration of Davidson algorithm (stored at dav.nDim position).
"""
function update_Heff!(dav::Davidson, prods)
  for i in 1:dav.nDim
    vec = loadtvecs(dav, i)
    res = vec ⋅ prods 
    dav.hmat[i,dav.nDim] = res
    dav.hmat[dav.nDim,i] = res
  end
end

"""
    update_Seff!(dav::Davidson, tvecs)

  Update effective overlap matrix.

  `tvecs` are trial vectors (for one state) 
  for the current iteration of Davidson algorithm (stored at dav.nDim position).
"""
function update_Seff!(dav::Davidson, tvecs)
  for i in 1:dav.nDim-1
    vec = loadtvecs(dav, i)
    res = vec ⋅ tvecs
    dav.smat[i,dav.nDim] = res
    dav.smat[dav.nDim,i] = res
  end
  thisDot = tvecs ⋅ tvecs
  dav.smat[dav.nDim,dav.nDim] = thisDot 
  return thisDot
end

"""
    addtrialvecs!(dav::Davidson, tvecs)

  Add trial vectors to Davidson object and update effective overlap matrix (in non-hermitian case).

  Note: `tvecs` will be orthogonalized and normalized.
"""
function addtrialvecs!(dav::Davidson, tvecs)
  @assert dav.nDimTrial < dav.maxdav*dav.nstates "Davidson: maximum number of trial vectors reached, but no restart done"
  orthogonalize!(dav, tvecs)
  dav.nDimTrial += 1
  savetvecs(dav, tvecs)
  if !hermitian(dav)
    update_Seff!(dav, tvecs)
  end
end

"""
    addtrialvecs4allstates!(dav::Davidson, tvecs_nst)

  Add trial vectors to Davidson object and update effective overlap matrix (in non-hermitian case).

  `tvecs_nst` are trial vectors for each state.
"""
function addtrialvecs4allstates!(dav::Davidson, tvecs_nst)
  @assert length(tvecs_nst) == dav.nstates
  for st in 1:dav.nstates
    addtrialvecs!(dav, tvecs_nst[st])
  end
end


"""
    addprodvecs!(dav::Davidson, prods)

  Add product vectors to Davidson object and update effective Hamiltonian matrix.
"""
function addprodvecs!(dav::Davidson, prods)
  @assert dav.nDim < dav.maxdav*dav.nstates "Davidson: maximum number of product vectors reached, but no restart done"
  dav.nDim += 1
  saveprods(dav, prods)
  update_Heff!(dav, tvecs, prods)
end

"""
    addvecs!(dav::Davidson, tvecs, prods)

  Add trial and product vectors to Davidson object and update effective Hamiltonian matrix.
"""
function addvecs!(dav::Davidson, tvecs, prods)
  @assert dav.nDim < dav.maxdav*dav.nstates "Davidson: maximum number of trial vectors reached, but no restart done"
  dav.nDim += 1
  savetvecs(dav, tvecs)
  saveprods(dav, prods)
  update_Heff!(dav, tvecs, prods)
end

"""
    diagonalize(dav::Davidson)

  Diagonalize effective Hamiltonian matrix.
"""
function diagonalize(dav::Davidson)
  if hermitian(dav)
    vals, vecs = eigen(Hermitian(dav.hmat[1:dav.nDim,1:dav.nDim]))
  else
    vals, vecs = eigen(dav.hmat[1:dav.nDim,1:dav.nDim], Hermitian(dav.smat[1:dav.nDim,1:dav.nDim]))
  end
  return vals, vecs
end

"""
    perform!(dav::Davidson, tvecs, prods)

  Perform Davidson algorithm.

  `tvecs` and `prods` are trial and product vectors for the first state.
"""
function perform!(dav::Davidson, tvecs, prods)
  perform!(dav, [tvecs], [prods], [1])
end

"""
    perform!(dav::Davidson, tvecs_nst, prods_nst, states)
  
  Perform Davidson algorithm.

  `tvecs_nst` and `prods_nst` are trial and product vectors for each state.
""" 
function perform!(dav::Davidson, tvecs_nst, prods_nst, states)
  @assert length(tvecs_nst) == length(prods_nst) == dav.nstates >= length(states) "Davidson: wrong number of states: $(dav.nstates) vs $(length(states))"
  for st in states
    addvecs!(dav, tvecs_nst[st], prods_nst[st])
  end
  println("Davidson: effective Hamiltonian")
  display(dav.hmat[1:dav.nDim,1:dav.nDim])
  if !hermitian(dav)
    println("Davidson: overlap matrix")
    display(dav.smat[1:dav.nDim,1:dav.nDim])
  end
  # solve effective Hamiltonian
  vals, vecs = diagonalize(dav)
  display(vecs * vecs')

  for st in 1:dav.nstates
    # eigenvectors
    combine!(dav, tvecs_nst[st], dav.tvecfiles, vecs[:,st])
    println("Norm of eigenvector $st: ", norm(tvecs_nst[st]))
    # product vectors
    combine!(dav, prods_nst[st], dav.prodfiles, vecs[:,st])
  end
  # display(tvecs_nst[1])
  # restart
  if dav.nDim + min(length(states)+1, dav.nstates) > dav.maxdav*dav.nstates
    println("Davidson: maximum number of trial vectors reached, restarting")
    dav.nDim = 0
    for st in 1:dav.nstates
      dav_normalize!(tvecs_nst[st], prods_nst[st])
      addvecs!(dav, tvecs_nst[st], prods_nst[st])
    end
    display(dav.hmat[1:dav.nDim,1:dav.nDim])
  end

  # residuals
  for st in 1:dav.nstates
    resnorm = 0.0
    for i in eachindex(prods_nst[st])
      prods_nst[st][i] .-= vals[st] * tvecs_nst[st][i]
      resnorm += prods_nst[st][i] ⋅ prods_nst[st][i]
    end
    println("Davidson: state $st, residual = $resnorm")
  end
  return vals[1:dav.nstates]
end

"""
    orthogonalize!(dav::Davidson, vecs)

  Orthogonalize vectors to trial vectors.
"""
function orthogonalize!(dav::Davidson, vecs)
  dav_normalize!(vecs)
  for i in 1:dav.nDim
    vec = loadtvecs(dav, i)
    overlap = vecs ⋅ vec
    for j in eachindex(vecs)
      vecs[j][:] .-= overlap * vec[j]
    end
  end
  dav_normalize!(vecs)
end

"""
    dav_normalize!(vecs)

  Normalize vectors.
"""
function dav_normalize!(vecs)
  vnorm = norm(vecs)
  for i in eachindex(vecs)
    vecs[i] ./= vnorm
  end
end

"""
    dav_normalize!(vecs, vecs2)

  Normalize vectors `vecs` and rescale `vecs2` accordingly.
"""
function dav_normalize!(vecs, vecs2)
  vnorm = norm(vecs)
  for i in eachindex(vecs)
    vecs[i] ./= vnorm
    vecs2[i] ./= vnorm
  end
end

end # module