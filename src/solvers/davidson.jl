
"""
    Davidson solver module
"""
module DavidsonSolver
using LinearAlgebra
using ..ElemCo.MIO
using ..ElemCo.ECInfos

export Davidson, perform!

export add_trial_vector!, get_current_trial_vector!, add_product_vector!
export get_residual!, get_eigenvector, get_eigenvector!
#export get_left_eigenvector

"""
  Davidson object
"""
mutable struct Davidson
    """ maximum number of trial vectors """
  maxdav::Int
  """ number of states """
  nstates::Int
  """ state index of each trial vector """
  states::Vector{Int}
  """ files for trial vectors """
  tvecfiles::Vector{String}
  """ files for product vectors """
  prodfiles::Vector{String}
  """ files for eigenvectors """
  eigvecfiles::Vector{String}
  """ size of the Davidson space 
  (i.e., number of trial vectors with corresponding product vectors) """
  nDim::Int
  """ number of added trial vectors (≥ `nDim`) """
  nDimTrial::Int
  """ effective matrix """
  hmat::Matrix{Float64}
  """ overlap matrix (length=0 for davidson.use_overlap=false (always there for non-hermitian)) """
  smat::Matrix{Float64}
  """ eigenvalues """
  eigvals::Vector{Float64}
  """ eigenvectors in the trial space """
  eigvecs::Matrix{Float64}
  """ hermitian flag """
  hermitian::Bool
  """
    Davidson(EC::ECInfo, nstates=1; maxdav::Int = EC.options.davidson.maxdav, hermitian::Bool = true)
  
  Create Davidson object for `nstates` states.
  """
  function Davidson(EC::ECInfo, nstates=1; maxdav::Int = -1, hermitian::Bool = true)
    if maxdav < 0
      maxdav = EC.options.davidson.maxdav
    end
    maxdav_tot = maxdav*nstates
    states = zeros(Int, maxdav_tot)
    tvecfiles = [ joinpath(EC.scr, "dav_tvec"*string(i)*EC.ext) for i in 1:maxdav_tot ]
    prodfiles = [ joinpath(EC.scr, "dav_prod"*string(i)*EC.ext) for i in 1:maxdav_tot ]
    eigvecfiles = [ joinpath(EC.scr, "dav_eigvec"*string(i)*EC.ext) for i in 1:nstates ]
    for i in 1:maxdav_tot
      add_file!(EC, "dav_tvec"*string(i), "tmp", overwrite=true)
      add_file!(EC, "dav_prod"*string(i), "tmp", overwrite=true)
      add_file!(EC, "dav_eigvec"*string(i), "tmp", overwrite=true)
    end
    if hermitian && !EC.options.davidson.use_overlap
      smat = zeros(0,0)
    else
      smat = zeros(maxdav_tot,maxdav_tot)
    end
    eigvals = zeros(nstates)
    eigvecs = zeros(maxdav_tot, nstates)
    new(maxdav, nstates, states, tvecfiles, prodfiles, eigvecfiles, 0, 0, 
        zeros(maxdav_tot,maxdav_tot), smat, eigvals, eigvecs, hermitian)
  end
end

"""
    use_overlap(dav::Davidson)

  Check if smat is used.
"""
use_overlap(dav::Davidson) = length(dav.smat) > 0

"""
    savetvecs(dav::Davidson, vecs, state)
  
  Save trial vectors to next file and store state index.
"""
function savetvecs(dav::Davidson, vecs, state)
  dav.states[dav.nDimTrial] = state
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
    saveeigvecs(dav::Davidson, vecs, state)

  Save eigenvectors to file for `state`.
"""
function saveeigvecs(dav::Davidson, vecs, state)
  miosave(dav.eigvecfiles[state], vecs...)
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
    update_Heff!(dav::Davidson, prods)

  Update effective Hamiltonian matrix.

  `prods` are product vectors (for one state) 
  for the current iteration of Davidson algorithm
  (stored at dav.nDim+1 position).
"""
function update_Heff!(dav::Davidson, prods)
  ipos = dav.nDim + 1
  for i in 1:dav.nDimTrial
    vec = loadtvecs(dav, i)
    res = vec ⋅ prods 
    dav.hmat[i,ipos] = res
    if dav.hermitian
      dav.hmat[ipos,i] = res
    end
  end
end

"""
    update_Heff_dagger!(dav::Davidson, tvecs)

  Update effective Hamiltonian matrix 
  (transpose, for non-hermitian problems).

  `tvecs` are trial vectors (for one state)
  for the current iteration of Davidson algorithm
  (stored at dav.nDimTrial+1 position).
"""
function update_Heff_dagger!(dav::Davidson, tvecs)
  ipos = dav.nDimTrial + 1
  for i in 1:dav.nDim
    prods = loadprods(dav, i)
    res = tvecs ⋅ prods 
    dav.hmat[ipos,i] = res
  end
end

"""
    update_Seff!(dav::Davidson, tvecs, state=1)

  Update effective overlap matrix.

  `tvecs` are trial vectors (for one state) 
  for the current iteration of Davidson algorithm (stored at dav.nDimTrial+1 position).
"""
function update_Seff!(dav::Davidson, tvecs, state=1)
  ipos = dav.nDimTrial + 1
  for i in 1:dav.nDimTrial
    vec = loadtvecs(dav, i)
    res = vec ⋅ tvecs
    dav.smat[i,ipos] = res
    dav.smat[ipos,i] = res
  end
  thisDot = tvecs ⋅ tvecs
  dav.smat[ipos,ipos] = thisDot 
  return thisDot
end

"""
    add_trial_vector!(dav::Davidson, tvecs, state=1)

  Add a trial vector for `state` to Davidson object and update effective overlap
  and Hamiltonian matrix.

  Note: the trial vector will be normalized and either orthogonalized to the existing trial vectors,
  or the effective overlap matrix will be updated (in non-hermitian case).
"""
function add_trial_vector!(dav::Davidson, tvecs, state=1)
  @assert dav.nDimTrial < dav.maxdav*dav.nstates "Davidson: maximum number of trial vectors reached, but no restart done"
  dav_normalize!(tvecs)
  if use_overlap(dav)
    update_Seff!(dav, tvecs, state)
  else
    orthogonalize!(dav, tvecs, state)
  end
  if !dav.hermitian
    update_Heff_dagger!(dav, tvecs)
  end
  dav.nDimTrial += 1
  savetvecs(dav, tvecs, state)
end

"""
    get_current_trial_vector!(dav::Davidson, tvecs, state=0)

  Copy the current trial vector from Davidson object to `tvecs`.
  If `state > 0`, check if the trial vector is for that state.
"""
function get_current_trial_vector!(dav::Davidson, tvecs, state=-1)
  @assert dav.nDimTrial > 0 "Davidson: no trial vectors"
  @assert dav.nDim < dav.nDimTrial "Davidson: all trial vectors used"
  ipos = dav.nDim + 1
  if state > 0
    @assert dav.states[ipos] == state "Davidson: trial vector for state $state not found"
  end
  t = loadtvecs(dav, ipos)
  @assert length(t) == length(tvecs) "Davidson: trial vector size mismatch"
  for i in eachindex(tvecs)
    tvecs[i][:] = t[i]
  end
  return tvecs
end

"""
    add_product_vector!(dav::Davidson, prods, state=0)
  
  Add a product vector for `state` to Davidson object and update effective Hamiltonian matrix.
"""
function add_product_vector!(dav::Davidson, prods, state=0)
  @assert dav.nDim < dav.maxdav*dav.nstates "Davidson: maximum number of product vectors reached, but no restart done"
  update_Heff!(dav, prods)
  dav.nDim += 1
  if state > 0
    @assert dav.states[dav.nDim] == state "Davidson: mismatch of product vector for state $state"
  end
  saveprods(dav, prods)
end

"""
    perform!(dav::Davidson)

  Perform Davidson diagonalization of effective Hamiltonian matrix.

  Store the eigenvalues and eigenvectors in the Davidson object.
"""
function perform!(dav::Davidson)
  @assert dav.nDim == dav.nDimTrial >= dav.nstates "Davidson: not enough trial or product vectors"
  # println("Davidson: effective Hamiltonian")
  # display(dav.hmat[1:dav.nDim,1:dav.nDim])
  # if use_overlap(dav)
  #   println("Davidson: overlap matrix")
  #   display(dav.smat[1:dav.nDim,1:dav.nDim])
  # end
  # solve effective Hamiltonian
  vals, vecs = diagonalize(dav)

  for st in 1:dav.nstates
    # eigenvectors
    eigvec = combine(dav, dav.tvecfiles, vecs[:,st])
    # println("Norm of eigenvector $st: ", norm(eigvec))
    saveeigvecs(dav, eigvec, st)
  end
  if dav.nDim + dav.nstates > dav.maxdav*dav.nstates
    # restart
    println("Davidson: maximum number of trial vectors reached, restarting")
    dav.nDim = dav.nDimTrial = 0
    for st in 1:dav.nstates
      evec = get_eigenvector(dav, st)
      add_trial_vector!(dav, evec, st)
    end
  end
  return vals[1:dav.nstates]
end

"""
    get_eigenvector(dav::Davidson, state)

  Get eigenvector for `state` from Davidson object.

  The eigenvector is loaded from the corresponding file.
"""
function get_eigenvector(dav::Davidson, state)
  eigvec = loadvecs(dav.eigvecfiles[state])
  return eigvec
end

"""
    get_eigenvector!(dav::Davidson, vecs, state)

  Get eigenvector for `state` from Davidson object and store it in `vecs`.

  The eigenvector is loaded from the corresponding file.
"""
function get_eigenvector!(dav::Davidson, vecs, state)
  eigvec = loadvecs(dav.eigvecfiles[state])
  @assert length(eigvec) == length(vecs) "Davidson: eigenvector size mismatch"
  for i in eachindex(vecs)
    vecs[i][:] = eigvec[i]
  end
  return vecs
end

function get_residual!(dav::Davidson, vecs, state)
  # calculate residual
  combine!(dav, vecs, dav.prodfiles, dav.eigvecs[:,state])
  en = dav.eigvals[state]
  eigvec = loadvecs(dav.eigvecfiles[state])
  @assert length(eigvec) == length(vecs) "Davidson: eigenvector size mismatch"
  for i in eachindex(vecs)
    vecs[i][:] .-= en * eigvec[i]
  end
  return vecs
end

"""
    diagonalize(dav::Davidson)

  Diagonalize effective Hamiltonian matrix.
"""
function diagonalize(dav::Davidson)
  if use_overlap(dav)
    vals, vecs = eigen(dav.hmat[1:dav.nDim,1:dav.nDim], Hermitian(dav.smat[1:dav.nDim,1:dav.nDim]))
  else
    vals, vecs = eigen(Hermitian(dav.hmat[1:dav.nDim,1:dav.nDim]))
  end
  dav.eigvals[1:dav.nstates] = vals[1:dav.nstates]
  dav.eigvecs[1:dav.nDim,1:dav.nstates] = vecs[:,1:dav.nstates]
  return vals, vecs
end

"""
    orthogonalize!(dav::Davidson, vecs, state=1)

  Orthogonalize vectors to trial vectors.
"""
function orthogonalize!(dav::Davidson, vecs, state=1)
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

end # module