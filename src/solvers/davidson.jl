
"""
    Davidson solver module
"""
module DavidsonSolver
using LinearAlgebra
using ..ElemCo.MIO
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools

export Davidson, perform!

export add_trial_vector!, get_current_trial_vector!, add_product_vector!
export get_residual!, get_eigenvector, get_eigenvector!
export refresh!, do_refresh
#export get_left_eigenvector

"""
  Davidson object
"""
mutable struct Davidson{T<:Number}
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
  hmat::Matrix{T}
  """ overlap matrix (length=0 for davidson.use_overlap=false (always there for non-hermitian)) """
  smat::Matrix{T}
  """ eigenvalues """
  eigvals::Vector{Float64}
  """ eigenvectors in the trial space """
  eigvecs::Matrix{T}
  """ hermitian flag """
  hermitian::Bool
  """
    Davidson(EC::ECInfo, nstates=1; maxdav::Int = EC.options.davidson.maxdav, hermitian::Bool = true)
  
  Create Davidson object for `nstates` states.
  """
  function Davidson(EC::ECInfo{T}, nstates=1; maxdav::Int = -1, hermitian::Bool = true) where T
    if maxdav < 0
      maxdav = EC.options.davidson.maxdav
    end
    maxdav_tot = maxdav*nstates
    states = zeros(Int, maxdav_tot)
    tvecfiles = [ fullfilename(EC, "dav_tvec"*string(i)) for i in 1:maxdav_tot ]
    prodfiles = [ fullfilename(EC, "dav_prod"*string(i)) for i in 1:maxdav_tot ]
    eigvecfiles = [ fullfilename(EC, "dav_eigvec"*string(i)) for i in 1:nstates ]
    for i in 1:maxdav_tot
      add_file!(EC, "dav_tvec"*string(i), "tmp", overwrite=true)
      add_file!(EC, "dav_prod"*string(i), "tmp", overwrite=true)
      add_file!(EC, "dav_eigvec"*string(i), "tmp", overwrite=true)
    end
    if hermitian && !EC.options.davidson.use_overlap
      smat = zeros(T, 0, 0)
    else
      smat = zeros(T, maxdav_tot, maxdav_tot)
    end
    eigvals = zeros(nstates)
    eigvecs = zeros(T, maxdav_tot, nstates)
    new{T}(maxdav, nstates, states, tvecfiles, prodfiles, eigvecfiles, 0, 0, 
        zeros(T, maxdav_tot, maxdav_tot), smat, eigvals, eigvecs, hermitian)
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
    loadvecs(dav::Davidson, file)

  Load vectors from file as `Vector{Vector{Float64}}`.
"""
function loadvecs(dav::Davidson{T}, file) where T
  return mioload(file, Val(1), T)
end

"""
    loadtvecs(dav::Davidson, ipos)

  Load trial vectors from file at position `ipos` as `Vector{Vector{Float64}}`.
"""
function loadtvecs(dav::Davidson, ipos)
  return loadvecs(dav, dav.tvecfiles[ipos])
end

"""
    loadprods(dav::Davidson, ipos)

  Load product vectors from file at position `ipos` as `Vector{Vector{Float64}}`.
"""
function loadprods(dav::Davidson, ipos)
  return loadvecs(dav, dav.prodfiles[ipos])
end

"""
    combine(dav::Davidson, vecfiles, coeffs)

  Combine vectors from files with coefficients.
"""
function combine(dav::Davidson, vecfiles, coeffs)
  outvecs = loadvecs(dav, vecfiles[1])
  for v in outvecs
    v .*= coeffs[1]
  end
  for i in 2:dav.nDim
    vect = loadvecs(dav, vecfiles[i])
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
    custom_dot(dav::Davidson, customdots, tens, vecs, state=0)

  Compute dot product of vectors
  using custom dot-product functions `customdots::Tuple`.

  `customdots` is a `Tuple` of custom dot-product functions for each tensor in the product
  vector of the form `f(tvec, prod, state)` or `f(tvec, prod)` for `state=0`.
  If `customdots` is empty, the standard dot product is used.
  `vecs` are reshaped to the shape of tensors `tens`.
"""
function custom_dot(dav::Davidson{T}, customdots, tens, vecs, state=0) where T
  if length(customdots) == 0
    dot = zero(T)
    for i in eachindex(tens)
      dot += vec(tens[i]) ⋅ vec(vecs[i])
    end
    return dot::T
  end
  @assert length(tens) == length(customdots)
  @assert length(tens) == length(vecs)
  dot = zero(T)
  for i in eachindex(tens)
    if state > 0
      dot += dispatch(customdots[i], tens[i], vecs[i], state)
    else
      dot += dispatch(customdots[i], tens[i], vecs[i])
    end
  end
  return dot::T
end

dispatch(f::F, v::Vector, t, state) where {F} = f(reshape(v, size(t)), t, state)
dispatch(f::F, v::Vector, t) where {F} = f(reshape(v, size(t)), t)
dispatch(f::F, t, v, state) where {F} = f(t, reshape(v, size(t)), state)
dispatch(f::F, t, v) where {F} = f(t, reshape(v, size(t)))

"""
    update_Heff!(dav::Davidson, prods, state, customdots=())

  Update effective Hamiltonian matrix.

  `prods` are product vectors (for one state) 
  for the current iteration of Davidson algorithm
  (stored at dav.nDim+1 position).
  `customdots` is a `Tuple` of custom dot-product functions for each tensor in the product
  vector of the form `f(tvec, prod, state)` or `f(tvec, prod)` for `state=0`.
"""
function update_Heff!(dav::Davidson, prods, state, customdots=())
  ipos = dav.nDim + 1
  for i in 1:dav.nDimTrial
    vec = loadtvecs(dav, i)
    res = custom_dot(dav, customdots, vec, prods, state)
    dav.hmat[i,ipos] = res
    if dav.hermitian
      dav.hmat[ipos,i] = conj(res)
    end
  end
end

"""
    update_Heff_dagger!(dav::Davidson, tvecs, state, customdots=())

  Update effective Hamiltonian matrix 
  (transpose, for non-hermitian problems).

  `tvecs` are trial vectors (for one state)
  for the current iteration of Davidson algorithm
  (stored at dav.nDimTrial+1 position).
  `customdots` is a `Tuple` of custom dot-product functions for each tensor in the product
  and trial vector of the form `f(tvec, prod, state)` or `f(tvec, prod)` for `state=0`.
"""
function update_Heff_dagger!(dav::Davidson, tvecs, state, customdots=())
  ipos = dav.nDimTrial + 1
  for i in 1:dav.nDim
    prods = loadprods(dav, i)
    res = custom_dot(dav, customdots, tvecs, prods, state)
    dav.hmat[ipos,i] = res
  end
end

"""
    update_Seff!(dav::Davidson, tvecs, state, customdots=())

  Update effective overlap matrix.

  `tvecs` are trial vectors (for one state) 
  for the current iteration of Davidson algorithm (stored at dav.nDimTrial+1 position).
  `customdots` is a `Tuple` of custom dot-product functions for each tensor in the
  trial vector of the form `f(tvec, prod, state)` or `f(tvec, prod)` for `state=0`.
"""
function update_Seff!(dav::Davidson, tvecs, state, customdots=())
  ipos = dav.nDimTrial + 1
  for i in 1:dav.nDimTrial
    vec = loadtvecs(dav, i)
    res = custom_dot(dav, customdots, tvecs, vec, state)
    dav.smat[ipos,i] = res
    dav.smat[i,ipos] = conj(res)
  end
  thisDot = custom_dot(dav, customdots, tvecs, tvecs, state)
  dav.smat[ipos,ipos] = thisDot 
  return thisDot
end

"""
    add_trial_vector!(dav::Davidson, tvecs, state=0, customdots=())

  Add a trial vector for `state` to Davidson object and update effective overlap
  and Hamiltonian matrix.

  Note: the trial vector will be normalized and either orthogonalized to the existing trial vectors,
  or the effective overlap matrix will be updated (in non-hermitian case).
  `customdots` is a `Tuple` of custom dot-product functions for each tensor in the trial vector
  of the form `f(tvec, prod, state)` or `f(tvec, prod)` for `state=0`.
"""
function add_trial_vector!(dav::Davidson, tvecs, state=0, customdots=())
  @assert dav.nDimTrial < dav.maxdav*dav.nstates "Davidson: maximum number of trial vectors reached, but no restart done"
  dav_normalize!(dav, tvecs, state, customdots)
  if use_overlap(dav)
    update_Seff!(dav, tvecs, state, customdots)
  else
    orthogonalize!(dav, tvecs, state, customdots)
  end
  if !dav.hermitian
    update_Heff_dagger!(dav, tvecs, state, customdots)
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
    if dav.states[ipos] != state
      error("Davidson: trial vector for state $state not found")
    end
  end
  t = loadtvecs(dav, ipos)
  @assert length(t) == length(tvecs) "Davidson: trial vector size mismatch"
  for i in eachindex(tvecs)
    tvecs[i][:] = t[i]
  end
  return tvecs
end

"""
    add_product_vector!(dav::Davidson, prods, state=0, customdots=())
  
  Add a product vector for `state` to Davidson object and update effective Hamiltonian matrix.

  `customdots()` is a `Tuple` of custom dot-product functions for each tensor in the product 
  vector of the form `f(tvec, prod, state)` or `f(tvec, prod)` for `state=0`.
"""
function add_product_vector!(dav::Davidson, prods, state=0, customdots=())
  @assert dav.nDim < dav.maxdav*dav.nstates "Davidson: maximum number of product vectors reached, but no restart done"
  update_Heff!(dav, prods, state, customdots)
  dav.nDim += 1
  if state > 0
    if dav.states[dav.nDim] != state
      error("Davidson: mismatch of product vector for state $state")
    end
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
  return vals[1:dav.nstates]
end

"""
    do_refresh(dav::Davidson, nstates=dav.nstates)

  Check if Davidson object needs to be refreshed.

  The Davidson object needs to be refreshed if the total number of trial vectors
  after the new iteration will exceed the maximum number of trial vectors
  times the number of states.
"""
function do_refresh(dav::Davidson, nstates)
  return dav.nDim + nstates > dav.maxdav*dav.nstates
end

"""
    refresh!(dav::Davidson, evec, custom_dots=())

  Refresh Davidson object by resetting the effective Hamiltonian and overlap matrices
  and adding the eigenvectors as trial vectors.
"""
function refresh!(dav::Davidson, evec, custom_dots=())
  println("Davidson: refresh")
  dav.nDim = dav.nDimTrial = 0
  dav.hmat .= 0.0
  if use_overlap(dav)
    dav.smat .= 0.0
  end
  for st in 1:dav.nstates
    get_eigenvector!(dav, evec, st)
    add_trial_vector!(dav, evec, st, custom_dots)
  end
end

"""
    get_eigenvector(dav::Davidson, state)

  Get eigenvector for `state` from Davidson object.

  The eigenvector is loaded from the corresponding file.
"""
function get_eigenvector(dav::Davidson, state)
  eigvec = loadvecs(dav, dav.eigvecfiles[state])
  return eigvec
end

"""
    get_eigenvector!(dav::Davidson, vecs, state)

  Get eigenvector for `state` from Davidson object and store it in `vecs`.

  The eigenvector is loaded from the corresponding file.
"""
function get_eigenvector!(dav::Davidson, vecs, state)
  eigvec = loadvecs(dav, dav.eigvecfiles[state])
  @assert length(eigvec) == length(vecs) "Davidson: eigenvector size mismatch"
  for i in eachindex(vecs)
    vecs[i][:] = eigvec[i]
  end
  return vecs
end

"""
    get_residual!(dav::Davidson, vecs, state)

  Calculate residual for `state` and store it in `vecs`.

  The residual is calculated as `res = H * eigvec - eigval * eigvec`.
  The eigenvector is loaded from the corresponding file.
"""
function get_residual!(dav::Davidson, vecs, state)
  # calculate residual
  combine!(dav, vecs, dav.prodfiles, dav.eigvecs[:,state])
  en = dav.eigvals[state]
  eigvec = loadvecs(dav, dav.eigvecfiles[state])
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
function diagonalize(dav::Davidson{T}) where T
  if use_overlap(dav)
    vals, vecs = eigen(dav.hmat[1:dav.nDim,1:dav.nDim], Hermitian(dav.smat[1:dav.nDim,1:dav.nDim]))
  else
    vals, vecs = eigen(Hermitian(dav.hmat[1:dav.nDim,1:dav.nDim]))
  end
  # rotate complex conjugate eigenpairs to real form (only for real Hamiltonians)
  if T <: Real
    vecs, vals = rotate_eigenvectors_to_real(vecs, vals)
  end
  dav.eigvals[1:dav.nstates] = real.(vals[1:dav.nstates])
  dav.eigvecs[1:dav.nDim,1:dav.nstates] = vecs[:, 1:dav.nstates]
  return vals, vecs
end

"""
    orthogonalize!(dav::Davidson, vecs, state, customdots=())

  Orthogonalize vectors to trial vectors.
"""
function orthogonalize!(dav::Davidson, vecs, state, customdots=())
  for i in 1:dav.nDim
    vec = loadtvecs(dav, i)
    overlap = custom_dot(dav, customdots, vecs, vec, state)
    for j in eachindex(vecs)
      vecs[j][:] .-= overlap * vec[j]
    end
  end
  dav_normalize!(dav, vecs, state, customdots)
end

"""
    dav_normalize!(dav, vecs, state, customdots)

  Normalize vectors.
"""
function dav_normalize!(dav::Davidson, vecs, state, customdots)
  vnorm = sqrt(abs(custom_dot(dav, customdots, vecs, vecs, state)))
  for i in eachindex(vecs)
    vecs[i] ./= vnorm
  end
end

end # module