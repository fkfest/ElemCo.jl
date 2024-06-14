""" tensor tools, 
    e.g., access to integrals, load/save intermediates... 
"""
module TensorTools
using LinearAlgebra
using ..ElemCo.ECInfos
using ..ElemCo.FciDump
using ..ElemCo.MIO

export save!, load, load_all, mmap, newmmap, closemmap, flushmmap
export load1idx, load2idx, load3idx, load4idx, load5idx, load6idx
export load1idx_all, load2idx_all, load3idx_all, load4idx_all, load5idx_all, load6idx_all
export ints1, ints2, detri_int2
export sqrtinvchol, invchol, rotate_eigenvectors_to_real, svd_thr
export get_spaceblocks
export print_nonzeros


"""
    save!(EC::ECInfo, fname::String, a::AbstractArray...; description="tmp", overwrite=true)

  Save array or tuple of arrays `a` to file `fname` in EC.scr directory.
  Add file to `EC.files` with `description`.
"""
function save!(EC::ECInfo, fname::String, a::AbstractArray...; description="tmp", overwrite=true)
  miosave(joinpath(EC.scr, fname*EC.ext), a...)
  add_file!(EC, fname, description; overwrite)
end

"""
    load(EC::ECInfo, fname::String)

  Load array from file `fname` in EC.scr directory.
"""
function load(EC::ECInfo, fname::String)
  return mioload(joinpath(EC.scr, fname*EC.ext))
end

"""
    load(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64 ) where {N}

  Type-stable load array from file `fname` in EC.scr directory.

  The type `T` and number of dimensions `N` are given explicitly.
"""
function load(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64) where {N}
  return mioload(joinpath(EC.scr, fname*EC.ext), Val(N), T)[1]
end

"""
    load_all(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64 ) where {N}

  Type-stable load arrays from file `fname` in EC.scr directory.

  The type `T` and number of dimensions `N` are given explicitly (have to be the same for all arrays).
  Return an array of arrays.
"""
function load_all(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64) where {N}
  return mioload(joinpath(EC.scr, fname*EC.ext), Val(N), T)
end

for N in 1:6
  loadN = Symbol("load$(N)idx")
  loadNall = Symbol("load$(N)idx_all")
  @eval begin
    function $loadN(EC::ECInfo, fname::String, T::Type=Float64)
      return load(EC, fname, Val($N), T)
    end
    function $loadNall(EC::ECInfo, fname::String, T::Type=Float64)
      return load_all(EC, fname, Val($N), T)
    end
  end
end

"""
    newmmap(EC::ECInfo, fname::String, dims::Tuple{Vararg{Int}}, Type=Float64; description="tmp")

  Create a new memory-map file for writing (overwrites existing file).
  Add file to `EC.files` with `description`.
  Return a pointer to the file and the mmaped array.
"""
function newmmap(EC::ECInfo, fname::String, dims::NTuple{N,Int}, Type=Float64; description="tmp") where {N}
  add_file!(EC, fname, description; overwrite=true)
  return mionewmmap(joinpath(EC.scr, fname*EC.ext), dims, Type)
end

"""
    closemmap(EC::ECInfo, file, array)

  Close memory-map file and flush to disk.
"""
function closemmap(EC::ECInfo, file, array)
  mioclosemmap(file, array)
end

"""
    flushmmap(EC::ECInfo, array)

  Flush memory-map array to disk.
"""
function flushmmap(EC::ECInfo, array)
  mioflushmmap(array)
end

"""
    mmap(EC::ECInfo, fname::String)

  Memory-map an existing file for reading.
  Return a pointer to the file and the mmaped array.
"""
function mmap(EC::ECInfo, fname::String)
  return miommap(joinpath(EC.scr, fname*EC.ext))
end

""" 
    ints1(EC::ECInfo, spaces::String, spincase = nothing)

  Return subset of 1e⁻ integrals according to spaces. 
  
  The `spincase`∈{`:α`,`:β`} can explicitly be given, or will be deduced 
  from upper/lower case of spaces specification. 
"""
function ints1(EC::ECInfo, spaces::String, spincase = nothing)
  sc = spincase
  if isnothing(sc)
    if isalphaspin(spaces[1],spaces[2])
      sc = :α
    else
      sc = :β
    end
  end
  return integ1(EC.fd, sc)[EC.space[spaces[1]],EC.space[spaces[2]]]
end

""" 
    triinds(norb, sp1::AbstractArray{Int}, sp2::AbstractArray{Int}, reverseCartInd = false)

  Generate set of CartesianIndex for addressing the lhs and 
  a bitmask for the rhs for transforming a triangular index from 1:norb  
  to two original indices in spaces sp1 and sp2.
  If `reverse`: the cartesian indices are reversed.
"""
function triinds(norb, sp1::AbstractArray{Int}, sp2::AbstractArray{Int}, reverseCartInd = false)
  # triangular index (TODO: save in EC or FDump)
  tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
  mask = falses(norb,norb)
  mask[sp1,sp2] .= true
  trimask = falses(norb,norb)
  trimask[tripp] .= true
  ci=CartesianIndices((length(sp1),length(sp2)))
  if reverseCartInd
    return CartesianIndex.(reverse.(Tuple.(ci[trimask[sp1,sp2]]))), mask[tripp]
  else
    return ci[trimask[sp1,sp2]], mask[tripp]
  end
end

""" 
    ints2(EC::ECInfo, spaces::String, spincase = nothing, detri = true)

  Return subset of 2e⁻ integrals according to spaces. 
  
  The `spincase`∈{`:α`,`:β`} can explicitly be given, or will be deduced 
  from upper/lower case of spaces specification.
  If the last two indices are stored as triangular and detri - make them full,
  otherwise return as a triangular cut.
"""
function ints2(EC::ECInfo, spaces::String, spincase = nothing, detri = true)
  if isnothing(spincase)
    second_el_alpha = isalphaspin(spaces[2],spaces[4])
    if isalphaspin(spaces[1],spaces[3])
      if second_el_alpha
        sc = :α
      else
        sc = :αβ
      end
    else
      !second_el_alpha || error("Use αβ integrals to get the βα block "*spaces)
      sc = :β
    end
  else 
    sc = spincase
  end
  allint = integ2(EC.fd, sc)
  norb = length(EC.space[':'])
  if ndims(allint) == 4
    return allint[EC.space[spaces[1]],EC.space[spaces[2]],EC.space[spaces[3]],EC.space[spaces[4]]]
  elseif detri
    # last two indices as a triangular index, desymmetrize
    return detri_int2(allint, norb, EC.space[spaces[1]], EC.space[spaces[2]], EC.space[spaces[3]], EC.space[spaces[4]])
  else
    cio, maski = triinds(norb, EC.space[spaces[3]], EC.space[spaces[4]])
    return allint[EC.space[spaces[1]],EC.space[spaces[2]],maski]
  end
end

""" 
    detri_int2(allint2, norb, sp1, sp2, sp3, sp4)

  Return full 2e⁻ integrals <sp1 sp2 | sp3 sp4> from allint2 with last two indices as a triangular index.
"""
function detri_int2(allint2, norb, sp1, sp2, sp3, sp4)
  @assert ndims(allint2) == 3
  out = Array{Float64,4}(undef,length(sp1),length(sp2),length(sp3),length(sp4))
  cio, maski = triinds(norb, sp3, sp4)
  out[:,:,cio] = allint2[sp1,sp2,maski]
  cio, maski = triinds(norb, sp4, sp3, true)
  out[:,:,cio] = permutedims(allint2[sp2,sp1,maski], (2,1,3))
  return out
end

""" 
    sqrtinvchol(A::AbstractMatrix; tol = 1e-8, verbose = false)

  Return NON-SYMMETRIC (pseudo)sqrt-inverse of a hermitian matrix using Cholesky decomposition.
  
  Starting from ``A^{-1} = A^{-1} L (A^{-1} L)^† = M M^†``
  with ``A = L L^†``.
  By solving the equation ``L^† M = 1`` (for low-rank: using QR decomposition).
  Return `M`.
"""
function sqrtinvchol(A::AbstractMatrix; tol = 1e-8, verbose = false)
  CA = cholesky(Symmetric(A), RowMaximum(), check = false, tol = tol)
  if CA.rank < size(A,1)
    if verbose
      redund = size(A,1) - CA.rank
      println("$redund vectors removed using Cholesky decomposition")
    end
    Umat = CA.U[1:CA.rank,:]
  else
    Umat = CA.U
  end
  return (Umat \ Matrix(I,CA.rank,CA.rank))[invperm(CA.p),:]
end

""" 
    invchol(A::AbstractMatrix; tol = 1e-8, verbose = false)

  Return (pseudo)inverse of a hermitian matrix using Cholesky decomposition .
    
  The inverse is calculated as ``A^{-1} = A^{-1} L (A^{-1} L)^† = M M^†``
  with ``A = L L^†``.
  By solving the equation ``L^† M = 1`` (for low-rank: using QR decomposition) 
"""
function invchol(A::AbstractMatrix; tol = 1e-8, verbose = false)
  M = sqrtinvchol(A, tol = tol, verbose = verbose)
  return M * M'
end

""" 
    rotate_eigenvectors_to_real(evecs::AbstractMatrix, evals::AbstractVector)

  Transform complex eigenvectors of a real matrix to a real space 
  such that they block-diagonalize the matrix.

  Return the eigenvectors and "eigenvalues" (the diagonal of the matrix) in the real space.
"""
function rotate_eigenvectors_to_real(evecs::AbstractMatrix, evals::AbstractVector)
  evecs_real::Matrix{Float64} = real.(evecs)
  evals_real::Vector{Float64} = real.(evals)
  npairs = 0
  skip = false
  for i in eachindex(evals)
    if skip 
      skip = false
      continue
    end
    if abs(imag(evals[i])) > 0.0
      println("complex: ",evals[i], " ",i)
      if evals[i] == conj(evals[i+1])
        @assert  evecs_real[:,i] == real.(evecs[:,i+1])
        evecs_real[:,i+1] = imag.(evecs[:,i+1])
        normalize!(evecs_real[:,i])
        normalize!(evecs_real[:,i+1])
        evals_real[i+1] = real(evals[i+1])
        npairs += 1
        skip = true
      else
        error("eigenvalue pair expected but not found: conj(",evals[i], ") != ",evals[i+1])
      end
    end
  end
  if npairs > 0
    println("$npairs eigenvector pairs rotated to the real space")
  end
  return evecs_real, evals_real
end

function rotate_eigenvectors_to_real(evecs::Matrix{Float64}, evals::Vector{Float64})
  return evecs, evals
end

""" 
    get_spaceblocks(space, maxblocksize=100, strict=false)

  Generate ranges for block indices for space (for loop over blocks).

  `space` is a range or an array of indices. 
  Even if `space` is non-contiguous, the blocks will be contiguous. 
  If `strict` is true, the blocks will be of size `maxblocksize` (except for the last block and non-contiguous index-ranges).
  Otherwise the actual block size will be as close as possible to `blocksize` such that
  the resulting blocks are of similar size.
"""
function get_spaceblocks(space, maxblocksize=100, strict=false)
  if length(space) == 0
    return []
  end
  if last(space) - first(space) + 1 == length(space)
    # contiguous
    cblks = [ first(space):last(space) ]
  else
    # create an array of contiguous ranges
    cblks = []
    begr = first(space)
    endr = begr - 1
    for idx in space
      if idx == endr + 1
        endr = idx
      else
        push!(cblks, begr:endr)
        endr = begr = idx
      end 
    end
    push!(cblks, begr:endr)
  end  

  allblks = []
  for range in cblks
    nblks = length(range) ÷ maxblocksize
    if nblks*maxblocksize < length(range)
      nblks += 1
    end
    if strict 
      blks = [ (i-1)*maxblocksize+first(range) : ((i == nblks) ? last(range) : i*maxblocksize+first(range)-1) for i in 1:nblks ]
    else
      blocksize = length(range) ÷ nblks
      n_largeblks = mod(length(range), nblks)
      blks = [ (i-1)*(blocksize+1)+first(range) : i*(blocksize+1)+first(range)-1 for i in 1:n_largeblks ]
      start = n_largeblks*(blocksize+1)+first(range)
      for i = n_largeblks+1:nblks
        push!(blks, start:start+blocksize-1)
        start += blocksize
      end
    end
    append!(allblks, blks)
  end
  return allblks
end

""" 
    print_nonzeros(tensor::AbstractArray; ϵ=1.e-12, fname::String="")

  Print cartesian index alongside value of array for elements with absolute value greater or equal than ϵ
  either to stdout or to a file.
"""
function print_nonzeros(tensor::AbstractArray; ϵ=1.e-12, fname::String="")
  cartindx = findall(x -> abs(x) >= ϵ, tensor)
  if isempty(fname)
    output=stdout
  else
    output = fname
  end
  redirect_stdio(stdout=output) do
    for indx in eachindex(cartindx)
      print(cartindx[indx])
      print("    ")
      print(tensor[cartindx][indx])
      println()
    end
  end
end

""" 
    svd_thr(Amat::AbstractMatrix, thr=1.e-12)

  Return SVD of a matrix with singular values below `thr` set to zero.
"""
function svd_thr(Amat::AbstractMatrix, thr=1.e-12)
  sA = svd(Amat)
  sA.S[sA.S .< thr] .= 0.0
  return sA
end

end #module
