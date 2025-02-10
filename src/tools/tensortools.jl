""" tensor tools, 
    e.g., access to integrals, load/save intermediates... 
"""
module TensorTools
using LinearAlgebra
using TensorOperations
# using ElemCoTensorOperations
using StridedViews
using ..ElemCo.ECInfos
using ..ElemCo.FciDumps
using ..ElemCo.MIO

export save!, load, load_all, load!, mmap, newmmap, closemmap, flushmmap
export load1idx, load2idx, load3idx, load4idx, load5idx, load6idx
export load1idx_all, load2idx_all, load3idx_all, load4idx_all, load5idx_all, load6idx_all
export mmap1idx, mmap2idx, mmap3idx, mmap4idx, mmap5idx, mmap6idx
export ints1, ints2, detri_int2
export ints2!, detri_int2!
export sqrtinvchol, invchol, rotate_eigenvectors_to_real, svd_thr
export get_spaceblocks
export print_nonzeros
export @mtensor, @mtensoropt
export @tensor, @tensoropt # reexport @tensor, @tensoropt
export @mview, mview

save_tensorcalls() = false

if save_tensorcalls()
  include("tensoranalyzer.jl")
  write_header4tensorcalls()
end

"""
    mtensor(ex)

Macro for tensor operations with manual allocator.
"""
macro mtensor(ex)
  if save_tensorcalls()
    print_tensor4tensorcalls(Symbol("@tensor"), ex)
  end
  return esc(:(@tensor $ex))
  # TODO: activate manual allocator
  # return esc(:(@mtensor allocator = TensorOperations.ManualAllocator() $ex))
end

macro mtensoropt(args::Vararg{Expr})
  if save_tensorcalls()
    print_tensor4tensorcalls(Symbol("@tensoropt"), args...)
  end
  return esc(:(@tensoropt $(args...)))
  # TODO: activate manual allocator
  # return esc(:(@mtensor allocator = TensorOperations.ManualAllocator() $ex))
end

"""
    @mview(ex)

  StridedView based version of `@view`.
"""
macro mview(ex)
  # NOTE it's largely based on the @view macro from Base.
  Meta.isexpr(ex, :ref) || throw(ArgumentError(
      "Invalid use of @mview macro: argument must be a reference expression A[...]."))
  ex = Base.replace_ref_begin_end!(ex)
  # NOTE We embed `view` as a function object itself directly into the AST.
  #      By doing this, we prevent the creation of function definitions like
  #      `view(A, idx) = xxx` in cases such as `@view(A[idx]) = xxx.`
  if Meta.isexpr(ex, :ref)
      ex = Expr(:call, mview, ex.args...)
  elseif Meta.isexpr(ex, :let) && (arg2 = ex.args[2]; Meta.isexpr(arg2, :ref))
      # ex replaced by let ...; foo[...]; end
      ex.args[2] = Expr(:call, mview, arg2.args...)
  else
      error("invalid expression")
  end
  return esc(ex)
end

"""
    mview(arr, args...)

  `StridedView` based version of `view`.

  The data array is enforced to be a vector, such that the view is always a `StridedView{..., Vector{...},...}`.
"""
function mview(arr, args...)
  return sview(reshape(view(vec(arr),:), size(arr)), args...)
end

"""
    mview(arr::StridedView, args...)

  StridedView based version of `view`, for `StridedView` input.

  Simply calls `StridedViews.sview`.
"""
function mview(arr::StridedView, args...)
  return sview(arr, args...)
end

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
    load(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64; skip_error=false) where {N}

  Type-stable load array from file `fname` in EC.scr directory.

  The type `T` and number of dimensions `N` are given explicitly.
  If `skip_error` is true, return empty `Array{T,N}` if the dimension/type is wrong.
"""
function load(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64; skip_error=false) where {N}
  return mioload(joinpath(EC.scr, fname*EC.ext), Val(N), T; skip_error)[1]
end

"""
    load_all(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64; skip_error=false) where {N}

  Type-stable load arrays from file `fname` in EC.scr directory.

  The type `T` and number of dimensions `N` are given explicitly (have to be the same for all arrays).
  Return an array of arrays.
  If `skip_error` is true, return empty `Array{T,N}[Array{T,N}()]` if the dimension/type is wrong.
"""
function load_all(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64; skip_error=false) where {N}
  return mioload(joinpath(EC.scr, fname*EC.ext), Val(N), T; skip_error)
end

for N in 1:6
  loadN = Symbol("load$(N)idx")
  loadNall = Symbol("load$(N)idx_all")
  @eval begin
    function $loadN(EC::ECInfo, fname::String, T::Type=Float64; skip_error=false)
      return load(EC, fname, Val($N), T; skip_error)
    end
    function $loadNall(EC::ECInfo, fname::String, T::Type=Float64; skip_error=false)
      return load_all(EC, fname, Val($N), T; skip_error)
    end
  end
end

"""
    load!(EC::ECInfo, fname::String, arrs::AbstractArray{T,N}...; skip_error=false)

  Load array(s) from file `fname` in EC.scr directory.

  The type and number of dimensions are deduced from the first array in `arrs`.
  If `skip_error` is true, return false if the dimension/type is wrong.
"""
function load!(EC::ECInfo, fname::String, arrs::AbstractArray{T,N}...; skip_error=false) where {T,N}
  return mioload!(joinpath(EC.scr, fname*EC.ext), arrs...; skip_error)
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

function mmap(EC::ECInfo, fname::String, ::Val{N}, T::Type=Float64) where {N}
  return miommap(joinpath(EC.scr, fname*EC.ext), Val(N), T)
end

for N in 1:6
  mmapN = Symbol("mmap$(N)idx")
  mmapNall = Symbol("mmap$(N)idx_all")
  @eval begin
    function $mmapN(EC::ECInfo, fname::String, T::Type=Float64)
      return mmap(EC, fname, Val($N), T)
    end
    function $mmapNall(EC::ECInfo, fname::String, T::Type=Float64)
      return load_all(EC, fname, Val($N), T)
    end
  end
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

function spincase_from_4spaces(spaces::String)
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
  return sc
end

""" 
    ints2(EC::ECInfo, spaces::String, spincase = nothing)

  Return subset of 2e⁻ integrals according to spaces. 
  
  The `spincase`∈{`:α`,`:β`} can explicitly be given, or will be deduced 
  from upper/lower case of spaces specification.
  If the last two indices are stored as triangular - make them full.
"""
function ints2(EC::ECInfo, spaces::String, spincase = nothing)
  if isnothing(spincase)
    sc = spincase_from_4spaces(spaces)
  else 
    sc::Symbol = spincase
  end
  SP = EC.space
  if EC.fd.uhf && sc == :αβ 
    return integ2_os(EC.fd)[SP[spaces[1]],SP[spaces[2]],SP[spaces[3]],SP[spaces[4]]]
  end
  allint = integ2_ss(EC.fd, sc)
  @assert ndims(allint) == 3
  norb = length(EC.space[':'])
  # last two indices as a triangular index, desymmetrize
  return detri_int2(allint, norb, SP[spaces[1]], SP[spaces[2]], SP[spaces[3]], SP[spaces[4]])
end

""" 
    ints2!(out::AbstractArray{Float64,4}, EC::ECInfo, spaces::String, spincase = nothing)

  Return subset of 2e⁻ integrals according to spaces. 
  
  The `spincase`∈{`:α`,`:β`} can explicitly be given, or will be deduced 
  from upper/lower case of spaces specification.
  If the last two indices are stored as triangular - make them full.
  The result is stored in `out`.
"""
function ints2!(out::AbstractArray{Float64,4}, EC::ECInfo, spaces::String, spincase = nothing)
  if isnothing(spincase)
    sc = spincase_from_4spaces(spaces)
  else 
    sc::Symbol = spincase
  end
  SP = EC.space
  if EC.fd.uhf && sc == :αβ
    @assert size(out) == (length(SP[spaces[1]]),length(SP[spaces[2]]),length(SP[spaces[3]]),length(SP[spaces[4]]))
    out .= @view integ2_os(EC.fd)[SP[spaces[1]],SP[spaces[2]],SP[spaces[3]],SP[spaces[4]]]
    return out
  end
  allint = integ2_ss(EC.fd, sc)
  @assert ndims(allint) == 3
  norb = length(EC.space[':'])
  # last two indices as a triangular index, desymmetrize
  return detri_int2!(out, allint, norb, SP[spaces[1]], SP[spaces[2]], SP[spaces[3]], SP[spaces[4]])
end

""" 
    detri_int2(allint2, norb, sp1, sp2, sp3, sp4)

  Return full 2e⁻ integrals <sp1 sp2 | sp3 sp4> from allint2 with last two indices as a triangular index.
"""
function detri_int2(allint2, norb, sp1, sp2, sp3, sp4)
  out = Array{Float64,4}(undef,length(sp1),length(sp2),length(sp3),length(sp4))
  return detri_int2!(out, allint2, norb, sp1, sp2, sp3, sp4)
end

"""
    detri_int2!(out, allint2, norb, sp1, sp2, sp3, sp4)

  Return full 2e⁻ integrals <sp1 sp2 | sp3 sp4> from allint2 with last two indices as a triangular index.
  The result is stored in `out`.
"""
function detri_int2!(out, allint2, norb, sp1, sp2, sp3, sp4)
  @assert ndims(allint2) == 3
  @assert size(out) == (length(sp1),length(sp2),length(sp3),length(sp4))
  cio, maski = triinds(norb, sp3, sp4)
  out[:,:,cio] .= @view(allint2[sp1,sp2,maski])
  cio, maski = triinds(norb, sp4, sp3, true)
  permutedims!(@view(out[:,:,cio]), @view(allint2[sp2,sp1,maski]), (2,1,3))
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
    get_spaceblocks(space, maxblocksize=128, strict=false)

  Generate ranges for block indices for space (for loop over blocks).

  `space` is a range or an array of indices. 
  Even if `space` is non-contiguous, the blocks will be contiguous. 
  If `strict` is true, the blocks will be of size `maxblocksize` (except for the last block and non-contiguous index-ranges).
  Otherwise the actual block size will be as close as possible to `blocksize` such that
  the resulting blocks are of similar size.
"""
function get_spaceblocks(space, maxblocksize=128, strict=false)
  if length(space) == 0
    return UnitRange{Int}[]
  end
  if last(space) - first(space) + 1 == length(space)
    # contiguous
    cblks = UnitRange{Int}[ first(space):last(space) ]
  else
    # create an array of contiguous ranges
    cblks = UnitRange{Int}[]
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

  allblks = UnitRange{Int}[]
  for range in cblks
    nblks::Int = length(range) ÷ maxblocksize
    if nblks*maxblocksize < length(range)
      nblks += 1
    end
    if strict 
      blks = UnitRange{Int}[ (i-1)*maxblocksize+first(range) : ((i == nblks) ? last(range) : i*maxblocksize+first(range)-1) for i in 1:nblks ]
    else
      blocksize = length(range) ÷ nblks
      n_largeblks = mod(length(range), nblks)
      blks = UnitRange{Int}[ (i-1)*(blocksize+1)+first(range) : i*(blocksize+1)+first(range)-1 for i in 1:n_largeblks ]
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
