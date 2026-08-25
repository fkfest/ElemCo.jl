""" tensor tools, 
    e.g., access to integrals, load/save intermediates... 
"""
module TensorTools
using LinearAlgebra
using ..ElemCo.ECInfos
using ..ElemCo.FciDumps
using ..ElemCo.MIO
using ..ElemCo.MTensorOperations
using ..ElemCo.ALPACADecomposition

export save!, load, load_all, load!, mmap, newmmap, closemmap, flushmmap
export load1idx, load2idx, load3idx, load4idx, load5idx, load6idx
export load1idx_all, load2idx_all, load3idx_all, load4idx_all, load5idx_all, load6idx_all
export mmap1idx, mmap2idx, mmap3idx, mmap4idx, mmap5idx, mmap6idx
export ints1, ints2, detri_int2
export ints2!, detri_int2!
export sqrtinvchol, invchol, rotate_eigenvectors_to_real, balance_norms!, svd_thr
export print_nonzeros
# reexport MTensorOperations
export @mtensor, @mtensoropt
export @tensor, @tensoropt 
export @mview, mview
export @buftensor

"""
    save!(EC::ECInfo, fname::String, a::AbstractArray...; description="tmp", overwrite=true)

  Save array or tuple of arrays `a` to file `fname` in EC.scr directory.
  Add file to `EC.files` with `description`.
"""
function save!(EC::ECInfo, fname::String, a::AbstractArray...; description="tmp", overwrite=true)
  miosave(fullfilename(EC, fname), a...)
  add_file!(EC, fname, description; overwrite)
end

"""
    save!(EC::ECInfo, fname::String, a::Tuple; description="tmp", overwrite=true)

  Save tuple of arrays `a` to file `fname` in EC.scr directory.
  Add file to `EC.files` with `description`.
"""
function save!(EC::ECInfo, fname::String, a::Tuple; description="tmp", overwrite=true)
  miosave(fullfilename(EC, fname), a...)
  add_file!(EC, fname, description; overwrite)
end

"""
    load(EC::ECInfo, fname::String)

  Load array from file `fname` in EC.scr directory.
"""
function load(EC::ECInfo, fname::String)
  return mioload(fullfilename(EC, fname))
end

"""
    load(EC::ECInfo{Ty}, fname::String, ::Val{N}, T::Type=Ty; skip_error=false) where {N, Ty}

  Type-stable load array from file `fname` in EC.scr directory.

  The type `T` and number of dimensions `N` are given explicitly.
  If `skip_error` is true, return empty `Array{T,N}` if the dimension/type is wrong.
"""
function load(EC::ECInfo{Ty}, fname::String, ::Val{N}, T::Type=Ty; skip_error=false) where {N, Ty}
  return mioload(fullfilename(EC, fname), Val(N), T; skip_error)[1]
end

"""
    load_all(EC::ECInfo{Ty}, fname::String, ::Val{N}, T::Type=Ty; skip_error=false) where {N, Ty}

  Type-stable load arrays from file `fname` in EC.scr directory.

  The type `T` and number of dimensions `N` are given explicitly (have to be the same for all arrays).
  Return an array of arrays.
  If `skip_error` is true, return empty `Array{T,N}[Array{T,N}()]` if the dimension/type is wrong.
"""
function load_all(EC::ECInfo{Ty}, fname::String, ::Val{N}, T::Type=Ty; skip_error=false) where {N, Ty}
  return mioload(fullfilename(EC, fname), Val(N), T; skip_error)
end

for N in 1:6
  loadN = Symbol("load$(N)idx")
  loadNall = Symbol("load$(N)idx_all")
  @eval begin
    function $loadN(EC::ECInfo{Ty}, fname::String, T::Type=Ty; skip_error=false) where Ty
      return load(EC, fname, Val($N), T; skip_error)
    end
    function $loadNall(EC::ECInfo{Ty}, fname::String, T::Type=Ty; skip_error=false) where Ty
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
  return mioload!(fullfilename(EC, fname), arrs...; skip_error)
end

"""
    newmmap(EC::ECInfo{Ty}, fname::String, dims::Tuple{Vararg{Int}}, Type=Ty; description="tmp")

  Create a new memory-map file for writing (overwrites existing file).
  Add file to `EC.files` with `description`.
  Return a pointer to the file and the mmaped array.
"""
function newmmap(EC::ECInfo{Ty}, fname::String, dims::NTuple{N,Int}, Type=Ty; description="tmp") where {N, Ty}
  add_file!(EC, fname, description; overwrite=true)
  return mionewmmap(fullfilename(EC, fname), dims, Type)
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
  return miommap(fullfilename(EC, fname))
end

function mmap(EC::ECInfo{Ty}, fname::String, ::Val{N}, T::Type=Ty; writable::Bool=false) where {N, Ty}
  return miommap(fullfilename(EC, fname), Val(N), T; writable)
end

for N in 1:6
  mmapN = Symbol("mmap$(N)idx")
  mmapNall = Symbol("mmap$(N)idx_all")
  @eval begin
    function $mmapN(EC::ECInfo{Ty}, fname::String, T::Type=Ty; writable::Bool=false) where Ty
      return mmap(EC, fname, Val($N), T; writable)
    end
    function $mmapNall(EC::ECInfo{Ty}, fname::String, T::Type=Ty) where Ty
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
    if occursin('p', spaces)
      sc = :p
    elseif isalphaspin(spaces[1], spaces[2])
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
    ints2!(out::AbstractArray{<:Number,4}, EC::ECInfo, sp1, sp2, sp3, sp4, spincase)

  Return subset of 2e⁻ integrals according to spaces `sp1`, `sp2`, `sp3`, `sp4`.

  The `sp1`, `sp2`, `sp3`, `sp4` are arrays or ranges of indices.
  The `spincase`∈{`:α`,`:β`,`:αβ`} has to be explicitly given. 
  If the last two indices are stored as triangular - make them full.
  The result is stored in `out`.
"""
function ints2!(out::AbstractArray{<:Number,4}, EC::ECInfo, sp1, sp2, sp3, sp4, spincase)
  if EC.fd.uhf && spincase == :αβ
    @assert size(out) == (length(sp1),length(sp2),length(sp3),length(sp4))
    out .= @view integ2_os(EC.fd)[sp1,sp2,sp3,sp4]
    return out
  end
  SP = EC.space
  if EC.options.wf.npositron > 0 && spincase == :p
    return integ2(EC.fd,spincase)[sp1,sp2,sp3,sp4]
  end
  allint = integ2_ss(EC.fd, spincase)
  @assert ndims(allint) == 3
  norb = length(EC.space[':'])
  # last two indices as a triangular index, desymmetrize
  return detri_int2!(out, allint, norb, sp1, sp2, sp3, sp4)
end

""" 
    ints2(EC::ECInfo, sp1, sp2, sp3, sp4, spincase)

  Return subset of 2e⁻ integrals according to spaces `sp1`, `sp2`, `sp3`, `sp4`.

  The `sp1`, `sp2`, `sp3`, `sp4` are arrays or ranges of indices.
  The `spincase`∈{`:α`,`:β`,`:αβ`} has to be explicitly given.
  If the last two indices are stored as triangular - make them full.
"""
function ints2(EC::ECInfo{T}, sp1, sp2, sp3, sp4, spincase) where T
  out = Array{T,4}(undef, length(sp1), length(sp2), length(sp3), length(sp4))
  return ints2!(out, EC, sp1, sp2, sp3, sp4, spincase)  
end

""" 
    ints2!(out::AbstractArray{<:Number,4}, EC::ECInfo, spaces::String, spincase = nothing)

  Return subset of 2e⁻ integrals according to spaces. 
  
  The `spincase`∈{`:α`,`:β`,`:αβ`} can explicitly be given, or will be deduced 
  from upper/lower case of spaces specification.
  If the last two indices are stored as triangular - make them full.
  The result is stored in `out`.
"""
function ints2!(out::AbstractArray{<:Number,4}, EC::ECInfo, spaces::String, spincase = nothing)
  if isnothing(spincase)
    sc = spincase_from_4spaces(spaces)
  else 
    sc::Symbol = spincase
  end
  SP = EC.space
  return ints2!(out, EC, SP[spaces[1]], SP[spaces[2]], SP[spaces[3]], SP[spaces[4]], sc)
end

""" 
    ints2(EC::ECInfo, spaces::String, spincase = nothing)

  Return subset of 2e⁻ integrals according to spaces. 
  
  The `spincase`∈{`:α`,`:β`,`:αβ`} can explicitly be given, or will be deduced 
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
  return ints2(EC, SP[spaces[1]], SP[spaces[2]], SP[spaces[3]], SP[spaces[4]], sc)
end

""" 
    detri_int2(allint2, norb, sp1, sp2, sp3, sp4)

  Return full 2e⁻ integrals <sp1 sp2 | sp3 sp4> from allint2 with last two indices as a triangular index.
"""
function detri_int2(allint2::AbstractArray{T,3}, norb, sp1, sp2, sp3, sp4) where T
  out = Array{T,4}(undef, length(sp1), length(sp2), length(sp3), length(sp4))
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
    sqrtinvchol(A::AbstractMatrix; tol = 1e-8, verbose = false, max_rank::Integer = 0)

  Return NON-SYMMETRIC (pseudo)sqrt-inverse of a hermitian matrix using Cholesky decomposition.
  
  Starting from ``A^{-1} = A^{-1} L (A^{-1} L)^† = M M^†``
  with ``A = L L^†``.
  By solving the equation ``L^† M = 1`` (for low-rank: using QR decomposition).

  If `max_rank > 0`, use LPACA decomposition with explicit rank control
  instead of threshold-based Cholesky truncation.

  Return `M`.
"""
function sqrtinvchol(A::AbstractMatrix; tol = 1e-8, verbose = false, max_rank::Integer = 0)
  if max_rank > 0
    result = lpaca(Hermitian(A); tol = tol, max_rank = max_rank)
    L = result.left
    r = size(L, 2)
    if verbose && r < size(A, 1)
      redund = size(A, 1) - r
      println("$redund vectors removed using ALPACA decomposition")
    end
    return L' \ Matrix(I, r, r)
  end
  CA = cholesky(Hermitian(A), RowMaximum(), check = false, tol = tol)
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
    rotate_eigenvectors_to_real(evecs::AbstractMatrix, evals::AbstractVector; verbose=true, warn_n_complex=0)

  Transform complex eigenvectors of a real matrix to a real space 
  such that they block-diagonalize the matrix.

  If verbose is false, only information about the first `warn_n_complex` eigenvalues will be printed.

  Return the eigenvectors and "eigenvalues" (the diagonal of the matrix) in the real space.
"""
function rotate_eigenvectors_to_real(evecs::AbstractMatrix, evals::AbstractVector; verbose=true, warn_n_complex=0)
  evecs_real::Matrix{Float64} = real.(evecs)
  evals_real::Vector{Float64} = real.(evals)
  npairs = 0
  # indices of complex eigenvalues
  idx = findall(x -> abs(imag(x)) > 0.0, evals)
  if length(idx) == 0
    return evecs_real, evals_real
  end
  if length(idx) % 2 != 0
    error("odd number of complex eigenvalues")
  end
  # find pairs of complex eigenvalues
  # and rotate the eigenvectors to the real space
  for ii in eachindex(idx)
    if idx[ii] < 0
      # skip this eigenvalue
      continue
    end
    i = idx[ii]
    if verbose || i <= warn_n_complex
      println("complex eigenvalue: ", evals[i], " ", i)
    end
    # find the complex conjugate eigenvalue
    # and the corresponding eigenvector
    iicc = ii+1
    while iicc <= length(idx) && !(evals[i] ≈ conj(evals[idx[iicc]]) && evecs_real[:,i] ≈ real.(evecs[:,idx[iicc]]))
      iicc += 1
    end
    if iicc > length(idx)
      error("complex eigenvalue pair expected but not found: conj(",evals[i], ") != ",evals[idx[ii+1]])
    end
    inext = idx[iicc]
    idx[iicc] = -inext
    evecs_real[:,inext] = imag.(@view(evecs[:,inext]))
    normalize!(@view(evecs_real[:,i]))
    normalize!(@view(evecs_real[:,inext]))
    evals_real[inext] = real(evals[inext])  
    npairs += 1
  end
  if verbose
    println("$npairs eigenvector pairs rotated to the real space")
  end
  return evecs_real, evals_real
end

function rotate_eigenvectors_to_real(evecs::Matrix{Float64}, evals::Vector{Float64}; verbose=true, warn_n_complex=0)
  return evecs, evals
end

""" 
    balance_norms!(evecs::AbstractMatrix, leftvecs=nothing)

  Balance the norms of left and right eigenvectors.

  Make each pair of left and right eigenvectors have the same norm.
"""
function balance_norms!(evecs::AbstractMatrix, leftvecs=nothing)
  if isnothing(leftvecs)
    leftvecs = transpose(inv(evecs))
  end
  for i in axes(evecs,2)
    nrm = norm(evecs[:,i])
    nrm_left = norm(leftvecs[:,i])
    scale = sqrt(nrm_left / nrm)
    evecs[:,i] .*= scale
    leftvecs[:,i] ./= scale
  end
  return evecs, leftvecs
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
