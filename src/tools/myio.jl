""" 
  EC-specific IO routines

  Use to store arrays in a file, and to load them back.
  Use memory-maps to store and load large arrays.
"""
module MIO
using Mmap

export miosave, mioload, mioload!, miommap, mionewmmap, mioclosemmap, mioflushmmap

const Types = [
  Bool,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float16,
  Float32,
  Float64,
  Complex{Float32},
  Complex{Float64},
]

const JuliaT2Int = Dict{DataType, Int}()

# function __init__()
  for (i,t) in enumerate(Types)
    JuliaT2Int[t] = i
  end
# end

""" 
    miosave(fname::String,arrs::AbstractArray{T}...) where T 

  Save arrays `arrs` in a file `fname`.
"""
function miosave(fname::String,arrs::AbstractArray{T}...) where T 
  io = open(fname, "w")
  # store type of numbers
  write(io, JuliaT2Int[T])
  # number of arrays in the file
  write(io, length(arrs))
  # store dimensions of the arrays
  for a in arrs
    write(io, ndims(a))
    for idim in 1:ndims(a)
      write(io, size(a,idim))
    end
  end
  # store the arrays
  for a in arrs
    write(io, a)
  end
  close(io)
end

"""
    mioload(fname::String, ::Val{N}, T::Type=Float64; skip_error=false) where {N}

  Type-stable load arrays from a file `fname`.

  Return an array of arrays. All arrays have the same type `T` and have `N` dimensions.
  For `N = 1`, return vectors even if the original array was a multi-dimensional array.
  If `skip_error` is set to true, the function will not throw an error if
  the type of the data/number of dimensions in the file does not match `T`/`N` 
  and an array with one empty Array{T,N} will be returned.
"""
function mioload(fname::String, ::Val{N}, T::Type=Float64; skip_error=false) where {N}
  io = open(fname)
  # type of numbers
  itype = read(io, Int)
  if itype > length(Types)
    if skip_error
      return Array{T,N}[Array{T,N}(undef, ntuple(i->0, Val(N)))]
    end
    error("Inconsistency in reading type of data!")
  end
  @assert T == Types[itype] "Inconsistency in reading type of data!"
  arrs = Array{T,N}[]
  # number of arrays in the file
  narray = read(io, Int)
  for ia in 1:narray
    ndim = read(io, Int)
    dims = Int[]
    if N == 1
      len = 1
      for idim in 1:ndim
        len *= read(io, Int)
      end
      append!(dims, len)
    else
      if ndim != N
        if skip_error
          return Array{T,N}[Array{T,N}(undef, ntuple(i->0, Val(N)))]
        end
        error("Inconsistency in reading dimensions of data! Expected $N, got $ndim.")
      end
      for idim in 1:ndim
        append!(dims, read(io, Int))
      end
    end
    push!(arrs, Array{T,N}(undef, (dims...)))
  end
  for ia in 1:narray
    read!(io, arrs[ia])
  end
  close(io)
  return arrs
end

"""
    mioload(fname::String; array_of_arrays = false)

  Load arrays from a file `fname`.

  Return an array of arrays.
  If there is only one array - return array itself
  (unless `array_of_arrays` is set to true).
"""
function mioload(fname::String; array_of_arrays=false)
  io = open(fname)
  # type of numbers
  itype = read(io, Int)
  if itype > length(Types)
    error("Inconsistency in reading type of data!")
  end
  T = Types[itype]
  arrs = Array{T}[]
  # number of arrays in the file
  narray = read(io, Int)
  for ia in 1:narray
    ndim = read(io, Int)
    dims = Int[]
    for idim in 1:ndim
      append!(dims, read(io, Int))
    end
    push!(arrs, Array{T}(undef, Tuple(dims)))
  end
  for ia in 1:narray
    read!(io, arrs[ia])
  end
  close(io)
  return (narray == 1 && !array_of_arrays) ? arrs[1] : arrs
end

"""
    mioload!(fname::String, arrs::AbstractArray{T,N}...; skip_error=false)

  Load arrays from a file `fname` into pre-allocated arrays `arrs`.
  Return true if successful, false otherwise.
  If `skip_error` is set to true, the function will not throw an error if
  the type of the data/number of dimensions in the file does not match `T`/`N`.
"""
function mioload!(fname::String, arrs::AbstractArray{T,N}...; skip_error=false) where {T,N}
  io = open(fname)
  # type of numbers
  itype = read(io, Int)
  if itype > length(Types)
    if skip_error
      return false
    end
    error("Inconsistency in reading type of data!")
  end
  @assert T == Types[itype] "Inconsistency in reading type of data!"
  # number of arrays in the file
  narray = read(io, Int)
  if narray != length(arrs)
    if skip_error
      return false
    end
    error("Inconsistency in reading number of arrays! Expected $(length(arrs)), got $narray.")
  end
  for ia in 1:narray
    ndim = read(io, Int)
    if N == 1
      len = 1
      for idim in 1:ndim
        len *= read(io, Int)
      end
      if length(arrs[ia]) != len
        if skip_error
          return false
        end
        error("Inconsistency in reading dimensions of data! Expected $(length(arrs[ia])), got $len.")
      end
    else
      if ndim != N
        if skip_error
          return false
        end
        error("Inconsistency in reading dimensions of data! Expected $N, got $ndim.")
      end
      dims = Int[]
      size_arr = size(arrs[ia])
      same = true
      for idim in 1:ndim
        append!(dims, read(io, Int))
        if dims[idim] != size_arr[idim]
          same = false
        end
      end
      if !same
        if skip_error
          return false
        end
        error("Inconsistency in reading dimensions of data! Expected $(size(arrs[ia])), got $(Tuple(dims)).")
      end
    end
  end
  for ia in 1:narray
    read!(io, arrs[ia])
  end
  close(io)
  return true
end

"""
    mionewmmap(fname::String, dims::Tuple{Vararg{Int}}, Type=Float64)

  Create a new memory-map file for writing (overwrites existing file).
  Return a pointer to the file and the mmaped array.
"""
function mionewmmap(fname::String, dims::NTuple{N, Int}, Type=Float64) where {N}
  io = open(fname, "w+")
  # store type of numbers
  write(io, JuliaT2Int[Type])
  # number of arrays in the file (1 for mmaps)
  write(io, 1)
  # store dimensions of the arrays
  write(io, length(dims))
  for dim in dims
    write(io, dim)
  end
  return io, mmap(io, Array{Type,N}, dims)
end

"""
    mioclosemmap(io::IO, array::AbstractArray)

  Close memory-map file and flush to disk.
"""
function mioclosemmap(io::IO, array::AbstractArray)
  Mmap.sync!(array)
  close(io)
end

"""
    mioflushmmap(io::IO, array::AbstractArray)

  Flush memory-map file to disk.
"""
function mioflushmmap(array::AbstractArray)
  Mmap.sync!(array)
end

"""
    miommap(fname::String)

  Memory-map an existing file for reading.
  Return a pointer to the file and the mmaped array.
"""
function miommap(fname::String)
  io = open(fname)
  # type of numbers
  itype = read(io, Int)
  if itype > length(Types)
    error("Inconsistency in reading type of data!")
  end
  T = Types[itype]
  # number of arrays in the file
  narray = read(io, Int)
  if narray != 1
    error("miommap can map only single arrays!")
  end
  ndim = read(io, Int)
  dims = Int[]
  for idim in 1:ndim
    append!(dims, read(io, Int))
  end
  return io, mmap(io, Array{T,ndim}, dims)
end

function miommap(fname::String, ::Val{N}, T::Type=Float64) where {N}
  io = open(fname)
  # type of numbers
  itype = read(io, Int)
  if itype > length(Types)
    error("Inconsistency in reading type of data!")
  end
  @assert T == Types[itype] "Inconsistency in reading type of data!"
  # number of arrays in the file
  narray = read(io, Int)
  if narray != 1
    error("miommap can map only single arrays!")
  end
  ndim = read(io, Int)
  dims = Int[]
  @assert N == ndim "Inconsistency in reading dimensions of data! Expected $N, got $ndim."
  for idim in 1:ndim
    append!(dims, read(io, Int))
  end
  return io, mmap(io, Array{T,N}, Tuple(dims)::NTuple{N,Int})
end


end #module
