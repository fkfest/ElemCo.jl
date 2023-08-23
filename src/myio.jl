""" 
  EC-specific IO routines

  Use to store arrays in a file, and to load them back.
  Use memory-maps to store and load large arrays.
"""
module MIO
using Mmap

export miosave, mioload, miommap, mionewmmap, mioclosemmap

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
    mioload(fname::String; array_of_arrays = false)

  Load arrays from a file `fname`.

  Return an array of arrays.
  If there is only one array - return array itself
  (unless `array_of_arrays` is set to true).
"""
function mioload(fname::String; array_of_arrays = false)
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
    mionewmmap(fname::String, Type, dims::Tuple{Vararg{Int}})

  Create a new memory-map file for writing (overwrites existing file).
  Return a pointer to the file and the mmaped array.
"""
function mionewmmap(fname::String, Type, dims::Tuple{Vararg{Int}})
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
  return io, mmap(io, Array{Type,length(dims)}, dims)
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
  return io, mmap(io, Array{T,ndim}, Tuple(dims))
end

end #module
