""" my IO routines

    use to store arrays

"""
module MyIO

export miosave, mioload, miommap

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

function __init__()
  for (i,t) in enumerate(Types)
    JuliaT2Int[t] = i
  end
end

""" 
save arrays in a file `fname`
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
load arrays from a file `fname`

return an array of arrays.
If there is only one array - return array itself.
"""
function mioload(fname::String)
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
  return (narray == 1) ? arrs[1] : arrs
end

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
  return mmap(io, Array{T}, Tuple(dims))
end

end
