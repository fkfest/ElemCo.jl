"""
A simplified version of NPZ.jl for mmaping npy files

NPY file format is described in
https://github.com/numpy/numpy/blob/v1.7.0/numpy/lib/format.py
"""
module MNPY
#NPZ.jl is licensed under the MIT License:
#
#Copyright (c) 2013: Fazlul Shahriar
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

using Mmap
import Base.CodeUnits

export mnpymmap

const NPYMagic = UInt8[0x93, 'N', 'U', 'M', 'P', 'Y']
const Version = UInt8[1, 0]

const MagicLen = length(NPYMagic)

const TypeMaps = [
  ("b1", Bool),
  ("i1", Int8),
  ("i2", Int16),
  ("i4", Int32),
  ("i8", Int64),
  ("u1", UInt8),
  ("u2", UInt16),
  ("u4", UInt32),
  ("u8", UInt64),
  ("f2", Float16),
  ("f4", Float32),
  ("f8", Float64),
  ("c8", Complex{Float32}),
  ("c16", Complex{Float64}),
]
const Numpy2Julia = Dict{String, DataType}(s => t for (s, t) in TypeMaps)

const Julia2Numpy = Dict{DataType, String}()

function __init__()
  for (s,t) in TypeMaps
    Julia2Numpy[t] = s
  end
end

# Julia2Numpy is a dictionary that uses Types as keys.
# This is problematic for precompilation because the
# hash of a Type changes everytime Julia is run.
# The hash of the keys when MNPY is precompiled will
# not be the same as when it is later run. This can
# be fixed by rehashing the Dict when the module is
# loaded.

readle(ios::IO, ::Type{T}) where T = ltoh(read(ios, T)) # ltoh is inverse of htol

function writecheck(io::IO, x::Any)
  n = write(io, x) # returns size in bytes
  n == sizeof(x) || error("short write") # sizeof is size in bytes
end

# Endianness only pertains to multi-byte things
writele(ios::IO, x::AbstractVector{UInt8}) = writecheck(ios, x)
writele(ios::IO, x::AbstractVector{CodeUnits{UInt8, <:Any}}) = writecheck(ios, x)
# codeunits returns vector of CodeUnits in 7+, uint in 6
writele(ios::IO, x::AbstractString) = writele(ios, codeunits(x))

writele(ios::IO, x::UInt16) = writecheck(ios, htol(x))

function parsechar(s::AbstractString, c::Char)
  firstchar = s[firstindex(s)]
  if  firstchar != c
    error("parsing header failed: expected character '$c', found '$firstchar'")
  end
  SubString(s, nextind(s, 1))
end

function parsestring(s::AbstractString)
  s = parsechar(s, '\'')
  parts = split(s, '\'', limit = 2)
  length(parts) != 2 && error("parsing header failed: malformed string")
  parts[1], parts[2]
end

function parsebool(s::AbstractString)
  if SubString(s, firstindex(s), thisind(s, 4)) == "True"
    return true, SubString(s, nextind(s, 4))
  elseif SubString(s, firstindex(s), thisind(s, 5)) == "False"
    return false, SubString(s, nextind(s, 5))
  end
  error("parsing header failed: excepted True or False")
end

function parseinteger(s::AbstractString)
  isdigit(s[firstindex(s)]) || error("parsing header failed: no digits")
  tail_idx = findfirst(c -> !isdigit(c), s)
  if isnothing(tail_idx)
    intstr = SubString(s, firstindex(s))
  else
    intstr = SubString(s, firstindex(s), prevind(s, tail_idx))
    if s[tail_idx] == 'L' # output of firstindex should be a valid code point
      tail_idx = nextind(s, tail_idx)
    end
  end
  n = parse(Int, intstr)
  return n, SubString(s, tail_idx)
end

function parsetuple(s::AbstractString)
  s = parsechar(s, '(')
  tup = Int[]
  while true
    s = strip(s)
    if s[firstindex(s)] == ')'
      break
    end
    n, s = parseinteger(s)
    push!(tup, n)
    s = strip(s)
    if s[firstindex(s)] == ')'
      break
    end
    s = parsechar(s, ',')
  end
  s = parsechar(s, ')')
  Tuple(tup), s
end

function parsedtype(s::AbstractString)
  dtype, s = parsestring(s)
  c = dtype[firstindex(s)]
  t = SubString(dtype, nextind(s, 1))
  if c == '<'
    Base.ENDIAN_BOM == 0x04030201 || error("Mmapping of little-endian npy files on big-endian architecture is not yet supported")
  elseif c == '>'
    Base.ENDIAN_BOM == 0x01020304 || error("Mmapping of big-endian npy files on little-endian architecture is not yet supported")
  elseif c == '|'
  else
    error("parsing header failed: unsupported endian character $c")
  end
  if !haskey(Numpy2Julia, t)
    error("parsing header failed: unsupported type $t")
  end
  Numpy2Julia[t], s
end

struct Header{T,N}
  shape::NTuple{N,Int}
end

Header{T}(shape::NTuple{N,Int}) where {T,N} = Header{T,N}(shape)
Base.size(hdr::Header) = hdr.shape
Base.eltype(hdr::Header{T}) where T = T
Base.ndims(hdr::Header{T,N}) where {T,N} = N

function parseheader(s::AbstractString)
    s = parsechar(s, '{')

    shape = Any
    T = Any
    for _ in 1:3
        s = strip(s)
        key, s = parsestring(s)
        s = strip(s)
        s = parsechar(s, ':')
        s = strip(s)
        if key == "descr"
            T, s = parsedtype(s)
        elseif key == "fortran_order"
            fortran_order, s = parsebool(s)
            fortran_order || error("Cannot mmap C-ordered npy arrays!")
        elseif key == "shape"
            shape, s = parsetuple(s)
        else
            error("parsing header failed: bad dictionary key")
        end
        s = strip(s)
        if s[firstindex(s)] == '}'
            break
        end
        s = parsechar(s, ',')
    end
    s = strip(s)
    s = parsechar(s, '}')
    s = strip(s)
    if s != ""
        error("malformed header")
    end
    Header{T}(shape)
end

function readheader(f::IO)
    b = read!(f, Vector{UInt8}(undef, length(NPYMagic)))
    if b != NPYMagic
        error("not a numpy array file")
    end
    b = read!(f, Vector{UInt8}(undef, length(Version)))

    # support for version 2 files
    if b[1] == 1
        hdrlen = UInt32(readle(f, UInt16))
    elseif b[1] == 2 
        hdrlen = UInt32(readle(f, UInt32))
    else
        error("unsupported NPY version")
    end
    hdr = ascii(String(read!(f, Vector{UInt8}(undef, hdrlen))))
    parseheader(strip(hdr))
end

function _mnpymmaparray(f, hdr::Header{T,N}) where {T,N}
    x = mmap(f, Array{T,N}, hdr.shape)
    ndims(x) == 0 ? x[1] : x
end

function mnpymmaparray(f::IO)
    hdr = readheader(f)
    _mnpymmaparray(f, hdr)
end

"""
    mnpymmap(filename::AbstractString)
Mmap a variable from `filename`. 
The input needs to be an `npy` file.
!!! note "Zero-dimensional arrays"
    Zero-dimensional arrays are stripped while being read in, and the values that they
    contain are returned. This is a notable difference from numpy, where 
    numerical values are written out and read back in as zero-dimensional arrays.
# Examples
```julia
julia> using NPZ
julia> npzwrite("temp.npy", ones(3))
julia> mnpymmap("temp.npy") # Mmaps the variable
3-element Vector{Float64}:
 1.0
 1.0
 1.0
```
"""
function mnpymmap(filename::AbstractString)
    # Detect if the file is a numpy npy array file
    f = open(filename)
    b = read!(f, Vector{UInt8}(undef, MagicLen))

    if b == NPYMagic
        seekstart(f)
        data = mnpymmaparray(f)
    else
        close(f)
        error("not a NPY file: $filename")
    end
    close(f)
    return data
end

"""
    readheader(filename)
Return a header corresponding to the variable contained in `filename`. 
The header contains information about the `eltype` and `size` of the array that may be extracted using 
the corresponding accessor functions.
"""
function readheader(filename::AbstractString)
    # Detect if the file is a numpy npy array file
    f = open(filename)
    b = read!(f, Vector{UInt8}(undef, MagicLen))

    if b == NPYMagic
        seekstart(f)
        data = readheader(f)
    else
        close(f)
        error("not a NPY file: $filename")
    end

    close(f)
    return data
end

end # module
