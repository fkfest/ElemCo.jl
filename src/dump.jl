#!/usr/bin/env julia

"""
Read and write fcidump format integrals.
Individual arrays of integrals can also be in *.npy format
"""
module FciDumps

# using LinearAlgebra
# using NPZ
using TensorOperations
using DocStringExtensions
using Printf
using ..ElemCo.MNPY
using ..ElemCo.QMTensors

export FDump, TFDump, QFDump 
export fd_exists, read_fcidump, write_fcidump, transform_fcidump
export headvar, headvars, integ1, integ2, triang
export reorder_orbs_int2, modify_header!
export int1_npy_filename, int2_npy_filename

# optional variables which won't be written if =0
const FDUMP_OPTIONAL=["IUHF", "ST", "III"]

"""prefered order of keys in fcidump header (optional keys are not included)"""
const FDUMP_KEYS=["NORB", "NELEC", "MS2", "ISYM", "ORBSYM" ]

"""
    FDumpHeader

  Header of fcidump file
"""
Base.@kwdef mutable struct FDumpHeader
  ihead::Dict{String,Vector{Int}} = Dict{String,Vector{Int}}()
  fhead::Dict{String,Vector{Float64}} = Dict{String,Vector{Float64}}()
  shead::Dict{String,Vector{String}} = Dict{String,Vector{String}}()
end

function Base.getindex(h::FDumpHeader, key::String)
  if haskey(h.ihead, key)
    return h.ihead[key]
  elseif haskey(h.fhead, key)
    return h.fhead[key]
  else
    return h.shead[key]
  end
end
Base.getindex(h::FDumpHeader, key::String, ::Type{<:Int}) = h.ihead[key]
Base.getindex(h::FDumpHeader, key::String, ::Type{Float64}) = h.fhead[key]
Base.getindex(h::FDumpHeader, key::String, ::Type{String}) = h.shead[key]

function Base.get(h::FDumpHeader, key::String, default) 
  if haskey(h.ihead, key)
    return h.ihead[key]
  elseif haskey(h.fhead, key)
    return h.fhead[key]
  elseif haskey(h.shead, key)
    return h.shead[key]
  else
    return default
  end
end
Base.get(h::FDumpHeader, key::String, ::Type{<:Int}, default) = get(h.ihead, key, default)
Base.get(h::FDumpHeader, key::String, ::Type{Float64}, default) = get(h.fhead, key, default)
Base.get(h::FDumpHeader, key::String, ::Type{String}, default) = get(h.shead, key, default)

Base.setindex!(h::FDumpHeader, val::Vector{Int}, key::String) = h.ihead[key] = val
Base.setindex!(h::FDumpHeader, val::Vector{Float64}, key::String) = h.fhead[key] = val
Base.setindex!(h::FDumpHeader, val::Vector{String}, key::String) = h.shead[key] = val

function Base.keys(h::FDumpHeader)
  return unique([keys(h.ihead); keys(h.fhead); keys(h.shead)])
end

Base.isempty(h::FDumpHeader) = isempty(h.ihead) && isempty(h.fhead) && isempty(h.shead)
Base.empty!(h::FDumpHeader) = empty!(h.ihead) && empty!(h.fhead) && empty!(h.shead)

function Base.iterate(h::FDumpHeader, state=1)
  ikeys = collect(keys(h.ihead))
  if state <= length(ikeys)
    return ikeys[state] => h.ihead[ikeys[state]], state+1
  end
  fkeys = collect(keys(h.fhead))
  fstate = state - length(ikeys)
  if fstate <= length(fkeys)
    return fkeys[fstate] => h.fhead[fkeys[fstate]], state+1
  end
  skeys = collect(keys(h.shead))
  sstate = fstate - length(fkeys)
  if sstate <= length(skeys)
    return skeys[sstate] => h.shead[skeys[sstate]], state+1
  end
  return nothing
end
  

"""
    FDump{N}

  Molecular integrals 

  The 2-e integrals are stored in the physicists' notation: `int2[pqrs]` ``= <pq|rs>=v_{pq}^{rs}``

  `N` denotes the number of indices in the 2-e-integral tensors,
  for `N=3` (usual) the last two indices are stored as a single uppertriangular index (r <= s)

  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct FDump{N}
  """ 2-e⁻ integrals for restricted orbitals fcidump. """
  int2::Array{Float64,N} = zeros(fill(0,N)...)
  """ αα 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2aa::Array{Float64,N} = zeros(fill(0,N)...)
  """ ββ 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2bb::Array{Float64,N} = zeros(fill(0,N)...)
  """ αβ 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2ab::Array{Float64,4} = zeros(0,0,0,0)
  """ 1-e⁻ integrals for restricted orbitals fcidump. """
  int1::Matrix{Float64} = zeros(0,0)
  """ α 1-e⁻ integrals for unrestricted orbitals fcidump. """
  int1a::Matrix{Float64} = zeros(0,0)
  """ β 1-e⁻ integrals for unrestricted orbitals fcidump. """
  int1b::Matrix{Float64} = zeros(0,0)
  """ core energy """
  int0::Float64 = 0.0
  """ header of fcidump file, a dictionary of arrays. """
  head::FDumpHeader = FDumpHeader()
  """`⟨false⟩` a convinience variable, has to coincide with `head["IUHF"][1] > 0`. """
  uhf::Bool = false
end

TFDump = FDump{3}
QFDump = FDump{4}

""" 
  is_triang(fd::FDump)
  
  If true: an uppertriangular index for last two indices of 2e⁻ integrals is used.
""" 
is_triang(fd::FDump{3}) = true
is_triang(fd::FDump{4}) = false

"""
    FDump(int2::Array{Float64,N}, int1::Matrix{Float64}, int0::Float64, head::FDumpHeader) where N

  Spin-free fcidump
"""
FDump(int2::Array{Float64,N}, int1::Matrix{Float64}, int0::Float64, head::FDumpHeader) where N = FDump(; int2, int1, int0, head)
"""
    FDump(int2aa::Array{Float64,N}, int2bb::Array{Float64,N}, int2ab::Array{Float64,4}, int1a::Matrix{Float64}, int1b::Matrix{Float64}, int0::Float64, head::FDumpHeader) where N

  Spin-polarized fcidump
"""
FDump(int2aa::Array{Float64,N}, int2bb::Array{Float64,N}, int2ab::Array{Float64,4}, int1a::Matrix{Float64}, int1b::Matrix{Float64}, int0::Float64, head::FDumpHeader) where N = FDump(; int2aa, int2bb, int2ab, int1a, int1b, int0, head)

"""
    FDump{N}(norb, nelec; ms2=0, isym=1, orbsym=[], uhf=false, simtra=false)

  Create a new FDump object
"""
function FDump{N}(norb::Int, nelec::Int; ms2::Int=0, isym::Int=1, orbsym::Vector{Int}=Int[], 
               uhf=false, simtra=false) where N
  fd = FDump{N}()
  fd.head["NORB"] = [norb]
  fd.head["NELEC"] = [nelec]
  fd.head["MS2"] = [ms2]
  fd.head["ISYM"] = [isym]
  if isempty(orbsym)
    fd.head["ORBSYM"] = ones(Int,norb)
  else
    fd.head["ORBSYM"] = orbsym
  end
  fd.head["IUHF"] = uhf ? [1] : [0]
  fd.head["ST"] = simtra ? [1] : [0]
  fd.uhf = uhf
  return fd
end

"""
    modify_header!(fd::FDump, norb, nelec; ms2=-1, isym=-1, orbsym=[])

  Modify header of FDump object
"""
function modify_header!(fd::FDump, norb::Int, nelec::Int; ms2::Int=-1, isym::Int=-1, orbsym::Vector{Int}=Int[])
  fd.head["NORB"] = [norb]
  fd.head["NELEC"] = [nelec]
  if ms2 >= 0
    fd.head["MS2"] = [ms2]
  end
  if isym >= 0
    fd.head["ISYM"] = [isym]
  end
  if isempty(orbsym)
    fd.head["ORBSYM"] = ones(Int,norb)
  else
    fd.head["ORBSYM"] = orbsym
  end
end

"""
    fd_exists(fd::FDump)

  Return true if the object is a non-empty FDump
"""
function fd_exists(fd::FDump)
  return !isempty(fd.head)
end

"""
    set_zero!(fd::FDump, norb::Int=0)

  Set all integrals to zero.

  If `norb` is not provided, the integrals are set to zero with the same dimensions as before.
"""
function set_zero!(fd::FDump, norb::Int=0)
  fd.int0 = 0.0
  if norb <= 0
    if fd.uhf
      fill!(fd.int1a, 0.0)
      fill!(fd.int1b, 0.0)
      fill!(fd.int2aa, 0.0)
      fill!(fd.int2bb, 0.0)
      fill!(fd.int2ab, 0.0)
    else
      fill!(fd.int1, 0.0)
      fill!(fd.int2, 0.0)
    end
  else
    if fd.uhf
      fd.int1a = zeros(norb,norb)
      fd.int1b = zeros(norb,norb)
      fd.int2aa = get_int2_zeros(fd.int2aa, norb)
      fd.int2bb = get_int2_zeros(fd.int2bb, norb)
      fd.int2ab = get_int2_zeros(fd.int2ab, norb)
    else
      fd.int1 = zeros(norb,norb)
      fd.int2 = get_int2_zeros(fd.int2, norb)
    end
  end
end

function get_int2_zeros(int2::Array{Float64,3}, norb)
  return zeros(norb,norb,(norb+1)*norb÷2)
end

function get_int2_zeros(int2::Array{Float64,4}, norb)
  return zeros(norb,norb,norb,norb)
end

"""
    integ1(fd::FDump, spincase::Symbol=:α)

  Return 1-e⁻ integrals (for UHF fcidump: for `spincase`).
  `spincase` can be `:α` or `:β`.
"""
function integ1(fd::FDump, spincase::Symbol=:α)
  if !fd.uhf
    return fd.int1
  elseif spincase == :α
    return fd.int1a
  else
    return fd.int1b
  end
end

"""
    integ2(fd::FDump, spincase::Symbol=:α)

  Return 2-e⁻ integrals (for UHF fcidump: for `spincase`).
  `spincase` can be `:α`, `:β` or `:αβ`.
"""
function integ2(fd::FDump, spincase::Symbol=:α)
  if !fd.uhf
    return fd.int2
  elseif spincase == :α
    return fd.int2aa
  elseif spincase == :β
    return fd.int2bb
  else
    return fd.int2ab
  end
end

"""
    read_fcidump(fcidump::String, ::Val{N})

  Read ascii file (possibly with integrals in npy files).
""" 
function read_fcidump(fcidump::String, ::Val{N}) where N
  fdf = open(fcidump)
  fd = FDump{N}()
  fd.head = read_header(fdf)
  fd.uhf = (headvar(fd, "IUHF", Int) > 0)
  simtra = (headvar(fd, "ST", Int) > 0)
  if simtra
    println("Non-Hermitian")
  end
  if isnothing(headvar(fd, "NPY2", String)) && isnothing(headvar(fd, "NPY2AA", String))
    # read integrals from fcidump file
    read_integrals!(fd, fdf)
    close(fdf)
  else
    close(fdf)
    # read integrals from npy files
    read_integrals!(fd, dirname(fcidump))
  end
  return fd
end

"""
    read_fcidump(fcidump::String)

  Read ascii file (possibly with integrals in npy files) to TFDump object.
"""
read_fcidump(fcidump::String) = read_fcidump(fcidump, Val(3))

"""
    read_header(fdfile::IOStream)

  Read header of fcidump file.
"""
function read_header(fdfile)
  # put some defaults...
  head = FDumpHeader()
  head["IUHF"] = [0]
  head["ST"] = [0]
  variable_name = ""
  vartype = Int
  for line in eachline(fdfile)
    #skip empty lines
    line = strip(line)
    if length(line) == 0
      continue
    end
    if line == "/" || line == "&END"
      # end of header
      break
    end
    line = replace(line,"=" => " = ")
    line_array = [var for var in split(line, [' ',',']) if !isempty(var)]
    # search for '=' and put element before it as the variable name, and everything
    # after (before the next variable name) as a vector of values
    prev_el = ""
    elements = []
    newvec = true
    if variable_name != ""
      # in case the elements of the last variable continue on new line...
      elements = head[variable_name, vartype]
      newvec = false
    end
    push!(line_array, "\n")
    for el in line_array
      if el == "="
        if prev_el != ""
          if variable_name != ""
            # store the previous array
            head[variable_name] = elements
          end
          # case-insensitive variable names in the header
          variable_name = uppercase(prev_el)
          newvec = true
          prev_el = ""
        else
          error("No variable name before '=':"*line)
        end
      else
        if prev_el != ""
          elem = tryparse(Int, prev_el)
          if isnothing(elem)
            elem = tryparse(Float64,prev_el)
            if isnothing(elem)
              elem = strip(prev_el, ['"','\''])
              vartype = String
            else
              vartype = Float64
            end 
          else
            vartype = Int
          end
          if newvec
            elements = Vector{vartype}()
            newvec = false
          end
          push!(elements, elem)
        end
        prev_el = el
      end
    end
    if variable_name != ""
      # store the previous array
      head[variable_name] = elements
    end
  end
  # print(head)
  return head
end

"""
    read_integrals!(fd::FDump, dir::AbstractString)

  Read integrals from npy files.
"""
function read_integrals!(fd::FDump, dir::AbstractString)
  println("Read npy files")
  if !fd.uhf
    fd.int2 = mmap_integrals(fd, dir, "NPY2")
    fd.int1 = mmap_integrals(fd, dir, "NPY1")
  else
    fd.int2aa = mmap_integrals(fd, dir, "NPY2AA")
    fd.int2bb = mmap_integrals(fd, dir, "NPY2BB")
    fd.int2ab = mmap_integrals(fd, dir, "NPY2AB")
    fd.int1a = mmap_integrals(fd, dir, "NPY1A")
    fd.int1b = mmap_integrals(fd, dir, "NPY1B")
  end
  enuc = headvar(fd, "ENUC", Float64)
  if isnothing(enuc)
    error("ENUC option not found in fcidump")
  end
  fd.int0 = enuc
end

"""
    set_int2!(int2::Array{Float64,3}, i1, i2, i3, i4, integ, simtra, ab)

  Set 2-e integral in `int2` array to `integ` considering permutational symmetries.

  For not `ab`: particle symmetry is assumed.
  Integrals are stored in physicists' notation.
"""
function set_int2!(int2::Array{Float64,3}, i1, i2, i3, i4, integ, simtra, ab)
  @assert !ab
  if i2 == i4
    i24 = uppertriangular_index(i2,i4)
    int2[i1,i3,i24] = integ
    int2[i3,i1,i24] = integ
  elseif i2 < i4 
    int2[i1,i3,uppertriangular_index(i2,i4)] = integ
  else
    int2[i3,i1,uppertriangular_index(i4,i2)] = integ
  end
  if !simtra
    if i2 == i3
      i23 = uppertriangular_index(i2,i3)
      int2[i1,i4,i23] = integ
      int2[i4,i1,i23] = integ
    elseif i2 < i3
      int2[i1,i4,uppertriangular_index(i2,i3)] = integ
    else
      int2[i4,i1,uppertriangular_index(i3,i2)] = integ
    end
    if i1 == i4
      i14 = uppertriangular_index(i1,i4)
      int2[i2,i3,i14] = integ
      int2[i3,i2,i14] = integ
    elseif i1 < i4
      int2[i2,i3,uppertriangular_index(i1,i4)] = integ
    else
      int2[i3,i2,uppertriangular_index(i4,i1)] = integ
    end
    if i1 == i3
      i13 = uppertriangular_index(i1,i3)
      int2[i2,i4,i13] = integ
      int2[i4,i2,i13] = integ
    elseif i1 < i3
      int2[i2,i4,uppertriangular_index(i1,i3)] = integ
    else
      int2[i4,i2,uppertriangular_index(i3,i1)] = integ
    end
  end
end

"""
    set_int2!(int2::Array{Float64,4}, i1, i2, i3, i4, integ, simtra, ab)

  Set 2-e integral in `int2` array to `integ` considering permutational symmetries.

  For not `ab`: particle symmetry is assumed.
  Integrals are stored in physicists' notation.
"""
function set_int2!(int2::Array{Float64,4}, i1, i2, i3, i4, integ, simtra, ab)
  int2[i1,i3,i2,i4] = integ
  if !ab
    int2[i3,i1,i4,i2] = integ
  end
  if !simtra
    int2[i1,i4,i2,i3] = integ
    int2[i2,i3,i1,i4] = integ
    int2[i2,i4,i1,i3] = integ
    if !ab
      int2[i4,i1,i3,i2] = integ
      int2[i3,i2,i4,i1] = integ
      int2[i4,i2,i3,i1] = integ
    end
  end
end

function set_int1!(int1, i1, i2, integ, simtra)
  int1[i1,i2] = integ
  if !simtra
    int1[i2,i1] = integ
  end
end

"""
    read_integrals!(fd::FDump{N}, fdfile::IOStream)

  Read integrals from fcidump file
"""
function read_integrals!(fd::FDump{N}, fdfile::IOStream) where N
  norb = headvar(fd, "NORB", Int)
  if isnothing(norb)
    error("NORB option not found in fcidump")
  end
  st = headvar(fd, "ST", Int)
  if isnothing(st)
    error("ST option not found in fcidump")
  end
  simtra = (st > 0)
  set_zero!(fd, norb)
  if fd.uhf
    print("UHF")
    fd.int0 = read_integrals!(fd.int1a, fd.int1b, fd.int2aa, fd.int2bb, fd.int2ab, norb, fdfile, simtra)
  else
    fd.int0 = read_integrals!(fd.int1, fd.int2, norb, fdfile, simtra)
  end
end

function read_integrals!(int1, int2, norb, fdfile, simtra)
  int0 = 0.0
  for linestr in eachline(fdfile)
    line = split(linestr)
    if length(line) != 5
      # println("Last line: ",linestr)
      # skip lines (in the case there is something left from header)...
      continue
    end
    integ = parse(Float64,line[1])
    i1 = parse(Int,line[2])
    i2 = parse(Int,line[3])
    i3 = parse(Int,line[4])
    i4 = parse(Int,line[5])
    if i1 > norb || i2 > norb || i3 > norb || i4 > norb
      error("Index larger than norb: "*linestr)
    end
    if i4 > 0
      set_int2!(int2, i1, i2, i3, i4, integ, simtra, false)
    elseif i2 > 0
      set_int1!(int1, i1, i2, integ, simtra)
    elseif i1 <= 0
      int0 = integ
    end
  end
  return int0
end

function read_integrals!(int1a, int1b, int2aa, int2bb, int2ab, norb, fdfile, simtra)
  int0 = 0.0
  spincase = 0 # aa, bb, ab, a, b
  for linestr in eachline(fdfile)
    line = split(linestr)
    if length(line) != 5
      # println("Last line: ",linestr)
      # skip lines (in the case there is something left from header)...
      continue
    end
    integ = parse(Float64,line[1])
    i1 = parse(Int,line[2])
    i2 = parse(Int,line[3])
    i3 = parse(Int,line[4])
    i4 = parse(Int,line[5])
    if i1 > norb || i2 > norb || i3 > norb || i4 > norb
      error("Index larger than norb: "*linestr)
    end
    if i4 > 0
      if spincase == 0
        set_int2!(int2aa, i1, i2, i3, i4, integ, simtra, false)
      elseif spincase == 1
        set_int2!(int2bb, i1, i2, i3, i4, integ, simtra, false)
      elseif spincase == 2
        set_int2!(int2ab, i1, i2, i3, i4, integ, simtra, true)
      else
          error("Unexpected 2-el integrals for spin-case "*string(spincase))
      end
    elseif i2 > 0
      if spincase == 3
        set_int1!(int1a, i1, i2, integ, simtra)
      elseif spincase == 4
        set_int1!(int1b, i1, i2, integ, simtra)
      else
        error("Unexpected 1-el integrals for spin-case "*string(spincase))
      end
    elseif i1 <= 0
      if spincase < 5
        spincase += 1
      else
        int0 = integ
      end
    end
  end
  return int0
end

"""
    headvar(head::FDumpHeader, key::String)

  Check header for `key`, return value if a list, 
  or the element or nothing if not there.
"""
function headvar(head::FDumpHeader, key::String)
  val = get(head, key, nothing)
  if isnothing(val)
    return val
  elseif length(val) == 1
    return val[1]
  else
    return val
  end
end

"""
    headvars(head::FDumpHeader, key::String, ::Type{T}) where {T}

  Check header for `key` of type `T`, return a vector of values or nothing if not there. 
"""
function headvars(head::FDumpHeader, key::String, ::Type{T}) where {T}
  return get(head, key, T, nothing)
end

"""
    headvar(head::FDumpHeader, key::String, ::Type{T}) where {T}

  Check header for `key` of type `T`, return the first element or nothing if not there. 
"""
function headvar(head::FDumpHeader, key::String, ::Type{T}) where {T}
  val = headvars(head, key, T)
  if isnothing(val)
    return nothing
  else
    return val[1]
  end
end

"""
    headvar(fd::FDump, key::String)

  Check header for `key`, return value if a list, 
  or the element or nothing if not there.
"""
function headvar(fd::FDump, key::String )
  return headvar(fd.head, key)
end

"""
    headvars(fd::FDump, key::String, ::Type{T}) where {T}

  Check header for `key`, return a vector of values or nothing if not there. 
"""
function headvars(fd::FDump, key::String, ::Type{T}) where {T}
  return headvars(fd.head, key, T)
end

"""
    headvar(fd::FDump, key::String, ::Type{T}) where {T}

  Check header for `key`, return the first element or nothing if not there. 
"""
function headvar(fd::FDump, key::String, ::Type{T}) where {T}
  return headvar(fd.head, key, T)
end

"""
    mmap_integrals(fd::FDump, dir::AbstractString, key::AbstractString)

  Memory-map integral file (from head[key])
"""
function mmap_integrals(fd::FDump, dir::AbstractString, key::AbstractString)
  file = headvar(fd, key, String)
  if isnothing(file)
    error(key*" option not found in fcidump")
  end
  if !isabspath(file)
    file = joinpath(dir,file)
  end
  # return npzread(file)
  return mnpymmap(file)
end

"""
    write_fcidump(fd::FDump, fcidump::String, tol=1e-12)

  Write fcidump file.
"""
function write_fcidump(fd::FDump, fcidump::String, tol=1e-12)
  println("Write fcidump $fcidump"...)
  fdf = open(fcidump, "w")
  write_header(fd, fdf)
  write_integrals(fd, fdf, tol)
  close(fdf)
end

"""
    write_header(fd::FDump, fdf)

  Write header of fcidump file.
"""
function write_header(fd::FDump, fdf)
  println(fdf, "&FCI")
  for key in FDUMP_KEYS
    val = headvar(fd, key)
    if !isnothing(val)
      println(fdf, " ", key, "=", join(val, ","), ",")
    end
  end
  for (key,val) in fd.head
    if key in FDUMP_KEYS
      continue
    end
    if key in FDUMP_OPTIONAL && val[1] == 0
      continue
    end
    if typeof(val[1]) <: AbstractString
      # add quotes around each element
      val = ["\"$v\"" for v in val]
    end
    println(fdf, " ", key, "=", join(val, ","), ",")
  end
  println(fdf, "/")
end

"""
    print_int_value(fdf, integ, i1, i2, i3, i4)

  Print integral value to fdf file.
"""
function print_int_value(fdf, integ, i1, i2, i3, i4)
  @printf(fdf, "%23.15e %3i %3i %3i %3i\n", integ, i1, i2, i3, i4)
end

"""
    write_integrals(fd::FDump, fdf, tol)

  Write integrals to fdf file.
"""
function write_integrals(fd::FDump, fdf, tol)
  st = headvar(fd, "ST", Int)
  if isnothing(st)
    error("ST option not found in fcidump")
  end
  simtra::Bool = (st > 0)
  if !fd.uhf
    write_integrals2(fd.int2, fdf, tol, simtra)
    write_integrals1(fd.int1, fdf, tol, simtra)
  else
    write_integrals2(fd.int2aa, fdf, tol, simtra)
    print_int_value(fdf,0.0,0,0,0,0)
    write_integrals2(fd.int2bb, fdf, tol, simtra)
    print_int_value(fdf,0.0,0,0,0,0)
    write_integrals2ab(fd.int2ab, fdf, tol, simtra)
    print_int_value(fdf,0.0,0,0,0,0)
    write_integrals1(fd.int1a, fdf, tol, simtra)
    print_int_value(fdf,0.0,0,0,0,0)
    write_integrals1(fd.int1b, fdf, tol, simtra)
    print_int_value(fdf,0.0,0,0,0,0)
  end
  print_int_value(fdf,fd.int0,0,0,0,0)
end

"""
    write_integrals2(int2::Array{Float64,3}, fdf, tol, simtra)

  Write 2-e integrals to fdf file.
"""
function write_integrals2(int2::Array{Float64,3}, fdf, tol, simtra)
  write_integrals2_ = simtra ? write_integrals2_simtra : write_integrals2_normal
  inds = (p,q,r,s) -> CartesianIndex(p,q,uppertriangular_index(r,s))
  indslow = (p,q,r,s) -> CartesianIndex(q,p,uppertriangular_index(s,r))
  write_integrals2_(int2, inds, indslow, fdf, tol)
end

function write_integrals2(int2::Array{Float64,4}, fdf, tol, simtra)
  write_integrals2_ = simtra ? write_integrals2_simtra : write_integrals2_normal
  inds = (p,q,r,s) -> CartesianIndex(p,q,r,s)
  write_integrals2_(int2, inds, inds, fdf, tol)
end

function write_integrals2_simtra(int2, inds, indslow, fdf, tol)
  norb = size(int2,1)
  for p = 1:norb
    for q = 1:norb
      for r = 1:p-1
        # lower triangle (q>s)
        for s = 1:q-1
          val = int2[indslow(p,r,q,s)]
          if abs(val) > tol
            print_int_value(fdf, val, p, q, r, s)
          end
        end
        # upper triangle (q<=s)
        for s = q:norb
          val = int2[inds(p,r,q,s)]
          if abs(val) > tol
            print_int_value(fdf, val, p, q, r, s)
          end
        end
      end
      # r==p case
      r = p
      for s = 1:q
        val = int2[indslow(p,r,q,s)]
        if abs(val) > tol
          print_int_value(fdf, val, p, q, r, s)
        end
      end
    end
  end
end
function write_integrals2_normal(int2, inds, indslow, fdf, tol)
  norb = size(int2,1)
  for p in 1:norb
    for q in 1:p
      for r in 1:p
        for s in 1:r
          if r*(r-1)/2+s <= p*(p-1)/2+q
            if s < q 
              # lower triangle
              val = int2[indslow(p,r,q,s)]
            else
              # upper triangle
              val = int2[inds(p,r,q,s)]
            end
            if abs(val) > tol
              print_int_value(fdf,val,p,q,r,s)
            end
          end
        end
      end
    end
  end
end

function write_integrals2ab(int2, fdf, tol, simtra)
  norb = size(int2,1)
  if simtra
    for p = 1:norb
      for q = 1:norb
        for r = 1:norb
          for s = 1:norb
            val = int2[p,r,q,s]
            if abs(val) > tol
              print_int_value(fdf,val,p,q,r,s)
            end
          end
        end
      end
    end
  else
    # normal αβ case
    for p in 1:norb
      for q in 1:p
        for r in 1:norb
          for s in 1:r
            val = int2[p,r,q,s]
            if abs(val) > tol
              print_int_value(fdf,val,p,q,r,s)
            end
          end
        end
      end
    end
  end
end

"""
    write_integrals1(int1, fdf, tol, simtra)

  Write 1-e integrals to fdf file.
"""
function write_integrals1(int1, fdf, tol, simtra)
  norb = size(int1,1)
  if simtra
    for p = 1:norb
      for q = 1:norb
        val = int1[p,q]
        if abs(val) > tol
          print_int_value(fdf,val,p,q,0,0)
        end
      end
    end
  else
    # normal case
    for p = 1:norb
      for q = 1:p
        val = int1[p,q]
        if abs(val) > tol
          print_int_value(fdf,val,p,q,0,0)
        end
      end
    end
  end
end

""" 
    transform_fcidump(fd::FDump, Tl::AbstractArray, Tr::AbstractArray)

  Transform integrals to new basis using Tl and Tr transformation matrices. 
  For UHF fcidump, Tl and Tr are arrays of matrices for α and β spin.
  If Tl and Tr are arrays of arrays, then the function transforms rhf fcidump to uhf fcidump.
"""
function transform_fcidump(fd::FDump, Tl::AbstractArray, Tr::AbstractArray) 
  println("Transform integrals...")
  if length(Tl) == 2 && typeof(Tl[1]) <: AbstractArray
    genuhfdump = true
  else
    genuhfdump = false
    @assert !fd.uhf # from uhf fcidump can generate only uhf fcidump
  end
  if fd.uhf
    fd.int2aa = transform_int2(fd.int2aa, Tl[1], Tl[1], Tr[1], Tr[1])
    fd.int2bb = transform_int2(fd.int2bb, Tl[2], Tl[2], Tr[2], Tr[2])
    fd.int2ab = transform_int2_Q(fd.int2ab, Tl[1], Tl[2], Tr[1], Tr[2])
    fd.int1a = transform_int1(fd.int1a, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1b, Tl[2], Tr[2])
  elseif genuhfdump
    # change fcidump from rhf to uhf format
    fd.int2aa = transform_int2(fd.int2, Tl[1], Tl[1], Tr[1], Tr[1])
    fd.int2bb = transform_int2(fd.int2, Tl[2], Tl[2], Tr[2], Tr[2])
    fd.int2ab = transform_int2_Q(fd.int2, Tl[1], Tl[2], Tr[1], Tr[2])
    fd.int1a = transform_int1(fd.int1, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1, Tl[2], Tr[2])
    fd.int2 = zeros(fill(0,ndims(fd.int2))...)
    fd.int1 = zeros(0,0)
    fd.head["IUHF"] = [1]
    fd.uhf = true
  else
    fd.int2 = transform_int2(fd.int2, Tl, Tl, Tr, Tr)
    fd.int1 = transform_int1(fd.int1, Tl, Tr)
  end
end

"""
    transform_int2(int2::Array{Float64,3}, Tl::AbstractArray, Tl2::AbstractArray, 
                   Tr::AbstractArray, Tr2::AbstractArray)

  Transform 2-e integrals to new basis using `Tl`/`Tl2` and `Tr`/`Tr2` transformation matrices.

  ``v_{pq}^{rs} = v_{p'q'}^{r's'}``* `Tl`[p',p] * `Tl2`[q',q] * `Tr`[r',r] * `Tr2`[s',s]

  The last two indices are stored as a single uppertriangular index.
"""
function transform_int2(int2::Array{Float64,3}, Tl::AbstractArray, Tl2::AbstractArray, 
                        Tr::AbstractArray, Tr2::AbstractArray)
  norb = size(int2,1)
  int2t = zeros(norb,norb,norb*(norb+1)÷2)
  int_3i = zeros(norb,norb,norb)
  for s = 1:norb
    rs = strict_uppertriangular_range(s)
    rrange = 1:s-1
    if length(rs) > 0
      @tensoropt int_3i[p,q,r] = int2[:,:,rs][p',q',r'] * Tl[p',p] * Tl2[q',q] * Tr[rrange,:][r',r]
    end
    # contribution from the diagonal <p'q'|s's'> 
    ss = uppertriangular_index(s, s)
    @tensoropt int_3i[p,q,r] += 0.5*int2[:,:,ss][p',q'] * Tl[p',p] * Tl2[q',q] * Tr[s,:][r]
    for s1 = 1:norb
      rs1 = uppertriangular_range(s1)
      rrange = 1:s1
      Tr2ss1 = Tr2[s,s1]
      @tensoropt int2t[:,:,rs1][p,q,r] += int_3i[:,:,rrange][p,q,r] * Tr2ss1
      @tensoropt int2t[:,:,rs1][p,q,r] += int_3i[:,:,s1][q,p] * Tr2[s,rrange][r]
    end
  end
  return int2t
end
function transform_int2(int2::Array{Float64,4}, Tl::AbstractArray, Tl2::AbstractArray, 
                        Tr::AbstractArray, Tr2::AbstractArray)
  return transform_int2_Q(int2, Tl, Tl2, Tr, Tr2)
end
"""
    transform_int2_Q(int2::Array{Float64,3}, Tl::AbstractArray, Tl2::AbstractArray, 
                   Tr::AbstractArray, Tr2::AbstractArray)

  Transform 2-e integrals to new basis using `Tl`/`Tl2` and `Tr`/`Tr2` transformation matrices.

  ``v_{pq}^{rs} = v_{p'q'}^{r's'}``* `Tl`[p',p] * `Tl2`[q',q] * `Tr`[r',r] * `Tr2`[s',s]

  The result is a full 4-index tensor.
"""
function transform_int2_Q(int2::Array{Float64,3}, Tl::AbstractArray, Tl2::AbstractArray, 
                        Tr::AbstractArray, Tr2::AbstractArray)
  norb = size(int2,1)
  int2t = zeros(norb,norb,norb,norb)
  int_3i = zeros(norb,norb,norb)
  int_3i2 = zeros(norb,norb,norb)
  for s = 1:norb
    rs = strict_uppertriangular_range(s)
    rrange = 1:s-1
    if length(rs) > 0
      @tensoropt int_3i[p,q,r] = int2[:,:,rs][p',q',r'] * Tl[p',p] * Tl2[q',q] * Tr[rrange,:][r',r]
      @tensoropt int_3i2[p,q,r] = int2[:,:,rs][p',q',r'] * Tl2[p',p] * Tl[q',q] * Tr2[rrange,:][r',r]
    end
    # contribution from the diagonal <p'q'|s's'> 
    ss = uppertriangular_index(s, s)
    @tensoropt int_3i[p,q,r] += 0.5*int2[:,:,ss][p',q'] * Tl[p',p] * Tl2[q',q] * Tr[s,:][r]
    @tensoropt int_3i2[p,q,r] += 0.5*int2[:,:,ss][p',q'] * Tl2[p',p] * Tl[q',q] * Tr2[s,:][r]

    @tensoropt int2t[p,q,r,s'] += int_3i[p,q,r] * Tr2[s,:][s']
    @tensoropt int2t[p,q,r,s'] += int_3i2[q,p,s'] * Tr[s,:][r]
  end
  return int2t
end
function transform_int2_Q(int2::Array{Float64,4}, Tl::AbstractArray, Tl2::AbstractArray, 
                        Tr::AbstractArray, Tr2::AbstractArray)
  @tensoropt int2t[p,q,r,s] := int2[p',q',r',s']*Tl[p',p]*Tl2[q',q]*Tr[r',r]*Tr2[s',s]
  return int2t
end

""" 
    transform_int1(int1::AbstractArray, Tl::AbstractArray,  Tr::AbstractArray)

  Transform 1-e integrals to new basis using `Tl` and `Tr` transformation matrices.
"""
function transform_int1(int1::AbstractArray, Tl::AbstractArray,  Tr::AbstractArray)
  @tensoropt int1t[p,q] := int1[p',q'] * Tl[p',p] * Tr[q',q]
  return int1t
end

"""
    reorder_orbs_int2(int2::AbstractArray, orbs)

  Reorder orbitals in 2-e integrals according to `orbs`.

  `orbs`can be a subset of orbitals or a permutation of orbitals.
  Return `int2[orbs[p],orbs[q],orbs[r],orbs[s]]` or the triangular version.
"""
function reorder_orbs_int2(int2::AbstractArray, orbs)
  norb = size(int2,1)
  norbnew = length(orbs)
  if orbs == 1:norb
    return int2
  end
  if norbnew == 0
    if ndims(int2) == 3
      return zeros(0,0,0)
    else
      return zeros(0,0,0,0)
    end
  end
  @assert maximum(orbs) <= norb && minimum(orbs) > 0 "Orbital index out of range"
  if ndims(int2) == 3
    # triangular
    int2t = zeros(norbnew, norbnew, norbnew*(norbnew+1)÷2)
    for s = 1:norbnew
      for r = 1:s
        ro = orbs[r]
        so = orbs[s]
        if ro <= so
          int2t[:,:,uppertriangular_index(r,s)] = int2[orbs,orbs,uppertriangular_index(ro, so)]
        else
          int2t[:,:,uppertriangular_index(r,s)] = permutedims(int2[orbs,orbs,uppertriangular_index(so, ro)], [2,1])
        end
      end
    end
  else
    int2t = int2[orbs,orbs,orbs,orbs]
  end
  return int2t
end

"""
    int1_npy_filename(fd::FDump, spincase::Symbol=:α)

  Return filename for 1-e integrals in npy format.
  `spincase` can be `:α` or `:β` for UHF fcidump.
"""
function int1_npy_filename(fd::FDump, spincase::Symbol=:α)
  if !fd.uhf
    file = headvar(fd, "NPY1", String)
    if isnothing(file)
      file = "int1.npy"
      # fd.head["NPY1"] = [file]
    end
  else
    if spincase == :α
      file = headvar(fd, "NPY1A", String)
      if isnothing(file)
        file = "int1a.npy"
        # fd.head["NPY1A"] = [file]
      end
    else
      file = headvar(fd, "NPY1B", String)
      if isnothing(file)
        file = "int1b.npy"
        # fd.head["NPY1B"] = [file]
      end
    end
  end
  return file::String
end

"""
    int2_npy_filename(fd::FDump, spincase::Symbol=:α)

  Return filename for 2-e integrals in npy format. 
  `spincase` can be `:α`, `:β` or `:αβ` for UHF fcidump.
"""
function int2_npy_filename(fd::FDump, spincase::Symbol=:α)
  if !fd.uhf
    file = headvar(fd, "NPY2", String)
    if isnothing(file)
      file = "int2.npy"
      # fd.head["NPY2"] = [file]
    end
  else
    if spincase == :α
      file = headvar(fd, "NPY2AA", String)
      if isnothing(file)
        file = "int2aa.npy"
        # fd.head["NPY2AA"] = [file]
      end
    elseif spincase == :β
      file = headvar(fd, "NPY2BB", String)
      if isnothing(file)
        file = "int2bb.npy"
        # fd.head["NPY2BB"] = [file]
      end
    else
      file = headvar(fd, "NPY2AB", String)
      if isnothing(file)
        file = "int2ab.npy"
        # fd.head["NPY2AB"] = [file]
      end
    end
  end
  return file::String
end

end #module
