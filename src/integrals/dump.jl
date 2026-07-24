"""
Read and write fcidump format integrals.
Individual arrays of integrals can also be in *.npy format
"""
module FciDumps

using DocStringExtensions
using Printf
using NPZ
using Buffers
using ..ElemCo.Utils
using ..ElemCo.MNPY
using ..ElemCo.MTensorOperations
using ..ElemCo.QMTensors

export FDump, TFDump, QFDump 
export fd_origin, fd_ismodified, read_fcidump, write_fcidump
export headvar, headvars, integ1, integ2, integ2_ss, integ2_os, triang
export reorder_orbs_int2, modify_header!
export int1_npy_filename, int2_npy_filename
export is_similarity_transformed

# optional variables which won't be written if =0
const FDUMP_OPTIONAL=["IUHF", "ST", "III", "ICMPLX"]

"""prefered order of keys in fcidump header (optional keys are not included)"""
const FDUMP_KEYS=["NORB", "NELEC", "MS2", "ISYM", "ORBSYM" ]

"""
    FDumpHeader

  Header of fcidump file
"""
@kwdef mutable struct FDumpHeader
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
    FDump{T,N}

  Molecular integrals 

  The 2-e integrals are stored in the physicists' notation: `int2[pqrs]` ``= <pq|rs>=v_{pq}^{rs}``

  `T` denotes the element type of integrals (`Float64` or `ComplexF64`)

  `N` denotes the number of indices in the 2-e-integral tensors,
  for `N=3` (usual) the last two indices are stored as a single uppertriangular index (r <= s)

  $(TYPEDFIELDS)
"""
@kwdef mutable struct FDump{T<:Number,N}
  """ 2-e⁻ integrals for restricted orbitals fcidump. """
  int2::Array{T,N} = zeros(T, ntuple(d->0,Val(N)))
  """ αα 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2aa::Array{T,N} = zeros(T, ntuple(d->0,Val(N)))
  """ ββ 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2bb::Array{T,N} = zeros(T, ntuple(d->0,Val(N)))
  """ αβ 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2ab::Array{T,4} = zeros(T, 0,0,0,0)
  """ e⁻e⁺ 2-body integrals for restricted orbitals fcidump. """
  int2ep::Array{T,4} = zeros(T, 0,0,0,0)
  """ 1-e⁻ integrals for restricted orbitals fcidump. """
  int1::Matrix{T} = zeros(T, 0,0)
  """ α 1-e⁻ integrals for unrestricted orbitals fcidump. """
  int1a::Matrix{T} = zeros(T, 0,0)
  """ β 1-e⁻ integrals for unrestricted orbitals fcidump. """
  int1b::Matrix{T} = zeros(T, 0,0)
  """ 1-e⁺ integrals for restricted orbitals fcidump. """
  int1p::Matrix{T} = zeros(T, 0,0)
  """ core energy """
  int0::Float64 = 0.0
  """ header of fcidump file, a dictionary of arrays. """
  head::FDumpHeader = FDumpHeader()
  """ path of the original fcidump file, empty if created from scratch. """
  origin::String = ""
  """`⟨false⟩` has the integrals been modified after reading? """
  modified::Bool = false
  """`⟨false⟩` a convinience variable, has to coincide with `head["IUHF"][1] > 0`. """
  uhf::Bool = false
  """`⟨false⟩` a convenience variable, has to coincide with `head["NPOS"][1] > 0`. """
  epdump::Bool = false
  """`⟨false⟩` 3-index DF integrals are stored in scratch (`mmL`) and need contraction to 4-index. """
  df3idx::Bool = false
  """ for ElemCo-generated reduced (frozen-core/deleted-virtual) dumps: the contiguous full-space
      (original) orbital range of the active orbitals (frozen core below it, deleted virtuals above
      it), i.e. active orbital `k` corresponds to full orbital `orig_orbs[k]`. Empty (`1:0`) for
      externally-read or non-reduced dumps. Used to translate user-supplied orbital lists
      (`occa`/`occb`/`active`), which always refer to the full MO space, to the active space. """
  orig_orbs::UnitRange{Int} = 1:0
end

const TFDump{T<:Number} = FDump{T,3}
const QFDump{T<:Number} = FDump{T,4}

"""
    FDump{T2,N}(fd::FDump{T1,N})

  Convert an `FDump{T1,N}` to `FDump{T2,N}` by converting all integral arrays.
"""
function FDump{T2,N}(fd::FDump{T1,N}) where {T1<:Number,T2<:Number,N}
  T1 === T2 && return fd
  FDump{T2,N}(
    int2 = Array{T2,N}(fd.int2),
    int2aa = Array{T2,N}(fd.int2aa),
    int2bb = Array{T2,N}(fd.int2bb),
    int2ab = Array{T2,4}(fd.int2ab),
    int1 = Matrix{T2}(fd.int1),
    int1a = Matrix{T2}(fd.int1a),
    int1b = Matrix{T2}(fd.int1b),
    int0 = fd.int0,
    head = fd.head,
    origin = fd.origin,
    modified = fd.modified,
    uhf = fd.uhf,
    df3idx = fd.df3idx,
    orig_orbs = fd.orig_orbs,
  )
end

Base.convert(::Type{FDump{T,N}}, fd::FDump{T,N}) where {T<:Number,N} = fd
Base.convert(::Type{FDump{T2,N}}, fd::FDump{T1,N}) where {T1<:Number,T2<:Number,N} = FDump{T2,N}(fd)

""" 
  is_triang(fd::FDump)
  
  If true: an uppertriangular index for last two indices of 2e⁻ integrals is used.
""" 
is_triang(fd::FDump{<:Number,3}) = true
is_triang(fd::FDump{<:Number,4}) = false

"""
    FDump(int2::Array{T,N}, int1::Matrix{T}, int0::Float64, head::FDumpHeader) where {T,N}

  Spin-free fcidump
"""
FDump(int2::Array{T,N}, int1::Matrix{T}, int0::Float64, head::FDumpHeader) where {T<:Number,N} = FDump{T,N}(; int2, int1, int0, head)
"""
    FDump(int2aa::Array{T,N}, int2bb::Array{T,N}, int2ab::Array{T,4}, int1a::Matrix{T}, int1b::Matrix{T}, int0::Float64, head::FDumpHeader) where {T,N}

  Spin-polarized fcidump
"""
FDump(int2aa::Array{T,N}, int2bb::Array{T,N}, int2ab::Array{T,4}, int1a::Matrix{T}, int1b::Matrix{T}, int0::Float64, head::FDumpHeader) where {T<:Number,N} = FDump{T,N}(; int2aa, int2bb, int2ab, int1a, int1b, int0, head, uhf=true)

"""
    FDump{T,N}(norb, nelec; ms2=0, isym=1, orbsym=[], uhf=false, simtra=false)

  Create a new FDump object with element type `T`.
"""
function FDump{T,N}(norb::Int, nelec::Int; npos::Int=0, ms2::Int=0, isym::Int=1, orbsym::Vector{Int}=Int[], 
               uhf=false, simtra=false) where {T<:Number,N}
  fd = FDump{T,N}()
  fd.head["NORB"] = [norb]
  fd.head["NELEC"] = [nelec]
  if npos > 0
    fd.head["NPOS"] = [npos]
    fd.epdump = true
  end
  fd.head["MS2"] = [ms2]
  fd.head["ISYM"] = [isym]
  if isempty(orbsym)
    fd.head["ORBSYM"] = ones(Int,norb)
  else
    fd.head["ORBSYM"] = orbsym
  end
  fd.head["IUHF"] = uhf ? [1] : [0]
  fd.head["ST"] = simtra ? [1] : [0]
  fd.head["ICMPLX"] = T <: Complex ? [1] : [0]
  fd.uhf = uhf
  return fd
end

"""
    modify_header!(fd::FDump, norb, nelec; ms2=-1, isym=-1, orbsym=[])

  Modify header of FDump object
"""
function modify_header!(fd::FDump, norb::Int, nelec::Int; npos::Int=-1, ms2::Int=-1, isym::Int=-1, orbsym::Vector{Int}=Int[])
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
  if npos >= 0
    fd.head["NPOS"] = [npos]
    if npos > 0
      fd.epdump = true
    else
      fd.epdump = false
    end
  end
end

function Base.isempty(fd::FDump)
  return isempty(fd.head)
end

"""
    fd_ismodified(fd::FDump)

  Return true if the object has been modified after reading
"""
function fd_ismodified(fd::FDump)
  return fd.modified
end

"""
    fd_origin(fd::FDump)

  Return the path of the original fcidump file, empty if created from scratch.
"""
function fd_origin(fd::FDump)
  return fd.origin
end

"""
    is_similarity_transformed(fd::FDump)

  Return true if the fcidump is similarity transformed.
  An empty dump (e.g. AO-direct runs, where `EC.fd` stays unpopulated) is not.
"""
is_similarity_transformed(fd::FDump) = !isempty(fd) && headvar(fd, "ST", Int) > 0

"""
    uses_reduced_permsym(fd::FDump{T}) where {T<:Number}

  Return true if the reduced (similarity-transformed) permutational symmetry has to be
  used when reading/writing the integrals.

  This is the case for similarity-transformed fcidumps (`ST=1`), and *always* for complex
  integrals, which lack the full permutational symmetry of real integrals. Complex integrals
  therefore use the same symmetry as `ST=1` without setting the `ST` flag.
"""
uses_reduced_permsym(fd::FDump{T}) where {T<:Number} = is_similarity_transformed(fd) || T <: Complex

"""
    set_zero!(fd::FDump, norb::Int=0)

  Set all integrals to zero.

  If `norb` is not provided, the integrals are set to zero with the same dimensions as before.
"""
function set_zero!(fd::FDump{T,N}, norb::Int=0) where {T,N}
  fd.int0 = 0.0
  if norb <= 0
    if fd.uhf
      fill!(fd.int1a, zero(T))
      fill!(fd.int1b, zero(T))
      fill!(fd.int2aa, zero(T))
      fill!(fd.int2bb, zero(T))
      fill!(fd.int2ab, zero(T))
    else
      fill!(fd.int1, zero(T))
      fill!(fd.int2, zero(T))
    end
    if fd.epdump
      fill!(fd.int1p, 0.0)
      fill!(fd.int2ep, 0.0)
    end
  else
    if fd.uhf
      fd.int1a = zeros(T, norb,norb)
      fd.int1b = zeros(T, norb,norb)
      fd.int2aa = get_int2_zeros(fd.int2aa, norb)
      fd.int2bb = get_int2_zeros(fd.int2bb, norb)
      fd.int2ab = get_int2_zeros(fd.int2ab, norb)
    else
      fd.int1 = zeros(T, norb,norb)
      fd.int2 = get_int2_zeros(fd.int2, norb)
      fd.int2ep = get_int2_zeros(fd.int2ep, norb)
    end
    if fd.epdump
      fd.int1p = zeros(norb,norb)
      fd.int2ep = get_int2_zeros(fd.int2ep, norb)
    end
  end
end

function get_int2_zeros(int2::Array{T,3}, norb) where T
  return zeros(T, norb,norb,(norb+1)*norb÷2)
end

function get_int2_zeros(int2::Array{T,4}, norb) where T
  return zeros(T, norb,norb,norb,norb)
end

"""
    integ1(fd::FDump, spincase::Symbol=:α)

  Return 1-e⁻ integrals (for UHF fcidump: for `spincase`).
  `spincase` can be `:α` or `:β` or `:p`.
"""
function integ1(fd::FDump, spincase::Symbol=:α)
  if spincase == :p
    @assert fd.epdump "Spincase :p only for positron fcidump"
    return fd.int1p
  end
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

  Return 2-e⁻ or e⁻e⁺ integrals (for UHF fcidump: for `spincase`).
  `spincase` can be `:α`, `:β`, `:αβ` or `:p`.

  Use type-stable versions instead: 
  [`integ2_ss`](@ref) for same-spin integrals and [`integ2_os`](@ref) for opposite-spin integrals.
"""
function integ2(fd::FDump, spincase::Symbol=:α)
  if spincase == :p
    @assert fd.epdump "Spincase :p only for positron fcidump"
    return fd.int2ep
  end
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
    integ2_ss(fd::FDump, spincase::Symbol=:α)

  Return 2-e⁻ or e⁻e⁺ integrals (for UHF fcidump: for `spincase`).
  `spincase` can be `:α`, `:β`, or `:p`.
"""
function integ2_ss(fd::FDump, spincase::Symbol=:α)
  if spincase == :p
    @assert fd.epdump "Spincase :p only for positron fcidump"
    return fd.int2ep
  end
  if !fd.uhf
    return fd.int2
  elseif spincase == :α
    return fd.int2aa
  elseif spincase == :β
    return fd.int2bb
  else
    error("Only α and β are allowed for spincase")
  end
end

"""
    integ2_os(fd::FDump)

  Return αβ 2-e⁻ integrals for UHF fcidump
"""
function integ2_os(fd::FDump)
  @assert fd.uhf "Only for UHF"
  return fd.int2ab
end

"""
    read_fcidump(fcidump::String, ::Type{T}, ::Val{N}) where {T<:Number, N}

  Read ascii file (possibly with integrals in npy files).
""" 
function read_fcidump(fcidump::String, ::Type{T}, ::Val{N}) where {T<:Number, N}
  fdf = open(fcidump)
  head = read_header(fdf)
  # auto-detect complex integrals from ICMPLX flag
  icmplx = headvar(head, "ICMPLX", Int)
  @assert (icmplx > 0) == (T <: Complex) "ICMPLX flag in fcidump header does not match the provided element type"
  fd = FDump{T,N}()
  fd.head = head
  fd.origin = fcidump
  fd.uhf = (headvar(fd, "IUHF", Int) > 0)
  simtra = (headvar(fd, "ST", Int) > 0)
  positron = headvar(fd, "NPOS", Int)
  if simtra
    println("Non-Hermitian")
  end
  if !isnothing(positron)
    fd.epdump = (positron > 0)
    if fd.epdump
      if fd.uhf
        error("UHF positron fcidump not supported")
      end
      println("Positron fcidump elements detected")
    end
  end
  done = false
  if !isnothing(headvar(fd, "NPY2", String)) || !isnothing(headvar(fd, "NPY2AA", String))
    # assert that no positrons present
    if !isnothing(positron) && positron > 0
      error("Positron fcidump with npy files not supported")
    end
    # try to read integrals from npy files
    done = read_integrals!(fd, dirname(fcidump))
  end
  if !done
    # read integrals from fcidump file
    read_integrals!(fd, fdf)
  end
  close(fdf)
  return fd
end

"""
    read_fcidump(fcidump::String, ::Type{T}=Float64) where {T<:Number}

  Read ascii file (possibly with integrals in npy files).
  The element type of the integrals is `T` (default: `Float64`).
"""
read_fcidump(fcidump::String, ::Type{T}=Float64) where {T<:Number} = read_fcidump(fcidump, T, Val(3))

"""
    read_header(fdfile::IOStream)

  Read header of fcidump file.
"""
function read_header(fdfile)
  # put some defaults...
  head = FDumpHeader()
  head["IUHF"] = [0]
  head["ST"] = [0]
  head["ICMPLX"] = [0]
  line_array = String[]
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
    append!(line_array, split(line, [' ',','], keepempty=false))
  end
  push!(line_array, "\n")
  # search for '=' and put element before it as the variable name, and everything
  # after (before the next variable name) as a vector of values
  comments = String[]
  variable_name, ipos = read_elements!(comments, line_array, 1)
  while ipos < length(line_array)
    ipos += 1
    el_str = line_array[ipos]
    elem = tryparse(Int, el_str)
    if !isnothing(elem)
      head[variable_name] = Int[elem]
      variable_name, ipos = read_elements!(head[variable_name,Int], line_array, ipos+1)
    else
      elem = tryparse(Float64, el_str)
      if !isnothing(elem)
        head[variable_name] = Float64[elem]
        variable_name, ipos = read_elements!(head[variable_name,Float64], line_array, ipos+1)
      else
        elem = strip(el_str, ['"','\''])
        head[variable_name] = String[elem]
        variable_name, ipos = read_elements!(head[variable_name,String], line_array, ipos+1)
      end 
    end
  end
  # print(head)
  return head
end

function read_elements!(elements::Vector{T}, line_array::Vector{String}, ipos::Int) where T
  variable_name = ""
  prev_el = ""
  while ipos <= length(line_array)
    el = line_array[ipos]
    if el == "=" 
      if prev_el != ""
        # case-insensitive variable names in the header
        variable_name = uppercase(prev_el)
        break
      else
        error("No variable name before '=': $(line_array)")
      end
    else
      if prev_el != ""
        if T == String
          push!(elements, strip(prev_el, ['"','\'']))
        else
          push!(elements, parse(T, prev_el))
        end
      end
      prev_el = el
    end
    ipos += 1
  end
  return variable_name, ipos
end

"""
    read_integrals!(fd::FDump, dir::AbstractString)

  Read integrals from npy files. 

Returns `true` if successful.
"""
function read_integrals!(fd::FDump, dir::AbstractString)
  println("Read npy files")
  if !fd.uhf
    fd.int2 = mmap_integrals(fd, dir, "NPY2", fd.int2)
    fd.int1 = mmap_integrals(fd, dir, "NPY1", fd.int1)
    success = length(fd.int2) > 0 && length(fd.int1) > 0
  else
    fd.int2aa = mmap_integrals(fd, dir, "NPY2AA", fd.int2aa)
    fd.int2bb = mmap_integrals(fd, dir, "NPY2BB", fd.int2bb)
    fd.int2ab = mmap_integrals(fd, dir, "NPY2AB", fd.int2ab)
    fd.int1a = mmap_integrals(fd, dir, "NPY1A", fd.int1a)
    fd.int1b = mmap_integrals(fd, dir, "NPY1B", fd.int1b)
    success = length(fd.int2aa) > 0 && length(fd.int2bb) > 0 && length(fd.int2ab) > 0 && length(fd.int1a) > 0 && length(fd.int1b) > 0
  end
  enuc = headvar(fd, "ENUC", Float64)
  if isnothing(enuc)
    error("ENUC option not found in fcidump")
  end
  fd.int0 = enuc
  return success
end

"""
    set_int2!(int2::Array{<:Number,3}, i1, i2, i3, i4, integ, simtra, ab)

  Set 2-e integral in `int2` array to `integ` considering permutational symmetries.

  For not `ab`: particle symmetry is assumed.
  Integrals are stored in physicists' notation.
"""
function set_int2!(int2::Array{<:Number,3}, i1, i2, i3, i4, integ, simtra, ab)
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
    set_int2!(int2::Array{<:Number,4}, i1, i2, i3, i4, integ, simtra, ab)

  Set 2-e integral in `int2` array to `integ` considering permutational symmetries.

  For not `ab`: particle symmetry is assumed.
  Integrals are stored in physicists' notation.
"""
function set_int2!(int2::Array{<:Number,4}, i1, i2, i3, i4, integ, simtra, ab)
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
    read_integrals!(fd::FDump{<:Number,N}, fdfile::IOStream)

  Read integrals from fcidump file

Returns `true` if successful.
"""
function read_integrals!(fd::FDump{<:Number,N}, fdfile::IOStream) where N
  norb = headvar(fd, "NORB", Int)
  if isnothing(norb)
    error("NORB option not found in fcidump")
  end
  st = headvar(fd, "ST", Int)
  if isnothing(st)
    error("ST option not found in fcidump")
  end
  simtra = uses_reduced_permsym(fd)
  set_zero!(fd, norb)
  if fd.uhf
    if fd.epdump
      error("Positron fcidump with UHF not supported")
    end
    print("UHF")
    fd.int0 = read_integrals!(fd.int1a, fd.int1b, fd.int2aa, fd.int2bb, fd.int2ab, norb, fdfile, simtra)
  else
    if fd.epdump
      fd.int0 = read_integrals!(fd.int1, fd.int2, fd.int1p, fd.int2ep, norb, fdfile, simtra)
    else
      fd.int0 = read_integrals!(fd.int1, fd.int2, norb, fdfile, simtra)
    end
  end
  return true
end

"""
    parse_integ_value(::Type{T}, linestr::AbstractString) where T

  Parse integral value and indices from a fcidump line.

  For complex integrals (`ICMPLX=1`), the format is `(real,imaginary) i1 i2 i3 i4`.
  For real integrals, the format is `value i1 i2 i3 i4`.
"""
function parse_integ_value(::Type{T}, linestr::AbstractString) where T<:Real
  line = split(linestr)
  length(line) == 5 || return nothing
  integ = T(parse(Float64, line[1]))
  i1 = parse(Int, line[2])
  i2 = parse(Int, line[3])
  i3 = parse(Int, line[4])
  i4 = parse(Int, line[5])
  return integ, i1, i2, i3, i4
end

function parse_integ_value(::Type{T}, linestr::AbstractString) where T<:Complex
  # format: (real,imaginary) i1 i2 i3 i4
  m = match(r"^\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", linestr)
  if isnothing(m)
    # fall back to real format (e.g., core energy or separator lines)
    line = split(linestr)
    length(line) == 5 || return nothing
    re = parse(real(T), line[1])
    integ = Complex{real(T)}(re, zero(real(T)))
    i1 = parse(Int, line[2])
    i2 = parse(Int, line[3])
    i3 = parse(Int, line[4])
    i4 = parse(Int, line[5])
    return integ, i1, i2, i3, i4
  end
  re = parse(real(T), m.captures[1])
  im = parse(real(T), m.captures[2])
  integ = Complex{real(T)}(re, im)
  i1 = parse(Int, m.captures[3])
  i2 = parse(Int, m.captures[4])
  i3 = parse(Int, m.captures[5])
  i4 = parse(Int, m.captures[6])
  return integ, i1, i2, i3, i4
end

function read_integrals!(int1::Matrix{T}, int2, norb, fdfile, simtra) where T
  int0 = 0.0
  readint0 = false
  for linestr in eachline(fdfile)
    parsed = parse_integ_value(T, linestr)
    isnothing(parsed) && continue
    integ, i1, i2, i3, i4 = parsed
    if i1 > norb || i2 > norb || i3 > norb || i4 > norb
      error("Index larger than norb: "*linestr)
    end
    if i4 > 0
      set_int2!(int2, i1, i2, i3, i4, integ, simtra, false)
    elseif i2 > 0
      set_int1!(int1, i1, i2, integ, simtra)
    elseif i1 <= 0
      int0 = real(integ)
      readint0 = true
    end
  end
  if !readint0
    error("No core energy found in fcidump. Incomplete file?")
  end
  return int0
end

function read_integrals!(int1a::Matrix{T}, int1b, int2aa, int2bb, int2ab, norb, fdfile, simtra) where T
  int0 = 0.0
  readint0 = false
  spincase = 0 # aa, bb, ab, a, b
  for linestr in eachline(fdfile)
    parsed = parse_integ_value(T, linestr)
    isnothing(parsed) && continue
    integ, i1, i2, i3, i4 = parsed
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
        int0 = real(integ)
        readint0 = true
      end
    end
  end
  if !readint0
    error("No core energy found in fcidump. Incomplete file?")
  end
  return int0
end

"""
    read_integrals!(int1::Matrix{T}, int2::Array{T,N},
                    int1p::Matrix{T}, int2ep::Array{T,4},
                    norb::Int, fdfile, simtra) where {T,N}

  Read integrals from fcidump file with positron. 
  We use a section counter to track which block we are reading:
  section==0: electron-electron 2-body (int2)
  section==1: electron-positron 2-body (int2ep)
  section==2: electron 1-body (int1)
  section==3: positron 1-body (int1p)
  When section==4 and a separator is encountered, the next line is core energy.
"""
function read_integrals!(int1::Matrix{T}, int2::Array{T,N},
                        int1p::Matrix{T}, int2ep::Array{T,4},
                        norb::Int, fdfile, simtra) where {T,N}
  int0 = 0.0
  readint0 = false
  section = 0

  for linestr in eachline(fdfile)
    parsed = parse_integ_value(T, linestr)
    isnothing(parsed) && continue
    integ, i1, i2, i3, i4 = parsed

    if i1 > norb || i2 > norb || i3 > norb || i4 > norb
      error("Index larger than norb: "*linestr)
    end

    if i4 > 0
      if section == 0
        set_int2!(int2, i1, i2, i3, i4, integ, simtra, false)
      elseif section == 1
        set_int2!(int2ep, i1, i2, i3, i4, integ, simtra, false)
      else
        error("Unexpected 2-electron integral line in section $(section)")
      end
    elseif i2 > 0
      if section == 2
        set_int1!(int1, i1, i2, integ, simtra)
      elseif section == 3
        set_int1!(int1p, i1, i2, integ, simtra)
      else
        error("Unexpected 1-electron integral line in section $(section)")
      end
    elseif i1 <= 0
      if section < 4
        section += 1
      else
      int0 = real(integ)
      readint0 = true
      end
    end
  end

  if !readint0
    error("No core energy found in fcidump. Incomplete file?")
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
    mmap_integrals(fd::FDump, dir::AbstractString, key::AbstractString, ::Array{T,N})

  Memory-map integral file (from head[key])
"""
function mmap_integrals(fd::FDump, dir::AbstractString, key::AbstractString, ::Array{T,N}) where {T,N}
  file = headvar(fd, key, String)
  if isnothing(file)
    error(key*" option not found in fcidump")
  end
  if !isabspath(file)
    file = joinpath(dir,file)
  end
  if !isfile(file)
    println("NPY-file $file not found. Continue with fcidump file.")
    return Array{T,N}(undef, ntuple(i->0, Val(N)))
  end
  # return npzread(file)
  return mnpymmap(file, Array{T,N})
end

"""
    write_fcidump(fd::FDump, fcidump::String; tol=-1.0, format=:ascii)

  Write fcidump file.

  If `tol` >= 0.0, integrals with absolute value smaller than `tol` are omitted.
  If `format` is `:npy`, integrals are written to npy files in the same directory,
  otherwise if `format` is `:ascii`, integrals are written to ascii fcidump file.
"""
function write_fcidump(fd::FDump, fcidump::String; tol=-1.0, format=:ascii)
  println("Write fcidump $fcidump"...)
  fdf = open(fcidump, "w")
  write_header(fd, fdf; npy=(format == :npy))
  if format == :ascii
    write_integrals(fd, fdf, tol)
  elseif format == :npy
    # copy integrals to npy files
    copy2npy(fd, dirname(fcidump))
  else
    error("Unknown format: "*string(format))
  end
  close(fdf)
end

"""
    write_header(fd::FDump, fdf; npy=false)

  Write header of fcidump file.

  If `npy` is true, write NPY file names for integrals.
"""
function write_header(fd::FDump{T}, fdf; npy=false) where T
  println(fdf, "&FCI")
  head = fd.head
  # set ICMPLX flag for complex integrals
  head["ICMPLX"] = T <: Complex ? [1] : [0]
  if npy
    if !fd.uhf
      head["NPY2"] = ["int2.npy"]
      head["NPY1"] = ["int1.npy"]
    else
      head["NPY2AA"] = ["int2aa.npy"]
      head["NPY2BB"] = ["int2bb.npy"]
      head["NPY2AB"] = ["int2ab.npy"]
      head["NPY1A"] = ["int1a.npy"]
      head["NPY1B"] = ["int1b.npy"]
    end
    head["ENUC"] = [fd.int0]
  else
    delete!(head.shead, "NPY2")
    delete!(head.shead, "NPY1")
    delete!(head.shead, "NPY2AA")
    delete!(head.shead, "NPY2BB")
    delete!(head.shead, "NPY2AB")
    delete!(head.shead, "NPY1A")
    delete!(head.shead, "NPY1B")
    delete!(head.fhead, "ENUC")
  end
  for key in FDUMP_KEYS
    val = headvar(fd, key)
    if !isnothing(val)
      println(fdf, " ", key, "=", join(val, ","), ",")
    end
  end
  for (key,val) in head
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

  For complex values, the format is `(real,imaginary) i1 i2 i3 i4`.
"""
function print_int_value(fdf, integ::Real, i1, i2, i3, i4)
  @printf(fdf, "%23.15e %3i %3i %3i %3i\n", integ, i1, i2, i3, i4)
end

function print_int_value(fdf, integ::Complex, i1, i2, i3, i4)
  @printf(fdf, "(%23.15e,%23.15e) %3i %3i %3i %3i\n", real(integ), imag(integ), i1, i2, i3, i4)
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
  simtra::Bool = uses_reduced_permsym(fd)
  if !fd.uhf
    write_integrals2(fd.int2, fdf, tol, simtra)
    if fd.epdump
      print_int_value(fdf,0.0,0,0,0,0)
      write_integrals2(fd.int2ep, fdf, tol, simtra)
      print_int_value(fdf,0.0,0,0,0,0)
    end
    write_integrals1(fd.int1, fdf, tol, simtra)
    if fd.epdump
      print_int_value(fdf,0.0,0,0,0,0)
      write_integrals1(fd.int1p, fdf, tol, simtra)
      print_int_value(fdf,0.0,0,0,0,0)
    end
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
    write_integrals2(int2::Array{<:Number,3}, fdf, tol, simtra)

  Write 2-e integrals to fdf file.
"""
function write_integrals2(int2::Array{<:Number,3}, fdf, tol, simtra)
  write_integrals2_ = simtra ? write_integrals2_simtra : write_integrals2_normal
  inds(p,q,r,s) = CartesianIndex(p,q,uppertriangular_index(r,s))
  indslow(p,q,r,s) = CartesianIndex(q,p,uppertriangular_index(s,r))
  write_integrals2_(int2, inds, indslow, fdf, tol)
end

function write_integrals2(int2::Array{<:Number,4}, fdf, tol, simtra)
  write_integrals2_ = simtra ? write_integrals2_simtra : write_integrals2_normal
  inds(p,q,r,s) = CartesianIndex(p,q,r,s)
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
    copy2npy(fd::FDump, dir::AbstractString)

  Copy integrals to npy files in `dir`.
"""
function copy2npy(fd::FDump, dir::AbstractString)
  println("Copy integrals to npy files in $dir")
  if !isdir(dir)
    mkpath(dir)
  end
  if !fd.uhf
    npzwrite(joinpath(dir,"int2.npy"), fd.int2)
    npzwrite(joinpath(dir,"int1.npy"), fd.int1)
  else
    npzwrite(joinpath(dir,"int2aa.npy"), fd.int2aa)
    npzwrite(joinpath(dir,"int2bb.npy"), fd.int2bb)
    npzwrite(joinpath(dir,"int2ab.npy"), fd.int2ab)
    npzwrite(joinpath(dir,"int1a.npy"), fd.int1a)
    npzwrite(joinpath(dir,"int1b.npy"), fd.int1b)
  end
end

"""
    reorder_orbs_int2(int2::AbstractArray, orbs; alloc=dims->zeros(eltype(int2), dims))

  Reorder orbitals in 2-e integrals according to `orbs`.

  `orbs`can be a subset of orbitals or a permutation of orbitals.
  Return `int2[orbs[p],orbs[q],orbs[r],orbs[s]]` or the triangular version.

  The reordered tensor is obtained from `alloc(dims)` (in-memory `zeros` by default) and filled
  one slice at a time; pass a memory-mapped allocator to keep the result on disk (and read the
  source `int2` directly, e.g. when reducing a large memory-mapped MO dump to the active space).
"""
function reorder_orbs_int2(int2::AbstractArray, orbs; alloc=dims->zeros(eltype(int2), dims))
  norb = size(int2,1)
  norbnew = length(orbs)
  if orbs == 1:norb
    return int2
  end
  if norbnew == 0
    return ndims(int2) == 3 ? alloc((0,0,0)) : alloc((0,0,0,0))
  end
  @assert maximum(orbs) <= norb && minimum(orbs) > 0 "Orbital index out of range"
  if ndims(int2) == 3
    # triangular
    int2t = alloc((norbnew, norbnew, norbnew*(norbnew+1)÷2))
    for s = 1:norbnew
      for r = 1:s
        ro = orbs[r]
        so = orbs[s]
        if ro <= so
          @views int2t[:,:,uppertriangular_index(r,s)] = int2[orbs,orbs,uppertriangular_index(ro, so)]
        else
          @views permutedims!(int2t[:,:,uppertriangular_index(r,s)], int2[orbs,orbs,uppertriangular_index(so, ro)], (2,1))
        end
      end
    end
  else
    int2t = alloc((norbnew, norbnew, norbnew, norbnew))
    int2t .= @view int2[orbs,orbs,orbs,orbs]
  end
  return int2t
end

"""
    int1_npy_filename(fd::FDump, spincase::Symbol=:α)

  Return filename for 1-e integrals in npy format.
  `spincase` can be `:α`, `:β`, or `:p` for UHF fcidump.
"""
function int1_npy_filename(fd::FDump, spincase::Symbol=:α)
  if spincase == :p
    file = headvar(fd, "NPY1P", String)
    if isnothing(file)
      file = "int1p.npy"
    end
    return file::String
  end
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
  `spincase` can be `:α`, `:β`, `:αβ`, or `:ep` for UHF fcidump.
"""
function int2_npy_filename(fd::FDump, spincase::Symbol=:α)
  if spincase == :ep
    file = headvar(fd, "NPY2EP", String)
    if isnothing(file)
      file = "int2ep.npy"
    end
    return file::String
  end
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
