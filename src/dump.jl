#!/usr/bin/env julia

"""
Read and write fcidump format integrals.
Individual arrays of integrals can also be in *.npy format
"""
module FciDump

# using LinearAlgebra
# using NPZ
using Parameters
using ..MNPY

export FDump, read_fcidump, headvar, SpinCase, SCα, SCβ, SCαβ, integ1, integ2

# optional variables which won't be written if =0
const FDUMP_OPTIONAL=["IUHF", "ST", "III"]

"""
molecular integrals 

the 2-e integrals are stored in the physcal notation: int2[pqrs] = <pq|rs>
"""
@with_kw mutable struct FDump
  int2::Array{Float64} = []
  int2aa::Array{Float64} = []
  int2bb::Array{Float64} = []
  int2ab::Array{Float64} = []
  int1::Array{Float64} = []
  int1a::Array{Float64} = []
  int1b::Array{Float64} = []
  int0::Float64 = 0.0
  head::Dict = Dict()
  # use an upper triangular index for last two indices of 2e⁻ integrals 
  triang::Bool = true
  # a convinience variable, has to coincide with `head["IUHF"][1] > 0``
  uhf::Bool = false
end

"""spin-free fcidump"""
FDump(int2::Array{Float64},int1::Array{Float64},int0::Float64,head::Dict) = FDump(int2,[],[],[],int1,[],[],int0,head)
"""spin-polarized fcidump"""
FDump(int2aa::Array{Float64},int2bb::Array{Float64},int2ab::Array{Float64},int1a::Array{Float64},int1b::Array{Float64},int0::Float64,head::Dict) = FDump([],int2aa,int2bb,int2ab,[],int1a,int1b,int0,head)

@enum SpinCase SCα SCβ SCαβ

"""return 1-e⁻ integrals """
function integ1(fd::FDump,spincase::SpinCase = SCα)
  if !fd.uhf
    return fd.int1
  elseif spincase == SCα
    return fd.int1a
  else
    return fd.int1b
  end
end

"""return 2-e⁻ integrals """
function integ2(fd::FDump,spincase::SpinCase = SCα)
  if !fd.uhf
    return fd.int2
  elseif spincase == SCα
    return fd.int2aa
  elseif spincase == SCβ
    return fd.int2bb
  else
    return fd.int2ab
  end
end

"""
read ascii file (possibly with integrals in npy files)
"""
function read_fcidump(fcidump::String)
  fdf = open(fcidump)
  fd = FDump()
  fd.head = read_header(fdf)
  fd.uhf = (headvar(fd, "IUHF") > 0)
  simtra = (headvar(fd, "ST") > 0)
  if simtra
    println("Non-Hermitian")
  end
  if isnothing(headvar(fd, "NPY2")) && isnothing(headvar(fd, "NPYAA"))
    # read integrals from fcidump file
    read_integrals!(fd,fdf)
    close(fdf)
  else
    close(fdf)
    # read integrals from npy files
    read_integrals!(fd,dirname(fcidump))
  end
  return fd
end

"""read header of fcidump file"""
function read_header(fdfile)
  # put some defaults...
  head = Dict()
  head["IUHF"] = [0]
  head["ST"] = [0]
  variable_name = ""
  for line in eachline(fdfile)
    #skip empty lines
    line = strip(line)
    if length(line) == 0
      continue
    end
    if line == "/"
      # end of header
      break
    end
    line = replace(line,"=" => " = ")
    line_array = [var for var in split(line, [' ',',']) if !isempty(var)]
    # search for '=' and put element before it as the variable name, and everything
    # after (before the next variable name) as a vector of values
    prev_el = ""
    elements = []
    if variable_name != ""
      # in case the elements of the last variable continue on new line...
      elements = head[variable_name]
    end
    push!(line_array, "\n")
    for el in line_array
      if el == "="
        if prev_el != ""
          if variable_name != ""
            # store the previous array
            head[variable_name] = elements
          end
          variable_name = prev_el
          elements = []
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
              elem = prev_el
            end 
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


"""read integrals from npy files"""
function read_integrals!(fd::FDump, dir::AbstractString)
  println("Read npy files")
  if !fd.uhf
    fd.int2 = mmap_integrals(fd, dir, "NPY2")
    fd.int1 = mmap_integrals(fd, dir, "NPY1")
  else
    fd.int2aa = mmap_integrals(fd, dir, "NPYAA")
    fd.int2bb = mmap_integrals(fd, dir, "NPYBB")
    fd.int2ab = mmap_integrals(fd, dir, "NPYAB")
    fd.int1a = mmap_integrals(fd, dir, "NPYA")
    fd.int1b = mmap_integrals(fd, dir, "NPYB")
  end
  if isnothing(headvar(fd, "ENUC"))
    error("ENUC option not found in fcidump")
  end
  fd.int0 = headvar(fd, "ENUC")
end

# return upper triangular index from two indices i1 <= i2
function uppertriangular(i1,i2)
  return i1+i2*(i2-1)÷2
end

"""for not ab: particle symmetry is assumed.
   Integrals are stored in physcal notation.
   if triang: the last two indices are stored as a single upper triangular index
"""
function set_int2!(int2::AbstractArray,i1,i2,i3,i4,integ,triang,simtra,ab)
  if triang
    @assert !ab
    if i2 == i4
      i24 = uppertriangular(i2,i4)
      int2[i1,i3,i24] = integ
      int2[i3,i1,i24] = integ
    elseif i2 < i4 
      int2[i1,i3,uppertriangular(i2,i4)] = integ
    else
      int2[i3,i1,uppertriangular(i4,i2)] = integ
    end
    if !simtra
      if i2 == i3
        i23 = uppertriangular(i2,i3)
        int2[i1,i4,i23] = integ
        int2[i4,i1,i23] = integ
      elseif i2 < i3
        int2[i1,i4,uppertriangular(i2,i3)] = integ
      else
        int2[i4,i1,uppertriangular(i3,i2)] = integ
      end
      if i1 == i4
        i14 = uppertriangular(i1,i4)
        int2[i2,i3,i14] = integ
        int2[i3,i2,i14] = integ
      elseif i1 < i4
        int2[i2,i3,uppertriangular(i1,i4)] = integ
      else
        int2[i3,i2,uppertriangular(i4,i1)] = integ
      end
      if i1 == i3
        i13 = uppertriangular(i1,i3)
        int2[i2,i4,i13] = integ
        int2[i4,i2,i13] = integ
      elseif i1 < i3
        int2[i2,i4,uppertriangular(i1,i3)] = integ
      else
        int2[i4,i2,uppertriangular(i3,i1)] = integ
      end
    end
  else
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
end

function set_int1!(int1, i1, i2, integ, simtra)
  int1[i1,i2] = integ
  if !simtra
    int1[i2,i1] = integ
  end
end

"""read integrals from fcidump file"""
function read_integrals!(fd::FDump, fdfile::IOStream)
  norb = headvar(fd, "NORB")
  simtra = (headvar(fd, "ST") > 0)
  if fd.uhf
    print("UHF")
    fd.int1a = zeros(norb,norb)
    fd.int1b = zeros(norb,norb)
    if fd.triang
      fd.int2aa = zeros(norb,norb,norb*(norb+1)÷2)
      fd.int2bb = zeros(norb,norb,norb*(norb+1)÷2)
    else
      fd.int2aa = zeros(norb,norb,norb,norb)
      fd.int2bb = zeros(norb,norb,norb,norb)
    end
    fd.int2ab = zeros(norb,norb,norb,norb)
  else
    fd.int1 = zeros(norb,norb)
    if fd.triang
      fd.int2 = zeros(norb,norb,norb*(norb+1)÷2)
    else
      fd.int2 = zeros(norb,norb,norb,norb)
    end
  end
  spincase = 0 # aa, bb, ab, a, b
  for linestr in eachline(fdfile)
    line = split(linestr)
    if length(line) != 5
      # println("Last line: ",linestr)
      break
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
        if fd.uhf
          set_int2!(fd.int2aa,i1,i2,i3,i4,integ,fd.triang,simtra,false)
        else
          set_int2!(fd.int2,i1,i2,i3,i4,integ,fd.triang,simtra,false)
        end
      elseif spincase == 1
        set_int2!(fd.int2bb,i1,i2,i3,i4,integ,fd.triang,simtra,false)
      elseif spincase == 2
        set_int2!(fd.int2ab,i1,i2,i3,i4,integ,false,simtra,true)
      else
          error("Unexpected 2-el integrals for spin-case "*string(spincase))
      end
    elseif i2 > 0
      if !fd.uhf 
        set_int1!(fd.int1,i1,i2,integ,simtra)
      elseif spincase == 3
        set_int1!(fd.int1a,i1,i2,integ,simtra)
      elseif spincase == 4
        set_int1!(fd.int1b,i1,i2,integ,simtra)
      else
        error("Unexpected 1-el integrals for spin-case "*string(spincase))
      end
    elseif i1 <= 0
      if fd.uhf && spincase < 5
        spincase += 1
      else
        fd.int0 = integ
      end
    end
  end
end

"""check header for the key, return value if a list, 
or the element or nothing if not there"""
function headvar(head::Dict, key::String)
  val = get(head, key, nothing)
  if isnothing(val)
    return val
  elseif length(val) == 1
    return val[1]
  else
    return val
  end
end

"""check header for the key, return value if a list, 
or the element or nothing if not there"""
function headvar(fd::FDump, key::String )
  return headvar(fd.head, key)
end

"""mmap integral file (from head[key])"""
function mmap_integrals(fd::FDump, dir::AbstractString, key::AbstractString)
  file = headvar(fd, key)
  if isnothing(file)
    error(key*" option not found in fcidump")
  end
  if !isabspath(file)
    file = joinpath(dir,file)
  end
  # return npzread(file)
  return mnpymmap(file)
end

end #module