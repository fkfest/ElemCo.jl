#!/usr/bin/env julia

"""
Read and write fcidump format integrals.
Individual arrays of integrals can also be in *.npy format
"""
module FciDump

#using LinearAlgebra
using NPZ
using Mmap
using Parameters

export FDump, read_fcidump, headvar

# optional variables which won't be written if =0
const FDUMP_OPTIONAL=["IUHF", "ST", "III"]

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
end

"""spin-free fcidump"""
FDump(int2::Array{Float64},int1::Array{Float64},int0::Float64,head::Dict) = FDump(int2,[],[],[],int1,[],[],int0,head)
"""spin-polarized fcidump"""
FDump(int2aa::Array{Float64},int2bb::Array{Float64},int2ab::Array{Float64},int1a::Array{Float64},int1b::Array{Float64},int0::Float64,head::Dict) = FDump([],int2aa,int2bb,int2ab,[],int1a,int1b,int0,head)

"""
read ascii file (possibly with integrals in npy files)
"""
function read_fcidump(fcidump::String)
  fdf = open(fcidump)
  fd = FDump()
  fd.head = read_header(fdf)
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
    read_integrals!(fd)
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
function read_integrals!(fd::FDump)
  println("Read npy files")
  if headvar(fd, "IUHF") <= 0
    fd.int2 = mmap_integrals(fd, "NPY2")
    fd.int1 = mmap_integrals(fd, "NPY1")
  else
    fd.int2aa = mmap_integrals(fd, "NPYAA")
    fd.int2bb = mmap_integrals(fd, "NPYBB")
    fd.int2ab = mmap_integrals(fd, "NPYAB")
    fd.int1a = mmap_integrals(fd, "NPYA")
    fd.int1b = mmap_integrals(fd, "NPYB")
  end
  if isnothing(headvar(fd, "ENUC"))
    error("ENUC option not found in fcidump")
  end
  fd.int0 = headvar(fd, "ENUC")
end

"""for not ab: particle symmetry is assumed """
function set_int2!(int2::AbstractArray,i1,i2,i3,i4,integ,simtra,ab)
  int2[i1,i2,i3,i4] = integ
  if !ab
      int2[i3,i4,i1,i2] = integ
  end
  if !simtra
    int2[i1,i2,i4,i3] = integ
    int2[i2,i1,i3,i4] = integ
    int2[i2,i1,i4,i3] = integ
    if !ab
      int2[i3,i4,i2,i1] = integ
      int2[i4,i3,i1,i2] = integ
      int2[i4,i3,i2,i1] = integ
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
function read_integrals!(fd::FDump, fdfile)
  norb = headvar(fd, "NORB")
  uhf = (headvar(fd, "IUHF") > 0)
  simtra = (headvar(fd, "ST") > 0)
  if uhf
    print("UHF")
    fd.int1a = zeros(norb,norb)
    fd.int1b = zeros(norb,norb)
    fd.int2aa = zeros(norb,norb,norb,norb)
    fd.int2bb = zeros(norb,norb,norb,norb)
    fd.int2ab = zeros(norb,norb,norb,norb)
  else
    fd.int1 = zeros(norb,norb)
    fd.int2 = zeros(norb,norb,norb,norb)
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
        if uhf
          set_int2!(fd.int2aa,i1,i2,i3,i4,integ,simtra,false)
        else
          set_int2!(fd.int2,i1,i2,i3,i4,integ,simtra,false)
        end
      elseif spincase == 1
        set_int2!(fd.int2bb,i1,i2,i3,i4,integ,simtra,false)
      elseif spincase == 2
        set_int2!(fd.int2ab,i1,i2,i3,i4,integ,simtra,true)
      else
          error("Unexpected 2-el integrals for spin-case "*string(spincase))
      end
    elseif i2 > 0
      if !uhf 
        set_int1!(fd.int1,i1,i2,integ,simtra)
      elseif spincase == 3
        set_int1!(fd.int1a,i1,i2,integ,simtra)
      elseif spincase == 4
        set_int1!(fd.int1b,i1,i2,integ,simtra)
      else
        error("Unexpected 1-el integrals for spin-case "*string(spincase))
      end
    elseif i1 <= 0
      if uhf && spincase < 5
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
function mmap_integrals(fd::FDump, key::String)
  file = headvar(fd, key)
  if isnothing(file)
    error(key*" option not found in fcidump")
  end
  return npzread(file)
end
end