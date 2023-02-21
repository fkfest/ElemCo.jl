#!/usr/bin/env julia

"""
Read and write fcidump format integrals.
Individual arrays of integrals can also be in *.npy format
"""
module FciDump

#using LinearAlgebra
using NPZ
using Mmap

export FDump, read_fcidump, headvar

# optional variables which won't be written if =0
const FDUMP_OPTIONAL=["IUHF", "ST", "III"]

struct FDump
  int2::Array{Float64}
  int2aa::Array{Float64}
  int2bb::Array{Float64}
  int2ab::Array{Float64}
  int1::Array{Float64}
  int1a::Array{Float64}
  int1b::Array{Float64}
  int0::Float64
  head::Dict
end

FDump() = FDump([],[],[],[],[],[],[],0.0,Dict())
"""spin-free fcidump"""
FDump(int2::Array{Float64},int1::Array{Float64},int0::Float64,head::Dict) = FDump(int2,[],[],[],int1,[],[],int0,head)
"""spin-polarized fcidump"""
FDump(int2aa::Array{Float64},int2bb::Array{Float64},int2ab::Array{Float64},int1a::Array{Float64},int1b::Array{Float64},int0::Float64,head::Dict) = FDump([],int2aa,int2bb,int2ab,[],int1a,int1b,int0,head)

"""
read ascii file (possibly with integrals in npy files)
"""
function read_fcidump(fcidump::String)
  fdf = open(fcidump)
  head = read_header(fdf)
  simtra = (headvar(head, "ST") > 0)
  if simtra
    println("Non-Hermitian")
  end
  int2, int1, int2aa, int2bb, int2ab, int1a, int1b = [[] for i in 1:7]
  int0 = 0.0
  if isnothing(headvar(head, "NPY2")) && isnothing(headvar(head, "NPYAA"))
    # read integrals from fcidump file
    int2,int2aa,int2bb,int2ab,int1,int1a,int1b,int0 = read_integrals(fdf,head)
    close(fdf)
  else
    close(fdf)
    # read integrals from npy files
    int2,int2aa,int2bb,int2ab,int1,int1a,int1b,int0 = read_integrals(head)
  end
  FDump(int2,int2aa,int2bb,int2ab,int1,int1a,int1b,int0,head)
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
function read_integrals(head::Dict)
  println("Read npy files")
  int2, int1, int2aa, int2bb, int2ab, int1a, int1b = [[] for i in 1:7]
  if headvar(head, "IUHF") <= 0
    int2 = mmap_integrals(head, "NPY2")
    int1 = mmap_integrals(head, "NPY1")
  else
    int2aa = mmap_integrals(head, "NPYAA")
    int2bb = mmap_integrals(head, "NPYBB")
    int2ab = mmap_integrals(head, "NPYAB")
    int1a = mmap_integrals(head, "NPYA")
    int1b = mmap_integrals(head, "NPYB")
  end
  if isnothing(headvar(head, "ENUC"))
    error("ENUC option not found in fcidump")
  end
  int0 = float(headvar(head, "ENUC"))

  return int2,int2aa,int2bb,int2ab,int1,int1a,int1b,int0
end

"""read integrals from fcidump file"""
function read_integrals(fdfile, head::Dict)
  norb = headvar(head, "NORB")
  uhf = (headvar(head, "IUHF") > 0)
  int2, int1, int2aa, int2bb, int2ab, int1a, int1b = [[] for i in 1:7]
  if uhf
    print("UHF")
    int1a = zeros(norb,norb)
    int1b = zeros(norb,norb)
    int2aa = zeros(norb,norb,norb,norb)
    int2bb = np.zeros([norb,norb,norb,norb])
    int2ab = np.zeros([norb,norb,norb,norb])
  else
    int1 = zeros(norb,norb)
    int2 = zeros(norb,norb,norb,norb)
  end

  #TODO

  return int2,int2aa,int2bb,int2ab,int1,int1a,int1b,int0
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
function mmap_integrals(head::Dict, key::String)
  file = headvar(head, key)
  if isnothing(file)
    error(key*" option not found in fcidump")
  end
  return npzread(file)
end
end