#!/usr/bin/env julia

"""
Read and write fcidump format integrals.
Individual arrays of integrals can also be in *.npy format
"""
module FciDump

# using LinearAlgebra
# using NPZ
using TensorOperations
using DocStringExtensions
using Printf
using ..ElemCo.MNPY

export FDump, fd_exists, read_fcidump, write_fcidump, transform_fcidump
export headvar, integ1, integ2, uppertriangular, uppertriangular_range

# optional variables which won't be written if =0
const FDUMP_OPTIONAL=["IUHF", "ST", "III"]

"""prefered order of keys in fcidump header (optional keys are not included)"""
const FDUMP_KEYS=["NORB", "NELEC", "MS2", "ISYM", "ORBSYM" ]

"""
    FDump

  Molecular integrals 

  The 2-e integrals are stored in the physicists' notation: int2[pqrs] = <pq|rs>
  and for `triang` the last two indices are stored as a single upper triangular index (r <= s)

  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct FDump
  """ 2-e⁻ integrals for restricted orbitals fcidump. """
  int2::Array{Float64} = []
  """ αα 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2aa::Array{Float64} = []
  """ ββ 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2bb::Array{Float64} = []
  """ αβ 2-e⁻ integrals for unrestricted orbitals fcidump. """
  int2ab::Array{Float64} = []
  """ 1-e⁻ integrals for restricted orbitals fcidump. """
  int1::Array{Float64} = []
  """ α 1-e⁻ integrals for unrestricted orbitals fcidump. """
  int1a::Array{Float64} = []
  """ β 1-e⁻ integrals for unrestricted orbitals fcidump. """
  int1b::Array{Float64} = []
  """ core energy """
  int0::Float64 = 0.0
  """ header of fcidump file, a dictionary of arrays. """
  head::Dict = Dict()
  """`⟨true⟩` use an upper triangular index for last two indices of 2e⁻ integrals.""" 
  triang::Bool = true
  """`⟨false⟩` a convinience variable, has to coincide with `head["IUHF"][1] > 0`. """
  uhf::Bool = false
end

"""
    FDump(int2::Array{Float64},int1::Array{Float64},int0::Float64,head::Dict)

  Spin-free fcidump
"""
FDump(int2::Array{Float64},int1::Array{Float64},int0::Float64,head::Dict) = FDump(int2,[],[],[],int1,[],[],int0,head)
"""
    FDump(int2aa::Array{Float64},int2bb::Array{Float64},int2ab::Array{Float64},int1::Array{Float64},int0::Float64,head::Dict)

  Spin-polarized fcidump
"""
FDump(int2aa::Array{Float64},int2bb::Array{Float64},int2ab::Array{Float64},int1a::Array{Float64},int1b::Array{Float64},int0::Float64,head::Dict) = FDump([],int2aa,int2bb,int2ab,[],int1a,int1b,int0,head)

"""
    FDump(norb,nelec;ms2=0,isym=1,orbsym=[],uhf=false,simtra=false,triang=true)

  Create a new FDump object
"""
function FDump(norb, nelec; ms2=0, isym=1, orbsym=[], 
               uhf=false, simtra=false, triang=true)
  fd = FDump()
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
  fd.triang = triang
  fd.uhf = uhf
  return fd
end

"""
    fd_exists(fd::FDump)

  Return true if the object is a non-empty FDump
"""
function fd_exists(fd::FDump)
  return !isempty(fd.head)
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
    read_fcidump(fcidump::String)

  Read ascii file (possibly with integrals in npy files).
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
  if isnothing(headvar(fd, "NPY2")) && isnothing(headvar(fd, "NPY2AA"))
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

"""
    read_header(fdfile::IOStream)

  Read header of fcidump file.
"""
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
          # case-insensitive variable names in the header
          variable_name = uppercase(prev_el)
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
  if isnothing(headvar(fd, "ENUC"))
    error("ENUC option not found in fcidump")
  end
  fd.int0 = headvar(fd, "ENUC")
end

""" 
    uppertriangular(i1, i2)

  Return upper triangular index from two indices i1 <= i2.
"""
function uppertriangular(i1, i2)
  return i1+i2*(i2-1)÷2
end

""" 
    uppertriangular(i1, i2, i3)

  Return upper triangular index from three indices i1 <= i2 <= i3.
"""
function uppertriangular(i1, i2, i3)
  return i1+i2*(i2-1)÷2+(i3+1)*i3*(i3-1)÷6
end

""" 
    uppertriangular_range(i2)

  Return range for the upper triangular index (i1 <= i2) for a given i2. 
"""
function uppertriangular_range(i2)
  return (i2*(i2-1)÷2+1):(i2*(i2+1)÷2)
end

""" 
    uppertriangular_diagonal(i2)

  Return index of diagonal of upper triangular index (i1 <= i2) for a given i2. 
"""
function uppertriangular_diagonal(i2)
  return (i2*(i2+1)÷2)
end

""" 
    strict_uppertriangular_range(i2)

  Return range for the upper triangular index (i1 <= i2) without diagonal (i1 < i2) for a given i2. 
"""
function strict_uppertriangular_range(i2)
  return (i2*(i2-1)÷2+1):(i2*(i2+1)÷2-1)
end

"""
    set_int2!(int2::AbstractArray, i1, i2, i3, i4, integ, triang, simtra, ab)

  Set 2-e integral in `int2` array to `integ` considering permutational symmetries.

  For not `ab`: particle symmetry is assumed.
  Integrals are stored in physicists' notation.
  If `triang`: the last two indices are stored as a single upper triangular index.
"""
function set_int2!(int2::AbstractArray, i1, i2, i3, i4, integ,
                   triang, simtra, ab)
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

"""
    read_integrals!(fd::FDump, fdfile::IOStream)

  Read integrals from fcidump file
"""
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

"""
    headvar(head::Dict, key::String)

  Check header for `key`, return value if a list, 
  or the element or nothing if not there.
"""
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

"""
    headvar(fd::FDump, key::String)

  Check header for `key`, return value if a list, 
  or the element or nothing if not there.
"""
function headvar(fd::FDump, key::String )
  return headvar(fd.head, key)
end

"""
    mmap_integrals(fd::FDump, dir::AbstractString, key::AbstractString)

  Memory-map integral file (from head[key])
"""
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

"""
    write_fcidump(fd::FDump, fcidump::String, tol=1e-12)

  Write fcidump file.
"""
function write_fcidump(fd::FDump, fcidump::String, tol=1e-12)
  fdf = open(fcidump,"w")
  write_header(fd,fdf)
  write_integrals(fd,fdf,tol)
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
  simtra = (headvar(fd, "ST") > 0)
  if !fd.uhf
    write_integrals2(fd.int2, fdf, tol, fd.triang, simtra)
    write_integrals1(fd.int1, fdf, tol, simtra)
  else
    write_integrals2(fd.int2aa, fdf, tol, fd.triang, simtra)
    print_int_value(fdf,0.0,0,0,0,0)
    write_integrals2(fd.int2bb, fdf, tol, fd.triang, simtra)
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
    write_integrals2(int2, fdf, tol, triang, simtra)

  Write 2-e integrals to fdf file.
"""
function write_integrals2(int2, fdf, tol, triang, simtra)
  norb = size(int2,1)
  if triang
    inds = (p,q,r,s) -> CartesianIndex(p,q,uppertriangular(r,s))
    indslow = (p,q,r,s) -> CartesianIndex(q,p,uppertriangular(s,r))
  else
    inds = (p,q,r,s) -> CartesianIndex(p,q,r,s)
    indslow = (p,q,r,s) -> CartesianIndex(p,q,r,s)
  end
  if simtra
    for p = 1:norb
      for q = 1:norb
        for r = 1:p-1
          # lower triangle (q>s)
          for s = 1:q-1
            val = int2[indslow(p,r,q,s)]
            if abs(val) > tol
              print_int_value(fdf,val,p,q,r,s)
            end
          end
          # upper triangle (q<=s)
          for s = q:norb
            val = int2[inds(p,r,q,s)]
            if abs(val) > tol
              print_int_value(fdf,val,p,q,r,s)
            end
          end
        end
        # r==p case
        r = p
        for s = 1:q
          val = int2[indslow(p,r,q,s)]
          if abs(val) > tol
            print_int_value(fdf,val,p,q,r,s)
          end
        end
      end
    end
  else
    # normal case
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
  if length(Tl) == 2 && typeof(Tl[1]) <: AbstractArray
    genuhfdump = true
  else
    genuhfdump = false
    @assert !fd.uhf # from uhf fcidump can generate only uhf fcidump
  end
  if fd.uhf
    fd.int2aa = transform_int2(fd.int2aa, Tl[1], Tl[1], Tr[1], Tr[1], fd.triang, fd.triang)
    fd.int2bb = transform_int2(fd.int2bb, Tl[2], Tl[2], Tr[2], Tr[2], fd.triang, fd.triang)
    fd.int2ab = transform_int2(fd.int2ab, Tl[1], Tl[2], Tr[1], Tr[2], false, false)
    fd.int1a = transform_int1(fd.int1a, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1b, Tl[2], Tr[2])
  elseif genuhfdump
    # change fcidump from rhf to uhf format
    fd.int2aa = transform_int2(fd.int2, Tl[1], Tl[1], Tr[1], Tr[1], fd.triang, fd.triang)
    fd.int2bb = transform_int2(fd.int2, Tl[2], Tl[2], Tr[2], Tr[2], fd.triang, fd.triang)
    fd.int2ab = transform_int2(fd.int2, Tl[1], Tl[2], Tr[1], Tr[2], fd.triang, false)
    fd.int1a = transform_int1(fd.int1, Tl[1], Tr[1])
    fd.int1b = transform_int1(fd.int1, Tl[2], Tr[2])
    fd.int2 = []
    fd.int1 = []
    fd.head["IUHF"] = [1]
    fd.uhf = true
  else
    fd.int2 = transform_int2(fd.int2, Tl, Tl, Tr, Tr, fd.triang, fd.triang)
    fd.int1 = transform_int1(fd.int1, Tl, Tr)
  end
end

"""
    transform_int2(int2::AbstractArray, Tl::AbstractArray, Tl2::AbstractArray, 
                   Tr::AbstractArray, Tr2::AbstractArray, triang_in, triang_out)

  Transform 2-e integrals to new basis using `Tl`/`Tl2` and `Tr`/`Tr2` transformation matrices.
  <pq|rs> = <p'q'|r's'> * Tl[p',p] * Tl2[q',q] * Tr[r',r] * Tr2[s',s]
  If `triang`: the last two indices are stored as a single upper triangular index.
"""
function transform_int2(int2::AbstractArray, Tl::AbstractArray, Tl2::AbstractArray, 
                        Tr::AbstractArray, Tr2::AbstractArray, triang_in, triang_out)
  norb = size(int2,1)
  if triang_in && triang_out
    int2t = zeros(norb,norb,norb*(norb+1)÷2)
    int_3i = zeros(norb,norb,norb)
    for s = 1:norb
      rs = strict_uppertriangular_range(s)
      rrange = 1:s-1
      if length(rs) > 0
        @tensoropt int_3i[p,q,r] = int2[:,:,rs][p',q',r'] * Tl[p',p] * Tl2[q',q] * Tr[rrange,:][r',r]
      end
      # contribution from the diagonal <p'q'|s's'> 
      ss = uppertriangular_diagonal(s)
      @tensoropt int_3i[p,q,r] += 0.5*int2[:,:,ss][p',q'] * Tl[p',p] * Tl2[q',q] * Tr[s,:][r]
      for s1 = 1:norb
        rs1 = uppertriangular_range(s1)
        rrange = 1:s1
        Tr2ss1 = Tr2[s,s1]
        @tensoropt int2t[:,:,rs1][p,q,r] += int_3i[:,:,rrange][p,q,r] * Tr2ss1
        @tensoropt int2t[:,:,rs1][p,q,r] += int_3i[:,:,s1][q,p] * Tr2[s,rrange][r]
      end
    end
  elseif triang_in && ! triang_out
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
      ss = uppertriangular_diagonal(s)
      @tensoropt int_3i[p,q,r] += 0.5*int2[:,:,ss][p',q'] * Tl[p',p] * Tl2[q',q] * Tr[s,:][r]
      @tensoropt int_3i2[p,q,r] += 0.5*int2[:,:,ss][p',q'] * Tl2[p',p] * Tl[q',q] * Tr2[s,:][r]

      @tensoropt int2t[p,q,r,s'] += int_3i[p,q,r] * Tr2[s,:][s']
      @tensoropt int2t[p,q,r,s'] += int_3i2[q,p,s'] * Tr[s,:][r]
    end
  elseif !triang_in && triang_out
    error("Can't transform from non-triangular to triangular")
  else
    @tensoropt int2t[p,q,r,s] := int2[p',q',r',s']*Tl[p',p]*Tl2[q',q]*Tr[r',r]*Tr2[s',s]
  end
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

end #module
