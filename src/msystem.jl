"""
info about molecular system (geometry)

macros to specify geometry and basis sets
"""
module MSystem
export MSys, Basis, ACenter, genxyz

const BOHR2ANGSTROM = 0.52917721
const ANGSTROM2BOHR = 1/BOHR2ANGSTROM
"""
basisset
"""
struct Basis
  # e.g., vdz
  name::String
  # coeffs, exponents..
end

""" 
set element specific basis from, e.g., Dict("ao"=>"cc-pVDZ; o=aug-cc-pVDZ","jkfit"=>"cc-pvdz-jkfit")
"""
function genbasis4element(basis::Dict,elem::AbstractString)
  elembasis = Dict{String,Basis}()
  elemUP = uppercase(elem)
  for (type,name) in basis
    names = strip.(split(name,[';',',']))
    elbas = ""
    if length(names) == 0
      error("Basis $type not defined!")
    else
      for bas in names
        eldef=strip.(split(bas,['=',':']))
        if length(eldef) == 1
          if elbas == ""
            elbas = eldef[1]
          end
        elseif length(eldef) != 2
          error("Something wrong in the basis definition $bas in $name")
        else
          if uppercase(eldef[1]) == elemUP
            elbas = eldef[2]
          end
        end
      end
    end
    elembasis[type] = Basis(elbas)
  end
  return elembasis
end

"""
atomic center
"""
mutable struct ACenter
  """name (as defined in input)"""
  name::String
  """coordinates in bohr"""
  coord::AbstractArray{Float64,1}
  """various basis sets (ao,mp2fit,jkfit)"""
  basis::Dict{String,Basis}
end

Base.show(io::IO, val::ACenter) = print(io, val.name, " ", val.coord[1], " ", val.coord[2], " ", val.coord[3]) 

""" transform from angstrom to bohr """
function a2b(vals,skip)
  if skip
    return vals
  else
    return vals*ANGSTROM2BOHR
  end
end

"""
  Create ACenter from a line `<Atom> x y z`. If !bohr: coords are in angstrom
  Returns the center and a bool success variable 
  If the line has a different format: return false
"""
function try2create_acenter(line::AbstractString, basis::Dict, bohr = true)
  coords = split(line)
  if length(coords) == 4
    xcoord = tryparse(Float64,coords[2])
    ycoord = tryparse(Float64,coords[3])
    zcoord = tryparse(Float64,coords[4])
    if !isnothing(xcoord) && !isnothing(ycoord) && !isnothing(zcoord)
      basis = genbasis4element(basis,coords[1])
      return ACenter(coords[1],a2b([xcoord,ycoord,zcoord],bohr),basis), true
    end
  end
  # not a center
  return ACenter("",[],Dict{String,Basis}()), false
end

"""
geometry and basis set for each element name in the geometry
"""
mutable struct MSys
  atoms::AbstractArray{ACenter,1}
  function MSys(xyz::AbstractString, basis::Dict)
    xyz_lines = split(xyz,"\n")
    if length(xyz_lines) == 0
      error("Empty geometry im MSys")
    elseif length(xyz_lines) == 1
      #check whether it's a file
      error("xyz-files not implemented yet!")
    end
    array_of_atoms = ACenter[]
    bohr = true
    for line in xyz_lines
      ac, success = try2create_acenter(line,basis,bohr)
      if success
        push!(array_of_atoms,ac)
      else
        if line == "bohr"
          bohr = true
        elseif line == "angstrom"
          bohr = false
        else
          error("Unsupported xyz line $line")
        end
      end
    end
    new(array_of_atoms)
  end
end

function Base.show(io::IO, val::MSys) 
  for atom in val.atoms
    println(io, atom)
  end
end

""" generate xyz string with element without numbers """
function genxyz(ac::ACenter, bohr=true)
  name = rstrip(ac.name,['0','1','2','3','4','5','6','7','8','9'])
  return string(name," ",ac.coord[1]," ",ac.coord[2]," ",ac.coord[3])
end
""" generate xyz string with elements without numbers """
function genxyz(ms::MSys, bohr=true)
  return join([genxyz(at,bohr) for at in ms.atoms],"\n")
end





end #module