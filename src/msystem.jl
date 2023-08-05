"""
Info about molecular system (geometry/basis).
"""
module MSystem
using ..ElemCo.ECInts
export MSys, ms_exists, Basis, ACenter, genxyz, nuclear_repulsion, bond_length, electron_distribution, guess_nelec, guess_norb

include("elements.jl")
include("minbas.jl")

const BOHR2ANGSTROM = 0.52917721
const ANGSTROM2BOHR = 1/BOHR2ANGSTROM

"""
Basis set
"""
struct Basis
  # e.g., vdz
  name::String
  # coeffs, exponents..
end

""" 
    genbasis4element(basis::Dict,elem::AbstractString)

  Set element specific basis from, e.g., Dict("ao"=>"cc-pVDZ; o=aug-cc-pVDZ","jkfit"=>"cc-pvdz-jkfit")
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
Atomic center
"""
mutable struct ACenter
  """name (as defined in input)"""
  name::String
  """nuclear charge (can be changed...)"""
  nuccharge::Float64
  """coordinates in bohr"""
  coord::AbstractArray{Float64,1}
  """various basis sets (ao,mp2fit,jkfit)"""
  basis::Dict{String,Basis}
end

Base.show(io::IO, val::ACenter) = print(io, val.name, " ", val.coord[1], " ", val.coord[2], " ", val.coord[3]) 
  
""" 
    element_name(name::AbstractString)

  Return element name without numbers.
"""
function element_name(name::AbstractString)
  return rstrip(name,['0','1','2','3','4','5','6','7','8','9'])
end

""" 
    element_NAME(name::AbstractString)

  Return element name in all caps and without numbers.
"""
function element_NAME(name::AbstractString)
  return uppercase(element_name(name))
end

""" 
    a2b(vals,skip)

  Transform from angstrom to bohr (if `skip`: no transformation). 
"""
function a2b(vals,skip)
  if skip
    return vals
  else
    return vals*ANGSTROM2BOHR
  end
end

""" 
    b2a(vals,skip)

  Transform from bohr to angstrom (if `skip`: no transformation). 
"""
function b2a(vals,skip)
  if skip
    return vals
  else
    return vals*BOHR2ANGSTROM
  end
end

"""
    try2create_acenter(line::AbstractString, basis::Dict, bohr = true)

  Create ACenter from a line `<Atom> x y z`. 
  
  If !bohr: coords are in angstrom.
  Returns the center and a bool success variable. 
  If the line has a different format: return dummy center and false.
"""
function try2create_acenter(line::AbstractString, basis::Dict, bohr = true)
  coords = split(line)
  if length(coords) == 4
    xcoord = tryparse(Float64,coords[2])
    ycoord = tryparse(Float64,coords[3])
    zcoord = tryparse(Float64,coords[4])
    if !isnothing(xcoord) && !isnothing(ycoord) && !isnothing(zcoord)
      basis = genbasis4element(basis,coords[1])
      charge = nuclear_charge_of_center(element_NAME(coords[1]))
      return ACenter(coords[1],charge,a2b([xcoord,ycoord,zcoord],bohr),basis), true
    end
  end
  # not a center
  return ACenter("",0.0,[],Dict{String,Basis}()), false
end

"""
Geometry and basis set for each element name in the geometry
"""
mutable struct MSys
  atoms::AbstractArray{ACenter,1}
  function MSys()
    new(ACenter[])
  end
  function MSys(xyz::AbstractString, basis::Dict)
    xyz_lines = strip.(split(xyz,"\n"))
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
        elseif line == ""
          # skip
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

""" 
    ms_exists(ms::MSys)

  Check whether the system is not empty.
"""
function ms_exists(ms::MSys)
  return length(ms.atoms) > 0
end

""" 
    genxyz(ac::ACenter; bohr=true)

  Generate xyz string with element without numbers.
  If bohr: return in coordinates in bohr.
"""
function genxyz(ac::ACenter; bohr=true)
  name = element_name(ac.name)
  return string(name," ",b2a(ac.coord[1],bohr)," ",b2a(ac.coord[2],bohr)," ",b2a(ac.coord[3],bohr))
end

""" 
    genxyz(ms::MSys; bohr=true)

  Generate xyz string with elements without numbers.
  If bohr: return in coordinates in bohr
"""
function genxyz(ms::MSys; bohr=true)
  return join([genxyz(at,bohr=bohr) for at in ms.atoms],"\n")
end

"""
    bond_length(cen1::ACenter, cen2::ACenter)

  Calculate bond length between two centers.
"""
function bond_length(cen1::ACenter, cen2::ACenter)
  return sqrt(sum(abs2,(cen1.coord-cen2.coord)))
end

""" 
    nuclear_repulsion(ms::MSys)

  Calculate nuclear repulsion energy.
"""
function nuclear_repulsion(ms::MSys)
  enuc = 0.0
  for i=2:length(ms.atoms)
    at1 = ms.atoms[i]
    z1 = at1.nuccharge
    for j=1:i-1
      at2 = ms.atoms[j]
      z2 = at2.nuccharge
      enuc += z1*z2/bond_length(at1,at2)
    end
  end
  return enuc
end

""" 
    guess_nelec(ms::MSys)

  Guess the number of electrons in the neutral system.
"""
function guess_nelec(ms::MSys)
  nelec = 0
  for at in ms.atoms
    nelec += nuclear_charge_of_center(element_NAME(at.name))
  end
  return nelec
end

"""
    guess_nalpha(ms::MSys)

  Guess the number of alpha electrons in the neutral system.
"""
function guess_nalpha(ms::MSys)
  nelec = guess_nelec(ms)
  if nelec % 2 == 0
    return nelec ÷ 2
  else
    return (nelec+1) ÷ 2
  end
end

"""
    guess_nbeta(ms::MSys)

  Guess the number of beta electrons in the neutral system.
"""
function guess_nbeta(ms::MSys)
  nelec = guess_nelec(ms)
  if nelec % 2 == 0
    return nelec ÷ 2
  else
    return (nelec-1) ÷ 2
  end
end

"""
    guess_nocc(ms::MSys)

  Guess the number of alpha and beta occupied orbitals in the neutral system.
"""
function guess_nocc(ms::MSys)
  nalpha = guess_nalpha(ms)
  nbeta = guess_nbeta(ms)
  return nalpha, nbeta
end

"""
    guess_norb(ms::MSys)

  Guess the number of orbitals in the system.
"""
function guess_norb(ms::MSys)
    # TODO: use element-specific basis!
  aobasis = lowercase(ms.atoms[1].basis["ao"].name)
  bao = BasisSet(aobasis,genxyz(ms,bohr=false))
  return bao.nbas 
end

""" 
    electron_distribution(elnam::AbstractString, minbas::AbstractString)

  Return the averaged number of electrons in the orbitals in the minimal basis set.
  
  Number of orbitals in the minimal basis set has to be specified in `minbas.jl`.
"""
function electron_distribution(ms::MSys, minbas::AbstractString)
  eldist = Float64[]
  for at in ms.atoms
    elnam = element_NAME(at.name)
    nnum = nuclear_charge_of_center(elnam)
    if nnum == 0
      continue
    end
    eldist = vcat(eldist,electron_distribution(elnam,nshell4l_minbas(nnum,uppercase(minbas))))
  end
  return eldist
end

end #module