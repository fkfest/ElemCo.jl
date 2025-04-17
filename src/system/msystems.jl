"""
Info about molecular system (geometry/basis).

`ACentre` and `MSystem` structures are used
for the atomic centres and the molecular system, respectively.
"""
module MSystems
using StaticArrays
using Printf
using DocStringExtensions
using ..ElemCo.Elements
using ..ElemCo.Constants
export ACentre, MSystem
export parse_geometry, system_exists, genxyz, nuclear_repulsion, bond_length, electron_distribution
export atomic_position
export guess_nelec, guess_norb, guess_ncore
export atomic_centre_label, element_fullname, element_label, element_LABEL, is_dummy, set_dummy!, unset_dummy!

include("minbas.jl")

"""
    ACentre

  An atomic centre with basis set information.

  $(TYPEDFIELDS)
"""
mutable struct ACentre
  """ atomic centre label (e.g., "H1")"""
  label::String
  """ atomic position in Bohr (3D vector)"""
  position::SVector{3, Float64}
  """ atomic number"""
  atomic_number::Int
  """ nuclear charge"""
  charge::Float64
  """ basis sets (e.g., "ao"=>"cc-pVDZ")"""
  basis::Dict{String,String}
  """ dummy atom"""
  dummy::Bool
end

function ACentre(label::AbstractString, x, y, z, atomic_number::Int, charge, basis::Dict{String,String}) 
  ACentre(label, SVector(x,y,z), atomic_number, charge, basis, false)
end

function ACentre(label::AbstractString, x, y, z, basis::Dict{String,String}) 
  ACentre(label, SVector(x,y,z), basis)
end

function ACentre(label::AbstractString, position::SVector{3, Float64}, basis::Dict{String,String}) 
  atomic_number = nuclear_charge_of_centre(element_LABEL(label))
  charge = atomic_number
  ACentre(label, position, atomic_number, charge, basis, false)
end

ACentre() = ACentre("", SVector(0.0,0.0,0.0), 0, 0, Dict{String,String}(), false)

function Base.show(io::IO, at::ACentre)
  # print(io, at.label, " ", at.position[1], " ", at.position[2], " ", at.position[3])
  @printf(io, "%-3s %16.10f %16.10f %16.10f", at.label, at.position[1], at.position[2], at.position[3])
end

function Base.:(==)(at1::ACentre, at2::ACentre)
  return at1.label == at2.label && at1.position == at2.position && 
        at1.atomic_number == at2.atomic_number && at1.charge == at2.charge && 
        at1.basis == at2.basis && at1.dummy == at2.dummy
end

function Base.isapprox(at1::ACentre, at2::ACentre; kwargs...)
  return at1.label == at2.label && isapprox(at1.position, at2.position; kwargs...) && 
        at1.atomic_number == at2.atomic_number && isapprox(at1.charge, at2.charge; kwargs...) && 
        at1.basis == at2.basis && at1.dummy == at2.dummy
end


"""
    MSystem

  A molecular system with atomic centres.

  $(TYPEDFIELDS)
"""
mutable struct MSystem
  """ atomic centres"""
  centres::Vector{ACentre}
end

MSystem() = MSystem(ACentre[])

Base.length(ms::MSystem) = length(ms.centres)
Base.getindex(ms::MSystem, i::Int) = ms.centres[i]
Base.firstindex(ms::MSystem) = 1
Base.lastindex(ms::MSystem) = length(ms.centres)
Base.first(ms::MSystem) = ms.centres[1]
Base.last(ms::MSystem) = ms.centres[end]
function Base.iterate(ms::MSystem, state::Int=1)
  if state <= length(ms.centres)
    return ms.centres[state], state+1
  else
    return nothing
  end
end
function Base.:(==)(ms1::MSystem, ms2::MSystem)
  return ms1.centres == ms2.centres
end
function Base.isapprox(ms1::MSystem, ms2::MSystem; atol=1e-10, rtol=atol>0 ? 0 : 1e-8)
  if length(ms1) != length(ms2)
    return false
  end
  for (at1,at2) in zip(ms1,ms2)
    if !isapprox(at1, at2; atol, rtol)
      return false
    end
  end
  return true
end

function Base.show(io::IO, ms::MSystem)
  for at in ms
    println(io, at)
  end
end

""" 
    element_label(name::AbstractString)

  Return element label without numbers.
"""
function element_label(name::AbstractString)
  return titlecase(rstrip(name,['0','1','2','3','4','5','6','7','8','9']),strict=true)
end

""" 
    element_label(atom::ACentre)

  Return element label without numbers.
"""
function element_label(atom::ACentre)
  return element_label(atomic_centre_label(atom))
end

""" 
    element_LABEL(name::AbstractString)

  Return element label in all caps and without numbers.
"""
function element_LABEL(name::AbstractString)
  return uppercase(element_label(name))
end

""" 
    element_LABEL(atom::ACentre)

  Return element label in all caps and without numbers.
"""
function element_LABEL(atom::ACentre)
  return element_LABEL(atomic_centre_label(atom))
end

""" 
    element_fullname(atom::ACentre)

  Return element full name.
"""
function element_fullname(atom::ACentre)
  return element_fullname_from_label(element_LABEL(atom))
end

"""
    atomic_centre_label(atom::ACentre)

  Return atomic centre label (i.e., chemical symbol possibly with a number).
"""
function atomic_centre_label(atom::ACentre)
  return atom.label
end

"""
    try2create_atom(line::AbstractString, basis::Dict, ang2bohr=false)

  Create `ACentre` from a line `<Atom> x y z`. 
  
  If ang2bohr is true: the coordinates are in angstrom, convert to bohr.
  Returns the centre and a bool success variable. 
  If the line has a different format: return dummy centre and false.
"""
function try2create_atom(line::AbstractString, basis::Dict, ang2bohr=false)
  coords = split(line)
  if length(coords) == 4
    xcoord = tryparse(Float64,coords[2])
    ycoord = tryparse(Float64,coords[3])
    zcoord = tryparse(Float64,coords[4])
    if !isnothing(xcoord) && !isnothing(ycoord) && !isnothing(zcoord)
      basis4a = genbasis4element(basis,coords[1])
      anum = nuclear_charge_of_centre(element_LABEL(coords[1]))
      if ang2bohr
        xcoord /= Constants.BOHR2ANGSTROM
        ycoord /= Constants.BOHR2ANGSTROM
        zcoord /= Constants.BOHR2ANGSTROM
      end
      return ACentre(coords[1], xcoord, ycoord, zcoord, anum, anum, basis4a), true
    end
  end
  # not a centre
  return ACentre(), false
end

""" 
    genbasis4element(basis::Dict, elem::AbstractString)

  Set element specific basis from, e.g., 
```julia
Dict("ao"=>"cc-pVDZ; o=aug-cc-pVDZ; 
  h={! hydrogen             (4s,1p) -> [2s,1p]
    s, H , 13.0100000, 1.9620000, 0.4446000, 0.1220000
    c, 1.4, 0.0196850, 0.1379770, 0.4781480, 0.5012400
    c, 4.4, 1.0000000
    p, H , 0.7270000
    c, 1.1, 1.0000000}",
    "jkfit"=>"cc-pvdz-jkfit")
```
"""
function genbasis4element(basis::Dict, elem::AbstractString)
  elembasis = Dict{String,String}()
  elemUP = uppercase(elem)
  elemLAB = element_LABEL(elem)
  for (type,name) in basis
    names = strip.(split(name,[';']))
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
          elLAB = element_LABEL(eldef[1])
          elUP = uppercase(eldef[1])
          # check whether the basis is defined for the centre or its element symbol
          if elUP == elemUP || (elUP == elLAB && elLAB == elemLAB)
            elbas = eldef[2]
          end
        end
      end
    end
    elembasis[type] = elbas
  end
  return elembasis
end

"""
    is_dummy(ac::ACentre)

  Check whether the atom is a dummy atom.
"""
is_dummy(ac::ACentre) = ac.dummy

"""
    set_dummy!(ac::ACentre)

  Set the atom as a dummy atom.
"""
function set_dummy!(ac::ACentre) 
  ac.dummy = true
end

"""
    unset_dummy!(ac::ACentre)

  Unset the atom as a dummy atom.
"""
function unset_dummy!(ac::ACentre)
  ac.dummy = false
end

"""
    set_dummy!(sys::MSystem, list)

  Set dummy atoms in the system.
  The list can contain atom indices or element labels.
  All atoms are unset first, i.e., only the atoms in the list are set as dummy.
"""
function set_dummy!(sys::MSystem, list)
  for at in sys
    unset_dummy!(at)
  end
  for a in list
    if a isa Int64
      if a < 1 || a > length(sys)
        error("Atom index $a out of range!")
      end
      set_dummy!(sys[a])
    else
      found = false
      for at in sys
        if uppercase(atomic_centre_label(at)) == uppercase(String(a))
          set_dummy!(at)
          found = true
        end
      end
      if !found
        error("Atom $a not found in the system!")
      end
    end
  end 
end

"""
    parse_geometry(geometry::AbstractString, basis::AbstractString)

  Parse geometry `geometry` and return `MSystem` object.
  The geometry can be in xyz format or in a file.
  The basis set can be defined for each element in the geometry.
"""
function parse_geometry(geometry::AbstractString, basis::AbstractString)
  return parse_geometry(geometry, Dict("ao"=>basis))
end

"""
    parse_geometry(geometry::AbstractString, basis::Dict)

  Parse geometry `geometry` and return `MSystem` object.
  The geometry can be in xyz format or in a file.
  The basis set can be defined for each element in the geometry.
"""
function parse_geometry(geometry::AbstractString, basis::Dict)
  geom_lines = strip.(split(geometry,"\n"))
  if length(geom_lines) == 0
    error("Empty geometry in parse_geometry!")
  end
  array_of_atoms, badline = parse_xyz_geometry(geom_lines, basis)
  if !isempty(badline) && length(geom_lines) == 1 && isfile(geom_lines[1])
    filename = geom_lines[1]
    geom_lines = strip.(readlines(filename))
    if length(geom_lines) == 0
      error("Empty geometry file $filename in parse_geometry!")
    end
    array_of_atoms, badline = parse_xyz_geometry(geom_lines, basis)
  end
  if !isempty(badline)
    error("Unsupported geometry line $badline")
  end
  return MSystem(array_of_atoms)
end

"""
    parse_xyz_geometry(xyz_lines::AbstractArray, basis::Dict)

  Parse xyz geometry `xyz_lines` stored as a vector of strings.
  Return array of `ACentre`s and an empty string in case of success.

  Empty lines are skipped.
  The default units are bohr. If the line is `bohr` or `angstrom`: change the units.
  If the first line is a number: assume xyz format and skip the second line
  (in this case, the default units are angstroms).
  If parsing fails: return empty array and the line that failed.
"""
function parse_xyz_geometry(xyz_lines::AbstractArray, basis::Dict)
  array_of_atoms = ACentre[]
  badline = ""
  angstrom = false
  firstline = true
  xyz_format = false
  for line in xyz_lines
    if xyz_format && firstline
      # skip second line in xyz format
      firstline = false
      continue
    end
    ac, success = try2create_atom(line, basis, angstrom)
    if success
      push!(array_of_atoms, ac)
    elseif line == "bohr"
      angstrom = false
      success = true
    elseif line == "angstrom"
      angstrom = true
      success = true
    elseif line == ""
      # skip
      continue
    elseif firstline
      # check whether it's the number of atoms in the xyz format
      natoms = tryparse(Int64, line)
      xyz_format = !isnothing(natoms)
      if xyz_format
        angstrom = true # assume angstrom in xyz format
        # skip the second line
        continue
      end
    end
    if !success
      badline = line
      break
    end
    firstline = false
  end
  return array_of_atoms, badline
end

""" 
    system_exists(ms::MSystem)

  Check whether the system is not empty.
"""
function system_exists(ms::MSystem)
  return length(ms) > 0
end

"""
    atomic_position(at::ACentre; bohr2ang=false)

  Return the position of the atom.
  If `bohr2ang` is true: convert the position to angstrom.
"""
function atomic_position(at::ACentre; bohr2ang=false) 
  if bohr2ang
    return at.position * BOHR2ANGSTROM
  else
    return at.position
  end
end

""" 
    genxyz(ac::ACentre; angstrom=true)

  Generate xyz string with element without numbers.
"""
function genxyz(ac::ACentre; angstrom=true)
  name = element_label(ac)
  coord = atomic_position(ac, bohr2ang=angstrom)
  return @sprintf("%-3s %16.10f %16.10f %16.10f",name,coord[1],coord[2],coord[3]) 
end

""" 
    genxyz(ms::MSystem; angstrom=true)

  Generate xyz string with elements without numbers.
"""
function genxyz(ms::MSystem; angstrom=true)
  return join([genxyz(at; angstrom) for at in ms],"\n")
end

"""
    bond_length(cen1::ACentre, cen2::ACentre)

  Calculate bond length in bohr between two centres.
"""
function bond_length(cen1::ACentre, cen2::ACentre)
  return sqrt(sum(abs2, cen1.position-cen2.position))
end

""" 
    nuclear_repulsion(ms::MSystem)

  Calculate nuclear repulsion energy.
"""
function nuclear_repulsion(ms::MSystem)
  enuc::Float64 = 0.0
  ncentre::Int = length(ms)
  for i = 2:ncentre
    at1 = ms[i]
    if is_dummy(at1)
      continue
    end
    z1 = at1.charge
    for j = 1:i-1
      at2 = ms[j]
      if is_dummy(at2)
        continue
      end
      z2 = at2.charge
      enuc += z1*z2/bond_length(at1, at2)
    end
  end
  return enuc
end

""" 
    guess_nelec(ms::MSystem)

  Guess the number of electrons in the neutral system.
"""
function guess_nelec(ms::MSystem)
  nelec = 0
  for at in ms if !is_dummy(at)
    nelec += at.atomic_number
  end end
  return nelec
end

"""
    guess_nalpha(ms::MSystem)

  Guess the number of alpha electrons in the neutral system.
"""
function guess_nalpha(ms::MSystem)
  nelec = guess_nelec(ms)
  if nelec % 2 == 0
    return nelec ÷ 2
  else
    return (nelec+1) ÷ 2
  end
end

"""
    guess_nbeta(ms::MSystem)

  Guess the number of beta electrons in the neutral system.
"""
function guess_nbeta(ms::MSystem)
  nelec = guess_nelec(ms)
  if nelec % 2 == 0
    return nelec ÷ 2
  else
    return (nelec-1) ÷ 2
  end
end

"""
    guess_nocc(ms::MSystem)

  Guess the number of alpha and beta occupied orbitals in the neutral system.
"""
function guess_nocc(ms::MSystem)
  nalpha = guess_nalpha(ms)
  nbeta = guess_nbeta(ms)
  return nalpha, nbeta
end


"""
    guess_ncore(ms::MSystem, coretype::Symbol=:large)

  Guess the number of core orbitals in the system.

  `coretype` as in [`Elements.ncoreorbs`](@ref Elements.ncoreorbs).
"""
function guess_ncore(ms::MSystem, coretype::Symbol=:large)
  return sum(Int[ncoreorbs(element_LABEL(at),coretype) for at in ms if !is_dummy(at)])
end

""" 
    electron_distribution(ms::MSystem, minbas::AbstractString)

  Return the averaged number of electrons in the orbitals in the minimal basis set.
  
  Number of orbitals in the minimal basis set has to be specified in `minbas.jl`.
"""
function electron_distribution(ms::MSystem, minbas::AbstractString)
  eldist = Float64[]
  for at in ms 
    elnam = element_LABEL(at)
    nnum = at.atomic_number
    eldist4el = electron_distribution4element(elnam, nshell4l_minbas(nnum, uppercase(minbas)))
    if is_dummy(at)
      eldist4el .= 0.0
    end 
    eldist = vcat(eldist, eldist4el)
  end
  return eldist
end


end #module