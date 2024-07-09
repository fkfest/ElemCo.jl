"""
Info about molecular system (geometry/basis).

`Atom` and `FlexibleSystem` from `AtomsBase` package are used
for the atoms and the molecular system, respectively.
"""
module MSystem
using Unitful, UnitfulAtomic
using AtomsBase
using Printf
using DocStringExtensions
using ..ElemCo.Elements
using ..ElemCo.Constants
export parse_geometry, system_exists, genxyz, nuclear_repulsion, bond_length, electron_distribution
export guess_nelec, guess_norb, guess_ncore
export element_name, element_symbol, element_SYMBOL, is_dummy, set_dummy!, unset_dummy!

include("minbas.jl")

""" 
    genbasis4element(basis::Dict,elem::AbstractString)

  Set element specific basis from, e.g., Dict("ao"=>"cc-pVDZ; o=aug-cc-pVDZ","jkfit"=>"cc-pvdz-jkfit")
"""
function genbasis4element(basis::Dict,elem::AbstractString)
  elembasis = Dict{String,String}()
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
    elembasis[type] = elbas
  end
  return elembasis
end

""" 
    element_symbol(name::AbstractString)

  Return element symbol without numbers.
"""
function element_symbol(name::AbstractString)
  return titlecase(rstrip(name,['0','1','2','3','4','5','6','7','8','9']),strict=true)
end

# """ 
#     element_symbol(atom::Atom)

#   Return element symbol without numbers.
# """
# function element_symbol(atom::Atom)
#   return element_symbol(string(atomic_symbol(atom)))
# end

""" 
    element_SYMBOL(name::AbstractString)

  Return element symbol in all caps and without numbers.
"""
function element_SYMBOL(name::AbstractString)
  return uppercase(element_symbol(name))
end

""" 
    element_SYMBOL(atom::Atom)

  Return element symbol in all caps and without numbers.
"""
function element_SYMBOL(atom::Atom)
  return element_SYMBOL(string(atomic_symbol(atom)))
end

""" 
    element_name(atom::Atom)

  Return element name.
"""
function element_name(atom::Atom)
  return element_name_from_symbol(element_SYMBOL(atom))
end

"""
    try2create_atom(line::AbstractString, basis::Dict, unit=u"bohr")

  Create `Atom` from a line `<Atom> x y z`. 
  
  `unit` is the unit of the coordinates.
  Returns the center and a bool success variable. 
  If the line has a different format: return dummy center and false.
"""
function try2create_atom(line::AbstractString, basis::Dict, unit=u"bohr")
  coords = split(line)
  if length(coords) == 4
    xcoord = tryparse(Float64,coords[2])
    ycoord = tryparse(Float64,coords[3])
    zcoord = tryparse(Float64,coords[4])
    if !isnothing(xcoord) && !isnothing(ycoord) && !isnothing(zcoord)
      basis4a = genbasis4element(basis,coords[1])
      anum = nuclear_charge_of_center(element_SYMBOL(coords[1]))
      return Atom(anum, [xcoord,ycoord,zcoord]*unit, atomic_symbol=Symbol(coords[1]), basis=basis4a), true
    end
  end
  # not a center
  return Atom(1, atomic_symbol=:X, [0.0,0.0,0.0]u"bohr"), false
end

"""
    is_dummy(ac::Atom)

  Check whether the atom is a dummy atom.
"""
function is_dummy(ac::Atom)
  return haskey(ac, :dummy)
end

"""
    set_dummy!(ac::Atom)

  Set the atom as a dummy atom.
"""
function set_dummy!(ac::Atom)
  ac.data[:dummy] = true
end

"""
    unset_dummy!(ac::Atom)

  Unset the atom as a dummy atom.
"""
function unset_dummy!(ac::Atom)
  delete!(ac.data, :dummy)
end

"""
    parse_geometry(geometry::AbstractString, basis::AbstractString)

  Parse geometry `geometry` and return `FlexibleSystem` object.
  The geometry can be in xyz format or in a file.
  The basis set can be defined for each element in the geometry.
"""
function parse_geometry(geometry::AbstractString, basis::AbstractString)
  return parse_geometry(geometry, Dict("ao"=>basis))
end

"""
    parse_geometry(geometry::AbstractString, basis::Dict)

  Parse geometry `geometry` and return `FlexibleSystem` object.
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
  return isolated_system(array_of_atoms)
end

"""
    parse_xyz_geometry(xyz_lines::AbstractArray, basis::Dict)

  Parse xyz geometry `xyz_lines` stored as a vector of strings.
  Return array of `Atom`s and an empty string in case of success.

  Empty lines are skipped.
  The default units are bohr. If the line is `bohr` or `angstrom`: change the units.
  If the first line is a number: assume xyz format and skip the second line
  (in this case, the default units are angstroms).
  If parsing fails: return empty array and the line that failed.
"""
function parse_xyz_geometry(xyz_lines::AbstractArray, basis::Dict)
  array_of_atoms = Atom[]
  badline = ""
  unit = u"bohr"
  firstline = true
  xyz_format = false
  for line in xyz_lines
    if xyz_format && firstline
      # skip second line in xyz format
      firstline = false
      continue
    end
    ac, success = try2create_atom(line, basis, unit)
    if success
      push!(array_of_atoms,ac)
    elseif line == "bohr"
      unit = u"bohr"
      success = true
    elseif line == "angstrom"
      unit = u"angstrom"
      success = true
    elseif line == ""
      # skip
      continue
    elseif firstline
      # check whether it's the number of atoms in the xyz format
      natoms = tryparse(Int64,line)
      xyz_format = !isnothing(natoms)
      if xyz_format
        unit = u"angstrom" # assume angstrom in xyz format
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
    system_exists(ms::AbstractSystem)

  Check whether the system is not empty.
"""
function system_exists(ms::AbstractSystem)
  n::Int = length(ms.particles)
  return n > 0
end

""" 
    genxyz(ac::Atom; unit=u"angstrom")

  Generate xyz string with element without numbers.
"""
function genxyz(ac::Atom; unit=u"angstrom")
  name = element_symbol(ac)
  coord = uconvert.(unit, ac.position)/unit
  return @sprintf("%-3s %16.10f %16.10f %16.10f",name,coord[1],coord[2],coord[3]) 
end

""" 
    genxyz(ms::AbstractSystem; unit=u"angstrom")

  Generate xyz string with elements without numbers.
"""
function genxyz(ms::AbstractSystem; unit=u"angstrom")
  return join([genxyz(at; unit) for at in ms],"\n")
end

"""
    bond_length(cen1::Atom, cen2::Atom)

  Calculate bond length in bohr between two centers.
"""
function bond_length(cen1::Atom, cen2::Atom)
  return sqrt(sum(abs2,uconvert.(u"bohr", cen1.position-cen2.position)/u"bohr"))
end

""" 
    nuclear_repulsion(ms::AbstractSystem)

  Calculate nuclear repulsion energy.
"""
function nuclear_repulsion(ms::AbstractSystem)
  enuc::Float64 = 0.0
  ncenter::Int = length(ms.particles)
  for i = 2:ncenter
    at1 = ms[i]
    if is_dummy(at1)
      continue
    end
    z1 = atomic_number(at1)
    for j = 1:i-1
      at2 = ms[j]
      if is_dummy(at2)
        continue
      end
      z2 = atomic_number(at2)
      enuc += z1*z2/bond_length(at1,at2)
    end
  end
  return enuc
end

""" 
    guess_nelec(ms::AbstractSystem)

  Guess the number of electrons in the neutral system.
"""
function guess_nelec(ms::AbstractSystem)
  nelec = 0
  for at in ms if !is_dummy(at)
    nelec += atomic_number(at)
  end end
  return nelec
end

"""
    guess_nalpha(ms::AbstractSystem)

  Guess the number of alpha electrons in the neutral system.
"""
function guess_nalpha(ms::AbstractSystem)
  nelec = guess_nelec(ms)
  if nelec % 2 == 0
    return nelec ÷ 2
  else
    return (nelec+1) ÷ 2
  end
end

"""
    guess_nbeta(ms::AbstractSystem)

  Guess the number of beta electrons in the neutral system.
"""
function guess_nbeta(ms::AbstractSystem)
  nelec = guess_nelec(ms)
  if nelec % 2 == 0
    return nelec ÷ 2
  else
    return (nelec-1) ÷ 2
  end
end

"""
    guess_nocc(ms::AbstractSystem)

  Guess the number of alpha and beta occupied orbitals in the neutral system.
"""
function guess_nocc(ms::AbstractSystem)
  nalpha = guess_nalpha(ms)
  nbeta = guess_nbeta(ms)
  return nalpha, nbeta
end


"""
    guess_ncore(ms::AbstractSystem, coretype::Symbol=:large)

  Guess the number of core orbitals in the system.

  `coretype` as in [`Elements.ncoreorbs`](@ref Elements.ncoreorbs).
"""
function guess_ncore(ms::AbstractSystem, coretype::Symbol=:large)
  return sum(Int[ncoreorbs(element_SYMBOL(at),coretype) for at in ms if !is_dummy(at)])
end

""" 
    electron_distribution(ms::AbstractSystem, minbas::AbstractString)

  Return the averaged number of electrons in the orbitals in the minimal basis set.
  
  Number of orbitals in the minimal basis set has to be specified in `minbas.jl`.
"""
function electron_distribution(ms::AbstractSystem, minbas::AbstractString)
  eldist = Float64[]
  for at in ms if !is_dummy(at)
    elnam = element_SYMBOL(at)
    nnum = atomic_number(at)
    eldist = vcat(eldist,electron_distribution4element(elnam,nshell4l_minbas(nnum,uppercase(minbas))))
  end end
  return eldist
end


end #module