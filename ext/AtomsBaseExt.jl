"""
    AtomsBaseExt

This module provides an interface to AtomsBase
"""
module AtomsBaseExt
using ElemCo
using AtomsBase
using Unitful, UnitfulAtomic
using ElemCo.MSystems
using ElemCo.ECInfos

function AtomsBase.FlexibleSystem(ms::MSystem)
  atoms = Atom[]
  for at in ms
    push!(atoms, Atom(at.atomic_number, at.position*u"bohr"))
  end
  return isolated_system(atoms)
end

function ElemCo.MSystems.MSystem(fs::FlexibleSystem, basis::AbstractString)
  return ElemCo.MSystems.MSystem(fs, Dict("ao" => basis))
end

function ElemCo.MSystems.MSystem(fs::FlexibleSystem, basis::Dict{String, String})
  atoms = ACentre[]
  for at in fs
    alabel = string(element_symbol(at))
    basis4a = ElemCo.MSystems.genbasis4element(basis, alabel)
    push!(atoms, ACentre(alabel, uconvert.(u"bohr", at.position)/u"bohr", basis4a))
  end
  return MSystem(atoms)
end

end # module