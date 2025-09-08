module Wavefunctions
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets
using ..ElemCo.TrexioInterface

export dump_orbitals

""" 
    dump_orbitals(EC::ECInfo, cMO::SpinMatrix; basis=nothing, type="HF", energies=nothing, occupations=nothing)

  Dump orbitals to TREXIO file. 
"""
function dump_orbitals(EC::ECInfo, cMO::SpinMatrix; basis=nothing, type="HF", energies=nothing, occupations=nothing)
  filename = EC.options.wf.dump
  oenergies = prepare_orb_vectors(energies, is_restricted(cMO))
  ooccupations = prepare_orb_vectors(occupations, is_restricted(cMO))
  classes = prepare_orb_classes(EC, is_restricted(cMO))
  println("Dumping orbitals to $filename ...")
  full_filename = joinpath(EC.scr, filename)
  open_trexio(full_filename, "w") do io
    write_trexio_system(io, EC.system)
    if isnothing(basis)
      basis = generate_basis(EC, "ao")
    end
    write_trexio_orbitals(io, cMO, basis; type, classes=classes, energies=oenergies, occupations=ooccupations)
  end
  return
end

"""
    prepare_orb_vectors(input, restricted)

  Prepare orbital info vectors for dumping.
"""
function prepare_orb_vectors(input, restricted)
  if isnothing(input)
    return (Float64[], Float64[])
  elseif restricted
    @assert isa(input, Vector{Float64}) "For restricted orbitals, provide input as a single vector."
    return (input, Float64[])
  else
    @assert length(input) == 2 "For unrestricted orbitals, provide input as a tuple of two vectors."
    return (input[1], input[2])
  end
end

function prepare_orb_classes(EC::ECInfo, restricted)
  classa = fill("Deleted", length(EC.space['m']))
  if restricted
    classa[EC.space['o']] .= "Inactive"
    classa[EC.space['v']] .= "Virtual"
    classa[EC.space['a']] .= "Active"
    classb = String[]
  else
    classb = fill("", length(EC.space['M']))
    classb[EC.space['O']] .= "Inactive"
    classb[EC.space['V']] .= "Virtual"
    classb[EC.space['a']] .= "Active"
  end
  return (classa, classb)
end

end #module