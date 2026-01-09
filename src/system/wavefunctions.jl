"""
    Wavefunctions

  Module for handling wavefunctions: dumping and fetching orbitals/amplitudes etc to/from TREXIO files.
"""
module Wavefunctions
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets
using ..ElemCo.TrexioInterface

export open_dump, close_dump
export dump_orbitals, fetch_orbitals
export dump_rotations, fetch_rotations, is_rotation, is_biorthogonal
export fetch_orbital_energies, fetch_orbital_occupations
export load_wavefunction, save_wavefunction, copy_wavefunction

"""
    dumpfile(EC::ECInfo, intent)

  Get the dump file name for wavefunction.

If `intent="w"`, the dump file for writing is returned. Otherwise, the dump file for reading is returned.
Returns `(filename::String, full_path_filename::String)`.
"""
function dumpfile(EC::ECInfo, intent)
  filename = ""
  if intent == "w"
    filename = EC.options.wf.store
  end
  if filename == ""
    filename = EC.options.wf.dump
  end
  full_filename = joinpath(EC.scr, filename)
  return filename, full_filename
end

"""
    open_dump(EC::ECInfo, intent) -> TrexioFile

  Open the dump file for wavefunction. 

`intent` can be "r", "w", or "u" (read, write, or update).
"""
function open_dump(EC::ECInfo, intent)
  filename, full_filename = dumpfile(EC, intent)
  mode = "reading"
  if intent == "w"
    mode = "writing"
  elseif intent == "u"
    mode = "updating (unsafe mode)"
  end
  println("Opening dump file $filename for $mode ...")
  return open_trexio(full_filename, intent)
end

"""
    open_dump(f::Function, EC::ECInfo, intent)

  Open the dump file for wavefunction, execute function `f` with the opened `TrexioFile`, and close the file.

To be used as `open_dump(EC, intent) do io ... end`.
`intent` can be "r", "w", or "u" (read, write, or update).
"""
function open_dump(f::Function, EC::ECInfo, intent)
  trexio = open_dump(EC, intent)
  try
    f(trexio)
  finally
    close_dump(trexio)
  end
end

"""
    close_dump(trexio::TrexioFile)

  Close the opened TREXIO file.
"""
function close_dump(trexio::TrexioFile)
  close_trexio(trexio)
end

""" 
    dump_orbitals([io::TrexioFile,] EC::ECInfo, cMO::SpinMatrix; basis=nothing, type="HF", 
                  energies=nothing, occupations=nothing, MO="mo")

  Dump orbitals to TREXIO file. 

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function dump_orbitals end
function dump_orbitals(EC::ECInfo, cMO::SpinMatrix; 
                       basis=nothing, type="HF", energies=nothing, occupations=nothing, MO="mo")
  open_dump(EC, "w") do io
    dump_orbitals(io, EC, cMO; basis=basis, type=type, energies=energies, occupations=occupations, MO=MO)
  end
  return
end
function dump_orbitals(io::TrexioFile, EC::ECInfo, cMO::SpinMatrix; 
                       basis=nothing, type="HF", energies=nothing, occupations=nothing, MO="mo")
  println("Dumping orbitals ...")
  oenergies = prepare_orb_vectors(energies, is_restricted(cMO))
  ooccupations = prepare_orb_vectors(occupations, is_restricted(cMO))
  classes = prepare_orb_classes(EC, is_restricted(cMO))
  write_trexio_system(io, EC.system)
  if isnothing(basis)
    basis = generate_basis(EC, "ao")
  end
  write_trexio_orbitals(io, cMO, basis; type, classes=classes, energies=oenergies, occupations=ooccupations, MO=MO)
  return
end

"""
    prepare_orb_vectors(input, restricted)

  Prepare orbital info vectors for dumping.
"""
function prepare_orb_vectors(input, restricted) 
  error("Unsupported input type for prepare_orb_vectors: $(typeof(input))")
end
function prepare_orb_vectors(input::Nothing, restricted)
  return (Float64[], Float64[])
end
function prepare_orb_vectors(input::Vector{Float64}, restricted)
  if restricted
    return (input, Float64[])
  else
    error("For unrestricted orbitals, provide input as a tuple of two vectors.")
  end
end
function prepare_orb_vectors(input::Tuple{Vector{Float64},Vector{Float64}}, restricted)
  if restricted
    return (input[1], Float64[])
  else
    return (input[1], input[2])
  end
end
function prepare_orb_vectors(input::Vector{Vector{Float64}}, restricted)
  if restricted
    @assert length(input) > 0 "Input vector must contain at least one vector for restricted orbitals."
    return (input[1], Float64[])
  else
    @assert length(input) == 2 "For unrestricted orbitals, provide input as a vector of two vectors."
    return (input[1], input[2])
  end
end

function prepare_orb_classes(EC::ECInfo, restricted)
  if !haskey(EC.space, 'm')
    setup_space_system!(EC)
  end
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

"""
    fetch_orbitals([io::TrexioFile,] EC::ECInfo, MO="mo") -> (SpinMatrix, String, BasisSet)

  Fetch the molecular orbitals from the trexio dump.

  Returns a `SpinMatrix` of the orbitals, `type::String` and the basis.

If the basis information is not stored in the dump file (i.e., we have a rotation instead of orbitals), 
`basis` is returned as empty `BasisSet`.
`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function fetch_orbitals end
function fetch_orbitals(EC::ECInfo, MO="mo")
  open_dump(EC, "r") do io
    return fetch_orbitals(io, EC, MO)
  end
end
function fetch_orbitals(io::TrexioFile, EC::ECInfo, MO="mo")
  println("Fetching orbitals ...")
  basis = read_trexio_basis(io)
  if isnothing(basis)
    return read_trexio_rotations(io; MO=MO)..., basis
  else
    return read_trexio_orbitals(io, basis; MO=MO)..., basis
  end
end

"""
    fetch_orbital_energies([io::TrexioFile,] EC::ECInfo, MO="mo") -> (Vector{Float64}, Vector{Float64})
  
  Fetch orbital energies from the trexio dump.
  
Returns tuples of orbital energies for alpha and beta spins.

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function fetch_orbital_energies end
function fetch_orbital_energies(ECInfo, MO="mo")
  open_dump(ECInfo, "r") do io
    return fetch_orbital_energies(io, ECInfo, MO)
  end
end
function fetch_orbital_energies(io::TrexioFile, EC::ECInfo, MO="mo")
  println("Fetching orbital energies ...")
  return read_trexio_orbital_energies(io, MO)
end

"""
    fetch_orbital_occupations([io::TrexioFile,] EC::ECInfo, MO="mo") -> (Vector{Float64}, Vector{Float64})

  Fetch orbital occupations from the trexio dump.

Returns tuples of orbital occupations for alpha and beta spins.

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function fetch_orbital_occupations end
function fetch_orbital_occupations(ECInfo, MO="mo")
  open_dump(ECInfo, "r") do io
    return fetch_orbital_occupations(io, ECInfo, MO)
  end
end
function fetch_orbital_occupations(io::TrexioFile, EC::ECInfo, MO="mo")
  println("Fetching orbital occupations ...")
  return read_trexio_orbital_occupations(io, MO)
end

""" 
    dump_rotations([io::TrexioFile,] EC::ECInfo, cRot::SpinMatrix; type="Rotation", energies=nothing, occupations=nothing, 
                   MO="mo", biorthogonal=false)

  Dump orbital rotations to TREXIO file. 

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function dump_rotations end
function dump_rotations(EC::ECInfo, cRot::SpinMatrix; 
                        type="", energies=nothing, occupations=nothing, MO="mo", biorthogonal=false)
  open_dump(EC, "w") do io
    dump_rotations(io, EC, cRot; type=type, energies=energies, occupations=occupations, MO=MO, biorthogonal=biorthogonal)
  end
  return
end
function dump_rotations(io::TrexioFile, EC::ECInfo, cRot::SpinMatrix; 
                        type="", energies=nothing, occupations=nothing, MO="mo", biorthogonal=false)
  println("Dumping orbital rotations ...")
  oenergies = prepare_orb_vectors(energies, is_restricted(cRot))
  ooccupations = prepare_orb_vectors(occupations, is_restricted(cRot))
  classes = prepare_orb_classes(EC, is_restricted(cRot))
  if biorthogonal && !is_biorthogonal(type)
    type *= " biorthogonal"
  end
  if !is_rotation(type)
    type *= " Rotation"
  end
  write_trexio_rotations(io, cRot; type, classes=classes, energies=oenergies, occupations=ooccupations, MO=MO)
  return
end

"""
    fetch_rotations([io::TrexioFile,] EC::ECInfo, MO="mo") -> (SpinMatrix, String)

  Fetch the molecular orbital rotations from the trexio dump.

  Returns a `SpinMatrix` of the rotations and `type::String`.

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function fetch_rotations end
function fetch_rotations(EC::ECInfo, MO="mo")
  open_dump(EC, "r") do io
    return fetch_rotations(io, EC, MO)
  end
end
function fetch_rotations(io::TrexioFile, EC::ECInfo, MO="mo")
  println("Fetching orbital rotations ...")
  return read_trexio_rotations(io, MO=MO)
end

"""
    load_wavefunction(EC::ECInfo, what::Vector{String})

  Load parts of the wavefunction from file [`WfOptions.dump`](@ref ECInfos.WfOptions).

  `what` can contain any of the following strings:
  - `"all"`: load everything (overrides other options)
  - `"orbitals"`: load orbitals (always loaded)
  - `"orbital_energies"`: load orbital energies
  - `"orbital_occupations"`: load orbital occupations

  Returns a `Dict{String,Any}` with the requested parts of the wavefunction.
"""
function load_wavefunction(EC::ECInfo, what::Vector{String})
  wf = Dict{String,Any}()
  println("what = ", what)
  all = "all" in what
  open_dump(EC, "r") do io
    wf["orbitals"], wf["orbital_type"], wf["basis"] = fetch_orbitals(io, EC)
    if all || "orbital_energies" in what
      wf["orbital_energies"] = fetch_orbital_energies(io, EC)
    end
    if all || "orbital_occupations" in what
      wf["orbital_occupations"] = fetch_orbital_occupations(io, EC)
    end
  end
  return wf
end

"""
    save_wavefunction(EC::ECInfo, wf::AbstractDict)

  Save parts of the wavefunction to file [`WfOptions.store`](@ref ECInfos.WfOptions) or 
  [`WfOptions.dump`](@ref ECInfos.WfOptions) (if `store` is empty).

  `wf` can contain any of the following keys:
  - `basis`: basis set information
  - `orbitals`: molecular orbitals
  - `orbital_type`: type of the orbitals (e.g., "RHF", "UHF", "ROHF", "MCSCF")
  - `orbital_energies`: molecular orbital energies
  - `orbital_occupations`: molecular orbital occupations
  - `amplitudes`: coupled cluster amplitudes
"""
function save_wavefunction(EC::ECInfo, wf::AbstractDict)
  open_dump(EC, "w") do io
    if haskey(wf, "orbitals") 
      if haskey(wf, "basis") && !isempty(wf["basis"])
        dump_orbitals(io, EC, wf["orbitals"]; 
                      basis=wf["basis"], type=get(wf, "orbital_type", "USER"), 
                      energies=get(wf, "orbital_energies", nothing), 
                      occupations=get(wf, "orbital_occupations", nothing))
      else
        dump_rotations(io, EC, wf["orbitals"]; 
                       type=get(wf, "orbital_type", "USER"), 
                       energies=get(wf, "orbital_energies", nothing), 
                       occupations=get(wf, "orbital_occupations", nothing),
                       biorthogonal=is_biorthogonal(get(wf, "orbital_type", "")))
      end
    elseif haskey(wf, "rotations") 
      dump_rotations(io, EC, wf["rotations"]; 
                     type=get(wf, "orbital_type", "USER"), 
                     energies=get(wf, "orbital_energies", nothing), 
                     occupations=get(wf, "orbital_occupations", nothing),
                     biorthogonal=is_biorthogonal(get(wf, "orbital_type", "")))
    end
    # if haskey(wf, "amplitudes")
    #   dump_amplitudes(io, EC, wf["amplitudes"]; type="CCSD")
    # end
  end
  return
end

"""
    copy_wavefunction(EC::ECInfo, tofile::AbstractString="")

  Copy the wavefunction dump file to `tofile`. If `tofile` is not given, copy to the current dump file for writing.

  Note: This does not check the contents of the files.
"""
function copy_wavefunction(EC::ECInfo, tofile::AbstractString="")
  cp(dumpfile(EC, "r")[2], tofile == "" ? dumpfile(EC, "w")[2] : tofile; force=true)
  return
end

"""
    is_rotation(type)

  Returns true if the given orbital type is an orbital rotation 
  (i.e., a unitary transformation of orbitals rather than LCAO coefficients).
"""
is_rotation(type) = "rotation" ∈ split(lowercase(type))

"""
    is_biorthogonal(type)

  Returns true if the given orbital type is a bi-orthogonal rotation.
"""
is_biorthogonal(type::AbstractString) = "biorthogonal" ∈ split(lowercase(type))

end #module