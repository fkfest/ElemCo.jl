"""
    Wavefunctions

  Module for handling wavefunctions: dumping and fetching orbitals/amplitudes etc to/from TREXIO files.
"""
module Wavefunctions
using LinearAlgebra
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets
using ..ElemCo.TrexioInterface

export open_dump, close_dump
export dump_orbitals, fetch_orbitals, fetch_orbital_classes
export dump_rotations, fetch_rotations, is_rotation, is_biorthogonal
export fetch_orbital_energies, fetch_orbital_occupations
export load_wavefunction, save_wavefunction, copy_wavefunction
export dump_amplitudes, has_amplitudes, has_dumpfile
export fetch_restricted_amplitudes, fetch_unrestricted_amplitudes
export transfer_orbitals_to_store!, OrbitalData, fetch_orbital_data

"""
    dumpfile(EC::ECInfo, intent; start=false)

  Get the dump file name for wavefunction.

If `intent="w"`, the dump file for writing is returned. Otherwise, the dump file for reading is returned.
If `start=true` and `intent != "w"`, the start file (`wf.start`) is returned.
Returns `(filename::String, full_path_filename::String)`.
"""
function dumpfile(EC::ECInfo, intent; start::Bool=false)
  filename = ""
  if intent == "w"
    filename = EC.options.wf.store
  elseif start
    filename = EC.options.wf.start
  end
  if filename == ""
    filename = EC.options.wf.dump
  end
  full_filename = joinpath(EC.scr, filename)
  return filename, full_filename
end

"""
    has_dumpfile(EC::ECInfo; start=false)

  Check if the dump file exists.

If `start=true`, checks for the start file (`wf.start`) instead.
Returns `true` if the file exists.
"""
function has_dumpfile(EC::ECInfo; start::Bool=false)
  _, full_filename = dumpfile(EC, "r"; start=start)
  return isfile(full_filename)
end

"""
    open_dump(EC::ECInfo, intent; start=false) -> TrexioFile

  Open the dump file for wavefunction. 

`intent` can be "r", "w", or "u" (read, write, or update).
If `start=true`, opens the start file (`wf.start`) instead.
"""
function open_dump(EC::ECInfo, intent; start::Bool=false)
  filename, full_filename = dumpfile(EC, intent; start=start)
  mode = "reading"
  if intent == "w"
    @assert !start "Cannot open start file for writing."
    mode = "writing"
  elseif intent == "u"
    mode = "updating (unsafe mode)"
  end
  println("Opening dump file $filename for $mode ...")
  return open_trexio(full_filename, intent)
end

"""
    open_dump(f::Function, EC::ECInfo, intent; start=false)

  Open the dump file for wavefunction, execute function `f` with the opened `TrexioFile`, and close the file.

To be used as `open_dump(EC, intent) do io ... end`.
`intent` can be "r", "w", or "u" (read, write, or update).
If `start=true`, opens the start file (`wf.start`) instead.
"""
function open_dump(f::Function, EC::ECInfo, intent; start::Bool=false)
  trexio = open_dump(EC, intent; start=start)
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
  
  # Save current space, apply freezing to determine core/deleted, then restore
  space_save = save_space(EC)
  freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc; verbose=false)
  freeze_nvirt!(EC, EC.options.wf.freeze_nvirt; verbose=false)
  
  # Now EC.space has the frozen configuration
  classa = fill("Deleted", length(EC.space['m']))
  classa[EC.space['o']] .= "Inactive"
  classa[EC.space['v']] .= "Virtual"
  classa[EC.space['a']] .= "Active"
  # Mark frozen core: compare with saved space to find removed occupied orbitals
  frozen_occ = setdiff(space_save['o'], EC.space['o'])
  classa[frozen_occ] .= "Core"
    
  if restricted
    classb = String[]
  else
    classb = fill("Deleted", length(EC.space['M']))
    classb[EC.space['O']] .= "Inactive"
    classb[EC.space['V']] .= "Virtual"
    classb[EC.space['a']] .= "Active"
    # Mark frozen core: compare with saved space to find removed occupied orbitals
    frozen_occ = setdiff(space_save['O'], EC.space['O'])
    classb[frozen_occ] .= "Core"
  end
    
  # Restore original space
  restore_space!(EC, space_save)
  
  return (classa, classb)
end

"""
    fetch_orbitals([io::TrexioFile,] EC::ECInfo; MO="mo", start=false) -> (SpinMatrix, String, BasisSet)

  Fetch the molecular orbitals from the trexio dump.

  Returns a `SpinMatrix` of the orbitals, `type::String` and the basis.

If the basis information is not stored in the dump file (i.e., we have a rotation instead of orbitals), 
`basis` is returned as empty `BasisSet`.
`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
If `start=true`, reads from the start file (`wf.start`) instead of the current dump file.
"""
function fetch_orbitals end
function fetch_orbitals(EC::ECInfo; MO="mo", start::Bool=false)
  open_dump(EC, "r"; start=start) do io
    return fetch_orbitals(io, EC; MO=MO)
  end
end
function fetch_orbitals(io::TrexioFile, EC::ECInfo; MO="mo")
  println("Fetching orbitals ...")
  basis = read_trexio_basis(io)
  if isempty(basis)
    cMO, type = read_trexio_rotations(io; MO=MO)
    return cMO, type, BasisSet()  # Return empty basis for rotations
  else
    return read_trexio_orbitals(io, basis; MO=MO)..., basis
  end
end

"""
    fetch_orbital_classes([io::TrexioFile,] EC::ECInfo; MO="mo", start=false) -> (Vector{String}, Vector{String})

  Fetch orbital classes from the trexio dump.

Returns tuples of orbital classes for alpha and beta spins.
Classes can be "Core", "Inactive", "Active", "Virtual", "Deleted".

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
If `start=true`, reads from the start file (`wf.start`) instead of the current dump file.
"""
function fetch_orbital_classes end
function fetch_orbital_classes(EC::ECInfo; MO="mo", start::Bool=false)
  open_dump(EC, "r"; start=start) do io
    return fetch_orbital_classes(io, EC; MO=MO)
  end
end
function fetch_orbital_classes(io::TrexioFile, EC::ECInfo; MO="mo")
  println("Fetching orbital classes ...")
  return read_trexio_orbital_classes(io, MO)
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
    fetch_orbital_occupations([io::TrexioFile,] EC::ECInfo, MO="mo"; start=false) -> (Vector{Float64}, Vector{Float64})

  Fetch orbital occupations from the trexio dump.

Returns tuples of orbital occupations for alpha and beta spins.

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
If `start=true`, reads from the start file (`wf.start`) instead of the current dump file.
"""
function fetch_orbital_occupations end
function fetch_orbital_occupations(EC::ECInfo, MO="mo"; start::Bool=false)
  open_dump(EC, "r"; start=start) do io
    return fetch_orbital_occupations(io, EC, MO)
  end
end
function fetch_orbital_occupations(io::TrexioFile, EC::ECInfo, MO="mo")
  println("Fetching orbital occupations ...")
  return read_trexio_orbital_occupations(io, MO)
end

"""
    OrbitalData

  Container for orbital data fetched from a dump file.
"""
struct OrbitalData
  cMO::SpinMatrix
  mo_type::String
  basis::BasisSet
  energies::NTuple{2,Vector{Float64}}
  occupations::NTuple{2,Vector{Float64}}
end

"""
    fetch_orbital_data(EC::ECInfo; MO="mo") -> Union{OrbitalData, Nothing}

  Fetch all orbital data from the dump file.
  
  Returns `nothing` if no dump file exists (FCIDUMP-only case).
  This should be called BEFORE opening the store file for writing,
  to avoid issues when dump and store are the same file.
"""
function fetch_orbital_data(EC::ECInfo; MO="mo")
  if !has_dumpfile(EC)
    return nothing
  end
  cMO, mo_type, basis = fetch_orbitals(EC; MO=MO)
  energies = fetch_orbital_energies(EC, MO)
  occupations = fetch_orbital_occupations(EC, MO)
  return OrbitalData(cMO, mo_type, basis, energies, occupations)
end

"""
    transfer_orbitals_to_store!(io_store::TrexioFile, EC::ECInfo, orbital_data::Union{OrbitalData, Nothing}=nothing; MO="mo")

  Transfer orbitals from the dump file to an already opened store file.

  If `orbital_data` is provided, uses it directly. Otherwise fetches from dump file.
  
  **Important**: When dump and store files are the same, `orbital_data` must be
  pre-fetched before opening the store file to avoid reading from a truncated file.
  
  This properly handles cases where frozen orbitals or geometry differ
  between dump and store files.
  
  For FCIDUMP-only calculations (no dump file), stores a unity rotation matrix
  instead of orbitals, allowing amplitude storage and restart.

# Arguments
- `io_store`: An already opened TrexioFile for writing
- `EC`: Electronic structure information object
- `orbital_data`: Pre-fetched orbital data, or `nothing` to fetch from dump
- `MO`: "mo" for molecular orbitals or "po" for positron orbitals
"""
function transfer_orbitals_to_store!(io_store::TrexioFile, EC::ECInfo, 
                                     orbital_data::Union{OrbitalData,Nothing}=nothing; MO="mo")
  # If no orbital data was pre-fetched, this is a FCIDUMP-only case.
  # Create a unity rotation for amplitude storage.
  if isnothing(orbital_data)
    println("No dump file found - storing unity rotation for FCIDUMP-only calculation ...")
    norb = n_orbs(EC)
    restricted = is_closed_shell(EC)
    
    # Create unity rotation matrix
    unity = Matrix{Float64}(I, norb, norb)
    if restricted
      cRot = SpinMatrix(unity)
    else
      cRot = SpinMatrix(unity, copy(unity))
    end
    
    # No classes for FCIDUMP-only
    classes = (String[], String[])
    
    # Write unity rotation to store file
    write_trexio_rotations(io_store, cRot; type="Rotation", classes=classes, MO=MO)
    return
  end
  
  cMO = orbital_data.cMO
  mo_type = orbital_data.mo_type
  basis = orbital_data.basis
  energies = orbital_data.energies
  occupations = orbital_data.occupations
  
  # If the basis is empty, we have rotations (not orbitals) - write as rotations
  if isempty(basis)
    println("Dumping rotations ...")
    classes = (String[], String[])
    write_trexio_rotations(io_store, cMO; type=mo_type, classes=classes, 
                          energies=energies, occupations=occupations, MO=MO)
    return
  end
  
  # Generate orbital classes from the system (not from dump file, 
  # because core orbital count may have changed)
  if !isempty(EC.system)
    # Save current space, setup from system, generate classes, then restore
    space_save = save_space(EC)
    setup_space_system!(EC; verbose=false)
    classes = prepare_orb_classes(EC, is_restricted(cMO))
    restore_space!(EC, space_save)
    # Use current basis for output (projection is done when retrieving amplitudes)
    basis = generate_basis(EC, "ao")
  else
    # No system available (FCIDUMP only) - skip class generation
    classes = (String[], String[])
  end
  
  # Write to store file
  println("Dumping orbitals ...")
  write_trexio_system(io_store, EC.system)
  write_trexio_orbitals(io_store, cMO, basis; type=mo_type, classes=classes,
                        energies=energies, occupations=occupations, MO=MO)
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

"""
    dump_amplitudes([io::TrexioFile,] EC::ECInfo, T1, T2)

  Dump closed-shell CC amplitudes to TREXIO file.

  `T1` is the singles amplitude matrix (nvirt × nocc).
  `T2` is the doubles amplitude tensor (nvirt × nvirt × nocc × nocc).
"""
function dump_amplitudes end
function dump_amplitudes(EC::ECInfo, T1::AbstractMatrix, T2::AbstractArray{<:Real,4})
  open_dump(EC, "u") do io
    dump_amplitudes(io, EC, T1, T2)
  end
  return
end
function dump_amplitudes(io::TrexioFile, EC::ECInfo, T1::AbstractMatrix, T2::AbstractArray{<:Real,4})
  println("Dumping amplitudes ...")
  write_trexio_amplitudes(io, T1, T2)
  return
end

"""
    dump_amplitudes([io::TrexioFile,] EC::ECInfo, T1a, T1b, T2a, T2b, T2ab)

  Dump unrestricted CC amplitudes to TREXIO file.

  `T1a`, `T1b` are the α and β singles amplitude matrices.
  `T2a`, `T2b`, `T2ab` are the αα, ββ, and αβ doubles amplitude tensors.
"""
function dump_amplitudes(EC::ECInfo, T1a::AbstractMatrix, T1b::AbstractMatrix, 
                         T2a::AbstractArray{<:Real,4}, T2b::AbstractArray{<:Real,4}, 
                         T2ab::AbstractArray{<:Real,4})
  open_dump(EC, "u") do io
    dump_amplitudes(io, EC, T1a, T1b, T2a, T2b, T2ab)
  end
  return
end
function dump_amplitudes(io::TrexioFile, EC::ECInfo, 
                         T1a::AbstractMatrix, T1b::AbstractMatrix, 
                         T2a::AbstractArray{<:Real,4}, T2b::AbstractArray{<:Real,4}, 
                         T2ab::AbstractArray{<:Real,4})
  println("Dumping amplitudes ...")
  write_trexio_amplitudes(io, T1a, T1b, T2a, T2b, T2ab)
  return
end

"""
    fetch_restricted_amplitudes([io::TrexioFile,] EC::ECInfo; start=false)

  Fetch restricted CC amplitudes from the trexio dump.

  Returns `(T1, T2)` for closed-shell case.
  If `start=true`, reads from the start file (`wf.start`) first.

  Returns empty arrays if amplitudes are not found in the dump file.
"""
function fetch_restricted_amplitudes end
function fetch_restricted_amplitudes(EC::ECInfo; start::Bool=false)
  open_dump(EC, "r"; start=start) do io
    return fetch_restricted_amplitudes(io, EC)
  end
end
function fetch_restricted_amplitudes(io::TrexioFile, EC::ECInfo)
  println("Fetching restricted amplitudes ...")
  T1 = read_trexio_singles(io)
  T2 = read_trexio_doubles(io)
  return (T1, T2)
end

"""
    fetch_unrestricted_amplitudes([io::TrexioFile,] EC::ECInfo; start=false)

  Fetch unrestricted CC amplitudes from the trexio dump.

  Returns `(T1a, T1b, T2a, T2b, T2ab)` for unrestricted case.
  If `start=true`, reads from the start file (`wf.start`) first.

  Returns empty arrays if amplitudes are not found in the dump file.
"""
function fetch_unrestricted_amplitudes end
function fetch_unrestricted_amplitudes(EC::ECInfo; start::Bool=false)
  open_dump(EC, "r"; start=start) do io
    return fetch_unrestricted_amplitudes(io, EC)
  end
end
function fetch_unrestricted_amplitudes(io::TrexioFile, EC::ECInfo)
  println("Fetching unrestricted amplitudes ...")
  T1a, T1b = read_trexio_unrestricted_singles(io)
  T2a, T2b, T2ab = read_trexio_unrestricted_doubles(io)
  return (T1a, T1b, T2a, T2b, T2ab)
end

"""
    has_amplitudes([io::TrexioFile,] EC::ECInfo; unrestricted=false, start=false)

  Check if amplitudes are stored in the dump file.
  If `start=true`, checks the start file (`wf.start`) first.

  Returns `true` if amplitudes (singles or doubles) are found.
"""
function has_amplitudes end
function has_amplitudes(EC::ECInfo; unrestricted::Bool=false, start::Bool=false)
  filename, full_filename = dumpfile(EC, "r"; start=start)
  if !isfile(full_filename)
    return false
  end
  open_dump(EC, "r"; start=start) do io
    return has_amplitudes(io, EC; unrestricted=unrestricted)
  end
end
function has_amplitudes(io::TrexioFile, EC::ECInfo; unrestricted::Bool=false)
  return has_trexio_amplitudes(io; unrestricted=unrestricted)
end

end #module