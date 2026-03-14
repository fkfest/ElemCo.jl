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
export transfer_orbitals_to_store!, OrbitalData, rotate_orbitaldata!, fetch_orbital_data
# Determinant I/O for CIPHI
export dump_determinants, fetch_determinants, has_determinants
export dump_determinants_multistate, state_filename

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
    # Register the file in EC.files
    add_file!(EC, filename, "TREXIO wavefunction dump"; overwrite=true)
  elseif intent == "u"
    mode = "updating (unsafe mode)"
  end
  println("Opening dump file $filename for $mode ...")
  return open_trexio(full_filename, intent)
end

"""
    open_dump(f, EC::ECInfo, intent; start=false)

  Open the dump file for wavefunction, execute function `f` with the opened `TrexioFile`, and close the file.

To be used as `open_dump(EC, intent) do io ... end`.
`intent` can be "r", "w", or "u" (read, write, or update).
If `start=true`, opens the start file (`wf.start`) instead.

Note: `f` is typed as a type parameter rather than `::Function` to enable
type inference of the return value through the closure.
"""
function open_dump(f::F, EC::ECInfo, intent; start::Bool=false) where {F}
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
function prepare_orb_vectors(input::Vector{ComplexF64}, restricted)
  return prepare_orb_vectors(real.(input), restricted)
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
function prepare_orb_vectors(input::Vector{Vector{ComplexF64}}, restricted)
  return prepare_orb_vectors([real.(v) for v in input], restricted)
end

function prepare_orb_classes(EC::ECInfo, restricted)
  space_save, space_b4freeze = restore_full_space!(EC)
  
  # Now EC.space has the frozen configuration
  classa = fill("Deleted", length(EC.space['m']))
  classa[EC.space['o']] .= "Inactive"
  classa[EC.space['v']] .= "Virtual"
  classa[EC.space['a']] .= "Active"
  # Mark frozen core: compare with space before freezing to find removed occupied orbitals
  frozen_occ = setdiff(space_b4freeze['o'], EC.space['o'])
  classa[frozen_occ] .= "Core"
    
  if restricted
    classb = String[]
  else
    classb = fill("Deleted", length(EC.space['M']))
    classb[EC.space['O']] .= "Inactive"
    classb[EC.space['V']] .= "Virtual"
    classb[EC.space['a']] .= "Active"
    # Mark frozen core: compare with space before freezing to find removed occupied orbitals
    frozen_occ = setdiff(space_b4freeze['O'], EC.space['O'])
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
  cMO::SpinMatrix{Float64}
  mo_type::String
  basis::BasisSet
  energies::NTuple{2,Vector{Float64}}
  occupations::NTuple{2,Vector{Float64}}
end

OrbitalData(cMO::SpinMatrix{Float64}) = OrbitalData(cMO, "Rotation", BasisSet(), (Float64[], Float64[]), (Float64[], Float64[]))

"""
    rotate_orbitaldata!(orbital_data::OrbitalData, Rpq_a::Matrix{Float64}, Rpq_b::Union{Matrix{Float64}, Nothing}=nothing) -> OrbitalData
  
  Rotate the orbitals in `orbital_data` using the provided rotation matrices `Rpq_a` and `Rpq_b`.

- If `Rpq_b` is `nothing` and the orbitals are restricted, only the alpha orbitals will be rotated, and the beta orbitals will follow the same rotation.
- If `Rpq_b` is `nothing` but the orbitals are unrestricted, the same rotation will be applied to both alpha and beta orbitals.
- If `Rpq_b` is provided and the orbitals are restricted, the beta orbitals will be rotated with `Rpq_b` if it is different from `Rpq_a`, effectively making the orbitals unrestricted.
- If `Rpq_b` is provided and the orbitals are already unrestricted, the beta orbitals will be rotated with `Rpq_b`.
"""
function rotate_orbitaldata!(orbital_data::OrbitalData, Rpq_a::Matrix{Float64}, Rpq_b::Union{Matrix{Float64}, Nothing}=nothing)
  if isnothing(Rpq_b) && is_restricted(orbital_data.cMO)
    # beta orbitals will be rotated automatically with the same rotation as alpha
  elseif isnothing(Rpq_b)
    # If we have unrestricted orbitals but only one rotation, apply the same rotation to beta
    orbital_data.cMO.β .= orbital_data.cMO.β * Rpq_a
  elseif is_restricted(orbital_data.cMO)
    # Make orbitals unrestricted if different rotations are provided
    if Rpq_a !== Rpq_b
      orbital_data.cMO.β = orbital_data.cMO.β * Rpq_b
    end
  else
    # Unrestricted orbitals with separate rotations
    orbital_data.cMO.β .= orbital_data.cMO.β * Rpq_b
  end

  orbital_data.cMO.α .= orbital_data.cMO.α * Rpq_a
  return orbital_data
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

  **Important**: `orbital_data` must be
  pre-fetched before opening the store file to avoid reading from a truncated file.
  
  This properly handles cases where frozen orbitals or geometry differ
  between dump and store files.
  
  For FCIDUMP-only calculations (no dump file), stores a unity rotation matrix
  instead of orbitals, allowing amplitude storage and restart.

# Arguments
- `io_store`: An already opened TrexioFile for writing
- `EC`: Electronic structure information object
- `orbital_data`: Pre-fetched orbital data, or `nothing` to store unity rotation
- `MO`: "mo" for molecular orbitals or "po" for positron orbitals
"""
function transfer_orbitals_to_store!(io_store::TrexioFile, EC::ECInfo, 
                                     orbital_data::OrbitalData; MO="mo")
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
    classes = prepare_orb_classes(EC, is_restricted(cMO))
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
  return
end

function transfer_orbitals_to_store!(io_store::TrexioFile, EC::ECInfo, 
                                     orbital_data::Nothing=nothing; MO="mo")
  # If no orbital data was pre-fetched, this is a FCIDUMP-only case.
  # Create a unity rotation for amplitude storage.
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
    load_wavefunction(EC::ECInfo, what::Vector{String}; start=false, state=1, OPattern=UInt64)

  Load parts of the wavefunction from file [`WfOptions.dump`](@ref ECInfos.WfOptions).

  `what` can contain any of the following strings:
  - `"all"`: load everything available (overrides other options)
  - `"orbitals"`: load orbitals (always loaded by default)
  - `"orbital_energies"`: load orbital energies
  - `"orbital_occupations"`: load orbital occupations
  - `"amplitudes"`: load CC amplitudes (restricted)
  - `"unrestricted_amplitudes"`: load CC amplitudes (unrestricted)
  - `"determinants"`: load selected CI determinants and coefficients

# Arguments
- `what::Vector{String}`: List of wavefunction parts to load
- `start::Bool=false`: If true, read from `wf.start` file instead of `wf.dump`
- `state::Int=1`: State number for determinants (1 = ground state)
- `OPattern::Type=UInt64`: Orbital pattern type for determinants (use UInt128 for >64 orbitals)

# Returns
A `Dict{String,Any}` with the requested parts of the wavefunction:
- `"orbitals"`: `SpinMatrix` of molecular orbitals
- `"orbital_type"`: Type string (e.g., "RHF", "UHF")
- `"basis"`: `BasisSet` (empty if rotation)
- `"orbital_energies"`: Tuple of (alpha, beta) energies
- `"orbital_occupations"`: Tuple of (alpha, beta) occupations
- `"T1"`, `"T2"`: Restricted amplitudes
- `"T1a"`, `"T1b"`, `"T2a"`, `"T2b"`, `"T2ab"`: Unrestricted amplitudes
- `"determinants"`: Vector of determinants
- `"ci_coefficients"`: Vector of CI coefficients

# Example
```julia
# Load orbitals and amplitudes
wf = load_wavefunction(EC, ["orbitals", "amplitudes"])

# Load everything
wf = load_wavefunction(EC, ["all"])

# Load determinants for excited state
wf = load_wavefunction(EC, ["determinants"]; state=2)
```
"""
function load_wavefunction(EC::ECInfo, what::Vector{String}; 
                           start::Bool=false, state::Int=1, OPattern::Type=UInt64)
  wf = Dict{String,Any}()
  all = "all" in what
  
  # Load orbitals (always loaded)
  open_dump(EC, "r"; start=start) do io
    wf["orbitals"], wf["orbital_type"], wf["basis"] = fetch_orbitals(io, EC)
    if all || "orbital_energies" in what
      wf["orbital_energies"] = fetch_orbital_energies(io, EC)
    end
    if all || "orbital_occupations" in what
      wf["orbital_occupations"] = fetch_orbital_occupations(io, EC)
    end
    # Load amplitudes
    if all || "amplitudes" in what
      if has_amplitudes(io, EC; unrestricted=false)
        T1, T2 = fetch_restricted_amplitudes(io, EC)
        if !isempty(T1)
          wf["T1"] = T1
        end
        if !isempty(T2)
          wf["T2"] = T2
        end
      end
    end
    if all || "unrestricted_amplitudes" in what
      if has_amplitudes(io, EC; unrestricted=true)
        T1a, T1b, T2a, T2b, T2ab = fetch_unrestricted_amplitudes(io, EC)
        if !isempty(T1a)
          wf["T1a"] = T1a
        end
        if !isempty(T1b)
          wf["T1b"] = T1b
        end
        if !isempty(T2a)
          wf["T2a"] = T2a
        end
        if !isempty(T2b)
          wf["T2b"] = T2b
        end
        if !isempty(T2ab)
          wf["T2ab"] = T2ab
        end
      end
    end
  end
  
  # Load determinants (stored in separate state-specific files)
  if all || "determinants" in what
    dets, coeffs = fetch_determinants(EC; start=start, OPattern=OPattern, state=state)
    if !isempty(dets)
      wf["determinants"] = dets
      wf["ci_coefficients"] = coeffs
    end
  end
  
  return wf
end

"""
    save_wavefunction(EC::ECInfo, wf::AbstractDict; state=1)

  Save parts of the wavefunction to file [`WfOptions.store`](@ref ECInfos.WfOptions) or 
  [`WfOptions.dump`](@ref ECInfos.WfOptions) (if `store` is empty).

  `wf` can contain any of the following keys:

**Orbital data:**
- `"basis"`: basis set information
- `"orbitals"`: molecular orbitals (`SpinMatrix`)
- `"rotations"`: orbital rotations (`SpinMatrix`) - alternative to `"orbitals"`
- `"orbital_type"`: type of the orbitals (e.g., "RHF", "UHF", "ROHF", "MCSCF")
- `"orbital_energies"`: molecular orbital energies
- `"orbital_occupations"`: molecular orbital occupations

**Restricted CC amplitudes:**
- `"T1"`: singles amplitudes (nvirt × nocc)
- `"T2"`: doubles amplitudes (nvirt × nvirt × nocc × nocc)

**Unrestricted CC amplitudes:**
- `"T1a"`, `"T1b"`: α and β singles amplitudes
- `"T2a"`, `"T2b"`, `"T2ab"`: αα, ββ, and αβ doubles amplitudes

**Selected CI (CIPHI) data:**
- `"determinants"`: vector of determinants
- `"ci_coefficients"`: CI coefficients (vector for single state, matrix for multi-state)

# Arguments
- `wf::AbstractDict`: Dictionary containing wavefunction data
- `state::Int=1`: State number for determinants (used when `ci_coefficients` is a vector)

# Example
```julia
# Save orbitals and amplitudes
save_wavefunction(EC, Dict(
    "orbitals" => cMO,
    "orbital_type" => "RHF",
    "T1" => T1,
    "T2" => T2
))

# Save determinants for ground state
save_wavefunction(EC, Dict(
    "determinants" => dets,
    "ci_coefficients" => coeffs
); state=1)

# Save multi-state determinants (each column is a state)
save_wavefunction(EC, Dict(
    "determinants" => dets,
    "ci_coefficients" => coeffs_matrix  # n_dets × n_states
))
```
"""
function save_wavefunction(EC::ECInfo, wf::AbstractDict; state::Int=1)
  # Save orbitals/rotations and amplitudes to the main dump file
  has_orbitals = haskey(wf, "orbitals") || haskey(wf, "rotations")
  has_amplitudes = haskey(wf, "T1") || haskey(wf, "T2") || 
                   haskey(wf, "T1a") || haskey(wf, "T1b") ||
                   haskey(wf, "T2a") || haskey(wf, "T2b") || haskey(wf, "T2ab")
  
  if has_orbitals
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
    end
  end
  
  # Save amplitudes (requires update mode if orbitals were written)
  if has_amplitudes
    mode = has_orbitals ? "u" : "w"
    open_dump(EC, mode) do io
      # Transfer orbitals if we didn't write them above but need them for amplitudes
      if !has_orbitals
        orbital_data = fetch_orbital_data(EC)
        transfer_orbitals_to_store!(io, EC, orbital_data)
      end
      
      # Restricted amplitudes
      if haskey(wf, "T1") || haskey(wf, "T2")
        T1 = get(wf, "T1", zeros(0, 0))
        T2 = get(wf, "T2", zeros(0, 0, 0, 0))
        dump_amplitudes(io, EC, T1, T2)
      end
      
      # Unrestricted amplitudes
      if haskey(wf, "T1a") || haskey(wf, "T1b") || 
         haskey(wf, "T2a") || haskey(wf, "T2b") || haskey(wf, "T2ab")
        T1a = get(wf, "T1a", zeros(0, 0))
        T1b = get(wf, "T1b", zeros(0, 0))
        T2a = get(wf, "T2a", zeros(0, 0, 0, 0))
        T2b = get(wf, "T2b", zeros(0, 0, 0, 0))
        T2ab = get(wf, "T2ab", zeros(0, 0, 0, 0))
        dump_amplitudes(io, EC, T1a, T1b, T2a, T2b, T2ab)
      end
    end
  end
  
  # Save determinants (stored in separate state-specific files)
  if haskey(wf, "determinants") && haskey(wf, "ci_coefficients")
    dets = wf["determinants"]
    coeffs = wf["ci_coefficients"]
    
    if coeffs isa AbstractMatrix
      # Multi-state: each column is a state
      dump_determinants_multistate(EC, dets, coeffs)
    else
      # Single state
      dump_determinants(EC, dets, coeffs; state=state)
    end
  end
  
  return
end

"""
    copy_wavefunction(EC::ECInfo, tofile::AbstractString=""; start=false, state=0)

  Copy the wavefunction dump file to `tofile`. If `tofile` is not given, copy to the current dump file for writing.

# Arguments
- `tofile::AbstractString=""`: Destination file path. If empty, copies to `wf.store`.
- `start::Bool=false`: If true, copy from `wf.start` file instead of `wf.dump`.
- `state::Int=0`: State number for determinant files. If 0, copies the main dump file.
                   If >0, copies the state-specific determinant file (e.g., `file_state2.h5`).

  Note: This does not check the contents of the files.

# Examples
```julia
# Copy current dump to store
copy_wavefunction(EC)

# Copy start file to a backup
copy_wavefunction(EC, "backup.h5"; start=true)

# Copy determinant file for state 2
copy_wavefunction(EC, "state2_backup.h5"; state=2)
```
"""
function copy_wavefunction(EC::ECInfo, tofile::AbstractString=""; start::Bool=false, state::Int=0)
  if state > 0
    # Copy state-specific determinant file
    base_filename, _ = dumpfile(EC, "r"; start=start)
    from_filename = state_filename(base_filename, state)
    from_fullpath = joinpath(EC.scr, from_filename)
    
    if !isfile(from_fullpath)
      @warn "State file $from_filename not found, nothing to copy."
      return
    end
    
    if tofile == ""
      # Default: copy to store location with state suffix
      base_store, _ = dumpfile(EC, "w")
      to_fullpath = joinpath(EC.scr, state_filename(base_store, state))
    else
      to_fullpath = tofile
    end
  else
    # Copy main dump file
    from_fullpath = dumpfile(EC, "r"; start=start)[2]
    to_fullpath = tofile == "" ? dumpfile(EC, "w")[2] : tofile
  end
  
  cp(from_fullpath, to_fullpath; force=true)
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

# ============================================================================
# Determinant I/O for CIPHI wave functions
# ============================================================================

"""
    state_filename(filename::String, state::Int) -> String

Generate state-specific filename for multi-state storage.
Per TREXIO standard, each state is stored in a separate file:
- State 1 (ground): `filename.h5`
- State 2: `filename_state2.h5`
- State n: `filename_state{n}.h5`
"""
function state_filename(filename::String, state::Int)
  state == 1 && return filename
  base, ext = splitext(filename)
  return "$(base)_state$(state)$(ext)"
end

"""
    dump_determinants([io::TrexioFile,] EC::ECInfo, dets, coeffs; state=1)

Dump CIPHI determinants and CI coefficients to TREXIO file.

For multi-state calculations, each state should be stored separately using
the `state` parameter. State 1 goes to the main file, state n goes to
`filename_state{n}.h5`.

# Arguments
- `dets::Vector{<:AbstractDeterminant}`: Determinants with alpha/beta occupation patterns
- `coeffs::AbstractVector{Float64}`: CI coefficients for this state
- `state::Int=1`: State number (1 = ground state)

# Example
```julia
# Store ground state
dump_determinants(EC, dets, coeffs[:, 1]; state=1)
# Store excited state
dump_determinants(EC, dets, coeffs[:, 2]; state=2)
```
"""
function dump_determinants end

function dump_determinants(EC::ECInfo, dets::Vector{D}, coeffs::AbstractVector{Float64}; 
                           state::Int=1) where {D}
  if EC.options.wf.store == ""
    return
  end
  filename = state_filename(EC.options.wf.store, state)
  full_filename = joinpath(EC.scr, filename)
  println("Storing determinants (state $state) to $filename ...")
  
  # Register the file in EC.files
  add_file!(EC, filename, "TREXIO CIPHI determinants state $state"; overwrite=true)
  
  # Pre-fetch orbital data BEFORE opening store file for writing
  orbital_data = fetch_orbital_data(EC)
  
  open_trexio(full_filename, "w") do io
    transfer_orbitals_to_store!(io, EC, orbital_data)
    dump_determinants(io, dets, coeffs)
  end
  return
end

function dump_determinants(io::TrexioFile, dets::Vector{D}, 
                           coeffs::AbstractVector{Float64}) where {D}
  write_trexio_determinants(io, dets, coeffs)
  return
end

"""
    fetch_determinants([io::TrexioFile,] EC::ECInfo; start=false, OPattern=UInt64, state=1)

Fetch CIPHI determinants and CI coefficients from TREXIO file.

# Arguments
- `start::Bool=false`: If true, read from `wf.start` file instead of `wf.dump`
- `OPattern::Type=UInt64`: Type for orbital patterns (use UInt128 for >64 orbitals)
- `state::Int=1`: State number to read

# Returns
- `(determinants, coefficients)`: Tuple of determinant vector and coefficient vector

# Example
```julia
dets, coeffs = fetch_determinants(EC; state=1)
# For systems with >64 orbitals:
dets, coeffs = fetch_determinants(EC; OPattern=UInt128)
```
"""
function fetch_determinants end

function fetch_determinants(EC::ECInfo; start::Bool=false, OPattern::Type=UInt64, state::Int=1)
  base_filename, base_full = dumpfile(EC, "r"; start=start)
  filename = state_filename(base_filename, state)
  full_filename = joinpath(EC.scr, filename)
  
  if !isfile(full_filename)
    println("Determinant file $filename not found.")
    return SimpleDeterminant{OPattern}[], Float64[]
  end
  
  println("Fetching determinants (state $state) from $filename ...")
  open_trexio(full_filename, "r") do io
    return fetch_determinants(io; OPattern=OPattern)
  end
end

function fetch_determinants(io::TrexioFile; OPattern::Type=UInt64)
  return read_trexio_determinants(io; OPattern=OPattern)
end

"""
    has_determinants([io::TrexioFile,] EC::ECInfo; start=false, state=1)

Check if determinants are stored in the TREXIO file.

# Arguments
- `start::Bool=false`: If true, check `wf.start` file instead of `wf.dump`
- `state::Int=1`: State number to check

# Returns
`true` if determinants are found in the file.
"""
function has_determinants end

function has_determinants(EC::ECInfo; start::Bool=false, state::Int=1)
  base_filename, _ = dumpfile(EC, "r"; start=start)
  filename = state_filename(base_filename, state)
  full_filename = joinpath(EC.scr, filename)
  
  if !isfile(full_filename)
    return false
  end
  
  open_trexio(full_filename, "r") do io
    return has_determinants(io)
  end
end

function has_determinants(io::TrexioFile)
  return has_trexio_determinants(io)
end

"""
    dump_determinants_multistate(EC::ECInfo, dets, coeffs_matrix)

Dump determinants and coefficients for multiple states to separate files.

# Arguments
- `EC::ECInfo`: Electronic structure information
- `dets::Vector{<:AbstractDeterminant}`: Determinants (same for all states)
- `coeffs_matrix::Matrix{Float64}`: CI coefficients matrix (n_dets × n_states)

Each state is stored in a separate file per TREXIO standard.
"""
function dump_determinants_multistate(EC::ECInfo, dets::Vector{D}, 
                                      coeffs_matrix::AbstractMatrix{Float64}) where {D}
  nstates = size(coeffs_matrix, 2)
  for state in 1:nstates
    dump_determinants(EC, dets, coeffs_matrix[:, state]; state=state)
  end
  return
end

end #module