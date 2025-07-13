"""
TREX Interface Module

This module provides functions to import and export data in TREX format,
which is a standardized format for quantum chemistry data exchange.

TREX (Table of Results Exchange) format specification:
- Based on HDF5 for efficient storage
- Standardized structure for quantum chemistry data
- Supports orbitals, amplitudes, integrals, and other QC data

See: https://trex-coe.github.io/trexio/lib.html
     https://arxiv.org/abs/2302.14793
"""
module TrexInterface

using HDF5
using Dates
using ..ElemCo.ECInfos
using ..ElemCo.MSystems
using ..ElemCo.Utils
using ..ElemCo.QMTensors
using ..ElemCo.Elements
using LinearAlgebra

export TrexFile, write_trex, read_trex
export write_trex_orbitals, read_trex_orbitals
export write_trex_amplitudes, read_trex_amplitudes
export write_trex_molecule, read_trex_molecule
export write_trex_basis, read_trex_basis
export open_trex, close_trex

"""
    TrexFile

Structure representing a TREX format file.
Contains the HDF5 file handle and metadata.
"""
mutable struct TrexFile
    filename::String
    file::Union{HDF5.File, Nothing}
    mode::String  # "r", "w", "r+"
    
    function TrexFile(filename::String, mode::String="r")
        new(filename, nothing, mode)
    end
end

"""
    open_trex(trex::TrexFile)

Open a TREX file for reading or writing.
"""
function open_trex(trex::TrexFile)
    if trex.file === nothing
        trex.file = h5open(trex.filename, trex.mode)
        # Ensure the main TREX group exists
        if trex.mode in ["w", "r+"] && !haskey(trex.file, "trex")
            create_group(trex.file, "trex")
        end
    end
    return trex.file
end

"""
    close_trex(trex::TrexFile)

Close a TREX file.
"""
function close_trex(trex::TrexFile)
    if trex.file !== nothing
        close(trex.file)
        trex.file = nothing
    end
end

"""
    write_trex_metadata(file::HDF5.File)

Write TREX format metadata to the file.
"""
function write_trex_metadata(file::HDF5.File)
    trex_group = file["trex"]
    
    # Write format version and metadata
    attrs(trex_group)["format_version"] = "2.4.0"
    attrs(trex_group)["created_by"] = "ElemCo.jl"
    attrs(trex_group)["created_date"] = string(Dates.now())
    
    return trex_group
end

"""
    write_trex_molecule(trex::TrexFile, system::MSystem)

Write molecular geometry and basis set information to TREX format.
"""
function write_trex_molecule(trex::TrexFile, system::MSystem)
    file = open_trex(trex)
    trex_group = haskey(file, "trex") ? file["trex"] : write_trex_metadata(file)
    
    # Create nucleus group for atomic information
    if haskey(trex_group, "nucleus")
        delete_object(trex_group, "nucleus")
    end
    nucleus_group = create_group(trex_group, "nucleus")
    
    # Extract atomic information
    natoms = length(system)
    nuclear_charges = Float64[]
    coordinates = zeros(Float64, 3, natoms)
    labels = String[]
    
    for (i, atom) in enumerate(system)
        push!(nuclear_charges, Float64(atom.atomic_number))
        coordinates[:, i] = Vector(atom.position)
        push!(labels, atom.label)
    end
    
    # Write nuclear data
    nucleus_group["num"] = Int64(natoms)
    nucleus_group["charge"] = nuclear_charges
    nucleus_group["coord"] = coordinates
    nucleus_group["label"] = labels
    
    # Add units information
    attrs(nucleus_group["coord"])["units"] = "bohr"
    
    return nucleus_group
end

"""
    read_trex_molecule(trex::TrexFile) -> MSystem

Read molecular geometry from TREX format and return MSystem.
"""
function read_trex_molecule(trex::TrexFile)
    file = open_trex(trex)
    
    if !haskey(file, "trex") || !haskey(file["trex"], "nucleus")
        error("No molecular data found in TREX file")
    end
    
    nucleus_group = file["trex"]["nucleus"]
    
    # Read atomic data
    natoms = read(nucleus_group["num"])
    charges = read(nucleus_group["charge"])
    coords = read(nucleus_group["coord"])
    labels = read(nucleus_group["label"])
    
    # Create atoms
    atoms = ACentre[]
    for i in 1:natoms
        # Convert charge to element symbol
        element = element_symbol(Int(charges[i]))
        pos = coords[:, i]
        
        # Create basic basis (this would need to be enhanced for real use)
        basis = Dict{String,String}()
        
        atom = ACentre(labels[i], pos[1], pos[2], pos[3], Int(charges[i]), charges[i], basis)
        push!(atoms, atom)
    end
    
    return MSystem(atoms)
end

"""
    write_trex_basis(trex::TrexFile, system::MSystem)

Write basis set information to TREX format.
"""
function write_trex_basis(trex::TrexFile, system::MSystem)
    file = open_trex(trex)
    trex_group = haskey(file, "trex") ? file["trex"] : write_trex_metadata(file)
    
    # Create basis group for basis set information
    if haskey(trex_group, "basis")
        delete_object(trex_group, "basis")
    end
    basis_group = create_group(trex_group, "basis")
    
    # Extract basis set information from atomic centres
    nbasis = length(system)
    basis_names = String[]
    atom_indices = Int[]
    
    # Collect unique basis sets and their associations with atoms
    for (i, atom) in enumerate(system)
        # Get the primary basis set (usually "ao")
        if haskey(atom.basis, "ao")
            push!(basis_names, atom.basis["ao"])
            push!(atom_indices, i)
        elseif !isempty(atom.basis)
            # Use the first available basis set if "ao" is not present
            first_basis = first(values(atom.basis))
            push!(basis_names, first_basis)
            push!(atom_indices, i)
        else
            # No basis set specified for this atom
            push!(basis_names, "")
            push!(atom_indices, i)
        end
    end
    
    # Write basis set data
    basis_group["num"] = Int64(nbasis)
    basis_group["nucleus_index"] = atom_indices
    basis_group["type"] = basis_names
    
    # Store additional basis set types if available
    all_basis_types = Set{String}()
    for atom in system
        for basis_type in keys(atom.basis)
            push!(all_basis_types, basis_type)
        end
    end
    
    if length(all_basis_types) > 1
        # Store all basis types as attributes
        attrs(basis_group)["available_types"] = collect(all_basis_types)
    end
    
    # Add metadata
    attrs(basis_group)["description"] = "Basis set information for molecular orbitals"
    
    return basis_group
end

"""
    read_trex_basis(trex::TrexFile) -> Dict{String, Any}

Read basis set information from TREX format.
"""
function read_trex_basis(trex::TrexFile)
    file = open_trex(trex)
    
    if !haskey(file, "trex") || !haskey(file["trex"], "basis")
        error("No basis set data found in TREX file")
    end
    
    basis_group = file["trex"]["basis"]
    
    # Read basis set data
    nbasis = read(basis_group["num"])
    atom_indices = read(basis_group["nucleus_index"])
    basis_types = read(basis_group["type"])
    
    basis_info = Dict{String, Any}(
        "num" => nbasis,
        "nucleus_index" => atom_indices,
        "type" => basis_types
    )
    
    # Read additional attributes if available
    if haskey(attrs(basis_group), "available_types")
        basis_info["available_types"] = read(attrs(basis_group)["available_types"])
    end
    
    return basis_info
end

"""
    write_trex_orbitals(trex::TrexFile, orbitals::AbstractMatrix; 
                       orbital_type="molecular", spin="restricted", system=nothing)

Write molecular orbitals to TREX format. If system is provided, 
basis set information will also be written.
"""
function write_trex_orbitals(trex::TrexFile, orbitals::AbstractMatrix; 
                            orbital_type="molecular", spin="restricted", system=nothing)
    file = open_trex(trex)
    trex_group = haskey(file, "trex") ? file["trex"] : write_trex_metadata(file)
    
    # Write basis set information if system is provided and not already written
    if !isnothing(system) && !haskey(trex_group, "basis")
        write_trex_basis(trex, system)
    end
    
    # Create MO group
    if haskey(trex_group, "mo")
        delete_object(trex_group, "mo")
    end
    mo_group = create_group(trex_group, "mo")
    
    # Write orbital information
    nmo, nbasis = size(orbitals)
    mo_group["num"] = Int64(nmo)
    mo_group["coefficient"] = orbitals
    
    # Add metadata
    attrs(mo_group)["type"] = orbital_type
    attrs(mo_group)["spin"] = spin
    attrs(mo_group)["basis_size"] = Int64(nbasis)
    
    return mo_group
end

"""
    read_trex_orbitals(trex::TrexFile) -> AbstractMatrix

Read molecular orbitals from TREX format.
"""
function read_trex_orbitals(trex::TrexFile)
    file = open_trex(trex)
    
    if !haskey(file, "trex") || !haskey(file["trex"], "mo")
        error("No orbital data found in TREX file")
    end
    
    mo_group = file["trex"]["mo"]
    return read(mo_group["coefficient"])
end

"""
    write_trex_amplitudes(trex::TrexFile, amplitudes::Dict{String, Any})

Write CC amplitudes to TREX format.
"""
function write_trex_amplitudes(trex::TrexFile, amplitudes::Dict{String, Any})
    file = open_trex(trex)
    trex_group = haskey(file, "trex") ? file["trex"] : write_trex_metadata(file)
    
    # Create amplitudes group
    if haskey(trex_group, "amplitudes")
        delete_object(trex_group, "amplitudes")
    end
    amp_group = create_group(trex_group, "amplitudes")
    
    # Write each amplitude tensor
    for (key, value) in amplitudes
        if isa(value, AbstractArray)
            amp_group[key] = value
            attrs(amp_group[key])["tensor_rank"] = length(size(value))
            attrs(amp_group[key])["dimensions"] = collect(size(value))
        end
    end
    
    return amp_group
end

"""
    read_trex_amplitudes(trex::TrexFile) -> Dict{String, Any}

Read CC amplitudes from TREX format.
"""
function read_trex_amplitudes(trex::TrexFile)
    file = open_trex(trex)
    
    if !haskey(file, "trex") || !haskey(file["trex"], "amplitudes")
        error("No amplitude data found in TREX file")
    end
    
    amp_group = file["trex"]["amplitudes"]
    amplitudes = Dict{String, Any}()
    
    for key in keys(amp_group)
        amplitudes[key] = read(amp_group[key])
    end
    
    return amplitudes
end

"""
    write_trex(filename::String, EC::ECInfo; kwargs...)

Write ElemCo data to TREX format file. When orbitals are included,
basis set information is automatically written as well.

# Keyword arguments
- `include_orbitals::Bool=true`: Include molecular orbitals and basis sets
- `include_amplitudes::Bool=false`: Include CC amplitudes  
- `include_molecule::Bool=true`: Include molecular geometry
"""
function write_trex(filename::String, EC::ECInfo; 
                   include_orbitals::Bool=true,
                   include_amplitudes::Bool=false,
                   include_molecule::Bool=true)
    
    trex = TrexFile(filename, "w")
    
    try
        # Write molecular information
        if include_molecule && !isnothing(EC.system)
            write_trex_molecule(trex, EC.system)
        end
        
        # Write orbitals if available
        if include_orbitals
            try
                # Check if orbitals file exists
                orb_file = fullfilename(EC, EC.options.wf.orb)
                if file_exists(EC, EC.options.wf.orb)
                    orbs = load(EC, EC.options.wf.orb)
                    if !isnothing(orbs)
                        # Pass system information to include basis set data
                        write_trex_orbitals(trex, orbs, system=EC.system)
                    end
                else
                    @warn "Orbital file $(EC.options.wf.orb) not found, skipping orbital export"
                end
            catch e
                @warn "Could not load orbitals for TREX export: $e"
            end
        end
        
        # Write amplitudes if requested and available
        if include_amplitudes
            # This would need to be implemented based on how amplitudes are stored in ElemCo
            @warn "Amplitude export not yet implemented"
        end
        
    finally
        close_trex(trex)
    end
    
    return filename
end

"""
    read_trex(filename::String) -> Dict{String, Any}

Read data from TREX format file.

Returns a dictionary with available data sections.
"""
function read_trex(filename::String)
    if !isfile(filename)
        error("TREX file not found: $filename")
    end
    
    trex = TrexFile(filename, "r")
    data = Dict{String, Any}()
    
    try
        file = open_trex(trex)
        
        if !haskey(file, "trex")
            error("Invalid TREX file format")
        end
        
        trex_group = file["trex"]
        
        # Read available sections
        if haskey(trex_group, "nucleus")
            data["molecule"] = read_trex_molecule(trex)
        end
        
        if haskey(trex_group, "basis")
            data["basis"] = read_trex_basis(trex)
        end
        
        if haskey(trex_group, "mo")
            data["orbitals"] = read_trex_orbitals(trex)
        end
        
        if haskey(trex_group, "amplitudes")
            data["amplitudes"] = read_trex_amplitudes(trex)
        end
        
    finally
        close_trex(trex)
    end
    
    return data
end

# Utility function to get element symbol from atomic number
function element_symbol(z::Int)
    # Use the Elements module if available, otherwise fall back to simple list
    try
        for (symbol, (atomic_num, _, _, _, _, _)) in ELEMENTS
            if atomic_num == z
                return symbol
            end
        end
    catch
        # Fallback to simple list
        elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
        return z <= length(elements) ? elements[z] : "X$z"
    end
    return "X$z"
end

end # module TrexInterface