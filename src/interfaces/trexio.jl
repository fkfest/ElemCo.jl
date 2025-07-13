"""
TREXIOIO Interface Module

This module provides functions to import and export data in TREXIOIO format,
which is a standardized format for quantum chemistry data exchange.

TREXIOIO (Table of Results Exchange Input/Output) format specification:
- Based on HDF5 for efficient storage
- Standardized structure for quantum chemistry data
- Supports orbitals, amplitudes, integrals, and other QC data

See: https://trexio-coe.github.io/trexio/lib.html
     https://arxiv.org/abs/2302.14793
"""
module TrexioInterface

using HDF5
using Dates
using ..ElemCo.ECInfos
using ..ElemCo.MSystems
using ..ElemCo.Utils
using ..ElemCo.QMTensors
using ..ElemCo.Elements
using ..ElemCo.OrbTools
using LinearAlgebra

export TrexioFile, write_trexio, read_trexio
export write_trexio_orbitals, read_trexio_orbitals
export write_trexio_amplitudes, read_trexio_amplitudes
export write_trexio_molecule, read_trexio_molecule
export write_trexio_basis, read_trexio_basis
export open_trexio, close_trexio

"""
    TrexioFile

Structure representing a TREXIOIO format file.
Contains the HDF5 file handle and metadata.
"""
mutable struct TrexioFile
    filename::String
    file::Union{HDF5.File, Nothing}
    mode::String  # "r", "w", "r+"
    
    function TrexioFile(filename::String, mode::String="r")
        new(filename, nothing, mode)
    end
end

"""
    open_trexio(trexio::TrexioFile)

Open a TREXIO file for reading or writing.
"""
function open_trexio(trexio::TrexioFile)
    if trexio.file === nothing
        trexio.file = h5open(trexio.filename, trexio.mode)
        # Ensure the main TREXIO group exists
        if trexio.mode in ["w", "r+"] && !haskey(trexio.file, "trexio")
            create_group(trexio.file, "trexio")
        end
    end
    return trexio.file
end

"""
    close_trexio(trexio::TrexioFile)

Close a TREXIO file.
"""
function close_trexio(trexio::TrexioFile)
    if trexio.file !== nothing
        close(trexio.file)
        trexio.file = nothing
    end
end

"""
    write_trexio_metadata(file::HDF5.File)

Write TREXIO format metadata to the file.
"""
function write_trexio_metadata(file::HDF5.File)
    trex_group = file["trexio"]
    
    # Write format version and metadata
    attrs(trex_group)["format_version"] = "2.4.0"
    attrs(trex_group)["created_by"] = "ElemCo.jl"
    attrs(trex_group)["created_date"] = string(Dates.now())
    
    return trex_group
end

"""
    write_trexio_molecule(trexio::TrexioFile, system::MSystem)

Write molecular geometry and basis set information to TREXIO format.
"""
function write_trexio_molecule(trexio::TrexioFile, system::MSystem)
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_trexio_metadata(file)
    
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
    read_trexio_molecule(trexio::TrexioFile) -> MSystem

Read molecular geometry from TREXIO format and return MSystem.
"""
function read_trexio_molecule(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "nucleus")
        error("No molecular data found in TREXIO file")
    end
    
    nucleus_group = file["trexio"]["nucleus"]
    
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
    write_trexio_basis(trexio::TrexioFile, system::MSystem; basisset=nothing)

Write basis set information to TREXIO format following TREXIOIO standard.
If basisset is provided, detailed shell information is stored.
Otherwise, only basic basis set names are stored.
"""
function write_trexio_basis(trexio::TrexioFile, system::MSystem; basisset=nothing)
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_trexio_metadata(file)
    
    # Create basis group for basis set information
    if haskey(trex_group, "basis")
        delete_object(trex_group, "basis")
    end
    basis_group = create_group(trex_group, "basis")
    
    if !isnothing(basisset)
        # Write detailed TREXIOIO-compliant basis set information
        _write_detailed_basis_trexio(basis_group, basisset)
    else
        # Fallback: write basic basis set names (original implementation)
        _write_basic_basis_names(basis_group, system)
    end
    
    # Add metadata
    attrs(basis_group)["description"] = "Basis set information following TREXIOIO standard"
    
    return basis_group
end

"""
Write detailed basis set information following TREXIOIO standard
"""
function _write_detailed_basis_trexio(basis_group, basisset)
    # Collect shell information following TREXIOIO standard
    shell_num = length(basisset)
    shell_nucleus_index = Int[]
    shell_ang_mom = Int[]
    shell_factor = Float64[]
    shell_range = Int[]
    
    prim_num_total = 0
    shell_index = 1
    
    # Count total primitives first
    for shell in basisset
        prim_num_total += length(shell.exponents)
    end
    
    # Collect exponents and coefficients
    exponent = Float64[]
    coefficient = Float64[]
    
    for shell in basisset
        # Get center index from shell_indices
        center_idx = basisset.shell_indices[shell_index][1]
        push!(shell_nucleus_index, center_idx)
        push!(shell_ang_mom, shell.l)
        push!(shell_factor, 1.0)  # Normalization factor
        push!(shell_range, length(shell.exponents))
        
        # Add exponents
        for exp in shell.exponents
            push!(exponent, exp)
        end
        
        # Add coefficients for each contraction
        for contraction in shell.subshells
            for coef in contraction.coefs
                push!(coefficient, coef)
            end
        end
        
        shell_index += 1
    end
    
    # Write TREXIOIO standard datasets
    basis_group["shell_num"] = Int64(shell_num)
    basis_group["prim_num"] = Int64(prim_num_total)
    basis_group["shell_nucleus_index"] = shell_nucleus_index
    basis_group["shell_ang_mom"] = shell_ang_mom
    basis_group["shell_factor"] = shell_factor
    basis_group["shell_range"] = shell_range
    basis_group["exponent"] = exponent
    basis_group["coefficient"] = coefficient
    
    # Store basis set type as attribute
    if !isempty(basisset.centres)
        attrs(basis_group)["type"] = basisset.centres[1].basis
    end
end

"""
Write basic basis set names (fallback for when detailed basis set is not available)
"""
function _write_basic_basis_names(basis_group, system)
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
    
    # Write basic basis set data (non-TREXIOIO standard, for compatibility)
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
end

"""
    read_trexio_basis(trexio::TrexioFile) -> Dict{String, Any}

Read basis set information from TREXIO format.
Handles both TREXIOIO-compliant format and legacy format.
"""
function read_trexio_basis(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "basis")
        error("No basis set data found in TREXIO file")
    end
    
    basis_group = file["trexio"]["basis"]
    
    # Check if this is TREXIOIO-compliant format (has shell_num) or legacy format (has num)
    if haskey(basis_group, "shell_num")
        # TREXIOIO-compliant format
        return _read_detailed_basis_trexio(basis_group)
    elseif haskey(basis_group, "num")
        # Legacy format (backward compatibility)
        return _read_basic_basis_legacy(basis_group)
    else
        error("Unknown basis set format in TREXIO file")
    end
end

"""
Read TREXIOIO-compliant detailed basis set information
"""
function _read_detailed_basis_trexio(basis_group)
    basis_info = Dict{String, Any}()
    
    # Read TREXIOIO standard datasets
    basis_info["shell_num"] = read(basis_group["shell_num"])
    basis_info["prim_num"] = read(basis_group["prim_num"])
    basis_info["shell_nucleus_index"] = read(basis_group["shell_nucleus_index"])
    basis_info["shell_ang_mom"] = read(basis_group["shell_ang_mom"])
    basis_info["shell_factor"] = read(basis_group["shell_factor"])
    basis_info["shell_range"] = read(basis_group["shell_range"])
    basis_info["exponent"] = read(basis_group["exponent"])
    basis_info["coefficient"] = read(basis_group["coefficient"])
    
    # Read basis set type from attributes
    if haskey(attrs(basis_group), "type")
        basis_info["type"] = read(attrs(basis_group)["type"])
    end
    
    # Mark as TREXIOIO format
    basis_info["format"] = "trexio"
    
    return basis_info
end

"""
Read legacy basis set format (for backward compatibility)
"""
function _read_basic_basis_legacy(basis_group)
    # Read legacy format data
    nbasis = read(basis_group["num"])
    atom_indices = read(basis_group["nucleus_index"])
    basis_types = read(basis_group["type"])
    
    basis_info = Dict{String, Any}(
        "num" => nbasis,
        "nucleus_index" => atom_indices,
        "type" => basis_types,
        "format" => "legacy"
    )
    
    # Read additional attributes if available
    if haskey(attrs(basis_group), "available_types")
        basis_info["available_types"] = read(attrs(basis_group)["available_types"])
    end
    
    return basis_info
end

"""
    write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix; 
                          orbital_type="molecular", system=nothing, basisset=nothing)

Write molecular orbitals to TREXIO format. If system is provided, 
basis set information will also be written. If basisset is provided,
detailed TREXIOIO-compliant basis information will be stored.
"""
function write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix; 
                            orbital_type="molecular", system=nothing, basisset=nothing)
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_trexio_metadata(file)
    
    # Write basis set information if system is provided and not already written
    if !isnothing(system) && !haskey(trex_group, "basis")
        write_trexio_basis(trexio, system; basisset=basisset)
    end
    
    # Create MO group
    if haskey(trex_group, "mo")
        delete_object(trex_group, "mo")
    end
    mo_group = create_group(trex_group, "mo")
    
    # Write orbital information
    nbasis, nmo = size(orbitals)
    if is_restricted(orbitals)
        mo_group["coefficient"] = orbitals[1]
    else
        mo_group["coefficient"] = hcat(orbitals...)
        attrs(mo_group)["spin"] = vcat(fill(0, nmo), fill(1, nmo))
        nmo *= 2  # For unrestricted, double the number of orbitals
    end
    mo_group["num"] = Int64(nmo)
    # Add metadata
    attrs(mo_group)["type"] = orbital_type
    attrs(mo_group)["basis_size"] = Int64(nbasis)
    
    return mo_group
end

"""
    read_trexio_orbitals(trexio::TrexioFile) -> AbstractMatrix

Read molecular orbitals from TREXIO format.
"""
function read_trexio_orbitals(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "mo")
        error("No orbital data found in TREXIO file")
    end
    
    mo_group = file["trexio"]["mo"]
    if haskey(mo_group, "spin")
        # Unrestricted orbitals
        spins = read(mo_group["spin"])
        if length(spins) == 2 * size(mo_group["coefficient"], 2)
            return SpinMatrix(read(mo_group["coefficient"][:, 1:end÷2]), read(mo_group["coefficient"][:, end÷2+1:end]))
        else
            error("Spin data does not match orbital coefficients")
        end
    end
    return SpinMatrix(read(mo_group["coefficient"]))
end

"""
    write_trexio_amplitudes(trexio::TrexioFile, amplitudes::Dict{String, Any})

Write CC amplitudes to TREXIO format.
"""
function write_trexio_amplitudes(trexio::TrexioFile, amplitudes::Dict{String, Any})
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_trexio_metadata(file)
    
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
    read_trexio_amplitudes(trexio::TrexioFile) -> Dict{String, Any}

Read CC amplitudes from TREXIO format.
"""
function read_trexio_amplitudes(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "amplitudes")
        error("No amplitude data found in TREXIO file")
    end
    
    amp_group = file["trexio"]["amplitudes"]
    amplitudes = Dict{String, Any}()
    
    for key in keys(amp_group)
        amplitudes[key] = read(amp_group[key])
    end
    
    return amplitudes
end

"""
    write_trexio(filename::String, EC::ECInfo; kwargs...)

Write ElemCo data to TREXIO format file. When orbitals are included,
basis set information is automatically written as well.

# Keyword arguments
- `include_orbitals::Bool=true`: Include molecular orbitals and basis sets
- `include_amplitudes::Bool=false`: Include CC amplitudes  
- `include_molecule::Bool=true`: Include molecular geometry
"""
function write_trexio(filename::String, EC::ECInfo; 
                   include_orbitals::Bool=true,
                   include_amplitudes::Bool=false,
                   include_molecule::Bool=true)
    
    trexio = TrexioFile(filename, "w")
    
    try
        # Write molecular information
        if include_molecule && !isnothing(EC.system)
            write_trexio_molecule(trexio, EC.system)
        end
        
        # Write orbitals if available
        if include_orbitals
            try
                # Check if orbitals file exists
                orb_file = fullfilename(EC, EC.options.wf.orb)
                if file_exists(EC, EC.options.wf.orb)
                    orbs = load_orbitals(EC, EC.options.wf.orb)
                    if !isnothing(orbs)
                        # Try to get basis set information for TREXIOIO-compliant storage
                        local basisset = nothing
                        try
                            # Try to import BasisSets module and generate basis set
                            basisset = ElemCo.BasisSets.generate_basis(EC, "ao")
                        catch e
                            @debug "Could not generate basis set for TREXIO export: $e"
                        end
                        
                        # Pass system and basis set information to include detailed basis set data
                        write_trexio_orbitals(trexio, orbs, system=EC.system, basisset=basisset)
                    end
                else
                    @warn "Orbital file $(EC.options.wf.orb) not found, skipping orbital export"
                end
            catch e
                @warn "Could not load orbitals for TREXIO export: $e"
            end
        end
        
        # Write amplitudes if requested and available
        if include_amplitudes
            # This would need to be implemented based on how amplitudes are stored in ElemCo
            @warn "Amplitude export not yet implemented"
        end
        
    finally
        close_trexio(trexio)
    end
    
    return filename
end

"""
    read_trexio(filename::String) -> Dict{String, Any}

Read data from TREXIO format file.

Returns a dictionary with available data sections.
"""
function read_trexio(filename::String)
    if !isfile(filename)
        error("TREXIO file not found: $filename")
    end
    
    trexio = TrexioFile(filename, "r")
    data = Dict{String, Any}()
    
    try
        file = open_trexio(trexio)
        
        if !haskey(file, "trexio")
            error("Invalid TREXIO file format")
        end
        
        trex_group = file["trexio"]
        
        # Read available sections
        if haskey(trex_group, "nucleus")
            data["molecule"] = read_trexio_molecule(trexio)
        end
        
        if haskey(trex_group, "basis")
            data["basis"] = read_trexio_basis(trexio)
        end
        
        if haskey(trex_group, "mo")
            data["orbitals"] = read_trexio_orbitals(trexio)
        end
        
        if haskey(trex_group, "amplitudes")
            data["amplitudes"] = read_trexio_amplitudes(trexio)
        end
        
    finally
        close_trexio(trexio)
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

end # module TrexioInterface