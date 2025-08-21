"""
TREXIO Interface Module for ElemCo.jl

This module provides ElemCo-specific functions to import and export data in TREXIO format,
which is a standardized format for quantum chemistry data exchange.
It uses the standalone TREXIO module for core operations and provides convenient
conversion between ElemCo types and TREXIO standard formats.

TREXIO format specification:
- Based on HDF5 for efficient storage
- Standardized structure for quantum chemistry data
- Supports orbitals, amplitudes, integrals, and other QC data

See: https://trexio-coe.github.io/trexio/lib.html
     https://arxiv.org/abs/2302.14793
"""
module TrexioInterface

using ..ElemCo.ECInfos
using ..ElemCo.MSystems
using ..ElemCo.Utils
using ..ElemCo.QMTensors
using ..ElemCo.Elements
using ..ElemCo.BasisSets
using ..ElemCo.OrbTools
using ..ElemCo.TREXIO  # Use the standalone TREXIO module
using LinearAlgebra

# Re-export the core TREXIO types and functions for backward compatibility
export TrexioFile, write_trexio, read_trexio
export write_trexio_orbitals, read_trexio_orbitals
export write_trexio_amplitudes, read_trexio_amplitudes
export write_trexio_molecule, read_trexio_molecule
export write_trexio_basis, read_trexio_basis
export open_trexio, close_trexio

# Re-export the standalone TREXIO types for compatibility
const TrexioFile = TREXIO.TrexioFile
const open_trexio = TREXIO.open_trexio
const close_trexio = TREXIO.close_trexio

"""
    write_trexio_molecule(trexio::TrexioFile, system::MSystem)

Write molecular geometry and basis set information to TREXIO format using ElemCo data structures.
"""
function write_trexio_molecule(trexio::TrexioFile, system::MSystem)
    # Convert ElemCo MSystem to TREXIO format
    natoms = length(system)
    nuclear_charges = Float64[]
    coordinates = zeros(Float64, 3, natoms)
    labels = String[]
    
    for (i, atom) in enumerate(system)
        push!(nuclear_charges, Float64(atom.atomic_number))
        coordinates[:, i] = Vector(atom.position)
        push!(labels, atom.label)
    end
    
    # Use the standalone TREXIO module to write the data
    return TREXIO.write_nucleus(trexio, nuclear_charges, coordinates, labels)
end

"""
    read_trexio_molecule(trexio::TrexioFile) -> MSystem

Read molecular geometry from TREXIO format and return ElemCo MSystem.
"""
function read_trexio_molecule(trexio::TrexioFile)
    # Read data using standalone TREXIO module
    nuclear_charges, coordinates, labels = TREXIO.read_nucleus(trexio)
    
    # Convert to ElemCo MSystem
    atoms = ACentre[]
    for i in 1:length(nuclear_charges)
        # Convert charge to element symbol
        element = element_symbol(Int(nuclear_charges[i]))
        pos = coordinates[:, i]
        
        # Create basic basis (this would need to be enhanced for real use)
        basis = Dict{String,String}()
        
        atom = ACentre(labels[i], pos[1], pos[2], pos[3], Int(nuclear_charges[i]), nuclear_charges[i], basis)
        push!(atoms, atom)
    end
    
    return MSystem(atoms)
end

"""
    write_trexio_basis(trexio::TrexioFile, system::MSystem; basisset=nothing)

Write basis set information to TREXIO format following TREXIO standard.
This is a simplified version that stores basic basis set names. 
For full TREXIO-compliant basis sets, the detailed basisset information would be needed.
"""
function write_trexio_basis(trexio::TrexioFile, system::MSystem; basisset=nothing)
    # For now, we'll implement a simplified version that stores basis set names
    # This maintains backward compatibility while the full TREXIO basis implementation is developed
    
    # Open file and ensure it has metadata
    file = open_trexio(trexio)
    if !haskey(file, "trexio")
        TREXIO.write_metadata(trexio, created_by="ElemCo.jl")
    end
    
    # For simplicity, store basis information as attributes for now
    # This is not fully TREXIO compliant but maintains functionality
    trex_group = file["trexio"]
    
    # Create a simple basis group with basic information
    if haskey(trex_group, "basis")
        delete_object(trex_group, "basis")
    end
    basis_group = create_group(trex_group, "basis")
    
    # Store basic basis set names from atoms
    basis_names = String[]
    for atom in system
        if haskey(atom.basis, "ao")
            push!(basis_names, atom.basis["ao"])
        else
            push!(basis_names, "unknown")
        end
    end
    
    # Store as legacy format for compatibility
    basis_group["num"] = Int64(length(system))
    basis_group["nucleus_index"] = collect(1:length(system))
    basis_group["type"] = basis_names
    attrs(basis_group)["format"] = "legacy"
    
    return basis_group
end

"""
    read_trexio_basis(trexio::TrexioFile) -> Dict{String, Any}

Read basis set information from TREXIO format.
Handles both TREXIO-compliant format and legacy format.
"""
function read_trexio_basis(trexio::TrexioFile)
    # Try to read using the standalone TREXIO module first
    try
        basis_data = TREXIO.read_basis(trexio)
        basis_data["format"] = "trexio"
        return basis_data
    catch
        # Fall back to legacy format reading
        file = open_trexio(trexio)
        
        if !haskey(file, "trexio") || !haskey(file["trexio"], "basis")
            error("No basis set data found in TREXIO file")
        end
        
        basis_group = file["trexio"]["basis"]
        
        # Read legacy format
        basis_data = Dict{String, Any}()
        basis_data["num"] = read(basis_group["num"])
        basis_data["nucleus_index"] = read(basis_group["nucleus_index"])
        basis_data["type"] = read(basis_group["type"])
        basis_data["format"] = "legacy"
        
        return basis_data
    end
end

"""
    write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix; 
                          orbital_type="molecular", system=nothing, basisset=nothing)

Write molecular orbitals to TREXIO format using ElemCo data structures.
If system is provided, basis set information will also be written.
"""
function write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix; 
                            orbital_type="molecular", system=nothing, basisset=nothing)
    
    # Write basis set information if system is provided
    if !isnothing(system)
        write_trexio_basis(trexio, system; basisset=basisset)
    end
    
    # Convert ElemCo orbital format to standard matrix format
    nbasis, nmo = size(orbitals)
    if is_restricted(orbitals)
        coefficients = orbitals[1]
        spin = nothing
    else
        coefficients = hcat(orbitals...)
        spin = vcat(fill(0, nmo), fill(1, nmo))  # α=0, β=1
        nmo *= 2  # For unrestricted, double the number of orbitals
    end
    
    # Use the standalone TREXIO module to write MO data
    return TREXIO.write_mo(trexio, coefficients, orbital_type=orbital_type, spin=spin)
end

"""
    read_trexio_orbitals(trexio::TrexioFile) -> SpinMatrix

Read molecular orbitals from TREXIO format and return ElemCo SpinMatrix.
"""
function read_trexio_orbitals(trexio::TrexioFile)
    # Read MO data using standalone TREXIO module
    mo_data = TREXIO.read_mo(trexio)
    
    coefficients = mo_data["coefficient"]
    
    # Check if spin information is available (unrestricted case)
    if haskey(mo_data, "spin")
        spins = mo_data["spin"]
        nmo_total = size(coefficients, 2)
        nmo_half = nmo_total ÷ 2
        
        # Split into α and β orbitals
        α_orbs = coefficients[:, 1:nmo_half]
        β_orbs = coefficients[:, nmo_half+1:end]
        
        return SpinMatrix(α_orbs, β_orbs)
    else
        # Restricted case
        return SpinMatrix(coefficients)
    end
end

"""
    write_trexio_amplitudes(trexio::TrexioFile, amplitudes::Dict{String, Any})

Write CC amplitudes to TREXIO format using the standalone TREXIO module.
This is a custom extension for storing amplitude data.
"""
function write_trexio_amplitudes(trexio::TrexioFile, amplitudes::Dict{String, Any})
    # Ensure metadata exists
    file = open_trexio(trexio)
    if !haskey(file, "trexio")
        TREXIO.write_metadata(trexio, created_by="ElemCo.jl")
    end
    
    trex_group = file["trexio"]
    
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

Write ElemCo data to TREXIO format file using the standalone TREXIO module.
When orbitals are included, basis set information is automatically written as well.

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
        # Write metadata
        TREXIO.write_metadata(trexio, created_by="ElemCo.jl")
        
        # Write molecular information
        if include_molecule && !isnothing(EC.system)
            write_trexio_molecule(trexio, EC.system)
        end
        
        # Write orbitals if available
        if include_orbitals
            try
                # Check if orbitals file exists
                if file_exists(EC, EC.options.wf.orb)
                    orbs = load_orbitals(EC, EC.options.wf.orb)
                    if !isnothing(orbs)
                        # Try to get basis set information for TREXIO-compliant storage
                        local basisset = nothing
                        try
                            # Try to generate basis set
                            basisset = generate_basis(EC, "ao")
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

Read data from TREXIO format file using the standalone TREXIO module.

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