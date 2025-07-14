"""
TREXIO - A standalone Julia implementation of the TREXIO format

This module provides a standalone implementation of the TREXIO (Table of Results Exchange) 
format for quantum chemistry data exchange. It follows the TREXIO specification closely
and can be used independently of any quantum chemistry package.

TREXIO format specification: https://trex-coe.github.io/trexio/trex.html
API documentation: https://trex-coe.github.io/trexio/index.html
Reference: https://arxiv.org/pdf/2302.14793

This implementation supports:
- HDF5 backend for efficient storage
- Column-major representation as specified by TREXIO
- Standard TREXIO data groups: nucleus, basis, mo, amplitudes, etc.
- Metadata and versioning
"""
module TREXIO

using HDF5
using Dates

export TrexioFile, open_trexio, close_trexio
export write_nucleus, read_nucleus
export write_basis, read_basis  
export write_mo, read_mo
export write_metadata, read_metadata
export create_trexio_file, read_trexio_file

"""
    TrexioFile

Structure representing a TREXIO format file following the standard specification.
Contains the HDF5 file handle and metadata.

# Fields
- `filename::String`: Path to the TREXIO file
- `file::Union{HDF5.File, Nothing}`: HDF5 file handle
- `mode::String`: File access mode ("r", "w", "r+")
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
    open_trexio(trexio::TrexioFile) -> HDF5.File

Open a TREXIO file for reading or writing. Creates the main TREXIO group structure
if opening in write mode.
"""
function open_trexio(trexio::TrexioFile)
    if trexio.file === nothing
        trexio.file = h5open(trexio.filename, trexio.mode)
        # Ensure the main TREXIO group exists following TREXIO standard
        if trexio.mode in ["w", "r+"] && !haskey(trexio.file, "trexio")
            create_group(trexio.file, "trexio")
        end
    end
    return trexio.file
end

"""
    close_trexio(trexio::TrexioFile)

Close a TREXIO file and release resources.
"""
function close_trexio(trexio::TrexioFile)
    if trexio.file !== nothing
        close(trexio.file)
        trexio.file = nothing
    end
end

"""
    write_metadata(trexio::TrexioFile; format_version="2.4.0", created_by="TREXIO.jl")

Write TREXIO format metadata following the standard specification.
"""
function write_metadata(trexio::TrexioFile; format_version="2.4.0", created_by="TREXIO.jl")
    file = open_trexio(trexio)
    trex_group = file["trexio"]
    
    # Write standard TREXIO metadata
    attrs(trex_group)["format_version"] = format_version
    attrs(trex_group)["created_by"] = created_by
    attrs(trex_group)["created_date"] = string(Dates.now())
    
    return trex_group
end

"""
    read_metadata(trexio::TrexioFile) -> Dict{String, String}

Read TREXIO format metadata.
"""
function read_metadata(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio")
        error("Invalid TREXIO file: missing main trexio group")
    end
    
    trex_group = file["trexio"]
    metadata = Dict{String, String}()
    
    # Read standard metadata attributes
    group_attrs = attrs(trex_group)
    for attr_name in ["format_version", "created_by", "created_date"]
        if haskey(group_attrs, attr_name)
            metadata[attr_name] = group_attrs[attr_name]
        end
    end
    
    return metadata
end

"""
    write_nucleus(trexio::TrexioFile, nuclear_charges::Vector{Float64}, 
                 coordinates::Matrix{Float64}, labels::Vector{String}; 
                 units="bohr")

Write nucleus data to TREXIO file following the standard specification.

# Arguments
- `nuclear_charges`: Array of nuclear charges (atomic numbers)
- `coordinates`: Matrix of atomic coordinates (3 × natoms, column-major as per TREXIO spec)
- `labels`: Array of atomic labels
- `units`: Coordinate units (default: "bohr")
"""
function write_nucleus(trexio::TrexioFile, nuclear_charges::Vector{Float64}, 
                      coordinates::Matrix{Float64}, labels::Vector{String}; 
                      units="bohr")
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_metadata(trexio)
    
    # Create or recreate nucleus group
    if haskey(trex_group, "nucleus")
        delete_object(trex_group, "nucleus")
    end
    nucleus_group = create_group(trex_group, "nucleus")
    
    natoms = length(nuclear_charges)
    
    # Validate input dimensions
    if size(coordinates) != (3, natoms)
        error("Coordinates must be 3×natoms matrix (column-major as per TREXIO spec)")
    end
    if length(labels) != natoms
        error("Labels array must have same length as nuclear_charges")
    end
    
    # Write nuclear data following TREXIO standard
    nucleus_group["num"] = Int64(natoms)
    nucleus_group["charge"] = nuclear_charges
    nucleus_group["coord"] = coordinates  # Column-major format
    nucleus_group["label"] = labels
    
    # Add units information as required by TREXIO
    attrs(nucleus_group["coord"])["units"] = units
    
    return nucleus_group
end

"""
    read_nucleus(trexio::TrexioFile) -> (Vector{Float64}, Matrix{Float64}, Vector{String})

Read nucleus data from TREXIO file. Returns (nuclear_charges, coordinates, labels).
"""
function read_nucleus(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "nucleus")
        error("No nucleus data found in TREXIO file")
    end
    
    nucleus_group = file["trexio"]["nucleus"]
    
    # Read nuclear data
    natoms = read(nucleus_group["num"])
    nuclear_charges = read(nucleus_group["charge"])
    coordinates = read(nucleus_group["coord"])  # Column-major format
    labels = read(nucleus_group["label"])
    
    return nuclear_charges, coordinates, labels
end

"""
    write_basis(trexio::TrexioFile, shell_num::Int, shell_nucleus_index::Vector{Int},
               shell_ang_mom::Vector{Int}, shell_factor::Vector{Float64},
               shell_range::Vector{Int}, exponent::Vector{Float64},
               coefficient::Vector{Float64})

Write basis set data to TREXIO file following the standard specification.

This follows the TREXIO standard for basis set representation with shells and primitives.
"""
function write_basis(trexio::TrexioFile, shell_num::Int, shell_nucleus_index::Vector{Int},
                    shell_ang_mom::Vector{Int}, shell_factor::Vector{Float64},
                    shell_range::Vector{Int}, exponent::Vector{Float64},
                    coefficient::Vector{Float64})
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_metadata(trexio)
    
    # Create or recreate basis group
    if haskey(trex_group, "basis")
        delete_object(trex_group, "basis")
    end
    basis_group = create_group(trex_group, "basis")
    
    # Write TREXIO standard basis set data
    basis_group["shell_num"] = Int64(shell_num)
    basis_group["prim_num"] = Int64(length(exponent))
    basis_group["shell_nucleus_index"] = shell_nucleus_index
    basis_group["shell_ang_mom"] = shell_ang_mom
    basis_group["shell_factor"] = shell_factor
    basis_group["shell_range"] = shell_range
    basis_group["exponent"] = exponent
    basis_group["coefficient"] = coefficient
    
    return basis_group
end

"""
    read_basis(trexio::TrexioFile) -> Dict{String, Any}

Read basis set data from TREXIO file following the standard specification.
"""
function read_basis(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "basis")
        error("No basis set data found in TREXIO file")
    end
    
    basis_group = file["trexio"]["basis"]
    basis_data = Dict{String, Any}()
    
    # Read TREXIO standard basis set data
    for key in ["shell_num", "prim_num", "shell_nucleus_index", "shell_ang_mom", 
                "shell_factor", "shell_range", "exponent", "coefficient"]
        if haskey(basis_group, key)
            basis_data[key] = read(basis_group[key])
        end
    end
    
    return basis_data
end

"""
    write_mo(trexio::TrexioFile, coefficients::Matrix{Float64}; 
            orbital_type="molecular", spin=nothing)

Write molecular orbital data to TREXIO file following the standard specification.

# Arguments
- `coefficients`: MO coefficient matrix (nbasis × nmo, column-major as per TREXIO spec)
- `orbital_type`: Type of orbitals (default: "molecular")
- `spin`: Optional spin array for unrestricted calculations
"""
function write_mo(trexio::TrexioFile, coefficients::Matrix{Float64}; 
                 orbital_type="molecular", spin=nothing)
    file = open_trexio(trexio)
    trex_group = haskey(file, "trexio") ? file["trexio"] : write_metadata(trexio)
    
    # Create or recreate MO group
    if haskey(trex_group, "mo")
        delete_object(trex_group, "mo")
    end
    mo_group = create_group(trex_group, "mo")
    
    nbasis, nmo = size(coefficients)
    
    # Write orbital data following TREXIO standard
    mo_group["coefficient"] = coefficients  # Column-major format
    mo_group["num"] = Int64(nmo)
    
    # Add optional spin information for unrestricted calculations
    if !isnothing(spin)
        mo_group["spin"] = spin
    end
    
    # Add metadata
    attrs(mo_group)["type"] = orbital_type
    attrs(mo_group)["basis_size"] = Int64(nbasis)
    
    return mo_group
end

"""
    read_mo(trexio::TrexioFile) -> Dict{String, Any}

Read molecular orbital data from TREXIO file following the standard specification.
"""
function read_mo(trexio::TrexioFile)
    file = open_trexio(trexio)
    
    if !haskey(file, "trexio") || !haskey(file["trexio"], "mo")
        error("No molecular orbital data found in TREXIO file")
    end
    
    mo_group = file["trexio"]["mo"]
    mo_data = Dict{String, Any}()
    
    # Read orbital data
    mo_data["coefficient"] = read(mo_group["coefficient"])
    mo_data["num"] = read(mo_group["num"])
    
    # Read optional spin information
    if haskey(mo_group, "spin")
        mo_data["spin"] = read(mo_group["spin"])
    end
    
    # Read metadata from attributes
    mo_attrs = attrs(mo_group)
    if haskey(mo_attrs, "type")
        mo_data["type"] = mo_attrs["type"]
    end
    if haskey(mo_attrs, "basis_size")
        mo_data["basis_size"] = mo_attrs["basis_size"]
    end
    
    return mo_data
end

"""
    create_trexio_file(filename::String, nucleus_data=nothing, basis_data=nothing, 
                      mo_data=nothing; kwargs...)

High-level function to create a complete TREXIO file with multiple data sections.

# Arguments
- `filename`: Output file path
- `nucleus_data`: Tuple of (charges, coordinates, labels) or nothing
- `basis_data`: Basis set data dictionary or nothing  
- `mo_data`: MO coefficient matrix or nothing
- `kwargs`: Additional metadata options
"""
function create_trexio_file(filename::String, nucleus_data=nothing, basis_data=nothing, 
                           mo_data=nothing; kwargs...)
    trexio = TrexioFile(filename, "w")
    
    try
        # Write metadata
        write_metadata(trexio; kwargs...)
        
        # Write nucleus data if provided
        if !isnothing(nucleus_data)
            charges, coords, labels = nucleus_data
            write_nucleus(trexio, charges, coords, labels)
        end
        
        # Write basis set data if provided
        if !isnothing(basis_data)
            write_basis(trexio, basis_data["shell_num"], basis_data["shell_nucleus_index"],
                       basis_data["shell_ang_mom"], basis_data["shell_factor"],
                       basis_data["shell_range"], basis_data["exponent"], 
                       basis_data["coefficient"])
        end
        
        # Write MO data if provided  
        if !isnothing(mo_data)
            write_mo(trexio, mo_data)
        end
        
    finally
        close_trexio(trexio)
    end
    
    return filename
end

"""
    read_trexio_file(filename::String) -> Dict{String, Any}

High-level function to read all available data from a TREXIO file.

Returns a dictionary with available data sections.
"""
function read_trexio_file(filename::String)
    if !isfile(filename)
        error("TREXIO file not found: $filename")
    end
    
    trexio = TrexioFile(filename, "r")
    data = Dict{String, Any}()
    
    try
        file = open_trexio(trexio)
        
        if !haskey(file, "trexio")
            error("Invalid TREXIO file format: missing main trexio group")
        end
        
        trex_group = file["trexio"]
        
        # Read metadata
        data["metadata"] = read_metadata(trexio)
        
        # Read available data sections
        if haskey(trex_group, "nucleus")
            charges, coords, labels = read_nucleus(trexio)
            data["nucleus"] = Dict("charge" => charges, "coord" => coords, "label" => labels)
        end
        
        if haskey(trex_group, "basis")
            data["basis"] = read_basis(trexio)
        end
        
        if haskey(trex_group, "mo")
            data["mo"] = read_mo(trexio)
        end
        
    finally
        close_trexio(trexio)
    end
    
    return data
end

end # module TREXIO