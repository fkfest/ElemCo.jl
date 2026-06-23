"""
TREXIO - A standalone Julia implementation of the TREXIO format

This module provides a standalone implementation of the TREXIO format for quantum chemistry 
data exchange. It follows the TREXIO specification closely
and can be used independently of any quantum chemistry package.

TREXIO format specification: https://trex-coe.github.io/trexio/trex.html
API documentation: https://trex-coe.github.io/trexio/index.html
Reference: https://arxiv.org/pdf/2302.14793

This implementation supports:
- HDF5 backend for efficient storage
- Column-major representation as specified by TREXIO
- Standard TREXIO data groups: nucleus, basis, mo, electron, ao, ecp, grid, amplitude, rdm, qmc
- Full TREXIO naming convention: trexio_[has|read|write]_<group>_<attribute>
- Metadata and versioning
- Error handling with TREXIO exit codes

The module contains a set of groups/attributes that extend the standard TREXIO format
to support additional features and data types. A suppressable warning is issued if the corresponding
has/read/write functions are called, and the doc-strings contain warnings as well.
"""
module TREXIO

using HDF5
using Dates

# exports `trexio_[has|read|write]_<group>_<attribute>` functions

# TREXIO exit codes following the standard
@enum TrexioExitCode begin
    TREXIO_FAILURE	            =-1     # Unknown failure
    TREXIO_SUCCESS	            = 0     # Success
    TREXIO_INVALID_ARG_1	    = 1	    # Invalid argument 1
    TREXIO_INVALID_ARG_2	    = 2	    # Invalid argument 2
    TREXIO_INVALID_ARG_3	    = 3	    # Invalid argument 3
    TREXIO_INVALID_ARG_4	    = 4	    # Invalid argument 4
    TREXIO_INVALID_ARG_5	    = 5	    # Invalid argument 5
    TREXIO_END	                = 6	    # End of file
    TREXIO_READONLY	            = 7	    # Read-only file
    TREXIO_ERRNO	            = 8	    # streerror(errno)
    TREXIO_INVALID_ID	        = 9	    # Invalid ID
    TREXIO_ALLOCATION_FAILED    = 10	# Allocation failed
    TREXIO_HAS_NOT	            = 11	# Element absent
    TREXIO_INVALID_NUM	        = 12	# Invalid (negative or 0) dimension
    TREXIO_ATTR_ALREADY_EXISTS  = 13	# Attribute already exists
    TREXIO_DSET_ALREADY_EXISTS	= 14	# Dataset already exists
    TREXIO_OPEN_ERROR	        = 15	# Error opening file
    TREXIO_LOCK_ERROR	        = 16	# Error locking file
    TREXIO_UNLOCK_ERROR	        = 17	# Error unlocking file
    TREXIO_FILE_ERROR	        = 18	# Invalid file
    TREXIO_GROUP_READ_ERROR	    = 19	# Error reading group
    TREXIO_GROUP_WRITE_ERROR	= 20	# Error writing group
    TREXIO_ELEM_READ_ERROR	    = 21	# Error reading element
    TREXIO_ELEM_WRITE_ERROR	    = 22	# Error writing element
    TREXIO_UNSAFE_ARRAY_DIM	    = 23	# Access to memory beyond allocated
    TREXIO_ATTR_MISSING	        = 24	# Attribute does not exist in the file
    TREXIO_DSET_MISSING	        = 25	# Dataset does not exist in the file
    TREXIO_BACK_END_MISSING	    = 26	# Requested back end is disabled
    TREXIO_INVALID_ARG_6	    = 27	# Invalid argument 6
    TREXIO_INVALID_ARG_7	    = 28	# Invalid argument 7
    TREXIO_INVALID_ARG_8	    = 29	# Invalid argument 8
    TREXIO_INVALID_STR_LEN	    = 30	# Invalid maxstrlen
    TREXIO_INT_SIZE_OVERFLOW	= 31	# Possible integer overflow
    TREXIO_SAFE_MODE	        = 32	# Unsafe operation in safe mode
    TREXIO_INVALID_ELECTRON_NUM	= 33	# Inconsistent number of electrons
    TREXIO_INVALID_DETERMINANT_NUM	= 34	# Inconsistent number of determinants
    TREXIO_INVALID_STATE	        = 35	# Inconsistent state of the file
    TREXIO_VERSION_PARSING_ISSUE	= 36	# Failed to parse package version
    TREXIO_PHASE_CHANGE	        = 37	# The function succeeded with a change of sign
    TREXIO_INVALID_MO_INDEX	    = 38	# Invalid MO index
    TREXIO_INVALID_ARG_9	    = 39	# Invalid argument 9
    TREXIO_INVALID_ARG_10	    = 40	# Invalid argument 10
    TREXIO_INVALID_ARG_11	    = 41	# Invalid argument 11
    TREXIO_INVALID_ARG_12	    = 42	# Invalid argument 12
    TREXIO_INVALID_ARG_13	    = 43	# Invalid argument 13
    TREXIO_INVALID_ARG_14	    = 44	# Invalid argument 14
    TREXIO_CORRUPTION_ATTEMPT	= 45 	# File offset is wrong, corruption risk
    TREXIO_GROUP_ERROR          = 46    # Error creating or accessing group
end

export TrexioFile, TrexioExitCode
export trexio_open, trexio_close
export trexio_check_read_status, trexio_check_write_status

"""
    trexio_check_read_status(status::TrexioExitCode, what::String="")

Check TREXIO read status and throw error if not successful.
`what` is an optional description of the data being read.
"""
function trexio_check_read_status(status::TrexioExitCode, what::String="")
  if status != TREXIO_SUCCESS
    error("TREXIO error: Failed to read $what from TREXIO with status $(string(status))")
  end
end

"""
    trexio_check_write_status(status::TrexioExitCode, what::String="")

Check TREXIO write status and throw error if not successful.
`what` is an optional description of the data being written.
"""
function trexio_check_write_status(status::TrexioExitCode, what::String="")
  if status != TREXIO_SUCCESS
    error("TREXIO error: Failed to write $what to TREXIO with status $(string(status))")
  end
end

"""
    TrexioFile

Structure representing a TREXIO format file following the standard specification.
Contains the HDF5 file handle and metadata.

# Fields
- `file::HDF5.File`: HDF5 file handle
- `filename::String`: Path to the TREXIO file
- `mode::String`: File access mode ("r", "w", "u")
"""
struct TrexioFile
    file::HDF5.File
    filename::String
    mode::String  # "r", "w", "u"
end

"""
    trexio_open(file_name::String, mode::String="u", back_end=0) -> TrexioFile

Open a TREXIO file for reading or writing. Creates the metadata group structure
if opening in write mode. Returns the TrexioFile object or nothing on failure.
"""
function trexio_open(file_name::String, mode::String="u", back_end=0)
    @assert back_end == 0 "TREXIO.jl supports only HDF5 backend (back_end=0)"
    h5mode = mode == "u" ? "cw" : mode
    try
        file = h5open(file_name, h5mode)
        # Ensure the metadata exists following TREXIO standard
        if mode in ["w", "u"] && !haskey(file, "metadata")
            # Write basic metadata for new files
            _write_basic_metadata(file)
        end
        return TrexioFile(file, file_name, mode)
    catch e
        throw(TREXIO_OPEN_ERROR)
    end
end

"""
    trexio_close(trexio::TrexioFile) -> TrexioExitCode

Close a TREXIO file and release resources. Returns TREXIO exit code.
"""
function trexio_close(trexio::TrexioFile)
    try
        close(trexio.file)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FILE_ERROR
    end
end

# Helper function to get or create group
function _get_or_create_group(trexio::TrexioFile, group_name)
    if !isopen(trexio.file)
        return nothing, TREXIO_OPEN_ERROR
    end
    
    # All groups are at root level of HDF5 file
    if !haskey(trexio.file, group_name)
        if trexio.mode in ["w", "u"]
            target_group = create_group(trexio.file, group_name)
        else
            return nothing, TREXIO_GROUP_ERROR
        end
    else
        target_group = trexio.file[group_name]
    end
    
    return target_group, TREXIO_SUCCESS
end

# Helper function to check if an HDF5 attribute exists on a group
function _has_hdf5_attribute(group::HDF5.Group, attr_name::AbstractString)
    return haskey(HDF5.attributes(group), String(attr_name))
end

# Helper function to check if a dataset exists in a group
function _has_hdf5_dataset(group::HDF5.Group, dset_name::AbstractString)
    return haskey(group, String(dset_name))
end

# Helper function to check if attribute/dataset exists (for scalar: attribute, for array: dataset)
# For sparse arrays, checks for the _values dataset
function _has_attribute(trexio::TrexioFile, group_name, attr_name; is_scalar=false, is_sparse=false)
    if !isopen(trexio.file)
        return false
    end
    
    if !haskey(trexio.file, group_name)
        return false
    end
    
    target_group = trexio.file[group_name]
    if is_scalar
        # Scalar values are stored as HDF5 attributes
        return _has_hdf5_attribute(target_group, attr_name)
    elseif is_sparse
        # Sparse arrays have _values suffix
        return _has_hdf5_dataset(target_group, attr_name * "_values")
    else
        # Array data stored as HDF5 datasets
        return _has_hdf5_dataset(target_group, attr_name)
    end
end

# TREXIO format version targeted by this writer. It must be a parseable
# `MAJOR.MINOR.PATCH` string: the official TREXIO library parses
# `metadata_package_version` when opening a file and refuses files it cannot parse.
const TREXIO_FORMAT_VERSION = "2.6.1"

# Helper function to write basic metadata for new TREXIO files
function _write_basic_metadata(trexio_file::HDF5.File)
    # Create metadata group at root level
    if !haskey(trexio_file, "metadata")
        metadata_group = create_group(trexio_file, "metadata")
    else
        metadata_group = trexio_file["metadata"]
    end

    # Write unsafe flag in metadata following TREXIO specification.
    # TREXIO stores scalar numerical values as HDF5 attributes, not datasets, and
    # names every object <group>_<attribute> (here: "metadata_unsafe").
    if _has_hdf5_attribute(metadata_group, "metadata_unsafe")
        delete_attribute(metadata_group, "metadata_unsafe")
    end
    HDF5.write_attribute(metadata_group, "metadata_unsafe", Int64(1))

    # Stamp a parseable package version (ASCII, fixed-length) unless one is already
    # present, so the file can be opened by the official TREXIO library. An explicit
    # `trexio_write_metadata_package_version` call later overrides this default.
    if !_has_hdf5_attribute(metadata_group, "metadata_package_version")
        _write_string_attribute(metadata_group, "metadata_package_version", TREXIO_FORMAT_VERSION)
    end
end

# ============================================================================
# Sparse Array Support (TREXIO format uses _indices/_values datasets)
# ============================================================================

"""
    _write_sparse_array(group, name, indices, values; compress_indices=false, size_max=0)

Write sparse array data in TREXIO format with separate _indices and _values datasets.
Values are stored as Float64.

If `compress_indices=true` and `size_max > 0`, indices are stored using the smallest
integer type that fits (matching official TREXIO behavior):
- size_max < 255: UInt8
- size_max < 65535: UInt16  
- otherwise: Int32

By default, indices are stored as Int32 for simplicity.
"""
function _write_sparse_array(group::HDF5.Group, name::String, indices::AbstractArray{<:Integer}, values::AbstractVector{Float64};
                             compress_indices::Bool=false, size_max::Int=0)
    indices_name = name * "_indices"
    values_name = name * "_values"
    
    # Delete existing if present
    if haskey(group, indices_name)
        delete_object(group, indices_name)
    end
    if haskey(group, values_name)
        delete_object(group, values_name)
    end
    
    # Convert indices - optionally compress based on size_max (official TREXIO behavior)
    if compress_indices && size_max > 0
        if size_max < 255
            indices_typed = convert(Array{UInt8}, indices)
        elseif size_max < 65535
            indices_typed = convert(Array{UInt16}, indices)
        else
            indices_typed = convert(Array{Int32}, indices)
        end
    else
        # Default: store as Int32 (always works, simpler)
        indices_typed = convert(Array{Int32}, indices)
    end
    
    # Flatten indices for storage
    indices_flat = vec(indices_typed)
    
    # Write as datasets
    group[indices_name] = indices_flat
    group[values_name] = values
end

"""
    _read_sparse_array(group, name, rank) -> (indices, values)

Read sparse array data from TREXIO format with _indices and _values datasets.
Returns indices as Int32 array and values as Float64 vector.

Note: Official TREXIO may compress indices as UInt8 or UInt16 depending on
the maximum index value. This function handles all cases and converts to Int32.
"""
function _read_sparse_array(group::HDF5.Group, name::String, rank::Int)
    indices_name = name * "_indices"
    values_name = name * "_values"
    
    if !haskey(group, values_name)
        return zeros(Int32, rank, 0), zeros(Float64, 0)
    end
    
    values = read(group[values_name])::Vector{Float64}
    
    # Read indices - official TREXIO may use UInt8, UInt16, or Int32 compression
    # We read as the stored type and convert to Int32
    indices_dset = group[indices_name]
    indices_raw = read(indices_dset)
    
    # Convert to Int32 regardless of storage type (handles UInt8, UInt16, Int32)
    indices_flat = convert(Vector{Int32}, vec(indices_raw))
    
    # Reshape indices: stored as flat array, reshape to (rank, n_elements)
    n_elements = length(values)
    if n_elements > 0
        indices = reshape(indices_flat, (rank, n_elements))
    else
        indices = zeros(Int32, rank, 0)
    end
    
    return indices, values
end

"""
    _has_sparse_array(group, name) -> Bool

Check if sparse array exists (checks for _values dataset).
"""
function _has_sparse_array(group::HDF5.Group, name::String)
    return haskey(group, name * "_values")
end

# ============================================================================
# TREXIO-compatible string storage
# ============================================================================
# The official TREXIO HDF5 back end stores strings as **ASCII** (not UTF-8):
# scalar strings as fixed-length, NUL-terminated HDF5 attributes and string
# arrays as variable-length datasets. HDF5.jl writes UTF-8 by default, which the
# TREXIO C reader cannot convert ("no appropriate function for conversion path"),
# so we build the ASCII datatypes explicitly to remain interoperable.

"""
    _ascii_string_dtype(nbytes) -> HDF5.Datatype

Build a NUL-terminated, ASCII HDF5 string datatype. `nbytes` is the storage size
in bytes (including the terminating NUL), or `HDF5.API.H5T_VARIABLE` for a
variable-length string.
"""
function _ascii_string_dtype(nbytes)
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, nbytes)
    HDF5.API.h5t_set_strpad(dt, HDF5.API.H5T_STR_NULLTERM)
    HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_ASCII)
    return dt
end

"""
    _write_string_attribute(group, name, s)

Write `s` as a fixed-length, NUL-terminated, ASCII HDF5 attribute (TREXIO scalar
string convention).
"""
function _write_string_attribute(group::HDF5.Group, name::AbstractString, s::AbstractString)
    if _has_hdf5_attribute(group, name)
        delete_attribute(group, name)
    end
    str = String(s)
    n = ncodeunits(str) + 1   # include terminating NUL
    dt = _ascii_string_dtype(n)
    buf = zeros(UInt8, n)
    copyto!(buf, codeunits(str))
    dspace = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))
    attr = HDF5.create_attribute(group, name, dt, dspace)
    GC.@preserve buf HDF5.API.h5a_write(attr, dt, pointer(buf))
    close(attr); close(dspace); close(dt)
    return nothing
end

"""
    _write_string_dataset(group, name, v)

Write `v` as a variable-length, ASCII HDF5 string dataset (TREXIO string-array
convention).
"""
function _write_string_dataset(group::HDF5.Group, name::AbstractString, v::AbstractVector{<:AbstractString})
    if haskey(group, name)
        delete_object(group, name)
    end
    dt = _ascii_string_dtype(HDF5.API.H5T_VARIABLE)
    dspace = HDF5.dataspace((length(v),))
    dset = HDF5.create_dataset(group, name, dt, dspace)
    owners = [Base.cconvert(Cstring, String(s)) for s in v]
    GC.@preserve owners begin
        buf = Cstring[Base.unsafe_convert(Cstring, o) for o in owners]
        HDF5.API.h5d_write(dset, dt, HDF5.API.H5S_ALL, HDF5.API.H5S_ALL, HDF5.API.H5P_DEFAULT, buf)
    end
    close(dset); close(dspace); close(dt)
    return nothing
end

# ============================================================================
# Generated TREXIO Functions System
# ============================================================================

"""
    TrexioField

Structure defining a TREXIO field with its properties for automatic function generation.

# Fields
- `group::String`: TREXIO group name (e.g., "nucleus", "mo", "basis")
- `attribute::String`: TREXIO attribute name within the group
- `type::DataType`: Julia data type (Int64, Float64, String)
- `dimensions::Vector{String}`: Data dimensionality (=0: scalar, =1: vector, etc); 
   each element is a string describing the size of the corresponding dimension (column-major)
- `description::String`: Human-readable description for documentation
- `sparse::Bool`: Indicates whether the field is a sparse array (key-arg, default: false)
- `violator::Bool`: Indicates whether the field violates the TREXIO standard (key-arg, default: false)
"""
struct TrexioField
    group::String
    attribute::String
    type::DataType
    dimensions::Vector{String}  # e.g., ["3", "nucleus.num"]
    description::String
    sparse::Bool
    violator::Bool
    function TrexioField(group, attribute, type, dimensions, description; sparse=false, violator=false)
        new(group, attribute, type, dimensions, description, sparse, violator)
    end
end

# Define all TREXIO fields according to https://trex-coe.github.io/trexio/trex.html

const SCALAR = String[]

# Standard TREXIO fields (auto-generated from trex.org)
include("trexio_standard_fields.jl")
# Non-standard field extensions (violate the TREXIO standard)
include("trexio_nonstandard_fields.jl")

# Combine all field definitions (standard + non-standard extensions)
const ALL_TREXIO_FIELDS = vcat(
    STANDARD_TREXIO_FIELDS,
    NONSTANDARD_TREXIO_FIELDS,
)

# Generate exports dynamically for all TREXIO fields
for field in ALL_TREXIO_FIELDS
    @eval export $(Symbol("trexio_write_$(field.group)_$(field.attribute)"))
    @eval export $(Symbol("trexio_read_$(field.group)_$(field.attribute)"))
    @eval export $(Symbol("trexio_has_$(field.group)_$(field.attribute)"))
end

"""
Generate documentation string for write functions.
"""
function generate_write_docstring(field::TrexioField)
    func_name = "trexio_write_$(field.group)_$(field.attribute)"
    
    # Use the description from the field structure
    description = field.description
    ndim = length(field.dimensions)
    # Determine the data type and format
    if ndim == 0
        # Scalar fields stored as HDF5 attributes (TREXIO standard)
        data_format = "stored as HDF5 attribute"
        type_str = "$(field.type)"
    elseif ndim == 1
        data_format = "as vector of $(field.type) values ($(field.dimensions[1]))"
        type_str = "Vector{$(field.type)}"
    else
        data_format = "in column-major format ($(join(field.dimensions,",")))"
        type_str = "Array{$(field.type), $(ndim)}"
    end
    if field.sparse
        data_format *= " (sparse: indices/values format)"
    end
    # Add a warning if the function violates the TREXIO standard
    warning = ""
    if field.violator
        warning *= """

        !!! warning
            The function `$func_name` violates the TREXIO standard.
        """
    end

    return """
    $func_name(trexio::TrexioFile, value::$type_str) -> TrexioExitCode

Write $description to TREXIO file $data_format.$warning
"""
end

"""
Generate documentation string for read functions.
"""
function generate_read_docstring(field::TrexioField)
    func_name = "trexio_read_$(field.group)_$(field.attribute)"
    
    # Use the description from the field structure
    description = field.description
    
    # Determine the return type
    ndim = length(field.dimensions)
    if ndim == 0
        type_str = "$(field.type)"
    elseif ndim == 1
        type_str = "Vector{$(field.type)}"
    elseif ndim == 2
        type_str = "Matrix{$(field.type)}"
    else
        type_str = "Array{$(field.type), $(ndim)}"
    end
    
    # Format information
    format_info = ""
    if ndim == 1
        format_info = " in vector format ($(field.dimensions[1]))"
    elseif ndim > 1
        format_info = " in column-major format ($(join(field.dimensions,",")))"
    end
    if field.sparse
        format_info *= " (sparse)"
    end
    # Add a warning if the function violates the TREXIO standard
    warning = ""
    if field.violator
        warning *= """

        !!! warning
            The function `$func_name` violates the TREXIO standard.
        """
    end
    
    return """
    $func_name(trexio::TrexioFile) -> ($type_str, TrexioExitCode)

Read $description from TREXIO file$format_info.

Returns `(dummy_info, TREXIO_HAS_NOT)` if not present, or `(dummy_info, error_code)` on failure.$warning
"""
end

"""
Generate documentation string for has functions.
"""
function generate_has_docstring(field::TrexioField)
    func_name = "trexio_has_$(field.group)_$(field.attribute)"
    
    # Use the description from the field structure
    description = field.description
    
    # Add a warning if the function violates the TREXIO standard
    warning = ""
    if field.violator
        warning *= """

        !!! warning
            The function `$func_name` violates the TREXIO standard.
        """
    end

    return """
    $func_name(trexio::TrexioFile) -> Bool

Check if $description exists in TREXIO file.$warning
"""
end

"""
Generate the function body for write functions.
Scalar fields are stored as HDF5 attributes (TREXIO standard).
Array fields are stored as HDF5 datasets.
Sparse fields use _indices/_values format.
Includes type and size validation.
"""
function generate_write_function(field::TrexioField)
    ndim = length(field.dimensions)
    # Official TREXIO names every HDF5 object <group>_<attribute> (e.g. "nucleus_charge").
    objname = field.group * "_" * field.attribute
    if ndim == 0
        # Scalar values - stored as HDF5 attributes (TREXIO standard).
        # Strings must be written as ASCII to stay readable by the TREXIO C library.
        write_scalar = field.type == String ?
            :(_write_string_attribute(group, $(objname), value)) :
            quote
                if _has_hdf5_attribute(group, $(objname))
                    delete_attribute(group, $(objname))
                end
                HDF5.write_attribute(group, $(objname), value)
            end
        return quote
            # Type validation for scalar values
            if !(value isa $(field.type))
                try
                    value = convert($(field.type), value)
                catch e
                    return TREXIO_INVALID_ARG_2
                end
            end

            group, status = _get_or_create_group(trexio, $(field.group))
            if isnothing(group) || status != TREXIO_SUCCESS
                return status
            end

            try
                $(write_scalar)
                return TREXIO_SUCCESS
            catch e
                @warn "$e"
                return TREXIO_FAILURE
            end
        end
    else
        # Arrays (1D, 2D, 3D, 4D, 6D, 8D) - validate type, dimensions, and size
        # Generate size validation based on field dimensions
        fixed_size_checks = []
        runtime_size_checks = []
        
        for (i, dim_str) in enumerate(field.dimensions)
            if occursin(".", dim_str)
                # Variable dimension - reference to another field (e.g., "nucleus.num")
                # Need runtime validation by reading the referenced field
                parts = split(dim_str, ".")
                ref_group = parts[1]
                # store the full HDF5 object name (<group>_<attribute>) for the referenced field
                ref_attr = ref_group * "_" * parts[2]
                push!(runtime_size_checks, (i, ref_group, ref_attr))
            else
                # fixed dimensions
                fixed_size = tryparse(Int, dim_str)
                if !isnothing(fixed_size)
                    push!(fixed_size_checks, :(size(value, $i) == $fixed_size))
                # else: just a description of the dimension, check not needed
                end
            end
        end
        
        if field.sparse
            # Sparse arrays: expect (indices, values) tuple or named tuple
            return quote
                # Sparse arrays: value should be a tuple (indices, values) or NamedTuple
                # indices: Array{Int32, 2} of shape (rank, n_elements)
                # values: Vector{Float64} of length n_elements
                local indices, values
                if value isa Tuple && length(value) == 2
                    indices, values = value
                elseif value isa NamedTuple && haskey(value, :indices) && haskey(value, :values)
                    indices, values = value.indices, value.values
                else
                    return TREXIO_INVALID_ARG_2
                end
                
                # Validate types
                if !(values isa AbstractVector{Float64})
                    return TREXIO_INVALID_ARG_2
                end
                if !(indices isa AbstractArray{<:Integer})
                    return TREXIO_INVALID_ARG_2
                end
                
                group, status = _get_or_create_group(trexio, $(field.group))
                if isnothing(group) || status != TREXIO_SUCCESS
                    return status
                end
                
                try
                    _write_sparse_array(group, $(objname), indices, Vector{Float64}(values))
                    return TREXIO_SUCCESS
                catch e
                    @warn "$e"
                    return TREXIO_FAILURE
                end
            end
        else
            # Dense arrays. String arrays need explicit ASCII storage for TREXIO.
            # `convert(Array, value)` materialises adjoints / transposes / strided
            # views into a contiguous Array (no-op for a plain Array); HDF5.jl cannot
            # write arrays with a non-`Array` stride.
            write_array = field.type == String ?
                :(_write_string_dataset(group, $(objname), value)) :
                quote
                    if haskey(group, $(objname))
                        delete_object(group, $(objname))
                    end
                    group[$(objname)] = convert(Array, value)
                end
            return quote
                # Type validation
                if !(value isa AbstractArray{$(field.type), $ndim})
                    return TREXIO_INVALID_ARG_2
                end
                
                # Fixed size validation
                $(if !isempty(fixed_size_checks)
                    quote
                        if !($(Expr(:&&, fixed_size_checks...)))
                            return TREXIO_INVALID_ARG_2
                        end
                    end
                else
                    quote end
                end)
                
                # Runtime size validation for fields referencing other fields
                $(if !isempty(runtime_size_checks)
                    quote
                        $(map(runtime_size_checks) do (dim_idx, ref_group, ref_attr)
                            quote
                                # Read the referenced field to get expected size
                                # In TREXIO, size fields must be written before dependent arrays
                                if !_has_attribute(trexio, $ref_group, $ref_attr, is_scalar=true)
                                    # Referenced field doesn't exist - this is an error
                                    return TREXIO_INVALID_ARG_2
                                end
                                
                                ref_group_obj, ref_status = _get_or_create_group(trexio, $ref_group)
                                if isnothing(ref_group_obj) || ref_status != TREXIO_SUCCESS
                                    return ref_status
                                end
                                
                                try
                                    # Read from HDF5 attribute (scalar dimension values)
                                    ref_value = HDF5.read_attribute(ref_group_obj, $ref_attr)
                                    expected_size = isa(ref_value, Array) ? first(ref_value) : ref_value
                                    if size(value, $dim_idx) != expected_size
                                        return TREXIO_INVALID_ARG_2
                                    end
                                catch e
                                    # Error reading the reference field
                                    return TREXIO_FAILURE
                                end
                            end
                        end...)
                    end
                else
                    quote end
                end)
                
                group, status = _get_or_create_group(trexio, $(field.group))
                if isnothing(group) || status != TREXIO_SUCCESS
                    return status
                end
                
                try
                    $(write_array)
                    return TREXIO_SUCCESS
                catch e
                    @warn "$e"
                    return TREXIO_FAILURE
                end
            end
        end
    end
end

"""
Generate the function body for read functions with type-stable returns.
Scalar fields are read from HDF5 attributes (TREXIO standard).
Array fields are read from HDF5 datasets.
Sparse fields use _indices/_values format.
"""
function generate_read_function(field::TrexioField)
    ndim = length(field.dimensions)
    # Official TREXIO names every HDF5 object <group>_<attribute> (e.g. "nucleus_charge").
    objname = field.group * "_" * field.attribute
    if ndim == 0
        # Scalar values stored as HDF5 attributes (TREXIO standard)
        # Handle all numeric types explicitly for type stability
        if field.type == Int || field.type == Int64
            default_val = 0
        elseif field.type == Float64
            default_val = 0.0
        elseif field.type == String
            default_val = ""
        else
            error("Unsupported field type: $(field.type)")
        end
        
        return quote
            if !_has_attribute(trexio, $(field.group), $(objname), is_scalar=true)
                return $(default_val), TREXIO_HAS_NOT
            end

            group, status = _get_or_create_group(trexio, $(field.group))
            if isnothing(group) || status != TREXIO_SUCCESS
                return $(default_val), status
            end

            try
                # Read from HDF5 attribute (TREXIO standard for scalars)
                value = HDF5.read_attribute(group, $(objname))
                val = convert($(field.type), value)::$(field.type)
                return val, TREXIO_SUCCESS
            catch e
                return $(default_val), TREXIO_FAILURE
            end
        end
    else
        # Array data
        if field.sparse
            # Sparse arrays: return (indices, values) tuple
            return quote
                if !_has_attribute(trexio, $(field.group), $(objname), is_sparse=true)
                    return (zeros(Int32, $(ndim), 0), zeros(Float64, 0)), TREXIO_HAS_NOT
                end

                group, status = _get_or_create_group(trexio, $(field.group))
                if isnothing(group) || status != TREXIO_SUCCESS
                    return (zeros(Int32, $(ndim), 0), zeros(Float64, 0)), status
                end

                try
                    indices, values = _read_sparse_array(group, $(objname), $(ndim))
                    return (indices, values), TREXIO_SUCCESS
                catch e
                    return (zeros(Int32, $(ndim), 0), zeros(Float64, 0)), TREXIO_FAILURE
                end
            end
        else
            # Dense arrays (1D, 2D, 3D, 4D, 6D, 8D)
            # Create appropriate empty array based on dimensionality
            if field.type == String
                default_val = fill("", ntuple(d->0, ndim))
            else
                default_val = zeros(field.type, ntuple(d->0, ndim))
            end
            return quote
                if !_has_attribute(trexio, $(field.group), $(objname))
                    return $(default_val), TREXIO_HAS_NOT
                end

                group, status = _get_or_create_group(trexio, $(field.group))
                if isnothing(group) || status != TREXIO_SUCCESS
                    return $(default_val), status
                end

                try
                    data = read(group[$(objname)])::Array{$(field.type), $(ndim)}
                    return data, TREXIO_SUCCESS
                catch e
                    return $(default_val), TREXIO_FAILURE
                end
            end
        end
    end
end

"""
Generate the function body for has functions.
"""
function generate_has_function(field::TrexioField)
    ndim = length(field.dimensions)
    is_scalar = ndim == 0
    is_sparse = field.sparse
    # Official TREXIO names every HDF5 object <group>_<attribute> (e.g. "nucleus_charge").
    objname = field.group * "_" * field.attribute
    return quote
        _has_attribute(trexio, $(field.group), $(objname), is_scalar=$(is_scalar), is_sparse=$(is_sparse))
    end
end

# Generate all the TREXIO functions using the field definitions
for field in ALL_TREXIO_FIELDS
    @eval begin
        # Generate write function
        @doc $(generate_write_docstring(field))
        function $(Symbol("trexio_write_$(field.group)_$(field.attribute)"))(trexio::TrexioFile, value)
        $(generate_write_function(field))
        end

        # Generate read function
        @doc $(generate_read_docstring(field))
        function $(Symbol("trexio_read_$(field.group)_$(field.attribute)"))(trexio::TrexioFile)
            $(generate_read_function(field))
        end

        # Generate has function
        @doc $(generate_has_docstring(field))
        function $(Symbol("trexio_has_$(field.group)_$(field.attribute)"))(trexio::TrexioFile)
            $(generate_has_function(field))
        end
    end
end

end # module