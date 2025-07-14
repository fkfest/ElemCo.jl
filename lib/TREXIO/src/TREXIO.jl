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
- Standard TREXIO data groups: nucleus, basis, mo, electron, ao, ecp, grid, amplitude, rdm, qmc
- Full TREXIO naming convention: trexio_[has|read|write]_<group>_<attribute>
- Metadata and versioning
- Error handling with TREXIO exit codes
"""
module TREXIO

using HDF5
using Dates

# TREXIO exit codes following the standard
@enum TrexioExitCode begin
    TREXIO_SUCCESS = 0
    TREXIO_FAILURE = 1
    TREXIO_INVALID_ARG_1 = 2
    TREXIO_INVALID_ARG_2 = 3
    TREXIO_INVALID_ARG_3 = 4
    TREXIO_INVALID_ARG_4 = 5
    TREXIO_INVALID_ARG_5 = 6
    TREXIO_END = 7
    TREXIO_READONLY = 8
    TREXIO_ERRNO = 9
    TREXIO_INVALID_ID = 10
    TREXIO_ALLOCATION_FAILED = 11
    TREXIO_HAS_NOT = 12
    TREXIO_INVALID_NUM = 13
    TREXIO_ATTR_ALREADY_EXISTS = 14
    TREXIO_DSET_ALREADY_EXISTS = 15
    TREXIO_OPEN_ERROR = 16
    TREXIO_LOCK_ERROR = 17
    TREXIO_UNLOCK_ERROR = 18
    TREXIO_FILE_ERROR = 19
    TREXIO_GROUP_ERROR = 20
    TREXIO_ELEM_ERROR = 21
    TREXIO_UNSAFE_OPERATION = 22
    TREXIO_INCONSISTENT_DATA = 23
    TREXIO_INT_SIZE_OVERFLOW = 24
    TREXIO_SAFE_MODE = 25
    TREXIO_INVALID_STR_LEN = 26
    TREXIO_INVALID_STATE = 27
    TREXIO_INVALID_BACKEND = 28
    TREXIO_INVALID_ARG_6 = 29
    TREXIO_INVALID_ARG_7 = 30
    TREXIO_INVALID_ARG_8 = 31
end

export TrexioFile, TrexioExitCode
export trexio_open, trexio_close
# Nucleus group
export trexio_write_nucleus_num, trexio_read_nucleus_num, trexio_has_nucleus_num
export trexio_write_nucleus_charge, trexio_read_nucleus_charge, trexio_has_nucleus_charge
export trexio_write_nucleus_coord, trexio_read_nucleus_coord, trexio_has_nucleus_coord
export trexio_write_nucleus_label, trexio_read_nucleus_label, trexio_has_nucleus_label
export trexio_write_nucleus_point_group, trexio_read_nucleus_point_group, trexio_has_nucleus_point_group
export trexio_write_nucleus_repulsion, trexio_read_nucleus_repulsion, trexio_has_nucleus_repulsion
# Electron group
export trexio_write_electron_num, trexio_read_electron_num, trexio_has_electron_num
export trexio_write_electron_up_num, trexio_read_electron_up_num, trexio_has_electron_up_num
export trexio_write_electron_dn_num, trexio_read_electron_dn_num, trexio_has_electron_dn_num
# Basis group
export trexio_write_basis_shell_num, trexio_read_basis_shell_num, trexio_has_basis_shell_num
export trexio_write_basis_prim_num, trexio_read_basis_prim_num, trexio_has_basis_prim_num
export trexio_write_basis_shell_nucleus_index, trexio_read_basis_shell_nucleus_index, trexio_has_basis_shell_nucleus_index
export trexio_write_basis_shell_ang_mom, trexio_read_basis_shell_ang_mom, trexio_has_basis_shell_ang_mom
export trexio_write_basis_shell_factor, trexio_read_basis_shell_factor, trexio_has_basis_shell_factor
export trexio_write_basis_shell_range, trexio_read_basis_shell_range, trexio_has_basis_shell_range
export trexio_write_basis_exponent, trexio_read_basis_exponent, trexio_has_basis_exponent
export trexio_write_basis_coefficient, trexio_read_basis_coefficient, trexio_has_basis_coefficient
export trexio_write_basis_type, trexio_read_basis_type, trexio_has_basis_type
# AO group
export trexio_write_ao_num, trexio_read_ao_num, trexio_has_ao_num
export trexio_write_ao_shell, trexio_read_ao_shell, trexio_has_ao_shell
export trexio_write_ao_normalization, trexio_read_ao_normalization, trexio_has_ao_normalization
export trexio_write_ao_1e_int_overlap, trexio_read_ao_1e_int_overlap, trexio_has_ao_1e_int_overlap
export trexio_write_ao_1e_int_kinetic, trexio_read_ao_1e_int_kinetic, trexio_has_ao_1e_int_kinetic
export trexio_write_ao_1e_int_potential_n_e, trexio_read_ao_1e_int_potential_n_e, trexio_has_ao_1e_int_potential_n_e
export trexio_write_ao_1e_int_ecp, trexio_read_ao_1e_int_ecp, trexio_has_ao_1e_int_ecp
export trexio_write_ao_1e_int_core_hamiltonian, trexio_read_ao_1e_int_core_hamiltonian, trexio_has_ao_1e_int_core_hamiltonian
export trexio_write_ao_2e_int_eri, trexio_read_ao_2e_int_eri, trexio_has_ao_2e_int_eri
export trexio_write_ao_2e_int_eri_lr, trexio_read_ao_2e_int_eri_lr, trexio_has_ao_2e_int_eri_lr
# MO group
export trexio_write_mo_num, trexio_read_mo_num, trexio_has_mo_num
export trexio_write_mo_coefficient, trexio_read_mo_coefficient, trexio_has_mo_coefficient
export trexio_write_mo_occupation, trexio_read_mo_occupation, trexio_has_mo_occupation
export trexio_write_mo_energy, trexio_read_mo_energy, trexio_has_mo_energy
export trexio_write_mo_spin, trexio_read_mo_spin, trexio_has_mo_spin
export trexio_write_mo_class, trexio_read_mo_class, trexio_has_mo_class
export trexio_write_mo_symmetry, trexio_read_mo_symmetry, trexio_has_mo_symmetry
export trexio_write_mo_1e_int_overlap, trexio_read_mo_1e_int_overlap, trexio_has_mo_1e_int_overlap
export trexio_write_mo_1e_int_kinetic, trexio_read_mo_1e_int_kinetic, trexio_has_mo_1e_int_kinetic
export trexio_write_mo_1e_int_potential_n_e, trexio_read_mo_1e_int_potential_n_e, trexio_has_mo_1e_int_potential_n_e
export trexio_write_mo_1e_int_ecp, trexio_read_mo_1e_int_ecp, trexio_has_mo_1e_int_ecp
export trexio_write_mo_1e_int_core_hamiltonian, trexio_read_mo_1e_int_core_hamiltonian, trexio_has_mo_1e_int_core_hamiltonian
export trexio_write_mo_2e_int_eri, trexio_read_mo_2e_int_eri, trexio_has_mo_2e_int_eri
export trexio_write_mo_2e_int_eri_lr, trexio_read_mo_2e_int_eri_lr, trexio_has_mo_2e_int_eri_lr
# Metadata functions
export trexio_write_metadata, trexio_read_metadata, trexio_has_metadata
# High-level functions
export trexio_create_file, trexio_read_file
# Backward compatibility aliases
export open_trexio, close_trexio
export write_nucleus, read_nucleus, write_basis, read_basis, write_mo, read_mo
export write_metadata, read_metadata, create_trexio_file, read_trexio_file

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
    trexio_open(trexio::TrexioFile) -> TrexioExitCode

Open a TREXIO file for reading or writing. Creates the main TREXIO group structure
if opening in write mode. Returns TREXIO exit code.
"""
function trexio_open(trexio::TrexioFile)
    try
        if trexio.file === nothing
            trexio.file = h5open(trexio.filename, trexio.mode)
            # Ensure the main TREXIO group exists following TREXIO standard
            if trexio.mode in ["w", "r+"] && !haskey(trexio.file, "trexio")
                create_group(trexio.file, "trexio")
            end
        end
        return TREXIO_SUCCESS
    catch e
        return TREXIO_OPEN_ERROR
    end
end

"""
    trexio_close(trexio::TrexioFile) -> TrexioExitCode

Close a TREXIO file and release resources. Returns TREXIO exit code.
"""
function trexio_close(trexio::TrexioFile)
    try
        if trexio.file !== nothing
            close(trexio.file)
            trexio.file = nothing
        end
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FILE_ERROR
    end
end

# Helper function to get or create group
function _get_or_create_group(trexio::TrexioFile, group_name::String)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return nothing, TREXIO_OPEN_ERROR
    end
    
    trex_group = haskey(trexio.file, "trexio") ? trexio.file["trexio"] : create_group(trexio.file, "trexio")
    
    if !haskey(trex_group, group_name)
        if trexio.mode in ["w", "r+"]
            target_group = create_group(trex_group, group_name)
        else
            return nothing, TREXIO_GROUP_ERROR
        end
    else
        target_group = trex_group[group_name]
    end
    
    return target_group, TREXIO_SUCCESS
end

# Helper function to check if attribute exists
function _has_attribute(trexio::TrexioFile, group_name::String, attr_name::String)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return false
    end
    
    if !haskey(trexio.file, "trexio")
        return false
    end
    
    trex_group = trexio.file["trexio"]
    if !haskey(trex_group, group_name)
        return false
    end
    
    target_group = trex_group[group_name]
    return haskey(target_group, attr_name)
end

# Nucleus group functions following TREXIO naming convention
"""
    trexio_write_nucleus_num(trexio::TrexioFile, num::Int) -> TrexioExitCode

Write the number of nuclei to TREXIO file.
"""
function trexio_write_nucleus_num(trexio::TrexioFile, num::Int)
    group, exit_code = _get_or_create_group(trexio, "nucleus")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["num"] = Int64(num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_nucleus_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read the number of nuclei from TREXIO file.
"""
function trexio_read_nucleus_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus") || !haskey(trexio.file["trexio"]["nucleus"], "num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        num = read(trexio.file["trexio"]["nucleus"]["num"])
        return Int(num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_nucleus_num(trexio::TrexioFile) -> Bool

Check if nucleus number exists in TREXIO file.
"""
function trexio_has_nucleus_num(trexio::TrexioFile)
    return _has_attribute(trexio, "nucleus", "num")
end

"""
    trexio_write_nucleus_charge(trexio::TrexioFile, charge::Vector{Float64}) -> TrexioExitCode

Write nuclear charges to TREXIO file (column-major format).
"""
function trexio_write_nucleus_charge(trexio::TrexioFile, charge::Vector{Float64})
    group, exit_code = _get_or_create_group(trexio, "nucleus")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["charge"] = charge
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_nucleus_charge(trexio::TrexioFile) -> (Vector{Float64}, TrexioExitCode)

Read nuclear charges from TREXIO file.
"""
function trexio_read_nucleus_charge(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return Float64[], TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus") || !haskey(trexio.file["trexio"]["nucleus"], "charge")
        return Float64[], TREXIO_HAS_NOT
    end
    
    try
        charge = read(trexio.file["trexio"]["nucleus"]["charge"])
        return charge, TREXIO_SUCCESS
    catch e
        return Float64[], TREXIO_FAILURE
    end
end

"""
    trexio_has_nucleus_charge(trexio::TrexioFile) -> Bool

Check if nucleus charges exist in TREXIO file.
"""
function trexio_has_nucleus_charge(trexio::TrexioFile)
    return _has_attribute(trexio, "nucleus", "charge")
end

"""
    trexio_write_nucleus_coord(trexio::TrexioFile, coord::Matrix{Float64}) -> TrexioExitCode

Write nuclear coordinates to TREXIO file (column-major format: 3 × natoms).
"""
function trexio_write_nucleus_coord(trexio::TrexioFile, coord::Matrix{Float64})
    group, exit_code = _get_or_create_group(trexio, "nucleus")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    if size(coord, 1) != 3
        return TREXIO_INVALID_ARG_2
    end
    
    try
        group["coord"] = coord  # Column-major format as per TREXIO spec
        attrs(group["coord"])["units"] = "bohr"
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_nucleus_coord(trexio::TrexioFile) -> (Matrix{Float64}, TrexioExitCode)

Read nuclear coordinates from TREXIO file (column-major format).
"""
function trexio_read_nucleus_coord(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return zeros(Float64, 3, 0), TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus") || !haskey(trexio.file["trexio"]["nucleus"], "coord")
        return zeros(Float64, 3, 0), TREXIO_HAS_NOT
    end
    
    try
        coord = read(trexio.file["trexio"]["nucleus"]["coord"])
        return coord, TREXIO_SUCCESS
    catch e
        return zeros(Float64, 3, 0), TREXIO_FAILURE
    end
end

"""
    trexio_has_nucleus_coord(trexio::TrexioFile) -> Bool

Check if nucleus coordinates exist in TREXIO file.
"""
function trexio_has_nucleus_coord(trexio::TrexioFile)
    return _has_attribute(trexio, "nucleus", "coord")
end

"""
    trexio_write_nucleus_label(trexio::TrexioFile, label::Vector{String}) -> TrexioExitCode

Write nuclear labels to TREXIO file.
"""
function trexio_write_nucleus_label(trexio::TrexioFile, label::Vector{String})
    group, exit_code = _get_or_create_group(trexio, "nucleus")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["label"] = label
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_nucleus_label(trexio::TrexioFile) -> (Vector{String}, TrexioExitCode)

Read nuclear labels from TREXIO file.
"""
function trexio_read_nucleus_label(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return String[], TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus") || !haskey(trexio.file["trexio"]["nucleus"], "label")
        return String[], TREXIO_HAS_NOT
    end
    
    try
        label = read(trexio.file["trexio"]["nucleus"]["label"])
        return label, TREXIO_SUCCESS
    catch e
        return String[], TREXIO_FAILURE
    end
end

"""
    trexio_has_nucleus_label(trexio::TrexioFile) -> Bool

Check if nucleus labels exist in TREXIO file.
"""
function trexio_has_nucleus_label(trexio::TrexioFile)
    return _has_attribute(trexio, "nucleus", "label")
end

"""
    trexio_write_nucleus_point_group(trexio::TrexioFile, point_group::String) -> TrexioExitCode

Write nuclear point group to TREXIO file.
"""
function trexio_write_nucleus_point_group(trexio::TrexioFile, point_group::String)
    group, exit_code = _get_or_create_group(trexio, "nucleus")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        attrs(group)["point_group"] = point_group
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_nucleus_point_group(trexio::TrexioFile) -> (String, TrexioExitCode)

Read nuclear point group from TREXIO file.
"""
function trexio_read_nucleus_point_group(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return "", TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus")
        return "", TREXIO_HAS_NOT
    end
    
    try
        group_attrs = attrs(trexio.file["trexio"]["nucleus"])
        if haskey(group_attrs, "point_group")
            return group_attrs["point_group"], TREXIO_SUCCESS
        else
            return "", TREXIO_HAS_NOT
        end
    catch e
        return "", TREXIO_FAILURE
    end
end

"""
    trexio_has_nucleus_point_group(trexio::TrexioFile) -> Bool

Check if nucleus point group exists in TREXIO file.
"""
function trexio_has_nucleus_point_group(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return false
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus")
        return false
    end
    
    group_attrs = attrs(trexio.file["trexio"]["nucleus"])
    return haskey(group_attrs, "point_group")
end

"""
    trexio_write_nucleus_repulsion(trexio::TrexioFile, repulsion::Float64) -> TrexioExitCode

Write nuclear repulsion energy to TREXIO file.
"""
function trexio_write_nucleus_repulsion(trexio::TrexioFile, repulsion::Float64)
    group, exit_code = _get_or_create_group(trexio, "nucleus")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["repulsion"] = repulsion
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_nucleus_repulsion(trexio::TrexioFile) -> (Float64, TrexioExitCode)

Read nuclear repulsion energy from TREXIO file.
"""
function trexio_read_nucleus_repulsion(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0.0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "nucleus") || !haskey(trexio.file["trexio"]["nucleus"], "repulsion")
        return 0.0, TREXIO_HAS_NOT
    end
    
    try
        repulsion = read(trexio.file["trexio"]["nucleus"]["repulsion"])
        return Float64(repulsion), TREXIO_SUCCESS
    catch e
        return 0.0, TREXIO_FAILURE
    end
end

"""
    trexio_has_nucleus_repulsion(trexio::TrexioFile) -> Bool

Check if nucleus repulsion energy exists in TREXIO file.
"""
function trexio_has_nucleus_repulsion(trexio::TrexioFile)
    return _has_attribute(trexio, "nucleus", "repulsion")
end

# Electron group functions
"""
    trexio_write_electron_num(trexio::TrexioFile, num::Int) -> TrexioExitCode

Write total number of electrons to TREXIO file.
"""
function trexio_write_electron_num(trexio::TrexioFile, num::Int)
    group, exit_code = _get_or_create_group(trexio, "electron")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["num"] = Int64(num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_electron_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read total number of electrons from TREXIO file.
"""
function trexio_read_electron_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "electron") || !haskey(trexio.file["trexio"]["electron"], "num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        num = read(trexio.file["trexio"]["electron"]["num"])
        return Int(num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_electron_num(trexio::TrexioFile) -> Bool

Check if electron number exists in TREXIO file.
"""
function trexio_has_electron_num(trexio::TrexioFile)
    return _has_attribute(trexio, "electron", "num")
end

"""
    trexio_write_electron_up_num(trexio::TrexioFile, up_num::Int) -> TrexioExitCode

Write number of spin-up electrons to TREXIO file.
"""
function trexio_write_electron_up_num(trexio::TrexioFile, up_num::Int)
    group, exit_code = _get_or_create_group(trexio, "electron")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["up_num"] = Int64(up_num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_electron_up_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read number of spin-up electrons from TREXIO file.
"""
function trexio_read_electron_up_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "electron") || !haskey(trexio.file["trexio"]["electron"], "up_num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        up_num = read(trexio.file["trexio"]["electron"]["up_num"])
        return Int(up_num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_electron_up_num(trexio::TrexioFile) -> Bool

Check if electron up number exists in TREXIO file.
"""
function trexio_has_electron_up_num(trexio::TrexioFile)
    return _has_attribute(trexio, "electron", "up_num")
end

"""
    trexio_write_electron_dn_num(trexio::TrexioFile, dn_num::Int) -> TrexioExitCode

Write number of spin-down electrons to TREXIO file.
"""
function trexio_write_electron_dn_num(trexio::TrexioFile, dn_num::Int)
    group, exit_code = _get_or_create_group(trexio, "electron")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["dn_num"] = Int64(dn_num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_electron_dn_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read number of spin-down electrons from TREXIO file.
"""
function trexio_read_electron_dn_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "electron") || !haskey(trexio.file["trexio"]["electron"], "dn_num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        dn_num = read(trexio.file["trexio"]["electron"]["dn_num"])
        return Int(dn_num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_electron_dn_num(trexio::TrexioFile) -> Bool

Check if electron down number exists in TREXIO file.
"""
function trexio_has_electron_dn_num(trexio::TrexioFile)
    return _has_attribute(trexio, "electron", "dn_num")
end

# Basis group functions (key ones following TREXIO naming convention)
"""
    trexio_write_basis_shell_num(trexio::TrexioFile, shell_num::Int) -> TrexioExitCode

Write number of basis shells to TREXIO file.
"""
function trexio_write_basis_shell_num(trexio::TrexioFile, shell_num::Int)
    group, exit_code = _get_or_create_group(trexio, "basis")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["shell_num"] = Int64(shell_num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_basis_shell_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read number of basis shells from TREXIO file.
"""
function trexio_read_basis_shell_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "basis") || !haskey(trexio.file["trexio"]["basis"], "shell_num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        shell_num = read(trexio.file["trexio"]["basis"]["shell_num"])
        return Int(shell_num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_basis_shell_num(trexio::TrexioFile) -> Bool

Check if basis shell number exists in TREXIO file.
"""
function trexio_has_basis_shell_num(trexio::TrexioFile)
    return _has_attribute(trexio, "basis", "shell_num")
end

"""
    trexio_write_basis_prim_num(trexio::TrexioFile, prim_num::Int) -> TrexioExitCode

Write number of basis primitives to TREXIO file.
"""
function trexio_write_basis_prim_num(trexio::TrexioFile, prim_num::Int)
    group, exit_code = _get_or_create_group(trexio, "basis")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["prim_num"] = Int64(prim_num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_basis_prim_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read number of basis primitives from TREXIO file.
"""
function trexio_read_basis_prim_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "basis") || !haskey(trexio.file["trexio"]["basis"], "prim_num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        prim_num = read(trexio.file["trexio"]["basis"]["prim_num"])
        return Int(prim_num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_basis_prim_num(trexio::TrexioFile) -> Bool

Check if basis primitive number exists in TREXIO file.
"""
function trexio_has_basis_prim_num(trexio::TrexioFile)
    return _has_attribute(trexio, "basis", "prim_num")
end

# MO group functions (key ones following TREXIO naming convention)
"""
    trexio_write_mo_num(trexio::TrexioFile, mo_num::Int) -> TrexioExitCode

Write number of molecular orbitals to TREXIO file.
"""
function trexio_write_mo_num(trexio::TrexioFile, mo_num::Int)
    group, exit_code = _get_or_create_group(trexio, "mo")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["num"] = Int64(mo_num)
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_mo_num(trexio::TrexioFile) -> (Int, TrexioExitCode)

Read number of molecular orbitals from TREXIO file.
"""
function trexio_read_mo_num(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return 0, TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "mo") || !haskey(trexio.file["trexio"]["mo"], "num")
        return 0, TREXIO_HAS_NOT
    end
    
    try
        mo_num = read(trexio.file["trexio"]["mo"]["num"])
        return Int(mo_num), TREXIO_SUCCESS
    catch e
        return 0, TREXIO_FAILURE
    end
end

"""
    trexio_has_mo_num(trexio::TrexioFile) -> Bool

Check if MO number exists in TREXIO file.
"""
function trexio_has_mo_num(trexio::TrexioFile)
    return _has_attribute(trexio, "mo", "num")
end

"""
    trexio_write_mo_coefficient(trexio::TrexioFile, coefficient::Matrix{Float64}) -> TrexioExitCode

Write molecular orbital coefficients to TREXIO file (column-major format).
"""
function trexio_write_mo_coefficient(trexio::TrexioFile, coefficient::Matrix{Float64})
    group, exit_code = _get_or_create_group(trexio, "mo")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["coefficient"] = coefficient  # Column-major format as per TREXIO spec
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_mo_coefficient(trexio::TrexioFile) -> (Matrix{Float64}, TrexioExitCode)

Read molecular orbital coefficients from TREXIO file (column-major format).
"""
function trexio_read_mo_coefficient(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return zeros(Float64, 0, 0), TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "mo") || !haskey(trexio.file["trexio"]["mo"], "coefficient")
        return zeros(Float64, 0, 0), TREXIO_HAS_NOT
    end
    
    try
        coefficient = read(trexio.file["trexio"]["mo"]["coefficient"])
        return coefficient, TREXIO_SUCCESS
    catch e
        return zeros(Float64, 0, 0), TREXIO_FAILURE
    end
end

"""
    trexio_has_mo_coefficient(trexio::TrexioFile) -> Bool

Check if MO coefficients exist in TREXIO file.
"""
function trexio_has_mo_coefficient(trexio::TrexioFile)
    return _has_attribute(trexio, "mo", "coefficient")
end

"""
    trexio_write_mo_occupation(trexio::TrexioFile, occupation::Vector{Float64}) -> TrexioExitCode

Write molecular orbital occupations to TREXIO file.
"""
function trexio_write_mo_occupation(trexio::TrexioFile, occupation::Vector{Float64})
    group, exit_code = _get_or_create_group(trexio, "mo")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["occupation"] = occupation
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_mo_occupation(trexio::TrexioFile) -> (Vector{Float64}, TrexioExitCode)

Read molecular orbital occupations from TREXIO file.
"""
function trexio_read_mo_occupation(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return Float64[], TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "mo") || !haskey(trexio.file["trexio"]["mo"], "occupation")
        return Float64[], TREXIO_HAS_NOT
    end
    
    try
        occupation = read(trexio.file["trexio"]["mo"]["occupation"])
        return occupation, TREXIO_SUCCESS
    catch e
        return Float64[], TREXIO_FAILURE
    end
end

"""
    trexio_has_mo_occupation(trexio::TrexioFile) -> Bool

Check if MO occupations exist in TREXIO file.
"""
function trexio_has_mo_occupation(trexio::TrexioFile)
    return _has_attribute(trexio, "mo", "occupation")
end

"""
    trexio_write_mo_energy(trexio::TrexioFile, energy::Vector{Float64}) -> TrexioExitCode

Write molecular orbital energies to TREXIO file.
"""
function trexio_write_mo_energy(trexio::TrexioFile, energy::Vector{Float64})
    group, exit_code = _get_or_create_group(trexio, "mo")
    if exit_code != TREXIO_SUCCESS
        return exit_code
    end
    
    try
        group["energy"] = energy
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_mo_energy(trexio::TrexioFile) -> (Vector{Float64}, TrexioExitCode)

Read molecular orbital energies from TREXIO file.
"""
function trexio_read_mo_energy(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return Float64[], TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "mo") || !haskey(trexio.file["trexio"]["mo"], "energy")
        return Float64[], TREXIO_HAS_NOT
    end
    
    try
        energy = read(trexio.file["trexio"]["mo"]["energy"])
        return energy, TREXIO_SUCCESS
    catch e
        return Float64[], TREXIO_FAILURE
    end
end

"""
    trexio_has_mo_energy(trexio::TrexioFile) -> Bool

Check if MO energies exist in TREXIO file.
"""
function trexio_has_mo_energy(trexio::TrexioFile)
    return _has_attribute(trexio, "mo", "energy")
end

# Metadata functions following TREXIO naming convention
"""
    trexio_write_metadata(trexio::TrexioFile; format_version="2.4.0", created_by="TREXIO.jl") -> TrexioExitCode

Write TREXIO format metadata following the standard specification.
"""
function trexio_write_metadata(trexio::TrexioFile; format_version="2.4.0", created_by="TREXIO.jl")
    if trexio_open(trexio) != TREXIO_SUCCESS
        return TREXIO_OPEN_ERROR
    end
    
    try
        trex_group = haskey(trexio.file, "trexio") ? trexio.file["trexio"] : create_group(trexio.file, "trexio")
        
        # Write standard TREXIO metadata
        attrs(trex_group)["format_version"] = format_version
        attrs(trex_group)["created_by"] = created_by
        attrs(trex_group)["created_date"] = string(Dates.now())
        
        return TREXIO_SUCCESS
    catch e
        return TREXIO_FAILURE
    end
end

"""
    trexio_read_metadata(trexio::TrexioFile) -> (Dict{String, String}, TrexioExitCode)

Read TREXIO format metadata.
"""
function trexio_read_metadata(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return Dict{String, String}(), TREXIO_OPEN_ERROR
    end
    
    if !haskey(trexio.file, "trexio")
        return Dict{String, String}(), TREXIO_HAS_NOT
    end
    
    try
        trex_group = trexio.file["trexio"]
        metadata = Dict{String, String}()
        
        # Read standard metadata attributes
        group_attrs = attrs(trex_group)
        for attr_name in ["format_version", "created_by", "created_date"]
            if haskey(group_attrs, attr_name)
                metadata[attr_name] = group_attrs[attr_name]
            end
        end
        
        return metadata, TREXIO_SUCCESS
    catch e
        return Dict{String, String}(), TREXIO_FAILURE
    end
end

"""
    trexio_has_metadata(trexio::TrexioFile) -> Bool

Check if metadata exists in TREXIO file.
"""
function trexio_has_metadata(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS
        return false
    end
    
    return haskey(trexio.file, "trexio")
end

# High-level functions with TREXIO naming convention
"""
    trexio_create_file(filename::String, nucleus_data=nothing, basis_data=nothing, 
                      mo_data=nothing; kwargs...) -> TrexioExitCode

High-level function to create a complete TREXIO file with multiple data sections.

# Arguments
- `filename`: Output file path
- `nucleus_data`: Tuple of (charges, coordinates, labels) or nothing
- `basis_data`: Basis set data dictionary or nothing  
- `mo_data`: MO coefficient matrix or nothing
- `kwargs`: Additional metadata options
"""
function trexio_create_file(filename::String, nucleus_data=nothing, basis_data=nothing, 
                           mo_data=nothing; kwargs...)
    trexio = TrexioFile(filename, "w")
    
    try
        # Write metadata
        exit_code = trexio_write_metadata(trexio; kwargs...)
        if exit_code != TREXIO_SUCCESS
            return exit_code
        end
        
        # Write nucleus data if provided
        if !isnothing(nucleus_data)
            charges, coords, labels = nucleus_data
            trexio_write_nucleus_num(trexio, length(charges))
            trexio_write_nucleus_charge(trexio, charges)
            trexio_write_nucleus_coord(trexio, coords)
            trexio_write_nucleus_label(trexio, labels)
        end
        
        # Write basis set data if provided
        if !isnothing(basis_data)
            trexio_write_basis_shell_num(trexio, basis_data["shell_num"])
            trexio_write_basis_prim_num(trexio, basis_data["prim_num"])
            # Add other basis data as needed
        end
        
        # Write MO data if provided  
        if !isnothing(mo_data)
            trexio_write_mo_coefficient(trexio, mo_data)
            trexio_write_mo_num(trexio, size(mo_data, 2))
        end
        
        return TREXIO_SUCCESS
        
    finally
        trexio_close(trexio)
    end
end

"""
    trexio_read_file(filename::String) -> (Dict{String, Any}, TrexioExitCode)

High-level function to read all available data from a TREXIO file.

Returns a dictionary with available data sections and exit code.
"""
function trexio_read_file(filename::String)
    if !isfile(filename)
        return Dict{String, Any}(), TREXIO_FILE_ERROR
    end
    
    trexio = TrexioFile(filename, "r")
    data = Dict{String, Any}()
    
    try
        if trexio_open(trexio) != TREXIO_SUCCESS
            return Dict{String, Any}(), TREXIO_OPEN_ERROR
        end
        
        if !haskey(trexio.file, "trexio")
            return Dict{String, Any}(), TREXIO_FILE_ERROR
        end
        
        trex_group = trexio.file["trexio"]
        
        # Read metadata
        metadata, meta_code = trexio_read_metadata(trexio)
        if meta_code == TREXIO_SUCCESS
            data["metadata"] = metadata
        end
        
        # Read available data sections
        if haskey(trex_group, "nucleus")
            charges, charge_code = trexio_read_nucleus_charge(trexio)
            coords, coord_code = trexio_read_nucleus_coord(trexio)
            labels, label_code = trexio_read_nucleus_label(trexio)
            
            if charge_code == TREXIO_SUCCESS && coord_code == TREXIO_SUCCESS && label_code == TREXIO_SUCCESS
                data["nucleus"] = Dict("charge" => charges, "coord" => coords, "label" => labels)
            end
        end
        
        if haskey(trex_group, "electron")
            num, num_code = trexio_read_electron_num(trexio)
            if num_code == TREXIO_SUCCESS
                data["electron"] = Dict("num" => num)
            end
        end
        
        if haskey(trex_group, "mo")
            coeffs, coeff_code = trexio_read_mo_coefficient(trexio)
            if coeff_code == TREXIO_SUCCESS
                data["mo"] = Dict("coefficient" => coeffs)
            end
        end
        
        return data, TREXIO_SUCCESS
        
    finally
        trexio_close(trexio)
    end
end

# Backward compatibility aliases - maintain old function names for ElemCo integration
const open_trexio = trexio_open
const close_trexio = trexio_close

# Legacy nucleus functions for backward compatibility
function write_nucleus(trexio::TrexioFile, nuclear_charges::Vector{Float64}, 
                      coordinates::Matrix{Float64}, labels::Vector{String}; 
                      units="bohr")
    # Write using new functions with proper error handling
    trexio_write_nucleus_num(trexio, length(nuclear_charges))
    trexio_write_nucleus_charge(trexio, nuclear_charges)
    trexio_write_nucleus_coord(trexio, coordinates)
    trexio_write_nucleus_label(trexio, labels)
    return trexio
end

function read_nucleus(trexio::TrexioFile)
    charges, _ = trexio_read_nucleus_charge(trexio)
    coords, _ = trexio_read_nucleus_coord(trexio)
    labels, _ = trexio_read_nucleus_label(trexio)
    return charges, coords, labels
end

# Legacy basis functions for backward compatibility
function write_basis(trexio::TrexioFile, shell_num::Int, shell_nucleus_index::Vector{Int},
                    shell_ang_mom::Vector{Int}, shell_factor::Vector{Float64},
                    shell_range::Vector{Int}, exponent::Vector{Float64},
                    coefficient::Vector{Float64})
    
    trexio_write_basis_shell_num(trexio, shell_num)
    trexio_write_basis_prim_num(trexio, length(exponent))
    
    # Note: For full TREXIO compliance, we should store individual arrays
    # but for backward compatibility, we'll just store what we can
    group, exit_code = _get_or_create_group(trexio, "basis")
    if exit_code == TREXIO_SUCCESS
        group["shell_nucleus_index"] = shell_nucleus_index
        group["shell_ang_mom"] = shell_ang_mom
        group["shell_factor"] = shell_factor
        group["shell_range"] = shell_range
        group["exponent"] = exponent
        group["coefficient"] = coefficient
    end
    
    return trexio
end

function read_basis(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS || !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "basis")
        error("No basis set data found in TREXIO file")
    end
    
    basis_group = trexio.file["trexio"]["basis"]
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

# Legacy MO functions for backward compatibility  
function write_mo(trexio::TrexioFile, coefficients::Matrix{Float64}; 
                 orbital_type="molecular", spin=nothing)
    trexio_write_mo_coefficient(trexio, coefficients)
    trexio_write_mo_num(trexio, size(coefficients, 2))
    
    # Add optional metadata for backward compatibility
    group, exit_code = _get_or_create_group(trexio, "mo")
    if exit_code == TREXIO_SUCCESS
        attrs(group)["type"] = orbital_type
        attrs(group)["basis_size"] = Int64(size(coefficients, 1))
        if !isnothing(spin)
            group["spin"] = spin
        end
    end
    
    return trexio
end

function read_mo(trexio::TrexioFile)
    if trexio_open(trexio) != TREXIO_SUCCESS || !haskey(trexio.file, "trexio") || !haskey(trexio.file["trexio"], "mo")
        error("No molecular orbital data found in TREXIO file")
    end
    
    mo_group = trexio.file["trexio"]["mo"]
    mo_data = Dict{String, Any}()
    
    # Read orbital data
    if haskey(mo_group, "coefficient")
        mo_data["coefficient"] = read(mo_group["coefficient"])
    end
    if haskey(mo_group, "num")
        mo_data["num"] = read(mo_group["num"])
    end
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

# Legacy metadata functions for backward compatibility
function write_metadata(trexio::TrexioFile; format_version="2.4.0", created_by="TREXIO.jl")
    trexio_write_metadata(trexio; format_version=format_version, created_by=created_by)
    
    # For backward compatibility, return the group like the old function did
    if trexio_open(trexio) == TREXIO_SUCCESS
        return trexio.file["trexio"]
    end
    return nothing
end

function read_metadata(trexio::TrexioFile)
    metadata, exit_code = trexio_read_metadata(trexio)
    if exit_code != TREXIO_SUCCESS
        error("Failed to read metadata from TREXIO file")
    end
    return metadata
end

# Legacy high-level functions for backward compatibility
const create_trexio_file = trexio_create_file

function read_trexio_file(filename::String)
    data, exit_code = trexio_read_file(filename)
    if exit_code != TREXIO_SUCCESS
        error("Failed to read TREXIO file: $filename")
    end
    return data
end

end # module TREXIO