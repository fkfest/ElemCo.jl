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
end

export TrexioFile, TrexioExitCode
export trexio_open, trexio_close

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

# Helper function to check if attribute exists
function _has_attribute(trexio::TrexioFile, group_name, attr_name)
    if !isopen(trexio.file)
        return false
    end
    
    if !haskey(trexio.file, group_name)
        return false
    end
    
    target_group = trexio.file[group_name]
    return haskey(target_group, attr_name)
end

# Helper function to write basic metadata for new TREXIO files
function _write_basic_metadata(trexio_file::HDF5.File)
    # Create metadata group at root level
    if !haskey(trexio_file, "metadata")
        metadata_group = create_group(trexio_file, "metadata")
    else
        metadata_group = trexio_file["metadata"]
    end
    
    # Write unsafe flag in metadata following TREXIO specification
    metadata_group["unsafe"] = 1
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

# Metadata fields (stored in metadata group at root level)
const TREXIO_METADATA_FIELDS = [
    TrexioField("metadata", "code_num", Int, SCALAR, "number of codes used to produce the file"),
    TrexioField("metadata", "code", String, ["metadata.code_num"], "names of the codes used"),
    TrexioField("metadata", "author_num", Int, SCALAR, "number of authors of the file"),
    TrexioField("metadata", "author", String, ["metadata.author_num"], "names of the authors of the file"),
    TrexioField("metadata", "package_version", String, SCALAR, "TREXIO version used to produce the file"),
    TrexioField("metadata", "description", String, SCALAR, "text describing the content of file"),
    TrexioField("metadata", "unsafe", Int, SCALAR, "indicates whether the file has been previously opened with 'u' mode; 1: true, 0: false"),
]

# 2. System fields
## 2.1 Nucleus (nucleus group)
const TREXIO_NUCLEUS_FIELDS = [
    TrexioField("nucleus", "num", Int, SCALAR, "number of nuclei"),
    TrexioField("nucleus", "charge", Float64, ["nucleus.num"], "charges of the nuclei"),
    TrexioField("nucleus", "coord", Float64, ["3", "nucleus.num"], "coordinates of the atoms"),
    TrexioField("nucleus", "label", String, ["nucleus.num"], "atom labels"),
    TrexioField("nucleus", "point_group", String, SCALAR, "symmetry point group"),
    TrexioField("nucleus", "repulsion", Float64, SCALAR, "nuclear repulsion energy"),
]

## 2.2 Cell (cell group)
const TREXIO_CELL_FIELDS = [
    TrexioField("cell", "a", Float64, ["3"], "first real space lattice vector"),
    TrexioField("cell", "b", Float64, ["3"], "second real space lattice vector"),
    TrexioField("cell", "c", Float64, ["3"], "third real space lattice vector"),
    TrexioField("cell", "g_a", Float64, ["3"], "first reciprocal space lattice vector"),
    TrexioField("cell", "g_b", Float64, ["3"], "second reciprocal space lattice vector"),
    TrexioField("cell", "g_c", Float64, ["3"], "third reciprocal space lattice vector"),
    TrexioField("cell", "two_pi", Int, SCALAR, "0 or 1; if two_pi=1, 2π is included in the reciprocal vectors"),
]

## 2.3 Periodic boundary calculations (pbc group)
const TREXIO_PBC_FIELDS = [
    TrexioField("pbc", "periodic", Int, SCALAR, "1: true or 0: false"),
    TrexioField("pbc", "k_point_num", Int, SCALAR, "number of k-points"),
    TrexioField("pbc", "k_point", Float64, ["3", "pbc.k_point_num"], "k-point sampling"),
    TrexioField("pbc", "k_point_weight", Float64, ["pbc.k_point_num"], "k-point weights"),
    TrexioField("pbc", "madelung", Float64, SCALAR, "Madelung correction of the Ewald probe charge method"),
]

## 2.4 Electron (electron group)
const TREXIO_ELECTRON_FIELDS = [
    TrexioField("electron", "num", Int, SCALAR, "number of electrons"),
    TrexioField("electron", "up_num", Int, SCALAR, "number of spin-up electrons"),
    TrexioField("electron", "dn_num", Int, SCALAR, "number of spin-down electrons"),
]

## 2.5 Ground or excited states (state group)
const TREXIO_STATE_FIELDS = [
    TrexioField("state", "num", Int, SCALAR, "number of states (including the ground state)"),
    TrexioField("state", "id", Int, SCALAR, "index of the current state (0 is ground state)"),
    TrexioField("state", "energy", Float64, SCALAR, "energy of the current state"),
    TrexioField("state", "current_label", String, SCALAR, "label of the current state"),
    TrexioField("state", "label", String, ["state.num"], "labels of all states"),
    TrexioField("state", "file_name", String, ["state.num"], "names of the TREXIO files linked to the current one (i.e. containing data for other states)"),
]

## 3.1 Basis set (basis group)
const TREXIO_BASIS_FIELDS = [
    TrexioField("basis", "type", String, SCALAR, "type of basis set: \"Gaussian\", \"Slater\", \"Numerical\" or \"PW\" for plane waves"),
    TrexioField("basis", "prim_num", Int, SCALAR, "total number of primitives"),
    TrexioField("basis", "shell_num", Int, SCALAR, "total number of shells"),
    TrexioField("basis", "nao_grid_num", Int, SCALAR, "total number of grid points for numerical orbitals"),
    TrexioField("basis", "interp_coeff_cnt", Int, SCALAR, "number of coefficients for the numerical orbital interpolator"),
    TrexioField("basis", "nucleus_index", Int, ["basis.shell_num"], "one-to-one correspondence between shells and atomic indices"),
    TrexioField("basis", "shell_ang_mom", Int, ["basis.shell_num"], "one-to-one correspondence between shells and angular momenta"),
    TrexioField("basis", "shell_factor", Float64, ["basis.shell_num"], "normalization factor for each shell (N_s)"),
    TrexioField("basis", "r_power", Int, ["basis.shell_num"], "power to which r is raised (N_s)"),
    TrexioField("basis", "nao_grid_start", Int, ["basis.shell_num"], "index of the first data point for a given numerical orbital"),
    TrexioField("basis", "nao_grid_size", Int, ["basis.shell_num"], "number of data points per numerical orbital"),
    TrexioField("basis", "shell_index", Int, ["basis.prim_num"], "one-to-one correspondence between primitives and shell index"),
    TrexioField("basis", "exponent", Float64, ["basis.prim_num"], "exponents of the primitives (\\gamma_ks)"),
    TrexioField("basis", "exponent_im", Float64, ["basis.prim_num"], "imaginary part of the exponents of the primitives (\\gamma_ks)"),
    TrexioField("basis", "coefficient", Float64, ["basis.prim_num"], "coefficients of the primitives (a_ks)"),
    TrexioField("basis", "coefficient_im", Float64, ["basis.prim_num"], "imaginary part of the coefficients of the primitives (a_ks)"),
    TrexioField("basis", "oscillation_arg", Float64, ["basis.prim_num"], "additional argument to have oscillating orbitals (\\beta_ks)"),
    TrexioField("basis", "oscillation_kind", String, SCALAR, "kind of oscillating function: \"Cos1\" or \"Cos2\""),
    TrexioField("basis", "prim_factor", Float64, ["basis.prim_num"], "normalization coefficients for the primitives (f_ks)"),
    TrexioField("basis", "e_cut", Float64, SCALAR, "energy cut-off for plane-wave calculations"),
    TrexioField("basis", "nao_grid_radius", Float64, ["basis.nao_grid_num"], "radii of grid points for numerical orbitals"),
    TrexioField("basis", "nao_grid_phi", Float64, ["basis.nao_grid_num"], "wave function values for numerical orbitals"),
    TrexioField("basis", "nao_grid_grad", Float64, ["basis.nao_grid_num"], "radial gradient of numerical orbitals"),
    TrexioField("basis", "nao_grid_lap", Float64, ["basis.nao_grid_num"], "Laplacian of numerical orbitals"),
    TrexioField("basis", "interpolator_kind", String, SCALAR, "Kind of spline, e.g. \"Polynomial\""),
    TrexioField("basis", "interpolator_phi", Float64, ["basis.interp_coeff_cnt", "basis.nao_grid_num"], "coefficients for numerical orbital interpolation function"),
    TrexioField("basis", "interpolator_grad", Float64, ["basis.interp_coeff_cnt", "basis.nao_grid_num"], "coefficients for numerical orbital gradient interpolation function"),
    TrexioField("basis", "interpolator_lap", Float64, ["basis.interp_coeff_cnt", "basis.nao_grid_num"], "coefficients for numerical orbital laplacian interpolation function"),
    # non-standard fields
    TrexioField("basis", "name", String, SCALAR, "name of the basis set", violator=true),
]

## 3.2 Effective core potentials (ecp group)
const TREXIO_ECP_FIELDS = [
    TrexioField("ecp", "max_ang_mom_plus_1", Int, ["nucleus.num"], "l_max+1, one higher than the max angular momentum in the removed core orbitals"),
    TrexioField("ecp", "z_core", Int, ["nucleus.num"], "number of core electrons to remove per atom"),
    TrexioField("ecp", "num", Int, SCALAR, "total number of ECP functions for all atoms and all values of l"),
    TrexioField("ecp", "ang_mom", Int, ["ecp.num"], "one-to-one correspondence between ECP items and the angular momentum l"),
    TrexioField("ecp", "nucleus_index", Int, ["ecp.num"], "one-to-one correspondence between ECP items and the atom index"),
    TrexioField("ecp", "exponent", Float64, ["ecp.num"], "all ECP exponents α_A_ql"),
    TrexioField("ecp", "coefficient", Float64, ["ecp.num"], "all ECP coefficients β_A_ql"),
    TrexioField("ecp", "power", Int, ["ecp.num"], "all ECP powers n_A_ql"),
]

## 3.3 Numerical integration grid (grid group)
const TREXIO_GRID_FIELDS = [
    TrexioField("grid", "description", String, SCALAR, "details about the used quadratures can go here"),
    TrexioField("grid", "rad_precision", Float64, SCALAR, "radial precision parameter (not used in some schemes like Krack-Köster)"),
    TrexioField("grid", "num", Int, SCALAR, "number of grid points"),
    TrexioField("grid", "max_ang_num", Int, SCALAR, "maximum number of angular grid points (for pruning)"),
    TrexioField("grid", "min_ang_num", Int, SCALAR, "minimum number of angular grid points (for pruning)"),
    TrexioField("grid", "coord", Float64, ["grid.num"], "discretized coordinate space"),
    TrexioField("grid", "weight", Float64, ["grid.num"], "grid weights according to a given partitioning (e.g. Becke)"),
    TrexioField("grid", "ang_num", Int, SCALAR, "number of angular integration points (if used)"),
    TrexioField("grid", "ang_coord", Float64, ["grid.ang_num"], "discretized angular space (if used)"),
    TrexioField("grid", "ang_weight", Float64, ["grid.ang_num"], "angular grid weights (if used)"),
    TrexioField("grid", "rad_num", Int, SCALAR, "number of radial integration points (if used)"),
    TrexioField("grid", "rad_coord", Float64, ["grid.rad_num"], "discretized radial space (if used)"),
    TrexioField("grid", "rad_weight", Float64, ["grid.rad_num"], "radial grid weights (if used)"),
]

# 4. Orbitals
## 4.1 Atomic orbitals (ao group)
const TREXIO_AO_FIELDS = [
    TrexioField("ao", "cartesian", Int, SCALAR, "1: true, 0: false"),
    TrexioField("ao", "num", Int, SCALAR, "total number of atomic orbitals"),
    TrexioField("ao", "shell", Int, ["ao.num"], "basis set shell for each AO"),
    TrexioField("ao", "normalization", Float64, ["ao.num"], "normalization factor N'_i"),
]

## 4.1.1 One-electron integrals (ao_1e_int group)
const TREXIO_AO_1E_INT_FIELDS = [
    TrexioField("ao_1e_int", "overlap", Float64, ["ao.num", "ao.num"], "overlap integrals ⟨p|q⟩"),
    TrexioField("ao_1e_int", "kinetic", Float64, ["ao.num", "ao.num"], "kinetic energy integrals ⟨p|T|q⟩"),
    TrexioField("ao_1e_int", "potential_n_e", Float64, ["ao.num", "ao.num"], "electron-nucleus potential integrals ⟨p|V_ne|q⟩"),
    TrexioField("ao_1e_int", "ecp", Float64, ["ao.num", "ao.num"], "effective core potential integrals ⟨p|V_ECP|q⟩"),
    TrexioField("ao_1e_int", "core_hamiltonian", Float64, ["ao.num", "ao.num"], "core Hamiltonian integrals ⟨p|h|q⟩"),
    TrexioField("ao_1e_int", "dipole_x", Float64, ["ao.num", "ao.num"], "dipole x component integrals ⟨p|μ_x|q⟩"),
    TrexioField("ao_1e_int", "dipole_y", Float64, ["ao.num", "ao.num"], "dipole y component integrals ⟨p|μ_y|q⟩"),
    TrexioField("ao_1e_int", "dipole_z", Float64, ["ao.num", "ao.num"], "dipole z component integrals ⟨p|μ_z|q⟩"),
    TrexioField("ao_1e_int", "overlap_im", Float64, ["ao.num", "ao.num"], "overlap integrals ⟨p|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "kinetic_im", Float64, ["ao.num", "ao.num"], "kinetic energy integrals ⟨p|T|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "potential_n_e_im", Float64, ["ao.num", "ao.num"], "electron-nucleus potential integrals ⟨p|V_ne|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "ecp_im", Float64, ["ao.num", "ao.num"], "effective core potential integrals ⟨p|V_ECP|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "core_hamiltonian_im", Float64, ["ao.num", "ao.num"], "core Hamiltonian integrals ⟨p|h|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "dipole_x_im", Float64, ["ao.num", "ao.num"], "dipole x component integrals ⟨p|μ_x|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "dipole_y_im", Float64, ["ao.num", "ao.num"], "dipole y component integrals ⟨p|μ_y|q⟩ (imaginary part)"),
    TrexioField("ao_1e_int", "dipole_z_im", Float64, ["ao.num", "ao.num"], "dipole z component integrals ⟨p|μ_z|q⟩ (imaginary part)"),
]

## 4.1.2 Two-electron integrals (ao_2e_int group)
const TREXIO_AO_2E_INT_FIELDS = [
    TrexioField("ao_2e_int", "eri", Float64, ["ao.num", "ao.num", "ao.num", "ao.num"], "electron repulsion integrals ⟨pq|rs⟩", sparse=true),
    TrexioField("ao_2e_int", "eri_lr", Float64, ["ao.num", "ao.num", "ao.num", "ao.num"], "long-range electron repulsion integrals", sparse=true),
    TrexioField("ao_2e_int", "eri_cholesky_num", Int, SCALAR, "number of Cholesky vectors for ERI"),
    TrexioField("ao_2e_int", "eri_cholesky", Float64, ["ao.num", "ao.num", "ao_2e_int.eri_cholesky_num"], "Cholesky decomposition of the ERI", sparse=true),
    TrexioField("ao_2e_int", "eri_lr_cholesky_num", Int, SCALAR, "number of Cholesky vectors for long range ERI"),
    TrexioField("ao_2e_int", "eri_lr_cholesky", Float64, ["ao.num", "ao.num", "ao_2e_int.eri_lr_cholesky_num"], "Cholesky decomposition of the long range ERI", sparse=true),
]

## 4.2 Molecular orbitals (mo group)
const TREXIO_MO_FIELDS = [
    TrexioField("mo", "type", String, SCALAR, "free text to identify the set of MOs (HF, Natural, Local, CASSCF, etc)"),
    TrexioField("mo", "num", Int, SCALAR, "number of MOs"),
    TrexioField("mo", "coefficient", Float64, ["ao.num", "mo.num"], "MO coefficients"),
    TrexioField("mo", "coefficient_im", Float64, ["ao.num", "mo.num"], "MO coefficients (imaginary part)"),
    TrexioField("mo", "class", String, ["mo.num"], "choose among: Core, Inactive, Active, Virtual, Deleted"),
    TrexioField("mo", "symmetry", String, ["mo.num"], "symmetry in the point group"),
    TrexioField("mo", "occupation", Float64, ["mo.num"], "occupation number"),
    TrexioField("mo", "energy", Float64, ["mo.num"], "for canonical MOs, corresponding eigenvalue"),
    TrexioField("mo", "spin", Int, ["mo.num"], "for UHF wave functions, 0 is ↑ and 1 is ↓"),
    TrexioField("mo", "k_point", Int, ["mo.num"], "for periodic calculations, the k point to which each MO belongs"),
]

## 4.2a Positron orbitals (po group) - non-standard, violator
const TREXIO_PO_FIELDS = [
    TrexioField("po", "type", String, SCALAR, "free text to identify the set of POs (HF, Natural, Local, CASSCF, etc)", violator=true),
    TrexioField("po", "num", Int, SCALAR, "number of POs", violator=true),
    TrexioField("po", "coefficient", Float64, ["ao.num", "po.num"], "PO coefficients", violator=true),
    TrexioField("po", "coefficient_im", Float64, ["ao.num", "po.num"], "PO coefficients (imaginary part)", violator=true),
    TrexioField("po", "class", String, ["po.num"], "choose among: Core, Inactive, Active, Virtual, Deleted", violator=true),
    TrexioField("po", "symmetry", String, ["po.num"], "symmetry in the point group", violator=true),
    TrexioField("po", "occupation", Float64, ["po.num"], "occupation number", violator=true),
    TrexioField("po", "energy", Float64, ["po.num"], "for canonical POs, corresponding eigenvalue", violator=true),
    TrexioField("po", "spin", Int, ["po.num"], "for UHF wave functions, 0 is ↑ and 1 is ↓", violator=true),
    TrexioField("po", "k_point", Int, ["po.num"], "for periodic calculations, the k point to which each PO belongs", violator=true),
]

## 4.2.1 One-electron integrals (mo_1e_int group)
const TREXIO_MO_1E_INT_FIELDS = [
    TrexioField("mo_1e_int", "overlap", Float64, ["mo.num", "mo.num"], "overlap integrals ⟨p|q⟩"),
    TrexioField("mo_1e_int", "kinetic", Float64, ["mo.num", "mo.num"], "kinetic energy integrals ⟨p|T|q⟩"),
    TrexioField("mo_1e_int", "potential_n_e", Float64, ["mo.num", "mo.num"], "electron-nucleus potential integrals ⟨p|V_ne|q⟩"),
    TrexioField("mo_1e_int", "ecp", Float64, ["mo.num", "mo.num"], "effective core potential integrals ⟨p|V_ECP|q⟩"),
    TrexioField("mo_1e_int", "core_hamiltonian", Float64, ["mo.num", "mo.num"], "core Hamiltonian integrals ⟨p|h|q⟩"),
    TrexioField("mo_1e_int", "dipole_x", Float64, ["mo.num", "mo.num"], "dipole x component integrals ⟨p|μ_x|q⟩"),
    TrexioField("mo_1e_int", "dipole_y", Float64, ["mo.num", "mo.num"], "dipole y component integrals ⟨p|μ_y|q⟩"),
    TrexioField("mo_1e_int", "dipole_z", Float64, ["mo.num", "mo.num"], "dipole z component integrals ⟨p|μ_z|q⟩"),
    TrexioField("mo_1e_int", "overlap_im", Float64, ["mo.num", "mo.num"], "overlap integrals ⟨p|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "kinetic_im", Float64, ["mo.num", "mo.num"], "kinetic energy integrals ⟨p|T|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "potential_n_e_im", Float64, ["mo.num", "mo.num"], "electron-nucleus potential integrals ⟨p|V_ne|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "ecp_im", Float64, ["mo.num", "mo.num"], "effective core potential integrals ⟨p|V_ECP|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "core_hamiltonian_im", Float64, ["mo.num", "mo.num"], "core Hamiltonian integrals ⟨p|h|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "dipole_x_im", Float64, ["mo.num", "mo.num"], "dipole x component integrals ⟨p|μ_x|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "dipole_y_im", Float64, ["mo.num", "mo.num"], "dipole y component integrals ⟨p|μ_y|q⟩ (imaginary part)"),
    TrexioField("mo_1e_int", "dipole_z_im", Float64, ["mo.num", "mo.num"], "dipole z component integrals ⟨p|μ_z|q⟩ (imaginary part)"),
]

## 4.2.2 Two-electron integrals (mo_2e_int group)
const TREXIO_MO_2E_INT_FIELDS = [
    TrexioField("mo_2e_int", "eri", Float64, ["mo.num", "mo.num", "mo.num", "mo.num"], "electron repulsion integrals ⟨pq|rs⟩", sparse=true),
    TrexioField("mo_2e_int", "eri_lr", Float64, ["mo.num", "mo.num", "mo.num", "mo.num"], "long-range electron repulsion integrals", sparse=true),
    TrexioField("mo_2e_int", "eri_cholesky_num", Int, SCALAR, "number of Cholesky vectors for ERI"),
    TrexioField("mo_2e_int", "eri_cholesky", Float64, ["mo.num", "mo.num", "mo_2e_int.eri_cholesky_num"], "Cholesky decomposition of the ERI", sparse=true),
    TrexioField("mo_2e_int", "eri_lr_cholesky_num", Int, SCALAR, "number of Cholesky vectors for long range ERI"),
    TrexioField("mo_2e_int", "eri_lr_cholesky", Float64, ["mo.num", "mo.num", "mo_2e_int.eri_lr_cholesky_num"], "Cholesky decomposition of the long range ERI", sparse=true),
]

# 5. Multi-determinant information
## 5.1 Slater determinants (determinant group)
const TREXIO_DETERMINANT_FIELDS = [
    TrexioField("determinant", "num", Int, SCALAR, "number of determinants"),
    TrexioField("determinant", "list", Int, ["determinant.num"], "list of determinants as integer bit fields"),
    TrexioField("determinant", "coefficient", Float64, ["determinant.num"], "coefficients of the determinants from the CI expansion"),
]

## 5.2 Configuration state functions (csf group)
const TREXIO_CSF_FIELDS = [
    TrexioField("csf", "num", Int, SCALAR, "number of CSFs"),
    TrexioField("csf", "coefficient", Float64, ["csf.num"], "coefficients of the CSF expansion"),
    TrexioField("csf", "det_coefficient", Float64, ["determinant.num", "csf.num"], "projection on the determinant basis", sparse=true),
]

## 5.3 Amplitudes (amplitude group)
const TREXIO_AMPLITUDE_FIELDS = [
    TrexioField("amplitude", "single", Float64, fill("mo.num", 2), "single excitation amplitudes", sparse=true),
    TrexioField("amplitude", "single_exp", Float64, fill("mo.num", 2), "exponentialized single excitation amplitudes", sparse=true),
    TrexioField("amplitude", "double", Float64, fill("mo.num", 4), "double excitation amplitudes", sparse=true),
    TrexioField("amplitude", "double_exp", Float64, fill("mo.num", 4), "exponentialized double excitation amplitudes", sparse=true),
    TrexioField("amplitude", "triple", Float64, fill("mo.num", 6), "triple excitation amplitudes", sparse=true),
    TrexioField("amplitude", "triple_exp", Float64, fill("mo.num", 6), "exponentialized triple excitation amplitudes", sparse=true),
    TrexioField("amplitude", "quadruple", Float64, fill("mo.num", 8), "quadruple excitation amplitudes", sparse=true),
    TrexioField("amplitude", "quadruple_exp", Float64, fill("mo.num", 8), "exponentialized quadruple excitation amplitudes", sparse=true),
    # non-standard fields
    TrexioField("amplitude", "single_dense", Float64, ["v", "o"], "single excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_dense", Float64, ["v", "v", "o(o+1)/2"], "double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "single_up_dense", Float64, ["v", "o"], "↑-spin component of the single excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "single_dn_dense", Float64, ["V", "O"], "↓-spin component of the single excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_upup_dense", Float64, ["v(v-1)/2", "o(o-1)/2"], "↑↑-spin component of the double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_dndn_dense", Float64, ["V(V-1)/2", "O(O-1)/2"], "↓↓-spin component of the double excitation amplitudes (dense)", violator=true),
    TrexioField("amplitude", "double_updn_dense", Float64, ["v", "V", "o", "O"], "↑↓-spin component of the double excitation amplitudes (dense)", violator=true),
]

## 5.4 Reduced density matrices (rdm group)
const TREXIO_RDM_FIELDS = [
    TrexioField("rdm", "1e", Float64, fill("mo.num", 2), "one body density matrix"),
    TrexioField("rdm", "1e_up", Float64, fill("mo.num", 2), "↑-spin component of the one body density matrix"),
    TrexioField("rdm", "1e_dn", Float64, fill("mo.num", 2), "↓-spin component of the one body density matrix"),
    TrexioField("rdm", "1e_transition", Float64, ["mo.num", "mo.num", "state.num", "state.num"], "one-particle transition density matrices"),
    TrexioField("rdm", "2e", Float64, fill("mo.num", 4), "two-body reduced density matrix (spin trace)", sparse=true),
    TrexioField("rdm", "2e_upup", Float64, fill("mo.num", 4), "↑↑ component of the two-body reduced density matrix", sparse=true),
    TrexioField("rdm", "2e_dndn", Float64, fill("mo.num", 4), "↓↓ component of the two-body reduced density matrix", sparse=true),
    TrexioField("rdm", "2e_updn", Float64, fill("mo.num", 4), "↑↓ component of the two-body reduced density matrix", sparse=true),
    TrexioField("rdm", "2e_transition", Float64, vcat(fill("mo.num", 4), fill("state.num", 2)), "two-particle transition density matrices", sparse=true),
    TrexioField("rdm", "2e_cholesky_num", Int, SCALAR, "number of Cholesky vectors"),
    TrexioField("rdm", "2e_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_cholesky_num"], "Cholesky decomposition of the two-body RDM (spin trace)", sparse=true),
    TrexioField("rdm", "2e_upup_cholesky_num", Int, SCALAR, "number of Cholesky vectors"),
    TrexioField("rdm", "2e_upup_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_upup_cholesky_num"], "Cholesky decomposition of the two-body RDM (↑↑)", sparse=true),
    TrexioField("rdm", "2e_dndn_cholesky_num", Int, SCALAR, "number of Cholesky vectors"),
    TrexioField("rdm", "2e_dndn_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_dndn_cholesky_num"], "Cholesky decomposition of the two-body RDM (↓↓)", sparse=true),
    TrexioField("rdm", "2e_updn_cholesky_num", Int, SCALAR, "number of Cholesky vectors"),
    TrexioField("rdm", "2e_updn_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_updn_cholesky_num"], "Cholesky decomposition of the two-body RDM (↑↓)", sparse=true),
]

# 6. Correlation factors
## 6.1 Jastrow factor (jastrow group)
const TREXIO_JASTROW_FIELDS = [
    TrexioField("jastrow", "type", String, SCALAR, "type of Jastrow factor: CHAMP or Mu"),
    TrexioField("jastrow", "en_num", Int, SCALAR, "number of Electron-nucleus parameters"),
    TrexioField("jastrow", "ee_num", Int, SCALAR, "number of Electron-electron parameters"),
    TrexioField("jastrow", "een_num", Int, SCALAR, "number of Electron-electron-nucleus parameters"),
    TrexioField("jastrow", "en", Float64, ["jastrow.en_num"], "electron-nucleus parameters"),
    TrexioField("jastrow", "ee", Float64, ["jastrow.ee_num"], "electron-electron parameters"),
    TrexioField("jastrow", "een", Float64, ["jastrow.een_num"], "electron-electron-nucleus parameters"),
    TrexioField("jastrow", "en_nucleus", Int, ["jastrow.en_num"], "nucleus relative to the eN parameter"),
    TrexioField("jastrow", "een_nucleus", Int, ["jastrow.een_num"], "nucleus relative to the eeN parameter"),
    TrexioField("jastrow", "ee_scaling", Float64, SCALAR, "κ_ee value in CHAMP Jastrow for electron-electron distances"),
    TrexioField("jastrow", "en_scaling", Float64, ["nucleus.num"], "κ_α value in CHAMP Jastrow for electron-nucleus distances"),
]

# 7. Quantum Monte Carlo data (qmc group)
const TREXIO_QMC_FIELDS = [
    TrexioField("qmc", "num", Int, SCALAR, "number of 3N-dimensional points"),
    TrexioField("qmc", "point", Float64, ["3", "electron.num", "qmc.num"], "3N-dimensional points"),
    TrexioField("qmc", "psi", Float64, ["qmc.num"], "wave function evaluated at the points"),
    TrexioField("qmc", "e_loc", Float64, ["qmc.num"], "local energy evaluated at the points"),
]

# Combine all field definitions
const ALL_TREXIO_FIELDS = vcat(
    TREXIO_METADATA_FIELDS, TREXIO_NUCLEUS_FIELDS, TREXIO_CELL_FIELDS,
    TREXIO_PBC_FIELDS, TREXIO_ELECTRON_FIELDS, TREXIO_STATE_FIELDS,
    TREXIO_BASIS_FIELDS, TREXIO_ECP_FIELDS, TREXIO_GRID_FIELDS,
    TREXIO_AO_FIELDS, TREXIO_AO_1E_INT_FIELDS, TREXIO_AO_2E_INT_FIELDS,
    TREXIO_MO_FIELDS, TREXIO_MO_1E_INT_FIELDS, TREXIO_MO_2E_INT_FIELDS,
    TREXIO_DETERMINANT_FIELDS, TREXIO_CSF_FIELDS, TREXIO_AMPLITUDE_FIELDS,
    TREXIO_RDM_FIELDS, TREXIO_JASTROW_FIELDS, TREXIO_QMC_FIELDS,
    # non-standard fields
    TREXIO_PO_FIELDS,
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
        # All scalar fields stored as HDF5 datasets (not attributes)
        data_format = "stored as HDF5 dataset"
        type_str = "$(field.type)"
    elseif ndim == 1
        data_format = "as vector of $(field.type) values ($(field.dimensions[1]))"
        type_str = "Vector{$(field.type)}"
    else
        data_format = "in column-major format ($(join(field.dimensions,",")))"
        type_str = "Array{$(field.type), $(ndim)}"
    end
    if field.sparse
        data_format *= " (sparse)"
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
All fields are stored as HDF5 datasets, not attributes.
Includes type and size validation.
"""
function generate_write_function(field::TrexioField)
    ndim = length(field.dimensions)
    if ndim == 0
        # Scalar values - validate type
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
                if haskey(group, $(field.attribute))
                    delete_object(group, $(field.attribute))
                end
                group[$(field.attribute)] = value
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
                ref_attr = parts[2]
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
        
        return quote
            # sparse arrays not implemented
            @assert !$(field.sparse) "Sparse arrays are not supported in TREXIO.jl write functions yet. Use the dense versions instead."
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
                            if !_has_attribute(trexio, $ref_group, $ref_attr)
                                # Referenced field doesn't exist - this is an error
                                return TREXIO_INVALID_ARG_2
                            end
                            
                            ref_group_obj, ref_status = _get_or_create_group(trexio, $ref_group)
                            if isnothing(ref_group_obj) || ref_status != TREXIO_SUCCESS
                                return ref_status
                            end
                            
                            try
                                ref_value = read(ref_group_obj[$ref_attr])
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
                if haskey(group, $(field.attribute))
                    delete_object(group, $(field.attribute))
                end
                group[$(field.attribute)] = value
                return TREXIO_SUCCESS
            catch e
                @warn "$e"
                return TREXIO_FAILURE
            end
        end
    end
end

"""
Generate the function body for read functions with type-stable returns.
All fields are read from HDF5 datasets, not attributes.
"""
function generate_read_function(field::TrexioField)
    ndim = length(field.dimensions)
    if ndim == 0
        # Scalar values stored as 1-element datasets
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
            if !_has_attribute(trexio, $(field.group), $(field.attribute))
                return $(default_val), TREXIO_HAS_NOT
            end
            
            group, status = _get_or_create_group(trexio, $(field.group))
            if isnothing(group) || status != TREXIO_SUCCESS
                return $(default_val), status
            end
            
            try
                value = read(group[$(field.attribute)])
                return convert($(field.type), value), TREXIO_SUCCESS
            catch e
                return $(default_val), TREXIO_FAILURE
            end
        end
    else
        # Array data
        # (1D, 2D, 3D, 4D, 6D, 8D)
        # Create appropriate empty array based on dimensionality
        if field.type == String
            default_val = fill("", ntuple(d->0, ndim))
        else
            default_val = zeros(field.type, ntuple(d->0, ndim))
        end
        return quote
            # sparse arrays not implemented
            @assert !$(field.sparse) "Sparse arrays are not supported in TREXIO.jl read functions yet. Use the dense versions instead."
            
            if !_has_attribute(trexio, $(field.group), $(field.attribute))
                return $(default_val), TREXIO_HAS_NOT
            end
            
            group, status = _get_or_create_group(trexio, $(field.group))
            if isnothing(group) || status != TREXIO_SUCCESS
                return $(default_val), status
            end
            
            try
                data = read(group[$(field.attribute)])::Array{$(field.type), $(ndim)}
                return data, TREXIO_SUCCESS
            catch e
                return $(default_val), TREXIO_FAILURE
            end
        end
    end
end

"""
Generate the function body for has functions.
"""
function generate_has_function(field::TrexioField)
    return quote
        _has_attribute(trexio, $(field.group), $(field.attribute))
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