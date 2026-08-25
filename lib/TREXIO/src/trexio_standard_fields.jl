#
# Standard TREXIO field definitions — AUTO-GENERATED from trex.org
# DO NOT EDIT MANUALLY.  Re-generate with:  julia generate_fields.jl
#
# Source: https://trex-coe.github.io/trexio/trex.html
#

# 1. Metadata fields (stored in metadata group at root level)
const TREXIO_METADATA_FIELDS = [
    TrexioField("metadata", "code_num", Int, SCALAR, "Number of codes used to produce the file"),
    TrexioField("metadata", "code", String, ["metadata.code_num"], "Names of the codes used"),
    TrexioField("metadata", "author_num", Int, SCALAR, "Number of authors of the file"),
    TrexioField("metadata", "author", String, ["metadata.author_num"], "Names of the authors of the file"),
    TrexioField("metadata", "package_version", String, SCALAR, "TREXIO version used to produce the file"),
    TrexioField("metadata", "description", String, SCALAR, "Text describing the content of file"),
    TrexioField("metadata", "unsafe", Int, SCALAR, "1: true, 0: false"),
]

# 2.1 Nucleus (nucleus group)
const TREXIO_NUCLEUS_FIELDS = [
    TrexioField("nucleus", "num", Int, SCALAR, "Number of nuclei"),
    TrexioField("nucleus", "charge", Float64, ["nucleus.num"], "Charges of the nuclei"),
    TrexioField("nucleus", "coord", Float64, ["3", "nucleus.num"], "Coordinates of the atoms"),
    TrexioField("nucleus", "label", String, ["nucleus.num"], "Atom labels"),
    TrexioField("nucleus", "point_group", String, SCALAR, "Symmetry point group"),
    TrexioField("nucleus", "repulsion", Float64, SCALAR, "Nuclear repulsion energy"),
]

# 2.2 Cell (cell group)
const TREXIO_CELL_FIELDS = [
    TrexioField("cell", "a", Float64, ["3"], "First real space lattice vector"),
    TrexioField("cell", "b", Float64, ["3"], "Second real space lattice vector"),
    TrexioField("cell", "c", Float64, ["3"], "Third real space lattice vector"),
    TrexioField("cell", "g_a", Float64, ["3"], "First reciprocal space lattice vector"),
    TrexioField("cell", "g_b", Float64, ["3"], "Second reciprocal space lattice vector"),
    TrexioField("cell", "g_c", Float64, ["3"], "Third reciprocal space lattice vector"),
    TrexioField("cell", "two_pi", Int, SCALAR, "0 or 1. If two_pi=1, \$2\\pi\$ is included in the reciprocal vectors."),
]

# 2.3 Periodic boundary calculations (pbc group)
const TREXIO_PBC_FIELDS = [
    TrexioField("pbc", "periodic", Int, SCALAR, "1: true or 0: false"),
    TrexioField("pbc", "k_point_num", Int, SCALAR, "Number of \$k\$-points"),
    TrexioField("pbc", "k_point", Float64, ["3", "pbc.k_point_num"], "\$k\$-point sampling"),
    TrexioField("pbc", "k_point_weight", Float64, ["pbc.k_point_num"], "\$k\$-point weight"),
    TrexioField("pbc", "madelung", Float64, SCALAR, "Madelung correction of the Ewald probe charge method"),
]

# 2.4 Electron (electron group)
const TREXIO_ELECTRON_FIELDS = [
    TrexioField("electron", "num", Int, SCALAR, "Number of electrons"),
    TrexioField("electron", "up_num", Int, SCALAR, "Number of \\uparrow-spin electrons"),
    TrexioField("electron", "dn_num", Int, SCALAR, "Number of \\downarrow-spin electrons"),
]

# 2.5 Ground or excited states (state group)
const TREXIO_STATE_FIELDS = [
    TrexioField("state", "num", Int, SCALAR, "Number of states (including the ground state)"),
    TrexioField("state", "id", Int, SCALAR, "Index of the current state (0 is ground state)"),
    TrexioField("state", "energy", Float64, SCALAR, "Energy of the current state"),
    TrexioField("state", "current_label", String, SCALAR, "Label of the current state"),
    TrexioField("state", "label", String, ["state.num"], "Labels of all states"),
    TrexioField("state", "file_name", String, ["state.num"], "Names of the TREXIO files linked to the current one (i.e. containing data for other states)"),
]

# 3.1 Basis set (basis group)
const TREXIO_BASIS_FIELDS = [
    TrexioField("basis", "type", String, SCALAR, "Type of basis set: \"Gaussian\", \"Slater\", \"Numerical\" or \"PW\" for plane waves"),
    TrexioField("basis", "prim_num", Int, SCALAR, "Total number of primitives"),
    TrexioField("basis", "shell_num", Int, SCALAR, "Total number of shells"),
    TrexioField("basis", "nao_grid_num", Int, SCALAR, "Total number of grid points for numerical orbitals"),
    TrexioField("basis", "interp_coeff_cnt", Int, SCALAR, "Number of coefficients for the numerical orbital interpolator"),
    TrexioField("basis", "nucleus_index", Int, ["basis.shell_num"], "One-to-one correspondence between shells and atomic indices"),
    TrexioField("basis", "shell_ang_mom", Int, ["basis.shell_num"], "One-to-one correspondence between shells and angular momenta"),
    TrexioField("basis", "shell_factor", Float64, ["basis.shell_num"], "Normalization factor of each shell (\$\\mathcal{N}_s\$)"),
    TrexioField("basis", "r_power", Int, ["basis.shell_num"], "Power to which \$r\$ is raised (\$n_s\$)"),
    TrexioField("basis", "nao_grid_start", Int, ["basis.shell_num"], "Index of the first data point for a given numerical orbital"),
    TrexioField("basis", "nao_grid_size", Int, ["basis.shell_num"], "Number of data points per numerical orbital"),
    TrexioField("basis", "shell_index", Int, ["basis.prim_num"], "One-to-one correspondence between primitives and shell index"),
    TrexioField("basis", "exponent", Float64, ["basis.prim_num"], "Exponents of the primitives (\$\\gamma_{ks}\$)"),
    TrexioField("basis", "exponent_im", Float64, ["basis.prim_num"], "Imaginary part of the exponents of the primitives (\$\\gamma_{ks}\$)"),
    TrexioField("basis", "coefficient", Float64, ["basis.prim_num"], "Coefficients of the primitives (\$a_{ks}\$)"),
    TrexioField("basis", "coefficient_im", Float64, ["basis.prim_num"], "Imaginary part of the coefficients of the primitives (\$a_{ks}\$)"),
    TrexioField("basis", "oscillation_arg", Float64, ["basis.prim_num"], "Additional argument to have oscillating orbitals (\$\\beta_{ks}\$)"),
    TrexioField("basis", "oscillation_kind", String, SCALAR, "Kind of Oscillating function:\"Cos1\" or \"Cos2\""),
    TrexioField("basis", "prim_factor", Float64, ["basis.prim_num"], "Normalization coefficients for the primitives (\$f_{ks}\$)"),
    TrexioField("basis", "e_cut", Float64, SCALAR, "Energy cut-off for plane-wave calculations"),
    TrexioField("basis", "nao_grid_radius", Float64, ["basis.nao_grid_num"], "Radii of grid points for numerical orbitals"),
    TrexioField("basis", "nao_grid_phi", Float64, ["basis.nao_grid_num"], "Wave function values for numerical orbitals"),
    TrexioField("basis", "nao_grid_grad", Float64, ["basis.nao_grid_num"], "Radial gradient of numerical orbitals"),
    TrexioField("basis", "nao_grid_lap", Float64, ["basis.nao_grid_num"], "Laplacian of numerical orbitals"),
    TrexioField("basis", "interpolator_kind", String, SCALAR, "Kind of spline, e.g. \"Polynomial\""),
    TrexioField("basis", "interpolator_phi", Float64, ["basis.interp_coeff_cnt", "basis.nao_grid_num"], "Coefficients for numerical orbital interpolation function"),
    TrexioField("basis", "interpolator_grad", Float64, ["basis.interp_coeff_cnt", "basis.nao_grid_num"], "Coefficients for numerical orbital gradient interpolation function"),
    TrexioField("basis", "interpolator_lap", Float64, ["basis.interp_coeff_cnt", "basis.nao_grid_num"], "Coefficients for numerical orbital laplacian interpolation function"),
]

# 3.2 Effective core potentials (ecp group)
const TREXIO_ECP_FIELDS = [
    TrexioField("ecp", "max_ang_mom_plus_1", Int, ["nucleus.num"], "\$\\ell_{\\max}+1\$, one higher than the max angular momentum in the removed core orbitals"),
    TrexioField("ecp", "z_core", Int, ["nucleus.num"], "Number of core electrons to remove per atom"),
    TrexioField("ecp", "num", Int, SCALAR, "Total number of ECP functions for all atoms and all values of \$\\ell\$"),
    TrexioField("ecp", "ang_mom", Int, ["ecp.num"], "One-to-one correspondence between ECP items and the angular momentum \$\\ell\$"),
    TrexioField("ecp", "nucleus_index", Int, ["ecp.num"], "One-to-one correspondence between ECP items and the atom index"),
    TrexioField("ecp", "exponent", Float64, ["ecp.num"], "\$\\alpha_{A q \\ell}\$ all ECP exponents"),
    TrexioField("ecp", "coefficient", Float64, ["ecp.num"], "\$\\beta_{A q \\ell}\$ all ECP coefficients"),
    TrexioField("ecp", "power", Int, ["ecp.num"], "\$n_{A q \\ell}\$ all ECP powers"),
]

# 3.3 Numerical integration grid (grid group)
const TREXIO_GRID_FIELDS = [
    TrexioField("grid", "description", String, SCALAR, "Details about the used quadratures can go here"),
    TrexioField("grid", "rad_precision", Float64, SCALAR, "Radial precision parameter (not used in some schemes like Krack-Köster)"),
    TrexioField("grid", "num", Int, SCALAR, "Number of grid points"),
    TrexioField("grid", "max_ang_num", Int, SCALAR, "Maximum number of angular grid points (for pruning)"),
    TrexioField("grid", "min_ang_num", Int, SCALAR, "Minimum number of angular grid points (for pruning)"),
    TrexioField("grid", "coord", Float64, ["grid.num"], "Discretized coordinate space"),
    TrexioField("grid", "weight", Float64, ["grid.num"], "Grid weights according to a given partitioning (e.g. Becke)"),
    TrexioField("grid", "ang_num", Int, SCALAR, "Number of angular integration points (if used)"),
    TrexioField("grid", "ang_coord", Float64, ["grid.ang_num"], "Discretized angular space (if used)"),
    TrexioField("grid", "ang_weight", Float64, ["grid.ang_num"], "Angular grid weights (if used)"),
    TrexioField("grid", "rad_num", Int, SCALAR, "Number of radial integration points (if used)"),
    TrexioField("grid", "rad_coord", Float64, ["grid.rad_num"], "Discretized radial space (if used)"),
    TrexioField("grid", "rad_weight", Float64, ["grid.rad_num"], "Radial grid weights  (if used)"),
]

# 4.1 Atomic orbitals (ao group)
const TREXIO_AO_FIELDS = [
    TrexioField("ao", "cartesian", Int, SCALAR, "1: true, 0: false"),
    TrexioField("ao", "num", Int, SCALAR, "Total number of atomic orbitals"),
    TrexioField("ao", "shell", Int, ["ao.num"], "Basis set shell for each AO"),
    TrexioField("ao", "normalization", Float64, ["ao.num"], "Normalization factor \$\\mathcal{N}_i\$"),
]

# 4.1.1 One-electron integrals (ao_1e_int group)
const TREXIO_AO_1E_INT_FIELDS = [
    TrexioField("ao_1e_int", "overlap", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "kinetic", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{T}_e \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "potential_n_e", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{V}_{\\text{ne}} \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "ecp", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{V}_{\\text{ecp}} \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "core_hamiltonian", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{h} \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "dipole_x", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{\\mu}_x \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "dipole_y", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{\\mu}_y \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "dipole_z", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{\\mu}_z \\vert q \\rangle\$"),
    TrexioField("ao_1e_int", "overlap_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert q \\rangle\$ (imaginary part)"),
    TrexioField("ao_1e_int", "kinetic_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{T}_e \\vert q \\rangle\$   (imaginary part)"),
    TrexioField("ao_1e_int", "potential_n_e_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{V}_{\\text{ne}} \\vert q \\rangle\$  (imaginary part)"),
    TrexioField("ao_1e_int", "ecp_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{V}_{\\text{ECP}} \\vert q \\rangle\$  (imaginary part)"),
    TrexioField("ao_1e_int", "core_hamiltonian_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{h} \\vert q \\rangle\$ (imaginary part)"),
    TrexioField("ao_1e_int", "dipole_x_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{\\mu}_x \\vert q \\rangle\$ (imaginary part)"),
    TrexioField("ao_1e_int", "dipole_y_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{\\mu}_y \\vert q \\rangle\$ (imaginary part)"),
    TrexioField("ao_1e_int", "dipole_z_im", Float64, ["ao.num", "ao.num"], "\$\\langle p \\vert \\hat{\\mu}_z \\vert q \\rangle\$ (imaginary part)"),
]

# 4.1.2 Two-electron integrals (ao_2e_int group)
const TREXIO_AO_2E_INT_FIELDS = [
    TrexioField("ao_2e_int", "eri", Float64, fill("ao.num", 4), "Electron repulsion integrals", sparse=true),
    TrexioField("ao_2e_int", "eri_lr", Float64, fill("ao.num", 4), "Long-range electron repulsion integrals", sparse=true),
    TrexioField("ao_2e_int", "eri_cholesky_num", Int, SCALAR, "Number of Cholesky vectors for ERI"),
    TrexioField("ao_2e_int", "eri_cholesky", Float64, ["ao.num", "ao.num", "ao_2e_int.eri_cholesky_num"], "Cholesky decomposition of the ERI", sparse=true),
    TrexioField("ao_2e_int", "eri_lr_cholesky_num", Int, SCALAR, "Number of Cholesky vectors for long range ERI"),
    TrexioField("ao_2e_int", "eri_lr_cholesky", Float64, ["ao.num", "ao.num", "ao_2e_int.eri_lr_cholesky_num"], "Cholesky decomposition of the long range ERI", sparse=true),
]

# 4.2 Molecular orbitals (mo group)
const TREXIO_MO_FIELDS = [
    TrexioField("mo", "type", String, SCALAR, "Free text to identify the set of MOs (HF, Natural, Local, CASSCF, /etc/)"),
    TrexioField("mo", "num", Int, SCALAR, "Number of MOs"),
    TrexioField("mo", "coefficient", Float64, ["ao.num", "mo.num"], "MO coefficients"),
    TrexioField("mo", "coefficient_im", Float64, ["ao.num", "mo.num"], "MO coefficients (imaginary part)"),
    TrexioField("mo", "class", String, ["mo.num"], "Choose among: Core, Inactive, Active, Virtual, Deleted"),
    TrexioField("mo", "symmetry", String, ["mo.num"], "Symmetry in the point group"),
    TrexioField("mo", "occupation", Float64, ["mo.num"], "Occupation number"),
    TrexioField("mo", "energy", Float64, ["mo.num"], "For canonical MOs, corresponding eigenvalue"),
    TrexioField("mo", "spin", Int, ["mo.num"], "For UHF wave functions, 0 is \$\\alpha\$ and 1 is \$\\beta\$"),
    TrexioField("mo", "k_point", Int, ["mo.num"], "For periodic calculations, the \$k\$ point to which each MO belongs"),
]

# 4.2.1 One-electron integrals (mo_1e_int group)
const TREXIO_MO_1E_INT_FIELDS = [
    TrexioField("mo_1e_int", "overlap", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "kinetic", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{T}_e \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "potential_n_e", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{V}_{\\text{ne}} \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "ecp", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{V}_{\\text{ECP}} \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "core_hamiltonian", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{h} \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "dipole_x", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{\\mu}_x \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "dipole_y", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{\\mu}_y \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "dipole_z", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{\\mu}_z \\vert j \\rangle\$"),
    TrexioField("mo_1e_int", "overlap_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert j \\rangle\$ (imaginary part)"),
    TrexioField("mo_1e_int", "kinetic_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{T}_e \\vert j \\rangle\$   (imaginary part)"),
    TrexioField("mo_1e_int", "potential_n_e_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{V}_{\\text{ne}} \\vert j \\rangle\$  (imaginary part)"),
    TrexioField("mo_1e_int", "ecp_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{V}_{\\text{ECP}} \\vert j \\rangle\$  (imaginary part)"),
    TrexioField("mo_1e_int", "core_hamiltonian_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{h} \\vert j \\rangle\$ (imaginary part)"),
    TrexioField("mo_1e_int", "dipole_x_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{\\mu}_x \\vert j \\rangle\$ (imaginary part)"),
    TrexioField("mo_1e_int", "dipole_y_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{\\mu}_y \\vert j \\rangle\$ (imaginary part)"),
    TrexioField("mo_1e_int", "dipole_z_im", Float64, ["mo.num", "mo.num"], "\$\\langle i \\vert \\hat{\\mu}_z \\vert j \\rangle\$ (imaginary part)"),
]

# 4.2.2 Two-electron integrals (mo_2e_int group)
const TREXIO_MO_2E_INT_FIELDS = [
    TrexioField("mo_2e_int", "eri", Float64, fill("mo.num", 4), "Electron repulsion integrals", sparse=true),
    TrexioField("mo_2e_int", "eri_lr", Float64, fill("mo.num", 4), "Long-range electron repulsion integrals", sparse=true),
    TrexioField("mo_2e_int", "eri_cholesky_num", Int, SCALAR, "Number of Cholesky vectors for ERI"),
    TrexioField("mo_2e_int", "eri_cholesky", Float64, ["mo.num", "mo.num", "mo_2e_int.eri_cholesky_num"], "Cholesky decomposition of the ERI", sparse=true),
    TrexioField("mo_2e_int", "eri_lr_cholesky_num", Int, SCALAR, "Number of Cholesky vectors for long range ERI"),
    TrexioField("mo_2e_int", "eri_lr_cholesky", Float64, ["mo.num", "mo.num", "mo_2e_int.eri_lr_cholesky_num"], "Cholesky decomposition of the long range ERI", sparse=true),
]

# 5.1 Slater determinants (determinant group)
const TREXIO_DETERMINANT_FIELDS = [
    TrexioField("determinant", "num", Int, SCALAR, "Number of determinants"),
    TrexioField("determinant", "list", Int, ["determinant.num"], "List of determinants as integer bit fields"),
    TrexioField("determinant", "coefficient", Float64, ["determinant.num"], "Coefficients of the determinants from the CI expansion"),
]

# 5.2 Configuration state functions (csf group)
const TREXIO_CSF_FIELDS = [
    TrexioField("csf", "num", Int, SCALAR, "Number of CSFs"),
    TrexioField("csf", "coefficient", Float64, ["csf.num"], "Coefficients \$C_I\$ of the CSF expansion"),
    TrexioField("csf", "det_coefficient", Float64, ["determinant.num", "csf.num"], "Projection on the determinant basis", sparse=true),
]

# 5.3 Amplitudes (amplitude group)
const TREXIO_AMPLITUDE_FIELDS = [
    TrexioField("amplitude", "single", Float64, ["mo.num", "mo.num"], "Single excitation amplitudes", sparse=true),
    TrexioField("amplitude", "single_exp", Float64, ["mo.num", "mo.num"], "Exponentialized single excitation amplitudes", sparse=true),
    TrexioField("amplitude", "double", Float64, fill("mo.num", 4), "Double excitation amplitudes", sparse=true),
    TrexioField("amplitude", "double_exp", Float64, fill("mo.num", 4), "Exponentialized double excitation amplitudes", sparse=true),
    TrexioField("amplitude", "triple", Float64, fill("mo.num", 6), "Triple excitation amplitudes", sparse=true),
    TrexioField("amplitude", "triple_exp", Float64, fill("mo.num", 6), "Exponentialized triple excitation amplitudes", sparse=true),
    TrexioField("amplitude", "quadruple", Float64, fill("mo.num", 8), "Quadruple excitation amplitudes", sparse=true),
    TrexioField("amplitude", "quadruple_exp", Float64, fill("mo.num", 8), "Exponentialized quadruple excitation amplitudes", sparse=true),
]

# 5.4 Reduced density matrices (rdm group)
const TREXIO_RDM_FIELDS = [
    TrexioField("rdm", "1e", Float64, ["mo.num", "mo.num"], "One body density matrix"),
    TrexioField("rdm", "1e_up", Float64, ["mo.num", "mo.num"], "\\uparrow-spin component of the one body density matrix"),
    TrexioField("rdm", "1e_dn", Float64, ["mo.num", "mo.num"], "\\downarrow-spin component of the one body density matrix"),
    TrexioField("rdm", "1e_transition", Float64, ["mo.num", "mo.num", "state.num", "state.num"], "One-particle transition density matrices"),
    TrexioField("rdm", "2e", Float64, fill("mo.num", 4), "Two-body reduced density matrix (spin trace)", sparse=true),
    TrexioField("rdm", "2e_upup", Float64, fill("mo.num", 4), "\\uparrow\\uparrow component of the two-body reduced density matrix", sparse=true),
    TrexioField("rdm", "2e_dndn", Float64, fill("mo.num", 4), "\\downarrow\\downarrow component of the two-body reduced density matrix", sparse=true),
    TrexioField("rdm", "2e_updn", Float64, fill("mo.num", 4), "\\uparrow\\downarrow component of the two-body reduced density matrix", sparse=true),
    TrexioField("rdm", "2e_transition", Float64, ["mo.num", "mo.num", "mo.num", "mo.num", "state.num", "state.num"], "Two-particle transition density matrices", sparse=true),
    TrexioField("rdm", "2e_cholesky_num", Int, SCALAR, "Number of Cholesky vectors"),
    TrexioField("rdm", "2e_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_cholesky_num"], "Cholesky decomposition of the two-body RDM (spin trace)", sparse=true),
    TrexioField("rdm", "2e_upup_cholesky_num", Int, SCALAR, "Number of Cholesky vectors"),
    TrexioField("rdm", "2e_upup_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_upup_cholesky_num"], "Cholesky decomposition of the two-body RDM (\\uparrow\\uparrow)", sparse=true),
    TrexioField("rdm", "2e_dndn_cholesky_num", Int, SCALAR, "Number of Cholesky vectors"),
    TrexioField("rdm", "2e_dndn_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_dndn_cholesky_num"], "Cholesky decomposition of the two-body RDM (\\downarrow\\downarrow)", sparse=true),
    TrexioField("rdm", "2e_updn_cholesky_num", Int, SCALAR, "Number of Cholesky vectors"),
    TrexioField("rdm", "2e_updn_cholesky", Float64, ["mo.num", "mo.num", "rdm.2e_updn_cholesky_num"], "Cholesky decomposition of the two-body RDM (\\uparrow\\downarrow)", sparse=true),
]

# 6.1 Jastrow factor (jastrow group)
const TREXIO_JASTROW_FIELDS = [
    TrexioField("jastrow", "type", String, SCALAR, "Type of Jastrow factor: CHAMP or Mu"),
    TrexioField("jastrow", "en_num", Int, SCALAR, "Number of Electron-nucleus parameters"),
    TrexioField("jastrow", "ee_num", Int, SCALAR, "Number of Electron-electron parameters"),
    TrexioField("jastrow", "een_num", Int, SCALAR, "Number of Electron-electron-nucleus parameters"),
    TrexioField("jastrow", "en", Float64, ["jastrow.en_num"], "Electron-nucleus parameters"),
    TrexioField("jastrow", "ee", Float64, ["jastrow.ee_num"], "Electron-electron parameters"),
    TrexioField("jastrow", "een", Float64, ["jastrow.een_num"], "Electron-electron-nucleus parameters"),
    TrexioField("jastrow", "en_nucleus", Int, ["jastrow.en_num"], "Nucleus relative to the eN parameter"),
    TrexioField("jastrow", "een_nucleus", Int, ["jastrow.een_num"], "Nucleus relative to the eeN parameter"),
    TrexioField("jastrow", "ee_scaling", Float64, SCALAR, "\$\\kappa\$ value in CHAMP Jastrow for electron-electron distances"),
    TrexioField("jastrow", "en_scaling", Float64, ["nucleus.num"], "\$\\kappa\$ value in CHAMP Jastrow for electron-nucleus distances"),
]

# 7. Quantum Monte Carlo data (qmc group)
const TREXIO_QMC_FIELDS = [
    TrexioField("qmc", "num", Int, SCALAR, "Number of 3N-dimensional points"),
    TrexioField("qmc", "point", Float64, ["3", "electron.num", "qmc.num"], "3N-dimensional points"),
    TrexioField("qmc", "psi", Float64, ["qmc.num"], "Wave function evaluated at the points"),
    TrexioField("qmc", "e_loc", Float64, ["qmc.num"], "Local energy evaluated at the points"),
]

# Combine all standard field definitions
const STANDARD_TREXIO_FIELDS = vcat(
    TREXIO_METADATA_FIELDS,
    TREXIO_NUCLEUS_FIELDS,
    TREXIO_CELL_FIELDS,
    TREXIO_PBC_FIELDS,
    TREXIO_ELECTRON_FIELDS,
    TREXIO_STATE_FIELDS,
    TREXIO_BASIS_FIELDS,
    TREXIO_ECP_FIELDS,
    TREXIO_GRID_FIELDS,
    TREXIO_AO_FIELDS,
    TREXIO_AO_1E_INT_FIELDS,
    TREXIO_AO_2E_INT_FIELDS,
    TREXIO_MO_FIELDS,
    TREXIO_MO_1E_INT_FIELDS,
    TREXIO_MO_2E_INT_FIELDS,
    TREXIO_DETERMINANT_FIELDS,
    TREXIO_CSF_FIELDS,
    TREXIO_AMPLITUDE_FIELDS,
    TREXIO_RDM_FIELDS,
    TREXIO_JASTROW_FIELDS,
    TREXIO_QMC_FIELDS,
)

