"""
FCI Module

This module contains a translation of the FCI C++ code of G. Knizia and
extensions for selected CI and heat-bath CI.
"""
module FCI

using LinearAlgebra
using Printf
using StridedViews

# Export main types and functions
export FCIContext, FCIVector, FCIOptions, FCIDump
export run_fci!, make_diagonal_h!, contract_hamiltonian!
export read_fcidump
export OrbStringAdrTable, SubstResult
export apply_1e_op!, apply_2e_op!
# Davidson diagonalization
export davidson_fci!, davidson_selected_ci!, update_ci_vector!
# RDM functionality
export make_1rdms!, add_1rdm_for_spin!, make_2rdm!
# Basic types
export OrbPattern, Address, Scalar
# P-space functionality
export PSpaceOptions, PSpaceData, Determinant, setup_pspace!
export build_hf_reference_determinant, address_from_determinant, determinant_from_address
export n_data
# Small-space initial guess
export SmallSpaceResult, initialize_multistate_from_small_space
export select_small_space_determinants, build_small_space_hamiltonian
# Selected CI functionality
export SelectedCIContext, SelectedCIDeterminants, ExcitationInfo
export contract_hamiltonian_selected!, compute_matrix_element_direct, compute_diagonal_element
export setup_selected_ci_from_determinants!, setup_selected_ci_from_addresses!
export project_selected_to_full!, extract_full_to_selected!
export diagonalize_selected_space
# Heat-Bath CI functionality
export HeatBathCIOptions, HBCandidate, HBCISetupData
export run_heatbath_ci!, hartree_fock_determinant, setup_hbci!
export generate_connected_determinants!, generate_excitations_with_threshold!
export compute_heatbath_probabilities!, compute_heatbath_probabilities_multistate!, select_determinants_heatbath!
# PT2 correction functionality
export PT2Options, PT2Result, compute_pt2_correction!

include("fci_types.jl")
include("fci_options.jl")
include("fci_dump.jl")
include("fci_vec.jl")
include("fci_ops.jl")
include("fci_main.jl")
include("fci_selected_ci.jl")
include("fci_pspace.jl")
include("fci_davidson.jl")

end # module FCI
