"""
    PSpaceOptions

Configuration options for P-space determinant selection and calculations.
Used to generate high-quality initial guess vectors and enhanced preconditioning.
"""
@kwdef mutable struct PSpaceOptions
  """ Maximum P-space size (typically 100-1000) """
  max_size::Int = 1000
  """ Maximum excitation level from HF reference (0=HF, 1=S, 2=SD, etc.) """
  max_excitation::Int = 4
  """ Energy cutoff for determinant inclusion """
  energy_threshold::Scalar = 5.0
  """ Selection method for P-space generation (:hybrid, :excitation, :energy, :hbci) """
  selection_method::Symbol = :hybrid
  """ Use Heat-Bath CI for P-space selection (overrides selection_method if true) """
  use_hbci::Bool = false
  """ HBCI selection threshold (epsilon_1) for P-space generation """
  hbci_epsilon::Scalar = 1e-4
  """ Enable setup for HBCI P-space generation """
  hbci_use_setup_phase::Bool = true
end

"""
    FCIOptions

FCI calculation options.
"""
@kwdef mutable struct FCIOptions
  """ Maximum number of iterations """
  max_iter::Int = 50
  """ Convergence tolerance for energy """
  conv_tol::Scalar = 1e-8
  """ Convergence tolerance for residual norm """
  res_tol::Scalar = 1e-6
  """ Number of roots to compute """
  n_roots::Int = 1
  """ Number of guess vectors to use """
  n_guess::Int = 2
  """ Maximum subspace size for Davidson diagonalization """
  subspace_size::Int = 8
  """ Level of printed output (0=none, 1=some, 2=detailed) """
  print_level::Int = 1
  """ P-space options """
  pspace_options::PSpaceOptions = PSpaceOptions()
  """ Whether to compute 1-RDMs after convergence """
  compute_rdms::Bool = true
  """ Whether to compute 2-RDM after convergence """
  compute_2rdm::Bool = false
  """ Use projected Jacobi-Davidson correction (prevents linear dependency) """
  jacobi_davidson::Bool = true
end
