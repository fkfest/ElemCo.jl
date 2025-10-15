
"""
    HeatBathCIOptions

Configuration for Heat-Bath CI selection algorithm.

Based on Holmes et al. (2016): "Heat-Bath Configuration Interaction: An Efficient 
Selected Configuration Interaction Algorithm Inspired by Heat-Bath Sampling"
"""
struct HeatBathCIOptions
  target_selection::Int              # Target number of determinants to select
  epsilon_h::Float64                # HCI selection threshold (default: 1e-4)
  epsilon_p::Float64                # CIPSI selection threshold (default: 1e-4)
  tol::Float64                      # convergence threshold for Davidson (default: 1e-8)
  max_iterations::Int               # Maximum HBCI iterations (default: 10)
  verbose::Bool                     # Print iteration details
  use_setup_phase::Bool             # Perform setup (all singles+doubles from HF)
  compute_pt2::Bool                 # Compute PT2 perturbative correction
  epsilon_pt2::Float64              # Threshold for PT2 contributions (default: 1e-6)
  n_roots::Int                      # Number of states to compute (default: 1 = ground state only)
  use_small_space_guess::Bool       # Use small-space Hamiltonian for initial guess
  small_space_size::Int             # Size of small space (0 = auto: max(100, target÷10, 5*n_roots))
  small_space_method::Symbol        # Selection method: :hybrid (energy + excitation)
  
  function HeatBathCIOptions(;
    target_selection::Int = 10000,
    epsilon_h::Float64 = 5e-4,
    epsilon_p::Float64 = epsilon_h,
    tol::Float64 = 1e-6,
    max_iterations::Int = 50,
    verbose::Bool = true,
    use_setup_phase::Bool = true,
    compute_pt2::Bool = true,
    epsilon_pt2::Float64 = 1e-6,
    n_roots::Int = 1,
    use_small_space_guess::Bool = true,
    small_space_size::Int = 0,
    small_space_method::Symbol = :hybrid
  )
    new(target_selection, epsilon_h, epsilon_p, tol, max_iterations, 
        verbose, use_setup_phase,
        compute_pt2, epsilon_pt2, n_roots,
        use_small_space_guess, small_space_size, small_space_method)
  end
end
