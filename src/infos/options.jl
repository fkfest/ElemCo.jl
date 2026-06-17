# Options

"""
  Options for wavefunction/orbitals.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct WfOptions
  """`⟨-1⟩` spin magnetic quantum number times two (2×mₛ) of the system. """
  ms2::Int = -1
  """`⟨-1⟩` number of electrons. If < 0, the number of electrons is 
  read from the FCIDump file or guessed for the neutral system. """
  nelec::Int = -1
  """`⟨0⟩` charge of the system (relative to nelec/FCIDump/neutral system!). """
  charge::Int = 0
  """`⟨"wf.h5"⟩` filename for wavefunction dump (stored in TREXIO format). """
  dump::String = "wf.h5"
  """`⟨""⟩` filename to store the output wavefunction dump (stored in TREXIO format). If empty, `dump` will be used. """
  store::String = ""
  """`⟨""⟩` filename to store correlated natural orbitals and occupations (stored in TREXIO format).
  If non-empty, the correlated 1-RDM is still written to the usual `dump`/`store` file. """
  natorb::String = ""
  """`⟨""⟩` filename to read starting amplitudes from (TREXIO format). 
  If empty, amplitudes are read from `dump`. If provided, amplitudes (and MOs/basis) 
  are read from this file and projected to the current MO basis. """
  start::String = ""
  """`⟨0⟩` Number of positrons. """
  npositron::Int = 0
  """`⟨:auto⟩` core type for frozen-core approximation:
  - `:auto` freeze the orbitals tagged `Core` in the dump (e.g. from `@region`) if present,
    otherwise fall back to `:large`,
  - `:none` no frozen-core approximation,
  - `:small` semi-core orbitals correlated,
  - `:large` semi-core orbitals frozen.
  Setting `core` to anything other than `:auto` (or setting `freeze_nocc`) overrides the
  dump-derived core. """
  core::Symbol = :auto
  """`⟨-1⟩` number of occupied (core) orbitals to freeze (overwrites core). """
  freeze_nocc::Int = -1
  """`⟨-1⟩` number of virtual (highest) orbitals to freeze. `-1` (auto) drops the orbitals
  tagged `Deleted` in the dump (e.g. from `@region`); a value `≥ 0` overrides this and freezes
  exactly that many highest virtuals. """
  freeze_nvirt::Int = -1
  """`⟨0⟩` number of virtual (highest) positron orbitals to freeze. """
  freeze_nvirt_pos::Int = 0
  """`⟨"-"⟩` occupied α (or closed-shell) orbitals. 
  The occupation strings can be given as a `+` separated list, e.g. `occa = 1+2+3` or equivalently `1-3`. 
  Additionally, the spatial symmetry of the orbitals can be specified with the syntax `orb.sym`, e.g. `occa = "-5.1+-2.2+-4.3"`. """
  occa::String = "-"
  """`⟨"-"⟩` occupied β orbitals. 
  If `occb::String` is empty, the occupied β orbitals are the same as the occupied α orbitals (closed-shell case)."""
  occb::String = "-"
  """`⟨"-"⟩` active space.
  The active space is defined by the occupation string (cf. `occa`) or in the `(#elec, #orb)` format. """
  active::String = "-"
  """`⟨false⟩` ignore various errors in sanity checks. """
  ignore_error::Bool = false
  """`⟨5⟩` number of largest orbitals to print. """
  print_nlargest::Int = 5
  """`⟨0.1⟩` threshold for orbital coefficients to print. """
  print_thr::Float64 = 0.1
end


""" 
  Options for SCF calculation.

  $(TYPEDFIELDS)    
"""
@kwdef mutable struct ScfOptions
  """`⟨1.e-10⟩` convergence threshold. """
  thr::Float64 = 1.e-10
  """`⟨sqrt(thr)*0.1⟩` energy convergence threshold (used additionally to `thr`). """
  thren::Float64 = -1.0
  """`⟨50⟩` maximum number of iterations. """
  maxit::Int = 50
  """`⟨1.e-8⟩` tolerance for imaginary part of MO coefs (for biorthogonal). """
  imagtol::Float64 = 1.e-8
  """`⟨1.e-8⟩` threshold for removing linearly-dependent (redundant) orbitals.
  Eigenvectors of the AO overlap matrix with eigenvalues below this threshold are
  discarded via canonical orthogonalization. This makes HF robust for redundant basis
  sets (e.g. Cartesian basis sets with 6 instead of 5 `d` functions). """
  redthr::Float64 = 1.e-8
  """`⟨false⟩` direct calculation without storing integrals. """
  direct::Bool = false
  """`⟨:SAD⟩` orbital guess:
  - `:HCORE` from core Hamiltonian
  - `:SAD` from atomic densities
  - `:GWH` not implemented yet
  - `:ORB` from previous orbitals stored in dump file [`WfOptions.dump`](@ref ECInfos.WfOptions)
  """
  guess::Symbol = :SAD
  """`⟨:HCORE⟩` positron orbital guess. Only `:HCORE` is implemented. """
  guess_pos::Symbol = :HCORE
  """`⟨0.5⟩` damping factor for bisection search in augmented Hessian tuning. """
  bisecdamp::Float64 = 0.5
  """`⟨3⟩` maximum number of iterations for searching for lambda value to get a reasonalbe guess within trust radius for MCSCF. """
  maxit4lambda::Int = 3
  """`⟨:SO_SCI⟩` Hessian Type for MCSCF:
  - `:SO` Second Order Approximation
  - `:SCI` Super CI
  - `:SO_SCI` Second Order Approximation combing Super CI
  """
  HessianType::Symbol = :SO_SCI
  """`⟨:GRADIENT_SETPLUS⟩` Initial Vectors Type for MCSCF:
  - `:INHERIT` from last macro/micro iterations
  - `:GRADIENT_SET` b0 as [1,0,0,...], b1 as gradient
  - `:GRADIENT_SETPLUS` b0, b1 as GRADIENT_SET, b2 as zeros but 1 at the first closed-virtual rotation parameter
  """
  initVecType::Symbol = :GRADIENT_SETPLUS
  """ `⟨0.0⟩` Fermi-Dirac temperature for starting guess (at the moment works only for BO-HF). """
  temperature_guess::Float64 = 0.0
  """ `⟨0.1⟩` the threshold of davidson convergence residure norm scaled to norm of g the gradient, for MCSCF. """
  gamaDavScale::Float64 = 0.1
  """ `⟨true⟩` if true then use the original SO_SCI Hessian"""
  SO_SCI_origin = true
  """ `⟨0.8⟩` the trust region of sqrt(sum(x.^2)) should be [trustScale,1] * trust"""
  trustScale = 0.8
  """ `⟨1000.0⟩` the maximum number of lambda when adjusting the level shift"""
  lambdaMax = 1000.0
  """ `⟨1e-6⟩` the minmum convergence threshold for davidson algorithm"""
  davErrorMin = 1e-6
  """ `⟨200⟩` the size of initial Davidson projected matrix"""
  iniDavMatSize = 200
  """ `⟨0.7⟩` the shrink scale of trust region"""
  trustShrinkScale = 0.7
  """ `⟨1.2⟩` the expand scale of trust region"""
  trustExpandScale = 1.2
  """ `⟨0.25⟩` when energy quotient is lower than this value, the trust value should be smaller"""
  enerQuotientLowerBound = 0.25
  """ `⟨0.75⟩` when energy quotient is higher than this value, the trust value should be larger"""
  enerQuotientUpperBound = 0.75
  """`⟨false⟩` Generate pseudo-canonical basis instead of solving the SCF problem,
  i.e., build and block-diagonalize the Fock matrix without changing the Fermi level.
  At the moment, it works only for BO-HF."""
  pseudo::Bool = false
  """`⟨""⟩` Minimal basis set for SAD guess. If empty, use `"minao"` type from basis Dict, or `"minao"` basis as fallback. """
  minao::String = ""
end
  
""" 
  Options for Coupled-Cluster calculation.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct CcOptions
  """`⟨1.e-10⟩` convergence threshold. """
  thr::Float64 = 1.e-10
  """`⟨0.1⟩` energy convergence factor. The energy convergence threshold is `sqrt(thr) * conven`. """
  conven::Float64 = 0.1
  """`⟨50⟩` maximum number of iterations. """
  maxit::Int = 50
  """`⟨0.15⟩` level shift for singles. """
  shifts::Float64 = 0.15
  """`⟨0.2⟩` level shift for doubles. """
  shiftp::Float64 = 0.2
  """`⟨0.2⟩` level shift for triples. """
  shiftt::Float64 = 0.2
  """`⟨false⟩` calculate properties. """
  properties::Bool = false
  """`⟨1.e-5⟩` amplitude decomposition threshold. """
  ampsvdtol::Float64 = 1.e-5
  """`⟨1.e-2⟩` tightening amplitude decomposition factor 
      (for the two-step decomposition). """
  ampsvdfac::Float64 = 1.e-2
  """`⟨true⟩` use kext for doubles residual. """
  use_kext::Bool = true
  """`⟨false⟩` use plus/minus factorization in kext. 
  This kext has two times less FLOPs, but is less cache-friendly and can be slower for small systems. """
  use_pm_kext::Bool = false
  """`⟨false⟩` calculate dressed <vv|vv>. """
  calc_d_vvvv::Bool = false
  """`⟨false⟩` calculate dressed <vv|vo>. """
  calc_d_vvvo::Bool = false
  """`⟨false⟩` calculate dressed <vo|vv>. """
  calc_d_vovv::Bool = false
  """`⟨false⟩` calculate dressed <vv|oo>. """
  calc_d_vvoo::Bool = false
  """`⟨1.e-6⟩` threshold for checking Fock matrix diagonality in (T). If negative, no check is performed. """
  fock_diag_thr::Float64 = 1.e-6
  """`⟨true⟩` use density fitting in SVD-DC-CCSDT instead of the integral decomposition. """
  usedf::Bool = true
  """`⟨true⟩` use Cholesky decomposition in SVD-DC-CCSDT instead of SVD in the integral decomposition. """
  usecholesky::Bool = true
  """`⟨false⟩` calculate (T) for decomposition. """
  calc_t3_for_decomposition::Bool = false
  """`⟨false⟩` skip (T) calculation in SVD-DC-CCSDT. """
  skip_pert_t::Bool = false
  """`⟨true⟩` project out the T^iii contribution from the density matrix in decomposition in SVD-DC-CCSDT. """
  project_t3iii::Bool = true
  """`⟨false⟩` calculated ``V_{aX}^{iL}`` in SVD-DC-CCSDT using a projection to the X space as
  ``V_{XZ}^{L} U^{iZ}_{a}``. This is an additional approximation, which reduces the scaling of the 
  most expensive steps and is useful for large systems. """
  project_voXL::Bool = false
  """`⟨false⟩` use dense SVD for the amplitude decomposition in SVD-DC-CCSDT instead of LLAMA low-rank approximation.
  This is useful for small systems, but becomes unfeasible for large systems. """
  use_dense_decomposition::Bool = false
  """`⟨:combined⟩` type of space for project_voXL. Possible values are :combined, :symcombined, :triples, :full. """ 
  space4voXL::Symbol = :combined
  """`⟨0.0⟩` imaginary shift for denominator in doubles decomposition. """
  deco_ishiftp::Float64 = 0.0
  """`⟨0.0⟩` imaginary shift for denominator in triples decomposition. """
  deco_ishiftt::Float64 = 0.0
  """`⟨false⟩` use a projected exchange for contravariant doubles amplitudes in SVD-DCSD,
  ``\\tilde T_{XY} = U^{†a}_{iX} U^{†b}_{jY} \\tilde T^{ij}_{ab}``. """
  use_projx::Bool = false
  """`⟨false⟩` use full doubles amplitudes in SVD-DCSD. 
  The decomposition is used only for ``N^6`` scaling terms. """
  use_full_t2::Bool = false
  """`⟨2⟩` what to project in ``v_{ak}^{ci} T^{kj}_{cb}`` in SVD-DCSD:
  0: both, 1: amplitudes, 2: residual, 3: robust fit. """
  project_vovo_t2::Int = 2
  """`⟨false⟩` use MP2 amplitudes for decomposition in SVD-DCSD. 
  If `false`, one iteration of SVD-DCD is used for decomposition instead. """
  decompose_using_mp2::Bool = false
  """`⟨false⟩` decompose full doubles amplitudes in SVD-DCSD (slow). """
  decompose_full_doubles::Bool = false
  """`⟨"cc_amplitudes"⟩` main part of filename for start amplitudes. 
      For example, the singles amplitudes are read from `start*"_1"`. """
  start::String = "cc_amplitudes"
  """`⟨"cc_amplitudes"⟩` main part of filename to save amplitudes.
      For example, the singles amplitudes are saved to `save*"_1"`. """
  save::String = "cc_amplitudes"
  """`⟨"cc_multipliers"⟩` main part of filename for start Lagrange multipliers. 
      For example, the singles Lagrange multipliers are read from `start_lm*"_1"`. """
  start_lm::String = "cc_multipliers"
  """`⟨"cc_multipliers"⟩` main part of filename to save Lagrange multipliers.
      For example, the singles Lagrange multipliers are saved to `save_lm*"_1"`. """
  save_lm::String = "cc_multipliers"
  """`⟨0⟩` Don't use MP2 amplitudes as starting guess for the CC amplitudes. """
  nomp2::Int = 0
  """`⟨0.33⟩` Factor for same-spin component in SCS-MP2. """
  mp2_ssfac::Float64 = 0.33
  """`⟨1.2⟩` Factor for opposite-spin component in SCS-MP2. """
  mp2_osfac::Float64 = 1.2
  """`⟨0.0⟩` Factor for open-shell component in SCS-MP2. """
  mp2_ofac::Float64 = 0.0
  """`⟨1.3⟩` Factor for opposite-spin component in SOS-MP2. """
  mp2_sosfac::Float64 = 1.3
  """`⟨1.13⟩` Factor for same-spin component in SCS-CCSD. """
  ccsd_ssfac::Float64 = 1.13
  """`⟨1.27⟩` Factor for opposite-spin component in SCS-CCSD. """
  ccsd_osfac::Float64 = 1.27
  """`⟨0.0⟩` Factor for open-shell component in SCS-CCSD. """
  ccsd_ofac::Float64 = 0.0
  """`⟨1.15⟩` Factor for same-spin component in SCS-DCSD. """
  dcsd_ssfac::Float64 = 1.15
  """`⟨1.05⟩` Factor for opposite-spin component in SCS-DCSD. """
  dcsd_osfac::Float64 = 1.05
  """`⟨0.15⟩` Factor for open-shell component in SCS-DCSD. """
  dcsd_ofac::Float64 = 0.15
  """`⟨false⟩` ignore various errors in sanity checks. """
  ignore_error::Bool = false
  """`⟨false⟩` keep the orbitals after rotations over iterations of orbital optimizations in the OQV-CCD/DCD."""
  keepOQVorbitals::Bool = false
  """`⟨:maxdim⟩` pivot tolerance mode for LLAMA decomposition:
  - `:adaptive` use LLAMA's internal adaptive `tol/sqrt(m_eff)` pivot tolerance
  - `:maxdim` use `tol/sqrt(max(m,n))` as pivot tolerance (more robust for difficult cases, e.g., ghost atoms)
  If `ampsvd_pivotol > 0`, use that explicit value instead (overrides mode). """
  ampsvd_pivotol_mode::Symbol = :maxdim
  """`⟨0.0⟩` explicit pivot tolerance for LLAMA. If > 0, overrides `ampsvd_pivotol_mode`. """
  ampsvd_pivotol::Float64 = 0.0
  """`⟨true⟩` localize orbitals (IBO for occupied, orthogonal PAOs for virtual)
  before amplitude decomposition in SVD-DC methods.
  The localization rotation is applied only to the matrices entering `svd_decompose`,
  and the resulting U vectors are transformed back to the canonical basis. """
  localize::Bool = true
end

"""
  Options for orbital localization.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct LocOptions
  """`⟨true⟩` localize virtual orbitals using orthogonal PAOs (OPAO) in addition to occupied. """
  virtual::Bool = true
  """`⟨0⟩` Localization exponent: 0 for automatic (4 for IBO, 2 for PM), or set explicitly. """
  exponent::Int = 0
  """`⟨"ibo"⟩` Localization method: `"ibo"` (Intrinsic Bond Orbitals), `"pm"` (Pipek-Mezey with Mulliken charges), or `"boys"` (Foster-Boys). """
  method::String = "ibo"
  """`⟨""⟩` Minimal basis set for IAO construction. If empty, use `"minao"` type from basis Dict, or `"minao"` basis as fallback. """
  minao::String = ""
  """`⟨false⟩` Localize core orbitals among themselves (separately from valence). """
  localize_core::Bool = false
end

"""
  Options for region-fragment orbital selection.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct RegionOptions
  """`⟨:inclusive⟩` occupied-selection mode for centers passed directly to `@region`.
  - `:inclusive`: treat `@region(centers)` as additional inclusive centers
  - `:exclusive`: treat `@region(centers)` as additional exclusive centers
  """
  mode::Symbol = :inclusive
  """`⟨Symbol[]⟩` additional center labels selected with the inclusive occupied-orbital rule, e.g. `[:O]`.
      The orbitals with significant contributions from these centers are added to the fragment. """
  inclusive_centers::Vector{Symbol} = Symbol[]
  """`⟨Symbol[]⟩` additional center labels selected with the exclusive occupied-orbital rule, e.g. `[:O, :H1]`. 
      The orbitals with significant contributions exclusively from these centers are added to the fragment. """
  exclusive_centers::Vector{Symbol} = Symbol[]
  """`⟨:none⟩` π-space selection mode.
  - `:none`: use the default IBO/PAO region selection
  - `:occupied`: select occupied π orbitals and keep PAO-defined virtuals
  - `:both`: select both occupied and virtual π orbitals
  """
  pi::Symbol = :none
  """`⟨-1⟩` override for the total number of π electrons used by the PiOS counting model.
    Use `-1` to keep the automatic chemistry-based count. """
  pi_electrons::Int = -1
  """`⟨-1⟩` override for the number of occupied π orbitals to keep in restricted PiOS runs.
    Use `-1` to keep the full automatically determined occupied π space. """
  pi_occupied::Int = -1
  """`⟨-1⟩` override for the number of virtual π orbitals to keep in restricted `region.pi=:both` runs.
    Use `-1` to keep the full automatically determined virtual π space. """
  pi_virtual::Int = -1
  """`⟨:complement⟩` fragment virtual-space construction mode.
  - `:complement`: build antibonding-like virtual targets by projecting fragment IAOs into the virtual space, then augment them with support-atom OPAOs selected from accumulated fragment charge
  - `:support_opao`: use the support-atom OPAO construction directly
  """
  virtual::Symbol = :complement
  """`⟨0.2⟩` threshold for selecting localized occupied orbitals from the requested centers. """
  occ_charge_thr::Float64 = 0.2
  """`⟨0.2⟩` threshold for adding atoms to the PAO support of the selected fragment. """
  atom_charge_thr::Float64 = 0.2
  """`⟨false⟩` pseudo-canonicalize the selected fragment occupied and virtual subspaces. """
  pseudo::Bool = false
end

"""
  Option for FCI calculations.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct FCIOptions
  """`⟨50⟩` Maximum number of iterations """
  max_iter::Int = 50
  """`⟨0.1⟩` Level shift to improve convergence """
  shift::Float64 = 0.1
  """`⟨1e-8⟩` Convergence tolerance for energy """
  conv_tol::Float64 = 1e-8
  """`⟨1e-6⟩` Convergence tolerance for residual norm """
  res_tol::Float64 = 1e-6
  """`⟨false⟩` calculate properties such as dipole moments. """
  properties::Bool = false
  """`⟨1⟩` Number of states to compute """
  nstates::Int = 1
  """`⟨2⟩` Number of guess vectors to use """
  n_guess::Int = 2
  """`⟨10⟩` Maximum subspace size for Davidson diagonalization """
  subspace_size::Int = 10
  """`⟨true⟩` Compute 1-RDMs after convergence """
  compute_rdms::Bool = true
  """`⟨false⟩` Compute 2-RDM after convergence """
  compute_2rdm::Bool = false
  """`⟨true⟩` Use projected Jacobi-Davidson correction (prevents linear dependency) """
  jacobi_davidson::Bool = true
  """`⟨1000⟩` Maximum P-space size (typically 100-1000) """
  max_pspace_size::Int = 1000
  """`⟨:hybrid⟩` Selection method for P-space generation (:hybrid, :excitation, :energy, :ciphi) """
  pspace_selection_method::Symbol = :hybrid
  """`⟨4⟩` Maximum excitation level from HF reference (0=HF, 1=S, 2=SD, etc.) """
  max_pspace_excitation::Int = 4
  """`⟨5.0⟩` Energy cutoff for determinant inclusion """
  pspace_energy_threshold::Float64 = 5.0
  """`⟨1e-3⟩` CIPHI selection threshold (epsilon_h) for P-space generation """
  pspace_ciphi_epsilon::Float64 = 1e-3
  """`⟨1⟩` Level of printed output (0=none, 1=some, 2=detailed) """
  print_level::Int = 1
  """`⟨1e-12⟩` Threshold for neglecting small Hamiltonian (etc) elements"""
  thr_negligible::Float64 = 1e-12
end

"""
  Option for CIΦ (CIPHI) calculations - Selected CI via Perturbation, Heat-Bath and Iterations.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct CIPHIOptions
  """`⟨1_000_000⟩` Target (i.e., maximum) number of determinants to select """
  target_selection::Int = 1_000_000
  """`⟨3e-4⟩` Selection threshold """
  epsilon::Float64 = 3e-4
  """`⟨epsilon/10⟩` CIPHI selection threshold. Note that a smaller value improves also the quality of the PT2 correction. """
  epsilon_h::Float64 = -1.0
  """`⟨epsilon_h⟩` instantaneous PT selection threshold. """
  epsilon_c::Float64 = -1.0
  """`⟨epsilon⟩` CIPSI selection threshold """
  epsilon_p::Float64 = -1.0
  """`⟨1e-6⟩` Energy convergence threshold """
  tol::Float64 = 1e-6
  """`⟨1e-6⟩` Convergence tolerance for residual norm """
  res_tol::Float64 = 1e-6
  """`⟨false⟩` calculate properties such as dipole moments. """
  properties::Bool = false
  """`⟨50⟩` Maximum CIPHI iterations """
  max_iter::Int = 50
  """`⟨0.1⟩` Level shift to improve convergence """
  shift::Float64 = 0.1
  """`⟨true⟩` Print iteration details """
  verbose::Bool = true
  """`⟨true⟩` Compute PT2 perturbative correction """
  compute_pt2::Bool = true
  """`⟨1e-6⟩` Threshold for PT2 contributions """
  epsilon_pt2::Float64 = 1e-6
  """`⟨epsilon_pt2/2⟩` Threshold for instantaneous PT2 contributions """
  epsilon_pt2_c::Float64 = -1.0
  """`⟨true⟩` Sort determinants by absolute value of coefficients before computing PT2 correction """
  sort4pt2::Bool = true
  """`⟨1e-10⟩` Small value added to denominators in PT2 to avoid divergences """
  pt2_shift::Float64 = 1e-10
  """`⟨false⟩` Use uncontracted MP2 instead of EN2 """
  use_mp2::Bool = false
  """`⟨false⟩` Use renormalized PT2 correction: `E_PT2 → E_PT2 / (1 + T2^2)` with `T2` being the PT2 amplitudes. """
  renorm_pt2::Bool = false
  """`⟨1⟩` Number of states to compute (default: 1 = ground state only) """
  nstates::Int = 1
  """`⟨2⟩` Number of steps in the iterative CIPHI selection (if > 1, the selection process is repeated after convergence)"""
  nsteps::Int = 2
  """`⟨true⟩` Use small-space Hamiltonian for initial guess """
  use_small_space_guess::Bool = true
  """`⟨0⟩` Size of small space (0 = auto: max(100, target÷10, 5*n_roots)) """
  small_space_size::Int = 0
  """`⟨:hybrid⟩` Selection method: :hybrid (energy + excitation) """
  small_space_method::Symbol = :hybrid
  """`⟨1⟩` Level of printed output (0=none, 1=some, 2=detailed) """
  print_level::Int = 1
  """`⟨false⟩` Skip variational CIPHI iterations and only compute PT2 correction (use with restart) """
  pt2_only::Bool = false
  """`⟨1e-10⟩` Threshold for neglecting small Hamiltonian (etc) elements"""
  thr_negligible::Float64 = 1e-10
end

"""
  Options for excited states calculation.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct EomOptions
  """`⟨1.e-6⟩` convergence threshold. """
  thr::Float64 = 1.e-6
  """`⟨1⟩` number of states to calculate. """
  nstates::Int = 1
  """`⟨50⟩` maximum number of iterations. """
  maxit::Int = 50
  """`⟨0.0⟩` level shift for the Davidson algorithm. """
  shift::Float64 = 0.0
  """`⟨"eigenvectors"⟩` main part of filename for start eigenvectors. 
      For example, the singles eigenvectors for state 2 are read from `start*"_1^2"`. """
  start::String = "eigenvectors"
  """`⟨"eigenvectors"⟩` main part of filename to save eigenvectors.
      For example, the singles eigenvectors for state 2 are saved to `save*"_1^2"`. """
  save::String = "eigenvectors"
  """`⟨1.e-5⟩` amplitude decomposition threshold. """
  ampsvdtol::Float64 = 1.e-5
  """`⟨6⟩` Choice how excited state SVD basis is generated. Default is decomposition of full U2. """
  svd_space_option::Int = 6
end

"""
  Options for Laplace quadrature.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct LaplaceOptions
  """`⟨8⟩` number of Laplace quadrature points. """
  npoints::Int = 8
  """`⟨:minimax⟩` algorithm for Laplace quadrature points. (`:minimax` or `:simplex`) """
  algo::Symbol = :minimax
end

""" 
  Options for DMRG calculation.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct DmrgOptions
  """`⟨10⟩` number of sweeps. """
  nsweeps::Int = 10
  """`⟨[100, 200]⟩` maximum size for the bond dimension. """
  maxdim::Vector{Int} = [100, 200]
  """`⟨1e-6⟩` cutoff for the singular value decomposition. """
  cutoff::Float64 = 1e-6
  """`⟨[1e-6, 1e-7, 1e-8, 0.0]⟩` strength of the noise term used to aid convergence. """
  noise::Vector{Float64} = [1e-6, 1e-7, 1e-8, 0.0]
end

"""
  Options for integral calculation.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct IntOptions
  """`⟨true⟩` use density-fitted integrals. """
  df::Bool = true
  """`⟨""⟩` store integrals in FCIDump format. """
  fcidump::String = ""
  """`⟨false⟩` use Cartesian subshells instead of Spherical. """
  cartesian::Bool = false
  """`⟨1000⟩` target batch length for the integral transformation. """
  target_batch_length::Int = 1000
  """`⟨false⟩` use fallback basis sets (in case of missing basis sets). """
  use_fallback_basis::Bool = false
  """`⟨true⟩` sanity check of the fit basis (i.e., that it's not an AO basis)"""
  check_fit_basis::Bool = true
  """`⟨true⟩` split independent angular shells (important for efficiency). """
  split_ashells::Bool = true
end

""" 
  Options for Cholesky decomposition.
    
  $(TYPEDFIELDS)
"""
@kwdef mutable struct CholeskyOptions
  """`⟨1.e-6⟩` threshold for elimination of redundancies in the auxiliary basis. """
  thred::Float64 = 1.e-6
  """`⟨1.e-5⟩` threshold for integral decomposition. """
  thr::Float64 = 1.e-5
  """`⟨0.01⟩` span factor for two-step Cholesky batch qualification. """
  sigma::Float64 = 0.01
  """`⟨false⟩` use SVD (real) / Takagi (complex) instead of Cholesky for the J matrix in step II. """
  usesvd::Bool = false
end

"""
  Options for DIIS.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct DiisOptions
  """`⟨6⟩` maximum number of DIIS vectors. """
  maxdiis::Int = 6
  """`⟨10.0⟩` DIIS residual threshold. """
  resthr::Float64 = 10.0
  """`⟨false⟩` CROP-DIIS (see [JCTC 11, 1518 (2015)](https://doi.org/10.1021/ct501114q)).
  Usually the DIIS dimension `maxcrop=3` is sufficient. """
  crop::Bool = false
  """`⟨3⟩` DIIS dimension for CROP-DIIS. """
  maxcrop::Int = 3
end

"""
  Options for Davidson.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct DavidsonOptions
  """`⟨10⟩` maximum number of Davidson vectors per state. """
  maxdav::Int = 10
  """`⟨true⟩` use overlap in hermitian Davidson. """
  use_overlap::Bool = true
end


"""
  Options for printing.

  $(TYPEDFIELDS)
"""
@kwdef mutable struct PrintOptions
  """`⟨2⟩` verbosity level for printing timings. """
  time::Int = 2
  """`⟨2⟩` verbosity level for printing memory usage. """
  memory::Int = 2
  """`⟨10⟩` maximum number of coefficients to print. """
  ncoeff::Int = 10
end

""" 
  Options for ElemCo.jl.

  $(TYPEDFIELDS)
"""  
@kwdef mutable struct Options
  """ Wavefunction options ([`WfOptions`](@ref)). """
  wf::WfOptions = WfOptions()
  """ SCF options ([`ScfOptions`](@ref)). """
  scf::ScfOptions = ScfOptions()
  """ Integral options ([`IntOptions`](@ref)). """
  int::IntOptions = IntOptions()
  """ Coupled-Cluster options ([`CcOptions`](@ref)). """
  cc::CcOptions = CcOptions()
  """ EOM options ([`EomOptions`](@ref)). """
  eom::EomOptions = EomOptions()
  """ FCI options ([`FCIOptions`](@ref)). """
  fci::FCIOptions = FCIOptions()
  """ CIPHI options ([`CIPHIOptions`](@ref)). """
  ciphi::CIPHIOptions = CIPHIOptions()
  """ DMRG options ([`DmrgOptions`](@ref)). """
  dmrg::DmrgOptions = DmrgOptions()
  """ Cholesky options ([`CholeskyOptions`](@ref)). """
  cholesky::CholeskyOptions = CholeskyOptions()
  """ Laplace options ([`LaplaceOptions`](@ref)). """
  laplace::LaplaceOptions = LaplaceOptions()
  """ DIIS options ([`DiisOptions`](@ref)). """
  diis::DiisOptions = DiisOptions()
  """ Davidson options ([`DavidsonOptions`](@ref)). """
  davidson::DavidsonOptions = DavidsonOptions()
  """ Print options ([`PrintOptions`](@ref)). """
  print::PrintOptions = PrintOptions()
  """ Localization options ([`LocOptions`](@ref)). """
  loc::LocOptions = LocOptions()
  """ Region selection options ([`RegionOptions`](@ref)). """
  region::RegionOptions = RegionOptions()
end
