# Options

"""
  Options for wavefunction/orbitals.

  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct WfOptions
  """`⟨-1⟩` spin magnetic quantum number times two (2×mₛ) of the system. """
  ms2::Int = -1
  """`⟨-1⟩` number of electrons. If < 0, the number of electrons is 
  read from the FCIDump file or guessed for the neutral system. """
  nelec::Int = -1
  """`⟨0⟩` charge of the system (relative to nelec/FCIDump/neutral system!). """
  charge::Int = 0
  """`⟨"C_Am"⟩` filename of MO coefficients. 
  Used by all programs to read and write orbitals from/to file. """
  orb::String = "C_Am"
  """`⟨"-left"⟩` addition to the filename for left orbitals (for biorthogonal calculations). """
  left::String = "-left"
  """`⟨:large⟩` core type for frozen-core approximation: 
  - `:none` no frozen-core approximation, 
  - `:small` semi-core orbitals correlated, 
  - `:large` semi-core orbitals frozen. """
  core::Symbol = :large
  """`⟨-1⟩` number of occupied (core) orbitals to freeze (overwrites core). """
  freeze_nocc::Int = -1
  """`⟨0⟩` number of virtual (highest) orbitals to freeze. """
  freeze_nvirt::Int = 0
  """`⟨"-"⟩` occupied α (or closed-shell) orbitals. """
  occa::String = "-"
  """`⟨"-"⟩` occupied β orbitals. 
  If `occb::String` is empty, the occupied β orbitals are the same as the occupied α orbitals (closed-shell case)."""
  occb::String = "-"
end


""" 
  Options for SCF calculation.

  $(TYPEDFIELDS)    
"""
Base.@kwdef mutable struct ScfOptions
  """`⟨1.e-10⟩` convergence threshold. """
  thr::Float64 = 1.e-10
  """`⟨sqrt(thr)*0.1⟩` energy convergence threshold (used additionally to `thr`). """
  thren::Float64 = -1.0
  """`⟨50⟩` maximum number of iterations. """
  maxit::Int = 50
  """`⟨1.e-8⟩` tolerance for imaginary part of MO coefs (for biorthogonal). """
  imagtol::Float64 = 1.e-8
  """`⟨false⟩` direct calculation without storing integrals. """
  direct::Bool = false
  """`⟨:SAD⟩` orbital guess:
  - `:HCORE` from core Hamiltonian
  - `:SAD` from atomic densities
  - `:GWH` not implemented yet
  - `:ORB` from previous orbitals stored in file [`WfOptions.orb`](@ref ECInfos.WfOptions)
  """
  guess::Symbol = :SAD
  """ `⟨0.0⟩` Fermi-Dirac temperature for starting guess (at the moment works only for BO-HF). """
  temperature_guess::Float64 = 0.0
  """`⟨false⟩` Generate pseudo-canonical basis instead of solving the SCF problem,
  i.e., build and block-diagonalize the Fock matrix without changing the Fermi level.
  At the moment, it works only for BO-HF."""
  pseudo::Bool = false
end

""" 
  Options for Coupled-Cluster calculation.

  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct CcOptions
  """`⟨1.e-10⟩` convergence threshold. """
  thr::Float64 = 1.e-10
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
  """`⟨1.e-3⟩` amplitude decomposition threshold. """
  ampsvdtol::Float64 = 1.e-3
  """`⟨1.e-2⟩` tightening amplitude decomposition factor 
      (for the two-step decomposition). """
  ampsvdfac::Float64 = 1.e-2
  """`⟨true⟩` use kext for doubles residual. """
  use_kext::Bool = true
  """`⟨false⟩` calculate dressed <vv|vv>. """
  calc_d_vvvv::Bool = false
  """`⟨false⟩` calculate dressed <vv|vo>. """
  calc_d_vvvo::Bool = false
  """`⟨false⟩` calculate dressed <vo|vv>. """
  calc_d_vovv::Bool = false
  """`⟨false⟩` calculate dressed <vv|oo>. """
  calc_d_vvoo::Bool = false
  """`⟨true⟩` use a triangular kext if possible. """
  triangular_kext::Bool = true
  """`⟨false⟩` calculate (T) for decomposition. """
  calc_t3_for_decomposition::Bool = false
  """`⟨0.0⟩` imaginary shift for denominator in doubles decomposition. """
  deco_ishiftp::Float64 = 0.0
  """`⟨0.0⟩` imaginary shift for denominator in triples decomposition. """
  deco_ishiftt::Float64 = 0.0
  """`⟨false⟩` use a projected exchange for contravariant doubles amplitudes in SVD-DCSD,
  ``\\tilde T_{XY} = U^{†a}_{iX} U^{†b}_{jY} \\tilde T^{ij}_{ab}``. """
  use_projx::Bool = false
  """`⟨false⟩` use full doubles amplitudes in SVD-DCSD. 
  The decomposition is used only for ``N^6`` scaling terms.  """
  use_full_t2::Bool = false
  """`⟨2⟩` what to project in ``v_{ak}^{ci} T^{kj}_{cb}`` in SVD-DCSD:
  0: both, 1: amplitudes, 2: residual, 3: robust fit. """
  project_vovo_t2::Int = 2
  """`⟨false⟩` decompose full doubles amplitudes in SVD-DCSD (slow). """
  decompose_full_doubles::Bool = false
  """`⟨"cc_amplitudes"⟩` main part of filename for start amplitudes. 
      For example, the singles amplitudes are read from `start*"_singles"` """
  start::String = "cc_amplitudes"
  """`⟨"cc_amplitudes"⟩` main part of filename to save amplitudes.
      For example, the singles amplitudes are saved to `save*"_singles"` """
  save::String = "cc_amplitudes"
  """`⟨"cc_multipliers"⟩` main part of filename for start Lagrange multipliers. 
      For example, the singles Lagrange multipliers are read from `start_lm*"_singles"` """
  start_lm::String = "cc_multipliers"
  """`⟨"cc_multipliers"⟩` main part of filename to save Lagrange multipliers.
      For example, the singles Lagrange multipliers are saved to `save_lm*"_singles"` """
  save_lm::String = "cc_multipliers"
  """`⟨0⟩` Don't use MP2 amplitudes as starting guess for the CC amplitudes """
  nomp2::Int = 0
end

"""
  Options for integral calculation.

  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct IntOptions
  """`⟨true⟩` use density-fitted integrals. """
  df::Bool = true
  """`⟨""⟩` store integrals in FCIDump format. """
  fcidump::String = ""
end

""" 
  Options for Cholesky decomposition.
    
  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct CholeskyOptions
  """`⟨1.e-6⟩` cholesky threshold. """
  thr::Float64 = 1.e-6
end

"""
  Options for DIIS.

  $(TYPEDFIELDS)
"""
Base.@kwdef mutable struct DiisOptions
  """`⟨6⟩` maximum number of DIIS vectors. """
  maxdiis::Int = 6
  """`⟨10.0⟩` DIIS residual threshold. """
  resthr::Float64 = 10.0
end

""" 
  Options for ElemCo.jl.

  $(TYPEDFIELDS)
"""  
Base.@kwdef mutable struct Options
  """ Wavefunction options ([`WfOptions`](@ref)). """
  wf::WfOptions = WfOptions()
  """ SCF options ([`ScfOptions`](@ref)). """
  scf::ScfOptions = ScfOptions()
  """ Integral options ([`IntOptions`](@ref)). """
  int::IntOptions = IntOptions()
  """ Coupled-Cluster options ([`CcOptions`](@ref)). """
  cc::CcOptions = CcOptions()
  """ Cholesky options ([`CholeskyOptions`](@ref)). """
  cholesky::CholeskyOptions = CholeskyOptions()
  """ DIIS options ([`DiisOptions`](@ref)). """
  diis::DiisOptions = DiisOptions()
end
