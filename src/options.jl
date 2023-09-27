# Options

"""
  Options for wavefunction/orbitals.

  $(FIELDS)
"""
@with_kw mutable struct WfOptions
  """ spin magnetic quantum number times two (2×mₛ) of the system. """
  ms2::Int = -1
  """ number of electrons. If < 0, the number of electrons is 
  read from the FCIDump file or guessed for the neutral system. """
  nelec::Int = -1
  """ charge of the system (relative to nelec/FCIDump/neutral system!). """
  charge::Int = 0
  """ filename of MO coefficients. 
  Used by all programs to read and write orbitals from/to file. """
  orb::String = "C_Am"
  """ addition to the filename for left orbitals (for biorthogonal calculations). """
  left::String = "-left"
  """ core type for frozen-core approximation: 
  - :none (no frozen-core approximation), 
  - :small (semi-core orbitals correlated), 
  - :large (semi-core orbitals frozen). """
  core::Symbol = :large
  """ number of occupied (core) orbitals to freeze (overwrites core). """
  freeze_nocc::Int = -1
  """ number of virtual (highest) orbitals to freeze. """
  freeze_nvirt::Int = 0
  """ occupied α (or closed-shell) orbitals. """
  occa::String = "-"
  """ occupied β orbitals. 
  If `occb::String` is empty, the occupied β orbitals are the same as the occupied α orbitals (closed-shell case)."""
  occb::String = "-"
end


""" 
  Options for SCF calculation.

  $(FIELDS)    
"""
@with_kw mutable struct ScfOptions
  """ convergence threshold. """
  thr::Float64 = 1.e-10
  """ maximum number of iterations. """
  maxit::Int = 50
  """ tolerance for imaginary part of MO coefs (for biorthogonal). """
  imagtol::Float64 = 1.e-8
  """ direct calculation without storing integrals. """
  direct::Bool = false
  """ orbital guess. """
  guess::Symbol = :SAD
end

""" 
  Options for Coupled-Cluster calculation.

  $(FIELDS)
"""
@with_kw mutable struct CcOptions
  """ convergence threshold. """
  thr::Float64 = 1.e-10
  """ maximum number of iterations. """
  maxit::Int = 50
  """ level shift for singles. """
  shifts::Float64 = 0.15
  """ level shift for doubles. """
  shiftp::Float64 = 0.2
  """ level shift for triples. """
  shiftt::Float64 = 0.2
  """ amplitude decomposition threshold. """
  ampsvdtol::Float64 = 1.e-3
  """ use kext for doubles residual. """
  use_kext::Bool = true
  """ calculate dressed <vv|vv>. """
  calc_d_vvvv::Bool = false
  """ calculate dressed <vv|vo>. """
  calc_d_vvvo::Bool = false
  """ calculate dressed <vo|vv>. """
  calc_d_vovv::Bool = false
  """ calculate dressed <vv|oo>. """
  calc_d_vvoo::Bool = false
  """ use a triangular kext if possible. """
  triangular_kext::Bool = true
  """ calculate (T) for decomposition. """
  calc_t3_for_decomposition::Bool = false
  """ imaginary shift for denominator in doubles decomposition. """
  deco_ishiftp::Float64 = 0.0
  """ imaginary shift for denominator in triples decomposition. """
  deco_ishiftt::Float64 = 0.0
  """ use a projected exchange for contravariant doubles amplitudes in SVD-DCSD,
  ``\\tilde T_{XY} = U^{†a}_{iX} U^{†b}_{jY} \\tilde T^{ij}_{ab}``. """
  use_projx::Bool = false
  """ use full doubles amplitudes in SVD-DCSD. 
  The decomposition is used only for ``N^6`` scaling terms.  """
  use_full_t2::Bool = false
  """ what to project in ``v_{ak}^{ci} T^{kj}_{cb}`` in SVD-DCSD:
  0: both, 1: amplitudes, 2: residual, 3: robust fit. """
  project_vovo_t2::Int = 2
  """ decompose full doubles amplitudes in SVD-DCSD (slow). """
  decompose_full_doubles::Bool = false
  """ main part of filename for start amplitudes. 
      For example, the singles amplitudes are read from `start*"_singles"` """
  start::String = "cc_amplitudes"
  """ main part of filename to save amplitudes.
      For example, the singles amplitudes are saved to `save*"_singles"` """
  save::String = "cc_amplitudes"
end

"""
  Options for integral calculation.

  $(FIELDS)
"""
@with_kw mutable struct IntOptions
  """ use density-fitted integrals. """
  df::Bool = true
  """ store integrals in FCIDump format. """
  fcidump::String = ""
end

""" 
  Options for Cholesky decomposition.
    
  $(FIELDS)
"""
@with_kw mutable struct CholeskyOptions
  """ cholesky threshold. """
  thr::Float64 = 1.e-6
end

"""
  Options for DIIS.

  $(FIELDS)
"""
@with_kw mutable struct DiisOptions
  """ maximum number of DIIS vectors. """
  maxdiis::Int = 6
  """ DIIS residual threshold. """
  resthr::Float64 = 10.0
end

""" 
  Options for ElemCo.jl.

  $(FIELDS)
"""  
@with_kw mutable struct Options
  """ Wavefunction options. """
  wf::WfOptions = WfOptions()
  """ SCF options. """
  scf::ScfOptions = ScfOptions()
  """ Integral options. """
  int::IntOptions = IntOptions()
  """ Coupled-Cluster options. """
  cc::CcOptions = CcOptions()
  """ Cholesky options. """
  cholesky::CholeskyOptions = CholeskyOptions()
  """ DIIS options. """
  diis::DiisOptions = DiisOptions()
end
