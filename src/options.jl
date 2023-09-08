# Options

""" 
  Options for SCF calculation

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
  """ filename of orbitals for orbital guess. """
  orbsguess::String = "C_Am"
  """ filename to save orbitals. """
  save::String = "C_Am"
  """ addition to the filename for left orbitals (for biorthogonal calculations). """
  left::String = "-left"
end

""" 
  Options for Coupled-Cluster calculation

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
  """ filename of orbitals (for non-fcidump calculations). """
  orbs::String = "C_Am"
  """ filename for start amplitudes. """
  start::String = "cc_amplitudes"
  """ filename to save amplitudes. """
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
  """ filename of orbitals for MO transformation. If empty: ScfOptions.save is used. """
  orbs::String = ""
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
  Options for ElemCo.jl

  $(FIELDS)
"""  
@with_kw mutable struct Options
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
