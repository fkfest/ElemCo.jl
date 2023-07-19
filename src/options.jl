# Options

""" SCF options """
@with_kw mutable struct ScfOptions
  """ convergence threshold """
  thr::Float64 = 1.e-10
  """ maximum number of iterations """
  maxit::Int = 50
end

""" Coupled-Cluster options """
@with_kw mutable struct CcOptions
  """ convergence threshold """
  thr::Float64 = 1.e-10
  """ maximum number of iterations """
  maxit::Int = 50
  """ level shift for singles """
  shifts::Float64 = 0.15
  """ level shift for doubles """
  shiftp::Float64 = 0.2
  """ level shift for triples """
  shiftt::Float64 = 0.2
  """ amplitude decomposition threshold """
  ampsvdtol::Float64 = 1.e-3
  """ use kext for doubles residual """
  use_kext::Bool = true
  """ calculate dressed <vv|vv> """
  calc_d_vvvv::Bool = false
  """ calculate dressed <vv|vo> """
  calc_d_vvvo::Bool = false
  """ calculate dressed <vo|vv> """
  calc_d_vovv::Bool = false
  """ calculate dressed <vv|oo> """
  calc_d_vvoo::Bool = false
  """ use a triangular kext if possible """
  triangular_kext::Bool = true
  """ calculate (T) for decomposition """
  calc_t3_for_decomposition::Bool = false
end

""" cholesky options """
@with_kw mutable struct CholeskyOptions
  """ cholesky threshold """
  thr::Float64 = 1.e-6
end

""" `ElemCo.jl` options """  
@with_kw mutable struct Options
  """ SCF options """
  scf::ScfOptions = ScfOptions()
  """ coupled-cluster options """
  cc::CcOptions = CcOptions()
  """ cholesky options """
  cholesky::CholeskyOptions = CholeskyOptions()
end