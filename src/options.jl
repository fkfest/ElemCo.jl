# Options

""" SCF options """
@with_kw mutable struct ScfOptions
  thr::Float64 = 1.e-10
  maxit::Int = 50
end

""" Coupled-Cluster options """
@with_kw mutable struct CcOptions
  thr::Float64 = 1.e-10
  maxit::Int = 50
  shifts::Float64 = 0.15
  shiftp::Float64 = 0.2
  shiftt::Float64 = 0.2
  # amplitude decomposition threshold
  ampsvdtol::Float64 = 1.e-3
  use_kext::Bool = true
  calc_d_vvvv::Bool = false
  calc_d_vvvo::Bool = false
  calc_d_vovv::Bool = false
  calc_d_vvoo::Bool = false
  triangular_kext::Bool = true
  calc_t3_for_decomposition::Bool = false
end

""" cholesky options """
@with_kw mutable struct CholeskyOptions
  # cholesky threshold
  thr::Float64 = 1.e-6
end

""" `ElemCo.jl` options """  
@with_kw mutable struct Options
  # SCF options
  scf::ScfOptions = ScfOptions()
  # coupled-cluster options
  cc::CcOptions = CcOptions()
  # cholesky options
  chol::CholeskyOptions = CholeskyOptions()
end