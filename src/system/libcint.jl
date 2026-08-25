"""
    Libcint

Minimal wrap around the integral library libcint. This module exposes
libcint functions to the Julia interface. 

(adapted from GaussianBasis.jl) 
"""
module Libcint

using ..ElemCo.BasisSets

export CINTcgtos_spheric, CINTcgtos_cart
export cint1e_kin_sph!, cint1e_nuc_sph!, cint1e_ovlp_sph!, cint2c2e_sph!, cint2e_sph!, cint3c2e_sph!
export cint1e_kin_cart!, cint1e_nuc_cart!, cint1e_ovlp_cart!, cint2c2e_cart!, cint2e_cart!, cint3c2e_cart!
export cint1e_ipkin_sph!, cint1e_ipnuc_sph!, cint1e_ipovlp_sph!, cint2e_ip1_sph!, cint1e_r_sph!
export cint1e_ipkin_cart!, cint1e_ipnuc_cart!, cint1e_ipovlp_cart!, cint2e_ip1_cart!, cint1e_r_cart!
export cint1e_r_sph!, cint1e_rr_sph!, cint1e_rrr_sph!, cint1e_rrrr_sph!
export CIntOpt, cint2e_sph_optimizer, cint2e_cart_optimizer, free_optimizer!
export cint1e_r_cart!, cint1e_rr_cart!, cint1e_rrr_cart!, cint1e_rrrr_cart!

using libcint_jll

const LIBCINT = libcint

function CINTcgtos_spheric(id::Integer, lib::ILibcint)
  id_c = Cint(id - 1)
  @ccall LIBCINT.CINTcgtos_spheric(id_c::Cint, lib.bas::Ptr{Cint})::Cint
end

function CINTcgtos_cart(id, lib::ILibcint)
  id_c = Cint(id - 1)
  @ccall LIBCINT.CINTcgtos_cart(id_c::Cint, lib.bas::Ptr{Cint})::Cint
end

# automatically generate functions for the 1-electron integrals
for suffix in ("sph", "cart")
  for type in ("ovlp", "kin", "nuc", "ipkin", "ipnuc", "ipovlp", "r", "rr", "rrr", "rrrr")
    jname = Symbol("cint1e_$(type)_$(suffix)!")
    cname = Symbol("cint1e_$(type)_$(suffix)")
    @eval begin
      function $jname(buf::Array{Cdouble}, cshls::AbstractArray{Cint}, lib::ILibcint)
        @ccall LIBCINT.$cname(
            buf  :: Ptr{Cdouble},
            cshls :: Ptr{Cint},
            lib.atm  :: Ptr{Cint},
            lib.natm :: Cint,
            lib.bas  :: Ptr{Cint},
            lib.nbas :: Cint,
            lib.env  :: Ptr{Cdouble}
          )::Cvoid
      end
    end
  end
end

"""
    CIntOpt

  A libcint `CINTOpt` optimizer handle: per-(integral type, basis) precomputed shell-pair data that
  libcint's kernels use to skip work — its intended fast path (without it every quartet call passes
  a NULL optimizer and recomputes the pair data). Create with the `*_optimizer` constructors (e.g.
  [`cint2e_sph_optimizer`](@ref)), pass to the optimizer-taking integral methods, and free with
  [`free_optimizer!`](@ref) (a finalizer is attached as a leak backstop, but generation sweeps free
  deterministically). The handle is READ-ONLY during integral evaluation, so one optimizer can be
  shared by all threads of a sweep.
"""
mutable struct CIntOpt
  ptr::Ptr{Cvoid}
  function CIntOpt(ptr::Ptr{Cvoid})
    opt = new(ptr)
    finalizer(free_optimizer!, opt)
    return opt
  end
end

"""
    free_optimizer!(opt::CIntOpt)

  Free the libcint optimizer (idempotent; also attached as a finalizer).
"""
function free_optimizer!(opt::CIntOpt)
  if opt.ptr != C_NULL
    ref = Ref{Ptr{Cvoid}}(opt.ptr)
    @ccall LIBCINT.CINTdel_optimizer(ref::Ptr{Ptr{Cvoid}})::Cvoid
    opt.ptr = C_NULL
  end
  return
end

# optimizer constructors for the plain 2-electron integrals (the 4-index ERI generation);
# other integral types keep the NULL-optimizer path until they need one
for suffix in ("sph", "cart")
  jname = Symbol("cint2e_$(suffix)_optimizer")
  cname = Symbol("cint2e_$(suffix)_optimizer")
  @eval begin
    """
        $($(string(jname)))(lib::ILibcint) -> CIntOpt

      Allocate the libcint optimizer for `cint2e_$($(string(suffix)))` on this basis.
    """
    function $jname(lib::ILibcint)
      ref = Ref{Ptr{Cvoid}}(C_NULL)
      @ccall LIBCINT.$cname(
          ref :: Ptr{Ptr{Cvoid}},
          lib.atm  :: Ptr{Cint},
          lib.natm :: Cint,
          lib.bas  :: Ptr{Cint},
          lib.nbas :: Cint,
          lib.env  :: Ptr{Cdouble}
        )::Cvoid
      return CIntOpt(ref[])
    end
  end
end

# automatically generate functions for the 2-electron integrals; the 4-arg methods take a CIntOpt
# (libcint treats a NULL optimizer as "compute the pair data on the fly", so the 3-arg methods
# simply forward a NULL handle)
for prefix in ("2e", "2c2e", "3c2e")
  for suffix in ("sph", "cart")
    for type in ("", "ip1_")
      jname = Symbol("cint$(prefix)_$(type)$(suffix)!")
      cname = Symbol("cint$(prefix)_$(type)$(suffix)")
      @eval begin
        function $jname(buf::AbstractArray{Cdouble}, cshls::AbstractArray{Cint}, lib::ILibcint,
                        opt::Ptr{Cvoid}=Ptr{Cvoid}(C_NULL))
          @ccall LIBCINT.$cname(
              buf  :: Ptr{Cdouble},
              cshls :: Ptr{Cint},
              lib.atm  :: Ptr{Cint},
              lib.natm :: Cint,
              lib.bas  :: Ptr{Cint},
              lib.nbas :: Cint,
              lib.env  :: Ptr{Cdouble},
              opt :: Ptr{Cvoid}
            )::Cvoid
        end
        $jname(buf::AbstractArray{Cdouble}, cshls::AbstractArray{Cint}, lib::ILibcint,
               opt::CIntOpt) = $jname(buf, cshls, lib, opt.ptr)
      end
    end
  end
end

end #module
