"""
    Libcint 5

Minimal wrap around the integral library libcint. This module exposes
libcint functions to the Julia interface. 

(adapted from GaussianBasis.jl) 
"""
module Libcint5

using ..ElemCo.BasisSets

export CINTcgtos_spheric, CINTcgtos_cart
export cint1e_kin_sph!, cint1e_nuc_sph!, cint1e_ovlp_sph!, cint2c2e_sph!, cint2e_sph!, cint3c2e_sph!
export cint1e_ipkin_sph!, cint1e_ipnuc_sph!, cint1e_ipovlp_sph!, cint2e_ip1_sph!, cint1e_r_sph!
export cint1e_r_sph!, cint1e_rr_sph!, cint1e_rrr_sph!, cint1e_rrrr_sph!

using libcint_jll

const LIBCINT = libcint

function CINTcgtos_spheric(id::Integer, lib::ILibcint5)
  id_c = Cint(id - 1)
  @ccall LIBCINT.CINTcgtos_spheric(id_c::Cint, lib.bas::Ptr{Cint})::Cint
end

function CINTcgtos_cart(id, lib::ILibcint5)
  id_c = Cint(id - 1)
  @ccall LIBCINT.CINTcgtos_cart(id_c::Cint, lib.bas::Ptr{Cint})::Cint
end

# automatically generate functions for the 1-electron integrals
for suffix in ("sph", "cart")
  for type in ("ovlp", "kin", "nuc", "ipkin", "ipnuc", "ipovlp", "r", "rr", "rrr", "rrrr")
    jname = Symbol("cint1e_$(type)_$(suffix)!")
    cname = Symbol("cint1e_$(type)_$(suffix)")
    @eval begin
      function $jname(buf::Array{Cdouble}, shls::Array{<:Integer}, lib::ILibcint5)
        cshls = Cint.(shls.-1)
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

# automatically generate functions for the 2-electron integrals
for prefix in ("2e", "2c2e", "3c2e")
  for suffix in ("sph", "cart")
    for type in ("", "ip1_")
      jname = Symbol("cint$(prefix)_$(type)$(suffix)!")
      cname = Symbol("cint$(prefix)_$(type)$(suffix)")
      @eval begin
        function $jname(buf::Array{Cdouble}, shls::Array{<:Integer}, lib::ILibcint5)
          opt = Ptr{UInt8}(C_NULL)
          cshls = Cint.(shls.-1)
          @ccall LIBCINT.$cname(
              buf  :: Ptr{Cdouble},
              cshls :: Ptr{Cint},
              lib.atm  :: Ptr{Cint},
              lib.natm :: Cint,
              lib.bas  :: Ptr{Cint},
              lib.nbas :: Cint,
              lib.env  :: Ptr{Cdouble},
              opt :: Ptr{UInt8}
            )::Cvoid
        end
      end
    end
  end
end

end #module
