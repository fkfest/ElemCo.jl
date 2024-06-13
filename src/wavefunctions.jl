module Wavefunctions
using ..ElemCo.BasisSets
export MOs, is_restricted_MO, unrestrict!, restrict!

"""
    MOs

A type to store the molecular orbitals of a system. 

The molecular orbitals are stored in a matrix, where each column is a molecular orbital. 
The first matrix is the alpha molecular orbitals, and the second matrix is the beta molecular orbitals. 
If the molecular orbitals are restricted, the beta molecular orbitals refer to the alpha molecular orbitals.
"""
mutable struct MOs
  α::Matrix{Float64}
  β::Matrix{Float64}
  function MOs(cmo::AbstractMatrix{Float64}, cmo2::AbstractMatrix{Float64})
    return new(cmo, cmo2)
  end
  function MOs(cmo::AbstractMatrix{Float64}=zeros(0,0))
    return new(cmo, cmo)
  end
  function MOs(cmos::Tuple{Matrix{Float64}, Matrix{Float64}})
    return new(cmos[1], cmos[2])
  end
end


Base.length(mos::MOs) = length(mos.α)

Base.size(mos::MOs) = size(mos.α)

Base.size(mos::MOs, i::Int) = size(mos.α, i)

Base.getindex(mos::MOs, spincase::Symbol) = getfield(mos, spincase)

Base.getindex(mos::MOs, i::Int) = getfield(mos, i)

function Base.setindex!(mos::MOs, cMO::AbstractMatrix{Float64}, spincase::Symbol) 
  if cMO isa Matrix{Float64}
    return setfield!(mos, spincase, cMO)
  else
    return setfield!(mos, spincase, copy(cMO))
  end
end

function Base.setindex!(mos::MOs, cMO::AbstractMatrix{Float64}, i::Int)
  if cMO isa Matrix{Float64}
    setfield!(mos, i, cMO)
  else
    setfield!(mos, i, copy(cMO))
  end
end

Base.iterate(mos::MOs, state=1) = state > 2 ? nothing : (mos[state], state+1) 

Base.Tuple(mos::MOs) = (mos.α, mos.β)

"""
    is_restricted_MO(cMO::MOs)

Check if the molecular orbitals are restricted.
"""
is_restricted_MO(cMO::MOs) = cMO.α === cMO.β

"""
    unrestrict!(cMO::MOs)

Unrestrict the molecular orbitals.
"""
function unrestrict!(cMO::MOs)
  if is_restricted_MO(cMO)
    cMO.β = copy(cMO.α)
  end
  return cMO
end

"""
    restrict!(cMO::MOs)

Restrict the molecular orbitals (β = α).
"""
function restrict!(cMO::MOs)
  cMO.β = cMO.α
  return cMO
end

struct Orbitals
    cMO::MOs
    basis::BasisSet
end



end #module