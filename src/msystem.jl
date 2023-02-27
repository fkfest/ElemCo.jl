"""
info about molecular system (geometry)

macros to specify geometry and basis sets
"""
module MSystem

"""
various basissets (ao,mp2fit,jkfit)
"""
mutable struct Basis
  basis::Dict{String,String}
end
"""
geometry and basis set for each element name in the geometry
"""
mutable struct MSys
  xyz::String
  bohr::Bool
  basis::Dict{String,Basis}
end



end #module