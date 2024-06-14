module Wavefunctions
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets

struct Orbitals{T<:Number}
    cMO::SpinMatrix{T}
    basis::BasisSet
end



end #module