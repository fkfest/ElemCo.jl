"""
QMTensors module

This module provides definitions for useful quantum-mechanical tensors. 
"""
module QMTensors
# from spinmatrix
export SpinMatrix, FSpinMatrix, CSpinMatrix, is_restricted, unrestrict!, restrict!
# uppertriangular functions from utensors
export lentri_from_norb, norb_from_lentri, uppertriangular_index, uppertriangular_range, strict_uppertriangular_range

include("spinmatrix.jl")
include("utensors.jl")

end #module