"""
QMTensors module

This module provides definitions for useful quantum-mechanical tensors. 
"""
module QMTensors
using Buffers
# from spinmatrix
export SpinMatrix, FSpinMatrix, CSpinMatrix, is_restricted, unrestrict!, restrict!
export SpinVector
# uppertriangular functions from utensors
export lentri_from_norb, norb_from_lentri
export strict_lentri_from_norb, norb_from_strict_lentri
export uppertriangular_index, uppertriangular_range, strict_uppertriangular_range
export uppertriangular_cut, strict_uppertriangular_cut
export uppertriangular_cut3, strict_uppertriangular_cut3
export swapped_uppertriangular_cut, swapped_strict_uppertriangular_cut
export detri_doubles, detri_samespin_doubles
export calc_tri_sym_antisym!

include("spinmatrix.jl")
include("utensors.jl")

end #module