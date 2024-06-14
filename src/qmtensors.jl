"""
QMTensors module

This module provides definitions for useful quantum-mechanical tensors. 
"""
module QMTensors
# from spinmatrix
export SpinMatrix, FSpinMatrix, CSpinMatrix, is_restricted, unrestrict!, restrict!

include("spinmatrix.jl")

end #module