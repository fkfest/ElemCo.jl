"""
FCI Module

This module contains a translation of the FCI C++ code of Gerald Knizia and
extensions for selected CI and CIPHI (CIΦ - Selected CI via Perturbation, Heat-Bath and Iterations).
"""
module FCI

using LinearAlgebra
using Printf
using StridedViews
using TensorOperations
using Buffers

using ..ElemCo.ECInfos
using ..ElemCo.FciDumps
using ..ElemCo.Utils
using ..ElemCo.AbstractEC: AbstractDeterminant

# Export main types and functions
export FCIContext
export CIPHIContext  # Lightweight context for CIPHI
export run_fci!
export run_ciphi!

include("fci_types.jl")
include("fci_vec.jl")
include("fci_ciphi_context.jl")
include("fci_ops.jl")
include("fci_main.jl")
include("sci_main.jl")
include("fci_pspace.jl")
include("fci_davidson.jl")

end # module FCI
