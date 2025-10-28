"""
FCI Module

This module contains a translation of the FCI C++ code of Gerald Knizia and
extensions for selected CI and heat-bath CI.
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

# Export main types and functions
export FCIContext
export HCIContext  # Lightweight context for Heat-Bath CI
export run_fci!
export run_heatbath_ci!

include("fci_types.jl")
include("fci_vec.jl")
include("fci_hci_context.jl")
include("fci_ops.jl")
include("fci_main.jl")
include("fci_selected_ci.jl")
include("fci_pspace.jl")
include("fci_davidson.jl")

end # module FCI
