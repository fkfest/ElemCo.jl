include("libcint5.jl")

"""
  Electron-repulsion (and other) integrals 
"""
module Integrals
using Buffers
using ..ElemCo.Utils
using ..ElemCo.BasisSets
using ..ElemCo.Libcint5

export generate_basis # from BasisSets
export overlap!, kinetic!, nuclear!
export overlap, kinetic, nuclear
export eri_2e4idx!, eri_2e3idx!, eri_2e2idx!
export eri_2e4idx, eri_2e3idx, eri_2e2idx
export n_ao4sphshell, n_ao4cartshell
export BasisBatcher, BasisBatch, buffer_size_3idx, max_batch_length

include("basisbatcher.jl")

"""
    n_ao4sphshell(id::Integer, info::ILibcint5)

  Return the number of AOs for a given spherical shell id.
"""
n_ao4sphshell(id::Integer, info::ILibcint5) = CINTcgtos_spheric(id, info)
"""
    n_ao4cartshell(id::Integer, info::ILibcint5)

  Return the number of AOs for a given cartesian shell id.
"""
n_ao4cartshell(id::Integer, info::ILibcint5) = CINTcgtos_cart(id, info)

include("integrals_2idx.jl")
include("integrals_2e3idx.jl")
include("integrals_2e4idx.jl")

end #module
