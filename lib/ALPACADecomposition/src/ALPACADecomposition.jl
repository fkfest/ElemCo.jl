module ALPACADecomposition

using LinearAlgebra
using Random: randperm

export AbstractALPACAMatrix
export DenseALPACAMatrix
export SymmetricALPACAMatrix, HermitianALPACAMatrix
export column!, row!, elements!
export AbstractPrincipalDescriptor
export PrincipalPairs, PrincipalTriples
export principal_pairs, principal_triples, normalize_principal_descriptor
export ALPACAOptions, ALPACAResult
export ALPACACache
export alpaca, lpaca, qrdalpaca
export alpaca_svd, lpaca_svd, qrdalpaca_svd
export alpaca_eigen, lpaca_eigen, qrdalpaca_eigen
export alpaca_takagi, lpaca_takagi, qrdalpaca_takagi
export alpaca_qr, lpaca_qr, qrdalpaca_qr
export LLAMAResult
export llama, llama_svd

include("access.jl")
include("descriptors.jl")
include("results.jl")
include("cache.jl")
include("kernels.jl")
include("pivots.jl")
include("alpaca.jl")
include("qrdalpaca.jl")
include("decompositions.jl")
include("llama.jl")

end