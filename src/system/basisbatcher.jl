
"""
    BasisBatcher

  A structure to loop of basis functions in a batched manner. This is useful for
  computing integrals in batches. The structure is initialized with two basis sets
  and provides a method to loop over all basis functions of the second basis in a batched manner:
```julia
function BasisBatcher(basis1::BasisSet, basis2::BasisSet, target_length::Int=100)
```
  The batch length is determined by the `target_length` parameter. The actual batch length depends
  on the number of basis functions in the shells of the second basis set and can be smaller or
  larger than the `target_length`. 
  One can use the [`max_batch_length`](@ref) function to determine the maximum batch length.
  
# Example
```julia
bbatches = BasisBatcher(basis1, basis2)
for batch in bbatches
  range = range(batch) # range of basis functions in the batch
  eri_2e3idx!(@view(pqP[:,:,range]), batch)
end
```
"""
struct BasisBatcher
  basis::BasisSet
  n4sh::Vector{Vector{Int}}
  n_max::Vector{Int}
  bas_offset::Vector{Vector{Int}}
  target_length::Int
  function BasisBatcher(basis1::BasisSet, basis2::BasisSet, target_length::Int=100)
    n4sh = Vector{Int}[]
    push!(n4sh, Int[n_ao(ash, basis1.cartesian) for ash in basis1])
    push!(n4sh, Int[n_ao(ash, basis2.cartesian) for ash in  basis2])
    n_max = maximum.(n4sh)
    # Offset list for each shell, used to map shell index to orbital index
    bas_offset = cumsum.(vcat.(0, n4sh))
    basis = combine(basis1, basis2)
    new(basis, n4sh, n_max, bas_offset, target_length)
  end
end

"""
    buffer_size_3idx(bb::BasisBatcher)

  Return the buffer size needed in the 3-index integral calculation.
  
  The buffer has to be of type `Buffer{Cdouble}(lenbuf)` or `ThreadsBuffer{Cdouble}(lenbuf)`.
"""
buffer_size_3idx(bb::BasisBatcher) = bb.n_max[1]^2*bb.n_max[2]

"""
    BasisBatch

  A structure to represent a batch of basis functions. The structure is used to loop over
  basis functions in a batched manner. The structure is used in conjunction with [`BasisBatcher`](@ref).
"""
struct BasisBatch
  range::UnitRange{Int}
  shrange::UnitRange{Int}
  bb::BasisBatcher
end

n4sh(bb::BasisBatch, i::Int) = bb.bb.n4sh[i]
n_max(bb::BasisBatch, i::Int) = bb.bb.n_max[i]
bas_offset(bb::BasisBatch, i::Int) = bb.bb.bas_offset[i]

"""
    range(bb::BasisBatch)

  Return the range of basis functions in the batch.
"""
Base.range(bb::BasisBatch) = bb.range

"""
    max_batch_length(bb::BasisBatcher)

  Return the maximum batch length.
"""
function max_batch_length(bb::BasisBatcher)
  max_len = 0
  for b in bb
    max_len = max(max_len, length(range(b)))
  end
  return max_len
end

function Base.iterate(bb::BasisBatcher, state=1)
  ib = length(bb.n4sh)
  full_shell_range = shell_range(bb.basis, ib)
  len_full_shell = length(full_shell_range)
  if state > len_full_shell
    return nothing
  end
  lenP = 0
  for P in state:len_full_shell
    nP = bb.n4sh[ib][P]
    new_lenP = lenP + nP
    # check what is closer to the target_length
    if abs(new_lenP - bb.target_length) < abs(lenP - bb.target_length) || lenP == 0
      lenP = new_lenP
    else
      range = (1:lenP) .+ bb.bas_offset[ib][state]
      return BasisBatch(range, full_shell_range[state:P-1], bb), P
    end
  end
  range = (1:lenP) .+ bb.bas_offset[ib][state]
  return BasisBatch(range, full_shell_range[state:end], bb), len_full_shell + 1
end