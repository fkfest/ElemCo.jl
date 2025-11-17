"""
    PairDict

A very lightweight immutable Dict that is just a Tuple of two values, i.e., 
it can hold only one key-value pair.

Useful for merging results into a Dict without creating intermediate Dicts.
"""
struct PairDict{K, V} <: AbstractDict{K, V}
    key::K
    value::V
end

Base.size(dict::PairDict) = 1
Base.length(dict::PairDict) = 1
Base.keys(dict::PairDict) = (dict.key,)
Base.values(dict::PairDict) = (dict.value,)
Base.iterate(dict::PairDict, state=1) = state > 1 ? nothing : ((dict.key, dict.value), state + 1)
Base.getindex(dict::PairDict{K, V}, key::K) where {K, V} = dict.key == key ? dict.value : throw(KeyError(key))

function Base.haskey(dict::PairDict{K, V}, key::K) where {K, V}
  return dict.key == key
end
function Base.pairs(dict::PairDict)
  return ((dict.key, dict.value),)
end