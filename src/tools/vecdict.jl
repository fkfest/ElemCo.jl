"""
    VecDicts

A module for a vector-based dictionary.

The module provides the `VecDict` type, which is a vector-based dictionary that stores
keys and values in vectors.

It supports standard dictionary operations such as indexing, setting values,
checking for keys, deleting keys, and iterating over key-value pairs.

The `VecDict` type is designed for efficiency of adding new values, but is much slower
for lookups compared to standard dictionaries, as it requires a linear search through the keys.
It is useful in scenarios where the order and speed of insertion matters and lookups are infrequent,
e.g., when collecting data before final processing or for merging to a standard dictionary.

# Examples
```julia
using VecDicts
dict = VecDict{String, Int}("a" => 1, "b" => 2)
dict["c"] = 3
println(dict["a"])  # Output: 1
println(keys(dict)) # Output: ["a", "b", "c"]
println(values(dict)) # Output: [1, 2, 3]
delete!(dict, "b")
println(length(dict)) # Output: 2
standard_dict = Dict("a" => 2, "d" => 4)
mergewith!(+, standard_dict, dict) 
println(standard_dict) # Output: Dict("c" => 3, "a" => 3, "d" => 4)
```
"""
module VecDicts

export VecDict, getvalue, setvalue!, values
export getkeyat, setkeyat!, getvalueat, setvalueat!, getat, setat!

"""
    VecDict{K, V}

A vector-based dictionary that maps keys of type `K` to values of type `V`. 
The values are stored in a vector, which means that the order of the key-value pairs is preserved.
"""
struct VecDict{K, V} <: AbstractDict{K, V}
  keys::Vector{K}
  values::Vector{V}
end

function VecDict{K, V}(pairs::Pair{K, V}...) where {K, V}
  keys = K[]
  values = V[]
  for (key, value) in pairs
    push!(keys, key)
    push!(values, value)
  end
  return VecDict(keys, values)
end

function VecDict(keys::AbstractVector{K}, values::AbstractVector{V}) where {K, V}
  @assert length(keys) == length(values) "Keys and values must have the same length"
  return VecDict(Vector{K}(keys), Vector{V}(values))
end



Base.@propagate_inbounds function Base.getindex(dict::VecDict{K, V}, key::K) where {K, V}
  index = findlast(isequal(key), dict.keys)
  if isnothing(index)
    throw(KeyError(key))
  end
  return dict.values[index]
end

Base.@propagate_inbounds function Base.setindex!(dict::VecDict{K, V}, value::V, key::K) where {K, V}
  push!(dict.keys, key)
  push!(dict.values, value)
end

"""
    getvalueat(dict::VecDict, index::Int)

Get the value at the specified index in the `VecDict`.
"""
Base.@propagate_inbounds function getvalueat(dict::VecDict{K, V}, index::Int) where {K, V}
  return dict.values[index]
end

"""
    setvalueat!(dict::VecDict, index::Int, value)

Set the value at the specified index in the `VecDict`.
"""
Base.@propagate_inbounds function setvalueat!(dict::VecDict{K, V}, index::Int, value::V) where {K, V}
  dict.values[index] = value
end

"""
    getkeyat(dict::VecDict, index::Int)

Get the key at the specified index in the `VecDict`.
"""
Base.@propagate_inbounds function getkeyat(dict::VecDict{K, V}, index::Int) where {K, V}
  return dict.keys[index]
end

"""
    setkeyat!(dict::VecDict, index::Int, key)

Set the key at the specified index in the `VecDict`.
"""
Base.@propagate_inbounds function setkeyat!(dict::VecDict{K, V}, index::Int, key::K) where {K, V}
  dict.keys[index] = key
end

"""
    getat(dict::VecDict, index::Int)::Tuple{K, V}

Get the key-value pair at the specified index in the `VecDict`.
"""
Base.@propagate_inbounds function getat(dict::VecDict{K, V}, index::Int) where {K, V}
  return (dict.keys[index], dict.values[index])
end

"""
    setat!(dict::VecDict, index::Int, key, value)

Set the key-value pair at the specified index in the `VecDict`.
"""
Base.@propagate_inbounds function setat!(dict::VecDict{K, V}, index::Int, key::K, value::V) where {K, V}
  dict.keys[index] = key
  dict.values[index] = value
end

@inline function Base.keys(dict::VecDict)
  return dict.keys
end

@inline function Base.values(dict::VecDict)
  return dict.values
end

function Base.getkey(dict::VecDict{K, V}, key::K, default) where {K, V}
  if key in dict.keys
    return key
  else
    return default
  end
end

function Base.haskey(dict::VecDict, key)
  return key in dict.keys
end

function Base.pairs(dict::VecDict)
  return zip(dict.keys, dict.values)
end

function Base.length(dict::VecDict)
  return length(dict.keys)
end

Base.@propagate_inbounds function Base.sizehint!(dict::VecDict, n::Integer)
  sizehint!(dict.keys, n)
  sizehint!(dict.values, n)
end

Base.@propagate_inbounds function Base.resize!(dict::VecDict, n::Integer)
  resize!(dict.keys, n)
  resize!(dict.values, n)
end

Base.@propagate_inbounds function Base.delete!(dict::VecDict{K, V}, key::K) where {K, V}
  index = findlast(isequal(key), dict.keys)
  if isnothing(index)
    throw(KeyError(key))
  end
  deleteat!(dict.keys, index)
  deleteat!(dict.values, index)
end

Base.@propagate_inbounds function Base.delete!(dict::VecDict{K, V}, keys::K...) where {K, V}
  for key in keys
    delete!(dict, key)
  end
end

function Base.empty!(dict::VecDict)
  empty!(dict.keys)
  empty!(dict.values)
end

Base.firstindex(dict::VecDict) = firstindex(dict.keys)
Base.lastindex(dict::VecDict) = lastindex(dict.keys)

function Base.first(dict::VecDict)
  return (first(dict.keys), first(dict.values))
end

function Base.last(dict::VecDict)
  return (last(dict.keys), last(dict.values))
end

Base.@propagate_inbounds function Base.iterate(dict::VecDict, state=1)
  if state > lastindex(dict)
    return nothing
  end
  key = dict.keys[state]
  value = dict.values[state]
  return ((key, value), state+1)
end

Base.@propagate_inbounds function Base.iterate(rdict::Iterators.Reverse{VecDict}, state=lastindex(rdict.itr))
  if state < 1
    return nothing
  end
  dict = rdict.itr
  key = dict.keys[state]
  value = dict.values[state]
  return ((key, value), state-1)
end

Base.@propagate_inbounds function Base.map!(f, dict::VecDict)
  for i in firstindex(dict):lastindex(dict)
    dict.values[i] = f(dict.values[i])
  end
  return dict
end

Base.@propagate_inbounds function Base.map(f, dict1::VecDict{K, V}, dict2::VecDict{K, V}) where {K, V}
  dict = copy(dict1)
  for i in firstindex(dict1):lastindex(dict1)
    @assert dict1.keys[i] == dict2.keys[i] "Keys do not match"
    dict.values[i] = f(dict1.values[i], dict2.values[i])
  end
  return dict
end

Base.@propagate_inbounds function Base.map(f, dict::VecDict{K, V}) where {K, V}
  dict1 = copy(dict)
  map!(f, dict1)
  return dict1
end

function Base.copy(dict::VecDict)
  return VecDict(copy(dict.keys), copy(dict.values))
end

Base.@propagate_inbounds function Base.push!(dict::VecDict{K, V}, key::K, value::V) where {K, V}
  push!(dict.keys, key)
  push!(dict.values, value)
  return dict
end

Base.@propagate_inbounds function Base.push!(dict::VecDict{K, V}, pair::Pair{K, V}) where {K, V}
  push!(dict, pair.first, pair.second)
end

Base.@propagate_inbounds function Base.push!(dict::VecDict{K, V}, pair::Pair{K, Tuple{V, String}}) where {K, V}
  push!(dict, pair.first, pair.second...)
end

Base.@propagate_inbounds function Base.push!(dict::VecDict{K, V}, pairs::Vararg{Pair{K, V},N}) where {K, V, N}
  for pair in pairs
    push!(dict, pair.first, pair.second::V)
  end
  return dict
end

Base.@propagate_inbounds function Base.push!(dict::VecDict{K, V}, dict2::VecDict{K, V}) where {K, V}
  for (key, value) in dict2
    push!(dict, key, value)
  end
  return dict
end

end #module