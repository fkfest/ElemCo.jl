"""
    DescDict

A module for an ordered descriptive dictionary.

The module provides the `ODDict` type, which is an ordered dictionary that stores a description
for each key-value pair.
"""
module DescDict

export ODDict, getdescription, setdescription!, descriptions
export last_key, last_value

"""
    ODDict{K, V}

An ordered descriptive dictionary that maps keys of type `K` to values of type `V`. 
Additionally, it stores a description of each key-value pair in the form of a string. 

The values are stored in an ordered dictionary, which means that the order of the key-value 
pairs is preserved.

### Examples

```julia
julia> dict = ODDict("a" => 1, "b" => 2)
ODDict{String, Int64}
  a => 1 ()
  b => 2 ()

julia> dict["a"]
1

julia> dict["c"] = 3
3

julia> push!(dict, "d" => 4)
ODDict{String, Int64}
  a => 1 ()
  b => 2 ()
  c => 3 ()
  d => 4 ()

julia> dict["a"] = (5, "this is a")
(5, "this is a")

julia> dict
ODDict{String, Int64}
  a => 5 (this is a)
  b => 2 ()
  c => 3 ()
  d => 4 ()

julia> push!(dict, "a" => (5,"this is a"), "e" => (6,"this is e"))
ODDict{String, Int64}
  b => 2 ()
  c => 3 ()
  d => 4 ()
  a => 5 (this is a)
  e => 6 (this is e)
```
"""
mutable struct ODDict{K, V} <: AbstractDict{K, V}
  keys::Vector{K}
  values::Vector{V}
  descriptions::Vector{String}
  function ODDict(keys::Vector{K}, values::Vector{V}, descriptions::Vector{String}) where {K, V}
    if length(keys) != length(values) || length(keys) != length(descriptions)
      throw(ArgumentError("Length of keys, values, and descriptions must be the same"))
    end
    new{K, V}(keys, values, descriptions)
  end
end

function ODDict{K, V}(pairs::Pair{K, V}...) where {K, V}
  keys = K[]
  values = V[]
  descriptions = String[]
  for (key, value) in pairs
    push!(keys, key)
    push!(values, value)
    push!(descriptions, "")
  end
  return ODDict(keys, values, descriptions)
end

function ODDict{K, V}(pairs::Pair{K, Tuple{V, String}}...) where {K, V}
  keys = K[]
  values = V[]
  descriptions = String[]
  for (key, value_desc) in pairs
    push!(keys, key)
    push!(values, value_desc[1])
    push!(descriptions, value_desc[2])
  end
  return ODDict(keys, values, descriptions)
end

function Base.getindex(dict::ODDict{K, V}, key::K) where {K, V}
  index = findfirst(isequal(key), dict.keys)
  if isnothing(index)
    throw(KeyError(key))
  end
  return dict.values[index]
end

function Base.setindex!(dict::ODDict{K, V}, value::V, key::K) where {K, V}
  index = findfirst(isequal(key), dict.keys)
  if isnothing(index)
    push!(dict.keys, key)
    push!(dict.values, value)
    push!(dict.descriptions, "")
  else
    dict.values[index] = value
  end
end

function Base.setindex!(dict::ODDict{K, V}, value_desc::Tuple{V,String}, key::K) where {K, V}
  index = findfirst(isequal(key), dict.keys)
  if isnothing(index)
    push!(dict.keys, key)
    push!(dict.values, value_desc[1])
    push!(dict.descriptions, value_desc[2])
  else
    dict.values[index] = value_desc[1]
    dict.descriptions[index] = value_desc[2]
  end
end

function Base.getkey(dict::ODDict{K, V}, key::K, default) where {K, V}
  if key in dict.keys
    return key
  else
    return default
  end
end

function Base.keys(dict::ODDict)
  return copy(dict.keys)
end

function Base.values(dict::ODDict)
  return copy(dict.values)
end

"""
    descriptions(dict::ODDict)

Get the descriptions of the key-value pairs in the dictionary.
"""
function descriptions(dict::ODDict)
  return copy(dict.descriptions)
end

function Base.haskey(dict::ODDict, key)
  return key in dict.keys
end

function Base.pairs(dict::ODDict)
  return zip(dict.keys, dict.values)
end

function Base.length(dict::ODDict)
  return length(dict.keys)
end

function Base.delete!(dict::ODDict{K, V}, key::K) where {K, V}
  index = findfirst(isequal(key), dict.keys)
  if isnothing(index)
    throw(KeyError(key))
  end
  deleteat!(dict.keys, index)
  deleteat!(dict.values, index)
  deleteat!(dict.descriptions, index)
end

"""
    getdescription(dict, key)

Get the description of the key-value pair with the given key.
"""
function getdescription(dict::ODDict{K, V}, key::K) where {K, V}
  index = findfirst(isequal(key), dict.keys)
  if isnothing(index)
    throw(KeyError(key))
  end
  return dict.descriptions[index]
end

"""
    setdescription!(dict, description, key)

Set the description of the key-value pair with the given key.
"""
function setdescription!(dict::ODDict{K, V}, description::String, key::K) where {K, V}
  index = findfirst(isequal(key), dict.keys)
  if isnothing(index)
    throw(KeyError(key))
  end
  dict.descriptions[index] = description
end

Base.firstindex(dict::ODDict) = firstindex(dict.keys)
Base.lastindex(dict::ODDict) = lastindex(dict.keys)

function Base.first(dict::ODDict)
  return (first(dict.keys), first(dict.values), first(dict.descriptions))
end

function Base.last(dict::ODDict)
  return (last(dict.keys), last(dict.values), last(dict.descriptions))
end

function Base.iterate(dict::ODDict, state=1)
  if state > lastindex(dict)
    return nothing
  end
  key = dict.keys[state]
  value = dict.values[state]
  description = dict.descriptions[state]
  return ((key, value, description), state+1)
end

function Base.iterate(rdict::Iterators.Reverse{ODDict}, state=lastindex(rdict.itr))
  if state < 1
    return nothing
  end
  dict = rdict.itr
  key = dict.keys[state]
  value = dict.values[state]
  description = dict.descriptions[state]
  return ((key, value, description), state-1)
end

function Base.map!(f, dict::ODDict)
  for i in firstindex(dict):lastindex(dict)
    @inbounds dict.values[i] = f(dict.values[i])
  end
  return dict
end

function Base.map(f, dict1::ODDict{K, V}, dict2::ODDict{K, V}) where {K, V}
  dict = copy(dict1)
  for i in firstindex(dict1):lastindex(dict1)
    @assert dict1.keys[i] == dict2.keys[i] "Keys do not match"
    @inbounds dict.values[i] = f(dict1.values[i], dict2.values[i])
  end
  return dict
end

function Base.map(f, dict::ODDict{K, V}) where {K, V}
  dict1 = copy(dict)
  map!(f, dict1)
  return dict1
end

function Base.merge(dict1::ODDict, dict2::ODDict)
  dict = copy(dict1)
  for (key, value, description) in dict2
    push!(dict, key, value, description)
  end
  return dict
end

function Base.merge(dict1::ODDict, pairs::Pair{K, V}...) where {K, V}
  dict = copy(dict1)
  push!(dict, pairs...)
  return dict
end

function Base.merge(dict1::ODDict, pairs::Pair{K, Tuple{V, String}}...) where {K, V}
  dict = copy(dict1)
  push!(dict, pairs...)
  return dict
end

function Base.copy(dict::ODDict)
  return ODDict(copy(dict.keys), copy(dict.values), copy(dict.descriptions))
end

function Base.push!(dict::ODDict{K, V}, key::K, value::V, description::String="") where {K, V}
  if key in dict.keys
    delete!(dict, key)
  end
  push!(dict.keys, key)
  push!(dict.values, value)
  push!(dict.descriptions, description)
  return dict
end

function Base.push!(dict::ODDict{K, V}, pair::Pair{K, V}, description::String="") where {K, V}
  push!(dict, pair.first, pair.second, description)
end

function Base.push!(dict::ODDict{K, V}, pair::Pair{K, Tuple{V, String}}) where {K, V}
  push!(dict, pair.first, pair.second...)
end

function Base.push!(dict::ODDict{K, V}, pairs::Pair{K, V}...) where {K, V}
  for pair in pairs
    push!(dict, pair)
  end
  return dict
end

function Base.push!(dict::ODDict{K, V}, pairs::Pair{K, Tuple{V, String}}...) where {K, V}
  for pair in pairs
    push!(dict, pair)
  end
  return dict
end

function Base.push!(dict::ODDict{K, V}, dict2::ODDict{K, V}) where {K, V}
  for (key, value, description) in dict2
    push!(dict, key, value, description)
  end
  return dict
end

# function Base.show(io::IO, dict::ODDict{K, V}) where {K, V}
#   println(io, "ODDict{", K, ", ", V, "}")
#   for (key, value, description) in zip(dict.keys, dict.values, dict.descriptions)
#     println(io, "  ", key, " => ", value, " ($description)")
#   end
# end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, dict::ODDict{K, V}) where {K, V}
  println(io, "ODDict{", K, ", ", V, "}")
  for (key, value, description) in zip(dict.keys, dict.values, dict.descriptions)
    println(io, "  ", key, " => ", value, " ($description)")
  end
end

"""
    last_key(dict::ODDict)

Get the last key in the dictionary.
"""
function last_key(dict::ODDict)
  return last(dict.keys)
end

"""
    last_value(dict::ODDict)

Get the last value in the dictionary.
"""
function last_value(dict::ODDict)
  return last(dict.values)
end

end #module