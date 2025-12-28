# bufvec.jl - Type-stable buffered vector with parametrized storage

"""
    BufVec{T,A<:AbstractVector{T}} <: AbstractVector{T}

Type-stable buffered vector with pre-allocated storage and parametrized buffer type.

The buffer does NOT auto-grow. Attempting to push beyond capacity will error
(unless `@inbounds` is used, which skips the check).

# Type Parameters
- `T` - Element type
- `A` - Buffer type (e.g., `Vector{T}`, `SubArray{T}`, etc.)

# Fields
- `data::A` - Pre-allocated storage buffer
- `length::Int` - Current number of elements in use

# Example
```julia
# Create a buffer from a pre-allocated vector
data = Vector{Float64}(undef, 100)
buf = BufVec(data)

# Add elements (errors if exceeding capacity)
push!(buf, 1.0)
push!(buf, 2.0)

# Access elements
x = buf[1]

# Iterate with SIMD
s = 0.0
@inbounds @simd for i in 1:length(buf)
  s += buf[i]
end

# Clear for reuse
empty!(buf)
```
"""
mutable struct BufVec{T,A<:AbstractVector{T}} <: AbstractVector{T}
  data::A
  len::Int
  
  function BufVec{T,A}(data::A) where {T,A<:AbstractVector{T}}
    new{T,A}(data, 0)
  end
  
  function BufVec{T,A}(data::A, len::Int) where {T,A<:AbstractVector{T}}
    if len < 0 || len > length(data)
      throw(ArgumentError("length must be between 0 and $(length(data)), got $len"))
    end
    new{T,A}(data, len)
  end
end

# Convenience constructors
BufVec(data::A) where {T,A<:AbstractVector{T}} = BufVec{T,A}(data)
BufVec(data::A, length::Int) where {T,A<:AbstractVector{T}} = BufVec{T,A}(data, length)

# AbstractArray interface
Base.size(buf::BufVec) = (buf.len,)
Base.length(buf::BufVec) = buf.len
Base.eltype(::Type{BufVec{T,A}}) where {T,A} = T

Base.@propagate_inbounds function Base.getindex(buf::BufVec, i::Int)
  @boundscheck checkbounds(buf, i)
  return buf.data[i]
end

Base.@propagate_inbounds function Base.setindex!(buf::BufVec, val, i::Int)
  @boundscheck checkbounds(buf, i)
  buf.data[i] = val
  return val
end

# Capacity management
@inline capacity(buf::BufVec) = length(buf.data)
@inline is_full(buf::BufVec) = buf.len == capacity(buf)
@inline Base.isempty(buf::BufVec) = buf.len == 0

# Modification operations
Base.@propagate_inbounds function Base.push!(buf::BufVec{T}, val) where {T}
  @boundscheck begin
    if is_full(buf)
      throw(ArgumentError("Buffer is full (capacity=$(capacity(buf)))"))
    end
  end
  buf.len += 1
  buf.data[buf.len] = val
  return buf
end

Base.@propagate_inbounds function Base.append!(buf::BufVec{T}, items) where {T}
  n_items = length(items)
  new_length = buf.len + n_items
  
  if new_length > capacity(buf)
    throw(ArgumentError("Cannot append $n_items items: would exceed capacity $(capacity(buf))"))
  end

  for item in items
    buf.len += 1
    buf.data[buf.len] = item
  end
  
  return buf
end

Base.@propagate_inbounds function Base.pop!(buf::BufVec)
  if isempty(buf)
    throw(ArgumentError("Buffer is empty"))
  end
  val = buf.data[buf.len]
  buf.len -= 1
  return val
end

function Base.empty!(buf::BufVec)
  buf.len = 0
  return buf
end

function Base.resize!(buf::BufVec, n::Int)
  if n < 0 || n > capacity(buf)
    throw(ArgumentError("Cannot resize to $n: must be between 0 and $(capacity(buf))"))
  end
  buf.len = n
  return buf
end

Base.sizehint!(buf::BufVec, n::Int) = buf

# Iteration
Base.@propagate_inbounds function Base.iterate(buf::BufVec, state=1)
  state > buf.len && return nothing
  val = buf.data[state]
  return (val, state + 1)
end

# Display
function Base.show(io::IO, buf::BufVec{T,A}) where {T,A}
  print(io, "BufVec{$T,$A}(length=$(buf.len), capacity=$(capacity(buf)))")
  if buf.len > 0
    print(io, " [")
    for i in 1:buf.len
      print(io, " ", buf[i])
    end
    print(io, "]")
  end
end

function Base.show(io::IO, ::MIME"text/plain", buf::BufVec{T,A}) where {T,A}
  if get(io, :compact, false)
    print(io, "BufVec{$T,$A}(length=$(buf.len), capacity=$(capacity(buf)))")
    if buf.len > 0
      print(io, " [")
      for i in 1:buf.len
        print(io, " ", buf[i])
      end
      print(io, "]")
    end
  else
    println(io, "BufVec{$T,$A} with $(buf.len) elements (capacity=$(capacity(buf))):")
    if buf.len > 0
      Base.print_array(io, buf[1:buf.len])
    end
  end
end

# Conversion and copying
Base.Vector(buf::BufVec{T}) where {T} = buf.len > 0 ? buf.data[1:buf.len] : T[]

function Base.copy(buf::BufVec{T,A}) where {T,A}
  new_data = copy(buf.data)
  return BufVec(new_data, buf.len)
end

Base.@propagate_inbounds function Base.copyto!(dest::BufVec, src::BufVec)
  if length(src) > capacity(dest)
    throw(ArgumentError("Source length $(length(src)) exceeds destination capacity $(capacity(dest))"))
  end
  copyto!(dest.data, 1, src.data, 1, src.len)
  dest.len = src.len
  return dest
end

# Comparison
@inline function Base.:(==)(a::BufVec, b::BufVec)
  a.len == b.len || return false
  @inbounds for i in 1:a.len
    a.data[i] == b.data[i] || return false
  end
  return true
end
