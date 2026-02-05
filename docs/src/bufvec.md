# BufVec - Pre-Allocated Vector

```@meta
CurrentModule = ElemCo.Utils
```

## Overview

`BufVec{T,A}` is a type-stable, pre-allocated vector implementation that implements the `AbstractVector` interface. It provides efficient memory management with **fixed capacity** (no auto-growth) and parametrized buffer type for maximum flexibility.

## Key Features

- **Type Stability**: All operations are type-stable, enabling Julia's compiler optimizations
- **Parametrized Buffer**: Can use any `AbstractVector` as storage (Vector, SubArray, etc.)
- **Fixed Capacity**: Does NOT auto-grow; errors when capacity exceeded (predictable memory usage)
- **AbstractVector Interface**: Full compliance with Julia's `AbstractVector` protocol
- **SIMD Support**: Works with Julia's native `@inbounds` and `@simd` annotations
- **Memory Efficient**: Separates capacity from length for efficient push operations

## Structure

```julia
mutable struct BufVec{T,A<:AbstractVector{T}} <: AbstractVector{T}
    data::A         # Pre-allocated storage buffer (parametrized)
    length::Int     # Current number of elements
end
```

## Construction

The buffer must be pre-allocated before creating a `BufVec`:

```julia
# Create from a pre-allocated vector
data = Vector{Float64}(undef, 100)
buf = BufVec(data)

# Create with initial length
data = [1.0, 2.0, 3.0, 0.0, 0.0]
buf = BufVec(data, 3)  # First 3 elements are "in use"

# Works with any AbstractVector
view_data = @view some_array[1:50]
buf = BufVec(view_data)
```

## Basic Operations

### Capacity Management

The buffer has **fixed capacity** set at construction. It will not auto-grow.

```julia
buf = BufVec(Vector{Int}(undef, 10))

# Query capacity
cap = capacity(buf)      # Returns 10
full = is_full(buf)       # Check if at capacity
empty = isempty(buf)     # Check if empty

# sizehint! is a no-op (provided for compatibility)
sizehint!(buf, 1000)     # Does nothing - capacity is fixed
```

**Important**: Attempting to push beyond capacity will throw an `ArgumentError`:

```julia
buf = BufVec(Vector{Int}(undef, 3))
push!(buf, 1)
push!(buf, 2)
push!(buf, 3)
push!(buf, 4)  # ERROR: Buffer is full (capacity=3)
```

You can skip the capacity check using `@inbounds` (unsafe!):

```julia
@inbounds push!(buf, 4)  # No error, but undefined behavior!
```

### Adding Elements

```julia
buf = BufVec(Vector{Int}(undef, 10))

# Add single element
push!(buf, 42)

# Add multiple elements
append!(buf, [1, 2, 3, 4])

# Chaining
push!(push!(buf, 10), 20)
```

**Note**: Both `push!` and `append!` will error if capacity is exceeded.

### Removing Elements

```julia
buf = BufVec(Vector{Int}(undef, 10))
append!(buf, [1, 2, 3])

val = pop!(buf)  # Returns 3, length now 2

empty!(buf)      # Clear all elements (capacity unchanged)
```

### Resizing

```julia
buf = BufVec(Vector{Float64}(undef, 10))
resize!(buf, 5)  # Set length to 5 (must be ≤ capacity)

# Initialize elements
for i in 1:5
    buf[i] = Float64(i)
end
```

**Note**: `resize!` will error if `n > capacity(buf)`.

## Indexing

`BufVec` supports standard Julia indexing:

```julia
buf = BufVec(Vector{Int}(undef, 10))
push!(buf, 42)
push!(buf, 17)

# Read
val = buf[1]          # 42
val = buf[end]        # 17

# Write
buf[1] = 100

# Bounds checking (skip with @inbounds)
@inbounds val = buf[1]
```

## Iteration

`BufVec` implements the full iteration interface:

```julia
buf = BufVec(Vector{Int}(undef, 10))
append!(buf, [1, 2, 3, 4, 5])

# Standard iteration
for val in buf
    println(val)
end

# Enumerate
for (i, val) in enumerate(buf)
    println("buf[$i] = $val")
end

# Collect
vec = collect(buf)  # Convert to Vector

# Map, filter, etc.
doubled = map(x -> 2x, buf)
evens = filter(iseven, buf)
```

## SIMD Support

`BufVec` works naturally with Julia's `@simd` macro:

```julia
function sum_buffer(buf::BufVec{Float64})
    s = 0.0
    @inbounds @simd for i in 1:length(buf)
        s += buf[i]
    end
    return s
end

buf = BufVec(Vector{Float64}(undef, 1000))
for i in 1:1000
    push!(buf, Float64(i))
end

total = sum_buffer(buf)
```

## Type Stability

All operations are designed for type stability:

```julia
using Test

data = Vector{Float64}(undef, 10)
buf = BufVec(data)

@inferred push!(buf, 1.0)
@inferred buf[1]
@inferred length(buf)
@inferred capacity(buf)

# Type-stable iteration
function sum_elements(buf::BufVec{Float64})
    s = 0.0
    for val in buf
        s += val
    end
    return s
end

@inferred sum_elements(buf)
```

## Performance Tips

1. **Pre-allocate with correct capacity**: Since the buffer doesn't auto-grow, allocate enough space upfront

   ```julia
   # Estimate maximum size
   max_size = estimate_max_results()
   data = Vector{Float64}(undef, max_size)
   buf = BufVec(data)
   ```

2. **Use @inbounds**: In performance-critical loops where you know bounds are safe

   ```julia
   @inbounds for i in 1:length(buf)
       buf[i] = compute_value(i)
   end
   ```

3. **Use @simd**: For simple operations on large arrays

   ```julia
   function process(buf::BufVec{Float64})
       @inbounds @simd for i in 1:length(buf)
           buf.data[i] *= 2.0
       end
   end
   ```

4. **Reuse buffers**: Clear and reuse instead of allocating new ones

   ```julia
   empty!(buf)  # Resets length to 0, capacity unchanged
   # Reuse buf in next iteration
   ```

## API Reference

### Constructors

- `BufVec(data::AbstractVector{T})`: Create from pre-allocated buffer with length=0
- `BufVec(data::AbstractVector{T}, length::Int)`: Create from buffer with initial length

### Capacity

- `capacity(buf)`: Current maximum capacity (fixed)
- `is_full(buf)`: Check if at capacity
- `isempty(buf)`: Check if empty
- `sizehint!(buf, n)`: No-op (provided for compatibility)

### Modification

- `push!(buf, val)`: Add element (errors if full)
- `append!(buf, iter)`: Add multiple elements (errors if capacity exceeded)
- `pop!(buf)`: Remove and return last element
- `empty!(buf)`: Remove all elements (capacity unchanged)
- `resize!(buf, n)`: Set length to n (must be ≤ capacity)

### Indexing

- `buf[i]`: Get element at index i
- `buf[i] = val`: Set element at index i
- `length(buf)`: Number of elements in use
- `size(buf)`: Tuple with length

### Iteration

- `for val in buf`: Iterate over values
- `collect(buf)`: Convert to Vector
- Standard iteration protocol

### Utilities

- `copy(buf)`: Deep copy
- `copyto!(dest, src)`: Copy contents
- `Vector(buf)`: Convert to Vector (copies only used elements)
- `buf1 == buf2`: Equality comparison
