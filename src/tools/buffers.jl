"""
    Buffers module

This module contains functions to handle buffers.

The [`Buffer`](@ref) object is used to store data of type `T` with an offset,
while the [`ThreadsBuffer`](@ref) object is used to store data of type `T` with an offset for each thread.

The buffers are used to store data in a contiguous memory block and to avoid memory allocation in loops.
The buffers can be used with [`alloc!`](@ref) to allocate tensors of given dimensions,
[`drop!`](@ref) to drop tensors from the buffer, and [`reset!`](@ref) to reset the buffer to the initial state.

Alternativelly, the buffers can be reshaped with [`reshape_buf!`](@ref) to use the same memory block for different tensors
or to allocate tensors with a specific offset.

In any case, the `::ThreadsBuffer` buffers should be released after use with [`Buffers.release!`](@ref) or [`reset!`](@ref).

If some functions complain about tensors being aliases or if the tensors will be used in C, 
the [`neuralyze`](@ref) function can be used to wipe the memory about the origin of the tensor.
"""
module Buffers

export Buffer, ThreadsBuffer
export alloc!, drop!, reset!
export reshape_buf!
export used, nbuffers, with_buffer
export neuralyze

"""
    Buffer{T}

Buffer object to store data of type `T` with an offset.

The buffer allocates an extra element at the beginning to store the current offset.
If the buffer is used with [`reshape_buf!`](@ref), the offset is set to zero.
"""
struct Buffer{T}
  data::Vector{T}
  function Buffer{T}(len::Int) where T
    data = Vector{T}(undef, len+1)
    data[1] = one(T)
    new(data)
  end
end

Buffer(len::Int) = Buffer{Float64}(len)

Base.length(buf::Buffer) = length(buf.data) - 1

"""
    used(buf::Buffer)

Return the number of elements used in buffer `buf`.

If the buffer is used with [`reshape_buf!`](@ref), `-1` is returned.
"""
function used(buf::Buffer)
  return Int(buf.data[1]) - 1
end

"""
    alloc!(buf::Buffer{T}, dims...)

  Allocate tensor of given dimensions in buffer `buf`.

# Example
```julia
julia> buf = Buffer(100000)
julia> A = alloc!(buf, 10, 10, 20) # 10x10x20 tensor
julia> B = alloc!(buf, 10, 10, 10) # 10x10x10 tensor starting after A
julia> C = alloc!(buf, 10, 20) # 10x20 tensor starting after B
julia> rand!(B)
julia> rand!(C)
julia> An = neuralyze(A) # tensor without origin
julia> @tensor An[i,j,k] = B[i,j,l] * C[l,k]
```
"""
function alloc!(buf::Buffer{T}, dims...) where {T}
  @assert buf.data[1] >= one(T) "Buffer is used with reshape_buf! and must be reset!"
  start = Int(buf.data[1]) + 1
  len = prod(dims)
  stop = start + len - 1
  @assert stop <= length(buf.data) "Buffer overflow!"
  buf.data[1] += len
  return reshape(view(buf.data, start:stop), dims)
end

"""
    drop!(buf::Buffer, tensor::AbstractArray...)

  Drop tensor(s) from buffer `buf`.

  Only last tensors can be dropped.
"""
function drop!(buf::Buffer, tensor::AbstractArray...)
  # order tensor from last to first
  order = sortperm([pointer(t) for t in tensor], rev=true)
  for i in order
    len = length(tensor[i])
    @assert pointer(tensor[i]) == pointer(buf.data, Int(buf.data[1])-len+1) "Tensor must be the last allocated!"
    buf.data[1] -= len
  end
end

"""
    reset!(buf::Buffer{T})

  Reset buffer `buf` to the initial state.
"""
function reset!(buf::Buffer{T}) where {T}
  buf.data[1] = one(T)
end

"""
    neuralyze(tensor::AbstractArray)

  Wipe the memory about origin of `tensor`.

  `tensor` is a (contiguous!) array that is a (possibly reshaped) view of a larger array.
  Return the same tensor pointing to the same memory, 
  but without the information about the origin.
  To be used together with [`alloc!`](@ref) or [`reshape_buf!`](@ref) to trick `Base.mightalias`.

!!! warning "Warning" 
    Note that this function is unsafe and should be used with caution!
    If too much memory is wiped, Julia might garbage-collect the
    original array and the tensor will point to invalid memory.

!!! tip "Tip" 
    One can use `GC.@preserve` to prevent the garbage collection of the original array.

# Example
```julia
julia> buf = Buffer(100000)
julia> A = alloc(buf, 10, 10, 20) # 10x10x20 tensor
julia> B = alloc(buf, 10, 10, 10) # 10x10x10 tensor starting after A
julia> C = alloc(buf, 10, 20) # 10x20 tensor starting after B
julia> rand!(B)
julia> rand!(C)
julia> An = neuralyze(A) # tensor without origin but pointing to the same memory
julia> @tensor An[i,j,k] = B[i,j,l] * C[l,k]
```
"""
function neuralyze(tensor::AbstractArray)
  @assert iscontiguous_tensor(tensor) "Tensor must be contiguous!"
  return unsafe_wrap(Array, pointer(tensor), size(tensor), own=false)
end

"""
    iscontiguous_tensor(tensor::AbstractArray)

  Check if `tensor` is contiguous.

  Return `true` if `tensor` is a `Vector` or a `SubArray` that is contiguous.
"""
function iscontiguous_tensor(tensor::AbstractArray)
  vtensor = vec(tensor)
  return typeof(vtensor) <: Array || ( typeof(vtensor) <: SubArray && Base.iscontiguous(vtensor) )
end

"""
    reshape_buf!(buf::Buffer{T}, dims...; offset=0)

  Reshape (part of) a buffer to given dimensions (without copying),
  using `offset`.

  It can be used, e.g., for itermediates in tensor contractions.

!!! warning "Warning" 
    Do not use this function together with [`alloc!`](@ref) or [`drop!`](@ref) on the same buffer!

# Example
```julia
julia> buf = Buffer(100000)
julia> A = reshape_buf!(buf, 10, 10, 20) # 10x10x20 tensor
julia> B = reshape_buf!(buf, 10, 10, 10, offset=2000) # 10x10x10 tensor starting at 2001
julia> B .= rand(10,10,10)
julia> C = rand(10,20)
julia> @tensor A[i,j,k] = B[i,j,l] * C[l,k]
```
"""
function reshape_buf!(buf::Buffer{T}, dims...; offset=0) where {T}
  @assert buf.data[1] == one(T) || buf.data[1] == zero(T) "Buffer is used with alloc! and must be reset!"
  buf.data[1] = zero(T)
  len = prod(dims)
  @assert len+offset+1 <= length(buf.data) "Buffer overflow!"
  return reshape(view(buf.data, 2+offset:len+offset+1), dims)
end

"""
    ThreadsBuffer{T}

Buffer object to store data of type `T` with an offset for each thread.

See [`Buffer`](@ref) for more details.

!!! warning "Warning"
    Always [`reset!`](@ref) or [`Buffers.release!`](@ref) the buffer after use!

# Example
```julia
julia> buf = Buffer(100000)
julia> C = alloc!(buf, 10, 10, 20) # 10x10x20 destination tensor on a single thread
julia> tbuf = ThreadsBuffer(100000) # 100000 elements buffer for nthreads() threads
julia> Threads.@threads for k = 1:20
          A = alloc!(tbuf, 10, 10) # 10x10 tensor
          B = alloc!(tbuf, 10, 10) # 10x10 tensor
          rand!(A)
          rand!(B)
          @tensor C[:,:,k][i,j] = A[i,l] * B[l,j]
          reset!(tbuf)
        end
```
""" 
struct ThreadsBuffer{T}
    buffers::Vector{Buffer{T}}
    condition::Threads.Condition
    id::Symbol
end

ThreadsBuffer{T}(len::Int, n::Int=Threads.nthreads()) where T = ThreadsBuffer{T}([Buffer{T}(len) for _ in 1:n], Threads.Condition(), gensym(:tbuffer))
ThreadsBuffer(len::Int, n::Int=Threads.nthreads()) = ThreadsBuffer{Float64}(len, n)

nbuffers(buf::ThreadsBuffer) = length(buf.buffers)

"""
    current_buffer(buf::ThreadsBuffer{T})

Return the buffer of the current thread.

If the buffer is not available, wait until it is released.
"""
function current_buffer(buf::ThreadsBuffer{T}) where {T}
  b::Buffer{T} = get!(task_local_storage(), buf.id) do 
    lock(buf.condition) do
      while isempty(buf.buffers)
        wait(buf.condition)
      end
      pop!(buf.buffers)
    end
  end
  return b
end

Base.length(buf::ThreadsBuffer) = length(current_buffer(buf))

"""
    used(buf::ThreadsBuffer)

Return the number of elements used in buffer `buf` of the current thread.
"""
used(buf::ThreadsBuffer) = used(current_buffer(buf))

"""
    alloc!(buf::ThreadsBuffer{T}, dims...)

  Allocate tensor of given dimensions in buffer `buf`.

  The tensor is allocated in the buffer of the current thread.
"""
function alloc!(buf::ThreadsBuffer{T}, dims...) where {T}
  return alloc!(current_buffer(buf), dims...) 
end

"""
    drop!(buf::ThreadsBuffer, tensor::AbstractArray...)

  Drop tensor(s) from buffer `buf`.

  The tensor is dropped from the buffer of the current thread.
"""
function drop!(buf::ThreadsBuffer, tensor::AbstractArray...)
  drop!(current_buffer(buf), tensor...)
end

"""
    reset!(buf::ThreadsBuffer{T})

  Reset buffer of the current thread to the initial state
  and release the buffer.
"""
function reset!(buf::ThreadsBuffer{T}) where {T}
  reset!(current_buffer(buf))
  release!(buf)
end

"""
    release!(buf::ThreadsBuffer)

Release buffer of the current thread.
"""
function release!(buf::ThreadsBuffer)
  lock(buf.condition) do
    @assert used(current_buffer(buf)) == 0 "Buffer is not empty! Use reset! to release the buffer."
    push!(buf.buffers, current_buffer(buf))
    delete!(task_local_storage(), buf.id)
    notify(buf.condition)
  end
end

"""
    reshape_buf!(buf::ThreadsBuffer{T}, dims...; offset=0)

  Reshape (part of) a buffer of the current thread to given dimensions,
  using `offset`.

  Do not use this function together with [`alloc!`](@ref) or [`drop!`](@ref) on the same buffer.
  Call [`reset!(::ThreadsBuffer)`](@ref) or [`release!`](@ref) after use.
"""
function reshape_buf!(buf::ThreadsBuffer{T}, dims...; offset=0) where {T}
  return reshape_buf!(current_buffer(buf), dims...; offset=offset)
end

"""
    with_buffer(f::Function, buf::ThreadsBuffer)

  Execute function `f` with buffer `buf`.

  The buffer is released after the function is executed.

# Example
```julia
julia> buf = Buffer(100000)
julia> C = alloc!(buf, 10, 10, 20) # 10x10x20 destination tensor on a single thread
julia> tbuf = ThreadsBuffer(100000)
julia> Threads.@threads for k = 1:20
          with_buffer(tbuf) do bu
            A = alloc!(bu, 10, 10) # 10x10 tensor
            B = alloc!(bu, 10, 10) # 10x10 tensor
            rand!(A)
            rand!(B)
            @tensor C[:,:,k][i,j] = A[i,l] * B[l,j]
          end
        end
```
"""
function with_buffer(f::Function, buf::ThreadsBuffer)
  b = current_buffer(buf)
  try
    f(b)
  finally
    reset!(buf)
  end
end

end # module