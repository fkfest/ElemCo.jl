"""
    MTensorOperations

Utilities for operations on tensors. This module provides wrappers around 
`TensorOperations.jl` and `StridedViews.jl` macros and functions.

The functions and macros are reexported by `TensorTools` module.
"""
module MTensorOperations
using TensorOperations
using StridedViews
using Buffers
export @mtensor, @mtensoropt
export @tensor, @tensoropt # reexport @tensor, @tensoropt
export @mview, mview
export @buftensor

save_tensorcalls() = false

if save_tensorcalls()
  include("tensoranalyzer.jl")
  write_header4tensorcalls()
end

"""
    mtensor(ex)

Macro for tensor operations with manual allocator.
"""
macro mtensor(ex)
  if save_tensorcalls()
    print_tensor4tensorcalls(Symbol("@tensor"), ex)
  end
  return esc(:(@tensor $ex))
  # TODO: activate manual allocator
  # return esc(:(@mtensor allocator = TensorOperations.ManualAllocator() $ex))
end

macro mtensoropt(args::Vararg{Expr})
  if save_tensorcalls()
    print_tensor4tensorcalls(Symbol("@tensoropt"), args...)
  end
  return esc(:(@tensoropt $(args...)))
  # TODO: activate manual allocator
  # return esc(:(@mtensor allocator = TensorOperations.ManualAllocator() $ex))
end

"""
    replace_ref_begin_end!(ex::Expr)

  Replace `begin` and `end` in reference expression `ex` with `firstindex` and `lastindex`.

  This is needed for macros that generate code with `begin` and `end` in references,
  because those macros are expanded before the actual indices are known.
"""
function replace_ref_begin_end!(ex::Expr)
  Meta.isexpr(ex, :ref) || return ex
  arr = ex.args[1]
  for (dim, arg) in enumerate(@view ex.args[2:end])
    ex.args[dim + 1] = _replace_begin_end(arg, arr, dim)
  end
  return ex
end

function _replace_begin_end(arg, arr, dim)
  if arg === :begin
    return :(firstindex($arr, $dim))
  elseif arg === :end
    return :(lastindex($arr, $dim))
  elseif arg isa Expr
    return Expr(arg.head, (_replace_begin_end(a, arr, dim) for a in arg.args)...)
  else
    return arg
  end
end

"""
    @mview(ex)

  StridedView based version of `@view`.
"""
macro mview(ex)
  # NOTE it's largely based on the @view macro from Base.
  Meta.isexpr(ex, :ref) || throw(ArgumentError(
      "Invalid use of @mview macro: argument must be a reference expression A[...]."))
  ex = replace_ref_begin_end!(ex)
  # NOTE We embed `view` as a function object itself directly into the AST.
  #      By doing this, we prevent the creation of function definitions like
  #      `view(A, idx) = xxx` in cases such as `@view(A[idx]) = xxx.`
  if Meta.isexpr(ex, :ref)
      ex = Expr(:call, mview, ex.args...)
  elseif Meta.isexpr(ex, :let) && (arg2 = ex.args[2]; Meta.isexpr(arg2, :ref))
      # ex replaced by let ...; foo[...]; end
      ex.args[2] = Expr(:call, mview, arg2.args...)
  else
      error("invalid expression")
  end
  return esc(ex)
end

"""
    mview(arr, args...)

  `StridedView` based version of `view`.

  The data array is enforced to be a vector, such that the view is always a `StridedView{..., Vector{...},...}`.
"""
function mview(arr, args...)
  return sview(reshape(view(vec(arr),:), size(arr)), args...)
end

"""
    mview(arr::StridedView, args...)

  StridedView based version of `view`, for `StridedView` input.

  Simply calls `StridedViews.sview`.
"""
function mview(arr::StridedView, args...)
  return sview(arr, args...)
end

# -----------------------------------------------------------------------------------------
# Buffer allocator for TensorOperations
# -----------------------------------------------------------------------------------------

"""
    TensorOperations.tensoralloc(::Type{A}, structure, ::Val{istemp}, buf::AbstractBuffer)

Allocate tensor temporaries from a `Buffer` instead of using Julia's default allocator.
Only temporary tensors (`istemp=true`) with non-zero dimensions are allocated from the buffer;
output tensors that escape the `@tensor` block use default allocation.
"""
function TensorOperations.tensoralloc(::Type{A}, structure, ::Val{istemp},
                                      buf::Buffers.AbstractBuffer) where {A<:AbstractArray,istemp}
  if istemp && length(structure) > 0
    return alloc!(buf, structure...)
  else
    return TensorOperations.tensoralloc(A, structure, Val(istemp))
  end
end

"""
    TensorOperations.tensorfree!(C, ::AbstractBuffer)

No-op for buffer allocator. Temporary memory is reclaimed by offset restore
in `blas_contract!` and `@buftensor`.
"""
TensorOperations.tensorfree!(C, ::Buffers.AbstractBuffer) = nothing

"""
    TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, buf::AbstractBuffer)

Override `blas_contract!` for buffer allocator. Saves and restores the buffer offset
around the generic implementation to reclaim per-contraction internal temporaries
(permutation buffers).
"""
function TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β,
                                         backend, buf::Buffers.AbstractBuffer)
  saved_offset = buf.offset[]
  try
    C = Base.@invoke TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β,
                                                      backend, buf::Any)
  finally
    buf.offset[] = saved_offset
  end
  return C
end

# -----------------------------------------------------------------------------------------
# Pointer-range aliasing check for StridedViews from shared buffer memory
# -----------------------------------------------------------------------------------------

"""
    _memory_span(sv::StridedView{T})

Compute the byte range `(lo, hi)` of memory accessed by a `StridedView`,
accounting for non-unit strides.
"""
function _memory_span(sv::StridedView{T}) where T
  lo = UInt(pointer(sv))
  hi = lo
  for d in 1:ndims(sv)
    n = size(sv, d)
    if n > 1
      hi += (n - 1) * UInt(abs(strides(sv)[d]) * sizeof(T))
    end
  end
  return (lo, hi + UInt(sizeof(T)))
end

"""
    _buf_mightalias(A::StridedView, B::StridedView)

Check if two `StridedView`s might alias, using pointer-range overlap instead of
`Base.mightalias`. This avoids false positives when both views reference
non-overlapping regions of the same buffer.
"""
function _buf_mightalias(A::StridedView, B::StridedView)
  A.parent === B.parent || return false
  loA, hiA = _memory_span(A)
  loB, hiB = _memory_span(B)
  return loA < hiB && loB < hiA
end

"""
    TensorOperations.stridedtensorcontract!(C, A, pA, B, pB, pAB, α, β,
                                             backend::StridedBLAS, allocator::AbstractBuffer)

Override for buffer allocator: uses pointer-range overlap check instead of
`Base.mightalias` to avoid false aliasing when arrays share a buffer parent.
"""
function TensorOperations.stridedtensorcontract!(
    C::StridedView,
    A::StridedView, pA::TensorOperations.Index2Tuple,
    B::StridedView, pB::TensorOperations.Index2Tuple,
    pAB::TensorOperations.Index2Tuple,
    α::Number, β::Number,
    backend::TensorOperations.StridedBLAS, allocator::Buffers.AbstractBuffer)
  TensorOperations.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
  TensorOperations.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)
  (_buf_mightalias(C, A) || _buf_mightalias(C, B)) &&
    throw(ArgumentError("output tensor must not be aliased with input tensor"))
  TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
  return C
end

# -----------------------------------------------------------------------------------------
# @buftensor macro
# -----------------------------------------------------------------------------------------

"""
    @buftensor buf tensor_expr

Use a `Buffer` from `Buffers.jl` to handle allocation of temporary tensors
in tensor operations. The buffer offset is saved before the expression and
restored afterwards, reclaiming all temporary allocations.

This is analogous to `@butensor` from TensorOperations (which uses Bumper.jl),
but uses ElemCo's `Buffers` package instead.

The buffer can be pre-allocated to avoid runtime growth:
```julia
buf = Buffer(nvir^3 * 4)  # pre-allocate for estimated temp usage
@buftensor buf begin
  C[a,b,c] = A[a,d] * B[d,b,c]
  C[a,b,c] += D[a,d] * E[d,b,c]
end
```

If the buffer argument is omitted, the variable `buf4tensor` is used by default:
```julia
buf4tensor = Buffer(nvir^3 * 4)
@buftensor C[a,b,c] = A[a,d] * B[d,b,c]
```

Additional `@tensor` keyword arguments can be passed:
```julia
@buftensor buf opt=true C[a,b,c] := A[a,d] * B[d,e] * E[e,b,c]
```
"""
macro buftensor(args...)
  length(args) >= 1 || throw(ArgumentError(
    "@buftensor requires at least a tensor expression"))

  if _is_tensor_expr(args[1])
    buf_expr = :buf4tensor
    tensor_args = args
  else
    length(args) >= 2 || throw(ArgumentError(
      "@buftensor with explicit buffer requires a tensor expression"))
    buf_expr = args[1]
    tensor_args = args[2:end]
  end

  buf_sym = gensym("buf")
  off_sym = gensym("offset")
  res_sym = gensym("result")

  newex = quote
    $buf_sym = $buf_expr
    $off_sym = $buf_sym.offset[]
    $res_sym = $(Expr(:macrocall, GlobalRef(TensorOperations, Symbol("@tensor")),
                      __source__, :(allocator = $buf_sym), tensor_args...))
    $buf_sym.offset[] = $off_sym
    $res_sym
  end
  return esc(Base.remove_linenums!(newex))
end

"""
    _is_tensor_expr(ex)

Check if an expression looks like a tensor expression (indexed assignment or begin block)
rather than a buffer variable/expression.
"""
_is_tensor_expr(::Any) = false
function _is_tensor_expr(ex::Expr)
  ex.head === :block && return true
  ex.head in (:(=), :(+=), :(-=)) || return false
  return ex.args[1] isa Expr && ex.args[1].head === :ref
end

end #module