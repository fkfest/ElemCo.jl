"""
    MTensorOperations

Utilities for operations on tensors. This module provides wrappers around 
`TensorOperations.jl` and `StridedViews.jl` macros and functions.

The functions and macros are reexported by `TensorTools` module.
"""
module MTensorOperations
using TensorOperations
using StridedViews
export @mtensor, @mtensoropt
export @tensor, @tensoropt # reexport @tensor, @tensoropt
export @mview, mview

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

end #module