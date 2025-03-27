
# analyze @tensor calls and write all of them in a single file
# together with tensor allocations. 
# Replace all tensors with dummy tensors.
# The final form will be like A<ndimA>[...] = B<ndimB1>[...] * B<ndimB2>[...]

const TENIO = open("tensorcalls.jl","w")
const _TensorExpressions = Set{Expr}()

function write_header4tensorcalls()
  io = TENIO
  println(io, "#using TensorOperations, StridedViews")
  println(io, "svzeros(dims...) = StridedView(reshape(view(vec(zeros(dims...)),:), dims))")
  println(io)
  println(io, "function tensorcalls()")
  println(io, "  alpha = 1.0")
  println(io, "  # tensor allocations")
  for sv in [""=>"", "v!"=>"sv"]
    for ndim in 1:6
      for ten in ['A', 'B']
        println(io, "  $(sv[1])$ten$ndim = $(sv[2])zeros($(join(fill(1, ndim), ", ")))")
      end
    end
  end
  println(io, "  # please add end at the end of this function")
  println(io, "  # tensor calls")
end

function print_tensor4tensorcalls(tencall, args::Vararg{Expr})
  io = TENIO
  ex = tensoranalyzer(args[end], tencall, args[1:(end - 1)])
  if ex in _TensorExpressions
    return
  end
  push!(_TensorExpressions, ex)
  println(io, ex)
end

function tensoranalyzer(ex, tencall, opts=[])
  _ex = ex
  Base.remove_linenums!(_ex)
  indices = Dict()
  _ex, indices = _analyze_tensorcontraction(_ex, indices, length(opts) > 0)
  if length(opts) > 0
    return Expr(:macrocall, tencall, :(), opts..., _ex)
  else
    return Expr(:macrocall, tencall, :(), _ex)
  end
end

const _INDICES4ANALYZER = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
is_expr(ex, head::Symbol) = Meta.isexpr(ex, head)

"""
    _analyze_tensorcontraction(ex::Expr, indices::Dict, keep_indices=false)

  Analyze expression `ex` from `@mtensor(ex)`, e.g., `@mtensor A[p,q,L] = B[p,r,L] * C[r,q]`.

  All tensor will be replaced with dummy tensors 
  (the lhs tensor is always A<ndim> or v!A<ndim>, and tensors on the right are always B<ndim> or v!B<ndim>) 
  and the indices replaced with "a,b,c,...", unless keep_indices is true.
  Returns a new expression and a dictionary of indices.
"""
function _analyze_tensorcontraction(ex, indices, keep_indices=false)
  if !(ex isa Expr)
    return nothing, indices
  end
  if is_expr(ex, :block)
    ex_args = []
    for exa in ex.args
      arg, indices = _analyze_tensorcontraction(exa, indices, keep_indices)
      if !isnothing(arg)
        push!(ex_args, arg)
      end
    end
    return Expr(:block, ex_args...), indices
  elseif ex.head in [:(=), :(:=), :(+=), :(-=)]
    lhs, indices = _analyze_tensor(ex.args[1], indices, keep_indices, "A")
    rhs, indices = _analyze_rhs(ex.args[2], indices, keep_indices, "B")
    return Expr(ex.head, lhs, rhs), indices
  else
    return nothing, indices
  end
end

function _analyze_rhs(rhs, indices, keep_indices, tenname)
  if is_expr(rhs, :call)
    rhs_args = []
    for i in 2:length(rhs.args)
      arg, indices = _analyze_rhs(rhs.args[i], indices, keep_indices, tenname)
      if !isnothing(arg)
        push!(rhs_args, arg)
      end
    end
    return Expr(:call, rhs.args[1], rhs_args...), indices
  else
    return _analyze_tensor(rhs, indices, keep_indices, tenname)
  end
end

"""
    _analyze_tensor(ten::Expr, indices::Dict, keep_indices, basetenname)

  Analyze tensor expression `ten`, e.g., `A[p,q,L]`.
"""
function _analyze_tensor(ten, indices, keep_indices, basetenname)
  if ten isa Number
    return ten, indices
  end
  if !(ten isa Expr)
    return :alpha, indices
  end
  if is_expr(ten, :ref)
    ndim = length(ten.args) - 1
    tn = "$(ten.args[1])"
    if length(tn) > 2 && chop(tn; tail=length(tn)-2) == "v!"
      # a StridedView
      tenname = "v!$(basetenname)$ndim"
    else
      tenname = "$(basetenname)$ndim"
    end
    inds = ten.args[2:end]
    if !keep_indices
      for (i, ind) in enumerate(inds)
        if !haskey(indices, ind)
          indices[ind] = Symbol(_INDICES4ANALYZER[length(indices) + 1])
        end
        inds[i] = indices[ind]
      end
    end
    return Expr(:ref, Symbol(tenname), inds...), indices
  else
    error("Unknown tensor expression: $ten")
  end
end