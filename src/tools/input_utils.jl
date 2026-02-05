
"""
    clean_exprstring(expr)

  Return a clean string from an expression, i.e., without empty spaces and extra parentheses.

  # Examples
```julia
julia> clean_exprstring(:(SVD-CCSD))
"SVD-CCSD"
julia> clean_exprstring(:(eom-svd-df-ccsd(t)))
"eom-svd-df-ccsd(t)"
```
"""
function clean_exprstring(expr)
  if !(expr isa Expr) || expr.head != :call || expr.args[1] ∉ [:-, :+, :*, :/]
    return string(expr)
  end
  return join([clean_exprstring(a) for a in expr.args[2:end]], string(expr.args[1]))
end

"""
    separate_kwargs(args)

  Separate keyword arguments from positional arguments in macro input.

  Returns `(positional_args, kwargs)` where:
  - `positional_args`: Vector of non-keyword arguments
  - `kwargs`: Vector of `Expr(:kw, key, esc(value))` for keyword arguments

  Keyword arguments are expressions with head `:(=)`.

# Example
```julia
# In a macro definition:
macro mymacro(args...)
  positional, kwargs = separate_kwargs(args)
  # positional contains non-keyword args
  # kwargs can be splatted: some_function(x; \$(kwargs...))
end
```
"""
function separate_kwargs(args)
  kwargs = Expr[]
  positional = []
  for arg in args
    if arg isa Expr && arg.head == :(=)
      push!(kwargs, Expr(:kw, arg.args[1], esc(arg.args[2])))
    else
      push!(positional, arg)
    end
  end
  return positional, kwargs
end

"""
    is_options_block(arg)

Check if `arg` is a `begin...end` block for local options.
"""
function is_options_block(arg)
  return arg isa Expr && arg.head == :block
end

"""
    add_kwarg!(opts_dict, opt_name, item)

Add a key=value pair `item` to `opts_dict` under category `opt_name`.

The `item` must be an expression with head `:(=)` or `:kw`.
The value is escaped so variables are evaluated in the caller's scope.
"""
function add_kwarg!(opts_dict::Dict{Symbol, Vector{Expr}}, opt_name::Symbol, item::Expr)
  if item.head == :(=) || item.head == :kw
    if !haskey(opts_dict, opt_name)
      opts_dict[opt_name] = Expr[]
    end
    # Escape the value so variables are evaluated in caller's scope
    push!(opts_dict[opt_name], Expr(:(=), item.args[1], esc(item.args[2])))
  else
    error("Expected key=value pair in options block, got: $item")
  end
end

"""
    parse_macro_options!(opts_dict, arg)

Parse a macro-style option expression (`@set category key=val ...`) and add entries to `opts_dict`.
"""
function parse_macro_options!(opts_dict::Dict{Symbol, Vector{Expr}}, arg::Expr)
  macro_sym = arg.args[1]
  if macro_sym != Symbol("@set") && macro_sym != Symbol("@opt")
    error("Unknown macro '$macro_sym' in options block. Use: @set category key=value")
  end
  # Parse @set style: @set wf charge=-1 ms2=1
  # args[1] = @set, args[2] = LineNumberNode, args[3] = option_name, args[4+] = key=value pairs
  opt_name = nothing
  for i in 2:length(arg.args)
    item = arg.args[i]
    item isa LineNumberNode && continue
    if isnothing(opt_name)
      # First non-LineNumberNode argument is the option category
      if item isa Symbol
        opt_name = item
      else
        error("Expected option category name after @set, got: $item")
      end
    else
      add_kwarg!(opts_dict, opt_name, item)
    end
  end
end

"""
    parse_call_options!(opts_dict, arg)

Parse a function-call style option expression (`category(key=val, ...)`) and add entries to `opts_dict`.
"""
function parse_call_options!(opts_dict::Dict{Symbol, Vector{Expr}}, arg::Expr)
  opt_name = arg.args[1]
  for i in 2:length(arg.args)
    add_kwarg!(opts_dict, opt_name, arg.args[i])
  end
end

"""
    parse_options_block(block::Expr)

Parse a `begin...end` block with local options and return code that constructs 
a NamedTuple for `with_local_options`.

Each line in the block should have one of these formats:
- Set-macro style: `@set option_name key1=val1 key2=val2 ...`
- Function-call style: `option_name(key1=val1, key2=val2, ...)`

Multiple lines for the same option category are merged together.

For example:
```julia
begin
  @set wf charge=-1 ms2=1
  @set cc maxit=30
  @set cc thr=1.e-10
end
# or equivalently:
begin
  wf(charge=-1, ms2=1)
  cc(maxit=30, thr=1.e-10)
end
```
Returns an expression that creates: `(wf=(charge=-1, ms2=1), cc=(maxit=30, thr=1.e-10))`
"""
function parse_options_block(block::Expr)
  @assert block.head == :block "Expected begin...end block"
  # Use a Dict to collect options per category (to allow merging)
  opts_dict = Dict{Symbol, Vector{Expr}}()
  
  for arg in block.args
    arg isa LineNumberNode && continue
    if arg isa Symbol
      error("Option category '$arg' specified without any settings")
    elseif arg isa Expr
      if arg.head == :macrocall
        parse_macro_options!(opts_dict, arg)
      elseif arg.head == :call
        parse_call_options!(opts_dict, arg)
      elseif arg.head == :(=)
        error("Direct assignment not supported in options block. Use: @set category key=value")
      else
        error("Unexpected expression in options block: $arg")
      end
    end
  end
  
  if isempty(opts_dict)
    return :(NamedTuple())
  end
  
  # Build the final NamedTuple expression from the collected options
  opts = Expr[]
  for (opt_name, kwargs) in opts_dict
    inner_tuple = Expr(:tuple, kwargs...)
    push!(opts, Expr(:(=), opt_name, inner_tuple))
  end
  return Expr(:tuple, opts...)
end

"""
    @var2string(var, strvar="", type=AbstractString)

  Return string representation of `var`.

  If `var` is a String (or `type`) variable, return the value of the variable.
  Otherwise, return the string representation of `var` (or `strvar` if provided).

  # Examples
```julia
julia> @var2string(CCSD)
"CCSD"
julia> CCSD = "UCCSD";
julia> @var2string(CCSD)
"UCCSD"
```
"""
macro var2string(var, strvar="", type=AbstractString)
  if strvar == ""
    strvar = clean_exprstring(var)
  end
  valvar = :($(esc(var)))
  return quote
    isvar = [false]
    try @assert(typeof($(esc(var))) <: $(esc(type)))  # check if var is defined and of correct type
      isvar[1] = true
    catch
    end
    if isvar[1]
      $valvar
    else
      $(esc(strvar))
    end
  end
end
