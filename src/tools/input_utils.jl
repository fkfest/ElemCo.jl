
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
    is_options_block(arg)

Check if `arg` is a `begin...end` block for local options.
"""
function is_options_block(arg)
  return arg isa Expr && arg.head == :block
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
      # Just option name without any settings - skip or error
      error("Option category '$arg' specified without any settings")
    elseif arg isa Expr
      if arg.head == :macrocall
        macro_sym = arg.args[1]
        if macro_sym == Symbol("@set") || macro_sym == Symbol("@opt")
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
            elseif item isa Expr && item.head == :(=)
              if !haskey(opts_dict, opt_name)
                opts_dict[opt_name] = Expr[]
              end
              # Escape the value so variables are evaluated in caller's scope
              push!(opts_dict[opt_name], Expr(:(=), item.args[1], esc(item.args[2])))
            elseif item isa Expr && item.head == :kw
              if !haskey(opts_dict, opt_name)
                opts_dict[opt_name] = Expr[]
              end
              # Escape the value so variables are evaluated in caller's scope
              push!(opts_dict[opt_name], Expr(:(=), item.args[1], esc(item.args[2])))
            else
              error("Expected key=value pair in options block, got: $item")
            end
          end
        else
          error("Unknown macro '$macro_sym' in options block. Use: @set category key=value")
        end
      elseif arg.head == :call
        # Parse function-call style: wf(charge=-1, ms2=1)
        opt_name = arg.args[1]
        for i in 2:length(arg.args)
          kw = arg.args[i]
          if kw isa Expr && kw.head == :(=)
            if !haskey(opts_dict, opt_name)
              opts_dict[opt_name] = Expr[]
            end
            # Escape the value so variables are evaluated in caller's scope
            push!(opts_dict[opt_name], Expr(:(=), kw.args[1], esc(kw.args[2])))
          elseif kw isa Expr && kw.head == :kw
            if !haskey(opts_dict, opt_name)
              opts_dict[opt_name] = Expr[]
            end
            # Escape the value so variables are evaluated in caller's scope
            push!(opts_dict[opt_name], Expr(:(=), kw.args[1], esc(kw.args[2])))
          else
            error("Expected key=value pair in options block, got: $kw")
          end
        end
      elseif arg.head == :(=)
        # Single assignment like: cc = saved_opts  
        # This would be for restoring, but we don't support it in blocks
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
