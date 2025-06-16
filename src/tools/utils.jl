""" various utilities """
module Utils
using MKL
using XML
using Printf
using ..ElemCo.AbstractEC
using ..ElemCo.DescDict
using ..ElemCo.Outputs

export NOTHING1idx, NOTHING2idx, NOTHING3idx, NOTHING4idx, NOTHING5idx, NOTHING6idx
export warn
export mainname, print_time, print_memory, free_memory
export draw_line, draw_wiggly_line, print_info, draw_endline, kwarg_provided_in_macro
export subspace_in_space, argmaxN
export @istoplevel
export substr
export amdmkl
export xpath
# from DescDict
export ODDict, getdescription, setdescription!, descriptions
export OutDict, last_energy

"""
    mainname(file::String)

Return the main name of a file, i.e. the part before the last dot
and the extension.

Examples:
```
julia> mainname("~/test.xyz")
("test", "xyz")

julia> mainname("test")
("test", "")
```
"""
function mainname(file::String)
  ffile = basename(file)
  afile = split(ffile,'.')
  if length(afile) == 1
    return afile[1], ""
  else
    return join(afile[1:end-1], '.'), afile[end]
  end
end

""" 
    print_time(EC::AbstractECInfo, t1, info::AbstractString, verb::Int)

  Print time with message `info` if verbosity `verb` is smaller than `PrintOptions.time`.
"""
function print_time(EC::AbstractECInfo, t1, info::AbstractString, verb::Int)
  t2 = time_ns()
  if verb < EC.options.print.time
    output_time(t2 - t1, info)
  end
  return t2
end

"""
    free_memory()

  Return the amount of free memory in bytes.
"""
free_memory() = Sys.free_memory()

""" 
    print_memory(EC::AbstractECInfo, mem1, info::AbstractString, verb::Int)

  Print memory usage with message `info` if verbosity `verb` is smaller than `PrintOptions.memory`.

  Note that memory is also used by other processes and the operating system, so the memory usage
  reported here is merely an estimate. 
"""
function print_memory(EC::AbstractECInfo, mem1, info::AbstractString, verb::Int)
  mem2 = free_memory()
  if verb < EC.options.print.memory
    output_memory(mem2 - mem1, info)
  end
  return mem2
end

"""
    warn(msg::AbstractString, err=false)

  Print a warning message. If `err` is `true`, the message is printed as an error message.

  The message is printed with a scull emoji.
  # Example
```julia
julia> warn("This is a warning message.")
```
"""
function warn(msg::AbstractString, err=false)
  if err
    error(msg)
  end
  println("☠️ Warning: ", msg)
end


"""
    OutDict

  An ordered descriptive dictionary that maps keys of type `String` to values of type `Float64`.
"""
const OutDict = ODDict{String, Float64}

for N in 1:6
  NOTHINGN = Symbol("NOTHING$(N)idx")
  @eval begin
    const $NOTHINGN = Array{Float64,$N}(undef, ntuple(i->0, Val($N)))
  end
end

"""
    last_energy(energies::OutDict)

  Return the last energy in `energies`.
"""
last_energy(energies::OutDict) = last_value(energies)

"""
    draw_line(n = 63)

  Print a thick line of `n` characters.
"""
function draw_line(n=63)
  println(repeat("━", n))
end

"""
    draw_thin_line(n = 63)

  Print a thin line of `n` characters.
"""
function draw_thin_line(n=63)
  println(repeat("─", n))
end

"""
    print_info(info::AbstractString, additional_info::AbstractString="")

  Print `info` between two lines.

  If `additional` not empty: additional info after main.
"""
function print_info(info::AbstractString, additional_info::AbstractString="")
  println()
  draw_line()
  println(info)
  draw_line()
  if additional_info != ""
    println(additional_info)
    draw_thin_line()
  end
  flush_output()
end

"""
    draw_endline()

  Print a line of ═.
"""
function draw_endline(n=63)
  println(repeat("═", n))
  flush_output()
end

"""
    kwarg_provided_in_macro(kwargs, key::Symbol)

  Check whether `key` is in `kwargs`. 

  This is used in macros to check whether a keyword argument is passed.
  The keyword argument in question `key` is passed as a symbol, e.g. `:thr`.
  `kwargs` is the keyword argument list passed to the macro.
"""
function kwarg_provided_in_macro(kwargs, key::Symbol)
  for kwarg in kwargs
    if typeof(kwarg) != Expr || kwarg.head != :(=)
      error("Not a keyword argument!")
    end
    if kwarg.args[1] == key
      return true
    end
  end
  return false
end

"""
    subspace_in_space(subspace, space)

  Return the positions of `subspace` in `space` 
  (with respect to `space`)

  `subspace` and `space` are lists of indices 
  with respect to the full space (e.g., `1:norb`).

  # Examples 
```julia
julia> get_subspace_of_space([1,3,5], [1,3,4,5])
3-element Array{Int64,1}:
  1
  2 
  4
```
"""
function subspace_in_space(subspace, space)
  idx = indexin(subspace, space)
  @assert all(!isnothing, idx) "Subspace not contained in space."
  return idx
end

"""
    subspace_in_space(subspace::UnitRange{Int}, space::UnitRange{Int})

  Return the positions of `subspace` in `space` 
  (with respect to `space`)

  `subspace` and `space` are ranges of indices 
  with respect to the full space (e.g., `1:norb`).

  # Examples 
```julia
julia> get_subspace_of_space(4:6, 2:7)
3:5
```
"""
function subspace_in_space(subspace::UnitRange{Int}, space::UnitRange{Int})
  start = subspace.start - space.start + 1
  stop = subspace.stop - space.start + 1
  @assert start > 0 && start <= stop <= length(space) "Subspace not contained in space."
  return start:stop
end


"""
    substr(string::AbstractString, start::Int, len::Int=-1)

  Return substring of `string`  starting at `start` spanning `len` characters 
  (including unicode).
  If `len` is not given, the substring spans to the end of `string`.

  Example:
```julia
julia> substr("λabδcd", 2, 3)
"abδ"
```
"""
function substr(string::AbstractString, start::Int, len::Int=-1)
  tail = length(string)-start-len+1
  if len < 0 || tail < 0
    tail = 0
  end
  return chop(string, head=start-1, tail=tail)
end

"""
    substr(string::AbstractString, range::UnitRange{Int})

  Return substring of `string` defined by `range` (including unicode).

  Example:
```julia
julia> substr("λabδcd", 2:4)
"abδ"
```
"""
function substr(string::AbstractString, range::UnitRange{Int})
  return substr(string, range.start, range.stop-range.start+1)
end

"""
    argmaxN(vals, N; by::Function=identity)

  Return the indices of the `N` largest elements in `vals`.

  The order of equal elements is preserved.
  The keyword argument `by` can be used to specify a function to compare the elements, i.e.,
  the function is applied to the elements before comparison.

  # Example
```julia
julia> argmaxN([1,2,3,4,5,6,7,8,9,10], 3)
3-element Vector{Int64}:
 10
  9
  8
julia> argmaxN([1,2,3,4,5,-6,-7,-8,-9,-10], 3; by=abs)
3-element Vector{Int64}:
 10
  9
  8
julia> argmaxN([1.0, 1.10, 1.112, -1.113, 1.09], 3; by=x->round(abs(x),digits=2))
3-element Vector{Int64}:
 3
 4
 2
```
"""
function argmaxN(vals, N; by::Function=identity)
  perm = sortperm(vals[1:N]; by, rev=true)
  smallest = by(vals[perm[N]])
  @inbounds for i in N+1:length(vals)
    el = by(vals[i])
    if smallest < el
      for j in 1:N
        if by(vals[perm[j]]) < el
          perm[j+1:end] = perm[j:end-1]
          perm[j] = i
          break
        end
      end
      smallest = by(vals[perm[N]])
    end
  end
  return perm
end

"""
    @istoplevel

  Macro to check if the current scope is the top level scope.

  (from https://discourse.julialang.org/t/is-there-a-way-to-determine-whether-code-is-toplevel)
"""
macro istoplevel()
  canary = gensym("canary")
  quote
    $(esc(canary)) = true
    Base.isdefined($__module__, $(QuoteNode(canary)))
  end
end

"""
    xspyder(xtag::AbstractString, node::Node)

Recursively search for nodes matching the xml tag `xtag` in the XML `node`.
Returns a vector of nodes that match the tag.
If `xtag` is an empty string, it returns the node itself.
"""
function xspyder(xtag::AbstractString, node::Node)
  matches = Node[]
  if xtag == ""
    push!(matches, node)
  else
    for child in children(node)
      if xtag == "*" || tag(child) == xtag
        push!(matches, child)
      end
      append!(matches, xspyder(xtag, child))
    end
  end
  return matches
end


"""
    xpath(xpath::AbstractString, node::Node)

Search for nodes matching the XPath `xpath` in the XML `node`.
Returns a vector of nodes that match the XPath expression.
If `xpath` is an empty string, it returns the node itself.

Supports basic XPath 1.0 syntax:
- `/tag1/tag2` - absolute path from root (here: root == current node)
- `tag1/tag2` - relative path from current node  
- `//tag` - descendant-or-self axis (recursive search)
- `/` - root node (here: root == current node)
- `*` - any element node
- `tag[@attr=value]` - elements with specific attribute values

Examples:
- `/node/child` - child elements named 'child' under the root 'node'
- `//item` - all 'item' elements anywhere in the tree
- `parent/child` - 'child' elements that are direct children of 'parent'
- `//variable[@name="x"]` - variable elements with name attribute equal to "x"
"""
function xpath(xpath::AbstractString, node::Node)
  if xpath == ""
    return [node]
  end
  
  # Handle root node selection
  if xpath == "/"
    # Find the root node by traversing up
    root = node
    # while XML.depth(root) !== 0
    #   root = XML.parent(root)
    # end
    return [root]
  end
  
  # Determine if this is an absolute path (starts with /)
  is_absolute = startswith(xpath, "/")
  if is_absolute
    xpath = xpath[2:end]  # Remove leading slash
    if startswith(xpath, "/")
      is_absolute = false  # If it starts with another slash, it's not absolute
      if startswith(xpath, "//")
        is_absolute = true  # If it starts with ///, it's absolute
        xpath = xpath[2:end]  # Remove double slash for descendant-or-self axis
      end
    end
  end
  
  # Start from root if absolute path, otherwise from current node
  if is_absolute
    root = node
    # while XML.depth(root) !== 0
    #   root = XML.parent(root)
    # end
    current_nodes = [root]
  else
    current_nodes = [node]
  end
  
  # Handle empty path after removing leading slash
  if xpath == ""
    return current_nodes
  end
  
  # Split path into steps, handling descendant-or-self axis (//)
  steps = String[]
  parts = split(xpath, "/")
  
  i = 1
  while i <= length(parts)
    if parts[i] == ""
      # Double slash (//) - descendant-or-self axis
      if i < length(parts)
        push!(steps, "//" * parts[i+1])
        i += 1
      end
    else
      push!(steps, parts[i])
    end
    i += 1
  end
  # Process each step
  for step in steps
    next_nodes = Node[]
    if startswith(step, "//")
      # Descendant-or-self axis - recursive search
      descendant_step = step[3:end]
      (target_tag, predicates) = parse_step_with_predicate(descendant_step)
      
      for current_node in current_nodes
        # Check if current node matches
        if (target_tag == "*" || target_tag == tag(current_node)) && 
           matches_predicates(current_node, predicates)
          push!(next_nodes, current_node)
        end
        
        # Recursively search descendants
        descendants = xspyder(target_tag, current_node)
        for desc in descendants
          if matches_predicates(desc, predicates)
            push!(next_nodes, desc)
          end
        end
      end
    else
      # Child axis - direct children only
      (target_tag, predicates) = parse_step_with_predicate(step)
      
      for current_node in current_nodes
        for child in children(current_node)
          if (target_tag == "*" || target_tag == tag(child)) && 
             matches_predicates(child, predicates)
            push!(next_nodes, child)
          end
        end
      end
    end
    current_nodes = next_nodes
  end
  return current_nodes
end

"""
    parse_step_with_predicate(step::AbstractString)

Parse a step that may contain predicates like "tag[@attr=value]".
Returns (tag_name, predicates) where predicates is a vector of predicate strings.
"""
function parse_step_with_predicate(step::AbstractString)
  # Check if step contains predicates
  if !occursin('[', step)
    return (step, String[])
  end
  
  # Find the tag name (before the first '[')
  bracket_pos = findfirst('[', step)
  tag_name = step[1:bracket_pos-1]
  
  # Extract predicates between brackets
  predicates = String[]
  remaining = step[bracket_pos:end]
  
  while occursin('[', remaining) && occursin(']', remaining)
    start_bracket = findfirst('[', remaining)
    end_bracket = findfirst(']', remaining)
    if start_bracket !== nothing && end_bracket !== nothing && end_bracket > start_bracket
      predicate = remaining[start_bracket+1:end_bracket-1]
      push!(predicates, predicate)
      remaining = remaining[end_bracket+1:end]
    else
      break
    end
  end
  
  return (tag_name, predicates)
end

"""
    evaluate_predicate(predicate::AbstractString, node::Node)

Evaluate a predicate against a node. Currently supports:
- @attr=value
"""
function evaluate_predicate(predicate::AbstractString, node::Node)
  # Handle attribute predicates like @name=value or @name=$var
  if occursin('=', predicate)
    parts = split(predicate, '=', limit=2)
    if length(parts) == 2
      attr_part = strip(parts[1])
      value_part = strip(parts[2])
      
      # Remove quotes from value if present
      if (startswith(value_part, '"') && endswith(value_part, '"')) || 
         (startswith(value_part, '\'') && endswith(value_part, '\''))
        value_part = value_part[2:end-1]
      end
      
      # Handle attribute access
      if startswith(attr_part, '@')
        attr_name = attr_part[2:end]
        # Get attribute value from the node
        attrs = attributes(node)
        attr = get(attrs, attr_name, nothing)
        if !isnothing(attr)
          # Check if the attribute value matches the expected value
          return attr == value_part
        else
          return false  # Attribute not found
        end
      end
    end
  end
  # Default: predicate not supported or doesn't match
  return false
end

"""
    matches_predicates(node::Node, predicates::Vector{String})

Check if a node matches all given predicates.
"""
function matches_predicates(node::Node, predicates::Vector{String})
  for predicate in predicates
    if !evaluate_predicate(predicate, node)
      return false
    end
  end
  return true
end

"""
    amdmkl(reset::Bool=false)

  Create a modified `libmkl_rt.so` and `libmkl_core.so` to make MKL work
  fast on "Zen" AMD machines (e.g., Ryzen series). Solution is based on
  [this forum post](https://discourse.julialang.org/t/how-to-circumvent-intels-amd-discrimination-in-mkl-from-v1-7-onwards).

  This function is only needed on AMD machines. In order to execute it,
  call `amdmkl()` in a separate Julia session (not in the same session
  where you want to run calculations).
  For example, your workflow could look like this:

```bash
> julia -e 'using ElemCo; ElemCo.amdmkl()'
> julia input.jl
```

  where `input.jl` is your script that uses `ElemCo.jl`.
  The changes can be reverted by calling `amdmkl(true)`.
"""
function amdmkl(reset::Bool=false)
  mklpath = dirname(MKL.MKL_jll.libmkl_rt_path)

  cd(mklpath)

  # check if a different process is modifying the files right now (e.g., another call to amdmkl)
  # and wait until the other process is done (max 5 minutes)
  while isfile("libamdmkl.c") && 0 < time() - mtime("libamdmkl.c") < 300
    sleep(5)
  end

  original = islink("libmkl_core.so") && islink("libmkl_rt.so")
  
  if reset
    if !original || isfile("libamdmkl.c")
      rm("libmkl_rt.so", force=true)
      rm("libmkl_core.so", force=true)
      rm("libamdmkl.c", force=true)
      symlink("libmkl_core.so.2","libmkl_core.so")
      symlink("libmkl_rt.so.2","libmkl_rt.so")
    end
  else
    if original || isfile("libamdmkl.c")
      try
        write("libamdmkl.c","int mkl_serv_intel_cpu_true() {return 1;}")
        rm("libmkl_core.so", force=true)
        run(`gcc -shared -o libmkl_core.so -Wl,-rpath=''\$ORIGIN'' libamdmkl.c libmkl_core.so.2`)
        rm("libmkl_rt.so", force=true)
        run(`gcc -shared -o libmkl_rt.so -Wl,-rpath=''\$ORIGIN'' libamdmkl.c libmkl_rt.so.2`)
        rm("libamdmkl.c")
      catch
        # if something goes wrong, revert to original
        println("Error: Reverting to original MKL libraries.")
        rm("libmkl_rt.so", force=true)
        rm("libmkl_core.so", force=true)
        rm("libamdmkl.c", force=true)
        symlink("libmkl_core.so.2","libmkl_core.so")
        symlink("libmkl_rt.so.2","libmkl_rt.so")
      end
    end
  end
end

end #module
