""" various utilities """
module Utils
using MKL
using Printf
using ..ElemCo.AbstractEC
using ..ElemCo.DescDict
using ..ElemCo.Outputs

export NOTHING1idx, NOTHING2idx, NOTHING3idx, NOTHING4idx, NOTHING5idx, NOTHING6idx
export mainname, print_time, draw_line, draw_wiggly_line, print_info, draw_endline, kwarg_provided_in_macro
export subspace_in_space, argmaxN
export substr, reshape_buf, create_buf
export amdmkl
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

  Print time with message `info` if verbosity `verb` is smaller than EC.verbosity.
"""
function print_time(EC::AbstractECInfo, t1, info::AbstractString, verb::Int)
  t2 = time_ns()
  if verb < EC.verbosity
    output_time(t2-t1, info)
  end
  return t2
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
    create_buf(len::Int, T=Float64)

  Create a buffer of length `len` of type `T`.
"""
function create_buf(len::Int, T=Float64)
  return Vector{T}(undef, len)
end

"""
    reshape_buf(buf::Vector{T}, dims...; offset=0)

  Reshape (part of) a buffer to given dimensions (without copying),
  using `offset`.

  It can be used, e.g., for itermediates in tensor contractions.

# Example
```julia
julia> buf = Vector{Float64}(undef, 100000)
julia> A = reshape_buf(buf, 10, 10, 20) # 10x10x20 tensor
julia> B = reshape_buf(buf, 10, 10, 10, offset=2000) # 10x10x10 tensor starting at 2001
julia> B .= rand(10,10,10)
julia> C = rand(10,20)
julia> @tensor A[i,j,k] = B[i,j,l] * C[l,k]
```
"""
function reshape_buf(buf::Vector{T}, dims...; offset=0) where {T}
  return reshape(view(buf, 1+offset:prod(dims)+offset), dims)
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
