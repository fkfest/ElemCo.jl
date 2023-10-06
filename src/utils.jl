""" various utilities """
module Utils
using Printf
using ..ElemCo.AbstractEC

export mainname, print_time, draw_line, draw_wiggly_line, print_info, draw_endline, kwarg_provided_in_macro

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
    @printf "Time for %s:\t %8.2f \n" info (t2-t1)/10^9
    flush(stdout)
  end
  return t2
end

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
  flush(stdout)
end

"""
    draw_endline()

  Print a line of ═.
"""
function draw_endline(n=63)
  println(repeat("═", n))
  flush(stdout)
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

end #module