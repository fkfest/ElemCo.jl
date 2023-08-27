""" various utilities """
module Utils
using Printf
using ..ElemCo.AbstractEC

export print_time, draw_line, print_info

""" 
    print_time(EC::AbstractECInfo, t1, info::AbstractString, verb::Int)

  Print time with message `info` if verbosity `verb` is smaller than EC.verbosity.
"""
function print_time(EC::AbstractECInfo, t1, info::AbstractString, verb::Int)
  t2 = time_ns()
  if verb < EC.verbosity
    @printf "Time for %s:\t %8.2f \n" info (t2-t1)/10^9
  end
  return t2
end

"""
    draw_line(n = 60)

  Print a line of `n` dashes.
"""
function draw_line(n=60)
  println(repeat("─", n))
end

"""
    print_info(info::AbstractString)

  Print `info` between two lines.
"""
function print_info(info::AbstractString)
  println()
  draw_line()
  println(info)
  draw_line()
end
end #module