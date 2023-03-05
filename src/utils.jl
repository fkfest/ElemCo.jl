""" various utilities """
module Utils
using Printf
using ..ECInfos

export print_time

""" print time with message `info` 
    if verbosity `verb` is smaller than EC.verbosity """
function print_time(EC::ECInfo, t1, info::AbstractString, verb::Int)
  t2 = time_ns()
  if verb < EC.verbosity
    @printf "Time for %s:\t %8.2f \n" info (t2-t1)/10^9
  end
  return t2
end

end #module