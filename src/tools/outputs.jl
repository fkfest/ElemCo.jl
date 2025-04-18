"""
    Outputs

Module for output functions.

The main purpose of this module is to hide the output functions from the JET.jl analyser.
"""
module Outputs
using Printf
export output_time, output_memory, flush_output
export output_iteration
export output_E_var, output_E_method, output_norms

"""
    output_time(Δtime, info::AbstractString)

  Output time with message `info`.
"""
function output_time(Δtime, info::AbstractString)
  @printf "Time for %s:\t %8.2f \n" info Δtime/10^9
  flush(stdout)
end

"""
    output_memory(Δmemory, info::AbstractString)

  Output memory usage (`Δmemory` in bytes) with message `info`.
"""
function output_memory(Δmemory, info::AbstractString)
  letters = " KMGTP"
  x = Δmemory
  i = 1
  y = x/1024
  while y > 1 && i < 6
    x = y
    y = x/1024
    i += 1
  end
  if i > 1
    x = round(x, digits=2)
  end
  println("Memory usage $info: $x$(letters[i])B")
  flush(stdout)
end

"""
    flush_output()

  Flush the output buffer.
"""
function flush_output()
  flush(stdout)
end

"""
    output_iteration(it, var, Δt, floats...)

  Output iteration number `it`, variance `var`, time step `Δt`, and additional floats.
"""
function output_iteration(it, var, Δt, floats...)
  @printf "%3i " it
  for f in floats
    @printf "%12.8f " f
  end
  @printf "%10.2e %8.2f \n" var Δt/10^9
  flush(stdout)
end

"""
    output_E_var(En, var, Δt)

  Output energy `En`, variance `var`, and time step `Δt`.
"""
function output_E_var(En, var, Δt)
  @printf "%12.8f %10.2e %8.2f \n" En var Δt/10^9
  flush(stdout)
end

"""
    output_E_method(En, method::AbstractString, info::AbstractString="")

  Output energy `En` with method `method` and additional info `info`.
"""
function output_E_method(En, method::AbstractString, info::AbstractString="")
  @printf "%s %s \t%16.12f \n" method info En
  flush(stdout)
end

"""
    output_norms(norms::Pair{String,Float64}...)

  Output squared norms. 

  The norms are pairs of strings and floats.
"""
function output_norms(norms::Pair{String,Float64}...)
  for norm in norms
    @printf "|%s|²: %12.8f   " norm.first norm.second
  end
  println()
  flush(stdout)
end

end #module