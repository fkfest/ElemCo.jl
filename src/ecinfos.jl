""" various global infos """
module ECInfos
using Parameters
using ..ElemCo.FciDump

export ECInfo, parse_orbstring, get_occvirt

@with_kw mutable struct ECInfo
  # path to scratch directory
  scr::String = joinpath(tempdir(),"elemcojlscr")
  thr::Float64 = 1.e-10
  maxit::Int = 50
  shifts::Float64 = 0.15
  shiftp::Float64 = 0.2
  shiftt::Float64 = 0.2
  verbosity::Int = 2
  # cholesky threshold
  choltol::Float64 = 1.e-6
  # amplitude decomposition threshold
  ampsvdtol::Float64 = 1.e-3
  fd::FDump = FDump()
  ignore_error::Bool = false
  # subspaces: 'o'ccupied, 'v'irtual, 'O'ccupied-β, 'V'irtual-β, ':' general
  space::Dict{Char,Any} = Dict{Char,Any}()
  fock::Array{Float64} = Float64[]
  fockb::Array{Float64} = Float64[]
  ϵo::Array{Float64} = Float64[]
  ϵv::Array{Float64} = Float64[]
  ϵob::Array{Float64} = Float64[]
  ϵvb::Array{Float64} = Float64[]
  use_kext::Bool = true
  calc_d_vvvv::Bool = false
  calc_d_vvvo::Bool = false
  calc_d_vovv::Bool = false
  calc_d_vvoo::Bool = false
  triangular_kext = true
  calc_t3_for_decomposition::Bool = false
end

"""
parse a string specifying some list of orbitals, e.g., 
`-3+5-8+10-12` → `[1 2 3 5 6 7 8 10 11 12]`
or use ':' and ';' instead of '-' and '+', respectively
"""
function parse_orbstring(orbs::String)
  # make it in julia syntax
  orbs1 = replace(orbs,"-"=>":")
  orbs1 = replace(orbs1,"+"=>";")
  orbs1 = replace(orbs1," "=>"")
  # println(orbs1)
  occursin(r"^[0-9:;]+$",orbs1) || error("Use only `0123456789:;+-` characters in the orbstring: $orbs")
  if first(orbs1) == ':'
    orbs1 = "1"*orbs1
  end
  orblist=Vector{Int}()
  for range in filter(!isempty,split(orbs1,';'))
    firstlast = filter(!isempty,split(range,':'))
    if length(firstlast) == 1
      # add the orbital
      orblist=push!(orblist,parse(Int,firstlast[1]))
    else
      length(firstlast) == 2 || error("Someting wrong in range $range in orbstring $orbs")
      firstorb = parse(Int,firstlast[1])
      lastorb = parse(Int,firstlast[2])
      # add the range
      orblist=vcat(orblist,[firstorb:lastorb]...)
    end
  end
  allunique(orblist) || error("Repeated orbitals found in orbstring $orbs")
  return orblist
end

"""
use a +/- string to specify the occupation. If occbs=="-", the occupation from occas is used (closed-shell).
if both are "-", the occupation is deduced from nelec.
"""
function get_occvirt(EC::ECInfo, occas::String, occbs::String, norb, nelec, ms2=0)
  if occas != "-"
    occa = parse_orbstring(occas)
    if occbs == "-"
      # copy occa to occb
      occb = deepcopy(occa)
    else
      occb = parse_orbstring(occbs)
    end
    if length(occa)+length(occb) != nelec && !EC.ignore_error
      error("Inconsistency in OCCA ($occas) and OCCB ($occbs) definitions and the number of electrons ($nelec). Use ignore_error (-f) to ignore.")
    end
  else 
    occa = [1:(nelec+ms2)÷2;]
    occb = [1:(nelec-ms2)÷2;]
  end
  virta = [ i for i in 1:norb if i ∉ occa ]
  virtb = [ i for i in 1:norb if i ∉ occb ]
  if occa == occb
    println("Occupied orbitals:", occa)
  else
    println("Occupied α orbitals:", occa)
    println("Occupied β orbitals:", occb)
  end
  return occa, virta, occb, virtb
end


end #module
