""" various global infos """
module ECInfos
using Parameters
using ..ElemCo.FciDump

export ECInfo, parse_orbstring, get_occvirt

include("options.jl")
@with_kw mutable struct ECInfo
  """ path to scratch directory """
  scr::String = joinpath(tempdir(),"elemcojlscr")
  """ output file """
  out = ""
  """ verbosity level """
  verbosity::Int = 2
  """ options """
  options::Options = Options()

  """ fcidump """
  fd::FDump = FDump()
  """ ignore various errors """
  ignore_error::Bool = false
  """ subspaces: 'o'ccupied, 'v'irtual, 'O'ccupied-β, 'V'irtual-β, ':' general """
  space::Dict{Char,Any} = Dict{Char,Any}()
  """ fock matrix (for UHF: α) """
  fock::Array{Float64} = Float64[]
  """ fock matrix (β) """
  fockb::Array{Float64} = Float64[]
  """ occupied orbital energies (for UHF: α) """
  ϵo::Array{Float64} = Float64[]
  """ virtual orbital energies (for UHF: α) """
  ϵv::Array{Float64} = Float64[]
  """ occupied orbital energies (β) """
  ϵob::Array{Float64} = Float64[]
  """ virtual orbital energies (β) """
  ϵvb::Array{Float64} = Float64[]
end

"""
parse a string specifying some list of orbitals, e.g., 
`-3+5-8+10-12` → `[1 2 3 5 6 7 8 10 11 12]`
or use ':' and ';' instead of '-' and '+', respectively
"""
function parse_orbstring(orbs::String, orbsym::Vector{Any})
  # make it in julia syntax
  orbs1 = replace(orbs,"-"=>":")
  orbs1 = replace(orbs1,"+"=>";")
  orbs1 = replace(orbs1," "=>"")
  symoffset = Dict{Int,Int}()
  if prod(orbsym) > 1 && occursin(".",orbs1)
    syms = [1,2,3,4]
    symlist = Dict{Int,Int}()
    for sym in syms
      symlist[sym] = count(isequal(sym),orbsym)
    end
    symoffset[1] = 0
    symoffset[2] = symlist[1]
    symoffset[3] = symlist[1] + symlist[2]
    symoffset[4] = symlist[1] + symlist[2] + symlist[3]
    symlist, syms = nothing, nothing
  elseif prod(orbsym) == 1 && occursin(".",orbs1)
    error("FCIDUMP without sym but orbital occupations with sym.")
  end
  # println(orbs1)
  occursin(r"^[0-9:;.]+$",orbs1) || error("Use only `0123456789:;+-.` characters in the orbstring: $orbs")
  if first(orbs1) == ':'
    orbs1 = "1"*orbs1
  end
  orblist=Vector{Int}()
  for range in filter(!isempty,split(orbs1,';'))
    firstlast = filter(!isempty,split(range,':'))
    if length(firstlast) == 1
      # add the orbital
      orblist=push!(orblist,symorb2orb(firstlast[1],symoffset))
    else
      length(firstlast) == 2 || error("Someting wrong in range $range in orbstring $orbs")
      firstorb = symorb2orb(firstlast[1],symoffset)
      lastorb = symorb2orb(firstlast[2],symoffset)
      # add the range
      orblist=vcat(orblist,[firstorb:lastorb]...)
    end
  end
  allunique(orblist) || error("Repeated orbitals found in orbstring $orbs")
  return orblist
end

"""
convert a symorb (like 1.3 [orb.sym]) to an orbital number.
If no sym given, just return the orbital number converted to Int.
"""
function symorb2orb(symorb::SubString, symoffset::Dict{Int,Int})
  if occursin(".",symorb)
    orb, sym = filter(!isempty,split(symorb,'.'))
    orb = parse(Int,orb)
    orb += symoffset[parse(Int,sym)]
    return orb
  else
    return parse(Int,symorb)
  end
end

"""
use a +/- string to specify the occupation. If occbs=="-", the occupation from occas is used (closed-shell).
if both are "-", the occupation is deduced from nelec.
"""
function get_occvirt(EC::ECInfo, occas::String, occbs::String, norb, nelec, orbsym::Vector{Any}, ms2=0)
  if occas != "-"
    occa = parse_orbstring(occas, orbsym)
    if occbs == "-"
      # copy occa to occb
      occb = deepcopy(occa)
    else
      occb = parse_orbstring(occbs, orbsym)
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
