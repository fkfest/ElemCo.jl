""" various global infos """
module ECInfos
using Parameters
using ..ElemCo.AbstractEC
using ..ElemCo.Utils
using ..ElemCo.FciDump
using ..ElemCo.MSystem

export ECInfo, setup, parse_orbstring, get_occvirt

include("options.jl")
@with_kw mutable struct ECInfo <: AbstractECInfo
  """ path to scratch directory """
  scr::String = joinpath(tempdir(),"elemcojlscr")
  """ output file """
  out = ""
  """ verbosity level """
  verbosity::Int = 2
  """ options """
  options::Options = Options()

  """ molecular system """
  ms::MSys = MSys()
  """ fcidump """
  fd::FDump = FDump()
  """ ignore various errors """
  ignore_error::Bool = false
  """ subspaces: 'o'ccupied, 'v'irtual, 'O'ccupied-β, 'V'irtual-β, ':' general """
  space::Dict{Char,Any} = Dict{Char,Any}()
  """ number of occupied orbitals (for UHF: α) """
  nocc::Int = 0
  """ number of occupied orbitals (β) """
  noccb::Int = 0
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

""" setup ECInfo """
function setup(EC::ECInfo; fcidump="", occa="-", occb="-", nelec=0, charge=0, ms2=0)
  t1 = time_ns()
  # create scratch directory
  mkpath(EC.scr)
  EC.scr = mktempdir(EC.scr)
  if fcidump != ""
    # read fcidump intergrals
    EC.fd = read_fcidump(fcidump)
    t1 = print_time(EC,t1,"read fcidump",1)
  end
  if fd_exists(EC.fd)
    println(size(EC.fd.int2))
    norb = headvar(EC.fd, "NORB")
    nelec = (nelec==0) ? headvar(EC.fd, "NELEC") : nelec
    nelec += charge
    ms2 = (ms2==0) ? headvar(EC.fd, "MS2") : ms2
  elseif ms_exists(EC.ms)
    norb = guess_norb(EC.ms) 
    nelec = (nelec==0) ? guess_nelec(EC.ms) : nelec
    nelec += charge
    ms2 = (ms2==0) ? mod(nelec,2) : ms2
  else
    error("No molecular system or fcidump specified!")
  end

  SP = EC.space

  SP['o'], SP['v'], SP['O'], SP['V'] = get_occvirt(EC, occa, occb, norb, nelec, ms2)
  SP[':'] = 1:norb
  EC.nocc = length(SP['o'])
  EC.noccb = length(SP['O'])
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
