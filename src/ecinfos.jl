""" Various global infos """
module ECInfos
using Parameters
using ..ElemCo.AbstractEC
using ..ElemCo.Utils
using ..ElemCo.FciDump
using ..ElemCo.MSystem

export ECInfo, setup!, set_options!, parse_orbstring, get_occvirt

include("options.jl")

"""
    ECInfo

  Global information for `ElemCo`.
"""
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

""" 
    setup!(EC::ECInfo; fcidump="", occa="-", occb="-", nelec=0, charge=0, ms2=0)

  Setup ECInfo from fcidump or molecular system.
"""
function setup!(EC::ECInfo; fcidump="", occa="-", occb="-", nelec=0, charge=0, ms2=0)
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
    orbsym = convert(Vector{Int},headvar(EC.fd, "ORBSYM"))
  elseif ms_exists(EC.ms)
    norb = guess_norb(EC.ms) 
    nelec = (nelec==0) ? guess_nelec(EC.ms) : nelec
    nelec += charge
    ms2 = (ms2==0) ? mod(nelec,2) : ms2
    orbsym = ones(Int,norb)
  else
    error("No molecular system or fcidump specified!")
  end

  SP = EC.space
  SP['o'], SP['v'], SP['O'], SP['V'] = get_occvirt(EC, occa, occb, norb, nelec; ms2, orbsym)
  SP[':'] = 1:norb
  EC.nocc = length(SP['o'])
  EC.noccb = length(SP['O'])
end

""" 
    set_options!(opt; kwargs...)

  Set options for option `opt` using keyword arguments.
"""
function set_options!(opt; kwargs...)
  for (key,value) in kwargs
    if hasproperty(opt, key)
      setproperty!(opt, key, value)
    else
      error("invalid option name: $key")
    end
  end
end

"""
    parse_orbstring(orbs::String; orbsym = Vector{Int})

  Parse a string specifying some list of orbitals, e.g., 
  `-3+5-8+10-12` → `[1 2 3 5 6 7 8 10 11 12]`
  or use ':' and ';' instead of '-' and '+', respectively.
"""
function parse_orbstring(orbs::String; orbsym = Vector{Int})
  # make it in julia syntax
  orbs1 = replace(orbs,"-"=>":")
  orbs1 = replace(orbs1,"+"=>";")
  orbs1 = replace(orbs1," "=>"")
  if prod(orbsym) > 1 && occursin(".",orbs1)
    @assert(issorted(orbsym),"Orbital symmetries are not sorted. Specify occa and occb without symmetry.")
    symoffset = zeros(Int,maximum(orbsym))
    symlist = zeros(Int,maximum(orbsym))
    for sym in eachindex(symlist)
      symlist[sym] = count(isequal(sym),orbsym)
    end
    for iter in eachindex(symoffset[2:end])
      symoffset[iter] = sum(symlist[1:iter-1])
    end
    symlist = nothing
  elseif prod(orbsym) == 1 && occursin(".",orbs1)
    error("FCIDUMP without sym but orbital occupations with sym.")
  else
    symoffset = zeros(Int,1)
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
    symorb2orb(symorb::SubString, symoffset::Vector{Int})

  Convert a symorb (like 1.3 [orb.sym]) to an orbital number.
  If no sym given, just return the orbital number converted to Int.
"""
function symorb2orb(symorb::SubString, symoffset::Vector{Int})
  if occursin(".",symorb)
    orb, sym = filter(!isempty,split(symorb,'.'))
    @assert(parse(Int,sym) <= length(symoffset),"Symmetry label $sym larger than maximum of orbsym vector.")
    orb = parse(Int,orb)
    orb += symoffset[parse(Int,sym)]
    return orb
  else
    return parse(Int,symorb)
  end
end

"""
    get_occvirt(EC::ECInfo, occas::String, occbs::String, norb, nelec; ms2=0, orbsym = Vector{Int})

  Use a +/- string to specify the occupation. If `occbs`=="-", the occupation from `occas` is used (closed-shell).
  If both are "-", the occupation is deduced from `nelec` and `ms2`.
  The optional argument `orbsym` is a vector with length norb of orbital symmetries (1 to 8) for each orbital.
"""
function get_occvirt(EC::ECInfo, occas::String, occbs::String, norb, nelec; ms2=0, orbsym = Vector{Int})
  @assert(isodd(ms2) == isodd(nelec), "Inconsistency in ms2 (2*S) and number of electrons.")
  if occas != "-"
    occa = parse_orbstring(occas; orbsym)
    if occbs == "-"
      # copy occa to occb
      occb = deepcopy(occa)
    else
      occb = parse_orbstring(occbs; orbsym)
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
