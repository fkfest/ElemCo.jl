#!/usr/bin/env julia

include("myio.jl")
include("mnpy.jl")
include("dump.jl")
include("diis.jl")

include("msystem.jl")
include("integrals.jl")

"""
Electron-Correlation methods 

"""
module eCo

using LinearAlgebra
using Mmap
using TensorOperations
using Printf
using Parameters
using ArgParse
using ..MyIO
using ..FciDump
using ..DIIS

@with_kw mutable struct ECInfo
  # path to scratch directory
  scr::String = "e-cojlscr"
  thr::Float64 = 1.e-10
  maxit::Int = 50
  shifts::Float64 = 0.15
  shiftp::Float64 = 0.2
  verbosity::Int = 2
  fd::FDump = FDump()
  # subspaces: 'o'ccupied, 'v'irtual, 'O'ccupied-β, 'V'irtual-β, ':' general
  space::Dict{Char,Any} = Dict{Char,Any}()
  fock::Array{Float64} = Float64[]
  ϵo::Array{Float64} = Float64[]
  ϵv::Array{Float64} = Float64[]
  use_kext::Bool = true
  calc_d_vvvv::Bool = false
  calc_d_vvvo::Bool = false
  calc_d_vovv::Bool = false
  calc_d_vvoo::Bool = false
  triangular_kext = true
end

EC::ECInfo = ECInfo()

function ecsave(fname::String,a::Array)
  miosave(joinpath(EC.scr, fname*".bin"), a)
end

function ecload(fname::String)
  return mioload(joinpath(EC.scr, fname*".bin"))
end

function ecmmap(fname::String)
  return miommap(joinpath(EC.scr, fname*".bin"))
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
function get_occvirt(occas::String, occbs::String, norb, nelec)
  if occas != "-"
    occa = parse_orbstring(occas)
    if occbs == "-"
      # copy occa to occb
      occb = deepcopy(occa)
    else
      occb = parse_orbstring(occbs)
    end
    if length(occa)+length(occb) != nelec
      error("Inconsistency in OCCA ($occas) and OCCB ($occbs) definitions and the number of electrons ($nelec)")
    end
  else 
    occa = [1:nelec÷2;]
    occb = [1:(nelec+1)÷2;]
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

function print_time(t1, info, verb)
  t2 = time_ns()
  if verb < EC.verbosity
    @printf "Time for %s:\t %8.2f \n" info (t2-t1)/10^9
  end
  return t2
end

const ExcLevels = "SDTQP"

@enum ExcType NoExc FullExc PertExc
"""
Description of the electron-correlation method
"""
struct ECMethod
  unrestricted::Bool
  """theory level: MP, CC, DC"""
  theory::String
  """ excitation level for each class (exclevel[1] for singles etc.)"""
  exclevel::Array{ExcType}

  function ECMethod(mname::AbstractString)
    if isempty(mname)
      error("Empty method name!")
    end
    unrestricted = false
    theory = ""
    exclevel = [NoExc for i in 1:length(ExcLevels)]
    ipos = 1
    if uppercase(mname[ipos:ipos+2]) == "EOM"
      error("EOM methods not implemented!")
      ipos += 3
      if mname[ipos] == '-'
        ipos += 1
      end
    end
    if uppercase(mname[ipos]) == 'U'
      unrestricted = true
      ipos += 1
    end
    if uppercase(mname[ipos:ipos+1]) == "CC"
      theory = "CC"
      ipos += 2
    elseif uppercase(mname[ipos:ipos+1]) == "DC"
      theory = "DC"
      ipos += 2
    elseif uppercase(mname[ipos:ipos+1]) == "MP"
      theory = "MP"
      ipos += 2
    else
      error("Theory not recognized in "*mname*": "*uppercase(mname[ipos:ipos+1]))
    end
    # loop over remaining letters to get excitation levels
    # currently case-insensitive, can change later...
    for char in uppercase(mname[ipos:end])
      if char == '2'
        if exclevel[1] == NoExc
          exclevel[1] = PertExc
        end
        exclevel[2] = PertExc
      else 
        #TODO:add parenthesis etc...
        iexc = findfirst(char,ExcLevels)
        if isnothing(iexc)
          error("Excitation level not recognized")
        end
        exclevel[iexc] = FullExc
      end
    end
    new(unrestricted,theory,exclevel)
  end
end

function SP(sp::Char)
  return EC.space[sp]
end

"""guess spin of an electron: lowcase α, uppercase β, non-letters skipped.
Returns true for α spin.  Throws an error if cannot decide"""
function isalphaspin(sp1::Char,sp2::Char)
  if isletter(sp1)
    return islowercase(sp1)
  elseif isletter(sp2)
    return islowercase(sp2)
  else
    error("Cannot guess spincase for $sp1 $sp2 . Specify the spincase explicitly!")
  end
end

""" return subset of 1e⁻ integrals according to spaces. The spincase can explicitly
    been given, or will be deduced from upper/lower case of spaces specification. 
"""
function ints1(spaces::String, spincase = nothing)
  sc = spincase
  if isnothing(sc)
    if isalphaspin(spaces[1],spaces[2])
      sc = SCα
    else
      sc = SCβ
    end
  end
  return integ1(EC.fd, sc)[EC.space[spaces[1]],EC.space[spaces[2]]]
end

""" generate set of CartesianIndex for addressing the lhs and 
    a bitmask for the rhs for transforming a triangular index from ':' 
    to two original indices in spaces sp1 and sp2.
    If `reverse`: the cartesian indices are reversed 
"""
function triinds(sp1::AbstractArray{Int}, sp2::AbstractArray{Int}, reverseCartInd = false)
  norb = length(SP(':'))
  # triangular index (TODO: save in EC or FDump)
  tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
  mask = falses(norb,norb)
  mask[sp1,sp2] .= true
  trimask = falses(norb,norb)
  trimask[tripp] .= true
  ci=CartesianIndices((length(sp1),length(sp2)))
  if reverseCartInd
    return CartesianIndex.(reverse.(Tuple.(ci[trimask[sp1,sp2]]))), mask[tripp]
  else
    return ci[trimask[sp1,sp2]], mask[tripp]
  end
end

""" return subset of 2e⁻ integrals according to spaces. The spincase can explicitly
    been given, or will be deduced from upper/lower case of spaces specification.
    if the last two indices are stored as triangular and detri - make them full,
    otherwise return as a triangular cut.
"""
function ints2(spaces::String, spincase = nothing, detri = true)
  sc = spincase
  if isnothing(sc)
    second_el_alpha = isalphaspin(spaces[2],spaces[4])
    if isalphaspin(spaces[1],spaces[3])
      if second_el_alpha
        sc = SCα
      else
        sc = SCαβ
      end
    else
      !second_el_alpha || error("Use αβ integrals to get the βα block "*spaces)
      sc = SCβ
    end
  end
  allint = integ2(EC.fd, sc)
  if ndims(allint) == 4
    return allint[EC.space[spaces[1]],EC.space[spaces[2]],EC.space[spaces[3]],EC.space[spaces[4]]]
  elseif detri
    # last two indices as a triangular index, desymmetrize
    @assert ndims(allint) == 3
    out = Array{Float64}(undef,length(EC.space[spaces[1]]),length(EC.space[spaces[2]]),length(EC.space[spaces[3]]),length(EC.space[spaces[4]]))
    cio, maski = triinds(EC.space[spaces[3]],EC.space[spaces[4]])
    out[:,:,cio] = allint[EC.space[spaces[1]],EC.space[spaces[2]],maski]
    cio, maski = triinds(EC.space[spaces[4]],EC.space[spaces[3]],true)
    out[:,:,cio] = permutedims(allint[EC.space[spaces[2]],EC.space[spaces[1]],maski],(2,1,3))
    return out
  else
    cio, maski = triinds(EC.space[spaces[3]],EC.space[spaces[4]])
    return allint[EC.space[spaces[1]],EC.space[spaces[2]],maski]
  end
end

function gen_fock(spincase::SpinCase)
  # calc fock matrix 
  # display(ints2(":o:o",spincase))
  @tensoropt fock[p,q] := integ1(EC.fd,spincase)[p,q] + 2.0*ints2(":o:o",spincase)[p,i,q,i] - ints2(":oo:",spincase)[p,i,i,q]
  return fock
end

function update_singles(R1, use_shift = true)
  ΔT1 = deepcopy(R1)
  shift = use_shift ? EC.shifts : 0.0
  for I ∈ CartesianIndices(ΔT1)
    a,i = Tuple(I)
    ΔT1[I] /= -(EC.ϵv[a] - EC.ϵo[i] + shift)
  end
  return ΔT1
end

function update_doubles(R2, use_shift = true)
  ΔT2 = deepcopy(R2)
  shift = use_shift ? EC.shiftp : 0.0
  for I ∈ CartesianIndices(ΔT2)
    a,b,i,j = Tuple(I)
    ΔT2[I] /= -(EC.ϵv[a] + EC.ϵv[b] - EC.ϵo[i] - EC.ϵo[j] + shift)
  end
  return ΔT2
end

function calc_singles_energy(T1)
  @tensoropt begin
    ET1 = scalar((2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*ints2("oovv")[i,j,a,b])
    ET1 += scalar(2.0*T1[a,i] * EC.fock[SP('o'),SP('v')][i,a])
  end
  return ET1
end

function calc_doubles_energy(T2)
  @tensoropt ET2 = scalar((2.0*T2[a,b,i,j] - T2[b,a,i,j]) * ints2("oovv")[i,j,a,b])
  return ET2
end

function calc_hylleraas(T1,T2,R1,R2)
  int2 = ints2("oovv")
  @tensoropt begin
    int2[i,j,a,b] += R2[a,b,i,j]
    ET2 = scalar((2.0*T2[a,b,i,j] - T2[b,a,i,j]) * int2[i,j,a,b])
  end
  if !isnothing(T1)
    dfock = ecload("dfock")
    fov = dfock[SP('o'),SP('v')] + EC.fock[SP('o'),SP('v')] # undressed part should be with factor two
    @tensoropt ET1 = scalar((fov[i,a] + 2.0 * R1[a,i])*T1[a,i])
    # ET1 = scalar(2.0*(EC.fock[SP('o'),SP('v')][i,a] + R1[a,i])*T1[a,i])
    # ET1 += scalar((2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*int2[i,j,a,b])
    ET2 += ET1
  end
  return ET2
end

function calc_singles_norm(T1)
  @tensor NormT1 = 2.0*scalar(T1[a,i]*T1[a,i])
  return NormT1
end

function calc_doubles_norm(T2)
  @tensor NormT2 = scalar((2.0*T2[a,b,i,j] - T2[b,a,i,j])*T2[a,b,i,j])
  return NormT2
end

function calc_dressed_ints(T1)
  t1 = time_ns()
  # first make half-transformed integrals
  if EC.calc_d_vvvv
    # <a\hat c|bd>
    hd_vvvv = ints2("vvvv")
    vovv = ints2("vovv")
    @tensoropt hd_vvvv[a,c,b,d] -= vovv[a,k,b,d] * T1[c,k]
    vovv = nothing
    ecsave("hd_vvvv",hd_vvvv)
    hd_vvvv = nothing
    t1 = print_time(t1,"dress hd_vvvv",3)
  end
  # <ik|j \hat l>
  hd_oooo = ints2("oooo")
  oovo = ints2("oovo")
  @tensoropt hd_oooo[j,i,l,k] += oovo[i,j,d,l] * T1[d,k]
  oovo = nothing
  t1 = print_time(t1,"dress hd_oooo",3)
  if EC.calc_d_vvoo
    # <a\hat c|j \hat l>
    hd_vvoo = ints2("vvoo")
    voov = ints2("voov")
    vooo = ints2("vooo")
    @tensoropt begin
      vooo[a,k,j,l] += voov[a,k,j,d] * T1[d,l]
      voov = nothing
      hd_vvoo[a,c,j,l] -= vooo[a,k,j,l] * T1[c,k]
      vooo = nothing
    end
    vvov = ints2("vvov")
    @tensoropt hd_vvoo[a,c,j,l] += vvov[a,c,j,d] * T1[d,l]
    vvov = nothing
    ecsave("hd_vvoo",hd_vvoo)
    hd_vvoo = nothing
    t1 = print_time(t1,"dress hd_vvoo",3)
  end
  # <\hat a k| \hat j l)
  hd_vooo = ints2("vooo")
  vovo = ints2("vovo")
  @tensoropt begin
    hd_vooo[a,k,j,l] -= hd_oooo[k,i,l,j] * T1[a,i]
    hd_vooo[a,k,j,l] += vovo[a,k,b,l] * T1[b,j]
  end
  t1 = print_time(t1,"dress hd_vooo",3)
  # some of the fully dressing moved here...
  # <ki\hat|dj>
  d_oovo = ints2("oovo")
  oovv = ints2("oovv")
  @tensoropt d_oovo[k,i,d,j] += oovv[k,i,d,b] * T1[b,j]
  ecsave("d_oovo",d_oovo)
  t1 = print_time(t1,"dress d_oovo",3)
  # <ak\hat|jd>
  d_voov = ints2("voov")
  vovv = ints2("vovv")
  @tensoropt begin
    d_voov[a,k,j,d] -= d_oovo[k,i,d,j] * T1[a,i]
    d_voov[a,k,j,d] += vovv[a,k,b,d] * T1[b,j]
  end
  ecsave("d_voov",d_voov)
  t1 = print_time(t1,"dress d_voov",3)
  # finish half-dressing
  # <ak|b \hat l>
  hd_vovo = ints2("vovo")
  @tensoropt hd_vovo[a,k,b,l] += vovv[a,k,b,d] * T1[d,l]
  vovv = nothing
  t1 = print_time(t1,"dress hd_vovo",3)
  if EC.calc_d_vvvo
    # <a\hat c|b \hat l>
    hd_vvvo = ints2("vvvo")
    vvvv = ints2("vvvv")
    @tensoropt begin
      hd_vvvo[a,c,b,l] -= hd_vovo[a,k,b,l] * T1[c,k]
      hd_vvvo[a,c,b,l] += vvvv[a,c,b,d] * T1[d,l]
    end
    vvvv = nothing
    ecsave("hd_vvvo",hd_vvvo)
    hd_vvvo = nothing
    t1 = print_time(t1,"dress hd_vvvo",3)
  end

  # fully dressed
  if EC.calc_d_vovv
    # <ak\hat|bd>
    d_vovv = ints2("vovv")
    @tensoropt d_vovv[a,k,b,d] -= oovv[i,k,b,d] * T1[a,i]
    ecsave("d_vovv",d_vovv)
    t1 = print_time(t1,"dress d_vovv",3)
  end
  oovv = nothing
  if EC.calc_d_vvvv
    # <ab\hat|cd>
    d_vvvv = ecload("hd_vvvv")
    if !EC.calc_d_vovv
      error("for calc_d_vvvv calc_d_vovv has to be True")
    end
    @tensoropt d_vvvv[a,c,b,d] -= d_vovv[c,i,d,b] * T1[a,i]
    d_vovv = nothing
    ecsave("d_vvvv",d_vvvv)
    d_vvvv = nothing
    t1 = print_time(t1,"dress d_vvvv",3)
  end
  # <ak\hat|bl>
  d_vovo = hd_vovo
  @tensoropt d_vovo[a,k,b,l] -= d_oovo[i,k,b,l] * T1[a,i]
  ecsave("d_vovo",d_vovo)
  d_vovo = nothing
  t1 = print_time(t1,"dress d_vovo",3)
  # <aj\hat|kl>
  d_vooo = hd_vooo
  @tensoropt d_vooo[a,k,j,l] += d_voov[a,k,j,d] * T1[d,l]
  ecsave("d_vooo",d_vooo)
  t1 = print_time(t1,"dress d_vooo",3)
  if EC.calc_d_vvvo
    # <ab\hat|cl>
    d_vvvo = ecload("hd_vvvo")
    @tensoropt d_vvvo[a,c,b,l] -= d_voov[c,i,l,b] * T1[a,i]
    ecsave("d_vvvo",d_vvvo)
    d_vvvo = nothing
    t1 = print_time(t1,"dress d_vvvo",3)
  end
  # <ij\hat|kl>
  d_oooo = hd_oooo
  @tensoropt d_oooo[i,k,j,l] += d_oovo[i,k,b,l] * T1[b,j]
  ecsave("d_oooo",d_oooo)
  t1 = print_time(t1,"dress d_oooo",3)
  if EC.calc_d_vvoo
    if !EC.calc_d_vvvo
      error("for calc_d_vvoo calc_d_vvvo has to be True")
    end
    # <ac\hat|jl>
    d_vvoo = ecload("hd_vvoo")
    hd_vvvo = ecload("hd_vvvo")
    @tensoropt begin
      d_vvoo[a,c,j,l] += hd_vvvo[a,c,b,l] * T1[b,j]
      hd_vvvo = nothing
      d_vvoo[a,c,j,l] -= d_vooo[c,i,l,j] * T1[a,i]
    end
    ecsave("d_vvoo",d_vvoo)
    t1 = print_time(t1,"dress d_vvoo",3)
  end
  # dress 1-el part
  d_int1 = deepcopy(integ1(EC.fd))
  dinter = ints1(":v")
  @tensoropt d_int1[:,SP('o')][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[SP('o'),:]
  @tensoropt d_int1[SP('v'),:][b,p] -= dinter[j,p] * T1[b,j]
  ecsave("dint1",d_int1)
  t1 = print_time(t1,"dress int1",3)

  # calc dressed fock
  dfock = d_int1
  @tensoropt begin
    foo[i,j] := 2.0*d_oooo[i,k,j,k] - d_oooo[i,k,k,j]
    fvo[a,i] := 2.0*d_vooo[a,k,i,k] - d_vooo[a,k,k,i]
    fov[i,a] := 2.0*d_oovo[i,k,a,k] - d_oovo[k,i,a,k]
    d_vovo = ecload("d_vovo")
    fvv[a,b] := 2.0*d_vovo[a,k,b,k]
    d_vovo = nothing
    fvv[a,b] -= d_voov[a,k,k,b]
  end
  dfock[SP('o'),SP('o')] += foo
  dfock[SP('v'),SP('o')] += fvo
  dfock[SP('o'),SP('v')] += fov
  dfock[SP('v'),SP('v')] += fvv

  ecsave("dfock",dfock)
  t1 = print_time(t1,"dress fock",3)
end

function calc_MP2()
  # calc MP2 energy and amplitudes, return (EMp2, T2)
  T2 = update_doubles(ints2("vvoo"), false)
  EMp2 = calc_doubles_energy(T2)
  return EMp2, T2
end

function method_name(T1, dc = false)
  if dc
    name = "DC"
  else
    name = "CC"
  end
  if isnothing(T1)
    name *= "D"
  else
    name *= "SD"
  end
  return name
end

""" 
calc D^{ij}_{pq} = T^{ij}_{cd} + T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl}

return as D[pqij] 

if `scalepp`: D[ppij] elements are scaled by 0.5 (for triangular summation)
"""
function calc_D2(T1, T2, scalepp = false)
    norb = length(SP(':'))
    nocc = length(SP('o'))
    D2 = Array{Float64}(undef,norb,norb,nocc,nocc)
    @tensoropt begin
      D2[SP('v'),SP('v'),:,:][a,b,i,j] = T2[a,b,i,j] + T1[a,i] * T1[b,j]
      D2[SP('o'),SP('v'),:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
      D2[SP('v'),SP('o'),:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
      D2[SP('o'),SP('o'),:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k]
    end
    diagindx = [CartesianIndex(i,i) for i in 1:norb]
    D2[diagindx,:,:] *= 0.5
    return D2
end

"""
Calculate CCSD or DCSD residual.
"""
function calc_ccsd_resid(T1,T2,dc)
  t1 = time_ns()
  calc_dressed_ints(T1)
  t1 = print_time(t1,"dressing",2)
  @tensor T2t[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  dfock = ecload("dfock")
  if EC.use_kext
    dint1 = ecload("dint1")
    R1 = dint1[SP('v'),SP('o')]
  else
    R1 = dfock[SP('v'),SP('o')]
    if !EC.calc_d_vovv
      error("for not use_kext calc_d_vovv has to be True")
    end
    int2 = ecload("d_vovv")
    @tensoropt R1[a,i] += int2[a,k,b,c] * T2t[c,b,k,i]
  end
  int2 = ecload("d_oovo")
  fov = dfock[SP('o'),SP('v')]
  @tensoropt begin
    R1[a,i] += T2t[a,b,i,j] * fov[j,b]
    R1[a,i] -= int2[k,j,c,i] * T2t[c,a,k,j]
  end
  t1 = print_time(t1,"singles residual",2)

  # <ab|ij>
  if EC.use_kext
    R2 = zeros((length(SP('v')),length(SP('v')),length(SP('o')),length(SP('o'))))
  else
    if !EC.calc_d_vvoo
      error("for not use_kext calc_d_vvoo has to be True")
    end
    R2 = ecload("d_vvoo")
  end
  t1 = print_time(t1,"<ab|ij>",2)
  klcd = ints2("oovv")
  t1 = print_time(t1,"<kl|cd>",2)
  int2 = ecload("d_oooo")
  if !dc
    # I_klij = <kl|ij>+<kl|cd>T^ij_cd
    @tensoropt int2[k,l,i,j] += klcd[k,l,c,d] * T2[c,d,i,j]
  end
  # I_klij T^kl_ab
  @tensoropt R2[a,b,i,j] += int2[k,l,i,j] * T2[a,b,k,l]
  t1 = print_time(t1,"I_klij T^kl_ab",2)
  # <kl|cd>\tilde T^ki_ca \tilde T^lj_db
  @tensoropt R2[a,b,i,j] += klcd[k,l,c,d] * T2t[c,a,k,i] * T2t[d,b,l,j]
  t1 = print_time(t1,"<kl|cd> tT^ki_ca tT^lj_db",2)
  if EC.use_kext
    int2 = integ2(EC.fd)
    if ndims(int2) == 4
      if EC.triangular_kext
        trioo = [CartesianIndex(i,j) for j in 1:length(SP('o')) for i in 1:j]
        D2 = calc_D2(T1, T2)[:,:,trioo]
        # <pq|rs> D^ij_rs
        @tensoropt R2pqx[p,r,x] := int2[p,r,q,s] * D2[q,s,x]
        D2 = nothing
        norb = length(SP(':'))
        nocc = length(SP('o'))
        Rpqoo = Array{Float64}(undef,norb,norb,nocc,nocc)
        Rpqoo[:,:,trioo] = R2pqx
        trioor = CartesianIndex.(reverse.(Tuple.(trioo)))
        @tensor Rpqoo[:,:,trioor][p,q,x] = R2pqx[q,p,x]
        R2pqx = nothing
        @tensor R2pq[a,b,i,j] := Rpqoo[a,b,i,j]
        Rpqoo = nothing
      else
        D2 = calc_D2(T1, T2)
        # <pq|rs> D^ij_rs
        @tensoropt R2pq[p,r,i,j] := int2[p,r,q,s] * D2[q,s,i,j]
        D2 = nothing
      end
    else
      # last two indices of integrals are stored as upper triangular 
      tripp = [CartesianIndex(i,j) for j in 1:length(SP(':')) for i in 1:j]
      D2 = calc_D2(T1, T2, true)[tripp,:,:]
      # <pq|rs> D^ij_rs
      @tensoropt rR2pq[p,r,i,j] := int2[p,r,x] * D2[x,i,j]
      D2 = nothing
      # symmetrize R
      @tensoropt R2pq[p,r,i,j] := rR2pq[p,r,i,j] + rR2pq[r,p,j,i]
    end
    R2 += R2pq[SP('v'),SP('v'),:,:]
    @tensoropt begin
      R2[a,b,i,j] -= R2pq[SP('o'),SP('v'),:,:][k,b,i,j] * T1[a,k]
      R2[a,b,i,j] -= R2pq[SP('v'),SP('o'),:,:][a,k,i,j] * T1[b,k]
      R2[a,b,i,j] += R2pq[SP('o'),SP('o'),:,:][k,l,i,j] * T1[a,k] * T1[b,l]
    # singles residual contributions
      R1[a,i] +=  2.0 * R2pq[SP('v'),SP('o'),:,:][a,k,i,k] - R2pq[SP('v'),SP('o'),:,:][a,k,k,i]
      x1[k,i] := 2.0 * R2pq[SP('o'),SP('o'),:,:][k,l,i,l] - R2pq[SP('o'),SP('o'),:,:][k,l,l,i]
      R1[a,i] -= x1[k,i] * T1[a,k]
    end
    x1 = nothing
    R2pq = nothing
    t1 = print_time(t1,"kext",2)
  else
    if !EC.calc_d_vvvv
      error("for not use_kext calc_d_vvvv has to be True")
    end
    int2 = ecload("d_vvvv")
    # <ab|cd> T^ij_cd
    @tensoropt R2[a,b,i,j] += int2[a,b,c,d] * T2[c,d,i,j]
    t1 = print_time(t1,"<ab|cd> T^ij_cd",2)
  end
  if !dc
    # <kl|cd> T^kj_ad T^il_cb
    @tensoropt R2[a,b,i,j] += klcd[k,l,c,d] * T2[a,d,k,j] * T2[c,b,i,l]
    t1 = print_time(t1,"<kl|cd> T^kj_ad T^il_cb",2)
  end

  fac = dc ? 0.5 : 1.0
  # x_ad = f_ad - <kl|cd> \tilde T^kl_ca
  # x_ki = f_ki + <kl|cd> \tilde T^il_cd
  xad = dfock[SP('v'),SP('v')]
  xki = dfock[SP('o'),SP('o')]
  @tensoropt begin
    xad[a,d] -= fac * klcd[k,l,c,d] * T2t[c,a,k,l]
    xki[k,i] += fac * klcd[k,l,c,d] * T2t[c,d,i,l]
  end
  t1 = print_time(t1,"xad, xki",2)

  # terms for P(ia;jb)
  @tensoropt begin
    # x_ad T^ij_db
    R2r[a,b,i,j] := xad[a,d] * T2[d,b,i,j]
    # -x_ki T^kj_ab
    R2r[a,b,i,j] -= xki[k,i] * T2[a,b,k,j]
  end
  t1 = print_time(t1,"x_ad T^ij_db -x_ki T^kj_ab",2)
  int2 = ecload("d_voov")
  # <ak|ic> \tilde T^kj_cb
  @tensoropt R2r[a,b,i,j] += int2[a,k,i,c] * T2t[c,b,k,j]
  t1 = print_time(t1,"<ak|ic> tT^kj_cb",2)
  if !dc
    # -<kl|cd> T^ki_da (T^lj_cb - T^lj_bc)
    T2t -= T2
    @tensoropt R2r[a,b,i,j] -= klcd[k,l,c,d] * T2[d,a,k,i] * T2t[c,b,l,j]
    t1 = print_time(t1,"-<kl|cd> T^ki_da (T^lj_cb - T^lj_bc)",2)
  end
  int2 = ecload("d_vovo")
  @tensoropt begin
    # -<ka|ic> T^kj_cb
    R2r[a,b,i,j] -= int2[a,k,c,i] * T2[c,b,k,j]
    # -<kb|ic> T^kj_ac
    R2r[a,b,i,j] -= int2[b,k,c,i] * T2[a,c,k,j]
    t1 = print_time(t1,"-<ka|ic> T^kj_cb -<kb|ic> T^kj_ac",2)

    R2[a,b,i,j] += R2r[a,b,i,j] + R2r[b,a,j,i]
  end
  t1 = print_time(t1,"P(ia;jb)",2)

  return R1,R2
end

"""
Calculate coupled cluster amplitudes.

If T1 is `nothing` on input, no singles will be calculated.
If dc: calculate distinguishable cluster.
"""
function calc_cc!(T1, T2, dc = false)
  println(method_name(T1,dc))
  diis = Diis(EC.scr)

  println("Iter     SqNorm      Energy      DE          Res         Time")
  NormR1 = 0.0
  NormT1 = 0.0
  NormT2 = 0.0
  R1 = nothing
  Eh = 0.0
  t0 = time_ns()
  for it in 1:EC.maxit
    t1 = time_ns()
    if isnothing(T1)
      # R2 = calc_ccd_resid(T2,dc)
      R2 = T2 #FIX
    else
      R1, R2 = calc_ccsd_resid(T1,T2,dc)
      NormT1 = calc_singles_norm(T1)
      NormR1 = calc_singles_norm(R1)
    end
    t1 = print_time(t1,"residual",2)
    NormT2 = calc_doubles_norm(T2)
    NormR2 = calc_doubles_norm(R2)
    Eh = calc_hylleraas(T1,T2,R1,R2)
    T2 += update_doubles(R2)
    if isnothing(T1)
      T2 = perform(diis,[T2],[R2])
      En = 0.0
    else
      T1 += update_singles(R1)
      T1,T2 = perform(diis,[T1,T2],[R1,R2])
      En = calc_singles_energy(T1)
    end
    En += calc_doubles_energy(T2)
    ΔE = En - Eh  
    NormR = NormR1 + NormR2
    NormT = 1.0 + NormT1 + NormT2
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %12.8f %10.2e %8.2f \n" it NormT Eh ΔE NormR tt
    if NormR < EC.thr
      break
    end
  end
  println()
  @printf "Sq.Norm of T1: %12.8f Sq.Norm of T2: %12.8f \n" NormT1 NormT2
  println()
  return Eh
end

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--method", "-m"
      help = "method or list of methods to calculate"
      arg_type = String
      default = "dcsd"
    "--scratch", "-s"
      help = "scratch directory"
      arg_type = String
      default = "e-cojlscr"
    "--verbosity", "-v"
      help = "verbosity"
      arg_type = Int
      default = 2
    "--occa"
      help = "occupied α orbitals (in '1-3+5' format)"
      arg_type = String
      default = "-"
    "--occb"
      help = "occupied β orbitals (in '1-3+6' format)"
      arg_type = String
      default = "-"
    "arg1"
      help = "input file (currently fcidump file)"
      default = "FCIDUMP"
  end
  args = parse_args(s)
  EC.scr = args["scratch"]
  EC.verbosity = args["verbosity"]
  fcidump_file = args["arg1"]
  method = args["method"]
  occa = args["occa"]
  occb = args["occb"]
  return fcidump_file, method, occa, occb
end

function main()
  t1 = time_ns()
  fcidump, method_string, occa, occb = parse_commandline()
  method_names = split(method_string)
  # create scratch directory
  mkpath(EC.scr)
  EC.scr = mktempdir(EC.scr)
  # read fcidump intergrals
  EC.fd = read_fcidump(fcidump)
  t1 = print_time(t1,"read fcidump",1)
  println(size(EC.fd.int2))
  norb = headvar(EC.fd, "NORB")
  nelec = headvar(EC.fd, "NELEC")
  # EC.shifts = 0.0
  # EC.shiftp = 0.0

  EC.space['o'], EC.space['v'], EC.space['O'], EC.space['V'] = get_occvirt(occa, occb, norb, nelec)
  EC.space[':'] = 1:headvar(EC.fd,"NORB")

  closed_shell = (EC.space['o'] == EC.space['O'] && !EC.fd.uhf)
  
  closed_shell || error("Open-shell methods not implemented yet")
  # calculate fock matrix 
  EC.fock = gen_fock(SCα)
  ϵ = diag(EC.fock)
  EC.ϵo = ϵ[SP('o')]
  EC.ϵv = ϵ[SP('v')]
  println("Occupied orbital energies: ",EC.ϵo)
  t1 = print_time(t1,"fock matrix",1)

  # calculate HF energy
  EHF = sum(EC.ϵo) + sum(diag(integ1(EC.fd))[SP('o')]) + EC.fd.int0
  println("HF energy: ",EHF)

  for mname in method_names
    println()
    println("Next method: ",mname)
    ecmethod = ECMethod(mname)
    if ecmethod.unrestricted
      error("unrestricted not implemented yet...")
    end
    # at the moment we always calculate MP2 first
    # calculate MP2
    EMp2, T2 = calc_MP2()
    println("MP2 correlation energy: ",EMp2)
    println("MP2 total energy: ",EMp2+EHF)
    t1 = print_time(t1,"MP2",1)

    if ecmethod.theory == "MP"
      continue
    end
    dc = (ecmethod.theory == "DC")
    T1 = nothing
    if ecmethod.exclevel[1] == FullExc
      T1 = zeros(size(SP('v'),1),size(SP('o'),1))
    end
    if ecmethod.exclevel[3] != NoExc
      error("no triples implemented yet...")
    end
    ECC = calc_cc!(T1, T2, dc)
    println(mname*" correlation energy: ",ECC)
    println(mname*" total energy: ",ECC+EHF)
    t1 = print_time(t1,"CC",1)
  end
end
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

end #module
