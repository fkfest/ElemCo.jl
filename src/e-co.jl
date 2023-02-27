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
  o::Array{Int} = Int[]
  v::Array{Int} = Int[]
  fock::Array{Float64} = Float64[]
  ϵo::Array{Float64} = Float64[]
  ϵv::Array{Float64} = Float64[]
  use_kext::Bool = true
  calc_d_vvvv::Bool = false
  calc_d_vvvo::Bool = false
  calc_d_vvov::Bool = false
  calc_d_vovo::Bool = false
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

function get_occvirt(norb, nelec, occs = nothing)
  if isnothing(occs)
    occs = [1:nelec÷2;]
  end
  virts = [ i for i in 1:norb if i ∉ occs ]
  println("Occupied orbitals:", occs)
  # println("Virtual orbitals:", virts)
  return occs, virts
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


function gen_fock(occs)
  # calc fock matrix 
  if headvar(EC.fd, "IUHF") != 0
    error("UHF-type integrals not implemented yet!")
  end
  @tensoropt fock[p,q] := EC.fd.int1[p,q] + 2.0*EC.fd.int2[:,:,occs,occs][p,q,i,i] - EC.fd.int2[:,occs,occs,:][p,i,i,q]
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
    a,i,b,j = Tuple(I)
    ΔT2[I] /= -(EC.ϵv[a] + EC.ϵv[b] - EC.ϵo[i] - EC.ϵo[j] + shift)
  end
  return ΔT2
end

function calc_singles_energy(T1)
  @tensoropt begin
    ET1 = scalar((2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*EC.fd.int2[EC.o,EC.v,EC.o,EC.v][i,a,j,b])
    ET1 += scalar(2.0*T1[a,i] * EC.fock[EC.o,EC.v][i,a])
  end
  return ET1
end

function calc_doubles_energy(T2)
  @tensoropt ET2 = scalar((2.0*T2[a,i,b,j] - T2[b,i,a,j]) * EC.fd.int2[EC.o,EC.v,EC.o,EC.v][i,a,j,b])
  return ET2
end

function calc_hylleraas(T1,T2,R1,R2)
  int2 = EC.fd.int2[EC.o,EC.v,EC.o,EC.v]
  @tensoropt begin
    int2[i,a,j,b] += R2[a,i,b,j]
    ET2 = scalar((2.0*T2[a,i,b,j] - T2[b,i,a,j]) * int2[i,a,j,b])
  end
  if !isnothing(T1)
    dfock = ecload("dfock")
    fov = dfock[EC.o,EC.v] + EC.fock[EC.o,EC.v] # undressed part should be with factor two
    @tensoropt ET1 = scalar((fov[i,a] + 2.0 * R1[a,i])*T1[a,i])
    # ET1 = scalar(2.0*(EC.fock[EC.o,EC.v][i,a] + R1[a,i])*T1[a,i])
    # ET1 += scalar((2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*int2[i,a,j,b])
    ET2 += ET1
  end
  return ET2
end

function calc_singles_norm(T1)
  @tensor NormT1 = 2.0*scalar(T1[a,i]*T1[a,i])
  return NormT1
end

function calc_doubles_norm(T2)
  @tensor NormT2 = scalar((2.0*T2[a,i,b,j] - T2[b,i,a,j])*T2[a,i,b,j])
  return NormT2
end

function calc_dressed_ints(T1)
  t1 = time_ns()
  # first make half-transformed integrals
  if EC.calc_d_vvvv
    # (ab|\hat c d)
    hd_vvvv = EC.fd.int2[EC.v,EC.v,EC.v,EC.v]
    vvov = EC.fd.int2[EC.v,EC.v,EC.o,EC.v]
    @tensoropt hd_vvvv[a,b,c,d] -= vvov[a,b,k,d] * T1[c,k]
    vvov = nothing
    ecsave("hd_vvvv",hd_vvvv)
    hd_vvvv = nothing
    t1 = print_time(t1,"dress hd_vvvv",3)
  end
  # (ij|k \hat l)
  hd_oooo = EC.fd.int2[EC.o,EC.o,EC.o,EC.o]
  ooov = EC.fd.int2[EC.o,EC.o,EC.o,EC.v]
  @tensoropt hd_oooo[i,j,k,l] += ooov[i,j,k,d] * T1[d,l]
  ooov = nothing
  t1 = print_time(t1,"dress hd_oooo",3)
  if EC.calc_d_vovo
    # (aj|\hat c \hat l)
    hd_vovo = EC.fd.int2[EC.v,EC.o,EC.v,EC.o]
    voov = EC.fd.int2[EC.v,EC.o,EC.o,EC.v]
    vooo = EC.fd.int2[EC.v,EC.o,EC.o,EC.o]
    @tensoropt begin
      vooo[a,j,k,l] += voov[a,j,k,d] * T1[d,l]
      voov = nothing
      hd_vovo[a,j,c,l] -= vooo[a,j,k,l] * T1[c,k]
      vooo = nothing
    end
    vovv = EC.fd.int2[EC.v,EC.o,EC.v,EC.v]
    @tensoropt hd_vovo[a,j,c,l] += vovv[a,j,c,d] * T1[d,l]
    vovv = nothing
    ecsave("hd_vovo",hd_vovo)
    hd_vovo = nothing
    t1 = print_time(t1,"dress hd_vovo",3)
  end
  # (\hat a \hat j |kl)
  hd_vooo = EC.fd.int2[EC.v,EC.o,EC.o,EC.o]
  vvoo = EC.fd.int2[EC.v,EC.v,EC.o,EC.o]
  @tensoropt begin
    hd_vooo[a,j,k,l] -= hd_oooo[k,l,i,j] * T1[a,i]
    hd_vooo[a,j,k,l] += vvoo[a,b,k,l] * T1[b,j]
  end
  t1 = print_time(t1,"dress hd_vooo",3)
  # some of the fully dressing moved here...
  # (kd\hat|ij)
  d_ovoo = EC.fd.int2[EC.o,EC.v,EC.o,EC.o]
  ovov = EC.fd.int2[EC.o,EC.v,EC.o,EC.v]
  @tensoropt d_ovoo[k,d,i,j] += ovov[k,d,i,b] * T1[b,j]
  ecsave("d_ovoo",d_ovoo)
  t1 = print_time(t1,"dress d_ovoo",3)
  # (aj\hat|kd)
  d_voov = EC.fd.int2[EC.v,EC.o,EC.o,EC.v]
  vvov = EC.fd.int2[EC.v,EC.v,EC.o,EC.v]
  @tensoropt begin
    d_voov[a,j,k,d] -= d_ovoo[k,d,i,j] * T1[a,i]
    d_voov[a,j,k,d] += vvov[a,b,k,d] * T1[b,j]
  end
  ecsave("d_voov",d_voov)
  t1 = print_time(t1,"dress d_voov",3)
  # finish half-dressing
  # (ab|k \hat l)
  hd_vvoo = EC.fd.int2[EC.v,EC.v,EC.o,EC.o]
  @tensoropt hd_vvoo[a,b,k,l] += vvov[a,b,k,d] * T1[d,l]
  vvov = nothing
  t1 = print_time(t1,"dress hd_vvoo",3)
  if EC.calc_d_vvvo
    # (ab | \hat c \hat l)
    hd_vvvo = EC.fd.int2[EC.v,EC.v,EC.v,EC.o]
    vvvv = EC.fd.int2[EC.v,EC.v,EC.v,EC.v]
    @tensoropt begin
      hd_vvvo[a,b,c,l] -= hd_vvoo[a,b,k,l] * T1[c,k]
      hd_vvvo[a,b,c,l] += vvvv[a,b,c,d] * T1[d,l]
    end
    vvvv = nothing
    ecsave("hd_vvvo",hd_vvvo)
    hd_vvvo = nothing
    t1 = print_time(t1,"dress hd_vvvo",3)
  end

  # fully dressed
  if EC.calc_d_vvov
    # (ab\hat|kd)
    d_vvov = EC.fd.int2[EC.v,EC.v,EC.o,EC.v]
    @tensoropt d_vvov[a,b,k,d] -= ovov[i,b,k,d] * T1[a,i]
    ecsave("d_vvov",d_vvov)
    t1 = print_time(t1,"dress d_vvov",3)
  end
  ovov = nothing
  if EC.calc_d_vvvv
    # (ab\hat|cd)
    d_vvvv = ecload("hd_vvvv")
    if !EC.calc_d_vvov
      error("for calc_d_vvvv calc_d_vvov has to be True")
    end
    @tensoropt d_vvvv[a,b,c,d] -= d_vvov[c,d,i,b] * T1[a,i]
    d_vvov = nothing
    ecsave("d_vvvv",d_vvvv)
    d_vvvv = nothing
    t1 = print_time(t1,"dress d_vvvv",3)
  end
  # (ab\hat|kl)
  d_vvoo = hd_vvoo
  @tensoropt d_vvoo[a,b,k,l] -= d_ovoo[i,b,k,l] * T1[a,i]
  ecsave("d_vvoo",d_vvoo)
  d_vvoo = nothing
  t1 = print_time(t1,"dress d_vvoo",3)
  # (aj\hat|kl)
  d_vooo = hd_vooo
  @tensoropt d_vooo[a,j,k,l] += d_voov[a,j,k,d] * T1[d,l]
  ecsave("d_vooo",d_vooo)
  t1 = print_time(t1,"dress d_vooo",3)
  if EC.calc_d_vvvo
    # (ab\hat|cl)
    d_vvvo = ecload("hd_vvvo")
    @tensoropt d_vvvo[a,b,c,l] -= d_voov[c,l,i,b] * T1[a,i]
    ecsave("d_vvvo",d_vvvo)
    d_vvvo = nothing
    t1 = print_time(t1,"dress d_vvvo",3)
  end
  # (ij\hat|kl)
  d_oooo = hd_oooo
  @tensoropt d_oooo[i,j,k,l] += d_ovoo[i,b,k,l] * T1[b,j]
  ecsave("d_oooo",d_oooo)
  t1 = print_time(t1,"dress d_oooo",3)
  if EC.calc_d_vovo
    if !EC.calc_d_vvvo
      error("for calc_d_vovo calc_d_vvvo has to be True")
    end
    # (aj\hat|cl)
    d_vovo = ecload("hd_vovo")
    hd_vvvo = ecload("hd_vvvo")
    @tensoropt begin
      d_vovo[a,j,c,l] += hd_vvvo[a,b,c,l] * T1[b,j]
      hd_vvvo = nothing
      d_vovo[a,j,c,l] -= d_vooo[c,l,i,j] * T1[a,i]
    end
    ecsave("d_vovo",d_vovo)
    t1 = print_time(t1,"dress d_vovo",3)
  end
  # dress 1-el part
  d_int1 = deepcopy(EC.fd.int1)
  dinter = EC.fd.int1[:,EC.v]
  @tensoropt d_int1[:,EC.o][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[EC.o,:]
  @tensoropt d_int1[EC.v,:][b,p] -= dinter[j,p] * T1[b,j]
  ecsave("dint1",d_int1)
  t1 = print_time(t1,"dress int1",3)

  # calc dressed fock
  dfock = d_int1
  @tensoropt begin
    foo[i,j] := 2.0*d_oooo[i,j,k,k] - d_oooo[i,k,k,j]
    fvo[a,i] := 2.0*d_vooo[a,i,k,k] - d_vooo[a,k,k,i]
    fov[i,a] := 2.0*d_ovoo[i,a,k,k] - d_ovoo[k,a,i,k]
    d_vvoo = ecload("d_vvoo")
    fvv[a,b] := 2.0*d_vvoo[a,b,k,k]
    d_vvoo = nothing
    fvv[a,b] -= d_voov[a,k,k,b]
  end
  dfock[EC.o,EC.o] += foo
  dfock[EC.v,EC.o] += fvo
  dfock[EC.o,EC.v] += fov
  dfock[EC.v,EC.v] += fvv

  ecsave("dfock",dfock)
  t1 = print_time(t1,"dress fock",3)
end

function calc_MP2()
  # calc MP2 energy and amplitudes, return (EMp2, T2)
  T2 = update_doubles(EC.fd.int2[EC.v,EC.o,EC.v,EC.o], false)
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
"""
function calc_D2(T1, T2)
    norb = size(EC.fd.int2,1)
    nocc = length(EC.o)
    D2 = Array{Float64}(undef,norb,norb,nocc,nocc)
    @tensoropt begin
      D2[EC.v,EC.v,:,:][a,b,i,j] = T2[a,i,b,j] + T1[a,i] * T1[b,j]
      D2[EC.o,EC.v,:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
      D2[EC.v,EC.o,:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
      D2[EC.o,EC.o,:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k]
    end
    return D2
end

"""
Calculate CCSD or DCSD residual.
"""
function calc_ccsd_resid(T1,T2,dc)
  t1 = time_ns()
  calc_dressed_ints(T1)
  t1 = print_time(t1,"dressing",2)
  @tensor T2t[a,i,b,j] := 2.0 * T2[a,i,b,j] - T2[b,i,a,j]
  dfock = ecload("dfock")
  if EC.use_kext
    dint1 = ecload("dint1")
    R1 = dint1[EC.v,EC.o]
  else
    R1 = dfock[EC.v,EC.o]
    if !EC.calc_d_vvov
      error("for not use_kext calc_d_vvov has to be True")
    end
    int2 = ecload("d_vvov")
    @tensoropt R1[a,i] += int2[a,b,k,c] * T2t[c,k,b,i]
  end
  int2 = ecload("d_ovoo")
  fov = dfock[EC.o,EC.v]
  @tensoropt begin
    R1[a,i] += T2t[a,i,b,j] * fov[j,b]
    R1[a,i] -= int2[k,c,j,i] * T2t[c,k,a,j]
  end
  t1 = print_time(t1,"singles residual",2)

  # (ai|bj)
  if EC.use_kext
    R2 = zeros((length(EC.v),length(EC.o),length(EC.v),length(EC.o)))
  else
    if !EC.calc_d_vovo
      error("for not use_kext calc_d_vovo has to be True")
    end
    R2 = ecload("d_vovo")
  end
  t1 = print_time(t1,"(ai|bj)",2)
  kcld = EC.fd.int2[EC.o,EC.v,EC.o,EC.v]
  t1 = print_time(t1,"(kc|ld)",2)
  int2 = ecload("d_oooo")
  if !dc
    # I_kilj = (ki|lj)+(kc|ld)T^ij_cd
    @tensoropt int2[k,i,l,j] += kcld[k,c,l,d] * T2[c,i,d,j]
  end
  # I_kilj T^kl_ab
  @tensoropt R2[a,i,b,j] += int2[k,i,l,j] * T2[a,k,b,l]
  t1 = print_time(t1,"I_kilj T^kl_ab",2)
  # (kc|ld)\tilde T^ki_ca \tilde T^lj_db
  @tensoropt R2[a,i,b,j] += kcld[k,c,l,d] * T2t[c,k,a,i] * T2t[d,l,b,j]
  t1 = print_time(t1,"(kc|ld) tT^ki_ca tT^lj_db",2)
  if EC.use_kext
    if EC.triangular_kext
      trioo = [CartesianIndex(i,j) for j in 1:length(EC.o) for i in 1:j]
      D2 = calc_D2(T1, T2)[:,:,trioo]
      # (pr|qs) D^ij_rs
      @tensoropt R2pqx[p,r,x] := EC.fd.int2[p,q,r,s] * D2[q,s,x]
      D2 = nothing
      norb = size(EC.fd.int2,1)
      nocc = length(EC.o)
      Rpqoo = Array{Float64}(undef,norb,norb,nocc,nocc)
      Rpqoo[:,:,trioo] = R2pqx
      trioor = CartesianIndex.(reverse.(Tuple.(trioo)))
      @tensor Rpqoo[:,:,trioor][p,q,x] = R2pqx[q,p,x]
      R2pqx = nothing
      @tensor R2pq[a,i,b,j] := Rpqoo[a,b,i,j]
      Rpqoo = nothing
    else
      D2 = calc_D2(T1, T2)
      # (pr|qs) D^ij_rs
      @tensoropt R2pq[p,i,r,j] := EC.fd.int2[p,q,r,s] * D2[q,s,i,j]
      D2 = nothing
    end
    R2 += R2pq[EC.v,:,EC.v,:]
    @tensoropt begin
      R2[a,i,b,j] -= R2pq[EC.o,:,EC.v,:][k,i,b,j] * T1[a,k]
      R2[a,i,b,j] -= R2pq[EC.v,:,EC.o,:][a,i,k,j] * T1[b,k]
      R2[a,i,b,j] += R2pq[EC.o,:,EC.o,:][k,i,l,j] * T1[a,k] * T1[b,l]
    # singles residual contributions
      R1[a,i] +=  2.0 * R2pq[EC.v,:,EC.o,:][a,i,k,k] - R2pq[EC.v,:,EC.o,:][a,k,k,i]
      x1[k,i] := 2.0 * R2pq[EC.o,:,EC.o,:][k,i,l,l] - R2pq[EC.o,:,EC.o,:][k,l,l,i]
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
    # (ac|bd) T^ij_cd
    @tensoropt R2[a,i,b,j] += int2[a,c,b,d] * T2[c,i,d,j]
    t1 = print_time(t1,"(ac|bd) T^ij_cd",2)
  end
  if !dc
    # (kc|ld) T^kj_ad T^il_cb
    @tensoropt R2[a,i,b,j] += kcld[k,c,l,d] * T2[a,k,d,j] * T2[c,i,b,l]
    t1 = print_time(t1,"(kc|ld) T^kj_ad T^il_cb",2)
  end

  fac = dc ? 0.5 : 1.0
  # x_ad = f_ad - (kc|ld) \tilde T^kl_ca
  # x_ki = f_ki + (kc|ld) \tilde T^il_cd
  xad = dfock[EC.v,EC.v]
  xki = dfock[EC.o,EC.o]
  @tensoropt begin
    xad[a,d] -= fac * kcld[k,c,l,d] * T2t[c,k,a,l]
    xki[k,i] += fac * kcld[k,c,l,d] * T2t[c,i,d,l]
  end
  t1 = print_time(t1,"xad, xki",2)

  # terms for P(ia;jb)
  @tensoropt begin
    # x_ad T^ij_db
    R2r[a,i,b,j] := xad[a,d] * T2[d,i,b,j]
    # -x_ki T^kj_ab
    R2r[a,i,b,j] -= xki[k,i] * T2[a,k,b,j]
  end
  t1 = print_time(t1,"x_ad T^ij_db -x_ki T^kj_ab",2)
  int2 = ecload("d_voov")
  # (ai|kc) \tilde T^kj_cb
  @tensoropt R2r[a,i,b,j] += int2[a,i,k,c] * T2t[c,k,b,j]
  t1 = print_time(t1,"(ai|kc) tT^kj_cb",2)
  if !dc
    # -(kc|ld) T^ki_da (T^lj_cb - T^lj_bc)
    T2t -= T2
    @tensoropt R2r[a,i,b,j] -= kcld[k,c,l,d] * T2[d,k,a,i] * T2t[c,l,b,j]
    t1 = print_time(t1,"-(kc|ld) T^ki_da (T^lj_cb - T^lj_bc)",2)
  end
  int2 = ecload("d_vvoo")
  @tensoropt begin
    # -(ki|ac) T^kj_cb
    R2r[a,i,b,j] -= int2[a,c,k,i] * T2[c,k,b,j]
    # -(ki|bc) T^kj_ac
    R2r[a,i,b,j] -= int2[b,c,k,i] * T2[a,k,c,j]
    t1 = print_time(t1,"-(ki|ac) T^kj_cb -(ki|bc) T^kj_ac",2)

    R2[a,i,b,j] += R2r[a,i,b,j] + R2r[b,j,a,i]
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
    "arg1"
      help = "input file (currently fcidump file)"
      default = "FCIDUMP"
  end
  args = parse_args(s)
  EC.scr = args["scratch"]
  EC.verbosity = args["verbosity"]
  fcidump_file = args["arg1"]
  method = args["method"]
  return fcidump_file, method
end

function main()
  t1 = time_ns()
  fcidump, method_string = parse_commandline()
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

  EC.o, EC.v = get_occvirt(norb, nelec)

  # calculate fock matrix
  EC.fock = gen_fock(EC.o)
  ϵ = diag(EC.fock)
  EC.ϵo = ϵ[EC.o]
  EC.ϵv = ϵ[EC.v]
  println("Occupied orbital energies: ",EC.ϵo)
  t1 = print_time(t1,"fock matrix",1)

  # calculate HF energy
  EHF = sum(EC.ϵo) + sum(diag(EC.fd.int1)[EC.o]) + EC.fd.int0
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
    dc = false
    if ecmethod.theory == "DC"
      dc = true
    end
    T1 = nothing
    if ecmethod.exclevel[1] == FullExc
      T1 = zeros(size(EC.v,1),size(EC.o,1))
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
