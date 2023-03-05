""" coupled-cluster methods """
module CoupledCluster

using LinearAlgebra
try
  using MKL
catch
  println("MKL package not found, using OpenBLAS.")
end
#BLAS.set_num_threads(1)
using TensorOperations
using Printf
using ..Utils
using ..ECInfos
using ..TensorTools
using ..FciDump
using ..DIIS

export calc_MP2, calc_UMP2, calc_cc!

function update_singles(R1, ϵo, ϵv, shift)
  ΔT1 = deepcopy(R1)
  for I ∈ CartesianIndices(ΔT1)
    a,i = Tuple(I)
    ΔT1[I] /= -(ϵv[a] - ϵo[i] + shift)
  end
  return ΔT1
end

function update_singles(EC::ECInfo, R1; spincase::SpinCase=SCα, use_shift=true)
  shift = use_shift ? EC.shifts : 0.0
  if spincase == SCα
    return update_singles(R1, EC.ϵo, EC.ϵv, shift)
  else
    return update_singles(R1, EC.ϵob, EC.ϵvb, shift)
  end
end

function update_doubles(R2, ϵo1, ϵv1, ϵo2, ϵv2, shift, antisymmetrize=false)
  ΔT2 = deepcopy(R2)
  if antisymmetrize
    ΔT2 -= permutedims(R2,(1,2,4,3))
  end
  for I ∈ CartesianIndices(ΔT2)
    a,b,i,j = Tuple(I)
    ΔT2[I] /= -(ϵv1[a] + ϵv2[b] - ϵo1[i] - ϵo2[j] + shift)
  end
  return ΔT2
end

function update_doubles(EC::ECInfo, R2; spincase::SpinCase=SCα, antisymmetrize=false, use_shift=true)
  shift = use_shift ? EC.shiftp : 0.0
  if spincase == SCα
    return update_doubles(R2, EC.ϵo, EC.ϵv, EC.ϵo, EC.ϵv, shift, antisymmetrize)
  elseif spincase == SCβ
    return update_doubles(R2, EC.ϵob, EC.ϵvb, EC.ϵob, EC.ϵvb, shift, antisymmetrize)
  else
    return update_doubles(R2, EC.ϵo, EC.ϵv, EC.ϵob, EC.ϵvb, shift, antisymmetrize)
  end
end

function calc_singles_energy(EC::ECInfo, T1; fock_only=false)
  SP(sp::Char) = EC.space[sp]
  ET1 = 0.0
  if !fock_only
    @tensoropt ET1 += scalar((2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*ints2(EC,"oovv")[i,j,a,b])
  end
  @tensoropt ET1 += scalar(2.0*T1[a,i] * EC.fock[SP('o'),SP('v')][i,a])
  return ET1
end

function calc_singles_energy(EC::ECInfo, T1a, T1b; fock_only=false)
  SP(sp::Char) = EC.space[sp]
  ET1 = 0.0
  if !fock_only
    @tensoropt begin
      ET1 += 0.5*scalar((T1a[a,i]*T1a[b,j]-T1a[b,i]*T1a[a,j])*ints2(EC,"oovv")[i,j,a,b])
      ET1 += 0.5*scalar((T1b[a,i]*T1b[b,j]-T1b[b,i]*T1b[a,j])*ints2(EC,"OOVV")[i,j,a,b])
      ET1 += scalar(T1a[a,i]*T1b[b,j]*ints2(EC,"oOvV")[i,j,a,b])
    end
  end
  @tensoropt begin
    ET1 += scalar(T1a[a,i] * EC.fock[SP('o'),SP('v')][i,a])
    ET1 += scalar(T1b[a,i] * EC.fockb[SP('O'),SP('V')][i,a])
  end
  return ET1
end

function calc_doubles_energy(EC::ECInfo, T2)
  @tensoropt ET2 = scalar((2.0*T2[a,b,i,j] - T2[b,a,i,j]) * ints2(EC,"oovv")[i,j,a,b])
  return ET2
end

function calc_doubles_energy(EC::ECInfo, T2a, T2b, T2ab)
  @tensoropt begin
    ET2 = 0.5*scalar(T2a[a,b,i,j] * ints2(EC,"oovv")[i,j,a,b])
    ET2 += 0.5*scalar(T2b[a,b,i,j] * ints2(EC,"OOVV")[i,j,a,b])
    ET2 += scalar(T2ab[a,b,i,j] * ints2(EC,"oOvV")[i,j,a,b])
  end
  return ET2
end

function calc_hylleraas(EC::ECInfo, T1,T2,R1,R2)
  SP(sp::Char) = EC.space[sp]
  int2 = ints2(EC,"oovv")
  @tensoropt begin
    int2[i,j,a,b] += R2[a,b,i,j]
    ET2 = scalar((2.0*T2[a,b,i,j] - T2[b,a,i,j]) * int2[i,j,a,b])
  end
  if !isnothing(T1)
    dfock = load(EC,"dfock")
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

function calc_singles_norm(T1a, T1b)
  @tensor begin
    NormT1 = scalar(T1a[a,i]*T1a[a,i])
    NormT1 += scalar(T1b[a,i]*T1b[a,i])
  end
  return NormT1
end

function calc_doubles_norm(T2)
  @tensoropt NormT2 = scalar((2.0*T2[a,b,i,j] - T2[b,a,i,j])*T2[a,b,i,j])
  return NormT2
end

function calc_doubles_norm(T2a, T2b, T2ab)
  @tensoropt begin
    NormT2 = 0.25*scalar(T2a[a,b,i,j]*T2a[a,b,i,j])
    NormT2 += 0.25*scalar(T2b[a,b,i,j]*T2b[a,b,i,j])
    NormT2 += scalar(T2ab[a,b,i,j]*T2ab[a,b,i,j])
  end
  return NormT2
end

"""dress integrals with singles"""
function calc_dressed_ints(EC::ECInfo, T1)
  t1 = time_ns()
  SP(sp::Char) = EC.space[sp]
  # first make half-transformed integrals
  if EC.calc_d_vvvv
    # <a\hat c|bd>
    hd_vvvv = ints2(EC,"vvvv")
    vovv = ints2(EC,"vovv")
    @tensoropt hd_vvvv[a,c,b,d] -= vovv[a,k,b,d] * T1[c,k]
    vovv = nothing
    save(EC,"hd_vvvv",hd_vvvv)
    hd_vvvv = nothing
    t1 = print_time(EC,t1,"dress hd_vvvv",3)
  end
  # <ik|j \hat l>
  hd_oooo = ints2(EC,"oooo")
  oovo = ints2(EC,"oovo")
  @tensoropt hd_oooo[j,i,l,k] += oovo[i,j,d,l] * T1[d,k]
  oovo = nothing
  t1 = print_time(EC,t1,"dress hd_oooo",3)
  if EC.calc_d_vvoo
    # <a\hat c|j \hat l>
    hd_vvoo = ints2(EC,"vvoo")
    voov = ints2(EC,"voov")
    vooo = ints2(EC,"vooo")
    @tensoropt begin
      vooo[a,k,j,l] += voov[a,k,j,d] * T1[d,l]
      voov = nothing
      hd_vvoo[a,c,j,l] -= vooo[a,k,j,l] * T1[c,k]
      vooo = nothing
    end
    vvov = ints2(EC,"vvov")
    @tensoropt hd_vvoo[a,c,j,l] += vvov[a,c,j,d] * T1[d,l]
    vvov = nothing
    save(EC,"hd_vvoo",hd_vvoo)
    hd_vvoo = nothing
    t1 = print_time(EC,t1,"dress hd_vvoo",3)
  end
  # <\hat a k| \hat j l)
  hd_vooo = ints2(EC,"vooo")
  vovo = ints2(EC,"vovo")
  @tensoropt begin
    hd_vooo[a,k,j,l] -= hd_oooo[k,i,l,j] * T1[a,i]
    hd_vooo[a,k,j,l] += vovo[a,k,b,l] * T1[b,j]
  end
  t1 = print_time(EC,t1,"dress hd_vooo",3)
  # some of the fully dressing moved here...
  # <ki\hat|dj>
  d_oovo = ints2(EC,"oovo")
  oovv = ints2(EC,"oovv")
  @tensoropt d_oovo[k,i,d,j] += oovv[k,i,d,b] * T1[b,j]
  save(EC,"d_oovo",d_oovo)
  t1 = print_time(EC,t1,"dress d_oovo",3)
  # <ak\hat|jd>
  d_voov = ints2(EC,"voov")
  vovv = ints2(EC,"vovv")
  @tensoropt begin
    d_voov[a,k,j,d] -= d_oovo[k,i,d,j] * T1[a,i]
    d_voov[a,k,j,d] += vovv[a,k,b,d] * T1[b,j]
  end
  save(EC,"d_voov",d_voov)
  t1 = print_time(EC,t1,"dress d_voov",3)
  # finish half-dressing
  # <ak|b \hat l>
  hd_vovo = ints2(EC,"vovo")
  @tensoropt hd_vovo[a,k,b,l] += vovv[a,k,b,d] * T1[d,l]
  vovv = nothing
  t1 = print_time(EC,t1,"dress hd_vovo",3)
  if EC.calc_d_vvvo
    # <a\hat c|b \hat l>
    hd_vvvo = ints2(EC,"vvvo")
    vvvv = ints2(EC,"vvvv")
    @tensoropt begin
      hd_vvvo[a,c,b,l] -= hd_vovo[a,k,b,l] * T1[c,k]
      hd_vvvo[a,c,b,l] += vvvv[a,c,b,d] * T1[d,l]
    end
    vvvv = nothing
    save(EC,"hd_vvvo",hd_vvvo)
    hd_vvvo = nothing
    t1 = print_time(EC,t1,"dress hd_vvvo",3)
  end

  # fully dressed
  if EC.calc_d_vovv
    # <ak\hat|bd>
    d_vovv = ints2(EC,"vovv")
    @tensoropt d_vovv[a,k,b,d] -= oovv[i,k,b,d] * T1[a,i]
    save(EC,"d_vovv",d_vovv)
    t1 = print_time(EC,t1,"dress d_vovv",3)
  end
  oovv = nothing
  if EC.calc_d_vvvv
    # <ab\hat|cd>
    d_vvvv = load(EC,"hd_vvvv")
    if !EC.calc_d_vovv
      error("for calc_d_vvvv calc_d_vovv has to be True")
    end
    @tensoropt d_vvvv[a,c,b,d] -= d_vovv[c,i,d,b] * T1[a,i]
    d_vovv = nothing
    save(EC,"d_vvvv",d_vvvv)
    d_vvvv = nothing
    t1 = print_time(EC,t1,"dress d_vvvv",3)
  end
  # <ak\hat|bl>
  d_vovo = hd_vovo
  @tensoropt d_vovo[a,k,b,l] -= d_oovo[i,k,b,l] * T1[a,i]
  save(EC,"d_vovo",d_vovo)
  d_vovo = nothing
  t1 = print_time(EC,t1,"dress d_vovo",3)
  # <aj\hat|kl>
  d_vooo = hd_vooo
  @tensoropt d_vooo[a,k,j,l] += d_voov[a,k,j,d] * T1[d,l]
  save(EC,"d_vooo",d_vooo)
  t1 = print_time(EC,t1,"dress d_vooo",3)
  if EC.calc_d_vvvo
    # <ab\hat|cl>
    d_vvvo = load(EC,"hd_vvvo")
    @tensoropt d_vvvo[a,c,b,l] -= d_voov[c,i,l,b] * T1[a,i]
    save(EC,"d_vvvo",d_vvvo)
    d_vvvo = nothing
    t1 = print_time(EC,t1,"dress d_vvvo",3)
  end
  # <ij\hat|kl>
  d_oooo = hd_oooo
  @tensoropt d_oooo[i,k,j,l] += d_oovo[i,k,b,l] * T1[b,j]
  save(EC,"d_oooo",d_oooo)
  t1 = print_time(EC,t1,"dress d_oooo",3)
  if EC.calc_d_vvoo
    if !EC.calc_d_vvvo
      error("for calc_d_vvoo calc_d_vvvo has to be True")
    end
    # <ac\hat|jl>
    d_vvoo = load(EC,"hd_vvoo")
    hd_vvvo = load(EC,"hd_vvvo")
    @tensoropt begin
      d_vvoo[a,c,j,l] += hd_vvvo[a,c,b,l] * T1[b,j]
      hd_vvvo = nothing
      d_vvoo[a,c,j,l] -= d_vooo[c,i,l,j] * T1[a,i]
    end
    save(EC,"d_vvoo",d_vvoo)
    t1 = print_time(EC,t1,"dress d_vvoo",3)
  end
  # dress 1-el part
  d_int1 = deepcopy(integ1(EC.fd))
  dinter = ints1(EC,":v")
  @tensoropt d_int1[:,SP('o')][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[SP('o'),:]
  @tensoropt d_int1[SP('v'),:][b,p] -= dinter[j,p] * T1[b,j]
  save(EC,"dint1",d_int1)
  t1 = print_time(EC,t1,"dress int1",3)

  # calc dressed fock
  dfock = d_int1
  @tensoropt begin
    foo[i,j] := 2.0*d_oooo[i,k,j,k] - d_oooo[i,k,k,j]
    fvo[a,i] := 2.0*d_vooo[a,k,i,k] - d_vooo[a,k,k,i]
    fov[i,a] := 2.0*d_oovo[i,k,a,k] - d_oovo[k,i,a,k]
    d_vovo = load(EC,"d_vovo")
    fvv[a,b] := 2.0*d_vovo[a,k,b,k]
    d_vovo = nothing
    fvv[a,b] -= d_voov[a,k,k,b]
  end
  dfock[SP('o'),SP('o')] += foo
  dfock[SP('v'),SP('o')] += fvo
  dfock[SP('o'),SP('v')] += fov
  dfock[SP('v'),SP('v')] += fvv

  save(EC,"dfock",dfock)
  t1 = print_time(EC,t1,"dress fock",3)
end

"""save non-dressed integrals in files instead of dressed integrals"""
function pseudo_dressed_ints(EC::ECInfo)
  t1 = time_ns()
  save(EC,"d_oovo",ints2(EC,"oovo"))
  save(EC,"d_voov",ints2(EC,"voov"))
  if EC.calc_d_vovv
    save(EC,"d_vovv",ints2(EC,"vovv"))
  end
  if EC.calc_d_vvvv
    save(EC,"d_vvvv",ints2(EC,"vvvv"))
  end
  save(EC,"d_vovo",ints2(EC,"vovo"))
  save(EC,"d_vooo",ints2(EC,"vooo"))
  if EC.calc_d_vvvo
    save(EC,"d_vvvo",ints2(EC,"vvvo"))
  end
  save(EC,"d_oooo",ints2(EC,"oooo"))
  if EC.calc_d_vvoo
    save(EC,"d_vvoo",ints2(EC,"vvoo"))
  end
  save(EC,"dint1",integ1(EC.fd))
  save(EC,"dfock",EC.fock)
  t1 = print_time(EC,t1,"pseudo-dressing",3)
end

""" Calculate closed-shell MP2 energy and amplitudes. 
    Return (EMp2, T2) """
function calc_MP2(EC::ECInfo)
  T2 = update_doubles(EC,ints2(EC,"vvoo"), use_shift=false)
  EMp2 = calc_doubles_energy(EC,T2)
  return EMp2, T2
end

""" Calculate unrestricted MP2 energy and amplitudes. 
    Return (EMp2, T2a, T2b, T2ab)"""
function calc_UMP2(EC::ECInfo, addsingles=true)
  SP(sp::Char) = EC.space[sp]
  T2a = update_doubles(EC,ints2(EC,"vvoo"), spincase=SCα, antisymmetrize = true, use_shift=false)
  T2b = update_doubles(EC,ints2(EC,"VVOO"), spincase=SCβ, antisymmetrize = true, use_shift=false)
  T2ab = update_doubles(EC,ints2(EC,"vVoO"), spincase=SCαβ, use_shift=false)
  EMp2 = calc_doubles_energy(EC,T2a,T2b,T2ab)
  if addsingles
    T1a = update_singles(EC,EC.fock[SP('v'),SP('o')], spincase=SCα, use_shift=false)
    T1b = update_singles(EC,EC.fockb[SP('V'),SP('O')], spincase=SCβ, use_shift=false)
    EMp2 += calc_singles_energy(EC, T1a, T1b, fock_only = true)
  end
  return EMp2, T2a, T2b, T2ab
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
function calc_D2(EC::ECInfo, T1, T2, scalepp = false)
  SP(sp::Char) = EC.space[sp]
  norb = length(SP(':'))
  nocc = length(SP('o'))
  if !isnothing(T1)
    D2 = Array{Float64}(undef,norb,norb,nocc,nocc)
  else
    D2 = zeros(norb,norb,nocc,nocc)
  end
  @tensoropt begin
    D2[SP('v'),SP('v'),:,:][a,b,i,j] = T2[a,b,i,j] 
    D2[SP('o'),SP('o'),:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k]
  end
  if !isnothing(T1)
    @tensoropt begin
      D2[SP('v'),SP('v'),:,:][a,b,i,j] += T1[a,i] * T1[b,j]
      D2[SP('o'),SP('v'),:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
      D2[SP('v'),SP('o'),:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
    end
  end
  diagindx = [CartesianIndex(i,i) for i in 1:norb]
  D2[diagindx,:,:] *= 0.5
  return D2
end

"""
Calculate CCSD or DCSD residual.
"""
function calc_ccsd_resid(EC::ECInfo, T1,T2,dc)
  t1 = time_ns()
  SP(sp::Char) = EC.space[sp]
  if !isnothing(T1)
    calc_dressed_ints(EC,T1)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC)
  end
  @tensor T2t[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  dfock = load(EC,"dfock")
  if !isnothing(T1)
    if EC.use_kext
      dint1 = load(EC,"dint1")
      R1 = dint1[SP('v'),SP('o')]
    else
      R1 = dfock[SP('v'),SP('o')]
      if !EC.calc_d_vovv
        error("for not use_kext calc_d_vovv has to be True")
      end
      int2 = load(EC,"d_vovv")
      @tensoropt R1[a,i] += int2[a,k,b,c] * T2t[c,b,k,i]
    end
    int2 = load(EC,"d_oovo")
    fov = dfock[SP('o'),SP('v')]
    @tensoropt begin
      R1[a,i] += T2t[a,b,i,j] * fov[j,b]
      R1[a,i] -= int2[k,j,c,i] * T2t[c,a,k,j]
    end
    t1 = print_time(EC,t1,"singles residual",2)
  else
    R1 = nothing
  end

  # <ab|ij>
  if EC.use_kext
    R2 = zeros((length(SP('v')),length(SP('v')),length(SP('o')),length(SP('o'))))
  else
    if !EC.calc_d_vvoo
      error("for not use_kext calc_d_vvoo has to be True")
    end
    R2 = load(EC,"d_vvoo")
  end
  t1 = print_time(EC,t1,"<ab|ij>",2)
  klcd = ints2(EC,"oovv")
  t1 = print_time(EC,t1,"<kl|cd>",2)
  int2 = load(EC,"d_oooo")
  if !dc
    # I_klij = <kl|ij>+<kl|cd>T^ij_cd
    @tensoropt int2[k,l,i,j] += klcd[k,l,c,d] * T2[c,d,i,j]
  end
  # I_klij T^kl_ab
  @tensoropt R2[a,b,i,j] += int2[k,l,i,j] * T2[a,b,k,l]
  t1 = print_time(EC,t1,"I_klij T^kl_ab",2)
  # <kl|cd>\tilde T^ki_ca \tilde T^lj_db
  @tensoropt R2[a,b,i,j] += klcd[k,l,c,d] * T2t[c,a,k,i] * T2t[d,b,l,j]
  t1 = print_time(EC,t1,"<kl|cd> tT^ki_ca tT^lj_db",2)
  if EC.use_kext
    int2 = integ2(EC.fd)
    if ndims(int2) == 4
      if EC.triangular_kext
        trioo = [CartesianIndex(i,j) for j in 1:length(SP('o')) for i in 1:j]
        D2 = calc_D2(EC, T1, T2)[:,:,trioo]
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
        D2 = calc_D2(EC, T1, T2)
        # <pq|rs> D^ij_rs
        @tensoropt R2pq[p,r,i,j] := int2[p,r,q,s] * D2[q,s,i,j]
        D2 = nothing
      end
    else
      # last two indices of integrals are stored as upper triangular 
      tripp = [CartesianIndex(i,j) for j in 1:length(SP(':')) for i in 1:j]
      D2 = calc_D2(EC, T1, T2, true)[tripp,:,:]
      # <pq|rs> D^ij_rs
      @tensoropt rR2pq[p,r,i,j] := int2[p,r,x] * D2[x,i,j]
      D2 = nothing
      # symmetrize R
      @tensoropt R2pq[p,r,i,j] := rR2pq[p,r,i,j] + rR2pq[r,p,j,i]
    end
    R2 += R2pq[SP('v'),SP('v'),:,:]
    if !isnothing(T1)
      @tensoropt begin
        R2[a,b,i,j] -= R2pq[SP('o'),SP('v'),:,:][k,b,i,j] * T1[a,k]
        R2[a,b,i,j] -= R2pq[SP('v'),SP('o'),:,:][a,k,i,j] * T1[b,k]
        R2[a,b,i,j] += R2pq[SP('o'),SP('o'),:,:][k,l,i,j] * T1[a,k] * T1[b,l]
        # singles residual contributions
        R1[a,i] +=  2.0 * R2pq[SP('v'),SP('o'),:,:][a,k,i,k] - R2pq[SP('v'),SP('o'),:,:][a,k,k,i]
        x1[k,i] := 2.0 * R2pq[SP('o'),SP('o'),:,:][k,l,i,l] - R2pq[SP('o'),SP('o'),:,:][k,l,l,i]
        R1[a,i] -= x1[k,i] * T1[a,k]
      end
    end
    x1 = nothing
    R2pq = nothing
    t1 = print_time(EC,t1,"kext",2)
  else
    if !EC.calc_d_vvvv
      error("for not use_kext calc_d_vvvv has to be True")
    end
    int2 = load(EC,"d_vvvv")
    # <ab|cd> T^ij_cd
    @tensoropt R2[a,b,i,j] += int2[a,b,c,d] * T2[c,d,i,j]
    t1 = print_time(EC,t1,"<ab|cd> T^ij_cd",2)
  end
  if !dc
    # <kl|cd> T^kj_ad T^il_cb
    @tensoropt R2[a,b,i,j] += klcd[k,l,c,d] * T2[a,d,k,j] * T2[c,b,i,l]
    t1 = print_time(EC,t1,"<kl|cd> T^kj_ad T^il_cb",2)
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
  t1 = print_time(EC,t1,"xad, xki",2)

  # terms for P(ia;jb)
  @tensoropt begin
    # x_ad T^ij_db
    R2r[a,b,i,j] := xad[a,d] * T2[d,b,i,j]
    # -x_ki T^kj_ab
    R2r[a,b,i,j] -= xki[k,i] * T2[a,b,k,j]
  end
  t1 = print_time(EC,t1,"x_ad T^ij_db -x_ki T^kj_ab",2)
  int2 = load(EC,"d_voov")
  # <ak|ic> \tilde T^kj_cb
  @tensoropt R2r[a,b,i,j] += int2[a,k,i,c] * T2t[c,b,k,j]
  t1 = print_time(EC,t1,"<ak|ic> tT^kj_cb",2)
  if !dc
    # -<kl|cd> T^ki_da (T^lj_cb - T^lj_bc)
    T2t -= T2
    @tensoropt R2r[a,b,i,j] -= klcd[k,l,c,d] * T2[d,a,k,i] * T2t[c,b,l,j]
    t1 = print_time(EC,t1,"-<kl|cd> T^ki_da (T^lj_cb - T^lj_bc)",2)
  end
  int2 = load(EC,"d_vovo")
  @tensoropt begin
    # -<ka|ic> T^kj_cb
    R2r[a,b,i,j] -= int2[a,k,c,i] * T2[c,b,k,j]
    # -<kb|ic> T^kj_ac
    R2r[a,b,i,j] -= int2[b,k,c,i] * T2[a,c,k,j]
    t1 = print_time(EC,t1,"-<ka|ic> T^kj_cb -<kb|ic> T^kj_ac",2)

    R2[a,b,i,j] += R2r[a,b,i,j] + R2r[b,a,j,i]
  end
  t1 = print_time(EC,t1,"P(ia;jb)",2)

  return R1,R2
end

"""
Calculate coupled cluster amplitudes.

If T1 is `nothing` on input, no singles will be calculated.
If dc: calculate distinguishable cluster.
"""
function calc_cc!(EC::ECInfo, T1, T2, dc = false)
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
    R1, R2 = calc_ccsd_resid(EC,T1,T2,dc)
    t1 = print_time(EC,t1,"residual",2)
    NormT2 = calc_doubles_norm(T2)
    NormR2 = calc_doubles_norm(R2)
    Eh = calc_hylleraas(EC,T1,T2,R1,R2)
    T2 += update_doubles(EC,R2)
    if isnothing(T1)
      T2, = perform(diis,[T2],[R2])
      En = 0.0
    else
      NormT1 = calc_singles_norm(T1)
      NormR1 = calc_singles_norm(R1)
      T1 += update_singles(EC,R1)
      T1,T2 = perform(diis,[T1,T2],[R1,R2])
      En = calc_singles_energy(EC, T1)
    end
    En += calc_doubles_energy(EC,T2)
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

end #module