""" coupled-cluster methods """
module CoupledCluster

try
  using MKL
catch
  #println("MKL package not found, using OpenBLAS.")
end
using LinearAlgebra
#BLAS.set_num_threads(1)
using TensorOperations
using Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.FciDump
using ..ElemCo.DIIS
using ..ElemCo.DecompTools
using ..ElemCo.DFCoupledCluster
using ..ElemCo.OrbTools
using ..ElemCo.CCTools

export calc_MP2, calc_UMP2, calc_cc, calc_pertT

include("cc_tests.jl")


"""
    calc_singles_energy(EC::ECInfo, T1; fock_only=false)

  Calculate coupled-cluster closed-shell singles energy.
"""
function calc_singles_energy(EC::ECInfo, T1; fock_only=false)
  SP = EC.space
  ET1 = 0.0
  if length(T1) > 0
    if !fock_only
      @tensoropt ET1 += (2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*ints2(EC,"oovv")[i,j,a,b]
    end
    @tensoropt ET1 += 2.0*T1[a,i] * load(EC,"f_mm")[SP['o'],SP['v']][i,a]
  end
  return ET1
end

"""
    calc_singles_energy(EC::ECInfo, T1a, T1b; fock_only=false)

  Calculate energy for α (T1a) and β (T1b) singles amplitudes.
"""
function calc_singles_energy(EC::ECInfo, T1a, T1b; fock_only=false)
  SP = EC.space
  ET1 = 0.0
  if !fock_only
    if length(T1a) > 0
      @tensoropt ET1 += 0.5*(T1a[a,i]*T1a[b,j]-T1a[b,i]*T1a[a,j])*ints2(EC,"oovv")[i,j,a,b]
    end
    if length(T1b) > 0
      @tensoropt ET1 += 0.5*(T1b[a,i]*T1b[b,j]-T1b[b,i]*T1b[a,j])*ints2(EC,"OOVV")[i,j,a,b]
      if length(T1a) > 0
        @tensoropt ET1 += T1a[a,i]*T1b[b,j]*ints2(EC,"oOvV")[i,j,a,b]
      end
    end
  end
  if length(T1a) > 0
    @tensoropt ET1 += T1a[a,i] * load(EC,"f_mm")[SP['o'],SP['v']][i,a]
  end
  if length(T1b) > 0
    @tensoropt ET1 += T1b[a,i] * load(EC,"f_MM")[SP['O'],SP['V']][i,a]
  end
  return ET1
end

"""
    calc_doubles_energy(EC::ECInfo, T2; fock_only=false)

  Calculate coupled-cluster closed-shell doubles energy.
"""
function calc_doubles_energy(EC::ECInfo, T2)
  @tensoropt ET2 = (2.0*T2[a,b,i,j] - T2[b,a,i,j]) * ints2(EC,"oovv")[i,j,a,b]
  return ET2
end

"""
    calc_doubles_energy(EC::ECInfo, T2a, T2b, T2ab; fock_only=false)

  Calculate energy for αα (T2a), ββ (T2b) and αβ (T2ab) doubles amplitudes.
"""
function calc_doubles_energy(EC::ECInfo, T2a, T2b, T2ab)
  @tensoropt begin
    ET2 = 0.5*T2a[a,b,i,j] * ints2(EC,"oovv")[i,j,a,b]
    ET2 += 0.5*T2b[a,b,i,j] * ints2(EC,"OOVV")[i,j,a,b]
    ET2 += T2ab[a,b,i,j] * ints2(EC,"oOvV")[i,j,a,b]
  end
  return ET2
end

"""
    calc_hylleraas(EC::ECInfo, T1, T2, R1, R2)

  Calculate closed-shell singles and doubles Hylleraas energy
"""
function calc_hylleraas(EC::ECInfo, T1, T2, R1, R2)
  SP = EC.space
  int2 = ints2(EC,"oovv")
  @tensoropt begin
    int2[i,j,a,b] += R2[a,b,i,j]
    ET2 = (2.0*T2[a,b,i,j] - T2[b,a,i,j]) * int2[i,j,a,b]
  end
  if length(T1) > 0
    mo = 'm'
    dfock = load(EC,"df_"*mo*mo)
    fov = dfock[SP['o'],SP['v']] + load(EC,"f_mm")[SP['o'],SP['v']] # undressed part should be with factor two
    @tensoropt ET1 = (fov[i,a] + 2.0 * R1[a,i])*T1[a,i]
    # ET1 = scalar(2.0*(load(EC,"f_mm")[SP['o'],SP['v']][i,a] + R1[a,i])*T1[a,i])
    # ET1 += scalar((2.0*T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j])*int2[i,j,a,b])
    ET2 += ET1
  end
  return ET2
end

"""
    calc_hylleraas4spincase(EC::ECInfo, o1, v1, o2, v2, T1, T2, R1, R2, fov)

  Calculate singles and doubles Hylleraas energy for one spin case.
"""
function calc_hylleraas4spincase(EC::ECInfo, o1, v1, o2, v2, T1, T2, R1, R2, fov)
  SP = EC.space
  int2 = ints2(EC,o1*o2*v1*v2)
  if o1 == o2
    fac = 0.5
  else
    fac = 1.0
  end
  @tensoropt begin
    int2[i,j,a,b] += fac*R2[a,b,i,j]
    ET2 = fac*T2[a,b,i,j] * int2[i,j,a,b]
  end
  if length(T1) > 0
    mo = space4spin('m', isalphaspin(o1,o1))
    dfock = load(EC,"df_"*mo*mo)
    dfov = dfock[SP[o1],SP[v1]] + fov # undressed part should be with factor two
    @tensoropt ET1 = (0.5*dfov[i,a] + R1[a,i])*T1[a,i]
    ET2 += ET1
  end
  return ET2
end

"""
    calc_hylleraas(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab, R1a, R1b, R2a, R2b, R2ab)

  Calculate singles and doubles Hylleraas energy.
"""
function calc_hylleraas(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab, R1a, R1b, R2a, R2b, R2ab)
  SP = EC.space
  Eh = calc_hylleraas4spincase(EC, "ovov"..., T1a, T2a, R1a, R2a, load(EC,"f_mm")[SP['o'],SP['v']])
  if n_occb_orbs(EC) > 0
    Eh += calc_hylleraas4spincase(EC, "OVOV"..., T1b, T2b, R1b, R2b, load(EC,"f_MM")[SP['O'],SP['V']])
    Eh += calc_hylleraas4spincase(EC, "ovOV"..., Float64[], T2ab, Float64[], R2ab, Float64[])
  end
  return Eh
end

""" 
    calc_dressed_ints(EC::ECInfo, T1, T12, o1::Char, v1::Char, o2::Char, v2::Char)

  Dress integrals with singles amplitudes. 

  The singles and orbspaces for first and second electron are `T1`, `o1`, `v1` and `T12, `o2`, `v2`, respectively.
  The integrals from EC.fd are used and dressed integrals are stored as `d_????`.
"""
function calc_dressed_ints(EC::ECInfo, T1, T12, o1::Char, v1::Char, o2::Char, v2::Char)
  t1 = time_ns()
  mixed = (o1 != o2)
  no1, no2 = len_spaces(EC,o1*o2)
  # first make half-transformed integrals
  if EC.options.cc.calc_d_vvvv
    # <a\hat c|bd>
    hd_vvvv = ints2(EC,v1*v2*v1*v2)
    vovv = ints2(EC,v1*o2*v1*v2)
    @tensoropt hd_vvvv[a,c,b,d] -= vovv[a,k,b,d] * T12[c,k]
    vovv = nothing
    save!(EC,"hd_"*v1*v2*v1*v2,hd_vvvv)
    hd_vvvv = nothing
    t1 = print_time(EC,t1,"dress hd_"*v1*v2*v1*v2,3)
  end
  # <ik|j \hat l>
  hd_oooo = ints2(EC,o1*o2*o1*o2)
  ooov = ints2(EC,o1*o2*o1*v2)
  @tensoropt hd_oooo[i,j,k,l] += ooov[i,j,k,d] * T12[d,l]
  ooov = nothing
  t1 = print_time(EC,t1,"dress hd_"*o1*o2*o1*o2,3)
  if EC.options.cc.calc_d_vvoo
    # <a\hat c|j \hat l>
    hd_vvoo = ints2(EC,v1*v2*o1*o2)
    voov = ints2(EC,v1*o2*o1*v2)
    vooo = ints2(EC,v1*o2*o1*o2)
    @tensoropt begin
      vooo[a,k,j,l] += voov[a,k,j,d] * T12[d,l]
      voov = nothing
      hd_vvoo[a,c,j,l] -= vooo[a,k,j,l] * T12[c,k]
      vooo = nothing
    end
    vvov = ints2(EC,v1*v2*o1*v2)
    @tensoropt hd_vvoo[a,c,j,l] += vvov[a,c,j,d] * T12[d,l]
    vvov = nothing
    save!(EC,"hd_"*v1*v2*o1*o2,hd_vvoo)
    hd_vvoo = nothing
    t1 = print_time(EC,t1,"dress hd_"*v1*v2*o1*o2,3)
  end
  # <\hat a k| \hat j l)
  hd_vooo = ints2(EC,v1*o2*o1*o2)
  vovo = ints2(EC,v1*o2*v1*o2)
  @tensoropt hd_vooo[a,k,j,l] -= hd_oooo[i,k,j,l] * T1[a,i]
  if no2 > 0
    @tensoropt hd_vooo[a,k,j,l] += vovo[a,k,b,l] * T1[b,j]
  end
  t1 = print_time(EC,t1,"dress hd_"*v1*o2*o1*o2,3)
  if mixed
    # <k\hat a | l\hat j )
    hd_ovoo = ints2(EC,o1*v2*o1*o2)
    ovov = ints2(EC,o1*v2*o1*v2)
    if no1 > 0 && no2 > 0
      @tensoropt begin
        hd_ovoo[k,a,l,j] -= hd_oooo[k,i,l,j] * T12[a,i]
        hd_ovoo[k,a,l,j] += ovov[k,a,l,b] * T12[b,j]
      end
    end
    t1 = print_time(EC,t1,"dress hd_"*o1*v2*o1*o2,3)
  end
  # some of the fully dressing moved here...
  # <ki\hat|dj>
  d_oovo = ints2(EC,o1*o2*v1*o2)
  oovv = ints2(EC,o1*o2*v1*v2)
  @tensoropt d_oovo[k,i,d,j] += oovv[k,i,d,b] * T12[b,j]
  save!(EC,"d_"*o1*o2*v1*o2,d_oovo)
  t1 = print_time(EC,t1,"dress d_"*o1*o2*v1*o2,3)
  # <ak\hat|jd>
  vovv = ints2(EC,v1*o2*v1*v2)
  d_voov = ints2(EC,v1*o2*o1*v2)
  if mixed
    # <oo|ov>
    oOvV = ints2(EC,o1*o2*v1*v2)
    d_ooov = ints2(EC,o1*o2*o1*v2)
    @tensoropt d_ooov[k,l,j,d] += oOvV[k,l,b,d] * T1[b,j]
    oOvV = nothing
    save!(EC,"d_"*o1*o2*o1*v2,d_ooov)
    t1 = print_time(EC,t1,"dress d_"*o1*o2*o1*v2,3)
    if no1 > 0 && no2 > 0
      @tensoropt begin
        d_voov[a,i,j,d] -= d_ooov[k,i,j,d] * T1[a,k]
        d_voov[a,i,j,d] += vovv[a,i,b,d] * T1[b,j]
      end
    end
    save!(EC,"d_"*v1*o2*o1*v2,d_voov)
  else
    if no1 > 0 && no2 > 0
      @tensoropt begin
        d_voov[a,k,j,d] -= d_oovo[k,i,d,j] * T1[a,i]
        d_voov[a,k,j,d] += vovv[a,k,b,d] * T1[b,j]
      end
    end
    save!(EC,"d_"*v1*o2*o1*v2,d_voov)
  end
  t1 = print_time(EC,t1,"dress d_"*v1*o2*o1*v2,3)
  # finish half-dressing
  # <ak|b \hat l>
  hd_vovo = ints2(EC,v1*o2*v1*o2)
  if no2 > 0
    @tensoropt hd_vovo[a,k,b,l] += vovv[a,k,b,d] * T12[d,l]
  end
  vovv = nothing
  if mixed
    # <k\hat a|dj>
    ovvv = ints2(EC,o1*v2*v1*v2)
    d_ovvo = ints2(EC,o1*v2*v1*o2)
    @tensoropt begin
      d_ovvo[i,A,b,J] -= d_oovo[i,K,b,J] * T12[A,K]
      d_ovvo[i,A,b,J] += ovvv[i,A,b,C] * T12[C,J]
    end
    save!(EC,"d_"*o1*v2*v1*o2,d_ovvo)
    t1 = print_time(EC,t1,"dress d_"*o1*v2*v1*o2,3)

    hd_ovov = ints2(EC,o1*v2*o1*v2)
    @tensoropt hd_ovov[k,a,l,b] += ovvv[k,a,d,b] * T1[d,l]
    ovvv = nothing
  end
  t1 = print_time(EC,t1,"dress hd_"*v1*o2*v1*o2,3)
  if EC.options.cc.calc_d_vvvo
    # <a\hat c|b \hat l>
    hd_vvvo = ints2(EC,v1*v2*v1*o2)
    vvvv = ints2(EC,v1*v2*v1*v2)
    @tensoropt begin
      hd_vvvo[a,c,b,l] -= hd_vovo[a,k,b,l] * T12[c,k]
      hd_vvvo[a,c,b,l] += vvvv[a,c,b,d] * T12[d,l]
    end
    save!(EC,"hd_"*v1*v2*v1*o2,hd_vvvo)
    hd_vvvo = nothing
    if mixed
      hd_vvov = ints2(EC,v1*v2*o1*v2)
      @tensoropt begin
        hd_vvov[a,c,l,b] -= hd_ovov[k,c,l,b] * T1[a,k]
        hd_vvov[a,c,l,b] += vvvv[a,c,d,b] * T1[d,l]
      end
      save!(EC,"hd_"*v1*v2*o1*v2,hd_vvov)
      hd_vvov = nothing
    end
    vvvv = nothing
    t1 = print_time(EC,t1,"dress hd_"*v1*v2*v1*o2,3)
  end

  # fully dressed
  if EC.options.cc.calc_d_vovv
    # <ak\hat|bd>
    d_vovv = ints2(EC,v1*o2*v1*v2)
    @tensoropt d_vovv[a,k,b,d] -= oovv[i,k,b,d] * T1[a,i]
    save!(EC,"d_"*v1*o2*v1*v2,d_vovv)
    t1 = print_time(EC,t1,"dress d_"*v1*o2*v1*v2,3)
    if mixed
      d_vovv = nothing
      d_ovvv = ints2(EC,o1*v2*v1*v2)
      @tensoropt d_ovvv[i,b,a,c] -= oovv[i,j,a,c] * T12[b,j]
      save!(EC,"d_"*o1*v2*v1*v2,d_ovvv)
      t1 = print_time(EC,t1,"dress d_"*o1*v2*v1*v2,3)
    end
  end
  oovv = nothing
  if EC.options.cc.calc_d_vvvv
    # <ab\hat|cd>
    d_vvvv = load(EC,"hd_"*v1*v2*v1*v2)
    if !EC.options.cc.calc_d_vovv
      error("for calc_d_vvvv calc_d_vovv has to be True")
    end
    if !mixed
      @tensoropt d_vvvv[a,c,b,d] -= d_vovv[c,i,d,b] * T1[a,i]
      d_vovv = nothing
    else
      @tensoropt d_vvvv[a,c,b,d] -= d_ovvv[i,c,b,d] * T1[a,i]
      d_ovvv = nothing
    end
    save!(EC,"d_"*v1*v2*v1*v2,d_vvvv)
    d_vvvv = nothing
    t1 = print_time(EC,t1,"dress d_"*v1*v2*v1*v2,3)
  end
  # <ak\hat|bl>
  d_vovo = hd_vovo
  @tensoropt d_vovo[a,k,b,l] -= d_oovo[i,k,b,l] * T1[a,i]
  save!(EC,"d_"*v1*o2*v1*o2,d_vovo)
  hd_vovo = nothing
  d_vovo = nothing
  if mixed
    d_ovov = hd_ovov
    @tensoropt d_ovov[k,a,l,b] -= d_ooov[k,i,l,b] * T12[a,i]
    save!(EC,"d_"*o1*v2*o1*v2,d_ovov)
    hd_ovov = nothing
    d_ovov = nothing
  end
  t1 = print_time(EC,t1,"dress d_"*v1*o2*v1*o2,3)
  # <aj\hat|kl>
  d_vooo = hd_vooo
  if no1 > 0 && no2 > 0
    @tensoropt d_vooo[a,k,j,l] += d_voov[a,k,j,d] * T12[d,l]
  end
  save!(EC,"d_"*v1*o2*o1*o2,d_vooo)
  if mixed
    d_ovoo = hd_ovoo
    @tensoropt d_ovoo[k,a,l,j] += d_ovvo[k,a,d,j] * T1[d,l]
    save!(EC,"d_"*o1*v2*o1*o2,d_ovoo)
  end
  t1 = print_time(EC,t1,"dress d_"*v1*o2*o1*o2,3)
  if EC.options.cc.calc_d_vvvo
    # <ab\hat|cl>
    if !mixed
      d_vvvo = load(EC,"hd_"*v1*v2*v1*o2)
      @tensoropt d_vvvo[a,c,b,l] -= d_voov[c,i,l,b] * T1[a,i]
      save!(EC,"d_"*v1*v2*v1*o2,d_vvvo)
      d_vvvo = nothing
    else
      d_vvvo = load(EC,"hd_"*v1*v2*v1*o2)
      @tensoropt d_vvvo[c,a,b,l] -= d_ovvo[i,a,b,l] * T1[c,i]
      save!(EC,"d_"*v1*v2*v1*o2,d_vvvo)
      d_vvvo = nothing
      d_vvov = load(EC,"hd_"*v1*v2*o1*v2)
      @tensoropt d_vvov[a,c,l,b] -= d_voov[a,i,l,b] * T1[c,i]
      save!(EC,"d_"*v1*v2*o1*v2,d_vvov)
      d_vvov = nothing
    end
    t1 = print_time(EC,t1,"dress d_"*v1*v2*v1*o2,3)
  end
  # <ij\hat|kl>
  d_oooo = hd_oooo
  @tensoropt d_oooo[i,k,j,l] += d_oovo[i,k,b,l] * T1[b,j]
  save!(EC,"d_"*o1*o2*o1*o2,d_oooo)
  t1 = print_time(EC,t1,"dress d_"*o1*o2*o1*o2,3)
  if EC.options.cc.calc_d_vvoo
    if !EC.options.cc.calc_d_vvvo
      error("for calc_d_vvoo calc_d_vvvo has to be True")
    end
    # <ac\hat|jl>
    d_vvoo = load(EC,"hd_"*v1*v2*o1*o2)
    hd_vvvo = load(EC,"hd_"*v1*v2*v1*o2)
    @tensoropt d_vvoo[a,c,j,l] += hd_vvvo[a,c,b,l] * T1[b,j]
    hd_vvvo = nothing
    if !mixed
      @tensoropt d_vvoo[a,c,j,l] -= d_vooo[c,i,l,j] * T1[a,i] 
    else
      @tensoropt d_vvoo[a,c,j,l] -= d_ovoo[i,c,j,l] * T1[a,i] 
    end
    save!(EC,"d_"*v1*v2*o1*o2,d_vvoo)
    d_vvoo = nothing
    t1 = print_time(EC,t1,"dress d_"*v1*v2*o1*o2,3)
  end
end

""" 
    dress_fock_closedshell(EC::ECInfo, T1)

  Dress the fock matrix (closed-shell). The dressed fock matrix is stored as `df_mm`.
"""
function dress_fock_closedshell(EC::ECInfo, T1)
  t1 = time_ns()
  SP = EC.space
  # dress 1-el part
  d_int1 = deepcopy(integ1(EC.fd))
  # display(d_int1[SP['v'],SP['o']])
  dinter = ints1(EC,":v")
  @tensoropt d_int1[:,SP['o']][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[SP['o'],:]
  @tensoropt d_int1[SP['v'],:][b,p] -= dinter[j,p] * T1[b,j]
  # display(d_int1[SP['v'],SP['o']])
  save!(EC,"dh_mm",d_int1)
  t1 = print_time(EC,t1,"dress int1",3)

  # calc dressed fock
  dfock = d_int1
  d_oooo = load(EC,"d_oooo")
  d_vooo = load(EC,"d_vooo")
  d_oovo = load(EC,"d_oovo")
  @tensoropt begin
    foo[i,j] := 2.0*d_oooo[i,k,j,k] - d_oooo[i,k,k,j]
    fvo[a,i] := 2.0*d_vooo[a,k,i,k] - d_vooo[a,k,k,i]
    fov[i,a] := 2.0*d_oovo[i,k,a,k] - d_oovo[k,i,a,k]
  end
  d_vovo = load(EC,"d_vovo")
  @tensoropt fvv[a,b] := 2.0*d_vovo[a,k,b,k]
  d_vovo = nothing
  d_voov = load(EC,"d_voov")
  @tensoropt fvv[a,b] -= d_voov[a,k,k,b]
  dfock[SP['o'],SP['o']] += foo
  dfock[SP['v'],SP['o']] += fvo
  dfock[SP['o'],SP['v']] += fov
  dfock[SP['v'],SP['v']] += fvv

  save!(EC,"df_mm",dfock)
  t1 = print_time(EC,t1,"dress fock",3)
end

""" 
    dress_fock_samespin(EC::ECInfo, T1, o1::Char, v1::Char)

  Dress the fock matrix (same-spin part). 
"""
function dress_fock_samespin(EC::ECInfo, T1, o1::Char, v1::Char)
  t1 = time_ns()
  SP = EC.space
  if isuppercase(o1)
    spin = :β
    no1 = n_occb_orbs(EC)
    mo = 'M'
  else
    spin = :α
    no1 = n_occ_orbs(EC)
    mo = 'm'
  end
  # dress 1-el part
  d_int1 = deepcopy(integ1(EC.fd,spin))
  dinter = ints1(EC,":"*v1)
  @tensoropt d_int1[:,SP[o1]][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[SP[o1],:]
  @tensoropt d_int1[SP[v1],:][b,p] -= dinter[j,p] * T1[b,j]
  save!(EC,"dh_"*mo*mo,d_int1)
  t1 = print_time(EC,t1,"dress int1",3)
  # calc dressed fock
  dfock = d_int1
  d_oooo = load(EC,"d_"*o1*o1*o1*o1)
  d_vooo = load(EC,"d_"*v1*o1*o1*o1)
  d_oovo = load(EC,"d_"*o1*o1*v1*o1)
  @tensoropt begin
    foo[i,j] := d_oooo[i,k,j,k] - d_oooo[i,k,k,j]
    fvo[a,i] := d_vooo[a,k,i,k] - d_vooo[a,k,k,i]
    fov[i,a] := d_oovo[i,k,a,k] - d_oovo[k,i,a,k] 
  end
  d_vovo = load(EC,"d_"*v1*o1*v1*o1)
  @tensoropt fvv[a,b] := d_vovo[a,k,b,k]
  d_vovo = nothing
  if no1 > 0 
    d_voov = load(EC,"d_"*v1*o1*o1*v1)
    @tensoropt fvv[a,b] -= d_voov[a,k,k,b]
    d_voov = nothing
  end
  dfock[SP[o1],SP[o1]] += foo
  dfock[SP[v1],SP[o1]] += fvo
  dfock[SP[o1],SP[v1]] += fov
  dfock[SP[v1],SP[v1]] += fvv
  save!(EC,"df_"*mo*mo,dfock)
  t1 = print_time(EC,t1,"dress fock",3)
end

""" 
    dress_fock_oppositespin(EC::ECInfo)

  Add the dressed opposite-spin part to the dressed Fock matrix. 
"""
function dress_fock_oppositespin(EC::ECInfo)
  t1 = time_ns()
  SP = EC.space
  d_oooo = load(EC,"d_oOoO")
  @tensoropt begin
    foo[i,j] := d_oooo[i,k,j,k]
    fOO[i,j] := d_oooo[k,i,k,j]
  end
  d_oooo = nothing
  d_vooo = load(EC,"d_vOoO")
  @tensoropt fvo[a,i] := d_vooo[a,k,i,k]
  d_vooo = nothing
  d_ovoo = load(EC,"d_oVoO")
  @tensoropt fVO[a,i] := d_ovoo[k,a,k,i]
  d_ovoo = nothing
  d_oovo = load(EC,"d_oOvO")
  @tensoropt fov[i,a] := d_oovo[i,k,a,k]
  d_oovo = nothing
  d_ooov = load(EC,"d_oOoV")
  @tensoropt fOV[i,a] := d_ooov[k,i,k,a]
  d_ooov = nothing
  d_vovo = load(EC,"d_vOvO")
  @tensoropt fvv[a,b] := d_vovo[a,k,b,k]
  d_vovo = nothing
  d_ovov = load(EC,"d_oVoV")
  @tensoropt fVV[a,b] := d_ovov[k,a,k,b]
  d_ovov = nothing

  dfocka = load(EC,"df_mm")
  dfocka[SP['o'],SP['o']] += foo
  dfocka[SP['o'],SP['v']] += fov
  dfocka[SP['v'],SP['o']] += fvo
  dfocka[SP['v'],SP['v']] += fvv
  save!(EC,"df_mm",dfocka)

  dfockb = load(EC,"df_MM")
  dfockb[SP['O'],SP['O']] += fOO
  dfockb[SP['O'],SP['V']] += fOV
  dfockb[SP['V'],SP['O']] += fVO
  dfockb[SP['V'],SP['V']] += fVV
  save!(EC,"df_MM",dfockb)
end

"""
    calc_dressed_ints(EC::ECInfo, T1a, T1b=Float64[])

  Dress integrals with singles.
"""
function calc_dressed_ints(EC::ECInfo, T1a, T1b=Float64[])
  if ndims(T1b) != 2
    calc_dressed_ints(EC,T1a,T1a,"ovov"...)
    dress_fock_closedshell(EC,T1a)
  else
    calc_dressed_ints(EC,T1a,T1a,"ovov"...)
    calc_dressed_ints(EC,T1b,T1b,"OVOV"...)
    calc_dressed_ints(EC,T1a,T1b,"ovOV"...)
    dress_fock_samespin(EC,T1a,"ov"...)
    dress_fock_samespin(EC,T1b,"OV"...)
    dress_fock_oppositespin(EC)
  end
end

"""
    pseudo_dressed_ints(EC::ECInfo, unrestricted=false)

  Save non-dressed integrals in files instead of dressed integrals.
"""
function pseudo_dressed_ints(EC::ECInfo, unrestricted=false)
  #TODO write like in itf with chars as arguments, so three calls for three spin cases...
  t1 = time_ns()
  save!(EC,"d_oovo",ints2(EC,"oovo"))
  save!(EC,"d_voov",ints2(EC,"voov"))
  if EC.options.cc.calc_d_vovv
    save!(EC,"d_vovv",ints2(EC,"vovv"))
  end
  if EC.options.cc.calc_d_vvvv
    save!(EC,"d_vvvv",ints2(EC,"vvvv"))
  end
  save!(EC,"d_vovo",ints2(EC,"vovo"))
  save!(EC,"d_vooo",ints2(EC,"vooo"))
  if EC.options.cc.calc_d_vvvo
    save!(EC,"d_vvvo",ints2(EC,"vvvo"))
  end
  save!(EC,"d_oooo",ints2(EC,"oooo"))
  if EC.options.cc.calc_d_vvoo
    save!(EC,"d_vvoo",ints2(EC,"vvoo"))
  end
  save!(EC,"dh_mm",integ1(EC.fd))
  save!(EC,"df_mm",load(EC,"f_mm"))
  save!(EC,"df_MM",load(EC,"f_MM"))
  t1 = print_time(EC,t1,"pseudo-dressing",3)
  if unrestricted
    save!(EC,"d_OOVO",ints2(EC,"OOVO"))
    save!(EC,"d_VVOO",ints2(EC,"VVOO"))
    save!(EC,"d_VVVV",ints2(EC,"VVVV"))
    save!(EC,"d_OOOO",ints2(EC,"OOOO"))
    save!(EC,"d_VOOO",ints2(EC,"VOOO"))
    save!(EC,"d_VOOV",ints2(EC,"VOOV"))
    save!(EC,"d_VOVO",ints2(EC,"VOVO"))
    save!(EC,"d_VOVV",ints2(EC,"VOVV"))

    save!(EC,"d_oOvO",ints2(EC,"oOvO"))
    save!(EC,"d_oOoV",ints2(EC,"oOoV"))
    save!(EC,"d_vVoO",ints2(EC,"vVoO"))
    save!(EC,"d_vVvV",ints2(EC,"vVvV"))
    save!(EC,"d_oOoO",ints2(EC,"oOoO"))
    # save!(EC,"d_voov",ints2(EC,"voov"))
    save!(EC,"d_oVvO",ints2(EC,"oVvO"))
    save!(EC,"d_vOoV",ints2(EC,"vOoV"))
    #vovo
    save!(EC,"d_vOvO",ints2(EC,"vOvO"))
    save!(EC,"d_oVoV",ints2(EC,"oVoV"))
    save!(EC,"d_vOoO",ints2(EC,"vOoO"))
    save!(EC,"d_oVoO",ints2(EC,"oVoO"))
    save!(EC,"d_vOvV",ints2(EC,"vOvV"))
    save!(EC,"d_oVvV",ints2(EC,"oVvV"))
    save!(EC,"dh_MM",integ1(EC.fd))
  end
end

""" 
    calc_MP2(EC::ECInfo, addsingles=true)

  Calculate closed-shell MP2 energy and amplitudes. 
  The amplitudes are stored in `T_vvoo` file.
  If `addsingles`: singles are also calculated and stored in `T_vo` file.
  Return EMp2 
"""
function calc_MP2(EC::ECInfo, addsingles=true)
  T2 = update_doubles(EC,ints2(EC,"vvoo"), use_shift=false)
  EMp2 = calc_doubles_energy(EC,T2)
  save!(EC, "T_vvoo", T2)
  if addsingles
    ϵo, ϵv = orbital_energies(EC)
    T1 = update_singles(load(EC,"f_mm")[EC.space['v'],EC.space['o']], ϵo, ϵv, 0.0)
    EMp2 += calc_singles_energy(EC,T1,fock_only=true)
    save!(EC, "T_vo", T1)
  end
  return EMp2
end

""" 
    calc_UMP2(EC::ECInfo, addsingles=true)

  Calculate unrestricted MP2 energy and amplitudes. 
  The amplitudes are stored in `T_vvoo`, `T_VVOO`, and `T_vVoO` files.
  If `addsingles`: singles are also calculated and stored in `T_vo` and `T_VO` files.
  Return EMp2
"""
function calc_UMP2(EC::ECInfo, addsingles=true)
  SP = EC.space
  T2a = update_doubles(EC,ints2(EC,"vvoo"), spincase=:α, antisymmetrize = true, use_shift=false)
  T2b = update_doubles(EC,ints2(EC,"VVOO"), spincase=:β, antisymmetrize = true, use_shift=false)
  T2ab = update_doubles(EC,ints2(EC,"vVoO"), spincase=:αβ, use_shift=false)
  EMp2 = calc_doubles_energy(EC,T2a,T2b,T2ab)
  save!(EC, "T_vvoo", T2a)
  save!(EC, "T_VVOO", T2b)
  save!(EC, "T_vVoO", T2ab)
  if addsingles
    T1a = update_singles(EC,load(EC,"f_mm")[SP['v'],SP['o']], spincase=:α, use_shift=false)
    T1b = update_singles(EC,load(EC,"f_MM")[SP['V'],SP['O']], spincase=:β, use_shift=false)
    EMp2 += calc_singles_energy(EC, T1a, T1b, fock_only = true)
    save!(EC, "T_vo", T1a)
    save!(EC, "T_VO", T1b)
  end
  return EMp2
end

""" 
    calc_D2(EC::ECInfo, T1, T2, scalepp = false)

  Calculate ``D^{ij}_{pq} = T^{ij}_{cd} + T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl}``.
  Return as D[pqij] 

  If `scalepp`: D[ppij] elements are scaled by 0.5 (for triangular summation).
"""
function calc_D2(EC::ECInfo, T1, T2, scalepp=false)
  SP = EC.space
  norb = n_orbs(EC)
  nocc = n_occ_orbs(EC)
  if length(T1) > 0
    D2 = Array{Float64}(undef,norb,norb,nocc,nocc)
    # D2 = zeros(norb,norb,nocc,nocc)
  else
    D2 = zeros(norb,norb,nocc,nocc)
  end
  @tensoropt begin
    D2[SP['v'],SP['v'],:,:][a,b,i,j] = T2[a,b,i,j] 
    D2[SP['o'],SP['o'],:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k]
  end
  if length(T1) > 0
    @tensoropt begin
      D2[SP['v'],SP['v'],:,:][a,b,i,j] += T1[a,i] * T1[b,j]
      D2[SP['o'],SP['v'],:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
      D2[SP['v'],SP['o'],:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1[a,k]
    end
  end
  if scalepp
    diagindx = [CartesianIndex(i,i) for i in 1:norb]
    D2[diagindx,:,:] *= 0.5
  end
  return D2
end

""" 
    calc_D2a(EC::ECInfo, T1a, T2a)

  Calculate ``^{αα}D^{ij}_{pq} = T^{ij}_{cd} + P_{ij}(T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl})``
  with ``P_{ij} X_{ij} = X_{ij} - X_{ji}``.
  Return as D[pqij] 
"""
function calc_D2a(EC::ECInfo, T1a, T2a)
  SP = EC.space
  norb = n_orbs(EC)
  nocc = n_occ_orbs(EC)
  if length(T1a) > 0
    D2a = Array{Float64}(undef,norb,norb,nocc,nocc)
  else
    D2a = zeros(norb,norb,nocc,nocc)
  end
  @tensoropt begin
    D2a[SP['v'],SP['v'],:,:][a,b,i,j] = T2a[a,b,i,j] 
    D2a[SP['o'],SP['o'],:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k] - Matrix(I,nocc,nocc)[k,j] * Matrix(I,nocc,nocc)[l,i]
  end
  if length(T1a) > 0
    @tensoropt begin
      D2a[SP['v'],SP['v'],:,:][a,b,i,j] += T1a[a,i] * T1a[b,j] - T1a[b,i] * T1a[a,j]
      D2a[SP['o'],SP['v'],:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1a[a,k] - Matrix(I,nocc,nocc)[k,j] * T1a[a,i]
      D2a[SP['v'],SP['o'],:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1a[a,k] - Matrix(I,nocc,nocc)[k,j] * T1a[a,i]
    end
  end
  return D2a
end

""" 
    calc_D2b(EC::ECInfo, T1b, T2b)

  Calculate ^{ββ}D^{ij}_{pq} = T^{ij}_{cd} + P_{ij}(T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl})
  with P_{ij} X_{ij} = X_{ij} - X_{ji}.
  Return as D[pqij] 
"""
function calc_D2b(EC::ECInfo, T1b, T2b)
  SP = EC.space
  norb = n_orbs(EC)
  nocc = n_occb_orbs(EC)
  if length(T1b) > 0
    D2b = Array{Float64}(undef,norb,norb,nocc,nocc)
  else
    D2b = zeros(norb,norb,nocc,nocc)
  end
  @tensoropt begin
    D2b[SP['V'],SP['V'],:,:][a,b,i,j] = T2b[a,b,i,j] 
    D2b[SP['O'],SP['O'],:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k] - Matrix(I,nocc,nocc)[k,j] * Matrix(I,nocc,nocc)[l,i]
  end
  if length(T1b) > 0
    @tensoropt begin
      D2b[SP['V'],SP['V'],:,:][a,b,i,j] += T1b[a,i] * T1b[b,j] - T1b[b,i] * T1b[a,j]
      D2b[SP['O'],SP['V'],:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1b[a,k] - Matrix(I,nocc,nocc)[k,j] * T1b[a,i]
      D2b[SP['V'],SP['O'],:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1b[a,k] - Matrix(I,nocc,nocc)[k,j] * T1b[a,i]
    end
  end
  return D2b
end

""" 
    calc_D2ab(EC::ECInfo, T1a, T1b, T2ab, scalepp=false)

  Calculate ^{αβ}D^{ij}_{pq} = T^{ij}_{cd} + T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl}
  Return as D[pqij] 

  If `scalepp`: D[ppij] elements are scaled by 0.5 (for triangular summation)
"""
function calc_D2ab(EC::ECInfo, T1a, T1b, T2ab, scalepp=false)
  SP = EC.space
  norb = n_orbs(EC)
  nocca = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  if length(T1a) > 0
    D2ab = Array{Float64}(undef,norb,norb,nocca,noccb)
  else
    D2ab = zeros(norb,norb,nocca,noccb)
  end
  @tensoropt begin
    D2ab[SP['v'],SP['V'],:,:][a,B,i,J] = T2ab[a,B,i,J] 
    D2ab[SP['o'],SP['O'],:,:][i,k,j,l] = Matrix(I,nocca,nocca)[i,j] * Matrix(I,noccb,noccb)[l,k]
  end
  if length(T1a) > 0
    @tensoropt begin
      D2ab[SP['v'],SP['V'],:,:][a,b,i,j] += T1a[a,i] * T1b[b,j]
      D2ab[SP['o'],SP['V'],:,:][j,a,i,k] = Matrix(I,nocca,nocca)[i,j] * T1b[a,k]
      D2ab[SP['v'],SP['O'],:,:][a,j,k,i] = Matrix(I,noccb,noccb)[i,j] * T1a[a,k]
    end
  end
  if scalepp
    diagindx = [CartesianIndex(i,i) for i in 1:norb]
    D2ab[diagindx,:,:] *= 0.5
  end
  return D2ab
end

"""
    calc_ccsd_resid(EC::ECInfo, T1, T2, dc)

  Calculate CCSD or DCSD closed-shell residual.
"""
function calc_ccsd_resid(EC::ECInfo, T1, T2; dc = false, tworef = false, fixref = false)
  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  norb = n_orbs(EC)
  if length(T1) > 0
    calc_dressed_ints(EC,T1)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC)
  end
  @tensor T2t[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  dfock = load(EC,"df_mm")
  if length(T1) > 0
    if EC.options.cc.use_kext
      dint1 = load(EC,"dh_mm")
      R1 = dint1[SP['v'],SP['o']]
    else
      R1 = dfock[SP['v'],SP['o']]
      if !EC.options.cc.calc_d_vovv
        error("for not use_kext calc_d_vovv has to be True")
      end
      int2 = load(EC,"d_vovv")
      @tensoropt R1[a,i] += int2[a,k,b,c] * T2t[c,b,k,i]
    end
    int2 = load(EC,"d_oovo")
    fov = dfock[SP['o'],SP['v']]
    @tensoropt begin
      R1[a,i] += T2t[a,b,i,j] * fov[j,b]
      R1[a,i] -= int2[k,j,c,i] * T2t[c,a,k,j]
    end
    t1 = print_time(EC,t1,"singles residual",2)
  else
    R1 = Float64[]
  end

  # <ab|ij>
  if EC.options.cc.use_kext
    R2 = zeros(nvirt,nvirt,nocc,nocc)
  else
    if !EC.options.cc.calc_d_vvoo
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
  if EC.options.cc.use_kext
    int2 = integ2(EC.fd)
    if ndims(int2) == 4
      if EC.options.cc.triangular_kext
        trioo = [CartesianIndex(i,j) for j in 1:nocc for i in 1:j]
        D2 = calc_D2(EC, T1, T2)[:,:,trioo]
        # <pq|rs> D^ij_rs
        @tensoropt R2pqx[p,r,x] := int2[p,r,q,s] * D2[q,s,x]
        D2 = nothing
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
      tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
      D2 = calc_D2(EC, T1, T2, true)[tripp,:,:]
      # <pq|rs> D^ij_rs
      @tensoropt rR2pq[p,r,i,j] := int2[p,r,x] * D2[x,i,j]
      D2 = nothing
      # symmetrize R
      @tensoropt R2pq[p,r,i,j] := rR2pq[p,r,i,j] + rR2pq[r,p,j,i]
    end
    R2 += R2pq[SP['v'],SP['v'],:,:]
    if length(T1) > 0
      @tensoropt begin
        R2[a,b,i,j] -= R2pq[SP['o'],SP['v'],:,:][k,b,i,j] * T1[a,k]
        R2[a,b,i,j] -= R2pq[SP['v'],SP['o'],:,:][a,k,i,j] * T1[b,k]
        R2[a,b,i,j] += R2pq[SP['o'],SP['o'],:,:][k,l,i,j] * T1[a,k] * T1[b,l]
        # singles residual contributions
        R1[a,i] +=  2.0 * R2pq[SP['v'],SP['o'],:,:][a,k,i,k] - R2pq[SP['v'],SP['o'],:,:][a,k,k,i]
        x1[k,i] := 2.0 * R2pq[SP['o'],SP['o'],:,:][k,l,i,l] - R2pq[SP['o'],SP['o'],:,:][k,l,l,i]
        R1[a,i] -= x1[k,i] * T1[a,k]
      end
    end
    x1 = nothing
    R2pq = nothing
    t1 = print_time(EC,t1,"kext",2)
  else
    if !EC.options.cc.calc_d_vvvv
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
  xad = dfock[SP['v'],SP['v']]
  xki = dfock[SP['o'],SP['o']]
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
    @notensor t1 = print_time(EC,t1,"-<ka|ic> T^kj_cb -<kb|ic> T^kj_ac",2)

    R2[a,b,i,j] += R2r[a,b,i,j] + R2r[b,a,j,i]
  end
  t1 = print_time(EC,t1,"P(ia;jb)",2)

  return R1,R2
end

"""
    calc_pertT(EC::ECInfo; save_t3=false)

  Calculate (T) correction for closed-shell CCSD.

  Return ( (T)-energy, [T]-energy))
"""
function calc_pertT(EC::ECInfo; save_t3=false)
  T1 = load(EC,"T_vo")
  T2 = load(EC,"T_vvoo")
  # <ab|ck>
  abck = ints2(EC,"vvvo")
  # <ia|jk>
  iajk = ints2(EC,"ovoo")
  # <ij|ab>
  ijab = ints2(EC,"oovv")
  nocc = n_occ_orbs(EC)
  nvir = n_virt_orbs(EC)
  ϵo, ϵv = orbital_energies(EC)
  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  if save_t3
    t3file, T3 = newmmap(EC,"T_vvvooo",Float64,(nvir,nvir,nvir,uppertriangular(nocc,nocc,nocc)))
  end
  for k = 1:nocc 
    for j = 1:k
      prefac = (j == k) ? 1.0 : 2.0
      for i = 1:j
        fac = prefac 
        if i == j 
          if j == k
            continue
          end 
          fac = 1.0
        end
        @tensoropt begin
          Kijk[a,b,c] := T2[:,:,i,j][a,d] * abck[:,:,:,k][d,c,b]
          Kijk[a,b,c] += T2[:,:,j,i][b,d] * abck[:,:,:,k][d,c,a]
          Kijk[a,b,c] += T2[:,:,i,k][a,d] * abck[:,:,:,j][d,b,c]
          Kijk[a,b,c] += T2[:,:,k,i][c,d] * abck[:,:,:,j][d,b,a]
          Kijk[a,b,c] += T2[:,:,j,k][b,d] * abck[:,:,:,i][d,a,c]
          Kijk[a,b,c] += T2[:,:,k,j][c,d] * abck[:,:,:,i][d,a,b]

          Kijk[a,b,c] -= T2[:,:,:,i][b,a,l] * iajk[:,:,j,k][l,c]
          Kijk[a,b,c] -= T2[:,:,:,j][a,b,l] * iajk[:,:,i,k][l,c]
          Kijk[a,b,c] -= T2[:,:,:,i][c,a,l] * iajk[:,:,k,j][l,b]
          Kijk[a,b,c] -= T2[:,:,:,k][a,c,l] * iajk[:,:,i,j][l,b]
          Kijk[a,b,c] -= T2[:,:,:,j][c,b,l] * iajk[:,:,k,i][l,a]
          Kijk[a,b,c] -= T2[:,:,:,k][b,c,l] * iajk[:,:,j,i][l,a]
        end
        if save_t3
          ijk = uppertriangular(i,j,k)
          T3[:,:,:,ijk] = Kijk
          for abc ∈ CartesianIndices(Kijk)
            a,b,c = Tuple(abc)
            T3[abc,ijk] /= ϵo[i] + ϵo[j] + ϵo[k] - ϵv[a] - ϵv[b] - ϵv[c]
          end
        end
        @tensoropt  X[a,b,c] := 4.0*Kijk[a,b,c] - 2.0*Kijk[a,c,b] - 2.0*Kijk[c,b,a] - 2.0*Kijk[b,a,c] + Kijk[c,a,b] + Kijk[b,c,a]
        for abc ∈ CartesianIndices(X)
          a,b,c = Tuple(abc)
          X[abc] /= ϵo[i] + ϵo[j] + ϵo[k] - ϵv[a] - ϵv[b] - ϵv[c]
        end

        @tensoropt Enb3 += fac * Kijk[a,b,c] * X[a,b,c]
      
        # julia 1.9 r1: cannot use @tensoropt begin/end here, since 
        # IntX[:,j] overwrites IntX[:,i] if j == i
        @tensoropt IntX[:,i][a] += fac * X[a,b,c] * ijab[j,k,:,:][b,c]
        @tensoropt IntX[:,j][b] += fac * X[a,b,c] * ijab[i,k,:,:][a,c]
        @tensoropt IntX[:,k][c] += fac * X[a,b,c] * ijab[i,j,:,:][a,b]
      end 
    end
  end
  if save_t3
    closemmap(EC,t3file,T3)
  end
  # singles contribution
  @tensoropt En3 = T1[a,i] * IntX[a,i]
  En3 += Enb3
  return En3, Enb3
end

"""
    calc_ccsd_resid(EC::ECInfo, T1a,T1b,T2a,T2b,T2ab,dc)

  Calculate UCCSD or UDCSD residual.
"""
function calc_ccsd_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab; dc = false, tworef = false, fixref = false)
  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nvirtb = n_virtb_orbs(EC)
  norb = n_orbs(EC)
  linearized::Bool = false
  if ndims(T1a) == 2
    if !EC.options.cc.use_kext
      error("open-shell CCSD only implemented with kext")
    end
    calc_dressed_ints(EC,T1a,T1b)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC,true)
  end

  if tworef
    morba, norbb, morbb, norba = active_orbitals(EC)
    T2ab[norba,morbb,morba,norbb] = 0
  end

  R1a = Float64[]
  R1b = Float64[]

  dfock = load(EC,"df_mm")
  dfockb = load(EC,"df_MM")

  fij = dfock[SP['o'],SP['o']]
  fab = dfock[SP['v'],SP['v']]
  fIJ = dfockb[SP['O'],SP['O']]
  fAB = dfockb[SP['V'],SP['V']]

  if length(T1a) > 0
    if EC.options.cc.use_kext
      dint1a = load(EC,"dh_mm")
      R1a = dint1a[SP['v'],SP['o']]
      dint1b = load(EC,"dh_MM")
      R1b = dint1b[SP['V'],SP['O']]
    else
      fai = dfock[SP['v'],SP['o']]
      fAI = dfockb[SP['V'],SP['O']]
      @tensoropt begin
        R1a[a,i] :=  fai[a,i]
        R1b[a,i] :=  fAI[a,i]
      end
      d_vovv = load(EC,"d_vovv")
      @tensoropt R1a[a,i] += d_vovv[a,k,b,d] * T2a[b,d,i,k]
      d_vovv = nothing
      d_VOVV = load(EC,"d_VOVV")
      @tensoropt R1b[A,I] += d_VOVV[A,K,B,D] * T2b[B,D,I,K]
      d_VOVV = nothing
      d_vOvV = load(EC,"d_vOvV")
      @tensoropt R1a[a,i] += d_vOvV[a,K,b,D] * T2ab[b,D,i,K]
      d_vOvV = nothing
      d_oVvV = load(EC,"d_oVvV")
      @tensoropt R1b[A,I] += d_oVvV[k,A,d,B] * T2ab[d,B,k,I]
      d_oVvV = nothing
    end
    fia = dfock[SP['o'],SP['v']]
    fIA = dfockb[SP['O'],SP['V']]
    @tensoropt begin
      R1a[a,i] += fia[j,b] * T2a[a,b,i,j]
      R1b[A,I] += fIA[J,B] * T2b[A,B,I,J]
      R1a[a,i] += fIA[J,B] * T2ab[a,B,i,J]
      R1b[A,I] += fia[j,b] * T2ab[b,A,j,I]
    end
    if n_occ_orbs(EC) > 0 
      d_oovo = load(EC,"d_oovo")
      @tensoropt R1a[a,i] -= d_oovo[k,j,d,i] * T2a[a,d,j,k]
      d_oovo = nothing
    end
    if n_occb_orbs(EC) > 0
      d_OOVO = load(EC,"d_OOVO")
      @tensoropt R1b[A,I] -= d_OOVO[K,J,D,I] * T2b[A,D,J,K]
      d_OOVO = nothing
      d_oOoV = load(EC,"d_oOoV")
      @tensoropt R1a[a,i] -= d_oOoV[j,K,i,D] * T2ab[a,D,j,K]
      d_oOoV = nothing
      d_oOvO = load(EC,"d_oOvO")
      @tensoropt R1b[A,I] -= d_oOvO[k,J,d,I] * T2ab[d,A,k,J]
    end
  end

  #driver terms
  if EC.options.cc.use_kext
    R2a = zeros(nvirt, nvirt, nocc, nocc)
    R2b = zeros(nvirtb, nvirtb, noccb, noccb)
    R2ab = zeros(nvirt, nvirtb, nocc, noccb)
  else
    d_vvoo = load(EC,"d_vvoo")
    R2a = deepcopy(d_vvoo)
    @tensoropt R2a[a,b,i,j] -= d_vvoo[b,a,i,j]
    d_vvoo = nothing
    d_VVOO = load(EC,"d_VVOO")
    R2b = deepcopy(d_VVOO)
    @tensoropt R2b[A,B,I,J] -= d_VVOO[B,A,I,J]
    d_VVOO = nothing
    R2ab = load(EC,"d_vVoO")
  end
  
  #ladder terms
  if EC.options.cc.use_kext
    # last two indices of integrals (apart from αβ) are stored as upper triangular 
    tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
    if(EC.fd.uhf)
      # αα
      int2a = integ2(EC.fd,:α)
      D2a = calc_D2a(EC, T1a, T2a)[tripp,:,:]
      @tensoropt rR2pqa[p,r,i,j] := int2a[p,r,x] * D2a[x,i,j]
      D2a = nothing
      int2a = nothing
      # symmetrize R
      @tensoropt R2pqa[p,r,i,j] := rR2pqa[p,r,i,j] + rR2pqa[r,p,j,i]
      rR2pqa = nothing
      R2a += R2pqa[SP['v'],SP['v'],:,:]
      if n_occb_orbs(EC) > 0
        # ββ
        int2b = integ2(EC.fd,:β)
        D2b = calc_D2b(EC, T1b, T2b)[tripp,:,:]
        @tensoropt rR2pqb[p,r,i,j] := int2b[p,r,x] * D2b[x,i,j]
        D2b = nothing
        int2b = nothing
        # symmetrize R
        @tensoropt R2pqb[p,r,i,j] := rR2pqb[p,r,i,j] + rR2pqb[r,p,j,i]
        rR2pqb = nothing
        R2b += R2pqb[SP['V'],SP['V'],:,:]
        # αβ
        int2ab = integ2(EC.fd,:αβ)
        D2ab = calc_D2ab(EC, T1a, T1b, T2ab)
        @tensoropt R2pqab[p,r,i,j] := int2ab[p,r,q,s] * D2ab[q,s,i,j]
        D2ab = nothing
        int2ab = nothing
        R2ab += R2pqab[SP['v'],SP['V'],:,:]
      end
    else
      int2 = integ2(EC.fd)
      # αα
      D2a = calc_D2a(EC, T1a, T2a)[tripp,:,:]
      @tensoropt rR2pqa[p,r,i,j] := int2[p,r,x] * D2a[x,i,j]
      D2a = nothing
      # symmetrize R
      @tensoropt R2pqa[p,r,i,j] := rR2pqa[p,r,i,j] + rR2pqa[r,p,j,i]
      rR2pqa = nothing
      R2a += R2pqa[SP['v'],SP['v'],:,:]
      if n_occb_orbs(EC) > 0
        # ββ
        D2b = calc_D2b(EC, T1b, T2b)[tripp,:,:]
        @tensoropt rR2pqb[p,r,i,j] := int2[p,r,x] * D2b[x,i,j]
        D2b = nothing
        # symmetrize R
        @tensoropt R2pqb[p,r,i,j] := rR2pqb[p,r,i,j] + rR2pqb[r,p,j,i]
        rR2pqb = nothing
        R2b += R2pqb[SP['V'],SP['V'],:,:]
        # αβ
        D2ab_full = calc_D2ab(EC, T1a, T1b, T2ab, true)
        D2ab = D2ab_full[tripp,:,:] 
        D2abT = permutedims(D2ab_full,(2,1,4,3))[tripp,:,:]
        D2ab_full = nothing
        @tensoropt R2pqab[p,r,i,j] := int2[p,r,x] * D2ab[x,i,j]
        @tensoropt R2pqab[p,r,i,j] += int2[r,p,x] * D2abT[x,j,i]
        D2ab = nothing
        D2abT = nothing
        R2ab += R2pqab[SP['v'],SP['V'],:,:]
      end
    end
    if length(T1a) > 0
      @tensoropt begin
        R2a[a,b,i,j] -= R2pqa[SP['o'],SP['v'],:,:][k,b,i,j] * T1a[a,k]
        R2a[a,b,i,j] -= R2pqa[SP['v'],SP['o'],:,:][a,k,i,j] * T1a[b,k]
        R2a[a,b,i,j] += R2pqa[SP['o'],SP['o'],:,:][k,l,i,j] * T1a[a,k] * T1a[b,l]
        # singles residual contributions
        R1a[a,i] +=  R2pqa[SP['v'],SP['o'],:,:][a,k,i,k] 
        x1a[k,i] :=  R2pqa[SP['o'],SP['o'],:,:][k,l,i,l]
        R1a[a,i] -= x1a[k,i] * T1a[a,k]
      end
    end
    if length(T1b) > 0
      @tensoropt begin
        R2b[a,b,i,j] -= R2pqb[SP['O'],SP['V'],:,:][k,b,i,j] * T1b[a,k]
        R2b[a,b,i,j] -= R2pqb[SP['V'],SP['O'],:,:][a,k,i,j] * T1b[b,k]
        R2b[a,b,i,j] += R2pqb[SP['O'],SP['O'],:,:][k,l,i,j] * T1b[a,k] * T1b[b,l]
        # singles residual contributions
        R1b[a,i] += R2pqb[SP['V'],SP['O'],:,:][a,k,i,k]
        x1b[k,i] := R2pqb[SP['O'],SP['O'],:,:][k,l,i,l]
        R1b[a,i] -= x1b[k,i] * T1b[a,k]
      end
    end
    if n_occ_orbs(EC) > 0 && n_occb_orbs(EC) > 0 && length(T1a) > 0
      @tensoropt begin
        R2ab[a,b,i,j] -= R2pqab[SP['o'],SP['V'],:,:][k,b,i,j] * T1a[a,k]
        R2ab[a,b,i,j] -= R2pqab[SP['v'],SP['O'],:,:][a,k,i,j] * T1b[b,k]
        R2ab[a,b,i,j] += R2pqab[SP['o'],SP['O'],:,:][k,l,i,j] * T1a[a,k] * T1b[b,l]
        R1a[a,i] += R2pqab[SP['v'],SP['O'],:,:][a,k,i,k] 
        x1a1[k,i] := R2pqab[SP['o'],SP['O'],:,:][k,l,i,l]
        R1a[a,i] -= x1a1[k,i] * T1a[a,k]
        R1b[a,i] += R2pqab[SP['o'],SP['V'],:,:][k,a,k,i] 
        x1b1[k,i] := R2pqab[SP['o'],SP['O'],:,:][l,k,l,i]
        R1b[a,i] -= x1b1[k,i] * T1b[a,k]
      end
    end
    (R2pqa, R2pqb, R2pqab) = (nothing, nothing, nothing)
    (x1a, x1b, x1ab) = (nothing, nothing, nothing)
  else
    d_vvvv = load(EC,"d_vvvv")
    @tensoropt R2a[a,b,i,j] += d_vvvv[a,b,c,d] * T2a[c,d,i,j]
    d_vvvv = nothing
    d_VVVV = load(EC,"d_VVVV")
    @tensoropt R2b[A,B,I,J] += d_VVVV[A,B,C,D] * T2b[C,D,I,J]
    d_VVVV = nothing
    d_vVvV = load(EC,"d_vVvV")
    @tensoropt R2ab[a,B,i,J] += d_vVvV[a,B,c,D] * T2ab[c,D,i,J]
    d_vVvV = nothing
  end

  @tensoropt begin
    xij[i,j] := fij[i,j]
    xIJ[i,j] := fIJ[i,j]
    xab[i,j] := fab[i,j]
    xAB[i,j] := fAB[i,j]
  end
  x_klij = load(EC,"d_oooo")
  x_KLIJ = load(EC,"d_OOOO")
  x_kLiJ = load(EC,"d_oOoO")
  if !linearized
    dcfac = dc ? 0.5 : 1.0
    oovv = ints2(EC,"oovv")
    @tensoropt begin
      xij[i,j] += dcfac * oovv[i,k,b,d] * T2a[b,d,j,k]
      xab[a,b] -= dcfac * oovv[i,k,b,d] * T2a[a,d,i,k]
    end
    !dc && @tensoropt x_klij[k,l,i,j] += 0.5 * oovv[k,l,c,d] * T2a[c,d,i,j]
    if n_occb_orbs(EC) > 0
      @tensoropt x_dAlI[d,A,l,I] := oovv[k,l,c,d] * T2ab[c,A,k,I]
      !dc && @tensoropt x_dAlI[d,A,l,I] -= oovv[k,l,d,c] * T2ab[c,A,k,I]
      @tensoropt begin
        rR2b[A,B,I,J] := x_dAlI[d,A,l,I] * T2ab[d,B,l,J]
        R2b[A,B,I,J] += rR2b[A,B,I,J] - rR2b[A,B,J,I]
      end
      x_dAlI, rR2b = nothing, nothing
    end
    @tensoropt x_adil[a,d,i,l] := 0.5 * oovv[k,l,c,d] *  T2a[a,c,i,k]
    !dc && @tensoropt x_adil[a,d,i,l] -= 0.5 * oovv[k,l,d,c] * T2a[a,c,i,k]
    @tensoropt R2ab[a,B,i,J] += x_adil[a,d,i,l] * T2ab[d,B,l,J]
    oovv = nothing
    if n_occb_orbs(EC) > 0
      OOVV = ints2(EC,"OOVV")
      @tensoropt begin
        xIJ[I,J] += dcfac * OOVV[I,K,B,D] * T2b[B,D,J,K]
        xAB[A,B] -= dcfac * OOVV[I,K,B,D] * T2b[A,D,I,K]
      end
      !dc && @tensoropt x_KLIJ[K,L,I,J] += 0.5 * OOVV[K,L,C,D] * T2b[C,D,I,J]
      @tensoropt x_ADIL[A,D,I,L] := 0.5 * OOVV[K,L,C,D] * T2b[A,C,I,K]
      !dc && @tensoropt x_ADIL[A,D,I,L] -= 0.5 * OOVV[K,L,D,C] * T2b[A,C,I,K]
      @tensoropt R2ab[b,A,j,I] += 2.0 * x_ADIL[A,D,I,L] * T2ab[b,D,j,L]
      @tensoropt x_vVoO[a,D,i,L] := OOVV[K,L,C,D] * T2ab[a,C,i,K]
      !dc && @tensoropt x_vVoO[a,D,i,L] -= OOVV[K,L,D,C] * T2ab[a,C,i,K]
      @tensoropt begin      
        rR2a[a,b,i,j] := x_vVoO[a,D,i,L] * T2ab[b,D,j,L]
        R2a[a,b,i,j] += rR2a[a,b,i,j] - rR2a[a,b,j,i] 
      end
      OOVV, x_vVoO, rR2a = nothing, nothing, nothing
      oOvV = ints2(EC,"oOvV")
      @tensoropt begin
        xij[i,j] += dcfac * oOvV[i,K,b,D] * T2ab[b,D,j,K]
        xab[a,b] -= dcfac * oOvV[i,K,b,D] * T2ab[a,D,i,K]
        xIJ[I,J] += dcfac * oOvV[k,I,d,B] * T2ab[d,B,k,J]
        xAB[A,B] -= dcfac * oOvV[k,I,d,B] * T2ab[d,A,k,I]
      end
      !dc && @tensoropt x_kLiJ[k,L,i,J] += oOvV[k,L,c,D] * T2ab[c,D,i,J]
      @tensoropt begin
        x_adil[a,d,i,l] += oOvV[l,K,d,C] * T2ab[a,C,i,K]
        R2ab[a,B,i,J] += x_adil[a,d,i,l] * T2ab[d,B,l,J]
        rR2a[a,b,i,j] := x_adil[a,d,i,l] *  T2a[b,d,j,l]
        R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i] - rR2a[a,b,j,i] - rR2a[b,a,i,j]
      end
      x_adil, rR2a = nothing, nothing
      @tensoropt begin
        x_ADIL[A,D,I,L] += oOvV[k,L,c,D] * T2ab[c,A,k,I]
        rR2b[A,B,I,J] := x_ADIL[A,D,I,L] * T2b[B,D,J,L]
        R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I] - rR2b[A,B,J,I] - rR2b[B,A,I,J]
      end 
      X_ADIL, rR2b = nothing, nothing
      @tensoropt begin
        x_vVoO[a,D,i,L] := oOvV[k,L,c,D] * T2a[a,c,i,k]
        R2ab[a,B,i,J] += x_vVoO[a,D,i,L] * T2b[B,D,J,L]
      end
      x_vVoO = nothing
      if !dc
        @tensoropt begin
          x_DBik[D,B,i,k] := oOvV[k,L,c,D] * T2ab[c,B,i,L]
          R2ab[a,B,i,J] += x_DBik[D,B,i,k] * T2ab[a,D,k,J]
        end
        x_DBik = nothing
      end
      oOvV = nothing
    end
  end

  @tensoropt R2a[a,b,i,j] += x_klij[k,l,i,j] *  T2a[a,b,k,l]
  if n_occb_orbs(EC) > 0
    @tensoropt begin
      R2b[A,B,I,J] += x_KLIJ[K,L,I,J] *  T2b[A,B,K,L]
      R2ab[a,B,i,J] += x_kLiJ[k,L,i,J] * T2ab[a,B,k,L]
    end
  end
  x_klij, x_KLIJ, x_kLiJ = nothing, nothing, nothing

  @tensoropt begin
    rR2a[a,b,i,j] := xab[a,c] * T2a[c,b,i,j]
    rR2a[a,b,i,j] -= xij[k,i] * T2a[a,b,k,j]
    R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i]
  end
  rR2a = nothing
  if n_occb_orbs(EC) > 0
    @tensoropt begin
      rR2b[A,B,I,J] := xAB[A,C] * T2b[C,B,I,J]
      rR2b[A,B,I,J] -= xIJ[K,I] * T2b[A,B,K,J]
      R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I]
    end
    rR2b = nothing
    @tensoropt begin
      R2ab[a,B,i,J] -= xij[k,i] * T2ab[a,B,k,J]
      R2ab[a,B,i,J] -= xIJ[K,J] * T2ab[a,B,i,K]
      R2ab[a,B,i,J] += xab[a,c] * T2ab[c,B,i,J]
      R2ab[a,B,i,J] += xAB[B,C] * T2ab[a,C,i,J]
    end
  end
  xij, xIJ, xab, xAB = nothing, nothing, nothing, nothing
  #ph-ab-ladder
  if n_occb_orbs(EC) > 0
    d_vOvO = load(EC,"d_vOvO")
    @tensoropt R2ab[a,B,i,J] -= d_vOvO[a,K,c,J] * T2ab[c,B,i,K]
    d_vOvO = nothing
    d_oVoV = load(EC,"d_oVoV")
    @tensoropt R2ab[a,B,i,J] -= d_oVoV[k,B,i,C] * T2ab[a,C,k,J]
    d_oVoV = nothing
  end

  #ring terms
  d_voov = load(EC,"d_voov")
  @tensoropt begin
    rR2a[a,b,i,j] := d_voov[b,k,j,c] * T2a[a,c,i,k]
    R2ab[a,B,i,J] += d_voov[a,k,i,c] * T2ab[c,B,k,J]
  end
  d_voov = nothing
  d_vOoV = load(EC,"d_vOoV")
  @tensoropt begin
    rR2a[a,b,i,j] += d_vOoV[b,K,j,C] * T2ab[a,C,i,K]
    R2ab[a,B,i,J] += d_vOoV[a,K,i,C] * T2b[B,C,J,K]
  end
  d_vOoV = nothing
  d_vovo = load(EC,"d_vovo")
  @tensoropt begin
    rR2a[a,b,i,j] -= d_vovo[b,k,c,j] * T2a[a,c,i,k]
    R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i] - rR2a[a,b,j,i] - rR2a[b,a,i,j]
    R2ab[a,B,i,J] -= d_vovo[a,k,c,i] * T2ab[c,B,k,J]
  end
  d_vovo, rR2a = nothing, nothing
  if n_occb_orbs(EC) > 0
    d_VOOV = load(EC,"d_VOOV")
    @tensoropt begin
      rR2b[A,B,I,J] := d_VOOV[B,K,J,C] * T2b[A,C,I,K]
      R2ab[a,B,i,J] += d_VOOV[B,K,J,C] * T2ab[a,C,i,K]
    end
    d_VOOV = nothing
    d_oVvO = load(EC,"d_oVvO")
    @tensoropt begin
      rR2b[A,B,I,J] += d_oVvO[k,B,c,J] * T2ab[c,A,k,I]
      R2ab[a,B,i,J] += d_oVvO[k,B,c,J] * T2a[a,c,i,k]
    end
    d_oVvO = nothing
    d_VOVO = load(EC,"d_VOVO")
    @tensoropt begin
      rR2b[A,B,I,J] -= d_VOVO[B,K,C,J] * T2b[A,C,I,K]
      R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I] - rR2b[A,B,J,I] - rR2b[B,A,I,J]
      R2ab[a,B,i,J] -= d_VOVO[B,K,C,J] * T2ab[a,C,i,K]
    end
    d_VOVO, rR2b = nothing, nothing
  end

  if tworef || fixref
    # 2D-CC assumes open-shell singlet reference morba and norbb occupied in Φ^A and morbb and norba in Φ^B.
    @assert length(setdiff(SP['o'],SP['O'])) == 1 && length(setdiff(SP['O'],SP['o'])) == 1 "2D-CCSD needs two open-shell alpha beta orbitals"
    morba, norbb, morbb, norba = active_orbitals(EC)
    if tworef
      activeorbs = (morba, norbb, morbb, norba)
      occcorea = collect(1:length(SP['o']))
      occcoreb = collect(1:length(SP['O']))
      filter!(x -> x != morba, occcorea)
      filter!(x -> x != norbb, occcoreb)
      occcore = (occcorea, occcoreb)
      virtualsa = collect(1:length(SP['v']))
      virtualsb = collect(1:length(SP['V']))
      filter!(x -> x != norba, virtualsa)
      filter!(x -> x != morbb, virtualsb)
      virtuals = (virtualsa, virtualsb)
      W = R2ab[norba,morbb,morba,norbb]
      R2ab[norba,morbb,morba,norbb] = 0.0
      if length(T1a) > 0
        M1a = calc_M1a(occcore,virtuals,T1b, T2ab, activeorbs)
        M1b = calc_M1b(occcore,virtuals,T1a, T2ab, activeorbs)
        @tensoropt R1a[a,i] += M1a[a,i] * W
        @tensoropt R1b[a,i] += M1b[a,i] * W
      end
      if !isempty(occcorea) && !isempty(occcoreb)
        M2a = calc_M2a(occcore,virtuals,T1a,T1b,T2b,T2ab, activeorbs)
        M2b = calc_M2b(occcore,virtuals,T1a,T1b,T2a,T2ab, activeorbs)
        @tensoropt R2a[a,b,i,j] += M2a[a,b,i,j] * W
        @tensoropt R2b[a,b,i,j] += M2b[a,b,i,j] * W
      end
      M2ab = calc_M2ab(occcore,virtuals,T1a,T1b,T2a,T2b,T2ab, activeorbs)
      @tensoropt R2ab[a,b,i,j] += M2ab[a,b,i,j] * W
      save!(EC,"2d_ccsd_W",[W])
    elseif fixref
      R2ab[norba,morbb,morba,norbb] = 0
    end
  end
  return R1a, R1b, R2a, R2b, R2ab
end

function active_orbitals(EC::ECInfo)
  SP = EC.space
  @assert length(setdiff(SP['o'],SP['O'])) == 1 && length(setdiff(SP['O'],SP['o'])) == 1 "Assumed two open-shell alpha beta orbitals here."
  morb = setdiff(SP['o'],SP['O'])[1]
  norb = setdiff(SP['O'],SP['o'])[1]
  morba = findfirst(isequal(morb),SP['o'])
  norbb = findfirst(isequal(norb),SP['O'])
  morbb = findfirst(isequal(morb),SP['V'])
  norba = findfirst(isequal(norb),SP['v'])
  return morba,norbb,morbb,norba
end

function calc_M1a(occcore,virtuals,T1, T2, activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M1 = zeros(Float64,size(T1))
  if !isempty(occcorea) && !isempty(occcoreb)
    @tensoropt M1[norba,occcorea][i] += T2[norba,morbb,morba,occcoreb][i]
    @tensoropt M1[virtualsa,occcorea][a,i] += T2[norba,morbb,morba,occcoreb][i] * T1[virtualsb,norbb][a]
    @tensoropt M1[virtualsa,occcorea][a,i] += T2[norba,virtualsb,morba,norbb][a] * T1[morbb,occcoreb][i]
  end
  @tensoropt M1[virtualsa,morba][a] -= T2[norba,virtualsb,morba,norbb][a]
  return M1
end

function calc_M1b(occcore,virtuals,T1, T2, activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M1 = zeros(Float64,size(T1))
  if !isempty(occcorea) && !isempty(occcoreb)
    @tensoropt M1[morbb,occcoreb][i] += T2[norba,morbb,occcorea,norbb][i]
    @tensoropt M1[virtualsb,occcoreb][a,i] += T2[norba,morbb,occcorea,norbb][i] * T1[virtualsa,morba][a]
    @tensoropt M1[virtualsb,occcoreb][a,i] += T2[virtualsa,morbb,morba,norbb][a] * T1[norba,occcorea][i]
  end
  @tensoropt M1[virtualsb,norbb][a] -= T2[virtualsa,morbb,morba,norbb][a]
  return M1
end

function calc_M2a(occcore,virtuals,T1a,T1b,T2b,T2ab,activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M2 = zeros(Float64,size(T2b))
  if length(T1a) > 0
    @tensoropt T1[a,i] := T1a[a,i] - T1b[a,i]
    @tensoropt T2t[a,b,i,j] := T2b[a,b,i,j] + T1b[a,i] * T1b[b,j]

    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * T1[virtualsa,occcorea][a,j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] += T2ab[norba,morbb,morba,occcoreb][j] * T1[virtualsa,occcorea][a,i]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] += T2ab[norba,virtualsb,morba,occcoreb][a,j] * T1b[morbb,occcoreb][i]

    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += T2ab[norba,virtualsb,morba,norbb][a] * T1[virtualsa,occcorea][b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= T2ab[norba,virtualsb,morba,norbb][a] * T1[virtualsa,occcorea][b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2t[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2t[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2t[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2t[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
  end
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,morbb,morba,occcoreb][j] * T2b[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[norba,morbb,morba,occcoreb][j] * T2b[virtualsb,virtualsb,norbb,occcoreb][a,b,i]

  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,virtualsb,morba,norbb][b] * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[norba,virtualsb,morba,norbb][b] * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]


  @tensoropt M2[virtualsa,norba,occcorea,morba][a,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i]
  return M2
end

function calc_M2b(occcore,virtuals,T1a,T1b,T2a,T2ab,activeorbs)
# morba, norbb, morbb, norba = activeorbs
  norbb, morba, norba, morbb = activeorbs
  occcoreb, occcorea = occcore
  virtualsb, virtualsa = virtuals
  P12 = (2,1,4,3)
  M2 = zeros(Float64,size(T2a))
  if length(T1a) > 0
    @tensoropt T1[a,i] := T1b[a,i] - T1a[a,i]
    @tensoropt T2t[a,b,i,j] := T2a[a,b,i,j] + T1a[a,i] * T1a[b,j]

    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1[virtualsa,occcorea][a,j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][j] * T1[virtualsa,occcorea][a,i]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,j] * T1a[morbb,occcoreb][i]

    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1[virtualsa,occcorea][b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1[virtualsa,occcorea][b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[virtualsb,norbb][b]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * T1[virtualsa,occcorea][b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * T1[virtualsa,occcorea][b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2t[morbb,virtualsb,norbb,occcoreb][a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2t[morbb,virtualsb,norbb,occcoreb][a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2t[morbb,virtualsb,norbb,occcoreb][a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2t[morbb,virtualsb,norbb,occcoreb][a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
  end
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][j] * T2a[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][j] * T2a[virtualsb,virtualsb,norbb,occcoreb][a,b,i]

  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][b] * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][b] * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]

  @tensoropt M2[virtualsa,norba,occcorea,morba][a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i]
  return M2
end

function calc_M2ab(occcore,virtuals,T1a,T1b,T2a,T2b,T2ab,activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M2 = zeros(Float64,size(T2ab))

  if length(T1a) > 0
    @tensoropt T1[a,i] := T1a[a,i] - T1b[a,i]
    @tensoropt T2ta[a,b,i,j] := T2a[a,b,i,j] + T1a[a,i] * T1a[b,j]
    @tensoropt T2tb[a,b,i,j] := T2b[a,b,i,j] + T1b[a,i] * T1b[b,j]
    @tensoropt T2tab[a,b,i,j] := T2ab[a,b,i,j] + T1a[a,i] * T1b[b,j]
    if !isempty(occcorea) && !isempty(occcoreb)
      @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * T1[virtualsb,occcoreb][a,j]
      @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] += T2ab[norba,morbb,occcorea,norbb][i] * T1[virtualsa,occcorea][a,j]
      @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2ab[virtualsa,morbb,morba,occcoreb][a,i] * T1a[norba,occcorea][j]
      @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] -= T2ab[norba,virtualsb,occcorea,norbb][a,i] * T1b[morbb,occcoreb][j]
      @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2tab[norba,morbb,occcorea,occcoreb][j,i] * T1a[virtualsa,morba][a]
      @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] -= T2tab[norba,morbb,occcorea,occcoreb][i,j] * T1b[virtualsb,norbb][a]

      @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2ab[norba,virtualsb,morba,norbb][a] * T1[virtualsb,occcoreb][b,i]
      @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] -= T2ab[virtualsa,morbb,morba,norbb][a] * T1[virtualsa,occcorea][b,i]
      @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2ab[norba,virtualsb,occcorea,norbb][a,i] * T1a[virtualsa,morba][b] 
      @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] += T2ab[virtualsa,morbb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b] 
      @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2tab[virtualsa,virtualsb,morba,norbb][b,a] * T1a[norba,occcorea][i]
      @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] += T2tab[virtualsa,virtualsb,morba,norbb][a,b] * T1b[morbb,occcoreb][i] 

      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * T1[virtualsb,occcoreb][b,j]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][b,a,j,i] += T2ab[norba,morbb,occcorea,norbb][i] * T1a[virtualsa,morba][a] * T1[virtualsa,occcorea][b,j]

      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * T1[virtualsb,occcoreb][b,j]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][b,a,j,i] += T2ab[virtualsa,morbb,morba,norbb][a] * T1a[norba,occcorea][i] * T1[virtualsa,occcorea][b,j]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2a[norba,virtualsa,morba,occcorea][b,j] * T1b[virtualsb,norbb][a] * T1b[morbb,occcoreb][i]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T1a[norba,occcorea][j] * T1a[virtualsa,morba][b]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[virtualsa,morbb,morba,occcoreb][b,i] * T1a[norba,occcorea][j] * T1b[virtualsb,norbb][a]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,virtualsb,occcorea,norbb][a,j] * T1a[virtualsa,morba][b] * T1b[morbb,occcoreb][i]
      @tensoropt M2[norba,morbb,morba,occcoreb][i] += T1a[norba,occcorea][i]
      @tensoropt M2[norba,morbb,occcorea,norbb][i] += T1b[morbb,occcoreb][i]
      @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2tab[norba,morbb,occcorea,occcoreb][i,j] * T2tab[virtualsa,virtualsb,morba,norbb][a,b]
      @tensoropt M2[norba,morbb,occcorea,occcoreb][i,j] -= T2tab[norba,morbb,occcorea,occcoreb][j,i]
      @tensoropt M2[norba,virtualsb,morba,occcoreb][a,i] += T2ta[norba,virtualsa,occcorea,morba][a,i]
      @tensoropt M2[virtualsa,morbb,occcorea,norbb][a,i] += T2tb[morbb,virtualsb,occcoreb,norbb][a,i]
      @tensoropt M2[virtualsa,morbb,morba,occcoreb][a,i] += T2tab[norba,virtualsb,occcorea,norbb][a,i]
      @tensoropt M2[norba,virtualsb,occcorea,norbb][a,i] += T2tab[virtualsa,morbb,morba,occcoreb][a,i]
    end
    @tensoropt M2[norba,virtualsb,morba,norbb][a] -= T1a[virtualsa,morba][a]
    @tensoropt M2[virtualsa,morbb,morba,norbb][a] -= T1b[virtualsb,norbb][a]
    @tensoropt M2[virtualsa,virtualsb,morba,norbb][a,b] -= T2tab[virtualsa,virtualsb,morba,norbb][b,a]
  end
  if !isempty(occcorea) && !isempty(occcoreb)
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[norba,morbb,occcorea,norbb][j] * T2ab[virtualsa,virtualsb,morba,occcoreb][b,a,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[norba,morbb,morba,occcoreb][i] * T2ab[virtualsa,virtualsb,occcorea,norbb][b,a,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[virtualsa,morbb,morba,norbb][b] * T2ab[norba,virtualsb,occcorea,occcoreb][a,j,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[norba,virtualsb,morba,norbb][a] * T2ab[virtualsa,morbb,occcorea,occcoreb][b,j,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2b[morbb,virtualsb,occcoreb,norbb][a,i] * T2a[norba,virtualsa,morba,occcorea][b,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T2ab[virtualsa,morbb,occcorea,norbb][b,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,virtualsb,occcorea,norbb][a,j] * T2ab[virtualsa,morbb,morba,occcoreb][b,i]
  end
  return M2
end

"""
    calc_cc(EC::ECInfo, method::ECMethod)

  Calculate coupled cluster amplitudes.

  Exact specification of the method is given by `method`.
"""
function calc_cc(EC::ECInfo, method::ECMethod)
  dc = (method.theory == "DC" || last(method.theory,2) == "DC")
  tworef = method.theory[1:2] == "2D"
  fixref = method.theory[1:2] == "FR"
  print_info(method_name(method))
  Amps, exc_ranges = starting_amplitudes(EC, method)
  singles, doubles, triples = exc_ranges[1:3]
  if method.unrestricted
    @assert (length(singles) == 2) && (length(doubles) == 3) && (length(triples) == 4)
  else
    @assert (length(singles) == 1) && (length(doubles) == 1) && (length(triples) == 1)
  end
  T2αβ = last(doubles)
  diis = Diis(EC)

  NormR1 = 0.0
  NormT1 = 0.0
  NormT2 = 0.0
  do_sing = (method.exclevel[1] == :full)
  Eh = 0.0
  En1 = 0.0
  converged = false
  t0 = time_ns()
  println("Iter     SqNorm      Energy      DE          Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    Res = calc_ccsd_resid(EC, Amps...; dc, tworef, fixref)
    t1 = print_time(EC, t1, "residual", 2)
    NormT2 = calc_doubles_norm(Amps[doubles]...)
    NormR2 = calc_doubles_norm(Res[doubles]...)
    Eh = calc_hylleraas(EC, Amps..., Res...)
    update_doubles!(EC, Amps[doubles]..., Res[doubles]...)
    if length(method.theory) > 2 && uppercase(method.theory[1:3]) == "FRS"
      morba, norbb, morbb, norba = active_orbitals(EC)
      Amps[T2αβ][norba,morbb,morba,norbb] = 1.0
    elseif length(method.theory) > 2 && uppercase(method.theory[1:3]) == "FRT"
      morba, norbb, morbb, norba = active_orbitals(EC)
      Amps[T2αβ][norba,morbb,morba,norbb] = -1.0
    elseif uppercase(method.theory[1:2]) == "2D"
      morba, norbb, morbb, norba = active_orbitals(EC)
      # println("T1a all internal: ", T1a[norba,morba])
      # println("T1b all internal: ", T1b[morbb,norbb])
      # T1a[norba,morba] = 0.0
      # T1b[morbb,norbb] = 0.0
    end
    if do_sing
      NormT1 = calc_singles_norm(Amps[singles]...)
      NormR1 = calc_singles_norm(Res[singles]...)
      update_singles!(EC, Amps[singles]..., Res[singles]...)
    end
    Amps = perform(diis, Amps, Res)
    if do_sing
      save_current_singles(EC, Amps[singles]...)
      En1 = calc_singles_energy(EC, Amps[singles]...)
    end
    save_current_doubles(EC, Amps[doubles]...)
    En2 = calc_doubles_energy(EC, Amps[doubles]...)
    En = En1 + En2
    ΔE = En - Eh  
    NormR = NormR1 + NormR2
    NormT = 1.0 + NormT1 + NormT2
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %12.8f %10.2e %8.2f \n" it NormT Eh ΔE NormR tt
    flush(stdout)
    if NormR < EC.options.cc.thr
      converged = true
      break
    end
  end
  if !converged
    println("WARNING: CC iterations did not converge!")
  end
  if do_sing
    try2save_singles!(EC, Amps[singles]...)
  end
  try2save_doubles!(EC, Amps[doubles]...)
  println()
  @printf "Sq.Norm of T1: %12.8f Sq.Norm of T2: %12.8f \n" NormT1 NormT2
  println()
  flush(stdout)
  return Eh
end

""" 
    calc_ccsdt(EC::ECInfo, useT3 = false, cc3 = false)

  Calculate decomposed closed-shell DC-CCSDT amplitudes.

  If `useT3`: (T) amplitudes from a preceding calculations will be used as starting guess.
  If cc3: calculate CC3 amplitudes.
"""
function calc_ccsdt(EC::ECInfo, useT3 = false, cc3 = false)
  if cc3
    print_info("CC3")
  else
    print_info("DC-CCSDT")
  end
  calc_integrals_decomposition(EC)
  T1 = read_starting_guess4amplitudes(EC, 1)
  T2 = read_starting_guess4amplitudes(EC, 2)
  if useT3
    calc_triples_decomposition(EC)
  else
    # calc_dressed_3idx(EC,zeros(size(T1)))
    calc_dressed_3idx(EC, T1)
    calc_triples_decomposition_without_triples(EC, T2)
  end
  diis = Diis(EC)

  println("Iter     SqNorm      Energy      DE          Res         Time")
  NormR1 = 0.0
  NormT1 = 0.0
  NormT2 = 0.0
  NormT3 = 0.0
  R1 = Float64[]
  Eh = 0.0
  t0 = time_ns()
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    #get dressed integrals
    calc_dressed_3idx(EC, T1)
    # test_dressed_ints(EC,T1) #DEBUG
    t1 = print_time(EC, t1, "dressed 3-idx integrals", 2)
    R1, R2 = calc_ccsd_resid(EC, T1, T2)
    t1 = print_time(EC, t1, "ccsd residual", 2)
    R1, R2 = add_to_singles_and_doubles_residuals(EC, R1, R2)
    t1 = print_time(EC, t1, "R1(T3) and R2(T3)", 2)
    calc_triples_residuals(EC, T1, T2, cc3)
    t1 = print_time(EC, t1, "R3", 2)
    NormT1 = calc_singles_norm(T1)
    NormT2 = calc_doubles_norm(T2)
    T3 = load(EC, "T_XXX")
    NormT3 = calc_deco_triples_norm(T3)
    NormR1 = calc_singles_norm(R1)
    NormR2 = calc_doubles_norm(R2)
    R3 = load(EC, "R_XXX")
    NormR3 = calc_deco_triples_norm(R3)
    Eh = calc_hylleraas(EC, T1, T2, R1, R2)
    T1 += update_singles(EC, R1)
    T2 += update_doubles(EC, R2)
    T3 += update_deco_triples(EC, R3)
    T1, T2, T3 = perform(diis, [T1,T2,T3], [R1,R2,R3])
    save!(EC, "T_XXX", T3)
    En = calc_singles_energy(EC, T1)
    En += calc_doubles_energy(EC, T2)
    ΔE = En - Eh
    NormR = NormR1 + NormR2 + NormR3
    NormT = 1.0 + NormT1 + NormT2 + NormT3
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %12.8f %10.2e %8.2f \n" it NormT Eh ΔE NormR tt
    flush(stdout)
    if NormR < EC.options.cc.thr
      break
    end
  end
  println()
  @printf "Sq.Norm of T1: %12.8f Sq.Norm of T2: %12.8f Sq.Norm of T3: %12.8f \n" NormT1 NormT2 NormT3
  println()
  flush(stdout)
  
  return Eh
end

"""
    add_to_singles_and_doubles_residuals(EC,R1,R2)

  Add contributions from triples to singles and doubles residuals.
"""
function add_to_singles_and_doubles_residuals(EC,R1,R2)
  SP = EC.space
  ooPfile, ooP = mmap(EC, "d_ooL")
  ovPfile, ovP = mmap(EC, "d_ovL")
  Txyz = load(EC, "T_XXX")
  
  U = load(EC, "C_voX")
  # println(size(U))

  @tensoropt Boo[i,j,P,X] := ovP[i,a,P] * U[a,j,X]
  @tensoropt A[P,X] := Boo[i,i,P,X] 
  @tensoropt BBU[Z,d,j] := (ovP[j,c,P] * ovP[k,d,P]) * U[c,k,Z]
  @tensoropt R1[a,i] += U[a,i,X] *(Txyz[X,Y,Z] *( 2.0*A[P,Y] * A[P,Z] - Boo[j,k,P,Z] * Boo[k,j,P,Y] ))
  @tensoropt R1[a,i] -= U[a,j,Y] *( 2.0*Boo[j,i,P,X]*(Txyz[X,Y,Z] * A[P,Z]) - Txyz[X,Y,Z] *(U[d,i,X]*BBU[Z,d,j] ))

  BBU = nothing

  @tensoropt Bov[i,a,P,X] := ooP[j,i,P] * U[a,j,X]
  vvPfile, vvP = mmap(EC, "d_vvL")
  @tensoropt Bvo[a,i,P,X] := vvP[a,b,P] * U[b,i,X]
  close(vvPfile)
  vvP = nothing
  dfock = load(EC, "df_mm")
  fov = dfock[SP['o'], SP['v']]
  # R2[abij] = RR2[abij] + RR2[baji]  
  @tensoropt RR2[a,b,i,j] := U[a,i,X] * (U[b,j,Y] * (Txyz[X,Y,Z] * (fov[k,c]*U[c,k,Z])) - (Txyz[X,Y,Z] * U[b,k,Z])* (fov[k,c]*U[c,j,Y]))
  @tensoropt RR2[a,b,i,j] += 2.0*U[b,j,Y] * ((Bvo[a,i,P,Z] - Bov[i,a,P,Z])*(Txyz[X,Y,Z] * A[P,X]))
  @tensoropt RR2[a,b,i,j] += (Bov[i,a,P,Z]  - Bvo[a,i,P,Z])*(Boo[k,j,P,Y] * (Txyz[X,Y,Z] * U[b,k,X]))
  @tensoropt RR2[a,b,i,j] -= U[b,j,Z] * (Txyz[X,Y,Z] * (Bvo[a,k,P,X] * Boo[k,i,P,Y] - U[a,k,Y] * (Bov[i,c,P,X] * ovP[k,c,P])))
  @tensoropt R2[a,b,i,j] += RR2[a,b,i,j] + RR2[b,a,j,i]
  close(ovPfile)
  close(ooPfile)

  return R1,R2
end


"""
    calc_triples_decomposition_without_triples(EC::ECInfo, T2)

  Decompose T^{ijk}_{abc} as U^{iX}_a * U^{jY}_b * U^{kZ}_c * T_{XYZ} 
  without explicit calculation of T^{ijk}_{abc}.

  Compute perturbative T^i_{aXY} and decompose D^{ij}_{ab} = (T^i_{aXY} T^j_{bXY}) to get U^{iX}_a.
"""
function calc_triples_decomposition_without_triples(EC::ECInfo, T2)
  println("T^ijk_abc-free-decomposition")
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)

  # first approx for U^iX_a from doubles decomposition
  tol2 = EC.options.cc.ampsvdtol*0.01
  UaiX = svd_decompose(reshape(permutedims(T2, (1,3,2,4)), (nocc*nvirt, nocc*nvirt)), nvirt, nocc, tol2)
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  D2 = calc_4idx_T3T3_XY(EC, T2, UaiX, ϵX) 
  # use tol^2 because D2 = (T3)^2
  UaiX = svd_decompose(reshape(D2, (nocc*nvirt, nocc*nvirt)), nvirt, nocc, EC.options.cc.ampsvdtol^2)
  # UaiX = eigen_decompose(reshape(D2, (nocc*nvirt, nocc*nvirt)), nvirt, nocc, EC.options.cc.ampsvdtol^2)
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save!(EC, "e_X", ϵX)
  #display(UaiX)
  naux = length(ϵX)
  save!(EC,"C_voX",UaiX)
  # TODO: calc starting guess for T3_XYZ from T2 and UvoX
  save!(EC,"T_XXX",zeros(naux, naux, naux))
end

"""
    calc_triples_decomposition(EC::ECInfo)

  Decompose T^{ijk}_{abc} as U^{iX}_a * U^{jY}_b * U^{kZ}_c * T_{XYZ}.
"""
function calc_triples_decomposition(EC::ECInfo)
  println("T^ijk_abc-decomposition")
  use_svd = true 
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)

  Triples_Amplitudes = zeros(nvirt, nocc, nvirt, nocc, nvirt, nocc)
  t3file, T3 = mmap(EC, "T_vvvooo")
  trippp = [CartesianIndex(i,j,k) for k in 1:nocc for j in 1:k for i in 1:j]
  for ijk in axes(T3,4)
    i,j,k = Tuple(trippp[ijk])                                            #trippp is giving the indices according to the joint index ijk as a tuple
    Triples_Amplitudes[:,i,:,j,:,k] = T3[:,:,:,ijk]
    Triples_Amplitudes[:,j,:,i,:,k] = permutedims(T3[:,:,:,ijk],(2,1,3))
    Triples_Amplitudes[:,i,:,k,:,j] = permutedims(T3[:,:,:,ijk],(1,3,2))
    Triples_Amplitudes[:,k,:,j,:,i] = permutedims(T3[:,:,:,ijk],(3,2,1))
    Triples_Amplitudes[:,j,:,k,:,i] = permutedims(T3[:,:,:,ijk],(2,3,1))
    Triples_Amplitudes[:,k,:,i,:,j] = permutedims(T3[:,:,:,ijk],(3,1,2))
  end
  close(t3file)
  if use_svd
    UaiX = svd_decompose(reshape(Triples_Amplitudes, (nocc*nvirt, nocc*nocc*nvirt*nvirt)), nvirt, nocc, EC.options.cc.ampsvdtol)
  else
    naux = nvirt * 2 
    UaiX = iter_svd_decompose(reshape(Triples_Amplitudes, (nocc*nvirt, nocc*nocc*nvirt*nvirt)), nvirt, nocc, naux)
  end
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save!(EC, "e_X", ϵX)
  #display(UaiX)
  save!(EC,"C_voX",UaiX)

  @tensoropt begin
    T3_decomp_starting_guess[X,Y,Z] := (((Triples_Amplitudes[a,i,b,j,c,k] * UaiX[a,i,X]) * UaiX[b,j,Y]) * UaiX[c,k,Z])
  end
  save!(EC,"T_XXX",T3_decomp_starting_guess)
  #display(T3_decomp_starting_guess)

  # @tensoropt begin
  #  T3_decomp_check[a,i,b,j,c,k] := T3_decomp_starting_guess[X,Y,Z] * UaiX2[a,i,X] * UaiX2[b,j,Y] * UaiX2[c,k,Z]
  # end
  # test_calc_pertT_from_T3(EC,T3_decomp_check)
end

"""
    calc_4idx_T3T3_XY(EC::ECInfo, T2, UvoX, ϵX)

  Calculate D^{ij}_{ab} = T^i_{aXY} T^j_{bXY} using half-decomposed imaginary-shifted perturbative triple amplitudes 
  T^i_{aXY} from T2 (and UvoX)
"""
function calc_4idx_T3T3_XY(EC::ECInfo, T2, UvoX, ϵX)
  voPfile, voP = mmap(EC, "d_voL")
  ooPfile, ooP = mmap(EC, "d_ooL")
  vvPfile, vvP = mmap(EC, "d_vvL")

  @tensoropt TXai[X,a,i] := UvoX[b,j,X] * T2[a,b,i,j]
  @tensoropt dU[P,X] := voP[c,k,P] * UvoX[c,k,X]

  @tensoropt RR[X,Y,a,i] := ((TXai[X,c,j] * vvP[b,c,P]) * UvoX[b,j,Y]) * voP[a,i,P]
  @tensoropt RR[X,Y,a,i] -= ((TXai[X,b,l] * ooP[l,j,P]) * UvoX[b,j,Y]) * voP[a,i,P]
  @tensoropt ddUv[a,d,X] := vvP[a,d,P] * dU[P,X]
  @tensoropt ddUo[l,j,X] := ooP[l,j,P] * dU[P,X]
  @tensoropt RR[X,Y,a,i] += ddUv[a,d,X] * TXai[Y,d,i]
  @tensoropt RR[X,Y,a,i] -= ddUo[l,i,X] * TXai[Y,a,l]
  TXai = nothing
  dU = nothing
  @tensoropt ddUU[X,Y,d,l] := ddUv[a,d,X] * UvoX[a,l,Y]
  @tensoropt ddUU[X,Y,d,l] -= ddUo[l,i,X] * UvoX[d,i,Y]
  @tensoropt RR[X,Y,a,i] += ddUU[X,Y,d,l] * T2[a,d,i,l]
  ddUU = nothing
  @tensoropt R[X,Y,a,i] := RR[X,Y,a,i] + RR[Y,X,a,i]
  RR = nothing
  close(voPfile)
  close(ooPfile)
  close(vvPfile)
  ϵo, ϵv = orbital_energies(EC)
  shifti = EC.options.cc.deco_ishiftt
  if shifti > 1.e-10
    # imaginary-shifted triples
    for I ∈ CartesianIndices(R)
      X,Y,a,i = Tuple(I)
      den = ϵX[X] + ϵX[Y] + ϵv[a] - ϵo[i]
      R[I] *= -den/(den^2 + shifti)
    end
  else
    for I ∈ CartesianIndices(R)
      X,Y,a,i = Tuple(I)
      R[I] /= -(ϵX[X] + ϵX[Y] + ϵv[a] - ϵo[i])
    end
  end
  nocc = n_occ_orbs(EC)
  naux = length(ϵX)
  # @tensoropt T3_decomp_check[a,i,b,j,c,k] := R[X,Y,a,i] * UvoX[c,k,X] * UvoX[b,j,Y]
  # for i = 1:nocc
  #   T3_decomp_check[:,i,:,i,:,i] .= 0.0
  # end
  # test_calc_pertT_from_T3(EC,T3_decomp_check)
  @tensoropt D2[a,i,b,j] := R[X,Y,a,i] * R[X,Y,b,j]
  # remove T^iii contributions from D2
  UU = zeros(naux,naux,nocc)
  for i = 1:nocc
    @tensoropt UU[:,:,i][X,Y] = UvoX[:,i,:][a,X] * UvoX[:,i,:][a,Y]
  end
  TUU4i = zeros(naux,naux,size(UvoX,1))
  ΔD2 = zeros(size(D2,1),size(D2,3))
  for i = 1:nocc
    @tensoropt TUU4i[X',Y',a] = (R[:,:,:,i][X,Y,a] * UU[:,:,i][X,X']) * UU[:,:,i][Y,Y']
    for j = 1:nocc
      @tensoropt ΔD2[a,b] = TUU4i[X,Y,a] * R[:,:,:,j][X,Y,b]
      @tensoropt D2[:,i,:,j][a,b] -= ΔD2[a,b]
      if i != j
        @tensoropt D2[:,j,:,i][b,a] -= ΔD2[a,b]
      end
    end
  end
  # display(D2)
  return D2
end

"""
    calc_triples_residuals(EC::ECInfo, T1, T2, cc3 = false)

  Calculate decomposed triples DC-CCSDT or CC3 residuals.
"""
function calc_triples_residuals(EC::ECInfo, T1, T2, cc3 = false)
  t1 = time_ns()
  UvoX = load(EC, "C_voX")
  #display(UvoX)

  #load decomposed amplitudes
  T3_XYZ = load(EC, "T_XXX")
  #display(T3_XYZ)

  #load df coeff
  ovPfile, ovP = mmap(EC, "d_ovL")
  voPfile, voP = mmap(EC, "d_voL")
  ooPfile, ooP = mmap(EC, "d_ooL")
  vvPfile, vvP = mmap(EC, "d_vvL")

  #load dressed fock matrices
  SP = EC.space
  dfock = load(EC, "df_mm")    
  dfoo = dfock[SP['o'], SP['o']]
  dfov = dfock[SP['o'], SP['v']]
  dfvv = dfock[SP['v'], SP['v']]
  
  @tensoropt Thetavirt[b,d,Z] := vvP[b,d,Q] * (voP[c,k,Q] * UvoX[c,k,Z]) #virt1
  @tensoropt Thetavirt[b,d,Z] += UvoX[c,k,Z] * (T2[c,b,l,m] * (ooP[l,k,Q] * ovP[m,d,Q])) #virt3
  @tensoropt Thetavirt[b,d,Z] -= ovP[l,d,Q] * (T2[b,e,l,k] * (UvoX[c,k,Z] * vvP[c,e,Q])) #virt6
  t1 = print_time(EC, t1, "1 Theta terms in R3(T3)", 2)
  
  @tensoropt Thetaocc[l,j,Z] := ooP[l,j,Q] * (voP[c,k,Q] * UvoX[c,k,Z]) #occ1
  @tensoropt Thetaocc[l,j,Z] -= UvoX[c,k,Z] * (T2[c,d,m,j] * (ovP[l,d,Q] * ooP[m,k,Q])) #occ4
  @tensoropt Thetaocc[l,j,Z] += UvoX[c,k,Z] * (T2[d,e,k,j]* (ovP[l,e,Q] * vvP[c,d,Q])) #occ5
  t1 = print_time(EC, t1, "2 Theta terms in R3(T3)", 2)
  if !cc3
    @tensoropt BooQX[i,j,Q,X] := ovP[i,a,Q] * UvoX[a,j,X]
    @tensoropt Thetavirt[b,d,Z] += 0.5* T3_XYZ[X',Y',Z] * (UvoX[b,m,Y'] * (ovP[l,d,Q] * BooQX[m,l,Q,X'])) #virt9
    @tensoropt Thetaocc[l,j,Z] -= 0.5 * T3_XYZ[X',Z,Z'] * (BooQX[l,m,Q,X'] * BooQX[m,j,Q,Z']) #occ8
    BooQX = nothing
    t1 = print_time(EC, t1, "3 Theta terms in R3(T3)", 2)

    @tensoropt A[Q,X] := ovP[i,a,Q] * UvoX[a,i,X]
    @tensoropt Thetavirt[b,d,Z] -= ovP[l,d,Q] * (UvoX[b,l,Z'] * (T3_XYZ[X',Z,Z'] * A[Q,X'])) #virt7
    @tensoropt Thetaocc[l,j,Z] += ovP[l,d,Q] * (UvoX[d,j,Z']* (T3_XYZ[X',Z,Z'] * A[Q,X']))   #occ6
    A = nothing
    t1 = print_time(EC, t1, "4 Theta terms in R3(T3)", 2)

    @tensoropt IntermediateTheta[Q,Z',Z] := ovP[m,e,Q] * (UvoX[e,k,Y'] * (T3_XYZ[X',Y',Z'] * (UvoX[c,m,X'] * UvoX[c,k,Z])))
    @tensoropt Thetavirt[b,d,Z] += 0.5* ovP[l,d,Q] * (UvoX[b,l,Z'] * IntermediateTheta[Q,Z',Z]) #virt8
    @tensoropt Thetaocc[l,j,Z] -= 0.5 * ovP[l,d,Q] * (UvoX[d,j,Z'] * IntermediateTheta[Q,Z',Z]) #occ7
    IntermediateTheta = nothing
    t1 = print_time(EC, t1, "5 Theta terms in R3(T3)", 2)
  end

  @tensoropt TaiX[a,i,X] := UvoX[b,j,X] * T2[a,b,i,j]
  @tensoropt TStrich[a,i,X] := 2* TaiX[a,i,X] - UvoX[b,j,X] * T2[b,a,i,j] 
  @tensoropt Thetavirt[b,d,Z] += vvP[b,d,Q] * (ovP[l,e,Q] * TStrich[e,l,Z]) #virt4
  @tensoropt Thetaocc[l,j,Z] += ooP[l,j,Q] * (ovP[m,d,Q] * TStrich[d,m,Z]) #occ2
  TStrich = nothing
  t1 = print_time(EC, t1, "6 Theta terms in R3(T3)", 2)

  @tensoropt Thetavirt[b,d,Z] -= dfov[l,d] * TaiX[b,l,Z] #virt2
  @tensoropt Thetavirt[b,d,Z] -= ovP[l,d,Q] * (vvP[b,e,Q] * TaiX[e,l,Z]) #virt5
  @tensoropt Thetaocc[l,j,Z] -= ooP[m,j,Q] * (ovP[l,d,Q] * TaiX[d,m,Z]) #occ3
  t1 = print_time(EC, t1, "7 Theta terms in R3(T3)", 2)
  
  @tensoropt Term1[X,Y,Z] := (TaiX[b,l,X] * Thetaocc[l,j,Z] - Thetavirt[b,d,Z] * TaiX[d,j,X]) * UvoX[b,j,Y]
  Thetaocc = nothing
  Thetavirt = nothing
  TaiX = nothing
  t1 = print_time(EC, t1, "Theta terms in R3(T3)", 2)

  @tensoropt R3decomp[X,Y,Z] := Term1[X,Y,Z] + Term1[Y,X,Z] + Term1[X,Z,Y] + Term1[Z,Y,X] + Term1[Z,X,Y] + Term1[Y,Z,X]
  Term1 = nothing
  t1 = print_time(EC, t1, "Symmetrization of Theta terms in R3(T3)", 2)


  @tensor TTilde[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  if cc3
    @tensoropt Term2[X,Y,Z] := T3_XYZ[X',Y,Z] * (UvoX[a,l,X'] * (dfoo[l,i]  * UvoX[a,i,X])) #1
    @tensoropt Term2[X,Y,Z] -= T3_XYZ[X',Y,Z] * (UvoX[a,i,X] *( dfvv[a,d] * UvoX[d,i,X'])) #2
  else
    @tensoropt Intermediate1Term2[l,d,m,e] := ovP[l,d,P] * ovP[m,e,P]
    @tensoropt Term2[X,Y,Z] := T3_XYZ[X',Y,Z] * (UvoX[a,l,X'] * ( (dfoo[l,i] + 0.5 * Intermediate1Term2[l,d,m,e] * TTilde[d,e,i,m]) * UvoX[a,i,X])) #1
    @tensoropt Term2[X,Y,Z] -= T3_XYZ[X',Y,Z] * (UvoX[a,i,X] *( (dfvv[a,d] - 0.5 * Intermediate1Term2[l,d,m,e] * TTilde[a,e,l,m]) * UvoX[d,i,X'])) #2
    Intermediate1Term2 = nothing
    t1 = print_time(EC, t1, "1 Chi terms in R3(T3)", 2)
    @tensoropt Term2[X,Y,Z] += (UvoX[a,i,X] * ((ooP[l,i,P] * vvP[a,d,P]) * UvoX[d,l,X'])) * (T3_XYZ[X',Y',Z] * (UvoX[b,j,Y] * UvoX[b,j,Y'])) #3
    @tensoropt Term2[X,Y,Z] -= 2* (T3_XYZ[X',Y,Z] *((voP[a,i,P] + ovP[m,e,P] * TTilde[a,e,i,m]) * UvoX[a,i,X]) * (ovP[l,d,P] * UvoX[d,l,X'])) #4
    @tensoropt Term2[X,Y,Z] -= T3_XYZ[X,Y',Z'] * (((UvoX[c,k,Z] * UvoX[c,m,Z']) * ooP[m,k,P]) * (UvoX[b,j,Y] * (ooP[l,j,P] * UvoX[b,l,Y']))) #5
    @tensoropt Intermediate2Term2[Y,Y',P] :=  UvoX[b,j,Y] * (vvP[b,d,P] * UvoX[d,j,Y'])
    @tensoropt Intermediate3Term2[X',Y,Z,P] :=  T3_XYZ[X',Y',Z] * (Intermediate2Term2[Y,Y',P])
    @tensoropt Term2[X,Y,Z] -= (T3_XYZ[X,Y',Z'] * Intermediate2Term2[Y,Y',P]) * (UvoX[e,k,Z'] * (UvoX[c,k,Z] * vvP[c,e,P])) #6
    Intermediate2Term2 = nothing
    t1 = print_time(EC,t1,"2 Chi terms in R3(T3)",2)
    @tensoropt Term2[X,Y,Z] += (ooP[l,i,P] * (UvoX[a,i,X] * UvoX[a,l,X'])) * (Intermediate3Term2[X',Y,Z,P] + Intermediate3Term2[X',Z,Y,P]) #7
    Intermediate3Term2 = nothing
    t1 = print_time(EC, t1, "3 Chi terms in R3(T3)", 2)
    @tensoropt Intermediate4Term2[l,d,a,i] := ovP[l,d,P] * (voP[a,i,P] + ovP[m,e,P] * TTilde[a,e,i,m])
    @tensoropt Term2[X,Y,Z] += UvoX[c,k,Z] * ((T3_XYZ[X',Y',Y] * UvoX[c,l,X']) * (UvoX[d,k,Y'] * (UvoX[a,i,X] * Intermediate4Term2[l,d,a,i]))) #8
    @tensoropt Term2[X,Y,Z] += UvoX[b,j,Y] * ((T3_XYZ[X',Y',Z] * UvoX[b,l,X']) * (UvoX[d,j,Y'] * (UvoX[a,i,X] * Intermediate4Term2[l,d,a,i]))) #9
    Intermediate4Term2 = nothing
    t1 = print_time(EC, t1, "4 Chi terms in R3(T3)", 2)
  end

  @tensoropt R3decomp[X,Y,Z] += Term2[X,Y,Z] + Term2[Y,X,Z] + Term2[Z,Y,X]
  Term2 = nothing
  t1 = print_time(EC, t1, "Symmetrization of Chi terms in R3(T3)", 2)

  #display(R3decomp)

  close(ovPfile)
  close(voPfile)
  close(ooPfile)
  close(vvPfile)

  save!(EC, "R_XXX", R3decomp)
  
end

end #module
