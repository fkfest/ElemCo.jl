@doc raw""" Coupled-cluster methods 

The following coupled-cluster methods are implemented in `ElemCo.jl`:
* `ccsd` - closed-shell implementation, for open-shell systems defaults to `uccsd`, 
* `uccsd` - unrestricted implementation,
* `rccsd` - restricted implementation (for high-spin RHF reference only),
* `ccsd(t)` - closed-shell implementation,
* `dcsd` - closed-shell implementation, for open-shell systems defaults to `udcsd`,
* `udcsd` - unrestricted implementation,
* `rdcsd` - restricted implementation (for high-spin RHF reference only),
* `λccsd` - calculation of Lagrange multipliers, closed-shell implementation,
* `λccsd(t)` - closed-shell implementation,
* `λdcsd` - calculation of Lagrange multipliers, closed-shell implementation.

The most efficient version of closed-shell CCSD/DCSD in `ElemCo.jl` combines the dressed factorization from [^Kats2013] with 
the `cckext` type of factorization from [^Hampel1992] and is given by
```math
\begin{align*}
\mathcal{L} &= v_{kl}^{cd} \tilde T^{kl}_{cd} + \left(\hat f_k^c + f_k^c\right) T^k_c
+ Λ_{ij}^{ab} \left(\hat v_{kl}^{ij} \red{+ v_{kl}^{cd} T^{ij}_{cd}}\right) T^{kl}_{ab}
+ Λ_{ij}^{ab} R^{ij}_{pq} δ_a^p δ_b^q 
\red{+Λ_{ij}^{ab} v_{kl}^{cd}T^{kj}_{ad}T^{il}_{cb}}\\
&+ Λ_{ij}^{ab} \mathcal{P}(ai;bj)\left\{\left(\hat f_a^c - \red{2\times}\frac{1}{2}v_{kl}^{cd} \tilde T^{kl}_{ad}\right)T^{ij}_{cb}
- \left(\hat f_k^i + \red{2\times}\frac{1}{2}v_{kl}^{cd}\tilde T^{il}_{cd}\right)T^{kj}_{ab} \right.\\
&+ \left(\hat v_{al}^{id}
+ \frac{1}{2} v_{kl}^{cd}\tilde T^{ik}_{ac}\right)\tilde T^{lj}_{db}
- \hat v_{ka}^{ic} T^{kj}_{cb} -\hat v_{kb}^{ic} T^{kj}_{ac}
\red{-v_{kl}^{cd}T^{ki}_{da}\left(T^{lj}_{cb}-T^{lj}_{bc}\right)}\\
&\left.- R^{ij}_{pq} \left(δ_k^p δ_b^q - \frac{1}{2} δ_k^p δ_l^q T^l_b\right) T^k_a \right\}
+Λ_i^a R^{ij}_{pq}\left( 2δ_a^p δ_j^q - δ_j^p δ_a^q \right)
-Λ_i^a T^k_a R^{ij}_{pq}\left( 2δ_k^p δ_j^q - δ_j^p δ_k^q \right)\\
&+Λ_i^a \hat h_a^i + Λ_i^a \hat f_j^b \tilde T^{ij}_{ab} 
- Λ_i^a \hat v_{jk}^{ic} \tilde T^{kj}_{ca},
\end{align*}
```
where
```math
R^{ij}_{pq} = v_{pq}^{rs} \left(\left(T^{ij}_{ab}+T^i_a T^j_b\right)δ_r^a δ_s^b 
+δ_r^i T^j_b δ_s^b + T^i_a δ_r^a δ_s^j + δ_r^i δ_s^j \right). 
```
The DCSD Lagrangian is obtained by removing terms in red.
Integrals with hats are dressed integrals, i.e. they are obtained by dressing the integrals with the singles amplitudes, e.g.,
``\hat v_{kl}^{id} = v_{kl}^{id} + v_{kl}^{cd} T^i_c``.

[^Kats2013]: D. Kats, and F.R. Manby, Sparse tensor framework for implementation of general local correlation methods, J. Chem. Phys. 138 (2013) 144101. doi:10.1063/1.4798940.

[^Hampel1992]: C. Hampel, K.A. Peterson, and H.-J. Werner, A comparison of the efficiency and accuracy of the quadratic configuration interaction (QCISD), coupled cluster (CCSD), and Brueckner coupled cluster (BCCD) methods, Chem. Phys. Lett. 190 (1992) 1. doi:10.1016/0009-2614(92)86093-W.


"""
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

export calc_MP2, calc_UMP2, calc_UMP2_energy 
export calc_cc, calc_pertT, calc_ΛpertT
export calc_lm_cc

include("cc_lagrange.jl")

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
    @tensoropt ET1 += 2.0*(T1[a,i] * load(EC,"f_mm")[SP['o'],SP['v']][i,a])
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
      @tensoropt ET1 += 0.5*((T1a[a,i]*T1a[b,j]-T1a[b,i]*T1a[a,j])*ints2(EC,"oovv")[i,j,a,b])
    end
    if length(T1b) > 0
      @tensoropt ET1 += 0.5*((T1b[a,i]*T1b[b,j]-T1b[b,i]*T1b[a,j])*ints2(EC,"OOVV")[i,j,a,b])
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
    ET2 = 0.5*(T2a[a,b,i,j] * ints2(EC,"oovv")[i,j,a,b])
    ET2 += 0.5*(T2b[a,b,i,j] * ints2(EC,"OOVV")[i,j,a,b])
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
    ET2 = fac*(T2[a,b,i,j] * int2[i,j,a,b])
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
    calc_dressed_ints(EC::ECInfo, T1, T12, o1::Char, v1::Char, o2::Char, v2::Char;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)

  Dress integrals with singles amplitudes. 

  The singles and orbspaces for first and second electron are `T1`, `o1`, `v1` and `T12`, `o2`, `v2`, respectively.
  The integrals from EC.fd are used and dressed integrals are stored as `d_????`.
  ``\\hat v_{ab}^{cd}``, ``\\hat v_{ab}^{ci}``, ``\\hat v_{ak}^{cd}`` and ``\\hat v_{ab}^{ij}`` are only 
  calculated if requested in `EC.options.cc` or using keyword-arguments.
"""
function calc_dressed_ints(EC::ECInfo, T1, T12, o1::Char, v1::Char, o2::Char, v2::Char;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)
  t1 = time_ns()
  mixed = (o1 != o2)
  no1, no2 = len_spaces(EC,o1*o2)
  # first make half-transformed integrals
  if calc_d_vvvv
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
  if calc_d_vvoo
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
  if calc_d_vvvo
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
  if calc_d_vovv
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
  if calc_d_vvvv
    # <ab\hat|cd>
    d_vvvv = load(EC,"hd_"*v1*v2*v1*v2)
    if !calc_d_vovv
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
  if calc_d_vvvo
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
  if calc_d_vvoo
    if !calc_d_vvvo
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
    calc_dressed_ints(EC::ECInfo, T1a, T1b=Float64[];
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)

  Dress integrals with singles.

  ``\\hat v_{ab}^{cd}``, ``\\hat v_{ab}^{ci}``, ``\\hat v_{ak}^{cd}`` and ``\\hat v_{ab}^{ij}`` are only 
  calculated if requested in `EC.options.cc` or using keyword-arguments.
"""
function calc_dressed_ints(EC::ECInfo, T1a, T1b=Float64[];
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)
  if ndims(T1b) != 2
    calc_dressed_ints(EC, T1a, T1a, "ovov"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
    dress_fock_closedshell(EC, T1a)
  else
    calc_dressed_ints(EC, T1a, T1a, "ovov"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
    calc_dressed_ints(EC, T1b, T1b, "OVOV"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
    calc_dressed_ints(EC, T1a, T1b, "ovOV"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
    dress_fock_samespin(EC, T1a, "ov"...)
    dress_fock_samespin(EC, T1b, "OV"...)
    dress_fock_oppositespin(EC)
  end
end

"""
    pseudo_dressed_ints(EC::ECInfo, unrestricted=false;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)

  Save non-dressed integrals in files instead of dressed integrals.
"""
function pseudo_dressed_ints(EC::ECInfo, unrestricted=false;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)
  #TODO write like in itf with chars as arguments, so three calls for three spin cases...
  t1 = time_ns()
  save!(EC,"d_oovo",ints2(EC,"oovo"))
  save!(EC,"d_voov",ints2(EC,"voov"))
  if calc_d_vovv
    save!(EC,"d_vovv",ints2(EC,"vovv"))
  end
  if calc_d_vvvv
    save!(EC,"d_vvvv",ints2(EC,"vvvv"))
  end
  save!(EC,"d_vovo",ints2(EC,"vovo"))
  save!(EC,"d_vooo",ints2(EC,"vooo"))
  if calc_d_vvvo
    save!(EC,"d_vvvo",ints2(EC,"vvvo"))
  end
  save!(EC,"d_oooo",ints2(EC,"oooo"))
  if calc_d_vvoo
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
  T2a = update_doubles(EC, ints2(EC,"vvoo"), spincase=:α, antisymmetrize = true, use_shift=false)
  T2b = update_doubles(EC, ints2(EC,"VVOO"), spincase=:β, antisymmetrize = true, use_shift=false)
  T2ab = update_doubles(EC, ints2(EC,"vVoO"), spincase=:αβ, use_shift=false)
  EMp2 = calc_doubles_energy(EC, T2a, T2b, T2ab)
  save!(EC, "T_vvoo", T2a)
  save!(EC, "T_VVOO", T2b)
  save!(EC, "T_vVoO", T2ab)
  if addsingles
    T1a = update_singles(EC,load(EC,"f_mm")[SP['v'],SP['o']], spincase=:α, use_shift=false)
    T1b = update_singles(EC,load(EC,"f_MM")[SP['V'],SP['O']], spincase=:β, use_shift=false)
    EMp2 += calc_singles_energy(EC, T1a, T1b, fock_only=true)
    save!(EC, "T_vo", T1a)
    save!(EC, "T_VO", T1b)
  end
  return EMp2
end

""" 
    calc_UMP2_energy(EC::ECInfo, addsingles=true)

  Calculate open-shell MP2 energy from precalculated amplitudes. 
  If `addsingles`: singles energy is also calculated.
  Return EMp2 
"""
function calc_UMP2_energy(EC::ECInfo, addsingles=true)
  T2a = load(EC,"T_vvoo")
  T2b = load(EC,"T_VVOO")
  T2ab = load(EC,"T_vVoO")
  EMp2 = calc_doubles_energy(EC, T2a, T2b, T2ab)
  if addsingles
    T1a = load(EC,"T_vo")
    T1b = load(EC,"T_VO")
    EMp2 += calc_singles_energy(EC, T1a, T1b, fock_only=true)
  end
  return EMp2
end

""" 
    calc_MP2(EC::ECInfo, addsingles=true)

  Calculate closed-shell MP2 energy and amplitudes. 
  The amplitudes are stored in `T_vvoo` file.
  If `addsingles`: singles are also calculated and stored in `T_vo` file.
  Return EMp2 
"""

""" 
    calc_D2(EC::ECInfo, T1, T2, scalepp=false)

  Calculate ``D^{ij}_{pq} = T^{ij}_{cd} + T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl}``.
  Return as `D[pqij]` 

  If `scalepp`: `D[ppij]` elements are scaled by 0.5 (for triangular summation).
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
    calc_D2(EC::ECInfo, T1, T2, spin::Symbol)

  Calculate ``^{σσ}D^{ij}_{pq} = T^{ij}_{cd} + P_{ij}(T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl})``
  with ``P_{ij} X_{ij} = X_{ij} - X_{ji}``.
  Return as `D[pqij]` 
"""
function calc_D2(EC::ECInfo, T1, T2, spin::Symbol)
  SP = EC.space
  norb = n_orbs(EC)
  if spin == :α
    virt = SP['v']
    occ = SP['o']
  else
    virt = SP['V']
    occ = SP['O']
  end
  nocc = length(occ)
  if length(T1) > 0
    D2 = Array{Float64}(undef,norb,norb,nocc,nocc)
  else
    D2 = zeros(norb,norb,nocc,nocc)
  end
  @tensoropt begin
    D2[virt,virt,:,:][a,b,i,j] = T2[a,b,i,j] 
    D2[occ,occ,:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k] - Matrix(I,nocc,nocc)[k,j] * Matrix(I,nocc,nocc)[l,i]
  end
  if length(T1) > 0
    @tensoropt begin
      D2[virt,virt,:,:][a,b,i,j] += T1[a,i] * T1[b,j] - T1[b,i] * T1[a,j]
      D2[occ,virt,:,:][j,a,i,k] = Matrix(I,nocc,nocc)[i,j] * T1[a,k] - Matrix(I,nocc,nocc)[k,j] * T1[a,i]
      D2[virt,occ,:,:][a,j,k,i] = Matrix(I,nocc,nocc)[i,j] * T1[a,k] - Matrix(I,nocc,nocc)[k,j] * T1[a,i]
    end
  end
  return D2
end

""" 
    calc_D2ab(EC::ECInfo, T1a, T1b, T2ab, scalepp=false)

  Calculate ``^{αβ}D^{ij}_{pq} = T^{ij}_{cd} + T^i_c T^j_d +δ_{ik} T^j_d + T^i_c δ_{jl} + δ_{ik} δ_{jl}``
  Return as `D[pqij]` 

  If `scalepp`: `D[ppij]` elements are scaled by 0.5 (for triangular summation)
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
    calc_ccsd_resid(EC::ECInfo, T1, T2; dc=false, tworef=false, fixref=false)

  Calculate CCSD or DCSD closed-shell residual.
"""
function calc_ccsd_resid(EC::ECInfo, T1, T2; dc=false, tworef=false, fixref=false)
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
  if EC.options.cc.use_kext
    int2 = integ2(EC.fd)
    if ndims(int2) == 4
      if EC.options.cc.triangular_kext
        trioo = [CartesianIndex(i,j) for j in 1:nocc for i in 1:j]
        D2 = calc_D2(EC, T1, T2)[:,:,trioo]
        # <pq|rs> D^ij_rs
        @tensoropt K2pqx[p,r,x] := int2[p,r,q,s] * D2[q,s,x]
        D2 = nothing
        K2pq = Array{Float64}(undef,norb,norb,nocc,nocc)
        K2pq[:,:,trioo] = K2pqx
        trioor = CartesianIndex.(reverse.(Tuple.(trioo)))
        @tensor K2pq[:,:,trioor][p,q,x] = K2pqx[q,p,x]
        K2pqx = nothing
      else
        D2 = calc_D2(EC, T1, T2)
        # <pq|rs> D^ij_rs
        @tensoropt K2pq[p,r,i,j] := int2[p,r,q,s] * D2[q,s,i,j]
        D2 = nothing
      end
    else
      # last two indices of integrals are stored as upper triangular 
      tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
      D2 = calc_D2(EC, T1, T2, true)[tripp,:,:]
      # <pq|rs> D^ij_rs
      @tensoropt rK2pq[p,r,i,j] := int2[p,r,x] * D2[x,i,j]
      D2 = nothing
      # symmetrize R
      @tensoropt K2pq[p,r,i,j] := rK2pq[p,r,i,j] + rK2pq[r,p,j,i]
      rK2pq = nothing
    end
    R2 += K2pq[SP['v'],SP['v'],:,:]
    if length(T1) > 0
      @tensoropt begin
        R2[a,b,i,j] -= K2pq[SP['o'],SP['v'],:,:][k,b,i,j] * T1[a,k]
        R2[a,b,i,j] -= K2pq[SP['v'],SP['o'],:,:][a,k,i,j] * T1[b,k]
        R2[a,b,i,j] += K2pq[SP['o'],SP['o'],:,:][k,l,i,j] * T1[a,k] * T1[b,l]
        # singles residual contributions
        R1[a,i] +=  2.0 * K2pq[SP['v'],SP['o'],:,:][a,k,i,k] - K2pq[SP['v'],SP['o'],:,:][a,k,k,i]
        x1[k,i] := 2.0 * K2pq[SP['o'],SP['o'],:,:][k,l,i,l] - K2pq[SP['o'],SP['o'],:,:][k,l,l,i]
        R1[a,i] -= x1[k,i] * T1[a,k]
      end
    end
    x1 = nothing
    K2pq = nothing
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
  # <kl|cd>\tilde T^ki_ca \tilde T^lj_db
  @tensoropt int2[a,k,i,c] += 0.5*klcd[k,l,c,d] * T2t[a,d,i,l] 
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
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC,"oovv"),[3,4,1,2])
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC,"vvvo")
  # ``v_{ia}^{jk}``
  ovoo = ints2(EC,"ovoo")
  nocc = n_occ_orbs(EC)
  nvir = n_virt_orbs(EC)
  ϵo, ϵv = orbital_energies(EC)
  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  IntY = zeros(nvir,nocc)
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
        T2ij = @view T2[:,:,i,j]
        T2ik = @view T2[:,:,i,k]
        T2jk = @view T2[:,:,j,k]
        T2i = @view T2[:,:,:,i]
        T2j = @view T2[:,:,:,j]
        T2k = @view T2[:,:,:,k]
        vvvk = @view vvvo[:,:,:,k]
        vvvj = @view vvvo[:,:,:,j]
        vvvi = @view vvvo[:,:,:,i]
        ovjk = @view ovoo[:,:,j,k]
        ovkj = @view ovoo[:,:,k,j]
        ovik = @view ovoo[:,:,i,k]
        ovki = @view ovoo[:,:,k,i]
        ovij = @view ovoo[:,:,i,j]
        ovji = @view ovoo[:,:,j,i]
        @tensoropt begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] := T2ij[a,d] * vvvk[b,c,d]
          Kijk[a,b,c] += T2ij[d,b] * vvvk[a,c,d]
          Kijk[a,b,c] += T2ik[a,d] * vvvj[c,b,d]
          Kijk[a,b,c] += T2ik[d,c] * vvvj[a,b,d]
          Kijk[a,b,c] += T2jk[b,d] * vvvi[c,a,d]
          Kijk[a,b,c] += T2jk[d,c] * vvvi[b,a,d]

          Kijk[a,b,c] -= T2i[b,a,l] * ovjk[l,c]
          Kijk[a,b,c] -= T2j[a,b,l] * ovik[l,c]
          Kijk[a,b,c] -= T2i[c,a,l] * ovkj[l,b]
          Kijk[a,b,c] -= T2k[a,c,l] * ovij[l,b]
          Kijk[a,b,c] -= T2j[c,b,l] * ovki[l,a]
          Kijk[a,b,c] -= T2k[b,c,l] * ovji[l,a]
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

        @tensoropt Enb3 += fac * (Kijk[a,b,c] * X[a,b,c])
        
        vv_jk = @view vv_oo[:,:,j,k]
        vv_ik = @view vv_oo[:,:,i,k]
        vv_ij = @view vv_oo[:,:,i,j]
        # julia 1.9 r1: cannot use @tensoropt begin/end here, since 
        # IntX[:,j] overwrites IntX[:,i] if j == i
        @tensoropt IntX[:,i][a] += fac * X[a,b,c] * vv_jk[b,c]
        @tensoropt IntX[:,j][b] += fac * X[a,b,c] * vv_ik[a,c]
        @tensoropt IntX[:,k][c] += fac * X[a,b,c] * vv_ij[a,b]
        @tensoropt IntY[:,i][a] += fac * X[a,b,c] * T2jk[b,c]
        @tensoropt IntY[:,j][b] += fac * X[a,b,c] * T2ik[a,c]
        @tensoropt IntY[:,k][c] += fac * X[a,b,c] * T2ij[a,b]
      end 
    end
  end
  if save_t3
    closemmap(EC,t3file,T3)
  end
  # singles contribution
  @tensoropt En3 = T1[a,i] * IntX[a,i]
  # fock contribution
  fov = load(EC,"f_mm")[EC.space['o'],EC.space['v']]
  @tensoropt En3 += fov[i,a] * IntY[a,i]
  En3 += Enb3
  return En3, Enb3
end

"""
    calc_ΛpertT(EC::ECInfo)

  Calculate (T) correction for closed-shell ΛCCSD(T).

  The amplitudes are stored in `T_vvoo` file, 
  and the Lagrangian multipliers are stored in `U_vvoo` file.
  Return ( (T) energy, [T] energy)
"""
function calc_ΛpertT(EC::ECInfo)
  T1 = load(EC,"T_vo")
  T2 = load(EC,"T_vvoo")
  U1 = load(EC,"U_vo")
  U2 = contra2covariant(load(EC,"U_vvoo"))
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC,"oovv"),[3,4,1,2])
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC,"vvvo")
  # ``v_{ia}^{jk}``
  ovoo = ints2(EC,"ovoo")
  # ``v_{ck}^{ab}``, reordered to ``v^{ab}_{ck}``
  vv_vo = permutedims(ints2(EC,"vovv"),[3,4,1,2])
  # ``v_{jk}^{ia}``, reordered to ``v^{ia}_{jk}``
  ov_oo = permutedims(ints2(EC,"ooov"),[3,4,1,2])
  nocc = n_occ_orbs(EC)
  nvir = n_virt_orbs(EC)
  ϵo, ϵv = orbital_energies(EC)
  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  IntY = zeros(nvir,nocc)
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
        T2ij = @view T2[:,:,i,j]
        T2ik = @view T2[:,:,i,k]
        T2jk = @view T2[:,:,j,k]
        T2i = @view T2[:,:,:,i]
        T2j = @view T2[:,:,:,j]
        T2k = @view T2[:,:,:,k]
        vvvk = @view vvvo[:,:,:,k]
        vvvj = @view vvvo[:,:,:,j]
        vvvi = @view vvvo[:,:,:,i]
        ovjk = @view ovoo[:,:,j,k]
        ovkj = @view ovoo[:,:,k,j]
        ovik = @view ovoo[:,:,i,k]
        ovki = @view ovoo[:,:,k,i]
        ovij = @view ovoo[:,:,i,j]
        ovji = @view ovoo[:,:,j,i]
        @tensoropt begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] := T2ij[a,d] * vvvk[b,c,d]
          Kijk[a,b,c] += T2ij[d,b] * vvvk[a,c,d]
          Kijk[a,b,c] += T2ik[a,d] * vvvj[c,b,d]
          Kijk[a,b,c] += T2ik[d,c] * vvvj[a,b,d]
          Kijk[a,b,c] += T2jk[b,d] * vvvi[c,a,d]
          Kijk[a,b,c] += T2jk[d,c] * vvvi[b,a,d]

          Kijk[a,b,c] -= T2i[b,a,l] * ovjk[l,c]
          Kijk[a,b,c] -= T2j[a,b,l] * ovik[l,c]
          Kijk[a,b,c] -= T2i[c,a,l] * ovkj[l,b]
          Kijk[a,b,c] -= T2k[a,c,l] * ovij[l,b]
          Kijk[a,b,c] -= T2j[c,b,l] * ovki[l,a]
          Kijk[a,b,c] -= T2k[b,c,l] * ovji[l,a]
        end
        @tensoropt  X[a,b,c] := 4.0*Kijk[a,b,c] - 2.0*Kijk[a,c,b] - 2.0*Kijk[c,b,a] - 2.0*Kijk[b,a,c] + Kijk[c,a,b] + Kijk[b,c,a]
        for abc ∈ CartesianIndices(X)
          a,b,c = Tuple(abc)
          X[abc] /= ϵo[i] + ϵo[j] + ϵo[k] - ϵv[a] - ϵv[b] - ϵv[c]
        end

        U2ij = @view U2[:,:,i,j]
        U2ik = @view U2[:,:,i,k]
        U2jk = @view U2[:,:,j,k]
        U2i = @view U2[:,:,:,i]
        U2j = @view U2[:,:,:,j]
        U2k = @view U2[:,:,:,k]
        vv_vk = @view vv_vo[:,:,:,k]
        vv_vj = @view vv_vo[:,:,:,j]
        vv_vi = @view vv_vo[:,:,:,i]
        ov_jk = @view ov_oo[:,:,j,k]
        ov_kj = @view ov_oo[:,:,k,j]
        ov_ik = @view ov_oo[:,:,i,k]
        ov_ki = @view ov_oo[:,:,k,i]
        ov_ij = @view ov_oo[:,:,i,j]
        ov_ji = @view ov_oo[:,:,j,i]
        @tensoropt begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = U2ij[a,d] * vv_vk[b,c,d]
          Kijk[a,b,c] += U2ij[d,b] * vv_vk[a,c,d]
          Kijk[a,b,c] += U2ik[a,d] * vv_vj[c,b,d]
          Kijk[a,b,c] += U2ik[d,c] * vv_vj[a,b,d]
          Kijk[a,b,c] += U2jk[b,d] * vv_vi[c,a,d]
          Kijk[a,b,c] += U2jk[d,c] * vv_vi[b,a,d]

          Kijk[a,b,c] -= U2i[b,a,l] * ov_jk[l,c]
          Kijk[a,b,c] -= U2j[a,b,l] * ov_ik[l,c]
          Kijk[a,b,c] -= U2i[c,a,l] * ov_kj[l,b]
          Kijk[a,b,c] -= U2k[a,c,l] * ov_ij[l,b]
          Kijk[a,b,c] -= U2j[c,b,l] * ov_ki[l,a]
          Kijk[a,b,c] -= U2k[b,c,l] * ov_ji[l,a]
        end
        @tensoropt Enb3 += fac * (Kijk[a,b,c] * X[a,b,c])
        
        vv_jk = @view vv_oo[:,:,j,k]
        vv_ik = @view vv_oo[:,:,i,k]
        vv_ij = @view vv_oo[:,:,i,j]
        # julia 1.9 r1: cannot use @tensoropt begin/end here, since 
        # IntX[:,j] overwrites IntX[:,i] if j == i
        @tensoropt IntX[:,i][a] += fac * X[a,b,c] * vv_jk[b,c]
        @tensoropt IntX[:,j][b] += fac * X[a,b,c] * vv_ik[a,c]
        @tensoropt IntX[:,k][c] += fac * X[a,b,c] * vv_ij[a,b]
        @tensoropt IntY[:,i][a] += fac * X[a,b,c] * U2jk[b,c]
        @tensoropt IntY[:,j][b] += fac * X[a,b,c] * U2ik[a,c]
        @tensoropt IntY[:,k][c] += fac * X[a,b,c] * U2ij[a,b]
      end 
    end
  end
  # singles contribution
  @tensoropt En3 = 0.5 * (U1[a,i] * IntX[a,i])
  # fock contribution
  fov = load(EC,"f_mm")[EC.space['o'],EC.space['v']]
  @tensoropt En3 += fov[i,a] * IntY[a,i]
  En3 += Enb3
  return En3, Enb3
end

"""
    calc_ccsd_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab; dc=false, tworef=false, fixref=false)

  Calculate UCCSD or UDCSD residual.
"""
function calc_ccsd_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab; dc=false, tworef=false, fixref=false)
  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nvirtb = n_virtb_orbs(EC)
  norb = n_orbs(EC)
  linearized::Bool = false

  if tworef
    morba, norbb, morbb, norba = active_orbitals(EC)
    T2ab[norba,morbb,morba,norbb] = 0.0
  end

  if ndims(T1a) == 2
    if !EC.options.cc.use_kext
      error("open-shell CCSD only implemented with kext")
    end
    calc_dressed_ints(EC,T1a,T1b)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC,true)
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
      @assert ndims(int2a) == 3 "Triangular storage of integrals expected!"
      D2a = calc_D2(EC, T1a, T2a, :α)[tripp,:,:]
      @tensoropt rK2pqa[p,r,i,j] := int2a[p,r,x] * D2a[x,i,j]
      D2a = nothing
      int2a = nothing
      # symmetrize R
      @tensoropt K2pqa[p,r,i,j] := rK2pqa[p,r,i,j] + rK2pqa[r,p,j,i]
      rK2pqa = nothing
      R2a += K2pqa[SP['v'],SP['v'],:,:]
      if n_occb_orbs(EC) > 0
        # ββ
        int2b = integ2(EC.fd,:β)
        D2b = calc_D2(EC, T1b, T2b, :β)[tripp,:,:]
        @tensoropt rK2pqb[p,r,i,j] := int2b[p,r,x] * D2b[x,i,j]
        D2b = nothing
        int2b = nothing
        # symmetrize R
        @tensoropt K2pqb[p,r,i,j] := rK2pqb[p,r,i,j] + rK2pqb[r,p,j,i]
        rK2pqb = nothing
        R2b += K2pqb[SP['V'],SP['V'],:,:]
        # αβ
        int2ab = integ2(EC.fd,:αβ)
        D2ab = calc_D2ab(EC, T1a, T1b, T2ab)
        @tensoropt K2pqab[p,r,i,j] := int2ab[p,r,q,s] * D2ab[q,s,i,j]
        D2ab = nothing
        int2ab = nothing
        R2ab += K2pqab[SP['v'],SP['V'],:,:]
      end
    else
      int2 = integ2(EC.fd)
      @assert ndims(int2) == 3 "Triangular storage of integrals expected!"
      # αα
      D2a = calc_D2(EC, T1a, T2a, :α)[tripp,:,:]
      @tensoropt rK2pqa[p,r,i,j] := int2[p,r,x] * D2a[x,i,j]
      D2a = nothing
      # symmetrize R
      @tensoropt K2pqa[p,r,i,j] := rK2pqa[p,r,i,j] + rK2pqa[r,p,j,i]
      rK2pqa = nothing
      R2a += K2pqa[SP['v'],SP['v'],:,:]
      if n_occb_orbs(EC) > 0
        # ββ
        D2b = calc_D2(EC, T1b, T2b, :β)[tripp,:,:]
        @tensoropt rK2pqb[p,r,i,j] := int2[p,r,x] * D2b[x,i,j]
        D2b = nothing
        # symmetrize R
        @tensoropt K2pqb[p,r,i,j] := rK2pqb[p,r,i,j] + rK2pqb[r,p,j,i]
        rK2pqb = nothing
        R2b += K2pqb[SP['V'],SP['V'],:,:]
        # αβ
        D2ab_full = calc_D2ab(EC, T1a, T1b, T2ab, true)
        D2ab = D2ab_full[tripp,:,:] 
        D2abT = permutedims(D2ab_full,(2,1,4,3))[tripp,:,:]
        D2ab_full = nothing
        @tensoropt K2pqab[p,r,i,j] := int2[p,r,x] * D2ab[x,i,j]
        @tensoropt K2pqab[p,r,i,j] += int2[r,p,x] * D2abT[x,j,i]
        D2ab = nothing
        D2abT = nothing
        R2ab += K2pqab[SP['v'],SP['V'],:,:]
      end
    end
    if length(T1a) > 0
      @tensoropt begin
        R2a[a,b,i,j] -= K2pqa[SP['o'],SP['v'],:,:][k,b,i,j] * T1a[a,k]
        R2a[a,b,i,j] -= K2pqa[SP['v'],SP['o'],:,:][a,k,i,j] * T1a[b,k]
        R2a[a,b,i,j] += K2pqa[SP['o'],SP['o'],:,:][k,l,i,j] * T1a[a,k] * T1a[b,l]
        # singles residual contributions
        R1a[a,i] +=  K2pqa[SP['v'],SP['o'],:,:][a,k,i,k] 
        x1a[k,i] :=  K2pqa[SP['o'],SP['o'],:,:][k,l,i,l]
        R1a[a,i] -= x1a[k,i] * T1a[a,k]
      end
    end
    if length(T1b) > 0
      @tensoropt begin
        R2b[a,b,i,j] -= K2pqb[SP['O'],SP['V'],:,:][k,b,i,j] * T1b[a,k]
        R2b[a,b,i,j] -= K2pqb[SP['V'],SP['O'],:,:][a,k,i,j] * T1b[b,k]
        R2b[a,b,i,j] += K2pqb[SP['O'],SP['O'],:,:][k,l,i,j] * T1b[a,k] * T1b[b,l]
        # singles residual contributions
        R1b[a,i] += K2pqb[SP['V'],SP['O'],:,:][a,k,i,k]
        x1b[k,i] := K2pqb[SP['O'],SP['O'],:,:][k,l,i,l]
        R1b[a,i] -= x1b[k,i] * T1b[a,k]
      end
    end
    if n_occ_orbs(EC) > 0 && n_occb_orbs(EC) > 0 && length(T1a) > 0
      @tensoropt begin
        R2ab[a,b,i,j] -= K2pqab[SP['o'],SP['V'],:,:][k,b,i,j] * T1a[a,k]
        R2ab[a,b,i,j] -= K2pqab[SP['v'],SP['O'],:,:][a,k,i,j] * T1b[b,k]
        R2ab[a,b,i,j] += K2pqab[SP['o'],SP['O'],:,:][k,l,i,j] * T1a[a,k] * T1b[b,l]
        R1a[a,i] += K2pqab[SP['v'],SP['O'],:,:][a,k,i,k] 
        x1a1[k,i] := K2pqab[SP['o'],SP['O'],:,:][k,l,i,l]
        R1a[a,i] -= x1a1[k,i] * T1a[a,k]
        R1b[a,i] += K2pqab[SP['o'],SP['V'],:,:][k,a,k,i] 
        x1b1[k,i] := K2pqab[SP['o'],SP['O'],:,:][l,k,l,i]
        R1b[a,i] -= x1b1[k,i] * T1b[a,k]
      end
    end
    (K2pqa, K2pqb, K2pqab) = (nothing, nothing, nothing)
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
  A_d_voov = load(EC,"d_voov") - permutedims(load(EC,"d_vovo"),(1,2,4,3))
  @tensoropt begin
    rR2a[a,b,i,j] := A_d_voov[b,k,j,c] * T2a[a,c,i,k]
    R2ab[a,B,i,J] += A_d_voov[a,k,i,c] * T2ab[c,B,k,J]
  end
  A_d_voov = nothing
  d_vOoV = load(EC,"d_vOoV")
  @tensoropt begin
    rR2a[a,b,i,j] += d_vOoV[b,K,j,C] * T2ab[a,C,i,K]
    R2ab[a,B,i,J] += d_vOoV[a,K,i,C] * T2b[B,C,J,K]
  end
  d_vOoV = nothing
  @tensoropt R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i] - rR2a[a,b,j,i] - rR2a[b,a,i,j]
  rR2a = nothing
  if n_occb_orbs(EC) > 0
    A_d_VOOV = load(EC,"d_VOOV") - permutedims(load(EC,"d_VOVO"),(1,2,4,3))
    @tensoropt begin
      rR2b[A,B,I,J] := A_d_VOOV[B,K,J,C] * T2b[A,C,I,K]
      R2ab[a,B,i,J] += A_d_VOOV[B,K,J,C] * T2ab[a,C,i,K]
    end
    A_d_VOOV = nothing
    d_oVvO = load(EC,"d_oVvO")
    @tensoropt begin
      rR2b[A,B,I,J] += d_oVvO[k,B,c,J] * T2ab[c,A,k,I]
      R2ab[a,B,i,J] += d_oVvO[k,B,c,J] * T2a[a,c,i,k]
    end
    d_oVvO = nothing
    @tensoropt R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I] - rR2b[A,B,J,I] - rR2b[B,A,I,J]
    rR2b = nothing
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
        M1a = calc_M1a(occcore, virtuals, T1a, T1b, T2b, T2ab, activeorbs)
        M1b = calc_M1b(occcore, virtuals, T1a, T1b, T2a, T2ab, activeorbs)
        @tensoropt R1a[a,i] += M1a[a,i] * W
        @tensoropt R1b[a,i] += M1b[a,i] * W
      end
      if !isempty(occcorea) && !isempty(occcoreb)
        M2a = calc_M2a(occcore, virtuals, T1a, T1b, T2b, T2ab, activeorbs)
        M2b = calc_M2b(occcore, virtuals, T1a, T1b, T2a, T2ab, activeorbs)
        @tensoropt R2a[a,b,i,j] += M2a[a,b,i,j] * W
        @tensoropt R2b[a,b,i,j] += M2b[a,b,i,j] * W
      end
      M2ab = calc_M2ab(occcore, virtuals, T1a, T1b, T2a, T2b, T2ab, activeorbs)
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

function calc_M1a(occcore, virtuals, T1a, T1b, T2b, T2ab, activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  internalT1a = T1a[norba,morba]
  M1 = zeros(Float64,size(T1a))
  if !isempty(occcorea) && !isempty(occcoreb)
    @tensoropt M1[norba,occcorea][i] += T2ab[norba,morbb,morba,occcoreb][i]
    @tensoropt M1[norba,occcorea][i] += internalT1a * T1b[morbb,occcoreb][i]
    @tensoropt M1[virtualsa,occcorea][a,i] += T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a]
    @tensoropt M1[virtualsa,occcorea][a,i] += internalT1a * T2b[morbb,virtualsb,occcoreb,norbb][a,i]
    @tensoropt M1[virtualsa,occcorea][a,i] += T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i]
    @tensoropt M1[virtualsa,occcorea][a,i] += internalT1a * T2b[morbb,virtualsb,occcoreb,norbb][a,i]
  end
  @tensoropt M1[virtualsa,morba][a] -= T2ab[norba,virtualsb,morba,norbb][a]
  @tensoropt M1[virtualsa,morba][a] -= internalT1a * T1b[virtualsb,norbb][a]
  M1[norba,morba] += internalT1a
  return M1
end

function calc_M1b(occcore, virtuals, T1a, T1b, T2a, T2ab, activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M1 = zeros(Float64,size(T1b))
  internalT1b = T1b[morbb,norbb]
  if !isempty(occcorea) && !isempty(occcoreb)
    @tensoropt M1[morbb,occcoreb][i] += T2ab[norba,morbb,occcorea,norbb][i]
    @tensoropt M1[morbb,occcoreb][i] += internalT1b * T1a[norba,occcorea][i]
    @tensoropt M1[virtualsb,occcoreb][a,i] += T2ab[norba,morbb,occcorea,norbb][i] * T1a[virtualsa,morba][a]
    @tensoropt M1[virtualsb,occcoreb][a,i] += internalT1b * T2a[norba,virtualsa,occcorea,morba][a,i]
    @tensoropt M1[virtualsb,occcoreb][a,i] += T2ab[virtualsa,morbb,morba,norbb][a] * T1a[norba,occcorea][i]
    @tensoropt M1[virtualsb,occcoreb][a,i] += internalT1b * T2a[virtualsa,norba,morba,occcorea][a,i]
  end
  @tensoropt M1[virtualsb,norbb][a] -= T2ab[virtualsa,morbb,morba,norbb][a]
  @tensoropt M1[virtualsb,norbb][a] -= internalT1b * T1a[virtualsa,morba][a]
  M1[morbb,norbb] += internalT1b
  return M1
end

function calc_M2a(occcore,virtuals,T1a,T1b,T2b,T2ab,activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M2 = zeros(Float64,size(T2b))
  if length(T1a) > 0
    internalT1a = T1a[norba,morba]
    internalT1b = T1b[morbb,norbb]
    @tensoropt TT1a[a,i] := T1a[virtualsa,occcorea][a,i] - T1b[virtualsb,occcoreb][a,i]
    @tensoropt TT1b[a,i] := T1b[virtualsb,occcoreb][a,i] - T1a[virtualsa,occcorea][a,i]

    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * TT1a[a,j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,j,i] += T2ab[norba,morbb,morba,occcoreb][i] * TT1a[a,j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,i,j] += T2ab[norba,morbb,morba,occcoreb][i] * TT1a[a,j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,j,i] -= T2ab[norba,morbb,morba,occcoreb][i] * TT1a[a,j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,j,i] += T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[morbb,occcoreb][j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,i,j] += T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[morbb,occcoreb][j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,j,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= internalT1a * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,i,j] += internalT1a * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]

    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += T2ab[norba,virtualsb,morba,norbb][a] * TT1a[b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= T2ab[norba,virtualsb,morba,norbb][a] * TT1a[b,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][a,b,i] -= T2ab[norba,virtualsb,morba,norbb][a] * TT1a[b,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][b,a,i] += T2ab[norba,virtualsb,morba,norbb][a] * TT1a[b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][a,b,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][b,a,i] += T2ab[norba,virtualsb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += internalT1a * T2b[virtualsb,virtualsb,norbb,occcoreb][b,a,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][a,b,i] -= internalT1a * T2b[virtualsb,virtualsb,norbb,occcoreb][b,a,i]
  
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * TT1a[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * TT1a[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * TT1a[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * TT1a[b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * TT1a[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * TT1a[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * TT1a[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * TT1a[b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1b * TT1b[a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += internalT1b * TT1b[a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += internalT1b * TT1b[a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= internalT1b * TT1b[a,i] * T2ab[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1a * TT1b[a,i] * T2b[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += internalT1a * TT1b[a,i] * T2b[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += internalT1a * TT1b[a,i] * T2b[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= internalT1a * TT1b[a,i] * T2b[morbb,virtualsb,norbb,occcoreb][b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1a * T1b[morbb,occcoreb][j] * T2b[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += internalT1a * T1b[morbb,occcoreb][j] * T2b[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
    
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1a * T1b[virtualsb,norbb][b] * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += internalT1a * T1b[virtualsb,norbb][b] * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
    
    @tensoropt M2[virtualsa,norba,occcorea,morba][a,i] -= internalT1a * TT1b[a,i]
    @tensoropt M2[norba,virtualsa,occcorea,morba][a,i] += internalT1a * TT1b[a,i]
    @tensoropt M2[virtualsa,norba,morba,occcorea][a,i] += internalT1a * TT1b[a,i]
    @tensoropt M2[norba,virtualsa,morba,occcorea][a,i] -= internalT1a * TT1b[a,i]
  end
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,morbb,morba,occcoreb][j] * T2b[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[norba,morbb,morba,occcoreb][j] * T2b[virtualsb,virtualsb,norbb,occcoreb][a,b,i]

  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[norba,virtualsb,morba,norbb][b] * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[norba,virtualsb,morba,norbb][b] * T2b[morbb,virtualsb,occcoreb,occcoreb][a,i,j]

  @tensoropt M2[virtualsa,norba,occcorea,morba][a,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i]
  @tensoropt M2[virtualsa,norba,morba,occcorea][a,i] += T2ab[norba,virtualsb,morba,occcoreb][a,i]
  @tensoropt M2[norba,virtualsa,occcorea,morba][a,i] += T2ab[norba,virtualsb,morba,occcoreb][a,i]
  @tensoropt M2[norba,virtualsa,morba,occcorea][a,i] -= T2ab[norba,virtualsb,morba,occcoreb][a,i]
  return M2
end

function calc_M2b(occcore,virtuals,T1a,T1b,T2a,T2ab,activeorbs)
#NOTE that we intentionally unpack in different order to reuse code from calc_M2a as much as possible
# morba, norbb, morbb, norba = activeorbs
  norbb, morba, norba, morbb = activeorbs
  occcoreb, occcorea = occcore
  virtualsb, virtualsa = virtuals
#ENDNOTE
  P12 = (2,1,4,3)
  M2 = zeros(Float64,size(T2a))
  if length(T1a) > 0
    internalT1a = T1a[morbb,norbb]
    internalT1b = T1b[norba,morba]
    @tensoropt TT1a[a,i] := T1a[virtualsb,occcoreb][a,i] - T1b[virtualsa,occcorea][a,i]
    @tensoropt TT1b[a,i] := T1b[virtualsa,occcorea][a,i] - T1a[virtualsb,occcoreb][a,i]

    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * TT1b[a,j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,j,i] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * TT1b[a,j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,i,j] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * TT1b[a,j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,j,i] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * TT1b[a,j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,j,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[morbb,occcoreb][j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,i,j] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[morbb,occcoreb][j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,j,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsa,occcorea,occcorea][a,i,j] -= internalT1b * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
    @tensoropt M2[virtualsa,norba,occcorea,occcorea][a,i,j] += internalT1b * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]

    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * TT1b[b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * TT1b[b,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][a,b,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * TT1b[b,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][b,a,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * TT1b[b,i]
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][b,a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[virtualsb,norbb][b]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][a,b,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][b,a,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i] * T1a[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsa,morba,occcorea][a,b,i] += internalT1b * T2a[virtualsb,virtualsb,norbb,occcoreb][b,a,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,morba][a,b,i] -= internalT1b * T2a[virtualsb,virtualsb,norbb,occcoreb][b,a,i]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][i] * T1a[virtualsb,norbb][a] * TT1b[b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][a] * T1a[morbb,occcoreb][i] * TT1b[b,j]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= T2ab[virtualsb,norba,occcoreb,morba][a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += T2ab[virtualsb,norba,occcoreb,morba][a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += T2ab[virtualsb,norba,occcoreb,morba][a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= T2ab[virtualsb,norba,occcoreb,morba][a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1a * TT1a[a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += internalT1a * TT1a[a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += internalT1a * TT1a[a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= internalT1a * TT1a[a,i] * permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1b * TT1a[a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += internalT1b * TT1a[a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += internalT1b * TT1a[a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,j,i] -= internalT1b * TT1a[a,i] * T2a[morbb,virtualsb,norbb,occcoreb][b,j]
    
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1b * T1a[morbb,occcoreb][j] * T2a[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += internalT1b * T1a[morbb,occcoreb][j] * T2a[virtualsb,virtualsb,norbb,occcoreb][a,b,i]

    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= internalT1b * T1a[virtualsb,norbb][b] * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
    @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += internalT1b * T1a[virtualsb,norbb][b] * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]

    @tensoropt M2[virtualsa,norba,occcorea,morba][a,i] -= internalT1b * TT1a[a,i]
    @tensoropt M2[norba,virtualsa,occcorea,morba][a,i] += internalT1b * TT1a[a,i]
    @tensoropt M2[virtualsa,norba,morba,occcorea][a,i] += internalT1b * TT1a[a,i]
    @tensoropt M2[norba,virtualsa,morba,occcorea][a,i] -= internalT1b * TT1a[a,i]
  end
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][j] * T2a[virtualsb,virtualsb,norbb,occcoreb][a,b,i]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,j,i] += permutedims(T2ab,P12)[norba,morbb,morba,occcoreb][j] * T2a[virtualsb,virtualsb,norbb,occcoreb][a,b,i]

  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][a,b,i,j] -= permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][b] * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]
  @tensoropt M2[virtualsa,virtualsa,occcorea,occcorea][b,a,i,j] += permutedims(T2ab,P12)[norba,virtualsb,morba,norbb][b] * T2a[morbb,virtualsb,occcoreb,occcoreb][a,i,j]

  @tensoropt M2[virtualsa,norba,occcorea,morba][a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i]
  @tensoropt M2[virtualsa,norba,morba,occcorea][a,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i]
  @tensoropt M2[norba,virtualsa,occcorea,morba][a,i] += permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i]
  @tensoropt M2[norba,virtualsa,morba,occcorea][a,i] -= permutedims(T2ab,P12)[norba,virtualsb,morba,occcoreb][a,i]
  return M2
end

function calc_M2ab(occcore,virtuals,T1a,T1b,T2a,T2b,T2ab,activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M2 = zeros(Float64,size(T2ab))
  # @assert isapprox(T2a[norba,virtualsa,morba,occcorea],-T2a[virtualsa,norba,morba,occcorea];atol=1.e-8)
  # @assert isapprox(T2a[norba,virtualsa,morba,occcorea],-T2a[norba,virtualsa,occcorea,morba];atol=1.e-8)
  # @assert isapprox(T2b[morbb,virtualsb,norbb,occcoreb],-T2b[virtualsb,morbb,norbb,occcoreb];atol=1.e-8)
  # @assert isapprox(T2b[morbb,virtualsb,norbb,occcoreb],-T2b[morbb,virtualsb,occcoreb,norbb];atol=1.e-8)
  if length(T1a) > 0
    internalT1a = T1a[norba,morba]
    internalT1b = T1b[morbb,norbb]
    @tensoropt TT1a[a,i] := T1a[virtualsa,occcorea][a,i] - T1b[virtualsb,occcoreb][a,i]
    @tensoropt TT1b[a,i] := T1b[virtualsb,occcoreb][a,i] - T1a[virtualsa,occcorea][a,i]
    @tensoropt T2ta[a,b,i,j] := T2a[a,b,i,j] + T1a[a,i] * T1a[b,j]
    @tensoropt T2tb[a,b,i,j] := T2b[a,b,i,j] + T1b[a,i] * T1b[b,j]
    @tensoropt T2tab[a,b,i,j] := T2ab[a,b,i,j] + T1a[a,i] * T1b[b,j]
    @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * TT1b[a,j]
    @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] -= T2ab[norba,morbb,occcorea,norbb][i] * TT1a[a,j]
    @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2ab[virtualsa,morbb,morba,occcoreb][a,i] * T1a[norba,occcorea][j]
    @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] -= T2ab[norba,virtualsb,occcorea,norbb][a,i] * T1b[morbb,occcoreb][j]
    @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2tab[norba,morbb,occcorea,occcoreb][j,i] * T1a[virtualsa,morba][a]
    @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] -= T2tab[norba,morbb,occcorea,occcoreb][i,j] * T1b[virtualsb,norbb][a]
    @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] -= T2a[virtualsa,norba,morba,occcorea][a,j] * T1b[morbb,occcoreb][i]
    @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] -= T2b[morbb,virtualsb,occcoreb,norbb][a,j] * T1a[norba,occcorea][i]
    @tensoropt M2[norba,virtualsb,occcorea,occcoreb][a,i,j] += internalT1a * T2ab[virtualsa,morbb,occcorea,occcoreb][a,j,i]
    @tensoropt M2[virtualsa,morbb,occcorea,occcoreb][a,j,i] += internalT1b * T2ab[norba,virtualsb,occcorea,occcoreb][a,i,j]

    @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2ab[norba,virtualsb,morba,norbb][a] * TT1b[b,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] += T2ab[virtualsa,morbb,morba,norbb][a] * TT1a[b,i]
    @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2ab[norba,virtualsb,occcorea,norbb][a,i] * T1a[virtualsa,morba][b] 
    @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] += T2ab[virtualsa,morbb,morba,occcoreb][a,i] * T1b[virtualsb,norbb][b] 
    @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2tab[virtualsa,virtualsb,morba,norbb][b,a] * T1a[norba,occcorea][i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] += T2tab[virtualsa,virtualsb,morba,norbb][a,b] * T1b[morbb,occcoreb][i] 
    @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] += T2a[norba,virtualsa,occcorea,morba][b,i] * T1b[virtualsb,norbb][a]
    @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] += T2b[morbb,virtualsb,occcoreb,norbb][b,i] * T1a[virtualsa,morba][a]
    @tensoropt M2[virtualsa,virtualsb,morba,occcoreb][a,b,i] -= internalT1a * T2ab[virtualsa,virtualsb,occcorea,norbb][b,a,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,norbb][b,a,i] -= internalT1b * T2ab[virtualsa,virtualsb,morba,occcoreb][a,b,i]

    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,morbb,morba,occcoreb][i] * T1b[virtualsb,norbb][a] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,morbb,occcorea,norbb][j] * T1a[virtualsa,morba][b] * TT1a[a,i]

    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,virtualsb,morba,norbb][a] * T1b[morbb,occcoreb][i] * TT1b[b,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[virtualsa,morbb,morba,norbb][b] * T1a[norba,occcorea][j] * TT1a[a,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2a[norba,virtualsa,morba,occcorea][b,j] * T1b[virtualsb,norbb][a] * T1b[morbb,occcoreb][i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T1a[norba,occcorea][j] * T1a[virtualsa,morba][b]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[virtualsa,morbb,morba,occcoreb][b,i] * T1a[norba,occcorea][j] * T1b[virtualsb,norbb][a]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2ab[norba,virtualsb,occcorea,norbb][a,j] * T1a[virtualsa,morba][b] * T1b[morbb,occcoreb][i]

    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= internalT1a * TT1b[b,j] * T2b[virtualsb,morbb,norbb,occcoreb][a,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= internalT1b * TT1a[a,i] * T2a[norba,virtualsa,occcorea,morba][b,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += internalT1a * T1b[morbb,occcoreb][i] * T2ab[virtualsa,virtualsb,occcorea,norbb][b,a,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += internalT1a * T1b[virtualsb,norbb][a] * T2ab[virtualsa,morbb,occcorea,occcoreb][b,j,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += internalT1b * T1a[virtualsa,morba][b] * T2ab[norba,virtualsb,occcorea,occcoreb][a,j,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += internalT1b * T1a[norba,occcorea][j] * T2ab[virtualsa,virtualsb,morba,occcoreb][b,a,i]

    @tensoropt M2[norba,morbb,morba,occcoreb][i] += T1a[norba,occcorea][i]
    @tensoropt M2[norba,morbb,occcorea,norbb][i] += T1b[morbb,occcoreb][i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2tab[norba,morbb,occcorea,occcoreb][j,i] * T2tab[virtualsa,virtualsb,morba,norbb][b,a]
    @tensoropt M2[norba,morbb,occcorea,occcoreb][i,j] -= T2tab[norba,morbb,occcorea,occcoreb][j,i]
    @tensoropt M2[norba,virtualsb,morba,occcoreb][a,i] += T2ta[norba,virtualsa,occcorea,morba][a,i]
    @tensoropt M2[virtualsa,morbb,occcorea,norbb][a,i] += T2tb[morbb,virtualsb,occcoreb,norbb][a,i]
    @tensoropt M2[virtualsa,morbb,morba,occcoreb][a,i] += T2tab[norba,virtualsb,occcorea,norbb][a,i]
    @tensoropt M2[norba,virtualsb,occcorea,norbb][a,i] += T2tab[virtualsa,morbb,morba,occcoreb][a,i]
    if !isempty(occcorea) && !isempty(occcoreb)
      @tensoropt M2[norba,virtualsb,morba,norbb][a] -= T1a[virtualsa,morba][a]
      @tensoropt M2[virtualsa,morbb,morba,norbb][a] -= T1b[virtualsb,norbb][a]
      @tensoropt M2[virtualsa,virtualsb,morba,norbb][a,b] -= T2tab[virtualsa,virtualsb,morba,norbb][b,a]
    end
  end
  if !isempty(occcorea) && !isempty(occcoreb)
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[norba,morbb,occcorea,norbb][j] * T2ab[virtualsa,virtualsb,morba,occcoreb][b,a,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[norba,morbb,morba,occcoreb][i] * T2ab[virtualsa,virtualsb,occcorea,norbb][b,a,j]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[virtualsa,morbb,morba,norbb][b] * T2ab[norba,virtualsb,occcorea,occcoreb][a,j,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] += T2ab[norba,virtualsb,morba,norbb][a] * T2ab[virtualsa,morbb,occcorea,occcoreb][b,j,i]
    @tensoropt M2[virtualsa,virtualsb,occcorea,occcoreb][a,b,i,j] -= T2b[morbb,virtualsb,norbb,occcoreb][a,i] * T2a[norba,virtualsa,morba,occcorea][b,j]
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
  dc = (method.theory == "DC")
  tworef = has_prefix(method, "2D")
  fixref = (has_prefix(method, "FRS") || has_prefix(method, "FRT"))
  restrict = has_prefix(method, "R")
  print_info(method_name(method))
  Amps, exc_ranges = starting_amplitudes(EC, method)
  singles, doubles, triples = exc_ranges[1:3]
  if is_unrestricted(method) || has_prefix(method, "R")
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
  Eias = 0.0
  converged = false
  t0 = time_ns()
  println("Iter     SqNorm      Energy      DE          Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    Res = calc_ccsd_resid(EC, Amps...; dc, tworef, fixref)
    if restrict
      spin_project!(EC, Res...)
    end
    t1 = print_time(EC, t1, "residual", 2)
    NormT2 = calc_doubles_norm(Amps[doubles]...)
    NormR2 = calc_doubles_norm(Res[doubles]...)
    Eh = calc_hylleraas(EC, Amps..., Res...)
    update_doubles!(EC, Amps[doubles]..., Res[doubles]...)
    if has_prefix(method, "FRS")
      morba, norbb, morbb, norba = active_orbitals(EC)
      Amps[T2αβ][norba,morbb,morba,norbb] = 1.0
    elseif has_prefix(method, "FRT")
      morba, norbb, morbb, norba = active_orbitals(EC)
      Amps[T2αβ][norba,morbb,morba,norbb] = -1.0
    elseif has_prefix(method, "2D") && do_sing
      morba, norbb, morbb, norba = active_orbitals(EC)
      T1α = first(singles)
      T1β = last(singles)
      W = load(EC,"2d_ccsd_W")[1]
      Eias = - W * Amps[T1α][norba,morba] * Amps[T1β][morbb,norbb]
    end
    if do_sing
      NormT1 = calc_singles_norm(Amps[singles]...)
      NormR1 = calc_singles_norm(Res[singles]...)
      update_singles!(EC, Amps[singles]..., Res[singles]...)
    end
    if restrict
      spin_project!(EC, Amps...)
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
  if has_prefix(method, "2D") && do_sing
    return Eh+Eias
  else
    return Eh
  end
end

""" 
    calc_ccsdt(EC::ECInfo, ccsd_pertt_energy, useT3=false, cc3=false)

  Calculate decomposed closed-shell DC-CCSDT amplitudes.

  If `useT3`: (T) amplitudes from a preceding calculations will be used as starting guess.
  If cc3: calculate CC3 amplitudes.
"""
function calc_ccsdt(EC::ECInfo, ccsd_pertt_energy, useT3=false, cc3=false)
  
  pert_svd_T = true
  
  if cc3
    print_info("CC3")
  end
  if pert_svd_T
    print_info("DC-CCSDT")
    println("DC-CCSDT with SVD-(T)")
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


# Charlottes block for svd-ccsd(t)  
  if pert_svd_T
    nocc = n_occ_orbs(EC)
    nvirt = n_virt_orbs(EC)
  
    calc_undressed_3idx(EC)
    calc_SVD_pert_T(EC, T2)
    R3 = load(EC, "R_XXX")
    T3 = load(EC, "T_XXX")
    T3 += update_deco_triples(EC, R3, false)
    save!(EC, "T_XXX", T3)
    
    R1 = zeros(nvirt,nocc)
    R2 = zeros(nvirt, nvirt, nocc, nocc)
    R1, R2 = SVD_pert_T_add_to_singles_and_doubles_residuals(EC, R1, R2)
  
    Eh_init = calc_hylleraas(EC, T1, T2, R1, R2)
    @printf "SVD-CCSD(T) energy:  %20.12f \n" Eh_init    
  end

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
    #comment DIIS out for testing purpose
    T1, T2, T3 = perform(diis, [T1,T2,T3], [R1,R2,R3])
    
    iii_aaa_remove = false
    if iii_aaa_remove == true
        #change to avoid potential iii and aaa contributions (also changed after amplitudes DIIS update)
        #attention! no complex conjugation is taken into account so far!
 
        @tensoropt TriplesDotProductOne = T3[X,Y,Z] * T3[X,Y,Z]
        println("Triples DotProduct without substraction:")
        println(TriplesDotProductOne)
 
        T_iii = zeros(nvirt,nvirt,nvirt)
        println("zero_T_iii:")
	display(T_iii)
	Diff_T_iii_aaa = zero(T3)
        
        UvoX = load(EC, "C_voX")
        for i in 1:nocc
           Ui = UvoX[:,i,:]
           println("Ui:")
           display(Ui)
	   @tensoropt T_iii[a,b,c] = (((T3[X,Y,Z] * Ui[a,X]) * Ui[b,Y]) * Ui[c,Z])

           for a in 1:nvirt
              T_iii[a,a,a] = 0.0
           end
           println("T_iii:")
	   display(T_iii)
	   @tensoropt Diff_T_iii_aaa[X,Y,Z] += (((T_iii[a,b,c] * Ui[a,X]) * Ui[b,Y]) * Ui[c,Z])
        end
 
        T_aaa = zeros(nocc,nocc,nocc)
 
        for a in 1:nvirt
           Ua = UvoX[a,:,:]
           @tensoropt T_aaa[i,j,k] = (((T3[X,Y,Z] * Ua[i,X]) * Ua[j,Y]) * Ua[k,Z])
           @tensoropt Diff_T_iii_aaa[X,Y,Z] += (((T_aaa[i,j,k] * Ua[i,X]) * Ua[j,Y]) * Ua[k,Z])
        end
        T3 .-= Diff_T_iii_aaa
 
        @tensoropt TriplesDotProductTwo = T3[X,Y,Z] * T3[X,Y,Z]
        println("Triples DotProduct with substraction:")
        println(TriplesDotProductTwo)
    end
	
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

  if pert_svd_T
    @printf "SVD-DC-CCSDT energy:  %20.12f \n" Eh
    @printf "SVD-DC-CCSDT - SVD-CCSD(T):  %20.12f \n" (Eh - Eh_init)
    @printf "CCSD(T):  %20.12f \n" ccsd_pertt_energy
    @printf "CCSD(T) - SVD-CCSD(T) + SVD-DC-CCSDT:  %20.12f \n" (ccsd_pertt_energy - Eh_init + Eh)
  end

  @printf "Sq.Norm of T1: %12.8f Sq.Norm of T2: %12.8f Sq.Norm of T3: %12.8f \n" NormT1 NormT2 NormT3
  println()
  flush(stdout)
  
  return Eh
end


"""
    SVD_pert_T_add_to_singles_and_doubles_residuals(EC, R1, R2)

  Add contributions for SVD-(T) from triples to singles and doubles residuals.
"""
function SVD_pert_T_add_to_singles_and_doubles_residuals(EC, R1, R2)
  SP = EC.space
  notd_ooPfile, notd_ooP = mmap(EC, "notd_ooL")
  notd_ovPfile, notd_ovP = mmap(EC, "notd_ovL")

  Txyz = load(EC, "T_XXX")

  U = load(EC, "C_voX")
  # println(size(U))

  @tensoropt Boo[i,j,P,X] := notd_ovP[i,a,P] * U[a,j,X]
  @tensoropt A[P,X] := Boo[i,i,P,X]
  @tensoropt BBU[Z,d,j] := (notd_ovP[j,c,P] * notd_ovP[k,d,P]) * U[c,k,Z]
  @tensoropt R1[a,i] += U[a,i,X] *(Txyz[X,Y,Z] *( 2.0*A[P,Y] * A[P,Z] - Boo[j,k,P,Z] * Boo[k,j,P,Y] ))
  @tensoropt R1[a,i] -= U[a,j,Y] *( 2.0*Boo[j,i,P,X]*(Txyz[X,Y,Z] * A[P,Z]) - Txyz[X,Y,Z] *(U[d,i,X]*BBU[Z,d,j] ))

  BBU = nothing

  @tensoropt Bov[i,a,P,X] := notd_ooP[j,i,P] * U[a,j,X]
  notd_vvPfile, notd_vvP = mmap(EC, "notd_vvL")
  @tensoropt Bvo[a,i,P,X] := notd_vvP[a,b,P] * U[b,i,X]
  close(notd_vvPfile)
  notd_vvP = nothing
  #dfock = load(EC, "df_mm")
  #fov = dfock[SP['o'], SP['v']]
  # R2[abij] = RR2[abij] + RR2[baji]
  #@tensoropt RR2[a,b,i,j] := U[a,i,X] * (U[b,j,Y] * (Txyz[X,Y,Z] * (fov[k,c]*U[c,k,Z])) - (Txyz[X,Y,Z] * U[b,k,Z])* (fov[k,c]*U[c,j,Y]))
  @tensoropt RR2[a,b,i,j] := 2.0*U[b,j,Y] * ((Bvo[a,i,P,Z] - Bov[i,a,P,Z])*(Txyz[X,Y,Z] * A[P,X]))
  @tensoropt RR2[a,b,i,j] += (Bov[i,a,P,Z]  - Bvo[a,i,P,Z])*(Boo[k,j,P,Y] * (Txyz[X,Y,Z] * U[b,k,X]))
  @tensoropt RR2[a,b,i,j] -= U[b,j,Z] * (Txyz[X,Y,Z] * (Bvo[a,k,P,X] * Boo[k,i,P,Y] - U[a,k,Y] * (Bov[i,c,P,X] * notd_ovP[k,c,P])))
  @tensoropt R2[a,b,i,j] += RR2[a,b,i,j] + RR2[b,a,j,i]
  close(notd_ovPfile)
  close(notd_ooPfile)

  return R1,R2
  GC.gc()
end


"""
    add_to_singles_and_doubles_residuals(EC, R1, R2)

  Add contributions from triples to singles and doubles residuals.
"""
function add_to_singles_and_doubles_residuals(EC, R1, R2)
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
  GC.gc()
end


"""
    calc_triples_decomposition_without_triples(EC::ECInfo, T2)

  Decompose ``T^{ijk}_{abc}`` as ``U^{iX}_a U^{jY}_b U^{kZ}_c T_{XYZ}`` 
  without explicit calculation of ``T^{ijk}_{abc}``.

  Compute perturbative ``T^i_{aXY}`` and decompose ``D^{ij}_{ab} = (T^i_{aXY} T^j_{bXY})`` to get ``U^{iX}_a``.
"""
function calc_triples_decomposition_without_triples(EC::ECInfo, T2)
  println("T^ijk_abc-free-decomposition")
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)

  # first approx for U^iX_a from doubles decomposition
  tol2 = EC.options.cc.ampsvdtol*EC.options.cc.ampsvdfac
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

  Decompose ``T^{ijk}_{abc}`` as ``U^{iX}_a U^{jY}_b U^{kZ}_c T_{XYZ}``.
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

  Calculate ``D^{ij}_{ab} = T^i_{aXY} T^j_{bXY}`` using half-decomposed imaginary-shifted perturbative triple amplitudes 
  ``T^i_{aXY}`` from `T2` (and `UvoX`)
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
 
  #Charlottes modification: making it possible to not project out iii and aaa contributions for constructing the SVD basis
  iii_aaa_remove = true
  if iii_aaa_remove == false
	  println("ATTENTION, PROJECTING OUT III AND AAA IN THE SVD BASIS IS DEACTIVATED")
  end
  if iii_aaa_remove == true
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
  end

  # display(D2)
  return D2
end

"""
    calc_SVD_pert_T(EC::ECInfo, T2)

  Calculate SVD-CCSD(T).
"""

function calc_SVD_pert_T(EC::ECInfo, T2)
  
  t1 = time_ns()
  UvoX = load(EC, "C_voX")
  #display(UvoX)

  #load decomposed amplitudes
  T3_XYZ = load(EC, "T_XXX")
  #display(T3_XYZ)
  #load df coeff
  notd_ooPfile, notd_ooP = mmap(EC, "notd_ooL")
  notd_voPfile, notd_voP = mmap(EC, "notd_voL")
  notd_vvPfile, notd_vvP = mmap(EC, "notd_vvL")

  @tensoropt Thetavirt[b,d,Z] := notd_vvP[b,d,Q] * (notd_voP[c,k,Q] * UvoX[c,k,Z]) #virt1
  notd_vvP = nothing
  #println(1)
  #flush(stdout)
  
  @tensoropt Thetaocc[l,j,Z] := notd_ooP[l,j,Q] * (notd_voP[c,k,Q] * UvoX[c,k,Z]) #occ1
  notd_voP = nothing
  notd_ooP = nothing
  #println(4)
  #flush(stdout)

  @tensoropt TaiX[a,i,X] := UvoX[b,j,X] * T2[a,b,i,j]
  #println(16)
  #flush(stdout)
  
  nocc = n_occ_orbs(EC)
  nsvd = size(T3_XYZ, 1)
  Term1 = zeros(nsvd,nsvd,nsvd)
  for j in 1:nocc
    ThetaoccCut = Thetaocc[:,j,:]
    @tensoropt IntermediateTerm11[b,X,Z] := TaiX[b,l,X] * ThetaoccCut[l,Z]
    ThetaoccCut = nothing
    TaiXCut = TaiX[:,j,:]
    @tensoropt IntermediateTerm11[b,X,Z] -= Thetavirt[b,d,Z] * TaiXCut[d,X]
    TaiXCut = nothing
    UvoXCut = UvoX[:,j,:]
    @tensoropt Term1[X,Y,Z] += IntermediateTerm11[b,X,Z] * UvoXCut[b,Y]
    IntermediateTerm11 = nothing
    UvoXCut = nothing
  end
  #println(23)
  #flush(stdout)
  Thetaocc = nothing
  Thetavirt = nothing
  TaiX = nothing
  t1 = print_time(EC, t1, "Theta terms in R3(T3)", 2)

  @tensoropt R3decomp[X,Y,Z] := Term1[X,Y,Z] + Term1[Y,X,Z] + Term1[X,Z,Y] + Term1[Z,Y,X] + Term1[Z,X,Y] + Term1[Y,Z,X]
  #println(24)
  #flush(stdout)
  Term1 = nothing
  t1 = print_time(EC, t1, "Symmetrization of Theta terms in R3(T3)", 2)

  #display(R3decomp)

  close(notd_vvPfile)
  close(notd_voPfile)
  close(notd_ooPfile)

  save!(EC, "R_XXX", R3decomp)
  #println(40)
  #flush(stdout)
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
 
  nocc = n_occ_orbs(EC)
  println("nocc_orbs:")
  println(nocc)
  nvirt = n_virt_orbs(EC)
  println("nvirt_orbs:")
  println(nvirt)
  nsvd = size(T3_XYZ, 1)
  println("nsvd_orbs:")
  println(nsvd)
  naux = size(voP, 3)
  println("naux_orbs:")
  println(naux)
  flush(stdout)

  @tensoropt Thetavirt[b,d,Z] := vvP[b,d,Q] * (voP[c,k,Q] * UvoX[c,k,Z]) #virt1
  #println(1)
  #flush(stdout)
  @tensoropt Thetavirt[b,d,Z] += UvoX[c,k,Z] * (T2[c,b,l,m] * (ooP[l,k,Q] * ovP[m,d,Q])) #virt3
  #println(2)
  #flush(stdout)

  for k in 1:nocc
    UvoXCut = UvoX[:,k,:]
    @tensoropt IntermediateV62[e,Q,Z] := UvoXCut[c,Z] * vvP[c,e,Q]
    UvoXCut = nothing
    T2Cut = T2[:,:,:,k]
    @tensoropt IntermediateV61[b,l,Q,Z] := T2Cut[b,e,l] * IntermediateV62[e,Q,Z]
    IntermediateV62 = nothing
    T2Cut = nothing
    @tensoropt Thetavirt[b,d,Z] -= ovP[l,d,Q] * IntermediateV61[b,l,Q,Z] #virt6
    IntermediateV61 = nothing
  end
  #println(3)
  #flush(stdout)
  t1 = print_time(EC, t1, "1 Theta terms in R3(T3)", 2)
  
  @tensoropt Thetaocc[l,j,Z] := ooP[l,j,Q] * (voP[c,k,Q] * UvoX[c,k,Z]) #occ1
  #println(4)
  #flush(stdout)
  @tensoropt Thetaocc[l,j,Z] -= UvoX[c,k,Z] * (T2[c,d,m,j] * (ovP[l,d,Q] * ooP[m,k,Q])) #occ4
  #println(5)
  #flush(stdout)
  @tensoropt Thetaocc[l,j,Z] += UvoX[c,k,Z] * (T2[d,e,k,j]* (ovP[l,e,Q] * vvP[c,d,Q])) #occ5
  #println(6)
  #flush(stdout)
  t1 = print_time(EC, t1, "2 Theta terms in R3(T3)", 2)
  
  if !cc3
    @tensoropt BooQX[i,j,Q,X] := ovP[i,a,Q] * UvoX[a,j,X]
    #println(7)
    #flush(stdout)

    for W in 1:nsvd
      BooQXCut = BooQX[:,:,:,W]
      @tensoropt IntermediateV92[d,m] := ovP[l,d,Q] * BooQXCut[m,l,Q]
      BooQXCut = nothing
      @tensoropt IntermediateV91[b,d,Y'] := UvoX[b,m,Y'] * IntermediateV92[d,m]
      IntermediateV92 = nothing
      T3_XYZCut = T3_XYZ[W,:,:]
      @tensoropt Thetavirt[b,d,Z] += 0.5* T3_XYZCut[Y',Z] * IntermediateV91[b,d,Y'] #virt9
      IntermediateV91 = nothing
      T3_XYZCut = nothing
    end

    #println(8)
    #flush(stdout)
    @tensoropt Thetaocc[l,j,Z] -= 0.5 * T3_XYZ[X',Z,Z'] * (BooQX[l,m,Q,X'] * BooQX[m,j,Q,Z']) #occ8
    #println(9)
    #flush(stdout)
    BooQX = nothing
    t1 = print_time(EC, t1, "3 Theta terms in R3(T3)", 2)

    @tensoropt A[Q,X] := ovP[i,a,Q] * UvoX[a,i,X]
    #println(10)
    #flush(stdout)
    
    @tensoropt IntermediateV72[Q,Z,Z'] := T3_XYZ[X',Z,Z'] * A[Q,X']
    for l in 1:nocc
      UvoXCut = UvoX[:,l,:]
      @tensoropt IntermediateV71[b,Q,Z] := UvoXCut[b,Z'] * IntermediateV72[Q,Z,Z']
      UvoXCut = nothing
      ovPCut = ovP[l,:,:]
      @tensoropt Thetavirt[b,d,Z] -= ovPCut[d,Q] * IntermediateV71[b,Q,Z] #virt7
      IntermediateV71 = nothing
      ovPCut = nothing
    end
    IntermediateV72 = nothing 
    
    #println(11)
    #flush(stdout)
   
    @tensoropt IntermediateO62[Q,Z,Z'] := T3_XYZ[X',Z,Z'] * A[Q,X']
    for d in 1:nvirt
      UvoXCut = UvoX[d,:,:]
      @tensoropt IntermediateO61[j,Q,Z] := UvoXCut[j,Z'] * IntermediateO62[Q,Z,Z']
      UvoXCut = nothing
      ovPCut = ovP[:,d,:]
      @tensoropt Thetaocc[l,j,Z] += ovPCut[l,Q] * IntermediateO61[j,Q,Z]   #occ6
      ovPCut = nothing
      IntermediateO61 = nothing
    end
    IntermediateO62 = nothing
    
    #println(12)
    #flush(stdout)
    A = nothing
    t1 = print_time(EC, t1, "4 Theta terms in R3(T3)", 2)

    IntermediateTheta = zeros(naux,nsvd,nsvd)
    @tensoropt IntermediateThetaV82[k,m,Q,Y'] := ovP[m,e,Q] * UvoX[e,k,Y']
    for W in 1:nsvd
      T3_XYZCut = T3_XYZ[:,:,W]
      @tensoropt IntermediateThetaV83[c,m,Y'] := UvoX[c,m,X'] * T3_XYZCut[X',Y']
      T3_XYZCut = nothing
      @tensoropt IntermediateThetaV81[c,k,Q] := IntermediateThetaV83[c,m,Y'] * IntermediateThetaV82[k,m,Q,Y']
      IntermediateThetaV83 = nothing
      @tensoropt IntermediateThetaCut[Q,Z] := IntermediateThetaV81[c,k,Q] * UvoX[c,k,Z]
      IntermediateThetaV81 = nothing
      IntermediateTheta[:,W,:] += IntermediateThetaCut
      IntermediateThetaCut = nothing
    #@tensoropt IntermediateThetaV83[k,m,X',Z] := UvoX[c,m,X'] * UvoX[c,k,Z]
    #@tensoropt IntermediateThetaV82[k,m,Y',Z,Z'] := T3_XYZ[X',Y',Z'] * IntermediateThetaV83[k,m,X',Z]
    #@tensoropt IntermediateThetaV81[e,m,Z,Z'] := UvoX[e,k,Y'] * IntermediateThetaV82[k,m,Y',Z,Z']
    #@tensoropt IntermediateTheta[Q,Z',Z] := ovP[m,e,Q] * IntermediateThetaV81[e,m,Z,Z']
    end
    IntermediateThetaV82 = nothing
    #println(13)
    #flush(stdout)

    #@tensoropt IntermediateThetaV83[k,m,X',Z] := UvoX[c,m,X'] * UvoX[c,k,Z]
    #@tensoropt IntermediateThetaV82[k,m,Y',Z,Z'] := T3_XYZ[X',Y',Z'] * IntermediateThetaV83[k,m,X',Z]
    #IntermediateThetaV83 = nothing
    #@tensoropt IntermediateThetaV81[e,m,Z,Z'] := UvoX[e,k,Y'] * IntermediateThetaV82[k,m,Y',Z,Z']
    #IntermediateThetaV82 = nothing
    #@tensoropt IntermediateTheta[Q,Z',Z] := ovP[m,e,Q] * IntermediateThetaV81[e,m,Z,Z']
    #IntermediateThetaV81 = nothing
    #println(13)
    #flush(stdout)
   
    for l in 1:nocc
      UvoXCut = UvoX[:,l,:]
      @tensoropt IntermediateV81[b,Q,Z] := UvoXCut[b,W] * IntermediateTheta[Q,W,Z]
      UvoXCut = nothing
      ovPCut = ovP[l,:,:]
      @tensoropt Thetavirt[b,d,Z] += 0.5 * ovPCut[d,Q] * IntermediateV81[b,Q,Z] #virt8
      ovPCut = nothing
      IntermediateV81 = nothing
    end
    #println(14)
    #flush(stdout)
    
    for d in 1:nvirt
      UvoXCut = UvoX[d,:,:]
      @tensoropt IntermediateO71[j,Q,Z] := UvoXCut[j,Z'] * IntermediateTheta[Q,Z',Z]
      UvoXCut = nothing
      ovPCut = ovP[:,d,:]
      @tensoropt Thetaocc[l,j,Z] -= 0.5 * ovPCut[l,Q] * IntermediateO71[j,Q,Z] #occ7
      IntermediateO71 = nothing
      ovPCut = nothing
    end
    #println(15)
    #flush(stdout)
    IntermediateTheta = nothing
    t1 = print_time(EC, t1, "5 Theta terms in R3(T3)", 2)
  end

  @tensoropt TaiX[a,i,X] := UvoX[b,j,X] * T2[a,b,i,j]
  #println(16)
  #flush(stdout)
  @tensoropt TStrich[a,i,X] := 2* TaiX[a,i,X] - UvoX[b,j,X] * T2[b,a,i,j] 
  #println(17)
  #flush(stdout)
  @tensoropt Thetavirt[b,d,Z] += vvP[b,d,Q] * (ovP[l,e,Q] * TStrich[e,l,Z]) #virt4
  #println(18)
  #flush(stdout)
  @tensoropt Thetaocc[l,j,Z] += ooP[l,j,Q] * (ovP[m,d,Q] * TStrich[d,m,Z]) #occ2
  #println(19)
  #flush(stdout)
  TStrich = nothing
  t1 = print_time(EC, t1, "6 Theta terms in R3(T3)", 2)

  @tensoropt Thetavirt[b,d,Z] -= dfov[l,d] * TaiX[b,l,Z] #virt2
  #println(20)
  #flush(stdout)
  
  for l in 1:nocc
    TaiXCut = TaiX[:,l,:]
    @tensoropt IntermediateV51[b,Q,Z] := vvP[b,e,Q] * TaiXCut[e,Z]
    TaiXCut = nothing
    ovPCut = ovP[l,:,:]
    @tensoropt Thetavirt[b,d,Z] -= ovPCut[d,Q] * IntermediateV51[b,Q,Z] #virt5
    IntermediateV51 = nothing
    ovPCut = nothing
  end
  #println(21)
  #flush(stdout)
 
  for m in 1:nocc
    TaiXCut = TaiX[:,m,:]
    @tensoropt IntermediateO31[l,Q,Z] := ovP[l,d,Q] * TaiXCut[d,Z]
    TaiXCut = nothing
    ooPCut = ooP[m,:,:]
    @tensoropt Thetaocc[l,j,Z] -= ooPCut[j,Q] * IntermediateO31[l,Q,Z] #occ3
    IntermediateO31 = nothing
    ooPCut = nothing
  end
  #println(22)
  #flush(stdout)
  t1 = print_time(EC, t1, "7 Theta terms in R3(T3)", 2)

  Term1 = zeros(nsvd,nsvd,nsvd)
  for j in 1:nocc
    ThetaoccCut = Thetaocc[:,j,:]
    @tensoropt IntermediateTerm11[b,X,Z] := TaiX[b,l,X] * ThetaoccCut[l,Z]
    ThetaoccCut = nothing
    TaiXCut = TaiX[:,j,:]
    @tensoropt IntermediateTerm11[b,X,Z] -= Thetavirt[b,d,Z] * TaiXCut[d,X]
    TaiXCut = nothing
    UvoXCut = UvoX[:,j,:]
    @tensoropt Term1[X,Y,Z] += IntermediateTerm11[b,X,Z] * UvoXCut[b,Y]
    IntermediateTerm11 = nothing
    UvoXCut = nothing
  end
  #println(23)
  #flush(stdout)
  Thetaocc = nothing
  Thetavirt = nothing
  TaiX = nothing
  t1 = print_time(EC, t1, "Theta terms in R3(T3)", 2)

  @tensoropt R3decomp[X,Y,Z] := Term1[X,Y,Z] + Term1[Y,X,Z] + Term1[X,Z,Y] + Term1[Z,Y,X] + Term1[Z,X,Y] + Term1[Y,Z,X]
  #println(24)
  #flush(stdout)
  Term1 = nothing
  t1 = print_time(EC, t1, "Symmetrization of Theta terms in R3(T3)", 2)


  @tensor TTilde[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  #println(25)
  #flush(stdout)

  if cc3
    @tensoropt Term2[X,Y,Z] := T3_XYZ[X',Y,Z] * (UvoX[a,l,X'] * (dfoo[l,i]  * UvoX[a,i,X])) #1
    #println(251)
    #flush(stdout)
    @tensoropt Term2[X,Y,Z] -= T3_XYZ[X',Y,Z] * (UvoX[a,i,X] *( dfvv[a,d] * UvoX[d,i,X'])) #2
    #println(252)
    #flush(stdout)
  else
    @tensoropt Intermediate1Term2[l,d,m,e] := ovP[l,d,P] * ovP[m,e,P]
    #println(26)
    #flush(stdout)
    @tensoropt Term2[X,Y,Z] := T3_XYZ[X',Y,Z] * (UvoX[a,l,X'] * ( (dfoo[l,i] + 0.5 * Intermediate1Term2[l,d,m,e] * TTilde[d,e,i,m]) * UvoX[a,i,X])) #1
    #println(27)
    #flush(stdout)
    @tensoropt Term2[X,Y,Z] -= T3_XYZ[X',Y,Z] * (UvoX[a,i,X] *( (dfvv[a,d] - 0.5 * Intermediate1Term2[l,d,m,e] * TTilde[a,e,l,m]) * UvoX[d,i,X'])) #2
    #println(28)
    #flush(stdout)
    Intermediate1Term2 = nothing
    t1 = print_time(EC, t1, "1 Chi terms in R3(T3)", 2)
    
    @tensoropt Term2[X,Y,Z] += (UvoX[a,i,X] * ((ooP[l,i,P] * vvP[a,d,P]) * UvoX[d,l,X'])) * (T3_XYZ[X',Y',Z] * (UvoX[b,j,Y] * UvoX[b,j,Y'])) #3
    #println(29)
    #flush(stdout)
    @tensoropt Term2[X,Y,Z] -= 2* (T3_XYZ[X',Y,Z] *((voP[a,i,P] + ovP[m,e,P] * TTilde[a,e,i,m]) * UvoX[a,i,X]) * (ovP[l,d,P] * UvoX[d,l,X'])) #4
    #println(30)
    #flush(stdout)

    """
    example for get_spaceblocks from Daniels dfcc.jl:

     W_LL = zeros(nL,nL)
     # generate ``W^{LL'} = v_a^{iL} v_a^{iL'}`` for SVD
     oBlks = get_spaceblocks(1:length(SP['o']))
     for oblk in oBlks
       voL = full_voL[:,oblk,:]
       @tensoropt W_LL[L,L'] += voL[a,i,L] * voL[a,i,L']
     end
    
    get_spaceblocks creates a list with ranges of the maximum size 100 up til the length which was defined
    e.g. get_spaceblocks(1:1500) gives back:
    Any[1:100, 101:200, 201:300, 301:400, 401:500, 501:600, 601:700, 701:800, 801:900, 901:1000, 1001:1100, 1101:1200, 1201:1300, 1301:1400, 1401:1500]
    """
    
    @tensoropt Intermediate52[Y,Y',P] := UvoX[b,j,Y] * (ooP[l,j,P] * UvoX[b,l,Y'])
    #println("30_1")
    #flush(stdout)
  


    #@tensoropt Intermediate51[Z,W,P] := (UvoX[c,k,Z] * UvoX[c,m,W]) * ooP[m,k,P]
    #@tensoropt Intermediate53[Y,Y',Z,W] := Intermediate51[Z,W,P] * Intermediate52[Y,Y',P]
    #@tensoropt Term2[X,Y,Z] -= T3_XYZ[X,Y',W] * Intermediate53[Y,Y',Z,W]


       
    @tensoropt Intermediate51[Z,W,P] := (UvoX[c,k,Z] * UvoX[c,m,W]) * ooP[m,k,P]
    #println("30_2")
    #println(sizeof(Intermediate51))
    #flush(stdout)
    #int1= Intermediate51[:,W,:] 
    
    for W in 1:nsvd
       Intermediate51Cut = Intermediate51[:,W,:] 
       #Intermediate51 = nothing
       #println("30_3")
       #println(sizeof(Intermediate51Cut))
       #display(Intermediate51Cut)
       flush(stdout)
       
       #Intermediate53 = zeros(nsvd,nsvd,nsvd)
       #display(Intermediate53)
       @tensoropt Intermediate53[Y,Y',Z] := Intermediate51Cut[Z,P] * Intermediate52[Y,Y',P]
       #println("30_4")
       #flush(stdout)
       Intermediate51Cut = nothing

       #println("30_5")
       #flush(stdout)
       
       T3_XYZCut = T3_XYZ[:,:,W]

       @tensoropt Term2[X,Y,Z] -= T3_XYZCut[X,Y'] * Intermediate53[Y,Y',Z] #5
       Intermediate53 = nothing
       T3_XYZCut = nothing

       #cut out W from all tensors, because [1,x,y] tensor should be represented as [x,y]

       #println("30_6")
       #flush(stdout)
    end
    
    Intermediate51 = nothing
    Intermediate52 = nothing

    #Intermediate53 = zeros(nsvd, nsvd, nsvd, nsvd)
    #println(size(Intermediate53))

    """
    svdBlks = get_spaceblocks(1:nsvd)

    for svdblk in svdBlks
       @tensoropt Intermediate53[Y,Y',Z,Z'] += Intermediate51[Z,Z',P] * Intermediate52[Y,Y',P]
       @tensoropt Term2[X,Y,Z] -= T3_XYZ[X,Y',Z'] * Intermediate53[Y,Y',Z,Z']
    end
    """   

    """@tensoropt Term2[X,Y,Z] -= T3_XYZ[X,Y',Z'] * (((UvoX[c,k,Z] * UvoX[c,m,Z']) * ooP[m,k,P]) * (UvoX[b,j,Y] * (ooP[l,j,P] * UvoX[b,l,Y']))) #5
    """

    #Intermediate51 = nothing
    #Intermediate52 = nothing
    #Intermediate53 = nothing
    #println(31)
    #flush(stdout)
   
    Intermediate2Term2 = zeros(nsvd,nsvd,naux)
    for j in 1:nocc
      UvoXCut = UvoX[:,j,:]
      @tensoropt IntermediateI2T21[b,P,Y'] := vvP[b,d,P] * UvoXCut[d,Y']
      @tensoropt Intermediate2Term2[Y,Y',P] +=  UvoXCut[b,Y] * IntermediateI2T21[b,P,Y']
      IntermediateI2T21 = nothing
      UvoXCut = nothing
    end
    #println(32)
    #flush(stdout)

    #hier vielleicht groeßere Scheiben schneiden mit get_spaceblocks Funktion??
    for P in 1:naux
      Intermediate2Term2Cut = Intermediate2Term2[:,:,P]
      @tensoropt IntermediateT2_1[X,Y,Z'] := T3_XYZ[X,Y',Z'] * Intermediate2Term2Cut[Y,Y']
      Intermediate2Term2Cut = nothing

      vvPCut = vvP[:,:,P]
      @tensoropt IntermediateT2_2[e,k,Z] := UvoX[c,k,Z] * vvPCut[c,e]
      @tensoropt IntermediateT2_3[Z,Z'] := UvoX[e,k,Z'] * IntermediateT2_2[e,k,Z]
      IntermediateT2_2 = nothing
    
      @tensoropt Term2[X,Y,Z] -= IntermediateT2_1[X,Y,Z'] * IntermediateT2_3[Z,Z'] #6
      IntermediateT2_1 = nothing
      IntermediateT2_3 = nothing
    end
    #println(34)
    #flush(stdout)
    t1 = print_time(EC,t1,"2 Chi terms in R3(T3)",2) #weil andere Termreihenfolge Print veraendern??
   
    for W in 1:nsvd 
      UvoXCut = UvoX[:,:,W]
      @tensoropt IntermediateT2_5[i,l,X] := UvoX[a,i,X] * UvoXCut[a,l]
      UvoXCut = nothing
      @tensoropt IntermediateT2_4[P,X] := ooP[l,i,P] * IntermediateT2_5[i,l,X]
      IntermediateT2_5 = nothing

      T3_XYZCut = T3_XYZ[W,:,:]
      @tensoropt Intermediate3Term2[Y,Z,P] :=  T3_XYZCut[Y',Z] * Intermediate2Term2[Y,Y',P]
      T3_XYZCut = nothing
      #println(33)
      #flush(stdout)

      @tensoropt Term2[X,Y,Z] += IntermediateT2_4[P,X] * (Intermediate3Term2[Y,Z,P] + Intermediate3Term2[Z,Y,P]) #7
      IntermediateT2_4 = nothing
      Intermediate3Term2 = nothing
    end

    Intermediate2Term2 = nothing
    #println(35)
    #flush(stdout)
    t1 = print_time(EC, t1, "3 Chi terms in R3(T3)", 2)
    
    @tensoropt Intermediate4Term2[l,d,a,i] := ovP[l,d,P] * (voP[a,i,P] + ovP[m,e,P] * TTilde[a,e,i,m])
    #println(36)
    #flush(stdout)
    
    @tensoropt IntermediateT2_9[d,l,X] := UvoX[a,i,X] * Intermediate4Term2[l,d,a,i]
    @tensoropt IntermediateT2_8[k,l,X,Y'] := UvoX[d,k,Y'] * IntermediateT2_9[d,l,X]
    IntermediateT2_9 = nothing
    for c in 1:nvirt
      UvoXCut2 = UvoX[c,:,:]
      @tensoropt IntermediateT2_7[l,Y,Y'] := T3_XYZ[X',Y',Y] * UvoXCut2[l,X']
      @tensoropt IntermediateT2_6[k,X,Y] := IntermediateT2_7[l,Y,Y'] * IntermediateT2_8[k,l,X,Y']
      IntermediateT2_7 = nothing
      @tensoropt Term2[X,Y,Z] += UvoXCut2[k,Z] * IntermediateT2_6[k,X,Y] #8
      IntermediateT2_6 = nothing
      UvoXCut2 = nothing
    end
    IntermediateT2_8 = nothing
    #println(37)
    #flush(stdout)
    #@tensoropt Term2[X,Y,Z] += UvoX[c,k,Z] * ((T3_XYZ[X',Y',Y] * UvoX[c,l,X']) * (UvoX[d,k,Y'] * (UvoX[a,i,X] * Intermediate4Term2[l,d,a,i]))) #8
   

    
    @tensoropt IntermediateT2_13[l,d,X] := UvoX[a,i,X] * Intermediate4Term2[l,d,a,i]
    @tensoropt IntermediateT2_12[j,l,X,Y'] := UvoX[d,j,Y'] * IntermediateT2_13[l,d,X]
    IntermediateT2_13 = nothing
    for b in 1:nvirt
      UvoXCut = UvoX[b,:,:]
      @tensoropt IntermediateT2_11[l,Y',Z] := T3_XYZ[X',Y',Z] * UvoXCut[l,X']
      @tensoropt IntermediateT2_10[j,X,Z] := IntermediateT2_11[l,Y',Z] * IntermediateT2_12[j,l,X,Y']
      IntermediateT2_11 = nothing
      @tensoropt Term2[X,Y,Z] += UvoXCut[j,Y] * IntermediateT2_10[j,X,Z] #9
      IntermediateT2_10 = nothing
      UvoXCut = nothing
    end
    IntermediateT2_12 = nothing
    Intermediate4Term2 = nothing
    
    #println(38)
    #flush(stdout)
    t1 = print_time(EC, t1, "4 Chi terms in R3(T3)", 2)

    #@tensoropt Term2[X,Y,Z] += UvoX[b,j,Y] * ((T3_XYZ[X',Y',Z] * UvoX[b,l,X']) * (UvoX[d,j,Y'] * (UvoX[a,i,X] * Intermediate4Term2[l,d,a,i]))) #9
    #Intermediate4Term2 = nothing
    #println(38)
    #flush(stdout)
    #t1 = print_time(EC, t1, "4 Chi terms in R3(T3)", 2)
  end

  @tensoropt R3decomp[X,Y,Z] += Term2[X,Y,Z] + Term2[Y,X,Z] + Term2[Z,Y,X]
  #println(39)
  #flush(stdout)
  Term2 = nothing
  t1 = print_time(EC, t1, "Symmetrization of Chi terms in R3(T3)", 2)

  iii_aaa_remove = false

  if iii_aaa_remove == true 
     #change to avoid potential iii and aaa contributions (also changed after amplitudes DIIS update)
     #attention! no complex conjugation is taken into account so far!

     @tensoropt ResidualsDotProductOne = R3decomp[X,Y,Z] * R3decomp[X,Y,Z]
     println("Residuals DotProduct without substraction:")
     println(ResidualsDotProductOne)
  
     R_iii = zeros(nvirt,nvirt,nvirt)
     Diff_R_iii_aaa = zero(R3decomp)
  
     for i in 1:nocc
        Ui = UvoX[:,i,:]
        @tensoropt R_iii[a,b,c] = (((R3decomp[X,Y,Z] * Ui[a,X]) * Ui[b,Y]) * Ui[c,Z])
     
        for a in 1:nvirt
           R_iii[a,a,a] = 0.0
        end
        @tensoropt Diff_R_iii_aaa[X,Y,Z] += (((R_iii[a,b,c] * Ui[a,X]) * Ui[b,Y]) * Ui[c,Z])
     end

     R_aaa = zeros(nocc,nocc,nocc)
  
     for a in 1:nvirt
        Ua = UvoX[a,:,:]
        @tensoropt R_aaa[i,j,k] = (((R3decomp[X,Y,Z] * Ua[i,X]) * Ua[j,Y]) * Ua[k,Z])
        @tensoropt Diff_R_iii_aaa[X,Y,Z] += (((R_aaa[i,j,k] * Ua[i,X]) * Ua[j,Y]) * Ua[k,Z])
     end
     R3decomp .-= Diff_R_iii_aaa

     @tensoropt ResidualsDotProductTwo = R3decomp[X,Y,Z] * R3decomp[X,Y,Z]
     println("Residuals DotProduct with substraction:")
     println(ResidualsDotProductTwo)

     #Daniels suggested code:
     #RR = zeros(nvirt,nvirt,nvirt)
     #DR = zero(R)
     #for i in 1:nocc
     #   Ui = U[:,i,:]
     #   @tensoropt RR[a,b,c] = R[X,Y,Z] * Ui[a,X] * Ui[b,Y] * Ui[c,Z]
     #   for a in 1:nvirt
     #      RR[a,a,a] = 0.0
     #   end
     #   @tensoropt DR[X,Y,Z] += RR[a,b,c] * Ui[a,X] * Ui[b,Y] * Ui[c,Z]
     #end
     #RRR = zeros(nocc,nocc,nocc)
     #for a in 1:nvirt
     #   Ua = U[a,:,:]
     #   @tensoropt RRR[i,j,k] = R[X,Y,Z] * Ua[i,X] * Ua[j,Y] * Ua[k,Z]
     #   @tensoropt DR[X,Y,Z] += RRR[i,j,k] * Ua[i,X] * Ua[j,Y] * Ua[k,Z]
     #end
     #R .-= DR
  end


  #display(R3decomp)

  close(ovPfile)
  close(voPfile)
  close(ooPfile)
  close(vvPfile)

  save!(EC, "R_XXX", R3decomp)
  GC.gc()
  #println(40)
  #flush(stdout)

end

end #module
