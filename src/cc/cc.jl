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
using Buffers
using ..ElemCo.Outputs
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.QMTensors
using ..ElemCo.TensorTools
using ..ElemCo.FciDumps
using ..ElemCo.MSystem
using ..ElemCo.DIIS
using ..ElemCo.DecompTools
using ..ElemCo.DFCoupledCluster
using ..ElemCo.OrbTools
using ..ElemCo.CCTools

export calc_MP2, calc_UMP2, calc_UMP2_energy 
export calc_cc, calc_pertT
export calc_lm_cc, calc_1RDM

include("cc_triples.jl")

include("cc_lagrange.jl")

include("cc_tests.jl")

include("../algo/ccsdt_singles.jl")
include("../algo/ccsdt_doubles.jl")
include("../algo/ccsdt_triples.jl")
include("../algo/dcccsdt_triples.jl")

"""
    calc_singles_energy(EC::ECInfo, T1; fock_only=false)

  Calculate coupled-cluster closed-shell singles energy.
  Returns total energy, SS, OS and Openshell (0.0) contributions
  as `OutDict` with keys (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_singles_energy(EC::ECInfo, T1; fock_only=false)
  SP = EC.space
  ET1 = ET1SS = ET1OS = 0.0
  if length(T1) > 0
    if !fock_only
      oovv = ints2(EC,"oovv")
      @mtensor begin
        ET1d = T1[a,i] * (T1[b,j] * oovv[i,j,a,b])
        ET1ex = T1[b,i] * (T1[a,j] * oovv[i,j,a,b])
      end
      ET1SS = ET1d - ET1ex
      ET1OS = ET1d
      ET1 = ET1SS + ET1OS
    end
    @mtensor ET1 += 2.0*(T1[a,i] * load2idx(EC,"f_mm")[SP['o'],SP['v']][i,a])
  end
  return OutDict("E"=>ET1, "ESS"=>ET1SS, "EOS"=>ET1OS, "EO"=>0.0)
end

"""
    calc_singles_energy(EC::ECInfo, T1a, T1b; fock_only=false)

  Calculate energy for α (T1a) and β (T1b) singles amplitudes.
  Returns total energy, SS, OS and Openshell contributions
  as `OutDict` with keys (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_singles_energy(EC::ECInfo, T1a, T1b; fock_only=false)
  SP = EC.space
  ET1 = ET1aa = ET1bb = ET1ab = 0.0
  if !fock_only
    if length(T1a) > 0
      @mtensor ET1aa = 0.5*((T1a[a,i]*T1a[b,j]-T1a[b,i]*T1a[a,j])*ints2(EC,"oovv")[i,j,a,b])
    end
    if length(T1b) > 0
      @mtensor ET1bb = 0.5*((T1b[a,i]*T1b[b,j]-T1b[b,i]*T1b[a,j])*ints2(EC,"OOVV")[i,j,a,b])
      if length(T1a) > 0
        @mtensor ET1ab = T1a[a,i]*(T1b[b,j]*ints2(EC,"oOvV")[i,j,a,b])
      end
    end
  end
  if length(T1a) > 0
    @mtensor ET1 += T1a[a,i] * load2idx(EC,"f_mm")[SP['o'],SP['v']][i,a]
  end
  if length(T1b) > 0
    @mtensor ET1 += T1b[a,i] * load2idx(EC,"f_MM")[SP['O'],SP['V']][i,a]
  end
  ET1 += ET1aa + ET1bb + ET1ab
  ET1SS = ET1aa + ET1bb
  ET1OS = ET1ab
  ET1O = ET1aa - ET1bb
  return OutDict("E"=>ET1, "ESS"=>ET1SS, "EOS"=>ET1OS, "EO"=>ET1O)
end

"""
    calc_doubles_energy(EC::ECInfo, T2)

  Calculate coupled-cluster closed-shell doubles energy.
  Returns total energy, SS, OS and Openshell (0.0) contributions
  as `OutDict` with keys (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_doubles_energy(EC::ECInfo, T2)
  oovv = ints2(EC,"oovv")
  @mtensor begin
    ET2d = T2[a,b,i,j] * oovv[i,j,a,b]
    ET2ex = T2[b,a,i,j] * oovv[i,j,a,b]
  end
  ET2SS = ET2d - ET2ex
  ET2OS = ET2d
  ET2 = ET2SS + ET2OS
  return OutDict("E"=>ET2, "ESS"=>ET2SS, "EOS"=>ET2OS, "EO"=>0.0)
end

"""
    calc_doubles_energy(EC::ECInfo, T2a, T2b, T2ab)

  Calculate energy for αα (T2a), ββ (T2b) and αβ (T2ab) doubles amplitudes.
  Returns total energy, SS, OS and Openshell contributions
  as `OutDict` with keys (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_doubles_energy(EC::ECInfo, T2a, T2b, T2ab)
  @mtensor begin
    ET2aa = 0.5*(T2a[a,b,i,j] * ints2(EC,"oovv")[i,j,a,b])
    ET2bb = 0.5*(T2b[a,b,i,j] * ints2(EC,"OOVV")[i,j,a,b])
    ET2OS = T2ab[a,b,i,j] * ints2(EC,"oOvV")[i,j,a,b]
  end
  ET2SS = ET2aa + ET2bb
  ET2O = ET2aa - ET2bb
  ET2 = ET2SS + ET2OS
  return OutDict("E"=>ET2, "ESS"=>ET2SS, "EOS"=>ET2OS, "EO"=>ET2O)
end

"""
    calc_hylleraas(EC::ECInfo, T1, T2, R1, R2)

  Calculate closed-shell singles and doubles Hylleraas energy.
  Returns total energy, SS, OS and Openshell (0.0) contributions
  as `OutDict` with keys (`pE`,`pESS`,`pEOS`,`pEO`,`E`,`ESS`,`EOS`,`EO`) where `pE` are projected energies.
"""
function calc_hylleraas(EC::ECInfo, T1, T2, R1, R2)
  SP = EC.space
  int2 = ints2(EC,"oovv")
  ET1 = ET1SS = ET1OS = 0.0
  if length(T1) > 0
    @mtensor begin
      ET1d = T1[a,i] * (T1[b,j] * int2[i,j,a,b])
      ET1ex = T1[b,i] * (T1[a,j] * int2[i,j,a,b])
    end
    ET1SS = ET1d - ET1ex
    ET1OS = ET1d
    ET1 = ET1SS + ET1OS
  end
  @mtensor begin
    pET2d = T2[a,b,i,j] * int2[i,j,a,b]
    pET2ex = T2[b,a,i,j] * int2[i,j,a,b]
    rET2d = T2[a,b,i,j] * R2[a,b,i,j]
    rET2ex = T2[b,a,i,j] * R2[a,b,i,j]
  end
  ET2d = pET2d + rET2d
  ET2ex = pET2ex + rET2ex
  ET2SS = ET2d - ET2ex
  ET2OS = ET2d
  ET2 = ET2SS + ET2OS
  pET2SS = pET2d - pET2ex
  pET2OS = pET2d
  pET2 = pET2SS + pET2OS
  if length(T1) > 0
    fov = load2idx(EC,"f_mm")[SP['o'],SP['v']] 
    @mtensor begin
      pET1 = 2.0*(fov[i,a] * T1[a,i])
      rET1 = 2.0*(R1[a,i] * T1[a,i])
    end
    ET1 += pET1 + rET1
  else
    pET1 = 0.0
  end
  pET1 += ET1SS + ET1OS
  ET2 += ET1
  ET2SS += ET1SS
  ET2OS += ET1OS
  pET2 += pET1
  pET2SS += ET1SS
  pET2OS += ET1OS
  return OutDict("pE"=>pET2, "pESS"=>pET2SS, "pEOS"=>pET2OS, "pEO"=>0.0,
                 "E"=>ET2, "ESS"=>ET2SS, "EOS"=>ET2OS, "EO"=>0.0)
end

"""
    calc_hylleraas4spincase(EC::ECInfo, o1, v1, o2, v2, T1, T1OS, T2, R1, R2, fov)

  Calculate singles and doubles Hylleraas energy for one spin case.
  
  Returns OutDict with keys (`pE2`,`pE1`,`pE1_2`,`E2`,`E1`,`E1_2`) where `pE` are projected energies
  and `E` are Hylleraas energies, and 2, 1, 1_2 are doubles, singles and quadratic singles contributions.
"""
function calc_hylleraas4spincase(EC::ECInfo, o1, v1, o2, v2, T1, T1OS, T2, R1, R2, fov)
  int2 = ints2(EC,o1*o2*v1*v2)
  if o1 == o2
    fac = 0.5
  else
    fac = 1.0
  end
  ET1 = ET1_2 = 0.0
  if length(T1) > 0
    if o1 == o2
      @mtensor ET1_2 = 0.5*((T1[a,i]*T1[b,j]-T1[b,i]*T1[a,j]) * int2[i,j,a,b])
    else
      @mtensor ET1_2 = T1[a,i] * (T1OS[b,j] * int2[i,j,a,b])
    end
  end
  @mtensor begin
    pET2 = fac*(T2[a,b,i,j] * int2[i,j,a,b])
    rET2 = fac*fac*(T2[a,b,i,j] * R2[a,b,i,j])
  end
  ET2 = pET2 + rET2
  pET1 = 0.0
  if length(R1) > 0
    @mtensor pET1 = fov[i,a] * T1[a,i]
    @mtensor ET1 = (fov[i,a] + R1[a,i]) * T1[a,i]
  end
  return OutDict("pE2"=>pET2, "pE1"=>pET1, "pE1_2"=>ET1_2, "E2"=>ET2, "E1"=>ET1, "E1_2"=>ET1_2)
end

"""
    calc_hylleraas(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab, R1a, R1b, R2a, R2b, R2ab)

  Calculate singles and doubles Hylleraas energy.
  Returns total energy, SS, OS and Openshell contributions
  as OutDict with keys (`pE`,`pESS`,`pEOS`,`pEO`,`E`,`ESS`,`EOS`,`EO`) where `pE` are projected energies.
"""
function calc_hylleraas(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab, R1a, R1b, R2a, R2b, R2ab)
  SP = EC.space
  # Eh2SSa, Eh1a, Eh1SSa = calc_hylleraas4spincase(EC, "ovov"..., T1a, T1b, T2a, R1a, R2a, load2idx(EC,"f_mm")[SP['o'],SP['v']])
  Ea = calc_hylleraas4spincase(EC, "ovov"..., T1a, T1b, T2a, R1a, R2a, load2idx(EC,"f_mm")[SP['o'],SP['v']])
  if n_occb_orbs(EC) > 0
    Eb = calc_hylleraas4spincase(EC, "OVOV"..., T1b, T1a, T2b, R1b, R2b, load2idx(EC,"f_MM")[SP['O'],SP['V']])
    Eab = calc_hylleraas4spincase(EC, "ovOV"..., T1a, T1b, T2ab, zeros(0,0), R2ab, zeros(0,0))
  else
    Eb = Eab = OutDict("pE2"=>0.0, "pE1"=>0.0, "pE1_2"=>0.0, "E2"=>0.0, "E1"=>0.0, "E1_2"=>0.0)
  end
  Eh = Ea["E2"] + Eb["E2"] + Eab["E2"] + Ea["E1"] + Eb["E1"] + Ea["E1_2"] + Eb["E1_2"] + Eab["E1"] + Eab["E1_2"]
  EhSS = Ea["E2"] + Eb["E2"] + Ea["E1_2"] + Eb["E1_2"]
  EhOS = Eab["E2"] + Eab["E1_2"]
  EhO = Ea["E2"] - Eb["E2"] + Ea["E1_2"] - Eb["E1_2"]
  Ep = Ea["pE2"] + Eb["pE2"] + Eab["pE2"] + Ea["pE1"] + Eb["pE1"] + Ea["pE1_2"] + Eb["pE1_2"] + Eab["pE1"] + Eab["pE1_2"]
  EpSS = Ea["pE2"] + Eb["pE2"] + Ea["pE1_2"] + Eb["pE1_2"]
  EpOS = Eab["pE2"] + Eab["pE1_2"]
  EpO = Ea["pE2"] - Eb["pE2"] + Ea["pE1_2"] - Eb["pE1_2"]
  return OutDict("pE"=>Ep, "pESS"=>EpSS, "pEOS"=>EpOS, "pEO"=>EpO,
                 "E"=>Eh, "ESS"=>EhSS, "EOS"=>EhOS, "EO"=>EhO)
end

# Function to calculate length for buffers buf1 buf2
# autogenerated by @print_buffer_usage
function auto_buf_length4calc_dressed_ints(no1, no2, nv1, nv2, calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo, mixed)
    buf1 = [0, 0]
    buf2 = [0, 0]
    if calc_d_vvvv
        hd_vvvv = pseudo_alloc!(buf1, nv1, nv2, nv1, nv2)
        vovv = pseudo_alloc!(buf2, nv1, no2, nv1, nv2)
        pseudo_drop!(buf2, vovv)
        pseudo_drop!(buf1, hd_vvvv)
    end
    hd_oooo = pseudo_alloc!(buf1, no1, no2, no1, no2)
    if mixed
        hd_oooo2 = pseudo_alloc!(buf1, no1, no2, no1, no2)
        oovo = pseudo_alloc!(buf2, no1, no2, nv1, no2)
        pseudo_drop!(buf2, oovo)
    end
    ooov = pseudo_alloc!(buf2, no1, no2, no1, nv2)
    pseudo_drop!(buf2, ooov)
    if calc_d_vvoo
        hd_vvoo = pseudo_alloc!(buf1, nv1, nv2, no1, no2)
        vooo = pseudo_alloc!(buf2, nv1, no2, no1, no2)
        voov = pseudo_alloc!(buf1, nv1, no2, no1, nv2)
        pseudo_drop!(buf1, voov)
        pseudo_drop!(buf2, vooo)
        vvov = pseudo_alloc!(buf2, nv1, nv2, no1, nv2)
        pseudo_drop!(buf2, vvov)
        pseudo_drop!(buf1, hd_vvoo)
    end
    hd_vooo = pseudo_alloc!(buf2, nv1, no2, no1, no2)
    if !mixed                                                                                                     nothing
    else
        pseudo_drop!(buf1, hd_oooo2)
    end
    if no2 > 0
        vovo = pseudo_alloc!(buf1, nv1, no2, nv1, no2)
        pseudo_drop!(buf1, vovo)
    end
    if mixed
        hd_ovoo = pseudo_alloc!(buf2, no1, nv2, no1, no2)
        ovov = pseudo_alloc!(buf1, no1, nv2, no1, nv2)
        pseudo_drop!(buf1, ovov)
    end
    d_oovo = pseudo_alloc!(buf2, no1, no2, nv1, no2)
    oovv = pseudo_alloc!(buf1, no1, no2, nv1, nv2)
    pseudo_drop!(buf1, oovv)
    pseudo_drop!(buf1, hd_oooo)
    if mixed
        d_ooov = pseudo_alloc!(buf2, no1, no2, no1, nv2)
    end
    vovv = pseudo_alloc!(buf2, nv1, no2, nv1, nv2)
    d_voov = pseudo_alloc!(buf1, nv1, no2, no1, nv2)
    if mixed
        oOvV = pseudo_alloc!(buf1, no1, no2, nv1, nv2)
        pseudo_drop!(buf1, oOvV)
    end
    if mixed
        d_ovvo = pseudo_alloc!(buf1, no1, nv2, nv1, no2)
    end
    hd_vovo = pseudo_alloc!(buf1, nv1, no2, nv1, no2)
    pseudo_drop!(buf2, vovv)
    if mixed
        ovvv = pseudo_alloc!(buf2, no1, nv2, nv1, nv2)
        hd_ovov = pseudo_alloc!(buf1, no1, nv2, no1, nv2)
        pseudo_drop!(buf2, ovvv)
    end
    if calc_d_vvvo
        vvvv = pseudo_alloc!(buf1, nv1, nv2, nv1, nv2)
        hd_vvvo = pseudo_alloc!(buf2, nv1, nv2, nv1, no2)
        pseudo_drop!(buf2, hd_vvvo)
        if mixed
            hd_vvov = pseudo_alloc!(buf2, nv1, nv2, no1, nv2)
            pseudo_drop!(buf2, hd_vvov)
        end
        pseudo_drop!(buf1, vvvv)
    end
    if mixed
        pseudo_drop!(buf1, hd_ovov)
    end
    pseudo_drop!(buf1, hd_vovo)
    if mixed
        pseudo_drop!(buf2, d_ooov)
    end
    pseudo_drop!(buf2, d_oovo)
    if calc_d_vovv
        d_vovv = pseudo_alloc!(buf2, nv1, no2, nv1, nv2)
        oovv = pseudo_alloc!(buf1, no1, no2, nv1, nv2)
        if mixed
            pseudo_drop!(buf2, d_vovv)
            d_ovvv = pseudo_alloc!(buf2, no1, nv2, nv1, nv2)
        end
        pseudo_drop!(buf1, oovv)
    end
    if calc_d_vvvv
        d_vvvv = pseudo_alloc!(buf1, nv1, nv2, nv1, nv2)
        if !mixed
            pseudo_drop!(buf2, d_vovv)
        else
            pseudo_drop!(buf2, d_ovvv)
        end
        pseudo_drop!(buf1, d_vvvv)
    elseif calc_d_vovv
        if mixed
            pseudo_drop!(buf2, d_ovvv)
        else
            pseudo_drop!(buf2, d_vovv)
        end
    end
    if calc_d_vvvo
        d_vvvo = pseudo_alloc!(buf2, nv1, nv2, nv1, no2)
        pseudo_drop!(buf2, d_vvvo)
        if mixed
            d_vvov = pseudo_alloc!(buf2, nv1, nv2, no1, nv2)
            pseudo_drop!(buf2, d_vvov)
        end
    end
    if mixed
        pseudo_drop!(buf1, d_ovvo)
    end
    pseudo_drop!(buf1, d_voov)
    if calc_d_vvoo
        d_vvoo = pseudo_alloc!(buf1, nv1, nv2, no1, no2)
        hd_vvvo = pseudo_alloc!(buf2, nv1, nv2, nv1, no2)
        pseudo_drop!(buf2, hd_vvvo)
        pseudo_drop!(buf1, d_vvoo)
    end
    if mixed
        pseudo_drop!(buf2, hd_ovoo)
    end
    pseudo_drop!(buf2, hd_vooo)
    return (buf1[2], buf2[2])
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
  mem1 = free_memory()
  mixed = (o1 != o2)
  no1, no2, nv1, nv2 = len_spaces(EC,o1*o2*v1*v2)
  lenbuf1, lenbuf2 = auto_buf_length4calc_dressed_ints(no1, no2, nv1, nv2, calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo, mixed)
  @buffer buf1(lenbuf1) buf2(lenbuf2) begin
  # @print_buffer_usage buf1 buf2 begin
  mem2 = print_memory(EC, mem1, "for buffers for dressed integrals", 2)
  # first make half-transformed integrals
  if calc_d_vvvv
    # <a\hat c|bd>
    hd_vvvv = alloc!(buf1, nv1,nv2,nv1,nv2)
    ints2!(hd_vvvv, EC, v1*v2*v1*v2)
    vovv = alloc!(buf2, nv1,no2,nv1,nv2)
    ints2!(vovv, EC, v1*o2*v1*v2)
    @mtensor hd_vvvv[a,c,b,d] -= vovv[a,k,b,d] * T12[c,k]
    save!(EC, "hd_"*v1*v2*v1*v2, hd_vvvv)
    drop!(buf2, vovv)
    drop!(buf1, hd_vvvv)
    t1 = print_time(EC, t1, "dress hd_"*v1*v2*v1*v2, 3)
  end
  # <ik|j \hat l>
  hd_oooo = alloc!(buf1, no1,no2,no1,no2)
  ints2!(hd_oooo, EC, o1*o2*o1*o2)
  if mixed
    # <ik|\hat j l>
    hd_oooo2 = alloc!(buf1, no1,no2,no1,no2)
    hd_oooo2 .= hd_oooo
    oovo = alloc!(buf2, no1,no2,nv1,no2)
    ints2!(oovo, EC, o1*o2*v1*o2)
    @mtensor hd_oooo2[i,j,k,l] += oovo[i,j,d,l] * T1[d,k]
    drop!(buf2, oovo)
  end
  ooov = alloc!(buf2, no1,no2,no1,nv2)
  ints2!(ooov, EC, o1*o2*o1*v2)
  @mtensor hd_oooo[i,j,k,l] += ooov[i,j,k,d] * T12[d,l]
  drop!(buf2, ooov)
  t1 = print_time(EC, t1, "dress hd_"*o1*o2*o1*o2, 3)
  if calc_d_vvoo
    # <a\hat c|j \hat l>
    hd_vvoo = alloc!(buf1, nv1,nv2,no1,no2)
    ints2!(hd_vvoo, EC, v1*v2*o1*o2)
    vooo = alloc!(buf2, nv1,no2,no1,no2)
    ints2!(vooo, EC, v1*o2*o1*o2)
    voov = alloc!(buf1, nv1,no2,no1,nv2)
    ints2!(voov, EC, v1*o2*o1*v2)
    @mtensor begin
      vooo[a,k,j,l] += voov[a,k,j,d] * T12[d,l]
      hd_vvoo[a,c,j,l] -= vooo[a,k,j,l] * T12[c,k]
    end
    drop!(buf1, voov)
    drop!(buf2, vooo)
    vvov = alloc!(buf2, nv1,nv2,no1,nv2)
    ints2!(vvov, EC, v1*v2*o1*v2)
    @mtensor hd_vvoo[a,c,j,l] += vvov[a,c,j,d] * T12[d,l]
    drop!(buf2, vvov)
    save!(EC, "hd_"*v1*v2*o1*o2, hd_vvoo)
    drop!(buf1, hd_vvoo)
    t1 = print_time(EC, t1, "dress hd_"*v1*v2*o1*o2, 3)
  end
  # <\hat a k| \hat j l>
  hd_vooo = alloc!(buf2, nv1,no2,no1,no2)
  ints2!(hd_vooo, EC, v1*o2*o1*o2)
  if !mixed
    @mtensor hd_vooo[a,k,j,l] -= hd_oooo[k,i,l,j] * T1[a,i]
  else
    @mtensor hd_vooo[a,k,j,l] -= hd_oooo2[i,k,j,l] * T1[a,i]
    drop!(buf1, hd_oooo2)
  end
  if no2 > 0
    vovo = alloc!(buf1, nv1,no2,nv1,no2)
    ints2!(vovo, EC, v1*o2*v1*o2)
    @mtensor hd_vooo[a,k,j,l] += vovo[a,k,b,l] * T1[b,j]
    drop!(buf1, vovo)
  end
  t1 = print_time(EC, t1, "dress hd_"*v1*o2*o1*o2, 3)
  if mixed
    # <k\hat a | l\hat j >
    hd_ovoo = alloc!(buf2, no1,nv2,no1,no2)
    ints2!(hd_ovoo, EC, o1*v2*o1*o2)
    ovov = alloc!(buf1, no1,nv2,no1,nv2)
    ints2!(ovov, EC, o1*v2*o1*v2)
    if no1 > 0 && no2 > 0
      @mtensor begin
        hd_ovoo[k,a,l,j] -= hd_oooo[k,i,l,j] * T12[a,i]
        hd_ovoo[k,a,l,j] += ovov[k,a,l,b] * T12[b,j]
      end
    end
    drop!(buf1, ovov)
    t1 = print_time(EC, t1, "dress hd_"*o1*v2*o1*o2, 3)
  end
  # some of the fully dressing moved here...
  # <ki\hat|dj>
  d_oovo = alloc!(buf2, no1,no2,nv1,no2)
  ints2!(d_oovo, EC, o1*o2*v1*o2)
  oovv = alloc!(buf1, no1,no2,nv1,nv2)
  ints2!(oovv, EC, o1*o2*v1*v2)
  @mtensor d_oovo[k,i,d,j] += oovv[k,i,d,b] * T12[b,j]
  save!(EC, "d_"*o1*o2*v1*o2, d_oovo)
  drop!(buf1, oovv)
  t1 = print_time(EC, t1, "dress d_"*o1*o2*v1*o2, 3)
  # <ij\hat|kl>
  @mtensor hd_oooo[i,k,j,l] += d_oovo[i,k,b,l] * T1[b,j]
  save!(EC, "d_"*o1*o2*o1*o2, hd_oooo)
  drop!(buf1, hd_oooo)
  if mixed
    d_ooov = alloc!(buf2, no1,no2,no1,nv2)
    ints2!(d_ooov, EC, o1*o2*o1*v2)
  end
  # <ak\hat|jd>
  vovv = alloc!(buf2, nv1,no2,nv1,nv2)
  ints2!(vovv, EC, v1*o2*v1*v2)
  d_voov = alloc!(buf1, nv1,no2,no1,nv2)
  ints2!(d_voov, EC, v1*o2*o1*v2)
  if mixed
    # <oo|ov>
    oOvV = alloc!(buf1, no1,no2,nv1,nv2)
    ints2!(oOvV, EC, o1*o2*v1*v2)
    @mtensor d_ooov[k,l,j,d] += oOvV[k,l,b,d] * T1[b,j]
    drop!(buf1, oOvV)
    save!(EC, "d_"*o1*o2*o1*v2, d_ooov)
    t1 = print_time(EC, t1, "dress d_"*o1*o2*o1*v2, 3)
    if no1 > 0 && no2 > 0
      @mtensor begin
        d_voov[a,i,j,d] -= d_ooov[k,i,j,d] * T1[a,k]
        d_voov[a,i,j,d] += vovv[a,i,b,d] * T1[b,j]
      end
    end
    save!(EC, "d_"*v1*o2*o1*v2, d_voov)
  else
    if no1 > 0 && no2 > 0
      @mtensor begin
        d_voov[a,k,j,d] -= d_oovo[k,i,d,j] * T1[a,i]
        d_voov[a,k,j,d] += vovv[a,k,b,d] * T1[b,j]
      end
    end
    save!(EC, "d_"*v1*o2*o1*v2, d_voov)
  end
  t1 = print_time(EC, t1, "dress d_"*v1*o2*o1*v2, 3)
  # finish half-dressing
  if mixed
    d_ovvo = alloc!(buf1, no1,nv2,nv1,no2)
    ints2!(d_ovvo, EC, o1*v2*v1*o2)
  end
  # <ak|b \hat l>
  hd_vovo = alloc!(buf1, nv1,no2,nv1,no2)
  ints2!(hd_vovo, EC, v1*o2*v1*o2)
  if no2 > 0
    @mtensor hd_vovo[a,k,b,l] += vovv[a,k,b,d] * T12[d,l]
  end
  drop!(buf2, vovv)
  if mixed
    # <k\hat a|dj>
    ovvv = alloc!(buf2, no1,nv2,nv1,nv2)
    ints2!(ovvv, EC, o1*v2*v1*v2)
    @mtensor begin
      d_ovvo[i,A,b,J] -= d_oovo[i,K,b,J] * T12[A,K]
      d_ovvo[i,A,b,J] += ovvv[i,A,b,C] * T12[C,J]
    end
    save!(EC, "d_"*o1*v2*v1*o2, d_ovvo)
    t1 = print_time(EC, t1, "dress d_"*o1*v2*v1*o2, 3)

    hd_ovov = alloc!(buf1, no1,nv2,no1,nv2)
    ints2!(hd_ovov, EC, o1*v2*o1*v2)
    @mtensor hd_ovov[k,a,l,b] += ovvv[k,a,d,b] * T1[d,l]
    drop!(buf2, ovvv)
  end
  t1 = print_time(EC, t1, "dress hd_"*v1*o2*v1*o2, 3)
  if calc_d_vvvo
    # <a\hat c|b \hat l>
    vvvv = alloc!(buf1, nv1,nv2,nv1,nv2)
    ints2!(vvvv, EC, v1*v2*v1*v2)
    hd_vvvo = alloc!(buf2, nv1,nv2,nv1,no2)
    ints2!(hd_vvvo, EC, v1*v2*v1*o2)
    @mtensor begin
      hd_vvvo[a,c,b,l] -= hd_vovo[a,k,b,l] * T12[c,k]
      hd_vvvo[a,c,b,l] += vvvv[a,c,b,d] * T12[d,l]
    end
    save!(EC, "hd_"*v1*v2*v1*o2, hd_vvvo)
    drop!(buf2, hd_vvvo)
    if mixed
      hd_vvov = alloc!(buf2, nv1,nv2,no1,nv2)
      ints2!(hd_vvov, EC, v1*v2*o1*v2)
      @mtensor begin
        hd_vvov[a,c,l,b] -= hd_ovov[k,c,l,b] * T1[a,k]
        hd_vvov[a,c,l,b] += vvvv[a,c,d,b] * T1[d,l]
      end
      save!(EC, "hd_"*v1*v2*o1*v2, hd_vvov)
      drop!(buf2, hd_vvov)
    end
    drop!(buf1, vvvv)
    t1 = print_time(EC, t1, "dress hd_"*v1*v2*v1*o2, 3)
  end

  # fully dressed
  # <ak\hat|bl>
  if mixed
    @mtensor hd_ovov[k,a,l,b] -= d_ooov[k,i,l,b] * T12[a,i]
    save!(EC, "d_"*o1*v2*o1*v2, hd_ovov)
    drop!(buf1, hd_ovov)
  end
  @mtensor hd_vovo[a,k,b,l] -= d_oovo[i,k,b,l] * T1[a,i]
  save!(EC, "d_"*v1*o2*v1*o2, hd_vovo)
  drop!(buf1, hd_vovo)
  if mixed
    drop!(buf2, d_ooov)
  end
  drop!(buf2, d_oovo)
  t1 = print_time(EC, t1, "dress d_"*v1*o2*v1*o2, 3)
  if calc_d_vovv
    # <ak\hat|bd>
    d_vovv = alloc!(buf2, nv1,no2,nv1,nv2)
    ints2!(d_vovv, EC, v1*o2*v1*v2)
    oovv = alloc!(buf1, no1,no2,nv1,nv2)
    ints2!(oovv, EC, o1*o2*v1*v2)
    @mtensor d_vovv[a,k,b,d] -= oovv[i,k,b,d] * T1[a,i]
    save!(EC, "d_"*v1*o2*v1*v2, d_vovv)
    t1 = print_time(EC, t1, "dress d_"*v1*o2*v1*v2, 3)
    if mixed
      drop!(buf2, d_vovv)
      d_ovvv = alloc!(buf2, no1,nv2,nv1,nv2)
      ints2!(d_ovvv, EC, o1*v2*v1*v2)
      @mtensor d_ovvv[i,b,a,c] -= oovv[i,j,a,c] * T12[b,j]
      save!(EC, "d_"*o1*v2*v1*v2, d_ovvv)
      t1 = print_time(EC, t1, "dress d_"*o1*v2*v1*v2, 3)
    end
    drop!(buf1, oovv)
  end
  if calc_d_vvvv
    # <ab\hat|cd>
    d_vvvv = alloc!(buf1, nv1,nv2,nv1,nv2)
    load!(EC, "hd_"*v1*v2*v1*v2, d_vvvv)
    if !calc_d_vovv
      error("for calc_d_vvvv calc_d_vovv has to be True")
    end
    if !mixed
      @mtensor d_vvvv[a,c,b,d] -= d_vovv[c,i,d,b] * T1[a,i]
      drop!(buf2, d_vovv)
    else
      @mtensor d_vvvv[a,c,b,d] -= d_ovvv[i,c,b,d] * T1[a,i]
      drop!(buf2, d_ovvv)
    end
    save!(EC, "d_"*v1*v2*v1*v2, d_vvvv)
    drop!(buf1, d_vvvv)
    t1 = print_time(EC, t1, "dress d_"*v1*v2*v1*v2, 3)
  elseif calc_d_vovv
    if mixed
      drop!(buf2, d_ovvv)
    else
      drop!(buf2, d_vovv)
    end
  end
  # <aj\hat|kl>
  d_vooo = hd_vooo
  if no1 > 0 && no2 > 0
    @mtensor d_vooo[a,k,j,l] += d_voov[a,k,j,d] * T12[d,l]
  end
  save!(EC, "d_"*v1*o2*o1*o2, d_vooo)
  if mixed
    d_ovoo = hd_ovoo
    @mtensor d_ovoo[k,a,l,j] += d_ovvo[k,a,d,j] * T1[d,l]
    save!(EC, "d_"*o1*v2*o1*o2, d_ovoo)
  end
  t1 = print_time(EC, t1, "dress d_"*v1*o2*o1*o2, 3)
  if calc_d_vvvo
    # <ab\hat|cl>
    d_vvvo = alloc!(buf2, nv1,nv2,nv1,no2)
    load!(EC, "hd_"*v1*v2*v1*o2, d_vvvo)
    if !mixed
      @mtensor d_vvvo[a,c,b,l] -= d_voov[c,i,l,b] * T1[a,i]
    else
      @mtensor d_vvvo[c,a,b,l] -= d_ovvo[i,a,b,l] * T1[c,i]
    end
    save!(EC, "d_"*v1*v2*v1*o2, d_vvvo)
    drop!(buf2, d_vvvo)
    if mixed
      d_vvov = alloc!(buf2, nv1,nv2,no1,nv2)
      load!(EC, "hd_"*v1*v2*o1*v2, d_vvov)
      @mtensor d_vvov[a,c,l,b] -= d_voov[a,i,l,b] * T12[c,i]
      save!(EC, "d_"*v1*v2*o1*v2, d_vvov)
      drop!(buf2, d_vvov)
    end
    t1 = print_time(EC, t1, "dress d_"*v1*v2*v1*o2, 3)
  end
  if mixed
    drop!(buf1, d_ovvo)
  end
  drop!(buf1, d_voov)
  t1 = print_time(EC, t1, "dress d_"*o1*o2*o1*o2, 3)
  if calc_d_vvoo
    if !calc_d_vvvo
      error("for calc_d_vvoo calc_d_vvvo has to be True")
    end
    # <ac\hat|jl>
    d_vvoo = alloc!(buf1, nv1,nv2,no1,no2)
    load!(EC, "hd_"*v1*v2*o1*o2, d_vvoo)
    hd_vvvo = alloc!(buf2, nv1,nv2,nv1,no2)
    load!(EC, "hd_"*v1*v2*v1*o2, hd_vvvo)
    @mtensor d_vvoo[a,c,j,l] += hd_vvvo[a,c,b,l] * T1[b,j]
    drop!(buf2, hd_vvvo)
    if !mixed
      @mtensor d_vvoo[a,c,j,l] -= d_vooo[c,i,l,j] * T1[a,i] 
    else
      @mtensor d_vvoo[a,c,j,l] -= d_ovoo[i,c,j,l] * T1[a,i] 
    end
    save!(EC, "d_"*v1*v2*o1*o2, d_vvoo)
    drop!(buf1, d_vvoo)
    t1 = print_time(EC, t1, "dress d_"*v1*v2*o1*o2, 3)
  end
  if mixed
    drop!(buf2, hd_ovoo)
  end
  drop!(buf2, hd_vooo)
  end # buffer
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
  @mtensor d_int1[:,SP['o']][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[SP['o'],:]
  @mtensor d_int1[SP['v'],:][b,p] -= dinter[j,p] * T1[b,j]
  # display(d_int1[SP['v'],SP['o']])
  save!(EC,"dh_mm",d_int1)
  t1 = print_time(EC,t1,"dress int1",3)

  # calc dressed fock
  dfock = d_int1
  d_oooo = load4idx(EC,"d_oooo")
  d_vooo = load4idx(EC,"d_vooo")
  d_oovo = load4idx(EC,"d_oovo")
  @mtensor begin
    foo[i,j] := 2.0*d_oooo[i,k,j,k] - d_oooo[i,k,k,j]
    fvo[a,i] := 2.0*d_vooo[a,k,i,k] - d_vooo[a,k,k,i]
    fov[i,a] := 2.0*d_oovo[i,k,a,k] - d_oovo[k,i,a,k]
  end
  d_vovo = load4idx(EC,"d_vovo")
  @mtensor fvv[a,b] := 2.0*d_vovo[a,k,b,k]
  d_vovo = NOTHING4idx
  d_voov = load4idx(EC,"d_voov")
  @mtensor fvv[a,b] -= d_voov[a,k,k,b]
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
  @mtensor d_int1[:,SP[o1]][p,j] += dinter[p,b] * T1[b,j]
  dinter = d_int1[SP[o1],:]
  @mtensor d_int1[SP[v1],:][b,p] -= dinter[j,p] * T1[b,j]
  save!(EC,"dh_"*mo*mo,d_int1)
  t1 = print_time(EC,t1,"dress int1",3)
  # calc dressed fock
  dfock = d_int1
  d_oooo = load4idx(EC,"d_"*o1*o1*o1*o1)
  d_vooo = load4idx(EC,"d_"*v1*o1*o1*o1)
  d_oovo = load4idx(EC,"d_"*o1*o1*v1*o1)
  @mtensor begin
    foo[i,j] := d_oooo[i,k,j,k] - d_oooo[i,k,k,j]
    fvo[a,i] := d_vooo[a,k,i,k] - d_vooo[a,k,k,i]
    fov[i,a] := d_oovo[i,k,a,k] - d_oovo[k,i,a,k] 
  end
  d_vovo = load4idx(EC,"d_"*v1*o1*v1*o1)
  @mtensor fvv[a,b] := d_vovo[a,k,b,k]
  d_vovo = NOTHING4idx
  if no1 > 0 
    d_voov = load4idx(EC,"d_"*v1*o1*o1*v1)
    @mtensor fvv[a,b] -= d_voov[a,k,k,b]
    d_voov = NOTHING4idx
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
  d_oooo = load4idx(EC,"d_oOoO")
  @mtensor begin
    foo[i,j] := d_oooo[i,k,j,k]
    fOO[i,j] := d_oooo[k,i,k,j]
  end
  d_oooo = NOTHING4idx
  d_vooo = load4idx(EC,"d_vOoO")
  @mtensor fvo[a,i] := d_vooo[a,k,i,k]
  d_vooo = NOTHING4idx
  d_ovoo = load4idx(EC,"d_oVoO")
  @mtensor fVO[a,i] := d_ovoo[k,a,k,i]
  d_ovoo = NOTHING4idx
  d_oovo = load4idx(EC,"d_oOvO")
  @mtensor fov[i,a] := d_oovo[i,k,a,k]
  d_oovo = NOTHING4idx
  d_ooov = load4idx(EC,"d_oOoV")
  @mtensor fOV[i,a] := d_ooov[k,i,k,a]
  d_ooov = NOTHING4idx
  d_vovo = load4idx(EC,"d_vOvO")
  @mtensor fvv[a,b] := d_vovo[a,k,b,k]
  d_vovo = NOTHING4idx
  d_ovov = load4idx(EC,"d_oVoV")
  @mtensor fVV[a,b] := d_ovov[k,a,k,b]
  d_ovov = NOTHING4idx

  dfocka = load2idx(EC,"df_mm")
  dfocka[SP['o'],SP['o']] += foo
  dfocka[SP['o'],SP['v']] += fov
  dfocka[SP['v'],SP['o']] += fvo
  dfocka[SP['v'],SP['v']] += fvv
  save!(EC,"df_mm",dfocka)

  dfockb = load2idx(EC,"df_MM")
  dfockb[SP['O'],SP['O']] += fOO
  dfockb[SP['O'],SP['V']] += fOV
  dfockb[SP['V'],SP['O']] += fVO
  dfockb[SP['V'],SP['V']] += fVV
  save!(EC,"df_MM",dfockb)
end

"""
    calc_dressed_ints(EC::ECInfo, T1;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)

  Dress integrals with singles.

  ``\\hat v_{ab}^{cd}``, ``\\hat v_{ab}^{ci}``, ``\\hat v_{ak}^{cd}`` and ``\\hat v_{ab}^{ij}`` are only 
  calculated if requested in `EC.options.cc` or using keyword-arguments.
"""
function calc_dressed_ints(EC::ECInfo, T1;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)
  calc_dressed_ints(EC, T1, T1, "ovov"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
  dress_fock_closedshell(EC, T1)
end
function calc_dressed_ints(EC::ECInfo, T1a, T1b;
              calc_d_vvvv=EC.options.cc.calc_d_vvvv, calc_d_vvvo=EC.options.cc.calc_d_vvvo,
              calc_d_vovv=EC.options.cc.calc_d_vovv, calc_d_vvoo=EC.options.cc.calc_d_vvoo)
  calc_dressed_ints(EC, T1a, T1a, "ovov"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
  calc_dressed_ints(EC, T1b, T1b, "OVOV"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
  calc_dressed_ints(EC, T1a, T1b, "ovOV"...; calc_d_vvvv, calc_d_vvvo, calc_d_vovv, calc_d_vvoo)
  dress_fock_samespin(EC, T1a, "ov"...)
  dress_fock_samespin(EC, T1b, "OV"...)
  dress_fock_oppositespin(EC)
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
  save!(EC,"df_mm",load2idx(EC,"f_mm"))
  save!(EC,"df_MM",load2idx(EC,"f_MM"))
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
  Return EMp2 `OutDict` with keys (`E`, `ESS`, `EOS`, `EO`).
"""
function calc_MP2(EC::ECInfo, addsingles=true)
  T2 = update_doubles(EC, ints2(EC,"vvoo"), use_shift=false)
  EMp2 = calc_doubles_energy(EC, T2)
  save!(EC, "T_vvoo", T2)
  if addsingles
    ϵo, ϵv = orbital_energies(EC)
    T1 = update_singles(load2idx(EC,"f_mm")[EC.space['v'],EC.space['o']], ϵo, ϵv, 0.0)
    EMp2s = calc_singles_energy(EC, T1, fock_only=true)
    save!(EC, "T_vo", T1)
    # add singles energies to MP2 energies
    EMp2 = map(+, EMp2, EMp2s) 
  end
  return EMp2
end

""" 
    calc_UMP2(EC::ECInfo, addsingles=true)

  Calculate unrestricted MP2 energy and amplitudes. 
  The amplitudes are stored in `T_vvoo`, `T_VVOO`, and `T_vVoO` files.
  If `addsingles`: singles are also calculated and stored in `T_vo` and `T_VO` files.
  Return EMp2 `OutDict` with keys (`E`, `ESS`, `EOS`, `EO`).
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
    T1a = update_singles(EC, load2idx(EC,"f_mm")[SP['v'],SP['o']], spincase=:α, use_shift=false)
    T1b = update_singles(EC, load2idx(EC,"f_MM")[SP['V'],SP['O']], spincase=:β, use_shift=false)
    EMp2s = calc_singles_energy(EC, T1a, T1b, fock_only=true)
    save!(EC, "T_vo", T1a)
    save!(EC, "T_VO", T1b)
    # add singles energies to MP2 energies
    EMp2 = map(+, EMp2, EMp2s)
  end
  return EMp2
end

""" 
    calc_UMP2_energy(EC::ECInfo, addsingles=true)

  Calculate open-shell MP2 energy from precalculated amplitudes. 
  If `addsingles`: singles energy is also calculated.
  Return EMp2 `OutDict` with keys (`E`, `ESS`, `EOS`, `EO`).
"""
function calc_UMP2_energy(EC::ECInfo, addsingles=true)
  T2a = load4idx(EC,"T_vvoo")
  T2b = load4idx(EC,"T_VVOO")
  T2ab = load4idx(EC,"T_vVoO")
  EMp2 = calc_doubles_energy(EC, T2a, T2b, T2ab)
  if addsingles
    T1a = load2idx(EC,"T_vo")
    T1b = load2idx(EC,"T_VO")
    EMp2s = calc_singles_energy(EC, T1a, T1b, fock_only=true)
    # add singles energies to MP2 energies
    EMp2 = map(+, EMp2, EMp2s)
  end
  return EMp2
end

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
  @mtensor begin
    D2[SP['v'],SP['v'],:,:][a,b,i,j] = T2[a,b,i,j] 
    D2[SP['o'],SP['o'],:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k]
  end
  if length(T1) > 0
    @mtensor begin
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
  @mtensor begin
    D2[virt,virt,:,:][a,b,i,j] = T2[a,b,i,j] 
    D2[occ,occ,:,:][i,k,j,l] = Matrix(I,nocc,nocc)[i,j] * Matrix(I,nocc,nocc)[l,k] - Matrix(I,nocc,nocc)[k,j] * Matrix(I,nocc,nocc)[l,i]
  end
  if length(T1) > 0
    @mtensor begin
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
  @mtensor begin
    D2ab[SP['v'],SP['V'],:,:][a,B,i,J] = T2ab[a,B,i,J] 
    D2ab[SP['o'],SP['O'],:,:][i,k,j,l] = Matrix(I,nocca,nocca)[i,j] * Matrix(I,noccb,noccb)[l,k]
  end
  if length(T1a) > 0
    @mtensor begin
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
    calc_E_Coe(eigenvalue_vector, q)

  Calculate the coefficient matrix in QV-CCD/DCD residual calculation.
"""
function calc_E_Coe(eigenvalue_vector, q, threshold=1e-10)
  coefficient_matrix = zeros(length(eigenvalue_vector), length(eigenvalue_vector))
  for i in eachindex(eigenvalue_vector)
    if eigenvalue_vector[i] < threshold
      println("WARNING: SMALL EIGENVALUE DETECTED IN calc_E_Coe() ", eigenvalue_vector[i])
    end
  end
  if q == 1
    evq = sqrt.(eigenvalue_vector)
  elseif q == 2
    evq = eigenvalue_vector
  else
    error("q must be 1 or 2")
  end
  for i in eachindex(eigenvalue_vector)
    for j in eachindex(eigenvalue_vector)
      if i == j
        coefficient_matrix[i,j] = 0      
      elseif q == 1
        coefficient_matrix[i,j] = -1/(evq[i]*evq[j]*(evq[i] + evq[j]))
      else
        coefficient_matrix[i,j] = -1/(evq[i] * evq[j])
      end
    end
  end
  return coefficient_matrix
end

"""
    calc_R_from_U_F(e, X, F, q)

  Calculate intermediate R with F and eigenvalue e, eigenvectors X of corresponding U in QV-CCD/DCD.
"""
function calc_R_from_U_F(e, X, F, q)
  E = calc_E_Coe(e, q)
  R = X' * F * X
  z = diag(R)
  R .= R .* E
  z .= e.^(-q/2.0-1.0) .* z .* (-q/2.0)
  R .+= Diagonal(z)
  R .= X * R * X'
  return R
end

"""
    calc_qvcc_resid(EC::ECInfo, T1, T2; dc=false)

  Calculate QV-CCD or QV-DCD closed-shell residual.
"""
function calc_qvcc_resid(EC::ECInfo, it::Int, T1, T2; dc=false)
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  I_ab = Matrix{Float64}(I,nvirt,nvirt)
  I_ij = Matrix{Float64}(I,nocc,nocc)
  @mtensor begin
    AU[b,a] := I_ab[b,a] + 2.0 * T2[a,c,i,j] * T2[b,c,i,j] - T2[a,c,i,j] * T2[c,b,i,j]
    BU[i,j] := I_ij[i,j] +  2.0 * T2[a,b,i,k] * T2[a,b,j,k] - T2[a,b,i,k] * T2[a,b,k,j]
    CU[i,j,k,l] := I_ij[i,k] * I_ij[j,l] + T2[a,b,k,l] * T2[a,b,i,j]
    Y[a,i,b,j] := I_ab[a,b] * I_ij[i,j] + 4.0 * T2[a,c,i,k] * T2[b,c,j,k] - 2.0 * T2[c,a,i,k] * T2[b,c,j,k] - 2.0 * T2[a,c,i,k] * T2[c,b,j,k] + T2[c,a,i,k] * T2[c,b,j,k]
    W[a,i,b,j] := I_ab[a,b] * I_ij[i,j] + T2[c,a,i,k] * T2[c,b,j,k]
  end

  AU .= 0.5 .* (AU .+ AU')
  BU .= 0.5 .* (BU .+ BU')
  CU .= 0.5 .* (CU .+ permutedims(CU, (3,4,1,2)))
  Y .= 0.5 .* (Y .+ permutedims(Y, (3,4,1,2)))
  W .= 0.5 .* (W .+ permutedims(W, (3,4,1,2)))
 
  # a function that can calculate the eigenvectors & eigenvalues of a matrix
  # CU[i,j,k,l] -> CU[ij,kl], Y[a,i,b,j] -> Y[ai,bj], W[a,i,b,j] -> W[ai,bj]
  # corresponding to \pre{_C}U^{ij}_{kl}, Y^{aj}_{bi}, W^{aj}_{bi}
  Ae, AX = eigen(Hermitian(AU))
  Be, BX = eigen(Hermitian(BU))
  Ce, CX = eigen(Hermitian(reshape(CU, nocc^2, nocc^2)))
  Ye, YX = eigen(Hermitian(reshape(Y, nvirt*nocc, nvirt*nocc)))
  We, WX = eigen(Hermitian(reshape(W, nvirt*nocc, nvirt*nocc)))
  G2 = zeros(nvirt, nvirt, nocc, nocc)
  E_qvccd = 0.0
  for q in [1.0, 2.0]
    AU1 = AX * Diagonal(Ae .^ (-q/2)) * AX' # AU1 = AU ^ (-q/2)
    BU1 = BX * Diagonal(Be .^ (-q/2)) * BX' # BU1 = BU ^ (-q/2)
    CU1 = reshape(CX * Diagonal(Ce .^ (-q/2)) * CX', nocc, nocc, nocc, nocc) # CU1 = CU ^ (-q/2)
    Y1 = reshape(YX * Diagonal(Ye .^ (-q/2)) * YX', nvirt, nocc, nvirt, nocc) # Y1 = reshape(Y ^ (-q/2), nvirt, nocc, nvirt, nocc)
    W1 = reshape(WX * Diagonal(We .^ (-q/2)) * WX', nvirt, nocc, nvirt, nocc) # W1 = reshape(W ^ (-q/2), nvirt, nocc, nvirt, nocc)

    @mtensor begin
      YWT[a,b,i,j] := Y1[c,k,a,i]* (T2[c,b,k,j] - 0.5 * T2[b,c,k,j]) + 
                    0.5*W1[c,k,a,i] * T2[b,c,k,j] + W1[c,k,a,j] * T2[c,b,i,k]
      qT[a,b,i,j] := AU1[a,c] * T2[c,b,i,j] + AU1[b,c] * T2[a,c,i,j] + BU1[k,i] * T2[a,b,k,j] + BU1[k,j] * T2[a,b,i,k] - 
                    CU1[i,j,k,l] * T2[a,b,k,l] - 0.5* YWT[a,b,i,j] - 0.5 * YWT[b,a,j,i]
    end
    if q == 1.0
      T1_0 = zeros(0,0)
      R1, qV = calc_cc_resid(EC, T1_0, qT; linearized=true) 
      qV .-= ints2(EC, "vvoo")
    else
      qV = ints2(EC, "vvoo")
    end
    qV .= 2.0 * qV .- permutedims(qV, (2,1,3,4))
    E_qvccd += sum(qV .* qT) * q

    @mtensor begin
      qAF[c,a] := qV[a,b,i,j] * T2[c,b,i,j]
      qBF[i,k] := qV[a,b,i,j] * T2[a,b,k,j]
      qCF[i,j,k,l] := qV[a,b,k,l] * T2[a,b,i,j]
      q1DF[a,i,c,k] := qV[a,b,i,j] * (T2[c,b,k,j] - 0.5*T2[b,c,k,j])
      q2DF[a,i,c,k] := 0.5*qV[a,b,i,j] * T2[b,c,k,j] + qV[a,b,j,i] * T2[c,b,j,k]
    end

    qCF = reshape(qCF, nocc^2, nocc^2)
    q1DF = reshape(q1DF, nvirt*nocc, nvirt*nocc)
    q2DF = reshape(q2DF, nvirt*nocc, nvirt*nocc)

    qAR = calc_R_from_U_F(Ae, AX, qAF, q)
    qBR = calc_R_from_U_F(Be, BX, qBF, q)
    qCR = calc_R_from_U_F(Ce, CX, qCF, q)
    q1DR = calc_R_from_U_F(Ye, YX, q1DF, q)
    q2DR = calc_R_from_U_F(We, WX, q2DF, q)

    qAR .= 0.5 .* (qAR .+ qAR')
    qBR .= 0.5 .* (qBR .+ qBR')
    qCR .= 0.5 .* (qCR .+ qCR')
    q1DR .= 0.5 .* (q1DR .+ q1DR')
    q2DR .= 0.5 .* (q2DR .+ q2DR')

    qCR = reshape(qCR, nocc, nocc, nocc, nocc)
    q1DR = reshape(q1DR, nvirt, nocc, nvirt, nocc)
    q2DR = reshape(q2DR, nvirt, nocc, nvirt, nocc)

    @mtensor qG[a,b,i,j] := 2.0 * qAR[d,a] * (2.0*T2[d,b,i,j] - T2[b,d,i,j]) + qV[c,b,i,j] * AU1[c,a] + 
                              2.0 * qBR[l,i] * (2.0*T2[a,b,l,j] - T2[b,a,l,j]) + qV[a,b,k,j] * BU1[i,k] +
                              (-0.5) * (2.0 * qCR[m,n,i,j] * T2[a,b,m,n] + qV[a,b,k,l] * CU1[k,l,i,j]) +
                              (-0.5) * (q1DR[a,i,c,k] * (8.0 * T2[c,b,k,j] - 4.0 * T2[b,c,k,j]) 
                              - q1DR[b,i,c,k]* (4.0 * T2[c,a,k,j] - 2.0 * T2[a,c,k,j])
                              + 2.0 * q2DR[b,i,c,k] * T2[a,c,k,j] 
                              + qV[c,b,k,j] * Y1[a,i,c,k] 
                              - 0.5 * qV[c,a,k,j] * Y1[b,i,c,k]
                              + 0.5 * qV[c,a,k,j] * W1[b,i,c,k] 
                              + qV[c,b,i,k] * W1[a,j,c,k])
    qG .+= permutedims(qG, (2,1,4,3))
    G2 += qG
  end
  G2 .= G2 .* 2.0/3.0 .+ permutedims(G2, (2,1,3,4)) .* 1.0/3.0
  return (T1,G2), E_qvccd
end


"""
    calc_cc_resid(EC::ECInfo, T1, T2; dc=false, tworef=false, fixref=false, linearized=false)

  Calculate CCSD or DCSD closed-shell residual.
"""
function calc_cc_resid(EC::ECInfo, T1, T2; dc=false, tworef=false, fixref=false, linearized=false)
  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  norb = n_orbs(EC)
  if length(T1) > 0
    calc_dressed_ints(EC, T1)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC)
  end
  @mtensor T2t[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  dfock = load2idx(EC,"df_mm")
  if length(T1) > 0
    if EC.options.cc.use_kext
      dint1 = load2idx(EC,"dh_mm")
      R1 = dint1[SP['v'],SP['o']]
    else
      R1 = dfock[SP['v'],SP['o']]
      if !EC.options.cc.calc_d_vovv
        error("for not use_kext calc_d_vovv has to be True")
      end
      int2 = load4idx(EC,"d_vovv")
      @mtensor R1[a,i] += int2[a,k,b,c] * T2t[c,b,k,i]
    end
    int2 = load4idx(EC,"d_oovo")
    fov = dfock[SP['o'],SP['v']]
    @mtensor begin
      R1[a,i] += T2t[a,b,i,j] * fov[j,b]
      R1[a,i] -= int2[k,j,c,i] * T2t[c,a,k,j]
    end
    t1 = print_time(EC,t1,"singles residual",2)
  else
    R1 = zero(T1)
  end

  # <ab|ij>
  if EC.options.cc.use_kext
    R2 = zeros(nvirt,nvirt,nocc,nocc)
  else
    if !EC.options.cc.calc_d_vvoo
      error("for not use_kext calc_d_vvoo has to be True")
    end
    R2 = load4idx(EC,"d_vvoo")
  end
  t1 = print_time(EC,t1,"<ab|ij>",2)
  klcd = ints2(EC,"oovv")
  t1 = print_time(EC,t1,"<kl|cd>",2)
  int2 = load4idx(EC,"d_oooo")
  if !dc && !linearized
    # I_klij = <kl|ij>+<kl|cd>T^ij_cd
    @mtensor int2[k,l,i,j] += klcd[k,l,c,d] * T2[c,d,i,j]
  end
  # I_klij T^kl_ab
  @mtensor R2[a,b,i,j] += int2[k,l,i,j] * T2[a,b,k,l]
  t1 = print_time(EC,t1,"I_klij T^kl_ab",2)
  if EC.options.cc.use_kext
    int2 = integ2_ss(EC.fd)
    # last two indices of integrals are stored as upper triangular 
    tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
    D2 = calc_D2(EC, T1, T2, true)[tripp,:,:]
    # <pq|rs> D^ij_rs
    @mtensor rK2pq[p,r,i,j] := int2[p,r,x] * D2[x,i,j]
    D2 = nothing
    # symmetrize R
    @mtensor K2pq[p,r,i,j] := rK2pq[p,r,i,j] + rK2pq[r,p,j,i]
    rK2pq = nothing
    R2 += K2pq[SP['v'],SP['v'],:,:]
    if length(T1) > 0
      @mtensor begin
        R2[a,b,i,j] -= K2pq[SP['o'],SP['v'],:,:][k,b,i,j] * T1[a,k]
        R2[a,b,i,j] -= K2pq[SP['v'],SP['o'],:,:][a,k,i,j] * T1[b,k]
        R2[a,b,i,j] += (K2pq[SP['o'],SP['o'],:,:][k,l,i,j] * T1[a,k]) * T1[b,l]
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
    int2 = load4idx(EC,"d_vvvv")
    # <ab|cd> T^ij_cd
    @mtensor R2[a,b,i,j] += int2[a,b,c,d] * T2[c,d,i,j]
    t1 = print_time(EC,t1,"<ab|cd> T^ij_cd",2)
  end
  if !dc && !linearized
    # <kl|cd> T^kj_ad T^il_cb
    @mtensor R2[a,b,i,j] += (klcd[k,l,c,d] * T2[a,d,k,j]) * T2[c,b,i,l]
    t1 = print_time(EC,t1,"<kl|cd> T^kj_ad T^il_cb",2)
  end

  fac = dc ? 0.5 : 1.0
  # x_ad = f_ad - <kl|cd> \tilde T^kl_ca
  # x_ki = f_ki + <kl|cd> \tilde T^il_cd
  xad = dfock[SP['v'],SP['v']]
  xki = dfock[SP['o'],SP['o']]
  if !linearized
    @mtensor begin
      xad[a,d] -= fac * (klcd[k,l,c,d] * T2t[c,a,k,l])
      xki[k,i] += fac * (klcd[k,l,c,d] * T2t[c,d,i,l])
    end
    t1 = print_time(EC,t1,"xad, xki",2)
  end

  # terms for P(ia;jb)
  @mtensor begin
    # x_ad T^ij_db
    R2r[a,b,i,j] := xad[a,d] * T2[d,b,i,j]
    # -x_ki T^kj_ab
    R2r[a,b,i,j] -= xki[k,i] * T2[a,b,k,j]
  end
  t1 = print_time(EC,t1,"x_ad T^ij_db -x_ki T^kj_ab",2)
  int2 = load4idx(EC,"d_voov")
  if !linearized
    # <kl|cd>\tilde T^ki_ca \tilde T^lj_db
    @mtensor int2[a,k,i,c] += 0.5*klcd[k,l,c,d] * T2t[a,d,i,l] 
  end
  # <ak|ic> \tilde T^kj_cb
  @mtensor R2r[a,b,i,j] += int2[a,k,i,c] * T2t[c,b,k,j]
  t1 = print_time(EC,t1,"<ak|ic> tT^kj_cb",2)
  if !dc && !linearized
    # -<kl|cd> T^ki_da (T^lj_cb - T^lj_bc)
    T2t -= T2
    @mtensor R2r[a,b,i,j] -= (klcd[k,l,c,d] * T2[d,a,k,i]) * T2t[c,b,l,j]
    t1 = print_time(EC,t1,"-<kl|cd> T^ki_da (T^lj_cb - T^lj_bc)",2)
  end
  int2 = load4idx(EC,"d_vovo")
  @mtensor begin
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
    calc_cc_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab; dc=false, tworef=false, fixref=false)

  Calculate UCCSD or UDCSD residual.
"""
function calc_cc_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab; dc=false, tworef=false, fixref=false)
  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nvirtb = n_virtb_orbs(EC)
  norb = n_orbs(EC)
  linearized::Bool = false

  if tworef
    active = oss_active_orbitals(EC)
    T2ab[active.ua,active.tb,active.ta,active.ub] = 0.0
  end

  if length(T1a) > 0 || length(T1b) > 0
    calc_dressed_ints(EC, T1a, T1b)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC, true)
  end

  R1a = zero(T1a)
  R1b = zero(T1b)

  dfock = load2idx(EC,"df_mm")
  dfockb = load2idx(EC,"df_MM")

  fij = dfock[SP['o'],SP['o']]
  fab = dfock[SP['v'],SP['v']]
  fIJ = dfockb[SP['O'],SP['O']]
  fAB = dfockb[SP['V'],SP['V']]

  if length(T1a) > 0
    if EC.options.cc.use_kext
      dint1a = load2idx(EC,"dh_mm")
      R1a = dint1a[SP['v'],SP['o']]
      dint1b = load2idx(EC,"dh_MM")
      R1b = dint1b[SP['V'],SP['O']]
    else
      fai = dfock[SP['v'],SP['o']]
      fAI = dfockb[SP['V'],SP['O']]
      @mtensor begin
        R1a[a,i] :=  fai[a,i]
        R1b[a,i] :=  fAI[a,i]
      end
      d_vovv = load4idx(EC,"d_vovv")
      @mtensor R1a[a,i] += d_vovv[a,k,b,d] * T2a[b,d,i,k]
      d_vovv = nothing
      d_VOVV = load4idx(EC,"d_VOVV")
      @mtensor R1b[A,I] += d_VOVV[A,K,B,D] * T2b[B,D,I,K]
      d_VOVV = nothing
      d_vOvV = load4idx(EC,"d_vOvV")
      @mtensor R1a[a,i] += d_vOvV[a,K,b,D] * T2ab[b,D,i,K]
      d_vOvV = nothing
      d_oVvV = load4idx(EC,"d_oVvV")
      @mtensor R1b[A,I] += d_oVvV[k,A,d,B] * T2ab[d,B,k,I]
      d_oVvV = nothing
      t1 = print_time(EC,t1,"``R_a^i += v_{ak}^{bd} T_{bd}^{ik}``",2)
    end
    fia = dfock[SP['o'],SP['v']]
    fIA = dfockb[SP['O'],SP['V']]
    @mtensor begin
      R1a[a,i] += fia[j,b] * T2a[a,b,i,j]
      R1b[A,I] += fIA[J,B] * T2b[A,B,I,J]
      R1a[a,i] += fIA[J,B] * T2ab[a,B,i,J]
      R1b[A,I] += fia[j,b] * T2ab[b,A,j,I]
    end
    t1 = print_time(EC,t1,"``R_a^i += f_j^b T_{ab}^{ij}``",2)
    if n_occ_orbs(EC) > 0 
      d_oovo = load4idx(EC,"d_oovo")
      @mtensor R1a[a,i] -= d_oovo[k,j,d,i] * T2a[a,d,j,k]
      d_oovo = nothing
    end
    if n_occb_orbs(EC) > 0
      d_OOVO = load4idx(EC,"d_OOVO")
      @mtensor R1b[A,I] -= d_OOVO[K,J,D,I] * T2b[A,D,J,K]
      d_OOVO = nothing
      d_oOoV = load4idx(EC,"d_oOoV")
      @mtensor R1a[a,i] -= d_oOoV[j,K,i,D] * T2ab[a,D,j,K]
      d_oOoV = nothing
      d_oOvO = load4idx(EC,"d_oOvO")
      @mtensor R1b[A,I] -= d_oOvO[k,J,d,I] * T2ab[d,A,k,J]
    end
    t1 = print_time(EC,t1,"``R_a^i -= v_{kj}^{di} T_{ad}^{jk}``",2)
  end

  #driver terms
  if EC.options.cc.use_kext
    R2a = zeros(nvirt, nvirt, nocc, nocc)
    R2b = zeros(nvirtb, nvirtb, noccb, noccb)
    R2ab = zeros(nvirt, nvirtb, nocc, noccb)
  else
    d_vvoo = load4idx(EC,"d_vvoo")
    R2a = deepcopy(d_vvoo)
    @mtensor R2a[a,b,i,j] -= d_vvoo[b,a,i,j]
    d_vvoo = nothing
    d_VVOO = load4idx(EC,"d_VVOO")
    R2b = deepcopy(d_VVOO)
    @mtensor R2b[A,B,I,J] -= d_VVOO[B,A,I,J]
    d_VVOO = nothing
    R2ab = load4idx(EC,"d_vVoO")
  end
  
  #ladder terms
  if EC.options.cc.use_kext
    # last two indices of integrals (apart from αβ) are stored as upper triangular 
    tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
    if EC.fd.uhf
      # αα
      int2a = integ2_ss(EC.fd, :α)
      D2a = calc_D2(EC, T1a, T2a, :α)[tripp,:,:]
      @mtensor rK2pqa[p,r,i,j] := int2a[p,r,x] * D2a[x,i,j]
      D2a = nothing
      int2a = nothing
      # symmetrize R
      @mtensor K2pqa[p,r,i,j] := rK2pqa[p,r,i,j] + rK2pqa[r,p,j,i]
      rK2pqa = nothing
      R2a += K2pqa[SP['v'],SP['v'],:,:]
      if n_occb_orbs(EC) > 0
        # ββ
        int2b = integ2_ss(EC.fd, :β)
        D2b = calc_D2(EC, T1b, T2b, :β)[tripp,:,:]
        @mtensor rK2pqb[p,r,i,j] := int2b[p,r,x] * D2b[x,i,j]
        D2b = nothing
        int2b = nothing
        # symmetrize R
        @mtensor K2pqb[p,r,i,j] := rK2pqb[p,r,i,j] + rK2pqb[r,p,j,i]
        rK2pqb = nothing
        R2b += K2pqb[SP['V'],SP['V'],:,:]
        # αβ
        int2ab = integ2_os(EC.fd)
        D2ab = calc_D2ab(EC, T1a, T1b, T2ab)
        @mtensor K2pqab[p,r,i,j] := int2ab[p,r,q,s] * D2ab[q,s,i,j]
        D2ab = nothing
        int2ab = nothing
        R2ab += K2pqab[SP['v'],SP['V'],:,:]
      end
    else
      int2 = integ2_ss(EC.fd)
      # αα
      D2a = calc_D2(EC, T1a, T2a, :α)[tripp,:,:]
      @mtensor rK2pqa[p,r,i,j] := int2[p,r,x] * D2a[x,i,j]
      D2a = nothing
      # symmetrize R
      @mtensor K2pqa[p,r,i,j] := rK2pqa[p,r,i,j] + rK2pqa[r,p,j,i]
      rK2pqa = nothing
      R2a += K2pqa[SP['v'],SP['v'],:,:]
      if n_occb_orbs(EC) > 0
        # ββ
        D2b = calc_D2(EC, T1b, T2b, :β)[tripp,:,:]
        @mtensor rK2pqb[p,r,i,j] := int2[p,r,x] * D2b[x,i,j]
        D2b = nothing
        # symmetrize R
        @mtensor K2pqb[p,r,i,j] := rK2pqb[p,r,i,j] + rK2pqb[r,p,j,i]
        rK2pqb = nothing
        R2b += K2pqb[SP['V'],SP['V'],:,:]
        # αβ
        D2ab_full = calc_D2ab(EC, T1a, T1b, T2ab, true)
        D2ab = D2ab_full[tripp,:,:] 
        D2abT = permutedims(D2ab_full,(2,1,4,3))[tripp,:,:]
        D2ab_full = nothing
        @mtensor K2pqab[p,r,i,j] := int2[p,r,x] * D2ab[x,i,j]
        @mtensor K2pqab[p,r,i,j] += int2[r,p,x] * D2abT[x,j,i]
        D2ab = nothing
        D2abT = nothing
        R2ab += K2pqab[SP['v'],SP['V'],:,:]
      end
    end
    if length(T1a) > 0
      @mtensor begin
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
      @mtensor begin
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
      @mtensor begin
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
    t1 = print_time(EC,t1,"kext",2)
  else
    d_vvvv = load4idx(EC,"d_vvvv")
    @mtensor R2a[a,b,i,j] += d_vvvv[a,b,c,d] * T2a[c,d,i,j]
    d_vvvv = nothing
    d_VVVV = load4idx(EC,"d_VVVV")
    @mtensor R2b[A,B,I,J] += d_VVVV[A,B,C,D] * T2b[C,D,I,J]
    d_VVVV = nothing
    d_vVvV = load4idx(EC,"d_vVvV")
    @mtensor R2ab[a,B,i,J] += d_vVvV[a,B,c,D] * T2ab[c,D,i,J]
    d_vVvV = nothing
    t1 = print_time(EC,t1,"``R_{ab}^{ij} += v_{ab}^{cd} T_{cd}^{ij}``",2)
  end

  @mtensor begin
    xij[i,j] := fij[i,j]
    xIJ[i,j] := fIJ[i,j]
    xab[i,j] := fab[i,j]
    xAB[i,j] := fAB[i,j]
  end
  x_klij = load4idx(EC,"d_oooo")
  x_KLIJ = load4idx(EC,"d_OOOO")
  x_kLiJ = load4idx(EC,"d_oOoO")
  if !linearized
    dcfac = dc ? 0.5 : 1.0
    oovv = ints2(EC,"oovv")
    @mtensor begin
      xij[i,j] += dcfac * (oovv[i,k,b,d] * T2a[b,d,j,k])
      xab[a,b] -= dcfac * (oovv[i,k,b,d] * T2a[a,d,i,k])
    end
    t1 = print_time(EC,t1,"``x_i^j and x_a^b``",2)
    !dc && @mtensor x_klij[k,l,i,j] += 0.5 * oovv[k,l,c,d] * T2a[c,d,i,j]
    if n_occb_orbs(EC) > 0
      @mtensor x_dAlI[d,A,l,I] := oovv[k,l,c,d] * T2ab[c,A,k,I]
      !dc && @mtensor x_dAlI[d,A,l,I] -= oovv[k,l,d,c] * T2ab[c,A,k,I]
      @mtensor begin
        rR2b[A,B,I,J] := x_dAlI[d,A,l,I] * T2ab[d,B,l,J]
        R2b[A,B,I,J] += rR2b[A,B,I,J] - rR2b[A,B,J,I]
      end
      x_dAlI, rR2b = nothing, nothing
      t1 = print_time(EC,t1,"``R_{AB}^{IJ} += x_{dA}^{lI} T_{dB}^{lJ}``",2)
    end
    @mtensor x_adil[a,d,i,l] := 0.5 * oovv[k,l,c,d] *  T2a[a,c,i,k]
    !dc && @mtensor x_adil[a,d,i,l] -= 0.5 * oovv[k,l,d,c] * T2a[a,c,i,k]
    @mtensor R2ab[a,B,i,J] += x_adil[a,d,i,l] * T2ab[d,B,l,J]
    oovv = nothing
    t1 = print_time(EC,t1,"``R_{aB}^{iJ} += x_{ad}^{il} T_{dB}^{lJ}``",2)
    if n_occb_orbs(EC) > 0
      OOVV = ints2(EC,"OOVV")
      @mtensor begin
        xIJ[I,J] += dcfac * (OOVV[I,K,B,D] * T2b[B,D,J,K])
        xAB[A,B] -= dcfac * (OOVV[I,K,B,D] * T2b[A,D,I,K])
      end
      t1 = print_time(EC,t1,"``x_I^J and x_A^B``",2)
      !dc && @mtensor x_KLIJ[K,L,I,J] += 0.5 * (OOVV[K,L,C,D] * T2b[C,D,I,J])
      @mtensor x_ADIL[A,D,I,L] := 0.5 * (OOVV[K,L,C,D] * T2b[A,C,I,K])
      !dc && @mtensor x_ADIL[A,D,I,L] -= 0.5 * (OOVV[K,L,D,C] * T2b[A,C,I,K])
      @mtensor R2ab[b,A,j,I] += 2.0 * x_ADIL[A,D,I,L] * T2ab[b,D,j,L]
      t1 = print_time(EC,t1,"``R_{bA}^{jI} += 2 x_{AD}^{IL} T_{bD}^{jL}``",2)
      @mtensor x_vVoO[a,D,i,L] := OOVV[K,L,C,D] * T2ab[a,C,i,K]
      !dc && @mtensor x_vVoO[a,D,i,L] -= OOVV[K,L,D,C] * T2ab[a,C,i,K]
      @mtensor begin      
        rR2a[a,b,i,j] := x_vVoO[a,D,i,L] * T2ab[b,D,j,L]
        R2a[a,b,i,j] += rR2a[a,b,i,j] - rR2a[a,b,j,i] 
      end
      OOVV, x_vVoO, rR2a = nothing, nothing, nothing
      t1 = print_time(EC,t1,"``R_{ab}^{ij} += x_{aL}^{Di} T_{bD}^{jL}``",2)
      oOvV = ints2(EC,"oOvV")
      @mtensor begin
        xij[i,j] += dcfac * (oOvV[i,K,b,D] * T2ab[b,D,j,K])
        xab[a,b] -= dcfac * (oOvV[i,K,b,D] * T2ab[a,D,i,K])
        xIJ[I,J] += dcfac * (oOvV[k,I,d,B] * T2ab[d,B,k,J])
        xAB[A,B] -= dcfac * (oOvV[k,I,d,B] * T2ab[d,A,k,I])
      end
      t1 = print_time(EC,t1,"``opposite spin for x_i^j, x_a^b, x_I^J, x_A^B``",2)
      !dc && @mtensor x_kLiJ[k,L,i,J] += oOvV[k,L,c,D] * T2ab[c,D,i,J]
      @mtensor begin
        x_adil[a,d,i,l] += oOvV[l,K,d,C] * T2ab[a,C,i,K]
        R2ab[a,B,i,J] += x_adil[a,d,i,l] * T2ab[d,B,l,J]
        rR2a[a,b,i,j] := x_adil[a,d,i,l] *  T2a[b,d,j,l]
        R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i] - rR2a[a,b,j,i] - rR2a[b,a,i,j]
      end
      x_adil, rR2a = nothing, nothing
      t1 = print_time(EC,t1,"``R_{ab}^{ij} += x_{al}^{id} T_{db}^{lj}``",2)
      @mtensor begin
        x_ADIL[A,D,I,L] += oOvV[k,L,c,D] * T2ab[c,A,k,I]
        rR2b[A,B,I,J] := x_ADIL[A,D,I,L] * T2b[B,D,J,L]
        R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I] - rR2b[A,B,J,I] - rR2b[B,A,I,J]
      end 
      X_ADIL, rR2b = nothing, nothing
      t1 = print_time(EC,t1,"``R_{AB}^{IJ} += x_{AL}^{ID} T_{BD}^{JL}``",2)
      @mtensor begin
        x_vVoO[a,D,i,L] := oOvV[k,L,c,D] * T2a[a,c,i,k]
        R2ab[a,B,i,J] += x_vVoO[a,D,i,L] * T2b[B,D,J,L]
      end
      x_vVoO = nothing
      t1 = print_time(EC,t1,"``R_{aB}^{iJ} += x_{aL}^{iD} T_{BD}^{JL}``",2)
      if !dc
        @mtensor begin
          x_DBik[D,B,i,k] := oOvV[k,L,c,D] * T2ab[c,B,i,L]
          R2ab[a,B,i,J] += x_DBik[D,B,i,k] * T2ab[a,D,k,J]
        end
        x_DBik = nothing
        t1 = print_time(EC,t1,"``R_{aB}^{iJ} += x_{DB}^{ik} T_{aD}^{kJ}``",2)
      end
      oOvV = nothing
    end
  end

  @mtensor R2a[a,b,i,j] += x_klij[k,l,i,j] *  T2a[a,b,k,l]
  if n_occb_orbs(EC) > 0
    @mtensor begin
      R2b[A,B,I,J] += x_KLIJ[K,L,I,J] *  T2b[A,B,K,L]
      R2ab[a,B,i,J] += x_kLiJ[k,L,i,J] * T2ab[a,B,k,L]
    end
  end
  x_klij, x_KLIJ, x_kLiJ = nothing, nothing, nothing
  t1 = print_time(EC,t1,"``R_{ab}^{ij} += x_{kl}^{ij} T_{ab}^{kl}``",2)

  @mtensor begin
    rR2a[a,b,i,j] := xab[a,c] * T2a[c,b,i,j]
    rR2a[a,b,i,j] -= xij[k,i] * T2a[a,b,k,j]
    R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i]
  end
  rR2a = nothing
  if n_occb_orbs(EC) > 0
    @mtensor begin
      rR2b[A,B,I,J] := xAB[A,C] * T2b[C,B,I,J]
      rR2b[A,B,I,J] -= xIJ[K,I] * T2b[A,B,K,J]
      R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I]
    end
    rR2b = nothing
    @mtensor begin
      R2ab[a,B,i,J] -= xij[k,i] * T2ab[a,B,k,J]
      R2ab[a,B,i,J] -= xIJ[K,J] * T2ab[a,B,i,K]
      R2ab[a,B,i,J] += xab[a,c] * T2ab[c,B,i,J]
      R2ab[a,B,i,J] += xAB[B,C] * T2ab[a,C,i,J]
    end
  end
  xij, xIJ, xab, xAB = nothing, nothing, nothing, nothing
  t1 = print_time(EC,t1,"``R_{ab}^{ij} += x_a^c T_{cb}^{ij} - x_k^i T_{ab}^{kj}``",2)
  #ph-ab-ladder
  if n_occb_orbs(EC) > 0
    d_vOvO = load4idx(EC,"d_vOvO")
    @mtensor R2ab[a,B,i,J] -= d_vOvO[a,K,c,J] * T2ab[c,B,i,K]
    d_vOvO = nothing
    d_oVoV = load4idx(EC,"d_oVoV")
    @mtensor R2ab[a,B,i,J] -= d_oVoV[k,B,i,C] * T2ab[a,C,k,J]
    d_oVoV = nothing
    t1 = print_time(EC,t1,"``R_{aB}^{iJ} -= v_{aK}^{cJ} T_{cB}^{iK}``",2)
  end

  #ring terms
  A_d_voov = load4idx(EC,"d_voov") - permutedims(load4idx(EC,"d_vovo"),(1,2,4,3))
  @mtensor begin
    rR2a[a,b,i,j] := A_d_voov[b,k,j,c] * T2a[a,c,i,k]
    R2ab[a,B,i,J] += A_d_voov[a,k,i,c] * T2ab[c,B,k,J]
  end
  A_d_voov = nothing
  t1 = print_time(EC,t1,"``R_{ab}^{ij} += \\bar v_{bk}^{jc} T_{ac}^{ik}``",2)
  d_vOoV = load4idx(EC,"d_vOoV")
  @mtensor begin
    rR2a[a,b,i,j] += d_vOoV[b,K,j,C] * T2ab[a,C,i,K]
    R2ab[a,B,i,J] += d_vOoV[a,K,i,C] * T2b[B,C,J,K]
  end
  d_vOoV = nothing
  t1 = print_time(EC,t1,"``R_{ab}^{ij} += v_{bK}^{jC} T_{aC}^{iK}``",2)
  @mtensor R2a[a,b,i,j] += rR2a[a,b,i,j] + rR2a[b,a,j,i] - rR2a[a,b,j,i] - rR2a[b,a,i,j]
  rR2a = nothing
  if n_occb_orbs(EC) > 0
    A_d_VOOV = load4idx(EC,"d_VOOV") - permutedims(load4idx(EC,"d_VOVO"),(1,2,4,3))
    @mtensor begin
      rR2b[A,B,I,J] := A_d_VOOV[B,K,J,C] * T2b[A,C,I,K]
      R2ab[a,B,i,J] += A_d_VOOV[B,K,J,C] * T2ab[a,C,i,K]
    end
    A_d_VOOV = nothing
    t1 = print_time(EC,t1,"``R_{AB}^{IJ} += \\bar v_{BK}^{JC} T_{AC}^{IK}``",2)
    d_oVvO = load4idx(EC,"d_oVvO")
    @mtensor begin
      rR2b[A,B,I,J] += d_oVvO[k,B,c,J] * T2ab[c,A,k,I]
      R2ab[a,B,i,J] += d_oVvO[k,B,c,J] * T2a[a,c,i,k]
    end
    d_oVvO = nothing
    t1 = print_time(EC,t1,"``R_{AB}^{IJ} += v_{kB}^{cJ} T_{cA}^{kI}``",2)
    @mtensor R2b[A,B,I,J] += rR2b[A,B,I,J] + rR2b[B,A,J,I] - rR2b[A,B,J,I] - rR2b[B,A,I,J]
    rR2b = nothing
  end

  if tworef || fixref
    # 2D-CC assumes open-shell singlet reference torba and uorbb occupied in Φ^A and torbb and uorba in Φ^B.
    @assert length(setdiff(SP['o'],SP['O'])) == 1 && length(setdiff(SP['O'],SP['o'])) == 1 "2D-CCSD needs two open-shell alpha beta orbitals"
    activeorbs = oss_active_orbitals(EC)
    if tworef
      occcorea = collect(1:length(SP['o']))
      occcoreb = collect(1:length(SP['O']))
      filter!(x -> x != activeorbs.ta, occcorea)
      filter!(x -> x != activeorbs.ub, occcoreb)
      occcore = (occcorea, occcoreb)
      virtualsa = collect(1:length(SP['v']))
      virtualsb = collect(1:length(SP['V']))
      filter!(x -> x != activeorbs.ua, virtualsa)
      filter!(x -> x != activeorbs.tb, virtualsb)
      virtuals = (virtualsa, virtualsb)
      W = R2ab[activeorbs.ua,activeorbs.tb,activeorbs.ta,activeorbs.ub]
      R2ab[activeorbs.ua,activeorbs.tb,activeorbs.ta,activeorbs.ub] = 0.0
      if length(T1a) > 0
        M1a = calc_M1a(occcore, virtuals, T1a, T1b, T2b, T2ab, activeorbs)
        M1b = calc_M1b(occcore, virtuals, T1a, T1b, T2a, T2ab, activeorbs)
        @mtensor R1a[a,i] += M1a[a,i] * W
        @mtensor R1b[a,i] += M1b[a,i] * W
      end
      if !isempty(occcorea) && !isempty(occcoreb)
        M2a = calc_M2a(occcore, virtuals, T1a, T1b, T2b, T2ab, activeorbs)
        M2b = calc_M2b(occcore, virtuals, T1a, T1b, T2a, T2ab, activeorbs)
        @mtensor R2a[a,b,i,j] += M2a[a,b,i,j] * W
        @mtensor R2b[a,b,i,j] += M2b[a,b,i,j] * W
      end
      M2ab = calc_M2ab(occcore, virtuals, T1a, T1b, T2a, T2b, T2ab, activeorbs)
      @mtensor R2ab[a,b,i,j] += M2ab[a,b,i,j] * W
      save!(EC,"2d_ccsd_W",[W])
      t1 = print_time(EC,t1,"``2D-CCSD additional terms``",2)
    elseif fixref
      R2ab[activeorbs.ua,activeorbs.tb,activeorbs.ta,activeorbs.ub] = 0
    end
  end
  return R1a, R1b, R2a, R2b, R2ab
end

"""
    calc_cc_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab, T3aaa, T3bbb, T3abb, T3aab; dc=false)

  Calculate UCCSDT or UDC-CCSDT residual.
"""
function calc_cc_resid(EC::ECInfo, T1a, T1b, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb; dc=false, tworef=false, fixref=false)
  EC.options.cc.use_kext = false
  EC.options.cc.calc_d_vvvv = true
  EC.options.cc.calc_d_vvvo = true
  EC.options.cc.calc_d_vovv = true
  EC.options.cc.calc_d_vvoo = true

  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nvirtb = n_virtb_orbs(EC)
  if length(T1a) > 0 || length(T1b) > 0
    calc_dressed_ints(EC, T1a, T1b)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC,true)
  end

  R1a = zero(T1a)
  R1b = zero(T1b)

  dfock = load2idx(EC,"df_mm")
  dfockb = load2idx(EC,"df_MM")

  fij = dfock[SP['o'],SP['o']]
  fab = dfock[SP['v'],SP['v']]
  fIJ = dfockb[SP['O'],SP['O']]
  fAB = dfockb[SP['V'],SP['V']]
  fai = dfock[SP['v'],SP['o']]
  fAI = dfockb[SP['V'],SP['O']]
  fia = dfock[SP['o'],SP['v']]
  fIA = dfockb[SP['O'],SP['V']]

  if length(T1a) > 0
    ccsdt_singles!(EC, R1a, R1b, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
    t1 = print_time(EC,t1,"ccsdt singles",2)
  end

  R2a = zeros(nvirt, nvirt, nocc, nocc)
  R2b = zeros(nvirtb, nvirtb, noccb, noccb)
  R2ab = zeros(nvirt, nvirtb, nocc, noccb)
  ccsdt_doubles!(EC, R2a, R2b, R2ab, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
  t1 = print_time(EC,t1,"ccsdt doubles",2)

  R3a = zeros(nvirt, nvirt, nvirt, nocc, nocc, nocc)
  R3b = zeros(nvirtb, nvirtb, nvirtb, noccb, noccb, noccb)
  R3abb = zeros(nvirt, nvirtb, nvirtb, nocc, noccb, noccb)
  R3aab = zeros(nvirt, nvirt, nvirtb, nocc, nocc, noccb)
  if dc
    dcccsdt_triples!(EC, R3a, R3b, R3aab, R3abb, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
  else
    ccsdt_triples!(EC, R3a, R3b, R3aab, R3abb, T2a, T2b, T2ab, T3a, T3b, T3aab, T3abb, fij, fab, fIJ, fAB, fai, fAI, fia, fIA)
  end
  t1 = print_time(EC,t1,"ccsdt triples",2)

  return R1a, R1b, R2a, R2b, R2ab, R3a, R3b, R3aab, R3abb
end

"""
    calc_cc_resid(EC::ECInfo, T1, T2, T3; dc=false, tworef=false, fixref=false)

  Calculate CCSDT or DC-CCSDT residual.
"""
function calc_cc_resid(EC::ECInfo, T1, T2, T3; dc=false, tworef=false, fixref=false)
  EC.options.cc.use_kext = false
  EC.options.cc.calc_d_vvvv = true
  EC.options.cc.calc_d_vvvo = true
  EC.options.cc.calc_d_vovv = true
  EC.options.cc.calc_d_vvoo = true

  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  if length(T1) > 0
    calc_dressed_ints(EC, T1)
    t1 = print_time(EC,t1,"dressing",2)
  else
    pseudo_dressed_ints(EC,true)
  end

  R1 = zeros(nvirt,nocc)

  dfock = load2idx(EC,"df_mm")

  fij = dfock[SP['o'],SP['o']]
  fab = dfock[SP['v'],SP['v']]
  fai = dfock[SP['v'],SP['o']]
  fia = dfock[SP['o'],SP['v']]

  if length(T1) > 0
    ccsdt_singles!(EC, R1, T2, T3, fij, fab, fai, fia)
    t1 = print_time(EC,t1,"ccsdt singles",2)
  end

  R2 = zeros(nvirt, nvirt, nocc, nocc)
  ccsdt_doubles!(EC, R2, T2, T3, fij, fab, fai, fia)
  t1 = print_time(EC,t1,"ccsdt doubles",2)

  R3 = zeros(nvirt, nvirt, nvirt, nocc, nocc, nocc)
  if dc
    dcccsdt_triples!(EC, R3, T2, T3, fij, fab, fai, fia)
  else
    ccsdt_triples!(EC, R3, T2, T3, fij, fab, fai, fia)
  end
  t1 = print_time(EC,t1,"ccsdt triples",2)

  return R1, R2, R3
end

"""
    oss_active_orbitals(EC::ECInfo)

  Return the four active orbitals of an (2e,2o) open-shell singlet problem based on a single determinant reference.
"""
function oss_active_orbitals(EC::ECInfo)
  SP = EC.space
  @assert length(setdiff(SP['o'],SP['O'])) == 1 && length(setdiff(SP['O'],SP['o'])) == 1 "Assumed two open-shell alpha beta orbitals here."
  torb = setdiff(SP['o'],SP['O'])[1]
  uorb = setdiff(SP['O'],SP['o'])[1]
  torba = findfirst(isequal(torb),SP['o'])
  @assert !isnothing(torba) "Active orbital not found in alpha orbitals."
  uorbb = findfirst(isequal(uorb),SP['O'])
  @assert !isnothing(uorbb) "Active orbital not found in beta orbitals."
  torbb = findfirst(isequal(torb),SP['V'])
  @assert !isnothing(torbb) "Active orbital not found in beta virtuals."
  uorba = findfirst(isequal(uorb),SP['v'])
  @assert !isnothing(uorba) "Active orbital not found in alpha virtuals."
  return (ta=torba,ub=uorbb,tb=torbb,ua=uorba)
end

function calc_M1a(occcore, virtuals, T1a, T1b, T2b, T2ab, activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  internalT1a = T1a[norba,morba]
  M1 = zeros(Float64,size(T1a))
  T2_nVmN = T2ab[norba,virtualsb,morba,norbb]
  T1_VN = T1b[virtualsb,norbb]
  if !isempty(occcorea) && !isempty(occcoreb)
    T2_nMmO = T2ab[norba,morbb,morba,occcoreb]
    T1_MO = T1b[morbb,occcoreb]
    @mtensor M1_no[i] := T2_nMmO[i]
    @mtensor M1_no[i] += internalT1a * T1_MO[i]
    @view(M1[norba,occcorea]) .+= M1_no 
    T2_MVON = T2b[morbb,virtualsb,occcoreb,norbb]
    @mtensor M1_vo[a,i] := T2_nMmO[i] * T1_VN[a]
    @mtensor M1_vo[a,i] += internalT1a * T2_MVON[a,i]
    @mtensor M1_vo[a,i] += T2_nVmN[a] * T1_MO[i]
    @mtensor M1_vo[a,i] += internalT1a * T1_VN[a] * T1_MO[i]
    @view(M1[virtualsa,occcorea]) .+= M1_vo
  end
  @mtensor M1_vm[a] := T2_nVmN[a]
  @mtensor M1_vm[a] += internalT1a * T1_VN[a]
  @view(M1[virtualsa,morba]) .-= M1_vm
  M1[norba,morba] += internalT1a
  return M1
end

function calc_M1b(occcore, virtuals, T1a, T1b, T2a, T2ab, activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M1 = zeros(Float64,size(T1b))
  internalT1b = T1b[morbb,norbb]
  T2_vMmN = T2ab[virtualsa,morbb,morba,norbb]
  T1_vm = T1a[virtualsa,morba]
  if !isempty(occcorea) && !isempty(occcoreb)
    T2_nMoN = T2ab[norba,morbb,occcorea,norbb]
    T1_no = T1a[norba,occcorea]
    @mtensor M1_MO[i] := T2_nMoN[i]
    @mtensor M1_MO[i] += internalT1b * T1_no[i]
    @view(M1[morbb,occcoreb]) .+= M1_MO
    T2_nvom = T2a[norba,virtualsa,occcorea,morba]
    @mtensor M1_VO[a,i] := T2_nMoN[i] * T1_vm[a]
    @mtensor M1_VO[a,i] += internalT1b * T2_nvom[a,i]
    @mtensor M1_VO[a,i] += T2_vMmN[a] * T1_no[i]
    @mtensor M1_VO[a,i] += internalT1b * T1_vm[a] * T1_no[i]
    @view(M1[virtualsb,occcoreb]) .+= M1_VO
  end
  @mtensor M1_VN[a] := T2_vMmN[a]
  @mtensor M1_VN[a] += internalT1b * T1_vm[a]
  @view(M1[virtualsb,norbb]) .-= M1_VN
  M1[morbb,norbb] += internalT1b
  return M1
end

function calc_M2a(occcore,virtuals,T1a,T1b,T2b,T2ab,activeorbs)
  morba, norbb, morbb, norba = activeorbs
  occcorea, occcoreb = occcore
  virtualsa, virtualsb = virtuals
  M2 = zeros(Float64,size(T2b))
  T2_nMmO = T2ab[norba,morbb,morba,occcoreb]
  T2_nVmO = T2ab[norba,virtualsb,morba,occcoreb]
  T2_MVOO = T2b[morbb,virtualsb,occcoreb,occcoreb]
  T2_nVmN = T2ab[norba,virtualsb,morba,norbb]
  T2_VVNO = T2b[virtualsb,virtualsb,norbb,occcoreb]
  @mtensor M2_vvoo[a,b,j,i] := T2_nMmO[j] * T2_VVNO[a,b,i]
  @mtensor M2_vvoo[a,b,i,j] -= T2_nMmO[j] * T2_VVNO[a,b,i]
  @mtensor M2_vvoo[b,a,i,j] += T2_nVmN[b] * T2_MVOO[a,i,j]
  @mtensor M2_vvoo[a,b,i,j] -= T2_nVmN[b] * T2_MVOO[a,i,j]
  @mtensor M2_vnom[a,i] := -T2_nVmO[a,i]
  @mtensor M2_vnmo[a,i] := T2_nVmO[a,i]
  @mtensor M2_nvom[a,i] := T2_nVmO[a,i]
  @mtensor M2_nvmo[a,i] := -T2_nVmO[a,i]
  if length(T1a) > 0
    internalT1a = T1a[norba,morba]
    internalT1b = T1b[morbb,norbb]
    T1_vo = T1a[virtualsa,occcorea]
    T1_VO = T1b[virtualsb,occcoreb]
    @mtensor TT1a[a,i] := T1_vo[a,i] - T1_VO[a,i]
    @mtensor TT1b[a,i] := T1_VO[a,i] - T1_vo[a,i]
    @mtensor M2_nvoo[a,j,i] := T2_nMmO[i] * TT1a[a,j]
    @mtensor M2_nvoo[a,i,j] -= T2_nMmO[i] * TT1a[a,j]
    @mtensor M2_vnoo[a,i,j] := T2_nMmO[i] * TT1a[a,j]
    @mtensor M2_vnoo[a,j,i] -= T2_nMmO[i] * TT1a[a,j]
    T1_MO = T1b[morbb,occcoreb]
    @mtensor M2_nvoo[a,i,j] -= T2_nVmO[a,i] * T1_MO[j]
    @mtensor M2_nvoo[a,j,i] += T2_nVmO[a,i] * T1_MO[j]
    @mtensor M2_vnoo[a,i,j] += T2_nVmO[a,i] * T1_MO[j]
    @mtensor M2_vnoo[a,j,i] -= T2_nVmO[a,i] * T1_MO[j]
    @mtensor M2_nvoo[a,i,j] -= internalT1a * T2_MVOO[a,i,j]
    @mtensor M2_vnoo[a,i,j] += internalT1a * T2_MVOO[a,i,j]
    @view(M2[norba,virtualsa,occcorea,occcorea]) .+= M2_nvoo
    @view(M2[virtualsa,norba,occcorea,occcorea]) .+= M2_vnoo

    @mtensor M2_vvmo[a,b,i] := T2_nVmN[a] * TT1a[b,i]
    @mtensor M2_vvmo[b,a,i] -= T2_nVmN[a] * TT1a[b,i]
    @mtensor M2_vvom[b,a,i] := T2_nVmN[a] * TT1a[b,i]
    @mtensor M2_vvom[a,b,i] -= T2_nVmN[a] * TT1a[b,i]
    T1_VN = T1b[virtualsb,norbb]
    @mtensor M2_vvmo[a,b,i] += T2_nVmO[a,i] * T1_VN[b] 
    @mtensor M2_vvmo[b,a,i] -= T2_nVmO[a,i] * T1_VN[b]
    @mtensor M2_vvom[a,b,i] -= T2_nVmO[a,i] * T1_VN[b] 
    @mtensor M2_vvom[b,a,i] += T2_nVmO[a,i] * T1_VN[b] 
    @mtensor M2_vvmo[a,b,i] += internalT1a * T2_VVNO[b,a,i]
    @mtensor M2_vvom[a,b,i] -= internalT1a * T2_VVNO[b,a,i]
    @view(M2[virtualsa,virtualsa,morba,occcorea]) .+= M2_vvmo
    @view(M2[virtualsa,virtualsa,occcorea,morba]) .+= M2_vvom

    @mtensor M2_vvoo[a,b,i,j] -= (T2_nMmO[i] * T1_VN[a]) * TT1a[b,j]
    @mtensor M2_vvoo[b,a,i,j] += (T2_nMmO[i] * T1_VN[a]) * TT1a[b,j]
    @mtensor M2_vvoo[a,b,j,i] += (T2_nMmO[i] * T1_VN[a]) * TT1a[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= (T2_nMmO[i] * T1_VN[a]) * TT1a[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= (T2_nVmN[a] * T1_MO[i]) * TT1a[b,j]
    @mtensor M2_vvoo[b,a,i,j] += (T2_nVmN[a] * T1_MO[i]) * TT1a[b,j]
    @mtensor M2_vvoo[a,b,j,i] += (T2_nVmN[a] * T1_MO[i]) * TT1a[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= (T2_nVmN[a] * T1_MO[i]) * TT1a[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= (T1_VN[a] * T1_MO[j]) * T2_nVmO[b,i]
    @mtensor M2_vvoo[b,a,i,j] += (T1_VN[a] * T1_MO[j]) * T2_nVmO[b,i]
    @mtensor M2_vvoo[a,b,j,i] += (T1_VN[a] * T1_MO[j]) * T2_nVmO[b,i]
    @mtensor M2_vvoo[b,a,j,i] -= (T1_VN[a] * T1_MO[j]) * T2_nVmO[b,i]
    T2_MVNO = T2b[morbb,virtualsb,norbb,occcoreb]
    @mtensor M2_vvoo[a,b,i,j] -= T2_MVNO[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[b,a,i,j] += T2_MVNO[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[a,b,j,i] += T2_MVNO[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= T2_MVNO[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1b * TT1b[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[b,a,i,j] += internalT1b * TT1b[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[a,b,j,i] += internalT1b * TT1b[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= internalT1b * TT1b[a,i] * T2_nVmO[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1a * TT1b[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[b,a,i,j] += internalT1a * TT1b[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[a,b,j,i] += internalT1a * TT1b[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= internalT1a * TT1b[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1a * T1_MO[j] * T2_VVNO[a,b,i]
    @mtensor M2_vvoo[a,b,j,i] += internalT1a * T1_MO[j] * T2_VVNO[a,b,i]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1a * T1_VN[b] * T2_MVOO[a,i,j]
    @mtensor M2_vvoo[b,a,i,j] += internalT1a * T1_VN[b] * T2_MVOO[a,i,j]
    @mtensor M2_vnom[a,i] -= internalT1a * TT1b[a,i]
    @mtensor M2_nvom[a,i] += internalT1a * TT1b[a,i]
    @mtensor M2_vnmo[a,i] += internalT1a * TT1b[a,i]
    @mtensor M2_nvmo[a,i] -= internalT1a * TT1b[a,i]
  end
  @view(M2[virtualsa,virtualsa,occcorea,occcorea]) .+= M2_vvoo
  @view(M2[virtualsa,norba,occcorea,morba]) .+= M2_vnom
  @view(M2[virtualsa,norba,morba,occcorea]) .+= M2_vnmo
  @view(M2[norba,virtualsa,occcorea,morba]) .+= M2_nvom
  @view(M2[norba,virtualsa,morba,occcorea]) .+= M2_nvmo
  return M2
end

function calc_M2b(occcore,virtuals,T1a,T1b,T2a,T2ab,activeorbs)
#NOTE that we intentionally unpack in different order to reuse code from calc_M2a as much as possible
# morba, norbb, morbb, norba = activeorbs
  norbb, morba, norba, morbb = activeorbs
  occcoreb, occcorea = occcore
  virtualsb, virtualsa = virtuals
#ENDNOTE
  M2 = zeros(Float64,size(T2a))
  T2_VVNO = T2a[virtualsb,virtualsb,norbb,occcoreb]
  T2_VnNm = T2ab[virtualsb,norba,norbb,morba]
  T2_VnOm = T2ab[virtualsb,norba,occcoreb,morba]
  T2_MnOm = T2ab[morbb,norba,occcoreb,morba]
  T2_MVOO = T2a[morbb,virtualsb,occcoreb,occcoreb]
  @mtensor M2_vvoo[a,b,j,i] := T2_MnOm[j] * T2_VVNO[a,b,i]
  @mtensor M2_vvoo[a,b,i,j] -= T2_MnOm[j] * T2_VVNO[a,b,i]
  @mtensor M2_vvoo[b,a,i,j] += T2_VnNm[b] * T2_MVOO[a,i,j]
  @mtensor M2_vvoo[a,b,i,j] -= T2_VnNm[b] * T2_MVOO[a,i,j]
  @mtensor M2_vnom[a,i] := -T2_VnOm[a,i]
  @mtensor M2_vnmo[a,i] := T2_VnOm[a,i]
  @mtensor M2_nvom[a,i] := T2_VnOm[a,i]
  @mtensor M2_nvmo[a,i] := -T2_VnOm[a,i]
  if length(T1a) > 0
    internalT1a = T1a[morbb,norbb]
    internalT1b = T1b[norba,morba]
    T1a_ai = T1a[virtualsb,occcoreb]
    T1b_ai = T1b[virtualsa,occcorea]
    @mtensor TT1a[a,i] := T1a_ai[a,i] - T1b_ai[a,i]
    @mtensor TT1b[a,i] := T1b_ai[a,i] - T1a_ai[a,i]
    @mtensor M2_nvoo[a,j,i] := T2_MnOm[i] * TT1b[a,j]
    @mtensor M2_nvoo[a,i,j] -= T2_MnOm[i] * TT1b[a,j]
    @mtensor M2_vnoo[a,i,j] := T2_MnOm[i] * TT1b[a,j]
    @mtensor M2_vnoo[a,j,i] -= T2_MnOm[i] * TT1b[a,j]
    T1_MO = T1a[morbb,occcoreb]
    @mtensor M2_nvoo[a,i,j] -= T2_VnOm[a,i] * T1_MO[j]
    @mtensor M2_nvoo[a,j,i] += T2_VnOm[a,i] * T1_MO[j]
    @mtensor M2_vnoo[a,i,j] += T2_VnOm[a,i] * T1_MO[j]
    @mtensor M2_vnoo[a,j,i] -= T2_VnOm[a,i] * T1_MO[j]
    @mtensor M2_nvoo[a,i,j] -= internalT1b * T2_MVOO[a,i,j]
    @mtensor M2_vnoo[a,i,j] += internalT1b * T2_MVOO[a,i,j]
    @view(M2[norba,virtualsa,occcorea,occcorea]) .+= M2_nvoo
    @view(M2[virtualsa,norba,occcorea,occcorea]) .+= M2_vnoo

    @mtensor M2_vvmo[a,b,i] := T2_VnNm[a] * TT1b[b,i]
    @mtensor M2_vvmo[b,a,i] -= T2_VnNm[a] * TT1b[b,i]
    @mtensor M2_vvom[b,a,i] := T2_VnNm[a] * TT1b[b,i]
    @mtensor M2_vvom[a,b,i] -= T2_VnNm[a] * TT1b[b,i]
    T1_VN = T1a[virtualsb,norbb]
    @mtensor M2_vvmo[a,b,i] += T2_VnOm[a,i] * T1_VN[b] 
    @mtensor M2_vvmo[b,a,i] -= T2_VnOm[a,i] * T1_VN[b]
    @mtensor M2_vvom[a,b,i] -= T2_VnOm[a,i] * T1_VN[b] 
    @mtensor M2_vvom[b,a,i] += T2_VnOm[a,i] * T1_VN[b] 
    @mtensor M2_vvmo[a,b,i] += internalT1b * T2_VVNO[b,a,i]
    @mtensor M2_vvom[a,b,i] -= internalT1b * T2_VVNO[b,a,i]
    @view(M2[virtualsa,virtualsa,morba,occcorea]) .+= M2_vvmo
    @view(M2[virtualsa,virtualsa,occcorea,morba]) .+= M2_vvom
   
    @mtensor M2_vvoo[a,b,i,j] -= (T2_MnOm[i] * T1_VN[a]) * TT1b[b,j]
    @mtensor M2_vvoo[b,a,i,j] += (T2_MnOm[i] * T1_VN[a]) * TT1b[b,j]
    @mtensor M2_vvoo[a,b,j,i] += (T2_MnOm[i] * T1_VN[a]) * TT1b[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= (T2_MnOm[i] * T1_VN[a]) * TT1b[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= (T2_VnNm[a] * T1_MO[i]) * TT1b[b,j]
    @mtensor M2_vvoo[b,a,i,j] += (T2_VnNm[a] * T1_MO[i]) * TT1b[b,j]
    @mtensor M2_vvoo[a,b,j,i] += (T2_VnNm[a] * T1_MO[i]) * TT1b[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= (T2_VnNm[a] * T1_MO[i]) * TT1b[b,j]
    
    @mtensor M2_vvoo[a,b,i,j] -= (T1_VN[a] * T1_MO[j]) * T2_VnOm[b,i]
    @mtensor M2_vvoo[b,a,i,j] += (T1_VN[a] * T1_MO[j]) * T2_VnOm[b,i]
    @mtensor M2_vvoo[a,b,j,i] += (T1_VN[a] * T1_MO[j]) * T2_VnOm[b,i]
    @mtensor M2_vvoo[b,a,j,i] -= (T1_VN[a] * T1_MO[j]) * T2_VnOm[b,i]
    T2_MVNO = T2a[morbb,virtualsb,norbb,occcoreb]
    @mtensor M2_vvoo[a,b,i,j] -= T2_VnOm[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[b,a,i,j] += T2_VnOm[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[a,b,j,i] += T2_VnOm[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= T2_VnOm[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1a * TT1a[a,i] * T2_VnOm[b,j]
    @mtensor M2_vvoo[b,a,i,j] += internalT1a * TT1a[a,i] * T2_VnOm[b,j]
    @mtensor M2_vvoo[a,b,j,i] += internalT1a * TT1a[a,i] * T2_VnOm[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= internalT1a * TT1a[a,i] * T2_VnOm[b,j]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1b * TT1a[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[b,a,i,j] += internalT1b * TT1a[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[a,b,j,i] += internalT1b * TT1a[a,i] * T2_MVNO[b,j]
    @mtensor M2_vvoo[b,a,j,i] -= internalT1b * TT1a[a,i] * T2_MVNO[b,j]
    
    @mtensor M2_vvoo[a,b,i,j] -= internalT1b * T1_MO[j] * T2_VVNO[a,b,i]
    @mtensor M2_vvoo[a,b,j,i] += internalT1b * T1_MO[j] * T2_VVNO[a,b,i]
    @mtensor M2_vvoo[a,b,i,j] -= internalT1b * T1_VN[b] * T2_MVOO[a,i,j]
    @mtensor M2_vvoo[b,a,i,j] += internalT1b * T1_VN[b] * T2_MVOO[a,i,j]

    @mtensor M2_vnom[a,i] -= internalT1b * TT1a[a,i]
    @mtensor M2_nvom[a,i] += internalT1b * TT1a[a,i]
    @mtensor M2_vnmo[a,i] += internalT1b * TT1a[a,i]
    @mtensor M2_nvmo[a,i] -= internalT1b * TT1a[a,i]
  end
  @view(M2[virtualsa,virtualsa,occcorea,occcorea]) .+= M2_vvoo
  @view(M2[virtualsa,norba,occcorea,morba]) .+= M2_vnom
  @view(M2[virtualsa,norba,morba,occcorea]) .+= M2_vnmo
  @view(M2[norba,virtualsa,occcorea,morba]) .+= M2_nvom
  @view(M2[norba,virtualsa,morba,occcorea]) .+= M2_nvmo
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
  T2_nMmO = T2ab[norba,morbb,morba,occcoreb]
  T2_nVmN = T2ab[norba,virtualsb,morba,norbb]
  T2_MVNO = T2b[morbb,virtualsb,norbb,occcoreb]
  T2_nMoN = T2ab[norba,morbb,occcorea,norbb]
  T2_vMmO = T2ab[virtualsa,morbb,morba,occcoreb]
  T2_nVoN = T2ab[norba,virtualsb,occcorea,norbb]
  T2_vMoO = T2ab[virtualsa,morbb,occcorea,occcoreb]
  T2_nVoO = T2ab[norba,virtualsb,occcorea,occcoreb]
  T2_vMmN = T2ab[virtualsa,morbb,morba,norbb]
  T2_vVoN = T2ab[virtualsa,virtualsb,occcorea,norbb]
  T2_vVmO = T2ab[virtualsa,virtualsb,morba,occcoreb]
  T2_nvmo = T2a[norba,virtualsa,morba,occcorea]
  if !isempty(occcorea) && !isempty(occcoreb)
    T2_nVmO = T2ab[norba,virtualsb,morba,occcoreb]
    @mtensor M2_vVoO[a,b,i,j] := T2_nMoN[j] * T2_vVmO[b,a,i]
    @mtensor M2_vVoO[a,b,i,j] += T2_nMmO[i] * T2_vVoN[b,a,j]
    @mtensor M2_vVoO[a,b,i,j] += T2_vMmN[b] * T2_nVoO[a,j,i]
    @mtensor M2_vVoO[a,b,i,j] += T2_nVmN[a] * T2_vMoO[b,j,i]
    @mtensor M2_vVoO[a,b,i,j] -= T2_MVNO[a,i] * T2_nvmo[b,j]
    T2_vMoN = T2ab[virtualsa,morbb,occcorea,norbb]
    @mtensor M2_vVoO[a,b,i,j] -= T2_nVmO[a,i] * T2_vMoN[b,j]
    @mtensor M2_vVoO[a,b,i,j] -= T2_nVoN[a,j] * T2_vMmO[b,i]
  end
  if length(T1a) > 0
    internalT1a = T1a[norba,morba]
    internalT1b = T1b[morbb,norbb]
    T1_vo = T1a[virtualsa,occcorea]
    T1_VO = T1b[virtualsb,occcoreb]
    @mtensor TT1a[a,i] := T1_vo[a,i] - T1_VO[a,i]
    @mtensor TT1b[a,i] := T1_VO[a,i] - T1_vo[a,i]
    @mtensor T2ta[a,b,i,j] := T2a[a,b,i,j] + T1a[a,i] * T1a[b,j]
    @mtensor T2tb[a,b,i,j] := T2b[a,b,i,j] + T1b[a,i] * T1b[b,j]
    @mtensor T2tab[a,b,i,j] := T2ab[a,b,i,j] + T1a[a,i] * T1b[b,j]
    T1_no = T1a[norba,occcorea]
    T1_MO = T1b[morbb,occcoreb]
    T1_vm = T1a[virtualsa,morba]
    T1_VN = T1b[virtualsb,norbb]
    @mtensor M2_nVoO[a,i,j] := internalT1a * T2_vMoO[a,j,i]
    @mtensor M2_vMoO[a,j,i] := internalT1b * T2_nVoO[a,i,j]
    @mtensor M2_nVoO[a,i,j] -= T2_nMmO[i] * TT1b[a,j]
    @mtensor M2_vMoO[a,j,i] -= T2_nMoN[i] * TT1a[a,j]
    @mtensor M2_nVoO[a,i,j] -= T2_vMmO[a,i] * T1_no[j]
    @mtensor M2_vMoO[a,j,i] -= T2_nVoN[a,i] * T1_MO[j]
    T2t_nMoO = T2tab[norba,morbb,occcorea,occcoreb]
    @mtensor M2_nVoO[a,i,j] -= T2t_nMoO[j,i] * T1_vm[a]
    @mtensor M2_vMoO[a,j,i] -= T2t_nMoO[i,j] * T1_VN[a]
    T2_vnmo = T2a[virtualsa,norba,morba,occcorea]
    @mtensor M2_nVoO[a,i,j] -= T2_vnmo[a,j] * T1_MO[i]
    T2_MVON = T2b[morbb,virtualsb,occcoreb,norbb]
    @mtensor M2_vMoO[a,j,i] -= T2_MVON[a,j] * T1_no[i]
    @view(M2[norba,virtualsb,occcorea,occcoreb]) .+= M2_nVoO
    @view(M2[virtualsa,morbb,occcorea,occcoreb]) .+= M2_vMoO

    @mtensor M2_vVmO[a,b,i] := T2_nVmN[a] * TT1b[b,i]
    @mtensor M2_vVoN[b,a,i] := T2_vMmN[a] * TT1a[b,i]
    @mtensor M2_vVmO[a,b,i] += T2_nVoN[a,i] * T1_vm[b] 
    @mtensor M2_vVoN[b,a,i] += T2_vMmO[a,i] * T1_VN[b] 
    T2t_vVmN = T2tab[virtualsa,virtualsb,morba,norbb]
    @mtensor M2_vVmO[a,b,i] += T2t_vVmN[b,a] * T1_no[i]
    @mtensor M2_vVoN[b,a,i] += T2t_vVmN[a,b] * T1_MO[i] 
    T2_nvom = T2a[norba,virtualsa,occcorea,morba]
    @mtensor M2_vVmO[a,b,i] += T2_nvom[b,i] * T1_VN[a]
    @mtensor M2_vVoN[b,a,i] += T2_MVON[b,i] * T1_vm[a]
    @mtensor M2_vVmO[a,b,i] -= internalT1a * T2_vVoN[b,a,i]
    @mtensor M2_vVoN[b,a,i] -= internalT1b * T2_vVmO[a,b,i]
    @view(M2[virtualsa,virtualsb,morba,occcoreb]) .+= M2_vVmO
    @view(M2[virtualsa,virtualsb,occcorea,norbb]) .+= M2_vVoN

    @mtensor M2_vVoO[a,b,i,j] -= (T2_nMmO[i] * T1_VN[a]) * TT1b[b,j]
    @mtensor M2_vVoO[a,b,i,j] -= (T2_nMoN[j] * T1_vm[b]) * TT1a[a,i]

    @mtensor M2_vVoO[a,b,i,j] -= T2_nVmN[a] * T1_MO[i] * TT1b[b,j]
    @mtensor M2_vVoO[a,b,i,j] -= T2_vMmN[b] * T1_no[j] * TT1a[a,i]
    @mtensor M2_vVoO[a,b,i,j] += T2_nvmo[b,j] * (T1_VN[a] * T1_MO[i])
    @mtensor M2_vVoO[a,b,i,j] += T2_MVNO[a,i] * (T1_no[j] * T1_vm[b])
    @mtensor M2_vVoO[a,b,i,j] -= T2_vMmO[b,i] * (T1_no[j] * T1_VN[a])
    @mtensor M2_vVoO[a,b,i,j] -= T2_nVoN[a,j] * (T1_vm[b] * T1_MO[i])
    T2_VMNO = T2b[virtualsb,morbb,norbb,occcoreb]
    @mtensor M2_vVoO[a,b,i,j] -= internalT1a * TT1b[b,j] * T2_VMNO[a,i]
    @mtensor M2_vVoO[a,b,i,j] -= internalT1b * TT1a[a,i] * T2_nvom[b,j]
    @mtensor M2_vVoO[a,b,i,j] += internalT1a * T1_MO[i] * T2_vVoN[b,a,j]
    @mtensor M2_vVoO[a,b,i,j] += internalT1a * T1_VN[a] * T2_vMoO[b,j,i]
    @mtensor M2_vVoO[a,b,i,j] += internalT1b * T1_vm[b] * T2_nVoO[a,j,i]
    @mtensor M2_vVoO[a,b,i,j] += internalT1b * T1_no[j] * T2_vVmO[b,a,i]
    @mtensor M2[norba,morbb,morba,occcoreb][i] += T1_no[i]
    @mtensor M2[norba,morbb,occcorea,norbb][i] += T1_MO[i]
    @mtensor M2_vVoO[a,b,i,j] -= T2t_nMoO[j,i] * T2t_vVmN[b,a]
    @mtensor M2[norba,morbb,occcorea,occcoreb][i,j] -= T2t_nMoO[j,i]
    T2t_nvom = T2ta[norba,virtualsa,occcorea,morba]
    @mtensor M2[norba,virtualsb,morba,occcoreb][a,i] += T2t_nvom[a,i]
    T2t_MVON = T2tb[morbb,virtualsb,occcoreb,norbb]
    @mtensor M2[virtualsa,morbb,occcorea,norbb][a,i] += T2t_MVON[a,i]
    T2t_nVoN = T2tab[norba,virtualsb,occcorea,norbb]
    @mtensor M2[virtualsa,morbb,morba,occcoreb][a,i] += T2t_nVoN[a,i]
    T2t_vMmO  = T2tab[virtualsa,morbb,morba,occcoreb]
    @mtensor M2[norba,virtualsb,occcorea,norbb][a,i] += T2t_vMmO[a,i]
    if !isempty(occcorea) && !isempty(occcoreb)
      @mtensor M2[norba,virtualsb,morba,norbb][a] -= T1_vm[a]
      @mtensor M2[virtualsa,morbb,morba,norbb][a] -= T1_VN[a]
      @mtensor M2[virtualsa,virtualsb,morba,norbb][a,b] -= T2t_vVmN[b,a]
    end
  end
  if !isempty(occcorea) && !isempty(occcoreb)
    @view(M2[virtualsa,virtualsb,occcorea,occcoreb]) .+= M2_vVoO
  end
  return M2
end

"""
    calc_cc(EC::ECInfo, method::ECMethod)

  Calculate coupled cluster amplitudes.

  Exact specification of the method is given by `method`.
  Returns energies `::OutDict` with the following keys:
  - `"E"` - correlation energy
  - `"ESS"` - same-spin component
  - `"EOS"` - opposite-spin component
  - `"EO"` - open-shell component (defined as ``E_{αα} - E_{ββ}``)
  - `"EIAS"` - internal-active singles (for 2D methods)
  - `"EW"` - singlet/triplet energy contribution (for 2D methods)
"""
function calc_cc(EC::ECInfo, method::ECMethod)
  t0 = time_ns()
  print_info(method_name(method))

  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 3
    error("only implemented upto triples")
  end
  if is_unrestricted(method) || has_prefix(method, "R")
    if method.exclevel[1] == :full
      T1a = read_starting_guess4amplitudes(EC, Val(1), :α)
      T1b = read_starting_guess4amplitudes(EC, Val(1), :β)
    else
      T1a = zeros(0,0)
      T1b = zeros(0,0)
    end
    if method.exclevel[2] != :full
      error("No doubles is not implemented")
    end
    T2a = read_starting_guess4amplitudes(EC, Val(2), :α, :α)
    T2b = read_starting_guess4amplitudes(EC, Val(2), :β, :β)
    T2ab = read_starting_guess4amplitudes(EC, Val(2), :α, :β)
    # custom functions for dot products in diis
    dots1 = (calc_u_singles_dot, calc_u_singles_dot)
    dots2 = (calc_samespin_doubles_dot, calc_samespin_doubles_dot, calc_ab_doubles_dot)
    if method.exclevel[3] != :full
      Eh = cc_iterations!((T1a,T1b), (T2a,T2b,T2ab), (), EC, method, (dots1..., dots2...), [1.0, 1.0, 2.0, 2.0, 1.0])
    else
      T3aaa = read_starting_guess4amplitudes(EC, Val(3), :α, :α, :α)
      T3bbb = read_starting_guess4amplitudes(EC, Val(3), :β, :β, :β)
      T3aab = read_starting_guess4amplitudes(EC, Val(3), :α, :α, :β)
      T3abb = read_starting_guess4amplitudes(EC, Val(3), :α, :β, :β)
      dots3 = (calc_samespin_triples_dot, calc_samespin_triples_dot, calc_mixedspin_triples_dot, calc_mixedspin_triples_dot)
      Eh = cc_iterations!((T1a,T1b), (T2a,T2b,T2ab), (T3aaa,T3bbb,T3aab,T3abb), EC, method, (dots1..., dots2..., dots3...),
                          [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    end
  else
    if method.exclevel[1] == :full
      T1 = read_starting_guess4amplitudes(EC, Val(1))
    else
      T1 = zeros(0,0)
    end
    if method.exclevel[2] != :full
      error("No doubles is not implemented")
    end
    T2 = read_starting_guess4amplitudes(EC, Val(2))
    # custom functions for dot products in diis
    dots1 = (calc_cs_singles_dot,)
    dots2 = (calc_cs_doubles_dot,)
    if method.exclevel[3] != :full
      Eh = cc_iterations!((T1,), (T2,), (), EC, method, (dots1..., dots2...))
    else
      T3 = read_starting_guess4amplitudes(EC, Val(3))
      dots3 = (calc_cs_triples_dot,)
      Eh = cc_iterations!((T1,), (T2,), (T3,), EC, method, (dots1..., dots2..., dots3...))
    end
  end

  if has_prefix(method, "2D")
    ene = Eh["E"] + Eh["EIAS"]
    W = load1idx(EC,"2d_ccsd_W")[1]
    push!(Eh, "EW"=>W, "E"=>ene)
  end
  t0 = print_time(EC, t0, "total", 1)
  return Eh
end

function cc_iterations!(Amps1, Amps2, Amps3, EC::ECInfo, method::ECMethod, dots=(), weights=Float64[])
  t0 = time_ns()
  dc = (method.theory[1:2] == "DC")
  tworef = has_prefix(method, "2D")
  fixref = (has_prefix(method, "FRS") || has_prefix(method, "FRT"))
  restrict = has_prefix(method, "R")
  qv = has_prefix(method, "QV")
  if is_unrestricted(method) || has_prefix(method, "R")
    @assert (length(Amps1) == 2) && (length(Amps2) == 3) && (length(Amps3) == 4 || length(Amps3) == 0)
  else
    @assert (length(Amps1) == 1) && (length(Amps2) == 1) && (length(Amps3) == 1 || length(Amps3) == 0)
  end
  Amps = (Amps1..., Amps2..., Amps3...) 
  T2αβ = last(Amps2)
  diis = Diis(EC, weights)

  NormR1 = 0.0
  NormT1::Float64 = 0.0
  NormT2::Float64 = 0.0
  NormT3::Float64 = 0.0
  do_sing = (method.exclevel[1] == :full)
  Eh = OutDict("E"=>0.0, "ESS"=>0.0, "EOS"=>0.0, "EO"=>0.0)
  En1 = 0.0
  Eias = 0.0
  converged = false
  thren = sqrt(EC.options.cc.thr) * EC.options.cc.conven
  t0 = print_time(EC, t0, "initialization", 1)
  println("Iter     SqNorm      Energy      DE          Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    if length(Amps3) == 0 && !do_sing && qv
      Res, E = calc_qvcc_resid(EC, it, Amps...; dc)
      Eh = OutDict("E"=>E)
    else
      Res = calc_cc_resid(EC, Amps...; dc, tworef, fixref)
    end
    @assert typeof(Res) == typeof(Amps)
    Res1 = Res[1:length(Amps1)]
    Res2 = Res[length(Amps1)+1:length(Amps1)+length(Amps2)]
    Res3 = Res[length(Amps1)+length(Amps2)+1:end]
    if length(Amps1) == 2 && restrict
      # spin_project!(EC, Res...)
      # at the moment we don't project the triples
      spin_project!(EC, Res1..., Res2...)
    end
    if length(Amps3) == 1
      clean_cs_triples!(Res3...)
    end
    t1 = print_time(EC, t1, "residual", 2)
    NormT2 = calc_doubles_norm(Amps2...)
    NormR2 = calc_doubles_norm(Res2...)
    if has_prefix(method, "FRS")
      active = oss_active_orbitals(EC)
      T2αβ[active.ua,active.tb,active.ta,active.ub] = 1.0
    elseif has_prefix(method, "FRT")
      active = oss_active_orbitals(EC)
      T2αβ[active.ua,active.tb,active.ta,active.ub] = -1.0
    end
    if !qv
      Eh = calc_hylleraas(EC, Amps1..., Amps2..., Res1..., Res2...)
    end
    update_doubles!(EC, Amps2..., Res2...)
    if length(Amps3) > 0 
      NormT3 = calc_triples_norm(Amps3...)
      NormR3 = calc_triples_norm(Res3...)
      update_triples!(EC, Amps3..., Res3...)
      if length(Amps3) == 1
        clean_cs_triples!(Amps3...)
      end
    end
    if do_sing
      NormT1 = calc_singles_norm(Amps1...)
      NormR1 = calc_singles_norm(Res1...)
      update_singles!(EC, Amps1..., Res1...)
    end
    if length(Amps1) == 2 && restrict
      # spin_project!(EC, Amps...)
      # at the moment we don't project the triples
      spin_project!(EC, Amps1..., Amps2...)
    end
    perform!(diis, Amps, Res, dots)
    save_current_doubles(EC, Amps2...)
    En2 = calc_doubles_energy(EC, Amps2...)
    En = En2["E"]
    if do_sing
      save_current_singles(EC, Amps1...)
      En1 = calc_singles_energy(EC, Amps1...)
      En += En1["E"]
      if has_prefix(method, "2D")
        active = oss_active_orbitals(EC)
        T1α = first(Amps1)
        T1β = last(Amps1)
        W = load1idx(EC,"2d_ccsd_W")[1]
        Eias = - W * T1α[active.ua,active.ta] * T1β[active.tb,active.ub]
      end
    end
    ΔE = En - Eh["E"]
    NormR = NormR1 + NormR2
    NormT = 1.0 + NormT1 + NormT2
    if length(Amps3) > 0
      NormR += NormR3
      NormT += NormT3
    end
    if qv
      ΔE = 0.0
    end
    output_iteration(it, NormR, time_ns() - t0, NormT, Eh["E"], ΔE) 
    if NormR < EC.options.cc.thr && abs(ΔE) < thren
      converged = true
      break
    end
  end
  if !converged
    println("WARNING: CC iterations did not converge!")
  end
  if tworef
    push!(Eh, "EIAS"=>Eias)
  end
  if do_sing
    try2save_singles!(EC, Amps1...)
  end
  try2save_doubles!(EC, Amps2...)
  println()
  if length(Amps3) > 0
    output_norms("T1"=>NormT1, "T2"=>NormT2, "T3"=>NormT3)
  else
    output_norms("T1"=>NormT1, "T2"=>NormT2)
  end
  println()
  return Eh 
end

""" 
    calc_ccsdt(EC::ECInfo, useT3=false, cc3=false)

  Calculate decomposed closed-shell DC-CCSDT amplitudes.

  If `useT3`: (T) amplitudes from a preceding calculations will be used as starting guess.
  If cc3: calculate CC3 amplitudes.
"""
function calc_ccsdt(EC::ECInfo, useT3=false, cc3=false)
  t0 = time_ns()
  pert_svd_T = true
  
  if cc3
    error("SVD-CC3 not implemented yet")
    print_info("SVD-CC3")
  else
    print_info("SVD-DC-CCSDT")
    if pert_svd_T
      println("SVD-DC-CCSDT with SVD-(T)")
    end
  end
  if EC.options.cc.usedf && system_exists(EC.system)
    println("Using density fitting")
    calc_df_integrals(EC)
  else
    println("Decomposing integrals")
    calc_integrals_decomposition(EC)
  end
  t0 = print_time(EC, t0, "integrals decomposition", 1)
  T1 = read_starting_guess4amplitudes(EC, Val(1))
  T2 = read_starting_guess4amplitudes(EC, Val(2))
  t0 = print_time(EC, t0, "starting guess", 2) 
  # calc_dressed_3idx(EC,zeros(size(T1)))
  calc_dressed_3idx(EC, T1)
  if useT3
    calc_triples_decomposition(EC)
  else
    calc_triples_decomposition_without_triples(EC, T2)
  end
  t0 = print_time(EC, t0, "triples decomposition", 1)
  diis = Diis(EC)
  thren = sqrt(EC.options.cc.thr) * EC.options.cc.conven

  if EC.options.cc.project_voXL
    calc_space4project_voXL(EC, T2)
    t0 = print_time(EC, t0, "space for project_voXL", 1)
  end
  # calc intermediates for SVD-T
  calc_intermediates4triples(EC)
  t0 = print_time(EC, t0, "intermediates for SVD-T", 1)
# svd-ccsd(t)  
  if pert_svd_T
    t1 = time_ns()
    save_pseudodressed_3idx(EC)
    save!(EC, "df_mm", load2idx(EC,"f_mm"))
    calc_SVD_pert_T(EC, T2)
    t1 = print_time(EC, t1, "R_XXX(T1,T2)", 2)
    R3 = load3idx(EC, "R_XXX")
    T3 = update_deco_triples(EC, R3, false)
    save!(EC, "T_XXX", T3)
    t1 = print_time(EC, t1, "T_XXX(T1,T2)", 2)
    
    R1, R2 = SVD_triples_to_singles_and_doubles_residuals(EC)
    t1 = print_time(EC, t1, "R1&R2(T_XXX)", 2)
  
    Eh_init = calc_hylleraas(EC, T1, T2, R1, R2)
    t1 = print_time(EC, t1, "energy(T_XXX)", 2)
    output_E_method(Eh_init["E"], "SVD-CCSD(T)", "correlation energy:")
    t0 = print_time(EC, t0, "SVD-CCSD(T)", 1)
    println()
  end

  println("Iter     SqNorm      Energy      DE          Res         Time")
  NormR1 = 0.0
  NormT1 = 0.0
  NormT2 = 0.0
  NormT3 = 0.0
  R1 = Float64[]
  Eh = OutDict("E"=>0.0, "ESS"=>0.0, "EOS"=>0.0, "EO"=>0.0)

  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    #get dressed integrals
    calc_dressed_3idx(EC, T1)
    # test_dressed_ints(EC,T1) #DEBUG
    t1 = print_time(EC, t1, "dressed 3-idx integrals", 2)
    R1, R2 = calc_cc_resid(EC, T1, T2)
    t1 = print_time(EC, t1, "ccsd residual", 2)
    calc_triples_residuals!(EC, R1, R2, T2)
    t1 = print_time(EC, t1, "triples residual", 2)
    NormT1 = calc_singles_norm(T1)
    NormT2 = calc_doubles_norm(T2)
    T3 = load3idx(EC, "T_XXX")
    NormT3 = calc_deco_triples_norm(T3)
    NormR1 = calc_singles_norm(R1)
    NormR2 = calc_doubles_norm(R2)
    R3 = load3idx(EC, "R_XXX")
    NormR3 = calc_deco_triples_norm(R3)
    Eh = calc_hylleraas(EC, T1, T2, R1, R2)
    T1 += update_singles(EC, R1)
    T2 += update_doubles(EC, R2)
    T3 += update_deco_triples(EC, R3)
    perform!(diis, (T1,T2,T3), (R1,R2,R3))
    save!(EC, "T_XXX", T3)
    En1 = calc_singles_energy(EC, T1)
    En2 = calc_doubles_energy(EC, T2)
    En = En1["E"] + En2["E"]
    ΔE = En - Eh["E"]
    NormR = NormR1 + NormR2 + NormR3
    NormT = 1.0 + NormT1 + NormT2 + NormT3
    output_iteration(it, NormR, time_ns() - t0, NormT, Eh["E"], ΔE)
    if NormR < EC.options.cc.thr && abs(ΔE) < thren
      break
    end
  end
  println()
  output_norms("T1"=>NormT1, "T2"=>NormT2, "T3"=>NormT3)
  println()
  if pert_svd_T
    push!(Eh, "SVD-CCSD(T)"=>Eh_init["E"])
  end
  print_time(EC, t0, "iterations", 1)
  return Eh
end

# Function to calculate length for buffer(s) buf
# autogenerated by @print_buffer_usage
function auto_buf_length4SVD_triples_to_singles_and_doubles_residuals(EC, nvirt, nX, nbX, nocc, nL, lenL, lenX, lenbX)
    buf = [0, 0]
    RR_vovo = pseudo_alloc!(buf, nvirt, nocc, nvirt, nocc)
    R_voX = pseudo_alloc!(buf, nvirt, nocc, nX)
    R_ooX = pseudo_alloc!(buf, nocc, nocc, nX)
    w_ovX = pseudo_alloc!(buf, nocc, nvirt, nX)
    if EC.options.cc.project_voXL
        RR_vobX = pseudo_alloc!(buf, nvirt, nocc, nbX)
    end
    fU_X = pseudo_alloc!(buf, nX)
    begin
        TvoXX = pseudo_alloc!(buf, nvirt, nocc, nX, lenX)
        Bv_vvX = pseudo_alloc!(buf, nvirt, nvirt, lenX)
        Bv_ooX = pseudo_alloc!(buf, nocc, nocc, lenX)
        if EC.options.cc.project_voXL
            bV_XbXL = pseudo_alloc!(buf, lenX, nbX, nL)
            begin
                TUU_XbXov = pseudo_alloc!(buf, lenX, lenbX, nocc, nvirt)
                pseudo_drop!(buf, TUU_XbXov)
            end
        end
        begin
            bB_voXL = pseudo_alloc!(buf, nvirt, nocc, lenX, lenL)
            B_voXL = pseudo_alloc!(buf, nvirt, nocc, lenX, lenL)
            pseudo_drop!(buf, B_voXL)
            V_XXL = pseudo_alloc!(buf, nX, lenX, lenL)
            if EC.options.cc.project_voXL
                nothing
            else
                V_voXL = pseudo_alloc!(buf, nvirt, nocc, lenX, lenL)
                pseudo_drop!(buf, V_voXL)
            end
            pseudo_drop!(buf, V_XXL)
            pseudo_drop!(buf, bB_voXL)
        end
        if EC.options.cc.project_voXL
            pseudo_drop!(buf, bV_XbXL)
        end
        UBv = pseudo_alloc!(buf, nvirt, nocc, nX, lenX)
        pseudo_drop!(buf, UBv)
        pseudo_drop!(buf, Bv_ooX, Bv_vvX)
        pseudo_drop!(buf, TvoXX)
    end
    pseudo_drop!(buf, fU_X)
    if EC.options.cc.project_voXL
        bUvoX = pseudo_alloc!(buf, nvirt, nocc, nbX)
        pseudo_drop!(buf, bUvoX, RR_vobX)
    end
    pseudo_drop!(buf, w_ovX)
    pseudo_drop!(buf, R_ooX)
    pseudo_drop!(buf, R_voX)
    pseudo_drop!(buf, RR_vovo)
    A_XL = pseudo_alloc!(buf, nX, nL)
    B_XX = pseudo_alloc!(buf, nX, nX)
    pseudo_drop!(buf, A_XL, B_XX)
    return buf[2]
end

"""
    SVD_triples_to_singles_and_doubles_residuals(EC)

  Calculate contributions from triples to singles and doubles residuals.
"""
function SVD_triples_to_singles_and_doubles_residuals(EC)
  t1 = time_ns()
  mem1 = free_memory()
  UvoX = load3idx(EC, "C_voX")
  #display(UvoX)

  #load decomposed amplitudes
  T_XXX = load3idx(EC, "T_XXX")
  #display(T_XXX)

  #load df coeff
  ovLfile, ovL = mmap3idx(EC, "d_ovL")
  ooLfile, ooL = mmap3idx(EC, "d_ooL")
  vvLfile, vvL = mmap3idx(EC, "d_vvL")

  if EC.options.cc.project_voXL
    UU_oXobXfile, UU_oXobX = mmap4idx(EC, "UU_oXo{bX}")
    nbX = size(UU_oXobX, 4)
  else
    nbX = size(T_XXX, 1)
  end

  #load dressed fock matrices
  SP = EC.space
  dfock = load2idx(EC, "df_mm")    
  dfov = dfock[SP['o'], SP['v']]
 
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nX = size(T_XXX, 1)
  nL = size(ovL, 3)

  LBlks = get_spaceblocks(1:nL)
  XBigBlks = get_spaceblocks(1:nX, 256)
  bXBlks = get_spaceblocks(1:nbX)

  maxL = maximum(length, LBlks)
  maxbX = maximum(length, bXBlks)
  maxXBig = maximum(length, XBigBlks)
  lenbuf = auto_buf_length4SVD_triples_to_singles_and_doubles_residuals(EC, nvirt, nX, nbX, nocc, nL, maxL, maxXBig, maxbX)
  @buffer buf(lenbuf) begin
  mem2 = print_memory(EC, mem1, "for buffer in SVD triples to singles and doubles residuals", 2)
  # @print_buffer_usage buf begin

  # RR[aibj] = ``RR^{ij}_{ab}``
  RR_vovo = alloc!(buf, nvirt, nocc, nvirt, nocc)
  RR_vovo .= 0.0
  R_voX = alloc!(buf, nvirt, nocc, nX)
  R_voX .= 0.0
  R_ooX = alloc!(buf,nocc,nocc,nX)
  R_ooX .= 0.0
  B_ooXLfile, B_ooXL = mmap4idx(EC, "B_ooXL")
  A_XLfile, A_XL = mmap2idx(EC, "A_XL")
  w_ovX = alloc!(buf, nocc, nvirt, nX)
  load!(EC, "w_ovX", w_ovX)
  if EC.options.cc.project_voXL
    RR_vobX = alloc!(buf, nvirt, nocc, nbX)
    RR_vobX .= 0.0
  end
  # ``fU^X = \hat f_k^c U^{kX}_c``
  fU_X = alloc!(buf, nX)
  @mtensor fU_X[X] = dfov[k,c] * UvoX[c,k,X]
  for X in XBigBlks
    lenX = length(X)
    v!T_XXX = @mview T_XXX[:,:,X]
    v!UvoX = @mview UvoX[:,:,X]
    # ``T^i_{aYX} = U^{iZ}_a T_{ZYX}``
    TvoXX = alloc!(buf, nvirt, nocc, nX, lenX)
    @mtensor TvoXX[a,i,Y,X] = UvoX[a,i,Z] * v!T_XXX[Z,Y,X]
    # ``R_{aY}^i += T_{aYX}^i fU^X``
    v!fU_X = @mview fU_X[X]
    @mtensor R_voX[a,i,Y] += TvoXX[a,i,Y,X] * v!fU_X[X]
    # ``R_{jX}^i -= T_{dYX}^i w_j^{dY}``
    v!R_ooX = @mview R_ooX[:,:,X]
    @mtensor v!R_ooX[j,i,X] = -TvoXX[d,i,Y,X] * w_ovX[j,d,Y]
    Bv_vvX = alloc!(buf, nvirt, nvirt, lenX)
    Bv_vvX .= 0.0
    Bv_ooX = alloc!(buf, nocc, nocc, lenX)
    Bv_ooX .= 0.0
    if EC.options.cc.project_voXL
      bV_XbXL = alloc!(buf, lenX, nbX, nL)
      for bX in bXBlks
        lenbX = length(bX)
        v!UU_oXobX = @mview UU_oXobX[:,:,:,bX]
        # ``TUU^j_{b\bar XX} = T_{bZX}^l UU_{\bar Xl}^{jZ}``
        TUU_XbXov = alloc!(buf, lenX, lenbX, nocc, nvirt)
        @mtensor TUU_XbXov[X,bX,j,b] = TvoXX[b,l,Z,X] * v!UU_oXobX[j,Z,l,bX]
        # ``\bar V_{\bar XX}^{L} = TUU^j_{b\bar XX} v_j^{bL}``
        v!bV_XbXL = @mview bV_XbXL[:,bX,:]
        @mtensor v!bV_XbXL[X,bX,L] = TUU_XbXov[X,bX,j,b] * ovL[j,b,L]
        drop!(buf, TUU_XbXov)
      end
    end
    for L in LBlks
      lenL = length(L)
      v!ooL = @mview ooL[:,:,L]
      v!vvL = @mview vvL[:,:,L]
      v!ovL = @mview ovL[:,:,L]
      # ``\bar B_a^{iXL} = \hat v_a^{bL} U^{iX}_b``
      bB_voXL = alloc!(buf, nvirt, nocc, lenX, lenL)
      @mtensor bB_voXL[a,i,X,L] = v!vvL[a,b,L] * v!UvoX[b,i,X]
      # ``B_a^{iXL} = \hat v_j^{iL} U^{jX}_a``
      B_voXL = alloc!(buf, nvirt, nocc, lenX, lenL)
      @mtensor B_voXL[a,i,X,L] = v!ooL[j,i,L] * v!UvoX[a,j,X]
      # ``Bv_a^{cX} = \bar B_a^{kXL} v_k^{cL}``
      @mtensor Bv_vvX[a,c,X] += bB_voXL[a,k,X,L] * v!ovL[k,c,L]
      # ``Bv_k^{iX} = B_c^{iXL} v_k^{cL}``
      @mtensor Bv_ooX[k,i,X] += B_voXL[c,i,X,L] * v!ovL[k,c,L]
      # ``\bar B_a^{iXL} -= B_a^{iXL}``
      @mtensor bB_voXL[a,i,X,L] -= B_voXL[a,i,X,L]
      drop!(buf, B_voXL)

      v!B_ooXL = @mview B_ooXL[:,:,:,L]
      v!A_XL = @mview A_XL[:,L]
      # ``V_{YX}^{L} = T_{ZYX} A^{ZL}``
      V_XXL = alloc!(buf, nX, lenX, lenL)
      @mtensor V_XXL[Y,X,L] = v!T_XXX[Z,Y,X] * v!A_XL[Z,L]
      # ``R_{jX}^{i} = 2 B_j^{iYL} V_{YX}^{L}``
      @mtensor v!R_ooX[j,i,X] += 2.0 * v!B_ooXL[j,i,Y,L] * V_XXL[Y,X,L]
      # ``R_{aZ}^i = 2 (\bar B - B)_a^{LiX} V_{ZX}^{L}``
      @mtensor R_voX[a,i,Z] += 2.0 * bB_voXL[a,i,X,L] * V_XXL[Z,X,L]
      if EC.options.cc.project_voXL
        v!bV_XbXL = @mview bV_XbXL[:,:,L]
        # ``RR^{i}_{a\bar X} -= (\bar B - B)_a^{LiX} \bar V_{X\bar X}^{L}``
        @mtensor RR_vobX[a,i,bX] -= bB_voXL[a,i,X,L] * v!bV_XbXL[X,bX,L]
      else
        # V[ajXL]=``V_{aX}^{jL} = T^i_{aYX} B_i^{jYL}``
        V_voXL = alloc!(buf, nvirt, nocc, lenX, lenL)
        @mtensor V_voXL[a,j,X,L] = TvoXX[a,i,Y,X] * v!B_ooXL[i,j,Y,L]
        # ``RR^{ij}_{ab} -= (\bar B - B)_a^{LiX} V_{bX}^{jL}``
        @mtensor RR_vovo[a,i,b,j] -= bB_voXL[a,i,X,L] * V_voXL[b,j,X,L]
        drop!(buf, V_voXL)
      end
      drop!(buf, V_XXL)
      drop!(buf, bB_voXL)
    end
    if EC.options.cc.project_voXL
      drop!(buf, bV_XbXL)
    end
    # ``Bv_k^{iX} -= \hat f_k^c U^{iX}_c``
    @mtensor Bv_ooX[k,i,X] -= dfov[k,c] * v!UvoX[c,i,X]
    # ``UBv_a^{iYX} = U^{iY}_c Bv_a^{cX}``
    UBv = alloc!(buf, nvirt, nocc, nX, lenX)
    @mtensor UBv[a,i,Y,X] = UvoX[c,i,Y] * Bv_vvX[a,c,X]
    # ``UBv_a^{iYX} -= U^{kY}_a Bv_k^{iX}``
    @mtensor UBv[a,i,Y,X] -= UvoX[a,k,Y] * Bv_ooX[k,i,X]
    # ``R_{aZ}^i -= T_{ZYX} UBv_a^{iYX}``
    @mtensor R_voX[a,i,Z] -= v!T_XXX[Z,Y,X] * UBv[a,i,Y,X]
    drop!(buf, UBv)
    drop!(buf, Bv_ooX, Bv_vvX)
    drop!(buf, TvoXX)
  end
  drop!(buf, fU_X)
  if EC.options.cc.project_voXL
    close(UU_oXobXfile)
    bUvoX = alloc!(buf, nvirt, nocc, nbX)
    load!(EC, "C_vo{bX}", bUvoX)
    @mtensor RR_vovo[a,i,b,j] += RR_vobX[a,i,bX] * bUvoX[b,j,bX]
    drop!(buf, bUvoX, RR_vobX)
  end
  drop!(buf, w_ovX)
  close(A_XLfile)
  close(B_ooXLfile)
  # ``R^i_a -= R_{jY}^{i} U^{jY}_a``
  @mtensor R1[a,i] := -(R_ooX[j,i,Y] * UvoX[a,j,Y])
  drop!(buf, R_ooX)
  # ``RR^{ij}_{ab} += R_{aZ}^i U^{jZ}_b``
  @mtensor RR_vovo[a,i,b,j] += R_voX[a,i,Z] * UvoX[b,j,Z]
  drop!(buf, R_voX)
  # ``R^{ij}_{ab} = RR^{ij}_{ab} + RR^{ji}_{ba}``
  @mtensor R2[a,b,i,j] := RR_vovo[a,i,b,j] + RR_vovo[b,j,a,i]
  drop!(buf, RR_vovo)

  A_XL = alloc!(buf, nX, nL)
  load!(EC, "A_XL", A_XL)
  B_XX = alloc!(buf, nX, nX)
  load!(EC, "B_XX", B_XX)
  # ``B^{XY} -= 2 A^{XL} A^{YL}``
  @mtensor B_XX[X,Y] -= 2.0 * A_XL[X,L] * A_XL[Y,L]
  # ``R^i_a -= U^{iX}_a (T_{XYZ} B^{YZ})
  @mtensor R1[a,i] -= UvoX[a,i,X] * (T_XXX[X,Y,Z] * B_XX[Y,Z])
  drop!(buf, A_XL, B_XX)

  close(ovLfile)
  close(ooLfile)
  close(vvLfile)
  end #buffer
  return R1, R2
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
  tol2 = sqrt(EC.options.cc.ampsvdtol*EC.options.cc.ampsvdfac)
  UaiX = svd_decompose(reshape(permutedims(T2, (1,3,2,4)), (nocc*nvirt, nocc*nvirt)), 
                       nvirt, nocc, tol2; description="Intermediate triples")
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  D2 = calc_4idx_T3T3_XY(EC, T2, UaiX, ϵX) 
  UaiX = svd_decompose(reshape(D2, (nocc*nvirt, nocc*nvirt)), 
                       nvirt, nocc, EC.options.cc.ampsvdtol; description="Triples")
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save!(EC, "e_X", ϵX)
  #display(UaiX)
  naux = length(ϵX)
  save!(EC,"C_voX",UaiX)
  # TODO: calc starting guess for T_XXX from T2 and UvoX
  save!(EC,"T_XXX",zeros(naux, naux, naux))
end

"""
    calc_triples_decomposition(EC::ECInfo)

  Decompose ``T^{ijk}_{abc}`` as ``U^{iX}_a U^{jY}_b U^{kZ}_c T_{XYZ}``.
"""
function calc_triples_decomposition(EC::ECInfo)
  println("T^ijk_abc-decomposition")
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)

  Triples_Amplitudes = zeros(nvirt, nocc, nvirt, nocc, nvirt, nocc)
  t3file, T3 = mmap4idx(EC, "T_vvvooo")
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
  UaiX = svd_decompose(reshape(Triples_Amplitudes, (nocc*nvirt, nocc*nocc*nvirt*nvirt)), 
                      nvirt, nocc, sqrt(EC.options.cc.ampsvdtol); description="Triples")
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save!(EC, "e_X", ϵX)
  #display(UaiX)
  save!(EC,"C_voX",UaiX)

  @mtensor T3_vovoX[b,j,c,k,X] := Triples_Amplitudes[a,i,b,j,c,k] * UaiX[a,i,X]
  @mtensor T3_voXX[c,k,X,Y] := T3_vovoX[b,j,c,k,X] * UaiX[b,j,Y]
  @mtensor T3_XXX[X,Y,Z] := T3_voXX[c,k,X,Y] * UaiX[c,k,Z]
  save!(EC,"T_XXX",T3_XXX)
  #display(T3_XXX)

  # @mtensor begin
  #  T3_decomp_check[a,i,b,j,c,k] := T3_XXX[X,Y,Z] * UaiX2[a,i,X] * UaiX2[b,j,Y] * UaiX2[c,k,Z]
  # end
  # test_calc_pertT_from_T3(EC,T3_decomp_check)
end

# Function to calculate length for buffer(s) buf
# autogenerated by @print_buffer_usage
function auto_buf_length4calc_4idx_T3T3_XY(EC::ECInfo, nvirt, nocc, nX, nL, lenL, lenX)
    buf = [0, 0]
    T_voX = pseudo_alloc!(buf, nvirt, nocc, nX)
    X_LXX = pseudo_alloc!(buf, nL, nX, nX)
    V_vvX = pseudo_alloc!(buf, nvirt, nvirt, nX)
    V_ooX = pseudo_alloc!(buf, nocc, nocc, nX)
    begin
        X_voXL = pseudo_alloc!(buf, nvirt, nocc, nX, lenL)
        UX_LXX = pseudo_alloc!(buf, lenL, nX, nX)
        pseudo_drop!(buf, UX_LXX, X_voXL)
        W_XL = pseudo_alloc!(buf, nX, lenL)
        pseudo_drop!(buf, W_XL)
    end
    if EC.options.cc.project_t3iii
        T3 = pseudo_alloc!(buf, nX, nX, nvirt, nocc)
    end
    begin
        R = pseudo_alloc!(buf, nvirt, nocc, lenX)
        E_voXY = pseudo_alloc!(buf, nvirt, nocc, lenX)
        pseudo_drop!(buf, E_voXY)
        pseudo_drop!(buf, R)
    end
    if EC.options.cc.project_t3iii
        UU = pseudo_alloc!(buf, nX, nX, nocc)
        TU = pseudo_alloc!(buf, nX, nX, nvirt)
        TUU4i = pseudo_alloc!(buf, nX, nX, nvirt)
        ΔD2 = pseudo_alloc!(buf, nvirt, nvirt)
        pseudo_drop!(buf, UU, TU, TUU4i, ΔD2)
    end
    pseudo_reset!(buf)
    return buf[2]
end

"""
    calc_4idx_T3T3_XY(EC::ECInfo, T2, UvoX, ϵX)

  Calculate ``D^{ij}_{ab} = T^i_{aXY} T^j_{bXY}`` using half-decomposed imaginary-shifted perturbative triple amplitudes 
  ``T^i_{aXY}`` from `T2` (and `UvoX`)
"""
function calc_4idx_T3T3_XY(EC::ECInfo, T2, UvoX, ϵX)
  mem1 = free_memory()
  voLfile, voL = mmap3idx(EC, "d_voL")
  ooLfile, ooL = mmap3idx(EC, "d_ooL")
  vvLfile, vvL = mmap3idx(EC, "d_vvL")

  nvirt, nocc, nX = size(UvoX)
  nL = size(voL, 3)

  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)

  lenbuf = auto_buf_length4calc_4idx_T3T3_XY(EC, nvirt, nocc, nX, nL, maxL, nX)
  @buffer buf(lenbuf) begin
  mem2 = print_memory(EC, mem1, "for buffer in half-decomposed T3 calculation", 2)
  # @print_buffer_usage buf begin
  # ``T^i_{aX} = U^{†b}_{jX} T^{ij}_{ab}``
  T_voX = alloc!(buf, nvirt, nocc, nX)
  @mtensor T_voX[a,i,X] = UvoX[b,j,X] * T2[a,b,i,j]
  X_LXX = alloc!(buf, nL, nX, nX)
  V_vvX = alloc!(buf, nvirt, nvirt, nX)
  V_vvX .= 0.0
  V_ooX = alloc!(buf, nocc, nocc, nX)
  V_ooX .= 0.0
  for L in LBlks
    lenL = length(L)
    v!voL = @mview voL[:,:,L]
    v!vvL = @mview vvL[:,:,L]
    v!ooL = @mview ooL[:,:,L]
    # ``X_{bX}^{jL} = T^j_{cX} \hat v_{b}^{cL} - T^l_{bX} \hat v_{l}^{jL}``
    X_voXL = alloc!(buf, nvirt, nocc, nX, lenL)
    @mtensor X_voXL[b,j,X,L] = T_voX[c,j,X] * v!vvL[b,c,L] 
    @mtensor X_voXL[b,j,X,L] -= T_voX[b,l,X] * v!ooL[l,j,L]
    # ``UX_{XY}^L = X_{bX}^{jL} U^{†b}_{jY}``
    UX_LXX = alloc!(buf, lenL, nX, nX)
    @mtensor UX_LXX[L,X,Y] = X_voXL[b,j,X,L] * UvoX[b,j,Y]
    v!X_LXX = @mview X_LXX[L,:,:]
    @mtensor v!X_LXX[L,X,Y] = UX_LXX[L,X,Y] + UX_LXX[L,Y,X]
    drop!(buf, UX_LXX, X_voXL)
    # ``W_{X}^{L} = \hat v_c^{kL} U^{†c}_{kX}``
    W_XL = alloc!(buf, nX, lenL)
    @mtensor W_XL[X,L] = v!voL[c,k,L] * UvoX[c,k,X]
    @mtensor V_vvX[a,d,X] += v!vvL[a,d,L] * W_XL[X,L]
    @mtensor V_ooX[l,j,X] += v!ooL[l,j,L] * W_XL[X,L]
    drop!(buf, W_XL)
  end
  D2 = zeros(nvirt, nocc, nvirt, nocc)
  ϵo, ϵv = orbital_energies(EC)
  shifti = EC.options.cc.deco_ishiftt
  if EC.options.cc.project_t3iii 
    # for the T^iii projection the full T^i_{aXY} is needed...
    T3 = alloc!(buf, nX, nX, nvirt, nocc)
  end
  # triangular loop over X,Y
  for Y = 1:nX
    X = 1:Y
    lenX = length(X)
    R = alloc!(buf, nvirt, nocc, lenX)
    v!X_LXY = @mview X_LXX[:,X,Y]
    @mtensor R[a,i,X] = v!X_LXY[L,X] * voL[a,i,L]
    v!T_voX = @mview T_voX[:,:,X]
    v!V_vvY = @mview V_vvX[:,:,Y]
    v!V_ooY = @mview V_ooX[:,:,Y]
    @mtensor R[a,i,X] += v!T_voX[c,i,X] * v!V_vvY[a,c]
    @mtensor R[a,i,X] -= v!T_voX[a,l,X] * v!V_ooY[l,i]
    v!T_voY = @mview T_voX[:,:,Y]
    v!V_vvX = @mview V_vvX[:,:,X]
    v!V_ooX = @mview V_ooX[:,:,X]
    @mtensor R[a,i,X] += v!T_voY[c,i] * v!V_vvX[a,c,X]
    @mtensor R[a,i,X] -= v!T_voY[a,l] * v!V_ooX[l,i,X]
    # ``E^d_{lXY} = V_{aX}^{d} U^{†a}_{lY} - V_{lX}^{i} U^{†d}_{iY}``
    E_voXY = alloc!(buf, nvirt, nocc, lenX)
    v!UvoY = @mview UvoX[:,:,Y]
    @mtensor E_voXY[d,l,X] = v!V_vvX[a,d,X] * v!UvoY[a,l] 
    @mtensor E_voXY[d,l,X] -= v!V_ooX[l,i,X] * v!UvoY[d,i]
    v!UvoX = @mview UvoX[:,:,X]
    @mtensor E_voXY[d,l,X] += v!V_vvY[a,d] * v!UvoX[a,l,X] 
    @mtensor E_voXY[d,l,X] -= v!V_ooY[l,i] * v!UvoX[d,i,X]
    # ``R^i_{aXY} += E^d_{lXY} T^il_{ad}``
    @mtensor R[a,i,X] += E_voXY[d,l,X] * T2[a,d,i,l]
    drop!(buf, E_voXY)
    # calc T^i_{aXY} = R^i_{aXY} / (ϵ_X + ϵ_Y + ϵ_v[a] - ϵ_o[i])
    if shifti > 1.e-10
      # imaginary-shifted triples
      for I ∈ CartesianIndices(R)
        a,i,iX = Tuple(I)
        den = ϵX[iX] + ϵX[Y] + ϵv[a] - ϵo[i]
        R[I] *= -den/(den^2 + shifti)
      end
    else
      for I ∈ CartesianIndices(R)
        a,i,iX = Tuple(I)
        R[I] /= -(ϵX[iX] + ϵX[Y] + ϵv[a] - ϵo[i])
      end
    end
    if Y > 1
      v!R = @mview R[:,:,1:(Y-1)]
      @mtensor D2[a,i,b,j] += 2.0 * v!R[a,i,X] * v!R[b,j,X]
    end
    # diagonal contribution X=Y
    v!RYY = @mview R[:,:,Y]
    @mtensor D2[a,i,b,j] += v!RYY[a,i] * v!RYY[b,j]
    if EC.options.cc.project_t3iii 
      permutedims!(@view(T3[X,Y,:,:]), R, (3,1,2))
      permutedims!(@view(T3[Y,X,:,:]), R, (3,1,2))
    end
    drop!(buf, R)
  end

  close(voLfile)
  close(ooLfile)
  close(vvLfile)
  if EC.options.cc.project_t3iii 
    # remove T^iii contributions from D2
    UU = alloc!(buf, nX, nX, nocc)
    for i = 1:nocc
      v!UvoX = @mview UvoX[:,i,:]
      v!UU = @mview UU[:,:,i]
      @mtensor v!UU[X,Y] = v!UvoX[a,X] * v!UvoX[a,Y]
    end
    TUU4i = alloc!(buf, nX, nX, nvirt)
    TU = alloc!(buf, nX, nX, nvirt)
    ΔD2 = alloc!(buf, nvirt, nvirt)
    for i = 1:nocc
      v!UUi = @mview UU[:,:,i]
      v!T_i = @mview T3[:,:,:,i]
      @mtensor TU[X',Y,a] = v!T_i[X,Y,a] * v!UUi[X,X']
      @mtensor TUU4i[X',Y',a] = TU[X',Y,a] * v!UUi[Y,Y']
      for j = 1:nocc
        v!T_j = @mview T3[:,:,:,j]
        @mtensor ΔD2[a,b] = TUU4i[X,Y,a] * v!T_j[X,Y,b]
        v!D2 = @mview D2[:,i,:,j]
        @mtensor v!D2[a,b] -= ΔD2[a,b]
        if i != j
          v!D2 = @mview D2[:,j,:,i]
          @mtensor v!D2[b,a] -= ΔD2[a,b]
        end
      end
    end
    drop!(buf, UU, TU, TUU4i, ΔD2)
  end
  reset!(buf)
  end # buffer
  # display(D2)
  return D2
end

# Function to calculate length for buffer(s) buf
# autogenerated by @print_buffer_usage
function auto_buf_length4calc_SVD_pert_T(nvirt, nocc, nX, nL, lenL, lenX)
    buf = [0, 0]
    V_vvX = pseudo_alloc!(buf, nvirt, nvirt, nX)
    V_ooX = pseudo_alloc!(buf, nocc, nocc, nX)
    begin
        W_XL = pseudo_alloc!(buf, nX, lenL)
        pseudo_drop!(buf, W_XL)
    end
    TvoX = pseudo_alloc!(buf, nvirt, nocc, nX)
    RR = pseudo_alloc!(buf, nX, nX, nX)
    begin
        RvoXX = pseudo_alloc!(buf, nvirt, nocc, nX, lenX)
        pseudo_drop!(buf, RvoXX)
    end
    R = pseudo_alloc!(buf, nX, nX, nX)
    pseudo_reset!(buf)
    return buf[2]
end


"""
    calc_SVD_pert_T(EC::ECInfo, T2)

  Calculate SVD-CCSD(T).
"""

function calc_SVD_pert_T(EC::ECInfo, T2)
  t1 = time_ns()
  mem1 = free_memory()
  UvoX = load3idx(EC, "C_voX")
  #display(UvoX)
  nvirt, nocc, nX = size(UvoX)

  #load df coeff
  ooLfile, ooL = mmap3idx(EC, "d_ooL")
  voLfile, voL = mmap3idx(EC, "d_voL")
  vvLfile, vvL = mmap3idx(EC, "d_vvL")

  nL = size(ooL, 3)

  LBlks = get_spaceblocks(1:nL)
  maxL = maximum(length, LBlks)
  XBlks = get_spaceblocks(1:nX)
  maxX = maximum(length, XBlks)

  lenbuf = auto_buf_length4calc_SVD_pert_T(nvirt, nocc, nX, nL, maxL, maxX)
  @buffer buf(lenbuf) begin
  mem2 = print_memory(EC, mem1, "for buffer in SVD-CCSD(T) calculation", 2)
  # @print_buffer_usage buf begin
  V_vvX = alloc!(buf, nvirt, nvirt, nX)
  V_vvX .= 0.0
  V_ooX = alloc!(buf, nocc, nocc, nX)
  V_ooX .= 0.0
  for L in LBlks
    lenL = length(L)
    v!ooL = @mview ooL[:,:,L]
    v!voL = @mview voL[:,:,L]
    v!vvL = @mview vvL[:,:,L]
    # ``W_{X}^{L} = v_c^{kL} U^{†c}_{kX}``
    W_XL = alloc!(buf, nX, lenL)
    @mtensor W_XL[X,L] = v!voL[c,k,L] * UvoX[c,k,X]
    # ``V_{aX}^{d} = v_{a}^{dL} W_{X}^{L}``
    @mtensor V_vvX[a,d,X] += v!vvL[a,d,L] * W_XL[X,L]
    # ``V_{lX}^{i} = v_{l}^{iL} W_{X}^{L}``
    @mtensor V_ooX[l,i,X] += v!ooL[l,i,L] * W_XL[X,L]
    drop!(buf, W_XL)
  end
  close(vvLfile)
  close(voLfile)
  close(ooLfile)
  TvoX = alloc!(buf, nvirt, nocc, nX)
  @mtensor TvoX[a,i,X] = UvoX[b,j,X] * T2[a,b,i,j]
  
  RR = alloc!(buf, nX, nX, nX)
  for X in XBlks
    lenX = length(X)
    v!TvoX = @mview TvoX[:,:,X]
    v!RR = @mview RR[:,:,X]
    RvoXX = alloc!(buf, nvirt, nocc, nX, lenX)
    @mtensor RvoXX[a,i,Y,X] = v!TvoX[a,k,X] * V_ooX[k,i,Y]
    @mtensor RvoXX[a,i,Y,X] -= v!TvoX[c,i,X] * V_vvX[a,c,Y]
    @mtensor v!RR[Z,Y,X] = RvoXX[a,i,Y,X] * UvoX[a,i,Z]
    drop!(buf, RvoXX)
  end
  R = alloc!(buf, nX, nX, nX)
  @mtensor R[X,Y,Z] = RR[X,Y,Z] + RR[Y,X,Z] + RR[X,Z,Y] + RR[Z,Y,X] + RR[Z,X,Y] + RR[Y,Z,X]

  save!(EC, "R_XXX", R)
  reset!(buf)
  end # buffer
end

"""
    calc_space4project_voXL(EC::ECInfo, T2)

  Calculate space for `project_voXL=true` approximation.
  
  It is a combination of spaces for triples and contravariant doubles. 
"""
function calc_space4project_voXL(EC::ECInfo, T2)
  nvirt = size(T2, 1)
  nocc = size(T2, 3)
  UvoX = load3idx(EC, "C_voX")
  if EC.options.cc.space4voXL == :triples
    println("Triples space for project_voXL")
    nbX = size(UvoX, 3) 
    UvobX = UvoX
  elseif EC.options.cc.space4voXL == :full
    println("Full space for project_voXL (not recommended, use project_voXL=false instead)")
    nbX = nvirt*nocc
    UvobX = reshape(Matrix{Float64}(I, nbX, nbX), (nvirt, nocc, nbX))
  elseif EC.options.cc.space4voXL in [:combined, :symcombined]
    @mtensor tT2[a,i,b,j] := 2.0 * T2[a,b,i,j] - T2[a,b,j,i]
    println("Combined space for project_voXL (triples + contravariant doubles space)")
    if EC.options.cc.space4voXL == :combined
      println("project the triples space from the doubles")
      # project the triples space contribution from \tilde T2
      @mtensor tT2X[X,b,j] := UvoX[a,i,X] * tT2[a,i,b,j]
      @mtensor tT2[a,i,b,j] -= UvoX[a,i,X] * tT2X[X,b,j]
      tT2X = nothing
    end
    # decompose ``\tilde T_2``
    tol2 = sqrt(EC.options.cc.ampsvdtol)
    UvoY = svd_decompose(reshape(tT2, (nocc*nvirt, nocc*nvirt)), 
                      nvirt, nocc, tol2; description="Contravariant doubles")
    # overlap of spaces
    @mtensor S_XY[X,Y] := UvoX[a,i,X] * UvoY[a,i,Y]
    nX, nY = size(S_XY)
    # full overlap
    S = Matrix{Float64}(I, nX+nY, nX+nY) 
    S[1:nX,nX+1:end] = S_XY
    S[nX+1:end,1:nX] = S_XY'
    TU_ZbX, Sigma = svd_decompose(S, tol2*tol2; description="Combined")
    # display(Sigma)
    TU_ZbX ./= sqrt.(Sigma')
    UvoZ = Array{Float64}(undef, nvirt, nocc, nX+nY)
    UvoZ[:,:,1:nX] = UvoX
    UvoZ[:,:,nX+1:end] = UvoY
    @mtensor UvobX[a,i,bX] := UvoZ[a,i,Z] * TU_ZbX[Z,bX]
  else
    error("Unknown space4voXL option: $(EC.options.cc.space4voXL)")
  end
  save!(EC, "C_vo{bX}", UvobX)
end

"""
    calc_intermediates4triples(EC::ECInfo)

  Calculate intermediates for decomposed triples independent of the amplitudes.
"""
function calc_intermediates4triples(EC::ECInfo)
  UvoX = load3idx(EC, "C_voX")
  ovLfile, ovL = mmap3idx(EC, "d_ovL")

  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nX = size(UvoX, 3)
  nL = size(ovL, 3)

  LBlks = get_spaceblocks(1:nL)

  A_XL = zeros(nX, nL)
  B_XX = zeros(nX, nX)
  w_ovX = zeros(nocc, nvirt, nX)
  B_ooXLfile, B_ooXL = newmmap(EC, "B_ooXL", (nocc,nocc,nX,nL))
  for L in LBlks
    v!ovL = @mview ovL[:,:,L]	  
    v!B_ooXL = @mview B_ooXL[:,:,:,L]
    # ``B_i^{jXL} = v_i^{aL} U^{jX}_a``
    @mtensor v!B_ooXL[i,j,X,L] = v!ovL[i,a,L] * UvoX[a,j,X]
    # ``A^{XL} = B_i^{iXL}``
    v!A_XL = @mview A_XL[:,L]
    @mtensor v!A_XL[X,L] = v!B_ooXL[i,i,X,L]
    # ``B^{XY} = B_j^{kXL} B_k^{jYL}``
    @mtensor B_XX[X,Y] += v!B_ooXL[j,k,X,L] * v!B_ooXL[k,j,Y,L]
    # ``w_k^{dX} = v_l^{dL} B_k^{lXL}``
    @mtensor w_ovX[k,d,X] += v!ovL[l,d,L] * v!B_ooXL[k,l,X,L]
  end
  closemmap(EC, B_ooXLfile, B_ooXL)
  close(ovLfile)
  save!(EC, "A_XL", A_XL)
  save!(EC, "B_XX", B_XX)
  save!(EC, "w_ovX", w_ovX)
  if EC.options.cc.project_voXL
    bUvoX = load3idx(EC, "C_vo{bX}")
    nbX = size(bUvoX, 3)
    # ``UU^{iX}_{j\bar X} = U^{iX}_a \bar U^{\dagger a}_{j\bar X}``
    UU_oXobXfile, UU_oXobX = newmmap(EC, "UU_oXo{bX}", (nocc,nX,nocc,nbX))
    bXBlks = get_spaceblocks(1:nbX)
    for bX in bXBlks
      v!bUvoX = @mview bUvoX[:,:,bX]
      v!UU_oXobX = @mview UU_oXobX[:,:,:,bX]
      @mtensor v!UU_oXobX[i,X,j,bX] = UvoX[a,i,X] * v!bUvoX[a,j,bX]
    end
    closemmap(EC, UU_oXobXfile, UU_oXobX)
    # trasformation matrix from \bar X to X space
    @mtensor C_bXX[bX,X] := bUvoX[a,i,bX] * UvoX[a,i,X]
    save!(EC, "C_{bX}X", C_bXX)
  end
end

# Function to calculate length for buffer(s) buf
# autogenerated by @print_buffer_usage
function auto_buf_length4calc_triples_residuals(EC, nvirt, nX, nbX, nocc, nL, lenL, lenX, lenY, lenZ, lenXt, lenbX, lena)
    buf = [0, 0]
    vY_ovX = pseudo_alloc!(buf, nocc, nvirt, nX)
    tT2 = pseudo_alloc!(buf, nvirt, nocc, nocc, nvirt)
    begin
        V_voL = pseudo_alloc!(buf, nvirt, nocc, lenL)
        pseudo_drop!(buf, V_voL)
    end
    pseudo_drop!(buf, tT2)
    UvY_oXoX = pseudo_alloc!(buf, nocc, nX, nocc, nX)
    pseudo_drop!(buf, UvY_oXoX)
    pseudo_drop!(buf, vY_ovX)
    RR_vovo = pseudo_alloc!(buf, nvirt, nocc, nvirt, nocc)
    R_voX = pseudo_alloc!(buf, nvirt, nocc, nX)
    R_ooX = pseudo_alloc!(buf, nocc, nocc, nX)
    G_ooX = pseudo_alloc!(buf, nocc, nocc, nX)
    w_ovX = pseudo_alloc!(buf, nocc, nvirt, nX)
    if EC.options.cc.project_voXL
        RR_vobX = pseudo_alloc!(buf, nvirt, nocc, nbX)
        C_bXX = pseudo_alloc!(buf, nbX, nX)
    end
    fU_X = pseudo_alloc!(buf, nX)
    begin
        TvoXX = pseudo_alloc!(buf, nvirt, nocc, nX, lenX)
        begin
            TUvY_voXX = pseudo_alloc!(buf, nvirt, nocc, lenY, lenX)
            a!Q_XXX = pseudo_alloc!(buf, nX, lenY, lenX)
            pseudo_drop!(buf, a!Q_XXX)
            pseudo_drop!(buf, TUvY_voXX)
        end
        Bv_vvX = pseudo_alloc!(buf, nvirt, nvirt, lenX)
        Bv_ooX = pseudo_alloc!(buf, nocc, nocc, lenX)
        if EC.options.cc.project_voXL
            bV_XbXL = pseudo_alloc!(buf, lenX, nbX, nL)
            begin
                TUU_XbXov = pseudo_alloc!(buf, lenX, lenbX, nocc, nvirt)
                pseudo_drop!(buf, TUU_XbXov)
            end
        end
        begin
            bB_voXL = pseudo_alloc!(buf, nvirt, nocc, lenX, lenL)
            B_voXL = pseudo_alloc!(buf, nvirt, nocc, lenX, lenL)
            pseudo_drop!(buf, B_voXL)
            a!W_XXL = pseudo_alloc!(buf, nX, lenX, lenL)
            pseudo_drop!(buf, a!W_XXL)
            V_XXL = pseudo_alloc!(buf, nX, lenX, lenL)
            if EC.options.cc.project_voXL
                nothing
            else
                V_voXL = pseudo_alloc!(buf, nvirt, nocc, lenX, lenL)
                pseudo_drop!(buf, V_voXL)
            end
            pseudo_drop!(buf, V_XXL)
            pseudo_drop!(buf, bB_voXL)
        end
        if EC.options.cc.project_voXL
            pseudo_drop!(buf, bV_XbXL)
        end
        UBv = pseudo_alloc!(buf, nvirt, nocc, nX, lenX)
        pseudo_drop!(buf, UBv)
        pseudo_drop!(buf, Bv_ooX, Bv_vvX)
        pseudo_drop!(buf, TvoXX)
    end
    pseudo_drop!(buf, fU_X)
    if EC.options.cc.project_voXL
        pseudo_drop!(buf, C_bXX)
        bUvoX = pseudo_alloc!(buf, nvirt, nocc, nbX)
        pseudo_drop!(buf, bUvoX, RR_vobX)
    end
    pseudo_drop!(buf, w_ovX)
    pseudo_drop!(buf, G_ooX, R_ooX)
    pseudo_drop!(buf, R_voX)
    pseudo_drop!(buf, RR_vovo)
    A_XL = pseudo_alloc!(buf, nX, nL)
    B_XX = pseudo_alloc!(buf, nX, nX)
    pseudo_drop!(buf, A_XL, B_XX)
    begin
        vV_XXov = pseudo_alloc!(buf, nX, lenX, nocc, nvirt)
        pseudo_drop!(buf, vV_XXov)
    end
    oovo = pseudo_alloc!(buf, nocc, nocc, nvirt, nocc)
    begin
        vvov = pseudo_alloc!(buf, lena, nvirt, nocc, nvirt)
        a!UvoX = pseudo_alloc!(buf, lena, nocc, nX)
        a!T2 = pseudo_alloc!(buf, nvirt, lena, nocc, nocc)
        vT_vooo = pseudo_alloc!(buf, lena, nocc, nocc, nocc)
        pseudo_drop!(buf, vT_vooo)
        vT_vvvo = pseudo_alloc!(buf, lena, nvirt, nvirt, nocc)
        pseudo_drop!(buf, vT_vvvo)
        pseudo_drop!(buf, a!T2, a!UvoX)
        pseudo_drop!(buf, vvov)
    end
    pseudo_drop!(buf, oovo)
    begin
        XT_voXX = pseudo_alloc!(buf, nvirt, nocc, nX, lenZ)
        pseudo_drop!(buf, XT_voXX)
    end
    W_LXX = pseudo_alloc!(buf, nL, nX, nX)
    begin
        W_XXX = pseudo_alloc!(buf, nX, nX, lenXt)
        qq_XX = pseudo_alloc!(buf, nX, lenXt)
        pseudo_drop!(buf, qq_XX, W_XXX)
    end
    pseudo_drop!(buf, W_LXX)
    q_voX = pseudo_alloc!(buf, nvirt, nocc, nX)
    vvoo = pseudo_alloc!(buf, nvirt, nvirt, nocc, nocc)
    pseudo_drop!(buf, vvoo)
    pseudo_drop!(buf, q_voX)
    A_XL = pseudo_alloc!(buf, nX, nL)
    pseudo_drop!(buf, A_XL)
    return buf[2]
end
"""
    calc_triples_residuals!(EC::ECInfo, R1, R2, T2)

  Calculate decomposed triples DC-CCSDT residuals.
"""
function calc_triples_residuals!(EC::ECInfo, R1, R2, T2)
  t1 = time_ns()
  mem1 = free_memory()
  UvoX = load3idx(EC, "C_voX")
  #display(UvoX)

  #load decomposed amplitudes
  T_XXX = load3idx(EC, "T_XXX")
  #display(T_XXX)

  #load df coeff
  ovLfile, ovL = mmap3idx(EC, "d_ovL")
  voLfile, voL = mmap3idx(EC, "d_voL")
  ooLfile, ooL = mmap3idx(EC, "d_ooL")
  vvLfile, vvL = mmap3idx(EC, "d_vvL")

  if EC.options.cc.project_voXL
    UU_oXobXfile, UU_oXobX = mmap4idx(EC, "UU_oXo{bX}")
    nbX = size(UU_oXobX, 4)
  else
    nbX = size(T_XXX, 1)
  end

  #load dressed fock matrices
  SP = EC.space
  dfock = load2idx(EC, "df_mm")    
  dfoo = dfock[SP['o'], SP['o']]
  dfov = dfock[SP['o'], SP['v']]
  dfvv = dfock[SP['v'], SP['v']]
 
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  nX = size(T_XXX, 1)
  nL = size(voL, 3)

  LBlks = get_spaceblocks(1:nL)
  XBlks = get_spaceblocks(1:nX)
  XBigBlks = get_spaceblocks(1:nX, 256)
  virtBlks = get_spaceblocks(1:nvirt)
  bXBlks = get_spaceblocks(1:nbX)

  maxL = maximum(length, LBlks)
  maxX = maximum(length, XBlks)
  maxbX = maximum(length, bXBlks)
  maxXBig = maximum(length, XBigBlks)
  maxa = maximum(length, virtBlks)
  lenbuf = auto_buf_length4calc_triples_residuals(EC, nvirt, nX, nbX, nocc, nL, maxL, maxXBig, maxX, maxX, nX, maxbX, maxa)
  @buffer buf(lenbuf) begin
  mem2 = print_memory(EC, mem1, "for buffer in SVD-DC-CCSDT triples residuals calculation", 2)
  # @print_buffer_usage buf begin

  Y_XL = zeros(nX, nL)
  x_oo = dfoo
  x_vv = dfvv
  # ``vY_{lX}^{d} = v_{l}^{dL} Y_{X}^{L}``
  vY_ovX = alloc!(buf, nocc, nvirt, nX)
  vY_ovX .= 0.0
  tT2 = alloc!(buf, nvirt, nocc, nocc, nvirt)
  @mtensor tT2[a,i,j,b] = 2.0 * T2[a,b,i,j] - T2[b,a,i,j]
  for L in LBlks
    lenL = length(L)
    v!ovL = @mview ovL[:,:,L]
    v!voL = @mview voL[:,:,L]
    v!Y_XL = @mview Y_XL[:,L]
    V_voL = alloc!(buf, nvirt, nocc, lenL)
    # ``V_{a}^{iL} = v_k^{cL} (2T^{ik}_{ac}- T^{ik}_{ca})``
    @mtensor V_voL[a,i,L] = tT2[a,i,k,c] * v!ovL[k,c,L]
    # ``x_l^i = \hat f_l^i + 0.5 V_{d}^{iL} v_{l}^{dL}``
    @mtensor x_oo[l,i] += 0.5 * v!ovL[l,d,L] * V_voL[d,i,L]
    # ``x_a^d = \hat f_a^d - 0.5 V_{a}^{lL} v_{l}^{dL}``
    @mtensor x_vv[a,d] -= 0.5 * v!ovL[l,d,L] * V_voL[a,l,L]
    # ``Y_{X}^{L} = U^{†a}_{iX} (\hat v_{a}^{iL} + V_{a}^{iL})``
    @mtensor V_voL[a,i,L] += v!voL[a,i,L]
    @mtensor v!Y_XL[X,L] = UvoX[a,i,X] * V_voL[a,i,L]
    drop!(buf, V_voL)
    # ``vY_{lX}^{d} = v_{l}^{dL} Y_{X}^{L}``
    @mtensor vY_ovX[l,d,X] += v!ovL[l,d,L] * v!Y_XL[X,L]
  end
  drop!(buf, tT2)
  # UvY[lYjX] = ``UvY_{lX}^{jY} = U^{jY}_d vY_{lX}^{d}``
  UvY_oXoX = alloc!(buf, nocc, nX, nocc, nX)
  @mtensor UvY_oXoX[l,Y,j,X] = UvoX[d,j,Y] * vY_ovX[l,d,X]
  save!(EC, "UvY_oXoX", UvY_oXoX)
  drop!(buf, UvY_oXoX)
  drop!(buf, vY_ovX)

  # ``T_{aX}^i = U^{†b}_{jX} T^{ij}_{ab}``
  @mtensor T_voX[a,i,X] := UvoX[b,j,X] * T2[a,b,i,j]
  # ``X_{bZ}^d = - \hat f_l^d T_{bZ}^{l} (+...)``
  @mtensor X_vvX[b,d,Z] := - dfov[l,d] * T_voX[b,l,Z]
  X_ooX = zeros(nocc, nocc, nX)
  for L in LBlks
    v!vvL = @mview vvL[:,:,L]
    v!ooL = @mview ooL[:,:,L]
    v!Y_XL = @mview Y_XL[:,L]
    # ``X_{bZ}^d += \hat v_b^{dL} Y_Z^{L}``
    @mtensor X_vvX[b,d,Z] += v!vvL[b,d,L] * v!Y_XL[Z,L]
    # ``X_{lZ}^j += \hat v_l^{jL} Y_Z^{L}``
    @mtensor X_ooX[l,j,Z] += v!ooL[l,j,L] * v!Y_XL[Z,L] 
  end
 
  Q_XXX = zeros(nX, nX, nX)
  # RR[aibj] = ``RR^{ij}_{ab}``
  RR_vovo = alloc!(buf, nvirt, nocc, nvirt, nocc)
  RR_vovo .= 0.0
  R_voX = alloc!(buf, nvirt, nocc, nX)
  R_voX .= 0.0
  R_ooX = alloc!(buf,nocc,nocc,nX)
  R_ooX .= 0.0
  G_ooX = alloc!(buf, nocc, nocc, nX)
  # ``W_{X}^{YL} = (\bar B - B)_b^{jYL} U^{\dagger b}_{jX}
  W_XXLfile, W_XXL = newmmap(EC, "W_XXL", (nX, nX, nL))
  # ``\tilde V_{YX}^{L} = T_{ZYX} A^{ZL} - 0.5 V_{aX}^{jL} U^{\dagger a}_{jY}``
  tV_XXLfile, tV_XXL = newmmap(EC, "tV_XXL", (nX, nX, nL))
  B_ooXLfile, B_ooXL = mmap4idx(EC, "B_ooXL")
  A_XLfile, A_XL = mmap2idx(EC, "A_XL")
  UvY_oXoXfile, UvY_oXoX = mmap4idx(EC, "UvY_oXoX")
  w_ovX = alloc!(buf, nocc, nvirt, nX)
  load!(EC, "w_ovX", w_ovX)
  if EC.options.cc.project_voXL
    RR_vobX = alloc!(buf, nvirt, nocc, nbX)
    RR_vobX .= 0.0
    C_bXX = alloc!(buf, nbX, nX)
    load!(EC, "C_{bX}X", C_bXX)
  end
  # ``fU^X = \hat f_k^c U^{kX}_c``
  fU_X = alloc!(buf, nX)
  @mtensor fU_X[X] = dfov[k,c] * UvoX[c,k,X]
  for X in XBigBlks
    lenX = length(X)
    v!T_XXX = @mview T_XXX[:,:,X]
    v!UvoX = @mview UvoX[:,:,X]
    # ``T^i_{aYX} = U^{iZ}_a T_{ZYX}``
    TvoXX = alloc!(buf, nvirt, nocc, nX, lenX)
    @mtensor TvoXX[a,i,Y,X] = UvoX[a,i,Z] * v!T_XXX[Z,Y,X]
    # ``R_{aY}^i += T_{aYX}^i fU^X``
    v!fU_X = @mview fU_X[X]
    @mtensor R_voX[a,i,Y] += TvoXX[a,i,Y,X] * v!fU_X[X]
    # ``X_{bZ}^d += 0.5 T^k_{bYX} w_{k}^{dY}``
    v!X_vvX = @mview X_vvX[:,:,X]
    @mtensor v!X_vvX[b,d,X] += 0.5 * TvoXX[b,k,Y,X] * w_ovX[k,d,Y]
    # ``G_{jX}^i = T_{dYX}^i w_j^{dY}``
    v!G_ooX = @mview G_ooX[:,:,X]
    @mtensor v!G_ooX[j,i,X] = TvoXX[d,i,Y,X] * w_ovX[j,d,Y]
    for Y in XBlks
      lenY = length(Y)
      v!UvY_oXoX = @mview UvY_oXoX[:,:,:,Y]
      # ``TUvY^j_{bYX} = T_{bZX}^l UvY_{Yl}^{jZ}``
      TUvY_voXX = alloc!(buf, nvirt, nocc, lenY, lenX)
      @mtensor TUvY_voXX[b,j,Y,X] = TvoXX[b,l,Z,X] * v!UvY_oXoX[l,Z,j,Y]
      # ``Q_{ZYX} = U^{\dagger b}_{jZ} TUvY^j_{bYX}``
      a!Q_XXX = alloc!(buf, nX, lenY, lenX)
      @mtensor a!Q_XXX[Z,Y,X] = UvoX[b,j,Z] * TUvY_voXX[b,j,Y,X]
      Q_XXX[:,Y,X] = a!Q_XXX
      drop!(buf, a!Q_XXX)
      drop!(buf, TUvY_voXX)
    end
    Bv_vvX = alloc!(buf, nvirt, nvirt, lenX)
    Bv_vvX .= 0.0
    Bv_ooX = alloc!(buf, nocc, nocc, lenX)
    Bv_ooX .= 0.0
    if EC.options.cc.project_voXL
      bV_XbXL = alloc!(buf, lenX, nbX, nL)
      for bX in bXBlks
        lenbX = length(bX)
        v!UU_oXobX = @mview UU_oXobX[:,:,:,bX]
        # ``TUU^j_{b\bar XX} = T_{bZX}^l UU_{\bar Xl}^{jZ}``
        TUU_XbXov = alloc!(buf, lenX, lenbX, nocc, nvirt)
        @mtensor TUU_XbXov[X,bX,j,b] = TvoXX[b,l,Z,X] * v!UU_oXobX[j,Z,l,bX]
        # ``\bar V_{\bar XX}^{L} = TUU^j_{b\bar XX} v_j^{bL}``
        v!bV_XbXL = @mview bV_XbXL[:,bX,:]
        @mtensor v!bV_XbXL[X,bX,L] = TUU_XbXov[X,bX,j,b] * ovL[j,b,L]
        drop!(buf, TUU_XbXov)
      end
    end
    v!R_ooX = @mview R_ooX[:,:,X]
    for L in LBlks
      lenL = length(L)
      v!ooL = @mview ooL[:,:,L]
      v!vvL = @mview vvL[:,:,L]
      v!ovL = @mview ovL[:,:,L]
      # ``\bar B_a^{iXL} = \hat v_a^{bL} U^{iX}_b``
      bB_voXL = alloc!(buf, nvirt, nocc, lenX, lenL)
      @mtensor bB_voXL[a,i,X,L] = v!vvL[a,b,L] * v!UvoX[b,i,X]
      # ``B_a^{iXL} = \hat v_j^{iL} U^{jX}_a``
      B_voXL = alloc!(buf, nvirt, nocc, lenX, lenL)
      @mtensor B_voXL[a,i,X,L] = v!ooL[j,i,L] * v!UvoX[a,j,X]
      # ``Bv_a^{cX} = \bar B_a^{kXL} v_k^{cL}``
      @mtensor Bv_vvX[a,c,X] += bB_voXL[a,k,X,L] * v!ovL[k,c,L]
      # ``Bv_k^{iX} = B_c^{iXL} v_k^{cL}``
      @mtensor Bv_ooX[k,i,X] += B_voXL[c,i,X,L] * v!ovL[k,c,L]
      # ``\bar B_a^{iXL} -= B_a^{iXL}``
      @mtensor bB_voXL[a,i,X,L] -= B_voXL[a,i,X,L]
      drop!(buf, B_voXL)
      # ``W_{Y}^{XL} = (\bar B - B)_b^{jXL} U^{\dagger b}_{jY}
      a!W_XXL = alloc!(buf, nX, lenX, lenL)
      @mtensor a!W_XXL[Y,X,L] = bB_voXL[b,j,X,L] * UvoX[b,j,Y]
      W_XXL[:,X,L] = a!W_XXL
      drop!(buf, a!W_XXL)

      v!B_ooXL = @mview B_ooXL[:,:,:,L]
      v!A_XL = @mview A_XL[:,L]
      # ``V_{YX}^{L} = T_{ZYX} A^{ZL}``
      V_XXL = alloc!(buf, nX, lenX, lenL)
      @mtensor V_XXL[Y,X,L] = v!T_XXX[Z,Y,X] * v!A_XL[Z,L]
      # ``R_{jX}^{i} = 2 B_j^{iYL} V_{YX}^{L}``
      @mtensor v!R_ooX[j,i,X] += 2.0 * v!B_ooXL[j,i,Y,L] * V_XXL[Y,X,L]
      # ``R_{aZ}^i = 2 (\bar B - B)_a^{LiX} V_{ZX}^{L}``
      @mtensor R_voX[a,i,Z] += 2.0 * bB_voXL[a,i,X,L] * V_XXL[Z,X,L]
      if EC.options.cc.project_voXL
        v!bV_XbXL = @mview bV_XbXL[:,:,L]
        # ``\tilde V_{YX}^{L} -= 0.5 \bar V_{X\bar X}^{L} C^{\bar X}_{Y}``
        @mtensor V_XXL[Y,X,L] -= 0.5 * v!bV_XbXL[X,bX,L] * C_bXX[bX,Y]
        # ``RR^{i}_{a\bar X} -= (\bar B - B)_a^{LiX} \bar V_{X\bar X}^{L}``
        @mtensor RR_vobX[a,i,bX] -= bB_voXL[a,i,X,L] * v!bV_XbXL[X,bX,L]
      else
        # V[ajXL]=``V_{aX}^{jL} = T^i_{aYX} B_i^{jYL}``
        V_voXL = alloc!(buf, nvirt, nocc, lenX, lenL)
        @mtensor V_voXL[a,j,X,L] = TvoXX[a,i,Y,X] * v!B_ooXL[i,j,Y,L]
        # ``\tilde V_{YX}^{L} -= 0.5 V_{aX}^{jL} U^{\dagger a}_{jY}``
        @mtensor V_XXL[Y,X,L] -= 0.5 * V_voXL[a,j,X,L] * UvoX[a,j,Y]
        # ``RR^{ij}_{ab} -= (\bar B - B)_a^{LiX} V_{bX}^{jL}``
        @mtensor RR_vovo[a,i,b,j] -= bB_voXL[a,i,X,L] * V_voXL[b,j,X,L]
        drop!(buf, V_voXL)
      end
      tV_XXL[:,X,L] = V_XXL
      drop!(buf, V_XXL)
      drop!(buf, bB_voXL)
    end
    if EC.options.cc.project_voXL
      drop!(buf, bV_XbXL)
    end
    # ``Bv_k^{iX} -= \hat f_k^c U^{iX}_c``
    @mtensor Bv_ooX[k,i,X] -= dfov[k,c] * v!UvoX[c,i,X]
    # ``UBv_a^{iYX} = U^{iY}_c Bv_a^{cX}``
    UBv = alloc!(buf, nvirt, nocc, nX, lenX)
    @mtensor UBv[a,i,Y,X] = UvoX[c,i,Y] * Bv_vvX[a,c,X]
    # ``UBv_a^{iYX} -= U^{kY}_a Bv_k^{iX}``
    @mtensor UBv[a,i,Y,X] -= UvoX[a,k,Y] * Bv_ooX[k,i,X]
    # ``R_{aZ}^i -= T_{ZYX} UBv_a^{iYX}``
    @mtensor R_voX[a,i,Z] -= v!T_XXX[Z,Y,X] * UBv[a,i,Y,X]
    drop!(buf, UBv)
    drop!(buf, Bv_ooX, Bv_vvX)
    drop!(buf, TvoXX)
  end
  drop!(buf, fU_X)
  if EC.options.cc.project_voXL
    drop!(buf, C_bXX)
    close(UU_oXobXfile)
    bUvoX = alloc!(buf, nvirt, nocc, nbX)
    load!(EC, "C_vo{bX}", bUvoX)
    @mtensor RR_vovo[a,i,b,j] += RR_vobX[a,i,bX] * bUvoX[b,j,bX]
    drop!(buf, bUvoX, RR_vobX)
  end
  drop!(buf, w_ovX)
  close(UvY_oXoXfile)
  close(A_XLfile)
  close(B_ooXLfile)
  closemmap(EC, W_XXLfile, W_XXL)
  closemmap(EC, tV_XXLfile, tV_XXL)
  # ``R_{jZ}^{i} -= G_{jZ}^{i}``
  @mtensor R_ooX[j,i,Z] -= G_ooX[j,i,Z]
  # ``R^i_a -= R_{jY}^{i} U^{jY}_a``
  @mtensor R1[a,i] -= R_ooX[j,i,Y] * UvoX[a,j,Y]
  # ``X_{jZ}^i -= 0.5 G_{jZ}^{i}``
  @mtensor X_ooX[j,i,Z] -= 0.5 * G_ooX[j,i,Z]
  drop!(buf, G_ooX, R_ooX)
  # ``RR^{ij}_{ab} += R_{aZ}^i U^{jZ}_b``
  @mtensor RR_vovo[a,i,b,j] += R_voX[a,i,Z] * UvoX[b,j,Z]
  drop!(buf, R_voX)
  # ``R^{ij}_{ab} = RR^{ij}_{ab} + RR^{ji}_{ba}``
  @mtensor R2[a,b,i,j] += RR_vovo[a,i,b,j] + RR_vovo[b,j,a,i]
  drop!(buf, RR_vovo)

  A_XL = alloc!(buf, nX, nL)
  load!(EC, "A_XL", A_XL)
  B_XX = alloc!(buf, nX, nX)
  load!(EC, "B_XX", B_XX)
  # ``B^{XY} -= 2 A^{XL} A^{YL}``
  @mtensor B_XX[X,Y] -= 2.0 * A_XL[X,L] * A_XL[Y,L]
  # ``R^i_a -= U^{iX}_a (T_{XYZ} B^{YZ})
  @mtensor R1[a,i] -= UvoX[a,i,X] * (T_XXX[X,Y,Z] * B_XX[Y,Z])
  drop!(buf, A_XL, B_XX)

  tV_XXLfile, tV_XXL = mmap3idx(EC, "tV_XXL")
  for X in XBigBlks
    lenX = length(X)
    # ``vV_{YXl}^d = v_{l}^{dL} \tilde V_{YX}^{L}``
    vV_XXov = alloc!(buf, nX, lenX, nocc, nvirt)
    vV_XXov .= 0.0
    for L in LBlks
      v!ovL = @mview ovL[:,:,L]
      v!tV_XXL = @mview tV_XXL[:,X,L]
      @mtensor vV_XXov[Y,X,l,d] += v!ovL[l,d,L] * v!tV_XXL[Y,X,L]
    end
    # ``X_{lY}^j += vV_{YXl}^d U^{jX}_d``
    v!UvoX = @mview UvoX[:,:,X]
    @mtensor X_ooX[l,j,Y] += vV_XXov[Y,X,l,d] * v!UvoX[d,j,X]
    # ``X_{bY}^d -= vV_{YXl}^d U^{lX}_b``
    @mtensor X_vvX[b,d,Y] -= vV_XXov[Y,X,l,d] * v!UvoX[b,l,X]
    drop!(buf, vV_XXov)
  end
  close(tV_XXLfile)

  # ``\hat v_{lk}^{di} = \hat v_{l}^{dL} \hat v_{k}^{iL}``
  oovo = alloc!(buf, nocc, nocc, nvirt, nocc)
  @mtensor oovo[l,k,d,i] = ovL[l,d,L] * ooL[k,i,L]
  for a in virtBlks
    lena = length(a)
    v!vvL = @mview vvL[a,:,:]
    # vvov[acld] = ``\hat v_{al}^{cd} = \hat v_{a}^{cL} v_{l}^{dL}``
    vvov = alloc!(buf, lena, nvirt, nocc, nvirt)
    @mtensor vvov[a,c,l,d] = v!vvL[a,c,L] * ovL[l,d,L]
    a!UvoX = alloc!(buf, lena, nocc, nX)
    a!UvoX .= @mview UvoX[a,:,:]
    a!T2 = alloc!(buf, nvirt, lena, nocc, nocc)
    a!T2 .= @mview T2[:,a,:,:]
    # ``vT_{al}^{ij} = \hat v_{al}^{cd} T_{cd}^{ij} - \hat v_{lk}^{di} T_{da}^{jk}``
    vT_vooo = alloc!(buf, lena, nocc, nocc, nocc)
    @mtensor vT_vooo[a,l,i,j] = vvov[a,c,l,d] * T2[c,d,i,j]
    @mtensor vT_vooo[a,l,i,j] -= oovo[l,k,d,i] * a!T2[d,a,j,k]
    # ``X_{lY}^j += vT_{al}^{ij} U^{\dagger a}_{iY}``
    @mtensor X_ooX[l,j,Y] += vT_vooo[a,l,i,j] * a!UvoX[a,i,Y]
    drop!(buf, vT_vooo)
    # vT[adbi] = ``vT_{ab}^{di} = \hat v_{lk}^{di} T_{ba}^{lk} - \hat v_{al}^{cd} T_{bc}^{li}``
    vT_vvvo = alloc!(buf, lena, nvirt, nvirt, nocc)
    @mtensor vT_vvvo[a,d,b,i] = oovo[l,k,d,i] * a!T2[b,a,l,k]
    @mtensor vT_vvvo[a,d,b,i] -= vvov[a,c,l,d] * T2[b,c,l,i]
    # ``X_{bY}^d += vT_{ab}^{di} U^{\dagger a}_{iY}``
    @mtensor X_vvX[b,d,Y] += vT_vvvo[a,d,b,i] * a!UvoX[a,i,Y]
    drop!(buf, vT_vvvo)
    drop!(buf, a!T2, a!UvoX)
    # ``X_{aY}^d -= \hat v_{al}^{cd} T_{cY}^{l}``
    v!X_vvX = @mview X_vvX[a,:,:]
    @mtensor v!X_vvX[a,d,Y] -= vvov[a,c,l,d] * T_voX[c,l,Y]
    drop!(buf, vvov)
  end
  # ``X_{lY}^j -= \hat v_{lk}^{dj} T_{dY}^{k}``
  @mtensor X_ooX[l,j,Y] -= oovo[l,k,d,j] * T_voX[d,k,Y]
  drop!(buf, oovo)

  # ``Q_{XYZ} += U^{\dagger b}_{jX} (X_{lY}^j T_{bZ}^l - X_{bY}^d T_{dZ}^j)``
  for Z in XBlks
    lenZ = length(Z)
    v!T_voX = @mview T_voX[:,:,Z]
    v!Q_XXX = @mview Q_XXX[:,:,Z]
    XT_voXX = alloc!(buf, nvirt, nocc, nX, lenZ)
    @mtensor XT_voXX[b,j,Y,Z] = X_ooX[l,j,Y] * v!T_voX[b,l,Z]
    @mtensor XT_voXX[b,j,Y,Z] -= X_vvX[b,d,Y] * v!T_voX[d,j,Z]
    @mtensor v!Q_XXX[X,Y,Z] += UvoX[b,j,X] * XT_voXX[b,j,Y,Z]
    drop!(buf, XT_voXX)
  end
  # ``R_{XYZ} += Q_{XYZ} + Q_{YXZ} + Q_{XZY} + Q_{ZXY} + Q_{ZYX} + Q_{YZX}``
  @mtensor R3decomp[X,Y,Z] := Q_XXX[X,Y,Z] + Q_XXX[Y,X,Z] + Q_XXX[X,Z,Y] + Q_XXX[Z,Y,X] + Q_XXX[Z,X,Y] + Q_XXX[Y,Z,X]

  # reuse memory
  q_XXX = Q_XXX
  q_XXX .= 0.0
  # reorder W_{X}^{YL} for a triangular contraction
  W_LXX = alloc!(buf, nL, nX, nX)
  W_XXLfile, W_XXL = mmap3idx(EC, "W_XXL")
  for L in LBlks
    v!W_XXL = @mview W_XXL[:,:,L]
    v!W_LXX = @mview W_LXX[L,:,:]
    permutedims!(v!W_LXX, v!W_XXL, (3,2,1))
  end
  close(W_XXLfile)
  for iY in 1:nX
    v!W_LX = @mview W_LXX[:,:,iY]
    X = 1:iY # only upper triangular part
    lenXt = length(X)
    v!W_LXX = @mview W_LXX[:,:,X]
    W_XXX = alloc!(buf, nX, nX, lenXt)
    @mtensor W_XXX[Y',X',X] = v!W_LXX[L,X',X] * v!W_LX[L,Y']
    # ``qq_{ZXY} += T_{ZX'Y'} W^{X'Y'}_{XY}``
    qq_XX = alloc!(buf, nX, lenXt)
    @mtensor qq_XX[Z,X] = T_XXX[Z,X',Y'] * W_XXX[X',Y',X]
    q_XXX[:,X,iY] = -qq_XX
    q_XXX[:,iY,X] = -qq_XX
    drop!(buf, qq_XX, W_XXX)
  end  
  drop!(buf, W_LXX)
  # ``q_{X}^{X'} = U^{\dagger a}_{iX} q_{a}^{iX'} - 2 Y_{X}^L A^{X'L}``
  # with ``q_{a}^{iX'} = x_{l}^{i} U_{a}^{lX'} - x_{a}^{d} U^{d}_{iX'} + (\hat v_{l}^{iL} \hat v_{a}^{dL}) U^{d}_{lX'}``
  q_voX = alloc!(buf, nvirt, nocc, nX)
  # vvoo[adli] = ``v_{al}^{di} = \hat v_{a}^{dL} \hat v_{l}^{iL}``
  vvoo = alloc!(buf, nvirt, nvirt, nocc, nocc)
  vvoo .= 0.0
  for L in LBlks
    v!vvL = @mview vvL[:,:,L]
    v!ooL = @mview ooL[:,:,L]
    @mtensor vvoo[a,d,l,i] += v!vvL[a,d,L] * v!ooL[l,i,L]
  end
  @mtensor q_voX[a,i,X] = vvoo[a,d,l,i] * UvoX[d,l,X]
  drop!(buf, vvoo)
  @mtensor q_voX[a,i,X] += x_oo[l,i] * UvoX[a,l,X] - x_vv[a,d] * UvoX[d,i,X]
  @mtensor q_XX[X,X'] := q_voX[a,i,X'] * UvoX[a,i,X]
  drop!(buf, q_voX)
  A_XL = alloc!(buf, nX, nL)
  load!(EC, "A_XL", A_XL)
  @mtensor q_XX[X,X'] -= 2.0 * Y_XL[X,L] * A_XL[X',L]
  drop!(buf, A_XL)
  # ``q_{XYZ} = T_{X'YZ} q_{X}^{X'}``
  @mtensor q_XXX[X,Y,Z] += T_XXX[X',Y,Z] * q_XX[X,X']

  @mtensor R3decomp[X,Y,Z] += q_XXX[X,Y,Z] + q_XXX[Y,X,Z] + q_XXX[Z,Y,X]

  close(ovLfile)
  close(voLfile)
  close(ooLfile)
  close(vvLfile)
  save!(EC, "R_XXX", R3decomp)
  end #buffer
end

end #module
