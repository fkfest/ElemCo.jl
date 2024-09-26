# This file contains the functions for the Lagrange multiplier method for the
# coupled cluster equations.
"""
    calc_ccsd_vector_times_Jacobian(EC::ECInfo, U1, U2; dc=false)

Calculate the vector times the Jacobian for the closed-shell CCSD or DCSD
equations.
"""
function calc_ccsd_vector_times_Jacobian(EC::ECInfo, U1, U2; dc=false)
  t1 = time_ns()
  SP = EC.space
  nocc = n_occ_orbs(EC)
  norb = n_orbs(EC)
  
  T1 = load2idx(EC, "T_vo")
  T2 = load4idx(EC, "T_vvoo")
  # Calculate 1RDM intermediates
  D1, dD1 = calc_1RDM(EC, U1, U2, T1, T2)
  t1 = print_time(EC, t1, "calculate 1RDM",2)

  fock = load2idx(EC,"f_mm")
  dfock = load2idx(EC,"df_mm")
  fov = fock[SP['o'],SP['v']]
  dfov = dfock[SP['o'],SP['v']]
  if length(U1) > 0
    @tensoropt R1[e,m] := 2.0 * fov[m,e]
  else
    R1 = U1
  end

  oovv = ints2(EC,"oovv")
  @tensoropt R2[e,f,m,n] := 2.0 * oovv[m,n,e,f] - oovv[n,m,e,f]
  int2 = load4idx(EC, "d_oooo")
  if !dc
    @tensoropt int2[m,n,i,j] += oovv[m,n,c,d] * T2[c,d,i,j]
  end
  # ``R^{ef}_{mn} += Λ_{ij}^{ef} (\hat v_{mn}^{ij} \red{+ v_{nm}^{cd} T^{ij}_{cd}})``
  @tensoropt R2[e,f,m,n] += int2[m,n,i,j] * U2[e,f,i,j]
  t1 = print_time(EC, t1, "R^{ef}_{mn} += U_{ij}^{ef} (\\hat v_{mn}^{ij} + v_{nm}^{cd} T^{ij}_{cd})",2)
  # the 4-external part
  if EC.options.cc.use_kext
    # the `kext` part
    int2 = integ2_ss(EC.fd)
    # last two indices of integrals are stored as upper triangular 
    dU2 = calc_dU2(EC, T1, T1, U2)
    # ``K_{mn}^{rs} = \hat U_{mn}^{pq} v_{pq}^{rs}``
    @tensoropt Kxoo[x,m,n] := int2[p,q,x] * dU2[p,q,m,n]
    dU2 = nothing
    Kmmoo = Array{Float64}(undef,norb,norb,nocc,nocc)
    tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
    Kmmoo[tripp,:,:] = Kxoo
    trippr = CartesianIndex.(reverse.(Tuple.(tripp)))
    @tensor Kmmoo[trippr,:,:][x,m,n] = Kxoo[x,n,m]
    Kxoo = nothing
    t1 = print_time(EC, t1, "K_{mn}^{rs} = \\hat U_{mn}^{pq} v_{pq}^{rs}",2)
    # ``R^{ef}_{mn} += K_{mn}^{rs} δ_r^e δ_s^f``
    R2 += Kmmoo[SP['v'],SP['v'],:,:]
    # ``R^e_m += 2 K_{mj}^{rs} δ_r^e (δ_s^j + δ_s^b T^j_b)``
    if length(U1) > 0
      @tensoropt R1[e,m] += 2.0 * Kmmoo[SP['v'],SP['o'],:,:][e,j,m,j] 
      if length(T1) > 0
        @tensoropt R1[e,m] += 2.0 * Kmmoo[SP['v'],SP['v'],:,:][e,b,m,j] * T1[b,j]
      end 
      t1 = print_time(EC, t1, "R^e_m += 2 K_{mj}^{rs} δ_r^e (δ_s^j + δ_s^b T^j_b)",2)
    end
    Kmmoo = nothing
  else
    error("non-kext Λ equations not implemented")
  end

  fac = dc ? 0.25 : 0.5
  # ``tR^{ef}_{mn} += f * D_m^k v_{kn}^{ef} - f * D_c^e v_{mn}^{cf}``
  @tensoropt begin 
    tR2[e,f,m,n] := fac * D1[SP['o'],SP['o']][m,k] * oovv[k,n,e,f]
    tR2[e,f,m,n] -= fac * D1[SP['v'],SP['v']][c,e] * oovv[m,n,c,f]
  end
  t1 = print_time(EC, t1, "tR^{ef}_{mn} += f * D_m^k v_{kn}^{ef} - f * D_c^e v_{mn}^{cf}",2)
  @tensoropt tT2[e,f,m,n] := 2.0 * T2[e,f,m,n] - T2[e,f,n,m]
  # ``x_m^i = \tilde T^{il}_{cd} v_{ml}^{cd}``
  @tensoropt xoo[m,i] := tT2[c,d,i,l] * oovv[m,l,c,d]
  t1 = print_time(EC, t1, "x_m^i = \\tilde T^{il}_{cd} v_{ml}^{cd}",2)
  # ``x_a^e = \tilde T^{kl}_{ac} v_{kl}^{ec}``
  @tensoropt xvv[a,e] := tT2[a,c,k,l] * oovv[k,l,e,c]
  t1 = print_time(EC, t1, "x_a^e = \\tilde T^{kl}_{ac} v_{kl}^{ec}",2)
  int2 = load4idx(EC, "d_voov")
  # ``tR^{ef}_{mn} += Λ_{in}^{af} (\hat v_{am}^{ie} + v_{km}^{ce} \tilde T^{ik}_{ac})``
  @tensoropt int2[a,m,i,e] += oovv[k,m,c,e] * tT2[a,c,i,k]
  @tensoropt tR2[e,f,m,n] += U2[a,f,i,n] * int2[a,m,i,e] 
  t1 = print_time(EC, t1, "tR^{ef}_{mn} += U_{in}^{af} (\\hat v_{am}^{ie} + v_{km}^{ce} \\tilde T^{ik}_{ac})",2)
  if length(U1) > 0
    # ``tR^{ef}_{mn} += 0.5 Λ_m^e \hat f_n^f``
    @tensoropt tR2[e,f,m,n] += 0.5 * U1[e,m] * dfov[n,f]
    t1 = print_time(EC, t1, "tR^{ef}_{mn} += 0.5 U_m^e \\hat f_n^f",2)
  end
  int2 = load4idx(EC, "d_vovv")
  if length(U1) > 0
    # ``tR^{ef}_{mn} += 0.5 Λ_n^a \hat v_{am}^{fe}``
    @tensoropt tR2[e,f,m,n] += 0.5 * U1[a,n] * int2[a,m,f,e]
    t1 = print_time(EC, t1, "tR^{ef}_{mn} += 0.5 U_n^a \\hat v_{am}^{fe}",2)
  end
  oovo = load4idx(EC, "d_oovo")
  if length(U1) > 0
    # ``tR^{ef}_{mn} -= 0.5 Λ_i^f \hat v_{mn}^{ei}``
    @tensoropt tR2[e,f,m,n] -= 0.5 * U1[f,i] * oovo[m,n,e,i]
    t1 = print_time(EC, t1, "tR^{ef}_{mn} -= 0.5 U_i^f \\hat v_{mn}^{ei}",2)
  end
  # ``pR^{ef}_{mn} = 2 tR^{ef}_{mn} - tR^{ef}_{nm}``
  @tensoropt pR2[e,f,m,n] := 2.0 * tR2[e,f,m,n] - tR2[e,f,n,m]
  t1 = print_time(EC, t1, "pR^{ef}_{mn} = 2 tR^{ef}_{mn} - tR^{ef}_{nm}",2)
  tR2 = nothing
  # calc ``D_{ib}^{aj} = Λ_{ik}^{ac} \tilde T^{kj}_{cb}``
  @tensoropt D2[a,b,i,j] := U2[a,c,i,k] * tT2[c,b,k,j]
  t1 = print_time(EC, t1, "D_{ib}^{aj} = U_{ik}^{ac} \\tilde T^{kj}_{cb}",2)
  tT2 = nothing
  if length(U1) > 0
    # ``R^{e}_{m} += 2 D_{md}^{al} \hat v_{al}^{ed}``
    @tensoropt R1[e,m] += 2.0 * D2[a,d,m,l] * int2[a,l,e,d]
    t1 = print_time(EC, t1, "R^{e}_{m} += 2 D_{md}^{al} \\hat v_{al}^{ed}",2)
    # ``R^{e}_{m} += -2 D_{id}^{el} \hat v_{ml}^{id}``
    @tensoropt R1[e,m] -= 2.0 * D2[e,d,i,l] * oovo[l,m,d,i]
    t1 = print_time(EC, t1, "R^{e}_{m} += -2 D_{id}^{el} \\hat v_{ml}^{id}",2)
  end
  if !dc
    # ``pR^{ef}_{mn} -= D_{nc}^{fl} v_{ml}^{ce}``
    @tensoropt pR2[e,f,m,n] -= D2[f,c,n,l] * oovv[m,l,c,e]
    t1 = print_time(EC, t1, "pR^{ef}_{mn} -= D_{nc}^{fl} v_{ml}^{ce}",2)
  end
  # calc ``\bar D_{ib}^{aj} = Λ_{ik}^{ac} T^{kj}_{cb}+ Λ_{ik}^{ca} T^{kj}_{bc}``
  @tensoropt begin 
    D2[a,b,i,j] = U2[a,c,i,k] * T2[c,b,k,j]
    D2[a,b,i,j] += U2[c,a,i,k] * T2[b,c,k,j]
  end
  t1 = print_time(EC, t1, "\\bar D_{ib}^{aj} = U_{ik}^{ac} T^{kj}_{cb}+ U_{ik}^{ca} T^{kj}_{bc}",2)
  if !dc
    # ``pR^{ef}_{mn} += \bar D_{nd}^{ek} v_{km}^{fd}``
    @tensoropt pR2[e,f,m,n] += D2[e,d,n,k] * oovv[k,m,f,d]
    t1 = print_time(EC, t1, "pR^{ef}_{mn} += \\bar D_{nd}^{ek} v_{km}^{fd}",2)
  end
  if length(U1) > 0
    # ``R^{e}_{m} -= 2 \bar D_{mc}^{ak} \hat v_{ak}^{ce}``
    @tensoropt R1[e,m] -= 2.0 * D2[a,c,m,k] * int2[a,k,c,e]
    t1 = print_time(EC, t1, "R^{e}_{m} -= 2 \\bar D_{mc}^{ak} \\hat v_{ak}^{ce}",2)
    # ``R^{e}_{m} += 2 \bar D_{ic}^{ek} \hat v_{km}^{ic}``
    @tensoropt R1[e,m] += 2.0 * D2[e,c,i,k] * oovo[m,k,c,i]
    t1 = print_time(EC, t1, "R^{e}_{m} += 2 \\bar D_{ic}^{ek} \\hat v_{km}^{ic}",2)
  end
  int2 = nothing
  D2 = nothing
  # calc ``D_{ij}^{kl} = Λ_{ij}^{cd} T^{kl}_{cd}``
  @tensoropt D2[i,j,k,l] := U2[c,d,i,j] * T2[c,d,k,l]
  t1 = print_time(EC, t1, "D_{ij}^{kl} = U_{ij}^{cd} T^{kl}_{cd}",2)
  if length(U1) > 0
    # ``R_e^m += 2 D_{mj}^{kl} \hat v_{kl}^{ej}``
    @tensoropt R1[e,m] += 2.0 * D2[m,j,k,l] * oovo[k,l,e,j]
    t1 = print_time(EC, t1, "R_e^m += 2 D_{mj}^{kl} \\hat v_{kl}^{ej}",2)
  end
  oovo = nothing
  if !dc
    # ``R^{ef}_{mn} += D_{mn}^{kl} v_{kl}^{ef}``
    @tensoropt R2[e,f,m,n] += D2[m,n,k,l] * oovv[k,l,e,f]
    t1 = print_time(EC, t1, "R^{ef}_{mn} += D_{mn}^{kl} v_{kl}^{ef}",2)
  end
  D2 = nothing
  if length(U1) > 0
    # ``R^e_m -= 2 Λ_{ij}^{eb} (v_{mb}^{cd} T^{ij}_{cd})`` 
    # ``(v_{mb}^{cd} T^{ij}_{cd})`` has to be precalculated
    vT_ovoo = load4idx(EC, "vT_ovoo")
    @tensoropt R1[e,m] -= 2.0 * U2[e,b,i,j] * vT_ovoo[m,b,i,j]
    vT_ovoo = nothing
    t1 = print_time(EC, t1, "R^e_m -= 2 U_{ij}^{eb} (v_{mb}^{cd} T^{ij}_{cd})",2)
  end
  fac = dc ? 0.5 : 1.0
  @tensoropt begin 
    # ``pR^{ef}_{mn} += Λ_{mn}^{af} (\hat f_a^e - f * x_a^e)``
    pR2[e,f,m,n] += U2[a,f,m,n] * (dfock[SP['v'],SP['v']][a,e] - fac * xvv[a,e])
    # ``pR^{ef}_{mn} -= Λ_{in}^{ef} (\hat f_m^i + f * x_m^i)``
    pR2[e,f,m,n] -= U2[e,f,i,n] * (dfock[SP['o'],SP['o']][m,i] + fac * xoo[m,i])
  end
  t1 = print_time(EC, t1, "pR^{ef}_{mn} += U_{in}^{af} (\\hat x_a^e δ_m^i - \\hat x_m^i δ_a^e)",2)
  int2 = load4idx(EC, "d_vovo")
  # ``pR^{ef}_{mn} -= Λ_{in}^{af} \hat v_{am}^{ei}``
  @tensoropt pR2[e,f,m,n] -= U2[a,f,i,n] * int2[a,m,e,i]
  t1 = print_time(EC, t1, "pR^{ef}_{mn} -= U_{in}^{af} \\hat v_{am}^{ei}",2)
  # ``pR^{ef}_{mn} -= Λ_{in}^{eb} \hat v_{mb}^{if}``
  @tensoropt pR2[e,f,m,n] -= U2[e,b,i,n] * int2[b,m,f,i]
  t1 = print_time(EC, t1, "pR^{ef}_{mn} -= U_{in}^{eb} \\hat v_{mb}^{if}",2)
  # ``R^{ef}_{mn} += pR^{ef}_{mn} + pR^{fe}_{nm}``
  @tensoropt R2[e,f,m,n] += pR2[e,f,m,n] + pR2[f,e,n,m]
  t1 = print_time(EC, t1, "R^{ef}_{mn} += pR^{ef}_{mn} + pR^{fe}_{nm}",2)
  pR2 = nothing

  if length(U1) > 0
    # ``R^e_m += \hat D_p^q (2 v_{mq}^{ep} - v_{mq}^{pe})``
    int2 = ints2(EC,"momm")
    @tensoropt R1[e,m] += dD1[p,q] * (2.0 * int2[:,:,:,SP['v']][q,m,p,e] - int2[:,:,SP['v'],:][q,m,e,p])
    t1 = print_time(EC, t1, "R^e_m += \\hat D_p^q (2 v_{mq}^{ep} - v_{mq}^{pe})",2)
    # ``R^e_m -= 2 Λ_{ij}^{eb} \hat v_{mb}^{ij}``
    int2 = load4idx(EC, "d_vooo")
    @tensoropt R1[e,m] -= 2.0 * U2[e,b,i,j] * int2[b,m,j,i]
    t1 = print_time(EC, t1, "R^e_m -= 2 U_{ij}^{eb} \\hat v_{mb}^{ij}",2)
    @tensoropt begin
      # ``R^e_m += D_m^k \hat f_k^e - D_d^e \hat f_m^d``
      R1[e,m] += D1[SP['o'],SP['o']][m,k] * dfov[k,e]
      R1[e,m] -= D1[SP['v'],SP['v']][d,e] * dfov[m,d]
      # ``R^e_m += Λ_m^a (\hat f_a^e - x_a^e) - Λ_i^e (\hat f_m^i + x_m^i)``
      R1[e,m] += U1[a,m] * (dfock[SP['v'],SP['v']][a,e] - xvv[a,e])
      R1[e,m] -= U1[e,i] * (dfock[SP['o'],SP['o']][m,i] + xoo[m,i])
    end
    t1 = print_time(EC, t1, "R^e_m += U_m^a (\\hat f_a^e - x_a^e) - U_i^e (\\hat f_m^i + x_m^i)",2)
  end
  return R1, R2
end

"""
    calc_dU2(EC::ECInfo, T1, T12, U2, o1='o', v1='v', o2='o', v2='v')

Calculate the "dressed" ``Λ_2`` for CCSD/DCSD.

`T12` is the T1 amplitude for the second electron of `U2` (=`T1` for closed-shell and same-spin `U2`). 
Return `dU2[p,q,m,n]`=``Λ_{mn}^{ab}δ_a^p δ_b^q - Λ_{mn}^{ab}T^i_a δ_i^p δ_b^q - Λ_{mn}^{ab}δ_a^p T^j_b δ_j^q + Λ_{mn}^{ab}T^i_a T^j_b δ_i^p δ_j^q``.
"""
function calc_dU2(EC::ECInfo, T1, T12, U2, o1='o', v1='v', o2='o', v2='v')
  SP = EC.space
  norb = n_orbs(EC)
  nocc1 = size(U2,3)
  nocc2 = size(U2,4)
  dU2 = zeros(norb,norb,nocc1,nocc2)
  dU2[SP[v1],SP[v2],:,:] = U2 
  if length(T1) > 0
    @tensoropt dUovoo[i,b,m,n] := U2[a,b,m,n] * T1[a,i]
    @tensoropt dU2[SP[v1],SP[o2],:,:][a,j,m,n] = - U2[a,b,m,n] * T12[b,j]
    @tensoropt dU2[SP[o1],SP[o2],:,:][i,j,m,n] = dUovoo[i,b,m,n] * T12[b,j]
    dU2[SP[o1],SP[v2],:,:] = -dUovoo
  end
  return dU2
end

"""
    calc_ccsd_vector_times_Jacobian(EC::ECInfo, U1a, U1b, U2a, U2b, U2ab; dc=false)

Calculate the vector times the Jacobian for the unresticted CCSD or DCSD
equations.

Return R1a, R1b, R2a, R2b, R2ab
"""
function calc_ccsd_vector_times_Jacobian(EC::ECInfo, U1a, U1b, U2a, U2b, U2ab; dc=false)
  t1 = time_ns()
 
  T2ab = load4idx(EC, "T_vVoO")
  # Calculate 1RDM intermediates
  T1 = load2idx(EC, "T_vo")
  T2 = load4idx(EC, "T_vvoo")
  D1α, dD1α = calc_1RDM(EC, U1a, U1b, U2a, U2ab, T1, T2, T2ab, :α)
  T1 = load2idx(EC, "T_VO")
  T2 = load4idx(EC, "T_VVOO")
  D1β, dD1β = calc_1RDM(EC, U1b, U1a, U2b, U2ab, T1, T2, T2ab, :β)
  T1 = T2 = T2ab = nothing
  t1 = print_time(EC, t1, "calculate 1RDM",2)

  R1a, R2a = calc_ccsd_vector_times_Jacobian4spin(EC, U1a, U2a, U2ab, D1α, dD1α, dD1β, :α; dc)
  t1 = print_time(EC, t1, "calculate R1a, R2a",2)
  R1b, R2b = calc_ccsd_vector_times_Jacobian4spin(EC, U1b, U2b, U2ab, D1β, dD1β, dD1α, :β; dc)
  t1 = print_time(EC, t1, "calculate R1b, R2b",2)
  ΔR1a, ΔR1b, R2ab = calc_ccsd_vector_times_Jacobian4ab(EC, U1a, U1b, U2a, U2b, U2ab, D1α, D1β; dc)
  R1a += ΔR1a
  R1b += ΔR1b
  t1 = print_time(EC, t1, "calculate ΔR1a, ΔR1b, R2ab",2)
  return R1a, R1b, R2a, R2b, R2ab
end

"""
    calc_ccsd_vector_times_Jacobian4spin(EC::ECInfo, U1a, U1b, U2a, U2b, U2ab, D1, dD1, dD1os, spin; dc=false)

Calculate the vector times the CCSD/DCSD Jacobian for the given `spin`
(same-spin residual for doubles). The singles residual is missing some terms
which are added in `calc_ccsd_vector_times_Jacobian4ab`.

Return R1 and R2 
"""
function calc_ccsd_vector_times_Jacobian4spin(EC::ECInfo, U1, U2, U2ab, 
            D1, dD1, dD1os, spin; dc=false)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  t1 = time_ns()

  SP = EC.space
  isα = (spin == :α)
  # spaces for the given spin
  o4s = space4spin('o', isα)
  v4s = space4spin('v', isα)
  m4s = space4spin('m', isα)
  # spaces for the opposite spin
  o4os = space4spin('o', !isα)
  v4os = space4spin('v', !isα)
  m4os = space4spin('m', !isα)

  norb = n_orbs(EC)

  fock = load2idx(EC,"f_"*m4s*m4s)
  dfock = load2idx(EC,"df_"*m4s*m4s)
  fov = fock[SP[o4s],SP[v4s]]
  dfov = dfock[SP[o4s],SP[v4s]] 
  if length(U1) > 0
    @tensoropt R1[e,m] := fov[m,e]
    # ``R^e_m += \hat D_p^q (v_{mq}^{ep} - v_{mq}^{pe})``
    int2 = ints2(EC,m4s*o4s*m4s*m4s)
    @tensoropt R1[e,m] += dD1[p,q] * (int2[:,:,:,SP[v4s]][q,m,p,e] - int2[:,:,SP[v4s],:][q,m,e,p])
    t1 = print_time(EC, t1, "R_e^m += \\hat D_p^q (v_{mq}^{ep} - v_{mq}^{pe})",2)
    if isα
      # ``R^e_m += \hat D_P^Q v_{mQ}^{eP}``
      int2 = ints2(EC,o4s*m4os*v4s*m4os)
      @tensoropt R1[e,m] += dD1os[p,q] * int2[:,:,:,:][m,q,e,p]
      t1 = print_time(EC, t1, "R_e^m += \\hat D_P^Q v_{mQ}^{eP}",2)
    else
      # ``R^e_m += \hat D_P^Q v_{Qm}^{Pe}``
      int2 = ints2(EC,m4os*o4s*m4os*v4s)
      @tensoropt R1[e,m] += dD1os[p,q] * int2[:,:,:,:][q,m,p,e]
      t1 = print_time(EC, t1, "R_e^m += \\hat D_P^Q v_{Qm}^{Pe}",2)
    end
  else
    R1 = U1
  end

  T1 = load2idx(EC, "T_"*v4s*o4s)
  T2 = load4idx(EC, "T_"*v4s*v4s*o4s*o4s)
  oovv = ints2(EC,o4s*o4s*v4s*v4s)
  @tensoropt R2[e,f,m,n] := oovv[m,n,e,f] - oovv[n,m,e,f]
  int2 = load4idx(EC, "d_"*o4s*o4s*o4s*o4s)
  if !dc
    @tensoropt int2[m,n,i,j] += 0.5 * oovv[m,n,c,d] * T2[c,d,i,j]
  end
  # ``R^{ef}_{mn} += Λ_{ij}^{ef} (\hat v_{mn}^{ij} \red{+ 0.5 v_{mn}^{cd} T^{ij}_{cd}})``
  @tensoropt R2[e,f,m,n] += int2[m,n,i,j] * U2[e,f,i,j]
  t1 = print_time(EC, t1, "R^{ef}_{mn} += U_{ij}^{ef} (\\hat v_{mn}^{ij} + 0.5 v_{mn}^{cd} T^{ij}_{cd})",2)

  # the 4-external part
  if EC.options.cc.use_kext
    # the `kext` part
    spin4int = EC.fd.uhf ? spin : :α
    int2 = integ2_ss(EC.fd, spin4int)
    dU2 = calc_dU2(EC, T1, T1, U2, o4s, v4s, o4s, v4s)
    if ndims(int2) == 4
      error("Non-triangular integrals not tested in kext equations in ΛUCCSD")
      # ``K_{mn}^{rs} = \hat U_{mn}^{pq} v_{pq}^{rs}``
      @tensoropt Kmmoo[r,s,m,n] := int2[p,q,r,s] * dU2[p,q,m,n]
      dU2 = nothing
    else
      # last two indices of integrals are stored as upper triangular 
      # ``K_{mn}^{rs} = \hat U_{mn}^{pq} v_{pq}^{rs}``
      @tensoropt Kxoo[x,m,n] := int2[p,q,x] * dU2[p,q,m,n]
      Kmmoo = dU2
      dU2 = nothing
      tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
      Kmmoo[tripp,:,:] = Kxoo
      trippr = CartesianIndex.(reverse.(Tuple.(tripp)))
      @tensor Kmmoo[trippr,:,:][x,m,n] = Kxoo[x,n,m]
      Kxoo = nothing
    end
    t1 = print_time(EC, t1, "K_{mn}^{rs} = \\hat U_{mn}^{pq} v_{pq}^{rs}",2)
    # ``R^{ef}_{mn} += K_{mn}^{rs} δ_r^e δ_s^f``
    R2 += Kmmoo[SP[v4s],SP[v4s],:,:]
    # ``R^e_m += K_{mj}^{rs} δ_r^e (δ_s^j + δ_s^b T^j_b)``
    if length(U1) > 0
      @tensoropt R1[e,m] += Kmmoo[SP[v4s],SP[o4s],:,:][e,j,m,j] 
      if length(T1) > 0
        @tensoropt R1[e,m] += Kmmoo[SP[v4s],SP[v4s],:,:][e,b,m,j] * T1[b,j]
      end 
    end
    t1 = print_time(EC, t1, "R^e_m += K_{mj}^{rs} δ_r^e (δ_s^j + δ_s^b T^j_b)",2)
    Kmmoo = nothing
  else
    error("non-kext Λ equations not implemented")
  end

  # ``D_{ij}^{kl} = 0.5 Λ_{ij}^{cd} T^{kl}_{cd}``
  @tensoropt D2[i,j,k,l] := 0.5 * U2[c,d,i,j] * T2[c,d,k,l]
  t1 = print_time(EC, t1, "D_{ij}^{kl} = 0.5 U_{ij}^{cd} T^{kl}_{cd}",2)
  if !dc
    # ``R^{ef}_{mn} += D_{mn}^{kl} v_{kl}^{ef}`` 
    @tensoropt R2[e,f,m,n] += D2[m,n,k,l] * oovv[k,l,e,f]
    t1 = print_time(EC, t1, "R^{ef}_{mn} += D_{mn}^{kl} v_{kl}^{ef}",2)
  end
  d_oovo = load4idx(EC, "d_"*o4s*o4s*v4s*o4s)
  if length(U1) > 0
    # ``R_e^m += D_{mj}^{kl} \hat v_{kl}^{ej}``
    @tensoropt R1[e,m] += D2[m,j,k,l] * d_oovo[k,l,e,j]
    t1 = print_time(EC, t1, "R_e^m += D_{mj}^{kl} \\hat v_{kl}^{ej}",2)
  end
  # ``D_{ib}^{aj} = Λ_{ik}^{ac} T^{jk}_{bc} + Λ_{iK}^{aC} T^{jK}_{bC}``
  T2ab = load4idx(EC, "T_vVoO")
  @tensoropt D2[a,b,i,j] := U2[a,c,i,k] * T2[b,c,j,k]
  if isα
    @tensoropt D2[a,b,i,j] += U2ab[a,C,i,K] * T2ab[b,C,j,K]
  else
    @tensoropt D2[A,B,I,J] += U2ab[c,A,k,I] * T2ab[c,B,k,J]
  end
  T2ab = nothing
  t1 = print_time(EC, t1, "D_{ib}^{aj} = U_{ik}^{ac} T^{jk}_{bc} + U_{iK}^{aC} T^{jK}_{bC}",2)
  if length(R1) > 0
    # ``R_e^m += D_{id}^{el} (\hat v_{ml}^{di}-\hat v_{lm}^{di})``
    @tensoropt R1[e,m] += D2[e,d,i,l] * (d_oovo[m,l,d,i] - d_oovo[l,m,d,i])
    t1 = print_time(EC, t1, "R_e^m += D_{id}^{el} (\\hat v_{ml}^{di}-\\hat v_{lm}^{di})",2)
  end
  int2 = load4idx(EC, "d_"*v4s*o4s*v4s*v4s)
  if length(R1) > 0
    # ``R_e^m += D_{md}^{al} (\hat v_{al}^{ed}-\hat v_{al}^{de})``
    @tensoropt R1[e,m] += D2[a,d,m,l] * (int2[a,l,e,d] - int2[a,l,d,e])
    t1 = print_time(EC, t1, "R_e^m += D_{md}^{al} (\\hat v_{al}^{ed}-\\hat v_{al}^{de})",2)
  end
  D2 = nothing
  # calc ``aR`` (to be antisymmetrized)
  if length(U1) > 0
    # ``aR^{ef}_{mn} += Λ_{m}^{a} \hat v_{an}^{ef}``
    @tensoropt aR2[e,f,m,n] := U1[a,m] * int2[a,n,e,f]
    int2 = nothing
    t1 = print_time(EC, t1, "aR^{ef}_{mn} += U_{m}^{a} \\hat v_{an}^{ef}",2)
    # ``aR^{ef}_{mn} -= Λ_{i}^{f} \hat v_{mn}^{ei}``
    @tensoropt aR2[e,f,m,n] -= U1[f,i] * d_oovo[m,n,e,i]
    d_oovo = nothing
    t1 = print_time(EC, t1, "aR^{ef}_{mn} -= U_{i}^{f} \\hat v_{mn}^{ei}",2)
    # ``aR^{ef}_{mn} += Λ_{m}^{e} \hat f_{n}^{f}``
    @tensoropt aR2[e,f,m,n] += U1[e,m] * dfov[n,f]
    t1 = print_time(EC, t1, "aR^{ef}_{mn} += U_{m}^{e} \\hat f_{n}^{f}",2)
  else
    aR2 = zeros(size(R2))
  end
  fac = dc ? 0.5 : 1.0
  # ``aR^{ef}_{mn} += 0.5(\hat x_c^e Λ_{mn}^{cf} - \hat x_m^k Λ_{kn}^{ef})``
  x_vv = load2idx(EC, "vT_"*v4s*v4s)
  x_oo = load2idx(EC, "vT_"*o4s*o4s)
  dx_vv = dfock[SP[v4s],SP[v4s]] - fac * x_vv
  dx_oo = dfock[SP[o4s],SP[o4s]] + fac * x_oo
  @tensoropt begin 
    aR2[e,f,m,n] += 0.5 * dx_vv[c,e] * U2[c,f,m,n]
    aR2[e,f,m,n] -= 0.5 * dx_oo[m,k] * U2[e,f,k,n]
  end
  t1 = print_time(EC, t1, "aR^{ef}_{mn} += 0.5(\\hat x_c^e U_{mn}^{cf} - \\hat x_m^k U_{kn}^{ef})",2)
  if length(U1) > 0
    # ``R^e_m += \hat x_c^e Λ_m^c - \hat x_m^k Λ_k^e``
    dx_vv -= (1.0 - fac) * x_vv
    dx_oo += (1.0 - fac) * x_oo
    @tensoropt begin 
      R1[e,m] += dx_vv[c,e] * U1[c,m]
      R1[e,m] -= dx_oo[m,k] * U1[e,k]
    end
    dx_vv = dx_oo = x_vv = x_oo = nothing
    t1 = print_time(EC, t1, "R^e_m += \\hat x_c^e U_m^c - \\hat x_m^k U_k^e",2)
  end
  # ``aR^{ef}_{mn} += f D_m^k v_{kn}^{ef} - f D_c^e v_{mn}^{cf}``
  @tensoropt begin 
    aR2[e,f,m,n] += fac * D1[SP[o4s],SP[o4s]][m,k] * oovv[k,n,e,f]
    aR2[e,f,m,n] -= fac * D1[SP[v4s],SP[v4s]][c,e] * oovv[m,n,c,f]
  end
  t1 = print_time(EC, t1, "aR^{ef}_{mn} += f D_m^k v_{kn}^{ef} - f D_c^e v_{mn}^{cf}",2)
  # ``aR^{ef}_{mn} += Λ_{in}^{af} \bar y_{am}^{ie}``
  vT_voov = load4idx(EC, "vT_"*v4s*o4s*o4s*v4s)
  @tensoropt aR2[e,f,m,n] += U2[a,f,i,n] * vT_voov[a,m,i,e]
  vT_voov = nothing
  t1 = print_time(EC, t1, "aR^{ef}_{mn} += U_{in}^{af} \\bar y_{am}^{ie}",2)
  # ``aR^{ef}_{mn} += Λ_{mJ}^{eB} \bar y_{Bn}^{Jf}``
  vT_VoOv = load4idx(EC, "vT_"*v4os*o4s*o4os*v4s) 
  if isα
    @tensoropt aR2[e,f,m,n] += U2ab[e,B,m,J] * vT_VoOv[B,n,J,f]
  else
    @tensoropt aR2[E,F,M,N] += U2ab[b,E,j,M] * vT_VoOv[b,N,j,F]
  end
  vT_VoOv = nothing
  t1 = print_time(EC, t1, "aR^{ef}_{mn} += U_{mJ}^{eB} \\bar y_{Bn}^{Jf}",2)
  # ``R^{ef}_{mn} += aR^{ef}_{mn} + aR^{fe}_{nm} - aR^{ef}_{nm} - aR^{fe}_{mn}``
  @tensoropt R2[e,f,m,n] += aR2[e,f,m,n] + aR2[f,e,n,m] - aR2[e,f,n,m] - aR2[f,e,m,n]
  aR2 = nothing
  t1 = print_time(EC, t1, "R^{ef}_{mn} += aR^{ef}_{mn} + aR^{fe}_{nm} - aR^{ef}_{nm} - aR^{fe}_{mn}",2)
  if length(R1) > 0
    # ``R^e_m -= Λ_{ij}^{eb} \hat v_{bm}^{ji}``
    int2 = load4idx(EC, "d_"*v4s*o4s*o4s*o4s)
    @tensoropt R1[e,m] -= U2[b,e,j,i] * int2[b,m,j,i]
    int2 = nothing
    t1 = print_time(EC, t1, "R^e_m -= U_{ij}^{eb} \\hat v_{bm}^{ji}",2)
    # ``R^e_m -= 0.5 Λ_{ij}^{eb} (\hat v_{mb}^{cd} T^{ij}_{cd})``
    vT_ovoo = load4idx(EC, "vT_"*o4s*v4s*o4s*o4s)
    @tensoropt R1[e,m] -= 0.5 * U2[e,b,i,j] * vT_ovoo[m,b,i,j]
    t1 = print_time(EC, t1, "R^e_m -= 0.5 U_{ij}^{eb} (\\hat v_{mb}^{cd} T^{ij}_{cd})",2)
    # ``R^e_m -= Λ_{iJ}^{eB} (\hat v_{mB}^{cD} T^{iJ}_{cD})``
    vT_ovoo = load4idx(EC, "vT_"*o4s*v4os*o4s*o4os)
    if isα
      @tensoropt R1[e,m] -= U2ab[e,B,i,J] * vT_ovoo[m,B,i,J]
    else
      @tensoropt R1[E,M] -= U2ab[b,E,j,I] * vT_ovoo[M,b,I,j]
    end
    vT_ovoo = nothing
    t1 = print_time(EC, t1, "R^e_m -= U_{iJ}^{eB} (\\hat v_{mB}^{cD} T^{iJ}_{cD})",2)
    # ``R^e_m += D_m^k \hat f_k^e - D_c^e \hat f_m^c``
    @tensoropt begin
      R1[e,m] += D1[SP[o4s],SP[o4s]][m,k] * dfov[k,e]
      R1[e,m] -= D1[SP[v4s],SP[v4s]][c,e] * dfov[m,c]
    end
    t1 = print_time(EC, t1, "R^e_m += D_m^k \\hat f_k^e - D_c^e \\hat f_m^c",2)
  end
  return R1, R2
end

"""
    calc_ccsd_vector_times_Jacobian4ab(EC::ECInfo, U1a, U1b, U2a, U2b, U2ab, D1a, D1b; dc=false)

Calculate the left vector times the CCSD/DCSD Jacobian for αβ component. 
Additionally, remaining contributions to the singles residual are calculated.

Return ΔR1a, ΔR1b, R2ab
"""
function calc_ccsd_vector_times_Jacobian4ab(EC::ECInfo, U1a::Matrix{Float64}, U1b::Matrix{Float64}, 
          U2a::Array{Float64,4}, U2b::Array{Float64}, U2ab::Array{Float64}, D1a, D1b; dc=false) 
  t1 = time_ns()

  SP = EC.space
  nocca = n_occ_orbs(EC)
  noccb = n_occb_orbs(EC)
  norb = n_orbs(EC)

  dfocka = load2idx(EC,"df_mm")
  dfockb = load2idx(EC,"df_MM")
  dfaov = dfocka[SP['o'],SP['v']] 
  dfbov = dfockb[SP['O'],SP['V']] 

  if length(U1a) > 0
    # ``R^e_m -= Λ_{iJ}^{eB} \hat v_{mB}^{iJ}``
    int2 = load4idx(EC, "d_oVoO")
    @tensoropt R1a[e,m] := - U2ab[e,B,i,J] * int2[m,B,i,J]
    int2 = NOTHING4idx
  else
    R1a = U1a
  end
  if length(U1b) > 0
    # ``R^E_M -= Λ_{jI}^{bE} \hat v_{bM}^{jI}``
    int2 = load4idx(EC, "d_vOoO")
    @tensoropt R1b[E,M] := - U2ab[b,E,j,I] * int2[b,M,j,I]
    int2 = NOTHING4idx
  else
    R1b = U1b
  end

  T1a = load2idx(EC, "T_vo")
  T1b = load2idx(EC, "T_VO")
  # the 4-external part
  if EC.options.cc.use_kext
    # the `kext` part
    dU2 = calc_dU2(EC, T1a, T1b, U2ab, 'o','v','O','V')
    Kmmoo = Array{Float64,4}(undef,norb,norb,nocca,noccb) 
    if EC.fd.uhf
      int2 = integ2_os(EC.fd)
      # ``K_{mN}^{rS} = \hat U_{mN}^{pQ} v_{pQ}^{rS}``
      @tensoropt Kmmoo[r,S,m,N] = int2[p,Q,r,S] * dU2[p,Q,m,N]
      int2 = NOTHING4idx
    else
      int2_3idx = integ2_ss(EC.fd)
      # ``K_{mN}^{rS} = \hat U_{mN}^{pQ} v_{pQ}^{rS}``, ``r ≤ S``
      @tensoropt Kxoo[x,m,N] := int2_3idx[p,Q,x] * dU2[p,Q,m,N]
      tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
      Kmmoo[tripp,:,:] = Kxoo
      # ``K_{mN}^{rS} = \hat U_{mN}^{pQ} v_{Qp}^{Sr}``, ``S ≤ r``
      @tensoropt Kxoo[x,m,N] = int2_3idx[Q,p,x] * dU2[p,Q,m,N]
      trippr = CartesianIndex.(reverse.(Tuple.(tripp)))
      @tensor Kmmoo[trippr,:,:][x,m,N] = Kxoo[x,m,N]
      Kxoo = NOTHING3idx
      int2_3idx = NOTHING3idx
    end
    dU2 = NOTHING4idx
    t1 = print_time(EC, t1, "K_{mN}^{rS} = \\hat U_{mN}^{pQ} v_{pQ}^{rS}",2)
    # ``R^{eF}_{mN} += K_{mN}^{rS} δ_r^e δ_S^F``
    R2 = Kmmoo[SP['v'],SP['V'],:,:]
    # ``R^e_m += K_{mJ}^{rS} δ_r^e (δ_S^J + δ_S^B T^J_B)``
    if length(R1a) > 0
      @tensoropt R1a[e,m] += Kmmoo[SP['v'],SP['O'],:,:][e,J,m,J] 
      if length(T1b) > 0
        @tensoropt R1a[e,m] += Kmmoo[SP['v'],SP['V'],:,:][e,B,m,J] * T1b[B,J]
      end 
    end
    t1 = print_time(EC, t1, "R^e_m += K_{mJ}^{rS} δ_r^e (δ_S^J + δ_S^B T^J_B)",2)
    # ``R^E_M += K_{jM}^{sR} δ_R^E (δ_s^j + δ_s^b T^j_b)``
    if length(R1b) > 0
      @tensoropt R1b[E,M] += Kmmoo[SP['o'],SP['V'],:,:][j,E,j,M] 
      if length(T1a) > 0
        @tensoropt R1b[E,M] += Kmmoo[SP['v'],SP['V'],:,:][b,E,j,M] * T1a[b,j]
      end 
    end
    t1 = print_time(EC, t1, "R^e_m += K_{mJ}^{rS} δ_r^e (δ_S^J + δ_S^B T^J_B)",2)
    Kmmoo = NOTHING4idx
  else
    error("non-kext Λ equations not implemented")
  end

  oOvV = ints2(EC,"oOvV")
  @tensoropt R2[e,F,m,N] += oOvV[m,N,e,F]
  T2ab = load4idx(EC, "T_vVoO")
  int2 = load4idx(EC, "d_oOoO")
  if !dc
    @tensoropt int2[m,N,i,J] += oOvV[m,N,c,D] * T2ab[c,D,i,J]
  end
  # ``R^{eF}_{mN} += Λ_{iJ}^{eF} (\hat v_{mN}^{iJ} \red{+ v_{mN}^{cD} T^{iJ}_{cD}})``
  @tensoropt R2[e,F,m,N] += int2[m,N,i,J] * U2ab[e,F,i,J] 
  t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{iJ}^{eF} (\\hat v_{mN}^{iJ} + v_{mN}^{cD} T^{iJ}_{cD})",2)
  
  # ``D_{iJ}^{kL} = Λ_{iJ}^{cD} T^{kL}_{cD}``
  @tensoropt D2[i,J,k,L] := U2ab[c,D,i,J] * T2ab[c,D,k,L]
  t1 = print_time(EC, t1, "D_{iJ}^{kL} = U_{iJ}^{cD} T^{kL}_{cD}",2)
  if !dc
    # ``R^{eF}_{mN} += D_{mN}^{kL} v_{kL}^{eF}`` 
    @tensoropt R2[e,F,m,N] += D2[m,N,k,L] * oOvV[k,L,e,F]
    t1 = print_time(EC, t1, "R^{eF}_{mN} += D_{mN}^{kL} v_{kL}^{eF}",2)
  end
  d_oOvO = load4idx(EC, "d_oOvO")
  if length(R1a) > 0
    # ``R^e_m += D_{mJ}^{kL} \hat v_{kL}^{eJ}``
    @tensoropt R1a[e,m] += D2[m,J,k,L] * d_oOvO[k,L,e,J]
    t1 = print_time(EC, t1, "R^e_m += D_{mJ}^{kL} \\hat v_{kL}^{eJ}",2)
  end
  d_oOoV = load4idx(EC, "d_oOoV")
  if length(R1b) > 0
    # ``R^E_M += D_{jM}^{lK} \hat v_{lK}^{jE}``
    @tensoropt R1b[E,M] += D2[j,M,l,K] * d_oOoV[l,K,j,E]
    t1 = print_time(EC, t1, "R^E_M += D_{jM}^{lK} \\hat v_{lK}^{jE}",2)
  end
  if length(R1a) > 0
    # ``\bar D_{iB}^{aJ} = Λ_{ik}^{ac} T^{kJ}_{cB} + Λ_{iK}^{aC} T^{JK}_{BC}``
    @tensoropt D2[a,B,i,J] := U2a[a,c,i,k] * T2ab[c,B,k,J]
    T2b = load4idx(EC, "T_VVOO")
    @tensoropt D2[a,B,i,J] += U2ab[a,C,i,K] * T2b[C,B,K,J]
    t1 = print_time(EC, t1, "\\bar D_{iB}^{aJ} = U_{ik}^{ac} T^{kJ}_{cB} + U_{iK}^{aC} T^{JK}_{BC}",2)
    T2b = nothing
    # ``R^e_m -= \bar D_{iD}^{eL} \hat v_{mL}^{iD}``
    @tensoropt R1a[e,m] -= D2[e,D,i,L] * d_oOoV[m,L,i,D]
    t1 = print_time(EC, t1, "R_e^m -= \\bar D_{iD}^{eL} \\hat v_{mL}^{iD}",2)
  end
  d_vOvV = load4idx(EC, "d_vOvV")
  if length(R1a) > 0
    # ``R^e_m += \bar D_{mD}^{aL} \hat v_{aL}^{eD}``
    @tensoropt R1a[e,m] += D2[a,D,m,L] * d_vOvV[a,L,e,D]
    t1 = print_time(EC, t1, "R_e^m += \\bar D_{mD}^{aL} \\hat v_{aL}^{eD}",2)
  end
  if length(R1a) > 0 || length(R1b) > 0
    # ``D_{Ib}^{aJ} = Λ_{kI}^{aC} T^{kJ}_{bC}``
    @tensoropt D2[a,b,I,J] := U2ab[a,C,k,I] * T2ab[b,C,k,J]
    t1 = print_time(EC, t1, "D_{Ib}^{aJ} = U_{kI}^{aC} T^{kJ}_{bC}",2)
  end
  if length(R1a) > 0
    # ``R^e_m += D_{Jc}^{eK} \hat v_{mK}^{cJ}``
    @tensoropt R1a[e,m] += D2[e,c,J,K] * d_oOvO[m,K,c,J]
    t1 = print_time(EC, t1, "R_e^m += D_{Jc}^{eK} \\hat v_{mK}^{cJ}",2)
  end
  if length(R1b) > 0
    # ``R^E_M -= D_{Md}^{bK} \hat v_{bK}^{dE}``
    @tensoropt R1b[E,M] -= D2[b,d,M,K] * d_vOvV[b,K,d,E]
    t1 = print_time(EC, t1, "R_E^M -= D_{Md}^{bK} \\hat v_{bK}^{dE}",2)
  end
  D2 = NOTHING4idx
  if length(U1a) > 0
    # ``R^{eF}_{mN} += Λ^a_m \hat v_{aN}^{eF}``
    @tensoropt R2[e,F,m,N] += U1a[a,m] * d_vOvV[a,N,e,F]
    t1 = print_time(EC, t1, "R^{eF}_{mN} += U^a_m \\hat v_{aN}^{eF}",2)
  end
  d_vOvV = NOTHING4idx

  if length(R1b) > 0
    # ``\bar D_{Ib}^{Aj} = Λ_{IK}^{AC} T^{jK}_{bC} + Λ_{kI}^{cA} T^{jk}_{bc}``
    @tensoropt D2[A,b,I,j] := U2b[A,C,I,K] * T2ab[b,C,j,K]
    T2a = load4idx(EC, "T_vvoo")
    @tensoropt D2[A,b,I,j] += U2ab[c,A,k,I] * T2a[c,b,k,j]
    t1 = print_time(EC, t1, "\\bar D_{Ib}^{Aj} = U_{IK}^{AC} T^{jK}_{bC} + U_{kI}^{cA} T^{jk}_{bc}",2)
    T2a = NOTHING4idx
    # ``R^E_M -= \bar D_{Id}^{El} \hat v_{lM}^{dI}``
    @tensoropt R1b[E,M] -= D2[E,d,I,l] * d_oOvO[l,M,d,I]
    t1 = print_time(EC, t1, "R_E^M -= \\bar D_{Id}^{El} \\hat v_{lM}^{dI}",2)
  end
  d_oVvV = load4idx(EC, "d_oVvV")
  if length(R1b) > 0
    # ``R^E_M += \bar D_{Md}^{Al} \hat v_{lA}^{dE}``
    @tensoropt R1b[E,M] += D2[A,d,M,l] * d_oVvV[l,A,d,E]
    t1 = print_time(EC, t1, "R_E^M += \\bar D_{Md}^{Al} \\hat v_{lA}^{dE}",2)
  end
  if length(R1a) > 0 || length(R1b) > 0
    # ``D_{iB}^{Aj} = Λ_{iK}^{cA} T^{jK}_{cB}``
    @tensoropt D2[A,B,i,j] := U2ab[c,A,i,K] * T2ab[c,B,j,K]
    t1 = print_time(EC, t1, "D_{iB}^{Aj} = U_{iK}^{cA} T^{jK}_{cB}",2)
  end
  if length(R1b) > 0
    # ``R^E_M += D_{jC}^{Ek} \hat v_{kM}^{jC}``
    @tensoropt R1b[E,M] += D2[E,C,j,k] * d_oOoV[k,M,j,C]
    t1 = print_time(EC, t1, "R_E^M += D_{jC}^{Ek} \\hat v_{kM}^{jC}",2)
  end
  if length(R1a) > 0
    # ``R^e_m -= D_{mD}^{Bk} \hat v_{kB}^{eD}``
    @tensoropt R1a[e,m] -= D2[B,D,m,k] * d_oVvV[k,B,e,D]
    t1 = print_time(EC, t1, "R_e^m -= D_{mD}^{Bk} \\hat v_{kB}^{eD}",2)
  end
  D2 = NOTHING4idx
  if length(U1b) > 0
    # ``R^{eF}_{mN} += Λ^A_N \hat v_{mA}^{eF}``
    @tensoropt R2[e,F,m,N] += U1b[A,N] * d_oVvV[m,A,e,F]
    t1 = print_time(EC, t1, "R^{eF}_{mN} += U^A_N \\hat v_{mA}^{eF}",2)
  end
  d_oVvV = NOTHING4idx
  T2ab = NOTHING4idx
  if length(U1a) > 0
    # ``R^{eF}_{mN} -= Λ_{i}^{e} \hat v_{mN}^{iF}``
    @tensoropt R2[e,F,m,N] -= U1a[e,i] * d_oOoV[m,N,i,F]
    t1 = print_time(EC, t1, "R^{eF}_{mN} -= U_{i}^{e} \\hat v_{mN}^{iF}",2)
    # ``R^{eF}_{mN} += Λ_{m}^{e} \hat f_{N}^{F}``
    @tensoropt R2[e,F,m,N] += U1a[e,m] * dfbov[N,F]
    t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{m}^{e} \\hat f_{N}^{F}",2)
  end
  d_oOoV = NOTHING4idx
  if length(U1b) > 0
    # ``R^{eF}_{mN} -= Λ_{I}^{F} \hat v_{mN}^{eI}``
    @tensoropt R2[e,F,m,N] -= U1b[F,I] * d_oOvO[m,N,e,I]
    t1 = print_time(EC, t1, "R^{eF}_{mN} -= U_{I}^{F} \\hat v_{mN}^{eI}",2)
    # ``R^{eF}_{mN} += Λ_{N}^{F} \hat f_{m}^{e}``
    @tensoropt R2[e,F,m,N] += U1b[F,N] * dfaov[m,e]
    t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{N}^{F} \\hat f_{m}^{e}",2)
  end
  d_oOvO = NOTHING4idx
  fac = dc ? 0.5 : 1.0
  # ``R^{eF}_{mN} += \hat x_c^e Λ_{mN}^{cF} - \hat x_m^k Λ_{kN}^{eF} + ...``
  x_vv = load2idx(EC, "vT_vv")
  x_VV = load2idx(EC, "vT_VV")
  x_oo = load2idx(EC, "vT_oo")
  x_OO = load2idx(EC, "vT_OO")
  dx_vv = dfocka[SP['v'],SP['v']] - fac * x_vv
  dx_VV = dfockb[SP['V'],SP['V']] - fac * x_VV
  dx_oo = dfocka[SP['o'],SP['o']] + fac * x_oo
  dx_OO = dfockb[SP['O'],SP['O']] + fac * x_OO
  @tensoropt begin 
    R2[e,F,m,N] += dx_vv[c,e] * U2ab[c,F,m,N]
    R2[e,F,m,N] += dx_VV[C,F] * U2ab[e,C,m,N]
    R2[e,F,m,N] -= dx_oo[m,k] * U2ab[e,F,k,N]
    R2[e,F,m,N] -= dx_OO[N,K] * U2ab[e,F,m,K]
  end
  t1 = print_time(EC, t1, "R^{eF}_{mN} += \\hat x_c^e U_{mN}^{cF} - \\hat x_m^k U_{kN}^{eF} + ...",2)
  # ``R^{eF}_{mN} += f D_m^k v_{kN}^{eF} - f D_c^e v_{mN}^{cF} + ...``
  @tensoropt begin 
    R2[e,F,m,N] += fac * D1a[SP['o'],SP['o']][m,k] * oOvV[k,N,e,F]
    R2[e,F,m,N] += fac * D1b[SP['O'],SP['O']][N,K] * oOvV[m,K,e,F]
    R2[e,F,m,N] -= fac * D1a[SP['v'],SP['v']][c,e] * oOvV[m,N,c,F]
    R2[e,F,m,N] -= fac * D1b[SP['V'],SP['V']][C,F] * oOvV[m,N,e,C]
  end
  t1 = print_time(EC, t1, "R^{eF}_{mN} += f D_m^k v_{kN}^{eF} - f D_c^e v_{mN}^{cF} + ...",2)
  # ``R^{eF}_{mN} += Λ_{iN}^{aF} \bar y_{am}^{ie}``
  vT_voov = load4idx(EC, "vT_voov")
  @tensoropt R2[e,F,m,N] += U2ab[a,F,i,N] * vT_voov[a,m,i,e]
  vT_voov = NOTHING4idx
  t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{iN}^{aF} \\bar y_{am}^{ie}",2)
  # ``R^{eF}_{mN} += Λ_{mJ}^{eB} \bar y_{BN}^{JF}``
  vT_VOOV = load4idx(EC, "vT_VOOV")
  @tensoropt R2[e,F,m,N] += U2ab[e,B,m,J] * vT_VOOV[B,N,J,F]
  vT_VOOV = NOTHING4idx
  t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{mJ}^{eB} \\bar y_{BN}^{JF}",2)
  # ``R^{eF}_{mN} += Λ_{im}^{ae} \bar y_{aN}^{iF}``
  vT_vOoV = load4idx(EC, "vT_vOoV")
  @tensoropt R2[e,F,m,N] += U2a[a,e,i,m] * vT_vOoV[a,N,i,F]
  vT_vOoV = NOTHING4idx
  t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{im}^{ae} \\bar y_{aN}^{iF}",2)
  # ``R^{eF}_{mN} += Λ_{IN}^{AF} \bar y_{Am}^{Ie}``
  vT_VoOv = load4idx(EC, "vT_VoOv")
  @tensoropt R2[e,F,m,N] += U2b[A,F,I,N] * vT_VoOv[A,m,I,e]
  vT_VoOv = NOTHING4idx
  t1 = print_time(EC, t1, "R^{eF}_{mN} += U_{IN}^{AF} \\bar y_{Am}^{Ie}",2)
  # ``R^{eF}_{mN} -= Λ_{mJ}^{aF} (\hat v_{aN}^{eJ} \red{- v_{kN}^{eD} T^{kJ}_{aD}})``
  int2 = load4idx(EC, "d_vOvO")
  if !dc
    T2ab = load4idx(EC, "T_vVoO")
    @tensoropt int2[a,N,e,J] -= oOvV[k,N,e,D] * T2ab[a,D,k,J]
  end
  @tensoropt R2[e,F,m,N] -= U2ab[a,F,m,J] * int2[a,N,e,J]
  t1 = print_time(EC, t1, "R^{eF}_{mN} -= U_{mJ}^{aF} (\\hat v_{aN}^{eJ} + v_{kN}^{eD} T^{kJ}_{aD})",2)
  # ``R^{eF}_{mN} -= Λ_{iN}^{eB} (\hat v_{mB}^{iF} \red{- v_{mL}^{cF} T^{iL}_{cB}})``
  int2 = load4idx(EC, "d_oVoV")
  if !dc
    @tensoropt int2[m,B,i,F] -= oOvV[m,L,c,F] * T2ab[c,B,i,L]
    T2ab = NOTHING4idx
  end
  @tensoropt R2[e,F,m,N] -= U2ab[e,B,i,N] * int2[m,B,i,F]
  t1 = print_time(EC, t1, "R^{eF}_{mN} -= U_{iN}^{eB} (\\hat v_{mB}^{iF} + v_{mL}^{cF} T^{iL}_{cB})",2)
  int2 = NOTHING4idx

  return R1a, R1b, R2
end

"""
    calc_1RDM(EC::ECInfo, U1, U2, T1, T2)

Calculate the 1RDM for the closed-shell CCSD or DCSD equations.

Return `D1[p,q]`=``D_p^q``, the 1RDM without T1 singles terms, 
and `dD1[p,q]`=``\\hat D_p^q``, the 1RDM with all T1 terms included.
"""
function calc_1RDM(EC::ECInfo, U1, U2, T1, T2)
  # 1RDM without T1 singles terms
  @tensoropt begin
    Doo[i,j] := -2.0 * U2[c,d,i,k] * T2[c,d,j,k]
    Dvv[a,b] := 2.0 * U2[b,c,k,l] * T2[a,c,k,l]
  end
  if length(U1) > 0
    @tensoropt begin
      Dov[i,a] := U1[a,i]
      Dvo[a,i] := 2.0 * U1[c,k] * T2[a,c,i,k] - U1[c,k] * T2[a,c,k,i]
    end
  else
    Dov = zeros(size(U2,3),size(U2,1))
    Dvo = zeros(size(U2,1),size(U2,3))
  end
  if length(T1) > 0
    # 1RDM (including all T1 terms)
    dDov = Dov
    @tensoropt begin
      dDoo[i,j] := Doo[i,j] - Dov[i,c] * T1[c,j]
      dDvv[a,b] := Dvv[a,b] + Dov[k,b] * T1[a,k]
    end
    @tensoropt dDvo[a,i] := Dvo[a,i] + 2.0 * T1[a,i] - Dvv[a,c] * T1[c,i] + dDoo[k,i] * T1[a,k]
  else
    dDov = Dov
    dDoo = Doo
    dDvv = Dvv
    dDvo = Dvo
  end
  SP = EC.space
  occ = SP['o']
  vir = SP['v']
  norb = n_orbs(EC)
  D1 = zeros(norb,norb)
  D1[occ,occ] = Doo
  D1[vir,vir] = Dvv
  D1[occ,vir] = Dov
  D1[vir,occ] = Dvo
  dD1 = zeros(norb,norb)
  dD1[occ,occ] = dDoo
  dD1[vir,vir] = dDvv
  dD1[occ,vir] = dDov
  dD1[vir,occ] = dDvo
  return D1, dD1
end

"""
    calc_1RDM(EC::ECInfo, U1, U1os, U2, U2ab, T1, T2, T2ab, spin)

Calculate the `spin`-1RDM for the unrestricted CCSD or DCSD equations.

`U1`, `U2`, `T1`, `T2` are the Lagrange multipliers and amplitudes for `spin`∈{`:α`,`:β`},
`U1os` are the singles Lagrange multipliers for opposite spin,
and `U2ab`, `T2ab` are the αβ Lagrange multipliers and amplitudes.

Return `D1[p,q]`=``D_p^q``, the 1RDM without T1 singles terms, 
and `dD1[p,q]`=``\\hat D_p^q``, the 1RDM with all T1 terms included.
"""
function calc_1RDM(EC::ECInfo, U1, U1os, U2, U2ab, T1, T2, T2ab, spin)
  # 1RDM without T1 singles terms
  @tensoropt begin 
    Doo[i,j] := -0.5 * U2[c,d,i,k] * T2[c,d,j,k]
    Dvv[a,b] := 0.5 * U2[b,c,k,l] * T2[a,c,k,l]
  end
  if length(U1) > 0
    @tensoropt begin
      Dov[i,a] := U1[a,i]
      Dvo[a,i] := U1[c,k] * T2[a,c,i,k] 
    end
  else
    Dov = zeros(size(U2,3),size(U2,1))
    Dvo = zeros(size(U2,1),size(U2,3))
  end
  if spin == :α
    @tensoropt begin 
      Doo[i,j] -= U2ab[c,D,i,K] * T2ab[c,D,j,K]
      Dvv[a,b] += U2ab[b,C,k,L] * T2ab[a,C,k,L]
    end
    if length(U1) > 0
      @tensoropt Dvo[a,i] += U1os[C,K] * T2ab[a,C,i,K] 
    end
  else
    @tensoropt begin 
      Doo[I,J] -= U2ab[c,D,k,I] * T2ab[c,D,k,J]
      Dvv[A,B] += U2ab[c,B,k,L] * T2ab[c,A,k,L]
    end
    if length(U1) > 0
      @tensoropt Dvo[A,I] += U1os[c,k] * T2ab[c,A,k,I] 
    end
  end
  if length(T1) > 0
    # 1RDM (including all T1 terms)
    dDov = Dov
    @tensoropt begin
      dDoo[i,j] := Doo[i,j] - Dov[i,c] * T1[c,j]
      dDvv[a,b] := Dvv[a,b] + Dov[k,b] * T1[a,k]
    end
    @tensoropt dDvo[a,i] := Dvo[a,i] + T1[a,i] - Dvv[a,c] * T1[c,i] + dDoo[k,i] * T1[a,k]
  else
    dDov = Dov
    dDoo = Doo
    dDvv = Dvv
    dDvo = Dvo
  end
  SP = EC.space
  
  occ = SP[space4spin('o', spin == :α)]
  vir = SP[space4spin('v', spin == :α)]
  norb = n_orbs(EC)
  D1 = zeros(norb,norb)
  D1[occ,occ] = Doo
  D1[vir,vir] = Dvv
  D1[occ,vir] = Dov
  D1[vir,occ] = Dvo
  dD1 = zeros(norb,norb)
  dD1[occ,occ] = dDoo
  dD1[vir,vir] = dDvv
  dD1[occ,vir] = dDov
  dD1[vir,occ] = dDvo
  return D1, dD1
end

"""
    calc_vT2_intermediates(EC::ECInfo, T2; dc=false)

  Calculate intermediates required in closed-shell [`calc_ccsd_vector_times_Jacobian`](@ref)
"""
function calc_vT2_intermediates(EC::ECInfo, T2; dc=false)
  t1 = time_ns()
  calc_3ext_times_T2(EC, T2, 'o', 'v', 'o', 'v')
  t1 = print_time(EC, t1, "calculate vT_ovoo",2)
end

"""
    calc_vT2_intermediates(EC::ECInfo, T2a, T2b, T2ab; dc=false)

  Calculate intermediates required in unrestricted [`calc_ccsd_vector_times_Jacobian`](@ref)
"""
function calc_vT2_intermediates(EC::ECInfo, T2a, T2b, T2ab; dc=false)
  t1 = time_ns()
  calc_3ext_times_T2(EC, T2a, T2b, T2ab)
  t1 = print_time(EC, t1, "calculate vT_ovoo, vT_OVOO, vT_oVoO, vT_OvOo",2)
  calc_focklike_vT2(EC, T2a, T2b, T2ab)
  t1 = print_time(EC, t1, "calculate vT_oo, vT_vv, vT_OO, vT_VV",2)
  calc_rings_vT2(EC, T2a, T2b, T2ab; dc)
  t1 = print_time(EC, t1, "calculate \\hat y_{am}^{ie} and \\hat y_{Bn}^{Jf} (all spin cases)",2)
end

"""
    calc_3ext_times_T2(EC::ECInfo, T2::AbstractArray, o1::Char='o', v1::Char='v', o2::Char='o', v2::Char='v')

  Calculate ``\\hat v_{mb}^{cd} T^{ij}_{cd}`` intermediate 
  required in [`calc_ccsd_vector_times_Jacobian`](@ref) 
  and store as `vT_ovoo[m,b,i,j]`.
"""
function calc_3ext_times_T2(EC::ECInfo, T2::AbstractArray, o1::Char='o', v1::Char='v', o2::Char='o', v2::Char='v')
  int2 = load4idx(EC, "d_"*v2*o1*v2*v1)
  @tensoropt vT_ovoo[m,b,i,j] := int2[b,m,d,c] * T2[d,c,j,i]
  save!(EC, "vT_"*o1*v2*o1*o2, vT_ovoo)
end

"""
    calc_3ext_times_T2(EC::ECInfo, T2a::AbstractArray, T2b::AbstractArray, T2ab::AbstractArray)

  Calculate ``\\hat v_{mb}^{cd} T^{ij}_{cd}`` intermediates 
  required in [`calc_ccsd_vector_times_Jacobian`](@ref) 
  for αα, ββ, and αβ amplitudes 
  and store as `vT_ovoo[m,b,i,j]`, `vT_OVOO[M,B,I,J]`, 
  `vT_oVoO[m,B,i,J]` and `vT_OvOo[M,b,I,j]`.
"""
function calc_3ext_times_T2(EC::ECInfo, T2a::AbstractArray, T2b::AbstractArray, T2ab::AbstractArray)
  calc_3ext_times_T2(EC, T2a, 'o', 'v', 'o', 'v')
  calc_3ext_times_T2(EC, T2b, 'O', 'V', 'O', 'V')
  int2 = load4idx(EC, "d_oVvV")
  @tensoropt vT_oVoO[m,B,i,J] := int2[m,B,c,D] * T2ab[c,D,i,J]
  save!(EC, "vT_oVoO", vT_oVoO)
  int2 = vT_oVoO = NOTHING4idx
  calc_3ext_times_T2(EC, T2ab, 'O', 'V', 'o', 'v')
end

"""
    calc_focklike_vT2(EC::ECInfo, T2a::AbstractArray, T2b::AbstractArray, T2ab::AbstractArray)

  Calculate the fock-like intermediates ``v_{kl}^{cd} T^{il}_{cd}`` and `v_{kl}^{cd} T^{kl}_{ad}` 
  required in [`calc_ccsd_vector_times_Jacobian`](@ref) 
  for αα/ββ, and αβ amplitudes.
  Store as `vT_oo[ki]`, `vT_vv[ac]`, `vT_OO[ki]` and `vT_VV[ac]`.
"""
function calc_focklike_vT2(EC::ECInfo, T2a::AbstractArray, T2b::AbstractArray, T2ab::AbstractArray)
  int2 = ints2(EC, "oovv")
  @tensoropt vT_oo[k,i] := int2[k,l,c,d] * T2a[c,d,i,l]
  @tensoropt vT_vv[a,c] := int2[k,l,c,d] * T2a[a,d,k,l]
  int2 = ints2(EC, "OOVV")
  @tensoropt vT_OO[k,i] := int2[k,l,c,d] * T2b[c,d,i,l]
  @tensoropt vT_VV[a,c] := int2[k,l,c,d] * T2b[a,d,k,l]
  int2 = ints2(EC, "oOvV")
  @tensoropt vT_oo[k,i] += int2[k,L,c,D] * T2ab[c,D,i,L]
  @tensoropt vT_vv[a,c] += int2[k,L,c,D] * T2ab[a,D,k,L]
  @tensoropt vT_OO[K,I] += int2[l,K,c,D] * T2ab[c,D,l,I]
  @tensoropt vT_VV[A,C] += int2[l,K,d,C] * T2ab[d,A,l,K]
  save!(EC, "vT_oo", vT_oo)
  save!(EC, "vT_vv", vT_vv)
  save!(EC, "vT_OO", vT_OO)
  save!(EC, "vT_VV", vT_VV)
end

"""
    calc_rings_vT2(EC::ECInfo, T2a::AbstractArray, T2b::AbstractArray, T2ab::AbstractArray; dc=false)

  Calculate the ring intermediates required in [`calc_ccsd_vector_times_Jacobian`](@ref)
  ``\\hat y_{am}^{ie} = \\hat v_{am}^{ie} - \\hat v_{am}^{ei} + 2x_{am}^{ie} + v_{mL}^{eD} T^{iL}_{aD}``
  and
  ``\\hat y_{Bn}^{Jf} = \\hat v_{nB}^{fJ} + 2x_{Bn}^{Jf} + v_{nL}^{fD} T^{LJ}_{DB}``
  and the spin-flip version of them,
  with 
  ``2x_{am}^{ie} = T^{il}_{ad} (v_{lm}^{de} \\red{- v_{ml}^{de}})``
  and
  ``2x_{Am}^{Ie} = T^{Il}_{Ad} (v_{lm}^{de} \\red{- v_{ml}^{de}})``
  
  The intermediates are stored as `vT_voov[amie]`, `vT_VOOV[AMIE]`, `vT_VoOv[BnJf]`, `vT_vOoV[bNjF]`,
"""
function calc_rings_vT2(EC::ECInfo, T2a::AbstractArray, T2b::AbstractArray, T2ab::AbstractArray; dc=false)
  # αα and βα 
  vT_voov = load4idx(EC, "d_voov")
  @tensoropt vT_voov[a,m,i,e] -= load4idx(EC, "d_vovo")[a,m,e,i]
  @tensoropt vT_VoOv[B,n,J,f] := load4idx(EC, "d_oVvO")[n,B,f,J]
  # add x terms
  oovv = ints2(EC, "oovv")
  if dc
    int2 = oovv
  else
    @tensoropt int2[k,l,c,d] := oovv[k,l,c,d] - oovv[l,k,c,d]
  end
  oovv = nothing
  @tensoropt vT_voov[a,m,i,e] += T2a[a,d,i,l] * int2[l,m,d,e]
  @tensoropt vT_VoOv[A,m,I,e] += T2ab[d,A,l,I] * int2[l,m,d,e]

  # ββ and αβ
  vT_VOOV = load4idx(EC, "d_VOOV")
  @tensoropt vT_VOOV[A,M,I,E] -= load4idx(EC, "d_VOVO")[A,M,E,I]
  vT_vOoV = load4idx(EC, "d_vOoV")
  # add x terms
  oovv = ints2(EC, "OOVV")
  if dc
    int2 = oovv
  else
    @tensoropt int2[K,L,C,D] := oovv[K,L,C,D] - oovv[L,K,C,D]
  end
  oovv = nothing
  @tensoropt vT_VOOV[A,M,I,E] += T2b[A,D,I,L] * int2[L,M,D,E]
  @tensoropt vT_vOoV[a,M,i,E] += T2ab[a,D,i,L] * int2[L,M,D,E]

  # add v_{kL}^{cD} terms
  int2 = ints2(EC, "oOvV")
  @tensoropt vT_voov[a,m,i,e] += T2ab[a,D,i,L] * int2[m,L,e,D]
  @tensoropt vT_VOOV[A,M,I,E] += T2ab[d,A,l,I] * int2[l,M,d,E]
  @tensoropt vT_VoOv[A,m,I,e] += T2b[A,D,I,L] * int2[m,L,e,D]
  @tensoropt vT_vOoV[a,M,i,E] += T2a[d,a,l,i] * int2[l,M,d,E]
  save!(EC, "vT_voov", vT_voov)
  save!(EC, "vT_VOOV", vT_VOOV)
  save!(EC, "vT_VoOv", vT_VoOv)
  save!(EC, "vT_vOoV", vT_vOoV)
end

"""
    calc_correlation_norm(EC::ECInfo, U1, U2)

  Calculate the norm of the correlation part of the CCSD or DCSD equations
  using Lagrange multipliers and amplitudes.

  Return ``⟨Λ|Ψ⟩ = ⟨Λ_1|T_1⟩ + ⟨Λ_2|T_2+\\frac{1}{2}T_1 T_1⟩``.
"""
function calc_correlation_norm(EC::ECInfo, U1, U2)
  T2 = load4idx(EC, "T_vvoo")
  Norm = 0.0
  @tensoropt Norm += U2[a,b,i,j] * T2[a,b,i,j]
  if length(U1) > 0
    T1 = load2idx(EC, "T_vo")
    if length(T1) > 0
      @tensoropt Norm += U2[a,b,i,j] * T1[a,i] * T1[b,j]
      @tensoropt Norm += U1[a,i] * T1[a,i]
    end
  end
  return Norm
end

"""
    calc_correlation_norm(EC::ECInfo, U1a, U1b, U2a, U2b, U2ab)

  Calculate the norm of the correlation part of the UCCSD or UDCSD equations
  using Lagrange multipliers and amplitudes.

  Return ``⟨Λ|Ψ⟩ = ⟨Λ_1|T_1⟩ + ⟨Λ_2|T_2+\\frac{1}{2}T_1 T_1⟩``.
"""
function calc_correlation_norm(EC::ECInfo, U1a, U1b, U2a, U2b, U2ab)
  T2 = load4idx(EC, "T_vvoo")
  Norm = 0.0
  @tensoropt Norm += 0.25*(U2a[a,b,i,j] * T2[a,b,i,j])
  T2 = load4idx(EC, "T_VVOO")
  @tensoropt Norm += 0.25*(U2b[a,b,i,j] * T2[a,b,i,j])
  T2 = load4idx(EC, "T_vVoO")
  @tensoropt Norm += U2ab[a,b,i,j] * T2[a,b,i,j]
  if length(U1a) > 0 || length(U1b) > 0
    T1a = load2idx(EC, "T_vo")
    T1b = load2idx(EC, "T_VO")
    if length(T1a) > 0
      @tensoropt Norm += 0.5*(U2a[a,b,i,j] * T1a[a,i] * T1a[b,j])
      @tensoropt Norm += U1a[a,i] * T1a[a,i]
    end
    if length(T1b) > 0
      @tensoropt Norm += 0.5*(U2b[a,b,i,j] * T1b[a,i] * T1b[b,j])
      @tensoropt Norm += U1b[a,i] * T1b[a,i]
    end
    if length(T1a) > 0 && length(T1b) > 0
      @tensoropt Norm += U2ab[a,b,i,j] * T1a[a,i] * T1b[b,j]
    end
  end
  return Norm
end

"""
    calc_lm_cc(EC::ECInfo, method::ECMethod)

  Calculate coupled cluster Lagrange multipliers.

  Exact specification of the method is given by `method`.
"""
function calc_lm_cc(EC::ECInfo, method::ECMethod)
  print_info(method_name(method)*" Lagrange multipliers")
  highest_full_exc = max_full_exc(method)
  if highest_full_exc > 2
    error("only implemented upto doubles")
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
    lm_cc_iterations!((T1a,T1b), (T2a,T2b,T2ab), EC, method)
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
    lm_cc_iterations!((T1,), (T2,), EC, method)
  end
end

function lm_cc_iterations!(LMs1, LMs2, EC::ECInfo, method::ECMethod)
  dc = (method.theory == "DC" || last(method.theory,2) == "DC")
  if is_unrestricted(method) || has_prefix(method, "R")
    @assert (length(LMs1) == 2) && (length(LMs2) == 3)
  else
    @assert (length(LMs1) == 1) && (length(LMs2) == 1) 
  end
  LMs = (LMs1..., LMs2...)
  # dress integrals
  t1 = time_ns()
  calc_dressed_ints(EC, LMs1...; calc_d_vovv=true)
  t1 = print_time(EC, t1, "dressing integrals",2)
  calc_vT2_intermediates(EC, LMs2...; dc)

  diis = Diis(EC)
  transform_amplitudes2lagrange_multipliers!(LMs1, LMs2)

  do_sing = (method.exclevel[1] == :full)

  NormR1 = 0.0
  NormLM1::Float64 = 0.0
  NormLM2::Float64 = 0.0
  ΛTNorm = 0.0
  converged = false
  thren = sqrt(EC.options.cc.thr) * EC.options.cc.conven
  t0 = time_ns()
  println("Iter     SqNorm     Corr.Norm   Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    Res = calc_ccsd_vector_times_Jacobian(EC, LMs...; dc)
    @assert typeof(Res) == typeof(LMs)
    Res1 = Res[1:length(LMs1)]
    Res2 = Res[length(LMs1)+1:end]
    t1 = print_time(EC, t1, "residual", 2)
    update_doubles!(EC, LMs2..., Res2...)
    NormLM2 = calc_contra_doubles_norm(LMs2...)
    NormR2 = calc_contra_doubles_norm(Res2...)
    if do_sing
      NormLM1 = calc_contra_singles_norm(LMs1...)
      NormR1 = calc_contra_singles_norm(Res1...)
      update_singles!(EC, LMs1..., Res1...)
    end
    perform!(diis, LMs, Res)
    if do_sing
      save_current_singles(EC, LMs1..., prefix="U")
    end
    save_current_doubles(EC, LMs2..., prefix="U")
    ΛTNorm_prev = ΛTNorm
    ΛTNorm = calc_correlation_norm(EC, LMs...)
    NormR = NormR1 + NormR2
    NormLM = 1.0 + NormLM1 + NormLM2
    output_iteration(it, NormR, time_ns()-t0, NormLM, ΛTNorm)
    if NormR < EC.options.cc.thr && abs(ΛTNorm-ΛTNorm_prev) < thren
      converged = true
      break
    end
  end
  if !converged
    println("WARNING: CC-LM iterations did not converge!")
  end
  if do_sing
    try2save_singles!(EC, LMs1...; type="LM")
  end
  try2save_doubles!(EC, LMs2...; type="LM")
  println()
  output_norms("LM1"=>NormLM1, "LM2"=>NormLM2)
  println()
  return
end
