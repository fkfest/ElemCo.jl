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
  
  T1 = load(EC, "T_vo")
  T2 = load(EC, "T_vvoo")
  # Calculate 1RDM intermediates
  D1, dD1 = calc_1RDM(U1, U2, T1, T2)

  fock = load(EC,"f_mm")
  dfock = load(EC,"df_mm")
  fov = fock[SP['o'],SP['v']]
  dfov = dfock[SP['o'],SP['v']]
  if length(U1) > 0
    @tensoropt R1[e,m] := 2.0 * fov[m,e]
  else
    R1 = []
  end

  oovv = ints2(EC,"oovv")
  @tensoropt R2[e,f,m,n] := 2.0 * oovv[m,n,e,f] - oovv[n,m,e,f]
  int2 = load(EC, "d_oooo")
  if !dc
    @tensoropt int2[m,n,i,j] += oovv[m,n,c,d] * T2[c,d,i,j]
  end
  # ``R^{ef}_{mn} += Λ_{ij}^{ef} (\hat v_{mn}^{ij} \red{+ v_{nm}^{cd} T^{ij}_{cd}})``
  @tensoropt R2[e,f,m,n] += int2[m,n,i,j] * U2[e,f,i,j]
  # the 4-external part
  if EC.options.cc.use_kext
    # the `kext` part
    int2 = integ2(EC.fd)
    if ndims(int2) == 4
      if EC.options.cc.triangular_kext
        trioo = [CartesianIndex(i,j) for j in 1:nocc for i in 1:j]
        dU2 = calc_dU2(EC, T1, U2)[:,:,trioo]
        # ``K_{mn}^{rs} = \hat U_{mn}^{pq} v_{pq}^{rs}``
        @tensoropt Kmmx[r,s,x] := int2[p,q,r,s] * dU2[p,q,x]
        dU2 = nothing
        Kmmoo = Array{Float64}(undef,norb,norb,nocc,nocc)
        Kmmoo[:,:,trioo] = Kmmx
        trioor = CartesianIndex.(reverse.(Tuple.(trioo)))
        @tensor Kmmoo[:,:,trioor][p,q,x] = Kmmx[q,p,x]
        Kmmx = nothing
      else
        dU2 = calc_dU2(EC, T1, U2)
        # ``K_{mn}^{rs} = \hat U_{mn}^{pq} v_{pq}^{rs}``
        @tensoropt Kmmoo[r,s,m,n] := int2[p,q,r,s] * dU2[p,q,m,n]
        dU2 = nothing
      end
    else
      # last two indices of integrals are stored as upper triangular 
      dU2 = calc_dU2(EC, T1, U2)
      # ``K_{mn}^{rs} = \hat U_{mn}^{pq} v_{pq}^{rs}``
      @tensoropt Kxoo[x,m,n] := int2[p,q,x] * dU2[p,q,m,n]
      dU2 = nothing
      Kmmoo = Array{Float64}(undef,norb,norb,nocc,nocc)
      tripp = [CartesianIndex(i,j) for j in 1:norb for i in 1:j]
      Kmmoo[tripp,:,:] = Kxoo
      trippr = CartesianIndex.(reverse.(Tuple.(tripp)))
      @tensor Kmmoo[trippr,:,:][x,m,n] = Kxoo[x,n,m]
      Kxoo = nothing
    end
    # ``R^{ef}_{mn} += K_{mn}^{rs} δ_r^e δ_s^f``
    R2 += Kmmoo[SP['v'],SP['v'],:,:]
    # ``R^e_m += 2 K_{mj}^{rs} δ_r^e (δ_s^j + δ_s^b T^j_b)``
    if length(U1) > 0
      @tensoropt R1[e,m] += 2.0 * Kmmoo[SP['v'],SP['o'],:,:][e,j,m,j] 
      if length(T1) > 0
        @tensoropt R1[e,m] += 2.0 * Kmmoo[SP['v'],SP['v'],:,:][e,b,m,j] * T1[b,j]
      end 
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
  @tensoropt tT2[e,f,m,n] := 2.0 * T2[e,f,m,n] - T2[e,f,n,m]
  # ``x_m^i = \tilde T^{il}_{cd} v_{ml}^{cd}``
  @tensoropt xoo[m,i] := tT2[c,d,i,l] * oovv[m,l,c,d]
  # ``x_a^e = \tilde T^{kl}_{ac} v_{kl}^{ec}``
  @tensoropt xvv[a,e] := tT2[a,c,k,l] * oovv[k,l,e,c]
  int2 = load(EC, "d_voov")
  # ``tR^{ef}_{mn} += Λ_{in}^{af} * (\hat v_{am}^{ie} + v_{km}^{ce} \tilde T^{ik}_{ac})``
  @tensoropt int2[a,m,i,e] += oovv[k,m,c,e] * tT2[a,c,i,k]
  @tensoropt tR2[e,f,m,n] += U2[a,f,i,n] * int2[a,m,i,e] 
  if length(U1) > 0
    # ``tR^{ef}_{mn} += 0.5 Λ_m^e \hat f_n^f``
    @tensoropt tR2[e,f,m,n] += 0.5 * U1[e,m] * dfov[n,f]
  end
  int2 = load(EC, "d_vovv")
  if length(U1) > 0
    # ``tR^{ef}_{mn} += 0.5 Λ_n^a \hat v_{am}^{fe}``
    @tensoropt tR2[e,f,m,n] += 0.5 * U1[a,n] * int2[a,m,f,e]
  end
  oovo = load(EC, "d_oovo")
  if length(U1) > 0
    # ``tR^{ef}_{mn} -= 0.5 Λ_i^f \hat v_{mn}^{ei}``
    @tensoropt tR2[e,f,m,n] -= 0.5 * U1[f,i] * oovo[m,n,e,i]
  end
  # ``pR^{ef}_{mn} = 2 tR^{ef}_{mn} - tR^{ef}_{nm}``
  @tensoropt pR2[e,f,m,n] := 2.0 * tR2[e,f,m,n] - tR2[e,f,n,m]  
  tR2 = nothing
  # calc ``D_{ib}^{aj} = Λ_{ik}^{ac} \tilde T^{kj}_{cb}``
  @tensoropt D2[a,b,i,j] := U2[a,c,i,k] * tT2[c,b,k,j]
  tT2 = nothing
  if length(U1) > 0
    # ``R^{e}_{m} += 2 D_{md}^{al} \hat v_{al}^{ed}``
    @tensoropt R1[e,m] += 2.0 * D2[a,d,m,l] * int2[a,l,e,d]
    # ``R^{e}_{m} += -2 D_{id}^{el} \hat v_{ml}^{id}``
    @tensoropt R1[e,m] -= 2.0 * D2[e,d,i,l] * oovo[l,m,d,i]
  end
  if !dc
    # ``pR^{ef}_{mn} -= D_{nc}^{fl} v_{ml}^{ce}``
    @tensoropt pR2[e,f,m,n] -= D2[f,c,n,l] * oovv[m,l,c,e]
  end
  # calc ``\bar D_{ib}^{aj} = Λ_{ik}^{ac} T^{kj}_{cb}+ Λ_{ik}^{ca} T^{kj}_{bc}``
  @tensoropt begin 
    D2[a,b,i,j] = U2[a,c,i,k] * T2[c,b,k,j]
    D2[a,b,i,j] += U2[c,a,i,k] * T2[b,c,k,j]
  end
  if !dc
    # ``pR^{ef}_{mn} += \bar D_{nd}^{ek} v_{km}^{fd}``
    @tensoropt pR2[e,f,m,n] += D2[e,d,n,k] * oovv[k,m,f,d]
  end
  if length(U1) > 0
    # ``R^{e}_{m} -= 2 \bar D_{mc}^{ak} \hat v_{ak}^{ce}``
    @tensoropt R1[e,m] -= 2.0 * D2[a,c,m,k] * int2[a,k,c,e]
    # ``R^{e}_{m} += 2 \bar D_{ic}^{ek} \hat v_{km}^{ic}``
    @tensoropt R1[e,m] += 2.0 * D2[e,c,i,k] * oovo[m,k,c,i]
  end
  int2 = nothing
  D2 = nothing
  # calc ``D_{ij}^{kl} = Λ_{ij}^{cd} T^{kl}_{cd}``
  @tensoropt D2[i,j,k,l] := U2[c,d,i,j] * T2[c,d,k,l]
  if length(U1) > 0
    # ``R_e^m += 2 D_{mj}^{kl} \hat v_{kl}^{ej}``
    @tensoropt R1[e,m] += 2.0 * D2[m,j,k,l] * oovo[k,l,e,j]
  end
  oovo = nothing
  if !dc
    # ``R^{ef}_{mn} += D_{mn}^{kl} v_{kl}^{ef}``
    @tensoropt R2[e,f,m,n] += D2[m,n,k,l] * oovv[k,l,e,f]
  end
  D2 = nothing
  if length(U1) > 0
    # ``R^e_m -= 2 Λ_{ij}^{eb} (v_{mb}^{cd} T^{ij}_{cd})`` 
    # ``(v_{mb}^{cd} T^{ij}_{cd})`` has to be precalculated
    vT_ovoo = load(EC, "vT_ovoo")
    @tensoropt R1[e,m] -= 2.0 * U2[e,b,i,j] * vT_ovoo[m,b,i,j]
    vT_ovoo = nothing
  end
  fac = dc ? 0.5 : 1.0
  @tensoropt begin 
    # ``pR^{ef}_{mn} += Λ_{mn}^{af} (\hat f_a^e - f * x_a^e)``
    pR2[e,f,m,n] += U2[a,f,m,n] * (dfock[SP['v'],SP['v']][a,e] - fac * xvv[a,e])
    # ``pR^{ef}_{mn} -= Λ_{in}^{ef} (\hat f_m^i + f * x_m^i)``
    pR2[e,f,m,n] -= U2[e,f,i,n] * (dfock[SP['o'],SP['o']][m,i] + fac * xoo[m,i])
  end
  int2 = load(EC, "d_vovo")
  # ``pR^{ef}_{mn} -= Λ_{in}^{af} \hat v_{am}^{ei}``
  @tensoropt pR2[e,f,m,n] -= U2[a,f,i,n] * int2[a,m,e,i]
  # ``pR^{ef}_{mn} -= Λ_{in}^{eb} \hat v_{mb}^{if}``
  @tensoropt pR2[e,f,m,n] -= U2[e,b,i,n] * int2[b,m,f,i]
  # ``R^{ef}_{mn} += pR^{ef}_{mn} + pR^{fe}_{nm}``
  @tensoropt R2[e,f,m,n] += pR2[e,f,m,n] + pR2[f,e,n,m]
  pR2 = nothing

  if length(U1) > 0
    # ``R_e^m += \hat D_p^q (2 v_{mq}^{ep} - v_{mq}^{pe})``
    int2 = ints2(EC,"momm")
    @tensoropt R1[e,m] += dD1[p,q] * (2.0 * int2[:,:,:,SP['v']][q,m,p,e] - int2[:,:,SP['v'],:][q,m,e,p])
    # ``R_e^m -= 2 Λ_{ij}^{eb} \hat v_{mb}^{ij}``
    int2 = load(EC, "d_vooo")
    @tensoropt R1[e,m] -= 2.0 * U2[e,b,i,j] * int2[b,m,j,i]
    @tensoropt begin
      # ``R_e^m += D_m^k \hat f_k^e - D_d^e \hat f_m^d``
      R1[e,m] += D1[SP['o'],SP['o']][m,k] * dfov[k,e]
      R1[e,m] -= D1[SP['v'],SP['v']][d,e] * dfov[m,d]
      # ``R_e^m += Λ_m^a (\hat f_a^e - x_a^e) - Λ_i^e (\hat f_m^i + x_m^i)``
      R1[e,m] += U1[a,m] * (dfock[SP['v'],SP['v']][a,e] - xvv[a,e])
      R1[e,m] -= U1[e,i] * (dfock[SP['o'],SP['o']][m,i] + xoo[m,i])
    end
  end
  return R1, R2
end

"""
    calc_dU2(EC::ECInfo, T1, U2)

Calculate the "dressed" ``Λ_2`` for the closed-shell CCSD/DCSD.

Return `dU2[p,q,m,n]`=``Λ_{mn}^{ab}δ_a^p δ_b^q - Λ_{mn}^{ab}T^i_a δ_i^p δ_b^q - Λ_{mn}^{ab}δ_a^p T^j_b δ_j^q + Λ_{mn}^{ab}T^i_a T^j_b δ_i^p δ_j^q``.
"""
function calc_dU2(EC::ECInfo, T1, U2)
  SP = EC.space
  norb = n_orbs(EC)
  nocc = n_occ_orbs(EC)
  if length(T1) > 0
    # dU2 = Array{Float64}(undef,norb,norb,nocc,nocc)
    dU2 = zeros(norb,norb,nocc,nocc)
  else
    dU2 = zeros(norb,norb,nocc,nocc)
  end
  dU2[SP['v'],SP['v'],:,:] = U2 
  if length(T1) > 0
    @tensoropt dUovoo[i,b,m,n] := U2[a,b,m,n] * T1[a,i]
    @tensoropt dU2[SP['v'],SP['o'],:,:][a,j,m,n] = - U2[a,b,m,n] * T1[b,j]
    @tensoropt dU2[SP['o'],SP['o'],:,:][i,j,m,n] = dUovoo[i,b,m,n] * T1[b,j]
    dU2[SP['o'],SP['v'],:,:] = -dUovoo
  end
  return dU2
end

"""
    calc_1RDM(U1, U2, T1, T2)

Calculate the 1RDM for the closed-shell CCSD or DCSD equations.

Return `D1[p,q]`=``D_p^q``, the 1RDM without T1 singles terms, 
and `dD1[p,q]`=``\\hat D_p^q``, the 1RDM with all T1 terms included.
"""
function calc_1RDM(U1, U2, T1, T2)
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
  D1 = [Doo Dov; Dvo Dvv]
  dD1 = [dDoo dDov; dDvo dDvv]
  return D1, dD1
end

"""
    calc_3ext_times_T2(EC::ECInfo)

  Calculate ``\\hat v_{mb}^{cd} T^{ij}_{cd}`` intermediate 
  required in [`calc_ccsd_vector_times_Jacobian`](@ref) 
  and store as `vT_ovoo[m,b,i,j]`.
"""
function calc_3ext_times_T2(EC::ECInfo)
  T2 = load(EC, "T_vvoo")
  int2 = load(EC, "d_vovv")
  @tensoropt vT_ovoo[m,b,i,j] := int2[b,m,d,c] * T2[d,c,j,i]
  save!(EC, "vT_ovoo", vT_ovoo)
end

"""
    calc_correlation_norm(EC::ECInfo, U1, U2)

  Calculate the norm of the correlation part of the CCSD or DCSD equations
  using Lagrange multipliers and amplitudes.

  Return ``⟨Λ|Ψ⟩ = ⟨Λ_1|T_1⟩ + ⟨Λ_2|T_2+\\frac{1}{2}T_1 T_1⟩``.
"""
function calc_correlation_norm(EC::ECInfo, U1, U2)
  T2 = load(EC, "T_vvoo")
  Norm = 0.0
  @tensoropt Norm += U2[a,b,i,j] * T2[a,b,i,j]
  if length(U1) > 0
    T1 = load(EC, "T_vo")
    if length(T1) > 0
      @tensoropt Norm += U2[a,b,i,j] * T1[a,i] * T1[b,j]
      @tensoropt Norm += U1[a,i] * T1[a,i]
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
  dc = (method.theory == "DC" || last(method.theory,2) == "DC")
  print_info(method_name(method)*" Lagrange multipliers")
  LMs, exc_ranges = starting_amplitudes(EC, method)
  singles, doubles, triples = exc_ranges[1:3]
  if is_unrestricted(method)
    @assert (length(singles) == 2) && (length(doubles) == 3) && (length(triples) == 4)
  else
    @assert (length(singles) == 1) && (length(doubles) == 1) && (length(triples) == 1)
  end
  transform_amplitudes2lagrange_multipliers!(LMs, exc_ranges)
  diis = Diis(EC)

  # dress integrals
  T1 = load(EC, "T_vo")
  calc_dressed_ints(EC, T1; calc_d_vovv=true)
  calc_3ext_times_T2(EC)
  NormR1 = 0.0
  NormLM1 = 0.0
  NormLM2 = 0.0
  do_sing = (method.exclevel[1] == :full)
  ΛTNorm = 0.0
  converged = false
  t0 = time_ns()
  println("Iter     SqNorm     Corr.Norm   Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    Res = calc_ccsd_vector_times_Jacobian(EC, LMs...; dc)
    t1 = print_time(EC, t1, "residual", 2)
    update_doubles!(EC, LMs[doubles]..., Res[doubles]...)
    NormLM2 = calc_contra_doubles_norm(LMs[doubles]...)
    NormR2 = calc_contra_doubles_norm(Res[doubles]...)
    if do_sing
      NormLM1 = calc_contra_singles_norm(LMs[singles]...)
      NormR1 = calc_contra_singles_norm(Res[singles]...)
      update_singles!(EC, LMs[singles]..., Res[singles]...)
    end
    LMs = perform(diis, LMs, Res)
    if do_sing
      save_current_singles(EC, LMs[singles]..., prefix="U")
    end
    save_current_doubles(EC, LMs[doubles]..., prefix="U")
    ΛTNorm = calc_correlation_norm(EC, LMs...)
    NormR = NormR1 + NormR2
    NormLM = 1.0 + NormLM1 + NormLM2
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %10.2e %8.2f \n" it NormLM ΛTNorm NormR tt
    flush(stdout)
    if NormR < EC.options.cc.thr
      converged = true
      break
    end
  end
  if !converged
    println("WARNING: CC-LM iterations did not converge!")
  end
  if do_sing
    try2save_singles!(EC, LMs[singles]...; type="LM")
  end
  try2save_doubles!(EC, LMs[doubles]...; type="LM")
  println()
  @printf "Sq.Norm of LM1: %12.8f Sq.Norm of LM2: %12.8f \n" NormLM1 NormLM2
  println()
  flush(stdout)
end