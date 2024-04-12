# triples routines

"""
    calc_pertT(EC::ECInfo, method::ECMethod; save_t3=false)

  Calculate (T) correction for [Λ][U]CCSD(T)

  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`. 
  If `save_t3` is true, the T3 amplitudes are saved in `T_vvvooo` file (only for closed-shell).
"""
function calc_pertT(EC::ECInfo, method::ECMethod; save_t3=false)
  if is_unrestricted(method) || has_prefix(method, "R")
    # unrestricted/restricted-open-shell branch
    if has_prefix(method, "Λ")
      @assert !save_t3 "Saving perturbative triples not implemented for ΛUCCSD(T)"
      return calc_ΛpertT_unrestricted(EC)
    else
      @assert !save_t3 "Saving perturbative triples not implemented for UCCSD(T)"
      return calc_pertT_unrestricted(EC)
    end
  else
    # closed-shell branch
    if has_prefix(method, "Λ")
      @assert !save_t3 "Saving perturbative triples not implemented for ΛCCSD(T)"
      return calc_ΛpertT_closed_shell(EC)
    else 
      return calc_pertT_closed_shell(EC; save_t3)
    end
  end
end

"""
    calc_pertT_closed_shell(EC::ECInfo; save_t3=false)

  Calculate (T) correction for closed-shell CCSD.

  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_pertT_closed_shell(EC::ECInfo; save_t3=false)
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

  X = zeros(nvir,nvir,nvir)
  Kijk = zeros(nvir,nvir,nvir)

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
          Kijk[a,b,c] = T2ij[a,d] * vvvk[b,c,d]
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
        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        if save_t3
          ijk = uppertriangular(i,j,k)
          T3[:,:,:,ijk] = Kijk
          for abc ∈ CartesianIndices(Kijk)
            a,b,c = Tuple(abc)
            T3[abc,ijk] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
          end
        end
        @tensoropt  X[a,b,c] = 4.0*Kijk[a,b,c] - 2.0*Kijk[a,c,b] - 2.0*Kijk[c,b,a] - 2.0*Kijk[b,a,c] + Kijk[c,a,b] + Kijk[b,c,a]
        for abc ∈ CartesianIndices(X)
          a,b,c = Tuple(abc)
          X[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
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
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_ΛpertT_closed_shell(EC::ECInfo)

  Calculate (T) correction for closed-shell ΛCCSD(T).

  The amplitudes are stored in `T_vvoo` file, 
  and the Lagrangian multipliers are stored in `U_vvoo` file.
  Return ( `ET3`=(T) energy, `ET3b`=[T] energy) `NamedTuple`.
"""
function calc_ΛpertT_closed_shell(EC::ECInfo)
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

        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        for abc ∈ CartesianIndices(X)
          a,b,c = Tuple(abc)
          X[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
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
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_pertT_unrestricted(EC::ECInfo)

  Calculate (T) correction for UCCSD.

  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_pertT_unrestricted(EC::ECInfo)
  T1a = load(EC,"T_vo")
  T2 = load(EC,"T_vvoo")
  En3a, Enb3a = calc_pertT_samespin(EC, T1a, T2, :α)

  T1b = load(EC,"T_VO")
  T2ba = permutedims(load(EC,"T_vVoO"), [2,1,4,3])
  En3ab, Enb3ab = calc_pertT_mixedspin(EC, T1a, T2, T1b, T2ba, :α)
  T2ba = nothing

  T2 = load(EC,"T_VVOO")
  En3b, Enb3b = calc_pertT_samespin(EC, T1b, T2, :β)

  T2ab = load(EC,"T_vVoO")
  En3ba, Enb3ba = calc_pertT_mixedspin(EC, T1b, T2, T1a, T2ab, :β)

  En3 = En3a + En3b + En3ab + En3ba 
  Enb3 = Enb3a + Enb3b + Enb3ab + Enb3ba
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_ΛpertT_unrestricted(EC::ECInfo)

  Calculate (T) correction for ΛUCCSD(T).

  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_ΛpertT_unrestricted(EC::ECInfo)
  U1a = load(EC,"U_vo")
  U1b = load(EC,"U_VO")
  T2 = load(EC,"T_vvoo")
  U2 = load(EC,"U_vvoo")
  En3a, Enb3a = calc_ΛpertT_samespin(EC, T2, U1a, U2, :α)

  T2ba = permutedims(load(EC,"T_vVoO"), [2,1,4,3])
  U2ba = permutedims(load(EC,"U_vVoO"), [2,1,4,3])
  En3ab, Enb3ab = calc_ΛpertT_mixedspin(EC, T2, T2ba, U1a, U2, U1b, U2ba, :α)
  T2ba = nothing
  U2ba = nothing

  T2 = load(EC,"T_VVOO")
  U2 = load(EC,"U_VVOO")
  En3b, Enb3b = calc_ΛpertT_samespin(EC, T2, U1b, U2, :β)

  T2ab = load(EC,"T_vVoO")
  U2ab = load(EC,"U_vVoO")
  En3ba, Enb3ba = calc_ΛpertT_mixedspin(EC, T2, T2ab, U1b, U2, U1a, U2ab, :β)

  En3 = En3a + En3b + En3ab + En3ba 
  Enb3 = Enb3a + Enb3b + Enb3ab + Enb3ba
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_pertT_samespin(EC::ECInfo, T1, T2, spin::Symbol)

  Calculate same-spin (T) correction for UCCSD(T) (i.e., ααα or βββ).
  `spin` ∈ (:α,:β)

  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_pertT_samespin(EC::ECInfo, T1, T2, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  o = space4spin('o', spin==:α)
  v = space4spin('v', spin==:α)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  # ``0.5(v_{ai}^{kj} - v_{ai}^{jk})``
  vooo = 0.5*ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  nocc = length(SP[o])
  nvir = length(SP[v])
  ϵo, ϵv = orbital_energies(EC, spin)

  T = zeros(nvir,nvir,nvir)
  Kijk = zeros(nvir,nvir,nvir)

  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  IntY = zeros(nvir,nocc)
  for k = 3:nocc 
    for j = 1:k-1
      for i = 1:j-1
        T2ij = @view T2[:,:,i,j]
        T2ik = @view T2[:,:,i,k]
        T2jk = @view T2[:,:,j,k]
        T2i = @view T2[:,:,:,i]
        T2j = @view T2[:,:,:,j]
        T2k = @view T2[:,:,:,k]
        vvvk = @view vvvo[:,:,:,k]
        vvvj = @view vvvo[:,:,:,j]
        vvvi = @view vvvo[:,:,:,i]
        vokj = @view vooo[:,:,k,j]
        voki = @view vooo[:,:,k,i]
        voij = @view vooo[:,:,i,j]
        @tensoropt begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = T2ij[a,d] * vvvk[b,c,d]
          Kijk[a,b,c] += T2ik[a,d] * vvvj[c,b,d]
          Kijk[a,b,c] += T2jk[d,c] * vvvi[b,a,d]

          Kijk[a,b,c] -= T2i[b,a,l] * vokj[c,l]
          Kijk[a,b,c] -= T2j[a,b,l] * voki[c,l]
          Kijk[a,b,c] -= T2k[b,c,l] * voij[a,l]
        end
        # antisymmetrize K = A(a,b,c) Kijk[a,b,c]
        @tensoropt  T[a,b,c] = Kijk[a,b,c] - Kijk[c,b,a]
        @tensoropt Kijk[a,b,c] = T[a,b,c] - T[b,a,c] - T[a,c,b]
        T .= Kijk
        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        for abc ∈ CartesianIndices(T)
          a,b,c = Tuple(abc)
          T[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
        end

        @tensoropt Enb3 += 1/6*(Kijk[a,b,c] * T[a,b,c])
        
        vv_jk = @view vv_oo[:,:,j,k]
        vv_ik = @view vv_oo[:,:,i,k]
        vv_ij = @view vv_oo[:,:,i,j]
        @tensoropt IntX[:,i][a] += T[a,b,c] * vv_jk[b,c]
        @tensoropt IntX[:,j][b] += T[a,b,c] * vv_ik[a,c]
        @tensoropt IntX[:,k][c] += T[a,b,c] * vv_ij[a,b]
        @tensoropt IntY[:,i][a] += T[a,b,c] * T2jk[b,c]
        @tensoropt IntY[:,j][b] += T[a,b,c] * T2ik[a,c]
        @tensoropt IntY[:,k][c] += T[a,b,c] * T2ij[a,b]
      end 
    end
  end
  # singles contribution
  @tensoropt En3 = T1[a,i] * IntX[a,i]
  # fock contribution
  m = space4spin('m', spin==:α)
  fov = load(EC,"f_"*m*m)[SP[o],SP[v]]
  @tensoropt En3 += 0.5 * (fov[i,a] * IntY[a,i])
  En3 += Enb3
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_pertT_mixedspin(EC::ECInfo, T1, T2, T1os, T2mix, spin::Symbol)

  Calculate mixed-spin (T) correction for UCCSD(T) (i.e., ααβ or ββα).

  `spin` ∈ (:α,:β)
  `T1` and `T2` are same-`spin` amplitudes, `T1os` are opposite-`spin` amplitudes,
  and `T2mix` are mixed-spin amplitudes with the *second* electron being `spin`,
  i.e., Tβα for `spin == :α` and Tαβ for `spin == :β`.
  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_pertT_mixedspin(EC::ECInfo, T1, T2, T1os, T2mix, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  isα = (spin == :α)
  o = space4spin('o', isα)
  v = space4spin('v', isα)
  O = space4spin('o', !isα)
  V = space4spin('v', !isα)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  # ``v_{ai}^{kj} - v_{ai}^{jk}``
  vooo = ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  if isα
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, o*O*v*V),[3,4,1,2])
    # ``v_{aB}^{cK}``
    vVvO = ints2(EC, v*V*v*O)
    # ``v_{Ab}^{Ck}``
    VvVo = permutedims(ints2(EC, v*V*o*V),[2,1,4,3])
    # ``v_{aI}^{kJ}``
    vOoO = ints2(EC, v*O*o*O)
    # ``v_{Ai}^{Kj}``
    VoOo = permutedims(ints2(EC, o*V*o*O),[2,1,4,3])
  else
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, O*o*V*v),[4,3,2,1])
    # ``v_{aB}^{cK}``
    vVvO = permutedims(ints2(EC, V*v*O*v),[2,1,4,3])
    # ``v_{Ab}^{Ck}``
    VvVo = ints2(EC, V*v*V*o)
    # ``v_{aI}^{kJ}``
    vOoO = permutedims(ints2(EC, O*v*O*o),[2,1,4,3])
    # ``v_{Ai}^{Kj}``
    VoOo = ints2(EC, V*o*O*o)
  end
  nocc = length(SP[o])
  nOcc = length(SP[O])
  nvir = length(SP[v])
  nVir = length(SP[V])
  ϵo, ϵv = orbital_energies(EC, spin)
  opspin = isα ? :β : :α
  ϵO, ϵV = orbital_energies(EC, opspin)

  T = zeros(nvir,nvir,nVir)
  Kijk = zeros(nvir,nvir,nVir)

  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  IntY = zeros(nvir,nocc)
  IntXos = zeros(nVir,nOcc)
  IntYos = zeros(nVir,nOcc)
  for K = 1:nOcc 
    T2K = T2mix[:,:,K,:]
    for j = 2:nocc
      for i = 1:j-1
        T2ij = @view T2[:,:,i,j]
        T2Ki = @view T2K[:,:,i]
        T2Kj = @view T2K[:,:,j]
        T2i = @view T2[:,:,:,i]
        T2j = @view T2[:,:,:,j]
        T2mixi = @view T2mix[:,:,:,i]
        T2mixj = @view T2mix[:,:,:,j]
        vVvK = @view vVvO[:,:,:,K]
        vvvi = @view vvvo[:,:,:,i]
        vvvj = @view vvvo[:,:,:,j]
        VvVi = @view VvVo[:,:,:,i]
        VvVj = @view VvVo[:,:,:,j]
        voij = @view vooo[:,:,i,j]
        vOjK = @view vOoO[:,:,j,K]
        vOiK = @view vOoO[:,:,i,K]
        VoKj = @view VoOo[:,:,K,j]
        VoKi = @view VoOo[:,:,K,i]
        @tensoropt begin
          # K_{abC}^{ijK} = v_{bC}^{dK} T^{ij}_{ad} + ...
          Kijk[a,b,C] = T2ij[a,d] * vVvK[b,C,d]
          Kijk[a,b,C] += T2Kj[C,d] * vvvi[b,a,d]
          Kijk[a,b,C] += T2Ki[C,d] * vvvj[a,b,d]
          Kijk[a,b,C] += T2Kj[D,b] * VvVi[C,a,D]
          Kijk[a,b,C] += T2Ki[D,a] * VvVj[C,b,D]
          Kijk[a,b,C] -= T2K[C,b,l] * voij[a,l]
          Kijk[a,b,C] -= T2mixi[C,a,L] * vOjK[b,L]
          Kijk[a,b,C] -= T2mixj[C,b,L] * vOiK[a,L]
        end
        # antisymmetrize ΔK = A(a,b) ΔKijk[a,b,C]
        Kijk -= permutedims(Kijk,[2,1,3])
        @tensoropt begin
          Kijk[a,b,C] -= T2i[b,a,l] * VoKj[C,l]
          Kijk[a,b,C] -= T2j[a,b,l] * VoKi[C,l]
        end
        T .= Kijk
        ϵoijK = ϵo[i] + ϵo[j] + ϵO[K]
        for abC ∈ CartesianIndices(T)
          a,b,C = Tuple(abC)
          T[abC] /= ϵoijK - ϵv[a] - ϵv[b] - ϵV[C]
        end

        @tensoropt Enb3 += 0.5 * (Kijk[a,b,C] * T[a,b,C])
        
        vV_jK = @view vV_oO[:,:,j,K]
        vV_iK = @view vV_oO[:,:,i,K]
        vv_ij = @view vv_oo[:,:,i,j]
        @tensoropt IntX[:,i][a] += T[a,b,C] * vV_jK[b,C]
        @tensoropt IntX[:,j][b] += T[a,b,C] * vV_iK[a,C]
        @tensoropt IntXos[:,K][C] += T[a,b,C] * vv_ij[a,b]
        @tensoropt IntY[:,i][a] += T[a,b,C] * T2Kj[C,b]
        @tensoropt IntY[:,j][b] += T[a,b,C] * T2Ki[C,a]
        @tensoropt IntYos[:,K][C] += T[a,b,C] * T2ij[a,b]
      end 
    end
  end
  # singles contribution
  @tensoropt En3 = T1[a,i] * IntX[a,i]
  @tensoropt En3 += T1os[A,I] * IntXos[A,I]
  # fock contribution
  m = space4spin('m', isα)
  fov = load(EC,"f_"*m*m)[SP[o],SP[v]]
  @tensoropt En3 += fov[i,a] * IntY[a,i]
  M = space4spin('m', !isα)
  fOV = load(EC,"f_"*M*M)[SP[O],SP[V]]
  @tensoropt En3 += 0.5 * (fOV[I,A] * IntYos[A,I])
  En3 += Enb3
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_ΛpertT_samespin(EC::ECInfo, T2, U1, U2, spin::Symbol)

  Calculate same-spin (T) correction for ΛUCCSD(T) (i.e., ααα or βββ).
  `spin` ∈ (:α,:β)

  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_ΛpertT_samespin(EC::ECInfo, T2, U1, U2, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  o = space4spin('o', spin==:α)
  v = space4spin('v', spin==:α)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  # ``0.5(v_{ai}^{kj} - v_{ai}^{jk})``
  vooo = 0.5*ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  # ``v_{ck}^{ab}``, reordered to ``v^{ab}_{ck}``
  vv_vo = permutedims(ints2(EC, v*o*v*v), [3,4,1,2])
  # ``0.5(v_{kj}^{ai} - v_{jk}^{ai})``, reordered to ``\bar v^{ai}_{kj}``
  vo_oo = 0.5*permutedims(ints2(EC, o*o*v*o), [3,4,1,2])
  vo_oo -= permutedims(vo_oo,[1,2,4,3])
  nocc = length(SP[o])
  nvir = length(SP[v])
  ϵo, ϵv = orbital_energies(EC, spin)

  T = zeros(nvir,nvir,nvir)
  Kijk = zeros(nvir,nvir,nvir)

  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  IntY = zeros(nvir,nocc)
  for k = 3:nocc 
    for j = 1:k-1
      for i = 1:j-1
        T2ij = @view T2[:,:,i,j]
        T2ik = @view T2[:,:,i,k]
        T2jk = @view T2[:,:,j,k]
        T2i = @view T2[:,:,:,i]
        T2j = @view T2[:,:,:,j]
        T2k = @view T2[:,:,:,k]
        vvvk = @view vvvo[:,:,:,k]
        vvvj = @view vvvo[:,:,:,j]
        vvvi = @view vvvo[:,:,:,i]
        vokj = @view vooo[:,:,k,j]
        voki = @view vooo[:,:,k,i]
        voij = @view vooo[:,:,i,j]
        @tensoropt begin
          # K_{abc}^{ijk} = v_{bc}^{dk} T^{ij}_{ad} + ...
          Kijk[a,b,c] = T2ij[a,d] * vvvk[b,c,d]
          Kijk[a,b,c] += T2ik[a,d] * vvvj[c,b,d]
          Kijk[a,b,c] += T2jk[d,c] * vvvi[b,a,d]

          Kijk[a,b,c] -= T2i[b,a,l] * vokj[c,l]
          Kijk[a,b,c] -= T2j[a,b,l] * voki[c,l]
          Kijk[a,b,c] -= T2k[b,c,l] * voij[a,l]
        end
        # antisymmetrize K = A(a,b,c) Kijk[a,b,c]
        @tensoropt  T[a,b,c] = Kijk[a,b,c] - Kijk[c,b,a]
        @tensoropt Kijk[a,b,c] = T[a,b,c] - T[b,a,c] - T[a,c,b]
        T .= Kijk
        ϵoijk = ϵo[i] + ϵo[j] + ϵo[k]
        for abc ∈ CartesianIndices(T)
          a,b,c = Tuple(abc)
          T[abc] /= ϵoijk - ϵv[a] - ϵv[b] - ϵv[c]
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
        vo_kj = @view vo_oo[:,:,k,j]
        vo_ki = @view vo_oo[:,:,k,i]
        vo_ij = @view vo_oo[:,:,i,j]
        @tensoropt begin
          # K^{abc}_{ijk} = v_{dk}^{bc} Λ_{ij}^{ad} + ...
          Kijk[a,b,c] := U2ij[a,d] * vv_vk[b,c,d]
          Kijk[a,b,c] += U2ik[a,d] * vv_vj[c,b,d]
          Kijk[a,b,c] += U2jk[d,c] * vv_vi[b,a,d]

          Kijk[a,b,c] -= U2i[b,a,l] * vo_kj[c,l]
          Kijk[a,b,c] -= U2j[a,b,l] * vo_ki[c,l]
          Kijk[a,b,c] -= U2k[b,c,l] * vo_ij[a,l]
        end
        # antisymmetrize K = A(a,b,c) Kijk[a,b,c]
        @tensoropt  X[a,b,c] := Kijk[a,b,c] - Kijk[c,b,a]
        @tensoropt Kijk[a,b,c] = X[a,b,c] - X[b,a,c] - X[a,c,b]

        @tensoropt Enb3 += 1/6*(Kijk[a,b,c] * T[a,b,c])
        
        vv_jk = @view vv_oo[:,:,j,k]
        vv_ik = @view vv_oo[:,:,i,k]
        vv_ij = @view vv_oo[:,:,i,j]
        @tensoropt IntX[:,i][a] += T[a,b,c] * vv_jk[b,c]
        @tensoropt IntX[:,j][b] += T[a,b,c] * vv_ik[a,c]
        @tensoropt IntX[:,k][c] += T[a,b,c] * vv_ij[a,b]
        @tensoropt IntY[:,i][a] += T[a,b,c] * U2jk[b,c]
        @tensoropt IntY[:,j][b] += T[a,b,c] * U2ik[a,c]
        @tensoropt IntY[:,k][c] += T[a,b,c] * U2ij[a,b]
      end 
    end
  end
  # singles contribution
  @tensoropt En3 = U1[a,i] * IntX[a,i]
  # fock contribution
  m = space4spin('m', spin==:α)
  fov = load(EC,"f_"*m*m)[SP[o],SP[v]]
  @tensoropt En3 += 0.5 * (fov[i,a] * IntY[a,i])
  En3 += Enb3
  return (ET3=En3, ET3b=Enb3)
end

"""
    calc_ΛpertT_mixedspin(EC::ECInfo, T2, T2mix, U1, U2, U1os, U2mix, spin::Symbol)

  Calculate mixed-spin (T) correction for ΛUCCSD(T) (i.e., ααβ or ββα).

  `spin` ∈ (:α,:β)
  `U1` and `U2`/`T2` are same-`spin` Lagrange multipliers/amplitudes,
  `U1os` are opposite-`spin` Lagrange multipliers,
  and `U2mix`/`T2mix` are mixed-spin Lagrange multipliers/amplitudes 
  with the *second* electron being `spin`,
  i.e., Tβα for `spin == :α` and Tαβ for `spin == :β`.
  Return ( `ET3`=(T)-energy, `ET3b`=[T]-energy)) `NamedTuple`.
"""
function calc_ΛpertT_mixedspin(EC::ECInfo, T2, T2mix, U1, U2, U1os, U2mix, spin::Symbol)
  @assert spin ∈ (:α,:β) "spin must be :α or :β"
  SP = EC.space
  isα = (spin == :α)
  o = space4spin('o', isα)
  v = space4spin('v', isα)
  O = space4spin('o', !isα)
  V = space4spin('v', !isα)
  # ``v_{ij}^{ab}``, reordered to ``v^{ab}_{ij}``
  vv_oo = permutedims(ints2(EC, o*o*v*v),[3,4,1,2])
  # ``v_{ab}^{ck}``
  vvvo = ints2(EC, v*v*v*o)
  # ``v_{ai}^{kj} - v_{ai}^{jk}``
  vooo = ints2(EC, v*o*o*o)
  vooo -= permutedims(vooo,[1,2,4,3])
  # ``v_{ck}^{ab}``, reordered to ``v^{ab}_{ck}``
  vv_vo = permutedims(ints2(EC, v*o*v*v), [3,4,1,2])
  # ``v_{kj}^{ai} - v_{jk}^{ai}``, reordered to ``\bar v^{ai}_{kj}``
  vo_oo = permutedims(ints2(EC, o*o*v*o), [3,4,1,2])
  vo_oo -= permutedims(vo_oo,[1,2,4,3])
  if isα
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, o*O*v*V),[3,4,1,2])
    # ``v_{aB}^{cK}``
    vVvO = ints2(EC, v*V*v*O)
    # ``v_{Ab}^{Ck}``
    VvVo = permutedims(ints2(EC, v*V*o*V),[2,1,4,3])
    # ``v_{aI}^{kJ}``
    vOoO = ints2(EC, v*O*o*O)
    # ``v_{Ai}^{Kj}``
    VoOo = permutedims(ints2(EC, o*V*o*O),[2,1,4,3])
    # ``v_{cK}^{aB}``, reordered to ``v^{aB}_{cK}``
    vV_vO = permutedims(ints2(EC, v*O*v*V), [3,4,1,2])
    # ``v_{Ck}^{Ab}``, reordered to ``v^{Ab}_{Ck}``
    Vv_Vo = permutedims(ints2(EC, o*V*v*V),[4,3,2,1])
    # ``v_{kJ}^{aI}``, reordered to ``v^{aI}_{kJ}``
    vO_oO = permutedims(ints2(EC, o*O*v*O), [3,4,1,2])
    # ``v_{Kj}^{Ai}``, reordered to ``v^{Ai}_{Kj}``
    Vo_Oo = permutedims(ints2(EC, o*O*o*V),[4,3,2,1])
  else
    # ``v_{iJ}^{aB}``, reordered to ``v^{aB}_{iJ}``
    vV_oO = permutedims(ints2(EC, O*o*V*v),[4,3,2,1])
    # ``v_{aB}^{cK}``
    vVvO = permutedims(ints2(EC, V*v*O*v),[2,1,4,3])
    # ``v_{Ab}^{Ck}``
    VvVo = ints2(EC, V*v*V*o)
    # ``v_{aI}^{kJ}``
    vOoO = permutedims(ints2(EC, O*v*O*o),[2,1,4,3])
    # ``v_{Ai}^{Kj}``
    VoOo = ints2(EC, V*o*O*o)
    # ``v_{cK}^{aB}``, reordered to ``v^{aB}_{cK}``
    vV_vO = permutedims(ints2(EC, O*v*V*v),[4,3,2,1])
    # ``v_{Ck}^{Ab}``, reordered to ``v^{Ab}_{Ck}``
    Vv_Vo = permutedims(ints2(EC, V*o*V*v),[3,4,1,2])
    # ``v_{kJ}^{aI}``, reordered to ``v^{aI}_{kJ}``
    vO_oO = permutedims(ints2(EC, O*o*O*v),[4,3,2,1])
    # ``v_{Kj}^{Ai}``, reordered to ``v^{Ai}_{Kj}``
    Vo_Oo = permutedims(ints2(EC, O*o*V*o),[3,4,1,2])
  end
  nocc = length(SP[o])
  nOcc = length(SP[O])
  nvir = length(SP[v])
  nVir = length(SP[V])
  ϵo, ϵv = orbital_energies(EC, spin)
  opspin = isα ? :β : :α
  ϵO, ϵV = orbital_energies(EC, opspin)

  T = zeros(nvir,nvir,nVir)
  Kijk = zeros(nvir,nvir,nVir)

  Enb3 = 0.0
  IntX = zeros(nvir,nocc)
  IntY = zeros(nvir,nocc)
  IntXos = zeros(nVir,nOcc)
  IntYos = zeros(nVir,nOcc)
  for K = 1:nOcc 
    T2K = T2mix[:,:,K,:]
    U2K = U2mix[:,:,K,:]
    for j = 2:nocc
      for i = 1:j-1
        T2ij = @view T2[:,:,i,j]
        T2Ki = @view T2K[:,:,i]
        T2Kj = @view T2K[:,:,j]
        T2i = @view T2[:,:,:,i]
        T2j = @view T2[:,:,:,j]
        T2mixi = @view T2mix[:,:,:,i]
        T2mixj = @view T2mix[:,:,:,j]
        vVvK = @view vVvO[:,:,:,K]
        vvvi = @view vvvo[:,:,:,i]
        vvvj = @view vvvo[:,:,:,j]
        VvVi = @view VvVo[:,:,:,i]
        VvVj = @view VvVo[:,:,:,j]
        voij = @view vooo[:,:,i,j]
        vOjK = @view vOoO[:,:,j,K]
        vOiK = @view vOoO[:,:,i,K]
        VoKj = @view VoOo[:,:,K,j]
        VoKi = @view VoOo[:,:,K,i]
        @tensoropt begin
          # K_{abC}^{ijK} = v_{bC}^{dK} T^{ij}_{ad} + ...
          Kijk[a,b,C] = T2ij[a,d] * vVvK[b,C,d]
          Kijk[a,b,C] += T2Kj[C,d] * vvvi[b,a,d]
          Kijk[a,b,C] += T2Ki[C,d] * vvvj[a,b,d]
          Kijk[a,b,C] += T2Kj[D,b] * VvVi[C,a,D]
          Kijk[a,b,C] += T2Ki[D,a] * VvVj[C,b,D]
          Kijk[a,b,C] -= T2K[C,b,l] * voij[a,l]
          Kijk[a,b,C] -= T2mixi[C,a,L] * vOjK[b,L]
          Kijk[a,b,C] -= T2mixj[C,b,L] * vOiK[a,L]
        end
        # antisymmetrize ΔK = A(a,b) ΔKijk[a,b,C]
        Kijk -= permutedims(Kijk,[2,1,3])
        @tensoropt begin
          Kijk[a,b,C] -= T2i[b,a,l] * VoKj[C,l]
          Kijk[a,b,C] -= T2j[a,b,l] * VoKi[C,l]
        end
        T .= Kijk
        ϵoijK = ϵo[i] + ϵo[j] + ϵO[K]
        for abC ∈ CartesianIndices(T)
          a,b,C = Tuple(abC)
          T[abC] /= ϵoijK - ϵv[a] - ϵv[b] - ϵV[C]
        end

        U2ij = @view U2[:,:,i,j]
        U2Ki = @view U2K[:,:,i]
        U2Kj = @view U2K[:,:,j]
        U2i = @view U2[:,:,:,i]
        U2j = @view U2[:,:,:,j]
        U2mixi = @view U2mix[:,:,:,i]
        U2mixj = @view U2mix[:,:,:,j]
        vV_vK = @view vV_vO[:,:,:,K]
        vv_vi = @view vv_vo[:,:,:,i]
        vv_vj = @view vv_vo[:,:,:,j]
        Vv_Vi = @view Vv_Vo[:,:,:,i]
        Vv_Vj = @view Vv_Vo[:,:,:,j]
        vo_ij = @view vo_oo[:,:,i,j]
        vO_jK = @view vO_oO[:,:,j,K]
        vO_iK = @view vO_oO[:,:,i,K]
        Vo_Kj = @view Vo_Oo[:,:,K,j]
        Vo_Ki = @view Vo_Oo[:,:,K,i]
        @tensoropt begin
          # K^{abC}_{ijK} = v_{dK}^{bC} Λ_{ij}^{ad} + ...
          Kijk[a,b,C] = U2ij[a,d] * vV_vK[b,C,d]
          Kijk[a,b,C] += U2Kj[C,d] * vv_vi[b,a,d]
          Kijk[a,b,C] += U2Ki[C,d] * vv_vj[a,b,d]
          Kijk[a,b,C] += U2Kj[D,b] * Vv_Vi[C,a,D]
          Kijk[a,b,C] += U2Ki[D,a] * Vv_Vj[C,b,D]
          Kijk[a,b,C] -= U2K[C,b,l] * vo_ij[a,l]
          Kijk[a,b,C] -= U2mixi[C,a,L] * vO_jK[b,L]
          Kijk[a,b,C] -= U2mixj[C,b,L] * vO_iK[a,L]
        end
        # antisymmetrize ΔK = A(a,b) ΔKijk[a,b,C]
        Kijk -= permutedims(Kijk,[2,1,3])
        @tensoropt begin
          Kijk[a,b,C] -= U2i[b,a,l] * Vo_Kj[C,l]
          Kijk[a,b,C] -= U2j[a,b,l] * Vo_Ki[C,l]
        end

        @tensoropt Enb3 += 0.5 * (Kijk[a,b,C] * T[a,b,C])
        
        vV_jK = @view vV_oO[:,:,j,K]
        vV_iK = @view vV_oO[:,:,i,K]
        vv_ij = @view vv_oo[:,:,i,j]
        @tensoropt IntX[:,i][a] += T[a,b,C] * vV_jK[b,C]
        @tensoropt IntX[:,j][b] += T[a,b,C] * vV_iK[a,C]
        @tensoropt IntXos[:,K][C] += T[a,b,C] * vv_ij[a,b]
        @tensoropt IntY[:,i][a] += T[a,b,C] * U2Kj[C,b]
        @tensoropt IntY[:,j][b] += T[a,b,C] * U2Ki[C,a]
        @tensoropt IntYos[:,K][C] += T[a,b,C] * U2ij[a,b]
      end 
    end
  end
  # singles contribution
  @tensoropt En3 = U1[a,i] * IntX[a,i]
  @tensoropt En3 += U1os[A,I] * IntXos[A,I]
  # fock contribution
  m = space4spin('m', isα)
  fov = load(EC,"f_"*m*m)[SP[o],SP[v]]
  @tensoropt En3 += fov[i,a] * IntY[a,i]
  M = space4spin('m', !isα)
  fOV = load(EC,"f_"*M*M)[SP[O],SP[V]]
  @tensoropt En3 += 0.5 * (fOV[I,A] * IntYos[A,I])
  En3 += Enb3
  return (ET3=En3, ET3b=Enb3)
end