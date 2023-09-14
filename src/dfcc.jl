"""
  DFCoupledCluster

  Density-fitted coupled-cluster methods.
"""
module DFCoupledCluster
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.DecompTools
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.CCTools
using ..ElemCo.DIIS

export calc_dressed_3idx, calc_svd_dc

"""
    get_vˣˣ(EC::ECInfo)

  Return ``v^{XY} = v^{XL} v^{YL'}δ_{LL'}``
  with ``v^{XL} = v_k^{cL} U^{kX}_c``. 

  The integrals will be read from file `d_^XX`. 
  If the file does not exist, the integrals will be calculated
  and stored in file `d_^XX`. 
  v_k^{cL} and U^{kX}_c are read from files `d_ovL` and `C_voX`.
"""
function get_vˣˣ(EC::ECInfo)
  if file_exists(EC, "d_^XX") 
    return load(EC, "d_^XX")
  end
  if !file_exists(EC, "d_ovL") || !file_exists(EC, "C_voX")
    error("Files d_ovL or C_voX do not exist!")
  end
  UvoX = load(EC, "C_voX")
  ovLfile, ovL = mmap(EC, "d_ovL")
  nL = size(ovL, 3)
  nX = size(UvoX, 3)
  vxx = zeros(nX, nX)
  LBlks = get_auxblks(nL)
  for L in LBlks
    V_ovL = @view ovL[:,:,L]
    @tensoropt W_XL[X,L] := UvoX[b,j,X] * V_ovL[j,b,L]
    @tensoropt vxx[X,Y] += W_XL[X,L] * W_XL[Y,L]
  end
  close(ovLfile)
  save(EC, "d_^XX", vxx)
  return vxx
end

"""
    calc_deco_hylleraas(EC::ECInfo, T1, tT2, R1, R2)

  Calculate closed-shell singles and doubles Hylleraas energy
  using contravariant decomposed doubles amplitudes `tT2`=``\\tilde T_{XY}``.
"""
function calc_deco_hylleraas(EC::ECInfo, T1, tT2, R1, R2)
  SP = EC.space
  vxx = get_vˣˣ(EC)
  @tensoropt ET2 = tT2[X,Y] * (vxx[X,Y] + R2[X,Y])
  if length(T1) > 0
    dfock = load(EC, "df_mm")
    fov = dfock[SP['o'],SP['v']] + load(EC,"f_mm")[SP['o'],SP['v']] # undressed part should be with factor two
    @tensoropt ET1 = (fov[i,a] + 2.0 * R1[a,i])*T1[a,i]
    ET2 += ET1
  end
  return ET2
end

"""
    calc_deco_doubles_energy(EC::ECInfo, tT2)

  Calculate closed-shell doubles energy
  using contravariant decomposed doubles amplitudes `tT2`=``\\tilde T_{XY}``.
"""
function calc_deco_doubles_energy(EC::ECInfo, tT2)
  vxx = get_vˣˣ(EC)
  @tensoropt ET2 = tT2[X,Y] * vxx[X,Y]
  return ET2
end

"""
    calc_dressed_3idx(EC, T1)

  Calculate dressed integrals for 3-index integrals from file `mmL`.
"""
function calc_dressed_3idx(EC, T1)
  mmLfile, mmL = mmap(EC, "mmL")
  # println(size(mmL))
  SP = EC.space
  nL = size(mmL, 3)
  nocc = length(SP['o'])
  nvirt = length(SP['v'])
  # create mmaps for dressed integrals
  ovLfile, ovL = newmmap(EC, "d_ovL", Float64, (nocc,nvirt,nL))
  voLfile, voL = newmmap(EC, "d_voL", Float64, (nvirt,nocc,nL))
  ooLfile, ooL = newmmap(EC, "d_ooL", Float64, (nocc,nocc,nL))
  vvLfile, vvL = newmmap(EC, "d_vvL", Float64, (nvirt,nvirt,nL))

  LBlks = get_auxblks(nL)
  for L in LBlks
    ovL[:,:,L] = mmL[SP['o'],SP['v'],L]
    V_ovL = @view ovL[:,:,L]
    V_vvL = mmL[SP['v'],SP['v'],L]
    @tensoropt V_vvL[a,b,L] -= T1[a,i] * V_ovL[i,b,L]
    V_voL = mmL[SP['v'],SP['o'],L]
    @tensoropt V_voL[a,i,L] += T1[b,i] * V_vvL[a,b,L]
    vvL[:,:,L] = V_vvL;   V_vvL = nothing
    V_ooL = mmL[SP['o'],SP['o'],L]
    @tensoropt V_voL[a,i,L] -= T1[a,j] * V_ooL[j,i,L]
    voL[:,:,L] = V_voL;   V_voL = nothing
    @tensoropt V_ooL[i,j,L] += T1[b,j] * V_ovL[i,b,L]
    ooL[:,:,L] = V_ooL;   V_ooL = nothing
  end
  closemmap(EC, ovLfile, ovL)
  closemmap(EC, voLfile, voL)
  closemmap(EC, ooLfile, ooL)
  closemmap(EC, vvLfile, vvL)
  close(mmLfile)
end

"""
    save_pseudodressed_3idx(EC)

  Save non-dressed 3-index integrals from file `mmL` to dressed files.
"""
function save_pseudodressed_3idx(EC)
  mmLfile, mmL = mmap(EC, "mmL")
  # println(size(mmL))
  SP = EC.space
  nL = size(mmL, 3)
  nocc = length(SP['o'])
  nvirt = length(SP['v'])
  # create mmaps for dressed integrals
  ovLfile, ovL = newmmap(EC,"d_ovL",Float64,(nocc,nvirt,nL))
  voLfile, voL = newmmap(EC,"d_voL",Float64,(nvirt,nocc,nL))
  ooLfile, ooL = newmmap(EC,"d_ooL",Float64,(nocc,nocc,nL))
  vvLfile, vvL = newmmap(EC,"d_vvL",Float64,(nvirt,nvirt,nL))

  LBlks = get_auxblks(nL)
  for L in LBlks
    ovL[:,:,L] = mmL[SP['o'],SP['v'],L]
    vvL[:,:,L] = mmL[SP['v'],SP['v'],L]
    voL[:,:,L] = mmL[SP['v'],SP['o'],L]
    ooL[:,:,L] = mmL[SP['o'],SP['o'],L]
  end
  closemmap(EC, ovLfile, ovL)
  closemmap(EC, voLfile, voL)
  closemmap(EC, ooLfile, ooL)
  closemmap(EC, vvLfile, vvL)
  close(mmLfile)
end

"""
    dress_df_fock(EC, T1)

  Dress DF fock matrix with DF 3-index integrals.

  The dress-contribution is added to the original fock matrix
  from file `f_mm`. The dressed fock matrix is stored in file `df_mm`.
"""
function dress_df_fock(EC, T1)
  dfock = load(EC, "f_mm")
  mmLfile, mmL = mmap(EC, "mmL")
  nL = size(mmL, 3)
  occ = EC.space['o']
  virt = EC.space['v']

  LBlks = get_auxblks(nL)
  for L in LBlks
    V_mmL = @view mmL[:,:,L]
    mvL = V_mmL[:,virt,:]
    @tensoropt vt_moL[p,i,L] := mvL[p,a,L]*T1[a,i]
    mvL = nothing
    @tensoropt vt_L[L] := vt_moL[occ,:,:][i,i,L]
    # exchange
    omL = V_mmL[occ,:,:]
    @tensoropt dfock[p,q] -= vt_moL[p,i,L]*omL[i,q,L]
    omL = nothing
    # coulomb
    @tensoropt dfock[p,q] += 2.0*V_mmL[p,q,L]*vt_L[L]
  end
  close(mmLfile)
  # dress external indices
  dinter = dfock[:,virt]
  @tensoropt dfock[:,occ][p,j] += dinter[p,b] * T1[b,j]
  dinter = dfock[occ,:]
  @tensoropt dfock[virt,:][b,p] -= dinter[j,p] * T1[b,j]
  save(EC, "df_mm", dfock)
end

"""
    save_pseudo_dress_df_fock(EC)

  Save non-dressed DF fock matrix from file `f_mm` to dressed file `df_mm`.
"""
function save_pseudo_dress_df_fock(EC)
  dfock = load(EC, "f_mm")
  save(EC, "df_mm", dfock)
end

"""
    calc_doubles_decomposition_without_doubles(EC::ECInfo)

  Decompose ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` without explicit
  calculation of ``T^{ij}_{ab}``.

  The decomposition is done in two steps:
  1. ``\\bar U^{i\\bar X}_a`` is calculated from ``v_a^{iL}`` using SVD (with threshold `EC.options.cc.ampsvdtol`×0.01);
  2. MP2 doubles ``T^{i}_{aX}`` are calculated from ``v_a^{iL}`` and ``U^{iX}_a`` and again decomposed using SVD and threshold `EC.options.cc.ampsvdtol`.
  The SVD-basis is rotated to pseudocanonical basis to diagonalize 
  orbital-energy differences, ``ϵ_X = U^{iX}_{a}(ϵ_a-ϵ_i)U^{iX}_a``.
  The imaginary shift `EC.options.cc.deco_ishiftp` is used in the denominator in the calculation of the MP2 amplitudes.
  The orbital energy differences are saved in file `e_X`.
  The SVD-coefficients ``U^{iX}_a`` are saved in file `C_voX`.
  The starting guess for doubles ``T_{XY}`` is saved in file `T_XX`.
"""
function calc_doubles_decomposition_without_doubles(EC::ECInfo)
  println("decomposition without doubles")
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  SP = EC.space
  # first approximation for U^{iX}_a from 3-index integrals v_a^{iL} 
  # TODO: add shifted Laplace transform!
  mmPfile, mmP = mmap(EC, "mmL")
  voL = mmP[SP['v'],SP['o'],:]
  nL = size(voL, 3)
  close(mmPfile)
  tol2 = EC.options.cc.ampsvdtol*0.01
  UaiX = svd_decompose(reshape(voL, (nvirt*nocc,nL)), nvirt, nocc, tol2)
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  # calculate rhs: v_{aX}^{i} = v_a^{iL} 𝟙_{LL} v_b^{jL} (U_b^{jX})^† 
  @tensoropt voX[a,i,X] := voL[a,i,L] * (voL[b,j,L] * UaiX[b,j,X])
  # calculate half-decomposed imaginary-shifted MP2 amplitudes 
  # T^i_{aX} = -v_{aX}^{i} * (ϵ_a - ϵ_i + ϵ_X)/((ϵ_a - ϵ_i - ϵ_X)^2 + ω)
  # TODO: use a better method than MP2
  ϵo, ϵv = orbital_energies(EC)
  shifti = EC.options.cc.deco_ishiftp
  for I ∈ CartesianIndices(voX)
    a,i,X = Tuple(I)
    den = ϵX[X] + ϵv[a] - ϵo[i]
    voX[I] *= -den/(den^2 + shifti)
  end
  naux = size(voX, 3)
  # decompose T^i_{aX}
  UaiX = svd_decompose(reshape(voX, (nvirt*nocc,naux)), nvirt, nocc, EC.options.cc.ampsvdtol)
  ϵX, UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save(EC, "e_X", ϵX)
  #display(UaiX)
  naux = length(ϵX)
  save(EC, "C_voX", UaiX)
  # calc starting guess for T_XY
  @tensoropt v_XL[X,L] := UaiX[a,i,X] * voL[a,i,L]
  @tensoropt v_XX[X,Y] := v_XL[X,L] * v_XL[Y,L]
  shifti = EC.options.cc.deco_ishiftp
  for I ∈ CartesianIndices(v_XX)
    X,Y = Tuple(I)
    den = ϵX[X] + ϵX[Y]
    v_XX[I] *= -den/(den^2 + shifti)
  end
  save(EC, "T_XX", v_XX)
  # save(EC, "T_XX", zeros(size(v_XX)))
end

"""
    contravariant_deco_doubles(EC::ECInfo, T2)

  Calculate contravariant doubles amplitudes
  ``T̃_{XY} = U^{†a}_{iX} U^{†b}_{jY} T̃^{ij}_{ab}``
  with ``T̃^{ij}_{ab} = 2T^{ij}_{ab} - T^{ij}_{ba}`` and
  ``T^{ij}_{ab} = U^{iX}_a U^{jY}_b T_{XY}``.
"""
function contravariant_deco_doubles(EC::ECInfo, T2)
  UvoX = load(EC, "C_voX")
  # calc T^{ij}_{ab} = U^{iX}_a U^{jY}_b T_{XY}
  @tensoropt Tabij[a,b,i,j] := UvoX[a,i,X] * (UvoX[b,j,Y] * T2[X,Y])
  # calc \tilde T_{XY}
  @tensoropt tT2[X,Y] := (2.0 * Tabij[a,b,i,j] - Tabij[b,a,i,j])*UvoX[a,i,X]*UvoX[b,j,Y]
  return tT2
end

"""
    calc_svd_dcsd_residual(EC::ECInfo, T1, T2, tT2)

  Calculate decomposed closed-shell DCSD residual with
  ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` and
  ``R_{XY}=U^{iX†}_a U^{jY†}_b R^{ij}_{ab}``.
  `tT2` contains contravariant amplitudes ``\\tilde T_{XY}`.
"""
function calc_svd_dcsd_residual(EC::ECInfo, T1, T2, tT2)
  t1 = time_ns()
  if length(T1) > 0
    #get dressed integrals
    calc_dressed_3idx(EC, T1)
    dress_df_fock(EC, T1)
    t1 = print_time(EC, t1, "dressed 3-idx integrals", 2)
  end
  SP = EC.space
  dfock = load(EC, "df_mm")
  if length(T1) > 0
    R1 = dfock[SP['v'],SP['o']]
  else
    R1 = Float64[]
  end
  R2 = zeros(size(T2))
  UvoX = load(EC, "C_voX")
  x_vv = dfock[SP['v'],SP['v']]
  x_oo = dfock[SP['o'],SP['o']]
  voLfile, voL = mmap(EC, "d_voL")
  ovLfile, ovL = mmap(EC, "d_ovL")
  v_voX = zeros(size(UvoX))
  vvLfile, vvL = mmap(EC, "d_vvL")
  ooLfile, ooL = mmap(EC, "d_ooL")
  f_ov = dfock[SP['o'],SP['v']]
  if length(R1) > 0
    # ``R^i_a += f_k^c U^{kX}_c \tilde T_{XY} U^{iY}_{a}``
    @tensoropt R1[a,i] += (f_ov[k,c] * UvoX[c,k,X]) * tT2[X,Y] * UvoX[a,i,Y]
  end
  f_ov = nothing
  nL = size(voL, 3)
  nX = size(UvoX, 3)
  LBlks = get_auxblks(nL)
  XBlks = get_auxblks(nX)
  for L in LBlks
    V_ovL = @view ovL[:,:,L]
    # ``Y_X^L = (v_k^{cL} U^{kY}_c) \tilde T_{XY}``) 
    @tensoropt W_XL[X,L] := (UvoX[b,j,Y] * V_ovL[j,b,L]) * tT2[X,Y] 
    # ``Y_a^{kL} = Y_X^L U^{kX}_a``
    @tensoropt Y_voL[a,k,L] := UvoX[a,k,X] * W_XL[X,L]
    # ``x_a^c -= 0.5 Y_a^{kL} v_k^{cL}``
    @tensoropt x_vv[a,c] -= 0.5 * Y_voL[a,k,L] * V_ovL[k,c,L]
    # ``x_k^i += 0.5 Y_c^{iL} v_k^{cL}``
    @tensoropt x_oo[k,i] += 0.5 * Y_voL[c,i,L] * V_ovL[k,c,L]
    # ``v_X^L = U^{†a}_{iX} v_a^{iL}``
    V_voL = @view voL[:,:,L]
    @tensoropt W_XL[X,L] += V_voL[a,i,L] * UvoX[a,i,X]
    # ``R_{XY} += (v+Y)^L_X (v+Y)^{L'}_Y δ_{LL'}``
    @tensoropt R2[X,Y] += W_XL[X,L] * W_XL[Y,L]

    V_vvL = @view vvL[:,:,L]
    V_ooL = @view ooL[:,:,L]
    if length(R1) > 0
      # ``R^i_a += v_a^{cL} Y_c^{iL}``
      @tensoropt R1[a,i] += V_vvL[a,c,L] * Y_voL[c,i,L]
      # ``R^i_a -= v_k^{iL} Y_a^{kL}``
      @tensoropt R1[a,i] -= V_ooL[k,i,L] * Y_voL[a,k,L]
    end
    # ``v_X^{X'L}`` as v[X',L,X]
    v_XLX = zeros(nX,length(L),nX)
    for X in XBlks
      V_UvoX = @view UvoX[:,:,X]
      V_v_voX = @view v_voX[:,:,X]
      V_v_XLX = @view v_XLX[:,:,X]
      # ``v_{kX}^{cL} = v_a^{cL} U^{†a}_{kX}``
      @tensoropt v_oXvL[k,X,c,L] := V_vvL[a,c,L] * V_UvoX[a,k,X]
      # ``v_{kX}^{c} = v_{iX}^{cL} v_{k}^{iL}``    
      @tensoropt V_v_voX[c,k,X] += v_oXvL[i,X,c,L] * V_ooL[k,i,L]
      # ``v_X^{X'L} = (v_{kX}^{cL} - v_k^{iL} U^{†c}_{iX}) U^{kX'}_c``
      @tensoropt V_v_XLX[X',L,X] = (v_oXvL[k,X,c,L] - V_ooL[k,i,L] * V_UvoX[c,i,X]) * UvoX[c,k,X']
    end
    @tensoropt R2[X,Y] += v_XLX[X',L,X] * (T2[X',Y'] * v_XLX[Y',L,Y])
  end
  close(voLfile)
  close(ovLfile)
  close(vvLfile)
  close(ooLfile)
  # ``W_X^{X'} = (x_a^c U^{†a}_{kX} - x_k^i U^{†c}_{iX} - v_{kX}^c) U^{kX'}_c``
  @tensoropt W_XX[X,X'] := (x_vv[a,c] * UvoX[a,k,X] - x_oo[k,i] * UvoX[c,i,X] - v_voX[c,k,X]) * UvoX[c,k,X']
  v_voX = nothing
  # ``R_{XY} += W_X^{X'} T_{X'Y} + T_{XY'} W_{Y}^{Y'}``
  @tensoropt R2[X,Y] += W_XX[X,X'] * T2[X',Y] + T2[X,Y'] * W_XX[Y,Y']
  W_XX = nothing

  return R1, R2
end

"""
    calc_svd_dc(EC::ECInfo, method::AbstractString)

  Calculate decomposed closed-shell DCSD or DCD with
  ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}``.
"""
function calc_svd_dc(EC::ECInfo, method::AbstractString)
  calc_svd_dc(EC, ECMethod(method))
end

"""
    calc_svd_dc(EC::ECInfo, method::ECMethod)

  Calculate decomposed closed-shell DCSD or DCD with
  ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}``.

  Currently only DC methods are supported. 
  The integrals are calculated using density fitting.
  The starting guess for SVD-coefficients is calculated without doubles,
  see [`calc_doubles_decomposition_without_doubles`](@ref).
"""
function calc_svd_dc(EC::ECInfo, method::ECMethod)
  t1 = time_ns()
  methodname = "SVD-"*method_name(method)
  print_info(methodname)

  if method.theory != "DC"
    error("Only DC methods are supported in SVD!")
  end
  do_sing = (method.exclevel[1] == :full)
  # integrals
  cMO = load_orbitals(EC, EC.options.cc.orbs)
  ERef = generate_DF_integrals(EC, cMO)
  println("Reference energy: ", ERef)
  println()
  # decomposition and starting guess
  calc_doubles_decomposition_without_doubles(EC)
  if do_sing
    T1 = read_starting_guess4amplitudes(EC, 1)
  else
    T1 = Float64[]
    save_pseudodressed_3idx(EC)
    save_pseudo_dress_df_fock(EC)
  end
  diis = Diis(EC)

  println("Iter     SqNorm      Energy      DE          Res         Time")
  NormR1 = 0.0
  NormT1 = 0.0
  NormT2 = 0.0
  R1 = Float64[]
  Eh = 0.0
  t0 = time_ns()
  T2 = load(EC,"T_XX")
  tT2 = contravariant_deco_doubles(EC, T2)
  converged = false
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    R1, R2 = calc_svd_dcsd_residual(EC, T1, T2, tT2)
    # println("R1: ", norm(R1))
    # println("R2: ", norm(R2))
    t1 = print_time(EC,t1,"ccsd residual",2)
    Eh = calc_deco_hylleraas(EC, T1, tT2, R1, R2)
    NormT2 = calc_deco_doubles_norm(T2, tT2)
    NormR2 = calc_deco_doubles_norm(R2)
    if do_sing
      NormT1 = calc_singles_norm(T1)
      NormR1 = calc_singles_norm(R1)
      T1 += update_singles(EC, R1)
    end
    T2 += update_deco_doubles(EC, R2)
    T1, T2 = perform(diis, [T1,T2], [R1,R2])
    tT2 = contravariant_deco_doubles(EC, T2)
    En = calc_deco_doubles_energy(EC, tT2)
    if do_sing
      En += calc_singles_energy_using_dfock(EC, T1)
    end
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
    println("$methodname not converged!")
  end
  println()
  @printf "Sq.Norm of T1: %12.8f Sq.Norm of T2: %12.8f \n" NormT1 NormT2
  println()
  println("$methodname correlation energy: ", Eh)
  println("$methodname total energy: ", Eh + ERef)
  println()
  flush(stdout)
  delete_temporary_files(EC)
  draw_endline()
  return Eh

end

end # module DFCoupledCluster