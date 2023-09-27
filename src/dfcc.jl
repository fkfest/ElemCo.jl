"""
  DFCoupledCluster

  Density-fitted coupled-cluster methods.
"""
module DFCoupledCluster
using LinearAlgebra, TensorOperations, Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.MSystem
using ..ElemCo.ECMethods
using ..ElemCo.TensorTools
using ..ElemCo.DecompTools
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.CCTools
using ..ElemCo.DIIS

export calc_dressed_3idx, calc_svd_dc

"""
    get_tildevˣˣ(EC::ECInfo)

  Return ``\\tilde v^{XY} = U^{kX}_c U^{lX}_d (2 v_{kl}^{cd} - v_{lk}^{cd} )``
  with ``v_{kl}^{cd} = v_k^{cL} v_l^{dL'}δ_{LL'}``. 

  The integrals will be read from file `td_^XX`. 
  If the file does not exist, the integrals will be calculated
  and stored in file `td_^XX`. 
  v_k^{cL} and U^{kX}_c are read from files `d_ovL` and `C_voX`.
"""
function get_tildevˣˣ(EC::ECInfo)
  if file_exists(EC, "td_^XX") 
    return load(EC, "td_^XX")
  end
  if !file_exists(EC, "d_ovL") || !file_exists(EC, "C_voX")
    error("Files d_ovL or C_voX do not exist!")
  end
  UvoX = load(EC, "C_voX")
  ovLfile, ovL = mmap(EC, "d_ovL")

  nocc, nvirt, nL = size(ovL)
  LBlks = get_auxblks(nL)
  vdagger_vvoo = zeros(nvirt,nvirt,nocc,nocc)
  for L in LBlks
    V_ovL = @view ovL[:,:,L]
    @tensoropt vdagger_vvoo[a,b,i,j] += V_ovL[i,a,L] * V_ovL[j,b,L]
  end
  close(ovLfile)
  @tensoropt vxx[X,Y] := (2.0*vdagger_vvoo[a,b,i,j] - vdagger_vvoo[a,b,j,i]) * UvoX[a,i,X] * UvoX[b,j,Y]
  save!(EC, "td_^XX", vxx)
  return vxx
end

"""
    calc_deco_hylleraas(EC::ECInfo, T1, T2, R1, R2)

  Calculate closed-shell singles and doubles Hylleraas energy
  using contravariant decomposed doubles amplitudes `T2`=``T_{XY}``
  or full contravariant doubles amplitude `T2`=``T^{ij}_{ab}``.
"""
function calc_deco_hylleraas(EC::ECInfo, T1, T2, R1, R2)
  SP = EC.space
  full_t2 = true
  if ndims(T2) == 2
    full_t2 = false
  elseif ndims(T2) != 4
    error("Wrong dimensionality of T2!")
  end
  if full_t2
    ovL = load(EC, "d_ovL")
    @tensoropt ET2 = (2.0 * T2[a,b,i,j] - T2[a,b,j,i]) * (R2[a,b,i,j] + ovL[i,a,L] * ovL[j,b,L])
    ovL = nothing
  else
    vxx = get_tildevˣˣ(EC)
    @tensoropt ET2 = T2[X,Y] * (vxx[X,Y] + 2.0*R2[X,Y])
    UvoX = load(EC, "C_voX")
    @tensoropt ET2 -= T2[X,Y] * ((((R2[X',Y'] * UvoX[a,i,X']) * UvoX[b,j,Y']) * UvoX[a,j,X]) * UvoX[b,i,Y])
    UvoX = nothing
  end
  if length(T1) > 0
    dfock = load(EC, "df_mm")
    fov = dfock[SP['o'],SP['v']] + load(EC,"f_mm")[SP['o'],SP['v']] # undressed part should be with factor two
    @tensoropt ET1 = (fov[i,a] + 2.0 * R1[a,i])*T1[a,i]
    ET2 += ET1
  end
  return ET2
end

"""
    calc_deco_doubles_energy(EC::ECInfo, T2)

  Calculate closed-shell doubles energy
  using decomposed doubles amplitudes `T2`=``T_{XY}``
  or `T2`=``T^{ij}_{ab}`` using density-fitted integrals.
"""
function calc_deco_doubles_energy(EC::ECInfo, T2)
  if ndims(T2) == 4
    return calc_df_doubles_energy(EC, T2)
  elseif ndims(T2) == 2
    tvxx = get_tildevˣˣ(EC)
    @tensoropt ET2 = T2[X,Y] * tvxx[X,Y]
    return ET2
  else
    error("Wrong dimensionality of T2: ", ndims(T2))
  end
end

"""
    calc_df_doubles_energy(EC::ECInfo, T2)

  Calculate closed-shell doubles energy using DF integrals 
  and `T2[a,b,i,j]` = ``T^{ij}_{ab}``.
"""
function calc_df_doubles_energy(EC::ECInfo, T2)
  if !file_exists(EC, "d_ovL")
    error("File d_ovL does not exist!")
  end
  ovL = load(EC, "d_ovL")
  @tensoropt ET2 = (2.0*T2[a,b,i,j] - T2[a,b,j,i]) * (ovL[i,a,L] * ovL[j,b,L])
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
  save!(EC, "df_mm", dfock)
end

"""
    save_pseudo_dress_df_fock(EC)

  Save non-dressed DF fock matrix from file `f_mm` to dressed file `df_mm`.
"""
function save_pseudo_dress_df_fock(EC)
  dfock = load(EC, "f_mm")
  save!(EC, "df_mm", dfock)
end

"""
    calc_doubles_decomposition(EC::ECInfo)

  Decompose ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}``
"""
function calc_doubles_decomposition(EC::ECInfo)
  if EC.options.cc.decompose_full_doubles
    return calc_doubles_decomposition_with_doubles(EC)
  else
    return calc_doubles_decomposition_without_doubles(EC)
  end
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
  Return full MP2 correlation energy (using the imaginary shift).
"""
function calc_doubles_decomposition_without_doubles(EC::ECInfo)
  println("Decomposition without doubles using threshold ", EC.options.cc.ampsvdtol)
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  SP = EC.space
  # first approximation for U^{iX}_a from 3-index integrals v_a^{iL} 
  # TODO: add shifted Laplace transform!
  mmLfile, mmL = mmap(EC, "mmL")
  nL = size(mmL, 3)
  tol2 = (EC.options.cc.ampsvdtol*0.01)
  voL = mmL[SP['v'],SP['o'],:]
  shifti = EC.options.cc.deco_ishiftp
  fullEMP2 = calc_MP2_from_3idx(EC, voL, shifti)
  if shifti ≈ 0.0
    println("MP2 correlation energy: ", fullEMP2)
  else
    println("MP2 imaginary shift for decomposition: ", shifti)
    println("MP2 imaginary shifted correlation energy: ", fullEMP2)
  end
  if EC.options.cc.use_full_t2
    T2 = try2start_doubles(EC)
    if size(T2) != (nvirt,nvirt,nocc,nocc)
      T2 = calc_MP2_amplitudes_from_3idx(EC, voL, shifti)
    end
    save!(EC, "T_vvoo", T2)
    T2 = nothing
  end
  UaiX = svd_decompose(reshape(voL, (nvirt*nocc,nL)), nvirt, nocc, tol2)
  # UaiX = calc_3idx_svd_decomposition(EC, voL) 
  ϵX,UaiX = rotate_U2pseudocanonical(EC, UaiX)
  # calculate rhs: v_{aX}^{i} = v_a^{iL} 𝟙_{LL} v_b^{jL} (U_b^{jX})^† 
  @tensoropt voX[a,i,X] := voL[a,i,L] * (voL[b,j,L] * UaiX[b,j,X])
  # calculate half-decomposed imaginary-shifted MP2 amplitudes 
  # T^i_{aX} = -v_{aX}^{i} * (ϵ_a - ϵ_i + ϵ_X)/((ϵ_a - ϵ_i - ϵ_X)^2 + ω)
  # TODO: use a better method than MP2
  ϵo, ϵv = orbital_energies(EC)
  for I ∈ CartesianIndices(voX)
    a,i,X = Tuple(I)
    den = ϵX[X] + ϵv[a] - ϵo[i]
    voX[I] *= -den/(den^2 + shifti)
  end
  naux = size(voX, 3)
  # decompose T^i_{aX}
  UaiX = svd_decompose(reshape(voX, (nvirt*nocc,naux)), nvirt, nocc, EC.options.cc.ampsvdtol)
  ϵX, UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save!(EC, "e_X", ϵX)
  #display(UaiX)
  naux = length(ϵX)
  save!(EC, "C_voX", UaiX)
  # calc starting guess for T_XY
  @tensoropt v_XL[X,L] := UaiX[a,i,X] * voL[a,i,L]
  @tensoropt v_XX[X,Y] := v_XL[X,L] * v_XL[Y,L]
  for I ∈ CartesianIndices(v_XX)
    X,Y = Tuple(I)
    den = ϵX[X] + ϵX[Y]
    v_XX[I] *= -den/(den^2 + shifti)
  end
  save!(EC, "T_XX", v_XX)
  # save!(EC, "T_XX", zeros(size(v_XX)))
  return fullEMP2
end

"""
    calc_doubles_decomposition_with_doubles(EC::ECInfo)

  Decompose ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` using explicit doubles amplitudes ``T^{ij}_{ab}``.
"""
function calc_doubles_decomposition_with_doubles(EC::ECInfo)
  println("Decomposition with doubles using threshold ", EC.options.cc.ampsvdtol)
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  SP = EC.space
  mmLfile, mmL = mmap(EC, "mmL")
  nL = size(mmL, 3)
  voL = mmL[SP['v'],SP['o'],:]
  T2 = try2start_doubles(EC)
  if size(T2) != (nvirt,nvirt,nocc,nocc)
    println("Use MP2 doubles for decomposition")
    shifti = EC.options.cc.deco_ishiftp
    T2 = calc_MP2_amplitudes_from_3idx(EC, voL, shifti)
  end
  if EC.options.cc.use_full_t2
    save!(EC, "T_vvoo", T2)
  end
  println("decompose full doubles (can be slow!)")
  UaiX = svd_decompose(reshape(T2, (nvirt*nocc,nvirt*nocc)), nvirt, nocc, EC.options.cc.ampsvdtol)
  ϵX, UaiX = rotate_U2pseudocanonical(EC, UaiX)
  save!(EC, "e_X", ϵX)
  #display(UaiX)
  naux = length(ϵX)
  save!(EC, "C_voX", UaiX)
  if !EC.options.cc.use_full_t2
    # calc starting guess for T_XY
    @tensoropt v_XL[X,L] := UaiX[a,i,X] * voL[a,i,L]
    @tensoropt v_XX[X,Y] := v_XL[X,L] * v_XL[Y,L]
    for I ∈ CartesianIndices(v_XX)
      X,Y = Tuple(I)
      den = ϵX[X] + ϵX[Y]
      v_XX[I] *= -den/(den^2 + shifti)
    end
    save!(EC, "T_XX", v_XX)
    # save!(EC, "T_XX", zeros(size(v_XX)))
  end
  return 0.0
end

"""
    calc_3idx_svd_decomposition(EC::ECInfo, full_voL::AbstractArray)

  Calculate ``U^{iX}_a`` from ``v_a^{iL}`` using SVD.

  Version without holding all ``v_a^{iL}`` integrals in memory.
  `full_voL` is the full 3-index integral ``v_a^{iL}`` (can be mmaped).
"""
function calc_3idx_svd_decomposition(EC::ECInfo, full_voL::AbstractArray)
  W_LL = zeros(nL,nL)
  # generate ``W^{LL'} = v_a^{iL} v_a^{iL'}`` for SVD
  oBlks = get_spaceblocks(1:length(SP['o']))
  for oblk in oBlks
    voL = full_voL[:,oblk,:]
    @tensoropt W_LL[L,L'] += voL[a,i,L] * voL[a,i,L']
  end
  # decompose W^{LL'} = V_{LX} Σ^{XX'} V^†_{X'L}
  Vmat, Σ = svd_decompose(W_LL, tol2)
  # calculate U^{iX}_a = v_a^{iL} V_{LX} Σ^{-1/2}_{XX}
  LBlks = get_auxblks(nL)
  nX = length(Σ)
  UaiX = zeros(nvirt,nocc,nX)
  for L in LBlks
    voL = full_voL[:,:,L]
    @tensoropt UaiX[a,i,X] += voL[a,i,L] * Vmat[L,:][L,X] 
  end
  UaiX = reshape(reshape(UaiX, nvirt*nocc, nX) ./= sqrt.(Σ)', nvirt, nocc, nX)
  return UaiX
end


"""
    calc_MP2_from_3idx(EC::ECInfo, voL::AbstractArray, ishift)

  Calculate MP2 energy from ``v_a^{iL}``.
  
  The imaginary shift ishift is used in the denominator in the calculation of the MP2 amplitudes.
"""
function calc_MP2_from_3idx(EC::ECInfo, voL::AbstractArray, ishift)
  @tensoropt vvoo[a,b,i,j] := voL[a,i,L] * voL[b,j,L]
  ϵo, ϵv = orbital_energies(EC)
  nocc = length(ϵo)
  nvirt = length(ϵv)
  EMP2 = 0.0
  t_vvo = zeros(nvirt,nvirt,nocc)
  for j = 1:nocc
    vvo = @view vvoo[:,:,:,j]
    if ishift ≈ 0.0
      for I ∈ CartesianIndices(vvo)
        a,b,i = Tuple(I)
        den = -ϵv[a] - ϵv[b] + ϵo[i] + ϵo[j]
        t_vvo[I] = vvo[I] / den
      end
    else
      for I ∈ CartesianIndices(vvo)
        a,b,i = Tuple(I)
        den = -ϵv[a] - ϵv[b] + ϵo[i] + ϵo[j]
        t_vvo[I] = vvo[I] * den/(den^2 + ishift)
      end
    end
    @tensoropt EMP2 += vvo[a,b,i] * (2.0*t_vvo[a,b,i]-t_vvo[b,a,i])
  end
  return EMP2
end

"""
    calc_MP2_amplitudes_from_3idx(EC::ECInfo, voL::AbstractArray, ishift)

  Calculate MP2 amplitudes from ``v_a^{iL}``.
  
  The imaginary shift ishift is used in the denominator in the calculation of the MP2 amplitudes.
"""
function calc_MP2_amplitudes_from_3idx(EC::ECInfo, voL::AbstractArray, ishift)
  @tensoropt vvoo[a,b,i,j] := voL[a,i,L] * voL[b,j,L]
  ϵo, ϵv = orbital_energies(EC)
  if ishift ≈ 0.0
    for I ∈ CartesianIndices(vvoo)
      a,b,i,j = Tuple(I)
      den = -ϵv[a] - ϵv[b] + ϵo[i] + ϵo[j]
      vvoo[I] /= den
    end
  else
    for I ∈ CartesianIndices(vvoo)
      a,b,i,j = Tuple(I)
      den = -ϵv[a] - ϵv[b] + ϵo[i] + ϵo[j]
      vvoo[I] *= den/(den^2 + ishift)
    end
  end
  return vvoo
end

"""
    contravariant_deco_doubles(EC::ECInfo, T2, projx=false)

  Calculate contravariant doubles amplitudes
  ``T̃^{ij}_{ab} = 2T^{ij}_{ab} - T^{ij}_{ba}``
   with
  ``T^{ij}_{ab} = U^{iX}_a U^{jY}_b T_{XY}``.
  If `projx` is true, the projected exchange is returned:
  ``T̃_{XY} = U^{†a}_{iX} U^{†b}_{jY} T̃^{ij}_{ab}``
"""
function contravariant_deco_doubles(EC::ECInfo, T2, projx=false)
  UvoX = load(EC, "C_voX")
  # calc T^{ij}_{ab} = U^{iX}_a U^{jY}_b T_{XY}
  @tensoropt Tabij[a,b,i,j] := UvoX[a,i,X] * (UvoX[b,j,Y] * T2[X,Y])
  if projx
    # calc \tilde T_{XY}
    @tensoropt tT2[X,Y] := (2.0 * Tabij[a,b,i,j] - Tabij[b,a,i,j])*UvoX[a,i,X]*UvoX[b,j,Y]
  else
    @tensoropt tT2[a,b,i,j] := 2.0 * Tabij[a,b,i,j] - Tabij[b,a,i,j]
  end
  return tT2
end

"""
    calc_vᵥᵒˣ(EC::ECInfo)

  Calculate ``\\hat v_a^{iX} = \\hat v_{ak}^{ci} U^{kX}_c`` 
  with ``\\hat v_{ak}^{ci} = \\hat v_a^{cL} \\hat v_k^{iL}`` 
  and ``U^{kX}_c`` from file `C_voX`.
"""
function calc_vᵥᵒˣ(EC::ECInfo)
  if !file_exists(EC, "C_voX")
    error("File C_voX does not exist!")
  end
  vvLfile, vvL = mmap(EC, "d_vvL")
  ooLfile, ooL = mmap(EC, "d_ooL")
  nL = size(vvL, 3)
  nocc = size(ooL,1)
  nvirt = size(vvL,1)
  vvoo = zeros(nvirt,nvirt,nocc,nocc) 
  LBlks = get_auxblks(nL)
  for L in LBlks
    V_vvL = @view vvL[:,:,L]
    V_ooL = @view ooL[:,:,L]
    # ``v_{ak}^{ci} = v_a^{cL} v_k^{iL}``
    @tensoropt vvoo[a,c,k,i] += V_vvL[a,c,L] * V_ooL[k,i,L]
  end
  close(vvLfile)
  close(ooLfile)
  UvoX = load(EC, "C_voX")
  @tensoropt v_voX[a,i,X] := vvoo[a,c,k,i] * UvoX[c,k,X]
  return v_voX
end


"""
    calc_svd_dcsd_residual(EC::ECInfo, T1, T2)

  Calculate decomposed closed-shell DCSD residual with
  ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` and
  ``R_{XY}=U^{iX†}_a U^{jY†}_b R^{ij}_{ab}``.
  `T2` contains decomposed amplitudes ``T_{XY}` or full amplitudes ``T^{ij}_{ab}``.
 
  If `T2` is ``T^{ij}_{ab}``, the residual is also returned in full form.
"""
function calc_svd_dcsd_residual(EC::ECInfo, T1, T2)
  t1 = time_ns()
  if length(T1) > 0
    #get dressed integrals
    calc_dressed_3idx(EC, T1)
    dress_df_fock(EC, T1)
    t1 = print_time(EC, t1, "dressed 3-idx integrals", 2)
  end
  UvoX = load(EC, "C_voX")
  use_projected_exchange = EC.options.cc.use_projx
  project_amps_vovo_t2 = EC.options.cc.project_vovo_t2 < 2

  if ndims(T2) == 4
    # T2 and R2 are full amplitudes/residuals ``T/R^{ij}_{ab}``
    full_t2 = true
    full_tt2 = true
    # project amplitudes onto SVD-basis
    @tensoropt dT2[X,Y] := (T2[a,b,i,j] * UvoX[a,i,X]) * UvoX[b,j,Y]
    @tensoropt tT2[a,b,i,j] := 2.0 * T2[a,b,i,j] - T2[a,b,j,i]
    if use_projected_exchange
      error("Projected exchange not implemented for full T2!")
    end
  elseif ndims(T2) == 2
    # T2 and R2 are decomposed amplitudes/residuals ``T/R_{XY}``
    full_t2 = false
    # if use_projected_exchange: the contravariant amplitudes are projected
    # onto the SVD-basis
    full_tt2 = !use_projected_exchange
    dT2 = T2
    tT2 = contravariant_deco_doubles(EC, T2, use_projected_exchange)
  else
    error("Wrong dimensionality of T2!")
  end
  SP = EC.space
  dfock = load(EC, "df_mm")
  if length(T1) > 0
    R1 = dfock[SP['v'],SP['o']]
  else
    R1 = Float64[]
  end
  R2 = zeros(size(T2))
  if full_t2
    dR2 = zeros(size(dT2))
  else
    dR2 = R2
  end
  x_vv = dfock[SP['v'],SP['v']]
  x_oo = dfock[SP['o'],SP['o']]
  voLfile, voL = mmap(EC, "d_voL")
  ovLfile, ovL = mmap(EC, "d_ovL")
  # if project_amps_vovo_t2: ``v_a^{iX} = v_a^{kXL} v_{k}^{iL}``
  # otherwise: ``v_{kX}^{c} = v_{iX}^{cL} v_{k}^{iL}``    
  v_voX = zeros(size(UvoX))
  vvLfile, vvL = mmap(EC, "d_vvL")
  ooLfile, ooL = mmap(EC, "d_ooL")
  f_ov = dfock[SP['o'],SP['v']]
  if length(R1) > 0
    if full_tt2
      # ``R^i_a += f_k^c \tilde T^{ik}_{ac}``
      @tensoropt R1[a,i] += f_ov[k,c] * tT2[a,c,i,k]
    else
      # ``R^i_a += f_k^c U^{kX}_c \tilde T_{XY} U^{iY}_{a}``
      @tensoropt R1[a,i] += (f_ov[k,c] * UvoX[c,k,X]) * tT2[X,Y] * UvoX[a,i,Y]
    end
  end
  f_ov = nothing
  nL = size(voL, 3)
  nX = size(UvoX, 3)
  LBlks = get_auxblks(nL)
  XBlks = get_auxblks(nX)
  for L in LBlks
    V_ovL = @view ovL[:,:,L]
    if full_tt2
      # ``Y_a^{iL} = v_k^{cL} \tilde T^{ik}_{ac}``
      @tensoropt Y_voL[a,i,L] := V_ovL[k,c,L] * tT2[a,c,i,k] 
      if !full_t2
        # ``Y_X^L = Y_a^{iL} U^{†a}_{iX}`` 
        @tensoropt W_XL[X,L] := Y_voL[a,i,L] * UvoX[a,i,X]
      end
    else
      # ``Y_X^L = (v_k^{cL} U^{kY}_c) \tilde T_{XY}``) 
      @tensoropt W_XL[X,L] := (UvoX[b,j,Y] * V_ovL[j,b,L]) * tT2[X,Y] 
      # ``Y_a^{kL} = Y_X^L U^{kX}_a``
      @tensoropt Y_voL[a,k,L] := UvoX[a,k,X] * W_XL[X,L]
    end
    # ``x_a^c -= 0.5 Y_a^{kL} v_k^{cL}``
    @tensoropt x_vv[a,c] -= 0.5 * Y_voL[a,k,L] * V_ovL[k,c,L]
    # ``x_k^i += 0.5 Y_c^{iL} v_k^{cL}``
    @tensoropt x_oo[k,i] += 0.5 * Y_voL[c,i,L] * V_ovL[k,c,L]
    # ``v_X^L = U^{†a}_{iX} v_a^{iL}``
    V_voL = @view voL[:,:,L]
    if full_t2
      @tensoropt vY_voL[a,i,L] := V_voL[a,i,L] + Y_voL[a,i,L]
      # ``R^{ij}_{ab} += (v+Y)_a^{iL} (v+Y)_b^{jL'}_Y δ_{LL'}``
      @tensoropt R2[a,b,i,j] += vY_voL[a,i,L] * vY_voL[b,j,L]
      vY_voL = nothing
    else
      @tensoropt W_XL[X,L] += V_voL[a,i,L] * UvoX[a,i,X]
      # ``R_{XY} += (v+Y)^L_X (v+Y)^{L'}_Y δ_{LL'}``
      @tensoropt R2[X,Y] += W_XL[X,L] * W_XL[Y,L]
    end

    V_vvL = @view vvL[:,:,L]
    V_ooL = @view ooL[:,:,L]
    if length(R1) > 0
      # ``R^i_a += v_a^{cL} Y_c^{iL}``
      @tensoropt R1[a,i] += V_vvL[a,c,L] * Y_voL[c,i,L]
      # ``R^i_a -= v_k^{iL} Y_a^{kL}``
      @tensoropt R1[a,i] -= V_ooL[k,i,L] * Y_voL[a,k,L]
    end
    v_XLX = zeros(nX,length(L),nX)
    if project_amps_vovo_t2
      # ``v_{X'}^{XL}`` as v[X',L,X]
      for X in XBlks
        V_UvoX = @view UvoX[:,:,X]
        V_v_voX = @view v_voX[:,:,X]
        V_v_XLX = @view v_XLX[:,:,X]
        # ``v_a^{kXL} = v_a^{cL} U^{kX}_{c}``
        @tensoropt v_voXL[a,k,X,L] := V_vvL[a,c,L] * V_UvoX[c,k,X]
        # ``v_a^{iX} = v_a^{kXL} v_{k}^{iL}``    
        @tensoropt V_v_voX[a,i,X] += v_voXL[a,k,X,L] * V_ooL[k,i,L]
        # ``v_{X'}^{XL} = (v_a^{iXL} - v_k^{iL} U^{kX}_a) U^{†a}_{iX'}``
        @tensoropt V_v_XLX[X',L,X] = (v_voXL[a,i,X,L] - V_ooL[k,i,L] * V_UvoX[a,k,X]) * UvoX[a,i,X'] 
      end
      @tensoropt dR2[X,Y] += v_XLX[X,L,X'] * (dT2[X',Y'] * v_XLX[Y,L,Y'])
    else
      # ``v_X^{X'L}`` as v[X',L,X]
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
      @tensoropt dR2[X,Y] += v_XLX[X',L,X] * (dT2[X',Y'] * v_XLX[Y',L,Y])
    end
  end
  close(voLfile)
  close(ovLfile)
  close(vvLfile)
  close(ooLfile)
  if full_t2
    @tensoropt RR2[a,b,i,j] := x_vv[a,c] * T2[c,b,i,j] - x_oo[k,i] * T2[a,b,k,j]
    if project_amps_vovo_t2
      if EC.options.cc.project_vovo_t2 == 0
        # project both sides
        @tensoropt W_XX[X,X'] := v_voX[a,i,X'] * UvoX[a,i,X]
        @tensoropt dR2[X,Y] -= W_XX[X,X'] * dT2[X',Y] + dT2[X,Y'] * W_XX[Y,Y']
      else
        @tensoropt RR2[a,b,i,j] -= v_voX[a,i,X] * (T2[c,b,k,j] * UvoX[c,k,X]) 
      end
    else
      @tensoropt RR2[a,b,i,j] -= UvoX[a,i,X] * (T2[c,b,k,j] * v_voX[c,k,X]) 
      if EC.options.cc.project_vovo_t2 == 3
        # robust fitting
        @tensoropt W_XX[X,X'] := v_voX[c,k,X] * UvoX[c,k,X']
        @tensoropt dR2[X,Y] += W_XX[X,X'] * dT2[X',Y] + dT2[X,Y'] * W_XX[Y,Y']
        v_voX = calc_vᵥᵒˣ(EC) 
        @tensoropt RR2[a,b,i,j] -= v_voX[a,i,X] * (T2[c,b,k,j] * UvoX[c,k,X]) 
      end
    end
    @tensoropt R2[a,b,i,j] += RR2[a,b,i,j] + RR2[b,a,j,i]
    # project dR2 to full basis
    @tensoropt R2[a,b,i,j] += (dR2[X,Y] * UvoX[a,i,X]) * UvoX[b,j,Y]
  else
    if project_amps_vovo_t2
      # ``W_X^{X'} = (x_a^c U^{iX'}_{c} - x_k^i U^{kX'}_{a} - v_a^{iX'}) U^{†a}_{iX}``
      @tensoropt W_XX[X,X'] := (x_vv[a,c] * UvoX[c,i,X'] - x_oo[k,i] * UvoX[a,k,X'] - v_voX[a,i,X']) * UvoX[a,i,X]
    else
      # ``W_X^{X'} = (x_a^c U^{†a}_{kX} - x_k^i U^{†c}_{iX} - v_{kX}^c) U^{kX'}_c``
      @tensoropt W_XX[X,X'] := (x_vv[a,c] * UvoX[a,k,X] - x_oo[k,i] * UvoX[c,i,X] - v_voX[c,k,X]) * UvoX[c,k,X']
    end
    v_voX = nothing
    # ``R_{XY} += W_X^{X'} T_{X'Y} + T_{XY'} W_{Y}^{Y'}``
    @tensoropt R2[X,Y] += W_XX[X,X'] * T2[X',Y] + T2[X,Y'] * W_XX[Y,Y']
    W_XX = nothing
  end

  return R1, R2
end

function additional_info(EC::ECInfo)
  return """Convergence threshold:  $(EC.options.cc.thr)
            Max. iterations:        $(EC.options.cc.maxit)
            Core type:              $(EC.options.wf.core)
            Level shifts:           $(EC.options.cc.shifts) $(EC.options.cc.shiftp)
            SVD-tolerance:          $(EC.options.cc.ampsvdtol)
            # occupied orbitals to freeze:    $(EC.options.wf.freeze_nocc)
            # virtual orbitals to freeze:     $(EC.options.wf.freeze_nvirt)
            Projected contravariant exchange: $(EC.options.cc.use_projx)
            Projection in pp-hh term:         $(EC.options.cc.project_vovo_t2)
            Use full T2 for N^5 terms:        $(EC.options.cc.use_full_t2)"""
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
  print_info(methodname, additional_info(EC))
  setup_space_ms!(EC)
  flush(stdout)
  if method.theory != "DC"
    error("Only DC methods are supported in SVD!")
  end
  do_sing = (method.exclevel[1] == :full)
  # integrals
  cMO = load_orbitals(EC, EC.options.wf.orb)
  ERef = generate_DF_integrals(EC, cMO)
  cMO = nothing
  println("Reference energy: ", ERef)
  println()

  space_save = save_space(EC)
  freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc)
  freeze_nvirt!(EC, EC.options.wf.freeze_nvirt)

  # decomposition and starting guess
  fullEMP2 = calc_doubles_decomposition(EC)
  if do_sing
    T1 = read_starting_guess4amplitudes(EC, 1)
  else
    T1 = Float64[]
  end
  save_pseudodressed_3idx(EC)
  save_pseudo_dress_df_fock(EC)
  diis = Diis(EC)

  NormR1 = 0.0
  NormT1 = 0.0
  NormT2 = 0.0
  R1 = Float64[]
  Eh = 0.0
  t0 = time_ns()
  if EC.options.cc.use_full_t2
    T2 = load(EC,"T_vvoo")
  else
    T2 = load(EC,"T_XX")
  end
  # calc starting guess energy 
  truncEMP2 = calc_deco_doubles_energy(EC, T2)
  println("Starting guess energy: ", truncEMP2)
  println()
  converged = false
  println("Iter     SqNorm      Energy      DE          Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    R1, R2 = calc_svd_dcsd_residual(EC, T1, T2)
    # println("R1: ", norm(R1))
    # println("R2: ", norm(R2))
    t1 = print_time(EC,t1,"ccsd residual",2)
    Eh = calc_deco_hylleraas(EC, T1, T2, R1, R2)
    NormT2 = calc_deco_doubles_norm(T2)
    NormR2 = calc_deco_doubles_norm(R2)
    if do_sing
      NormT1 = calc_singles_norm(T1)
      NormR1 = calc_singles_norm(R1)
      T1 += update_singles(EC, R1)
    end
    T2 += update_deco_doubles(EC, R2)
    T1, T2 = perform(diis, [T1,T2], [R1,R2])
    En = calc_deco_doubles_energy(EC, T2)
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
  try2save_singles!(EC, T1)
  try2save_doubles!(EC, T2)
  println()
  @printf "Sq.Norm of T1: %12.8f Sq.Norm of T2: %12.8f \n" NormT1 NormT2
  println()
  println("$methodname correlation energy: ", Eh)
  println("$methodname total energy: ", Eh + ERef)
  println()
  if !EC.options.cc.use_full_t2
    println("$methodname corrected correlation energy: ", Eh + fullEMP2 - truncEMP2)
    println("$methodname corrected total energy: ", Eh + ERef + fullEMP2 - truncEMP2)
    println()
  end
  flush(stdout)
  delete_temporary_files!(EC)
  restore_space!(EC, space_save)
  draw_endline()
  return Eh

end

end # module DFCoupledCluster