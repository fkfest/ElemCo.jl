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
    get_ssv_osvˣˣ(EC::ECInfo)

  Return ``ssv^{XY} = U^{kX}_c U^{lX}_d (v_{kl}^{cd} - v_{lk}^{cd} )``
  and ``osv^{XY} = U^{kX}_c U^{lX}_d v_{kl}^{cd}``
  with ``v_{kl}^{cd} = v_k^{cL} v_l^{dL'}δ_{LL'}``. 

  The integrals will be read from files `ssd_^XX` and `osd_^XX`. 
  If the files do not exist, the integrals will be calculated
  and stored in files `ssd_^XX` and `osd_^XX`.
  v_k^{cL} and U^{kX}_c are read from files `d_ovL` and `C_voX`.
"""
function get_ssv_osvˣˣ(EC::ECInfo)
  if file_exists(EC, "ssd_^XX") 
    return load(EC, "ssd_^XX"), load(EC, "osd_^XX")
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
  @tensoropt begin
    ssvxx[X,Y] := (vdagger_vvoo[a,b,i,j] - vdagger_vvoo[a,b,j,i]) * UvoX[a,i,X] * UvoX[b,j,Y]
    osvxx[X,Y] := vdagger_vvoo[a,b,i,j] * UvoX[a,i,X] * UvoX[b,j,Y]
  end
  save!(EC, "ssd_^XX", ssvxx)
  save!(EC, "osd_^XX", osvxx)
  return ssvxx, osvxx
end

"""
    gen_vₓˣᴸ(EC::ECInfo)

  Generate ``v_X^{X'L} = v_a^{cL} U^{†a}_{kX} U^{kX'}_c`` using bare integrals.

  The integrals and the SVD-coefficients are read from files `mmL` and `C_voX`,
  and the result is stored in file `X^XL`.
"""
function gen_vₓˣᴸ(EC::ECInfo)
  t1 = time_ns()
  UvoX = load(EC, "C_voX")
  mmLfile, mmL = mmap(EC, "mmL")
  SP = EC.space
  nL = size(mmL, 3)
  nX = size(UvoX, 3)
  # create mmap for the v_X^{X'L} intermediate
  vXXLfile, v_XXL = newmmap(EC, "X^XL", (nX,nX,nL))
  LBlks = get_auxblks(nL)
  XBlks = get_auxblks(nX)
  for L in LBlks
    vvL = mmL[SP['v'],SP['v'],L]
    for X in XBlks
      V_UvoX = @view UvoX[:,:,X]
      # ``v_{X'}^{XL} = (v_a^{cL} U^{kX}_{c}) U^{†a}_{kX'}``
      @tensoropt v_XXL[:,X,L][X',X,L] = (vvL[a,c,L] * V_UvoX[c,k,X]) * UvoX[a,k,X'] 
    end
  end
  closemmap(EC, vXXLfile, v_XXL)
  close(mmLfile)
  t1 = print_time(EC, t1, "v_X^{X'L}", 2)
end

"""
    calc_deco_hylleraas(EC::ECInfo, T1, T2, R1, R2)

  Calculate closed-shell singles and doubles Hylleraas energy
  using contravariant decomposed doubles amplitudes `T2`=``T_{XY}``
  or full contravariant doubles amplitude `T2`=``T^{ij}_{ab}``.

  Returns total energy, SS, OS and Openshell (0.0) contributions
  as a NamedTuple (`E`,`ESS`,`EOS`,`EO`).
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
    @tensoropt begin
      int2[a,b,i,j] := R2[a,b,i,j] + ovL[i,a,L] * ovL[j,b,L]
      ET2d = T2[a,b,i,j] * int2[a,b,i,j]
      ET2ex = T2[a,b,j,i] * int2[a,b,i,j]
    end
    ET2SS = ET2d - ET2ex
    ET2OS = ET2d
    ET2 = ET2SS + ET2OS
    ovL = nothing
  else
    ssvxx, osvxx = get_ssv_osvˣˣ(EC)
    @tensoropt begin
      ET2OS = T2[X,Y] * (osvxx[X,Y] + R2[X,Y])
      ET2SS = T2[Y,X] * (ssvxx[X,Y] + R2[X,Y])
    end
    UvoX = load(EC, "C_voX")
    @tensoropt ET2SS -= T2[X,Y] * ((((R2[X',Y'] * UvoX[a,i,X']) * UvoX[b,j,Y']) * UvoX[a,j,X]) * UvoX[b,i,Y])
    UvoX = nothing
    ET2 = ET2SS + ET2OS
  end
  if length(T1) > 0
    dfockc_ov = load(EC, "dfc_ov")
    dfocke_ov = load(EC, "dfe_ov")
    @tensoropt begin
      ET1d = T1[a,i] * dfockc_ov[i,a] 
      ET1ex = T1[a,i] * dfocke_ov[i,a]
    end
    ET1SS = ET1d - ET1ex
    ET1OS = ET1d
    ET1 = ET1SS + ET1OS
    fov = load(EC,"f_mm")[SP['o'],SP['v']] 
    @tensoropt ET1 += 2.0*((fov[i,a] + 2.0 * R1[a,i])*T1[a,i])
    ET2 += ET1
    ET2SS += ET1SS
    ET2OS += ET1OS
  end
  return (E=ET2, ESS=ET2SS, EOS=ET2OS, EO=0.0)
end

"""
    calc_deco_doubles_energy(EC::ECInfo, T2)

  Calculate closed-shell doubles energy
  using decomposed doubles amplitudes `T2`=``T_{XY}``
  or `T2`=``T^{ij}_{ab}`` using density-fitted integrals.

  Returns total energy, SS, OS and Openshell (0.0) contributions
  as a NamedTuple (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_deco_doubles_energy(EC::ECInfo, T2)
  if ndims(T2) == 4
    return calc_df_doubles_energy(EC, T2)
  elseif ndims(T2) == 2
    ssvxx, osvxx = get_ssv_osvˣˣ(EC)
    @tensoropt begin
      ET2OS = T2[X,Y] * osvxx[X,Y] 
      ET2SS = T2[Y,X] * ssvxx[X,Y]
    end
    ET2 = ET2SS + ET2OS
    return (E=ET2, ESS=ET2SS, EOS=ET2OS, EO=0.0)
  else
    error("Wrong dimensionality of T2: ", ndims(T2))
  end
end

"""
    calc_df_doubles_energy(EC::ECInfo, T2)

  Calculate closed-shell doubles energy using DF integrals 
  and `T2[a,b,i,j]` = ``T^{ij}_{ab}``.

  Returns total energy, SS, OS and Openshell (0.0) contributions
  as a NamedTuple (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_df_doubles_energy(EC::ECInfo, T2)
  if !file_exists(EC, "d_ovL")
    error("File d_ovL does not exist!")
  end
  ovL = load(EC, "d_ovL")
  @tensoropt begin
    int2[a,b,i,j] := ovL[i,a,L] * ovL[j,b,L]
    ET2d = T2[a,b,i,j] * int2[a,b,i,j]
    ET2ex = T2[b,a,i,j] * int2[a,b,i,j]
  end
  ET2SS = ET2d - ET2ex
  ET2OS = ET2d
  ET2 = ET2SS + ET2OS
  return (E=ET2, ESS=ET2SS, EOS=ET2OS, EO=0.0)
end

"""
    calc_dressed_3idx(EC::ECInfo, T1)

  Calculate dressed integrals for 3-index integrals from file `mmL`.
"""
function calc_dressed_3idx(EC::ECInfo, T1)
  mmLfile, mmL = mmap(EC, "mmL")
  # println(size(mmL))
  SP = EC.space
  nL = size(mmL, 3)
  nocc = length(SP['o'])
  nvirt = length(SP['v'])
  # create mmaps for dressed integrals
  ovLfile, ovL = newmmap(EC, "d_ovL", (nocc,nvirt,nL))
  voLfile, voL = newmmap(EC, "d_voL", (nvirt,nocc,nL))
  ooLfile, ooL = newmmap(EC, "d_ooL", (nocc,nocc,nL))
  vvLfile, vvL = newmmap(EC, "d_vvL", (nvirt,nvirt,nL))

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
    save_pseudodressed_3idx(EC::ECInfo)

  Save non-dressed 3-index integrals from file `mmL` to dressed files.
"""
function save_pseudodressed_3idx(EC::ECInfo)
  mmLfile, mmL = mmap(EC, "mmL")
  # println(size(mmL))
  SP = EC.space
  nL = size(mmL, 3)
  nocc = length(SP['o'])
  nvirt = length(SP['v'])
  # create mmaps for dressed integrals
  ovLfile, ovL = newmmap(EC, "d_ovL", (nocc,nvirt,nL))
  voLfile, voL = newmmap(EC, "d_voL", (nvirt,nocc,nL))
  ooLfile, ooL = newmmap(EC, "d_ooL", (nocc,nocc,nL))
  vvLfile, vvL = newmmap(EC, "d_vvL", (nvirt,nvirt,nL))

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
    dress_df_fock(EC::ECInfo, T1)

  Dress DF fock matrix with DF 3-index integrals.

  The dress-contribution is added to the original fock matrix
  from file `f_mm`. The dressed fock matrix is stored in file `df_mm`.
  Additionally, the coulomb and exchange dressing contributions to ``\\hat f_k^c`` 
  are stored in files `dfc_ov` and `dfe_ov`.
"""
function dress_df_fock(EC::ECInfo, T1)
  dfock = load(EC, "f_mm")
  mmLfile, mmL = mmap(EC, "mmL")
  nL = size(mmL, 3)
  occ = EC.space['o']
  virt = EC.space['v']

  LBlks = get_auxblks(nL)
  dfockc = zeros(size(dfock))
  dfocke = zeros(size(dfock))
  for L in LBlks
    V_mmL = @view mmL[:,:,L]
    mvL = V_mmL[:,virt,:]
    @tensoropt vt_moL[p,i,L] := mvL[p,a,L]*T1[a,i]
    mvL = nothing
    @tensoropt vt_L[L] := vt_moL[occ,:,:][i,i,L]
    # exchange
    omL = V_mmL[occ,:,:]
    @tensoropt dfocke[p,q] += vt_moL[p,i,L]*omL[i,q,L]
    omL = nothing
    # coulomb
    @tensoropt dfockc[p,q] += V_mmL[p,q,L]*vt_L[L]
  end
  close(mmLfile)
  dfock += 2.0*dfockc - dfocke
  save!(EC, "dfc_ov", dfockc[occ,virt], description="tmp Coulomb-Dressed-Part-Fock")
  save!(EC, "dfe_ov", dfocke[occ,virt], description="tmp Exchange-Dressed-Part-Fock")
  # dress external indices
  dinter = dfock[:,virt]
  @tensoropt dfock[:,occ][p,j] += dinter[p,b] * T1[b,j]
  dinter = dfock[occ,:]
  @tensoropt dfock[virt,:][b,p] -= dinter[j,p] * T1[b,j]
  save!(EC, "df_mm", dfock)
end

"""
    save_pseudo_dress_df_fock(EC::ECInfo)

  Save non-dressed DF fock matrix from file `f_mm` to dressed file `df_mm`.
"""
function save_pseudo_dress_df_fock(EC::ECInfo)
  dfock = load(EC, "f_mm")
  dfc = zeros(size(dfock))
  occ = EC.space['o']
  virt = EC.space['v']
  save!(EC, "dfc_ov", dfc[occ,virt])
  save!(EC, "dfe_ov", dfc[occ,virt])
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
  1. ``\\bar U^{i\\bar X}_a`` is calculated from ``v_a^{iL}`` using SVD (with threshold [`CcOptions.ampsvdtol`](@ref ECInfos.CcOptions)×`CcOptions.ampsvdfac`);
  2. MP2 doubles ``T^{i}_{aX}`` are calculated from ``v_a^{iL}`` and ``U^{iX}_a`` and again decomposed using SVD and threshold [`CcOptions.ampsvdtol`](@ref ECInfos.CcOptions).
  The SVD-basis is rotated to pseudocanonical basis to diagonalize 
  orbital-energy differences, ``ϵ_X = U^{iX}_{a}(ϵ_a-ϵ_i)U^{iX}_a``.
  The imaginary shift [`CcOptions.deco_ishiftp`](@ref ECInfos.CcOptions) is used in the denominator in the calculation of the MP2 amplitudes.
  The orbital energy differences are saved in file `e_X`.
  The SVD-coefficients ``U^{iX}_a`` are saved in file `C_voX`.
  The starting guess for doubles ``T_{XY}`` is saved in file `T_XX`.
  Return full MP2 correlation energy, SS, OS, and Openshell(0.0) (using the imaginary shift)
  as a NamedTuple (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_doubles_decomposition_without_doubles(EC::ECInfo)
  t1 = time_ns()
  println("Decomposition without doubles using threshold ", EC.options.cc.ampsvdtol)
  flush(stdout)
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  SP = EC.space
  # first approximation for U^{iX}_a from 3-index integrals v_a^{iL} 
  # TODO: add shifted Laplace transform!
  mmLfile, mmL = mmap(EC, "mmL")
  nL = size(mmL, 3)
  tol2 = (EC.options.cc.ampsvdtol*EC.options.cc.ampsvdfac)
  voL = mmL[SP['v'],SP['o'],:]
  shifti = EC.options.cc.deco_ishiftp
  fullEMP2 = calc_MP2_from_3idx(EC, voL, shifti)
  if shifti ≈ 0.0
    println("MP2 correlation energy: ", fullEMP2.E)
  else
    println("MP2 imaginary shift for decomposition: ", shifti)
    println("MP2 imaginary shifted correlation energy: ", fullEMP2.E)
  end
  t1 = print_time(EC, t1, "MP2 from 3idx", 2)
  flush(stdout)
  if EC.options.cc.use_full_t2
    T2 = try2start_doubles(EC)
    if size(T2) != (nvirt,nvirt,nocc,nocc)
      T2 = calc_MP2_amplitudes_from_3idx(EC, voL, shifti)
    end
    save!(EC, "T_vvoo", T2)
    T2 = nothing
  end
  UaiX = svd_decompose(reshape(voL, (nvirt*nocc,nL)), nvirt, nocc, tol2)
  t1 = print_time(EC, t1, "SVD decomposition", 2)
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
  t1 = print_time(EC, t1, "half-decomposed MP2", 2)
  naux = size(voX, 3)
  # decompose T^i_{aX}
  UaiX = svd_decompose(reshape(voX, (nvirt*nocc,naux)), nvirt, nocc, EC.options.cc.ampsvdtol)
  t1 = print_time(EC, t1, "T^i_{aX} SVD decomposition", 2)
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
  t1 = print_time(EC, t1, "starting guess T_{XY}", 2)
  return fullEMP2
end

"""
    calc_doubles_decomposition_with_doubles(EC::ECInfo)

  Decompose ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` using explicit doubles amplitudes ``T^{ij}_{ab}``.
"""
function calc_doubles_decomposition_with_doubles(EC::ECInfo)
  t1 = time_ns()
  println("Decomposition with doubles using threshold ", EC.options.cc.ampsvdtol)
  flush(stdout)
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  SP = EC.space
  mmLfile, mmL = mmap(EC, "mmL")
  nL = size(mmL, 3)
  voL = mmL[SP['v'],SP['o'],:]
  T2 = try2start_doubles(EC)
  if size(T2) != (nvirt,nvirt,nocc,nocc)
    println("Use MP2 doubles for decomposition")
    flush(stdout)
    shifti = EC.options.cc.deco_ishiftp
    T2 = calc_MP2_amplitudes_from_3idx(EC, voL, shifti)
  end
  if EC.options.cc.use_full_t2
    save!(EC, "T_vvoo", T2)
  end
  t1 = print_time(EC, t1, "MP2 from 3idx", 2)
  println("decompose full doubles (can be slow!)")
  flush(stdout)
  UaiX = svd_decompose(reshape(permutedims(T2, (1,3,2,4)), (nvirt*nocc,nvirt*nocc)), nvirt, nocc, EC.options.cc.ampsvdtol)
  t1 = print_time(EC, t1, "SVD decomposition", 2)
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
    t1 = print_time(EC, t1, "starting guess T_{XY}", 2)
  end
  return (E=0.0,)
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
  Returns total energy, SS, OS and Openshell (0.0) contributions
  as a NamedTuple (`E`,`ESS`,`EOS`,`EO`).
"""
function calc_MP2_from_3idx(EC::ECInfo, voL::AbstractArray, ishift)
  @tensoropt vvoo[a,b,i,j] := voL[a,i,L] * voL[b,j,L]
  ϵo, ϵv = orbital_energies(EC)
  nocc = length(ϵo)
  nvirt = length(ϵv)
  ET2d = 0.0
  ET2ex = 0.0
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
    @tensoropt begin
      ET2d += vvo[a,b,i] * t_vvo[a,b,i]
      ET2ex += vvo[a,b,i] * t_vvo[b,a,i]
    end
  end
  ET2SS = ET2d - ET2ex
  ET2OS = ET2d
  ET2 = ET2SS + ET2OS
  return (E=ET2, ESS=ET2SS, EOS=ET2OS, EO=0.0)
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
    calc_voX(EC::ECInfo; calc_vᵥᵒˣ=false, calc_vᵛₒₓ=false)

  Calculate ``\\hat v_a^{iX} = \\hat v_{ak}^{ci} U^{kX}_c`` 
  and/or ``\\hat v^c_{kX} = \\hat v_{ak}^{ci} U^{†a}_{kX}``
  with ``\\hat v_{ak}^{ci} = \\hat v_a^{cL} \\hat v_k^{iL}`` 
  and ``U^{kX}_c`` from file `C_voX`.

  Return a tuple (vᵥᵒˣ, vᵛₒₓ) (not calculated intermediates are empty arrays).
"""
function calc_voX(EC::ECInfo; calc_vᵥᵒˣ=false, calc_vᵛₒₓ=false)
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
  v_voX = v_voX2 = similar(UvoX, 0)
  if calc_vᵥᵒˣ
    # ``v_a^{iX} = v_{ac}^{ki} U^{kX}_c``
    @tensoropt v_voX[a,i,X] := vvoo[a,c,k,i] * UvoX[c,k,X]
  end
  if calc_vᵛₒₓ
    # ``v_{kX}^{c} = v_{ak}^{ci} U^{†a}_{kX}``
    @tensoropt v_voX2[c,k,X] := vvoo[a,c,k,i] * UvoX[a,i,X]
  end
  return (v_voX, v_voX2)
end

"""
    calc_svd_dcsd_residual(EC::ECInfo, T1, T2)

  Calculate decomposed closed-shell DCSD residual with
  ``T^{ij}_{ab}=U^{iX}_a U^{jY}_b T_{XY}`` and
  ``R_{XY}=U^{iX†}_a U^{jY†}_b R^{ij}_{ab}``.
  `T2` contains decomposed amplitudes ``T_{XY}`` or full amplitudes ``T^{ij}_{ab}``.
 
  If `T2` is ``T^{ij}_{ab}``, the residual is also returned in full form.
"""
function calc_svd_dcsd_residual(EC::ECInfo, T1, T2)
  t1 = time_ns()
  if length(T1) > 0
    #get dressed integrals
    calc_dressed_3idx(EC, T1)
    t1 = print_time(EC, t1, "dressed 3-idx integrals", 2)
    dress_df_fock(EC, T1)
    t1 = print_time(EC, t1, "dressed fock", 2)
  end
  UvoX = load(EC, "C_voX")
  use_projected_exchange = EC.options.cc.use_projx
  project_amps_vovo_t2 = EC.options.cc.project_vovo_t2 != 2
  project_resid_vovo_t2 = EC.options.cc.project_vovo_t2 >= 2

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
  vvLfile, vvL = mmap(EC, "d_vvL")
  ooLfile, ooL = mmap(EC, "d_ooL")
  f_ov = dfock[SP['o'],SP['v']]
  if length(R1) > 0
    if full_tt2
      # ``R^i_a += f_k^c \tilde T^{ik}_{ac}``
      @tensoropt R1[a,i] += f_ov[k,c] * tT2[a,c,i,k]
      t1 = print_time(EC, t1, "``R^i_a += f_k^c \\tilde T^{ik}_{ac}``", 2)
    else
      # ``R^i_a += f_k^c U^{kX}_c \tilde T_{XY} U^{iY}_{a}``
      @tensoropt R1[a,i] += (f_ov[k,c] * UvoX[c,k,X]) * tT2[X,Y] * UvoX[a,i,Y]
      t1 = print_time(EC, t1, "``R^i_a += f_k^c U^{kX}_c \\tilde T_{XY} U^{iY}_{a}``", 2)
    end
  end
  f_ov = nothing
  nL = size(voL, 3)
  nX = size(UvoX, 3)
  LBlks = get_auxblks(nL)
  for L in LBlks
    V_ovL = @view ovL[:,:,L]
    if full_tt2
      # ``Y_a^{iL} = v_k^{cL} \tilde T^{ik}_{ac}``
      @tensoropt Y_voL[a,i,L] := V_ovL[k,c,L] * tT2[a,c,i,k] 
      t1 = print_time(EC, t1, "``Y_a^{iL} = v_k^{cL} \\tilde T^{ik}_{ac}``", 2)
      if !full_t2
        # ``Y_X^L = Y_a^{iL} U^{†a}_{iX}`` 
        @tensoropt W_XL[X,L] := Y_voL[a,i,L] * UvoX[a,i,X]
        t1 = print_time(EC, t1, "``Y_X^L = Y_a^{iL} U^{†a}_{iX}``", 2)
      end
    else
      # ``Y_X^L = (v_k^{cL} U^{kY}_c) \tilde T_{XY}``) 
      @tensoropt W_XL[X,L] := (UvoX[b,j,Y] * V_ovL[j,b,L]) * tT2[X,Y]
      t1 = print_time(EC, t1, "``Y_X^L = (v_k^{cL} U^{kY}_c) \\tilde T_{XY}``", 2)
      # ``Y_a^{kL} = Y_X^L U^{kX}_a``
      @tensoropt Y_voL[a,k,L] := UvoX[a,k,X] * W_XL[X,L]
      t1 = print_time(EC, t1, "``Y_a^{kL} = Y_X^L U^{kX}_a``", 2)
    end
    # ``x_a^c -= 0.5 Y_a^{kL} v_k^{cL}``
    @tensoropt x_vv[a,c] -= 0.5 * Y_voL[a,k,L] * V_ovL[k,c,L]
    t1 = print_time(EC, t1, "``x_a^c -= 0.5 Y_a^{kL} v_k^{cL}``", 2)
    # ``x_k^i += 0.5 Y_c^{iL} v_k^{cL}``
    @tensoropt x_oo[k,i] += 0.5 * Y_voL[c,i,L] * V_ovL[k,c,L]
    t1 = print_time(EC, t1, "``x_k^i += 0.5 Y_c^{iL} v_k^{cL}``", 2)
    # ``v_X^L = U^{†a}_{iX} v_a^{iL}``
    V_voL = @view voL[:,:,L]
    if full_t2
      @tensoropt vY_voL[a,i,L] := V_voL[a,i,L] + Y_voL[a,i,L]
      # ``R^{ij}_{ab} += (v+Y)_a^{iL} (v+Y)_b^{jL'}_Y δ_{LL'}``
      @tensoropt R2[a,b,i,j] += vY_voL[a,i,L] * vY_voL[b,j,L]
      vY_voL = nothing
      t1 = print_time(EC, t1, "``R^{ij}_{ab} += (v+Y)_a^{iL} (v+Y)_b^{jL'}_Y δ_{LL'}``", 2)
    else
      @tensoropt W_XL[X,L] += V_voL[a,i,L] * UvoX[a,i,X]
      # ``R_{XY} += (v+Y)^L_X (v+Y)^{L'}_Y δ_{LL'}``
      @tensoropt R2[X,Y] += W_XL[X,L] * W_XL[Y,L]
      t1 = print_time(EC, t1, "``R_{XY} += (v+Y)^L_X (v+Y)^{L'}_Y δ_{LL'}``", 2)
    end

    V_vvL = @view vvL[:,:,L]
    V_ooL = @view ooL[:,:,L]
    if length(R1) > 0
      # ``R^i_a += v_a^{cL} Y_c^{iL}``
      @tensoropt R1[a,i] += V_vvL[a,c,L] * Y_voL[c,i,L]
      t1 = print_time(EC, t1, "``R^i_a += v_a^{cL} Y_c^{iL}``", 2)
      # ``R^i_a -= v_k^{iL} Y_a^{kL}``
      @tensoropt R1[a,i] -= V_ooL[k,i,L] * Y_voL[a,k,L]
      t1 = print_time(EC, t1, "``R^i_a -= v_k^{iL} Y_a^{kL}``", 2)
    end
  end
  voL = nothing
  close(voLfile)
  vvL = nothing
  close(vvLfile)
  if length(T1) > 0
    # ``U^{i}_{jX} = U^{†b}_{jX} T^i_b``
    @tensoropt UTooX[i,j,X] := UvoX[b,j,X] * T1[b,i]
    t1 = print_time(EC, t1, "``U^{i}_{jX} = U^{†b}_{jX} T^i_b``", 2)
  end
  XBigBlks = get_auxblks(nX, 512)
  XXLfile, XXL = mmap(EC, "X^XL")
  d_XXLfile, d_XXL = newmmap(EC, "d_X^XL", (nX,nX,nL))
  for X in XBigBlks
    V_UvoX = @view UvoX[:,:,X]
    # ``U_{iX}^{jY} = U^{†c}_{iX} U^{jY}_{c}``
    @tensoropt UUooXX[i,j,X,Y] := UvoX[c,i,X] * V_UvoX[c,j,Y]
    t1 = print_time(EC, t1, "``U_{iX}^{jY} = U^{†c}_{iX} U^{jY}_{c}``", 2)
    for L in LBlks
      V_ooL = @view ooL[:,:,L]
      V_ovL = @view ovL[:,:,L]
      v_XXL = deepcopy(XXL[:,X,L])
      if length(T1) > 0
        # ``v_X^{YL} -= v_{l}^{cL} U^{kY}_c U^{l}_{kX}``
        @tensoropt v_XXL[X,Y,L] -= (V_ovL[l,c,L] * V_UvoX[c,k,Y]) * UTooX[l,k,X]
        t1 = print_time(EC, t1, "``v_X^{YL} -= v_{l}^{cL} U^{kY}_c U^{l}_{kX}``", 2)
      end
      # ``v_X^{YL} -= \hat v_{j}^{iL} U_{iX}^{jY}``
      @tensoropt v_XXL[X,Y,L] -= V_ooL[j,i,L] * UUooXX[i,j,X,Y]
      t1 = print_time(EC, t1, "``v_X^{YL} -= \\hat v_{j}^{iL} U_{iX}^{jY}``", 2)
      d_XXL[:,X,L] = v_XXL
    end
  end
  closemmap(EC, d_XXLfile, d_XXL)
  ovL = nothing
  close(ovLfile)
  ooL = nothing
  close(ooLfile)
  XXL = nothing
  close(XXLfile)
  UTooX = nothing
  d_XXLfile, d_XXL = mmap(EC, "d_X^XL")
  for L in LBlks
    v_XXL = @view d_XXL[:,:,L]
    # ``R_{XY} += v_X^{X'L} T_{X'Y'} v_Y^{Y'L}``
    @tensoropt dR2[X,Y] += v_XXL[X,X',L] * (dT2[X',Y'] * v_XXL[Y,Y',L])
    t1 = print_time(EC, t1, "``R_{XY} += v_{X}^{X'L} T_{X'Y'} v_{Y}^{Y'L}``", 2)
  end
  d_XXL = nothing
  close(d_XXLfile)
  calc_vᵥᵒˣ = !full_t2 || project_amps_vovo_t2
  calc_vᵛₒₓ = full_t2 && project_resid_vovo_t2
  vᵥᵒˣ, vᵛₒₓ = calc_voX(EC; calc_vᵥᵒˣ, calc_vᵛₒₓ) 
  t1 = print_time(EC, t1, "calc v_a^{iX}", 2)
  if full_t2
    @tensoropt RR2[a,b,i,j] := x_vv[a,c] * T2[c,b,i,j] - x_oo[k,i] * T2[a,b,k,j]
    t1 = print_time(EC, t1, "``R_{ab}^{ij} += x_a^c T_{cb}^{ij} - x_k^i T_{ab}^{kj}``", 2)
    if project_resid_vovo_t2
      @tensoropt RR2[a,b,i,j] -= UvoX[a,i,X] * (T2[c,b,k,j] * vᵛₒₓ[c,k,X]) 
      t1 = print_time(EC, t1, "``R_{ab}^{ij} -= U^{iX}_a T_{cb}^{kj} v_{kX}^{c}``", 2)
      if project_amps_vovo_t2
        # robust fitting
        @tensoropt W_XX[X,X'] := vᵛₒₓ[c,k,X] * UvoX[c,k,X']
        @tensoropt dR2[X,Y] += W_XX[X,X'] * dT2[X',Y] + dT2[X,Y'] * W_XX[Y,Y']
        t1 = print_time(EC, t1, "``R_{XY} += v_{X}^{X'} T_{X'Y} + T_{XY'} v_{Y}^{Y'}``", 2)
        @tensoropt RR2[a,b,i,j] -= vᵥᵒˣ[a,i,X] * (T2[c,b,k,j] * UvoX[c,k,X]) 
        t1 = print_time(EC, t1, "``R_{ab}^{ij} -= v_a^{iX} T_{cb}^{kj} U^{c}_{kX}``", 2)
      end
    else
      if EC.options.cc.project_vovo_t2 == 0
        # project both sides
        @tensoropt W_XX[X,X'] := vᵥᵒˣ[a,i,X'] * UvoX[a,i,X]
        @tensoropt dR2[X,Y] -= W_XX[X,X'] * dT2[X',Y] + dT2[X,Y'] * W_XX[Y,Y']
        t1 = print_time(EC, t1, "``R_{XY} += v_{X}^{X'} T_{X'Y} + T_{XY'} v_{Y}^{Y'}``", 2)
      else
        @tensoropt RR2[a,b,i,j] -= vᵥᵒˣ[a,i,X] * (T2[c,b,k,j] * UvoX[c,k,X]) 
        t1 = print_time(EC, t1, "``R_{ab}^{ij} -= v_a^{iX} T_{cb}^{kj} U^{c}_{kX}``", 2)
      end
    end
    @tensoropt R2[a,b,i,j] += RR2[a,b,i,j] + RR2[b,a,j,i]
    # project dR2 to full basis
    @tensoropt R2[a,b,i,j] += (dR2[X,Y] * UvoX[a,i,X]) * UvoX[b,j,Y]
    t1 = print_time(EC, t1, "project R2 to full basis", 2)
  else
    # ``W_X^{X'} = (x_a^c U^{iX'}_{c} - x_k^i U^{kX'}_{a} - v_a^{iX'}) U^{†a}_{iX}``
    @tensoropt W_XX[X,X'] := (x_vv[a,c] * UvoX[c,i,X'] - x_oo[k,i] * UvoX[a,k,X'] - vᵥᵒˣ[a,i,X']) * UvoX[a,i,X]
    t1 = print_time(EC, t1, "``W_X^{X'} = (x_a^c U^{iX'}_{c} - x_k^i U^{kX'}_{a} - v_a^{iX'}) U^{†a}_{iX}``", 2)
    # ``R_{XY} += W_X^{X'} T_{X'Y} + T_{XY'} W_{Y}^{Y'}``
    @tensoropt R2[X,Y] += W_XX[X,X'] * T2[X',Y] + T2[X,Y'] * W_XX[Y,Y']
    W_XX = nothing
    t1 = print_time(EC, t1, "``R_{XY} += W_X^{X'} T_{X'Y} + T_{XY'} W_{Y}^{Y'}``", 2)
  end

  return R1, R2
end

function additional_info(EC::ECInfo)
  return """Convergence threshold:  $(EC.options.cc.thr)
            Max. iterations:        $(EC.options.cc.maxit)
            Core type:              $(EC.options.wf.core)
            Level shifts:           $(EC.options.cc.shifts) $(EC.options.cc.shiftp)
            SVD-tolerance:          $(EC.options.cc.ampsvdtol)
            SVD-factor for 2-step:  $(EC.options.cc.ampsvdfac)
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
  DF integrals are used (have to be calculated before).
  The starting guess for SVD-coefficients is calculated without doubles,
  see [`calc_doubles_decomposition_without_doubles`](@ref).
"""
function calc_svd_dc(EC::ECInfo, method::ECMethod)
  t1 = time_ns()
  print_info(method_name(method), additional_info(EC))
  if method.theory != "DC"
    error("Only DC methods are supported in SVD!")
  end
  do_sing = (method.exclevel[1] == :full)

  # decomposition and starting guess
  fullEMP2 = calc_doubles_decomposition(EC)
  t1 = print_time(EC, t1, "doubles decomposition", 2)
  if do_sing
    T1 = read_starting_guess4amplitudes(EC, 1)
  else
    T1 = Float64[]
  end
  save_pseudodressed_3idx(EC)
  save_pseudo_dress_df_fock(EC)
  t1 = print_time(EC, t1, "save pseudodressed 3-idx and fock", 2)

  println("Calculating intermediates...")
  gen_vₓˣᴸ(EC)
  t1 = print_time(EC, t1, "intermediates", 2)

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
  t1 = print_time(EC, t1, "calc starting guess energy", 2)
  println("Starting guess energy: ", truncEMP2.E)
  println()
  converged = false
  println("Iter     SqNorm      Energy      DE          Res         Time")
  for it in 1:EC.options.cc.maxit
    t1 = time_ns()
    R1, R2 = calc_svd_dcsd_residual(EC, T1, T2)
    # println("R1: ", norm(R1))
    # println("R2: ", norm(R2))
    t1 = print_time(EC,t1,"dcsd residual",2)
    Eh = calc_deco_hylleraas(EC, T1, T2, R1, R2)
    t1 = print_time(EC,t1,"calc hylleraas",2)
    NormT2 = calc_deco_doubles_norm(T2)
    NormR2 = calc_deco_doubles_norm(R2)
    if do_sing
      NormT1 = calc_singles_norm(T1)
      NormR1 = calc_singles_norm(R1)
      T1 += update_singles(EC, R1)
    end
    T2 += update_deco_doubles(EC, R2)
    t1 = print_time(EC,t1,"update amplitudes",2)
    perform!(diis, [T1,T2], [R1,R2])
    t1 = print_time(EC,t1,"DIIS",2)
    En2 = calc_deco_doubles_energy(EC, T2)
    En = En2.E
    if do_sing
      En1 = calc_singles_energy_using_dfock(EC, T1)
      En += En1.E
    end
    ΔE = En - Eh.E
    NormR = NormR1 + NormR2
    NormT = 1.0 + NormT1 + NormT2
    tt = (time_ns() - t0)/10^9
    @printf "%3i %12.8f %12.8f %12.8f %10.2e %8.2f \n" it NormT Eh.E ΔE NormR tt
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
  flush(stdout)
  if !EC.options.cc.use_full_t2
    # ΔMP2 correction
    Eh = (; Eh..., Ecorrection=fullEMP2.E - truncEMP2.E)
  end
  return Eh
end

end # module DFCoupledCluster
