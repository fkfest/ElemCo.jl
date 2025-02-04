"""
This module contains functions for tensor decomposition methods.
"""
module DecompTools
using LinearAlgebra, ElemCoTensorOperations
# using TSVD
using IterativeSolvers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.QMTensors

export calc_integrals_decomposition, calc_df_integrals
export eigen_decompose, svd_decompose, iter_svd_decompose
export rotate_U2pseudocanonical

"""
    calc_integrals_decomposition(EC::ECInfo)

  Decompose ``v_{pr}^{qs}`` as ``v_p^{qL} v_r^{sL}`` and store as `mmL`.
"""
function calc_integrals_decomposition(EC::ECInfo)
  pqrs = permutedims(ints2(EC,"::::",:α),(1,3,2,4))
  n = size(pqrs,1)
  if EC.options.cc.usecholesky
    CA = cholesky(Symmetric(reshape(pqrs, (n^2,n^2))), RowMaximum(), check = false, tol = EC.options.cholesky.thr)
    pqrs = nothing
    naux1 = CA.rank
    pqP = CA.U[1:naux1,invperm(CA.p)]'
  else
    B, S, = svd(reshape(pqrs, (n^2,n^2)))
    # display(S)
    pqrs = nothing

    naux1 = 0
    for s in S
      if s > EC.options.cholesky.thr
        naux1 += 1
      else
        break
      end
    end
    #println(naux1)
  
    #get integral decomposition
    pqP = B[:,1:naux1].*sqrt.(S[1:naux1]')
  end
  println("Integral auxiliary space size: ",naux1)
  save!(EC, "mmL", reshape(pqP, (n,n,naux1)))
  #B_comparison = pqP * pqP'
  #println( B_comparison ≈ reshape(pqrs, (n^2,n^2)) )
end

"""
    calc_df_integrals(EC::ECInfo)

  Calculate 3-index integrals and store them in `mmL` file.
  The routine is intended to be used in a combination with FDump integrals.
"""
function calc_df_integrals(EC::ECInfo)
  space_save = save_space(EC)
  setup_space_system!(EC; verbose=false)
  cMO = load_orbitals(EC, EC.options.wf.orb)
  freeze_core!(EC, EC.options.wf.core, EC.options.wf.freeze_nocc; verbose=false)
  freeze_nvirt!(EC, EC.options.wf.freeze_nvirt; verbose=false)
  # correlated MOs
  SP = EC.space
  if is_restricted(cMO) && SP['o'] == SP['O']
    coMO = SpinMatrix(cMO[1][:,vcat(SP['o'],SP['v'])])
  else
    coMO = SpinMatrix(cMO[1][:,vcat(SP['o'],SP['v'])], cMO[2][:,vcat(SP['O'],SP['V'])])
  end
  generate_3idx_integrals(EC, coMO, "mpfit")
  restore_space!(EC, space_save)
end

"""
    eigen_decompose(T2mat, nvirt, nocc, tol=1e-6)

  Eigenvector-decompose symmetric doubles `T2[ai,bj]` matrix: 
  ``T^{ij}_{ab} = U^{iX}_a T_{XY} U^{jY}_b δ_{XY}``.
  Return ``U^iX_a`` as `U[a,i,X]` for ``T_{XX}`` > `tol`
"""
function eigen_decompose(T2mat, nvirt, nocc, tol=1e-6)
  Tval, U = eigen(Symmetric(-T2mat))
  naux = 0
  for s in Tval
    if -s < tol
      break
    end
    naux += 1
  end
  # display(Tval[1:naux])
  # println(naux)
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose(Amat, nvirt, nocc, tol=1e-6)

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose(Amat, nvirt, nocc, tol=1e-6)
  U, S, = svd(Amat)
  # display(S)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  # display(S[1:naux])
  println("SVD-basis size: ",naux)
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose(Amat, tol=1e-6)

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose(Amat, tol=1e-6)
  U, S, = svd(Amat)
  # display(S)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  # display(S[1:naux])
  println("SVD-basis size: ",naux)
  return U[:,1:naux], S[1:naux]
end

"""
    iter_svd_decompose(Amat, nvirt, nocc, naux)

  Iteratively decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^Y_ξ``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for first `naux` ``Σ_X``
"""
function iter_svd_decompose(Amat, nvirt, nocc, naux)
  # U, S2, Vt = tsvd(Amat, naux )
  # UaiX = reshape(U[:,1:naux], (nvirt,nocc,naux))
  # U = nothing
  # S2 = nothing
  # Vt = nothing
  S2, L = svdl(Amat, nsv = naux )
  # display(S2[1:naux])
  return reshape(L.P[:,1:naux], (nvirt,nocc,naux))
  # display(UaiX)
end

""" 
    rotate_U2pseudocanonical(EC::ECInfo, UaiX)

  Diagonalize ϵv - ϵo transformed with UaiX (for update).
  Return eigenvalues and rotated UaiX
"""
function rotate_U2pseudocanonical(EC::ECInfo, UaiX)
  SP = EC.space
  nocc = n_occ_orbs(EC)
  nvirt = n_virt_orbs(EC)
  UaiX2 = deepcopy(UaiX)
  ϵo, ϵv = orbital_energies(EC)
  for a in 1:nvirt
    for i in 1:nocc
      UaiX2[a,i,:] *= ϵv[a] - ϵo[i]
    end
  end

  @mtensor Fdiff[X,Y] := UaiX[a,i,X] * UaiX2[a,i,Y]
  diagFdiff = eigen(Symmetric(Fdiff))

  @mtensor UaiX2[a,i,Y] = diagFdiff.vectors[X,Y] * UaiX[a,i,X]
  return diagFdiff.values, UaiX2
end


end #module
