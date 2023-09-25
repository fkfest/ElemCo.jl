"""
This module contains functions for tensor decomposition methods.
"""
module DecompTools
using LinearAlgebra, TensorOperations
# using TSVD
using IterativeSolvers
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.OrbTools

export calc_integrals_decomposition, eigen_decompose, svd_decompose, iter_svd_decompose
export rotate_U2pseudocanonical

"""
    calc_integrals_decomposition(EC::ECInfo)

  Decompose ``v_{pr}^{qs}`` as ``v_p^{qL} v_r^{sL}`` and store as `mmL`.
"""
function calc_integrals_decomposition(EC::ECInfo)
  pqrs = permutedims(ints2(EC,"::::",:α),(1,3,2,4))
  n = size(pqrs,1)
  B, S, Bt = svd(reshape(pqrs, (n^2,n^2)))
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
  save!(EC, "mmL", reshape(pqP, (n,n,naux1)))
  #B_comparison = pqP * pqP'
  #println( B_comparison ≈ reshape(pqrs, (n^2,n^2)) )
end

"""
    eigen_decompose(T2mat, nvirt, nocc, tol = 1e-6)

  Eigenvector-decompose symmetric doubles T2[ai,bj] matrix: 
  ``T^{ij}_{ab} = U^{iX}_a * T_{XY} * U^{jY}_b δ_{XY}``.
  Return ``U^iX_a`` as `U[a,i,X]` for ``T_{XX}`` > `tol`
"""
function eigen_decompose(T2mat, nvirt, nocc, tol = 1e-6)
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
    svd_decompose(Amat, nvirt, nocc, tol = 1e-6)

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a S_X δ_{XY} V^{Y}_{ξ}``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``S_X`` > `tol`
"""
function svd_decompose(Amat, nvirt, nocc, tol = 1e-6)
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
    svd_decompose(Amat, tol = 1e-6)

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} S_X δ_{XY} V^{Y}_{ξ'}``.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` for ``S_X`` > `tol`
"""
function svd_decompose(Amat, tol = 1e-6)
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

  Iteratively decompose `A[ai,ξ]` as ``U^{iX}_a S_X δ_{XY} V^Y_ξ``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for first `naux` ``S_X``
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

  @tensoropt Fdiff[X,Y] := UaiX[a,i,X] * UaiX2[a,i,Y]
  diagFdiff = eigen(Symmetric(Fdiff))

  @tensoropt UaiX2[a,i,Y] = diagFdiff.vectors[X,Y] * UaiX[a,i,X]
  return diagFdiff.values, UaiX2
end


end #module