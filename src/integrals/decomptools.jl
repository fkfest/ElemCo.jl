"""
This module contains functions for tensor decomposition methods.
"""
module DecompTools
using LinearAlgebra
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.TensorTools
using ..ElemCo.OrbTools
using ..ElemCo.DFTools
using ..ElemCo.QMTensors

export calc_integrals_decomposition, calc_df_integrals
export eigen_decompose, svd_decompose
export rotate_U2pseudocanonical

"""
    symmetric_pivoted_cholesky(M, tol)

  Pivoted Cholesky factorization for a complex symmetric positive semi-definite
  matrix: ``M = P^T L L^T P``, where ``L`` is lower triangular.
  This differs from the standard Hermitian Cholesky (``M = L L^†``) by using
  transpose instead of adjoint, which is required for complex symmetric matrices
  arising from two-electron integrals with complex MO coefficients.

  Returns `(pqP, rank)` where `pqP[i,L]` are the unpivoted Cholesky vectors
  such that ``M ≈ pqP \\cdot pqP^T``.
"""
function symmetric_pivoted_cholesky(M_in::AbstractMatrix{T}, tol) where T<:Complex
  #TODO implement a more efficient version or check if using SVD is faster
  n = size(M_in, 1)
  M = copy(M_in)
  L = zeros(T, n, n)
  perm = collect(1:n)
  rank = 0
  for k in 1:n
    max_diag = 0.0
    max_idx = k
    for i in k:n
      d = abs(M[i,i])
      if d > max_diag
        max_diag = d
        max_idx = i
      end
    end
    max_diag < tol && break
    if max_idx != k
      for j in 1:n
        M[j,k], M[j,max_idx] = M[j,max_idx], M[j,k]
      end
      for j in 1:n
        M[k,j], M[max_idx,j] = M[max_idx,j], M[k,j]
      end
      for j in 1:k-1
        L[k,j], L[max_idx,j] = L[max_idx,j], L[k,j]
      end
      perm[k], perm[max_idx] = perm[max_idx], perm[k]
    end
    L[k,k] = sqrt(M[k,k])
    for i in (k+1):n
      L[i,k] = M[i,k] / L[k,k]
    end
    # Update remaining submatrix: M -= L[:,k] * L[:,k]^T (symmetric, not Hermitian)
    for j in (k+1):n
      for i in j:n
        M[i,j] -= L[i,k] * L[j,k]
        if i != j
          M[j,i] = M[i,j]
        end
      end
    end
    rank += 1
  end
  pqP = L[invperm(perm), 1:rank]
  return pqP, rank
end

"""
    calc_integrals_decomposition(EC::ECInfo)

  Decompose ``v_{pr}^{qs}`` as ``v_p^{qL} v_r^{sL}`` and store as `mmL`.
"""
function calc_integrals_decomposition(EC::ECInfo)
  pqrs = permutedims(ints2(EC,"::::",:α),(1,3,2,4))
  n = size(pqrs,1)
  if EC.options.cc.usecholesky
    Mmat = reshape(pqrs, (n^2,n^2))
    pqrs = nothing
    if ec_eltype(EC) <: Complex
      # Complex symmetric PSD matrix: need M = L*L^T (not L*L†)
      pqP, naux1 = symmetric_pivoted_cholesky(Mmat, EC.options.cholesky.thr)
    else
      CA = cholesky(Hermitian(Mmat), RowMaximum(), check = false, tol = EC.options.cholesky.thr)
      naux1 = CA.rank
      pqP = CA.U[1:naux1,invperm(CA.p)]'
    end
  else
    F = svd(reshape(pqrs, (n^2,n^2)))
    S = F.S
    pqrs = nothing

    naux1 = 0
    for s in S
      if s > EC.options.cholesky.thr
        naux1 += 1
      else
        break
      end
    end

    if ec_eltype(EC) <: Complex
      # Takagi factorization for complex symmetric M = U Σ U^T
      # From SVD M = A Σ B†, phases: e^{iφ_k} = conj(Aₖ^T Bₖ)
      A = F.U[:, 1:naux1]
      B = F.V[:, 1:naux1]
      phases = [conj(sum(A[:,k] .* B[:,k])) for k in 1:naux1]
      pqP = A .* transpose(sqrt.(phases) .* sqrt.(S[1:naux1]))
    else
      pqP = F.U[:,1:naux1].*sqrt.(S[1:naux1]')
    end
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
  space_save, _ = restore_system_space!(EC)
  cMO = load_orbitals(EC)
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
  Tval, U = eigen(Hermitian(-T2mat))
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
    svd_decompose(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")
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
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose(Amat, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose(Amat, tol=1e-6; verbose=true, description="")
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
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return U[:,1:naux], S[1:naux]
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

  @mtensor Fdiff[X,Y] := conj(UaiX[a,i,X]) * UaiX2[a,i,Y]
  diagFdiff = eigen(Hermitian(Fdiff))

  @mtensor UaiX2[a,i,Y] = diagFdiff.vectors[X,Y] * UaiX[a,i,X]
  return eltype(Fdiff).(diagFdiff.values), UaiX2
end


end #module
