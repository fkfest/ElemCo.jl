"""
This module contains functions for tensor decomposition methods.
"""
module DecompTools
using LinearAlgebra
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.FciDumps
using ..ElemCo.TensorTools
using ..ElemCo.QMTensors
using ..ElemCo.ALPACADecomposition
using ..ElemCo.OrbLocalization

export calc_integrals_decomposition
export svd_decompose, svd_decompose_dense, svd_decompose_llama
export get_localization_rotations
export prepare_doubles_for_decomposition, backtransform_svd_vectors!

"""
    IntegralMatrix{T} <: AbstractALPACAMatrix{T}

Matrix-free representation of the two-electron integral matrix for ALPACA decomposition.

Uses the raw integral array from `integ2_ss(EC.fd)` directly (mmapped, physicist
notation order, upper triangular storage for the last two indices):
`int2[p, q, tri(r,s)]` = ``v_{pq}^{rs}`` with ``r ≤ s``.

The matrix has compound indices ``I = (r-1) n + p`` and ``J = (s-1) n + q``
(column-major order), with elements ``M_{IJ} = v_{pq}^{rs} = ⟨pq|rs⟩``.

The matrix is symmetric: ``M^T = M`` (complex symmetric for complex MOs).
"""
struct IntegralMatrix{T} <: AbstractALPACAMatrix{T}
  "Two-electron integrals int2[p,q,tri(r,s)] in physicist notation, upper triangular"
  int2::Array{T,3}
  n::Int
end

Base.size(mat::IntegralMatrix) = (mat.n^2, mat.n^2)

function ALPACADecomposition.column!(buffer::AbstractVector, mat::IntegralMatrix, j::Integer)
  n = mat.n
  int2 = mat.int2
  q = ((j - 1) % n) + 1
  s = ((j - 1) ÷ n) + 1
  # buffer[I] = v_{pq}^{rs} where I = (r-1)*n + p
  # For r = 1:s, tri(r,s) = r + s*(s-1)÷2 are contiguous → batch copy
  tri_start = 1 + s * (s - 1) ÷ 2
  @views copyto!(reshape(buffer[1:s*n], n, s), int2[:, q, tri_start:tri_start+s-1])
  # For r > s: v_{pq}^{rs} = v_{qp}^{sr} = int2[q, p, tri(s,r)], use strided view copy
  @inbounds for r in s+1:n
    tri = s + r * (r - 1) ÷ 2
    off = (r - 1) * n
    @views copyto!(buffer[off+1:off+n], int2[q, :, tri])
  end
  return buffer
end

"""
    calc_integrals_decomposition(EC::ECInfo)

  Decompose ``v_{pr}^{qs}`` as ``v_p^{qL} v_r^{sL}`` and store as `mmL`.

  Uses the ALPACA algorithm with a matrix-free interface that accesses
  elements of the ``n^2 \\times n^2`` integral matrix on demand
  directly from the mmapped integral array, avoiding materialization
  of the full dense matrix.

  Diagonal elements ``⟨pq|pq⟩`` are pre-computed and passed as
  `ALPACADecomposition.PrincipalTriples` to avoid scattered I/O via `elements!`.
"""
function calc_integrals_decomposition(EC::ECInfo)
  int2 = integ2_ss(EC.fd)
  n = n_orbs(EC)
  tol = EC.options.cholesky.thr

  # Pre-compute diagonal elements M[I,I] = ⟨pp|rr⟩ where I = (r-1)*n + p
  n2 = n^2
  diag_pairs = Vector{Tuple{Int,Int}}(undef, n2)
  diag_values = Vector{eltype(int2)}(undef, n2)
  @inbounds for r in 1:n
    tri_rr = r * (r + 1) ÷ 2
    off = (r - 1) * n
    for p in 1:n
      I = off + p
      diag_pairs[I] = (I, I)
      diag_values[I] = int2[p, p, tri_rr]
    end
  end
  principal = PrincipalTriples(diag_pairs, diag_values)

  mat = IntegralMatrix(int2, n)
  opts = ALPACAOptions(tol=tol, symmetry=:symmetric)
  result = alpaca(mat; principal=principal, options=opts)

  naux1 = size(result.left, 2)
  if !isempty(result.neg_indices)
    @warn "ALPACA found $(length(result.neg_indices)) negative eigenvalues in integral matrix"
  end
  println("Integral auxiliary space size: ",naux1)
  save!(EC, "mmL", reshape(result.left, (n,n,naux1)))
end

"""
    _check_llama_reconstruction(Amat, Q, tol, description) -> Float64

Check the actual reconstruction error ``\\|A - Q Q^H A\\|`` after LLAMA
decomposition and warn if it exceeds the requested tolerance.
Returns the reconstruction error.
"""
function _check_llama_reconstruction(Amat, Q, tol, description)
  # Estimate spectral norm of R = (I - Q*Q')*A via power iteration on R'R.
  # Cost: O(m*n) per iteration vs O(m*n*min(m,n)) for full SVD.
  n = size(Amat, 2)
  nk = size(Q, 2)
  nk == 0 && return real(eltype(Amat))(Inf)
  v = randn(eltype(Amat), n)
  v ./= norm(v)
  recon_err = zero(real(eltype(Amat)))
  prev_err = zero(real(eltype(Amat)))
  for _ in 1:50
    w = Amat * v            # m-vector
    w .-= Q * (Q' * w)      # project out Q (w now ⊥ Q)
    recon_err = norm(w)
    recon_err < eps(real(eltype(Amat))) && break
    abs(recon_err - prev_err) < 1.0e-3 * recon_err && break
    prev_err = recon_err
    v = Amat' * w            # n-vector (= R'*w since w ⊥ Q)
    v ./= norm(v)
  end
  if recon_err > tol
    ratio = recon_err / tol
    @warn "$(description) LLAMA reconstruction error ($recon_err) exceeds " *
          "tolerance ($tol) by factor $(round(ratio; digits=1)). " *
          "Consider using dense decomposition (use_dense_decomposition=true) " *
          "or providing an explicit pivotol to llama."
  end
  return recon_err
end

"""
    _compute_pivotol(Amat, tol, pivotol, pivotol_mode)

Compute the pivot tolerance for LLAMA based on the mode.
If `pivotol > 0`, use it directly (manual override).
Otherwise, use the mode:
- `:maxdim` → `tol / sqrt(max(size(A)...))`
- `:adaptive` → `NaN` (let LLAMA decide internally)
"""
function _compute_pivotol(Amat, tol, pivotol, pivotol_mode)
  if pivotol > 0
    return pivotol
  elseif pivotol_mode == :maxdim
    return tol / sqrt(max(size(Amat)...))
  elseif pivotol_mode == :adaptive
    return NaN
  else
    throw(ArgumentError("Unknown pivotol_mode: $pivotol_mode. Use :maxdim or :adaptive."))
  end
end

"""
    svd_decompose_llama(Amat, nvirt, nocc, tol=1e-6; pivotol=0.0, pivotol_mode=:maxdim, verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``
  using LLAMA low-rank approximation.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`.
  If `pivotol > 0`, it overrides LLAMA's pivot tolerance.
  Otherwise, `pivotol_mode` controls the computed pivot tolerance:
  `:maxdim` (default) uses `tol/sqrt(max(m,n))`, `:adaptive` delegates to LLAMA's internal logic.
"""
function svd_decompose_llama(Amat, nvirt, nocc, tol=1e-6; pivotol=0.0, pivotol_mode=:maxdim,
                             verbose=true, description="")
  effective_pivotol = _compute_pivotol(Amat, tol, pivotol, pivotol_mode)
  result = llama(Amat; tol=tol, pivotol=effective_pivotol)
  naux = length(result.singular_values)
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  Q = result.Q
  recon_err = _check_llama_reconstruction(Amat, Q, tol, description)
  return reshape(Q, (nvirt,nocc,naux))
end

"""
    svd_decompose_llama(Amat, tol=1e-6; pivotol=0.0, pivotol_mode=:maxdim, verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``
  using LLAMA low-rank approximation.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` and ``Σ_X`` for ``Σ_X`` > `tol`.
  If `pivotol > 0`, it overrides LLAMA's pivot tolerance.
  Otherwise, `pivotol_mode` controls the computed pivot tolerance:
  `:maxdim` (default) uses `tol/sqrt(max(m,n))`, `:adaptive` delegates to LLAMA's internal logic.
"""
function svd_decompose_llama(Amat, tol=1e-6; pivotol=0.0, pivotol_mode=:maxdim, verbose=true, description="")
  effective_pivotol = _compute_pivotol(Amat, tol, pivotol, pivotol_mode)
  result = llama(Amat; tol=tol, pivotol=effective_pivotol)
  naux = length(result.singular_values)
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  Q = result.Q
  recon_err = _check_llama_reconstruction(Amat, Q, tol, description)
  return Q, result.singular_values
end

"""
    svd_decompose_dense(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``
  using full dense SVD.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`
"""
function svd_decompose_dense(Amat, nvirt, nocc, tol=1e-6; verbose=true, description="")
  U, S, = svd(Amat; full=false)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return reshape(U[:,1:naux], (nvirt,nocc,naux))
end

"""
    svd_decompose_dense(Amat, tol=1e-6; verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  using full dense SVD.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` and ``Σ_X`` for ``Σ_X`` > `tol`
"""
function svd_decompose_dense(Amat, tol=1e-6; verbose=true, description="")
  U, S, = svd(Amat; full=false)
  naux = 0
  for s in S
    if s > tol
      naux += 1
    else
      break
    end
  end
  if verbose
    println(description, " SVD-basis size: ", naux)
  end
  return U[:,1:naux], S[1:naux]
end

"""
    svd_decompose(Amat, nvirt, nocc, tol=1e-6; method=:llama, pivotol=0.0, pivotol_mode=:maxdim, verbose=true, description="")

  SVD-decompose `A[ai,ξ]` as ``U^{iX}_a Σ_X δ_{XY} V^{Y}_{ξ}``.
  Return ``U^{iX}_a`` as `U[a,i,X]` for ``Σ_X`` > `tol`.

  `method` selects the algorithm: `:llama` (default) or `:dense`.
  `pivotol` if > 0 overrides LLAMA's pivot tolerance (ignored for `:dense`).
  `pivotol_mode` controls automatic pivot tolerance: `:maxdim` or `:adaptive`.
"""
function svd_decompose(Amat, nvirt, nocc, tol=1e-6; method=:llama, pivotol=0.0, pivotol_mode=:maxdim, 
                       verbose=true, description="")
  if method == :llama
    return svd_decompose_llama(Amat, nvirt, nocc, tol; pivotol, pivotol_mode, verbose, description)
  elseif method == :dense
    return svd_decompose_dense(Amat, nvirt, nocc, tol; verbose, description)
  else
    throw(ArgumentError("Unknown SVD method: $method. Use :llama or :dense."))
  end
end

"""
    svd_decompose(Amat, tol=1e-6; method=:llama, pivotol=0.0, pivotol_mode=:maxdim, verbose=true, description="")

  SVD-decompose `A[ξ,ξ']` as ``U^{X}_{ξ} Σ_X δ_{XY} V^{Y}_{ξ'}``.
  Return ``U^{X}_{ξ}`` as `U[ξ,X]` and ``Σ_X`` for ``Σ_X`` > `tol`.

  `method` selects the algorithm: `:llama` (default) or `:dense`.
  `pivotol` if > 0 overrides LLAMA's pivot tolerance (ignored for `:dense`).
  `pivotol_mode` controls automatic pivot tolerance: `:maxdim` or `:adaptive`.
"""
function svd_decompose(Amat, tol=1e-6; method=:llama, pivotol=0.0, pivotol_mode=:maxdim, 
                       verbose=true, description="")
  if method == :llama
    return svd_decompose_llama(Amat, tol; pivotol, pivotol_mode, verbose, description)
  elseif method == :dense
    return svd_decompose_dense(Amat, tol; verbose, description)
  else
    throw(ArgumentError("Unknown SVD method: $method. Use :llama or :dense."))
  end
end

"""
    prepare_doubles_for_decomposition(EC::ECInfo, T2::AbstractArray{T,4}; permuted=false, R_occ=nothing, R_virt=nothing)

Prepare doubles amplitudes for SVD decomposition.

Transforms ``T^{ij}_{ab}`` to a localized orbital basis (if localization is enabled
and a molecular system is available), permutes to ``T^i_a{}^j_b`` layout,
and reshapes to a flat ``[ai,bj]`` matrix.

If `permuted=false` (default), `T2` is in ``[a,b,i,j]`` (vvoo) layout and will be
permuted to ``[a,i,b,j]``.  If `permuted=true`, `T2` is already in ``[a,i,b,j]`` layout.

Pre-computed rotation matrices can be passed via `R_occ`/`R_virt` to avoid
recomputing them when multiple decompositions share the same localization.

# Returns
`(T2_flat, R_occ, R_virt)` where `T2_flat` is ready for [`svd_decompose`](@ref)
and `R_occ`/`R_virt` are the rotation matrices (or `nothing`).
Pass them to [`backtransform_svd_vectors!`](@ref) after decomposition.
"""
function prepare_doubles_for_decomposition(EC::ECInfo, T2::AbstractArray{T,4};
                                           permuted::Bool=false,
                                           R_occ=nothing, R_virt=nothing) where T
  if isnothing(R_occ) || isnothing(R_virt)
    R_occ, R_virt = get_localization_rotations(EC)
  end
  T2_flat = _prepare_doubles_impl(T2, R_occ, R_virt, permuted)
  return T2_flat, R_occ, R_virt
end

function _prepare_doubles_impl(T2::AbstractArray, ::Nothing, ::Nothing, permuted::Bool)
  if !permuted
    T2 = permutedims(T2, (1,3,2,4))
  end
  nv, no = size(T2, 1), size(T2, 2)
  return reshape(T2, (nv*no, nv*no))
end

function _prepare_doubles_impl(T2::AbstractArray, R_occ::AbstractMatrix, R_virt::AbstractMatrix, permuted::Bool)
  if permuted
    # T2[a,i,b,j]: dims 1,3 virtual; dims 2,4 occupied
    @mtensor T2a[a,i,b,j] := R_virt[a,a2] * T2[a2,i,b,j]
    @mtensor T2b[a,i,b,j] := R_occ[i,i2] * T2a[a,i2,b,j]
    @mtensor T2a[a,i,b,j] = R_virt[b,b2] * T2b[a,i,b2,j]
    @mtensor T2b[a,i,b,j] = R_occ[j,j2] * T2a[a,i,b,j2]
  else
    # T2[a,b,i,j]: dims 1,2 virtual; dims 3,4 occupied
    @mtensor T2a[a,b,i,j] := R_virt[a,a2] * T2[a2,b,i,j]
    @mtensor T2b[a,b,i,j] := R_virt[b,b2] * T2a[a,b2,i,j]
    @mtensor T2a[a,b,i,j] = R_occ[i,i2] * T2b[a,b,i2,j]
    @mtensor T2b[a,i,b,j] := R_occ[j,j2] * T2a[a,b,i,j2]
  end
  nv, no = size(T2b, 1), size(T2b, 2)
  T2_flat = reshape(T2b, (nv*no, nv*no))
  return T2_flat
end

"""
    backtransform_svd_vectors!(UaiX::AbstractArray{T,3}, R_occ, R_virt)

Transform SVD vectors ``U^{iX}_a`` from a localized back to the canonical orbital basis.

Applies ``U_{\\mathrm{can}}[a,i,X] = R_v^T[a,a'] R_o^T[i,i'] U_{\\mathrm{loc}}[a',i',X]``.
If `R_occ` or `R_virt` are `nothing`, `UaiX` is returned unchanged.
"""
function backtransform_svd_vectors!(UaiX::AbstractArray{T,3}, ::Nothing, ::Nothing) where T
  return UaiX
end

function backtransform_svd_vectors!(UaiX::AbstractArray{T,3}, R_occ::AbstractMatrix, R_virt::AbstractMatrix) where T
  U_tmp = similar(UaiX)
  @mtensor U_tmp[a,i,X] = R_virt[a2,a] * UaiX[a2,i,X]
  @mtensor UaiX[a,i,X] = R_occ[i2,i] * U_tmp[a,i2,X]
  return UaiX
end

"""
    get_localization_rotations(EC::ECInfo)

Compute localization rotation matrices if `EC.options.cc.localize` is `true`
and a molecular system is available.

Returns `(R_occ, R_virt)` where `R_occ` (nocc × nocc) and `R_virt` (nvirt × nvirt)
are orthogonal rotation matrices from canonical to localized orbitals,
or `(nothing, nothing)` if localization is not used.

If localization is requested but no molecular system is set up (FCIDUMP-only),
prints a warning and returns `(nothing, nothing)`.
"""
function get_localization_rotations(EC::ECInfo)
  if !EC.options.cc.localize
    return nothing, nothing
  end
  if isempty(EC.system)
    println("WARNING: Localized SVD requested but no molecular system available (FCIDUMP-only). ",
            "Using non-localized SVD.")
    return nothing, nothing
  end
  return compute_localization_rotations(EC)
end

"""
    svd_decompose(EC::ECInfo, Amat, nvirt, nocc, tol; description="", kwargs...)

SVD-decompose `A[ai,ξ]` reading decomposition options from `EC.options.cc`.

Reads `method` (dense vs llama), `pivotol`, and `pivotol_mode` from EC options
and forwards to the plain [`svd_decompose`](@ref).

For doubles-type quantities, use [`prepare_doubles_for_decomposition`](@ref) before
calling this function and [`backtransform_svd_vectors!`](@ref) afterwards.

# Arguments
- `EC::ECInfo`: Electronic structure information object
- `Amat`: Matrix to decompose, shape `(nvirt*nocc, ξ)`
- `nvirt`: Number of virtual orbitals
- `nocc`: Number of occupied orbitals
- `tol`: SVD threshold

# Keyword Arguments
- `description=""`: Description for output
- Additional `kwargs` are forwarded to the underlying `svd_decompose`

# Returns
- `Array{T,3}`: `U[a,i,X]` decomposition vectors
"""
function svd_decompose(EC::ECInfo, Amat, nvirt, nocc, tol; description="", kwargs...)
  method = EC.options.cc.use_dense_decomposition ? :dense : :llama
  pivotol_mode = EC.options.cc.ampsvd_pivotol_mode
  pivotol = EC.options.cc.ampsvd_pivotol
  return svd_decompose(Amat, nvirt, nocc, tol;
                       method, pivotol_mode, pivotol, description, kwargs...)
end

end #module
