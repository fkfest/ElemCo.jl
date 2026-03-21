"""
    VaspInterface

Interface module for reading VASP/CC4S tensor files into ElemCo.

Reads YAML descriptor files (version 100) and their companion `.elements` files
(text or IEEE binary) as produced by the VASP → CC4S pipeline.

Key tensors:
- `EigenEnergies`: orbital eigenvalues
- `CoulombVertex`: 3-index density-fitted integrals Γ[F,p,q]
- `State`: dimension properties (occupied/virtual partitioning)

"""
module VaspInterface

using YAML
using LinearAlgebra
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.FciDumps
using ..ElemCo.TensorTools
using ..ElemCo.DecompTools: symmetric_pivoted_cholesky

export load_vasp, setup_vasp!

const VASP_TENSOR_VERSION = 100

"""
    VaspTensorMeta

Parsed metadata from a VASP tensor YAML file.
"""
struct VaspTensorMeta
  version::Int
  scalar_type::Symbol        # :Real64 or :Complex64
  dims::Vector{@NamedTuple{length::Int, type::String}}
  elements_type::Symbol      # :TextFile or :IeeeBinaryFile
  unit::Float64
  half_grid::Bool
  metadata::Dict{String,Any}
end

"""
    VaspData{T}

Container for all tensors loaded from a VASP/CC4S directory.
`T` is the element type of the Coulomb vertex (`Float64` or `ComplexF64`).
"""
struct VaspData{T<:Union{Float64, ComplexF64}}
  "Orbital eigenvalues in Hartree, length norb"
  eigen_energies::Vector{Float64}
  "Fermi energy in Hartree"
  fermi_energy::Float64
  "3-index Coulomb vertex Γ[F,p,q] in a.u., shape (naux, norb, norb)"
  coulomb_vertex::Array{T,3}
  "Number of occupied orbitals"
  n_occupied::Int
  "Number of virtual orbitals"
  n_virtual::Int
  "Coulomb potential v[G] (optional)"
  coulomb_potential::Vector{Float64}
  "Grid vectors G[3,nG] (optional)"
  grid_vectors::Matrix{Float64}
  "Coulomb vertex singular vectors U[nG,naux] (optional)"
  coulomb_vertex_singular_vectors::Matrix{T}
  "Delta integrals HH Δ[i,j] (optional)"
  delta_integrals_hh::Matrix{T}
  "Delta integrals PPHH Δ[a,b,i,j] (optional)"
  delta_integrals_pphh::Array{T,4}
  "MP2 pair energies ε[i,j] (optional)"
  mp2_pair_energies::Matrix{Float64}
end

"""
    read_vasp_yaml(filepath::String) → VaspTensorMeta

Parse a VASP tensor YAML file and return structured metadata.
"""
function read_vasp_yaml(filepath::String)
  y = YAML.load_file(filepath)
  version = y["version"]::Int
  if version != VASP_TENSOR_VERSION
    error("Unsupported VASP tensor version $version (expected $VASP_TENSOR_VERSION) in $filepath")
  end

  # Tensor files have "type" == "Tensor"; dimension files have "dimensionType"
  if !haskey(y, "type") || y["type"] != "Tensor"
    error("File $filepath is not a tensor YAML (type != Tensor)")
  end

  scalar_type = Symbol(y["scalarType"]::String)
  if scalar_type ∉ (:Real64, :Complex64)
    error("Unsupported scalarType $scalar_type in $filepath")
  end

  dims_raw = y["dimensions"]
  dims = [@NamedTuple{length::Int, type::String}((
    d["length"]::Int,
    get(d, "type", "")::String
  )) for d in dims_raw]

  elements = y["elements"]
  elements_type = Symbol(elements["type"]::String)
  if elements_type ∉ (:TextFile, :IeeeBinaryFile)
    error("Unsupported elements type $elements_type in $filepath")
  end

  unit = Float64(y["unit"])

  md = get(y, "metaData", Dict{String,Any}())
  half_grid = get(md, "halfGrid", 0) == 1

  return VaspTensorMeta(version, scalar_type, dims, elements_type, unit, half_grid, md)
end

"""
    read_vasp_elements(elements_path::String, meta::VaspTensorMeta) → Array

Read the `.elements` companion file according to the tensor metadata.
Returns a Julia array with proper shape and type, in column-major order.

For `halfGrid=1`: Complex64 data is reinterpreted as Real64 with doubled
AuxiliaryField dimension (following CC4S convention).
"""
function read_vasp_elements(elements_path::String, meta::VaspTensorMeta)
  # Compute the effective dimensions and julia element type
  effective_scalar = meta.scalar_type
  if meta.half_grid
    effective_scalar = :Real64
  end

  lens = [d.length for d in meta.dims]
  if meta.half_grid
    # Double the AuxiliaryField dimension
    aux_idx = findfirst(d -> d.type == "AuxiliaryField", meta.dims)
    if isnothing(aux_idx)
      error("halfGrid=1 but no AuxiliaryField dimension found")
    end
    lens[aux_idx] *= 2
  end

  total_elements = prod(lens)

  if meta.elements_type == :TextFile
    data = read_text_elements(elements_path, total_elements, effective_scalar)
  else
    data = read_binary_elements(elements_path, total_elements, effective_scalar, meta.scalar_type, meta.half_grid)
  end

  if length(lens) == 1
    return data
  else
    # VASP stores tensors in column-major order (same as Julia/Fortran)
    return reshape(data, Tuple(lens))
  end
end

"""
    read_text_elements(path, n, scalar_type) → Vector

Read `n` elements from a text file, one number per line.
"""
function read_text_elements(path::String, n::Int, scalar_type::Symbol)
  T = scalar_type == :Complex64 ? ComplexF64 : Float64
  data = Vector{T}(undef, n)
  open(path) do io
    for i in 1:n
      line = readline(io)
      if T == Float64
        data[i] = parse(Float64, strip(line))
      else
        # Complex text: two floats per line (real imag) or  (real,imag)
        parts = split(strip(line))
        if length(parts) == 2
          data[i] = complex(parse(Float64, parts[1]), parse(Float64, parts[2]))
        else
          data[i] = complex(parse(Float64, parts[1]), 0.0)
        end
      end
    end
  end
  return data
end

"""
    read_binary_elements(path, n, effective_type, file_type, half_grid) → Vector

Read elements from an IEEE binary file.
For `half_grid=1`: reads as Complex64 then reinterprets as Float64.
"""
function read_binary_elements(path::String, n::Int, effective_type::Symbol,
                              file_type::Symbol, half_grid::Bool)
  if half_grid
    # File stores Complex64, but we want Real64 with doubled aux dim
    n_complex = n ÷ 2
    raw = Vector{ComplexF64}(undef, n_complex)
    open(path) do io
      read!(io, raw)
    end
    return reinterpret(Float64, raw)
  else
    T = file_type == :Complex64 ? ComplexF64 : Float64
    data = Vector{T}(undef, n)
    open(path) do io
      read!(io, data)
    end
    return data
  end
end

"""
    load_vasp(dirpath::String) → VaspData

Load all VASP/CC4S tensors from a directory containing YAML descriptor files
and their companion `.elements` files.

Required files:
- `EigenEnergies.yaml` + `.elements`
- `CoulombVertex.yaml` + `.elements`

For occupied/virtual partitioning, `State.yaml` is read if present.
Otherwise, the user must set occupation explicitly.
"""
function load_vasp(dirpath::String)
  # --- EigenEnergies (required) ---
  ee_yaml = joinpath(dirpath, "EigenEnergies.yaml")
  isfile(ee_yaml) || error("EigenEnergies.yaml not found in $dirpath")
  ee_meta = read_vasp_yaml(ee_yaml)
  ee_raw = read_vasp_elements(joinpath(dirpath, "EigenEnergies.elements"), ee_meta)
  # Convert to Hartree using unit factor
  eigen_energies = real.(ee_raw) .* ee_meta.unit

  fermi_energy = 0.0
  if haskey(ee_meta.metadata, "fermiEnergy")
    fermi_energy = Float64(ee_meta.metadata["fermiEnergy"]) * ee_meta.unit
  end

  norb = length(eigen_energies)

  # --- Determine occupied / virtual from State.yaml or fermiEnergy ---
  n_occupied, n_virtual = determine_occupation(dirpath, norb, eigen_energies, fermi_energy, ee_meta)

  # --- CoulombVertex (required) ---
  cv_yaml = joinpath(dirpath, "CoulombVertex.yaml")
  isfile(cv_yaml) || error("CoulombVertex.yaml not found in $dirpath")
  cv_meta = read_vasp_yaml(cv_yaml)
  cv_raw = read_vasp_elements(joinpath(dirpath, "CoulombVertex.elements"), cv_meta)

  # Convert to a.u., preserving element type (Float64 or ComplexF64)
  coulomb_vertex = cv_raw .* cv_meta.unit

  # --- Optional tensors ---
  T = eltype(coulomb_vertex)
  coulomb_potential = load_optional_1d(dirpath, "CoulombPotential")
  grid_vectors = load_optional_2d(dirpath, "GridVectors")
  cv_sv = T.(load_optional_2d(dirpath, "CoulombVertexSingularVectors"))
  delta_hh = T.(load_optional_2d(dirpath, "DeltaIntegralsHH"))
  delta_pphh = T.(load_optional_4d(dirpath, "DeltaIntegralsPPHH"))
  mp2_pe = Float64.(load_optional_2d(dirpath, "Mp2PairEnergies"))

  println("VASP data loaded from $dirpath")
  println("  Orbitals: $norb ($n_occupied occupied, $n_virtual virtual)")
  println("  Coulomb vertex: ", size(coulomb_vertex))
  println("  Fermi energy: ", fermi_energy, " Eh")

  return VaspData(
    eigen_energies, fermi_energy, coulomb_vertex,
    n_occupied, n_virtual,
    coulomb_potential, grid_vectors, cv_sv,
    delta_hh, delta_pphh, mp2_pe
  )
end

"""
    determine_occupation(dirpath, norb, eigen_energies, fermi_energy, ee_meta) → (nocc, nvirt)

Determine occupied/virtual split. Uses metadata `energies` field if available
to count states below the Fermi energy. Falls back to counting eigenvalues < fermi_energy.
"""
function determine_occupation(dirpath::String, norb::Int,
                              eigen_energies::Vector{Float64},
                              fermi_energy::Float64,
                              ee_meta::VaspTensorMeta)
  # First check if State.yaml explicitly provides occupation info
  state_yaml = joinpath(dirpath, "State.yaml")
  if isfile(state_yaml)
    y = YAML.load_file(state_yaml)
    # State.yaml doesn't directly encode nocc, but the properties sometimes help
    # We'll still use fermi energy for the split
  end

  # Use the energies from metadata (in eV, pre-conversion) or the converted eigenvalues
  if haskey(ee_meta.metadata, "energies")
    raw_energies_ev = Float64.(ee_meta.metadata["energies"])
    fermi_ev = haskey(ee_meta.metadata, "fermiEnergy") ?
      Float64(ee_meta.metadata["fermiEnergy"]) : 0.0
    n_occupied = count(e -> e < fermi_ev, raw_energies_ev)
  else
    # Count eigenvalues below fermi energy (already in Hartree)
    n_occupied = count(e -> e < fermi_energy, eigen_energies)
  end

  if n_occupied == 0 || n_occupied == norb
    error("Could not determine occupation: nocc=$n_occupied, norb=$norb. " *
          "Set occupation manually via EC.options.wf")
  end

  return n_occupied, norb - n_occupied
end

"""
    decompose_vasp_vertex(Gamma::Array{T,3}, tol::Float64; sigma::Float64=0.01, usesvd::Bool=false) where T → (B, naux_new)

Two-step pivoted Cholesky decomposition of the VASP Coulomb vertex,
following the algorithm of Folkestad et al. [JCP 150, 194112 (2019)].

Given ``\\Gamma^L_{pq}`` (shape `(naux, norb, norb)`), the 4-index integrals in 
chemist notation are:
``(pq|rs) = \\sum_L \\overline{\\Gamma^L_{qp}} \\Gamma^L_{rs}``

This function produces ``B^J_{pq}`` (shape `(norb, norb, naux_new)`) such that:
``(pq|rs) = \\sum_J B^J_{pq} B^J_{rs}``

**Step I** determines the pivot set ``\\mathcal{B}`` using span-factor batching
with diagonal screening, without building full-length Cholesky vectors.

**Step II** constructs the final Cholesky vectors via the inner-projection
(RI) formula using BLAS-3 matrix operations:
``L^J_{pq} = \\sum_{K \\in \\mathcal{B}} V_{(pq),K} (K^{-T})_{KJ}``
where ``V_{(pq),K} = \\sum_L \\overline{\\Gamma^L_{qp}} \\Gamma^L_{r_K s_K}``
and ``J = K K^T`` is the Coulomb matrix among pivots.

# Arguments
- `Gamma`: VASP Coulomb vertex, shape `(naux, norb, norb)`
- `tol`: Cholesky decomposition threshold (absolute value of diagonal residual)
- `sigma`: span factor for batch qualification (default: 0.01)
- `usesvd`: if true, use SVD (real) / Takagi (complex) instead of Cholesky for J in step II (default: false)

# Returns
- `B`: decomposed 3-index integrals, shape `(norb, norb, naux_new)`
- `naux_new`: number of Cholesky vectors

# References
- S.D. Folkestad, E.F. Kjønstad, H. Koch, JCP 150, 194112 (2019)
"""
function decompose_vasp_vertex(Gamma::Array{T,3}, tol::Float64; sigma::Float64=0.01, usesvd::Bool=false) where T
  naux, norb, _ = size(Gamma)
  n2 = norb * norb

  # Precompute reshaped views for BLAS:
  # Gtilde[L, I] = conj(Gamma[L, q, p]) where I = (q-1)*norb + p  (compound index for (p,q))
  # G[L, I] = Gamma[L, p, q]             where I = (q-1)*norb + p  (compound index for (p,q))
  # Note: G[L, I] with I=(q-1)*norb+p gives Gamma[L, p, q], which for the pivot column formula
  # V[(p,q), (r,s)] = Σ_L Gtilde[L,(p,q)] * G[L,(r,s)] = Σ_L conj(Γ[L,q,p]) * Γ[L,r,s]
  # is exactly (pq|rs) in chemist notation.
  Gtilde = zeros(T, naux, n2)
  for q in 1:norb, p in 1:norb
    I = (q - 1) * norb + p
    @inbounds for L in 1:naux
      Gtilde[L, I] = conj(Gamma[L, q, p])
    end
  end
  G = reshape(Gamma, naux, n2)

  # ═══════════════════════════════════════════════════════════════
  # STEP I: Determine pivot set B
  # ═══════════════════════════════════════════════════════════════
  # Compute initial diagonals: d[I] = V[I,I] = Σ_L Gtilde[L,I] * G[L,I]
  d = zeros(real(T), n2)
  for I in 1:n2
    s = zero(T)
    @inbounds for L in 1:naux
      s += Gtilde[L, I] * G[L, I]
    end
    d[I] = real(s)  # diagonal of a positive semi-definite matrix is real and non-negative
  end

  # D: set of significant diagonal indices (those above threshold)
  D_indices = findall(di -> di >= tol, d)
  nD = length(D_indices)
  D_map = zeros(Int, n2)
  @inbounds for (k, I) in enumerate(D_indices)
    D_map[I] = k
  end

  # Pre-allocate Cholesky vector storage with amortized doubling (avoids hcat)
  est_cap = min(n2, max(64, isqrt(nD)))
  L_storage = Matrix{T}(undef, nD, est_cap)
  n_stored = 0

  pivots = Int[]
  sizehint!(pivots, est_cap)
  is_pivot = falses(n2)  # BitVector for O(1) pivot lookup

  # Pre-allocate work buffers
  V_col = Vector{T}(undef, nD)
  coeffs = Vector{T}(undef, est_cap)
  Q_batch = Vector{Int}(undef, n2)
  new_D_buf = Vector{Int}(undef, n2)

  max_rank = min(n2, naux)

  while nD > 0
    # Find D_max = max d[I] for I ∈ D
    D_max = 0.0
    @inbounds for I in D_indices
      d[I] > D_max && (D_max = d[I])
    end
    D_max < tol && break

    # Qualify batch: Q = {I ∈ D : d[I] ≥ σ * D_max}
    threshold = sigma * D_max
    nQ = 0
    @inbounds for I in D_indices
      if d[I] >= threshold
        nQ += 1
        Q_batch[nQ] = I
      end
    end
    Q_view = @view Q_batch[1:nQ]
    sort!(Q_view, by=I -> d[I], rev=true)

    # Compute columns V[D, Q] using BLAS-3
    # Copy non-contiguous columns into contiguous buffers for efficient BLAS-3
    # (fancy-indexed SubArrays bypass optimized BLAS routines)
    Gt_D = Gtilde[:, D_indices]       # contiguous copy (naux × nD)
    G_Q = G[:, @view Q_batch[1:nQ]]   # contiguous copy (naux × nQ)
    V_DQ = transpose(Gt_D) * G_Q      # BLAS-3 gemm (nD × nQ)

    # Subtract previous Cholesky contributions using BLAS-3 (Eq. 9)
    if n_stored > 0
      # Extract Q rows from L_storage into contiguous buffer for efficient BLAS-3
      L_Q_buf = Matrix{T}(undef, nQ, n_stored)
      @inbounds for j in 1:nQ
        q_local = D_map[Q_batch[j]]
        for m in 1:n_stored
          L_Q_buf[j, m] = L_storage[q_local, m]
        end
      end
      # V_DQ -= L_storage[:, 1:n_stored] * L_Q_buf'
      mul!(V_DQ, @view(L_storage[:, 1:n_stored]), transpose(L_Q_buf), -one(T), one(T))
    end

    # Inner pivoted Cholesky within the batch Q
    n_stored_before = n_stored
    n_batch = 0
    for _ in 1:nQ
      # Find max diagonal in Q
      best_j = 0
      best_val = 0.0
      @inbounds for j in 1:nQ
        if d[Q_batch[j]] > best_val
          best_val = d[Q_batch[j]]
          best_j = j
        end
      end
      best_val < tol && break

      q = Q_batch[best_j]  # full index of the new pivot
      q_local = D_map[q]
      n_batch += 1

      # Copy column into V_col buffer (no allocation)
      @inbounds for k in 1:nD
        V_col[k] = V_DQ[k, best_j]
      end

      # Subtract current-batch contributions using BLAS-2 gemv
      n_current = n_stored - n_stored_before
      if n_current > 0
        @inbounds for j in 1:n_current
          coeffs[j] = L_storage[q_local, n_stored_before + j]
        end
        mul!(V_col, @view(L_storage[:, n_stored_before+1:n_stored]),
             @view(coeffs[1:n_current]), -one(T), one(T))
      end

      # Normalize by sqrt(d[q])
      diag_sqrt = sqrt(d[q])
      @inbounds for k in 1:nD
        V_col[k] /= diag_sqrt
      end

      # Update diagonals: d[I] -= (L^q_I)² for I ∈ D
      @inbounds for (k, I) in enumerate(D_indices)
        d[I] -= real(V_col[k]^2)
        d[I] < 0 && (d[I] = 0.0)
      end
      d[q] = 0.0

      # Ensure L_storage capacity (amortized doubling)
      if n_stored + 1 > size(L_storage, 2)
        new_cap = 2 * size(L_storage, 2)
        L_new = Matrix{T}(undef, nD, new_cap)
        @views L_new[:, 1:n_stored] .= L_storage[:, 1:n_stored]
        L_storage = L_new
        coeffs = Vector{T}(undef, new_cap)
      end

      # Store Cholesky vector directly into L_storage (no hcat)
      @inbounds for k in 1:nD
        L_storage[k, n_stored + 1] = V_col[k]
      end
      n_stored += 1

      push!(pivots, q)
      is_pivot[q] = true

      n_stored >= max_rank && break
    end

    # No progress in this batch — stop
    n_batch == 0 && break

    # Screen D: remove indices with d[I] < tol
    nD_new = 0
    @inbounds for I in D_indices
      if !is_pivot[I] && d[I] >= tol
        nD_new += 1
        new_D_buf[nD_new] = I
      end
    end

    if nD_new < nD
      old_D_indices = D_indices
      D_indices = @view(new_D_buf[1:nD_new]) |> collect
      nD = nD_new
      fill!(D_map, 0)
      @inbounds for (k, I) in enumerate(D_indices)
        D_map[I] = k
      end
      # Compress L_storage rows to match new D_indices
      if nD > 0
        L_compressed = Matrix{T}(undef, nD, size(L_storage, 2))
        @inbounds for j in 1:n_stored
          for (old_k, I) in enumerate(old_D_indices)
            new_k = D_map[I]
            if new_k > 0
              L_compressed[new_k, j] = L_storage[old_k, j]
            end
          end
        end
        L_storage = L_compressed
      end
      V_col = Vector{T}(undef, nD)
    else
      D_indices = @view(new_D_buf[1:nD_new]) |> collect
      nD = nD_new
    end

    n_stored >= max_rank && break
  end

  nB = length(pivots)
  println("  Step I: $nB pivots determined (from $naux original auxiliary functions)")

  # ═══════════════════════════════════════════════════════════════
  # STEP II: Construct Cholesky vectors via inner projection (RI)
  # ═══════════════════════════════════════════════════════════════
  # Form Coulomb matrix among pivots: J[p,p'] = V[pivots[p], pivots[p']]
  # J = Gtilde[:, pivots]' * G[:, pivots]  — (nB × nB) matrix
  # Copy pivot columns into contiguous buffer for efficient BLAS-3
  G_B = G[:, pivots]  # contiguous copy (naux × nB)
  J = transpose(Gtilde[:, pivots]) * G_B  # BLAS-3; shape (nB, nB)

  # Cholesky decompose J = K * K^T (transpose, NOT conjugate transpose)
  # or alternatively use SVD/Takagi decomposition for better numerical stability.
  # For complex symmetric case, use our symmetric_pivoted_cholesky or Takagi.
  # For real symmetric case, use Julia's built-in pivoted Cholesky or SVD.
  if usesvd
    # SVD-based decomposition: J = U Σ U^T (Takagi for complex, eigendecomposition for real)
    F = svd(J)
    nB_final = count(s -> s > tol, F.S)
    if T <: Complex
      # Takagi factorization for complex symmetric: J = U_T Σ U_T^T
      # From SVD J = A Σ B†, Takagi phases: e^{iφ_k} = conj(A_k^T B_k)
      # U_T = A * diag(√phases)
      # J^{-1} = conj(U_T) Σ^{-1} conj(U_T)^T, so C = conj(U_T) Σ^{-1/2}
      A = F.U[:, 1:nB_final]
      B = F.Vt[1:nB_final, :]'  # = F.V[:, 1:nB_final]
      phases = [conj(sum(A[:,k] .* B[:,k])) for k in 1:nB_final]
      inv_sqrt_S = 1.0 ./ sqrt.(F.S[1:nB_final])
      K_mat = conj(A) .* transpose(sqrt.(conj.(phases)) .* inv_sqrt_S)
    else
      # Real symmetric: J = U S U^T (symmetric SVD = eigendecomposition)
      inv_sqrt_S = 1.0 ./ sqrt.(F.S[1:nB_final])
      K_mat = F.U[:, 1:nB_final] .* inv_sqrt_S'
    end
    # K_mat: (nB, nB_final), such that L = V_all_B * K_mat gives Cholesky vectors
    # with L * L^T ≈ V_all_B * J^{-1} * V_all_B^T
  else
    if T <: Complex
      K_mat, rank_J = symmetric_pivoted_cholesky(J, tol * 1e-2)
      # K_mat: shape (nB, rank_J), J ≈ K_mat * transpose(K_mat)
      nB_final = rank_J
    else
      J_herm = Hermitian(J)
      CA = cholesky(J_herm, RowMaximum(), check=false, tol=tol * 1e-2)
      rank_J = CA.rank
      # Unpivot: CA gives J[CA.p, CA.p] = CA.L * CA.U where CA.U = CA.L'
      K_mat = CA.U[1:rank_J, invperm(CA.p)]'  # (nB, rank_J)
      nB_final = rank_J
    end
  end

  # V_all_B[I, k] = V[(p,q), pivots[k]] = Σ_L Gtilde[L, I] * G[L, pivots[k]]
  # This is the key BLAS-3 operation: (n2, naux) × (naux, nB) → (n2, nB)
  # G_B already holds contiguous pivot columns from J computation above
  V_all_B = transpose(Gtilde) * G_B  # shape (n2, nB)

  # Final Cholesky vectors via RI formula.
  # SVD path: L = V_all_B * K_mat (K_mat = U * Σ^{-1/2})
  # Cholesky path: L = V_all_B * K^{-T}, i.e. K * L^T = V_B^T
  if usesvd
    L_final = V_all_B * K_mat  # (n2, nB) × (nB, nB_final) → (n2, nB_final)
  else
    # K_mat: (nB, nB_final), transpose(V_all_B): (nB, n2)
    # Overdetermined solve → L_final_T: (nB_final, n2)
    L_final_T = K_mat \ transpose(V_all_B)
    L_final = transpose(L_final_T)  # (n2, nB_final)
  end

  # Free intermediate arrays
  Gtilde = nothing
  V_all_B = nothing
  L_storage = nothing

  println("  Step II: $nB_final Cholesky vectors constructed")

  # Reshape to B[norb, norb, nB_final]
  B_out = reshape(L_final, norb, norb, nB_final)

  return B_out, nB_final
end

"""
    setup_vasp!(EC::ECInfo, data::VaspData; ms2::Int=0)

Populate `EC` from loaded VASP data for use with `ccdriver`.

Sets up:
- `EC.fd` header and core Hamiltonian h₀ reconstructed from eigenvalues and Coulomb vertex
- 3-index Coulomb vertex saved as `mmL` to EC scratch directory
- Orbital spaces via `setup_space_fd!`

The core Hamiltonian is computed as:
``h_{pq} = F_{pq} - \\sum_i [2 J_i(p,q) - K_i(p,q)]``
where ``F_{pq} = \\varepsilon_p \\delta_{pq}`` (canonical MOs) and ``J``, ``K``
are the Coulomb and exchange contributions from occupied orbitals.
"""
function setup_vasp!(EC::ECInfo, data::VaspData; ms2::Int=0)
  norb = data.n_occupied + data.n_virtual
  nelec = 2 * data.n_occupied  # restricted, doubly occupied
  nocc = data.n_occupied

  # Create FDump with header, element type matching the Coulomb vertex
  T = eltype(data.coulomb_vertex)
  EC.fd = FDump{T,3}(norb, nelec; ms2=ms2)

  # Mark that 3-index DF integrals will be available in scratch (mmL)
  EC.fd.df3idx = true

  # Nuclear repulsion / reference energy is zero for VASP periodic calculations
  EC.fd.int0 = 0.0

  naux = size(data.coulomb_vertex, 1)
  println("  Setting up DF integrals: naux=$naux, norb=$norb")

  # Reconstruct core Hamiltonian using ORIGINAL Gamma before Cholesky decomposition.
  # VASP convention: (pq|rs) = Σ_L conj(Γ[L,q,p]) * Γ[L,r,s]
  # J[p,q] = Σ_i (pq|ii) = Σ_L conj(Γ[L,q,p]) * Γ[L,i,i]
  # K[p,q] = Σ_i (pi|iq) = Σ_L conj(Γ[L,i,p]) * Γ[L,i,q]
  println("  Reconstructing core Hamiltonian h₀ from Fock eigenvalues and Coulomb vertex...")

  Gamma = data.coulomb_vertex  # Γ[L,p,q], shape (naux, norb, norb)
  # Two-step Cholesky decomposition of the Coulomb vertex to produce
  # symmetric 3-index integrals: (pq|rs) = Σ_J B[p,q,J] * B[r,s,J]
  thr = EC.options.cholesky.thr
  sigma = EC.options.cholesky.sigma
  usesvd = EC.options.cholesky.usesvd
  println("  Performing two-step Cholesky decomposition (thr=$thr, σ=$sigma, usesvd=$usesvd)...")
  mmL_data, naux_new = decompose_vasp_vertex(Gamma, thr; sigma, usesvd)

  # Save via memory-mapped file
  mmLfile, mmL = newmmap(EC, "mmL", (norb, norb, naux_new), T; description="vasp_coulomb_vertex")
  mmL .= mmL_data
  closemmap(EC, mmLfile, mmL)

  occ = 1:nocc
  # Coulomb: v_J[L] = Σ_i Γ[L,i,i], then J[p,q] = Σ_L conj(Γ[L,q,p]) * v_J[L]
  ooL = @view(mmL[occ, occ, :])  # (nocc, nocc, naux_new)
  @mtensor v_J[L] := ooL[i, i, L]
  @mtensor J[p, q] := mmL[p, q, L] * v_J[L]

  # Exchange: K[p,q] = Σ_i (pi|iq) = Σ_{i,L} conj(Γ[L,i,p]) * Γ[L,i,q]
  moL = @view(mmL[:, occ, :])  # (norb, nocc, naux_new)
  omL = @view(mmL[occ, :, :])  # (nocc, norb, naux_new)
  @mtensor K[p, q] := moL[p, i, L] * omL[i, q, L]

  # h_{pq} = F_{pq} - 2*J[p,q] + K[p,q]
  h_full = -2 .* J .+ K
  for p in 1:norb
    h_full[p, p] += data.eigen_energies[p]
  end
  EC.fd.int1 = h_full


  # Set up orbital spaces
  setup_space_fd!(EC)

  println("  VASP data mapped to ECInfo:")
  println("    NORB = $norb, NELEC = $nelec, MS2 = $ms2")
  println("    Occupied: ", EC.space['o'])
  println("    Virtual:  ", EC.space['v'])
end

# --- Optional tensor loaders ---

function load_optional_tensor(dirpath::String, name::String)
  yaml_path = joinpath(dirpath, name * ".yaml")
  if !isfile(yaml_path)
    return nothing
  end
  meta = read_vasp_yaml(yaml_path)
  elements_path = joinpath(dirpath, name * ".elements")
  if !isfile(elements_path)
    return nothing
  end
  raw = read_vasp_elements(elements_path, meta)
  return raw .* meta.unit
end

function load_optional_1d(dirpath::String, name::String)
  data = load_optional_tensor(dirpath, name)
  return isnothing(data) ? Float64[] : vec(data)
end

function load_optional_2d(dirpath::String, name::String)
  data = load_optional_tensor(dirpath, name)
  if isnothing(data)
    return zeros(0, 0)
  end
  return ndims(data) == 2 ? data : reshape(data, size(data, 1), :)
end

function load_optional_4d(dirpath::String, name::String)
  data = load_optional_tensor(dirpath, name)
  if isnothing(data)
    return zeros(0, 0, 0, 0)
  end
  return data
end

end # module VaspInterface
