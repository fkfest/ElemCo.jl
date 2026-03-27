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
using ..ElemCo.ALPACADecomposition

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
    VaspCoulombMatrix{T} <: AbstractALPACAMatrix

Matrix-free representation of the VASP Coulomb matrix for ALPACA decomposition.

The matrix has compound indices ``I = (q-1) n + p`` (pair ``(p,q)``) and
``J = (s-1) n + r`` (pair ``(r,s)``), with elements:
``V_{IJ} = \\sum_L \\overline{\\Gamma^L_{qp}} \\, \\Gamma^L_{rs} = (pq|rs)``

The matrix is complex symmetric: ``V^T = V``.
"""
struct VaspCoulombMatrix{T} <: AbstractALPACAMatrix{T}
  "conj(Γ[L,q,p]) reshaped as (naux, n²)"
  Gtilde::Matrix{T}
  "Original Γ[L,p,q], shape (naux, norb, norb)"
  Gamma::Array{T,3}
  norb::Int
end

Base.size(mat::VaspCoulombMatrix) = (mat.norb^2, mat.norb^2)

function ALPACADecomposition.column!(buffer::AbstractVector, mat::VaspCoulombMatrix, j::Integer)
  norb = mat.norb
  # j maps to pair (r, s): j = (s-1)*norb + r
  r = ((j - 1) % norb) + 1
  s = ((j - 1) ÷ norb) + 1
  # buffer = Gtilde' * Gamma[:, r, s]
  mul!(buffer, transpose(mat.Gtilde), view(mat.Gamma, :, r, s))
  return buffer
end

function ALPACADecomposition.elements!(buffer::AbstractVector, mat::VaspCoulombMatrix{T},
                   pairs::AbstractVector{<:Tuple{<:Integer,<:Integer}}) where T
  norb = mat.norb
  @inbounds for k in eachindex(pairs)
    I, J = pairs[k]
    r = ((J - 1) % norb) + 1
    s = ((J - 1) ÷ norb) + 1
    # V[I,J] = Gtilde[:,I]' * Gamma[:,r,s]
    buffer[k] = transpose(view(mat.Gtilde, :, I)) * view(mat.Gamma, :, r, s)
  end
  return buffer
end

"""
    decompose_vasp_vertex(Gamma::Array{T,3}, tol::Float64; sigma::Float64=0.01) where T → (B, naux_new)

ALPACA decomposition of the VASP Coulomb vertex using a matrix-free interface.

Given ``\\Gamma^L_{pq}`` (shape `(naux, norb, norb)`), the 4-index integrals in
chemist notation are:
``(pq|rs) = \\sum_L \\overline{\\Gamma^L_{qp}} \\Gamma^L_{rs}``

This function produces ``B^J_{pq}`` (shape `(norb, norb, naux_new)`) such that:
``(pq|rs) \\approx \\sum_J B^J_{pq} B^J_{rs}``

Uses the ALPACA (Amended Low-rank Principal-element Adaptive Cross
Approximation) algorithm with a matrix-free interface that computes columns
of the ``n^2 \\times n^2`` Coulomb matrix on demand via BLAS-2 operations,
avoiding materialization of the full matrix.  Principal elements default to
the diagonal.

# Arguments
- `Gamma`: VASP Coulomb vertex, shape `(naux, norb, norb)`
- `tol`: decomposition threshold
- `sigma`: span factor for ALPACA batch screening (default: 0.01)

# Returns
- `B`: decomposed 3-index integrals, shape `(norb, norb, naux_new)`
- `naux_new`: number of decomposition vectors
"""
function decompose_vasp_vertex(Gamma::Array{T,3}, tol::Float64; sigma::Float64=0.01) where T
  naux, norb, _ = size(Gamma)
  n2 = norb * norb

  # Precompute Gtilde[L, I] = conj(Gamma[L, q, p]) where I = (q-1)*norb + p
  Gtilde = Matrix{T}(undef, naux, n2)
  for q in 1:norb, p in 1:norb
    I = (q - 1) * norb + p
    @inbounds for L in 1:naux
      Gtilde[L, I] = conj(Gamma[L, q, p])
    end
  end

  mat = VaspCoulombMatrix(Gtilde, Gamma, norb)

  # Coulomb matrix is complex symmetric (or real symmetric for Γ-point)
  opts = ALPACAOptions(tol=tol, sigma=sigma, symmetry=:symmetric, max_rank=min(n2, naux))

  result = alpaca(mat; options=opts)

  r = size(result.left, 2)
  nB = length(result.pivot_indices)
  println("  ALPACA decomposition: $r vectors from $naux original auxiliary functions ($nB pivots)")

  if !isempty(result.neg_indices)
    @warn "ALPACA found $(length(result.neg_indices)) negative eigenvalues in Coulomb matrix"
  end

  # For symmetric: A ≈ L * D * L^T, where D is diagonal with ±1 from neg_indices
  # For complex symmetric (typical VASP case): neg_indices is empty, A ≈ L * L^T
  B_out = reshape(result.left, norb, norb, r)

  return B_out, r
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
  # ALPACA decomposition of the Coulomb vertex to produce
  # symmetric 3-index integrals: (pq|rs) = Σ_J B[p,q,J] * B[r,s,J]
  thr = EC.options.cholesky.thr
  sigma = EC.options.cholesky.sigma
  println("  Performing ALPACA decomposition (thr=$thr, σ=$sigma)...")
  mmL_data, naux_new = decompose_vasp_vertex(Gamma, thr; sigma)

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
