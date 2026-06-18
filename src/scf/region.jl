"""
Region-tagged orbital construction.

Implements fragment occupied/virtual selection and PiOS-style π-space
construction on top of the orbital-localization primitives from
`OrbLocalization`.
"""
module OrbRegion
using LinearAlgebra
using Printf
using ..ElemCo.Constants: BOHR2ANGSTROM
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.MSystems
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.OrbTools
using ..ElemCo.TensorTools
using ..ElemCo.QMTensors
using ..ElemCo.Wavefunctions
using ..OrbLocalization: compute_iaos, localize_ibo, _compute_proto_iaos, _fix_sign_convention!

export region_orbitals

"""
    _region_exponent(EC::ECInfo)

  Resolve the effective localization exponent for region-tagged occupied-space
  construction, using the IBO default of `4` when `loc.exponent == 0`.
"""
function _region_exponent(EC::ECInfo)
  exponent = EC.options.loc.exponent
  return exponent == 0 ? 4 : exponent
end

"""
    _normalize_region_centers(centers)

  Normalize user-provided region-center input to a flat collection that can be
  resolved against the molecular system.

  Scalars are wrapped as a one-element vector, while iterable containers are
  collected into a flat array.
"""
function _normalize_region_centers(centers)
  if centers isa Integer || centers isa Symbol || centers isa AbstractString
    return Any[centers]
  elseif centers isa AbstractArray || centers isa Tuple || centers isa AbstractSet || centers isa AbstractRange
    return collect(centers)
  elseif applicable(iterate, centers)
    return collect(centers)
  end

  return Any[centers]
end

"""
    _resolve_region_centers(EC::ECInfo, centers; allow_empty=false)

  Resolve atom indices or center labels to the global and non-dummy local atom
  indices used by the region-selection machinery.
"""
function _resolve_region_centers(EC::ECInfo, centers; allow_empty::Bool=false)
  center_list = _normalize_region_centers(centers)
  isempty(center_list) && allow_empty && return Int[], Int[]
  isempty(center_list) && error("@region requires at least one atom center.")

  selected_global = Int[]
  for center in center_list
    if center isa Integer
      idx = Int(center)
      1 <= idx <= length(EC.system) || error("Atom index $idx out of range!")
      push!(selected_global, idx)
    else
      label = uppercase(String(center))
      found = false
      for (idx, atom) in enumerate(EC.system)
        if uppercase(atomic_centre_label(atom)) == label
          push!(selected_global, idx)
          found = true
        end
      end
      found || error("Atom $center not found in the system!")
    end
  end
  unique!(selected_global)

  atom_map = Dict{Int,Int}()
  real_counter = 0
  for (idx, atom) in enumerate(EC.system)
    if !is_dummy(atom)
      real_counter += 1
      atom_map[idx] = real_counter
    end
  end

  # Dummy (ghost) atoms have no IAO-charge row, so they are kept only in the global list
  # (used for the AO/PAO virtual-space support) and skipped in the non-dummy local numbering.
  selected_local = Int[]
  for idx in selected_global
    is_dummy(EC.system[idx]) && continue
    push!(selected_local, atom_map[idx])
  end
  unique!(selected_local)

  return selected_global, selected_local
end

"""
    _merge_region_centers(primary::Vector{Int}, extras::Vector{Int})

  Merge two center-index lists while preserving the existing order of the first
  occurrence of each center.
"""
function _merge_region_centers(primary::Vector{Int}, extras::Vector{Int})
  merged = copy(primary)
  append!(merged, extras)
  unique!(merged)
  return merged
end

"""
    _resolve_region_selection(EC::ECInfo, centers, mode::Symbol,
                              inclusive_centers, exclusive_centers)

  Resolve the effective inclusive and exclusive region-center sets by combining
  the explicit `@region` argument with the option-level center lists.
"""
function _resolve_region_selection(EC::ECInfo, centers, mode::Symbol,
                                  inclusive_centers, exclusive_centers)
  selected_global, selected_local = _resolve_region_centers(EC, centers; allow_empty=true)
  inclusive_global, inclusive_local = _resolve_region_centers(EC, inclusive_centers; allow_empty=true)
  exclusive_global, exclusive_local = _resolve_region_centers(EC, exclusive_centers; allow_empty=true)

  if mode == :inclusive
    inclusive_global = _merge_region_centers(selected_global, inclusive_global)
    inclusive_local = _merge_region_centers(selected_local, inclusive_local)
  else
    exclusive_global = _merge_region_centers(selected_global, exclusive_global)
    exclusive_local = _merge_region_centers(selected_local, exclusive_local)
  end

  all_global = _merge_region_centers(inclusive_global, exclusive_global)
  all_local = _merge_region_centers(inclusive_local, exclusive_local)
  isempty(all_global) && error("@region requires at least one center, either through the macro argument or region.inclusive_centers / region.exclusive_centers.")

  return (; all_global, all_local, inclusive_global, inclusive_local, exclusive_global, exclusive_local)
end

"""
    _difference_preserve_order(all_items::Vector{Int}, kept_items::Vector{Int})

  Return the elements of `all_items` that are not in `kept_items`, preserving
  the original order from `all_items`.
"""
function _difference_preserve_order(all_items::Vector{Int}, kept_items::Vector{Int})
  kept = Set(kept_items)
  return [item for item in all_items if item ∉ kept]
end

"""
    _region_space_ranges(EC::ECInfo, restricted::Bool)

  Collect the active occupied, frozen-core, and virtual orbital ranges used for
  region construction, returning separate alpha/beta tuples for restricted and
  unrestricted references.

  This helper snapshots the temporary system-space dictionaries and restores the
  caller's original `EC.space` before building the returned tuples.
"""
function _region_space_ranges(EC::ECInfo, restricted::Bool)
  space_save, space_b4freeze = restore_system_space!(EC; verbose=false)
  active_space = save_space(EC)
  restore_space!(EC, space_save)

  occ_a = active_space['o']
  all_occ_a = space_b4freeze['o']
  core_a = _difference_preserve_order(all_occ_a, occ_a)
  virt_a = active_space['v']
  alpha = (all_occ=all_occ_a, occ=occ_a, core=core_a, virt=virt_a)

  if restricted
    beta = (all_occ=Int[], occ=Int[], core=Int[], virt=Int[])
  else
    occ_b = active_space['O']
    all_occ_b = space_b4freeze['O']
    core_b = _difference_preserve_order(all_occ_b, occ_b)
    virt_b = active_space['V']
    beta = (all_occ=all_occ_b, occ=occ_b, core=core_b, virt=virt_b)
  end

  return alpha, beta
end

"""
    _compute_iao_basis_metadata(EC::ECInfo)

  Build the IAO basis metadata needed for π-space selection: the retained MINAO
  AO descriptors, their global atom indices, and their compact non-dummy atom
  numbering.
"""
function _compute_iao_basis_metadata(EC::ECInfo)
  bminao = generate_minao_basis(EC, EC.options.loc.minao)
  aos_min_full = ao_list(bminao)
  ghost_mask = [is_dummy(EC.system[Int(ao.icentre)]) for ao in aos_min_full]
  real_idx = findall(.!ghost_mask)
  iao_aos = aos_min_full[real_idx]
  iao_global_atoms = [Int(ao.icentre) for ao in iao_aos]
  unique_atoms = sort(unique(iao_global_atoms))
  atom_remap = Dict(a => idx for (idx, a) in enumerate(unique_atoms))
  iao_local_atoms = [atom_remap[a] for a in iao_global_atoms]
  return iao_aos, iao_global_atoms, iao_local_atoms
end

"""
    _compute_raw_iaos(EC::ECInfo, cMO_occ::AbstractMatrix)

  Construct the non-orthogonalized proto-IAOs before the final Lowdin step.
  The returned AO-space columns use the revised Senjean/Knizia formulation and
  are used to define π-target directions for `region.pi` selection.
"""
function _compute_raw_iaos(EC::ECInfo, cMO_occ::AbstractMatrix{T}) where T
  proto_iao, _, _, _ = _compute_proto_iaos(EC, cMO_occ)
  return proto_iao
end

const PI_COVALENT_RADII_ANGSTROM = Dict{String,Float64}(
  "H" => 0.31,
  "HE" => 0.28,
  "B" => 0.85,
  "C" => 0.76,
  "N" => 0.71,
  "O" => 0.66,
  "F" => 0.57,
  "NE" => 0.58,
  "AL" => 1.21,
  "SI" => 1.11,
  "P" => 1.07,
  "S" => 1.05,
  "CL" => 1.02,
  "AR" => 1.06,
  "GA" => 1.22,
  "GE" => 1.20,
  "AS" => 1.19,
  "SE" => 1.20,
  "BR" => 1.20,
  "KR" => 1.16,
  "IN" => 1.42,
  "SN" => 1.39,
  "SB" => 1.39,
  "TE" => 1.38,
  "I" => 1.39,
  "XE" => 1.40,
)

_pi_covalent_radius(atom) = get(PI_COVALENT_RADII_ANGSTROM, element_LABEL(atom), 1.20)

"""
    _rank_neighbors_by_distance(EC::ECInfo, atom_index::Int, candidates::Vector{Int})

  Sort candidate neighbor atoms by their distance from `atom_index`, excluding
  the center atom itself and dummy atoms.
"""
function _rank_neighbors_by_distance(EC::ECInfo, atom_index::Int, candidates::Vector{Int})
  center = atomic_position(EC.system[atom_index]) * BOHR2ANGSTROM
  ranked = Tuple{Float64,Int}[]
  for idx in candidates
    idx == atom_index && continue
    is_dummy(EC.system[idx]) && continue
    distance = norm(atomic_position(EC.system[idx]) * BOHR2ANGSTROM - center)
    push!(ranked, (distance, idx))
  end
  sort!(ranked, by=first)
  return [idx for (_, idx) in ranked]
end

"""
    _bonded_neighbors(EC::ECInfo, atom_index::Int)

  Identify atoms bonded to `atom_index` using a covalent-radius cutoff tailored
  for the local π-plane heuristics.
"""
function _bonded_neighbors(EC::ECInfo, atom_index::Int)
  center = atomic_position(EC.system[atom_index]) * BOHR2ANGSTROM
  radius = _pi_covalent_radius(EC.system[atom_index])
  bonded = Tuple{Float64,Int}[]
  for (idx, atom) in enumerate(EC.system)
    idx == atom_index && continue
    is_dummy(atom) && continue
    cutoff = 1.3 * (radius + _pi_covalent_radius(atom))
    distance = norm(atomic_position(atom) * BOHR2ANGSTROM - center)
    distance < cutoff || continue
    push!(bonded, (distance, idx))
  end
  sort!(bonded, by=first)
  return [idx for (_, idx) in bonded]
end

"""
    _nearest_neighbors(EC::ECInfo, atom_index::Int; exclude=Int[], max_neighbors=3)

  Return up to `max_neighbors` nearest non-dummy atoms around `atom_index`,
  skipping any atoms listed in `exclude`.
"""
function _nearest_neighbors(EC::ECInfo, atom_index::Int; exclude::Vector{Int}=Int[], max_neighbors::Int=3)
  candidates = Int[]
  for (idx, atom) in enumerate(EC.system)
    idx == atom_index && continue
    idx in exclude && continue
    is_dummy(atom) && continue
    push!(candidates, idx)
  end
  ranked = _rank_neighbors_by_distance(EC, atom_index, candidates)
  return ranked[1:min(max_neighbors, length(ranked))]
end

"""
    _local_plane_atoms(EC::ECInfo, atom_index::Int)

  Choose up to three nearby atoms that define the local π-plane around the
  selected center, preferring bonded neighbors and falling back to nearest
  atoms when necessary.
"""
function _local_plane_atoms(EC::ECInfo, atom_index::Int)
  plane_atoms = _bonded_neighbors(EC, atom_index)

  if length(plane_atoms) == 1
    anchor = plane_atoms[1]
    anchor_neighbors = [idx for idx in _bonded_neighbors(EC, anchor) if idx != atom_index]
    append!(plane_atoms, _rank_neighbors_by_distance(EC, atom_index, anchor_neighbors))
  end

  if length(plane_atoms) < 2
    extras = _nearest_neighbors(EC, atom_index; exclude=vcat([atom_index], plane_atoms), max_neighbors=3)
    append!(plane_atoms, extras)
  end

  unique!(plane_atoms)
  plane_atoms = _rank_neighbors_by_distance(EC, atom_index, plane_atoms)
  length(plane_atoms) >= 2 || error("Atom $(atomic_centre_label(EC.system[atom_index])) needs at least two atoms to define a local π-plane.")
  return plane_atoms[1:min(3, length(plane_atoms))]
end

"""
    _local_plane_normal_from_vectors(vectors)

  Determine the normal vector of the best-fit plane through the supplied bond
  vectors by diagonalizing the corresponding inertia-like matrix.
"""
function _local_plane_normal_from_vectors(vectors)
  inertia = zeros(Float64, 3, 3)
  for vec in vectors
    inertia .+= vec * vec'
  end
  eigenpairs = eigen(Hermitian(inertia))
  return eigenpairs.vectors[:, argmin(eigenpairs.values)], eigenpairs.values
end

"""
    _local_plane_normal(EC::ECInfo, atom_index::Int)

  Compute a normalized local π-axis for `atom_index` from its bonded-neighbor
  geometry, augmenting the plane definition with nearby atoms when the initial
  plane is degenerate.
"""
function _local_plane_normal(EC::ECInfo, atom_index::Int)
  center = atomic_position(EC.system[atom_index])
  plane_atoms = _local_plane_atoms(EC, atom_index)
  vectors = [atomic_position(EC.system[idx]) - center for idx in plane_atoms]
  normal, eigenvalues = _local_plane_normal_from_vectors(vectors)

  if eigenvalues[2] - eigenvalues[1] <= 1.e-10
    extras = _nearest_neighbors(EC, atom_index; exclude=vcat([atom_index], plane_atoms), max_neighbors=3)
    for idx in extras
      push!(vectors, atomic_position(EC.system[idx]) - center)
      normal, eigenvalues = _local_plane_normal_from_vectors(vectors)
      eigenvalues[2] - eigenvalues[1] > 1.e-10 && break
    end
  end

  norm_normal = norm(normal)
  norm_normal > 1.e-10 || error("Failed to determine a unique local π-axis for atom $(atomic_centre_label(EC.system[atom_index])).")
  eigenvalues[2] - eigenvalues[1] > 1.e-10 || error("Failed to determine a local π-plane for atom $(atomic_centre_label(EC.system[atom_index])).")
  return normal / norm_normal
end

"""
    _select_valence_p_iaos(EC::ECInfo, iao_aos, iao_global_atoms::Vector{Int}, atom_index::Int)

  Select the highest-principal-quantum-number Cartesian p-shell IAOs centered
  on `atom_index`, returning the `(p_x, p_y, p_z)` basis indices.
"""
function _select_valence_p_iaos(EC::ECInfo, iao_aos, iao_global_atoms::Vector{Int}, atom_index::Int)
  p_candidates = Tuple{Int,Any}[]
  for (idx, ao) in enumerate(iao_aos)
    iao_global_atoms[idx] == atom_index || continue
    Int(ao.l) == 1 || continue
    push!(p_candidates, (idx, ao))
  end
  isempty(p_candidates) && error("No valence p IAOs found for atom $(atomic_centre_label(EC.system[atom_index])).")

  max_n = maximum(Int(ao.n) for (_, ao) in p_candidates)
  p_shell = Dict{Int,Int}()
  for (idx, ao) in p_candidates
    Int(ao.n) == max_n || continue
    p_shell[Int(ao.ml)] = idx
  end

  for ml in (-1, 0, 1)
    haskey(p_shell, ml) || error("Incomplete valence p shell found for atom $(atomic_centre_label(EC.system[atom_index])).")
  end

  return p_shell[-1], p_shell[0], p_shell[1]
end

"""
    _build_pi_target_orbitals(EC::ECInfo, C_iao::AbstractMatrix, selected_global::Vector{Int})

  Build one AO-space π-target orbital per selected center by projecting the
  local π-axis onto the corresponding valence p-shell IAOs.
"""
function _build_pi_target_orbitals(EC::ECInfo, C_iao::AbstractMatrix{T}, selected_global::Vector{Int}) where T
  iao_aos, iao_global_atoms, _ = _compute_iao_basis_metadata(EC)
  PZ_AO = zeros(T, size(C_iao, 1), length(selected_global))

  for (icol, atom_index) in enumerate(selected_global)
    normal = _local_plane_normal(EC, atom_index)
    ix, iy, iz = _select_valence_p_iaos(EC, iao_aos, iao_global_atoms, atom_index)
    PZ_AO[:, icol] = normal[1] * C_iao[:, ix] + normal[2] * C_iao[:, iy] + normal[3] * C_iao[:, iz]
  end

  return PZ_AO
end

"""
    _pi_projection_rotation(cMO_block::AbstractMatrix, PZ_AO::AbstractMatrix, S::AbstractMatrix)

  Diagonalize the projected π-overlap metric for a molecular-orbital block and
  return the rotation that orders the block by decreasing π-character together
  with the corresponding overlap scores.
"""
function _pi_projection_rotation(cMO_block::AbstractMatrix{Tc}, PZ_AO::AbstractMatrix{Tp},
                                 S::AbstractMatrix) where {Tc,Tp}
  T = promote_type(Tc, Tp, eltype(S))
  nblock = size(cMO_block, 2)
  nblock == 0 && return Matrix{T}(I, 0, 0), Float64[]

  X = PZ_AO' * S * cMO_block
  target_overlap = Hermitian(PZ_AO' * S * PZ_AO)
  overlap = Hermitian(X' * (target_overlap \ X))
  eigenpairs = eigen(overlap)
  perm = sortperm(real.(eigenpairs.values); rev=true)
  R = Matrix{T}(eigenpairs.vectors[:, perm])
  scores = real.(eigenpairs.values[perm])
  _fix_sign_convention!(R)

  return R, scores
end

const PI_GROUP13 = Set(["B", "AL", "GA", "IN", "TL"])
const PI_GROUP14 = Set(["C", "SI", "GE", "SN", "PB"])
const PI_GROUP15 = Set(["N", "P", "AS", "SB", "BI"])
const PI_GROUP16 = Set(["O", "S", "SE", "TE", "PO"])
const PI_GROUP17 = Set(["F", "CL", "BR", "I", "AT"])

"""
    _pi_electron_contribution(EC::ECInfo, atom_index::Int)

  Estimate the number of π electrons contributed by the selected atom based on
  its element and local bonding pattern.
"""
function _pi_electron_contribution(EC::ECInfo, atom_index::Int)
  atom = EC.system[atom_index]
  label = element_LABEL(atom)
  degree = length(_bonded_neighbors(EC, atom_index))

  if label in PI_GROUP13
    return 0
  elseif label in PI_GROUP14
    return 1
  elseif label in PI_GROUP15
    return degree >= 3 ? 2 : 1
  elseif label in PI_GROUP16
    return degree >= 2 ? 2 : 1
  elseif label in PI_GROUP17
    return 2
  end

  error("region.pi electron counting is only implemented for main-group p-block atoms. Unsupported center $(atomic_centre_label(atom)).")
end

"""
    _estimate_pi_requested_counts(EC::ECInfo, selected_global::Vector{Int}, alpha_occ::Int,
                                  beta_occ::Int, restricted::Bool; pi_electrons=-1,
                                  pi_occupied=-1, pi_virtual=-1)

  Estimate how many occupied and virtual π orbitals should be retained for the
  selected region centers, optionally overriding the automatic PiOS electron
  count or restricting the retained occupied/virtual frontier subset.
"""
function _estimate_pi_requested_counts(EC::ECInfo, selected_global::Vector{Int}, alpha_occ::Int,
                                       beta_occ::Int, restricted::Bool;
                                       pi_electrons::Int=-1,
                                       pi_occupied::Int=-1,
                                       pi_virtual::Int=-1)
  ncenters = length(selected_global)
  ncenters > 0 || error("No π centers selected for region.pi.")
  pi_electrons >= -1 || error("region.pi_electrons must be -1 or a non-negative integer.")
  pi_occupied >= -1 || error("region.pi_occupied must be -1 or a non-negative integer.")
  pi_virtual >= -1 || error("region.pi_virtual must be -1 or a non-negative integer.")

  npi_e = pi_electrons >= 0 ? pi_electrons : sum(_pi_electron_contribution(EC, idx) for idx in selected_global)
  npi_e <= 2 * ncenters || error("Estimated π-electron count ($npi_e) exceeds the capacity of the selected π centers ($ncenters).")

  if restricted
    iseven(npi_e) || error("Estimated an odd number of π electrons ($npi_e) for a restricted reference.")
    default_occ = div(npi_e, 2)
    default_virt = max(ncenters - default_occ, 0)
    requested_occ = pi_occupied >= 0 ? pi_occupied : default_occ
    requested_virt = pi_virtual >= 0 ? pi_virtual : default_virt
    requested_occ > 0 || error("region.pi_occupied must select at least one occupied π orbital for a restricted reference.")
    requested_virt >= 0 || error("region.pi_virtual must be non-negative.")
    requested_occ + requested_virt <= ncenters || error("Requested restricted π subset ($(requested_occ) occupied + $(requested_virt) virtual) exceeds the number of local p'_z orbitals ($ncenters).")
    requested_occ <= alpha_occ || error("Requested $(requested_occ) occupied π orbitals, but only $(alpha_occ) occupied orbitals are available.")
    return requested_occ, requested_virt, 0, 0
  end

  (pi_occupied >= 0 || pi_virtual >= 0) && error("region.pi_occupied and region.pi_virtual currently require a restricted reference; use region.pi_electrons or the automatic PiOS count for unrestricted references.")

  ms2 = alpha_occ - beta_occ
  iseven(npi_e + ms2) || error("Estimated π-electron count ($npi_e) is incompatible with the reference spin state (M_S=$(ms2 / 2)).")
  requested_occ_a = div(npi_e + ms2, 2)
  requested_occ_b = div(npi_e - ms2, 2)
  requested_occ_a <= alpha_occ || error("Estimated $(requested_occ_a) occupied α π orbitals, but only $(alpha_occ) occupied α orbitals are available.")
  requested_occ_b <= beta_occ || error("Estimated $(requested_occ_b) occupied β π orbitals, but only $(beta_occ) occupied β orbitals are available.")
  requested_virt_a = max(ncenters - requested_occ_a, 0)
  requested_virt_b = max(ncenters - requested_occ_b, 0)
  return requested_occ_a, requested_virt_a, requested_occ_b, requested_virt_b
end

"""
    _select_region_occupied(charges::AbstractMatrix{<:Real}, inclusive_centers::Vector{Int},
                            exclusive_centers::Vector{Int}, threshold::Float64)

  Select localized occupied orbitals whose large atomic charges satisfy either
  the inclusive-center rule or the exclusive-center rule.
"""
function _select_region_occupied(charges::AbstractMatrix{<:Real}, inclusive_centers::Vector{Int},
                                 exclusive_centers::Vector{Int}, threshold::Float64)
  inclusive_set = Set(inclusive_centers)
  exclusive_set = Set(exclusive_centers)
  selected = Int[]

  for imo in 1:size(charges, 2)
    large_atoms = Int[]
    for iatom in 1:size(charges, 1)
      charges[iatom, imo] >= threshold || continue
      push!(large_atoms, iatom)
    end
    isempty(large_atoms) && continue

    inclusive_match = !isempty(inclusive_set) && any(iatom in inclusive_set for iatom in large_atoms)
    exclusive_match = !isempty(exclusive_set) &&
      any(iatom in exclusive_set for iatom in large_atoms) &&
      all(iatom in exclusive_set for iatom in large_atoms)

    if inclusive_match || exclusive_match
      push!(selected, imo)
    end
  end

  return selected
end

"""
    _collect_support_atoms(charges::AbstractMatrix{<:Real}, nfrag_occ::Int,
                           selected_centers::Vector{Int}, threshold::Float64)

  Expand the fragment-support atom set from the selected centers by adding any
  atom that carries at least `threshold` charge on one of the kept occupied
  fragment orbitals.
"""
function _collect_support_atoms(charges::AbstractMatrix{<:Real}, nfrag_occ::Int,
                                selected_centers::Vector{Int}, threshold::Float64)
  support = Set(selected_centers)
  for imo in 1:nfrag_occ, iatom in 1:size(charges, 1)
    if charges[iatom, imo] >= threshold
      push!(support, iatom)
    end
  end
  return sort!(collect(support))
end

"""
    _accumulated_fragment_atom_charges(charges::AbstractMatrix{<:Real}, nfrag_occ::Int)

  Sum the atom-resolved IAO charges over the selected fragment occupied
  orbitals, returning one accumulated charge per atom.
"""
function _accumulated_fragment_atom_charges(charges::AbstractMatrix{<:Real}, nfrag_occ::Int)
  nfrag_occ == 0 && return zeros(eltype(charges), size(charges, 1))
  return vec(sum(@view(charges[:, 1:nfrag_occ]); dims=2))
end

"""
    _collect_support_atoms_accumulated(charges::AbstractMatrix{<:Real}, nfrag_occ::Int,
                                       selected_centers::Vector{Int}, threshold::Float64)

  Expand the virtual-support atom set from the selected centers by summing the
  fragment IAO charge over all kept occupied fragment orbitals and retaining
  atoms whose accumulated charge exceeds `threshold`.
"""
function _collect_support_atoms_accumulated(charges::AbstractMatrix{<:Real}, nfrag_occ::Int,
                                            selected_centers::Vector{Int}, threshold::Float64)
  support = Set(selected_centers)
  accumulated = _accumulated_fragment_atom_charges(charges, nfrag_occ)
  for iatom in eachindex(accumulated)
    if accumulated[iatom] >= threshold
      push!(support, iatom)
    end
  end
  return sort!(collect(support))
end

"""
    _accumulated_lowdin_pops(cMO_frag_occ::AbstractMatrix, S::AbstractMatrix,
                             ao_atoms_global::Vector{Int}, atoms::Vector{Int}) -> Dict{Int,Float64}

  Accumulated Löwdin population ``\\sum_i \\sum_{\\mu\\in A} (S^{1/2} C)_{\\mu i}^2`` of the fragment
  occupied orbitals `cMO_frag_occ` on each requested (global) atom index. Used to measure the
  contribution of atoms — including dummy/ghost atoms, which carry no IAOs — to the fragment.
"""
function _accumulated_lowdin_pops(cMO_frag_occ::AbstractMatrix, S::AbstractMatrix,
                                  ao_atoms_global::Vector{Int}, atoms::Vector{Int})
  pops = Dict{Int,Float64}()
  (isempty(atoms) || size(cMO_frag_occ, 2) == 0) && return pops
  Shalf = real.(sqrt(Hermitian(Matrix(S))))
  pop = abs2.(Shalf * cMO_frag_occ)
  for a in atoms
    aos = [μ for μ in eachindex(ao_atoms_global) if ao_atoms_global[μ] == a]
    pops[a] = isempty(aos) ? 0.0 : sum(@view pop[aos, :])
  end
  return pops
end

"""
    _collect_ghost_support(EC::ECInfo, cMO_frag_occ::AbstractMatrix, S::AbstractMatrix,
                           ao_atoms_global::Vector{Int}, threshold::Float64) -> (support, pops)

  Return the global indices of dummy (ghost) atoms whose accumulated Löwdin population over the
  fragment occupied orbitals reaches `threshold`, together with the population of every ghost
  atom (`Dict` global index → accumulated population). Ghost atoms carry no IAOs, so their
  contribution is measured directly from the AO basis (see [`_accumulated_lowdin_pops`](@ref)).
"""
function _collect_ghost_support(EC::ECInfo, cMO_frag_occ::AbstractMatrix, S::AbstractMatrix,
                                ao_atoms_global::Vector{Int}, threshold::Float64)
  ghost_atoms = sort!(unique(Int[a for a in ao_atoms_global if is_dummy(EC.system[a])]))
  pops = _accumulated_lowdin_pops(cMO_frag_occ, S, ao_atoms_global, ghost_atoms)
  support = sort!(Int[g for g in ghost_atoms if get(pops, g, 0.0) >= threshold])
  return support, pops
end

"""
    _fragment_iao_targets(C_iao::AbstractMatrix, iao_atoms::Vector{Int}, support_atoms::Vector{Int})

  Select the IAO columns centered on the requested support atoms. These are
  later projected into the virtual space to build antibonding-like fragment
  virtual targets.
"""
function _fragment_iao_targets(C_iao::AbstractMatrix{T}, iao_atoms::Vector{Int},
                               support_atoms::Vector{Int}) where T
  isempty(support_atoms) && return Matrix{T}(undef, size(C_iao, 1), 0)
  support_set = Set(support_atoms)
  iao_subset = [i for i in eachindex(iao_atoms) if iao_atoms[i] in support_set]
  isempty(iao_subset) && return Matrix{T}(undef, size(C_iao, 1), 0)
  return Matrix{T}(C_iao[:, iao_subset])
end

"""
    _support_ao_targets(cMO_virt::AbstractMatrix, ao_atoms::Vector{Int}, support_atoms::Vector{Int})

  Build AO-basis unit-vector columns for the AOs centered on `support_atoms`.
  These columns are used as the legacy OPAO augmentation targets.
"""
function _support_ao_targets(cMO_virt::AbstractMatrix{T}, ao_atoms::Vector{Int},
                             support_atoms::Vector{Int}) where T
  isempty(support_atoms) && return Matrix{T}(undef, size(cMO_virt, 1), 0)
  support_set = Set(support_atoms)
  ao_subset = [i for i in eachindex(ao_atoms) if ao_atoms[i] in support_set]
  nao = size(cMO_virt, 1)
  C_target = zeros(T, nao, length(ao_subset))
  for (icol, iao) in enumerate(ao_subset)
    C_target[iao, icol] = one(T)
  end
  return C_target
end

"""
    _project_virtual_targets(cMO_virt::AbstractMatrix, S::AbstractMatrix,
                             C_target::AbstractMatrix; existing=nothing, tol=1e-8)

  Project arbitrary AO-space target vectors into the canonical virtual space,
  optionally removing components already spanned by the orthonormal AO-space
  vectors in `existing`, and return both the AO-space orthonormalized targets
  and the corresponding virtual-space rotation columns.
"""
function _project_virtual_targets(cMO_virt::AbstractMatrix{Tc}, S::AbstractMatrix,
                                  C_target::AbstractMatrix{Tt};
                                  existing=nothing,
                                  tol::Float64=1e-8) where {Tc,Tt}
  T = promote_type(Tc, Tt, eltype(S))
  nvirt = size(cMO_virt, 2)
  ntarget = size(C_target, 2)
  ntarget == 0 && return Matrix{T}(undef, size(cMO_virt, 1), 0), Matrix{T}(undef, nvirt, 0)

  C_proj = Matrix{T}(cMO_virt * (cMO_virt' * S * C_target))
  if !isnothing(existing) && size(existing, 2) > 0
    C_proj .-= existing * (existing' * S * C_proj)
  end

  metric = Hermitian(C_proj' * S * C_proj)
  M = sqrtinvchol(metric; tol=tol, max_rank=nvirt)
  nsel = size(M, 2)
  nsel == 0 && return Matrix{T}(undef, size(cMO_virt, 1), 0), Matrix{T}(undef, nvirt, 0)

  C_sel = Matrix{T}(C_proj * M)
  R_sel = Matrix{T}(cMO_virt' * S * C_sel)
  _fix_sign_convention!(R_sel)
  return C_sel, R_sel
end

"""
    _fragment_complement_virtual_rotation(cMO_virt::AbstractMatrix, S::AbstractMatrix,
                                          C_iao::AbstractMatrix, iao_atoms::Vector{Int},
                                          ao_atoms::Vector{Int}, support_atoms::Vector{Int}; tol=1e-8)

  Build the default fragment virtual-space rotation by first projecting the
  fragment IAOs into the virtual space to obtain antibonding-like complement
  vectors and then augmenting that subspace with support-atom OPAO targets.
"""
function _fragment_complement_virtual_rotation(cMO_virt::AbstractMatrix{T}, S::AbstractMatrix,
                                               C_iao::AbstractMatrix, iao_atoms::Vector{Int},
                                               ao_atoms::Vector{Int}, support_atoms::Vector{Int};
                                               tol::Float64=1e-8) where T
  nvirt = size(cMO_virt, 2)
  nvirt == 0 && return Matrix{T}(I, 0, 0), 0
  isempty(support_atoms) && return Matrix{T}(I, nvirt, nvirt), 0

  C_iao_target = _fragment_iao_targets(C_iao, iao_atoms, support_atoms)
  C_complement, R_complement = _project_virtual_targets(cMO_virt, S, C_iao_target; tol=tol)

  C_ao_target = _support_ao_targets(cMO_virt, ao_atoms, support_atoms)
  _, R_aug = _project_virtual_targets(cMO_virt, S, C_ao_target; existing=C_complement, tol=tol)

  nfrag_virt = size(R_complement, 2) + size(R_aug, 2)
  nfrag_virt == 0 && return Matrix{T}(I, nvirt, nvirt), 0

  R_frag = hcat(R_complement, R_aug)
  if nfrag_virt == nvirt
    R_virt = R_frag
  else
    R_rest = Matrix{T}(nullspace(adjoint(R_frag)))
    R_virt = hcat(R_frag, R_rest)
  end

  _fix_sign_convention!(R_virt)
  return R_virt, nfrag_virt
end

"""
    _fragment_opao_rotation(cMO_virt::AbstractMatrix, S::AbstractMatrix,
                            ao_atoms::Vector{Int}, support_atoms::Vector{Int}; tol=1e-8)

  Build the orthogonal PAO rotation that places fragment-supported virtual
  orbitals first, followed by an orthogonal complement for the remaining
  virtual space.
"""
function _fragment_opao_rotation(cMO_virt::AbstractMatrix{T}, S::AbstractMatrix,
                                 ao_atoms::Vector{Int}, support_atoms::Vector{Int};
                                 tol::Float64=1e-8) where T
  nvirt = size(cMO_virt, 2)
  nvirt == 0 && return Matrix{T}(I, 0, 0), 0
  isempty(support_atoms) && return Matrix{T}(I, nvirt, nvirt), 0

  support_set = Set(support_atoms)
  ao_subset = [iao for iao in eachindex(ao_atoms) if ao_atoms[iao] in support_set]
  isempty(ao_subset) && return Matrix{T}(I, nvirt, nvirt), 0

  C_PAO = cMO_virt * (cMO_virt' * S[:, ao_subset])
  S_PAO = Hermitian(C_PAO' * S * C_PAO)
  M = sqrtinvchol(S_PAO; tol=tol, max_rank=nvirt)
  nfrag_virt = size(M, 2)
  nfrag_virt == 0 && return Matrix{T}(I, nvirt, nvirt), 0

  C_OPAO = C_PAO * M
  R_frag = Matrix{T}(cMO_virt' * S * C_OPAO)
  _fix_sign_convention!(R_frag)

  if nfrag_virt == nvirt
    R_virt = R_frag
  else
    R_comp = Matrix{T}(nullspace(adjoint(R_frag)))
    R_virt = hcat(R_frag, R_comp)
  end

  _fix_sign_convention!(R_virt)
  return R_virt, nfrag_virt
end

"""
    _reconstruct_fock_matrix(cMO::AbstractMatrix, S::AbstractMatrix, energies::Vector{Float64})

  Reconstruct an AO-space Fock matrix from orbital coefficients and diagonal
  orbital energies for the pseudo-canonical region post-processing.
"""
function _reconstruct_fock_matrix(cMO::AbstractMatrix{T}, S::AbstractMatrix,
                                  energies::Vector{Float64}) where T
  isempty(energies) && error("region.pseudo requires orbital energies in the input dump.")
  return S * cMO * Diagonal(T.(energies)) * cMO' * S
end

"""
    _pseudocanonicalize_block!(cMO_region::AbstractMatrix, energies::Vector{Float64},
                               F_AO::AbstractMatrix, block_indices::Vector{Int})

  Semicanonicalize the selected orbital block with respect to the AO-space Fock
  matrix and overwrite both the rotated orbitals and the corresponding orbital
  energies in place.
"""
function _pseudocanonicalize_block!(cMO_region::AbstractMatrix{T}, energies::Vector{Float64},
                                    F_AO::AbstractMatrix, block_indices::Vector{Int}) where T
  isempty(block_indices) && return

  C_block = cMO_region[:, block_indices]
  F_block = Hermitian(C_block' * F_AO * C_block)
  eigenpairs = eigen(F_block)
  R_block = Matrix{T}(eigenpairs.vectors)
  _fix_sign_convention!(R_block)
  cMO_region[:, block_indices] = C_block * R_block
  energies[block_indices] = real.(eigenpairs.values)
end

"""
    _build_region_spin(cMO_full::AbstractMatrix, energies::Vector{Float64}, occupied_data,
                       S::AbstractMatrix, C_iao::AbstractMatrix, iao_atoms::Vector{Int},
                       natom::Int, ao_atoms::Vector{Int}, inclusive_centers::Vector{Int},
                       exclusive_centers::Vector{Int}; ...)

  Build the region-tagged occupied and fragment-virtual spaces for one spin
  block using IBO-selected occupied orbitals and OPAO-selected fragment
  virtuals.
"""
function _build_region_spin(EC::ECInfo, cMO_full::AbstractMatrix{T}, energies::Vector{Float64},
                            occupied_data, S::AbstractMatrix, C_iao::AbstractMatrix,
                            iao_atoms::Vector{Int}, natom::Int, ao_atoms_global::Vector{Int},
                            real_globals::Vector{Int},
                            inclusive_centers::Vector{Int}, exclusive_centers::Vector{Int};
                            virtual_mode::Symbol,
                            occ_charge_thr::Float64, atom_charge_thr::Float64,
                            exponent::Int, spin_label::String="",
                            pao_centers::Vector{Int}=Int[],
                            pseudo::Bool=false, F_AO=nothing) where T
  valence_occ_range = occupied_data.occ
  virt_range = occupied_data.virt
  core_range = occupied_data.core

  isempty(valence_occ_range) && error("No $(spin_label)occupied orbitals available for @region.")

  cMO_occ = cMO_full[:, valence_occ_range]
  R_occ, charges = localize_ibo(cMO_occ, S, C_iao, iao_atoms, natom;
    exponent=exponent)

  fragment_occ = _select_region_occupied(charges, inclusive_centers, exclusive_centers, occ_charge_thr)
  isempty(fragment_occ) && error("No $(spin_label)occupied fragment orbitals selected; lower region.occ_charge_thr or adjust the requested region centers.")

  # Keep the selected fragment orbitals first while collecting the virtual-space
  # support atoms (the charge-based helpers expect the fragment block in columns 1:nfrag_occ).
  rest_occ = _difference_preserve_order(collect(1:size(R_occ, 2)), fragment_occ)
  occ_perm = vcat(fragment_occ, rest_occ)
  R_occ = R_occ[:, occ_perm]
  charges = charges[:, occ_perm]
  nfrag_occ = length(fragment_occ)

  support_centers = _merge_region_centers(inclusive_centers, exclusive_centers)
  accumulated = virtual_mode == :complement
  if accumulated
    support_atoms = _collect_support_atoms_accumulated(charges, nfrag_occ, support_centers, atom_charge_thr)
  elseif virtual_mode in (:support_opao, :opao)
    support_atoms = _collect_support_atoms(charges, nfrag_occ, support_centers, atom_charge_thr)
  else
    error("Unknown region.virtual = $virtual_mode. Valid modes are :complement and :support_opao.")
  end
  # real PAO centers extend the (local) charge-based support used for the IAO-complement targets
  local_of = Dict(g => k for (k, g) in enumerate(real_globals))
  pao_real_local = Int[local_of[p] for p in pao_centers if haskey(local_of, p)]
  isempty(pao_real_local) || (support_atoms = sort!(union(support_atoms, pao_real_local)))
  # Global support for the AO/OPAO virtual targets: the real charge-based support, plus dummy
  # (ghost) atoms with sizable Löwdin population on the fragment occupied orbitals, plus any
  # explicitly requested PAO centers (which may be dummy atoms).
  iao_atoms_global = Int[real_globals[k] for k in iao_atoms]
  cMO_frag_occ = cMO_occ * R_occ[:, 1:nfrag_occ]
  ghost_support, ghost_pops = _collect_ghost_support(EC, cMO_frag_occ, S, ao_atoms_global, atom_charge_thr)
  support_global = sort!(union(Int[real_globals[k] for k in support_atoms], ghost_support, pao_centers))
  # accumulated charge on each support center (IAO charge for real atoms, Löwdin population for ghosts)
  real_acc = _accumulated_fragment_atom_charges(charges, nfrag_occ)
  support_charges = Float64[haskey(local_of, a) ? real_acc[local_of[a]] : get(ghost_pops, a, 0.0) for a in support_global]

  # Now move the fragment occupied orbitals to the TOP of the occupied block (just below
  # the Fermi level); the environment orbitals below them form a contiguous block that is
  # frozen as core in subsequent correlated calculations. charges columns follow along.
  occ_to_top = vcat(collect(nfrag_occ+1:size(R_occ, 2)), collect(1:nfrag_occ))
  R_occ = R_occ[:, occ_to_top]
  charges = charges[:, occ_to_top]

  cMO_region = copy(cMO_full)
  cMO_region[:, valence_occ_range] = cMO_occ * R_occ

  nfrag_virt = 0
  if !isempty(virt_range)
    cMO_virt = cMO_full[:, virt_range]
    if virtual_mode == :complement
      R_virt, nfrag_virt = _fragment_complement_virtual_rotation(cMO_virt, S, C_iao, iao_atoms_global, ao_atoms_global, support_global)
    else
      R_virt, nfrag_virt = _fragment_opao_rotation(cMO_virt, S, ao_atoms_global, support_global)
    end
    nfrag_virt > 0 || error("No $(spin_label)fragment virtual orbitals constructed; lower region.atom_charge_thr.")
    cMO_region[:, virt_range] = cMO_virt * R_virt
    if !isempty(energies)
      ε_virt = energies[virt_range]
      energies[virt_range] = real.(diag(R_virt' * Diagonal(ε_virt) * R_virt))
    end
  end

  if !isempty(energies)
    ε_occ = energies[valence_occ_range]
    energies[valence_occ_range] = real.(diag(R_occ' * Diagonal(ε_occ) * R_occ))
  end

  if pseudo
    isnothing(F_AO) && error("Missing Fock matrix for region pseudo-canonicalization.")
    fragment_occ_range = collect(valence_occ_range[end-nfrag_occ+1:end])
    _pseudocanonicalize_block!(cMO_region, energies, F_AO, fragment_occ_range)
    if nfrag_virt > 0
      fragment_virt_range = collect(virt_range[1:nfrag_virt])
      _pseudocanonicalize_block!(cMO_region, energies, F_AO, fragment_virt_range)
    end
  end

  classes = fill("Deleted", size(cMO_full, 2))
  classes[core_range] .= "Core"
  nocc = length(valence_occ_range)
  # Environment (non-selected) occupied orbitals sit below the region and are frozen
  # as core; the selected fragment orbitals are the active (Inactive) occupied block.
  classes[valence_occ_range[1:nocc-nfrag_occ]] .= "Core"
  classes[valence_occ_range[nocc-nfrag_occ+1:nocc]] .= "Inactive"
  if nfrag_virt > 0
    classes[virt_range[1:nfrag_virt]] .= "Virtual"
  end

  region_info = (charges=charges, support_atoms=support_global, support_charges=support_charges, scores=Float64[])
  return cMO_region, energies, classes, nfrag_occ, nfrag_virt, region_info
end

"""
    _build_region_spin_pi(cMO_full::AbstractMatrix, energies::Vector{Float64}, occupied_data,
                          S::AbstractMatrix, PZ_AO::AbstractMatrix, ao_atoms::Vector{Int},
                          selected_centers::Vector{Int}; ...)

  Build the π-selected region spaces for one spin block, ordering occupied and
  optionally virtual orbitals by their projected π-character or fragment-OPAO
  support.
"""
function _build_region_spin_pi(EC::ECInfo, cMO_full::AbstractMatrix{T}, energies::Vector{Float64},
                               occupied_data, S::AbstractMatrix, PZ_AO::AbstractMatrix,
                               ao_atoms_global::Vector{Int}, real_globals::Vector{Int},
                               selected_centers::Vector{Int};
                               pi_mode::Symbol, requested_occ::Int, requested_virt::Int,
                               spin_label::String="", atom_charge_thr::Float64=0.2,
                               pao_centers::Vector{Int}=Int[],
                               pseudo::Bool=false, F_AO=nothing) where T
  valence_occ_range = occupied_data.occ
  virt_range = occupied_data.virt
  core_range = occupied_data.core

  isempty(valence_occ_range) && error("No $(spin_label)occupied orbitals available for @region.")

  cMO_occ = cMO_full[:, valence_occ_range]
  requested_occ = clamp(requested_occ, 0, length(valence_occ_range))
  requested_occ > 0 || error("No $(spin_label)occupied π orbitals requested for the chosen centers.")
  R_occ, region_scores = _pi_projection_rotation(cMO_occ, PZ_AO, S)
  # Move the selected (highest π-character) occupied orbitals to the top of the
  # occupied block (just below the Fermi level); environment orbitals go below them.
  occ_to_top = vcat(collect(requested_occ+1:length(valence_occ_range)), collect(1:requested_occ))
  R_occ = R_occ[:, occ_to_top]
  isempty(region_scores) || (region_scores = region_scores[occ_to_top])

  cMO_region = copy(cMO_full)
  cMO_region[:, valence_occ_range] = cMO_occ * R_occ
  if !isempty(energies)
    ε_occ = energies[valence_occ_range]
    energies[valence_occ_range] = real.(diag(R_occ' * Diagonal(ε_occ) * R_occ))
  end

  nfrag_occ = requested_occ
  # dummy (ghost) atoms with sizable Löwdin population on the selected π occupied orbitals,
  # together with any explicitly requested PAO centers, extend the virtual-space support
  cMO_frag_occ = cMO_region[:, valence_occ_range[end-nfrag_occ+1:end]]
  ghost_support, _ = _collect_ghost_support(EC, cMO_frag_occ, S, ao_atoms_global, atom_charge_thr)
  extra_support = sort!(union(ghost_support, pao_centers))   # global; beyond the π centers themselves
  nfrag_virt = 0
  if !isempty(virt_range)
    cMO_virt = cMO_full[:, virt_range]
    if pi_mode == :both
      requested_virt = clamp(requested_virt, 0, length(virt_range))
      requested_virt > 0 || error("No $(spin_label)virtual π orbitals requested for the chosen centers.")
      R_pi, _ = _pi_projection_rotation(cMO_virt, PZ_AO, S)
      R_frag = R_pi[:, 1:requested_virt]
      if !isempty(extra_support)
        # augment the π virtual fragment with OPAOs on the extra (ghost + PAO) centers,
        # orthogonalized against the already-selected π virtuals
        C_pi = cMO_virt * R_frag
        C_ao_target = _support_ao_targets(cMO_virt, ao_atoms_global, extra_support)
        _, R_aug = _project_virtual_targets(cMO_virt, S, C_ao_target; existing=C_pi)
        R_frag = hcat(R_frag, R_aug)
      end
      nfrag_virt = size(R_frag, 2)
      R_virt = nfrag_virt < length(virt_range) ? hcat(R_frag, Matrix{T}(nullspace(adjoint(R_frag)))) : Matrix{T}(R_frag)
      _fix_sign_convention!(R_virt)
      cMO_region[:, virt_range] = cMO_virt * R_virt
      if !isempty(energies)
        ε_virt = energies[virt_range]
        energies[virt_range] = real.(diag(R_virt' * Diagonal(ε_virt) * R_virt))
      end
    else
      virt_centers = sort!(union(Int[real_globals[k] for k in selected_centers], extra_support))
      R_virt, nfrag_virt = _fragment_opao_rotation(cMO_virt, S, ao_atoms_global, virt_centers)
      nfrag_virt > 0 || error("No $(spin_label)fragment virtual orbitals constructed from the selected π-system atoms.")
      cMO_region[:, virt_range] = cMO_virt * R_virt
      if !isempty(energies)
        ε_virt = energies[virt_range]
        energies[virt_range] = real.(diag(R_virt' * Diagonal(ε_virt) * R_virt))
      end
    end
  end

  if pseudo
    isnothing(F_AO) && error("Missing Fock matrix for region pseudo-canonicalization.")
    fragment_occ_range = collect(valence_occ_range[end-nfrag_occ+1:end])
    _pseudocanonicalize_block!(cMO_region, energies, F_AO, fragment_occ_range)
    if nfrag_virt > 0
      fragment_virt_range = collect(virt_range[1:nfrag_virt])
      _pseudocanonicalize_block!(cMO_region, energies, F_AO, fragment_virt_range)
    end
  end

  classes = fill("Deleted", size(cMO_full, 2))
  classes[core_range] .= "Core"
  nocc = length(valence_occ_range)
  # Environment (non-selected) occupied orbitals sit below the region and are frozen
  # as core; the selected fragment orbitals are the active (Inactive) occupied block.
  classes[valence_occ_range[1:nocc-nfrag_occ]] .= "Core"
  classes[valence_occ_range[nocc-nfrag_occ+1:nocc]] .= "Inactive"
  if nfrag_virt > 0
    classes[virt_range[1:nfrag_virt]] .= "Virtual"
  end

  # :both virtuals come from the π projector and are augmented by the extra (ghost + PAO)
  # support; the OPAO (:occupied) virtuals are supported on the π centers plus the extra support.
  support_for_info = pi_mode == :both ? extra_support :
    sort!(union(Int[real_globals[k] for k in selected_centers], extra_support))
  # accumulated Löwdin population of each support center on the selected π occupied orbitals
  support_pops = _accumulated_lowdin_pops(cMO_frag_occ, S, ao_atoms_global, support_for_info)
  support_charges = Float64[get(support_pops, a, 0.0) for a in support_for_info]
  region_info = (charges=zeros(Float64, 0, 0), support_atoms=support_for_info,
                 support_charges=support_charges, scores=region_scores)
  return cMO_region, energies, classes, nfrag_occ, nfrag_virt, region_info
end

"""
    _dominant_charges_str(col, label_of; topn=3, thr=0.05)

  Format the largest atomic IAO partial charges of one occupied orbital as a compact
  `label:charge` list, keeping at most `topn` atoms with charge at least `thr`.
"""
function _dominant_charges_str(col, label_of; topn::Int=3, thr::Float64=0.05)
  isempty(col) && return "—"
  parts = String[]
  for a in sortperm(collect(col); rev=true)
    (length(parts) >= topn || col[a] < thr) && break
    push!(parts, @sprintf("%s:%.2f", label_of(a), col[a]))
  end
  return isempty(parts) ? "—" : join(parts, "  ")
end

"""
    _print_region_orbitals(space, energies, occupations, nfrag_occ, nfrag_virt, info,
                           virtual_mode, pi_mode, pseudo, real_atom_labels; spin_label="")

  Print a per-spin summary of the constructed region: the selected occupied orbitals with
  their energies, occupations and dominant IAO partial charges (plus the π-projection score
  in π modes), the region virtual orbitals, the virtual-space support/PAO centers, and a
  note when the fragment blocks were pseudo-canonicalized.
"""
function _print_region_orbitals(EC::ECInfo, space, energies, occupations, nfrag_occ, nfrag_virt, info,
                                virtual_mode, pi_mode, pseudo, real_atom_labels; spin_label::String="")
  nfrag_occ > 0 || return nothing
  occ_global = space.occ[end-nfrag_occ+1:end]
  has_scores = !isempty(info.scores)
  has_charges = size(info.charges, 1) > 0 && size(info.charges, 2) >= nfrag_occ
  getocc(i) = (i >= 1 && i <= length(occupations)) ? occupations[i] : 0.0
  label_of(a) = (a >= 1 && a <= length(real_atom_labels)) ? real_atom_labels[a] : "atom$a"

  println()
  println("  ", spin_label, "region occupied orbitals ($nfrag_occ):")
  header = has_scores ? "    orbital       energy    occ.   π-score   dominant IAO charges" :
                        "    orbital       energy    occ.   dominant IAO charges"
  println(header)
  println("    ", repeat("─", length(header) - 4))
  for k in 1:nfrag_occ
    iorb = occ_global[k]
    chg = has_charges ? _dominant_charges_str(view(info.charges, :, size(info.charges, 2) - nfrag_occ + k), label_of) : "—"
    if has_scores
      sc = info.scores[length(info.scores) - nfrag_occ + k]
      @printf("    %5i  %12.6f  %5.2f  %8.4f   %s\n", iorb, energies[iorb], getocc(iorb), sc, chg)
    else
      @printf("    %5i  %12.6f  %5.2f   %s\n", iorb, energies[iorb], getocc(iorb), chg)
    end
  end

  if nfrag_virt > 0
    vmode = pi_mode == :both ? "π" : (pi_mode != :none ? "support-OPAO" : String(virtual_mode))
    println()
    println("  ", spin_label, "region virtual orbitals ($nfrag_virt, $vmode):")
    println("    orbital       energy")
    println("    ", repeat("─", 22))
    for iorb in space.virt[1:nfrag_virt]
      @printf("    %5i  %12.6f\n", iorb, energies[iorb])
    end
  end

  if !isempty(info.support_atoms)
    # support atoms are global indices (may include dummy/ghost atoms); show the accumulated
    # charge/population that each center carries on the fragment occupied orbitals
    charges = get(info, :support_charges, Float64[])
    entries = String[]
    for (i, a) in enumerate(info.support_atoms)
      lab = atomic_centre_label(EC.system[a])
      push!(entries, i <= length(charges) ? @sprintf("%s:%.2f", lab, charges[i]) : lab)
    end
    println()
    println("  ", spin_label, "virtual-space support / PAO centers (accumulated charge): ", join(entries, "  "))
  end
  pseudo && println("  ", spin_label, "fragment occupied/virtual blocks were pseudo-canonicalized")
  return nothing
end

"""
    region_orbitals(EC::ECInfo, centers)

Construct a region-tagged orbital dump from localized occupied orbitals and fragment OPAOs.

The occupied orbitals are localized with the IBO criterion, selected according to
`EC.options.region`, and written with explicit TREXIO classes:
- fragment occupied orbitals → `Inactive`
- fragment virtual orbitals → `Virtual`
- frozen occupied orbitals and the non-selected environment occupied orbitals → `Core`
- all remaining complements → `Deleted`

The fragment (`Inactive`/`Virtual`) orbitals are placed at the Fermi level, so that the
`Core` block sits contiguously below them and the `Deleted` virtuals above them. With the
default `wf.core = :auto` (and `wf.freeze_nvirt = -1`), a subsequent correlated calculation
honors these classes automatically (see [`freeze_orbitals!`](@ref)).

For unrestricted references, alpha and beta spaces are treated independently.
"""
function region_orbitals(EC::ECInfo, centers)
  mode = EC.options.region.mode
  mode in (:inclusive, :exclusive) || error("Unknown region.mode = $mode. Valid modes are :inclusive and :exclusive.")
  configured_inclusive = EC.options.region.inclusive_centers
  configured_exclusive = EC.options.region.exclusive_centers
  virtual_mode = EC.options.region.virtual
  pi_mode = EC.options.region.pi
  pi_mode in (:none, :occupied, :both) || error("Unknown region.pi = $pi_mode. Valid modes are :none, :occupied, and :both.")
  pi_electrons = EC.options.region.pi_electrons
  pi_occupied = EC.options.region.pi_occupied
  pi_virtual = EC.options.region.pi_virtual
  occ_charge_thr = EC.options.region.occ_charge_thr
  atom_charge_thr = EC.options.region.atom_charge_thr
  pseudo = EC.options.region.pseudo
  occ_charge_thr >= 0 || error("region.occ_charge_thr must be non-negative.")
  atom_charge_thr >= 0 || error("region.atom_charge_thr must be non-negative.")

  selection = _resolve_region_selection(EC, centers, mode, configured_inclusive, configured_exclusive)
  selected_global = selection.all_global
  selected_local = selection.all_local
  # extra atom centers whose PAOs are added to the fragment virtual space (global indices,
  # which may include dummy/ghost atoms)
  pao_centers, _ = _resolve_region_centers(EC, EC.options.region.pao_centers; allow_empty=true)
  inclusive_labels = [atomic_centre_label(EC.system[idx]) for idx in selection.inclusive_global]
  exclusive_labels = [atomic_centre_label(EC.system[idx]) for idx in selection.exclusive_global]
  if !isempty(inclusive_labels) && !isempty(exclusive_labels)
    println("Building region-tagged orbitals for inclusive centers: ", join(inclusive_labels, ", "))
    println("  Exclusive centers: ", join(exclusive_labels, ", "))
  elseif !isempty(inclusive_labels)
    println("Building region-tagged orbitals for inclusive centers: ", join(inclusive_labels, ", "))
  else
    println("Building region-tagged orbitals for exclusive centers: ", join(exclusive_labels, ", "))
  end

  use_start = EC.options.wf.start != ""
  cMO, _, basis = fetch_orbitals(EC; start=use_start)
  isempty(basis) && error("@region requires orbitals with basis information; rotations alone are insufficient.")

  basis_ao = generate_basis(EC, "ao")
  cMO = project_onto_basis(cMO, basis, basis_ao; check=true)
  energies = fetch_orbital_energies(EC; start=use_start)
  occupations = fetch_orbital_occupations(EC; start=use_start)
  restricted = is_restricted(cMO)

  exponent = _region_exponent(EC)
  S = overlap(basis_ao)
  # global atom index per AO (including dummy/ghost atoms) and the local→global map for the
  # non-dummy atoms used by the IAO partial charges
  ao_atoms_global = Int[Int(ao.icentre) for ao in ao_list(basis_ao)]
  real_globals = sort!(unique(Int[a for a in ao_atoms_global if !is_dummy(EC.system[a])]))
  alpha_space, beta_space = _region_space_ranges(EC, restricted)
  requested_occ_a = 0
  requested_virt_a = 0
  requested_occ_b = 0
  requested_virt_b = 0
  if pi_mode != :none
    requested_occ_a, requested_virt_a, requested_occ_b, requested_virt_b = _estimate_pi_requested_counts(
      EC, selected_global, length(alpha_space.occ), restricted ? 0 : length(beta_space.occ), restricted;
      pi_electrons=pi_electrons, pi_occupied=pi_occupied, pi_virtual=pi_virtual)
  end

  F_AO_a = nothing
  F_AO_b = nothing
  if pseudo
    F_AO_a = _reconstruct_fock_matrix(cMO[1], S, energies[1])
    if !restricted
      F_AO_b = _reconstruct_fock_matrix(cMO[2], S, energies[2])
    end
  end

  # labels of the non-dummy atoms, in the numbering used by the IAO partial charges
  real_atom_labels = String[atomic_centre_label(atom) for atom in EC.system if !is_dummy(atom)]

  if restricted
    cMO_alpha = cMO[1]
    energies_a = copy(energies[1])
    if pi_mode == :none
      C_iao, iao_atoms, natom = compute_iaos(EC, cMO_alpha[:, alpha_space.all_occ])
      cMO_region_a, energies_region_a, classa, nfrag_occ_a, nfrag_virt_a, info_a = _build_region_spin(EC,
        cMO_alpha, energies_a, alpha_space, S, C_iao, iao_atoms, natom, ao_atoms_global, real_globals,
        selection.inclusive_local, selection.exclusive_local;
        virtual_mode=virtual_mode,
        occ_charge_thr=occ_charge_thr, atom_charge_thr=atom_charge_thr,
        exponent=exponent, pao_centers=pao_centers, pseudo=pseudo, F_AO=F_AO_a)
      region_type = isempty(alpha_space.virt) ? "Region-IBO" : "Region-IBO+OPAO"
      pseudo && (region_type *= "-Pseudo")
    else
      C_iao_raw = _compute_raw_iaos(EC, cMO_alpha[:, alpha_space.all_occ])
      PZ_AO = _build_pi_target_orbitals(EC, C_iao_raw, selected_global)
      cMO_region_a, energies_region_a, classa, nfrag_occ_a, nfrag_virt_a, info_a = _build_region_spin_pi(EC,
        cMO_alpha, energies_a, alpha_space, S, PZ_AO, ao_atoms_global, real_globals, selected_local;
        pi_mode=pi_mode, requested_occ=requested_occ_a, requested_virt=requested_virt_a,
        atom_charge_thr=atom_charge_thr, pao_centers=pao_centers, pseudo=pseudo, F_AO=F_AO_a)
      if pi_mode == :both
        region_type = "Region-PiOS"
      elseif isempty(alpha_space.virt)
        region_type = "Region-PiOcc"
      else
        region_type = "Region-PiOcc+OPAO"
      end
      pseudo && pi_mode == :occupied && (region_type *= "-Pseudo")
    end
    println()
    println("Region orbital construction: ", region_type)
    _print_region_orbitals(EC, alpha_space, energies_region_a, occupations[1], nfrag_occ_a, nfrag_virt_a,
      info_a, virtual_mode, pi_mode, pseudo, real_atom_labels)
    dump_orbitals(EC, SpinMatrix(cMO_region_a);
      basis=basis_ao,
      type=region_type,
      energies=(energies_region_a, energies[2]),
      occupations=occupations,
      classes=(classa, String[]))

    println()
    println("  Region dump written: $(nfrag_occ_a) active occupied / $(nfrag_virt_a) active virtual orbital(s); ",
            "$(length(alpha_space.occ) - nfrag_occ_a) occupied frozen as core")
  else
    cMO_alpha = cMO[1]
    cMO_beta = cMO[2]
    energies_a = copy(energies[1])
    energies_b = copy(energies[2])

    if pi_mode == :none
      C_iao_a, iao_atoms_a, natom_a = compute_iaos(EC, cMO_alpha[:, alpha_space.all_occ])
      C_iao_b, iao_atoms_b, natom_b = compute_iaos(EC, cMO_beta[:, beta_space.all_occ])
      cMO_region_a, energies_region_a, classa, nfrag_occ_a, nfrag_virt_a, info_a = _build_region_spin(EC,
        cMO_alpha, energies_a, alpha_space, S, C_iao_a, iao_atoms_a, natom_a, ao_atoms_global, real_globals,
        selection.inclusive_local, selection.exclusive_local;
        virtual_mode=virtual_mode,
        occ_charge_thr=occ_charge_thr, atom_charge_thr=atom_charge_thr,
        exponent=exponent, spin_label="alpha ", pao_centers=pao_centers, pseudo=pseudo, F_AO=F_AO_a)
      cMO_region_b, energies_region_b, classb, nfrag_occ_b, nfrag_virt_b, info_b = _build_region_spin(EC,
        cMO_beta, energies_b, beta_space, S, C_iao_b, iao_atoms_b, natom_b, ao_atoms_global, real_globals,
        selection.inclusive_local, selection.exclusive_local;
        virtual_mode=virtual_mode,
        occ_charge_thr=occ_charge_thr, atom_charge_thr=atom_charge_thr,
        exponent=exponent, spin_label="beta ", pao_centers=pao_centers, pseudo=pseudo, F_AO=F_AO_b)
      region_type = (!isempty(alpha_space.virt) || !isempty(beta_space.virt)) ? "Region-IBO+OPAO" : "Region-IBO"
      pseudo && (region_type *= "-Pseudo")
    else
      C_iao_raw_a = _compute_raw_iaos(EC, cMO_alpha[:, alpha_space.all_occ])
      C_iao_raw_b = _compute_raw_iaos(EC, cMO_beta[:, beta_space.all_occ])
      PZ_AO_a = _build_pi_target_orbitals(EC, C_iao_raw_a, selected_global)
      PZ_AO_b = _build_pi_target_orbitals(EC, C_iao_raw_b, selected_global)
      cMO_region_a, energies_region_a, classa, nfrag_occ_a, nfrag_virt_a, info_a = _build_region_spin_pi(EC,
        cMO_alpha, energies_a, alpha_space, S, PZ_AO_a, ao_atoms_global, real_globals, selected_local;
        pi_mode=pi_mode, requested_occ=requested_occ_a, requested_virt=requested_virt_a,
        spin_label="alpha ", atom_charge_thr=atom_charge_thr, pao_centers=pao_centers, pseudo=pseudo, F_AO=F_AO_a)
      cMO_region_b, energies_region_b, classb, nfrag_occ_b, nfrag_virt_b, info_b = _build_region_spin_pi(EC,
        cMO_beta, energies_b, beta_space, S, PZ_AO_b, ao_atoms_global, real_globals, selected_local;
        pi_mode=pi_mode, requested_occ=requested_occ_b, requested_virt=requested_virt_b,
        spin_label="beta ", atom_charge_thr=atom_charge_thr, pao_centers=pao_centers, pseudo=pseudo, F_AO=F_AO_b)
      if pi_mode == :both
        region_type = "Region-PiOS"
      elseif !isempty(alpha_space.virt) || !isempty(beta_space.virt)
        region_type = "Region-PiOcc+OPAO"
      else
        region_type = "Region-PiOcc"
      end
      pseudo && pi_mode == :occupied && (region_type *= "-Pseudo")
    end
    println()
    println("Region orbital construction: ", region_type)
    _print_region_orbitals(EC, alpha_space, energies_region_a, occupations[1], nfrag_occ_a, nfrag_virt_a,
      info_a, virtual_mode, pi_mode, pseudo, real_atom_labels; spin_label="α ")
    _print_region_orbitals(EC, beta_space, energies_region_b, occupations[2], nfrag_occ_b, nfrag_virt_b,
      info_b, virtual_mode, pi_mode, pseudo, real_atom_labels; spin_label="β ")
    dump_orbitals(EC, SpinMatrix(cMO_region_a, cMO_region_b);
      basis=basis_ao,
      type=region_type,
      energies=(energies_region_a, energies_region_b),
      occupations=occupations,
      classes=(classa, classb))

    println()
    println("  Region dump written: α $(nfrag_occ_a) occ / $(nfrag_virt_a) virt, β $(nfrag_occ_b) occ / $(nfrag_virt_b) virt; ",
            "frozen as core: α $(length(alpha_space.occ) - nfrag_occ_a), β $(length(beta_space.occ) - nfrag_occ_b)")
  end

  return nothing
end

end # module