"""
    Properties

  Shared property post-processing helpers for SCF, correlated, and CI methods.
"""
module Properties
using LinearAlgebra

using ..ElemCo.ECInfos
using ..ElemCo.Utils: OutDict
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets
using ..ElemCo.Integrals
using ..ElemCo.MSystems
using ..ElemCo.Constants: AU2DEBYE

export add_dipole_entries, calc_dipole_moment, natural_orbital_rotation

"""
    _dipole_output_pairs(method::String, dipole; include_method_aliases=false)

  Build the standard dipole output key/value/description tuples for insertion
  into an `OutDict`, optionally including method-qualified aliases.
"""
function _dipole_output_pairs(method::String, dipole; include_method_aliases::Bool=false)
  base_pairs = (
    "DMX" => (dipole[1], "$method dipole x component [Debye]"),
    "DMY" => (dipole[2], "$method dipole y component [Debye]"),
    "DMZ" => (dipole[3], "$method dipole z component [Debye]"),
    "DM" => (dipole[4], "$method dipole magnitude [Debye]"),
    "mux" => (dipole[1], "$method dipole x component [Debye]"),
    "muy" => (dipole[2], "$method dipole y component [Debye]"),
    "muz" => (dipole[3], "$method dipole z component [Debye]"),
    "mu" => (dipole[4], "$method dipole magnitude [Debye]")
  )
  if !include_method_aliases
    return base_pairs
  end
  return (
    method*"-DMX" => (dipole[1], "$method dipole x component [Debye]"),
    method*"-DMY" => (dipole[2], "$method dipole y component [Debye]"),
    method*"-DMZ" => (dipole[3], "$method dipole z component [Debye]"),
    method*"-DM" => (dipole[4], "$method dipole magnitude [Debye]"),
    method*"-mux" => (dipole[1], "$method dipole x component [Debye]"),
    method*"-muy" => (dipole[2], "$method dipole y component [Debye]"),
    method*"-muz" => (dipole[3], "$method dipole z component [Debye]"),
    method*"-mu" => (dipole[4], "$method dipole magnitude [Debye]"),
    base_pairs...
  )
end

"""
    _insert_before_last(energies::OutDict, pairs...)

  Insert additional output entries immediately before the final element of an
  `OutDict`, preserving the convention that the last entry remains the primary
  energy.
"""
function _insert_before_last(energies::OutDict, pairs...)
  isempty(energies) && return OutDict(pairs...)

  energies_out = copy(energies)
  last_key, last_value, last_description = last(energies_out)
  delete!(energies_out, last_key)
  push!(energies_out, pairs...)
  push!(energies_out, last_key, last_value, last_description)
  return energies_out
end

"""
    real_part_if_small(EC::ECInfo, value, label::AbstractString)

  Return the real part of `value` when its imaginary contribution is below the
  configured SCF tolerance, otherwise emit a warning before dropping it.
"""
function real_part_if_small(EC::ECInfo, value, label::AbstractString)
  eltype(value) <: Real && return value
  value_real = real.(value)
  diff = sum(abs2, value) - sum(abs2, value_real)
  if diff > EC.options.scf.imagtol
    println("Large imaginary part in $label neglected!")
    println("Difference between squared norms:", diff)
  end
  return value_real
end

"""
    nuclear_dipole_components(system::MSystem)

  Compute the Cartesian nuclear dipole contribution in atomic units from the
  molecular geometry stored in `system`.
"""
function nuclear_dipole_components(system::MSystem)
  dip = zeros(Float64, 3)
  for atom in system
    dip .+= Float64(atom.atomic_number) .* Vector(atom.position)
  end
  return dip
end

"""
    dipole_mo_components(cMO::AbstractMatrix, Dx::AbstractMatrix,
                         Dy::AbstractMatrix, Dz::AbstractMatrix)

  Transform AO dipole integrals to the MO basis defined by `cMO`.
"""
function dipole_mo_components(cMO::AbstractMatrix, Dx::AbstractMatrix,
                              Dy::AbstractMatrix, Dz::AbstractMatrix)
  return (cMO' * Dx * cMO, cMO' * Dy * cMO, cMO' * Dz * cMO)
end

"""
    natural_orbital_rotation(EC::ECInfo, rdm)

  Diagonalize a MO-space 1-RDM and return the descending occupations together
  with the rotation matrix that transforms the current orbitals into natural
  orbitals. For unrestricted densities, alpha and beta blocks are treated
  independently.
"""
function natural_orbital_rotation(EC::ECInfo, rdm::AbstractMatrix)
  rdmh = 0.5 * (rdm + rdm')
  rdmr = real_part_if_small(EC, rdmh, "1-RDM")
  eig = eigen(Hermitian(rdmr))
  perm = sortperm(eig.values; rev=true)
  occ = Vector{Float64}(eig.values[perm])
  rot = Matrix{Float64}(eig.vectors[:, perm])
  return occ, rot
end

function natural_orbital_rotation(EC::ECInfo, rdm::SpinMatrix)
  if is_restricted(rdm)
    occ, rot = natural_orbital_rotation(EC, rdm.α)
    return SpinMatrix(rot), (occ, Float64[])
  end
  occa, rota = natural_orbital_rotation(EC, rdm.α)
  occb, rotb = natural_orbital_rotation(EC, rdm.β)
  return SpinMatrix(rota, rotb), (occa, occb)
end

"""
    add_dipole_entries(energies::OutDict, method::String, dipole; include_method_aliases=false)

  Merge dipole components and magnitude into an output dictionary using the
  standard `DM*` and `mu*` keys. When `include_method_aliases=true`, also add
  method-qualified aliases such as `METHOD-DMX` and `METHOD-mu`.
"""
function add_dipole_entries(energies::OutDict, method::String, dipole; include_method_aliases=false)
  return _insert_before_last(energies,
    _dipole_output_pairs(method, dipole; include_method_aliases=include_method_aliases)...)
end

"""
    calc_dipole_moment(EC::ECInfo, cMO::SpinMatrix, rdm::SpinMatrix; basis=nothing)

  Calculate total molecular dipole components in Debye from a MO-space 1-RDM.
  Returns `(μx, μy, μz, |μ|)` or `nothing` when the AO basis / geometry is unavailable.
"""
function calc_dipole_moment(EC::ECInfo, cMO::SpinMatrix, rdm::SpinMatrix; basis=nothing)
  if isempty(EC.system)
    return nothing
  end
  if isnothing(basis)
    basis = generate_basis(EC, "ao")
  elseif isempty(basis)
    return nothing
  end
  Dx, Dy, Dz = dipole(basis)
  μnuc = nuclear_dipole_components(EC.system)
  if is_restricted(rdm)
    dipm = dipole_mo_components(cMO.α, Dx, Dy, Dz)
    μel = [tr(rdm.α * dipm[i]) for i in 1:3]
  else
    dipma = dipole_mo_components(cMO.α, Dx, Dy, Dz)
    dipmb = dipole_mo_components(cMO.β, Dx, Dy, Dz)
    μel = [tr(rdm.α * dipma[i]) + tr(rdm.β * dipmb[i]) for i in 1:3]
  end
  μel = real_part_if_small(EC, μel, "dipole moment")
  μtot = AU2DEBYE .* (μnuc .- μel)
  return (μtot[1], μtot[2], μtot[3], norm(μtot))
end

end # module