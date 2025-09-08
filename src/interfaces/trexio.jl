"""
TREXIO Interface Module for ElemCo.jl

This module provides ElemCo-specific functions to import and export data in TREXIO format,
which is a standardized format for quantum chemistry data exchange.
It uses the standalone TREXIO module for core operations and provides convenient
conversion between ElemCo types and TREXIO standard formats.

TREXIO format specification:
- Based on HDF5 for efficient storage
- Standardized structure for quantum chemistry data
- Supports orbitals, amplitudes, integrals, and other QC data

See: https://trexio-coe.github.io/trexio/lib.html
     https://arxiv.org/abs/2302.14793
"""
module TrexioInterface

using ..ElemCo.MSystems
using ..ElemCo.Utils
using ..ElemCo.QMTensors
using ..ElemCo.BasisSets
using ..ElemCo.TREXIO  # Use the standalone TREXIO module
using LinearAlgebra

# Re-export the core TREXIO types and functions for backward compatibility
export TrexioFile
export open_trexio, close_trexio
export write_trexio_system, read_trexio_system
export write_trexio_basis, read_trexio_basis
export write_trexio_orbitals, read_trexio_orbitals
export read_trexio_orbital_classes, read_trexio_orbital_energies, read_trexio_orbital_occupations
export write_trexio_amplitudes
export read_trexio_singles, read_trexio_doubles
export read_trexio_unrestricted_singles, read_trexio_unrestricted_doubles

# Re-export the standalone TREXIO types for compatibility
const TrexioFile = TREXIO.TrexioFile

"""
    TREXIO2LIBCINT_PERMUTATION

  Permutation of the atomic orbitals from the TREXIO to the libcint order.
"""
const TREXIO2LIBCINT_PERMUTATION = [        # TREXIO order:
  [1],                                      # s
  [2,3,1],                                  # p 0, +1, -1 (z,x,y)
  [5,3,1,2,4],                              # d 0, +1, -1 , +2, -2 (z^2, xz, yz, x^2-y^2, xy)
  [7,5,3,1,2,4,6],                          # f 0, +1, -1, +2, -2, +3, -3
  [9,7,5,3,1,2,4,6,8],                      # g 0, +1, -1, +2, -2, +3, -3, +4, -4
  [11,9,7,5,3,1,2,4,6,8,10],                # h 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5
  [13,11,9,7,5,3,1,2,4,6,8,10,12]           # i 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5, +6, -6
      ]

"""
    TREXIO2LIBCINT_PERMUTATION_CART

  Permutation of the atomic orbitals from the TREXIO to the libcint order for cartesian basis sets.
"""
const TREXIO2LIBCINT_PERMUTATION_CART = [   # TREXIO order (lexicographical, same as libcint)
  [1],                                      # s
  [1:3;],                                   # p x,y,z
  [1:6;],                                   # d xx,xy,xz,yy,yz,zz
  [1:10;],                                  # f xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz
  [1:15;],                                  # g
  [1:21;],                                  # h
  [1:28;]                                   # i
      ]

"""
    order4l(basis::BasisSet) -> Vector{Int}

  Return order for each l from TREXIO order to libcint order.
"""
order4l(basis) = is_cartesian(basis) ? TREXIO2LIBCINT_PERMUTATION_CART : TREXIO2LIBCINT_PERMUTATION
"""
    open_trexio(filename::String, mode::String="r") -> TrexioFile

Open a TREXIO file. Returns an opened TrexioFile object.
Compatible wrapper around the standard TREXIO API.

# Arguments
- `filename::String`: Path to the TREXIO file
- `mode::String`: Access mode ("r" for read, "w" for write, "u" for read-write)
"""
function open_trexio(filename::String, mode::String="r")
  trexio = TREXIO.trexio_open(filename, mode)
  if isnothing(trexio)
    error("Failed to open TREXIO file: $filename")
  end
  return trexio
end

function open_trexio(f::Function, filename::String, mode::String="r")
  trexio = open_trexio(filename, mode)
  try
    f(trexio)
  finally
    close_trexio(trexio)
  end
end

"""
    close_trexio(trexio::TrexioFile)

Close a TREXIO file and release resources.
Compatible wrapper around the standard TREXIO API.
"""
function close_trexio(trexio::TrexioFile)
  status = TREXIO.trexio_close(trexio)
  if status != TREXIO.TREXIO_SUCCESS
    @warn "Warning: Failed to properly close TREXIO file"
  end
end

"""
    write_trexio_system(trexio::TrexioFile, system::MSystem)

Write molecular geometry and basis set information to TREXIO format using ElemCo data structures.
"""
function write_trexio_system(trexio::TrexioFile, system::MSystem)
  # Convert ElemCo MSystem to TREXIO format
  natoms = length(system)
  nuclear_charges = Float64[]
  coordinates = zeros(Float64, 3, natoms)
  labels = String[]
  
  for (i, atom) in enumerate(system)
    push!(nuclear_charges, Float64(atom.atomic_number))
    coordinates[:, i] = Vector(atom.position)
    push!(labels, atom.label)
  end
  
  # Use the standalone TREXIO module to write the data
  trexio_write_nucleus_num(trexio, natoms)
  trexio_write_nucleus_charge(trexio, nuclear_charges)
  trexio_write_nucleus_coord(trexio, coordinates)
  trexio_write_nucleus_label(trexio, labels)
  trexio_write_nucleus_point_group(trexio, "C1") # we don't have point group info
  trexio_write_nucleus_repulsion(trexio, nuclear_repulsion(system)) # no repulsion energy

  # write_trexio_basis(trexio, generate_basis(system, "ao"))
end

"""
    read_trexio_system(trexio::TrexioFile) -> MSystem

Read molecular geometry from TREXIO format and return ElemCo MSystem.
"""
function read_trexio_system(trexio::TrexioFile)
  # Read data using standalone TREXIO module
  natoms, status = trexio_read_nucleus_num(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
      error("No nucleus data found in TREXIO file")
  end
  nuclear_charges, status = trexio_read_nucleus_charge(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read nuclear charges from TREXIO file"
  coordinates, status = trexio_read_nucleus_coord(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read nuclear coordinates from TREXIO file"
  labels, status = trexio_read_nucleus_label(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read nuclear labels from TREXIO file"
  sym, status = trexio_read_nucleus_point_group(trexio)
  if status == TREXIO.TREXIO_SUCCESS && sym != "C1"
    error("Point group symmetry not supported: $sym")
  end
  
  # Convert to ElemCo MSystem
  atoms = ACentre[]
  for (i, charge) in enumerate(nuclear_charges)
    # Convert charge to element symbol
    pos = coordinates[:, i]
    
    # Create basic basis (this would need to be enhanced for real use)
    basis = Dict{String,String}()

    atom = ACentre(labels[i], pos[1], pos[2], pos[3], charge, basis)
    push!(atoms, atom)
  end
  
  return MSystem(atoms)
end

"""
    write_trexio_basis(trexio::TrexioFile, basis::BasisSet)

Write basis set information to TREXIO format following TREXIO standard.
"""
function write_trexio_basis(trexio::TrexioFile, basis::BasisSet)
  nsh = n_subshells(basis)

  trexio_write_basis_type(trexio, "Gaussian")
  nucleus_index = Int[]
  shell_ang_mom = Int[]
  ish = 0
  shell_index = Int[]
  exponent = Float64[]
  coefficient = Float64[]
  prim_factor = Float64[]
  basisname = ""
  for (i, centre) in enumerate(basis.centres)
    append!(nucleus_index, fill(i-1, n_subshells(centre)))
    if basisname == ""
      basisname = centre.name
    elseif basisname != centre.name
      basisname = "Mixed"
    end
    for ash in angularshells(centre)
      append!(shell_ang_mom, fill(ash.l, n_subshells(ash)))
      for bc in ash.subshells
        normalized_contraction = normalize_contraction(bc, ash, basis.cartesian)
        ic = 1
        for prim in bc.exprange
          push!(shell_index, ish)
          push!(exponent, ash.exponents[prim])
          push!(coefficient, bc.coefs[ic])
          push!(prim_factor, normalized_contraction[ic]/bc.coefs[ic])
          ic += 1
        end
        ish += 1
      end
    end
  end
  status = trexio_write_basis_prim_num(trexio, length(exponent))
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write prim_num to TREXIO with status $status"
  status = trexio_write_basis_shell_num(trexio, nsh)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write shell_num to TREXIO with status $status"
  status = trexio_write_basis_nucleus_index(trexio, nucleus_index)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write nucleus_index to TREXIO with status $status"
  status = trexio_write_basis_shell_ang_mom(trexio, shell_ang_mom)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write shell_ang_mom to TREXIO with status $status"
  status = trexio_write_basis_shell_factor(trexio, fill(1.0, nsh))  # no normalization
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write shell_factor to TREXIO with status $status"
  status = trexio_write_basis_shell_index(trexio, shell_index)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write shell_index to TREXIO with status $status"
  status = trexio_write_basis_exponent(trexio, exponent)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write exponent to TREXIO with status $status"
  status = trexio_write_basis_coefficient(trexio, coefficient)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write coefficient to TREXIO with status $status"
  status = trexio_write_basis_prim_factor(trexio, prim_factor)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write prim_factor to TREXIO with status $status"
  status = trexio_write_basis_name(trexio, basisname)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write basisname to TREXIO with status $status"
  
  # write AO information
  aolist = ao_list(basis)
  status = trexio_write_ao_cartesian(trexio, basis.cartesian ? 1 : 0)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write ao_cartesian to TREXIO with status $status"
  status = trexio_write_ao_num(trexio, length(aolist))
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write ao_num to TREXIO with status $status"
  shell = Int[]
  ishell = -1
  iash = issh = 0
  for ao in aolist
    if ao.iangularshell != iash || ao.isubshell != issh
      ishell += 1
      iash = ao.iangularshell
      issh = ao.isubshell
    end
    push!(shell, ishell)
  end
  status = trexio_write_ao_shell(trexio, shell)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write ao_shell to TREXIO with status $status"
  status = trexio_write_ao_normalization(trexio, fill(1.0, length(aolist)))
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write ao_normalization to TREXIO with status $status"
end

"""
    read_trexio_basis(trexio::TrexioFile) -> BasisSet

Read basis set information from TREXIO file.
"""
function read_trexio_basis(trexio::TrexioFile)
  system = read_trexio_system(trexio)

  type, status = trexio_read_basis_type(trexio)
  if type != "Gaussian"
    error("Unsupported basis set type: $type")
  end

  prim_num, status = trexio_read_basis_prim_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read prim_num from TREXIO with status $status"
  shell_num, status = trexio_read_basis_shell_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read shell_num from TREXIO with status $status"
  nucleus_index, status = trexio_read_basis_nucleus_index(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read nucleus_index from TREXIO with status $status"
  shell_ang_mom, status = trexio_read_basis_shell_ang_mom(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read shell_ang_mom from TREXIO with status $status"
  shell_factor, status = trexio_read_basis_shell_factor(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read shell_factor from TREXIO with status $status"
  shell_index, status = trexio_read_basis_shell_index(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read shell_index from TREXIO with status $status"
  exponent, status = trexio_read_basis_exponent(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read exponent from TREXIO with status $status"
  coefficient, status = trexio_read_basis_coefficient(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read coefficient from TREXIO with status $status"
  prim_factor, status = trexio_read_basis_prim_factor(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read prim_factor from TREXIO with status $status"
  basisname, status = trexio_read_basis_name(trexio)
  if status != TREXIO.TREXIO_SUCCESS
    basisname = "Unknown"
  end

  @assert length(nucleus_index) == length(shell_ang_mom) == length(shell_factor) == shell_num "Mismatch in shell_num lengths"
  @assert length(shell_index) == length(exponent) == length(coefficient) == length(prim_factor) == prim_num "Mismatch in prim_num lengths"

  array_of_centres = BasisCentre[]
  id = 1
  iat = 0
  sh_start = 1
  pr_start = 1
  for atom in system
    element = element_label(atom)
    basisfunctions = AngularShell[]
    for ish in sh_start:shell_num
      if nucleus_index[ish] != iat
        sh_start = ish
        break  # Move to the next atom
      end
      l = shell_ang_mom[ish]
      # get range of primitives for this shell
      pr_end = pr_start - 1
      for ipr in pr_start:prim_num
        if shell_index[ipr] == ish - 1
          pr_end = ipr
        else
          break
        end
      end
      if length(basisfunctions) == 0 || basisfunctions[end].l != l ||
          length(basisfunctions[end].exponents) != pr_end - pr_start + 1 ||
          !isapprox(basisfunctions[end].exponents, exponent[pr_start:pr_end])
        # new angular shell with new exponents
        push!(basisfunctions, generate_angularshell(element, l, exponent[pr_start:pr_end]))
      end
      add_subshell!(basisfunctions[end], 1:(pr_end-pr_start+1), coefficient[pr_start:pr_end])
      pr_start = pr_end + 1
    end
    push!(array_of_centres, BasisCentre(atom, basisname, basisfunctions))
    iat += 1
  end
  return BasisSet(array_of_centres)
end

"""
    write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix, basis::BasisSet;
                          type="HF", classes=(String[], String[]),
                          energies=(Float64[], Float64[]), occupations=(Float64[], Float64[]))

  Write molecular orbitals to TREXIO file. 

`classes`, `energies`, and `occupations` are optional and can be provided as tuples for alpha and beta spins.
`classes` entries can be "Core", "Inactive", "Active", "Virtual", "Deleted"
"""
function write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix, basis::BasisSet;
                               type="HF", classes=(String[], String[]),
                               energies=(Float64[], Float64[]), occupations=(Float64[], Float64[]))
  write_trexio_basis(trexio, basis)
  # Convert ElemCo orbital format to standard matrix format
  nbasis, nmo = size(orbitals)
  nao = n_ao(basis)
  @assert nao == nbasis "Basis size mismatch: basis has $nao, orbitals have $nbasis"
  order = ao_order2internal(basis, order4l(basis), true)
  status = trexio_write_mo_type(trexio, type)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_type to TREXIO with status $status"
  if is_restricted(orbitals)
    status = trexio_write_mo_num(trexio, nmo)
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_num to TREXIO with status $status"
    status = trexio_write_mo_coefficient(trexio, orbitals[1][order, :])
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_coefficient to TREXIO with status $status"
    if length(classes[1]) > 0
      status = trexio_write_mo_class(trexio, classes[1])
      @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_class to TREXIO with status $status"
    end
    if length(energies[1]) > 0
      status = trexio_write_mo_energy(trexio, energies[1])
      @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_energy to TREXIO with status $status"
    end
    if length(occupations[1]) > 0
      status = trexio_write_mo_occupation(trexio, occupations[1])
      @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_occupation to TREXIO with status $status"
    end
  else
    status = trexio_write_mo_num(trexio, 2*nmo) # For unrestricted, double the number of orbitals
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_num to TREXIO with status $status"
    status = trexio_write_mo_coefficient(trexio, hcat(orbitals...)[order, :])
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_coefficient to TREXIO with status $status"
    status = trexio_write_mo_spin(trexio, vcat(fill(0, nmo), fill(1, nmo)))  # α=0, β=1
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_spin to TREXIO with status $status"
    if length(classes[1]) > 0
      status = trexio_write_mo_class(trexio, vcat(classes...))
      @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_class to TREXIO with status $status"
    end
    if length(energies[1]) > 0
      status = trexio_write_mo_energy(trexio, vcat(energies...))
      @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_energy to TREXIO with status $status"
    end
    if length(occupations[1]) > 0
      status = trexio_write_mo_occupation(trexio, vcat(occupations...))
      @assert status == TREXIO.TREXIO_SUCCESS "Failed to write mo_occupation to TREXIO with status $status"
    end
  end
  return  
end

"""
    read_trexio_orbitals(trexio::TrexioFile, basis=nothing; verbose=true) -> SpinMatrix

Read molecular orbitals from TREXIO file and return `SpinMatrix`.
"""
function read_trexio_orbitals(trexio::TrexioFile, basis=nothing; verbose=true)
  # Read basis first
  if isnothing(basis)
    basis = read_trexio_basis(trexio)
  end
  order = ao_order2internal(basis, order4l(basis))
  nao = length(order)
  # Read MO data using standalone TREXIO module
  type, status = TREXIO.trexio_read_mo_type(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read mo_type from TREXIO with status $status"
  if verbose
    println("Read $type molecular orbitals from TREXIO file")
  end
  nmo, status = TREXIO.trexio_read_mo_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read mo_num from TREXIO with status $status"
  coefficients, status = TREXIO.trexio_read_mo_coefficient(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read mo_coefficient from TREXIO with status $status"
  @assert size(coefficients, 1) == nao "Basis size mismatch: basis has $nao, orbitals have $(size(coefficients, 1))"
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo)
  if length(beta_iorbs) == 0
    orbs = SpinMatrix(coefficients[order,:])
  else
    orbs = SpinMatrix(coefficients[order,alpha_iorbs],coefficients[order,beta_iorbs])
  end
  return orbs
end

"""
    alphabeta_orbital_indices(trexio::TrexioFile, nmo)

  Return the indices of alpha and beta orbitals.

  For restricted orbitals the list of beta orbitals is empty.
"""
function alphabeta_orbital_indices(trexio::TrexioFile, nmo)
  spins, status = TREXIO.trexio_read_mo_spin(trexio)
  if status == TREXIO.TREXIO_HAS_NOT || length(spins) != nmo
    return ([1:nmo;], Int[])
  else
    alpha_indices = findall(spins .== 0)
    beta_indices = findall(spins .== 1)
    if length(alpha_indices) == 0 || length(beta_indices) == 0
      # assume restricted case
      return ([1:nmo;], Int[])
    elseif length(alpha_indices) != length(beta_indices) || length(alpha_indices) + length(beta_indices) != nmo
      error("Inconsistent spin information in TREXIO file")
    end
    return (alpha_indices, beta_indices)
  end
end

"""
    read_trexio_orbital_classes(trexio::TrexioFile) -> (classa::Vector{String}, classb::Vector{String})

Read molecular orbital classes from TREXIO file and return as two vectors (alpha, beta).

For restricted orbitals the list of beta orbitals is empty.
If no orbital classes are found, empty vectors are returned.
"""
function read_trexio_orbital_classes(trexio::TrexioFile)
  # Read MO data using standalone TREXIO module
  nmo, status = TREXIO.trexio_read_mo_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read mo_num from TREXIO with status $status"
  classes, status = TREXIO.trexio_read_mo_class(trexio)
  if status != TREXIO.TREXIO_SUCCESS 
    println("Failed to read mo_class from TREXIO with status $status")
    return (String[], String[])
  end
  @assert length(classes) == nmo "Inconsistent number of orbital classes: expected $nmo, got $(length(classes))"
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo)
  if length(beta_iorbs) == 0
    return (classes, String[])
  else
    return (classes[alpha_iorbs], classes[beta_iorbs])
  end
end

"""
    read_trexio_orbital_energies(trexio::TrexioFile) -> (epsa, epsb)

Read molecular orbital energies from TREXIO file and return as two vectors (alpha, beta).

For restricted orbitals the list of beta orbitals is empty.
If no orbital energies are found, empty vectors are returned.
"""
function read_trexio_orbital_energies(trexio::TrexioFile)
  # Read MO data using standalone TREXIO module
  nmo, status = TREXIO.trexio_read_mo_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read mo_num from TREXIO with status $status"
  energies, status = TREXIO.trexio_read_mo_energy(trexio)
  if status != TREXIO.TREXIO_SUCCESS 
    println("Failed to read mo_energy from TREXIO with status $status")
    return (Float64[], Float64[])
  end
  @assert length(energies) == nmo "Inconsistent number of orbital energies: expected $nmo, got $(length(energies))"
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo)
  if length(beta_iorbs) == 0
    return (energies, Float64[])
  else
    return (energies[alpha_iorbs], energies[beta_iorbs])
  end
end

"""
    read_trexio_orbital_occupations(trexio::TrexioFile) -> (occa, occb)

Read molecular orbital occupations from TREXIO file and return as two vectors (alpha, beta).

For restricted orbitals the list of beta orbitals is empty.
If no orbital occupations are found, empty vectors are returned.
"""
function read_trexio_orbital_occupations(trexio::TrexioFile)
  # Read MO data using standalone TREXIO module
  nmo, status = TREXIO.trexio_read_mo_num(trexio)
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read mo_num from TREXIO with status $status"
  occupations, status = TREXIO.trexio_read_mo_occupation(trexio)
  if status != TREXIO.TREXIO_SUCCESS 
    println("Failed to read mo_occupation from TREXIO with status $status")
    return (Float64[], Float64[])
  end
  @assert length(occupations) == nmo "Inconsistent number of orbital occupations: expected $nmo, got $(length(occupations))"
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo)
  if length(beta_iorbs) == 0
    return (occupations, Float64[])
  else
    return (occupations[alpha_iorbs], occupations[beta_iorbs])
  end
end

"""
    write_trexio_amplitudes(trexio::TrexioFile, T1::AbstractArray{Float64,2}, T2::AbstractArray{Float64,4})

Write CC amplitudes to TREXIO format using the standalone TREXIO module.
This is a custom extension for storing amplitude data.
"""
function write_trexio_amplitudes(trexio::TrexioFile, T1::AbstractArray{Float64,2}, T2::AbstractArray{Float64,4})
  if length(T1) > 0
    status = TREXIO.trexio_write_amplitude_single_dense(trexio, T1)
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T1 amplitudes to TREXIO with status $status"
  end
  if length(T2) > 0
    a,b,i,j = size(T2)
    @assert a == b && i == j "T2 amplitudes must be in vvoo order with equal dimensions"
    status = TREXIO.trexio_write_amplitude_double_dense(trexio, T2[:,:,uppertriangular_cut(i)])
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T2 amplitudes to TREXIO with status $status"
  end
end

function write_trexio_amplitudes(trexio::TrexioFile, T1a::AbstractArray{Float64,2}, T1b::AbstractArray{Float64,2},
                                 T2a::AbstractArray{Float64,4}, T2b::AbstractArray{Float64,4}, T2ab::AbstractArray{Float64,4})
  if length(T1a) > 0
    status = TREXIO.trexio_write_amplitude_single_up_dense(trexio, T1a)
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T1a amplitudes to TREXIO with status $status"
  end
  if length(T1b) > 0
    status = TREXIO.trexio_write_amplitude_single_dn_dense(trexio, T1b)
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T1b amplitudes to TREXIO with status $status"
  end
  if length(T2a) > 0
    a,b,i,j = size(T2a)
    @assert a == b && i == j "T2a amplitudes must be in vvoo order with equal dimensions"
    status = TREXIO.trexio_write_amplitude_double_upup_dense(trexio, T2a[strict_uppertriangular_cut(a),strict_uppertriangular_cut(i)])
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T2a amplitudes to TREXIO with status $status"
  end
  if length(T2b) > 0
    a,b,i,j = size(T2b)
    @assert a == b && i == j "T2b amplitudes must be in VVOO order with equal dimensions"
    status = TREXIO.trexio_write_amplitude_double_dndn_dense(trexio, T2b[strict_uppertriangular_cut(a),strict_uppertriangular_cut(i)])
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T2b amplitudes to TREXIO with status $status"
  end
  if length(T2ab) > 0
    status = TREXIO.trexio_write_amplitude_double_updn_dense(trexio, T2ab)
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to write T2ab amplitudes to TREXIO with status $status"
  end
end

"""
    read_trexio_singles(trexio::TrexioFile) -> T1

Read T1 amplitudes from TREXIO file.
"""
function read_trexio_singles(trexio::TrexioFile)
  T1, status = TREXIO.trexio_read_amplitude_single_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    return zeros(0, 0)  # No T1 amplitudes found
  end
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T1 amplitudes from TREXIO with status $status"
  return T1
end

"""
    read_trexio_doubles(trexio::TrexioFile) -> T2

Read T2 amplitudes from TREXIO file.
"""
function read_trexio_doubles(trexio::TrexioFile)
  T2, status = TREXIO.trexio_read_amplitude_double_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    return zeros(0, 0, 0, 0)  # No T2 amplitudes found
  end
  @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T2 amplitudes from TREXIO with status $status"
  return detri_doubles(T2)
end

"""
    read_trexio_unrestricted_singles(trexio::TrexioFile) -> (T1a, T1b)

Read T1a and T1b amplitudes from TREXIO file.
"""
function read_trexio_unrestricted_singles(trexio::TrexioFile)
  T1a, status = TREXIO.trexio_read_amplitude_single_up_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T1a = zeros(0, 0)
  else
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T1a amplitudes from TREXIO with status $status"
  end
  T1b, status = TREXIO.trexio_read_amplitude_single_dn_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T1b = zeros(0, 0)
  else
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T1b amplitudes from TREXIO with status $status"
  end
  return (T1a, T1b)
end

"""
    read_trexio_unrestricted_doubles(trexio::TrexioFile) -> (T2a, T2b, T2ab)

Read T2a, T2b and T2ab amplitudes from TREXIO file.
"""
function read_trexio_unrestricted_doubles(trexio::TrexioFile)
  T2a, status = TREXIO.trexio_read_amplitude_double_upup_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T2a_full = zeros(0, 0, 0, 0)
  else
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T2a amplitudes from TREXIO with status $status"
    T2a_full = detri_samespin_doubles(T2a)
  end
  T2b, status = TREXIO.trexio_read_amplitude_double_dndn_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T2b_full = zeros(0, 0, 0, 0)
  else
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T2b amplitudes from TREXIO with status $status"
    T2b_full = detri_samespin_doubles(T2b)
  end
  T2ab, status = TREXIO.trexio_read_amplitude_double_updn_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T2ab = zeros(0, 0, 0, 0)
  else
    @assert status == TREXIO.TREXIO_SUCCESS "Failed to read T2ab amplitudes from TREXIO with status $status"
  end
  return (T2a_full, T2b_full, T2ab)
end

end # module TrexioInterface