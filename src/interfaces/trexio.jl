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
using ..ElemCo.AbstractEC: AbstractDeterminant
using ..ElemCo.BasisSets
using ..ElemCo.TREXIO  # Use the standalone TREXIO module
using LinearAlgebra

# Re-export the core TREXIO types and functions for backward compatibility
export TrexioFile
export open_trexio, close_trexio
export write_trexio_system, read_trexio_system
export write_trexio_basis, read_trexio_basis
export write_trexio_orbitals, read_trexio_orbitals
export write_trexio_rotations, read_trexio_rotations
export read_trexio_orbital_classes, read_trexio_orbital_energies, read_trexio_orbital_occupations
export write_trexio_amplitudes
export read_trexio_singles, read_trexio_doubles
export read_trexio_unrestricted_singles, read_trexio_unrestricted_doubles
export has_trexio_amplitudes
export orbital_indices_from_classes, occupied_virtual_from_classes, occupied_virtual_from_occupations
# FCI determinant I/O
export write_trexio_determinants, read_trexio_determinants, has_trexio_determinants

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

function open_trexio(f::F, filename::String, mode::String="r") where {F}
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
    return MSystem()  # empty system
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
  trexio_check_write_status(status, "prim_num")
  status = trexio_write_basis_shell_num(trexio, nsh)
  trexio_check_write_status(status, "shell_num")
  status = trexio_write_basis_nucleus_index(trexio, nucleus_index)
  trexio_check_write_status(status, "nucleus_index")
  status = trexio_write_basis_shell_ang_mom(trexio, shell_ang_mom)
  trexio_check_write_status(status, "shell_ang_mom")
  status = trexio_write_basis_shell_factor(trexio, fill(1.0, nsh))  # no normalization
  trexio_check_write_status(status, "shell_factor")
  status = trexio_write_basis_shell_index(trexio, shell_index)
  trexio_check_write_status(status, "shell_index")
  status = trexio_write_basis_exponent(trexio, exponent)
  trexio_check_write_status(status, "exponent")
  status = trexio_write_basis_coefficient(trexio, coefficient)
  trexio_check_write_status(status, "coefficient")
  status = trexio_write_basis_prim_factor(trexio, prim_factor)
  trexio_check_write_status(status, "prim_factor")
  status = trexio_write_basis_name(trexio, basisname)
  trexio_check_write_status(status, "basisname")
  
  # write AO information
  aolist = ao_list(basis)
  status = trexio_write_ao_cartesian(trexio, basis.cartesian ? 1 : 0)
  trexio_check_write_status(status, "ao_cartesian")
  status = trexio_write_ao_num(trexio, length(aolist))
  trexio_check_write_status(status, "ao_num")
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
  trexio_check_write_status(status, "ao_shell")
  status = trexio_write_ao_normalization(trexio, fill(1.0, length(aolist)))
  trexio_check_write_status(status, "ao_normalization")
end

"""
    read_trexio_basis(trexio::TrexioFile) -> BasisSet

Read basis set information from TREXIO file.

Returns empty `BasisSet()` if no molecular system or basis is stored (e.g., FCIDUMP-only case).
"""
function read_trexio_basis(trexio::TrexioFile)
  system = read_trexio_system(trexio)
  if isempty(system)
    return BasisSet()  # No system stored (FCIDUMP-only case)
  end

  type, status = trexio_read_basis_type(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    return BasisSet()  # No basis set stored
  end
  if type != "Gaussian"
    error("Unsupported basis set type: $type")
  end

  prim_num, status = trexio_read_basis_prim_num(trexio)
  trexio_check_read_status(status, "prim_num")
  shell_num, status = trexio_read_basis_shell_num(trexio)
  trexio_check_read_status(status, "shell_num")
  nucleus_index, status = trexio_read_basis_nucleus_index(trexio)
  trexio_check_read_status(status, "nucleus_index")
  shell_ang_mom, status = trexio_read_basis_shell_ang_mom(trexio)
  trexio_check_read_status(status, "shell_ang_mom")
  shell_factor, status = trexio_read_basis_shell_factor(trexio)
  trexio_check_read_status(status, "shell_factor")
  shell_index, status = trexio_read_basis_shell_index(trexio)
  trexio_check_read_status(status, "shell_index")
  exponent, status = trexio_read_basis_exponent(trexio)
  trexio_check_read_status(status, "exponent")
  coefficient, status = trexio_read_basis_coefficient(trexio)
  trexio_check_read_status(status, "coefficient")
  prim_factor, status = trexio_read_basis_prim_factor(trexio)
  trexio_check_read_status(status, "prim_factor")
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

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function write_trexio_orbitals(trexio::TrexioFile, orbitals::SpinMatrix, basis::BasisSet;
                               type="HF", classes=(String[], String[]),
                               energies=(Float64[], Float64[]), occupations=(Float64[], Float64[]),
                               MO="mo")
  write_trexio_basis(trexio, basis)
  nbasis, nmo = size(orbitals)
  nao = n_ao(basis)
  if nao != nbasis
    error("Basis size mismatch: basis has $nao, orbitals have $nbasis")
  end
  order = ao_order2internal(basis, order4l(basis), true)
  _write_trexio_orbital_transformation_data(trexio, orbitals, order, type, classes, energies, occupations, MO)
end

"""
    read_trexio_orbitals(trexio::TrexioFile, basis=nothing; verbose=true, MO="mo") -> (SpinMatrix, String)

  Read molecular orbitals from TREXIO file and return `SpinMatrix` and `type::String`.

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function read_trexio_orbitals(trexio::TrexioFile, basis=nothing; verbose=true, MO="mo")
  # Read basis first
  if isnothing(basis) || isempty(basis)
    basis = read_trexio_basis(trexio)
  end
  order = ao_order2internal(basis, order4l(basis))
  return _read_trexio_orbital_transformations(trexio, order, verbose, MO)
end

"""
    write_trexio_rotations(trexio::TrexioFile, rotations::SpinMatrix;
                          type="Rotation", classes=(String[], String[]),
                          energies=(Float64[], Float64[]), occupations=(Float64[], Float64[]), MO="mo")

  Write rotations of molecular orbitals to TREXIO file. 

The rotations are written in place of orbitals, i.e., it is not possible to write rotations 
to a TREXIO file, which contains orbitals.
`classes`, `energies`, and `occupations` are optional and can be provided as tuples for alpha and beta spins.
`classes` entries can be "Core", "Inactive", "Active", "Virtual", "Deleted"

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function write_trexio_rotations(trexio::TrexioFile, rotations::SpinMatrix;
                               type="HF", classes=(String[], String[]),
                               energies=(Float64[], Float64[]), occupations=(Float64[], Float64[]), MO="mo")
  nbasis, nmo = size(rotations)
  status = trexio_write_ao_num(trexio, nbasis)
  trexio_check_write_status(status, "ao_num")
  _write_trexio_orbital_transformation_data(trexio, rotations, collect(1:nbasis), type, classes, energies, occupations, MO)
end

"""
    read_trexio_rotations(trexio::TrexioFile; verbose=true, MO="mo") -> (SpinMatrix, String)

  Read molecular orbital rotations from TREXIO file and return `SpinMatrix` and `type::String`.

`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function read_trexio_rotations(trexio::TrexioFile; verbose=true, MO="mo")
  nao, status = trexio_read_ao_num(trexio)
  trexio_check_read_status(status, "ao_num")
  return _read_trexio_orbital_transformations(trexio, collect(1:nao), verbose, MO)
end

# Generate wrapper functions for mo and po variants of TREXIO orbital functions
# Uses type-stable dispatch via Val{Symbol} instead of runtime string comparison
for action in ("has", "write", "read")
  for field in ("num", "type", "spin", "coefficient", "class", "energy", "occupation")
    fname = Symbol("trexio_" * action * "_MO_" * field)
    fmoname = Symbol("trexio_" * action * "_mo_" * field)
    fponame = Symbol("trexio_" * action * "_po_" * field)
    @eval begin
      # Type-stable dispatch on Val{Symbol}
      function ($fname)(::Val{:mo}, trexio::TrexioFile, args...)
        return $fmoname(trexio, args...)
      end
      function ($fname)(::Val{:po}, trexio::TrexioFile, args...)
        return $fponame(trexio, args...)
      end
      # String wrapper converts to Val for type-stable dispatch
      function ($fname)(MO::AbstractString, trexio::TrexioFile, args...)
        return ($fname)(Val(Symbol(MO)), trexio, args...)
      end
    end
  end
end

"""
    _write_trexio_orbital_transformation_data(trexio::TrexioFile, coefficients::SpinMatrix, 
                                             order, type, classes, energies, occupations, MO="mo")

  Internal function to write orbital transformation data to TREXIO file.
  
  `MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
""" 
function _write_trexio_orbital_transformation_data(trexio::TrexioFile, coefficients::SpinMatrix, 
                                                   order, type, classes, energies, occupations, MO="mo")
  nbasis, nmo = size(coefficients)
  if length(order) != nbasis
    error("Basis size mismatch: basis has $nbasis, order has $(length(order))")
  end
  status = trexio_write_MO_type(MO, trexio, type)
  trexio_check_write_status(status, "$(MO)_type")
  if is_restricted(coefficients)
    status = trexio_write_MO_num(MO, trexio, nmo)
    trexio_check_write_status(status, "$(MO)_num")
    status = trexio_write_MO_coefficient(MO, trexio, coefficients[1][order, :])
    trexio_check_write_status(status, "$(MO)_coefficient")
    if length(classes[1]) > 0
      status = trexio_write_MO_class(MO, trexio, classes[1])
      trexio_check_write_status(status, "$(MO)_class")
    end
    if length(energies[1]) > 0
      status = trexio_write_MO_energy(MO, trexio, energies[1])
      trexio_check_write_status(status, "$(MO)_energy")
    end
    if length(occupations[1]) > 0
      status = trexio_write_MO_occupation(MO, trexio, occupations[1])
      trexio_check_write_status(status, "$(MO)_occupation")
    end
  else
    status = trexio_write_MO_num(MO, trexio, 2*nmo) # For unrestricted, double the number of coefficients
    trexio_check_write_status(status, "$(MO)_num")
    status = trexio_write_MO_coefficient(MO, trexio, hcat(coefficients...)[order, :])
    trexio_check_write_status(status, "$(MO)_coefficient")
    status = trexio_write_MO_spin(MO, trexio, vcat(fill(0, nmo), fill(1, nmo)))  # α=0, β=1
    trexio_check_write_status(status, "$(MO)_spin")
    if length(classes[1]) > 0
      status = trexio_write_MO_class(MO, trexio, vcat(classes...))
      trexio_check_write_status(status, "$(MO)_class")
    end
    if length(energies[1]) > 0
      status = trexio_write_MO_energy(MO, trexio, vcat(energies...))
      trexio_check_write_status(status, "$(MO)_energy")
    end
    if length(occupations[1]) > 0
      status = trexio_write_MO_occupation(MO, trexio, vcat(occupations...))
      trexio_check_write_status(status, "$(MO)_occupation")
    end
  end
  return
end

"""
    _read_trexio_orbital_transformations(trexio::TrexioFile, order, verbose, MO="mo") -> (SpinMatrix, String)

  Internal function to read orbital transformation coefficients from TREXIO file
  
  `MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function _read_trexio_orbital_transformations(trexio::TrexioFile, order, verbose, MO="mo")
  nao = length(order)
  # Read MO data using standalone TREXIO module
  type, status = trexio_read_MO_type(MO, trexio)
  trexio_check_read_status(status, "$(MO)_type")
  if verbose
    println("Read $type molecular orbitals from TREXIO file")
  end
  nmo, status = trexio_read_MO_num(MO, trexio)
  trexio_check_read_status(status, "$(MO)_num")
  coefficients, status = trexio_read_MO_coefficient(MO, trexio)
  trexio_check_read_status(status, "$(MO)_coefficient")
  if size(coefficients, 1) != nao
    error("Basis size mismatch: basis has $nao, orbitals have $(size(coefficients, 1))")
  end
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo, MO)
  if length(beta_iorbs) == 0
    coeffs = SpinMatrix(coefficients[order,:])
  else
    coeffs = SpinMatrix(coefficients[order,alpha_iorbs],coefficients[order,beta_iorbs])
  end
  return coeffs, type
end


"""
    alphabeta_orbital_indices(trexio::TrexioFile, nmo, MO="mo") -> (alpha_indices::Vector{Int}, beta_indices::Vector{Int})

  Return the indices of alpha and beta orbitals.

  For restricted orbitals the list of beta orbitals is empty.
  
  `MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function alphabeta_orbital_indices(trexio::TrexioFile, nmo, MO="mo")
  spins, status = trexio_read_MO_spin(MO, trexio)
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
    read_trexio_orbital_classes(trexio::TrexioFile, MO="mo") -> (classa::Vector{String}, classb::Vector{String})

Read molecular orbital classes from TREXIO file and return as two vectors (alpha, beta).

For restricted orbitals the list of beta orbitals is empty.
If no orbital classes are found, empty vectors are returned.
  
`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function read_trexio_orbital_classes(trexio::TrexioFile, MO="mo")
  # Read MO data using standalone TREXIO module
  nmo, status = trexio_read_MO_num(MO, trexio)
  trexio_check_read_status(status, "$(MO)_num")
  classes, status = trexio_read_MO_class(MO, trexio)
  if status != TREXIO.TREXIO_SUCCESS
    # println("Failed to read $(MO)_class from TREXIO with status $(string(status))")
    return (String[], String[])
  end
  if length(classes) != nmo
    error("Inconsistent number of orbital classes: expected $nmo, got $(length(classes))")
  end
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo, MO)
  if length(beta_iorbs) == 0
    return (classes, String[])
  else
    return (classes[alpha_iorbs], classes[beta_iorbs])
  end
end

"""
    read_trexio_orbital_energies(trexio::TrexioFile, MO="mo") -> (epsa, epsb)

Read molecular orbital energies from TREXIO file and return as two vectors (alpha, beta).

For restricted orbitals the list of beta orbitals is empty.
If no orbital energies are found, empty vectors are returned.
  
`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function read_trexio_orbital_energies(trexio::TrexioFile, MO="mo")
  # Read MO data using standalone TREXIO module
  nmo, status = trexio_read_MO_num(MO, trexio)
  trexio_check_read_status(status, "$(MO)_num")
  energies, status = trexio_read_MO_energy(MO, trexio)
  if status != TREXIO.TREXIO_SUCCESS
    # println("Failed to read $(MO)_energy from TREXIO with status $(string(status))")
    return (Float64[], Float64[])
  end
  if length(energies) != nmo
    error("Inconsistent number of orbital energies: expected $nmo, got $(length(energies))")
  end
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo, MO)
  if length(beta_iorbs) == 0
    return (energies, Float64[])
  else
    return (energies[alpha_iorbs], energies[beta_iorbs])
  end
end

"""
    read_trexio_orbital_occupations(trexio::TrexioFile, MO="mo") -> (occa, occb)

Read molecular orbital occupations from TREXIO file and return as two vectors (alpha, beta).

For restricted orbitals the list of beta orbitals is empty.
If no orbital occupations are found, empty vectors are returned.
  
`MO` can be "mo" for molecular orbitals or "po" for positron orbitals.
"""
function read_trexio_orbital_occupations(trexio::TrexioFile, MO="mo")
  # Read MO data using standalone TREXIO module
  nmo, status = trexio_read_MO_num(MO, trexio)
  trexio_check_read_status(status, "$(MO)_num")
  occupations, status = trexio_read_MO_occupation(MO, trexio)
  if status != TREXIO.TREXIO_SUCCESS
    # println("Failed to read $(MO)_occupation from TREXIO with status $(string(status))")
    return (Float64[], Float64[])
  end
  if length(occupations) != nmo
    error("Inconsistent number of orbital occupations: expected $nmo, got $(length(occupations))")
  end
  alpha_iorbs, beta_iorbs = alphabeta_orbital_indices(trexio, nmo, MO)
  if length(beta_iorbs) == 0
    return (occupations, Float64[])
  else
    return (occupations[alpha_iorbs], occupations[beta_iorbs])
  end
end

"""
    write_trexio_amplitudes(trexio::TrexioFile, T1::AbstractArray{<:Number,2}, T2::AbstractArray{<:Number,4})

Write CC amplitudes to TREXIO format using the standalone TREXIO module.
This is a custom extension for storing amplitude data.
"""
function write_trexio_amplitudes(trexio::TrexioFile, T1::AbstractArray{<:Number,2}, T2::AbstractArray{<:Number,4})
  if length(T1) > 0
    status = TREXIO.trexio_write_amplitude_single_dense(trexio, T1)
    trexio_check_write_status(status, "T1 amplitudes")
  end
  if length(T2) > 0
    a,b,i,j = size(T2)
    @assert a == b && i == j "T2 amplitudes must be in vvoo order with equal dimensions"
    status = TREXIO.trexio_write_amplitude_double_dense(trexio, T2[:,:,uppertriangular_cut(i)])
    trexio_check_write_status(status, "T2 amplitudes")
  end
end

function write_trexio_amplitudes(trexio::TrexioFile, T1a::AbstractArray{<:Number,2}, T1b::AbstractArray{<:Number,2},
                                 T2a::AbstractArray{<:Number,4}, T2b::AbstractArray{<:Number,4}, T2ab::AbstractArray{<:Number,4})
  if length(T1a) > 0
    status = TREXIO.trexio_write_amplitude_single_up_dense(trexio, T1a)
    trexio_check_write_status(status, "T1a amplitudes")
  end
  if length(T1b) > 0
    status = TREXIO.trexio_write_amplitude_single_dn_dense(trexio, T1b)
    trexio_check_write_status(status, "T1b amplitudes")
  end
  if length(T2a) > 0
    a,b,i,j = size(T2a)
    @assert a == b && i == j "T2a amplitudes must be in vvoo order with equal dimensions"
    status = TREXIO.trexio_write_amplitude_double_upup_dense(trexio, T2a[strict_uppertriangular_cut(a),strict_uppertriangular_cut(i)])
    trexio_check_write_status(status, "T2a amplitudes")
  end
  if length(T2b) > 0
    a,b,i,j = size(T2b)
    @assert a == b && i == j "T2b amplitudes must be in VVOO order with equal dimensions"
    status = TREXIO.trexio_write_amplitude_double_dndn_dense(trexio, T2b[strict_uppertriangular_cut(a),strict_uppertriangular_cut(i)])
    trexio_check_write_status(status, "T2b amplitudes")
  end
  if length(T2ab) > 0
    status = TREXIO.trexio_write_amplitude_double_updn_dense(trexio, T2ab)
    trexio_check_write_status(status, "T2ab amplitudes")
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
  trexio_check_read_status(status, "T1 amplitudes")
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
  trexio_check_read_status(status, "T2 amplitudes")
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
    trexio_check_read_status(status, "T1a amplitudes")
  end
  T1b, status = TREXIO.trexio_read_amplitude_single_dn_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T1b = zeros(0, 0)
  else
    trexio_check_read_status(status, "T1b amplitudes")
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
    trexio_check_read_status(status, "T2a amplitudes")
    T2a_full = detri_samespin_doubles(T2a)
  end
  T2b, status = TREXIO.trexio_read_amplitude_double_dndn_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T2b_full = zeros(0, 0, 0, 0)
  else
    trexio_check_read_status(status, "T2b amplitudes")
    T2b_full = detri_samespin_doubles(T2b)
  end
  T2ab, status = TREXIO.trexio_read_amplitude_double_updn_dense(trexio)
  if status == TREXIO.TREXIO_HAS_NOT
    T2ab = zeros(0, 0, 0, 0)
  else
    trexio_check_read_status(status, "T2ab amplitudes")
  end
  return (T2a_full, T2b_full, T2ab)
end

"""
    has_trexio_amplitudes(trexio::TrexioFile; unrestricted::Bool=false)

Check if amplitudes are stored in the TREXIO file using TREXIO has_* functions.

Returns `true` if any amplitude data (singles or doubles) is found.
This function does not read the amplitudes, only checks for their presence.
"""
function has_trexio_amplitudes(trexio::TrexioFile; unrestricted::Bool=false)
  if unrestricted
    # Check for unrestricted amplitudes
    has_t1a = TREXIO.trexio_has_amplitude_single_up_dense(trexio)
    has_t1b = TREXIO.trexio_has_amplitude_single_dn_dense(trexio)
    has_t2a = TREXIO.trexio_has_amplitude_double_upup_dense(trexio)
    has_t2b = TREXIO.trexio_has_amplitude_double_dndn_dense(trexio)
    has_t2ab = TREXIO.trexio_has_amplitude_double_updn_dense(trexio)
    return has_t1a || has_t1b || has_t2a || has_t2b || has_t2ab
  else
    # Check for restricted amplitudes
    has_t1 = TREXIO.trexio_has_amplitude_single_dense(trexio)
    has_t2 = TREXIO.trexio_has_amplitude_double_dense(trexio)
    return has_t1 || has_t2
  end
end

"""
    orbital_indices_from_classes(classes::Vector{String}; include::Vector{String}=["Inactive", "Active", "Virtual"])

Extract orbital indices for each orbital class type from orbital class labels.

# Arguments
- `classes`: Vector of orbital class labels ("Core", "Inactive", "Active", "Virtual", "Deleted")
- `include`: Which classes to include in the result (default: ["Inactive", "Active", "Virtual"])

# Returns
A `Dict{String,Vector{Int}}` mapping each class name to the orbital indices with that class.

# Example
```julia
classes = ["Core", "Inactive", "Inactive", "Virtual", "Virtual", "Virtual", "Deleted"]
result = orbital_indices_from_classes(classes)
# result["Inactive"] = [2, 3]
# result["Active"] = Int[]
# result["Virtual"] = [4, 5, 6]
```
"""
function orbital_indices_from_classes(classes::Vector{String}; 
                                       include::Vector{String}=["Inactive", "Active", "Virtual"])
  result = Dict{String,Vector{Int}}()
  for class_name in include
    result[class_name] = Int[]
  end
  for (i, c) in enumerate(classes)
    if c in include
      push!(result[c], i)
    end
  end
  return result
end

"""
    occupied_virtual_from_classes(classes::Vector{String})

Extract occupied and virtual orbital indices from orbital class labels.

Returns `(occ_indices, virt_indices)` where:
- `occ_indices` are orbitals with class "Inactive" or "Active"
- `virt_indices` are orbitals with class "Virtual"

Note: "Core" and "Deleted" orbitals are not included in either list.
"""
function occupied_virtual_from_classes(classes::Vector{String})
  occ_indices = Int[]
  virt_indices = Int[]
  for (i, c) in enumerate(classes)
    if c == "Inactive" || c == "Active"
      push!(occ_indices, i)
    elseif c == "Virtual"
      push!(virt_indices, i)
    end
  end
  return occ_indices, virt_indices
end

"""
    occupied_virtual_from_occupations(occupations::Vector{Float64}, classes::Vector{String}=String[]; 
                                      occ_threshold::Float64=0.5)

Extract occupied and virtual orbital indices from occupation numbers and optionally class labels.

Orbitals with occupation ≥ `occ_threshold` are considered occupied.
If classes are provided, "Core" and "Deleted" orbitals are excluded from both lists.

Returns `(occ_indices, virt_indices)`.

This is more reliable than `occupied_virtual_from_classes` for UHF orbitals where the same
orbital index may be occupied for one spin but virtual for the other.
"""
function occupied_virtual_from_occupations(occupations::Vector{Float64}, 
                                           classes::Vector{String}=String[];
                                           occ_threshold::Float64=0.5)
  occ_indices = Int[]
  virt_indices = Int[]
  for (i, occ) in enumerate(occupations)
    # Skip Core and Deleted orbitals if classes are provided
    if !isempty(classes) && length(classes) >= i
      c = classes[i]
      if c == "Core" || c == "Deleted"
        continue
      end
    end
    if occ >= occ_threshold
      push!(occ_indices, i)
    else
      push!(virt_indices, i)
    end
  end
  return occ_indices, virt_indices
end

include("trexio_fci.jl")

end # module TrexioInterface