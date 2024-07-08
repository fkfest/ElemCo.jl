# adapted from GaussianBasis.jl

"""
    AbstractILib

Abstract type for infos for integral libraries.
"""
abstract type AbstractILib end

"""
    ILibcint5

  Infos for Libcint5 integral library.
"""
struct ILibcint5 <: AbstractILib
  atm::Vector{Cint}
  natm::Cint
  bas::Vector{Cint}
  nbas::Cint
  env::Vector{Cdouble}
end

Base.show(io::IO, ilib::ILibcint5) = print(io, "libcint v5")

"""
    ILibcint5(atoms::Vector{BasisCenter}, cartesian::Bool)

  Prepare the infos for Libcint5 integral library.
"""
function ILibcint5(atoms::Vector{BasisCenter}, cartesian::Bool)
  ATM_SLOTS = 6
  BAS_SLOTS = 8

  natoms = length(atoms)

  nashells = n_angularshells(atoms)
  nprim_tot = n_primitives(atoms)
  ncoefs_tot = 0
  for atom in atoms, ashell in atom.shells
    ncoefs_tot += n_coefficients_1mat(ashell)
  end

  lc_atm = zeros(Cint, natoms*ATM_SLOTS)
  lc_bas = zeros(Cint, nashells*BAS_SLOTS)
  env = zeros(Cdouble, 20+4*natoms+nprim_tot+ncoefs_tot)

  # Prepare the lc_atm input
  off = 20
  for (i, atom) in enumerate(atoms)
    # lc_atom has ATM_SLOTS (6) "spaces" for each atom
    # The first one (Z_INDEX) is the atomic number
    lc_atm[1 + ATM_SLOTS*(i-1)] = atom.atomic_number
    # The second one is the env index address for xyz
    lc_atm[2 + ATM_SLOTS*(i-1)] = off
    env[off+1:off+3] .= atom.position
    off += 4 # Skip an extra slot for the kappa (nuclear model parameter)
    # The remaining 4 slots are zero.
  end

  # Prepare the lc_bas input
  ib = 0
  for (i, atom) in enumerate(atoms), ashell in atom.shells
    nprim = n_primitives(ashell)
    ncoefs = n_coefficients_1mat(ashell)
    # lc_bas has BAS_SLOTS for each basis set block (angular shell)
    # The first one is the index of the atom starting from 0
    lc_bas[1 + BAS_SLOTS*ib] = i - 1
    # The second one is the angular momentum
    lc_bas[2 + BAS_SLOTS*ib] = ashell.l
    # The third is the number of primitive functions
    lc_bas[3 + BAS_SLOTS*ib] = nprim
    # The fourth is the number of contracted functions
    lc_bas[4 + BAS_SLOTS*ib] = n_subshells(ashell)
    # The fifth is a κ parameter
    lc_bas[5 + BAS_SLOTS*ib] = 0
    # Sixth is the env index address for exponents
    lc_bas[6 + BAS_SLOTS*ib] = off
    env[off+1:off+nprim] .= ashell.exponents
    off += nprim
    # Seventh is the env index address for contraction coeff
    lc_bas[7 + BAS_SLOTS*ib] = off
    env[off+1:off+ncoefs] .= coefficients_1mat(ashell, cartesian)[:]
    off += ncoefs
    # Eigth, nothing
    ib += 1
  end

  return ILibcint5(lc_atm, Cint(natoms), lc_bas, Cint(ib), env)
end