

const ML_cart = [[""],
                ["x","y","z"],
                ["x²","xy","xz","y²","yz","z²"],
                ["x³","x²y","x²z","xy²","xyz","xz²","y³","y²z", "yz²","z³"],
                ["x⁴","x³y","x³z","x²y²","x²yz","x²z²","xy³","xy²z","xyz²","xz³","y⁴","y³z","y²z²","yz³","z⁴"],
                ["x⁵","x⁴y","x⁴z","x³y²","x³yz","x³z²","x²y³","x²y²z","x²yz²","x²z³","xy⁴","xy³z","xy²z²","xyz³","xz⁴","y⁵","y⁴z","y³z²","y²z³","yz⁴","z⁵"],
                ["x⁶","x⁵y","x⁵z","x⁴y²","x⁴yz","x⁴z²","x³y³","x³y²z","x³yz²","x³z³","x²y⁴","x²y³z","x²y²z²","x²yz³","x²z⁴","xy⁵","xy⁴z","xy³z²","xy²z³","xyz⁴","xz⁵","y⁶","y⁵z","y⁴z²","y³z³","y²z⁴","yz⁵","z⁶"]]

const ML_sph = [[""],                                                         # s
                ["x","y","z"],                                                # p
                ["xy","yz","z²","xz","x²-y²"]]                                # d

abstract type AbstractAtomicOrbital end

"""
  SphericalAtomicOrbital

  Represents an atomic orbital in a spherical basis set.

  $(TYPEDFIELDS)
"""
struct SphericalAtomicOrbital <: AbstractAtomicOrbital
  """ index of the centre in the basis set object """
  icentre::UInt16
  """ index of the angular shell in the basis set object """
  iangularshell::UInt16
  """ index of the contraction in the angular shell """
  isubshell::UInt8
  """ principal quantum number """
  n::UInt8
  """ orbital angular momentum quantum number """
  l::UInt8
  """ magnetic quantum number (ml = -l:l)"""
  ml::Int8
end

"""
  CartesianAtomicOrbital

  Represents an atomic orbital in a cartesian basis set.

  $(TYPEDFIELDS)
"""
struct CartesianAtomicOrbital <: AbstractAtomicOrbital
  """ index of the centre in the basis set object """
  icentre::UInt16
  """ index of the angular shell in the basis set object """
  iangularshell::UInt16
  """ index of the subshell in the angular shell """
  isubshell::UInt8
  """ principal quantum number """
  n::UInt8
  """ orbital angular momentum quantum number """
  l::UInt8
  """ magnetic "quantum number" (ml = -l:l(l+1)/2)"""
  ml::Int8
end

function Base.show(io::IO, ao::SphericalAtomicOrbital) 
  print(io, "$(ao.n)", SUBSHELLS_NAMES[ao.l+1])
  if ao.l == 0
    # do nothing for s orbitals
  elseif ao.l < length(ML_sph)
    print(io, "{", ML_sph[ao.l+1][ao.ml+ao.l+1], "}")
  else
    print(io, "{$(ao.ml)}")
  end
end

function Base.show(io::IO, ao::CartesianAtomicOrbital) 
  print(io, "$(ao.n)", SUBSHELLS_NAMES[ao.l+1])
  if ao.l == 0
    # do nothing for s orbitals
  elseif ao.l < length(ML_cart)
    print(io, "{", ML_cart[ao.l+1][ao.ml+ao.l+1], "}")
  else
    print(io, "{$(ao.ml)}")
  end
end


