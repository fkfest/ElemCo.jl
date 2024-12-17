"""
Molden interface

This module provides an interface to Molden to read and write orbitals and other data.
"""
module MoldenInterface
using Unitful, UnitfulAtomic
using AtomsBase
using Printf
using ..ElemCo.Utils
using ..ElemCo.ECInfos
using ..ElemCo.QMTensors
using ..ElemCo.MSystem
using ..ElemCo.BasisSets
using ..ElemCo.Wavefunctions
using ..ElemCo.OrbTools

export is_molden_file, write_molden_orbitals

"""
    MOLDEN2LIBCINT_PERMUTATION

  Permutation of the atomic orbitals from the Molden to the libcint order.
"""
const MOLDEN2LIBCINT_PERMUTATION = [        # Molden order:
  [1],                                      # s 
  [1,2,3],                                  # p x,y,z
  [5,3,1,2,4],                              # d 0, +1, -1 , +2, -2 (z^2, xz, yz, x^2-y^2, xy)
  [7,5,3,1,2,4,6],                          # f 0, +1, -1, +2, -2, +3, -3
  [9,7,5,3,1,2,4,6,8],                      # g 0, +1, -1, +2, -2, +3, -3, +4, -4
  [11,9,7,5,3,1,2,4,6,8,10],                # h 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5
  [13,11,9,7,5,3,1,2,4,6,8,10,12]           # i 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5, +6, -6
      ]

"""
    MOLDEN2LIBCINT_PERMUTATION_CART

  Permutation of the atomic orbitals from the Molden to the libcint order for cartesian basis sets.
"""
const MOLDEN2LIBCINT_PERMUTATION_CART = [   # Molden order:
  [1],                                      # s 
  [1,2,3],                                  # p x,y,z
  [1,4,5,2,6,3],                            # d x²,y²,z²,xy,xz,yz 
  [1,5,6,4,10,7,2,9,8,3],                   # f x³,y³,z³,xy²,x²y,x²z,xz²,yz²,y²z,xyz 
  [1,4,5,10,13,11,6,14,15,8,2,7,12,9,3]     # g x⁴,y⁴,z⁴,x³y,x³z,xy³,y³z,xz³,yz³,x²y²,x²z²,y²z²,x²yz,xy²z,xyz²   
      ]
"""
    is_molden_file(filename::String)

  Check if the file `filename` is a Molden file.
"""
function is_molden_file(filename::String)
  open(filename) do f
    for line in eachline(f)
      if occursin(r"^\s*\[Molden Format\]", line)
        return true
      end
    end
  end
  return false
end

"""
    write_molden_orbitals(EC::ECInfo, filename::String)

  Write the current orbitals to a Molden file.
"""
function write_molden_orbitals(EC::ECInfo, filename::String)
  basisset = generate_basis(EC, "ao")
  order = ao_permutation(EC, true)
  orbs = load_orbitals(EC)
  SP = EC.space
  if is_restricted(orbs)
    occ = [2*ones(Int, length(SP['o'])); zeros(Int, length(SP['v']))]
    ϵo, ϵv = orbital_energies(EC)
    eps = append!(ϵo, ϵv)
  else
    occa = [ones(Int, length(SP['o'])); zeros(Int, length(SP['v']))]
    occb = [ones(Int, length(SP['O'])); zeros(Int, length(SP['V']))]
    ϵoa, ϵva = orbital_energies(EC, :α)
    ϵob, ϵvb = orbital_energies(EC, :β)
    epsa = [ϵoa; ϵva]
    epsb = [ϵob; ϵvb]
  end
  has_positron = EC.options.wf.npositron > 0
  if has_positron
    orbs_pos = load_positron_orbitals(EC)
    eps_pos = load_positron_epsilon(EC)
    occ_pos = zeros(Int, length(SP['m']))
    occ_pos[1] = 1
  end
  open(filename, "w") do f
    println(f, "[Molden Format]")
    distunit = unit(EC.system[1].position[1])
    if distunit == u"bohr"
      println(f, "[Atoms] AU")
    else
      distunit = u"angstrom"
      println(f, "[Atoms] Angs")
    end
    for (iat,atom) in enumerate(EC.system)
      coord = uconvert.(distunit, atom.position)/distunit
      @printf(f, "%s %i %i %16.10f %16.10f %16.10f\n", 
              atomic_center_symbol(atom), iat, atomic_number(atom), coord[1], coord[2], coord[3])
    end
    println(f, "[GTO]")
    for ic in center_range(basisset)
      println(f, "   ", ic, " ", 0)
      for ash in basisset.centers[ic].shells
        for con in ash.subshells
          println(f, " ", subshell_char(ash.l), " ", length(con.exprange))
          for (i, iex) in enumerate(con.exprange)
            @printf(f, "%.10E %.10E\n", ash.exponents[iex], con.coefs[i])
          end
        end
      end
      println(f)
    end
    println(f, "[MO]")
    if !is_cartesian(basisset)
      maxl = max_l(basisset)
      maxl > 1 && println(f, "[5D]")
      maxl > 2 && println(f, "[7F]")
      maxl > 3 && println(f, "[9G]")
      maxl > 4 && println(f, "[11H]")
      maxl > 5 && println(f, "[13I]")
    end
    if is_restricted(orbs)
      cmo = orbs[1]
      energies = eps
      occupation = occ
      printmos(f, cmo, order, energies, occupation)
    else
      printmos(f, orbs[1], order, epsa, occa)
      printmos(f, orbs[2], order, epsb, occb, "Beta")
    end
  end
  if (has_positron)
    println("Writing also positron orbitals to $(filename)_positron")
    open(filename*"_positron", "w") do f
      println(f, "[Molden Format]")
      distunit = unit(EC.system[1].position[1])
      if distunit == u"bohr"
        println(f, "[Atoms] AU")
      else
        distunit = u"angstrom"
        println(f, "[Atoms] Angs")
      end
      for (iat,atom) in enumerate(EC.system)
        coord = uconvert.(distunit, atom.position)/distunit
        @printf(f, "%s %i %i %16.10f %16.10f %16.10f\n", 
                atomic_center_symbol(atom), iat, atomic_number(atom), coord[1], coord[2], coord[3])
      end
      println(f, "[GTO]")
      for ic in center_range(basisset)
        println(f, "   ", ic, " ", 0)
        for ash in basisset.centers[ic].shells
          for con in ash.subshells
            println(f, " ", subshell_char(ash.l), " ", length(con.exprange))
            for (i, iex) in enumerate(con.exprange)
              @printf(f, "%.10E %.10E\n", ash.exponents[iex], con.coefs[i])
            end
          end
        end
        println(f)
      end
      println(f, "[MO]")
      if !is_cartesian(basisset)
        maxl = max_l(basisset)
        maxl > 1 && println(f, "[5D]")
        maxl > 2 && println(f, "[7F]")
        maxl > 3 && println(f, "[9G]")
        maxl > 4 && println(f, "[11H]")
        maxl > 5 && println(f, "[13I]")
      end
      cmo = orbs_pos[1]
      energies = eps_pos
      occupation = occ_pos
      printmos(f, cmo, order, energies, occupation)
    end
  end
end

"""
    printmos(f, orbs, order, energies, occupation, spin="Alpha")

  Print the molecular orbital coefficients to a Molden file.
"""
function printmos(f, orbs, order, energies, occupation, spin="Alpha")
  nmo = size(orbs,2)
  for imo = 1:nmo
    println(f, " Sym=  ", imo, ".1")
    println(f, " Ene=  ", energies[imo])
    println(f, " Spin= ", spin)
    println(f, " Occup= ", occupation[imo])
    for (i, iao) in enumerate(order)
      @printf(f, "%i %.15f\n", i, orbs[iao,imo])
    end
  end
end

"""
    ao_permutation(EC::ECInfo, back=false)

  Return the permutation of the atomic orbitals from the Molden to the libcint order 
  such that `μ(molden)[ao_permutation(EC)] = μ(libcint)`.

  If `back` is `true`, the permutation is for the libcint to Molden order.
"""
function ao_permutation(EC::ECInfo, back=false)
  basisset = generate_basis(EC, "ao")
  permutation = is_cartesian(basisset) ? MOLDEN2LIBCINT_PERMUTATION_CART : MOLDEN2LIBCINT_PERMUTATION
  action = back ? invperm : identity
  order = Int[]
  for ash in basisset
    for ish = 1:n_subshells(ash)
      append!(order, action(permutation[ash.l+1]) .+ length(order))
    end
  end
  return order
end
end # module
