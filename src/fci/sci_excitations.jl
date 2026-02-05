# ===========================================
# Orbital Excitation Analysis
# ===========================================

"""
    find_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern) where OPattern -> (Int, Int)

Find the two orbitals involved in a single excitation i -> a.
"""
function find_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern)::Tuple{Int, Int} where OPattern
  diff = pattern_i ⊻ pattern_j
  occ = pattern_i & diff
  virt = pattern_j & diff
 
  orb_a = trailing_zeros(virt) + 1  # Orbital being created
  orb_i = trailing_zeros(occ) + 1  # Orbital being destroyed

  return (orb_i, orb_a)
end

"""
    find_double_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern) where OPattern -> (Int, Int, Int, Int)

Find the four orbitals involved in a double excitation ij -> ab.
"""
function find_double_excitation_orbitals(pattern_i::OPattern, pattern_j::OPattern) where OPattern
  diff = pattern_i ⊻ pattern_j
  occ = pattern_i & diff
  virt = pattern_j & diff
 
  orb_a = trailing_zeros(virt)  # Orbital being created
  virt ⊻= (OPattern(1) << orb_a)
  orb_b = trailing_zeros(virt)  # Second orbital being created
  orb_i = trailing_zeros(occ)  # Orbital being destroyed
  occ ⊻= (OPattern(1) << orb_i)
  orb_j = trailing_zeros(occ)  # Second orbital being destroyed
  return (orb_i+1, orb_j+1, orb_a+1, orb_b+1)
end

"""
    calculate_excitation_phase(pattern::OPattern, orb_i::Int, orb_a::Int) where OPattern -> Int

Calculate phase factor for single excitation i -> a.
"""
function calculate_excitation_phase(pattern::OPattern, orb_i::Int, orb_a::Int)::Int where OPattern
  # Count electrons between orb_i and orb_a
  min_orb = min(orb_i, orb_a)
  max_orb = max(orb_i, orb_a) - 1
  # pattern between min_orb and max_orb (exclusive)
  sub_pattern = pattern & ((OPattern(1) << max_orb) - 1) & ~( (OPattern(1) << min_orb) - 1)
  # Count number of electrons in sub_pattern
  n_electrons = count_ones(sub_pattern)
  # Phase is (-1)^n_electrons
  return (n_electrons % 2 == 0) ? 1 : -1
end

"""
    calculate_double_excitation_phase(pattern::OPattern, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) where OPattern -> Int

Calculate phase factor for double excitation ij -> ab.

The phase is calculated by decomposing into two successive single excitations:
  1. First excitation:  i -> a, with phase φ₁
  2. Second excitation: j -> b in modified determinant, with phase φ₂
  Total phase = φ₁ × φ₂
  
This approach correctly handles the fermionic anticommutation relations.
"""
function calculate_double_excitation_phase(pattern::OPattern, orb_i::Int, orb_j::Int,
                                           orb_a::Int, orb_b::Int)::Int where OPattern
  # First excitation: i -> a
  phase1 = calculate_excitation_phase(pattern, orb_i, orb_a)
  
  # Create intermediate determinant: remove electron from orb_i, add to orb_a
  intermediate = pattern
  intermediate &= ~(OPattern(1) << (orb_i - 1))  # Remove electron from i
  intermediate |= (OPattern(1) << (orb_a - 1))   # Add electron to a

  # Second excitation: j -> b in the intermediate determinant
  phase2 = calculate_excitation_phase(intermediate, orb_j, orb_b)
  
  # Total phase is the product
  return phase1 * phase2
end

"""
    single_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int) -> (Determinant{OPattern}, Int)

Create determinant and calculate phase factor for single alpha excitation i -> a.
"""
function single_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int)
  new_det = single_excitation_alpha(det, orb_i, orb_a)
  return new_det, calculate_excitation_phase(det.alpha, orb_i, orb_a)
end
"""
    single_beta_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int) -> (Determinant{OPattern}, Int)

Create determinant and calculate phase factor for single beta excitation i -> a.
"""
function single_beta_excitation_phase(det::Determinant, orb_i::Int, orb_a::Int)
  new_det = single_excitation_beta(det, orb_i, orb_a)
  return new_det, calculate_excitation_phase(det.beta, orb_i, orb_a)
end

"""
    double_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> (Determinant, Int)

Create determinant and calculate phase factor for double alpha excitation ij -> ab.
"""
function double_alpha_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int,
                                       orb_a::Int, orb_b::Int)
  new_det = double_excitation_alpha(det, orb_i, orb_j, orb_a, orb_b)
  return new_det, calculate_double_excitation_phase(det.alpha, orb_i, orb_j, orb_a, orb_b) 
end
"""
    double_beta_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int, orb_a::Int, orb_b::Int) -> (Determinant, Int)

Create determinant and calculate phase factor for double beta excitation ij -> ab.
"""
function double_beta_excitation_phase(det::Determinant, orb_i::Int, orb_j::Int,
                                      orb_a::Int, orb_b::Int)
  new_det = double_excitation_beta(det, orb_i, orb_j, orb_a, orb_b)
  return new_det, calculate_double_excitation_phase(det.beta, orb_i, orb_j, orb_a, orb_b)
end

"""
    double_alpha_beta_excitation_phase(det::Determinant, orb_i_alpha::Int, orb_i_beta::Int,
                                       orb_a_alpha::Int, orb_a_beta::Int) -> (Determinant, Int)

Calculate phase factor for double alpha-beta excitation iα jβ -> aα bβ.
"""
function double_alpha_beta_excitation_phase(det::Determinant, orb_i_alpha::Int, orb_i_beta::Int,
                                               orb_a_alpha::Int, orb_a_beta::Int)
  new_det = double_excitation_mixed(det, orb_i_alpha, orb_i_beta, orb_a_alpha, orb_a_beta)
  phase_alpha = calculate_excitation_phase(det.alpha, orb_i_alpha, orb_a_alpha)
  phase_beta = calculate_excitation_phase(det.beta, orb_i_beta, orb_a_beta)
  return new_det, phase_alpha * phase_beta
end

"""
    single_excitation_alpha(det::Determinant{OPattern}, i::Int, a::Int) where OPattern -> Determinant{OPattern}

Create determinant with alpha electron moved from orbital i to orbital a.
"""
function single_excitation_alpha(det::Determinant{OPattern}, i::Int, a::Int)::Determinant{OPattern} where OPattern
  # Remove electron from orbital i, add to orbital a
  new_alpha = det.alpha & ~(OPattern(1) << (i-1))  # Remove i
  new_alpha |= (OPattern(1) << (a-1))              # Add a
  return Determinant(new_alpha, det.beta)
end

"""
    single_excitation_beta(det::Determinant{OPattern}, i::Int, a::Int) where OPattern -> Determinant{OPattern}

Create determinant with beta electron moved from orbital i to orbital a.
"""
function single_excitation_beta(det::Determinant{OPattern}, i::Int, a::Int)::Determinant{OPattern} where OPattern
  new_beta = det.beta & ~(OPattern(1) << (i-1))
  new_beta |= (OPattern(1) << (a-1))
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_alpha(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int) where OPattern -> Determinant{OPattern}

Create determinant with alpha electrons moved from orbitals i,j to orbitals a,b.
"""
function double_excitation_alpha(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int)::Determinant{OPattern} where OPattern
  new_alpha = det.alpha & ~(OPattern(1) << (i-1))  # Remove i
  new_alpha &= ~(OPattern(1) << (j-1))              # Remove j
  new_alpha |= (OPattern(1) << (a-1))               # Add a
  new_alpha |= (OPattern(1) << (b-1))               # Add b
  return Determinant(new_alpha, det.beta)
end

"""
    double_excitation_beta(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int) where OPattern -> Determinant{OPattern}

Create determinant with beta electrons moved from orbitals i,j to orbitals a,b.
"""
function double_excitation_beta(det::Determinant{OPattern}, i::Int, j::Int, a::Int, b::Int)::Determinant{OPattern} where OPattern
  new_beta = det.beta & ~(OPattern(1) << (i-1))
  new_beta &= ~(OPattern(1) << (j-1))
  new_beta |= (OPattern(1) << (a-1))
  new_beta |= (OPattern(1) << (b-1))
  return Determinant(det.alpha, new_beta)
end

"""
    double_excitation_mixed(det::Determinant, i_alpha::Int, i_beta::Int, a_alpha::Int, a_beta::Int) -> Determinant

Create determinant with one alpha excitation i_alpha->a_alpha and one beta excitation i_beta->a_beta.
"""
function double_excitation_mixed(det::Determinant{OPattern}, i_alpha::Int, i_beta::Int, 
                                 a_alpha::Int, a_beta::Int)::Determinant{OPattern} where OPattern
  new_alpha = det.alpha & ~(OPattern(1) << (i_alpha-1))
  new_alpha |= (OPattern(1) << (a_alpha-1))
  new_beta = det.beta & ~(OPattern(1) << (i_beta-1))
  new_beta |= (OPattern(1) << (a_beta-1))
  return Determinant(new_alpha, new_beta)
end