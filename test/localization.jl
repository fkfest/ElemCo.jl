using ElemCo
using LinearAlgebra

@testset "Canonical localization ordering tie-breakers" begin
  charges = [
    1.0   1.0   0.669 0.669;
    0.0   0.0   0.331 0.331
  ]
  coeffs = [
    0.95  0.10  0.95  0.10;
    0.10  0.95  0.10  0.95;
    0.00  0.00  0.05  0.90;
    0.00  0.00  0.90  0.05
  ]
  labels = ["core", "lone_pair", "bond_a", "bond_b"]

  # Same-atom orbitals have identical charge keys here, so canonical ordering
  # must fall back to the coefficient fingerprint rather than the input order.
  input_order = [2, 4, 1, 3]
  charges_perm = charges[:, input_order]
  coeffs_perm = coeffs[:, input_order]
  labels_perm = labels[input_order]

  perm = ElemCo.OrbLocalization._canonical_orbital_order(
    charges_perm, coeffs_perm, size(charges_perm, 2), size(charges_perm, 1))

  @test labels_perm[perm] == ["core", "lone_pair", "bond_a", "bond_b"]
end

@testset "Orbital Localization Test" begin
  epsilon = 1.e-6

  xyz = "bohr
       O      0.000000000    0.000000000   -0.130186067
       H1     0.000000000    1.489124508    1.033245507
       H2     0.000000000   -1.489124508    1.033245507"

  basis = Dict("ao" => "cc-pVDZ",
               "jkfit" => "cc-pvtz-jkfit",
               "mpfit" => "cc-pvdz-mpfit")

  EC = ElemCo.ECInfo(system = ElemCo.parse_geometry(xyz, basis))

  @dfhf

  # Save canonical orbital energies for comparison
  energies_canon = ElemCo.Wavefunctions.fetch_orbital_energies(EC)

  # Get reference DCSD energy with canonical orbitals
  energies_canon_cc = @cc dcsd
  EDCSD_ref = energies_canon_cc["DCSD"]

  # Test IBO+OPAO localization
  @localize
  cMO, type, bs = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type == "IBO+OPAO"

  # Check orthonormality
  C = cMO[1]
  bao = ElemCo.Integrals.generate_basis(EC, "ao")
  S = ElemCo.Integrals.overlap(bao)
  CtSC = C' * S * C
  @test norm(CtSC - I(size(C, 2))) < 1.e-10

  # Check that energies were rotated (not identical to canonical)
  energies_loc = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
  nocc = length(EC.space['o'])
  # Core orbital energy should be unchanged (not rotated)
  @test energies_loc[1][1] ≈ energies_canon[1][1]
  # Valence occupied energies should differ from canonical
  @test norm(energies_loc[1][2:nocc+1] - energies_canon[1][2:nocc+1]) > 0.01

  # DCSD energy should be invariant under orbital rotation
  energies_loc_cc = @cc dcsd
  @test abs(energies_loc_cc["DCSD"] - EDCSD_ref) < epsilon

  # Test IBO-only localization
  @set loc virtual=false
  @dfhf
  @localize
  cMO2, type2, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type2 == "IBO"

  C2 = cMO2[1]
  CtSC2 = C2' * S * C2
  @test norm(CtSC2 - I(size(C2, 2))) < 1.e-10

  # Compare IBO charges with Molpro IBBA reference
  # Molpro uses MINAO basis, we use ANO-RCC-MB → expect ~0.01 difference
  # Molpro reference (per-orbital IAO charges, occupancy-normalized):
  #   core:    O=1.000
  #   OH bond: O=0.669, H=0.331
  #   LP:      O=1.000
  occ_range = EC.space['o']
  C_loc_occ = C2[:, occ_range]
  C_all_occ = C2[:, 1:maximum(occ_range)]
  C_iao, iao_atoms, natom = ElemCo.OrbLocalization.compute_iaos(EC, C_all_occ)
  Q = C_iao' * S * C_loc_occ
  niao = size(C_iao, 2)
  nocc_loc = size(C_loc_occ, 2)
  charges = zeros(natom, nocc_loc)
  for i in 1:nocc_loc, mu in 1:niao
    A = iao_atoms[mu]
    A == 0 && continue
    charges[A, i] += abs2(Q[mu, i])
  end
  # Each orbital's charges should sum to 1
  for i in 1:nocc_loc
    @test abs(sum(charges[:, i]) - 1.0) < 1.e-10
  end
  # Identify orbital types by O-charge and compare with Molpro
  n_core = 0; n_lp = 0; n_bond = 0
  for i in 1:nocc_loc
    q_O = charges[1, i]
    if q_O > 0.99
      # core or lone pair
      n_lp += 1
      @test q_O ≈ 1.0 atol=0.01
    else
      # OH bond: Molpro q_O=0.669, q_H=0.331
      n_bond += 1
      @test q_O ≈ 0.669 atol=0.01
      q_H = maximum(charges[2:end, i])
      @test q_H ≈ 0.331 atol=0.01
    end
  end
  @test n_lp == 3   # 1 core + 2 lone pairs
  @test n_bond == 2  # 2 OH bonds

  # Test localize_core option
  @set loc virtual=false localize_core=true
  @dfhf
  @localize
  cMO_lc, type_lc, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type_lc == "IBO"
  C_lc = cMO_lc[1]
  # Core orbital should now be localized (different from canonical)
  # but still orthonormal
  CtSC_lc = C_lc' * S * C_lc
  @test norm(CtSC_lc - I(size(C_lc, 2))) < 1.e-10

  # Test Boys localization
  @set loc method="boys" virtual=false localize_core=false
  @dfhf
  @localize
  cMO_boys, type_boys, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type_boys == "Boys"

  C_boys = cMO_boys[1]
  CtSC_boys = C_boys' * S * C_boys
  @test norm(CtSC_boys - I(size(C_boys, 2))) < 1.e-10

  # Boys+OPAO localization
  @set loc method="boys" virtual=true
  @dfhf
  @localize
  cMO_boys2, type_boys2, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type_boys2 == "Boys+OPAO"

  C_boys2 = cMO_boys2[1]
  CtSC_boys2 = C_boys2' * S * C_boys2
  @test norm(CtSC_boys2 - I(size(C_boys2, 2))) < 1.e-10

  # DCSD energy should be invariant under Boys+OPAO rotation
  energies_boys_cc = @cc dcsd
  @test abs(energies_boys_cc["DCSD"] - EDCSD_ref) < epsilon

  # Test PM localization
  @set loc method="pm" virtual=false
  @dfhf
  @localize
  cMO_pm, type_pm, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type_pm == "PM"

  C_pm = cMO_pm[1]
  CtSC_pm = C_pm' * S * C_pm
  @test norm(CtSC_pm - I(size(C_pm, 2))) < 1.e-10
end

@testset "IBO orbital comparison with Molpro IBBA" begin
  # Compare IBO localized orbitals with Molpro's IBBA implementation.
  # Use minimum singular value of the occupied subspace overlap matrix
  # as the primary metric (robust against degenerate orbital rotations).
  # Per-orbital overlaps are tested where degeneracies are absent.

  refdir = joinpath(@__DIR__, "files")

  """
      subspace_overlap(EC, molpro_orbs_file, molpro_overlap_file, nocc)

  Compute subspace and per-orbital overlaps between ElemCo and Molpro IBOs.
  Returns (min_singular_value, per_orbital_overlaps).
  """
  function subspace_overlap(EC, molpro_orbs_file, molpro_overlap_file, nocc)
    C_molpro = ElemCo.Interfaces.import_matrix(EC, molpro_orbs_file)
    S_ref = ElemCo.Interfaces.import_matrix(EC, molpro_overlap_file)
    cMO, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
    C_elemco = cMO[1]
    S_cross = C_elemco[:, 1:nocc]' * S_ref * C_molpro[:, 1:nocc]
    sv = svdvals(S_cross)
    # Per-orbital: greedy matching of |<i|j>|
    per_orb = Float64[]
    for i in 1:nocc
      push!(per_orb, maximum(abs.(S_cross[i, :])))
    end
    return minimum(sv), per_orb
  end

  # --- H2O / cc-pVDZ ---
  geometry = "angstrom
O      0.000000000    0.000000000   -0.068891500
H1     0.000000000    0.788010754    0.546769976
H2     0.000000000   -0.788010754    0.546769976"
  basis = "cc-pVDZ"

  @ECinit
  @dfhf
  @set loc virtual=false localize_core=true
  @localize

  min_sv, per_orb = subspace_overlap(EC,
    joinpath(refdir, "h2o_vdz_ibo_orbs.dat"),
    joinpath(refdir, "h2o_vdz_overlap.dat"), 5)
  @test min_sv > 0.999  # subspace essentially identical

  # --- CH2O / cc-pVDZ (formaldehyde, has π bond) ---
  geometry = "angstrom
C      0.000000000    0.000000000   -0.529177000
O      0.000000000    0.000000000    0.667323000
H      0.000000000    0.935307000   -1.109577000
H      0.000000000   -0.935307000   -1.109577000"
  basis = "cc-pVDZ"

  @ECinit
  @dfhf
  @set loc virtual=false localize_core=true
  @localize

  min_sv, per_orb = subspace_overlap(EC,
    joinpath(refdir, "ch2o_vdz_ibo_orbs.dat"),
    joinpath(refdir, "ch2o_vdz_overlap.dat"), 8)
  @test min_sv > 0.999
  @test minimum(per_orb) > 0.98  # no degeneracy in CH2O

  # --- C2H4 / cc-pVDZ (ethylene, conjugated π) ---
  geometry = "angstrom
C     0.000000000    0.000000000    0.665850000
C     0.000000000    0.000000000   -0.665850000
H     0.000000000    0.922683000    1.232790000
H     0.000000000   -0.922683000    1.232790000
H     0.000000000    0.922683000   -1.232790000
H     0.000000000   -0.922683000   -1.232790000"
  basis = "cc-pVDZ"

  @ECinit
  @dfhf
  @set loc virtual=false localize_core=true
  @localize

  min_sv, per_orb = subspace_overlap(EC,
    joinpath(refdir, "c2h4_vdz_ibo_orbs.dat"),
    joinpath(refdir, "c2h4_vdz_overlap.dat"), 8)
  @test min_sv > 0.999
  @test minimum(per_orb) > 0.99  # all orbitals match with core localized
end

@testset "Boys orbital comparison with Molpro" begin
  # Compare Foster-Boys localized orbitals with Molpro's LOCALI,BOYS implementation.
  # Use minimum singular value of the occupied subspace overlap matrix
  # as the primary metric (robust against degenerate orbital rotations).

  refdir = joinpath(@__DIR__, "files")

  """
      subspace_overlap_boys(EC, molpro_orbs_file, molpro_overlap_file, nocc)

  Compute subspace and per-orbital overlaps between ElemCo and Molpro Boys orbitals.
  Returns (min_singular_value, per_orbital_overlaps).
  """
  function subspace_overlap_boys(EC, molpro_orbs_file, molpro_overlap_file, nocc)
    C_molpro = ElemCo.Interfaces.import_matrix(EC, molpro_orbs_file)
    S_ref = ElemCo.Interfaces.import_matrix(EC, molpro_overlap_file)
    cMO, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
    C_elemco = cMO[1]
    S_cross = C_elemco[:, 1:nocc]' * S_ref * C_molpro[:, 1:nocc]
    sv = svdvals(S_cross)
    per_orb = Float64[]
    for i in 1:nocc
      push!(per_orb, maximum(abs.(S_cross[i, :])))
    end
    return minimum(sv), per_orb
  end

  # --- H2O / cc-pVDZ ---
  geometry = "angstrom
O      0.000000000    0.000000000   -0.068891500
H1     0.000000000    0.788010754    0.546769976
H2     0.000000000   -0.788010754    0.546769976"
  basis = "cc-pVDZ"

  @ECinit
  @dfhf
  @set loc method="boys" virtual=false localize_core=true
  @localize

  min_sv, per_orb = subspace_overlap_boys(EC,
    joinpath(refdir, "h2o_vdz_boys_orbs.dat"),
    joinpath(refdir, "h2o_vdz_boys_overlap.dat"), 5)
  @test min_sv > 0.999  # subspace essentially identical

  # --- CH2O / cc-pVDZ (formaldehyde, has π bond) ---
  geometry = "angstrom
C      0.000000000    0.000000000   -0.529177000
O      0.000000000    0.000000000    0.667323000
H      0.000000000    0.935307000   -1.109577000
H      0.000000000   -0.935307000   -1.109577000"
  basis = "cc-pVDZ"

  @ECinit
  @dfhf
  @set loc method="boys" virtual=false localize_core=true
  @localize

  min_sv, per_orb = subspace_overlap_boys(EC,
    joinpath(refdir, "ch2o_vdz_boys_orbs.dat"),
    joinpath(refdir, "ch2o_vdz_boys_overlap.dat"), 8)
  @test min_sv > 0.999
  @test minimum(per_orb) > 0.98  # no degeneracy in CH2O
end
