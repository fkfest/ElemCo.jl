@testitem "localization" tags=[:df, :quick] begin
using ElemCo
using ElemCo.TrexioInterface
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

@testset "Pi local planes use bonded neighbors" begin
  xyz = "angstrom
       O1    0.000000000    0.000000000    0.000000000
       C1    1.220000000    0.000000000    0.000000000
       H1    1.820000000    0.940000000    0.000000000
       H2    1.820000000   -0.940000000    0.000000000
       Ne1   0.000000000    0.000000000    1.900000000"

  EC = ElemCo.ECInfo(system = ElemCo.parse_geometry(xyz, Dict("ao" => "sto-3g")))

  bonded = ElemCo.OrbRegion._bonded_neighbors(EC, 1)
  @test [ElemCo.MSystems.atomic_centre_label(EC.system[idx]) for idx in bonded] == ["C1"]

  normal = ElemCo.OrbRegion._local_plane_normal(EC, 1)
  @test abs(normal[3]) > 0.99
  @test abs(normal[1]) < 1.e-8
  @test abs(normal[2]) < 1.e-8
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
  bao = ElemCo.Integrals.generate_basis(EC, "ao")
  S = ElemCo.Integrals.overlap(bao)
  cMO_canon, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  C_occ_canon = cMO_canon[1][:, EC.space['o']]
  C_iao_canon, iao_atoms_canon, natom_canon = ElemCo.OrbLocalization.compute_iaos(EC, C_occ_canon)
  R_ibo_canon, charges_ibo_canon = ElemCo.OrbLocalization.localize_ibo(
    C_occ_canon, S, C_iao_canon, iao_atoms_canon, natom_canon; exponent=4)
  @test size(R_ibo_canon) == (length(EC.space['o']), length(EC.space['o']))
  @test size(charges_ibo_canon) == (natom_canon, length(EC.space['o']))
  for i in axes(charges_ibo_canon, 2)
    @test abs(sum(charges_ibo_canon[:, i]) - 1.0) < 1.e-10
  end

  # Get reference DCSD energy with canonical orbitals
  energies_canon_cc = @cc dcsd
  EDCSD_ref = energies_canon_cc["DCSD"]

  # Test IBO+OPAO localization
  @localize
  cMO, type, bs = ElemCo.Wavefunctions.fetch_orbitals(EC)
  @test type == "IBO+OPAO"

  # Check orthonormality
  C = cMO[1]
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

@testset "Region-tagged orbital dumps" begin
  function load_region_dump(path)
    io = open_trexio(path, "r")
    try
      dump_basis = read_trexio_basis(io)
      orbitals, type = read_trexio_orbitals(io, dump_basis)
      classes = read_trexio_orbital_classes(io)
      return dump_basis, orbitals, type, classes
    finally
      close_trexio(io)
    end
  end

  function reconstruct_fock(cMO, energies, S)
    return S * cMO * Diagonal(eltype(cMO).(energies)) * cMO' * S
  end

  function load_region_fock(path)
    io = open_trexio(path, "r")
    try
      dump_basis = read_trexio_basis(io)
      return read_trexio_ao_fock(io, dump_basis)
    finally
      close_trexio(io)
    end
  end

  offdiag_norm(mat) = norm(mat - Diagonal(diag(mat)))

  # The region-selected (Inactive) occupied orbitals must sit at the Fermi level, i.e.
  # the Core block is contiguous from index 1 and the Inactive block immediately follows.
  function check_region_occ_ordering(cls)
    core = findall(==("Core"), cls)
    inact = findall(==("Inactive"), cls)
    @test core == collect(1:length(core))
    @test inact == collect(length(core)+1 : length(core)+length(inact))
  end

  geometry = "bohr
       O      0.000000000    0.000000000   -0.130186067
       H1     0.000000000    1.489124508    1.033245507
       H2     0.000000000   -1.489124508    1.033245507"

  basis = Dict("ao" => "cc-pVDZ",
               "jkfit" => "cc-pvtz-jkfit",
               "mpfit" => "cc-pvdz-mpfit")

  @ECinit
  @dfhf

  inclusive_store = "region_inclusive.h5"
  @set wf store=inclusive_store
  @set region mode=:inclusive occ_charge_thr=0.2 atom_charge_thr=0.2
  @region [2]

  basis_inc, cMO_inc, type_inc, classes_inc = load_region_dump(joinpath(EC.scr, inclusive_store))
  @test type_inc == "Region-IBO+OPAO"
  classa_inc = classes_inc[1]
  # water has 5 occupied orbitals; 1 is selected, the remaining 4 become Core (frozen)
  @test count(==("Core"), classa_inc) == 4
  @test count(==("Inactive"), classa_inc) == 1
  @test count(==("Virtual"), classa_inc) > 0
  check_region_occ_ordering(classa_inc)
  S_inc = ElemCo.Integrals.overlap(basis_inc)
  @test norm(cMO_inc[1]' * S_inc * cMO_inc[1] - I(size(cMO_inc[1], 2))) < 1.e-10

  legacy_virtual_store = "region_inclusive_legacy_virtuals.h5"
  @set wf store=legacy_virtual_store
  @set region mode=:inclusive virtual=:support_opao occ_charge_thr=0.2 atom_charge_thr=0.2
  @region [2]

  basis_inc_legacy, cMO_inc_legacy, _, classes_inc_legacy = load_region_dump(joinpath(EC.scr, legacy_virtual_store))
  classa_inc_legacy = classes_inc_legacy[1]
  if count(==("Virtual"), classa_inc) == count(==("Virtual"), classa_inc_legacy)
    virt_inc = findall(==("Virtual"), classa_inc)
    virt_inc_legacy = findall(==("Virtual"), classa_inc_legacy)
    S_inc_legacy = ElemCo.Integrals.overlap(basis_inc_legacy)
    overlap_diag = abs.(diag(cMO_inc[1][:, virt_inc]' * S_inc_legacy * cMO_inc_legacy[1][:, virt_inc_legacy]))
    @test minimum(overlap_diag) < 0.95
  else
    # support_opao prunes the redundant fragment PAO (a clean ~1e-8 eigenvalue of the
    # O+H support-PAO overlap that the old absolute-threshold orthogonalization kept),
    # so it yields one fewer active virtual than the complement construction, which
    # spans the full virtual space via its IAO-antibonding targets.
    @test count(==("Virtual"), classa_inc) > count(==("Virtual"), classa_inc_legacy)
  end

  basis = Dict("ao" => "cc-pVDZ",
               "jkfit" => "cc-pvtz-jkfit",
               "mpfit" => "cc-pvdz-mpfit")
  exclusive_store = "region_exclusive.h5"
  @set wf store=exclusive_store
  @set region mode=:exclusive occ_charge_thr=0.2 atom_charge_thr=0.2
  @region [1, 2]

  _, _, _, classes_exc = load_region_dump(joinpath(EC.scr, exclusive_store))
  classa_exc = classes_exc[1]
  # 3 occupied orbitals are selected; the remaining 2 of the 5 become Core (frozen)
  @test count(==("Core"), classa_exc) == 2
  @test count(==("Inactive"), classa_exc) == 3
  @test count(==("Virtual"), classa_exc) > 0
  check_region_occ_ordering(classa_exc)

  oxygen_exclusive_store = "region_exclusive_oxygen.h5"
  @set wf store=oxygen_exclusive_store
  @set region mode=:exclusive occ_charge_thr=0.2 atom_charge_thr=0.2
  @region [1]

  _, _, _, classes_ox_exc = load_region_dump(joinpath(EC.scr, oxygen_exclusive_store))
  classa_ox_exc = classes_ox_exc[1]
  @test count(==("Inactive"), classa_ox_exc) > 0

  mixed_store = "region_mixed_centers.h5"
  @set wf store=mixed_store
  @set region mode=:inclusive occ_charge_thr=0.2 atom_charge_thr=0.2
  @region begin
    @set region inclusive_centers=[2] exclusive_centers=[1]
  end

  _, _, _, classes_mixed = load_region_dump(joinpath(EC.scr, mixed_store))
  classa_mixed = classes_mixed[1]
  @test count(==("Inactive"), classa_mixed) ==
    count(==("Inactive"), classa_inc) + count(==("Inactive"), classa_ox_exc)

  cMO_ref, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  energies_ref = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
  # the HF dump stores the exact converged AO Fock; it must agree with the canonical reconstruction
  F_stored_sm = ElemCo.Wavefunctions.fetch_ao_fock(EC)
  @test !isnothing(F_stored_sm)
  F_ref = F_stored_sm[1]
  @test norm(F_ref - reconstruct_fock(cMO_ref[1], energies_ref[1], S_inc)) < 1.e-3
  pseudo_store = "region_pseudo.h5"
  @set wf store=pseudo_store
  @set region mode=:exclusive occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=true
  @region [1, 2]

  _, cMO_pseudo, type_pseudo, classes_pseudo = load_region_dump(joinpath(EC.scr, pseudo_store))
  @test type_pseudo == "Region-IBO+OPAO-Pseudo"
  frag_occ = findall(==("Inactive"), classes_pseudo[1])
  frag_virt = findall(==("Virtual"), classes_pseudo[1])
  # pseudo uses the exact stored Fock, so the fragment blocks diagonalize it to machine precision
  @test offdiag_norm(cMO_pseudo[1][:, frag_occ]' * F_ref * cMO_pseudo[1][:, frag_occ]) < 1.e-9
  @test offdiag_norm(cMO_pseudo[1][:, frag_virt]' * F_ref * cMO_pseudo[1][:, frag_virt]) < 1.e-9
  # the region dump carries the AO Fock forward unchanged (for chained pseudo runs)
  F_carried = load_region_fock(joinpath(EC.scr, pseudo_store))
  @test !isnothing(F_carried)
  @test norm(F_carried[1] - F_ref) < 1.e-12

  # Regression for the chained-region pseudo bug: run region WITHOUT pseudo, then run region WITH
  # pseudo reading that first (non-canonical, IBO) dump. The carried-forward Fock lets the second
  # run pseudo-canonicalize against the TRUE Fock instead of the IBO-basis Fock diagonal.
  chain1_store = "region_chain_nopseudo.h5"
  @set wf store=chain1_store
  @set region mode=:exclusive occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=false
  @region [1, 2]
  _, cMO_chain1, _, classes_chain1 = load_region_dump(joinpath(EC.scr, chain1_store))
  frag_occ_chain1 = findall(==("Inactive"), classes_chain1[1])
  # the localized (no-pseudo) region orbitals are IBOs: their fragment Fock block is NOT diagonal
  @test offdiag_norm(cMO_chain1[1][:, frag_occ_chain1]' * F_ref * cMO_chain1[1][:, frag_occ_chain1]) > 1.e-3
  F_chain1 = load_region_fock(joinpath(EC.scr, chain1_store))
  @test !isnothing(F_chain1)
  @test norm(F_chain1[1] - F_ref) < 1.e-12

  chain2_store = "region_chain_pseudo.h5"
  @set wf start=chain1_store store=chain2_store
  @set region mode=:exclusive occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=true
  @region [1, 2]
  @set wf start=""
  _, cMO_chain2, _, classes_chain2 = load_region_dump(joinpath(EC.scr, chain2_store))
  frag_occ_chain2 = findall(==("Inactive"), classes_chain2[1])
  # using the carried Fock, pseudo diagonalizes the TRUE Fock block to machine precision
  @test offdiag_norm(cMO_chain2[1][:, frag_occ_chain2]' * F_ref * cMO_chain2[1][:, frag_occ_chain2]) < 1.e-9

  # region.pao_centers extends the virtual-space support: with the auto support disabled
  # (high atom_charge_thr), adding atom 3 (H2) as a PAO center adds OPAO virtuals.
  pao_base_store = "region_pao_base.h5"
  @set wf store=pao_base_store
  @set region mode=:inclusive virtual=:support_opao occ_charge_thr=0.2 atom_charge_thr=10.0 pseudo=false pao_centers=Int[]
  @region [2]
  _, _, _, classes_pao_base = load_region_dump(joinpath(EC.scr, pao_base_store))
  nvirt_pao_base = count(==("Virtual"), classes_pao_base[1])

  pao_ext_store = "region_pao_ext.h5"
  @set wf store=pao_ext_store
  @set region pao_centers=[3]
  @region [2]
  _, _, _, classes_pao_ext = load_region_dump(joinpath(EC.scr, pao_ext_store))
  nvirt_pao_ext = count(==("Virtual"), classes_pao_ext[1])
  @test nvirt_pao_ext > nvirt_pao_base
  @set region virtual=:complement atom_charge_thr=0.2 pao_centers=Int[]

  basis = Dict("ao" => "cc-pVDZ",
               "jkfit" => "cc-pvtz-jkfit",
               "mpfit" => "cc-pvdz-mpfit")
  unrestricted_store = "region_uhf.h5"
  @set wf store="" ms2=2
  @set region mode=:inclusive occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=false
  @dfuhf
  @set wf store=unrestricted_store
  @set region mode=:inclusive occ_charge_thr=0.2 atom_charge_thr=0.2
  @region [1]

  basis_uhf, cMO_uhf, type_uhf, classes_uhf = load_region_dump(joinpath(EC.scr, unrestricted_store))
  @test type_uhf == "Region-IBO+OPAO"
  classa_uhf, classb_uhf = classes_uhf
  # environment occupied orbitals are now frozen as Core in both spin blocks
  @test count(==("Core"), classa_uhf) >= 1
  @test count(==("Core"), classb_uhf) >= 1
  @test count(==("Inactive"), classa_uhf) > 0
  @test count(==("Inactive"), classb_uhf) > 0
  @test count(==("Virtual"), classa_uhf) > 0
  @test count(==("Virtual"), classb_uhf) > 0
  check_region_occ_ordering(classa_uhf)
  check_region_occ_ordering(classb_uhf)
  S_uhf = ElemCo.Integrals.overlap(basis_uhf)
  @test norm(cMO_uhf[1]' * S_uhf * cMO_uhf[1] - I(size(cMO_uhf[1], 2))) < 1.e-10
  @test norm(cMO_uhf[2]' * S_uhf * cMO_uhf[2] - I(size(cMO_uhf[2], 2))) < 1.e-10

  cMO_uhf_ref, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  energies_uhf_ref = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
  # the UHF dump stores the exact converged alpha/beta AO Fock matrices
  F_uhf_sm = ElemCo.Wavefunctions.fetch_ao_fock(EC)
  @test !isnothing(F_uhf_sm)
  @test !ElemCo.QMTensors.is_restricted(F_uhf_sm)
  F_uhf_a = F_uhf_sm[1]
  F_uhf_b = F_uhf_sm[2]
  @test norm(F_uhf_a - reconstruct_fock(cMO_uhf_ref[1], energies_uhf_ref[1], S_uhf)) < 1.e-3
  @test norm(F_uhf_b - reconstruct_fock(cMO_uhf_ref[2], energies_uhf_ref[2], S_uhf)) < 1.e-3
  pseudo_uhf_store = "region_uhf_pseudo.h5"
  @set wf store=pseudo_uhf_store
  @set region mode=:inclusive occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=true
  @region [1]

  _, cMO_uhf_pseudo, type_uhf_pseudo, classes_uhf_pseudo = load_region_dump(joinpath(EC.scr, pseudo_uhf_store))
  @test type_uhf_pseudo == "Region-IBO+OPAO-Pseudo"
  frag_occ_a = findall(==("Inactive"), classes_uhf_pseudo[1])
  frag_virt_a = findall(==("Virtual"), classes_uhf_pseudo[1])
  frag_occ_b = findall(==("Inactive"), classes_uhf_pseudo[2])
  frag_virt_b = findall(==("Virtual"), classes_uhf_pseudo[2])
  @test offdiag_norm(cMO_uhf_pseudo[1][:, frag_occ_a]' * F_uhf_a * cMO_uhf_pseudo[1][:, frag_occ_a]) < 1.e-9
  @test offdiag_norm(cMO_uhf_pseudo[1][:, frag_virt_a]' * F_uhf_a * cMO_uhf_pseudo[1][:, frag_virt_a]) < 1.e-9
  @test offdiag_norm(cMO_uhf_pseudo[2][:, frag_occ_b]' * F_uhf_b * cMO_uhf_pseudo[2][:, frag_occ_b]) < 1.e-9
  @test offdiag_norm(cMO_uhf_pseudo[2][:, frag_virt_b]' * F_uhf_b * cMO_uhf_pseudo[2][:, frag_virt_b]) < 1.e-9

  @set wf store="" ms2=0
  @set region mode=:inclusive virtual=:complement occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=false
end

@testset "Pi-system region selection" begin
  function load_pi_region_dump(path)
    io = open_trexio(path, "r")
    try
      dump_basis = read_trexio_basis(io)
      orbitals, type = read_trexio_orbitals(io, dump_basis)
      classes = read_trexio_orbital_classes(io)
      return dump_basis, orbitals, type, classes
    finally
      close_trexio(io)
    end
  end

  reconstruct_fock(cMO, energies, S) = S * cMO * Diagonal(eltype(cMO).(energies)) * cMO' * S
  offdiag_norm(mat) = norm(mat - Diagonal(diag(mat)))

  geometry = "angstrom
C1    -2.602000000    0.000000000    0.000000000
C2    -0.867000000    0.000000000    0.000000000
C3     0.867000000    0.000000000    0.000000000
C4     2.602000000    0.000000000    0.000000000
H1    -3.166000000    0.929000000    0.000000000
H2    -3.166000000   -0.929000000    0.000000000
H3    -0.302000000    0.929000000    0.000000000
H4     0.302000000   -0.929000000    0.000000000
H5     3.166000000    0.929000000    0.000000000
H6     3.166000000   -0.929000000    0.000000000"
  basis = Dict("ao"=>"cc-pVDZ",
               "jkfit"=>"cc-pvtz-jkfit",
               "mpfit"=>"cc-pvdz-mpfit")

  @ECinit
  @dfhf
  cMO_ref, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  energies_ref = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
  S_ref = ElemCo.Integrals.overlap(ElemCo.Integrals.generate_basis(EC, "ao"))
  # pseudo uses the exact stored AO Fock; verify it matches the canonical reconstruction
  F_ref = ElemCo.Wavefunctions.fetch_ao_fock(EC)[1]
  @test norm(F_ref - reconstruct_fock(cMO_ref[1], energies_ref[1], S_ref)) < 1.e-3

  pi_both_store = "region_pi_both.h5"
  @set wf store=pi_both_store
  @set region pi=:both pseudo=true
  @region [1, 2, 3, 4]

  basis_pi, cMO_pi, type_pi, classes_pi = load_pi_region_dump(joinpath(EC.scr, pi_both_store))
  @test type_pi == "Region-PiOS"
  classa_pi = classes_pi[1]
  @test count(==("Inactive"), classa_pi) == 2
  @test count(==("Virtual"), classa_pi) == 2
  @test norm(cMO_pi[1]' * ElemCo.Integrals.overlap(basis_pi) * cMO_pi[1] - I(size(cMO_pi[1], 2))) < 1.e-10
  pi_occ = findall(==("Inactive"), classa_pi)
  pi_virt = findall(==("Virtual"), classa_pi)
  @test offdiag_norm(cMO_pi[1][:, pi_occ]' * F_ref * cMO_pi[1][:, pi_occ]) < 1.e-10
  @test offdiag_norm(cMO_pi[1][:, pi_virt]' * F_ref * cMO_pi[1][:, pi_virt]) < 1.e-10

  # region.pao_centers also augments the π=:both virtual space with OPAOs (orthogonal to the
  # π virtuals); the occupied π space is unchanged.
  pi_pao_store = "region_pi_both_pao.h5"
  @set wf store=pi_pao_store
  @set region pi=:both pseudo=false pao_centers=[5]
  @region [1, 2, 3, 4]
  basis_pi_pao, cMO_pi_pao, _, classes_pi_pao = load_pi_region_dump(joinpath(EC.scr, pi_pao_store))
  @test count(==("Inactive"), classes_pi_pao[1]) == 2
  @test count(==("Virtual"), classes_pi_pao[1]) > 2
  @test norm(cMO_pi_pao[1]' * ElemCo.Integrals.overlap(basis_pi_pao) * cMO_pi_pao[1] - I(size(cMO_pi_pao[1], 2))) < 1.e-10
  @set region pao_centers=Int[]

  pi_frontier_store = "region_pi_frontier.h5"
  @set wf store=pi_frontier_store
  @set region pi=:both pi_occupied=1 pi_virtual=1 pseudo=false
  @region [1, 2, 3, 4]

  _, _, type_pi_frontier, classes_pi_frontier = load_pi_region_dump(joinpath(EC.scr, pi_frontier_store))
  @test type_pi_frontier == "Region-PiOS"
  classa_pi_frontier = classes_pi_frontier[1]
  @test count(==("Inactive"), classa_pi_frontier) == 1
  @test count(==("Virtual"), classa_pi_frontier) == 1

  pi_electron_override_store = "region_pi_electron_override.h5"
  @set wf store=pi_electron_override_store
  @set region pi=:both pi_electrons=2 pi_occupied=-1 pi_virtual=-1 pseudo=false
  @region [1, 2, 3, 4]

  _, _, type_pi_override, classes_pi_override = load_pi_region_dump(joinpath(EC.scr, pi_electron_override_store))
  @test type_pi_override == "Region-PiOS"
  classa_pi_override = classes_pi_override[1]
  @test count(==("Inactive"), classa_pi_override) == 1
  @test count(==("Virtual"), classa_pi_override) == 3

  pi_occ_store = "region_pi_occ.h5"
  @set wf store=pi_occ_store
  @set region pi=:occupied pi_electrons=-1 pi_occupied=-1 pi_virtual=-1 pseudo=true
  @region [1, 2, 3, 4]

  _, cMO_pi_occ, type_pi_occ, classes_pi_occ = load_pi_region_dump(joinpath(EC.scr, pi_occ_store))
  @test type_pi_occ == "Region-PiOcc+OPAO-Pseudo"
  classa_pi_occ = classes_pi_occ[1]
  @test count(==("Inactive"), classa_pi_occ) == 2
  @test count(==("Virtual"), classa_pi_occ) > 2
  pi_occ_only = findall(==("Inactive"), classa_pi_occ)
  @test offdiag_norm(cMO_pi_occ[1][:, pi_occ_only]' * F_ref * cMO_pi_occ[1][:, pi_occ_only]) < 1.e-10

  geometry = "angstrom
C     0.000000000    0.000000000    0.665850000
O     0.000000000    0.000000000   -0.665850000
H1    0.000000000    0.922683000    1.232790000
H2    0.000000000   -0.922683000    1.232790000"
  @ECinit
  @dfhf
  cMO_ref_co, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
  energies_ref_co = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
  S_ref_co = ElemCo.Integrals.overlap(ElemCo.Integrals.generate_basis(EC, "ao"))
  # pseudo uses the exact stored AO Fock; verify it matches the canonical reconstruction
  F_ref_co = ElemCo.Wavefunctions.fetch_ao_fock(EC)[1]
  @test norm(F_ref_co - reconstruct_fock(cMO_ref_co[1], energies_ref_co[1], S_ref_co)) < 1.e-3

  pi_carbonyl_store = "region_pi_carbonyl.h5"
  @set wf store=pi_carbonyl_store
  @set region pi=:both pseudo=true
  @region [1, 2]

  _, cMO_pi_co, type_pi_co, classes_pi_co = load_pi_region_dump(joinpath(EC.scr, pi_carbonyl_store))
  @test type_pi_co == "Region-PiOS"
  classa_pi_co = classes_pi_co[1]
  @test count(==("Inactive"), classa_pi_co) == 1
  @test count(==("Virtual"), classa_pi_co) == 1
  pi_occ_co = findall(==("Inactive"), classa_pi_co)
  pi_virt_co = findall(==("Virtual"), classa_pi_co)
  @test offdiag_norm(cMO_pi_co[1][:, pi_occ_co]' * F_ref_co * cMO_pi_co[1][:, pi_occ_co]) < 1.e-10
  @test offdiag_norm(cMO_pi_co[1][:, pi_virt_co]' * F_ref_co * cMO_pi_co[1][:, pi_virt_co]) < 1.e-10

  @set wf store=""
  @set region pi=:none pi_electrons=-1 pi_occupied=-1 pi_virtual=-1 pseudo=false
end
end

@testitem "OPAO redundancy removal (diffuse basis)" tags=[:df, :quick] begin
using ElemCo
using ElemCo.Integrals: overlap, generate_basis
using ElemCo.OrbTools: select_lowdin_orth
using ElemCo.OrbLocalization: compute_ao_atoms
using ElemCo.OrbRegion: _fragment_opao_rotation
using LinearAlgebra

# With a diffuse/augmented basis the O-centered PAOs projected onto the virtual space
# carry a near-linear-dependency (eigenvalue ~1e-7*λmax). The old absolute-threshold
# ALPACA orthogonalization kept it and amplified it ~1000x into a junk active virtual.
# select_lowdin_orth detects it with a RELATIVE threshold (loc.opaothr) and drops it,
# while keeping the OPAOs atom-centered (symmetric Löwdin on the selected pivots).
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao" => "avdz")

@ECinit
@dfhf

cMO, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
S = overlap(generate_basis(EC, "ao"))
cv = cMO[1][:, EC.space['v']]
nvirt = size(cv, 2)
ao_atoms, natom = compute_ao_atoms(EC)
o_aos = findall(==(1), ao_atoms)                # O is atom 1

# PAO overlap of the O fragment: there is a clean ~1000x gap above one redundancy
S_PAO = Hermitian(transpose(cv * (transpose(cv) * S[:, o_aos])) * S * (cv * (transpose(cv) * S[:, o_aos])))
ev = eigvals(S_PAO)                              # sorted ascending for a Hermitian matrix
@test count(>(1e-5 * ev[end]), ev) == 22        # eigen rank at the default threshold
@test count(>(1e-7 * ev[end]), ev) == 23        # one near-redundant direction sits in (1e-7,1e-5)*λmax

# select_lowdin_orth returns exactly the non-redundant, S-orthonormal set
M = select_lowdin_orth(S_PAO; relthr=EC.options.loc.opaothr)
@test size(M, 2) == 22
COPAO = (cv * (transpose(cv) * S[:, o_aos])) * M
@test maximum(abs.(transpose(COPAO) * S * COPAO - I)) < 1e-8   # well-conditioned, unlike the kept-23 case

# the real fragment-OPAO rotation drops the redundancy and stays a square, orthogonal rotation
R, nfrag = _fragment_opao_rotation(cv, S, ao_atoms, [1]; relthr=EC.options.loc.opaothr)
@test nfrag == 22
@test size(R) == (nvirt, nvirt)
@test maximum(abs.(transpose(R) * R - I)) < 1e-8
# a looser threshold keeps the junk orbital (documents the behaviour being fixed)
@test _fragment_opao_rotation(cv, S, ao_atoms, [1]; relthr=1e-12)[2] == 23

# the 22 active OPAOs stay local (predominantly O-centered)
Cact = cv * R[:, 1:nfrag]
SC = S * Cact
maxpop = [maximum([sum(real(Cact[mu, j] * SC[mu, j]) for mu in findall(==(a), ao_atoms)) for a in 1:natom]) for j in 1:nfrag]
@test sum(maxpop) / nfrag > 0.9                  # mean dominant-atom population (~0.97)
end
