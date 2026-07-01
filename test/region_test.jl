@testitem "region downstream class honoring" tags=[:df, :region, :quick] begin
using ElemCo
using ElemCo.TrexioInterface
using LinearAlgebra

# Pseudo-canonicalized region MP2: the region active occupied/virtual blocks are
# semicanonicalized, so the MP2 is well defined (a raw localized/OPAO active space has
# no meaningful orbital energies for the MP2 denominators) and invariant to the OPAO
# orthogonalization scheme.
E_MP2 = -76.0441333320342

geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao" => "cc-pVDZ",
             "jkfit" => "cc-pvtz-jkfit",
             "mpfit" => "cc-pvdz-mpfit")

@ECinit
@dfhf

# Apply the driver's freeze logic (freeze_orbitals! handles core, redundant, and the
# freeze_nvirt count) to the current dump and return the resulting (n_active_occ, n_active_virt);
# restore the space + options.
function active_after_freeze(EC; core=:auto, freeze_nocc=-1, freeze_nvirt=-1)
  sp = ElemCo.save_space(EC)
  c0, fn0, fv0 = EC.options.wf.core, EC.options.wf.freeze_nocc, EC.options.wf.freeze_nvirt
  EC.options.wf.core, EC.options.wf.freeze_nocc, EC.options.wf.freeze_nvirt = core, freeze_nocc, freeze_nvirt
  ElemCo.ECInfos.setup_space_system!(EC; verbose=false)
  ElemCo.OrbTools.freeze_orbitals!(EC; verbose=false)
  res = (length(EC.space['o']), length(EC.space['v']))
  ElemCo.restore_space!(EC, sp)
  EC.options.wf.core, EC.options.wf.freeze_nocc, EC.options.wf.freeze_nvirt = c0, fn0, fv0
  return res
end

# water/cc-pVDZ: 5 occupied, 19 virtual, chemical (:large) core = 1 (O 1s).
# Ordinary dump: :auto reproduces the standard :large frozen core; explicit settings override.
@test active_after_freeze(EC; core=:auto)[1]  == 4   # :auto -> chemical core (1 frozen)
@test active_after_freeze(EC; core=:large)[1] == 4
@test active_after_freeze(EC; core=:none)[1]  == 5   # override: nothing frozen
@test active_after_freeze(EC; freeze_nocc=2)[1] == 3 # override: freeze 2 lowest

# Region dump (written back to wf.dump). pseudo-canonicalize so the downstream MP2 has
# well-defined orbital energies (see E_MP2 note above).
@set region mode=:inclusive occ_charge_thr=0.2 atom_charge_thr=0.2 pseudo=true
@region [2]
io = open_trexio(joinpath(EC.scr, EC.options.wf.dump), "r")
classa = try
  read_trexio_orbital_classes(io)[1]
finally
  close_trexio(io)
end
n_inact = count(==("Inactive"), classa)
n_virt = count(==("Virtual"), classa)
@test n_inact == 1

# :auto honors the region; user settings override the region's prescription.
@test active_after_freeze(EC; core=:auto)       == (n_inact, n_virt)  # region active space
@test active_after_freeze(EC; core=:none)[1]    == 5                  # override core: correlate all occ
@test active_after_freeze(EC; freeze_nocc=2)[1] == 3                  # override core: freeze 2 lowest
@test active_after_freeze(EC; freeze_nvirt=5)[2] == n_virt - 5        # override virt: freeze 5 highest

# end-to-end: default :auto restricts @dfmp2 to the region
energies = @dfmp2
@test abs(energies["MP2"] - E_MP2) < 1e-8
end

@testitem "region redundant freezing by index" tags=[:df, :region, :quick] begin
using ElemCo
using ElemCo.TrexioInterface
using LinearAlgebra

geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao" => "cc-pVDZ", "jkfit" => "cc-pvtz-jkfit", "mpfit" => "cc-pvdz-mpfit")

@ECinit
@dfhf

cMO, _, basis_ao = ElemCo.Wavefunctions.fetch_orbitals(EC)
en = ElemCo.Wavefunctions.fetch_orbital_energies(EC)
occ = ElemCo.Wavefunctions.fetch_orbital_occupations(EC)
norb = size(cMO[1], 2)

# Craft a region-like dump in which two redundant (sentinel-energy) "Deleted" orbitals sit at
# LOW virtual indices (6,7) while the active "Virtual" orbitals occupy the HIGH indices. A
# top-index freeze (the old freeze_nvirt!(nredund) approach) would wrongly delete the
# high-index region virtuals; freeze_orbitals! must remove the redundant orbitals by their
# actual indices instead.
classes = fill("Virtual", norb)
classes[1] = "Core"
classes[2:5] .= "Inactive"
classes[6:7] .= "Deleted"
en_mod = copy(en[1])
en_mod[6:7] .= ElemCo.OrbTools.REDUNDANT_ORBITAL_ENERGY
ElemCo.Wavefunctions.dump_orbitals(EC, ElemCo.QMTensors.SpinMatrix(cMO[1]);
  basis=basis_ao, type="Region-test",
  energies=(en_mod, Float64[]), occupations=occ, classes=(classes, String[]))

sp = ElemCo.save_space(EC)
ElemCo.ECInfos.setup_space_system!(EC; verbose=false)
ElemCo.OrbTools.freeze_orbitals!(EC; verbose=false)
@test EC.space['o'] == [2, 3, 4, 5]                  # "Core" (1) frozen, region occ kept
@test 6 ∉ EC.space['v'] && 7 ∉ EC.space['v']         # redundant removed by their actual indices
@test issubset([8, norb], EC.space['v'])             # high-index region virtuals NOT frozen by mistake
@test length(EC.space['v']) == norb - 7              # 1 core + 4 inactive + 2 redundant removed
ElemCo.restore_space!(EC, sp)
end

@testitem "region ghost PAO centers" tags=[:df, :region, :quick] begin
using ElemCo
using ElemCo.TrexioInterface
using ElemCo.Integrals: overlap, generate_basis, ao_list
using LinearAlgebra

# water plus an extra ghost H ("H3") above O: basis functions only, no electrons, so the
# molecule stays neutral closed-shell. The ghost carries sizable density from the O orbitals.
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507
     H3     0.000000000    0.000000000    2.000000000"
basis = Dict("ao" => "cc-pVDZ", "jkfit" => "cc-pvtz-jkfit", "mpfit" => "cc-pvdz-mpfit")

@ECinit
@dummy ["H3"]
@dfhf

ghost = 4   # global atom index of the ghost H3
basis_ao = generate_basis(EC, "ao")
S = overlap(basis_ao)
ao_atoms_global = Int[Int(ao.icentre) for ao in ao_list(basis_ao)]
ghost_aos = [i for i in eachindex(ao_atoms_global) if ao_atoms_global[i] == ghost]
@test ElemCo.MSystems.is_dummy(EC.system[ghost])

# total Löwdin weight of the ghost AOs in a coefficient block
Shalf = real.(sqrt(Hermitian(Matrix(S))))
ghost_weight(blk) = sum(abs2.(Shalf * blk)[ghost_aos, :])
function load_dump(path)
  io = open_trexio(path, "r")
  try
    b = read_trexio_basis(io)
    orbs, _ = read_trexio_orbitals(io, b)
    return orbs, read_trexio_orbital_classes(io)
  finally
    close_trexio(io)
  end
end

# --- automatic detection (Löwdin population over the occupied orbitals) ---
cMO, _, _ = ElemCo.Wavefunctions.fetch_orbitals(EC)
cMO_occ = cMO[1][:, EC.space['o']]
@test ElemCo.OrbRegion._collect_ghost_support(EC, cMO_occ, S, ao_atoms_global, 0.1)[1] == [ghost]
@test isempty(ElemCo.OrbRegion._collect_ghost_support(EC, cMO_occ, S, ao_atoms_global, 0.5)[1])
# the population of the ghost is reported (≈0.18) and reaches the 0.1 threshold
@test ElemCo.OrbRegion._collect_ghost_support(EC, cMO_occ, S, ao_atoms_global, 0.1)[2][ghost] > 0.1

# --- manual route: pao_centers includes the ghost; its AOs enter the Virtual space ---
@set region mode=:inclusive virtual=:support_opao occ_charge_thr=0.2 atom_charge_thr=10.0 pao_centers=Int[]
@set wf store="region_ghost_base.h5"
@region [2]
orbs0, cls0 = load_dump(joinpath(EC.scr, "region_ghost_base.h5"))
v0 = findall(==("Virtual"), cls0[1])
@test ghost_weight(orbs0[1][:, v0]) < 1.0    # ghost not in the support -> only incidental weight

@set region pao_centers=[4]
@set wf store="region_ghost_pao.h5"
@region [2]
orbs1, cls1 = load_dump(joinpath(EC.scr, "region_ghost_pao.h5"))
v1 = findall(==("Virtual"), cls1[1])
@test length(v1) > length(v0)                # ghost AOs added as OPAO virtuals
@test ghost_weight(orbs1[1][:, v1]) > 2.0    # ghost now carries substantial virtual weight

# --- automatic route end-to-end: a region on O auto-detects the ghost ---
@set region virtual=:support_opao occ_charge_thr=0.2 atom_charge_thr=0.1 pao_centers=Int[]
@set wf store="region_ghost_auto.h5"
@region [1]
orbs2, cls2 = load_dump(joinpath(EC.scr, "region_ghost_auto.h5"))
v2 = findall(==("Virtual"), cls2[1])
@test ghost_weight(orbs2[1][:, v2]) > 2.0    # ghost auto-included in the virtual space

# a ghost atom cannot be a fragment center (only a PAO center) — must error clearly and early
@set region pao_centers=Int[]
@test_throws ErrorException ElemCo.region_orbitals(EC, [ghost])
end
