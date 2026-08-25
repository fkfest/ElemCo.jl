@testitem "redundant_hf" tags=[:df, :quick] begin
using ElemCo
using ElemCo.OrbTools
using LinearAlgebra

@testset "Redundant/Cartesian DF-HF Test" begin
epsilon = 1.e-6

# --- Unit test of the canonical orthogonalization used to remove redundancies ---
# nearly singular overlap (first two basis functions almost identical)
S = [1.0 (1.0 - 1.e-9) 0.0;
     (1.0 - 1.e-9) 1.0 0.0;
     0.0           0.0 1.0]
F = [0.5 0.1 0.2; 0.1 0.6 0.3; 0.2 0.3 0.7]
X, Xredundant = canonical_orthogonalization(S, 1.e-8)
@test size(X, 2) == 2            # one redundant direction removed
@test size(Xredundant, 2) == 1
@test isapprox(X' * S * X, I, atol=1.e-9)   # orthonormal in the S metric
ϵ, cMO = eigen_orth(F, X, Xredundant)
@test size(cMO) == (3, 3)        # coefficient matrix stays square
@test ϵ[end] > 1.e5              # redundant orbital parked at high energy
# the old generalized eigensolver crashes on an exactly singular overlap:
@test_throws Exception eigen(Hermitian(F), Hermitian([1.0 1.0 0.0; 1.0 1.0 0.0; 0.0 0.0 1.0]))

# --- Cartesian basis set (6 d functions instead of 5): the user's scenario ---
ECART_HF_test  = -76.0218062865106
ECART_UHF_test = -75.79311670650866

# Use the variable names `geometry`/`basis` that the @set/@dfhf macros pick up, so the
# system is re-derived from them (a stale `geometry` leaked by an earlier test file in
# the shared test module would otherwise overwrite EC.system via @setupEC).
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = Dict("ao"=>"cc-pVDZ", "jkfit"=>"cc-pvtz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(geometry, basis))
@set int cartesian=true
@set scf direct=true
@test ElemCo.guess_norb(EC) == 25    # spherical would be 24 (one extra Cartesian d)
hf = @dfhf
@test abs(hf["HF"] - ECART_HF_test) < epsilon
@set wf ms2=2
uhf = @dfuhf
@test abs(uhf["HF"] - ECART_UHF_test) < epsilon

# --- Genuinely redundant basis: must remove orbitals, converge, and freeze them post-HF ---
# An explicit H basis that duplicates one s contraction makes the AO overlap exactly
# rank-deficient (one redundant function per H, i.e. 2 for H2). The overlap of two
# identical normalized contracted functions is exactly 1, so the redundant eigenvalues
# are exactly zero (to machine precision) regardless of the BLAS/integral build -- the
# count is structural, platform-independent, and uses only ordinary (non-coincident)
# integrals. (the @set/@dfhf macros operate on a variable named `EC`, so we reassign it.)
ERED_HF_test  = -1.1287103855054323
EMP2_red_test = -1.1550781799131038
ECCSD_red_test = -1.1634434300922467
hbasis = "{
  s, H, 13.01, 1.962, 0.4446, 0.122
  c, 1.4, 0.019685, 0.137977, 0.478148, 0.50124
  c, 4.4, 1.0
  c, 4.4, 1.0
  p, H, 0.727
  c, 1.1, 1.0}"
geometry = "bohr
     H1 0.0 0.0 0.0
     H2 0.0 0.0 1.4"
basis = Dict("ao"=>hbasis, "jkfit"=>"cc-pvdz-jkfit", "mpfit"=>"cc-pvdz-mpfit")
EC = ElemCo.ECInfo(system=ElemCo.parse_geometry(geometry, basis))
@set int cartesian=false
@set scf direct=true redthr=1.e-6
@test ElemCo.OrbTools.n_redundant_orbitals(EC) == 2   # one duplicated function per H projected out
hfr = @dfhf
@test abs(hfr["HF"] - ERED_HF_test) < 1.e-6

# HF must record the redundant orbitals as "Deleted" in the wavefunction dump,
# and post-HF must read that count back
classa, classb = ElemCo.Wavefunctions.fetch_orbital_classes(EC)
@test count(==("Deleted"), classa) == 2
@test ElemCo.OrbTools.n_deleted_orbitals(EC) == 2

# post-HF must freeze the redundant orbitals out of the correlation treatment
mp2r = @dfmp2
@test abs(mp2r["MP2"] - EMP2_red_test) < 1.e-6

# FCIDump route: the dump is generated with the redundant orbitals already removed
fdump_red = "redundant_hf_test.FCIDUMP"
@set int fcidump=fdump_red
@set scf direct=false
@dfints
ccr = ElemCo.ccdriver(EC, "ccsd"; fcidump=fdump_red)
@test abs(ccr["CCSD"] - ECCSD_red_test) < 1.e-6
rm(fdump_red)
end
end
