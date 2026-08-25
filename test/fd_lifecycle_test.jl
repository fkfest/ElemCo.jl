@testitem "fd_lifecycle" tags=[:cc, :quick] begin
using ElemCo
using Test

# Creator-responsibility lifecycle of the MO-integral FCIDUMP (`EC.fd`): orbital-dependent integrals
# persist only when the user created them. A driver (@cc/@fci/@ciphi) that has to create the fd
# builds it from the CURRENT correlation orbitals and deletes it at the end of its run — so it can
# never be silently reused after the orbitals changed. Explicit integrals (@dfints, an fcidump file)
# persist, and refreshing them after orbital changes is the user's responsibility.
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
basis = "vdz"
epsilon = 1.e-9

@testset "driver-created fd is per-run; @dfints persists" begin
  @dfhf
  e_implicit = @cc ccsd
  # the driver created the DF integrals for itself and deleted them afterwards.
  # DISCRIMINATING: on the previous behavior the implicitly created fd persisted here.
  @test isempty(EC.fd)

  @dfints
  @test !isempty(EC.fd)
  e_explicit = @cc ccsd
  @test !isempty(EC.fd)                              # user-created: the driver leaves it alone
  @test abs(e_implicit["CCSD"] - e_explicit["CCSD"]) < epsilon

  # ... and it keeps serving further runs without regeneration
  e_again = @cc ccsd
  @test e_again["CCSD"] == e_explicit["CCSD"]
end

@testset "regeneration picks up changed orbitals" begin
  # the case the rule exists for: orbitals change at fixed geometry (here: localization).
  # With per-run integrals the second @cc runs on the NEW orbitals — DCSD is invariant under
  # occupied/virtual rotations, so the energy must match the canonical one. (Under the old
  # behavior the second @cc silently reused the canonical-orbital fd, testing nothing.)
  @dfhf
  e_canon = @cc dcsd
  @localize
  e_loc = @cc dcsd
  @test abs(e_loc["DCSD"] - e_canon["DCSD"]) < 1.e-7
end

@testset "@dfints + @write_ints flow" begin
  # (@write_ints takes a literal file name — it does not interpolate variables)
  @dfhf
  @dfints
  @write_ints "TESTDUMP_fdlife"
  @test isfile("TESTDUMP_fdlife")
  e = @cc mp2
  @test haskey(e, "MP2")
  @test !isempty(EC.fd)                              # still the user's
  rm("TESTDUMP_fdlife"; force=true)
end
end
