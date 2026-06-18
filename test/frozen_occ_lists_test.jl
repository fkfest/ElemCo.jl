@testitem "frozen occ lists (issue #191)" tags=[:df, :quick] begin
using ElemCo
using ElemCo.ECInfos: translate_orbs_to_active

@testset "full→active orbital translation" begin
  # core orbital 1 frozen ⇒ active orbital k corresponds to full orbital k+1
  orig = 2:10
  @test translate_orbs_to_active([1, 2, 3, 4, 5], orig) == [1, 2, 3, 4]  # core 1 dropped, 2-5→1-4
  @test translate_orbs_to_active([2, 4, 7], orig) == [1, 3, 6]
  @test translate_orbs_to_active([1], orig) == Int[]                     # only frozen core → empty
  @test translate_orbs_to_active([3, 5], 1:0) == [3, 5]                 # no map → indices unchanged
  # orbitals outside the active range (frozen virtual / out of range) error, not silently dropped
  @test_throws ErrorException translate_orbs_to_active([11], orig)       # > hi
  @test_throws ErrorException translate_orbs_to_active([0], orig)        # < 1
end

@testset "occa/occb refer to the full MO space (@cc with frozen core)" begin
  geometry = "bohr
       O      0.000000000    0.000000000   -0.130186067
       H1     0.000000000    1.489124508    1.033245507
       H2     0.000000000   -1.489124508    1.033245507"
  basis = Dict("ao" => "cc-pVDZ",
               "jkfit" => "cc-pvtz-jkfit",
               "mpfit" => "cc-pvdz-mpfit")

  @ECinit
  @dfhf
  # default frozen-core DCSD reference
  e_default = @cc dcsd

  # occa/occb given in the FULL MO space (including the frozen core orbital 1). Before issue #191
  # this errored ("Inconsistency in OCCA (1-5) ... number of electrons (8)"); now the lists are
  # translated to the active dump (core dropped, valence renumbered) and reproduce the default.
  @set wf occa="1-5" occb="1-5"
  e_full = @cc dcsd
  @test e_full["DCSD"] ≈ e_default["DCSD"] rtol=1e-8
end
end
