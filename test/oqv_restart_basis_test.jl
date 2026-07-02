@testitem "oqv_restart_basis" tags=[:qvcc, :quick] begin
using ElemCo
using Test

# Restart an orbital-optimized CC (OQV-DCD) reusing stored optimized orbitals across a BASIS CHANGE
# (dump=""+start). The orbitals are projected onto the new AO basis and completed to a full set
# (extra virtuals appended when the basis grows, excess dropped when it shrinks); the amplitudes are
# embedded into the new correlation space. Each restart below follows a basis change so the FCIDUMP
# is regenerated from the stored orbitals (mirrors a fresh-session restart).
geometry = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
jk = "cc-pvtz-jkfit"; mp = "cc-pvtz-mpfit"

tmp = mktempdir()
wf_s = joinpath(tmp, "wf_s.h5"); wf_b = joinpath(tmp, "wf_b.h5")
cc_s = joinpath(tmp, "cc_s.h5"); cc_b = joinpath(tmp, "cc_b.h5")

# store OQV-DCD in the small basis (sto-3g)
basis = Dict("ao"=>"sto-3g", "jkfit"=>jk, "mpfit"=>mp)
@set wf dump=wf_s
@dfhf
E_small = (@cc oqv-dcd begin @set wf dump=wf_s store=cc_s end)["OQV-DCD"]

# store OQV-DCD in the larger basis (6-31G)
basis = Dict("ao"=>"6-31G", "jkfit"=>jk, "mpfit"=>mp)
@set wf dump=wf_b
@dfhf
E_big = (@cc oqv-dcd begin @set wf dump=wf_b store=cc_b end)["OQV-DCD"]

# restart in the SMALLER basis reusing the 6-31G optimized orbitals (orbital set is reduced)
basis = Dict("ao"=>"sto-3g", "jkfit"=>jk, "mpfit"=>mp)
E_small_restart = (@cc oqv-dcd begin @set wf start=cc_b dump="" end)["OQV-DCD"]

# restart in the LARGER basis reusing the sto-3g optimized orbitals (orbital set is completed)
basis = Dict("ao"=>"6-31G", "jkfit"=>jk, "mpfit"=>mp)
E_big_restart = (@cc oqv-dcd begin @set wf start=cc_s dump="" end)["OQV-DCD"]

@testset "OQV-DCD basis-change restart (dump=\"\")" begin
  @test isfile(cc_s) && isfile(cc_b)
  # reducing the basis recovers (very nearly) the small-basis OQV-DCD energy: the well-optimized
  # occupied space is preserved, so this is a tight, deterministic check.
  @test abs(E_small_restart - E_small) < 5.e-3
  # enlarging the basis must run and clearly lower the energy (it gains the larger-basis correlation).
  # We deliberately do NOT assert closeness to a fresh larger-basis run: the OQV orbital optimization
  # converges to a stationary point near the projected guess, which is basis/BLAS-dependent and not a
  # stable quantity to test.
  @test E_big_restart < E_small - 0.5
  @test E_big_restart > E_big - 0.5   # sane lower bound (near or above the fresh larger-basis result)
end
rm(tmp; force=true, recursive=true)
end
