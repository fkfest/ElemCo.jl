@testitem "oqv_restart_geometry" tags=[:qvcc, :quick] begin
using ElemCo
using Test

# Restart an orbital-optimized CC (OQV-DCD) across a geometry change, then restart AGAIN at the
# unchanged geometry (both via dump=""+start). The second restart must resume at the stored
# solution in a single iteration.
#
# Regression: the wavefunction store rotated the *raw* orbitals read from the start file rather than
# the *correlation reference* the FCIDUMP was built from. Across a geometry change these differ (the
# reference is projected+re-orthonormalized into the current AO basis), so the store wrote old-basis
# coefficients tagged with the new geometry's basis — inconsistent with the stored amplitudes. A
# subsequent same-geometry restart then failed to resume and re-optimized over several iterations
# (and to a different energy). See `correlation_reference_orbital_data`.
G1 = "bohr
     O      0.000000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
G2 = "bohr
     O      0.001000000    0.000000000   -0.130186067
     H1     0.000000000    1.489124508    1.033245507
     H2     0.000000000   -1.489124508    1.033245507"
# cc-pVDZ (not a minimal basis) so the projected reference differs appreciably from the raw stored
# orbitals — the regression (storing raw instead of the reference) then shifts the resumed energy by
# ~7e-5, well clear of the tolerance below; a minimal basis would mask it at ~1e-7.
basis = "vdz"

@testset "OQV-DCD geometry-change restart resumes in one iteration (dump=\"\")" begin
  tmp = mktempdir()
  ccfile = joinpath(tmp, "cc.h5")

  # optimize at geometry G1 and store the wavefunction
  geometry = G1
  @dfhf
  @cc oqv-dcd begin @set wf store=ccfile end

  # restart at the displaced geometry G2 (rebuilds the FCIDUMP from the projected reference orbitals)
  geometry = G2
  e2 = @cc oqv-dcd begin @set wf start=ccfile dump="" store=ccfile end

  # restart AGAIN at the same geometry G2 reusing the just-stored solution. `maxit=2` makes this a
  # true resume test: without the fix the stored orbitals do not match the amplitudes, so two
  # iterations do not reproduce the previous energy (and it would need several more).
  e3 = @cc oqv-dcd begin
    @set wf start=ccfile dump="" store=ccfile
    @set cc maxit=2
  end
  @test abs(e3["OQV-DCD"] - e2["OQV-DCD"]) < 1.e-7

  # storing to a *different* file (not overwriting `start`) must resume identically
  ccfile2 = joinpath(tmp, "cc2.h5")
  e3b = @cc oqv-dcd begin
    @set wf start=ccfile dump="" store=ccfile2
    @set cc maxit=2
  end
  @test abs(e3b["OQV-DCD"] - e2["OQV-DCD"]) < 1.e-7

  rm(tmp; force=true, recursive=true)
end
end
